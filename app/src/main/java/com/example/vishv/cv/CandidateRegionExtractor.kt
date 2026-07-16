package com.example.vishv.cv

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.SystemClock
import com.example.vishv.model.CandidateExtractionResult
import com.example.vishv.model.CandidateRegion
import com.example.vishv.model.CandidateStatus
import com.example.vishv.model.CleanupSettings
import com.example.vishv.model.ForegroundDetectionResult
import com.example.vishv.model.GmmModelStatus
import com.example.vishv.model.GmmSkipReason
import com.example.vishv.viewmodel.DisplayMode
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * Stateless morphological cleanup and contour-based candidate extraction.
 * Consumes rawMaskBitmap from ForegroundDetectionResult; produces CandidateExtractionResult.
 *
 * pendingSettings is @Volatile — updated by the main thread, read by the analysis thread.
 * No per-source state; the GMM already isolates source state upstream.
 */
class CandidateRegionExtractor {

    @Volatile private var pendingSettings = CleanupSettings()

    fun updateSettings(settings: CleanupSettings) { pendingSettings = settings }

    fun process(
        fgResult: ForegroundDetectionResult,
        displayMode: DisplayMode,
    ): CandidateExtractionResult {
        val totalStart = SystemClock.elapsedRealtime()
        val settings = pendingSettings

        if (!fgResult.gmmApplied) {
            val status = when (fgResult.skipReason) {
                GmmSkipReason.STABILIZATION_UNAVAILABLE -> CandidateStatus.STABILIZATION_UNAVAILABLE
                GmmSkipReason.DUPLICATE_FRAME -> CandidateStatus.DUPLICATE_FRAME
                else -> CandidateStatus.SKIPPED
            }
            return makeSkipResult(fgResult, status, "GMM not applied: ${fgResult.skipReason}", totalStart)
        }

        if (fgResult.modelStatus == GmmModelStatus.WARMING_UP) {
            return makeSkipResult(fgResult, CandidateStatus.WARMING_UP, null, totalStart)
        }

        val rawBitmap = fgResult.rawMaskBitmap
        if (rawBitmap == null || rawBitmap.isRecycled) {
            return makeSkipResult(fgResult, CandidateStatus.SKIPPED, "rawMaskBitmap unavailable", totalStart)
        }

        val morphStart = SystemClock.elapsedRealtime()

        // rawMaskBitmap is ARGB_8888 produced by COLOR_GRAY2RGBA so R=G=B=original fgMask value.
        val rgbaMat = Mat()
        Utils.bitmapToMat(rawBitmap, rgbaMat)
        val grayMat = Mat()
        Imgproc.cvtColor(rgbaMat, grayMat, Imgproc.COLOR_RGBA2GRAY)
        rgbaMat.release()

        // Binary threshold: fg=255 only, or include shadow=127 pixels.
        val thresholdValue = if (settings.includeShadows) 126.0 else 254.0
        val binaryMat = Mat()
        Imgproc.threshold(grayMat, binaryMat, thresholdValue, 255.0, Imgproc.THRESH_BINARY)
        grayMat.release()

        // Morphological opening (noise removal) → closing (gap filling).
        val ks = settings.morphKernelSize.let { if (it % 2 == 0) it + 1 else it }
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(ks.toDouble(), ks.toDouble()))
        val openedMat = Mat()
        Imgproc.morphologyEx(binaryMat, openedMat, Imgproc.MORPH_OPEN, kernel, Point(-1.0, -1.0), settings.openingIterations)
        binaryMat.release()
        val cleanedMat = Mat()
        Imgproc.morphologyEx(openedMat, cleanedMat, Imgproc.MORPH_CLOSE, kernel, Point(-1.0, -1.0), settings.closingIterations)
        openedMat.release()
        kernel.release()

        val totalPixels = cleanedMat.rows() * cleanedMat.cols()
        val cleanedFgCount = Core.countNonZero(cleanedMat)
        val cleanedFgPct = if (totalPixels > 0) cleanedFgCount.toFloat() / totalPixels * 100f else 0f
        val morphTimeMs = SystemClock.elapsedRealtime() - morphStart

        if (totalPixels > 0 && cleanedFgCount.toFloat() / totalPixels > settings.maxForegroundCoveragePct) {
            val cleanedBmp = if (displayMode == DisplayMode.CLEANED_MASK) matToArgbBitmap(cleanedMat) else null
            cleanedMat.release()
            return CandidateExtractionResult(
                source = fgResult.source,
                timestampMs = fgResult.timestampMs,
                candidateStatus = CandidateStatus.GLOBAL_CHANGE,
                isGlobalChange = true,
                rawForegroundPixels = fgResult.foregroundPixelCount,
                cleanedForegroundPixels = cleanedFgCount,
                totalPixels = totalPixels,
                rawForegroundPct = fgResult.foregroundPct,
                cleanedForegroundPct = cleanedFgPct,
                rawContourCount = 0,
                acceptedCount = 0,
                rejectedSmallCount = 0,
                rejectedLargeCount = 0,
                rejectedShapeCount = 0,
                candidateRegions = emptyList(),
                morphologyTimeMs = morphTimeMs,
                contourTimeMs = 0L,
                totalTimeMs = SystemClock.elapsedRealtime() - totalStart,
                cleanedMaskBitmap = cleanedBmp,
                candidateOverlayBitmap = null,
                skipReason = null,
            )
        }

        // Capture cleaned mask bitmap before findContours may modify the mat.
        val cleanedMaskBitmap = if (displayMode == DisplayMode.CLEANED_MASK) matToArgbBitmap(cleanedMat) else null

        val contourStart = SystemClock.elapsedRealtime()
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(cleanedMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        hierarchy.release()
        cleanedMat.release()

        val rawContourCount = contours.size
        val maxContourArea = settings.maxContourAreaFraction * totalPixels

        val accepted = mutableListOf<CandidateRegion>()
        var rejectedSmall = 0
        var rejectedLarge = 0
        var rejectedShape = 0
        var regionId = 1

        for (contour in contours) {
            val area = Imgproc.contourArea(contour).toFloat()
            val rect = Imgproc.boundingRect(contour)
            val w = rect.width
            val h = rect.height

            if (area < settings.minContourArea || w < settings.minWidth || h < settings.minHeight) {
                rejectedSmall++
                contour.release()
                continue
            }
            if (area > maxContourArea) {
                rejectedLarge++
                contour.release()
                continue
            }

            val bbArea = w * h
            val fillRatio = if (bbArea > 0) area / bbArea else 0f
            if (settings.minFillRatio > 0f && fillRatio < settings.minFillRatio) {
                rejectedShape++
                contour.release()
                continue
            }

            val m = Imgproc.moments(contour)
            val cx = if (m.m00 > 0) (m.m10 / m.m00).toFloat() else rect.x + w / 2f
            val cy = if (m.m00 > 0) (m.m01 / m.m00).toFloat() else rect.y + h / 2f

            if (accepted.size < settings.maxRegionsDisplayed) {
                accepted.add(
                    CandidateRegion(
                        id = regionId++,
                        left = rect.x, top = rect.y,
                        right = rect.x + w, bottom = rect.y + h,
                        width = w, height = h,
                        centroidX = cx, centroidY = cy,
                        contourArea = area,
                        boundingBoxArea = bbArea,
                        aspectRatio = if (h > 0) w.toFloat() / h else 0f,
                        fillRatio = fillRatio,
                    )
                )
            }
            contour.release()
        }

        val contourTimeMs = SystemClock.elapsedRealtime() - contourStart

        val candidateStatus = if (accepted.isEmpty()) CandidateStatus.NO_MOTION else CandidateStatus.ACTIVE
        val overlayBitmap = if (displayMode == DisplayMode.CANDIDATE_REGIONS && accepted.isNotEmpty())
            buildOverlayBitmap(rawBitmap.width, rawBitmap.height, accepted) else null

        return CandidateExtractionResult(
            source = fgResult.source,
            timestampMs = fgResult.timestampMs,
            candidateStatus = candidateStatus,
            isGlobalChange = false,
            rawForegroundPixels = fgResult.foregroundPixelCount,
            cleanedForegroundPixels = cleanedFgCount,
            totalPixels = totalPixels,
            rawForegroundPct = fgResult.foregroundPct,
            cleanedForegroundPct = cleanedFgPct,
            rawContourCount = rawContourCount,
            acceptedCount = accepted.size,
            rejectedSmallCount = rejectedSmall,
            rejectedLargeCount = rejectedLarge,
            rejectedShapeCount = rejectedShape,
            candidateRegions = accepted,
            morphologyTimeMs = morphTimeMs,
            contourTimeMs = contourTimeMs,
            totalTimeMs = SystemClock.elapsedRealtime() - totalStart,
            cleanedMaskBitmap = cleanedMaskBitmap,
            candidateOverlayBitmap = overlayBitmap,
            skipReason = null,
        )
    }

    private fun buildOverlayBitmap(width: Int, height: Int, candidates: List<CandidateRegion>): Bitmap {
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmp)
        val fillPaint = Paint().apply {
            style = Paint.Style.FILL
            color = Color.argb(50, 0, 255, 0)
        }
        val strokePaint = Paint().apply {
            style = Paint.Style.STROKE
            color = Color.GREEN
            strokeWidth = 2f
            isAntiAlias = false
        }
        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = (height * 0.03f).coerceAtLeast(10f)
            isFakeBoldText = true
            isAntiAlias = true
        }
        for (r in candidates) {
            canvas.drawRect(r.left.toFloat(), r.top.toFloat(), r.right.toFloat(), r.bottom.toFloat(), fillPaint)
            canvas.drawRect(r.left.toFloat(), r.top.toFloat(), r.right.toFloat(), r.bottom.toFloat(), strokePaint)
            canvas.drawText("R${r.id}", r.left.toFloat() + 2f, r.top + textPaint.textSize, textPaint)
        }
        return bmp
    }

    private fun matToArgbBitmap(mat: Mat): Bitmap {
        val rgbaOut = Mat()
        Imgproc.cvtColor(mat, rgbaOut, Imgproc.COLOR_GRAY2RGBA)
        val bmp = Bitmap.createBitmap(rgbaOut.cols(), rgbaOut.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rgbaOut, bmp)
        rgbaOut.release()
        return bmp
    }

    private fun makeSkipResult(
        fgResult: ForegroundDetectionResult,
        status: CandidateStatus,
        skipReason: String?,
        totalStart: Long,
    ) = CandidateExtractionResult(
        source = fgResult.source,
        timestampMs = fgResult.timestampMs,
        candidateStatus = status,
        isGlobalChange = false,
        rawForegroundPixels = fgResult.foregroundPixelCount,
        cleanedForegroundPixels = 0,
        totalPixels = fgResult.totalPixelCount,
        rawForegroundPct = fgResult.foregroundPct,
        cleanedForegroundPct = 0f,
        rawContourCount = 0,
        acceptedCount = 0,
        rejectedSmallCount = 0,
        rejectedLargeCount = 0,
        rejectedShapeCount = 0,
        candidateRegions = emptyList(),
        morphologyTimeMs = 0L,
        contourTimeMs = 0L,
        totalTimeMs = SystemClock.elapsedRealtime() - totalStart,
        cleanedMaskBitmap = null,
        candidateOverlayBitmap = null,
        skipReason = skipReason,
    )
}
