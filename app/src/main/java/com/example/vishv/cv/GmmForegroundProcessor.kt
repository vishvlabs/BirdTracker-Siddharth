package com.example.vishv.cv

import android.graphics.Bitmap
import android.os.SystemClock
import com.example.vishv.model.ForegroundDetectionResult
import com.example.vishv.model.FrameSource
import com.example.vishv.model.GmmModelStatus
import com.example.vishv.model.GmmSettings
import com.example.vishv.model.GmmSkipReason
import com.example.vishv.model.StabilizationFailureReason
import com.example.vishv.model.StabilizationResult
import com.example.vishv.viewmodel.DisplayMode
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import org.opencv.video.BackgroundSubtractorMOG2
import org.opencv.video.Video

/**
 * Per-source GMM background subtraction. Consumes the stabilized frame from StabilizationResult
 * so that camera shake does not contaminate the background model.
 *
 * Failure policy: when stabilization failed for this frame, the GMM is skipped and the background
 * model is left unchanged. This prevents misaligned frames from corrupting the distribution.
 *
 * Thread safety: stateBySource is guarded by stateLock. warmUpFramesCompleted is mutated only
 * from the processing thread; since camera and video never run simultaneously, no additional
 * locking is required for that field. pendingSettings is @Volatile and written by the main thread.
 */
class GmmForegroundProcessor {

    private val stateLock = Object()
    private val stateBySource = mutableMapOf<FrameSource, SourceState>()

    @Volatile private var pendingSettings = GmmSettings()

    fun updateSettings(settings: GmmSettings) { pendingSettings = settings }

    fun resetSource(source: FrameSource) {
        synchronized(stateLock) { stateBySource.remove(source) }
    }

    fun resetAll() {
        synchronized(stateLock) { stateBySource.clear() }
    }

    fun process(
        source: FrameSource,
        timestampMs: Long,
        stabResult: StabilizationResult,
        isDuplicate: Boolean,
        displayMode: DisplayMode,
    ): ForegroundDetectionResult {
        val totalStart = SystemClock.elapsedRealtime()
        val settings = pendingSettings

        if (isDuplicate) {
            return makeSkipResult(source, timestampMs, settings, GmmSkipReason.DUPLICATE_FRAME, null, totalStart)
        }
        if (!stabResult.transformSuccess) {
            return makeSkipResult(
                source, timestampMs, settings,
                GmmSkipReason.STABILIZATION_UNAVAILABLE, stabResult.failureReason, totalStart,
            )
        }
        val stabilizedBitmap = stabResult.stabilizedBitmap
        if (stabilizedBitmap == null || stabilizedBitmap.isRecycled) {
            return makeSkipResult(source, timestampMs, settings, GmmSkipReason.NO_STABILIZED_FRAME, null, totalStart)
        }

        // Get or create source state; recreate if settings changed so MOG2 history/threshold/shadow
        // parameters take effect immediately (MOG2 has no setter for these after construction).
        var state = synchronized(stateLock) { stateBySource[source] }
        if (state == null || state.appliedSettings != settings) {
            state = SourceState(
                mog2 = Video.createBackgroundSubtractorMOG2(
                    settings.history, settings.varThreshold, settings.detectShadows,
                ),
                warmUpFramesCompleted = 0,
                appliedSettings = settings,
            )
            synchronized(stateLock) { stateBySource[source] = state }
        }

        val gmmStart = SystemClock.elapsedRealtime()

        // Convert stabilized bitmap → grayscale for MOG2 (faster than 3-channel processing).
        val rgbaMat = Mat()
        Utils.bitmapToMat(stabilizedBitmap, rgbaMat)
        val grayMat = Mat()
        Imgproc.cvtColor(rgbaMat, grayMat, Imgproc.COLOR_RGBA2GRAY)
        rgbaMat.release()

        // 0.0 maps to OpenCV's automatic learning rate (-1.0); any positive value is used as-is.
        val effectiveLearningRate = if (settings.learningRate == 0.0) -1.0 else settings.learningRate
        val fgMask = Mat()
        state.mog2.apply(grayMat, fgMask, effectiveLearningRate)
        grayMat.release()

        val gmmTimeMs = SystemClock.elapsedRealtime() - gmmStart

        // fgMask is CV_8UC1: 0 = background, 127 = shadow (if detectShadows), 255 = foreground.
        val totalPixels = fgMask.rows() * fgMask.cols()

        val fgBinary = Mat()
        Core.compare(fgMask, Scalar(255.0), fgBinary, Core.CMP_EQ)
        val fgCount = Core.countNonZero(fgBinary)
        fgBinary.release()

        val shadowCount = if (settings.detectShadows) {
            val shadowBinary = Mat()
            Core.compare(fgMask, Scalar(127.0), shadowBinary, Core.CMP_EQ)
            val count = Core.countNonZero(shadowBinary)
            shadowBinary.release()
            count
        } else 0

        // Advance warm-up counter AFTER applying the frame so the count reflects applied frames.
        state.warmUpFramesCompleted++
        val modelStatus = if (state.warmUpFramesCompleted < settings.warmUpFrames)
            GmmModelStatus.WARMING_UP else GmmModelStatus.ACTIVE

        // Always produce rawMaskBitmap so CandidateRegionExtractor can consume it downstream.
        val rgbaOut = Mat()
        Imgproc.cvtColor(fgMask, rgbaOut, Imgproc.COLOR_GRAY2RGBA)
        val rawMaskBitmap = Bitmap.createBitmap(rgbaOut.cols(), rgbaOut.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rgbaOut, rawMaskBitmap)
        rgbaOut.release()
        fgMask.release()

        // maskBitmap reuses rawMaskBitmap for the FOREGROUND_MASK display mode — no second conversion.
        val maskBitmap: Bitmap? = if (displayMode == DisplayMode.FOREGROUND_MASK) rawMaskBitmap else null

        val fgPct = if (totalPixels > 0) fgCount.toFloat() / totalPixels * 100f else 0f
        val shadowPct = if (totalPixels > 0) shadowCount.toFloat() / totalPixels * 100f else 0f

        return ForegroundDetectionResult(
            source = source,
            timestampMs = timestampMs,
            modelStatus = modelStatus,
            warmUpFramesCompleted = state.warmUpFramesCompleted,
            warmUpFramesRequired = settings.warmUpFrames,
            gmmApplied = true,
            skipReason = GmmSkipReason.NONE,
            stabFailureReason = null,
            foregroundPixelCount = fgCount,
            shadowPixelCount = shadowCount,
            totalPixelCount = totalPixels,
            foregroundPct = fgPct,
            shadowPct = shadowPct,
            shadowDetectionEnabled = settings.detectShadows,
            gmmProcessingTimeMs = gmmTimeMs,
            maskBitmap = maskBitmap,
            rawMaskBitmap = rawMaskBitmap,
        )
    }

    private fun makeSkipResult(
        source: FrameSource,
        timestampMs: Long,
        settings: GmmSettings,
        skipReason: GmmSkipReason,
        stabFailureReason: StabilizationFailureReason?,
        totalStart: Long,
    ): ForegroundDetectionResult {
        val warmUpCount = synchronized(stateLock) { stateBySource[source]?.warmUpFramesCompleted ?: 0 }
        val modelStatus = if (warmUpCount < settings.warmUpFrames) GmmModelStatus.WARMING_UP else GmmModelStatus.ACTIVE
        return ForegroundDetectionResult(
            source = source,
            timestampMs = timestampMs,
            modelStatus = modelStatus,
            warmUpFramesCompleted = warmUpCount,
            warmUpFramesRequired = settings.warmUpFrames,
            gmmApplied = false,
            skipReason = skipReason,
            stabFailureReason = stabFailureReason,
            foregroundPixelCount = 0,
            shadowPixelCount = 0,
            totalPixelCount = 0,
            foregroundPct = 0f,
            shadowPct = 0f,
            shadowDetectionEnabled = settings.detectShadows,
            gmmProcessingTimeMs = SystemClock.elapsedRealtime() - totalStart,
            maskBitmap = null,
            rawMaskBitmap = null,
        )
    }

    private class SourceState(
        val mog2: BackgroundSubtractorMOG2,
        var warmUpFramesCompleted: Int,
        val appliedSettings: GmmSettings,
    )
}
