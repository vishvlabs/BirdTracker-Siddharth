package com.example.vishv.cv

import android.graphics.Bitmap
import android.os.SystemClock
import com.example.vishv.model.AnalysisFrame
import com.example.vishv.model.FastSettings
import com.example.vishv.model.FrameSource
import com.example.vishv.model.StabilizationFailureReason
import com.example.vishv.model.StabilizationResult
import com.example.vishv.model.StabilizationSettings
import com.example.vishv.viewmodel.DisplayMode
import org.opencv.android.Utils
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.FastFeatureDetector
import org.opencv.features2d.ORB
import org.opencv.imgproc.Imgproc
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.sqrt

/**
 * Performs pairwise stabilization between consecutive frames from the same source.
 *
 * Pipeline per call:
 *   Bitmap → RGBA Mat → Gray Mat
 *   → FAST keypoints (top-K by response)
 *   → ORB descriptors
 *   → BFMatcher cross-check matching against previous frame
 *   → Sort by distance, take best N
 *   → estimateAffinePartial2D (RANSAC)
 *   → Validate transform
 *   → warpAffine (current → previous coordinate system)
 *   → absdiff before / after
 *   → optional debug bitmap
 *
 * Mat ownership:
 *   All temporary Mats are released before returning.
 *   FrameState holds grayMat and descriptors across frames; these are released in resetSource(),
 *   resetAll(), and when replaced by the next frame's state.
 *
 * Thread safety:
 *   stateBySource is guarded by stateLock.
 *   fastDet is only written from the processing thread (the single camera executor or the single
 *   IO coroutine per source); volatile pendingThreshold/pendingNms are written by the main thread.
 *   Since camera and video never run simultaneously, a single processing thread accesses the
 *   processor at any one time.
 */
class StabilizationProcessor {

    companion object {
        private const val MAX_TIMESTAMP_GAP_MS = 2_000L
        private const val MAX_DISPLAY_LINES = 80
        private const val VIZ_SCALE = 0.4     // Scale for side-by-side match visualization
        private const val DIFF_AMPLIFY = 3.0  // Constant multiplier for diff bitmaps
    }

    // ── Per-source temporal state ───────────────────────────────────────────────────────────────

    private val stateLock = Object()
    private val stateBySource = mutableMapOf<FrameSource, FrameState>()

    // ── Native processors ────────────────────────────────────────────────────────────────────────
    // Created once; fastDet is rebuilt when FAST settings change.

    @Volatile private var pendingThreshold = 20
    @Volatile private var pendingNms = true
    private var currentThreshold = -1
    private var currentNms = true
    private var fastDet: FastFeatureDetector = FastFeatureDetector.create(20, true)
    private val orb: ORB = ORB.create()
    // crossCheck=true: a match is valid only if it's mutually best in both directions.
    private val matcher: BFMatcher = BFMatcher.create(Core.NORM_HAMMING, true)

    // ── Public API ───────────────────────────────────────────────────────────────────────────────

    fun updateFastSettings(threshold: Int, nms: Boolean) {
        pendingThreshold = threshold
        pendingNms = nms
    }

    fun resetSource(source: FrameSource) {
        synchronized(stateLock) {
            stateBySource.remove(source)?.release()
        }
    }

    fun resetAll() {
        synchronized(stateLock) {
            stateBySource.values.forEach { it.release() }
            stateBySource.clear()
        }
    }

    fun release() = resetAll()

    // ── Main processing entry ────────────────────────────────────────────────────────────────────

    fun process(
        frame: AnalysisFrame,
        fastSettings: FastSettings,
        stabSettings: StabilizationSettings,
        isDuplicate: Boolean,
        displayMode: DisplayMode,
    ): StabilizationResult {
        val totalStart = SystemClock.elapsedRealtime()

        // Rebuild FAST detector if settings changed.
        val threshold = pendingThreshold
        val nms = pendingNms
        if (threshold != currentThreshold || nms != currentNms) {
            fastDet = FastFeatureDetector.create(threshold, nms)
            currentThreshold = threshold
            currentNms = nms
        }

        // ── 1. Bitmap → gray ────────────────────────────────────────────────────────────────────
        val fpStart = SystemClock.elapsedRealtime()
        val rgbaMat = Mat()
        Utils.bitmapToMat(frame.bitmap, rgbaMat)
        val grayMat = Mat()
        Imgproc.cvtColor(rgbaMat, grayMat, Imgproc.COLOR_RGBA2GRAY)

        // ── 2. FAST detection (top-K by response) ───────────────────────────────────────────────
        val allKpsMat = MatOfKeyPoint()
        fastDet.detect(grayMat, allKpsMat)
        val topKps = allKpsMat.toArray()
            .sortedByDescending { it.response }
            .take(stabSettings.maxKeypointsForStab)
        allKpsMat.release()

        // ── 3. ORB descriptors ───────────────────────────────────────────────────────────────────
        val kpsMat = MatOfKeyPoint(*topKps.toTypedArray())
        val descriptors = Mat()
        orb.compute(grayMat, kpsMat, descriptors)
        val finalKps = kpsMat.toArray()   // ORB may remove border keypoints
        kpsMat.release()
        val featureProcessingTimeMs = SystemClock.elapsedRealtime() - fpStart

        // ── 4. Duplicate frame (video only) — retain previous state ─────────────────────────────
        if (isDuplicate) {
            grayMat.release()
            descriptors.release()
            rgbaMat.release()
            return emptyResult(
                frame, featureProcessingTimeMs, finalKps.size, descriptors.rows(),
                StabilizationFailureReason.DUPLICATE_FRAME, totalStart,
            )
        }

        // ── 5. Load previous state ───────────────────────────────────────────────────────────────
        val prevState: FrameState? = synchronized(stateLock) { stateBySource[frame.source] }

        if (finalKps.isEmpty() || descriptors.empty()) {
            // Too few features — store current as new previous (blank-wall scenario).
            storeState(frame.source, FrameState(grayMat, finalKps, descriptors, frame.timestampMs))
            rgbaMat.release()
            return emptyResult(
                frame, featureProcessingTimeMs, finalKps.size, 0,
                StabilizationFailureReason.TOO_FEW_FEATURES, totalStart,
            )
        }

        if (prevState == null) {
            storeState(frame.source, FrameState(grayMat, finalKps, descriptors, frame.timestampMs))
            rgbaMat.release()
            return emptyResult(
                frame, featureProcessingTimeMs, finalKps.size, descriptors.rows(),
                StabilizationFailureReason.FIRST_FRAME, totalStart,
            )
        }

        // ── 6. Timestamp gap (detect seek / restart) ─────────────────────────────────────────────
        val gapMs = frame.timestampMs - prevState.timestampMs
        if (gapMs < 0 || gapMs > MAX_TIMESTAMP_GAP_MS) {
            storeState(frame.source, FrameState(grayMat, finalKps, descriptors, frame.timestampMs))
            rgbaMat.release()
            return emptyResult(
                frame, featureProcessingTimeMs, finalKps.size, descriptors.rows(),
                StabilizationFailureReason.TIMESTAMP_GAP, totalStart,
            )
        }

        // ── 7. Descriptor matching ───────────────────────────────────────────────────────────────
        val matchStart = SystemClock.elapsedRealtime()
        val matchesMat = MatOfDMatch()
        // query = current descriptors, train = previous descriptors
        matcher.match(descriptors, prevState.descriptors, matchesMat)
        val allMatches = matchesMat.toArray()
        matchesMat.release()
        val filteredMatches = allMatches.sortedBy { it.distance }.take(stabSettings.maxMatchesUsed)
        val matchingTimeMs = SystemClock.elapsedRealtime() - matchStart

        if (filteredMatches.size < stabSettings.minMatches) {
            storeState(frame.source, FrameState(grayMat, finalKps, descriptors, frame.timestampMs))
            rgbaMat.release()
            return buildResult(
                frame, prevState, featureProcessingTimeMs, matchingTimeMs, 0L, 0L,
                finalKps.size, descriptors.rows(), allMatches.size, filteredMatches.size,
                0, 0f, StabilizationFailureReason.TOO_FEW_MATCHES,
                0f, 0f, 0f, 1f, null, null, null, null, null, totalStart,
            )
        }

        // ── 8. Build point correspondences ───────────────────────────────────────────────────────
        // from = current frame points; to = previous frame points.
        // estimateAffinePartial2D(from, to) returns M such that M maps from → to,
        // i.e., M maps current frame points into the previous frame's coordinate system.
        // warpAffine(currentRgba, stabilized, M, size) then aligns current to previous.
        val estStart = SystemClock.elapsedRealtime()
        val fromList = mutableListOf<Point>()
        val toList = mutableListOf<Point>()
        for (m in filteredMatches) {
            fromList.add(Point(finalKps[m.queryIdx].pt.x, finalKps[m.queryIdx].pt.y))
            toList.add(Point(prevState.keypoints[m.trainIdx].pt.x, prevState.keypoints[m.trainIdx].pt.y))
        }
        val fromPts = MatOfPoint2f(); fromPts.fromList(fromList)
        val toPts = MatOfPoint2f(); toPts.fromList(toList)

        val inliersMask = Mat()
        val transformMat = Calib3d.estimateAffinePartial2D(
            fromPts, toPts, inliersMask,
            Calib3d.RANSAC, stabSettings.ransacThreshold,
            2000L, 0.995, 10L,
        )
        fromPts.release(); toPts.release()
        val transformEstimationTimeMs = SystemClock.elapsedRealtime() - estStart

        // ── 9. Validate transform ─────────────────────────────────────────────────────────────────
        if (transformMat == null || transformMat.empty()) {
            inliersMask.release()
            val debugBmp = if (displayMode == DisplayMode.FEATURE_MATCHES)
                buildMatchBitmap(prevState, rgbaMat, finalKps, filteredMatches, null)
            else null
            storeState(frame.source, FrameState(grayMat, finalKps, descriptors, frame.timestampMs))
            rgbaMat.release()
            return buildResult(
                frame, prevState, featureProcessingTimeMs, matchingTimeMs,
                transformEstimationTimeMs, 0L,
                finalKps.size, descriptors.rows(), allMatches.size, filteredMatches.size,
                0, 0f, StabilizationFailureReason.ESTIMATOR_FAILED,
                0f, 0f, 0f, 1f, null, null, null, debugBmp, null, totalStart,
            )
        }

        val inlierCount = Core.countNonZero(inliersMask)
        val inlierRatio = inlierCount.toFloat() / filteredMatches.size

        val a = transformMat.get(0, 0)[0]
        val b = transformMat.get(1, 0)[0]
        val tx = transformMat.get(0, 2)[0].toFloat()
        val ty = transformMat.get(1, 2)[0].toFloat()
        val scale = sqrt(a * a + b * b).toFloat()
        val rotDeg = (atan2(b, a) * 180.0 / PI).toFloat()

        val failureReason = when {
            a.isNaN() || b.isNaN() || tx.isNaN() || ty.isNaN() ->
                StabilizationFailureReason.INVALID_MATRIX
            inlierCount < stabSettings.minMatches ->
                StabilizationFailureReason.TOO_FEW_INLIERS
            inlierRatio < stabSettings.minInlierRatio ->
                StabilizationFailureReason.LOW_INLIER_RATIO
            abs(tx) > stabSettings.maxTranslationPx || abs(ty) > stabSettings.maxTranslationPx ->
                StabilizationFailureReason.TRANSLATION_TOO_LARGE
            abs(rotDeg) > stabSettings.maxRotationDeg ->
                StabilizationFailureReason.ROTATION_TOO_LARGE
            abs(scale - 1f) > stabSettings.maxScaleChange ->
                StabilizationFailureReason.SCALE_CHANGE_TOO_LARGE
            else -> StabilizationFailureReason.NONE
        }

        // ── 10. Warp and diff (only when transform is valid) ──────────────────────────────────────
        val warpStart = SystemClock.elapsedRealtime()
        var stabilizedRgba: Mat? = null
        var diffBefore: Mat? = null
        var diffAfter: Mat? = null
        var madBefore: Float? = null
        var madAfter: Float? = null

        if (failureReason == StabilizationFailureReason.NONE) {
            stabilizedRgba = Mat()
            Imgproc.warpAffine(
                rgbaMat, stabilizedRgba, transformMat,
                Size(rgbaMat.cols().toDouble(), rgbaMat.rows().toDouble()),
            )
            // Before diff: |prevGray − currentGray|
            diffBefore = Mat()
            Core.absdiff(prevState.grayMat, grayMat, diffBefore)
            madBefore = Core.mean(diffBefore).`val`[0].toFloat()

            // After diff: |prevGray − stabilizedGray|
            val stabilizedGray = Mat()
            Imgproc.cvtColor(stabilizedRgba, stabilizedGray, Imgproc.COLOR_RGBA2GRAY)
            diffAfter = Mat()
            Core.absdiff(prevState.grayMat, stabilizedGray, diffAfter)
            madAfter = Core.mean(diffAfter).`val`[0].toFloat()
            stabilizedGray.release()
        } else {
            // Still compute before-diff even on transform failure — useful for display.
            if (displayMode == DisplayMode.DIFF_BEFORE) {
                diffBefore = Mat()
                Core.absdiff(prevState.grayMat, grayMat, diffBefore)
                madBefore = Core.mean(diffBefore).`val`[0].toFloat()
            }
        }
        val warpTimeMs = SystemClock.elapsedRealtime() - warpStart

        // ── 11. Debug bitmap ──────────────────────────────────────────────────────────────────────
        // Always produce the stabilized bitmap so the downstream GMM stage can consume it
        // without re-running warpAffine. When the display mode is STABILIZED, reuse this
        // bitmap for the overlay rather than creating a second copy.
        val stabilizedBmp: Bitmap? = stabilizedRgba?.let { matToRgbaBitmap(it) }

        val debugBmp: Bitmap? = when (displayMode) {
            DisplayMode.FEATURE_MATCHES ->
                buildMatchBitmap(prevState, rgbaMat, finalKps, filteredMatches, inliersMask)
            DisplayMode.STABILIZED -> stabilizedBmp
            DisplayMode.DIFF_BEFORE ->
                diffBefore?.let { grayDiffToBitmap(it) }
            DisplayMode.DIFF_AFTER ->
                diffAfter?.let { grayDiffToBitmap(it) } ?: diffBefore?.let { grayDiffToBitmap(it) }
            else -> null   // M4 modes (GRAYSCALE, FAST_FEATURES) handled by FastFeatureProcessor
        }

        // ── 12. Release temporaries ───────────────────────────────────────────────────────────────
        transformMat.release()
        inliersMask.release()
        stabilizedRgba?.release()
        diffBefore?.release()
        diffAfter?.release()

        // ── 13. Update previous state ─────────────────────────────────────────────────────────────
        storeState(frame.source, FrameState(grayMat, finalKps, descriptors, frame.timestampMs))
        rgbaMat.release()

        val diffReductionPct = if (madBefore != null && madAfter != null && madBefore > 0f)
            (madBefore - madAfter) / madBefore * 100f else null

        return buildResult(
            frame, prevState,
            featureProcessingTimeMs, matchingTimeMs, transformEstimationTimeMs, warpTimeMs,
            finalKps.size, descriptors.rows(), allMatches.size, filteredMatches.size,
            inlierCount, inlierRatio, failureReason,
            tx, ty, rotDeg, scale, madBefore, madAfter, diffReductionPct, debugBmp,
            stabilizedBmp, totalStart,
        )
    }

    // ── State management ─────────────────────────────────────────────────────────────────────────

    private fun storeState(source: FrameSource, newState: FrameState) {
        synchronized(stateLock) {
            stateBySource[source]?.release()
            stateBySource[source] = newState
        }
    }

    // ── Debug bitmap builders ─────────────────────────────────────────────────────────────────────

    private fun buildMatchBitmap(
        prevState: FrameState,
        currentRgba: Mat,
        currentKps: Array<org.opencv.core.KeyPoint>,
        filteredMatches: List<org.opencv.core.DMatch>,
        inliersMask: Mat?,
    ): Bitmap {
        val scaledW = (currentRgba.cols() * VIZ_SCALE).toInt().coerceAtLeast(1)
        val scaledH = (currentRgba.rows() * VIZ_SCALE).toInt().coerceAtLeast(1)
        val sz = Size(scaledW.toDouble(), scaledH.toDouble())

        // Previous frame: convert gray → RGBA, scale down.
        val prevRgba = Mat()
        Imgproc.cvtColor(prevState.grayMat, prevRgba, Imgproc.COLOR_GRAY2RGBA)
        val prevSmall = Mat(); Imgproc.resize(prevRgba, prevSmall, sz); prevRgba.release()

        // Current frame: scale down.
        val currSmall = Mat(); Imgproc.resize(currentRgba, currSmall, sz)

        // Side-by-side canvas.
        val canvas = Mat(scaledH, scaledW * 2, CvType.CV_8UC4, Scalar(20.0, 20.0, 20.0, 255.0))
        prevSmall.copyTo(canvas.submat(0, scaledH, 0, scaledW))
        currSmall.copyTo(canvas.submat(0, scaledH, scaledW, scaledW * 2))
        prevSmall.release(); currSmall.release()

        // Draw match lines.
        val green = Scalar(0.0, 255.0, 0.0, 255.0)
        val red = Scalar(255.0, 60.0, 60.0, 255.0)
        filteredMatches.take(MAX_DISPLAY_LINES).forEachIndexed { i, match ->
            val isInlier = inliersMask?.get(i, 0)?.getOrNull(0)?.let { it > 0.0 } ?: false
            val color = if (isInlier) green else red
            val kpPrev = prevState.keypoints.getOrNull(match.trainIdx)?.pt ?: return@forEachIndexed
            val kpCurr = currentKps.getOrNull(match.queryIdx)?.pt ?: return@forEachIndexed
            Imgproc.line(
                canvas,
                Point(kpPrev.x * VIZ_SCALE, kpPrev.y * VIZ_SCALE),
                Point(kpCurr.x * VIZ_SCALE + scaledW, kpCurr.y * VIZ_SCALE),
                color, 1,
            )
        }

        val bmp = Bitmap.createBitmap(canvas.cols(), canvas.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(canvas, bmp)
        canvas.release()
        return bmp
    }

    private fun matToRgbaBitmap(mat: Mat): Bitmap {
        val bmp = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, bmp)
        return bmp
    }

    private fun grayDiffToBitmap(diffMat: Mat): Bitmap {
        val enhanced = Mat()
        diffMat.convertTo(enhanced, -1, DIFF_AMPLIFY, 0.0)
        val rgba = Mat()
        Imgproc.cvtColor(enhanced, rgba, Imgproc.COLOR_GRAY2RGBA)
        enhanced.release()
        val bmp = Bitmap.createBitmap(rgba.cols(), rgba.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rgba, bmp)
        rgba.release()
        return bmp
    }

    // ── Result builders ───────────────────────────────────────────────────────────────────────────

    private fun emptyResult(
        frame: AnalysisFrame,
        featureProcessingTimeMs: Long,
        featureCount: Int,
        descriptorCount: Int,
        reason: StabilizationFailureReason,
        totalStart: Long,
    ) = StabilizationResult(
        source = frame.source, previousTimestampMs = null, currentTimestampMs = frame.timestampMs,
        featureCount = featureCount, descriptorCount = descriptorCount,
        rawMatchCount = 0, filteredMatchCount = 0, inlierCount = 0, inlierRatio = 0f,
        transformSuccess = false, failureReason = reason,
        translationX = 0f, translationY = 0f, rotationDeg = 0f, scale = 1f,
        featureProcessingTimeMs = featureProcessingTimeMs, matchingTimeMs = 0L,
        transformEstimationTimeMs = 0L, warpTimeMs = 0L,
        totalStabilizationTimeMs = SystemClock.elapsedRealtime() - totalStart,
        meanAbsDiffBefore = null, meanAbsDiffAfter = null, diffReductionPct = null,
        debugBitmap = null,
        stabilizedBitmap = null,
    )

    @Suppress("LongParameterList")
    private fun buildResult(
        frame: AnalysisFrame,
        prevState: FrameState,
        featureProcessingTimeMs: Long,
        matchingTimeMs: Long,
        transformEstimationTimeMs: Long,
        warpTimeMs: Long,
        featureCount: Int,
        descriptorCount: Int,
        rawMatchCount: Int,
        filteredMatchCount: Int,
        inlierCount: Int,
        inlierRatio: Float,
        failureReason: StabilizationFailureReason,
        tx: Float, ty: Float, rotDeg: Float, scale: Float,
        madBefore: Float?, madAfter: Float?, diffReductionPct: Float?,
        debugBmp: Bitmap?,
        stabilizedBitmap: Bitmap?,
        totalStart: Long,
    ) = StabilizationResult(
        source = frame.source,
        previousTimestampMs = prevState.timestampMs,
        currentTimestampMs = frame.timestampMs,
        featureCount = featureCount, descriptorCount = descriptorCount,
        rawMatchCount = rawMatchCount, filteredMatchCount = filteredMatchCount,
        inlierCount = inlierCount, inlierRatio = inlierRatio,
        transformSuccess = failureReason == StabilizationFailureReason.NONE,
        failureReason = failureReason,
        translationX = tx, translationY = ty, rotationDeg = rotDeg, scale = scale,
        featureProcessingTimeMs = featureProcessingTimeMs,
        matchingTimeMs = matchingTimeMs,
        transformEstimationTimeMs = transformEstimationTimeMs,
        warpTimeMs = warpTimeMs,
        totalStabilizationTimeMs = SystemClock.elapsedRealtime() - totalStart,
        meanAbsDiffBefore = madBefore, meanAbsDiffAfter = madAfter,
        diffReductionPct = diffReductionPct,
        debugBitmap = debugBmp,
        stabilizedBitmap = stabilizedBitmap,
    )

    // ── Internal state holder ─────────────────────────────────────────────────────────────────────

    private class FrameState(
        val grayMat: Mat,
        val keypoints: Array<org.opencv.core.KeyPoint>,
        val descriptors: Mat,
        val timestampMs: Long,
    ) {
        fun release() {
            grayMat.release()
            descriptors.release()
            // KeyPoint[] has no native resources.
        }
    }
}
