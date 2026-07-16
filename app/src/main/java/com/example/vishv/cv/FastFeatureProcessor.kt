package com.example.vishv.cv

import android.graphics.Bitmap
import android.os.SystemClock
import com.example.vishv.model.AnalysisFrame
import com.example.vishv.model.FastKeyPoint
import com.example.vishv.model.FastSettings
import com.example.vishv.model.FeatureDetectionResult
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.features2d.FastFeatureDetector
import org.opencv.imgproc.Imgproc

/**
 * Converts an AnalysisFrame to grayscale, runs FAST corner detection, and optionally renders a
 * debug bitmap. Stateless; safe to share across calls from a single background thread at a time.
 *
 * Mat ownership: all Mats created here are released before returning.
 * Bitmap ownership: debugBitmap (when produced) is returned to the caller; the caller must not
 *   recycle the input frame.bitmap before this call returns.
 *
 * A new FastFeatureDetector is created on each call to avoid shared-state concurrency issues.
 * At 5 FPS the allocation cost is negligible; native memory is reclaimed by the JVM finalizer.
 */
class FastFeatureProcessor {

    fun process(
        frame: AnalysisFrame,
        settings: FastSettings,
        renderGrayscale: Boolean,
        renderKeypoints: Boolean,
    ): FeatureDetectionResult {
        val totalStart = SystemClock.elapsedRealtime()

        // 1. Bitmap → RGBA Mat.
        // Utils.bitmapToMat produces a 4-channel Mat with channel order R, G, B, A.
        val rgbaMat = Mat()
        Utils.bitmapToMat(frame.bitmap, rgbaMat)

        // 2. Grayscale conversion.
        val grayStart = SystemClock.elapsedRealtime()
        val grayMat = Mat()
        Imgproc.cvtColor(rgbaMat, grayMat, Imgproc.COLOR_RGBA2GRAY)
        val grayscaleTimeMs = SystemClock.elapsedRealtime() - grayStart

        // 3. FAST detection on the grayscale Mat.
        val fastStart = SystemClock.elapsedRealtime()
        val keypoints = MatOfKeyPoint()
        FastFeatureDetector
            .create(settings.threshold, settings.nonMaxSuppression)
            .detect(grayMat, keypoints)
        val fastTimeMs = SystemClock.elapsedRealtime() - fastStart

        val allKps = keypoints.toArray()
        val featurePoints = allKps
            .take(settings.maxFeaturePointsShown)
            .map { FastKeyPoint(it.pt.x.toFloat(), it.pt.y.toFloat(), it.response.toFloat()) }

        // 4. Optional debug bitmap.
        val debugBitmap: Bitmap? = when {
            renderGrayscale -> buildGrayscaleBitmap(grayMat)
            renderKeypoints -> buildKeypointsBitmap(rgbaMat, allKps, settings.maxFeaturePointsShown)
            else -> null
        }

        // 5. Release all Mats.
        rgbaMat.release()
        grayMat.release()
        keypoints.release()

        return FeatureDetectionResult(
            source = frame.source,
            timestampMs = frame.timestampMs,
            frameWidth = frame.width,
            frameHeight = frame.height,
            grayscaleTimeMs = grayscaleTimeMs,
            fastTimeMs = fastTimeMs,
            totalProcessingTimeMs = SystemClock.elapsedRealtime() - totalStart,
            featureCount = allKps.size,
            featurePoints = featurePoints,
            debugBitmap = debugBitmap,
        )
    }

    private fun buildGrayscaleBitmap(grayMat: Mat): Bitmap {
        val grayRgba = Mat()
        Imgproc.cvtColor(grayMat, grayRgba, Imgproc.COLOR_GRAY2RGBA)
        val bmp = Bitmap.createBitmap(grayRgba.cols(), grayRgba.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(grayRgba, bmp)
        grayRgba.release()
        return bmp
    }

    private fun buildKeypointsBitmap(
        rgbaMat: Mat,
        keypoints: Array<org.opencv.core.KeyPoint>,
        maxPoints: Int,
    ): Bitmap {
        // Clone so we don't modify the original (needed for potential future pipeline stages).
        val displayMat = rgbaMat.clone()
        val green = Scalar(0.0, 255.0, 0.0, 255.0) // RGBA green
        for (kp in keypoints.take(maxPoints)) {
            // Draw a circle at each keypoint with radius 5. Thick outline (2px) is visible
            // even on high-res frames without overwhelming the image at lower keypoint counts.
            Imgproc.circle(displayMat, Point(kp.pt.x, kp.pt.y), 5, green, 2)
        }
        val bmp = Bitmap.createBitmap(displayMat.cols(), displayMat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(displayMat, bmp)
        displayMat.release()
        return bmp
    }
}
