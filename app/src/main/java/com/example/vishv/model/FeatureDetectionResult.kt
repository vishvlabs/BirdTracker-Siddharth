package com.example.vishv.model

import android.graphics.Bitmap

/** A single detected FAST corner. Response is the sum-of-absolute-differences score. */
data class FastKeyPoint(val x: Float, val y: Float, val response: Float)

/**
 * Result produced by FastFeatureProcessor for one AnalysisFrame.
 *
 * Ownership: debugBitmap (if non-null) is created by FastFeatureProcessor and transferred to
 * the caller. It must NOT be recycled while it is referenced by a StateFlow collector. It is
 * intentionally not recycled explicitly — GC handles it once the StateFlow value is replaced.
 */
data class FeatureDetectionResult(
    val source: FrameSource,
    val timestampMs: Long,
    val frameWidth: Int,
    val frameHeight: Int,
    val grayscaleTimeMs: Long,
    val fastTimeMs: Long,
    val totalProcessingTimeMs: Long,
    val featureCount: Int,
    val featurePoints: List<FastKeyPoint>,
    /** Non-null only when DisplayMode is GRAYSCALE or FAST_FEATURES. */
    val debugBitmap: Bitmap?,
)
