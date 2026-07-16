package com.example.vishv.model

import android.graphics.Bitmap

enum class StabilizationFailureReason {
    NONE,               // Success
    FIRST_FRAME,        // No previous frame stored yet
    DUPLICATE_FRAME,    // Consecutive video frames were identical (hash check)
    TIMESTAMP_GAP,      // Gap between frames too large — likely a seek or restart
    TOO_FEW_FEATURES,   // FAST+ORB produced no usable keypoints
    TOO_FEW_MATCHES,    // Fewer than minMatches after cross-check filtering
    TOO_FEW_INLIERS,    // Fewer than minMatches inliers after RANSAC
    LOW_INLIER_RATIO,   // Inlier/filtered-match ratio below threshold
    TRANSLATION_TOO_LARGE,
    ROTATION_TOO_LARGE,
    SCALE_CHANGE_TOO_LARGE,
    ESTIMATOR_FAILED,   // estimateAffinePartial2D returned empty Mat
    INVALID_MATRIX,     // NaN or Inf in transform coefficients
}

/**
 * Result of one stabilization pass. Contains no OpenCV-native types — safe to post to StateFlow
 * and observe from Compose.
 *
 * Bitmap ownership: debugBitmap and stabilizedBitmap are created by StabilizationProcessor and
 * transferred here. They are not recycled explicitly; GC reclaims them when the StateFlow value
 * is replaced and Compose finishes rendering the previous frame.
 */
data class StabilizationResult(
    val source: FrameSource,
    val previousTimestampMs: Long?,
    val currentTimestampMs: Long,
    // --- Feature / matching counts ---
    val featureCount: Int,
    val descriptorCount: Int,
    val rawMatchCount: Int,
    val filteredMatchCount: Int,
    val inlierCount: Int,
    val inlierRatio: Float,
    // --- Transform ---
    val transformSuccess: Boolean,
    val failureReason: StabilizationFailureReason,
    val translationX: Float,
    val translationY: Float,
    val rotationDeg: Float,
    val scale: Float,
    // --- Timing ---
    val featureProcessingTimeMs: Long,
    val matchingTimeMs: Long,
    val transformEstimationTimeMs: Long,
    val warpTimeMs: Long,
    val totalStabilizationTimeMs: Long,
    // --- Difference diagnostics ---
    val meanAbsDiffBefore: Float?,
    val meanAbsDiffAfter: Float?,
    val diffReductionPct: Float?,
    // --- Debug bitmap (mode-dependent, may be null) ---
    val debugBitmap: Bitmap?,
    // --- Stabilized frame bitmap (always set when transformSuccess == true) ---
    // Consumed by GmmForegroundProcessor to keep the background model camera-shake-free.
    val stabilizedBitmap: Bitmap?,
)
