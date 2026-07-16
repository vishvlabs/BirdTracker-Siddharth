package com.example.vishv.model

import android.graphics.Bitmap

enum class GmmModelStatus {
    WARMING_UP, // Still accumulating frames; foreground mask is not yet reliable.
    ACTIVE,     // Background model is trained; mask is meaningful.
}

enum class GmmSkipReason {
    NONE,
    STABILIZATION_UNAVAILABLE, // Stabilization failed; frame skipped to protect background model.
    DUPLICATE_FRAME,           // Identical to previous frame; skip to avoid model stagnation.
    NO_STABILIZED_FRAME,       // stabilizedBitmap unexpectedly absent despite transform success.
}

/**
 * Result of one GMM foreground-detection pass. Contains no OpenCV-native types — safe to
 * post to StateFlow and observe from Compose.
 *
 * Bitmap ownership: maskBitmap is created by GmmForegroundProcessor and transferred here.
 * It is not recycled explicitly; GC reclaims it when the StateFlow value is replaced.
 */
data class ForegroundDetectionResult(
    val source: FrameSource,
    val timestampMs: Long,
    // --- Model state ---
    val modelStatus: GmmModelStatus,
    val warmUpFramesCompleted: Int,
    val warmUpFramesRequired: Int,
    // --- Application ---
    val gmmApplied: Boolean,
    val skipReason: GmmSkipReason,
    val stabFailureReason: StabilizationFailureReason?,
    // --- Pixel counts (0 when warming up or skipped) ---
    val foregroundPixelCount: Int,
    val shadowPixelCount: Int,
    val totalPixelCount: Int,
    val foregroundPct: Float,
    val shadowPct: Float,
    val shadowDetectionEnabled: Boolean,
    // --- Timing ---
    val gmmProcessingTimeMs: Long,
    // --- Debug bitmap (only populated when displayMode == FOREGROUND_MASK) ---
    val maskBitmap: Bitmap?,
)
