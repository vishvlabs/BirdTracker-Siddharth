package com.example.vishv.model

import android.graphics.Bitmap

enum class CandidateStatus {
    WARMING_UP,                 // GMM still in warm-up; no mask available.
    GLOBAL_CHANGE,              // Cleaned foreground exceeded maxForegroundCoveragePct; probable camera motion or scene cut.
    STABILIZATION_UNAVAILABLE,  // Stabilization failed; GMM skipped this frame.
    DUPLICATE_FRAME,            // Identical to the previous frame; GMM skipped.
    NO_MOTION,                  // GMM applied and mask cleaned, but no contours passed the size filters.
    ACTIVE,                     // One or more candidate regions accepted.
    SKIPPED,                    // Any other skip (e.g., rawMaskBitmap null).
}

data class CandidateExtractionResult(
    val source: FrameSource,
    val timestampMs: Long,
    val candidateStatus: CandidateStatus,
    val isGlobalChange: Boolean,
    // --- Foreground pixel counts ---
    val rawForegroundPixels: Int,
    val cleanedForegroundPixels: Int,
    val totalPixels: Int,
    val rawForegroundPct: Float,
    val cleanedForegroundPct: Float,
    // --- Contour stats ---
    val rawContourCount: Int,
    val acceptedCount: Int,
    val rejectedSmallCount: Int,
    val rejectedLargeCount: Int,
    val rejectedShapeCount: Int,
    val candidateRegions: List<CandidateRegion>,
    // --- Timing ---
    val morphologyTimeMs: Long,
    val contourTimeMs: Long,
    val totalTimeMs: Long,
    // --- Bitmaps (display only) ---
    val cleanedMaskBitmap: Bitmap?,      // Only when displayMode == CLEANED_MASK.
    val candidateOverlayBitmap: Bitmap?, // Only when displayMode == CANDIDATE_REGIONS && candidates non-empty.
    val skipReason: String?,
)
