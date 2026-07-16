package com.example.vishv.model

data class StabilizationSettings(
    /** Minimum filtered matches required to attempt transform estimation. */
    val minMatches: Int = 8,
    /** Best-N matches (by Hamming distance) fed to RANSAC. */
    val maxMatchesUsed: Int = 100,
    /** RANSAC reprojection threshold in pixels. */
    val ransacThreshold: Double = 3.0,
    /** Minimum inlier-to-filtered-match ratio to accept the transform. */
    val minInlierRatio: Float = 0.25f,
    /** Reject transforms with |tx| or |ty| above this value (pixels). */
    val maxTranslationPx: Float = 300f,
    /** Reject transforms with |rotation| above this value (degrees). */
    val maxRotationDeg: Float = 30f,
    /** Reject transforms where |scale - 1| exceeds this fraction. */
    val maxScaleChange: Float = 0.3f,
    /** Cap FAST keypoints fed to ORB to keep descriptor/matching cost bounded. */
    val maxKeypointsForStab: Int = 200,
)
