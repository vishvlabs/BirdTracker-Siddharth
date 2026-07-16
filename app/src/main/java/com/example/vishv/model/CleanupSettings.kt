package com.example.vishv.model

data class CleanupSettings(
    // Binary threshold: include shadow pixels (127) as foreground. Default false — shadows excluded.
    val includeShadows: Boolean = false,
    // Structuring element size for morphological ops. Must be odd (3, 5, 7).
    val morphKernelSize: Int = 5,
    // Erosion-then-dilation passes to remove small isolated noise.
    val openingIterations: Int = 1,
    // Dilation-then-erosion passes to fill small gaps inside regions.
    val closingIterations: Int = 2,
    // Minimum contour area in pixels. Rejects tiny blobs below this size.
    // Default 150 ≈ 0.05% of a 640×480 frame; adjust for larger analysis resolutions.
    val minContourArea: Int = 150,
    // Maximum contour area as a fraction of total frame pixels. Rejects near-full-frame blobs.
    val maxContourAreaFraction: Float = 0.25f,
    // Minimum bounding-box dimension to accept a region.
    val minWidth: Int = 8,
    val minHeight: Int = 8,
    // Minimum fill ratio (contourArea / boundingBoxArea). 0.0 disables the check.
    val minFillRatio: Float = 0.0f,
    // If cleaned foreground exceeds this fraction of the frame, emit GLOBAL_CHANGE instead of candidates.
    val maxForegroundCoveragePct: Float = 0.50f,
    // Upper bound on how many candidate regions are stored and drawn per frame.
    val maxRegionsDisplayed: Int = 30,
)
