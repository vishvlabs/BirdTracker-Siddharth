package com.example.vishv.model

data class GmmSettings(
    // Number of recent frames used to build the background distribution.
    val history: Int = 200,
    // Mahalanobis distance threshold: lower → more sensitive, higher → more conservative.
    val varThreshold: Double = 16.0,
    // When true, MOG2 emits 127 for shadow pixels and 255 for foreground. When false, only 0/255.
    val detectShadows: Boolean = true,
    // 0.0 = pass -1.0 to OpenCV (automatic adaptation); otherwise a fixed rate in (0, 1).
    val learningRate: Double = 0.0,
    // Frames to feed the model before treating its foreground mask as reliable.
    val warmUpFrames: Int = 20,
)
