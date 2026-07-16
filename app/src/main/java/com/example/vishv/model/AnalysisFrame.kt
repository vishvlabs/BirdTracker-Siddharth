package com.example.vishv.model

import android.graphics.Bitmap

enum class FrameSource { CAMERA_REAR, CAMERA_FRONT, VIDEO }

/**
 * Lightweight metadata snapshot stored in UI state.
 * Does not hold a Bitmap so the state is cheap to copy and diff.
 */
data class FrameMetadata(
    val width: Int,
    val height: Int,
    val rotationDegrees: Int,
    val timestampMs: Long,
    val source: FrameSource,
)

/**
 * Full frame passed through the analysis pipeline.
 * Callers must recycle the bitmap when they are done with it.
 */
data class AnalysisFrame(
    val bitmap: Bitmap,
    val width: Int,
    val height: Int,
    val rotationDegrees: Int,
    val timestampMs: Long,
    val source: FrameSource,
)
