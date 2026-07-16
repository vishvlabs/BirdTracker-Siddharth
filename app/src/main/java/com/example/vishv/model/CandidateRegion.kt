package com.example.vishv.model

data class CandidateRegion(
    val id: Int,                // Frame-local label only — not a persistent tracking ID.
    val left: Int,
    val top: Int,
    val right: Int,
    val bottom: Int,
    val width: Int,
    val height: Int,
    val centroidX: Float,
    val centroidY: Float,
    val contourArea: Float,
    val boundingBoxArea: Int,
    val aspectRatio: Float,     // width / height
    val fillRatio: Float,       // contourArea / boundingBoxArea
)
