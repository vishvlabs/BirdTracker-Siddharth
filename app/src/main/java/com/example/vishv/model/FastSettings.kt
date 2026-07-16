package com.example.vishv.model

data class FastSettings(
    val threshold: Int = 20,
    val nonMaxSuppression: Boolean = true,
    val maxFeaturePointsShown: Int = 500,
)
