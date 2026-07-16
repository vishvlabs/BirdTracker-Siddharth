package com.example.vishv.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.SystemClock
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.example.vishv.model.FastSettings
import com.example.vishv.model.FeatureDetectionResult
import com.example.vishv.model.FrameSource
import com.example.vishv.model.StabilizationFailureReason
import com.example.vishv.model.StabilizationResult
import com.example.vishv.model.StabilizationSettings
import com.example.vishv.viewmodel.AnalysisState
import com.example.vishv.viewmodel.DisplayMode
import com.example.vishv.viewmodel.InputMode
import com.example.vishv.viewmodel.LensFacing
import com.example.vishv.viewmodel.MainViewModel
import com.example.vishv.viewmodel.VideoExtractionDiag
import com.example.vishv.viewmodel.VideoUiState
import kotlinx.coroutines.delay
import kotlin.math.roundToInt

// ─── Root ─────────────────────────────────────────────────────────────────────

@Composable
fun MainScreen(viewModel: MainViewModel) {
    val context = LocalContext.current
    val inputMode       by viewModel.inputMode.collectAsState()
    val cameraUiState   by viewModel.cameraUiState.collectAsState()
    val videoUiState    by viewModel.videoUiState.collectAsState()
    val analysisState   by viewModel.analysisState.collectAsState()
    val fastSettings    by viewModel.fastSettings.collectAsState()
    val stabSettings    by viewModel.stabSettings.collectAsState()
    val displayMode     by viewModel.displayMode.collectAsState()
    val latestDetection by viewModel.latestDetectionResult.collectAsState()
    val latestStabResult by viewModel.latestStabResult.collectAsState()
    val videoDiag       by viewModel.videoExtractionDiag.collectAsState()

    val hasCameraPermission = ContextCompat.checkSelfPermission(
        context, Manifest.permission.CAMERA
    ) == PackageManager.PERMISSION_GRANTED

    // Tick every second for session duration display.
    val sessionStartMs = remember { SystemClock.elapsedRealtime() }
    var sessionElapsedSec by remember { mutableLongStateOf(0L) }
    LaunchedEffect(Unit) {
        while (true) { delay(1_000); sessionElapsedSec = (SystemClock.elapsedRealtime() - sessionStartMs) / 1_000 }
    }

    val status = deriveStatus(
        opencvReady = viewModel.opencvReady,
        hasCameraPermission = hasCameraPermission,
        inputMode = inputMode,
        cameraRunning = cameraUiState.isRunning,
        videoUiState = videoUiState,
    )

    Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding),
        ) {
            // Source selector
            TabRow(selectedTabIndex = inputMode.ordinal) {
                Tab(
                    selected = inputMode == InputMode.CAMERA,
                    onClick = { viewModel.selectMode(InputMode.CAMERA) },
                    text = { Text("Live Camera") },
                )
                Tab(
                    selected = inputMode == InputMode.VIDEO,
                    onClick = { viewModel.selectMode(InputMode.VIDEO) },
                    text = { Text("Test Video") },
                )
            }

            // Preview — primary focus, takes all remaining height above the info panel.
            Box(modifier = Modifier.weight(1f).fillMaxWidth()) {
                when (inputMode) {
                    InputMode.CAMERA -> CameraScreen(viewModel = viewModel)
                    InputMode.VIDEO  -> VideoScreen(viewModel = viewModel)
                }
            }

            // Info panel — scrollable, capped so preview always stays visible.
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(max = 340.dp)
                    .verticalScroll(rememberScrollState()),
            ) {
                StatusStrip(
                    status = status,
                    inputMode = inputMode,
                    lensFacing = cameraUiState.lensFacing,
                    videoFileName = videoUiState.fileName,
                    analysisRateFps = analysisState.analysisRateFps,
                    sessionElapsedSec = sessionElapsedSec,
                )

                HorizontalDivider()

                VisualizationSection(
                    displayMode = displayMode,
                    onDisplayModeChange = viewModel::setDisplayMode,
                )

                HorizontalDivider()

                ObservedEntitiesPanel(
                    framesAnalyzed = analysisState.framesAnalyzed,
                    detection = latestDetection,
                    stabResult = latestStabResult,
                )

                HorizontalDivider()

                AdvancedDebugSection(
                    analysisState = analysisState,
                    fastSettings = fastSettings,
                    stabSettings = stabSettings,
                    detection = latestDetection,
                    stabResult = latestStabResult,
                    videoDiag = videoDiag,
                    onThresholdChange = viewModel::setFastThreshold,
                    onNmsToggle = viewModel::toggleNonMaxSuppression,
                    onMaxPointsChange = viewModel::setMaxFeaturePoints,
                    onRansacThresholdChange = viewModel::setRansacThreshold,
                    onMinInlierRatioChange = viewModel::setMinInlierRatio,
                )
            }
        }
    }
}

// ─── Status strip ─────────────────────────────────────────────────────────────

private fun deriveStatus(
    opencvReady: Boolean,
    hasCameraPermission: Boolean,
    inputMode: InputMode,
    cameraRunning: Boolean,
    videoUiState: VideoUiState,
): String = when {
    !opencvReady -> "OpenCV Failed"
    inputMode == InputMode.CAMERA && !hasCameraPermission -> "Camera Permission Required"
    inputMode == InputMode.CAMERA -> if (cameraRunning) "Analyzing" else "Camera Stopped"
    videoUiState.uri == null -> "No Video Selected"
    !videoUiState.isPrepared -> "Loading..."
    videoUiState.isPlaying -> "Analyzing"
    else -> "Paused"
}

@Composable
private fun StatusStrip(
    status: String,
    inputMode: InputMode,
    lensFacing: LensFacing,
    videoFileName: String?,
    analysisRateFps: Int,
    sessionElapsedSec: Long,
) {
    val isAnalyzing = status == "Analyzing"
    val isError = status == "OpenCV Failed" || status == "Camera Permission Required"

    val statusColor = when {
        isAnalyzing -> MaterialTheme.colorScheme.primary
        isError     -> MaterialTheme.colorScheme.error
        else        -> MaterialTheme.colorScheme.onSurfaceVariant
    }

    val sourceLabel = when (inputMode) {
        InputMode.CAMERA -> if (lensFacing == LensFacing.BACK) "Rear Camera" else "Front Camera"
        InputMode.VIDEO  -> videoFileName?.let { truncate(it, 22) } ?: "Test Video"
    }

    Surface(tonalElevation = 2.dp) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = if (isAnalyzing) "●" else "○",
                color = statusColor,
                style = MaterialTheme.typography.labelLarge,
            )
            Spacer(Modifier.width(6.dp))
            Text(
                text = status,
                color = statusColor,
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.SemiBold,
            )
            Spacer(Modifier.width(10.dp))
            Text("│", color = MaterialTheme.colorScheme.outlineVariant, style = MaterialTheme.typography.labelMedium)
            Spacer(Modifier.width(10.dp))
            Text(
                text = sourceLabel,
                style = MaterialTheme.typography.labelMedium,
                modifier = Modifier.weight(1f),
            )
            if (isAnalyzing) {
                Text(
                    text = "$analysisRateFps FPS",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Spacer(Modifier.width(10.dp))
            }
            Text(
                text = formatDuration(sessionElapsedSec),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

// ─── Visualization mode ───────────────────────────────────────────────────────

private data class ModeInfo(val chipLabel: String, val explanation: String)

private val MODE_INFO = mapOf(
    DisplayMode.ORIGINAL to ModeInfo(
        chipLabel = "Original",
        explanation = "Raw camera or video output with no processing applied.",
    ),
    DisplayMode.GRAYSCALE to ModeInfo(
        chipLabel = "Grayscale",
        explanation = "Grayscale conversion used as input to the feature detection stage.",
    ),
    DisplayMode.FAST_FEATURES to ModeInfo(
        chipLabel = "Feature Points",
        explanation = "Visual corners and textured regions detected in the scene. These appear on stationary background objects and are not detected birds.",
    ),
    DisplayMode.FEATURE_MATCHES to ModeInfo(
        chipLabel = "Feature Matching",
        explanation = "Lines connecting matched background features between consecutive frames. Green = reliable match, red = rejected.",
    ),
    DisplayMode.STABILIZED to ModeInfo(
        chipLabel = "Stabilized",
        explanation = "The current frame aligned to the previous frame's position, compensating for camera movement.",
    ),
    DisplayMode.DIFF_BEFORE to ModeInfo(
        chipLabel = "Difference (Raw)",
        explanation = "Pixel changes between frames before stabilization. Bright areas include all motion, including camera shake.",
    ),
    DisplayMode.DIFF_AFTER to ModeInfo(
        chipLabel = "Difference (Stabilized)",
        explanation = "Pixel changes between frames after stabilization. Camera shake has been removed — bright areas represent scene changes only.",
    ),
)

@Composable
private fun VisualizationSection(
    displayMode: DisplayMode,
    onDisplayModeChange: (DisplayMode) -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 10.dp, bottom = 6.dp),
    ) {
        Text(
            text = "VISUALIZATION",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(horizontal = 16.dp),
        )
        Spacer(Modifier.height(6.dp))

        // Horizontally scrollable chip row — one chip per display mode.
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState())
                .padding(horizontal = 12.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            DisplayMode.entries.forEach { mode ->
                val info = MODE_INFO[mode] ?: return@forEach
                FilterChip(
                    selected = mode == displayMode,
                    onClick = { onDisplayModeChange(mode) },
                    label = { Text(info.chipLabel, style = MaterialTheme.typography.labelMedium) },
                )
            }
        }

        // One-sentence explanation for the selected mode.
        val info = MODE_INFO[displayMode]
        if (info != null) {
            Surface(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 6.dp),
                color = MaterialTheme.colorScheme.surfaceVariant,
                shape = MaterialTheme.shapes.small,
            ) {
                Column(modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp)) {
                    Text(
                        text = info.chipLabel,
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.SemiBold,
                    )
                    Text(
                        text = info.explanation,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

// ─── Observed Entities ────────────────────────────────────────────────────────

@Composable
private fun ObservedEntitiesPanel(
    framesAnalyzed: Long,
    detection: FeatureDetectionResult?,
    stabResult: StabilizationResult?,
) {
    val stabLabel = stabResult?.let { stabStatusLabel(it.failureReason) }
    val stabColor: Color? = stabResult?.let { stabStatusColor(it.failureReason) }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 10.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text(
            text = "OBSERVED ENTITIES",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            EntityTile("Frames Processed", framesAnalyzed.toString(), modifier = Modifier.weight(1f))
            EntityTile("Visual Features", detection?.featureCount?.toString() ?: "–", modifier = Modifier.weight(1f))
        }
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            EntityTile("Reliable Matches", stabResult?.inlierCount?.toString() ?: "–", modifier = Modifier.weight(1f))
            EntityTile("Stabilization", stabLabel ?: "–", valueColor = stabColor, modifier = Modifier.weight(1f))
        }
    }
}

@Composable
private fun EntityTile(
    label: String,
    value: String,
    modifier: Modifier = Modifier,
    valueColor: Color? = null,
) {
    Surface(modifier = modifier, tonalElevation = 2.dp, shape = MaterialTheme.shapes.small) {
        Column(modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp)) {
            Text(
                text = label,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                text = value,
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
                color = valueColor ?: MaterialTheme.colorScheme.onSurface,
            )
        }
    }
}

@Composable
private fun stabStatusColor(reason: StabilizationFailureReason): Color = when (reason) {
    StabilizationFailureReason.NONE                         -> MaterialTheme.colorScheme.primary
    StabilizationFailureReason.FIRST_FRAME,
    StabilizationFailureReason.DUPLICATE_FRAME,
    StabilizationFailureReason.TIMESTAMP_GAP               -> MaterialTheme.colorScheme.onSurfaceVariant
    else                                                    -> MaterialTheme.colorScheme.error
}

private fun stabStatusLabel(reason: StabilizationFailureReason): String = when (reason) {
    StabilizationFailureReason.NONE                         -> "Active"
    StabilizationFailureReason.FIRST_FRAME                  -> "Computing..."
    StabilizationFailureReason.DUPLICATE_FRAME              -> "Paused"
    StabilizationFailureReason.TIMESTAMP_GAP               -> "Resetting"
    StabilizationFailureReason.TOO_FEW_FEATURES            -> "Too Few Features"
    StabilizationFailureReason.TOO_FEW_MATCHES             -> "Too Few Matches"
    StabilizationFailureReason.TOO_FEW_INLIERS,
    StabilizationFailureReason.LOW_INLIER_RATIO            -> "Low Confidence"
    StabilizationFailureReason.TRANSLATION_TOO_LARGE,
    StabilizationFailureReason.ROTATION_TOO_LARGE,
    StabilizationFailureReason.SCALE_CHANGE_TOO_LARGE      -> "Motion Too Large"
    StabilizationFailureReason.ESTIMATOR_FAILED,
    StabilizationFailureReason.INVALID_MATRIX              -> "Estimation Failed"
}

// ─── Advanced Debug ───────────────────────────────────────────────────────────

@Composable
private fun AdvancedDebugSection(
    analysisState: AnalysisState,
    fastSettings: FastSettings,
    stabSettings: StabilizationSettings,
    detection: FeatureDetectionResult?,
    stabResult: StabilizationResult?,
    videoDiag: VideoExtractionDiag?,
    onThresholdChange: (Int) -> Unit,
    onNmsToggle: () -> Unit,
    onMaxPointsChange: (Int) -> Unit,
    onRansacThresholdChange: (Double) -> Unit,
    onMinInlierRatioChange: (Float) -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }

    // Tappable header row
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { expanded = !expanded }
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = if (expanded) "▲  Advanced Debug" else "▼  Advanced Debug",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.weight(1f),
        )
    }

    AnimatedVisibility(visible = expanded) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 4.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            // ── Frame info ─────────────────────────────────────────────────────
            DebugGroup("Frame Info") {
                val meta = analysisState.lastFrameMeta
                if (meta != null) {
                    DebugRow("Source", when (meta.source) {
                        FrameSource.CAMERA_REAR  -> "Rear Camera"
                        FrameSource.CAMERA_FRONT -> "Front Camera"
                        FrameSource.VIDEO        -> "Video"
                    })
                    DebugRow("Dimensions",      "${meta.width} × ${meta.height}")
                    DebugRow("Frame Rotation",  "${meta.rotationDegrees}°")
                    DebugRow("Pipeline Time",   "${analysisState.lastProcessingTimeMs} ms")
                    DebugRow("Analyzed",        "${analysisState.framesAnalyzed}")
                    DebugRow("Skipped",         "${analysisState.framesSkipped}")
                } else {
                    DebugRow("Status", "No frames received yet")
                }
            }

            // ── FAST controls ──────────────────────────────────────────────────
            DebugGroup("Feature Point Controls (FAST)") {
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Text("Threshold: ${fastSettings.threshold}", style = MaterialTheme.typography.labelSmall, modifier = Modifier.width(110.dp))
                    Slider(
                        value = fastSettings.threshold.toFloat(),
                        onValueChange = { onThresholdChange(it.roundToInt()) },
                        valueRange = 5f..100f,
                        modifier = Modifier.weight(1f),
                    )
                }
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Text("Non-Max Suppression", style = MaterialTheme.typography.labelSmall, modifier = Modifier.weight(1f))
                    Switch(checked = fastSettings.nonMaxSuppression, onCheckedChange = { onNmsToggle() })
                }
                Text("Max Feature Points Shown", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    listOf(100, 250, 500, 1000).forEach { n ->
                        if (n == fastSettings.maxFeaturePointsShown) {
                            Button(onClick = {}, modifier = Modifier.weight(1f)) { Text("$n", style = MaterialTheme.typography.labelSmall) }
                        } else {
                            OutlinedButton(onClick = { onMaxPointsChange(n) }, modifier = Modifier.weight(1f)) { Text("$n", style = MaterialTheme.typography.labelSmall) }
                        }
                    }
                }
            }

            // ── Stabilization controls ─────────────────────────────────────────
            DebugGroup("Stabilization Controls") {
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Text("RANSAC: ${"%.1f".format(stabSettings.ransacThreshold)} px", style = MaterialTheme.typography.labelSmall, modifier = Modifier.width(110.dp))
                    Slider(
                        value = stabSettings.ransacThreshold.toFloat(),
                        onValueChange = { onRansacThresholdChange(it.toDouble()) },
                        valueRange = 0.5f..20f,
                        modifier = Modifier.weight(1f),
                    )
                }
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Text("Min Inlier Ratio: ${"%.2f".format(stabSettings.minInlierRatio)}", style = MaterialTheme.typography.labelSmall, modifier = Modifier.width(110.dp))
                    Slider(
                        value = stabSettings.minInlierRatio,
                        onValueChange = { onMinInlierRatioChange(it) },
                        valueRange = 0.05f..0.95f,
                        modifier = Modifier.weight(1f),
                    )
                }
            }

            // ── Feature detection stats ────────────────────────────────────────
            if (detection != null) {
                DebugGroup("Feature Detection Stats") {
                    DebugRow("FAST Features",    "${detection.featureCount}")
                    DebugRow("Grayscale Time",   "${detection.grayscaleTimeMs} ms")
                    DebugRow("FAST Time",        "${detection.fastTimeMs} ms")
                    DebugRow("Total Time",       "${detection.totalProcessingTimeMs} ms")
                }
            }

            // ── Stabilization stats ────────────────────────────────────────────
            if (stabResult != null) {
                DebugGroup("Stabilization Stats") {
                    DebugRow("Keypoints",        "${stabResult.featureCount}")
                    DebugRow("Descriptors",      "${stabResult.descriptorCount}")
                    DebugRow("Raw Matches",      "${stabResult.rawMatchCount}")
                    DebugRow("Filtered Matches", "${stabResult.filteredMatchCount}")
                    DebugRow("Inliers",          "${stabResult.inlierCount}")
                    DebugRow("Inlier Ratio",     "${"%.2f".format(stabResult.inlierRatio)}")
                    if (stabResult.transformSuccess) {
                        DebugRow("Translation X",    "${"%.1f".format(stabResult.translationX)} px")
                        DebugRow("Translation Y",    "${"%.1f".format(stabResult.translationY)} px")
                        DebugRow("Rotation",         "${"%.2f".format(stabResult.rotationDeg)}°")
                        DebugRow("Scale",            "${"%.3f".format(stabResult.scale)}")
                    }
                    DebugRow("Status",           stabResult.failureReason.name)
                    DebugRow("Feature Time",     "${stabResult.featureProcessingTimeMs} ms")
                    DebugRow("Matching Time",    "${stabResult.matchingTimeMs} ms")
                    DebugRow("Estimation Time",  "${stabResult.transformEstimationTimeMs} ms")
                    DebugRow("Warp Time",        "${stabResult.warpTimeMs} ms")
                    DebugRow("Total Stab Time",  "${stabResult.totalStabilizationTimeMs} ms")
                    stabResult.meanAbsDiffBefore?.let  { DebugRow("MAD Before",      "${"%.1f".format(it)}") }
                    stabResult.meanAbsDiffAfter?.let   { DebugRow("MAD After",       "${"%.1f".format(it)}") }
                    stabResult.diffReductionPct?.let   { DebugRow("Diff Reduction",  "${"%.1f".format(it)}%") }
                }
            }

            // ── Video diagnostics ──────────────────────────────────────────────
            if (videoDiag != null) {
                DebugGroup("Video Diagnostics") {
                    DebugRow("Requested Position", "${videoDiag.requestedMs} ms")
                    DebugRow("Actual Frame Time",  videoDiag.actualFrameMs?.let { "$it ms" } ?: "N/A")
                    DebugRow(
                        label = "Duplicate Frame",
                        value = if (videoDiag.identicalToPrevious) "Yes" else "No",
                        valueColor = if (videoDiag.identicalToPrevious) MaterialTheme.colorScheme.error else null,
                    )
                }
            }

            Spacer(Modifier.height(8.dp))
        }
    }
}

// ─── Debug layout helpers ─────────────────────────────────────────────────────

@Composable
private fun DebugGroup(title: String, content: @Composable () -> Unit) {
    Column(modifier = Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(3.dp)) {
        Text(
            text = title,
            style = MaterialTheme.typography.labelSmall,
            fontWeight = FontWeight.SemiBold,
            color = MaterialTheme.colorScheme.primary,
            modifier = Modifier.padding(bottom = 2.dp),
        )
        content()
        HorizontalDivider(modifier = Modifier.padding(top = 4.dp))
    }
}

@Composable
private fun DebugRow(label: String, value: String, valueColor: Color? = null) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(value, style = MaterialTheme.typography.labelSmall, color = valueColor ?: MaterialTheme.colorScheme.onSurface)
    }
}

// ─── Utilities ────────────────────────────────────────────────────────────────

private fun formatDuration(seconds: Long): String {
    val h = seconds / 3600; val m = (seconds % 3600) / 60; val s = seconds % 60
    return when {
        h > 0 -> "${h}h ${m}m"
        m > 0 -> "${m}m ${s}s"
        else  -> "${s}s"
    }
}

private fun truncate(text: String, maxLen: Int) =
    if (text.length <= maxLen) text else text.take(maxLen - 1) + "…"
