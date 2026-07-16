package com.example.vishv.ui

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.OpenableColumns
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.ui.draw.rotate
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.media3.ui.PlayerView
import com.example.vishv.viewmodel.DisplayMode
import com.example.vishv.viewmodel.MainViewModel
import kotlinx.coroutines.delay

@Composable
fun VideoScreen(viewModel: MainViewModel) {
    val videoUiState by viewModel.videoUiState.collectAsState()
    val displayMode by viewModel.displayMode.collectAsState()
    val latestDetection by viewModel.latestDetectionResult.collectAsState()
    val latestStabResult by viewModel.latestStabResult.collectAsState()
    val analysisState by viewModel.analysisState.collectAsState()
    val context = LocalContext.current

    // SAF file picker — no storage permission needed.
    val pickVideoLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        uri ?: return@rememberLauncherForActivityResult
        // Take a persistable grant so the URI stays valid if the app is backgrounded.
        try {
            context.contentResolver.takePersistableUriPermission(
                uri, Intent.FLAG_GRANT_READ_URI_PERMISSION
            )
        } catch (_: SecurityException) {
            // Some providers do not support persistable grants; session access is sufficient.
        }
        viewModel.selectVideo(uri, resolveFileName(context, uri))
    }

    val playerView = remember {
        PlayerView(context).apply {
            useController = false
            setBackgroundColor(android.graphics.Color.BLACK)
        }
    }

    // Attach the player to the view on entry; detach on exit to avoid leaks.
    DisposableEffect(Unit) {
        playerView.player = viewModel.player
        onDispose { playerView.player = null }
    }

    // Poll playback position while playing; snapshot it once when paused.
    var positionMs by remember { mutableLongStateOf(0L) }
    val isPlaying = videoUiState.isPlaying
    val isPrepared = videoUiState.isPrepared

    // Reset position display when a new file is selected.
    LaunchedEffect(videoUiState.uri) { positionMs = 0L }

    LaunchedEffect(isPlaying) {
        if (isPlaying) {
            while (true) {
                positionMs = viewModel.player.currentPosition.coerceAtLeast(0L)
                delay(250)
            }
        } else if (isPrepared) {
            positionMs = viewModel.player.currentPosition.coerceAtLeast(0L)
        }
    }

    Column(modifier = Modifier.fillMaxSize()) {

        // Video preview area
        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .background(Color.Black)
        ) {
            AndroidView(
                factory = { playerView },
                modifier = Modifier.fillMaxSize()
            )

            // Debug overlay: shown on top of the PlayerView when a processed bitmap exists.
            // GRAYSCALE/FAST_FEATURES bitmaps come from FastFeatureProcessor (latestDetection).
            // FEATURE_MATCHES/STABILIZED/DIFF_BEFORE/DIFF_AFTER come from StabilizationProcessor.
            val debugBitmap = when (displayMode) {
                DisplayMode.GRAYSCALE, DisplayMode.FAST_FEATURES -> latestDetection?.debugBitmap
                DisplayMode.FEATURE_MATCHES, DisplayMode.STABILIZED,
                DisplayMode.DIFF_BEFORE, DisplayMode.DIFF_AFTER -> latestStabResult?.debugBitmap
                else -> null
            }
            // videoRotationDegrees (from METADATA_KEY_VIDEO_ROTATION) is carried through as
            // rotationDegrees on every AnalysisFrame, so the same value that oriented the
            // pipeline also rotates the overlay to match the PlayerView's display.
            val overlayRotation = analysisState.lastFrameMeta?.rotationDegrees?.toFloat() ?: 0f
            if (displayMode != DisplayMode.ORIGINAL && debugBitmap != null && !debugBitmap.isRecycled) {
                Image(
                    bitmap = debugBitmap.asImageBitmap(),
                    contentDescription = null,
                    modifier = Modifier
                        .fillMaxSize()
                        .rotate(overlayRotation),
                    contentScale = ContentScale.Fit,
                )
            }

            if (!isPrepared) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "Tap \"Select Video\" to load a file",
                        color = Color.White.copy(alpha = 0.7f),
                        style = MaterialTheme.typography.bodyMedium,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(24.dp)
                    )
                }
            }
        }

        // Metadata row
        Surface(tonalElevation = 1.dp) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 8.dp)
            ) {
                Text(
                    text = videoUiState.fileName ?: "No video selected",
                    style = MaterialTheme.typography.bodyMedium,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                Text(
                    text = "${formatMs(positionMs)}  /  ${formatMs(videoUiState.durationMs)}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }

        // Playback controls
        Surface(tonalElevation = 3.dp) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp)
            ) {
                Button(
                    onClick = { pickVideoLauncher.launch(arrayOf("video/*")) },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Select Video")
                }

                Spacer(modifier = Modifier.height(8.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Button(
                        onClick = {
                            if (isPlaying) viewModel.pauseVideo() else viewModel.playVideo()
                        },
                        enabled = isPrepared,
                        modifier = Modifier.weight(1f)
                    ) {
                        Text(if (isPlaying) "Pause" else "Play")
                    }

                    OutlinedButton(
                        onClick = { viewModel.restartVideo() },
                        enabled = isPrepared,
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Restart")
                    }

                    OutlinedButton(
                        onClick = { viewModel.stopVideo() },
                        enabled = isPrepared,
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Stop")
                    }
                }
            }
        }
    }
}

private fun resolveFileName(context: Context, uri: Uri): String {
    return context.contentResolver.query(
        uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null
    )?.use { cursor ->
        if (cursor.moveToFirst()) {
            val idx = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (idx >= 0) cursor.getString(idx) else null
        } else null
    } ?: uri.lastPathSegment ?: "video"
}

private fun formatMs(ms: Long): String {
    val totalSec = ms / 1000
    val h = totalSec / 3600
    val m = (totalSec % 3600) / 60
    val s = totalSec % 60
    return if (h > 0) "%d:%02d:%02d".format(h, m, s) else "%02d:%02d".format(m, s)
}
