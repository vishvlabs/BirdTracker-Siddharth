package com.example.vishv.ui

import android.Manifest
import android.content.pm.PackageManager
import android.view.Surface
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
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
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.example.vishv.model.FrameSource
import com.example.vishv.viewmodel.DisplayMode
import com.example.vishv.viewmodel.LensFacing
import com.example.vishv.viewmodel.MainViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

@Composable
fun CameraScreen(viewModel: MainViewModel) {
    val uiState by viewModel.cameraUiState.collectAsState()
    val displayMode by viewModel.displayMode.collectAsState()
    val latestDetection by viewModel.latestDetectionResult.collectAsState()
    val latestStabResult by viewModel.latestStabResult.collectAsState()
    val latestFgResult by viewModel.latestFgResult.collectAsState()
    val analysisState by viewModel.analysisState.collectAsState()
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    // Display rotation drives targetRotation on both CameraX use cases so that
    // imageProxy.imageInfo.rotationDegrees is always the correct value to make the
    // sensor frame upright relative to the current display orientation.
    val displayRotation = LocalView.current.display?.rotation ?: Surface.ROTATION_0

    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> hasCameraPermission = granted }

    if (!hasCameraPermission) {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Text(
                    text = "Camera permission is required.",
                    style = MaterialTheme.typography.bodyMedium
                )
                Button(onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) }) {
                    Text("Grant Permission")
                }
            }
        }
        return
    }

    val previewView = remember {
        PreviewView(context).apply {
            // COMPATIBLE uses TextureView, which composites correctly inside Compose.
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }

    // Store the resolved provider in Compose state so DisposableEffect reads the live value.
    var cameraProvider: ProcessCameraProvider? by remember { mutableStateOf(null) }

    // Single background thread for image analysis — keeps the main thread free.
    val analysisExecutor = remember { Executors.newSingleThreadExecutor() }

    // Resolve provider once on entry.
    LaunchedEffect(Unit) {
        cameraProvider = try {
            withContext(Dispatchers.IO) { cameraProviderFuture.get() }
        } catch (e: Exception) {
            null
        }
    }

    // Release camera and shut down executor when this composable leaves composition
    // (e.g., the user switches to Test Video mode).
    DisposableEffect(Unit) {
        onDispose {
            cameraProvider?.unbindAll()
            analysisExecutor.shutdown()
        }
    }

    // Rebind whenever provider, running flag, lens selection, or display rotation changes.
    // Including displayRotation ensures targetRotation is updated after a physical rotation,
    // keeping imageProxy.imageInfo.rotationDegrees correct for the current display orientation.
    LaunchedEffect(cameraProvider, uiState.isRunning, uiState.lensFacing, displayRotation) {
        val provider = cameraProvider ?: return@LaunchedEffect
        provider.unbindAll()
        if (!uiState.isRunning) return@LaunchedEffect

        val source = if (uiState.lensFacing == LensFacing.BACK)
            FrameSource.CAMERA_REAR else FrameSource.CAMERA_FRONT

        // targetRotation tells CameraX which display orientation to produce rotationDegrees for.
        val preview = Preview.Builder()
            .setTargetRotation(displayRotation)
            .build().also {
                it.surfaceProvider = previewView.surfaceProvider
            }

        // STRATEGY_KEEP_ONLY_LATEST drops frames automatically when the analyzer is busy,
        // preventing frame queue build-up. RGBA_8888 avoids a YUV→Bitmap conversion step.
        val imageAnalysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setTargetRotation(displayRotation)
            .build()
            .also { analysis ->
                analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                    // ViewModel handles rate-limiting and always closes imageProxy.
                    viewModel.onCameraFrameReceived(imageProxy, source)
                }
            }

        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(
                if (uiState.lensFacing == LensFacing.BACK)
                    CameraSelector.LENS_FACING_BACK
                else
                    CameraSelector.LENS_FACING_FRONT
            )
            .build()
        try {
            provider.bindToLifecycle(lifecycleOwner, cameraSelector, preview, imageAnalysis)
        } catch (e: Exception) {
            // Selected lens unavailable on this device.
        }
    }

    // No Scaffold here — MainScreen owns the single Scaffold.
    Column(modifier = Modifier.fillMaxSize()) {
        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
        ) {
            AndroidView(
                factory = { previewView },
                modifier = Modifier.fillMaxSize()
            )

            if (!uiState.isRunning) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(Color.Black),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "Press Start to begin",
                        color = Color.White,
                        style = MaterialTheme.typography.bodyLarge
                    )
                }
            } else {
                // Debug overlay: shown on top of the live preview when a processed bitmap exists.
                // GRAYSCALE/FAST_FEATURES bitmaps come from FastFeatureProcessor (latestDetection).
                // FEATURE_MATCHES/STABILIZED/DIFF_BEFORE/DIFF_AFTER come from StabilizationProcessor.
                val debugBitmap = when (displayMode) {
                    DisplayMode.GRAYSCALE, DisplayMode.FAST_FEATURES -> latestDetection?.debugBitmap
                    DisplayMode.FEATURE_MATCHES, DisplayMode.STABILIZED,
                    DisplayMode.DIFF_BEFORE, DisplayMode.DIFF_AFTER -> latestStabResult?.debugBitmap
                    DisplayMode.FOREGROUND_MASK -> latestFgResult?.maskBitmap
                    else -> null
                }
                // rotationDegrees is the CW rotation needed to make the raw sensor buffer upright
                // for the current display orientation. Applying it to the overlay aligns it with
                // the PreviewView which handles its own rotation internally.
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

                Surface(
                    modifier = Modifier
                        .align(Alignment.TopStart)
                        .padding(12.dp),
                    color = Color.Black.copy(alpha = 0.55f),
                    shape = MaterialTheme.shapes.small
                ) {
                    Text(
                        text = if (uiState.lensFacing == LensFacing.BACK) "Rear Camera"
                               else "Front Camera",
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
                        color = Color.White,
                        style = MaterialTheme.typography.labelLarge
                    )
                }
            }
        }

        Surface(tonalElevation = 3.dp) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Button(
                    onClick = {
                        if (uiState.isRunning) viewModel.stopCamera()
                        else viewModel.startCamera()
                    },
                    modifier = Modifier.weight(1f)
                ) {
                    Text(if (uiState.isRunning) "Stop" else "Start")
                }

                OutlinedButton(
                    onClick = { viewModel.switchLens() },
                    modifier = Modifier.weight(1f)
                ) {
                    Text(
                        if (uiState.lensFacing == LensFacing.BACK) "Switch to Front"
                        else "Switch to Rear"
                    )
                }
            }
        }
    }
}
