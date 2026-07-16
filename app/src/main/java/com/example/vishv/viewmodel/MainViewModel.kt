package com.example.vishv.viewmodel

import android.app.Application
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.SystemClock
import androidx.camera.core.ImageProxy
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.media3.common.MediaItem
import androidx.media3.common.Player
import androidx.media3.exoplayer.ExoPlayer
import com.example.vishv.cv.FastFeatureProcessor
import com.example.vishv.cv.GmmForegroundProcessor
import com.example.vishv.cv.StabilizationProcessor
import com.example.vishv.model.AnalysisFrame
import com.example.vishv.model.FastSettings
import com.example.vishv.model.FeatureDetectionResult
import com.example.vishv.model.ForegroundDetectionResult
import com.example.vishv.model.FrameMetadata
import com.example.vishv.model.FrameSource
import com.example.vishv.model.GmmSettings
import com.example.vishv.model.StabilizationFailureReason
import com.example.vishv.model.StabilizationResult
import com.example.vishv.model.StabilizationSettings
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader

enum class LensFacing { BACK, FRONT }
enum class InputMode { CAMERA, VIDEO }
enum class DisplayMode {
    ORIGINAL,
    GRAYSCALE,
    FAST_FEATURES,
    FEATURE_MATCHES,
    STABILIZED,
    DIFF_BEFORE,
    DIFF_AFTER,
    FOREGROUND_MASK,
}

data class CameraUiState(
    val lensFacing: LensFacing = LensFacing.BACK,
    val isRunning: Boolean = false,
)

data class VideoUiState(
    val uri: Uri? = null,
    val fileName: String? = null,
    val durationMs: Long = 0L,
    val isPlaying: Boolean = false,
    val isPrepared: Boolean = false,
)

data class AnalysisState(
    val cameraFramesReceived: Long = 0L,
    val videoFramesReceived: Long = 0L,
    val framesAnalyzed: Long = 0L,
    /** Frames intentionally rejected by the application's rate limiter. */
    val framesSkipped: Long = 0L,
    val lastProcessingTimeMs: Long = 0L,
    val lastFrameMeta: FrameMetadata? = null,
    val analysisRateFps: Int = 5,
)

/**
 * Diagnostic for a single video frame retrieval. Useful for verifying that OPTION_CLOSEST
 * returns distinct frames rather than repeating the same keyframe.
 *
 * actualFrameMs: null when the API level is below 30 or the decoder did not report a frame time.
 */
data class VideoExtractionDiag(
    val requestedMs: Long,
    val actualFrameMs: Long?,
    val identicalToPrevious: Boolean,
)

class MainViewModel(application: Application) : AndroidViewModel(application) {

    // --- OpenCV ---
    // initLocal() loads the bundled libopencv_java4.so from the APK (Maven Central distribution).
    val opencvReady: Boolean = OpenCVLoader.initLocal()
    private val fastProcessor = FastFeatureProcessor()
    private val stabProcessor = StabilizationProcessor()
    private val gmmProcessor = GmmForegroundProcessor()

    // --- Input mode ---
    private val _inputMode = MutableStateFlow(InputMode.CAMERA)
    val inputMode: StateFlow<InputMode> = _inputMode.asStateFlow()

    // --- Camera state ---
    private val _cameraUiState = MutableStateFlow(CameraUiState())
    val cameraUiState: StateFlow<CameraUiState> = _cameraUiState.asStateFlow()

    // --- Video state ---
    private val _videoUiState = MutableStateFlow(VideoUiState())
    val videoUiState: StateFlow<VideoUiState> = _videoUiState.asStateFlow()

    // --- Analysis counters ---
    private val _analysisState = MutableStateFlow(AnalysisState())
    val analysisState: StateFlow<AnalysisState> = _analysisState.asStateFlow()

    // --- FAST settings ---
    private val _fastSettings = MutableStateFlow(FastSettings())
    val fastSettings: StateFlow<FastSettings> = _fastSettings.asStateFlow()

    // --- Stabilization settings ---
    private val _stabSettings = MutableStateFlow(StabilizationSettings())
    val stabSettings: StateFlow<StabilizationSettings> = _stabSettings.asStateFlow()

    // --- GMM settings ---
    private val _gmmSettings = MutableStateFlow(GmmSettings())
    val gmmSettings: StateFlow<GmmSettings> = _gmmSettings.asStateFlow()

    // --- Display mode (controls which debug bitmap is rendered) ---
    private val _displayMode = MutableStateFlow(DisplayMode.ORIGINAL)
    val displayMode: StateFlow<DisplayMode> = _displayMode.asStateFlow()

    // --- Latest FAST detection result ---
    private val _latestDetectionResult = MutableStateFlow<FeatureDetectionResult?>(null)
    val latestDetectionResult: StateFlow<FeatureDetectionResult?> = _latestDetectionResult.asStateFlow()

    // --- Latest stabilization result ---
    private val _latestStabResult = MutableStateFlow<StabilizationResult?>(null)
    val latestStabResult: StateFlow<StabilizationResult?> = _latestStabResult.asStateFlow()

    // --- Latest foreground detection result ---
    private val _latestFgResult = MutableStateFlow<ForegroundDetectionResult?>(null)
    val latestFgResult: StateFlow<ForegroundDetectionResult?> = _latestFgResult.asStateFlow()

    // --- Video frame extraction diagnostic ---
    private val _videoExtractionDiag = MutableStateFlow<VideoExtractionDiag?>(null)
    val videoExtractionDiag: StateFlow<VideoExtractionDiag?> = _videoExtractionDiag.asStateFlow()

    // Rate limiter for CameraX: tracks the elapsedRealtime of the last analyzed camera frame.
    @Volatile private var lastCameraFrameAnalyzedMs = 0L

    // Cheap per-frame hash for detecting repeated video frames. Reset on each new extraction session.
    @Volatile private var prevVideoFrameHash = Int.MIN_VALUE

    // Rotation embedded in the video container (from METADATA_KEY_VIDEO_ROTATION).
    // MediaMetadataRetriever.getFrameAtTime() returns raw encoded pixels without applying this
    // rotation, so we carry it forward as rotationDegrees on every AnalysisFrame from video.
    @Volatile private var videoRotationDegrees: Int = 0

    private var videoExtractionJob: Job? = null
    private var retriever: MediaMetadataRetriever? = null

    // ExoPlayer lives for the ViewModel lifetime and is released in onCleared().
    val player: ExoPlayer = ExoPlayer.Builder(application).build().also { exoPlayer ->
        exoPlayer.addListener(object : Player.Listener {
            override fun onIsPlayingChanged(isPlaying: Boolean) {
                _videoUiState.update { it.copy(isPlaying = isPlaying) }
                if (isPlaying) startVideoExtraction() else stopVideoExtraction()
            }

            override fun onPlaybackStateChanged(playbackState: Int) {
                when (playbackState) {
                    Player.STATE_READY -> _videoUiState.update { state ->
                        state.copy(
                            durationMs = exoPlayer.duration.coerceAtLeast(0L),
                            isPrepared = true,
                        )
                    }
                    Player.STATE_IDLE -> _videoUiState.update { it.copy(isPrepared = false) }
                    else -> Unit
                }
            }
        })
    }

    // --- Mode switching ---

    fun selectMode(mode: InputMode) {
        if (_inputMode.value == mode) return
        when (mode) {
            InputMode.CAMERA -> {
                player.pause()
                stopVideoExtraction()
                lastCameraFrameAnalyzedMs = 0L
            }
            InputMode.VIDEO -> {
                lastCameraFrameAnalyzedMs = 0L
            }
        }
        _inputMode.value = mode
        stabProcessor.resetAll()
        gmmProcessor.resetAll()
        clearStaleResults()
    }

    // --- Camera actions ---

    fun startCamera() { _cameraUiState.update { it.copy(isRunning = true) } }

    fun stopCamera() { _cameraUiState.update { it.copy(isRunning = false) } }

    fun switchLens() {
        _cameraUiState.update { state ->
            val next = if (state.lensFacing == LensFacing.BACK) LensFacing.FRONT else LensFacing.BACK
            state.copy(lensFacing = next)
        }
        lastCameraFrameAnalyzedMs = 0L
        stabProcessor.resetSource(FrameSource.CAMERA_REAR)
        stabProcessor.resetSource(FrameSource.CAMERA_FRONT)
        gmmProcessor.resetSource(FrameSource.CAMERA_REAR)
        gmmProcessor.resetSource(FrameSource.CAMERA_FRONT)
        clearStaleResults()
    }

    /**
     * Entry point for all CameraX frames. Called from the ImageAnalysis executor thread.
     * Applies the configured rate limit, converts the proxy, and forwards to consumeFrame().
     * Always closes the ImageProxy before returning.
     */
    fun onCameraFrameReceived(imageProxy: ImageProxy, source: FrameSource) {
        val now = SystemClock.elapsedRealtime()
        _analysisState.update { it.copy(cameraFramesReceived = it.cameraFramesReceived + 1) }

        val targetIntervalMs = 1000L / _analysisState.value.analysisRateFps
        if (now - lastCameraFrameAnalyzedMs < targetIntervalMs) {
            _analysisState.update { it.copy(framesSkipped = it.framesSkipped + 1) }
            imageProxy.close()
            return
        }
        lastCameraFrameAnalyzedMs = now

        val bitmap = imageProxy.toBitmap()
        val width = imageProxy.width
        val height = imageProxy.height
        // CameraX frame rotation is discrete metadata — normally 0, 90, 180, or 270 degrees.
        // It tells consumers how many degrees to rotate the sensor frame CW to make it upright
        // relative to the use case's targetRotation. The pixel buffer itself is never rotated.
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val timestampMs = imageProxy.imageInfo.timestamp / 1_000_000L
        imageProxy.close()

        consumeFrame(
            AnalysisFrame(bitmap, width, height, rotationDegrees, timestampMs, source),
            now,
            isDuplicate = false,
        )
    }

    // --- Video actions ---

    fun selectVideo(uri: Uri, fileName: String) {
        stopVideoExtraction()
        releaseRetriever()
        _videoUiState.update { VideoUiState(uri = uri, fileName = fileName) }
        stabProcessor.resetSource(FrameSource.VIDEO)
        gmmProcessor.resetSource(FrameSource.VIDEO)
        clearStaleResults()
        player.stop()
        player.clearMediaItems()
        player.setMediaItem(MediaItem.fromUri(uri))
        player.prepare()
        retriever = MediaMetadataRetriever().apply { setDataSource(getApplication(), uri) }
        videoRotationDegrees = retriever
            ?.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)
            ?.toIntOrNull() ?: 0
    }

    fun playVideo() { player.play() }

    fun pauseVideo() { player.pause() }

    fun restartVideo() {
        stabProcessor.resetSource(FrameSource.VIDEO)
        gmmProcessor.resetSource(FrameSource.VIDEO)
        player.seekTo(0L)
        player.play()
    }

    fun stopVideo() {
        stabProcessor.resetSource(FrameSource.VIDEO)
        gmmProcessor.resetSource(FrameSource.VIDEO)
        player.pause()
        player.seekTo(0L)
    }

    // --- Display mode ---

    fun setDisplayMode(mode: DisplayMode) {
        _displayMode.value = mode
        clearStaleResults()
    }

    // --- FAST settings ---

    fun setFastThreshold(threshold: Int) {
        val clamped = threshold.coerceIn(5, 100)
        _fastSettings.update { it.copy(threshold = clamped) }
        stabProcessor.updateFastSettings(clamped, _fastSettings.value.nonMaxSuppression)
    }

    fun toggleNonMaxSuppression() {
        val newNms = !_fastSettings.value.nonMaxSuppression
        _fastSettings.update { it.copy(nonMaxSuppression = newNms) }
        stabProcessor.updateFastSettings(_fastSettings.value.threshold, newNms)
    }

    fun setMaxFeaturePoints(count: Int) {
        _fastSettings.update { it.copy(maxFeaturePointsShown = count) }
    }

    // --- Stabilization settings ---

    fun setRansacThreshold(threshold: Double) {
        _stabSettings.update { it.copy(ransacThreshold = threshold.coerceIn(0.5, 20.0)) }
    }

    fun setMinInlierRatio(ratio: Float) {
        _stabSettings.update { it.copy(minInlierRatio = ratio.coerceIn(0.05f, 0.95f)) }
    }

    // --- GMM settings ---

    fun setGmmHistory(history: Int) {
        val newSettings = _gmmSettings.value.copy(history = history.coerceIn(50, 500))
        _gmmSettings.value = newSettings
        gmmProcessor.updateSettings(newSettings)
    }

    fun setGmmVarThreshold(threshold: Double) {
        val newSettings = _gmmSettings.value.copy(varThreshold = threshold.coerceIn(4.0, 64.0))
        _gmmSettings.value = newSettings
        gmmProcessor.updateSettings(newSettings)
    }

    fun setGmmLearningRate(rate: Double) {
        val newSettings = _gmmSettings.value.copy(learningRate = rate.coerceIn(0.0, 0.5))
        _gmmSettings.value = newSettings
        gmmProcessor.updateSettings(newSettings)
    }

    fun toggleGmmShadowDetection() {
        val newSettings = _gmmSettings.value.copy(detectShadows = !_gmmSettings.value.detectShadows)
        _gmmSettings.value = newSettings
        gmmProcessor.updateSettings(newSettings)
    }

    fun setGmmWarmUpFrames(frames: Int) {
        val newSettings = _gmmSettings.value.copy(warmUpFrames = frames.coerceIn(5, 100))
        _gmmSettings.value = newSettings
        gmmProcessor.updateSettings(newSettings)
    }

    // --- Shared frame consumer ---

    /**
     * Common processing entry for every frame regardless of source.
     * Runs FAST detection and stabilization, then updates all state.
     * Recycles the input bitmap when done.
     *
     * Caller must not touch frame.bitmap after this call returns.
     *
     * @param isDuplicate true when the video extraction loop detected this frame is identical to
     *   the previous one (hash check). StabilizationProcessor will not update per-source state for
     *   duplicate frames, preserving continuity for the next unique frame.
     */
    private fun consumeFrame(frame: AnalysisFrame, receivedAtMs: Long, isDuplicate: Boolean) {
        val settings = _fastSettings.value
        val sSettings = _stabSettings.value
        val mode = _displayMode.value

        val fastResult: FeatureDetectionResult? = if (opencvReady) {
            try {
                fastProcessor.process(
                    frame,
                    settings,
                    renderGrayscale = mode == DisplayMode.GRAYSCALE,
                    renderKeypoints = mode == DisplayMode.FAST_FEATURES,
                )
            } catch (e: Exception) {
                null
            }
        } else null

        val stabResult: StabilizationResult? = if (opencvReady) {
            try {
                stabProcessor.process(frame, settings, sSettings, isDuplicate, mode)
            } catch (e: Exception) {
                null
            }
        } else null

        // A timestamp gap means a video seek; reset the background model so the next frames
        // start fresh rather than contaminating the model with a sudden scene change.
        if (stabResult?.failureReason == StabilizationFailureReason.TIMESTAMP_GAP) {
            gmmProcessor.resetSource(frame.source)
        }

        val fgResult: ForegroundDetectionResult? = if (opencvReady && stabResult != null) {
            try {
                gmmProcessor.process(frame.source, frame.timestampMs, stabResult, isDuplicate, mode)
            } catch (e: Exception) {
                null
            }
        } else null

        val processingTimeMs = SystemClock.elapsedRealtime() - receivedAtMs

        _analysisState.update { state ->
            state.copy(
                framesAnalyzed = state.framesAnalyzed + 1,
                lastProcessingTimeMs = (stabResult?.totalStabilizationTimeMs
                    ?: fastResult?.totalProcessingTimeMs
                    ?: processingTimeMs),
                lastFrameMeta = FrameMetadata(
                    width = frame.width,
                    height = frame.height,
                    rotationDegrees = frame.rotationDegrees,
                    timestampMs = frame.timestampMs,
                    source = frame.source,
                ),
            )
        }

        if (fastResult != null) _latestDetectionResult.value = fastResult
        if (stabResult != null) _latestStabResult.value = stabResult
        if (fgResult != null) _latestFgResult.value = fgResult

        frame.bitmap.recycle()
    }

    // --- Video frame extraction ---

    /**
     * Runs on Dispatchers.IO. Uses MediaMetadataRetriever.getFrameAtTime with OPTION_CLOSEST
     * (not OPTION_CLOSEST_SYNC) so the retriever returns the frame nearest to the requested
     * timestamp rather than snapping to the closest keyframe only.
     *
     * OPTION_CLOSEST requires decoding from the prior keyframe forward, which is slower than
     * OPTION_CLOSEST_SYNC on high-bitrate content. If decode time exceeds the analysis budget
     * (1000 / rateFps ms) the effective rate drops below the configured FPS — this is expected
     * and safe; the delay() call is simply skipped.
     *
     * Seek detection: if the player position jumps backward (user tapped restart) or forward by
     * more than MAX_TIMESTAMP_GAP_MS, StabilizationProcessor's timestamp-gap guard handles it
     * internally — no separate seek flag is needed here.
     */
    private fun startVideoExtraction() {
        videoExtractionJob?.cancel()
        prevVideoFrameHash = Int.MIN_VALUE
        stabProcessor.resetSource(FrameSource.VIDEO)
        gmmProcessor.resetSource(FrameSource.VIDEO)
        videoExtractionJob = viewModelScope.launch(Dispatchers.IO) {
            while (isActive) {
                val startMs = SystemClock.elapsedRealtime()
                val targetIntervalMs = 1000L / _analysisState.value.analysisRateFps

                val positionMs = withContext(Dispatchers.Main) {
                    if (player.isPlaying) player.currentPosition else -1L
                }
                if (positionMs < 0L) break

                val r = retriever ?: break
                val (bitmap, actualFrameMs) = try {
                    retrieveVideoFrame(r, positionMs * 1_000L)
                } catch (e: Exception) {
                    break  // Retriever released mid-decode (selectVideo called concurrently).
                }

                if (bitmap != null) {
                    val hash = cheapBitmapHash(bitmap)
                    val identical = (hash == prevVideoFrameHash)
                    prevVideoFrameHash = hash

                    _videoExtractionDiag.value = VideoExtractionDiag(
                        requestedMs = positionMs,
                        actualFrameMs = actualFrameMs,
                        identicalToPrevious = identical,
                    )

                    _analysisState.update { it.copy(videoFramesReceived = it.videoFramesReceived + 1) }
                    consumeFrame(
                        AnalysisFrame(
                            bitmap = bitmap,
                            width = bitmap.width,
                            height = bitmap.height,
                            rotationDegrees = videoRotationDegrees,
                            timestampMs = actualFrameMs ?: positionMs,
                            source = FrameSource.VIDEO,
                        ),
                        startMs,
                        isDuplicate = identical,
                    )
                }

                val elapsed = SystemClock.elapsedRealtime() - startMs
                val remaining = targetIntervalMs - elapsed
                if (remaining > 0L) delay(remaining)
            }
        }
    }

    /**
     * MediaMetadataRetriever does not report the actual decoded frame timestamp — BitmapParams
     * (API 30) only exposes the pixel config, not timing. The second element of the pair is
     * therefore always null. The diagnostic uses the requested position as the best available
     * timestamp, and the identical-frame hash to detect when the retriever returned the same
     * keyframe for consecutive requests.
     */
    private fun retrieveVideoFrame(
        retriever: MediaMetadataRetriever,
        positionUs: Long,
    ): Pair<Bitmap?, Long?> {
        return Pair(
            retriever.getFrameAtTime(positionUs, MediaMetadataRetriever.OPTION_CLOSEST),
            null,
        )
    }

    /**
     * Samples 16 evenly-spaced pixels along the centre row.
     * Cheap O(1) indicator of whether two consecutive frames are identical.
     * False negatives (different frames with the same hash) are acceptable for a debug diagnostic.
     */
    private fun cheapBitmapHash(bitmap: Bitmap): Int {
        val y = bitmap.height / 2
        val step = maxOf(1, bitmap.width / 16)
        var hash = 1
        var x = 0
        while (x < bitmap.width) {
            hash = 31 * hash + bitmap.getPixel(x, y)
            x += step
        }
        return hash
    }

    private fun stopVideoExtraction() {
        videoExtractionJob?.cancel()
        videoExtractionJob = null
        _videoExtractionDiag.value = null
    }

    private fun clearStaleResults() {
        _latestDetectionResult.value = null
        _latestStabResult.value = null
        _latestFgResult.value = null
    }

    private fun releaseRetriever() {
        try { retriever?.release() } catch (e: Exception) { /* already released */ }
        retriever = null
    }

    override fun onCleared() {
        stopVideoExtraction()
        releaseRetriever()
        stabProcessor.release()
        player.release()
        super.onCleared()
    }
}
