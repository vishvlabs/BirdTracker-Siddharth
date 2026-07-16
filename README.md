# VishV

Agricultural bird-deterrent computer-vision app for Android. Detects candidate birds by tracking independently-moving objects against a stabilized background using classical CV — no neural networks, no cloud services.

## Requirements

- Android Studio Ladybug or later
- Android device with camera, minSdk 24 (Android 7.0)
- compileSdk / targetSdk 37

## Build

```bash
./gradlew assembleDebug
# APK: app/build/outputs/apk/debug/app-debug.apk
```

Install directly:

```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

## Key dependencies

| Library | Version |
|---|---|
| Jetpack Compose BOM | 2024.09.00 |
| CameraX | 1.5.0 |
| Media3 ExoPlayer | 1.5.0 |
| OpenCV (Maven Central) | 4.10.0 |
| Lifecycle / ViewModel | 2.10.0 |
| Coroutines | 1.9.0 |

OpenCV is loaded from the bundled AAR via `OpenCVLoader.initLocal()` — no OpenCV Manager app required.

## Input modes

**Live Camera** — CameraX `ImageAnalysis` with `STRATEGY_KEEP_ONLY_LATEST` and `OUTPUT_IMAGE_FORMAT_RGBA_8888`. Both rear and front lenses supported. Frame rotation tracks the display orientation so `imageProxy.imageInfo.rotationDegrees` is always correct for the current physical orientation.

**Test Video** — SAF file picker (no storage permission). Media3 ExoPlayer for playback; `MediaMetadataRetriever.getFrameAtTime(OPTION_CLOSEST)` extracts frames at the analysis rate. Both sources feed the same shared pipeline.

## Analysis pipeline (runs at 5 FPS, same code path for camera and video)

1. `Bitmap` → RGBA `Mat` → grayscale `Mat`
2. FAST corner detection (`FastFeatureDetector`)
3. ORB descriptor computation on top-K FAST keypoints
4. BFMatcher cross-check matching against the previous frame from the same source
5. `estimateAffinePartial2D` (RANSAC) → 4-DOF partial affine transform
6. `warpAffine` — aligns current frame to previous frame coordinate system
7. `absdiff` before/after stabilization → mean absolute difference diagnostics

## Display modes

| Mode | Source | Shows |
|---|---|---|
| Original | — | Raw camera / video, no overlay |
| Grayscale | FAST | Grayscale frame |
| FAST Features | FAST | FAST keypoints (green circles) |
| Matches | Stabilization | Side-by-side previous (gray) / current (color) with green inlier and red outlier match lines |
| Stabilized | Stabilization | Current frame warped to align with previous |
| Diff-B | Stabilization | Amplified pixel difference before stabilization |
| Diff-A | Stabilization | Amplified pixel difference after stabilization |

**FAST Features** are corners and textured image locations. They are expected to appear on stationary background objects. They are not a motion detector.

## Stats bar

- **Analyzed / Skipped** — total frames processed vs. rate-limiter rejections
- **Frame Rotation** — `imageProxy.imageInfo.rotationDegrees`; discrete CameraX metadata (0, 90, 180, or 270°) indicating how many degrees to rotate the raw sensor buffer CW to make it display-upright. Not phone tilt.
- **FAST Features** — keypoint count and per-stage timing (grayscale / FAST / total)
- **Stab** — keypoint count, raw → filtered → inlier match counts, inlier ratio, transform (tx, ty, rotation, scale), MAD before/after/reduction
- **Video diag** — requested frame position, actual frame time (always N/A — `MediaMetadataRetriever` does not report it), duplicate-frame hash indicator

## Temporal state

Each source (`CAMERA_REAR`, `CAMERA_FRONT`, `VIDEO`) maintains independent per-source frame state in `StabilizationProcessor`. State resets on:
- Switching input modes
- Switching camera lens
- `selectVideo`, `restartVideo`, `stopVideo`
- Timestamp gap > 2 s (seek / loop detection)
- Duplicate video frame (hash check) — state is not updated, continuity preserved for the next unique frame

## Architecture rules

- No YOLO, TensorFlow Lite, ONNX, ML Kit, neural networks, cloud services, databases, remote APIs.
- Camera and video use the same CV pipeline — no separate analysis paths.
- All OpenCV `Mat` objects are explicitly released. `ImageProxy` is always closed.
- Debug bitmaps are not explicitly recycled — GC reclaims them when the `StateFlow` value is replaced and Compose finishes rendering.

## Milestone status

| # | Description | Status |
|---|---|---|
| M0 | Project setup, Compose scaffold | Done |
| M1 | CameraX live preview, front/rear lens switch | Done |
| M2 | Local video selection and playback (SAF + ExoPlayer) | Done |
| M3 | Shared `AnalysisFrame` pipeline, frame extraction | Done |
| M4 | OpenCV FAST feature detection, display modes, diagnostic stats | Done |
| M5 | Feature correspondence, partial affine stabilization, diff modes | Done |
| M6 | Background subtraction / motion detection | Planned |
