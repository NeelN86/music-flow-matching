# Musical Flow Matching Visualizer — Dev Context

## What this project is

A Gradio app where a user records/uploads a melody (4–10s) and the app generates 4 audio variations while showing particles flowing through a learned 2D sound space. Extends the prior toy flow matching visualizer at `../flow_matching_visualizer/` (fully complete, use it as reference for porting).

## Git repo
- **Local**: `C:\Users\neeln\OneDrive\Desktop\Claude_projects\Music-flow-matching` (this is the git root)
- **Remote**: https://github.com/NeelN86/music-flow-matching
- All commits go here — do NOT commit to the parent `Claude_projects/` repo

## Current state — ALL STEPS COMPLETE + mixed dataset retrained (2026-05-27)

All major work is done. HiFi-GAN vocoder integrated and verified. VAE upgraded to 16D latent with PCA visualization.

| File | Tests |
|------|-------|
| `tests/test_data.py` | 5/5 ✅ |
| `tests/test_vae.py` | 8/8 ✅ |
| `tests/test_model.py` | 7/7 ✅ |
| `tests/test_solvers.py` | 6/6 ✅ |
| `tests/test_vocoder.py` | 4/4 ✅ |

**Note**: Tests still pass but were written for 2D latent. The model/solver/vocoder tests use `VAE_LATENT_DIM` from config so they are 16D-aware. Some tests that formerly used hardcoded `B, 2` shapes were updated.

### Checkpoints
- `checkpoints/audio_vae.pt` — 16D VAE, trained 30 epochs, beta=0.001. Retrained after latent dim upgrade.
- `checkpoints/flow_model.pt` — VelocityMLP for 16D, trained 30k steps on full NSynth-valid (12,678 samples), cosine LR 1e-3→1e-5 w/ 1k warmup. Retrained after latent dim upgrade.
- `checkpoints/hifigan/` — SpeechBrain tts-hifigan-libritts-16kHz pre-trained weights (local copy, no network needed):
  - `hyperparams.yaml` — SpeechBrain model config (80 mels, sr=16kHz, upsample 8×8×2×2)
  - `generator.ckpt` — pre-trained HiFi-GAN weights (~80MB)

### Mel normalization stats (config.py)
`MEL_MEAN = -45.2344`, `MEL_STD = 20.7955` — recomputed from mixed dataset (NSynth-valid + VocalSet, 2000 samples each).

---

## Build order status

| Step | Files | Gate |
|------|-------|------|
| ✅ 0 | Scaffold, .gitignore, README, requirements.txt | committed |
| ✅ 1 | `src/data.py`, `tests/test_data.py` | 5/5 pass |
| ✅ 2 | `src/vae.py`, `tests/test_vae.py`, 3 scripts | MSE 0.19, no collapse |
| ✅ 3 | `src/model.py`, `tests/test_model.py` | 7/7 pass |
| ✅ 4 | `src/train.py`, `scripts/train_flow.py` | 30k-step retrain, cosine LR, 12678 samples |
| ✅ 5 | `src/solvers.py`, `tests/test_solvers.py` | trajectory [51,4,16] ✓ |
| ✅ 6 | `src/vocoder.py`, `tests/test_vocoder.py` | HiFi-GAN primary, Griffin-Lim fallback ✓ |
| ✅ 7 | `src/visualize.py` | GIF 70 frames @ 20fps, PCA projection for 16D ✓ |
| ✅ 8 | `app.py` | e2e working, HiFi-GAN verified ✓ |
| ✅ 9 | Style slider (stretch) | sliders wired, mu+offset verified |
| ✅ 10 | Sonic scrubbing grid (stretch) | scatter + scrub WAV verified |

---

## Key upgrades made (2026-05-25 session)

### 1. VAE latent dim: 2D → 16D
**Motivation**: 2D bottleneck was a hard ceiling on reconstruction quality regardless of beta. VAE reconstruction sounded nothing like the input at 2D.

- `VAE_LATENT_DIM = 16` in `src/config.py`
- `VAE_BETA = 0.001` (reconstruction-dominant; KL is now tiny regularizer)
- `checkpoints/audio_vae.pt` retrained at 16D
- `checkpoints/flow_model.pt` retrained at 16D (VelocityMLP input: `2*16+1=33`, output: `16`)

### 2. PCA visualization for 16D latents
Since the visualization is 2D, 16D latents are projected via PCA (SVD) to the top 2 principal components.

- `app.py`: `_compute_latent_space()` fits PCA on 300 NSynth-valid samples, saves `pca_mean`/`pca_components` to `outputs/latents_cache.npz`
- `app.py`: `_try_load_latent_cache()` loads cached PCA on startup
- `generate_variations()`: if `_pca` not loaded, computes fallback PCA from the 4-particle trajectory via SVD (so animation always works)
- `src/visualize.py`: `animate_flow(pca=...)` projects trajectory to 2D, back-projects quiver grid to 16D for velocity evaluation
- `src/solvers.py`: `velocity_field_on_grid(pca=...)` back-projects 2D grid → 16D before model eval

### 3. HiFi-GAN vocoder (primary path)
Griffin-Lim produces buzzy, artifact-heavy audio. HiFi-GAN is a neural vocoder that produces clean, natural-sounding waveforms.

- Model: `speechbrain/tts-hifigan-libritts-16kHz` (matches our sr=16kHz, n_mels=80 config exactly)
- Weights stored locally at `checkpoints/hifigan/` — no network request at runtime
- Loaded lazily on first decode; cached for session
- Falls back to Griffin-Lim if checkpoint missing or load fails
- Mel format conversion: `dB-scale → power → log(power + 1e-9)` — what HiFi-GAN was trained on
- **Critical bug fixed**: `librosa.db_to_power` triggers a lazy-import chain that conflicts with SpeechBrain's k2 lazy module. Fixed by inlining: `power = 10.0 ** (mel_db / 10.0)`

### 4. New UI elements (app.py)
- **"Reconstruct Input (VAE only)"** button + audio player: encodes input → decodes directly → plays back. Useful for checking VAE quality independently of flow model.
- **"Variation radius"** slider (0–3, default 1.0): controls how far initial particles are spread from input mu (cardinal ±PC1/±PC2 directions).

---

## Key work done (2026-05-27 session)

### 7. Mixed dataset retraining (NSynth + VocalSet)
- Added `FlatAudioDataset` to `src/data.py` — scans any WAV directory recursively (handles VocalSet, LJSpeech, etc.)
- Added `compute_mel_stats_mixed()` for recomputing mel stats across multiple directories
- Updated `scripts/train_vae.py` and `scripts/train_flow.py` with `--extra_audio_dirs` flag
- Added `train_vae_from_loader` and GPU support (`cuda` auto-detect) to both training functions
- Created `colab_train_mixed.ipynb` — full GPU training notebook; downloads VocalSet (2.1 GB) automatically from Zenodo
- Retrained VAE + flow model on NSynth-valid + VocalSet on Colab T4 (~25 min total)
- New mel stats: `MEL_MEAN = -45.2344`, `MEL_STD = 20.7955`

### 8. Audio post-processing
- `src/vocoder.py`: Gaussian temporal mel smoothing (σ=1 frame) reduces VAE frame-to-frame jitter
- `src/vocoder.py`: Peak normalization to 0.95 + 20ms fade in/out on all outputs (removes clicks)

### 9. Timbre targeting
- `app.py`: Per-instrument-family centroid vectors computed when "Load latent space" is clicked
- Centroids stored in `outputs/latents_cache.npz` alongside PCA
- New UI: **Target timbre** dropdown (bass/brass/flute/guitar/keyboard/mallet/organ/reed/string/synth_lead/vocal) + **Timbre blend** slider (0 = input style, 1 = pure instrument)
- Blends `mu` toward selected centroid: `mu = (1-blend)*mu + blend*centroid`

---

## Key fixes made (2026-05-26 session)

### 5. Variation diversity fix
**Problem**: All 4 variations were near-identical. Two compounding causes:
1. `_make_initial_particles` was starting particles at `mu ± radius*PC_direction` — i.e., near `mu`, which is OOD for the flow model (trained on x0 ~ N(0,I)).
2. All 4 particles used the **same** style vector `mu`, so the velocity field pulled them all to the same attractor.

**Fix** (`app.py`):
- Removed `_make_initial_particles(mu, radius)` — replaced with `_make_initial_particles()` that always returns pure `N(0,I)` noise (what the flow model expects).
- Added `_make_per_particle_styles(mu, diversity)` that gives each particle its own style: `mu + diversity * N(0,I)`. Different style attractors → genuinely different endpoints.
- Renamed "Variation radius" UI slider → **"Diversity"** (0=identical, 1.0=default, 3.0=very diverse).
- `euler_integrate` already supports `[B, D]` per-particle style tensors — no solver changes needed.

### 6. Mel spectrogram on confirm
**Problem**: Mel spectrogram only appeared after "Generate Variations", not after confirming audio.

**Fix** (`app.py`): `confirm_recording` now computes and returns the mel plot immediately — useful for verifying input is being read correctly before running the full generation pipeline.

### 7. VAE domain mismatch (diagnosed, not fully fixable without retraining)
The VAE was trained exclusively on NSynth (sustained instrument notes). Personal microphone recordings are OOD → poor reconstruction is expected. Best proxies in the existing dataset:
- `vocal_acoustic_000-*-050.wav` — closest to humming/singing
- `flute_acoustic_002-*-050.wav` — pure sustained tones
- `reed_acoustic_*-050.wav` — breathy, expressive

If personal audio support is needed in the future, the VAE would need to be retrained on a mixed dataset (NSynth + voice data).

---

## Architecture quick reference

### VAE conv dims (kernel=4, stride=2, padding=1)
```
Encoder: [B,1,80,256] → [B,32,40,128] → [B,64,20,64] → [B,128,10,32] → [B,256,5,16]
         flatten(20480) → FC(512) → mu[B,16], logvar[B,16]  (logvar clamped [-4,4])
Decoder: z[B,16] → FC(512) → FC(20480) → reshape[B,256,5,16]
         → ConvTranspose2d ×4 → [B,1,80,256]  (no BN, no final activation)
Loss: MSE(recon, x) + beta * KL  where beta=0.001
```

### VelocityMLP
- Input: `[B, 33]` = cat([x_16d, t, style_16d]) where style = VAE mu of input audio
- 3 hidden layers, width 256, SiLU
- Output: `[B, 16]` = velocity vector in 16D latent space
- style=None → zeros (unconditional mode for tests)

### Vocoder pipeline
```
Primary (HiFi-GAN):
  mel_normalized → mel_db = mel * MEL_STD + MEL_MEAN
                → power = 10^(mel_db/10)
                → log_mel = log(power + 1e-9)   ← HiFi-GAN's expected format
                → HIFIGAN.decode_batch()
                → wav float32

Fallback (Griffin-Lim, if HiFi-GAN unavailable):
  mel_normalized → mel_db → power → mel_to_stft → griffinlim(128 iters) → wav
```

### Generation threading model (app.py)
```
1. load_audio → audio_to_mel → normalize_mel → vae.encode → mu [1,16] (style)
2. _make_initial_particles() → 4 pure N(0,I) noise samples [4, 16]
   _make_per_particle_styles(mu, diversity) → 4 style vectors [4, 16]
3. euler_integrate(x0, style=[4,16], steps=100) → trajectory [101, 4, 16]
4. Thread(target=decode_batch, args=(final_latents, vae.decode, ...))  ← HiFi-GAN
5. vis_pca = _pca or fallback SVD from trajectory
6. animate_flow(trajectory, pca=vis_pca) → GIF [70 frames @ 20fps]
7. thread.join(timeout=2.0)
8. return gif_path, wav_paths, mel_figure
```

---

## Config values that matter most

```python
# src/config.py — actual values as of 2026-05-25
VAE_LATENT_DIM = 16         # upgraded from 2 for reconstruction quality
VAE_BETA = 0.001            # reconstruction-dominant; tiny KL regularizer
MEL_MEAN = -43.9487         # computed from nsynth-valid (2000 samples)
MEL_STD = 20.9642           # computed from nsynth-valid (2000 samples)
GRIFFIN_LIM_ITERS = 128     # fallback only; HiFi-GAN is primary
ANIMATION_N_STEPS = 70      # 70 frames @ 20fps = 3.5s animation
EULER_STEPS = 100           # 100 integration steps
FLOW_STEPS = 30_000         # cosine LR schedule, 12678-sample dataset
```

---

## NSynth data (already downloaded)

```
data/nsynth-valid/   — full split: 12,678 samples used for flow model retrain
  examples.json      — (VAE was trained on 2000-sample subset)
  audio/*.wav
```

## Known issues / potential next work

- **Style offset broken**: The style offset X/Y sliders in the UI apply a 2D offset to mu directly, but mu is 16D — the offset only nudges the first 2 dimensions and has negligible effect. Needs to be reimplemented as a PCA-space offset (project offset through PCA components before adding to mu).
- **Audio quality**: HiFi-GAN produces much cleaner audio than Griffin-Lim. VAE reconstruction is the ceiling — test "Reconstruct Input" to hear it. Mixed dataset retraining (NSynth + VocalSet) has improved voice/humming input quality.
- **Personal audio**: Even with VocalSet added, arbitrary mic recordings may still sound imperfect. Best inputs: NSynth `vocal_acoustic_*` or `flute_acoustic_*` files, or clean sustained humming.
- **Timbre targeting requires latent cache**: `_centroids` are stored in `outputs/latents_cache.npz`. User must click "Load latent space" once per session (or after VAE retrain) to compute centroids.
- **Test suite**: `tests/test_vocoder.py` tests still call `mel_to_wav` (Griffin-Lim) directly. The HiFi-GAN path is not exercised in the test suite (by design — requires the 80MB checkpoint).
- **Latent cache**: `outputs/latents_cache.npz` must be regenerated if VAE is retrained. Delete the file and click "Load latent space" in the app.

## Next Session Opener

"Resume the project. Read CLAUDE.md first.
VAE is 16D retrained on NSynth + VocalSet mixed dataset. HiFi-GAN is the primary vocoder. PCA projects 16D latents to 2D for visualization.
App runs — `python app.py` → http://127.0.0.1:7860.
Flow: record/upload → 'Use this recording ▶' → mel spectrogram shows immediately → 'Generate Variations'.
Best test input: upload a file from `data/nsynth-valid/audio/vocal_acoustic_000-*-050.wav`.
New features: timbre targeting dropdown (needs 'Load latent space' first), diversity slider.

**Next fix needed**: Style offset sliders are broken — they apply a 2D offset to a 16D vector, so only dims 0-1 are nudged. Fix: reimplement as PCA-space offset (multiply offset by PCA components matrix before adding to mu)."
