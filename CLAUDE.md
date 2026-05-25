# Musical Flow Matching Visualizer — Dev Context

## What this project is

A Gradio app where a user records/uploads a melody (4–10s) and the app generates 4 audio variations while showing particles flowing through a learned 2D sound space. Extends the prior toy flow matching visualizer at `../flow_matching_visualizer/` (fully complete, use it as reference for porting).

## Git repo
- **Local**: `C:\Users\neeln\OneDrive\Desktop\Claude_projects\Music-flow-matching` (this is the git root)
- **Remote**: https://github.com/NeelN86/music-flow-matching
- All commits go here — do NOT commit to the parent `Claude_projects/` repo

## Current state — ALL STEPS COMPLETE (2026-05-25)

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
`MEL_MEAN = -43.9487`, `MEL_STD = 20.9642` — computed from nsynth-valid (2000 samples).

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
2. _make_initial_particles(mu, radius) → 4 start points near mu in PC1/PC2 dirs
3. euler_integrate(4 noise samples, style=mu, steps=100) → trajectory [101, 4, 16]
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

- **Audio quality**: HiFi-GAN should be much cleaner than Griffin-Lim, but variations are generated from 16D flow endpoints that may not reconstruct perfectly. The VAE reconstruction is the ceiling — test "Reconstruct Input" button to hear the VAE ceiling.
- **Test suite**: `tests/test_vocoder.py` tests still call `mel_to_wav` (Griffin-Lim) directly and use `GRIFFIN_LIM_ITERS`. The HiFi-GAN path is not exercised in the test suite (by design — it requires the 80MB checkpoint).
- **Latent cache**: `outputs/latents_cache.npz` must be regenerated if VAE is retrained. Delete the file and click "Load latent space" in the app.

## Next Session Opener

"Resume the project. Read CLAUDE.md first.
VAE is 16D, HiFi-GAN is the primary vocoder (loaded from `checkpoints/hifigan/`), PCA projects 16D latents to 2D for visualization.
App runs — `python app.py` → http://127.0.0.1:7860.
Flow: record/upload → 'Use this recording ▶' → preview → 'Generate Variations'.
Check if audio quality is acceptable with HiFi-GAN. If variations still sound bad, the bottleneck is VAE reconstruction quality (test with 'Reconstruct Input' button)."
