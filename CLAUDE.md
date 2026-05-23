# Musical Flow Matching Visualizer — Dev Context

## What this project is

A Gradio app where a user records/uploads a melody (4–10s) and the app generates 4 audio variations while showing particles flowing through a learned 2D sound space. Extends the prior toy flow matching visualizer at `../flow_matching_visualizer/` (fully complete, use it as reference for porting).

## Git repo
- **Local**: `C:\Users\neeln\OneDrive\Desktop\Claude_projects\Music-flow-matching` (this is the git root)
- **Remote**: https://github.com/NeelN86/music-flow-matching
- All commits go here — do NOT commit to the parent `Claude_projects/` repo

## Current state — CORE APP COMPLETE, e2e verified (2026-05-23)

All 30 unit tests pass. Both checkpoints trained. Full pipeline verified at 7.8s on CPU.

| File | Tests |
|------|-------|
| `tests/test_data.py` | 5/5 ✅ |
| `tests/test_vae.py` | 8/8 ✅ |
| `tests/test_model.py` | 7/7 ✅ |
| `tests/test_solvers.py` | 6/6 ✅ |
| `tests/test_vocoder.py` | 4/4 ✅ |

### Checkpoints
- `checkpoints/audio_vae.pt` — trained 30 epochs, beta=0.1 w/ 5-epoch warmup. MSE 0.19, latent range x=[-2.4,1.9] y=[-2.7,3.8], 10 instrument families separable.
- `checkpoints/flow_model.pt` — trained 10,000 steps on NSynth-valid (2000 samples).

### Mel normalization stats (config.py)
`MEL_MEAN = -43.9487`, `MEL_STD = 20.9642` — computed from nsynth-valid (2000 samples).

## Known issues fixed

**Posterior collapse** (2026-05-22): VAE_BETA lowered 0.5→0.1, KL warmup over 5 epochs.
- Retrain completed: MSE 0.19 (PASS), latent range x=[-2.4,1.9] y=[-2.7,3.8], 10 families.

**Flow model training** (2026-05-23): `checkpoints/flow_model.pt` written at 03:34 AM.

**animate_flow performance** (2026-05-23): batched quiver precomputation + smaller figure.
- e2e time 10.1s → 7.8s (under 8s target), GIF 8MB → 3MB.

## Current state — e2e VERIFIED WORKING (2026-05-23)

Full pipeline validated:
1. VAE encodes input audio → style mu [1,2] ✅
2. euler_integrate produces trajectory [51,4,2] ✅
3. decode_batch writes 4 WAV variations ✅
4. animate_flow writes `outputs/flow.gif` (3MB, 70 frames @ 20fps) ✅
5. Total e2e: **7.8s on CPU** ✅

## Resume here next session

**The core app is complete.** Stretch goals remaining:

**Step 9 — Style slider** (in app.py):
Add a gr.Slider that lets the user offset the style mu (e.g., +/- 1 in each dimension)
before running euler_integrate. Wires into `mu = mu + style_offset`.

**Step 10 — Sonic scrubbing grid** (in app.py):
gr.Plot showing latent_scatter of training data. Click → emit a [1,2] coordinate
as style input and immediately generate one variation at that point.

Both are additive UI features — no changes to ML code needed.

---

## Build order status

| Step | Files | Gate |
|------|-------|------|
| ✅ 0 | Scaffold, .gitignore, README, requirements.txt | committed |
| ✅ 1 | `src/data.py`, `tests/test_data.py` | 5/5 pass |
| ✅ 2 | `src/vae.py`, `tests/test_vae.py`, 3 scripts | MSE 0.19, no collapse |
| ✅ 3 | `src/model.py`, `tests/test_model.py` | 7/7 pass |
| ✅ 4 | `src/train.py` | 10k-step run complete |
| ✅ 5 | `src/solvers.py`, `tests/test_solvers.py` | trajectory [51,4,2] ✓ |
| ✅ 6 | `src/vocoder.py`, `tests/test_vocoder.py` | 4 WAVs created ✓ |
| ✅ 7 | `src/visualize.py` | GIF 70 frames @ 20fps ✓ |
| ✅ 8 | `app.py` | e2e 7.8s on CPU ✓ |
| ⬜ 9 | Style slider (stretch) | — |
| ⬜ 10 | Sonic scrubbing grid (stretch) | — |

---

## Architecture quick reference

### VAE conv dims (kernel=4, stride=2, padding=1)
```
Encoder: [B,1,80,256] → [B,32,40,128] → [B,64,20,64] → [B,128,10,32] → [B,256,5,16]
         flatten(20480) → FC(512) → mu[B,2], logvar[B,2]  (logvar clamped [-4,4])
Decoder: z[B,2] → FC(512) → FC(20480) → reshape[B,256,5,16]
         → ConvTranspose2d ×4 → [B,1,80,256]  (no BN, no final activation)
Loss: MSE(recon, x) + beta * KL  where beta=0.1 (warmup 5 epochs)
```

### VelocityMLP
- Input: [B,5] = cat([x, y, t, style_x, style_y])  where style = VAE mu of input audio
- 3 hidden layers, width 256, SiLU
- Output: [B,2] = (vx, vy)
- style=None → zeros (unconditional mode for tests)

### Generation threading model (app.py)
```
1. load_audio → audio_to_mel → normalize_mel → vae.encode → mu (style)
2. euler_integrate(4 noise samples, style=mu) → trajectory [51, 4, 2]
3. Thread(target=decode_batch, args=(final_latents, ...))  ← start background
4. animate_flow → GIF [70 frames, 4×4 fig, batched quiver precompute]
5. thread.join(timeout=2.0)
6. return gif_path, wav_paths, mel_figure
```

---

## Config values that matter most

```python
# src/config.py — actual values as of 2026-05-23
VAE_BETA = 0.1          # lowered from 0.5 to fix posterior collapse
MEL_MEAN = -43.9487     # computed from nsynth-valid (2000 samples)
MEL_STD = 20.9642       # computed from nsynth-valid (2000 samples)
GRIFFIN_LIM_ITERS = 60  # vocoder quality vs speed trade-off
ANIMATION_N_STEPS = 70  # 70 frames @ 20fps = 3.5s animation
```

---

## NSynth data (already downloaded)

```
data/nsynth-valid/   — 2000-sample subset used for all training
  examples.json
  audio/*.wav
```

## Next Session Opener

"Resume the project. Read CLAUDE.md first.
Core app is complete and e2e verified at 7.8s. Both checkpoints exist.
Next work: stretch goals Step 9 (style slider) and Step 10 (sonic scrubbing grid) in app.py."