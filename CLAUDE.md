# Musical Flow Matching Visualizer — Dev Context

## What this project is

A Gradio app where a user records/uploads a melody (4–10s) and the app generates 4 audio variations while showing particles flowing through a learned 2D sound space. Extends the prior toy flow matching visualizer at `../flow_matching_visualizer/` (fully complete, use it as reference for porting).

## Git repo
- **Local**: `C:\Users\neeln\OneDrive\Desktop\Claude_projects\Music-flow-matching` (this is the git root)
- **Remote**: https://github.com/NeelN86/music-flow-matching
- All commits go here — do NOT commit to the parent `Claude_projects/` repo

## Current state — ALL code complete, training in progress

### All steps implemented (commit baacc4b, 2026-05-22)
All 30 unit tests pass across 5 test files:

| File | Tests |
|------|-------|
| `tests/test_data.py` | 5/5 ✅ |
| `tests/test_vae.py` | 8/8 ✅ |
| `tests/test_model.py` | 7/7 ✅ |
| `tests/test_solvers.py` | 6/6 ✅ |
| `tests/test_vocoder.py` | 4/4 ✅ |

Implemented files (all stubs replaced):
- `src/vae.py` — AudioVAE with logvar clamp, train_vae, load_vae
- `src/vocoder.py` — mel_to_wav + decode_batch (parallel Griffin-Lim)
- `src/visualize.py` — mel_thumbnail, latent_scatter, animate_flow, render_static_quiver
- `src/model.py` — VelocityMLP (style-conditioned, input_dim=5)
- `src/train.py` — train_flow + _cycle
- `src/solvers.py` — euler_integrate + velocity_field_on_grid
- `scripts/train_vae.py`, `scripts/eyeball_latent.py`, `scripts/eyeball_reconstruct.py`
- `app.py` — Gradio UI fully wired with threading model

### Mel normalization stats (config.py)
`MEL_MEAN = -43.9487`, `MEL_STD = 20.9642` — computed from nsynth-valid (2000 samples).

## Known issue fixed (2026-05-22)

**Posterior collapse** was observed when training with beta=0.5 (KL → 0 at epoch 3,
all latents at origin). Fixed by:
- `VAE_BETA = 0.1` in `src/config.py`
- KL warmup over 5 epochs in `train_vae` (linearly ramps beta 0 → target)

Retrain running in background: `python scripts/train_vae.py --data_dir data/nsynth-valid --max_samples 2000 --epochs 30 --beta 0.1 --beta_warmup 5`

## Resume here next session

**Step A — verify VAE retrain** (should be done; check checkpoint was updated):
```
python scripts/eyeball_latent.py --data_dir data/nsynth-valid --max_samples 2000
# PASS: ≥3 families form separable clusters, latent range not collapsed at origin
python scripts/eyeball_reconstruct.py --data_dir data/nsynth-valid
# PASS: MSE < 1.5
```
If collapse persists: try `--beta 0.05` or `--beta_warmup 10`.

**Step B — train flow model** (after VAE checkpoint is good):
```
python -c "
from src.vae import load_vae
from src.data import NSynthDataset, make_dataloader
from src.train import train_flow
vae = load_vae()
ds = NSynthDataset('data/nsynth-valid/examples.json', 'data/nsynth-valid/audio', normalize=True)
loader = make_dataloader(ds, batch_size=256)
train_flow(vae, loader, steps=10000)
"
```

**Step C — launch and test app e2e:**
```
python app.py
```
Record or upload a short melody. Expected: GIF animation appears, 4 WAV variations download.
Target: e2e under 8s on CPU. Check `outputs/flow.gif` was created.

**Stretch goals (Steps 9+10):** style slider, sonic scrubbing grid — implement in app.py.

---

## Full build order (for reference)

| Step | Files | Gate |
|------|-------|------|
| ✅ 0 | Scaffold, .gitignore, README, requirements.txt | committed |
| ✅ 1 | `src/data.py`, `tests/test_data.py` | 5/5 pass |
| 🔄 2 | `src/vae.py`, `tests/test_vae.py`, 3 scripts | **MILESTONE GATE** (scripts done; awaiting NSynth data to run) |
| ⬜ 3 | `src/model.py`, `tests/test_model.py` | 7 tests pass |
| ⬜ 4 | `src/train.py` | loss decreases on 100-step smoke run |
| ⬜ 5 | `src/solvers.py`, `tests/test_solvers.py` | trajectory shape [51,4,2] |
| ⬜ 6 | `src/vocoder.py`, `tests/test_vocoder.py` | WAVs created, parallel faster |
| ⬜ 7 | `src/visualize.py` | GIF renders with 3-layer zorder |
| ⬜ 8 | `app.py` | e2e under 8s on CPU |
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
Loss: MSE(recon, x) + beta * KL  where beta=0.5 in config.py
```

### VelocityMLP (Step 3 — not yet started)
Port of `../flow_matching_visualizer/src/model.py`, only change is input_dim 3→5:
- Input: [B,5] = cat([x, y, t, style_x, style_y])  where style = VAE mu of input audio
- 3 hidden layers, width 256, SiLU
- Output: [B,2] = (vx, vy)
- style=None → zeros (unconditional mode for tests)

### Generation threading model (Step 8 — app.py)
```
1. load_audio → audio_to_mel → normalize_mel → vae.encode → mu (style)
2. euler_integrate(4 noise samples, style=mu) → trajectory [51, 4, 2]
3. Thread(target=decode_batch, args=(final_latents, ...))  ← start background
4. animate_flow(trajectory) → GIF  [~3.5s in main thread]
5. thread.join(timeout=1.0)
6. return gif_path, wav_paths, mel_figure
```

---

## Key files to reference from prior project

| Need | Source file |
|------|-------------|
| VelocityMLP to port | `../flow_matching_visualizer/src/model.py` |
| Euler + quiver grid to port | `../flow_matching_visualizer/src/solvers.py` |
| Quiver+particle animation pattern | `../flow_matching_visualizer/src/visualize.py` |
| Flow matching training loop | `../flow_matching_visualizer/src/train.py` |
| VAE structure reference | `../flow_matching_visualizer/src/vae.py` (MNIST, simpler) |

---

## Config values that matter most

```python
# src/config.py — key knobs
VAE_BETA = 0.5          # β-VAE KL weight; adjust based on milestone gate result
MEL_MEAN = -5.0         # placeholder — run compute_mel_stats() after data download
MEL_STD = 2.5           # placeholder — same
GRIFFIN_LIM_ITERS = 60  # vocoder quality vs speed trade-off
ANIMATION_N_STEPS = 70  # 70 frames @ 20fps = 3.5s animation window
```

---

## NSynth data setup (if not yet downloaded)

```
data/
└── nsynth-valid/
    ├── examples.json
    └── audio/
        ├── bass_acoustic_000-024-050.wav
        └── ...
```

Download from: https://magenta.tensorflow.org/datasets/nsynth  
Size: ~2.5 GB for nsynth-valid. Use `--max_samples 500` for fast dev iteration before full training.

## Next Session Opener
Paste this to resume:

"Resume the project. Read CLAUDE.md fully before doing anything.
Current state: paused mid Step 2. src/vae.py and tests/test_vae.py
are committed but VAE test results unverified — logvar clamp fix was applied.
Start by running: python tests/test_vae.py
Report results before proceeding."