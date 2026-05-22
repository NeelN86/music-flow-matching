# Musical Flow Matching Visualizer

A Gradio app where you record or upload a short melody (5–10 seconds) and the app generates 4 audio variations while showing particles flowing through a learned sound space in real time.

## Two-Project Arc

This is the second project in a two-part series:

**Part 1 — [Flow Matching Visualizer](https://github.com/NeelN86/flow-matching-visualizer)**
A 2D toy that trains a velocity-field MLP to transport Gaussian noise to a target distribution (two moons, GMM, checkerboard), then visualizes the learned field and particle trajectories interactively. Established the core architecture: VelocityMLP, Euler/RK4 ODE integration, layered quiver+particle visualization.

**Part 2 — This project**
Ports the entire architecture into the audio domain. The key insight: if the audio VAE has a 2D bottleneck, the velocity field lives in the same 2D space as the prior project — no PCA approximation needed. Every design decision from Part 1 carries over directly; only the data modality changes.

## ML Architecture

```
NSynth WAV
    ↓ librosa (mel spectrogram)
[B, 1, 80, 256]
    ↓ AudioEncoder (4× Conv2d stride-2 + FC)
mu [B, 2], logvar [B, 2]          ← 2D bottleneck (visualizable)
    ↓
    ├─ mu  →  style conditioning for VelocityMLP
    └─ mu + ε·σ  →  x1 (flow matching target)

x0 ~ N(0,I)   t ~ U[0,1]
    ↓
x_t = (1-t)·x0 + t·x1            ← linear interpolation path
    ↓ VelocityMLP([x, y, t, style_x, style_y])
v_pred [B, 2]
    ↓ MSE vs (x1 - x0)            ← flow matching loss

── INFERENCE ──
input audio → mu (style)
4 noise samples x0 ~ N(0,I)
    ↓ Euler ODE (50 steps)
trajectories [51, 4, 2]
    ├─ → particle animation (GIF, 3.5s)        [main thread]
    └─ → AudioDecoder + Griffin-Lim → 4 WAVs  [background thread]
```

### Components

| Module | Role |
|--------|------|
| `src/data.py` | librosa mel pipeline, NSynth dataset loader |
| `src/vae.py` | β-VAE with 2D bottleneck, trained on mel spectrograms |
| `src/model.py` | VelocityMLP — style-conditioned velocity field |
| `src/train.py` | Flow matching training loop (VAE frozen) |
| `src/solvers.py` | Euler ODE integration, velocity grid evaluation |
| `src/vocoder.py` | Griffin-Lim phase recovery, parallel WAV decoding |
| `src/visualize.py` | Quiver + particle animation, mel thumbnails |
| `src/config.py` | All hyperparameters in one place |
| `app.py` | Gradio UI wiring only |

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NSynth data

Download the NSynth-valid split (~2.5 GB) from:
https://magenta.tensorflow.org/datasets/nsynth

Extract so your layout is:
```
data/
└── nsynth-valid/
    ├── examples.json
    └── audio/
        ├── bass_acoustic_000-024-050.wav
        └── ...
```

### 3. Train the VAE

```bash
python scripts/train_vae.py --data_dir data/nsynth-valid --epochs 30
```

After training, inspect the latent space and reconstruction quality:

```bash
python scripts/eyeball_latent.py --data_dir data/nsynth-valid
python scripts/eyeball_reconstruct.py --data_dir data/nsynth-valid
```

**Milestone gate:** Do not proceed until instrument families cluster in the scatter plot and reconstructed mel spectrograms look clean.

### 4. Train the flow matching model

```bash
python scripts/train_flow.py --data_dir data/nsynth-valid
```

### 5. Launch the app

```bash
python app.py
```

## Design Decisions

- **2D latent space** — non-negotiable. Makes the velocity field directly visualizable (same axes as particle animation) without any dimensionality reduction.
- **style_emb = VAE mu** — the encoded input audio's latent position is used directly as the style conditioning signal. No separate style encoder; the VAE encoder is the learned embedding.
- **β-VAE (β=0.5)** — prioritizes reconstruction quality over latent regularity. Tunable in `src/config.py`.
- **Griffin-Lim vocoder** — fast, no extra model checkpoint. Known limitation: metallic artifacts on transient instruments. Upgrade path: HiFi-GAN.
- **Custom Euler for animation** — `torchdiffeq.odeint` is in dependencies for optional high-quality generation, but animation requires uniform time steps, so custom Euler is used for trajectory pre-computation.

## Known Limitations

- Griffin-Lim produces audible phase artifacts, especially on guitar/mallet sounds.
- NSynth-valid (12k samples) produces weaker latent clustering than NSynth-train (300k). Use NSynth-train for best results.
- CPU-only. Full VAE training on NSynth-valid takes ~2–4 hours on CPU.
