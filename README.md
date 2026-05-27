# Musical Flow Matching Visualizer

A Gradio app where you record or upload a short melody (4–10 seconds) and the app generates 4 audio variations while showing particles flowing through a learned sound space in real time.

## Two-Project Arc

This is the second project in a two-part series:

**Part 1 — [Flow Matching Visualizer](https://github.com/NeelN86/flow-matching-visualizer)**
A 2D toy that trains a velocity-field MLP to transport Gaussian noise to a target distribution (two moons, GMM, checkerboard), then visualizes the learned field and particle trajectories interactively.

**Part 2 — This project**
Ports the architecture into the audio domain. The VAE encodes audio into a 16D latent space; PCA projects to 2D for visualization. Every design decision from Part 1 carries over; only the data modality and dimensionality change.

## ML Architecture

```
NSynth + VocalSet WAV
    ↓ librosa (mel spectrogram, 80 mels, 16kHz)
[B, 1, 80, 256]
    ↓ AudioEncoder (4× Conv2d stride-2 + FC)
mu [B, 16], logvar [B, 16]        ← 16D bottleneck
    ↓ PCA (SVD, top-2 components) ← for visualization only
    ├─ mu  →  style conditioning for VelocityMLP
    └─ mu + ε·σ  →  x1 (flow matching target)

x0 ~ N(0,I)   t ~ U[0,1]
    ↓
x_t = (1-t)·x0 + t·x1            ← linear interpolation path
    ↓ VelocityMLP([x_16d, t, style_16d])  ← input dim 33
v_pred [B, 16]
    ↓ MSE vs (x1 - x0)

── INFERENCE ──
input audio → mu [1, 16] (style)
4 noise samples x0 ~ N(0,I) [4, 16]
per-particle styles = mu + diversity × N(0,I)
    ↓ Euler ODE (100 steps)
trajectories [101, 4, 16]
    ├─ PCA → 2D → particle animation (GIF, 3.5s)   [main thread]
    └─ VAE decode → HiFi-GAN → 4 WAVs              [background thread]
```

### Components

| Module | Role |
|--------|------|
| `src/data.py` | librosa mel pipeline, NSynth dataset loader, `FlatAudioDataset` for mixed training |
| `src/vae.py` | β-VAE with 16D bottleneck (β=0.001), trained on mel spectrograms |
| `src/model.py` | VelocityMLP — style-conditioned velocity field, input dim 33 |
| `src/train.py` | Flow matching training loop (VAE frozen), CUDA-aware |
| `src/solvers.py` | Euler ODE integration, velocity grid evaluation with PCA back-projection |
| `src/vocoder.py` | HiFi-GAN primary vocoder (SpeechBrain), Griffin-Lim fallback, mel smoothing + audio normalization |
| `src/visualize.py` | Quiver + particle animation with PCA projection, mel thumbnails |
| `src/config.py` | All hyperparameters in one place |
| `app.py` | Gradio UI — timbre targeting, diversity slider, VAE reconstruction baseline |

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NSynth data

Download the NSynth-valid split (~3.4 GB) from:
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

### 3. Train the models (CPU or Colab GPU)

**Option A — local CPU** (slow, ~4 hours total):
```bash
python scripts/train_vae.py --data_dir data/nsynth-valid --epochs 30
python scripts/train_flow.py --data_dir data/nsynth-valid --steps 30000
```

**Option B — Google Colab GPU** (recommended, ~25 minutes total):
Open `colab_train_mixed.ipynb` in Colab with a T4 GPU runtime. The notebook downloads VocalSet automatically, retrains on the mixed dataset, and saves checkpoints to Google Drive.

**Mixed dataset training** (NSynth + VocalSet voice data):
```bash
python scripts/train_vae.py --data_dir data/nsynth-valid --extra_audio_dirs data/vocalset
python scripts/train_flow.py --data_dir data/nsynth-valid --extra_audio_dirs data/vocalset
```

### 4. Download HiFi-GAN vocoder

The app uses SpeechBrain's `tts-hifigan-libritts-16kHz` model. Place the checkpoint files at:
```
checkpoints/hifigan/
    hyperparams.yaml
    generator.ckpt
```

### 5. Launch the app

```bash
python app.py
```

Open http://127.0.0.1:7860

## UI Features

- **Record or upload** a 4–10s audio clip (works best with sustained notes; NSynth vocal/flute files give cleanest results)
- **Reconstruct Input (VAE only)** — hear the VAE's quality ceiling before running the flow model
- **Diversity slider** — controls how different the 4 variations sound from each other
- **Target timbre** — blend your input's style toward a specific instrument family (piano, guitar, flute, etc.); requires clicking **Load latent space** first
- **Sound Space Explorer** — scrub through the 2D PCA latent space and generate audio at any point
- **Velocity field preview** — inspect the learned flow field at any time step t

## Design Decisions

- **16D latent space** — upgraded from 2D to improve reconstruction quality. 2D was a hard ceiling regardless of training; 16D gives the VAE enough capacity to encode timbral detail. PCA projects to 2D for visualization with no architectural change.
- **β=0.001** — reconstruction-dominant loss. KL is a tiny regularizer rather than a strong constraint; this prioritizes audio quality over latent space structure.
- **HiFi-GAN vocoder** — SpeechBrain's `tts-hifigan-libritts-16kHz` matches our mel config exactly (sr=16kHz, n_mels=80). Produces dramatically cleaner audio than Griffin-Lim. Loaded lazily, cached for session.
- **Mixed dataset (NSynth + VocalSet)** — NSynth alone (sustained instrument notes) is out-of-distribution for voice/humming input. Adding VocalSet (20 singers, 10+ hours) extends the VAE's domain to singing voice.
- **Per-particle style vectors** — each of the 4 particles gets its own style `mu + diversity × noise`, so they converge to different attractors rather than all landing at the same point.
- **Timbre targeting** — instrument centroids (mean latent across each NSynth family) let users blend toward a specific timbre without retraining.

## Known Limitations

- **VAE domain gap** — even with VocalSet added, arbitrary microphone recordings may reconstruct imperfectly. Best inputs: NSynth vocal/flute files, or clean sustained humming/singing.
- **Style offset broken** — the PCA-space style offset sliders in the UI don't correctly map to 16D latent space yet (pending fix).
- **Timbre targeting requires latent cache** — click **Load latent space** before using the target timbre dropdown.
- **HiFi-GAN checkpoint not included** — download separately from SpeechBrain (tts-hifigan-libritts-16kHz). Falls back to Griffin-Lim if missing.
