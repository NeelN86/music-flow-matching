# All hyperparameters in one place. Every other module imports from here.

# ── Audio pipeline ────────────────────────────────────────────────────────────
SAMPLE_RATE: int = 16_000
CLIP_DURATION: float = 4.0          # seconds; matches NSynth clip length
N_MELS: int = 80
N_FFT: int = 1024
HOP_LENGTH: int = 256               # 16000 * 4 / 256 = 250 frames → padded to 256
N_FRAMES: int = 256                 # target width after pad/truncate

# ── VAE architecture ──────────────────────────────────────────────────────────
# Encoder: 4 Conv2d(stride=2) stages → [256, 5, 16] → flatten → FC(512) → mu/logvar
VAE_CHANNELS: list = [1, 32, 64, 128, 256]
VAE_FLAT_DIM: int = 256 * 5 * 16   # 20480
VAE_BOTTLENECK_FC: int = 512
VAE_LATENT_DIM: int = 2            # 2D bottleneck — non-negotiable for visualization

# ── VAE training ──────────────────────────────────────────────────────────────
VAE_BETA: float = 0.5              # β-VAE KL weight; <1.0 prioritizes reconstruction
VAE_LR: float = 1e-3
VAE_BATCH_SIZE: int = 64
VAE_EPOCHS: int = 30

# ── Mel normalization (global stats over NSynth-valid) ────────────────────────
# Placeholders — overwrite after running data.compute_mel_stats()
MEL_MEAN: float = -43.9487   # computed from nsynth-valid (2000-sample subset)
MEL_STD: float = 20.9642    # computed from nsynth-valid (2000-sample subset)

# ── Flow matching ─────────────────────────────────────────────────────────────
FLOW_HIDDEN_WIDTH: int = 256
FLOW_N_HIDDEN: int = 3
FLOW_LR: float = 1e-3
FLOW_BATCH_SIZE: int = 256
FLOW_STEPS: int = 10_000

# ── Inference / generation ────────────────────────────────────────────────────
N_VARIATIONS: int = 4
EULER_STEPS: int = 50

# ── Animation ─────────────────────────────────────────────────────────────────
ANIMATION_FPS: int = 20
ANIMATION_DURATION_S: float = 3.5  # 70 frames at 20fps
ANIMATION_N_STEPS: int = 70        # = ANIMATION_FPS * ANIMATION_DURATION_S

# ── Vocoder ───────────────────────────────────────────────────────────────────
GRIFFIN_LIM_ITERS: int = 60

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR: str = "checkpoints"
OUTPUT_DIR: str = "outputs"
DATA_DIR: str = "data"
VAE_CHECKPOINT: str = "checkpoints/audio_vae.pt"
FLOW_CHECKPOINT: str = "checkpoints/flow_model.pt"
