# Gradio entry point — wiring only, no ML logic here.
#
# Generation flow (threading model):
#   1. load_audio + audio_to_mel + normalize_mel + vae.encode → mu (style)
#   2. euler_integrate(4 noise samples) → trajectory [51, 4, 2]
#   3. start Thread(target=decode_batch) — VAE decode + Griffin-Lim
#   4. animate_flow → GIF (3.5s rendering in main thread)
#   5. thread.join(timeout=1.0) — should already be done
#   6. return gif_path, wav_paths, mel_figure
#
# This keeps total e2e time under 8s on CPU:
#   trajectory: <50ms, animation: 3.5s, decode (concurrent): ~2.5s

from __future__ import annotations

import math
import os
import threading

import gradio as gr
import numpy as np
import torch

from src.config import (
    ANIMATION_N_STEPS,
    EULER_STEPS,
    FLOW_CHECKPOINT,
    N_VARIATIONS,
    OUTPUT_DIR,
    VAE_CHECKPOINT,
    VAE_LATENT_DIM,
)
from src.data import audio_to_mel, load_audio, normalize_mel
from src.model import VelocityMLP
from src.solvers import euler_integrate
from src.vae import AudioVAE, load_vae
from src.visualize import animate_flow, latent_scatter, mel_thumbnail, render_static_quiver
from src.vocoder import decode_batch

# Models loaded once at startup
_vae: AudioVAE | None = None
_flow_model: VelocityMLP | None = None

# Step 10: cached latent space for the scrubbing grid
_latents: np.ndarray | None = None        # [N, VAE_LATENT_DIM] full-dim latents
_latents_2d: np.ndarray | None = None     # [N, 2] PCA projections for display
_latent_labels: list[str] | None = None
_pca: tuple[np.ndarray, np.ndarray] | None = None  # (mean [D], components [2, D])
_LATENT_CACHE = os.path.join(OUTPUT_DIR, "latents_cache.npz")

# Timbre targeting: per-instrument-family centroid vectors in latent space
_centroids: dict[str, np.ndarray] | None = None
_KNOWN_FAMILIES = [
    "bass", "brass", "flute", "guitar", "keyboard",
    "mallet", "organ", "reed", "string", "synth_lead", "vocal",
]


def _load_models() -> tuple[str, str]:
    """Load VAE and flow model from checkpoints. Returns (vae_status, flow_status)."""
    global _vae, _flow_model

    vae_status = "not found"
    if os.path.exists(VAE_CHECKPOINT):
        _vae = load_vae(VAE_CHECKPOINT)
        vae_status = f"loaded ({VAE_CHECKPOINT})"

    flow_status = "not found"
    if os.path.exists(FLOW_CHECKPOINT):
        _flow_model = VelocityMLP()
        _flow_model.load_state_dict(
            torch.load(FLOW_CHECKPOINT, map_location="cpu", weights_only=True)
        )
        _flow_model.eval()
        flow_status = f"loaded ({FLOW_CHECKPOINT})"

    # Eagerly load latent cache if it already exists (fast, no computation needed)
    _try_load_latent_cache()

    return vae_status, flow_status


# ── Step 10: latent space helpers ─────────────────────────────────────────────

def _try_load_latent_cache() -> bool:
    """Load latent cache from disk if available. Returns True on success."""
    global _latents, _latents_2d, _latent_labels, _pca, _centroids
    if not os.path.exists(_LATENT_CACHE):
        return False
    data = np.load(_LATENT_CACHE, allow_pickle=True)
    _latents = data["latents"]
    _latent_labels = list(data["labels"])
    if "pca_mean" in data and "pca_components" in data:
        _pca = (data["pca_mean"], data["pca_components"])
        _latents_2d = (_latents - _pca[0]) @ _pca[1].T  # [N, 2]
    else:
        _latents_2d = _latents  # fallback: assume already 2D
    _centroids = {k[9:]: data[k] for k in data.files if k.startswith("centroid_")}
    return True


def _compute_latent_space(max_samples: int = 300) -> str:
    """Encode NSynth-valid samples through the VAE, save cache, return status."""
    global _latents, _latent_labels
    if _vae is None:
        return "VAE not loaded — cannot compute latent space."

    data_dir = "data/nsynth-valid"
    if not os.path.exists(data_dir):
        return f"NSynth data not found at {data_dir}."

    from torch.utils.data import DataLoader
    from src.data import NSynthDataset

    dataset = NSynthDataset(
        os.path.join(data_dir, "examples.json"),
        os.path.join(data_dir, "audio"),
        max_samples=max_samples,
        normalize=True,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    all_mu: list[np.ndarray] = []
    all_labels: list[str] = []
    with torch.no_grad():
        for mels, families, _ in loader:
            mu, _ = _vae.encode(mels)
            all_mu.append(mu.numpy())
            all_labels.extend(families)

    _latents = np.concatenate(all_mu, axis=0)       # [N, D]
    _latent_labels = all_labels

    # Fit PCA: project D-dim latents to 2D for visualization
    mean = _latents.mean(axis=0)                     # [D]
    centered = _latents - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:2]                              # [2, D] top-2 principal components
    _pca = (mean, components)
    _latents_2d = centered @ components.T            # [N, 2]

    # Compute per-family centroids for timbre targeting
    family_groups: dict[str, list[int]] = {}
    for i, fam in enumerate(_latent_labels):
        family_groups.setdefault(fam, []).append(i)
    _centroids = {fam: _latents[idxs].mean(axis=0) for fam, idxs in family_groups.items()}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    centroid_kwargs = {f"centroid_{k}": v for k, v in _centroids.items()}
    np.savez(
        _LATENT_CACHE,
        latents=_latents,
        labels=np.array(_latent_labels),
        pca_mean=mean,
        pca_components=components,
        **centroid_kwargs,
    )
    n_centroids = len(_centroids)
    return f"Loaded {len(_latents)} points ({len(set(_latent_labels))} families, {n_centroids} timbre centroids computed)."


def _scrub_plot(scrub_x: float, scrub_y: float):
    """Build the latent scatter with a white cursor dot at (scrub_x, scrub_y)."""
    import matplotlib.pyplot as plt

    if _latents is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(
            0.5, 0.5, 'Click "Load latent space" to begin',
            ha="center", va="center", transform=ax.transAxes, fontsize=12,
        )
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    highlight = np.array([[scrub_x, scrub_y]])
    return latent_scatter(_latents_2d, _latent_labels, highlight=highlight)


def _load_space_handler():
    """Button handler: load (from cache) or compute the latent space."""
    if not _try_load_latent_cache():
        status = _compute_latent_space()
    else:
        status = f"Loaded {len(_latents)} points from cache."
    return _scrub_plot(0.0, 0.0), status


# ── Particle initialisation helpers ───────────────────────────────────────────

def _make_initial_particles() -> torch.Tensor:
    """N_VARIATIONS starting positions sampled from N(0,I).

    Pure noise start is what the flow model was trained on (x0 ~ N(0,I) → x1 ~ data).
    Starting near mu instead produces OOD inputs for the flow model and reduces diversity.
    """
    return torch.randn(N_VARIATIONS, VAE_LATENT_DIM)


def _make_per_particle_styles(mu: torch.Tensor, diversity: float) -> torch.Tensor:
    """Per-particle style vectors: mu + diversity * N(0,I).

    diversity=0   → all 4 particles use identical style (mu) → near-identical outputs.
    diversity>0   → each particle gets a different style attractor → diverse outputs.
    Recommended range: 0.5–2.0.
    """
    noise = torch.randn(N_VARIATIONS, VAE_LATENT_DIM) * diversity
    return mu.expand(N_VARIATIONS, -1).clone() + noise


# ── Core generation ────────────────────────────────────────────────────────────

def _resolve_audio_path(audio_path) -> str:
    """Gradio 6 can pass a dict {path:...} or a plain str — normalise to str."""
    if audio_path is None:
        raise gr.Error("No audio set. Record or upload audio and click 'Use this recording' first.")
    if isinstance(audio_path, dict):
        audio_path = audio_path.get("path") or audio_path.get("name", "")
    if not audio_path or not os.path.exists(str(audio_path)):
        raise gr.Error(f"Audio file not found: {audio_path}")
    return str(audio_path)


def confirm_recording(audio_input):
    """Copy the recorded/uploaded audio into confirmed state, preview, and mel plot."""
    if audio_input is None:
        return None, None, "No audio — record or upload first.", None
    path = _resolve_audio_path(audio_input)
    name = os.path.basename(path)
    try:
        wav = load_audio(path)
        mel = audio_to_mel(wav)
        mel_t = normalize_mel(mel).unsqueeze(0).unsqueeze(0)
        mel_fig = mel_thumbnail(mel_t[0], title="Input mel")
    except Exception:
        mel_fig = None
    return path, path, f"Ready: {name}", mel_fig


def reconstruct_input(audio_path) -> str:
    """Encode → decode through the VAE only, bypassing the flow model.

    This is the best reconstruction possible from the 2D bottleneck and serves
    as a quality baseline: if this already sounds bad the issue is the VAE/vocoder,
    not the flow model.
    """
    if _vae is None:
        raise gr.Error("VAE not loaded.")
    audio_path = _resolve_audio_path(audio_path)
    wav = load_audio(audio_path)
    mel = audio_to_mel(wav)
    mel_t = normalize_mel(mel).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        mu, _ = _vae.encode(mel_t)
    recon_dir = os.path.join(OUTPUT_DIR, "recon")
    os.makedirs(recon_dir, exist_ok=True)
    paths = decode_batch(mu, _vae.decode, output_dir=recon_dir)
    return paths[0]


def generate_variations(
    audio_path,
    style_offset_x: float = 0.0,
    style_offset_y: float = 0.0,
    diversity: float = 1.0,
    target_timbre: str = "Match input",
    timbre_blend: float = 0.5,
) -> tuple:
    """Main handler: input audio → animation GIF + 4 WAV outputs + mel figure."""
    if _vae is None or _flow_model is None:
        raise gr.Error("Models not loaded. Check that VAE and flow checkpoints exist.")

    audio_path = _resolve_audio_path(audio_path)

    # 1. Audio → style
    wav = load_audio(audio_path)
    mel = audio_to_mel(wav)
    mel_t = normalize_mel(mel).unsqueeze(0).unsqueeze(0)  # [1, 1, N_MELS, N_FRAMES]
    with torch.no_grad():
        mu, _ = _vae.encode(mel_t)  # [1, VAE_LATENT_DIM]

    # Step 9: apply style offset
    if style_offset_x != 0.0 or style_offset_y != 0.0:
        offset = torch.tensor([[style_offset_x, style_offset_y]])
        mu = mu + offset

    # Timbre targeting: blend mu toward the chosen instrument family centroid
    if target_timbre != "Match input" and _centroids and target_timbre in _centroids:
        centroid = torch.from_numpy(_centroids[target_timbre]).float().unsqueeze(0)
        mu = (1.0 - timbre_blend) * mu + timbre_blend * centroid
    elif target_timbre != "Match input" and not _centroids:
        print(f"Timbre targeting: centroids not loaded yet — click 'Load latent space' first.")

    # 2. Pure noise start (what the flow model was trained on) + per-particle styles
    x0 = _make_initial_particles()
    styles = _make_per_particle_styles(mu, diversity)  # [N_VARIATIONS, D]
    trajectory = euler_integrate(_flow_model, x0, n_steps=EULER_STEPS, style=styles)
    final_latents = trajectory[-1]  # [N_VARIATIONS, VAE_LATENT_DIM]

    # 3. Decode in background so it runs concurrently with animate_flow
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wav_paths: list[str] = []

    def _decode() -> None:
        paths = decode_batch(final_latents, _vae.decode, output_dir=OUTPUT_DIR)
        wav_paths.extend(paths)

    decode_thread = threading.Thread(target=_decode, daemon=True)
    decode_thread.start()

    # 4. Determine PCA for visualisation — use global if loaded, else fit quickly
    #    on the trajectory itself so the animation always works even before the
    #    user clicks "Load latent space".
    vis_pca = _pca
    if vis_pca is None and VAE_LATENT_DIM > 2:
        traj_np = trajectory.numpy()                          # [steps+1, B, D]
        pts = traj_np.reshape(-1, VAE_LATENT_DIM)
        mean = pts.mean(axis=0)
        centered = pts - mean
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        vis_pca = (mean, Vt[:2])

    # 5. Animate (~3.5s — covers decode_batch runtime)
    gif_path = animate_flow(
        _flow_model, x0, style=mu, pca=vis_pca,
        n_steps=ANIMATION_N_STEPS,
        out_path=os.path.join(OUTPUT_DIR, "flow.gif"),
    )

    # 5. Join decode thread — should already be done; 2s safety margin
    decode_thread.join(timeout=2.0)

    # 6. Mel thumbnail for display
    mel_fig = mel_thumbnail(mel_t[0], title="Input mel")

    return gif_path, *wav_paths, mel_fig


def generate_at_point(scrub_x: float, scrub_y: float) -> str:
    """Step 10: generate one variation at a 2D PCA coordinate.

    The (scrub_x, scrub_y) position is back-projected from 2D PCA space to
    the full latent space before being used as the style vector.
    """
    if _vae is None or _flow_model is None:
        raise gr.Error("Models not loaded.")
    if _pca is not None:
        pca_mean, pca_comp = _pca
        coord = np.array([[scrub_x, scrub_y]])          # [1, 2]
        style_np = coord @ pca_comp + pca_mean          # [1, D]
        style = torch.from_numpy(style_np).float()
    else:
        style = torch.tensor([[scrub_x, scrub_y]])
    x0 = torch.randn(1, VAE_LATENT_DIM)
    traj = euler_integrate(_flow_model, x0, n_steps=EULER_STEPS, style=style)
    scrub_out_dir = os.path.join(OUTPUT_DIR, "scrub")
    os.makedirs(scrub_out_dir, exist_ok=True)
    paths = decode_batch(traj[-1], _vae.decode, output_dir=scrub_out_dir)
    return paths[0]


def update_quiver_preview(t: float) -> object:
    """Gradio slider callback: show velocity field at time t (unconditional)."""
    if _flow_model is None:
        return None
    return render_static_quiver(_flow_model, t, pca=_pca)


# ── UI definition ─────────────────────────────────────────────────────────────

with gr.Blocks(title="Musical Flow Matching") as demo:
    gr.Markdown("# Musical Flow Matching Visualizer")

    confirmed_audio = gr.State(value=None)

    # ── Step 1: Record or upload ──────────────────────────────────────────────
    gr.Markdown("### Step 1 — Record or upload your audio (4–10s)")
    with gr.Row():
        audio_recorder = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            format="wav",
            label="Record or upload",
        )
        with gr.Column():
            confirm_btn = gr.Button("Use this recording ▶", variant="secondary")
            input_status = gr.Textbox(
                value="No audio set — record or upload, then click above.",
                label="Input status",
                interactive=False,
            )

    # ── Step 2: Preview & Generate ────────────────────────────────────────────
    gr.Markdown("### Step 2 — Preview & generate")
    with gr.Row():
        audio_preview = gr.Audio(
            label="Confirmed input (preview here before generating)",
            type="filepath",
            interactive=False,
        )
        mel_plot = gr.Plot(label="Input mel spectrogram")

    confirm_btn.click(
        confirm_recording,
        inputs=[audio_recorder],
        outputs=[confirmed_audio, audio_preview, input_status, mel_plot],
    )

    # Fix A — VAE reconstruction baseline
    gr.Markdown(
        "**Optional:** Reconstruct your input through the VAE only (no flow model). "
        "This is the best quality the decoder can produce — if it sounds bad, "
        "the VAE bottleneck/vocoder is the limiting factor."
    )
    with gr.Row():
        reconstruct_btn = gr.Button("Reconstruct Input (VAE only)", variant="secondary")
        reconstruction_audio = gr.Audio(
            label="VAE Reconstruction (quality baseline)",
            type="filepath",
            interactive=False,
        )

    reconstruct_btn.click(
        reconstruct_input,
        inputs=[confirmed_audio],
        outputs=[reconstruction_audio],
    )

    # Style offsets, diversity, timbre targeting
    with gr.Accordion("Style offsets, diversity & timbre", open=False):
        gr.Markdown(
            "**Style offsets:** shift the input's style vector before generating. "
            "Positive X → brighter timbres; Y shifts pitch character."
        )
        with gr.Row():
            style_x = gr.Slider(-1.0, 1.0, value=0.0, step=0.1, label="Style offset X")
            style_y = gr.Slider(-1.0, 1.0, value=0.0, step=0.1, label="Style offset Y")
        gr.Markdown(
            "**Diversity:** how different each variation sounds from the others. "
            "0 = all 4 are near-identical (strongly anchored to your input). "
            "Higher = more varied outputs but further from original character."
        )
        diversity_slider = gr.Slider(
            0.0, 3.0, value=1.0, step=0.1,
            label="Diversity",
        )
        gr.Markdown(
            "**Target timbre:** blend your input's style toward a specific instrument family. "
            "Requires clicking **Load latent space** first to compute centroids. "
            "Blend = 0 keeps your input's character; Blend = 1 is the pure instrument sound."
        )
        with gr.Row():
            timbre_dropdown = gr.Dropdown(
                choices=["Match input"] + _KNOWN_FAMILIES,
                value="Match input",
                label="Target timbre",
            )
            timbre_blend_slider = gr.Slider(
                0.0, 1.0, value=0.5, step=0.05,
                label="Timbre blend",
            )

    generate_btn = gr.Button("Generate Variations", variant="primary")
    animation_output = gr.Image(label="Particle flow animation", type="filepath")

    with gr.Row():
        variation_outputs = [
            gr.Audio(label=f"Variation {i + 1}", type="filepath")
            for i in range(N_VARIATIONS)
        ]

    # Step 10 — sonic scrubbing grid
    with gr.Accordion("Sound Space Explorer", open=False):
        gr.Markdown(
            "Navigate the learned 2D latent space. "
            "Move the sliders to place the cursor (white dot), then click **Generate at this point**."
        )
        with gr.Row():
            scrub_x = gr.Slider(-3.0, 3.0, value=0.0, step=0.05, label="Latent X (z₀)")
            scrub_y = gr.Slider(-3.0, 3.0, value=0.0, step=0.05, label="Latent Y (z₁)")
        latent_plot = gr.Plot(label="Latent space")
        with gr.Row():
            load_space_btn = gr.Button("Load latent space")
            space_status = gr.Textbox(label="Status", interactive=False, scale=3)
        scrub_generate_btn = gr.Button("Generate at this point")
        scrub_audio = gr.Audio(label="Variation at cursor", type="filepath")

    with gr.Accordion("Velocity field preview", open=False):
        t_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Time t")
        quiver_plot = gr.Plot(label="Velocity field at t")
        t_slider.change(update_quiver_preview, inputs=t_slider, outputs=quiver_plot)

    generate_btn.click(
        generate_variations,
        inputs=[confirmed_audio, style_x, style_y, diversity_slider, timbre_dropdown, timbre_blend_slider],
        outputs=[animation_output, *variation_outputs, mel_plot],
    )

    # Step 10: latent space explorer events
    load_space_btn.click(_load_space_handler, outputs=[latent_plot, space_status])
    scrub_x.change(_scrub_plot, inputs=[scrub_x, scrub_y], outputs=latent_plot)
    scrub_y.change(_scrub_plot, inputs=[scrub_x, scrub_y], outputs=latent_plot)
    scrub_generate_btn.click(
        generate_at_point, inputs=[scrub_x, scrub_y], outputs=scrub_audio
    )


if __name__ == "__main__":
    vae_s, flow_s = _load_models()
    print(f"VAE:  {vae_s}")
    print(f"Flow: {flow_s}")
    demo.launch(theme=gr.themes.Soft())
