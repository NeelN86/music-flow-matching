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

import os
import threading

import gradio as gr
import torch

from src.config import (
    ANIMATION_N_STEPS,
    EULER_STEPS,
    FLOW_CHECKPOINT,
    N_VARIATIONS,
    OUTPUT_DIR,
    VAE_CHECKPOINT,
)
from src.data import audio_to_mel, load_audio, normalize_mel
from src.model import VelocityMLP
from src.solvers import euler_integrate
from src.vae import AudioVAE, load_vae
from src.visualize import animate_flow, mel_thumbnail, render_static_quiver
from src.vocoder import decode_batch

# Models loaded once at startup
_vae: AudioVAE | None = None
_flow_model: VelocityMLP | None = None


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

    return vae_status, flow_status


def generate_variations(audio_path: str) -> tuple:
    """Main handler: input audio → animation GIF + 4 WAV outputs + mel figure.

    Threading model (see module docstring):
      1. Encode input audio with VAE → style mu [1, 2]
      2. Euler integrate 4 noise samples → trajectory [51, 4, 2]
      3. Kick off decode_batch in a background thread
      4. animate_flow in main thread (~3.5s)
      5. join decode thread (should already be done)
      6. Return gif_path, 4×wav_path, mel_figure
    """
    if _vae is None or _flow_model is None:
        raise gr.Error("Models not loaded. Check that VAE and flow checkpoints exist.")

    # 1. Audio → style
    wav = load_audio(audio_path)
    mel = audio_to_mel(wav)
    mel_t = normalize_mel(mel).unsqueeze(0).unsqueeze(0)  # [1, 1, N_MELS, N_FRAMES]
    with torch.no_grad():
        mu, _ = _vae.encode(mel_t)  # [1, 2] — the style conditioning vector

    # 2. Integrate N_VARIATIONS noise particles
    x0 = torch.randn(N_VARIATIONS, 2)
    trajectory = euler_integrate(_flow_model, x0, n_steps=EULER_STEPS, style=mu)
    final_latents = trajectory[-1]  # [N_VARIATIONS, 2]

    # 3. Decode in background so it runs concurrently with animate_flow
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wav_paths: list[str] = []

    def _decode() -> None:
        paths = decode_batch(final_latents, _vae.decode, output_dir=OUTPUT_DIR)
        wav_paths.extend(paths)

    decode_thread = threading.Thread(target=_decode, daemon=True)
    decode_thread.start()

    # 4. Animate (~3.5s — covers decode_batch runtime)
    gif_path = animate_flow(
        _flow_model, x0, style=mu,
        n_steps=ANIMATION_N_STEPS,
        out_path=os.path.join(OUTPUT_DIR, "flow.gif"),
    )

    # 5. Join decode thread — should already be done; 2s safety margin
    decode_thread.join(timeout=2.0)

    # 6. Mel thumbnail for display
    mel_fig = mel_thumbnail(mel_t[0], title="Input mel")

    return gif_path, *wav_paths, mel_fig


def update_quiver_preview(t: float) -> object:
    """Gradio slider callback: show velocity field at time t (unconditional)."""
    if _flow_model is None:
        return None
    return render_static_quiver(_flow_model, t)


# ── UI definition ─────────────────────────────────────────────────────────────

with gr.Blocks(title="Musical Flow Matching", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Musical Flow Matching Visualizer")
    gr.Markdown(
        "Upload or record a short melody (4–10s). "
        "The app generates 4 audio variations while showing particles "
        "flow through the learned sound space."
    )

    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Input audio",
        )
        mel_plot = gr.Plot(label="Input mel spectrogram")

    generate_btn = gr.Button("Generate Variations", variant="primary")
    animation_output = gr.Image(label="Particle flow animation", type="filepath")

    with gr.Row():
        variation_outputs = [
            gr.Audio(label=f"Variation {i + 1}", type="filepath")
            for i in range(N_VARIATIONS)
        ]

    with gr.Accordion("Velocity field preview", open=False):
        t_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Time t")
        quiver_plot = gr.Plot(label="Velocity field at t")
        t_slider.change(update_quiver_preview, inputs=t_slider, outputs=quiver_plot)

    generate_btn.click(
        generate_variations,
        inputs=audio_input,
        outputs=[animation_output, *variation_outputs, mel_plot],
    )


if __name__ == "__main__":
    vae_s, flow_s = _load_models()
    print(f"VAE:  {vae_s}")
    print(f"Flow: {flow_s}")
    demo.launch()
