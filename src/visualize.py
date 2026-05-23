# Quiver + particle animation and mel spectrogram thumbnails.
#
# animate_flow is a direct port of flow-matching-visualizer/src/visualize.py
# with a `style` parameter added. The three-layer zorder convention is unchanged:
#   layer 1 (zorder=1, alpha=0.4): quiver velocity field (background)
#   layer 2 (zorder=2):            fading trail segments (LineCollection)
#   layer 3 (zorder=3):            particles (white scatter, foreground)
#
# Key timing: n_steps=ANIMATION_N_STEPS (70) at FPS=20 → exactly 3.5s of animation,
# long enough for decode_batch to finish in the background thread.

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import numpy as np
import torch
from torch import Tensor

from src.config import (
    ANIMATION_FPS,
    ANIMATION_N_STEPS,
    OUTPUT_DIR,
)
from src.model import VelocityMLP
from src.solvers import euler_integrate, velocity_field_on_grid


def animate_flow(
    model: VelocityMLP,
    x0: Tensor,
    style: Tensor | None = None,
    n_steps: int = ANIMATION_N_STEPS,
    bounds: tuple[float, float, float, float] = (-3.0, 3.0, -3.0, 3.0),
    grid_res: int = 20,
    trail_len: int = 12,
    out_path: str = "outputs/flow.gif",
    fps: int = ANIMATION_FPS,
) -> str:
    """Animate particles flowing through the style-conditioned velocity field.

    Pre-computes the full trajectory and all quiver grids before the animation
    starts, so the FuncAnimation callback is lightweight (just array swaps).

    Args:
        model:     Trained VelocityMLP.
        x0:        [B, 2] initial noise positions.
        style:     [1, 2] or [B, 2] VAE mu conditioning. None → unconditional.
        n_steps:   Integration steps = animation frames.
        bounds:    (x_min, x_max, y_min, y_max).
        grid_res:  Quiver grid resolution.
        trail_len: Number of past frames shown per particle.
        out_path:  GIF output path.
        fps:       Playback FPS.

    Returns:
        Absolute path to the saved GIF.
    """
    model.eval()

    # Pre-compute full trajectory [n_steps+1, B, 2]
    traj = euler_integrate(model, x0, n_steps=n_steps, style=style).numpy()
    B = x0.shape[0]

    # Pre-compute quiver grids for every frame (keeps animation callback lightweight)
    quiver_frames: list[tuple] = []
    with torch.no_grad():
        for i in range(n_steps + 1):
            t = i / n_steps
            X, Y, U, V = velocity_field_on_grid(model, t, bounds, grid_res, style=style)
            mag = np.hypot(U.numpy(), V.numpy())
            quiver_frames.append((X.numpy(), Y.numpy(), U.numpy(), V.numpy(), mag))

    all_mags = np.stack([f[4] for f in quiver_frames])
    vmin, vmax = float(all_mags.min()), float(all_mags.max())

    # Figure setup
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    # Layer 1: quiver field (background)
    X0, Y0, U0, V0, mag0 = quiver_frames[0]
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    quiv = ax.quiver(
        X0, Y0, U0, V0, mag0,
        cmap="plasma", norm=norm, alpha=0.4, zorder=1,
        angles="xy", scale_units="xy", scale=1.5,
    )

    # Layer 2: trail LineCollection
    lc = LineCollection([], linewidths=1.0, zorder=2)
    ax.add_collection(lc)

    # Layer 3: particle scatter (foreground)
    scat = ax.scatter(traj[0, :, 0], traj[0, :, 1], s=10, c="white", zorder=3, linewidths=0)
    title = ax.set_title("t = 0.00", color="white", pad=8)

    def _update(frame: int):
        # 1. Quiver
        _, _, U, V, mag = quiver_frames[frame]
        quiv.set_UVC(U, V, mag)

        # 2. Trails — alpha ramps oldest→0 newest→0.7
        segments: list[np.ndarray] = []
        rgba: list[tuple] = []
        start = max(0, frame - trail_len)
        window = traj[start: frame + 1]  # [k, B, 2]
        k = window.shape[0]
        if k >= 2:
            alphas = np.linspace(0.0, 0.7, k - 1)
            for j in range(B):
                pts = window[:, j, :]
                segs = np.stack([pts[:-1], pts[1:]], axis=1)  # [k-1, 2, 2]
                segments.extend(segs)
                rgba.extend((0.55, 0.78, 1.0, float(a)) for a in alphas)
        lc.set_segments(segments)
        lc.set_color(rgba)

        # 3. Particles
        scat.set_offsets(traj[frame])
        title.set_text(f"t = {frame / n_steps:.2f}")
        return quiv, lc, scat, title

    anim = FuncAnimation(fig, _update, frames=n_steps + 1, interval=1000 // fps, blit=False)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    anim.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)
    return os.path.abspath(out_path)


def render_static_quiver(
    model: VelocityMLP,
    t: float,
    style: Tensor | None = None,
    bounds: tuple[float, float, float, float] = (-3.0, 3.0, -3.0, 3.0),
    res: int = 25,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Render velocity field at a fixed time t. Used for Gradio slider preview."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("#111111")
    else:
        fig = ax.figure

    X, Y, U, V = velocity_field_on_grid(model, t, bounds, res, style=style)
    X, Y = X.numpy(), Y.numpy()
    U, V = U.numpy(), V.numpy()
    mag = np.hypot(U, V)

    ax.set_facecolor("#111111")
    ax.quiver(X, Y, U, V, mag, cmap="plasma", alpha=0.4, zorder=1)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal")
    ax.set_title(f"velocity field   t = {t:.2f}", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    return fig


def mel_thumbnail(
    mel_normalized: Tensor,
    title: str = "",
    figsize: tuple[float, float] = (4, 2),
) -> plt.Figure:
    """Render a normalized mel spectrogram as a matplotlib figure.

    Denormalizes before display. Used for the input audio panel in Gradio
    and in scripts/eyeball_reconstruct.py.

    Args:
        mel_normalized: [1, N_MELS, N_FRAMES] or [N_MELS, N_FRAMES] tensor.
        title:          Figure title.

    Returns:
        Figure with a single imshow (viridis colormap, origin='lower').
    """
    from src.data import denormalize_mel

    mel = mel_normalized
    if mel.ndim == 3:
        mel = mel[0]  # [N_MELS, N_FRAMES]

    mel_db = denormalize_mel(mel).numpy()

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(mel_db, origin="lower", aspect="auto", cmap="viridis")
    if title:
        ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mel bin")
    fig.tight_layout()
    return fig


def latent_scatter(
    latents: np.ndarray,
    labels: list[str],
    highlight: np.ndarray | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Scatter plot of 2D latent points colored by label string.

    Used by scripts/eyeball_latent.py and optionally the Gradio UI.

    Args:
        latents:   [N, 2] array of (x, y) latent positions.
        labels:    Length-N list of label strings (NSynth instrument families).
        highlight: [M, 2] optional extra points shown in white (e.g. query audio).
        ax:        Existing axes; creates new figure if None.

    Returns:
        matplotlib Figure.
    """
    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))
    color_map = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    label_arr = np.asarray(labels)
    for lbl in unique_labels:
        mask = label_arr == lbl
        ax.scatter(
            latents[mask, 0], latents[mask, 1],
            c=[color_map[lbl]], label=lbl, s=12, alpha=0.7, linewidths=0,
        )

    if highlight is not None:
        ax.scatter(
            highlight[:, 0], highlight[:, 1],
            c="white", s=60, zorder=5, edgecolors="black", linewidths=0.8,
        )

    ax.legend(loc="upper right", fontsize=8, markerscale=2, framealpha=0.7)
    ax.set_xlabel("z₀")
    ax.set_ylabel("z₁")
    ax.set_title("Latent Space — instrument families")
    fig.tight_layout()
    return fig
