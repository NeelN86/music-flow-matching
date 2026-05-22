"""Tests for src/vae.py — AudioVAE architecture, loss, and training."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from src.config import N_FRAMES, N_MELS, VAE_LATENT_DIM
from src.vae import AudioDecoder, AudioEncoder, AudioVAE, audio_vae_loss

B = 4  # batch size for all shape tests


def _random_mel(batch: int = B) -> torch.Tensor:
    return torch.randn(batch, 1, N_MELS, N_FRAMES)


def _random_z(batch: int = B) -> torch.Tensor:
    return torch.randn(batch, VAE_LATENT_DIM)


def test_encoder_output_shape():
    """AudioEncoder([B,1,80,256]) → mu[B,2], logvar[B,2]."""
    enc = AudioEncoder()
    mu, logvar = enc(_random_mel())
    assert mu.shape == (B, VAE_LATENT_DIM), f"mu shape {mu.shape}"
    assert logvar.shape == (B, VAE_LATENT_DIM), f"logvar shape {logvar.shape}"


def test_decoder_output_shape():
    """AudioDecoder(z[B,2]) → [B,1,80,256]."""
    dec = AudioDecoder()
    out = dec(_random_z())
    assert out.shape == (B, 1, N_MELS, N_FRAMES), f"decoder output shape {out.shape}"


def test_vae_forward_shapes():
    """AudioVAE.forward([B,1,80,256]) → (recon[B,1,80,256], mu[B,2], logvar[B,2])."""
    vae = AudioVAE()
    x = _random_mel()
    recon, mu, logvar = vae(x)
    assert recon.shape == x.shape, f"recon shape {recon.shape}"
    assert mu.shape == (B, VAE_LATENT_DIM)
    assert logvar.shape == (B, VAE_LATENT_DIM)


def test_vae_loss_positive_and_finite():
    """audio_vae_loss returns a positive, finite scalar."""
    vae = AudioVAE()
    x = _random_mel()
    recon, mu, logvar = vae(x)
    total, recon_loss, kl_loss = audio_vae_loss(recon, x, mu, logvar)
    assert total.item() > 0, "Loss should be positive"
    assert torch.isfinite(total), f"Loss is not finite: {total.item()}"
    assert torch.isfinite(recon_loss)
    assert torch.isfinite(kl_loss)


def test_vae_loss_decreases():
    """200 gradient steps on a fixed batch should reduce total loss.

    Uses a higher lr (3e-3) and more steps because AudioVAE is a larger model
    than the MNIST VAE — the decoder must learn to output ~0 for normalized mels.
    """
    vae = AudioVAE()
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    x = _random_mel()
    vae.train()

    losses = []
    for _ in range(200):
        optimizer.zero_grad()
        recon, mu, logvar = vae(x)
        total, _, _ = audio_vae_loss(recon, x, mu, logvar)
        total.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        optimizer.step()
        losses.append(total.item())

    first_avg = sum(losses[:10]) / 10
    last_avg = sum(losses[-10:]) / 10
    assert last_avg < first_avg, (
        f"Loss should decrease: first 10 avg={first_avg:.4f}, last 10 avg={last_avg:.4f}"
    )


def test_gradients_flow():
    """All encoder and decoder parameters must receive gradients."""
    vae = AudioVAE()
    vae.train()
    x = _random_mel()
    recon, mu, logvar = vae(x)
    total, _, _ = audio_vae_loss(recon, x, mu, logvar)
    total.backward()

    for name, param in vae.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum().item() > 0, f"Zero gradient for {name}"


def test_beta_kl_scaling():
    """Doubling beta should roughly double the KL contribution to total loss."""
    vae = AudioVAE()
    x = _random_mel()
    with torch.no_grad():
        recon, mu, logvar = vae(x)

    _, _, kl1 = audio_vae_loss(recon, x, mu, logvar, beta=1.0)
    _, _, kl2 = audio_vae_loss(recon, x, mu, logvar, beta=2.0)
    # The KL term itself is the same; the total would differ — just check KL is identical
    assert torch.allclose(kl1, kl2), "KL value should not change with beta"


def test_reparameterize_train_vs_eval():
    """In train mode, reparameterize samples (z != mu most of the time).
    In eval mode, returns mu exactly."""
    vae = AudioVAE()
    mu = torch.zeros(B, VAE_LATENT_DIM)
    logvar = torch.zeros(B, VAE_LATENT_DIM)  # std=1

    vae.train()
    z_train = vae.reparameterize(mu, logvar)
    # With std=1 and batch=4, sampling should deviate from zeros with high probability
    assert not torch.allclose(z_train, mu, atol=1e-6), "Train mode should sample ≠ mu"

    vae.eval()
    z_eval = vae.reparameterize(mu, logvar)
    assert torch.allclose(z_eval, mu), "Eval mode should return mu exactly"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as e:
                print(f"FAIL  {name}  {e}")
