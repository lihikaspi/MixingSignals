"""
Run this before training to find optimal scaling factors
"""
import torch
import numpy as np


def diagnose_embedding_scales(model, sample_items=100):
    """
    Analyze the scale of different embedding components
    to find optimal balance factors
    """
    device = next(model.parameters()).device

    # Sample random items
    sample_idx = torch.randint(0, model.num_items, (sample_items,))

    with torch.no_grad():
        # Get raw embeddings
        embedding_layer = model.embedding_layer

        audio = embedding_layer.item_audio_emb[sample_idx].to(device)
        artist = embedding_layer.artist_emb(embedding_layer.artist_ids[sample_idx].to(device))
        album = embedding_layer.album_emb(embedding_layer.album_ids[sample_idx].to(device))

        # Compute norms
        audio_norm = audio.norm(dim=-1).mean().item()
        artist_norm = artist.norm(dim=-1).mean().item()
        album_norm = album.norm(dim=-1).mean().item()
        metadata_norm = (artist + album).norm(dim=-1).mean().item()

        print("=" * 60)
        print("EMBEDDING SCALE ANALYSIS")
        print("=" * 60)
        print(f"Audio norm:        {audio_norm:.6f}")
        print(f"Artist norm:       {artist_norm:.6f}")
        print(f"Album norm:        {album_norm:.6f}")
        print(f"Metadata combined: {metadata_norm:.6f}")
        print(f"\nRatio (audio/metadata): {audio_norm / metadata_norm:.2f}x")

        # Recommended scales
        target_audio = 0.5  # We want audio to contribute ~30%
        target_metadata = 0.5  # metadata to contribute ~70%

        rec_audio_scale = target_audio / audio_norm
        rec_metadata_scale = target_metadata / metadata_norm

        print("\n" + "=" * 60)
        print("RECOMMENDED SCALING FACTORS")
        print("=" * 60)
        print(f"audio_scale = {rec_audio_scale:.4f}")
        print(f"metadata_scale = {rec_metadata_scale:.4f}")

        # Verify
        audio_scaled = (audio * rec_audio_scale).norm(dim=-1).mean().item()
        metadata_scaled = ((artist + album) * rec_metadata_scale).norm(dim=-1).mean().item()

        print(f"\nAfter scaling:")
        print(f"  Audio contribution:    {audio_scaled:.4f} ({target_audio * 100:.0f}%)")
        print(f"  Metadata contribution: {metadata_scaled:.4f} ({target_metadata * 100:.0f}%)")
        print("=" * 60)

        return rec_audio_scale, rec_metadata_scale

# Usage in your training script:
# Before training, run:
# audio_scale, metadata_scale = diagnose_embedding_scales(model)
# Then update your model:
# model.audio_scale = audio_scale
# model.metadata_scale = metadata_scale