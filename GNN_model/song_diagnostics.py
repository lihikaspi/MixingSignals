import torch


def diagnose_embedding_scales(model, sample_items=100, print_values: bool = True):
    """
    Analyze the scale of different embedding components
    to find optimal balance factors
    """
    device = next(model.parameters()).device

    # Sample random items
    sample_idx = torch.randint(0, model.num_items, (sample_items,))

    with torch.no_grad():
        # Get raw embeddings
        audio = model.item_audio_emb[sample_idx].to(device)
        artist = model.artist_emb(model.artist_ids[sample_idx].to(device))
        album = model.album_emb(model.album_ids[sample_idx].to(device))

        # Compute norms
        audio_norm = audio.norm(dim=-1).mean().item()
        artist_norm = artist.norm(dim=-1).mean().item()
        album_norm = album.norm(dim=-1).mean().item()
        metadata_norm = (artist + album).norm(dim=-1).mean().item()

        if print_values:
            print("=" * 60)
            print("EMBEDDING SCALE ANALYSIS")
            print("=" * 60)
            print(f"Audio norm:        {audio_norm:.6f}")
            print(f"Artist norm:       {artist_norm:.6f}")
            print(f"Album norm:        {album_norm:.6f}")
            print(f"Metadata combined: {metadata_norm:.6f}")
            print(f"\nRatio (audio/metadata): {audio_norm / metadata_norm:.2f}x")

        # Recommended scales
        target_audio = 0.5  # We want audio to contribute ~50%
        target_metadata = 0.5  # metadata to contribute ~50%

        rec_audio_scale = target_audio / audio_norm
        rec_metadata_scale = target_metadata / metadata_norm

        if print_values:
            print("\n" + "=" * 60)
            print("RECOMMENDED SCALING FACTORS")
            print("=" * 60)
            print(f"audio_scale = {rec_audio_scale:.4f}")
            print(f"metadata_scale = {rec_metadata_scale:.4f}")

        # Verify
        audio_scaled = (audio * rec_audio_scale).norm(dim=-1).mean().item()
        metadata_scaled = ((artist + album) * rec_metadata_scale).norm(dim=-1).mean().item()

        if print_values:
            print(f"\nAfter scaling:")
            print(f"  Audio contribution:    {audio_scaled:.4f} ({target_audio * 100:.0f}%)")
            print(f"  Metadata contribution: {metadata_scaled:.4f} ({target_metadata * 100:.0f}%)")
            print("=" * 60)

        return rec_audio_scale, rec_metadata_scale

