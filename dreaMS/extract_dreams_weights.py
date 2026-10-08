"""Extract the weights of DreaMS's embedding model into a plain state dict for dreams_lite.py.

The official checkpoint (embedding_model.ckpt, 1.2 GB) is a PyTorch Lightning file whose metadata refers to classes of
the dreams/msml packages, so it can only be read where they are installed. The extracted file holds tensors only
(~470 MB, float32) and loads with torch alone.

Runs once, in the DreaMS environment, from the dreaMS folder:
    .venv-dreams/bin/python extract_dreams_weights.py
"""

from pathlib import Path

import torch

from dreams_lite import FOURIER_DEPTH, N_LAYERS, DreamsEmbedder, fourier_frequencies

CHECKPOINT_PATH = Path(".venv-dreams/lib/python3.11/site-packages/dreams/models/pretrained/embedding_model.ckpt")
OUTPUT_PATH = Path("dreams_cache/dreams_embedding_weights.pt")


def main():
    official = torch.load(CHECKPOINT_PATH, map_location="cpu")["state_dict"]

    weights = {}
    # The stored frequencies are NOT the formula's values (they moved during the embedding model's fine-tuning,
    # although the pre-training settings mark them as fixed): the model uses the stored ones, so they are copied
    weights["frequencies"] = official["backbone.fourier_enc.b"]
    assert weights["frequencies"].shape == fourier_frequencies().unsqueeze(0).shape

    weights["peak_layer.weight"] = official["backbone.ff_peak.ff.0.weight"]
    weights["peak_layer.bias"] = official["backbone.ff_peak.ff.0.bias"]

    # official ff_fourier.ff = [Linear, Dropout, ReLU] x 4 + [Linear, ReLU]: Linear layers at 0, 3, 6, 9, 12
    # dreams_lite fourier_network = [Linear, ReLU] x 5: Linear layers at 0, 2, 4, 6, 8
    for depth in range(FOURIER_DEPTH):
        for name in ["weight", "bias"]:
            weights[f"fourier_network.{2 * depth}.{name}"] = official[f"backbone.ff_fourier.ff.{3 * depth}.{name}"]

    encoder = "backbone.transformer_encoder"
    for i in range(N_LAYERS):
        weights[f"layers.{i}.qkvo_weights"] = official[f"{encoder}.atts.{i}.weights"]
        weights[f"layers.{i}.feed_forward_in.weight"] = official[f"{encoder}.ffs.{i}.in_proj.weight"]
        weights[f"layers.{i}.feed_forward_out.weight"] = official[f"{encoder}.ffs.{i}.out_proj.weight"]
        for name in ["weight", "bias"]:
            # scales = one LayerNorm before each attention (2i) and feed-forward (2i + 1), then a final one
            weights[f"layers.{i}.attention_norm.{name}"] = official[f"{encoder}.scales.{2 * i}.{name}"]
            weights[f"layers.{i}.feed_forward_norm.{name}"] = official[f"{encoder}.scales.{2 * i + 1}.{name}"]
    for name in ["weight", "bias"]:
        weights[f"final_norm.{name}"] = official[f"{encoder}.scales.{2 * N_LAYERS}.{name}"]
        weights[f"head.{name}"] = official[f"head.{name}"]

    # every tensor of dreams_lite is filled, and only the training-only heads of the checkpoint are left out
    DreamsEmbedder().load_state_dict(weights, strict=True)
    unused = sorted(set(official) - {"backbone.fourier_enc.Fourier frequencies"} - {
        key for key in official if key.startswith(("backbone.ff_peak", "backbone.ff_fourier", encoder, "head."))
    } - {"backbone.fourier_enc.b"})
    print("checkpoint tensors not used (training heads):", unused)

    weights = {name: tensor.float().contiguous() for name, tensor in weights.items()}
    torch.save(weights, OUTPUT_PATH)
    n_parameters = sum(tensor.numel() for tensor in weights.values())
    print(f"saved {len(weights)} tensors, {n_parameters / 1e6:.1f}M numbers, to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
