"""Store original head IDs and reload structurally pruned Wav2Vec2 models."""
import json
from pathlib import Path


def validate_heads(heads, layers, per_layer):
    result = {i: [] for i in range(layers)}
    for key, values in heads.items():
        layer = int(key)
        if layer not in result or not isinstance(values, list):
            raise ValueError("Invalid pruning metadata")
        if any(type(h) is not int or not 0 <= h < per_layer for h in values):
            raise ValueError("Head IDs must be integers within the original layer")
        if len(set(values)) != len(values) or len(values) >= per_layer:
            raise ValueError("Head IDs must be unique and leave at least one head per layer")
        result[layer] = sorted(values)
    return result


def save_checkpoint(model, processor, directory, heads, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    processor.save_pretrained(directory)
    document = {**metadata, "format_version": 1,
                "pruned_heads": {str(k): sorted(v) for k, v in heads.items()}}
    (directory / "pruning.json").write_text(json.dumps(document, indent=2) + "\n")


def load_checkpoint(directory):
    from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Processor
    from safetensors.torch import load_file
    from irhp.pruning import prune_wav2vec2_attention
    directory = Path(directory)
    metadata = json.loads((directory / "pruning.json").read_text())
    if metadata.get("format_version") != 1:
        raise ValueError("Unsupported pruning metadata format")
    cfg = Wav2Vec2Config.from_pretrained(directory, local_files_only=True)
    cfg._attn_implementation = "eager"
    cfg.output_attentions = False
    heads = validate_heads(metadata["pruned_heads"], cfg.num_hidden_layers, cfg.num_attention_heads)
    model = Wav2Vec2ForCTC(cfg)
    prune_wav2vec2_attention(model, heads)
    model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True)
    processor = Wav2Vec2Processor.from_pretrained(directory, local_files_only=True)
    return model, processor, metadata
