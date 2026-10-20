from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import (Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Processor,
                          Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor)
from irhp.pruning import prune_wav2vec2_attention, pad_attention_tensors_to_max
from irhp.checkpoint import save_checkpoint, load_checkpoint, validate_heads


def tiny_model():
    cfg = Wav2Vec2Config(hidden_size=16, num_attention_heads=4, num_hidden_layers=2,
                        intermediate_size=32, conv_dim=(8, 8, 8), conv_stride=(2, 2, 2),
                        conv_kernel=(3, 3, 3), num_conv_pos_embedding_groups=2,
                        num_conv_pos_embeddings=16, vocab_size=8,
                        mask_time_prob=0, layerdrop=0, attention_dropout=0)
    cfg._attn_implementation = "eager"
    return Wav2Vec2ForCTC(cfg).eval()


class PruningTests(unittest.TestCase):
    def test_structural_pruning_matches_zeroed_attention_heads(self):
        torch.manual_seed(7)
        dense = tiny_model()
        pruned = deepcopy(dense)
        hidden = torch.randn(2, 12, 16)
        original_attention = dense.wav2vec2.encoder.layers[0].attention
        with torch.no_grad():
            expected = original_attention(hidden, layer_head_mask=torch.tensor([1., 0., 1., 0.]))[0]
            prune_wav2vec2_attention(pruned, {0: [1, 3]})
            actual = pruned.wav2vec2.encoder.layers[0].attention(hidden)[0]
        torch.testing.assert_close(actual, expected)
        self.assertEqual(pruned.wav2vec2.encoder.layers[0].attention.q_proj.out_features, 8)

    def test_repeated_pruning_uses_original_head_ids(self):
        torch.manual_seed(7)
        direct = tiny_model()
        repeated = deepcopy(direct)
        prune_wav2vec2_attention(direct, {0: [1, 2, 3]})
        prune_wav2vec2_attention(repeated, {0: [1, 3]})
        prune_wav2vec2_attention(repeated, {0: [2]})
        for key, value in direct.state_dict().items():
            torch.testing.assert_close(value, repeated.state_dict()[key])

    def test_padding_restores_original_head_positions(self):
        cfg = tiny_model().config
        maps = [torch.ones(2, 3, 3), torch.full((3, 3, 3), 2.)]
        result = pad_attention_tensors_to_max(maps, cfg, {0: {1, 3}, 1: {0}})
        self.assertEqual(tuple(result.shape), (2, 4, 3, 3))
        self.assertEqual(result[0, 1].sum().item(), 0)
        self.assertEqual(result[1, 0].sum().item(), 0)
        self.assertEqual(result[1, 3].mean().item(), 2.)

    def test_checkpoint_roundtrip_preserves_logits_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "A": 5, "B": 6, "C": 7}
            (path / "vocab.json").write_text(json.dumps(vocab))
            processor = Wav2Vec2Processor(
                Wav2Vec2FeatureExtractor(sampling_rate=16000, return_attention_mask=True),
                Wav2Vec2CTCTokenizer(str(path / "vocab.json"), word_delimiter_token="|"))
            model = tiny_model()
            heads = {0: [1, 3], 1: [0]}
            prune_wav2vec2_attention(model, heads)
            audio = torch.randn(2, 800)
            with torch.no_grad():
                expected = model(audio).logits
            save_checkpoint(model, processor, path / "export", heads, {"test_fixture": True})
            restored, _, metadata = load_checkpoint(path / "export")
            restored.eval()
            with torch.no_grad():
                actual = restored(audio).logits
            torch.testing.assert_close(actual, expected)
            self.assertEqual(metadata["pruned_heads"]["0"], [1, 3])

    def test_invalid_masks_are_rejected(self):
        for heads in ({0: [0, 1, 2, 3]}, {0: [1, 1]}, {0: [4]}, {5: [0]}):
            with self.subTest(heads=heads), self.assertRaises(ValueError):
                validate_heads(heads, 2, 4)
