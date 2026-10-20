"""Historical Conformer experiment; see docs/conformer.md before use."""
from pathlib import Path
import re
import os
import torch
import tempfile
import shutil
import json
import argparse
import torch.nn as nn
from ruamel.yaml import YAML
import nemo.collections.asr as nemo_asr
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from datasets import load_dataset, DownloadConfig, config
import aiohttp
from omegaconf import OmegaConf
from copy import deepcopy
from nemo.utils.app_state import AppState
from torch.utils.data import DataLoader
import glob
import zipfile
import torch.nn.functional as F
from transformers import Wav2Vec2Config, Wav2Vec2Processor
from irhp.conformer.pruning import get_avg_attention_matrices, cluster_and_select_heads, prune_conformer_attention, find_remaining_heads, find_heads_to_keep, compute_pairwise_distances, get_token_based_distance, build_prune_dict
import numpy as np
from sklearn.cluster import SpectralClustering
from irhp.conformer.pruning import prune_conformer_attention
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from dataclasses import dataclass
from typing import Union, Optional
from transformers import Wav2Vec2FeatureExtractor
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

@dataclass
class ConformerCTCDataCollator:
    feature_extractor: Wav2Vec2FeatureExtractor
    tokenizer: SentencePieceTokenizer
    padding: Union[bool, str] = True
    sampling_rate: int = 16000

    def __call__(self, features):
        batch_audio = [f['audio']['array'] for f in features]
        inputs = self.feature_extractor(batch_audio, sampling_rate=self.sampling_rate, return_tensors='pt', padding=True, return_attention_mask=True)
        input_signal = inputs['input_values'].float()
        input_signal_length = inputs['attention_mask'].sum(dim=-1)
        raw_texts = [f['text'] for f in features]
        ids_list = [self.tokenizer.text_to_ids(text) for text in raw_texts]
        batch_size = len(ids_list)
        max_len = max((len(ids) for ids in ids_list))
        pad_id = self.tokenizer.pad_id
        labels = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        for i, ids in enumerate(ids_list):
            labels[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels = labels.masked_fill(labels == pad_id, -100)
        label_lengths = (labels != -100).sum(-1)
        return (input_signal, input_signal_length, labels, label_lengths)

def build_manifest_from_hf(ds, manifest_path: str, cache_dir: str):
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    extract_root = os.path.join(cache_dir, 'extracted')
    with open(manifest_path, 'w') as fout:
        for sample in ds:
            audio = sample['audio']
            orig_path = audio['path']
            if not os.path.isfile(orig_path):
                filename = os.path.basename(orig_path)
                pattern = os.path.join(extract_root, '**', filename)
                matches = glob.glob(pattern, recursive=True)
                if not matches:
                    raise FileNotFoundError(f'Audio file not found: {filename} \nOriginal path: {orig_path}\nSearch pattern: {pattern}')
                orig_path = matches[0]
            duration = len(audio['array']) / audio['sampling_rate']
            entry = {'audio_filepath': orig_path, 'duration': duration, 'text': sample['text'].lower().strip()}
            fout.write(json.dumps(entry, ensure_ascii=False) + '\n')

def release_nemoAPI(model):
    meta = AppState().get_model_metadata_from_guid(model.model_guid)
    nemo_file = meta.restoration_path
    connector = SaveRestoreConnector()
    archive = tempfile.mkdtemp(prefix='irhp-nemo-')
    connector._unpack_nemo_file(nemo_file, out_folder=archive)
    model._save_restore_connector.model_extracted_dir = archive
    AppState().nemo_file_folder = archive

def save_weights_only_nemo(model, checkpoint_path, save_path):
    pack_dir = tempfile.mkdtemp()
    try:
        cfg_file = os.path.join(pack_dir, 'model_config.yaml')
        OmegaConf.save(config=model.cfg, f=cfg_file)
        ckpt_name = os.path.basename(checkpoint_path)
        shutil.copy(checkpoint_path, os.path.join(pack_dir, ckpt_name))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with zipfile.ZipFile(save_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(pack_dir):
                for fname in files:
                    fullpath = os.path.join(root, fname)
                    relpath = os.path.relpath(fullpath, pack_dir)
                    zf.write(fullpath, relpath)
        print(f'✅ Saved weights-only .nemo to {save_path}')
    finally:
        shutil.rmtree(pack_dir)

def setup_nemo_datasets_and_cfg(model, train_ds, val_ds, test_ds, collator, train_manifest, val_manifest, test_manifest, args):
    train_cfg = model.cfg.train_ds
    if isinstance(train_cfg, dict):
        is_tarred = train_cfg.get('is_tarred', False)
    else:
        is_tarred = getattr(train_cfg, 'is_tarred', False)
    if is_tarred:
        if isinstance(train_cfg, dict):
            train_cfg['is_tarred'] = False
            train_cfg['tarred_audio_filepaths'] = None
        else:
            train_cfg.is_tarred = False
            train_cfg.tarred_audio_filepaths = None
    if isinstance(train_cfg, dict):
        train_cfg['manifest_filepath'] = train_manifest
        train_cfg['sample_rate'] = args.data_sample_rate
        train_cfg['batch_size'] = args.batch_size
    else:
        train_cfg.manifest_filepath = train_manifest
        train_cfg.sample_rate = args.data_sample_rate
        train_cfg.batch_size = args.batch_size
    val_cfg = model.cfg.validation_ds
    if isinstance(val_cfg, dict):
        is_tarred = val_cfg.get('is_tarred', False)
    else:
        is_tarred = getattr(val_cfg, 'is_tarred', False)
    if is_tarred:
        if isinstance(val_cfg, dict):
            val_cfg['is_tarred'] = False
            val_cfg['tarred_audio_filepaths'] = None
        else:
            val_cfg.is_tarred = False
            val_cfg.tarred_audio_filepaths = None
    if isinstance(val_cfg, dict):
        val_cfg['manifest_filepath'] = val_manifest
        val_cfg['sample_rate'] = args.data_sample_rate
        val_cfg['batch_size'] = args.batch_size
    else:
        val_cfg.manifest_filepath = val_manifest
        val_cfg.sample_rate = args.data_sample_rate
        val_cfg.batch_size = args.batch_size
    test_cfg = model.cfg.test_ds
    if isinstance(test_cfg, dict):
        is_tarred = test_cfg.get('is_tarred', False)
    else:
        is_tarred = getattr(test_cfg, 'is_tarred', False)
    if is_tarred:
        if isinstance(test_cfg, dict):
            test_cfg['is_tarred'] = False
            test_cfg['tarred_audio_filepaths'] = None
        else:
            test_cfg.is_tarred = False
            test_cfg.tarred_audio_filepaths = None
    if isinstance(test_cfg, dict):
        test_cfg['manifest_filepath'] = test_manifest
        test_cfg['sample_rate'] = args.data_sample_rate
        test_cfg['batch_size'] = args.batch_size
    else:
        test_cfg.manifest_filepath = test_manifest
        test_cfg.sample_rate = args.data_sample_rate
        test_cfg.batch_size = args.batch_size
    model.train_dataset = train_ds
    model.val_dataset = val_ds
    model.test_dataset = test_ds
    model.batch_size = args.batch_size
    model.data_collator = collator
    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)
    model.setup_test_data(model.cfg.test_ds)

class My_EncDecCTCModelBPE(nemo_asr.models.EncDecCTCModelBPE):

    def __init__(self, cfg, trainer=None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.data_collator = None

    @classmethod
    def from_pretrained(cls, *args, train_ds=None, val_ds=None, test_ds=None, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        if train_ds is not None:
            model.train_dataset = train_ds
        if val_ds is not None:
            model.val_dataset = val_ds
        if test_ds is not None:
            model.test_dataset = test_ds
        return model

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=self.data_collator)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=self.data_collator)

    def test_dataloader(self):
        if hasattr(self, '_test_dl') and self._test_dl is not None:
            return self._test_dl
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=self.data_collator)

def load_datasets(dataset_name):
    if dataset_name == 'librispeech':
        train_dataset = load_dataset('openslr/librispeech_asr', 'clean', split='train.100', trust_remote_code=True)
        dev_dataset = load_dataset('openslr/librispeech_asr', 'clean', split='validation', trust_remote_code=True)
        test_dataset = load_dataset('openslr/librispeech_asr', 'clean', split='test', trust_remote_code=True)
    elif dataset_name == 'tedlium':
        train_dataset = load_dataset(str(Path(__file__).resolve().parents[2] / 'data_scripts' / 'tedlium_test.py'), 'release1', split='train', trust_remote_code=True)
        dev_dataset = load_dataset(str(Path(__file__).resolve().parents[2] / 'data_scripts' / 'tedlium_test.py'), 'release1', split='validation', trust_remote_code=True)
        test_dataset = load_dataset(str(Path(__file__).resolve().parents[2] / 'data_scripts' / 'tedlium_test.py'), 'release1', split='test', trust_remote_code=True)
        MAX_SAMPLES = 320000
        train_dataset = train_dataset.filter(lambda x: x['audio']['array'].shape[0] <= MAX_SAMPLES)
        dev_dataset = dev_dataset.filter(lambda x: x['audio']['array'].shape[0] <= MAX_SAMPLES)
        test_dataset = test_dataset.filter(lambda x: x['audio']['array'].shape[0] <= MAX_SAMPLES)
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return (train_dataset, dev_dataset, test_dataset)

def preprocess_function(batch, processor):
    input_values = processor(batch['audio']['array'], sampling_rate=16000).input_values[0]
    batch['text'] = clean_transcript(batch['text'])
    labels = processor.tokenizer(batch['text']).input_ids
    return {'input_values': input_values, 'labels': labels}

def clean_transcript(text):
    text = re.sub('\\{.*?\\}', '', text)
    text = re.sub('\\(.*?\\)', '', text)
    text = text.replace('<sil>', '')
    text = text.upper()
    text = re.sub("[^A-Z' ]+", '', text)
    text = re.sub('\\s+', ' ', text).strip()
    return text

def main():
    parser = argparse.ArgumentParser(description='Train halved-dimension Conformer CTC student on LibriSpeech 100h')
    parser.add_argument('--data_dir', type=str, default='data', help='Data dir.')
    parser.add_argument('--config', type=str, default='configs/conformer_ctc_bpe.yaml', help='Config.')
    parser.add_argument('--gpus', type=int, default=1, help='Gpus.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--data_sample_rate', type=int, default=16000, help='Data sample rate.')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output dir.')
    parser.add_argument('--data_script_path', type=str, default=str(Path(__file__).resolve().parents[2] / 'data_scripts' / 'librispeech_asr.py'), help='Data script path.')
    parser.add_argument('--data_config_name', type=str, default='train_100', help='Data config name.')
    parser.add_argument('--data_train_split', type=str, default='train.clean.100', help='Data train split.')
    parser.add_argument('--data_val_split', type=str, default='dev.clean', help='Data val split.')
    parser.add_argument('--data_test_split', type=str, default='test.clean', help='Data test split.')
    parser.add_argument('--train_teacher_model', type=bool, default=False, help='Train teacher model.')
    parser.add_argument('--logit_distillation', type=bool, default=False, help='Logit distillation.')
    parser.add_argument('--kd_alpha', type=float, default=1.0, help='Kd alpha.')
    parser.add_argument('--kd_temperature', type=float, default=1.0, help='Kd temperature.')
    parser.add_argument('--layerwise_distillation', type=bool, default=False, help='Layerwise distillation.')
    parser.add_argument('--layer_kd_alpha', type=float, default=1.0, help='Layer kd alpha.')
    parser.add_argument('--test_mode', action='store_true', help='Test mode.')
    parser.add_argument('--iterative_finetune_epochs', type=int, default=2, help='Iterative finetune epochs.')
    parser.add_argument('--final_finetune_epochs', type=int, default=100, help='Final finetune epochs.')
    parser.add_argument('--method', type=str, default='redundancy-based', choices=['redundancy-based', 'magnitude-based', 'l0-based', 'baseline', 'one-shot'], help='Method.')
    parser.add_argument('--prune_ratio', type=float, default=0.8, help='Prune ratio.')
    parser.add_argument('--iterations', type=int, default=10, help='Iterations.')
    parser.add_argument('--distance_metric', type=str, default='cosine', help='Distance metric.')
    parser.add_argument('--dataset_name', type=str, default='librispeech', help='[librispeech, tedlium]')
    parser.add_argument('--model_name_or_path', type=str, default='nvidia/stt_en_conformer_ctc_small', help='Model name or path.')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    manifest_dir = os.path.join(args.data_dir, f'manifests_{args.dataset_name}_{args.data_config_name}')
    os.makedirs(manifest_dir, exist_ok=True)
    train_manifest = os.path.join(manifest_dir, 'train.json')
    val_manifest = os.path.join(manifest_dir, 'val.json')
    test_manifest = os.path.join(manifest_dir, 'test.json')
    print('Datasets cache dir:', config.HF_DATASETS_CACHE)
    cache_dir = os.path.join(args.data_dir, 'cache')
    config.HF_DATASETS_CACHE = cache_dir
    dl_cfg = DownloadConfig(cache_dir=cache_dir, force_download=False, resume_download=True, max_retries=10, disable_tqdm=False, download_desc='Downloading LibriSpeech ASR', storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=10800)}}, delete_extracted=False, extract_compressed_file=True, force_extract=True)
    if args.dataset_name == 'librispeech':
        train_ds = load_dataset(args.data_script_path, args.data_config_name, split=args.data_train_split, trust_remote_code=True, download_config=dl_cfg, cache_dir=cache_dir)
        val_ds = load_dataset(args.data_script_path, args.data_config_name, split=args.data_val_split, trust_remote_code=True, download_config=dl_cfg, cache_dir=cache_dir)
        test_ds = load_dataset(args.data_script_path, args.data_config_name, split=args.data_test_split, trust_remote_code=True, download_config=dl_cfg, cache_dir=cache_dir)
    elif args.dataset_name == 'tedlium':
        train_ds, val_ds, test_ds = load_datasets(dataset_name=args.dataset_name)
        wav2vec2config = Wav2Vec2Config.from_pretrained('facebook/wav2vec2-base-100h', output_attentions=True)
        processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-100h', config=wav2vec2config)
        if args.test_mode:
            train_ds = train_ds.select(range(300))
            val_ds = val_ds.select(range(300))
            test_ds = test_ds.select(range(300))
        train_ds = train_ds.map(preprocess_function, fn_kwargs={'processor': processor}, num_proc=4)
        val_ds = val_ds.map(preprocess_function, fn_kwargs={'processor': processor}, num_proc=4)
        test_ds = test_ds.map(preprocess_function, fn_kwargs={'processor': processor}, num_proc=4)
    eval_datasets = {'dev': val_ds, 'test': test_ds}
    print(f'train_ds.cache_files: {train_ds.cache_files}')
    print('building manifest files...')
    if not os.path.isfile(train_manifest):
        build_manifest_from_hf(train_ds, train_manifest, cache_dir)
        print(f'train_manifest DONE: {train_manifest}')
    if not os.path.isfile(val_manifest):
        build_manifest_from_hf(val_ds, val_manifest, cache_dir)
        print(f'val_manifest DONE: {val_manifest}')
    if not os.path.isfile(test_manifest):
        build_manifest_from_hf(test_ds, test_manifest, cache_dir)
        print(f'test_manifest DONE: {test_manifest}')
    print('manifest files built.')
    if args.test_mode:
        train_ds = train_ds.select(range(300))
        val_ds = val_ds.select(range(300))
        test_ds = test_ds.select(range(300))
        args.iterative_finetune_epochs = 2
        args.final_finetune_epochs = 5
        test_train_manifest = os.path.join(manifest_dir, 'train_testmode.json')
        test_val_manifest = os.path.join(manifest_dir, 'val_testmode.json')
        test_test_manifest = os.path.join(manifest_dir, 'test_testmode.json')
        build_manifest_from_hf(train_ds, test_train_manifest, cache_dir)
        build_manifest_from_hf(val_ds, test_val_manifest, cache_dir)
        build_manifest_from_hf(test_ds, test_test_manifest, cache_dir)
        train_manifest = test_train_manifest
        val_manifest = test_val_manifest
        test_manifest = test_test_manifest
    prj_name = os.getenv('PRJ_NAME')
    exp_name = os.getenv('EXP_NAME')
    iterative_checkpoint_callback = ModelCheckpoint(dirpath=args.output_dir, filename='best', save_top_k=1, verbose=True, monitor='val_loss', mode='min')
    last_checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(args.output_dir, 'final_finetune'), filename='last', save_top_k=0, verbose=True, save_last=True)
    best_checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(args.output_dir, 'final_finetune'), filename='best', save_top_k=2, verbose=True, monitor='val_wer', mode='min')
    cfg_dict = vars(args)
    wandb_logger = WandbLogger(project=prj_name, name=exp_name, save_dir=args.output_dir, config=cfg_dict)
    trainer = pl.Trainer(devices=args.gpus, accelerator='gpu', max_epochs=args.final_finetune_epochs, default_root_dir=args.output_dir, logger=wandb_logger, callbacks=[last_checkpoint_callback, best_checkpoint_callback])
    model = My_EncDecCTCModelBPE.from_pretrained(model_name=args.model_name_or_path, map_location='cuda:0', trainer=trainer)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h')
    tokenizer = model.tokenizer
    collator = ConformerCTCDataCollator(feature_extractor=feature_extractor, tokenizer=tokenizer, padding=True, sampling_rate=args.data_sample_rate)
    setup_nemo_datasets_and_cfg(model, train_ds, val_ds, test_ds, collator, train_manifest, val_manifest, test_manifest, args)
    release_nemoAPI(model)
    AppState().nemo_file_folder = model._save_restore_connector.model_extracted_dir
    init_num_layers = model.cfg.encoder.n_layers
    init_num_heads = model.cfg.encoder.n_heads
    init_num_total_heads = init_num_layers * init_num_heads
    init_head_dim = model.encoder.layers[0].self_attn.d_k
    heads_to_remove_per_iter = int(init_num_total_heads * args.prune_ratio / args.iterations)
    already_pruned_heads_dict = {i: set() for i in range(init_num_layers)}
    if args.method == 'redundancy-based':
        avg_attn_matrices = get_avg_attention_matrices(model, dataset=train_ds, data_collator=collator, sample_size=10, already_pruned_heads=already_pruned_heads_dict)
        heads_to_keep = find_heads_to_keep(avg_attn_matrices, already_pruned_heads_dict, init_num_total_heads, args)
    else:
        heads_to_keep = []
    if args.method == 'baseline':
        print('▶ Baseline pruning: no pruning applied.')
    elif args.method == 'one-shot':
        print('▶ One-shot pruning: computing average attention …')
        avg_attn_matrices = get_avg_attention_matrices(model, dataset=train_ds, data_collator=collator, sample_size=10, already_pruned_heads=already_pruned_heads_dict)
        L, H, T, _ = avg_attn_matrices.shape
        flat_attn = avg_attn_matrices.view(-1, T, T)
        head_distance_matrix = compute_pairwise_distances(flat_attn, distance_func=get_token_based_distance, mode='token', metric=args.distance_metric)
        head_distance_matrix = head_distance_matrix / head_distance_matrix.max()
        head_similarity_matrix = 1.0 - head_distance_matrix
        total_heads = init_num_total_heads
        n_keep = int(total_heads * (1.0 - args.prune_ratio))
        clustering = SpectralClustering(n_clusters=n_keep, affinity='precomputed', assign_labels='kmeans', random_state=42)
        cluster_labels = clustering.fit_predict(head_similarity_matrix)
        cluster_to_rep = {}
        for head_idx, c in enumerate(cluster_labels):
            cluster_to_rep.setdefault(c, head_idx)
        layer_to_keep = {i: [] for i in range(init_num_layers)}
        for rep in cluster_to_rep.values():
            lyr = rep // init_num_heads
            h_in_layer = rep % init_num_heads
            layer_to_keep[lyr].append(h_in_layer)
        for i in range(init_num_layers):
            if not layer_to_keep[i]:
                layer_to_keep[i] = [0]
        heads_to_prune = build_prune_dict(layer_to_keep, init_num_layers, init_num_heads)
        print(f'▶ One-shot prune plan: {heads_to_prune}')
        prune_conformer_attention(model, heads_to_prune)
        print('▶ One-shot head pruning complete.')
    else:
        for i in range(args.iterations):
            iter_last_checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(args.output_dir, f'iteration_{i + 1}'), filename=f'last', save_top_k=0, verbose=True, save_last=True)
            trainer_i = pl.Trainer(devices=args.gpus, accelerator='gpu', max_epochs=args.iterative_finetune_epochs, default_root_dir=args.output_dir, logger=wandb_logger, callbacks=[iter_last_checkpoint_callback])
            model_i = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=args.model_name_or_path, map_location='cuda:0', trainer=trainer_i)
            setup_nemo_datasets_and_cfg(model_i, train_ds, val_ds, test_ds, collator, train_manifest, val_manifest, test_manifest, args)
            prune_conformer_attention(model_i, already_pruned_heads_dict)
            _orig = wandb_logger.log_hyperparams
            wandb_logger.log_hyperparams = lambda *args, **kwargs: None
            trainer_i.fit(model_i)
            wandb_logger.log_hyperparams = _orig
            if args.method == 'redundancy-based':
                avg_attn_matrices = get_avg_attention_matrices(model_i, dataset=train_ds.select(range(1)), data_collator=collator, sample_size=1, already_pruned_heads=already_pruned_heads_dict)
                del model_i
                del trainer_i
                torch.cuda.empty_cache()
                n_remove = heads_to_remove_per_iter
                remove_candidates = cluster_and_select_heads(avg_attn_matrices, n_remove=n_remove, init_num_total_heads=init_num_total_heads, already_pruned_heads=already_pruned_heads_dict, test_mode=args.test_mode, init_num_heads=init_num_heads, heads_to_keep=heads_to_keep)
                current_num_heads = find_remaining_heads(already_pruned_heads_dict, init_num_total_heads)
                for lyr_idx, h_idx in remove_candidates:
                    if h_idx in already_pruned_heads_dict[lyr_idx]:
                        continue
                    h_g_idx = lyr_idx * init_num_heads + h_idx
                    if h_g_idx in heads_to_keep:
                        continue
                    if len(already_pruned_heads_dict[lyr_idx]) + 1 == init_num_heads:
                        continue
                    already_pruned_heads_dict[lyr_idx].add(h_idx)
            elif args.method == 'magnitude-based':
                head_importance = []
                for layer_idx in range(init_num_layers):
                    attn = model_i.encoder.layers[layer_idx].self_attn
                    q_weight = attn.linear_q.weight.data
                    k_weight = attn.linear_k.weight.data
                    v_weight = attn.linear_v.weight.data
                    o_weight = attn.linear_out.weight.data
                    for h in range(init_num_heads):
                        if h in already_pruned_heads_dict[layer_idx]:
                            continue
                        start, end = (h * init_head_dim, (h + 1) * init_head_dim)
                        q_norm = q_weight[start:end, :].abs().sum()
                        k_norm = k_weight[start:end, :].abs().sum()
                        v_norm = v_weight[start:end, :].abs().sum()
                        o_norm = o_weight[:, start:end].abs().sum()
                        magnitude = (q_norm + k_norm + v_norm + o_norm).item()
                        head_importance.append((layer_idx, h, magnitude))
                head_importance.sort(key=lambda x: x[2])
                current_num_heads = find_remaining_heads(already_pruned_heads_dict, init_num_total_heads)
                n_remove = heads_to_remove_per_iter
                if current_num_heads <= n_remove:
                    n_remove = current_num_heads - 1
                remove_candidates = head_importance[:n_remove]
                cnt = 0
                for lyr_idx, h_idx, _ in remove_candidates:
                    while len(already_pruned_heads_dict[lyr_idx]) + 1 == init_num_heads:
                        lyr_idx, h_idx, _ = head_importance[n_remove + cnt]
                        cnt += 1
                    already_pruned_heads_dict[lyr_idx].add(h_idx)
            elif args.method == 'l0-based':
                eps = 0.001
                head_importance = []
                for layer_idx in range(init_num_layers):
                    attn = model_i.encoder.layers[layer_idx].self_attn
                    q_weight = attn.linear_q.weight.data
                    k_weight = attn.linear_k.weight.data
                    v_weight = attn.linear_v.weight.data
                    o_weight = attn.linear_out.weight.data
                    for h in range(init_num_heads):
                        if h in already_pruned_heads_dict[layer_idx]:
                            continue
                        start, end = (h * init_head_dim, (h + 1) * init_head_dim)
                        sub_q = q_weight[start:end, :]
                        sub_k = k_weight[start:end, :]
                        sub_v = v_weight[start:end, :]
                        sub_o = o_weight[:, start:end]
                        nonzero_q = torch.count_nonzero(sub_q.abs() > eps)
                        nonzero_k = torch.count_nonzero(sub_k.abs() > eps)
                        nonzero_v = torch.count_nonzero(sub_v.abs() > eps)
                        nonzero_o = torch.count_nonzero(sub_o.abs() > eps)
                        total_nonzero = (nonzero_q + nonzero_k + nonzero_v + nonzero_o).item()
                        head_importance.append((layer_idx, h, total_nonzero))
                head_importance.sort(key=lambda x: x[2])
                current_num_heads = find_remaining_heads(already_pruned_heads_dict, init_num_total_heads)
                n_remove = heads_to_remove_per_iter
                if current_num_heads <= n_remove:
                    n_remove = current_num_heads - 1
                remove_candidates = head_importance[:n_remove]
                cnt = 0
                for lyr_idx, h_idx, _ in remove_candidates:
                    while len(already_pruned_heads_dict[lyr_idx]) + 1 == init_num_heads:
                        lyr_idx, h_idx, _ = head_importance[n_remove + cnt]
                        cnt += 1
                    already_pruned_heads_dict[lyr_idx].add(h_idx)
            else:
                raise ValueError(f'Unknown method: {args.method}')
        trainer = pl.Trainer(devices=args.gpus, accelerator='gpu', max_epochs=args.final_finetune_epochs, default_root_dir=args.output_dir, logger=wandb_logger, callbacks=[best_checkpoint_callback, last_checkpoint_callback])
        model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=args.model_name_or_path, map_location='cuda:0', trainer=trainer)
        setup_nemo_datasets_and_cfg(model, train_ds, val_ds, test_ds, collator, train_manifest, val_manifest, test_manifest, args)
        prune_conformer_attention(model, already_pruned_heads_dict)
    _orig = wandb_logger.log_hyperparams
    wandb_logger.log_hyperparams = lambda *args, **kwargs: None
    trainer.fit(model)
    wandb_logger.log_hyperparams = _orig
    final_epoch_ckpt_path = os.path.join(args.output_dir, 'final_finetune', 'final_epoch.ckpt')
    os.makedirs(os.path.dirname(final_epoch_ckpt_path), exist_ok=True)
    trainer.save_checkpoint(final_epoch_ckpt_path)
    with open(os.path.join(args.output_dir, 'pruning.json'), 'w') as stream:
        json.dump({str(k): sorted(v) for k, v in already_pruned_heads_dict.items()}, stream, indent=2)
    print(f'✅ Final checkpoint saved to {final_epoch_ckpt_path}')
    MAX_SAMPLES = 160000
    if args.dataset_name == 'librispeech':
        split_names = ['dev.clean', 'dev.other', 'test.clean', 'test.other']
        script = args.data_script_path
        config_name = args.data_config_name
        load_kwargs = {'path': script, 'name': config_name, 'trust_remote_code': True, 'download_config': dl_cfg, 'cache_dir': cache_dir}
    else:
        split_names = ['dev', 'test']
        load_kwargs = {'path': str(Path(__file__).resolve().parents[2] / 'data_scripts' / 'tedlium_test.py'), 'name': 'release1', 'trust_remote_code': True}
    metrics = {}
    for split_name in split_names:
        print(f'\n===== Evaluating on split: {split_name} =====')
        model.eval()
        test_i_ds = load_dataset(load_kwargs['path'], load_kwargs['name'], split=split_name, **{k: v for k, v in load_kwargs.items() if k not in ['path', 'name']})
        if args.dataset_name == 'tedlium':
            test_i_ds = test_i_ds.filter(lambda x: x['audio']['array'].shape[0] <= MAX_SAMPLES)
            test_i_ds = test_i_ds.map(preprocess_function, fn_kwargs={'processor': processor}, num_proc=4)
        if args.test_mode:
            test_i_ds = test_i_ds.select(range(300))
        json_name = split_name.replace('.', '_') + '.json'
        manifest_i = os.path.join(manifest_dir, json_name)
        build_manifest_from_hf(test_i_ds, manifest_i, cache_dir)
        test_cfg = deepcopy(model.cfg.test_ds)
        test_cfg.manifest_filepath = manifest_i
        test_cfg.shuffle = False
        model.setup_test_data(test_cfg)
        dl = model.test_dataloader()
        _orig_log_hparams = wandb_logger.log_hyperparams
        wandb_logger.log_hyperparams = lambda *args, **kwargs: None
        results = trainer.test(model=model, dataloaders=[dl], ckpt_path=final_epoch_ckpt_path or None, verbose=True)
        wandb_logger.log_hyperparams = _orig_log_hparams
        res = results[0]
        wer = res.get('test_wer', res.get('wer', None))
        loss = res.get('test_loss', res.get('loss', None))
        print(f'  → split={split_name} | loss={loss:.4f} | wer={wer:.2%}')
        key_prefix = split_name.replace('.', '_')
        metric = {f'{key_prefix}/wer': wer, f'{key_prefix}/loss': loss}
        metrics[f'{key_prefix}/wer'] = wer
        metrics[f'{key_prefix}/loss'] = loss
        wandb_logger.log_metrics(metric, step=trainer.current_epoch)
    print(f'metrics: {metrics}')
    wandb_logger.log_metrics(metrics, step=trainer.current_epoch)
if __name__ == '__main__':
    main()
