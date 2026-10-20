"""Historical Conformer experiment; see docs/conformer.md before use."""
from pathlib import Path
import numpy as np
import torch
import random
import torch.nn as nn
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from sklearn.cluster import SpectralClustering
from tqdm import tqdm
from transformers.pytorch_utils import prune_linear_layer

def cosine_distance(vec1, vec2):
    """1 - cosine similarity"""
    return cosine(vec1, vec2)

def pearson_distance(vec1, vec2):
    """1 - Pearson correlation"""
    corr, _ = pearsonr(vec1, vec2)
    return 1 - corr

def jensen_shannon_distance(p, q):
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)

    def kl_div(a, b):
        a = np.where(a == 0, 1e-12, a)
        b = np.where(b == 0, 1e-12, b)
        return np.sum(a * np.log(a / b))
    js_div = 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)
    return np.sqrt(js_div)

def bhattacharyya_distance(p, q):
    """Bhattacharyya distance"""
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)
    p /= p.sum()
    q /= q.sum()
    bc = np.sum(np.sqrt(p * q))
    return -np.log(bc + 1e-12)

def get_token_based_distance(matA, matB, metric='cosine'):
    distances = []
    for i in range(matA.shape[0]):
        vecA = matA[i, :]
        vecB = matB[i, :]
        if metric == 'cosine':
            dist = cosine_distance(vecA, vecB)
        elif metric == 'corr':
            dist = pearson_distance(vecA, vecB)
        elif metric == 'js':
            dist = jensen_shannon_distance(vecA, vecB)
        elif metric == 'bc':
            dist = bhattacharyya_distance(vecA, vecB)
        else:
            raise ValueError('Unknown token-based metric')
        distances.append(dist)
    return np.mean(distances)

def distance_correlation(A, B):
    A_flat = A.flatten()
    B_flat = B.flatten()
    corr, _ = pearsonr(A_flat, B_flat)
    return 1 - abs(corr)

def procrustes_distance(A, B):
    A_norm = A / (np.linalg.norm(A) + 1e-12)
    B_norm = B / (np.linalg.norm(B) + 1e-12)
    fro_diff = np.linalg.norm(A_norm - B_norm)
    return fro_diff / 2.0

def canonical_correlation_distance(A, B):
    A_flat = A.flatten()
    B_flat = B.flatten()
    corr, _ = pearsonr(A_flat, B_flat)
    return 1 - corr

def get_sentence_based_distance(matA, matB, metric='dCor'):
    if metric == 'dCor':
        return distance_correlation(matA, matB)
    elif metric == 'PC':
        return procrustes_distance(matA, matB)
    elif metric == 'CC':
        return canonical_correlation_distance(matA, matB)
    else:
        raise ValueError('Unknown sentence-based metric')

def compute_pairwise_distances(attention_mats, distance_func, mode='token', metric='cosine'):
    num_heads = attention_mats.shape[0]
    dist_matrix = np.zeros((num_heads, num_heads))
    for i in tqdm(range(num_heads)):
        for j in range(num_heads):
            dist_val = distance_func(attention_mats[i], attention_mats[j], metric)
            if isinstance(dist_val, (int, float)) and np.isfinite(dist_val):
                dist_matrix[i, j] = dist_val
                dist_matrix[j, i] = dist_val
            else:
                dist_matrix[i, j] = 2.0
                dist_matrix[j, i] = 2.0
    return dist_matrix

def build_prune_dict(layer_to_keep_dict, layers_num, heads_num):
    num_attention_heads = heads_num
    heads_to_prune = {}
    for layer_idx in range(layers_num):
        keep_heads = set(layer_to_keep_dict.get(layer_idx, []))
        if not keep_heads:
            keep_heads = {0}
        all_heads = set(range(num_attention_heads))
        prune_heads = sorted(list(all_heads - keep_heads))
        if prune_heads:
            heads_to_prune[layer_idx] = prune_heads
    return heads_to_prune

def find_pruneable_heads_and_indices(heads, num_heads, head_dim, already_pruned_heads=None, num_attention_heads=None):
    if already_pruned_heads is None:
        already_pruned_heads = set()
    heads = set(heads) - already_pruned_heads
    mask = torch.ones(num_heads, head_dim)
    for head in heads:
        less_than_head = 0
        less_than_head += sum((1 if h < head else 0 for h in already_pruned_heads))
        mask[head - less_than_head] = 0
    mask = mask.view(-1).eq(1)
    index = torch.arange(num_heads * head_dim)[mask]
    return (heads, index)

def prune_conformer_block_attention(layer_module, heads_to_prune):
    attn_module = layer_module.self_attn
    if not heads_to_prune:
        return
    num_heads = attn_module.h
    head_dim = attn_module.d_k
    d_model = num_heads * head_dim
    already_pruned = getattr(attn_module, 'pruned_heads', set())
    heads, index = find_pruneable_heads_and_indices(heads_to_prune, num_heads, head_dim, already_pruned, num_attention_heads=num_heads)
    head_indices = torch.arange(num_heads)[~torch.tensor([h in heads for h in range(num_heads)])]
    head_indices = head_indices.to(attn_module.pos_bias_u.device)
    attn_module.linear_q = prune_linear_layer(attn_module.linear_q, index, dim=0)
    attn_module.linear_k = prune_linear_layer(attn_module.linear_k, index, dim=0)
    attn_module.linear_v = prune_linear_layer(attn_module.linear_v, index, dim=0)
    attn_module.linear_out = prune_linear_layer(attn_module.linear_out, index, dim=1)
    attn_module.linear_pos = prune_linear_layer(attn_module.linear_pos, index, dim=0)
    if hasattr(attn_module, 'pos_bias_u') and attn_module.pos_bias_u is not None:
        attn_module.pos_bias_u = torch.nn.Parameter(attn_module.pos_bias_u[head_indices].detach().clone())
    if hasattr(attn_module, 'pos_bias_v') and attn_module.pos_bias_v is not None:
        attn_module.pos_bias_v = torch.nn.Parameter(attn_module.pos_bias_v[head_indices].detach().clone())
    attn_module.h = num_heads - len(heads)
    layer_module.n_heads = attn_module.h
    attn_module.pruned_heads = already_pruned.union(heads)

def prune_conformer_attention(model, heads_to_prune_dict):
    for layer_idx, prune_head_list in heads_to_prune_dict.items():
        layer_module = model.encoder.layers[layer_idx]
        prune_conformer_block_attention(layer_module, prune_head_list)

def my_compute_loss_ctc(outputs, labels, processor=None, blank_token_id=None, num_items_in_batch=None):
    if blank_token_id is None:
        blank_token_id = processor.tokenizer.pad_token_id
    logits = outputs.logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs = log_probs.transpose(0, 1)
    batch_size, T, _ = logits.shape
    device = logits.device
    label_lengths = (labels != -100).sum(dim=1)
    targets_list = [label_seq[label_seq != -100] for label_seq in labels]
    targets_list = [t for t in targets_list if t.numel() > 0]
    targets = torch.cat(targets_list).to(device) if targets_list else torch.tensor([], dtype=torch.long, device=device)
    input_lengths = torch.full(size=(batch_size,), fill_value=T, dtype=torch.long).to(device)
    ctc_loss_fn = nn.CTCLoss(blank=blank_token_id, zero_infinity=True)
    loss = ctc_loss_fn(log_probs, targets, input_lengths, label_lengths)
    return loss

def pad_attention_tensors_to_max(avg_attn_per_layer, model_config, already_pruned_heads):
    init_num_layer = model_config.encoder.n_layers
    init_num_head = model_config.encoder.n_heads
    seq_len = avg_attn_per_layer[0].size(1)
    device = avg_attn_per_layer[0].device
    dtype = avg_attn_per_layer[0].dtype
    padded_head_tensors = []
    padded_layer_tensors = []
    for layer_idx, tensor in enumerate(avg_attn_per_layer):
        head_tensor_list = []
        cnt = 0
        for head_idx in range(init_num_head):
            if head_idx in already_pruned_heads[layer_idx]:
                cnt += 1
                head_tensor_list.append(torch.zeros(seq_len, seq_len, device=device, dtype=dtype))
            else:
                head_tensor_list.append(avg_attn_per_layer[layer_idx][head_idx - cnt])
        padded_head_tensors = torch.stack(head_tensor_list, dim=0)
        padded_layer_tensors.append(padded_head_tensors)
    padded_tensors = torch.stack(padded_layer_tensors, dim=0)
    return padded_tensors

def get_avg_attention_matrices(model, dataset, data_collator, sample_size=10, already_pruned_heads=None):
    device_before = next(model.parameters()).device
    model.cpu().eval()
    collected = []
    handles = []
    num_layers = model.encoder.n_layers
    for layer_idx in range(num_layers):
        mha = model.encoder.layers[layer_idx].self_attn

        def make_hook(li):

            def hook(mod, inp, out):
                if not isinstance(out, tuple) or len(out) < 2 or out[1].ndim != 4:
                    raise RuntimeError('Conformer research code requires a NeMo attention patch returning (features, attention_maps). See docs/conformer.md.')
                attn = out[1]
                collected.append((li, attn.cpu()))
            return hook
        handles.append(mha.register_forward_hook(make_hook(layer_idx)))
    idxs = random.sample(range(len(dataset)), sample_size)
    sampled = [dataset[i] for i in idxs]
    input_signal, input_signal_length, _, _ = data_collator(sampled)
    with torch.no_grad():
        _ = model(input_signal=input_signal.to(model.device), input_signal_length=input_signal_length.to(model.device))
    for h in handles:
        h.remove()
    from collections import defaultdict
    layer_dict = defaultdict(list)
    for layer_idx, attn in collected:
        layer_dict[layer_idx].append(attn)
    avg_attn_per_layer = []
    for li in range(num_layers):
        attn_list = layer_dict[li]
        attn_cat = torch.cat(attn_list, dim=0)
        avg = attn_cat.mean(dim=0)
        avg_attn_per_layer.append(avg)
    padded = pad_attention_tensors_to_max(avg_attn_per_layer, model.cfg, already_pruned_heads)
    model.to(device_before)
    return padded

def delete_pruned_heads_from_similarity_matrix(similarity_matrix, already_pruned_heads, heads_to_keep, init_num_heads=12):
    delete_heads_list = list((layer_idx * init_num_heads + head_idx for layer_idx, pruned_heads in already_pruned_heads.items() for head_idx in pruned_heads))
    for keep_head_global_idx in heads_to_keep:
        delete_heads_list.append(keep_head_global_idx)
    deleted_similarity_matrix = np.delete(similarity_matrix, delete_heads_list, axis=0)
    deleted_similarity_matrix = np.delete(deleted_similarity_matrix, delete_heads_list, axis=1)
    return deleted_similarity_matrix

def restore_pruned_heads_to_cluster_labels(cluster_labels, already_pruned_heads, heads_to_keep, init_num_heads):
    deleted_heads_list = list((layer_idx * init_num_heads + head_idx for layer_idx, pruned_heads in already_pruned_heads.items() for head_idx in pruned_heads))
    for keep_head_global_idx in heads_to_keep:
        deleted_heads_list.append(keep_head_global_idx)
    for pruned_head_g_idx in sorted(deleted_heads_list):
        cluster_labels = np.insert(cluster_labels, pruned_head_g_idx, -1)
    return cluster_labels

def cluster_and_select_heads(attn_matrices, distance_metric='cosine', n_remove=5, init_num_total_heads=144, already_pruned_heads=None, test_mode=False, init_num_heads=12, heads_to_keep=None):
    print(f'--------- START cluster_and_select_heads FUNCTION ---------')
    current_num_heads = find_remaining_heads(already_pruned_heads, init_num_total_heads)
    seq_len = attn_matrices.size(-1)
    reshaped_attn = attn_matrices.view(-1, seq_len, seq_len)
    print(f' - Calculating pairwise distances ...')
    if not test_mode:
        distance_matrix = compute_pairwise_distances(reshaped_attn, distance_func=get_token_based_distance, mode='token', metric=distance_metric)
    else:
        distance_matrix = np.random.rand(init_num_total_heads, init_num_total_heads) * 2.0
    print(f' - Distance matrix shape: {distance_matrix.shape}')
    distance_matrix = distance_matrix / 2.0
    similarity_matrix = 1.0 - distance_matrix
    n_clusters = max(1, current_num_heads - n_remove - len(heads_to_keep))
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans', random_state=42)
    deleted_similarity_matrix = delete_pruned_heads_from_similarity_matrix(similarity_matrix, already_pruned_heads, heads_to_keep, init_num_heads)
    print(f'similarity_matrix.shape: {similarity_matrix.shape}')
    print(f'already_pruned_heads: {already_pruned_heads}')
    cluster_labels = clustering.fit_predict(deleted_similarity_matrix)
    print(f'Pruned cluster_labels: {cluster_labels}')
    cluster_labels = restore_pruned_heads_to_cluster_labels(cluster_labels, already_pruned_heads, heads_to_keep, init_num_heads)
    print(f'After Restore cluster_labels: {cluster_labels}')
    try:
        sc_score = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
        print(f' - SpectralClustering Silhouette Score: {sc_score:.4f}')
    except:
        pass
    cluster_to_rep = {}
    cluster_head_dict = {}
    for head_idx, c_id in enumerate(cluster_labels):
        cluster_head_dict[c_id] = cluster_head_dict.get(c_id, []) + [head_idx]
    print(f'cluster_head_dict: {cluster_head_dict}')
    for c_id, head_indices in cluster_head_dict.items():
        if c_id == -1:
            continue
        rep_head_idx = np.random.choice(head_indices)
        cluster_to_rep[c_id] = rep_head_idx
    print(f'cluster_to_rep: {cluster_to_rep}')
    keep_set = set(cluster_to_rep.values())
    all_heads_set = set(range(init_num_total_heads))
    already_pruned_heads_set = set([layer_idx * init_num_heads + head_idx for layer_idx, pruned_heads in already_pruned_heads.items() for head_idx in pruned_heads])
    prune_set = all_heads_set - already_pruned_heads_set - keep_set - set(heads_to_keep)
    remove_list = []
    num_heads = attn_matrices.size(1)
    for g_idx in prune_set:
        layer_idx = g_idx // num_heads
        head_idx_in_layer = g_idx % num_heads
        remove_list.append((layer_idx, head_idx_in_layer))
    print(f' - Heads to remove: {remove_list}')
    print(f'--------- END cluster_and_select_heads FUNCTION ---------')
    return remove_list

def find_remaining_heads(already_pruned_heads_dict, init_num_total_heads):
    pruned_heads_num = 0
    for layer_idx, pruned_heads in already_pruned_heads_dict.items():
        pruned_heads_num += len(pruned_heads)
    remaining_heads_num = init_num_total_heads - pruned_heads_num
    return remaining_heads_num

def find_heads_to_keep(attn_matrices, already_pruned_heads_dict, init_num_total_heads, args):
    seq_len = attn_matrices.size(-1)
    reshaped_attn = attn_matrices.view(-1, seq_len, seq_len)
    print(f' - Calculating pairwise distances ...')
    if not args.test_mode:
        distance_matrix = compute_pairwise_distances(reshaped_attn, distance_func=get_token_based_distance, mode='token', metric='cosine')
    else:
        distance_matrix = np.random.rand(init_num_total_heads, init_num_total_heads) * 2.0
    print(f' - Distance matrix shape: {distance_matrix.shape}')
    distance_matrix = distance_matrix / 2.0
    similarity_matrix = 1.0 - distance_matrix
    heads_to_keep = []
    for head_idx in range(init_num_total_heads):
        avg_similarity = similarity_matrix[head_idx].mean()
        heads_to_keep.append((head_idx, avg_similarity))
    heads_to_keep = sorted(heads_to_keep, key=lambda x: x[1])
    heads_to_keep = heads_to_keep[:10]
    heads_to_keep = [head_idx for head_idx, _ in heads_to_keep]
    print(f' - Heads to keep: {heads_to_keep}')
    return heads_to_keep
