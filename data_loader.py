# data_loader.py
import glob
import json
import os
import re
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


TOKEN_RE = re.compile(r"[\w']+|[.,!?;]")


class Sent140Dataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def _load_leaf_dir(split_dir: str):
    files = sorted(glob.glob(os.path.join(split_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"Nenhum JSON encontrado em {split_dir}")

    users = {}
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            obj = json.load(f)

        for user in obj["users"]:
            if user not in users:
                users[user] = {"x": [], "y": []}
            users[user]["x"].extend(obj["user_data"][user]["x"])
            users[user]["y"].extend(obj["user_data"][user]["y"])

    return users


def _extract_text(sample):
    if isinstance(sample, str):
        return sample

    if isinstance(sample, dict):
        for key in ("text", "tweet", "sentence"):
            if key in sample:
                return str(sample[key])
        return str(sample)

    if isinstance(sample, (list, tuple)):
        if len(sample) >= 5 and isinstance(sample[4], str):
            return sample[4]
        for item in reversed(sample):
            if isinstance(item, str) and item.strip():
                return item
        return str(sample)

    return str(sample)


def _tokenize(text: str):
    return TOKEN_RE.findall(text)


def _encode_label(y):
    y = int(float(y))
    return 0 if y == 0 else 1


def _encode_text(text, stoi, seq_len, unk_idx):
    tokens = _tokenize(text)
    ids = [stoi.get(tok, unk_idx) for tok in tokens[:seq_len]]
    ids += [unk_idx] * (seq_len - len(ids))
    return ids


def _load_filtered_glove(glove_path, vocab, embedding_dim=300):
    glove = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word not in vocab:
                continue
            vec = np.asarray(parts[1:], dtype=np.float32)
            if vec.shape[0] == embedding_dim:
                glove[word] = vec
    return glove


def _select_users_for_paper_stats(common_users, train_users, test_users, config):
    if len(common_users) < config.num_clients:
        raise RuntimeError(
            f"LEAF gerou apenas {len(common_users)} usuarios, mas o config espera {config.num_clients}."
        )

    totals = np.array(
        [len(train_users[u]["y"]) + len(test_users[u]["y"]) for u in common_users],
        dtype=np.int64,
    )

    if len(common_users) == config.num_clients and int(totals.sum()) == config.target_total_samples:
        return list(common_users)

    best_indices = None
    best_gap = None
    for seed in [config.client_selection_seed] + list(range(20000)):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(common_users), size=config.num_clients, replace=False)
        total = int(totals[indices].sum())
        gap = abs(total - config.target_total_samples)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_indices = indices
        if gap == 0:
            break

    if best_indices is None:
        raise RuntimeError("Falha ao selecionar usuarios do Sent140.")

    return [common_users[i] for i in best_indices]


def load_sent140_partitions(config):
    train_dir = os.path.join(config.leaf_root, "train")
    test_dir = os.path.join(config.leaf_root, "test")

    print("Lendo Sent140 pre-processado pelo LEAF...")
    train_users = _load_leaf_dir(train_dir)
    test_users = _load_leaf_dir(test_dir)

    common_users = sorted(set(train_users.keys()) & set(test_users.keys()))
    selected_users = _select_users_for_paper_stats(common_users, train_users, test_users, config)

    selected_sizes = np.array(
        [len(train_users[u]["y"]) + len(test_users[u]["y"]) for u in selected_users],
        dtype=np.int64,
    )
    print(
        f"[Selecao Sent140] clientes={len(selected_users)}, total={selected_sizes.sum()}, "
        f"media={selected_sizes.mean():.1f}, std={selected_sizes.std(ddof=0):.1f}, "
        f"min={selected_sizes.min()}, max={selected_sizes.max()}"
    )

    token_counter = Counter()
    for user in selected_users:
        for x in train_users[user]["x"]:
            token_counter.update(_tokenize(_extract_text(x))[: config.seq_len])

    pad_token = "<pad>"
    unk_token = "<unk>"
    pad_idx = 0
    unk_idx = 1
    vocab_tokens = sorted(token_counter.keys())
    stoi = {pad_token: pad_idx, unk_token: unk_idx}
    for tok in vocab_tokens:
        stoi[tok] = len(stoi)

    print("Carregando GloVe filtrado para o vocabulario do Sent140...")
    glove = _load_filtered_glove(
        glove_path=config.glove_path,
        vocab=set(vocab_tokens),
        embedding_dim=config.embedding_dim,
    )

    embedding_matrix = np.random.normal(
        loc=0.0,
        scale=0.1,
        size=(len(stoi), config.embedding_dim),
    ).astype(np.float32)
    embedding_matrix[pad_idx] = 0.0

    found = 0
    for tok, idx in stoi.items():
        if tok in glove:
            embedding_matrix[idx] = glove[tok]
            found += 1

    config.embedding_matrix = embedding_matrix
    config.pad_idx = pad_idx
    config.vocab_size = len(stoi)

    train_loaders = {}
    global_test_sequences = []
    global_test_labels = []
    sizes = []
    total_train_samples = 0
    total_test_samples = 0

    for client_id, user in enumerate(selected_users):
        seqs_train, ys_train = [], []
        seqs_test, ys_test = [], []

        for x, y in zip(train_users[user]["x"], train_users[user]["y"]):
            seqs_train.append(_encode_text(_extract_text(x), stoi, config.seq_len, unk_idx))
            ys_train.append(_encode_label(y))

        for x, y in zip(test_users[user]["x"], test_users[user]["y"]):
            seqs_test.append(_encode_text(_extract_text(x), stoi, config.seq_len, unk_idx))
            ys_test.append(_encode_label(y))

        train_dataset = Sent140Dataset(seqs_train, ys_train)
        train_loaders[client_id] = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
        )

        global_test_sequences.extend(seqs_test)
        global_test_labels.extend(ys_test)
        total_train_samples += len(seqs_train)
        total_test_samples += len(seqs_test)
        sizes.append(len(seqs_train) + len(seqs_test))

    test_dataset = Sent140Dataset(global_test_sequences, global_test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    config.num_clients = len(train_loaders)
    config.clients_per_round = 10
    config.frac = config.clients_per_round / config.num_clients

    sizes = np.array(sizes)
    print("[Sent140] Particao concluida.")
    print(f"  Clientes: {len(train_loaders)}")
    print(f"  Total de amostras: {sizes.sum()}")
    print(
        f"  Tamanho por cliente -> min: {sizes.min()}, max: {sizes.max()}, "
        f"media: {sizes.mean():.1f}, std: {sizes.std(ddof=0):.1f}"
    )
    print(f"  Treino total: {total_train_samples} | Teste total: {total_test_samples}")
    print(f"  Vocabulario: {config.vocab_size} | Tokens com GloVe encontrado: {found}")

    return train_loaders, test_loader
