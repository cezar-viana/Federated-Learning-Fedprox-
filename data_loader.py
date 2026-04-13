# data_loader.py
import glob
import json
import os
import re
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


TOKEN_RE = re.compile(r"[A-Za-z0-9_']+|[^\w\s]")


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
    """
    LEAF Sent140 costuma armazenar cada x como algo parecido com:
    [tweet_id, date, query, user, text]
    Esta função é robusta o bastante para lidar com variantes simples.
    """
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
        # fallback: pega a última string "grande"
        for item in reversed(sample):
            if isinstance(item, str) and len(item.strip()) > 0:
                return item
        return str(sample)

    return str(sample)


def _tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


def _encode_label(y):
    # Sent140 costuma usar 0 e 4
    y = int(float(y))
    return 0 if y == 0 else 1


def _pad_or_truncate(ids, seq_len, pad_idx):
    if len(ids) >= seq_len:
        return ids[:seq_len]
    return ids + [pad_idx] * (seq_len - len(ids))


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


def load_sent140_partitions(config):
    train_dir = os.path.join(config.leaf_root, "train")
    test_dir = os.path.join(config.leaf_root, "test")

    print("Lendo Sent140 pré-processado pelo LEAF...")
    train_users = _load_leaf_dir(train_dir)
    test_users = _load_leaf_dir(test_dir)

    common_users = sorted(set(train_users.keys()) & set(test_users.keys()))
    
    if len(common_users) < config.num_clients:
        raise RuntimeError(
            f"LEAF gerou apenas {len(common_users)} usuários, mas o config espera {config.num_clients}."
        )
    
    rng = np.random.default_rng(42)
    common_users = list(rng.choice(common_users, size=config.num_clients, replace=False))
    
    selected_sizes = np.array(
        [len(train_users[u]["y"]) + len(test_users[u]["y"]) for u in common_users],
        dtype=np.int64
    )
    
    print(
        f"[Pré-seleção Sent140] clientes={len(common_users)}, "
        f"total={selected_sizes.sum()}, média={selected_sizes.mean():.1f}, "
        f"std={selected_sizes.std(ddof=0):.1f}, min={selected_sizes.min()}, max={selected_sizes.max()}"
    )

    # constroi vocabulário a partir do treino
    token_counter = Counter()
    total_train_samples = 0
    total_test_samples = 0

    for user in common_users:
        for x in train_users[user]["x"]:
            token_counter.update(_tokenize(_extract_text(x))[: config.seq_len])
        total_train_samples += len(train_users[user]["y"])
        total_test_samples += len(test_users[user]["y"])

    pad_token = "<pad>"
    unk_token = "<unk>"
    pad_idx = 0
    unk_idx = 1

    vocab_tokens = sorted(token_counter.keys())
    stoi = {pad_token: pad_idx, unk_token: unk_idx}
    for tok in vocab_tokens:
        stoi[tok] = len(stoi)

    print("Carregando GloVe filtrado para o vocabulário do Sent140...")
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

    for client_id, user in enumerate(common_users):
        seqs_train, ys_train = [], []
        seqs_test, ys_test = [], []

        for x, y in zip(train_users[user]["x"], train_users[user]["y"]):
            toks = _tokenize(_extract_text(x))
            ids = [stoi.get(tok, unk_idx) for tok in toks]
            ids = _pad_or_truncate(ids, config.seq_len, pad_idx)
            seqs_train.append(ids)
            ys_train.append(_encode_label(y))

        for x, y in zip(test_users[user]["x"], test_users[user]["y"]):
            toks = _tokenize(_extract_text(x))
            ids = [stoi.get(tok, unk_idx) for tok in toks]
            ids = _pad_or_truncate(ids, config.seq_len, pad_idx)
            seqs_test.append(ids)
            ys_test.append(_encode_label(y))

        train_dataset = Sent140Dataset(seqs_train, ys_train)
        train_loaders[client_id] = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
        )

        global_test_sequences.extend(seqs_test)
        global_test_labels.extend(ys_test)
        sizes.append(len(seqs_train) + len(seqs_test))

    test_dataset = Sent140Dataset(global_test_sequences, global_test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # atualiza config para refletir exatamente o que foi carregado
    config.num_clients = len(train_loaders)
    config.clients_per_round = 10
    config.frac = config.clients_per_round / config.num_clients

    sizes = np.array(sizes)
    print("[Sent140] Partição concluída.")
    print(f"  Clientes: {len(train_loaders)}")
    print(f"  Total de amostras: {sizes.sum()}")
    print(
        f"  Tamanho por cliente -> min: {sizes.min()}, max: {sizes.max()}, "
        f"média: {sizes.mean():.1f}, std: {sizes.std(ddof=0):.1f}"
    )
    print(f"  Treino total: {total_train_samples} | Teste total: {total_test_samples}")
    print(f"  Vocabulário: {config.vocab_size} | Tokens com GloVe encontrado: {found}")

    return train_loaders, test_loader