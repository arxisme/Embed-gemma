
# Generate Triplet Embeddings + Train Compressors + Benchmark
# Compares: BERT (128/256/512/768) vs Gemma (2B/2-2B) vs
#           PCA compression vs Our InfoNCE+MSE neural compressor

import os
import gc
import json
import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")

# Local paths
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR       = os.path.join(BASE_DIR, "contrastive_embs")
COMPRESSOR_DIR = os.path.join(BASE_DIR, "multi_compressors")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(COMPRESSOR_DIR, exist_ok=True)

# Model Registry
# Each entry: dim -> {"name": HF model id, "is_decoder": bool}
# is_decoder=True  → Gemma-style causal LMs (need left-pad, pad=eos)
# is_decoder=False → BERT-style encoder models
MODEL_CONFIGS = {
    #128:  {"name": "google/bert_uncased_L-2_H-128_A-2",  "is_decoder": False},
    #256:  {"name": "google/bert_uncased_L-4_H-256_A-4",  "is_decoder": False},
    #512:  {"name": "google/bert_uncased_L-4_H-512_A-8",  "is_decoder": False},
    #768:  {"name": "bert-base-uncased",                  "is_decoder": False},
    # Gemma models
    # Requires HuggingFace token: huggingface-cli login
    2048: {"name": "google/gemma-2b",                   "is_decoder": True},
    2304: {"name": "google/gemma-2-2b",                 "is_decoder": True},
}

# BERT models used to train the compressor (768-dim anchor)
BERT_DIMS   = [128, 256, 512, 768]
# Gemma models added as native baselines in the benchmark
GEMMA_DIMS  = [2048, 2304]
# Compressor target dims (768-dim BERT → these sizes)
COMPRESS_DIMS = [512, 256, 128]

BATCH_SIZE   = 512
MAX_LENGTH   = 128
USE_FP16     = True
MAX_TRIPLETS = None   # e.g. 50_000 for a quick smoke test


# Model / Tokenizer Loader
def load_model_tokenizer(dim: int):
    """Load model+tokenizer from MODEL_CONFIGS[dim].
    Handles both BERT-style encoders and Gemma-style decoder LMs."""
    cfg       = MODEL_CONFIGS[dim]
    name      = cfg["name"]
    is_dec    = cfg["is_decoder"]

    tokenizer = AutoTokenizer.from_pretrained(name)
    if is_dec:
        # Causal LMs need left-padding so the last real token is last
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token    = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModel.from_pretrained(name).eval().to(DEVICE)
    if USE_FP16 and DEVICE == "cuda":
        model = model.half()

    return model, tokenizer


# Mean Pooling (attention-mask aware)
def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


# Sentence Encoder
@torch.no_grad()
def encode(sentences: list[str], model, tokenizer, desc="encoding") -> np.ndarray:
    """Encode a list of sentences into mean-pooled, L2-normalised embeddings."""
    all_embs = []
    for i in tqdm(range(0, len(sentences), BATCH_SIZE), desc=f"  {desc}", leave=False):
        batch = sentences[i : i + BATCH_SIZE]
        enc   = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt"
        ).to(DEVICE)
        out = model(**enc)
        emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        all_embs.append(emb.cpu().float().numpy())
    return np.vstack(all_embs)


# PHASE 1 - Build Triplets & Generate Embeddings

def build_triplets(dataset) -> list[tuple[str, str, str]]:
    prem2hyp: dict[str, dict[int, list[str]]] = {}
    for row in tqdm(dataset, desc="  indexing", leave=False):
        lbl = row["label"]
        if lbl not in (0, 2):
            continue
        prem = row["premise"].strip()
        hyp  = row["hypothesis"].strip()
        if prem not in prem2hyp:
            prem2hyp[prem] = {0: [], 2: []}
        prem2hyp[prem][lbl].append(hyp)

    triplets = []
    for prem, hyps in prem2hyp.items():
        if hyps[0] and hyps[2]:
            triplets.append((prem, hyps[0][0], hyps[2][0]))
    return triplets


print("\n[Phase 1] Loading NLI data for triplets…")
snli_raw     = load_dataset("stanfordnlp/snli",    split="train", trust_remote_code=True)
snli_triplets = build_triplets(snli_raw);  del snli_raw;  gc.collect()

mnli_raw     = load_dataset("nyu-mll/multi_nli",   split="train", trust_remote_code=True)
mnli_triplets = build_triplets(mnli_raw);  del mnli_raw;  gc.collect()

all_triplets = snli_triplets + mnli_triplets
if MAX_TRIPLETS:
    random.shuffle(all_triplets)
    all_triplets = all_triplets[:MAX_TRIPLETS]

anchors   = [t[0] for t in all_triplets]
positives = [t[1] for t in all_triplets]
negatives = [t[2] for t in all_triplets]
print(f"      Total triplets : {len(all_triplets):,}")
del all_triplets, snli_triplets, mnli_triplets; gc.collect()

# Encode with every BERT + Gemma model and save .npz files
print("\n[Phase 1] Encoding triplets across all models…")
metadata = {"max_length": MAX_LENGTH, "pooling": "mean", "total_triplets": len(anchors)}

for dim, cfg in MODEL_CONFIGS.items():
    npz_path = os.path.join(SAVE_DIR, f"triplet_embeddings_{dim}.npz")
    if os.path.exists(npz_path):
        print(f"  Skipping {dim}-dim : {cfg['name']} (embeddings already exist at {npz_path})")
        metadata[f"model_{dim}"] = cfg["name"]
        continue

    print(f"\n{'='*55}")
    tag = "Gemma" if cfg["is_decoder"] else "BERT"
    print(f"  [{tag}] Encoding {dim}-dim : {cfg['name']}")
    print(f"{'='*55}")

    model, tokenizer = load_model_tokenizer(dim)

    t0          = time.time()
    anchor_embs = encode(anchors,   model, tokenizer, desc=f"anchors-{dim}")
    pos_embs    = encode(positives, model, tokenizer, desc=f"pos-{dim}")
    neg_embs    = encode(negatives, model, tokenizer, desc=f"neg-{dim}")
    print(f"  Done in {(time.time()-t0)/60:.1f} min")

    np.savez_compressed(npz_path,
                        anchors=anchor_embs, positives=pos_embs, negatives=neg_embs)
    metadata[f"model_{dim}"] = cfg["name"]
    print(f"  Saved → {npz_path}  ({os.path.getsize(npz_path)/1e6:.1f} MB)")

    del model, tokenizer, anchor_embs, pos_embs, neg_embs
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# Save raw sentences once
np.savez_compressed(
    os.path.join(SAVE_DIR, "triplet_sentences.npz"),
    anchors   = np.array(anchors,   dtype=object),
    positives = np.array(positives, dtype=object),
    negatives = np.array(negatives, dtype=object),
)
with open(os.path.join(SAVE_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n Phase 1 complete - embeddings saved to {SAVE_DIR}")


# PHASE 2 - Train Neural Compressors (BERT-768 → 512 / 256 / 128)

# Config
CFG = {
    "input_dim"    : 768,
    "target_dims"  : COMPRESS_DIMS,
    "hidden_dim"   : 512,
    "temperature"  : 0.05,
    "mse_weight"   : 1.0,
    "batch_size"   : 256,
    "lr"           : 1e-3,
    "weight_decay" : 1e-4,
    "epochs"       : 20,
    "warmup_frac"  : 0.05,
    "max_grad_norm": 1.0,
    "num_workers"  : 0,        # 0 required on Windows
    "emb_path"     : os.path.join(SAVE_DIR, "triplet_embeddings_768.npz"),
    "save_dir"     : COMPRESSOR_DIR,
}


# Dataset
class TripletEmbDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.anchors   = F.normalize(torch.from_numpy(data["anchors"]).float(),   p=2, dim=-1)
        self.positives = F.normalize(torch.from_numpy(data["positives"]).float(), p=2, dim=-1)
        self.negatives = F.normalize(torch.from_numpy(data["negatives"]).float(), p=2, dim=-1)
        print(f"  Loaded {len(self.anchors):,} triplets | dim={self.anchors.shape[1]}")

    def __len__(self): return len(self.anchors)

    def __getitem__(self, idx):
        return self.anchors[idx], self.positives[idx], self.negatives[idx]


# Autoencoder
class EmbeddingAutoencoder(nn.Module):
    """Neural compressor: in_dim → hid_dim → out_dim (+ decoder for MSE)."""
    def __init__(self, in_dim=768, hid_dim=512, out_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hid_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hid_dim, out_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, hid_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hid_dim, in_dim),
        )

    def forward(self, x):
        encoded = F.normalize(self.encoder(x), p=2, dim=-1, eps=1e-6)
        decoded = F.normalize(self.decoder(encoded), p=2, dim=-1, eps=1e-6)
        return encoded, decoded

    def encode_only(self, x):
        return F.normalize(self.encoder(x), p=2, dim=-1, eps=1e-6)


# Hybrid Loss: InfoNCE + MSE
class InfoNCEWithHardNegAndMSE(nn.Module):
    def __init__(self, temperature=0.05, mse_weight=1.0):
        super().__init__()
        self.tau         = temperature
        self.mse_weight  = mse_weight
        self.mse_loss_fn = nn.MSELoss()

    def forward(self, z_a, z_p, z_n, rec_a, rec_p, rec_n, orig_a, orig_p, orig_n):
        B = z_a.size(0)

        pos_sim      = (z_a * z_p).sum(dim=1) / self.tau
        hard_neg_sim = (z_a * z_n).sum(dim=1) / self.tau
        sim_ap = torch.mm(z_a, z_p.T) / self.tau
        sim_an = torch.mm(z_a, z_n.T) / self.tau

        eye    = torch.eye(B, device=z_a.device).bool()
        sim_ap = sim_ap.masked_fill(eye, float("-inf"))

        all_logits   = torch.cat([pos_sim.unsqueeze(1), hard_neg_sim.unsqueeze(1), sim_ap, sim_an], dim=1)
        log_denom    = torch.logsumexp(all_logits, dim=1)
        loss_infonce = -(pos_sim - log_denom).mean()

        loss_mse = (self.mse_loss_fn(rec_a, orig_a) +
                    self.mse_loss_fn(rec_p, orig_p) +
                    self.mse_loss_fn(rec_n, orig_n)) / 3.0

        return loss_infonce + self.mse_weight * loss_mse, loss_infonce, loss_mse


# Cosine LR with Warmup
def get_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# Training Loop
def train_compressor(cfg, target_dim, loader):
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPRESSOR : BERT-768 → {target_dim}")
    print(f"{'='*60}")

    model     = EmbeddingAutoencoder(cfg["input_dim"], cfg["hidden_dim"], target_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_steps  = len(loader) * cfg["epochs"]
    warmup_steps = int(cfg["warmup_frac"] * total_steps)
    scheduler    = get_scheduler(optimizer, warmup_steps, total_steps)
    criterion    = InfoNCEWithHardNegAndMSE(cfg["temperature"], cfg["mse_weight"])
    best_loss    = float("inf")

    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1:02d}/{cfg['epochs']}", leave=False)

        for anc, pos, neg in pbar:
            anc, pos, neg = anc.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
            z_a, rec_a = model(anc)
            z_p, rec_p = model(pos)
            z_n, rec_n = model(neg)

            loss, _, _ = criterion(z_a, z_p, z_n, rec_a, rec_p, rec_n, anc, pos, neg)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            optimizer.step(); scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = epoch_loss / len(loader)
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(),
                       os.path.join(cfg["save_dir"], f"compressor_{target_dim}_best.pt"))

    print(f"   Finished 768 → {target_dim}  (Best Loss: {best_loss:.4f})")
    return model


# PHASE 3 - Ultimate Benchmark
#   Rows: Native-BERT (128/256/512/768), Native-Gemma (2048/2304),
#         PCA-{512/256/128}, Ours-{512/256/128}
#   Cols: STS-B, SST-2, QNLI, Recall@10, MRR@10 | Stor, Speed, Enc, VRAM

# Benchmark helpers (defined at module level for reuse)
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
from collections import defaultdict

_bench_native_models     = {}
_bench_native_tokenizers = {}
_bench_compressors       = {}


@torch.no_grad()
def bench_encode_native(sentences, dim, batch_size=256, max_len=128):
    model     = _bench_native_models[dim]
    tokenizer = _bench_native_tokenizers[dim]
    embs = []
    for i in tqdm(range(0, len(sentences), batch_size), desc=f"  Native-{dim}", leave=False):
        b    = sentences[i : i + batch_size]
        toks = tokenizer(b, padding=True, truncation=True,
                         max_length=max_len, return_tensors="pt").to(DEVICE)
        out  = model(**toks)
        mask = toks["attention_mask"].unsqueeze(-1).float()
        e    = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        embs.append(F.normalize(e, p=2, dim=-1).cpu())
    return torch.cat(embs, dim=0)


@torch.no_grad()
def bench_compress(embs_in, dim, batch_size=256):
    comp  = _bench_compressors[dim]
    parts = []
    for i in range(0, len(embs_in), batch_size):
        b = embs_in[i : i + batch_size].to(DEVICE)
        parts.append(comp.encode_only(b).cpu())
    return torch.cat(parts, dim=0)


def bench_fit_pca(embs_tensor, n_components):
    pca = SklearnPCA(n_components=n_components, random_state=42)
    pca.fit(embs_tensor.numpy())
    return pca


def bench_apply_pca(pca, embs_tensor):
    out   = pca.transform(embs_tensor.numpy())
    norms = np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
    return torch.from_numpy(out / norms).float()


def pair_features(e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    return np.concatenate([np.abs(e1 - e2), e1 * e2], axis=1)


def search_speed_and_mem(d_cpu, q_cpu):
    if DEVICE == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    d_gpu, q_gpu = d_cpu.to(DEVICE), q_cpu.to(DEVICE)
    _ = torch.mm(q_gpu[:10], d_gpu.T)          # warmup
    if DEVICE == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    _  = torch.mm(q_gpu, d_gpu.T).topk(10, dim=1)
    if DEVICE == "cuda": torch.cuda.synchronize()
    ms  = (time.perf_counter() - t0) * 1000
    mem = torch.cuda.max_memory_allocated() / (1024**2) if DEVICE == "cuda" else 0.0
    return ms, mem


# All benchmark native dims (BERT + Gemma)
ALL_NATIVE_DIMS = BERT_DIMS + GEMMA_DIMS   # [128, 256, 512, 768, 2048, 2304]


# Entry point
if __name__ == "__main__":

    # Phase 2: Train compressors
    print("\n[Phase 2] Training neural compressors on BERT-768 triplets…")
    if not os.path.exists(CFG["emb_path"]):
        print(f"  ⚠️  Skipping Phase 2: {CFG['emb_path']} not found. (Did you run Phase 1 with BERT-768 enabled?)")
    else:
        dataset = TripletEmbDataset(CFG["emb_path"])
        loader  = DataLoader(
            dataset,
            batch_size  = CFG["batch_size"],
            shuffle     = True,
            num_workers = CFG["num_workers"],
            pin_memory  = (DEVICE == "cuda"),
            drop_last   = True,
        )

        for target_dim in CFG["target_dims"]:
            pt_path = os.path.join(CFG["save_dir"], f"compressor_{target_dim}_best.pt")
            if os.path.exists(pt_path):
                print(f"  Compressor 768 → {target_dim} already exists at {pt_path}. Skipping training.")
                continue
            train_compressor(CFG, target_dim, loader)

        json.dump(CFG, open(os.path.join(CFG["save_dir"], "config.json"), "w"), indent=2)
        print(f"\nPhase 2 complete - compressors saved to {COMPRESSOR_DIR}")


    # Phase 3: Benchmark
    print("\n[Phase 3] Loading all native models for benchmark…")
    print("  NOTE: Gemma models are gated - run `huggingface-cli login` first.\n")

    # Load native models (BERT + Gemma)
    for dim in ALL_NATIVE_DIMS:
        if dim not in MODEL_CONFIGS:
            print(f"  ⚠️  Skipping {dim}-dim: Not configured in MODEL_CONFIGS")
            continue
        cfg_entry = MODEL_CONFIGS[dim]
        print(f"  Loading {dim}-dim : {cfg_entry['name']}")
        try:
            m, tok = load_model_tokenizer(dim)
            _bench_native_models[dim]     = m
            _bench_native_tokenizers[dim] = tok
        except Exception as e:
            print(f"  ⚠️  Skipping {dim}-dim ({cfg_entry['name']}): {e}")

    # Load trained compressors
    print("\n[Phase 3] Loading trained compressors…")
    for dim in COMPRESS_DIMS:
        pt_path = os.path.join(COMPRESSOR_DIR, f"compressor_{dim}_best.pt")
        if not os.path.exists(pt_path):
            print(f"  ⚠️  Compressor file 768→{dim} not found at {pt_path}. Skipping.")
            continue
        comp    = EmbeddingAutoencoder(768, 512, dim).to(DEVICE)
        try:
            comp.load_state_dict(torch.load(pt_path, map_location=DEVICE, weights_only=True))
            print(f"  Loaded compressor 768→{dim}")
            comp.eval()
            _bench_compressors[dim] = comp
        except Exception as e:
            print(f"  ⚠️  Failed to load compressor 768→{dim}: {e}")

    results = defaultdict(dict)

    # Helper: only evaluate dims that were successfully loaded
    avail_native = [d for d in ALL_NATIVE_DIMS if d in _bench_native_models]

    # 1. STS-B
    print("\n[3a] Evaluating STS-B…")
    if avail_native:
        ds      = load_dataset("nyu-mll/glue", "stsb", split="validation")
        sents1  = ds["sentence1"]; sents2 = ds["sentence2"]
        sts_scores = np.array(ds["label"])

        e1_nat = {d: bench_encode_native(sents1, d) for d in avail_native}
        e2_nat = {d: bench_encode_native(sents2, d) for d in avail_native}

        pcas_sts = {}
        if 768 in avail_native:
            pcas_sts = {d: bench_fit_pca(torch.cat([e1_nat[768], e2_nat[768]]), d)
                        for d in COMPRESS_DIMS}

        def spearman_cos(a, b):
            cos   = F.cosine_similarity(a, b).numpy()
            sp, _ = spearmanr(cos, sts_scores)
            return float(sp)

        for d in avail_native:
            results[f"nat_{d}"]["sts"] = spearman_cos(e1_nat[d], e2_nat[d])
        if 768 in avail_native:
            for d in COMPRESS_DIMS:
                if d in pcas_sts:
                    results[f"pca_{d}"]["sts"]  = spearman_cos(
                        bench_apply_pca(pcas_sts[d], e1_nat[768]),
                        bench_apply_pca(pcas_sts[d], e2_nat[768]))
                if d in _bench_compressors:
                    results[f"ours_{d}"]["sts"] = spearman_cos(
                        bench_compress(e1_nat[768], d),
                        bench_compress(e2_nat[768], d))

    # 2. SST-2
    print("[3b] Evaluating SST-2…")
    if avail_native:
        tr_ds   = load_dataset("nyu-mll/glue", "sst2", split="train")
        va_ds   = load_dataset("nyu-mll/glue", "sst2", split="validation")
        tr_sent, tr_lbl = tr_ds["sentence"], tr_ds["label"]
        va_sent, va_lbl = va_ds["sentence"], va_ds["label"]

        tr_nat = {d: bench_encode_native(tr_sent, d) for d in avail_native}
        va_nat = {d: bench_encode_native(va_sent, d) for d in avail_native}
        pcas_sst = {}
        if 768 in avail_native:
            pcas_sst = {d: bench_fit_pca(tr_nat[768], d) for d in COMPRESS_DIMS}

        def lr_acc(tr_x, va_x, tr_y=tr_lbl, va_y=va_lbl):
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
            clf.fit(tr_x.numpy(), tr_y)
            return float(accuracy_score(va_y, clf.predict(va_x.numpy())))

        for d in avail_native:
            results[f"nat_{d}"]["sst2"] = lr_acc(tr_nat[d], va_nat[d])
        if 768 in avail_native:
            for d in COMPRESS_DIMS:
                if d in pcas_sst:
                    results[f"pca_{d}"]["sst2"]  = lr_acc(
                        bench_apply_pca(pcas_sst[d], tr_nat[768]),
                        bench_apply_pca(pcas_sst[d], va_nat[768]))
                if d in _bench_compressors:
                    results[f"ours_{d}"]["sst2"] = lr_acc(
                        bench_compress(tr_nat[768], d),
                        bench_compress(va_nat[768], d))

    # 3. QNLI
    print("[3c] Evaluating QNLI…")
    if avail_native:
        tr_q_ds = load_dataset("nyu-mll/glue", "qnli", split="train")
        va_q_ds = load_dataset("nyu-mll/glue", "qnli", split="validation")
        tr_q, tr_s, tr_ql = tr_q_ds["question"], tr_q_ds["sentence"], tr_q_ds["label"]
        va_q, va_s, va_ql = va_q_ds["question"], va_q_ds["sentence"], va_q_ds["label"]

        trq_nat = {d: bench_encode_native(tr_q, d) for d in avail_native}
        trs_nat = {d: bench_encode_native(tr_s, d) for d in avail_native}
        vaq_nat = {d: bench_encode_native(va_q, d) for d in avail_native}
        vas_nat = {d: bench_encode_native(va_s, d) for d in avail_native}

        pcas_qnli = {}
        if 768 in avail_native:
            pcas_qnli = {d: bench_fit_pca(torch.cat([trq_nat[768], trs_nat[768]]), d)
                         for d in COMPRESS_DIMS}

        def pair_acc(trq, trs, vaq, vas, tr_y=tr_ql, va_y=va_ql):
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
            clf.fit(pair_features(trq.numpy(), trs.numpy()), tr_y)
            return float(accuracy_score(va_y, clf.predict(pair_features(vaq.numpy(), vas.numpy()))))

        for d in avail_native:
            results[f"nat_{d}"]["qnli"] = pair_acc(trq_nat[d], trs_nat[d], vaq_nat[d], vas_nat[d])
        if 768 in avail_native:
            for d in COMPRESS_DIMS:
                if d in pcas_qnli:
                    results[f"pca_{d}"]["qnli"]  = pair_acc(
                        bench_apply_pca(pcas_qnli[d], trq_nat[768]),
                        bench_apply_pca(pcas_qnli[d], trs_nat[768]),
                        bench_apply_pca(pcas_qnli[d], vaq_nat[768]),
                        bench_apply_pca(pcas_qnli[d], vas_nat[768]))
                if d in _bench_compressors:
                    results[f"ours_{d}"]["qnli"] = pair_acc(
                        bench_compress(trq_nat[768], d), bench_compress(trs_nat[768], d),
                        bench_compress(vaq_nat[768], d), bench_compress(vas_nat[768], d))

    # 4. Quora Retrieval (Recall@10 / MRR@10)
    print("[3d] Evaluating Quora Retrieval…")
    if avail_native:
        qqp_ds    = load_dataset("nyu-mll/glue", "qqp", split="validation")
        pos_pairs = [r for r in qqp_ds if r["label"] == 1][:5000]
        qqp_q     = [r["question1"] for r in pos_pairs]
        qqp_d     = [r["question2"] for r in pos_pairs]
        N_Q       = len(qqp_q)

        eq_nat = {d: bench_encode_native(qqp_q, d) for d in avail_native}
        ed_nat = {d: bench_encode_native(qqp_d, d) for d in avail_native}
        pcas_qqp = {}
        if 768 in avail_native:
            pcas_qqp = {d: bench_fit_pca(ed_nat[768], d) for d in COMPRESS_DIMS}

        def retrieval_metrics(eq, ed):
            eq, ed   = eq.to(DEVICE), ed.to(DEVICE)
            sim      = torch.mm(eq, ed.T)
            _, top10 = sim.topk(10, dim=1)
            target   = torch.arange(N_Q, device=DEVICE).unsqueeze(1)
            matches  = (top10 == target)
            r10  = matches.any(dim=1).float().mean().item()
            nz   = matches.nonzero()
            mrr  = (1.0 / (nz[:, 1] + 1).float()).sum().item() / N_Q if len(nz) else 0.0
            return float(r10), float(mrr)

        for d in avail_native:
            r10, mrr = retrieval_metrics(eq_nat[d], ed_nat[d])
            results[f"nat_{d}"]["r10"], results[f"nat_{d}"]["mrr"] = r10, mrr
        if 768 in avail_native:
            for d in COMPRESS_DIMS:
                if d in pcas_qqp:
                    r10, mrr = retrieval_metrics(bench_apply_pca(pcas_qqp[d], eq_nat[768]),
                                                 bench_apply_pca(pcas_qqp[d], ed_nat[768]))
                    results[f"pca_{d}"]["r10"], results[f"pca_{d}"]["mrr"] = r10, mrr
                if d in _bench_compressors:
                    r10, mrr = retrieval_metrics(bench_compress(eq_nat[768], d),
                                                 bench_compress(ed_nat[768], d))
                    results[f"ours_{d}"]["r10"], results[f"ours_{d}"]["mrr"] = r10, mrr

    # 5. Efficiency (Storage / Search Speed / VRAM)
    print("[3e] Evaluating Efficiency…")
    N_DOCS, N_BENCH_Q = 100_000, 1_000

    # Storage is determined by embedding dimension
    for d in avail_native:
        results[f"nat_{d}"]["stor"] = N_DOCS * d * 4 / (1024**2)
        rand_d = F.normalize(torch.randn(N_DOCS,   d), p=2, dim=1)
        rand_q = F.normalize(torch.randn(N_BENCH_Q, d), p=2, dim=1)
        s, mem = search_speed_and_mem(rand_d, rand_q)
        results[f"nat_{d}"]["s_ms"] = s
        results[f"nat_{d}"]["mem"]  = mem

    if 768 in avail_native:
        embs_768_rand  = F.normalize(torch.randn(N_DOCS,    768), p=2, dim=1)
        query_768_rand = F.normalize(torch.randn(N_BENCH_Q, 768), p=2, dim=1)
        for d in COMPRESS_DIMS:
            stor = N_DOCS * d * 4 / (1024**2)
            # PCA
            if d in pcas_qqp:
                t0 = time.perf_counter()
                pca_docs = bench_apply_pca(pcas_qqp[d], embs_768_rand)
                t_pca    = (time.perf_counter() - t0) * 1000
                results[f"pca_{d}"]["stor"] = stor
                results[f"pca_{d}"]["enc"]  = t_pca
                s, mem = search_speed_and_mem(pca_docs, bench_apply_pca(pcas_qqp[d], query_768_rand))
                results[f"pca_{d}"]["s_ms"] = s
                results[f"pca_{d}"]["mem"]  = mem
            # Ours
            if d in _bench_compressors:
                t0 = time.perf_counter()
                comp_docs = bench_compress(embs_768_rand, d)
                t_comp    = (time.perf_counter() - t0) * 1000
                results[f"ours_{d}"]["stor"] = stor
                results[f"ours_{d}"]["enc"]  = t_comp
                s, mem = search_speed_and_mem(comp_docs, bench_compress(query_768_rand, d))
                results[f"ours_{d}"]["s_ms"] = s
                results[f"ours_{d}"]["mem"]  = mem

    # Print Results Table
    W = 125
    print("\n" + "=" * W)
    print(f"  BENCHMARK RESULTS")
    print("=" * W)
    header = (f"{'Model':<16} | {'STS-B':>7} | {'SST-2':>7} | {'QNLI':>7} | "
              f"{'Rcl@10':>7} | {'MRR@10':>7} || {'Stor(MB)':>9} | "
              f"{'Srch(ms)':>9} | {'Enc(ms)':>9} | {'VRAM(MB)':>9}")
    print(header)
    print("-" * W)

    def print_row(key, label):
        r   = results.get(key, {})
        if not r:
            return
        sts  = f"{r['sts']:7.4f}"   if "sts"  in r else "    N/A"
        sst  = f"{r['sst2']:7.4f}"  if "sst2" in r else "    N/A"
        qnli = f"{r['qnli']:7.4f}"  if "qnli" in r else "    N/A"
        r10  = f"{r['r10']:7.4f}"   if "r10"  in r else "    N/A"
        mrr  = f"{r['mrr']:7.4f}"   if "mrr"  in r else "    N/A"
        stor = f"{r['stor']:9.1f}"  if "stor" in r else "      N/A"
        sms  = f"{r['s_ms']:9.1f}"  if "s_ms" in r else "      N/A"
        enc  = f"{r['enc']:9.1f}"   if "enc"  in r else "      N/A"
        mem  = f"{r['mem']:9.1f}"   if "mem"  in r else "      N/A"
        print(f"{label:<16} | {sts} | {sst} | {qnli} | {r10} | {mrr} || {stor} | {sms} | {enc} | {mem}")

    # BERT native rows
    print("── BERT Native ──")
    for d in BERT_DIMS:
        if d in avail_native:
            print_row(f"nat_{d}", f"BERT-{d}")
    print("-" * W)

    # Gemma native rows
    print("── Gemma Native ──")
    gemma_labels = {2048: "Gemma-2B-2048", 2304: "Gemma2-2B-2304"}
    for d in GEMMA_DIMS:
        if d in avail_native:
            print_row(f"nat_{d}", gemma_labels.get(d, f"Gemma-{d}"))
    print("-" * W)

    # Compression rows
    print("── Compression (BERT-768 source) ──")
    for d in COMPRESS_DIMS:
        print_row(f"pca_{d}",  f"PCA-{d}")
        print_row(f"ours_{d}", f"Ours-{d}")
        print("-" * W)

    # Save results as JSON
    out_path = os.path.join(BASE_DIR, "benchmark_results.json")
    json.dump(dict(results), open(out_path, "w"), indent=2)
    print(f"\n Full results saved → {out_path}")