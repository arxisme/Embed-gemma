
# ================================================================
# Generate Triplet Embeddings for Multiple BERT Dimensions
# (768, 512, 256, 128)
# ================================================================

import os
import gc
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")

# Dictionary of models mapped to their hidden dimensions
MODELS = {
    768: "bert-base-uncased",
    512: "google/bert_uncased_L-4_H-512_A-8",
    256: "google/bert_uncased_L-4_H-256_A-4",
    128: "google/bert_uncased_L-2_H-128_A-2"
}

BATCH_SIZE  = 128
MAX_LENGTH  = 128
USE_FP16    = True
MAX_TRIPLETS = None   # set e.g. 100_000 to quick-test
SAVE_DIR    = "/kaggle/working/contrastive_embs"

os.makedirs(SAVE_DIR, exist_ok=True)

# ── 1. Build Triplets ────────────────────────────────────────────
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

print("\n[1/2] Loading Data…")
snli_raw = load_dataset("stanfordnlp/snli", split="train", trust_remote_code=True)
snli_triplets = build_triplets(snli_raw)
del snli_raw; gc.collect()

mnli_raw = load_dataset("nyu-mll/multi_nli", split="train", trust_remote_code=True)
mnli_triplets = build_triplets(mnli_raw)
del mnli_raw; gc.collect()

all_triplets = snli_triplets + mnli_triplets
if MAX_TRIPLETS:
    import random; random.shuffle(all_triplets)
    all_triplets = all_triplets[:MAX_TRIPLETS]

anchors   = [t[0] for t in all_triplets]
positives = [t[1] for t in all_triplets]
negatives = [t[2] for t in all_triplets]

print(f"      Total triplets : {len(all_triplets):,}")
del all_triplets, snli_triplets, mnli_triplets; gc.collect()

# ── 2. Helper Functions ──────────────────────────────────────────
def mean_pool(last_hidden_state, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * mask_expanded, 1) / \
           torch.clamp(mask_expanded.sum(1), min=1e-9)

@torch.no_grad()
def encode(sentences: list[str], model, tokenizer) -> np.ndarray:
    all_embs = []
    for i in tqdm(range(0, len(sentences), BATCH_SIZE), desc="  encoding", leave=False):
        batch = sentences[i : i + BATCH_SIZE]
        enc   = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
        out   = model(**enc)
        emb   = mean_pool(out.last_hidden_state, enc["attention_mask"])
        emb   = F.normalize(emb, p=2, dim=1)
        all_embs.append(emb.cpu().float().numpy())
    return np.vstack(all_embs)

# ── 3. Loop Over Models & Encode ─────────────────────────────────
print("\n[2/2] Encoding across all dimensions…")
metadata = {"max_length": MAX_LENGTH, "pooling": "mean", "total_triplets": len(anchors)}

for dim, model_name in MODELS.items():
    print(f"\n{'='*50}")
    print(f"  Encoding {dim}-dim : {model_name}")
    print(f"{'='*50}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).eval().to(DEVICE)
    if USE_FP16 and DEVICE == "cuda":
        model = model.half()

    t0 = time.time()
    anchor_embs = encode(anchors, model, tokenizer)
    pos_embs    = encode(positives, model, tokenizer)
    neg_embs    = encode(negatives, model, tokenizer)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f} min")

    npz_path = os.path.join(SAVE_DIR, f"triplet_embeddings_{dim}.npz")
    np.savez_compressed(npz_path, anchors=anchor_embs, positives=pos_embs, negatives=neg_embs)

    metadata[f"model_{dim}"] = model_name
    print(f"  Saved → {npz_path} ({os.path.getsize(npz_path)/1e6:.1f} MB)")

    # Free memory before loading next model
    del model, tokenizer, anchor_embs, pos_embs, neg_embs
    gc.collect(); torch.cuda.empty_cache()

# -- Save text reference just once
text_path = os.path.join(SAVE_DIR, "triplet_sentences.npz")
np.savez_compressed(text_path, anchors=np.array(anchors, dtype=object),
                    positives=np.array(positives, dtype=object), negatives=np.array(negatives, dtype=object))

with open(os.path.join(SAVE_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ All embeddings saved successfully to {SAVE_DIR}!")

"""Train"""

# ================================================================
# Train Compressors for 512, 256, and 128 Dimensions
# Uses InfoNCE + MSE Reconstruction
# ================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os, json, math

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Config ───────────────────────────────────────────────────────
CFG = {
    "input_dim"     : 768,
    "target_dims"   : [512, 256, 128],  # Will train 3 separate models
    "hidden_dim"    : 512,              # Fixed hidden for encoder/decoder
    "temperature"   : 0.05,
    "mse_weight"    : 1.0,
    "batch_size"    : 256,
    "lr"            : 1e-3,
    "weight_decay"  : 1e-4,
    "epochs"        : 20,
    "warmup_frac"   : 0.05,
    "max_grad_norm" : 1.0,
    "num_workers"   : 2,
    "emb_path"      : "/kaggle/working/contrastive_embs/triplet_embeddings_768.npz",
    "save_dir"      : "/kaggle/working/multi_compressors",
}
os.makedirs(CFG["save_dir"], exist_ok=True)


# ── Dataset ──────────────────────────────────────────────────────
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


# ── Autoencoder ──────────────────────────────────────────────────
class EmbeddingAutoencoder(nn.Module):
    def __init__(self, in_dim=768, hid_dim=512, out_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hid_dim, out_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hid_dim, in_dim),
        )

    def forward(self, x):
        encoded = F.normalize(self.encoder(x), p=2, dim=-1, eps=1e-6)
        decoded = F.normalize(self.decoder(encoded), p=2, dim=-1, eps=1e-6)
        return encoded, decoded


# ── Hybrid Loss ──────────────────────────────────────────────────
class InfoNCEWithHardNegAndMSE(nn.Module):
    def __init__(self, temperature=0.05, mse_weight=1.0):
        super().__init__()
        self.tau = temperature
        self.mse_weight = mse_weight
        self.mse_loss_fn = nn.MSELoss()

    def forward(self, z_a, z_p, z_n, rec_a, rec_p, rec_n, orig_a, orig_p, orig_n):
        B = z_a.size(0)

        # InfoNCE
        pos_sim = (z_a * z_p).sum(dim=1) / self.tau
        hard_neg_sim = (z_a * z_n).sum(dim=1) / self.tau
        sim_ap = torch.mm(z_a, z_p.T) / self.tau
        sim_an = torch.mm(z_a, z_n.T) / self.tau

        eye = torch.eye(B, device=z_a.device).bool()
        sim_ap = sim_ap.masked_fill(eye, float('-inf'))

        all_logits = torch.cat([pos_sim.unsqueeze(1), hard_neg_sim.unsqueeze(1), sim_ap, sim_an], dim=1)
        log_denom = torch.logsumexp(all_logits, dim=1)
        loss_infonce = -(pos_sim - log_denom).mean()

        # MSE
        loss_mse = (self.mse_loss_fn(rec_a, orig_a) +
                    self.mse_loss_fn(rec_p, orig_p) +
                    self.mse_loss_fn(rec_n, orig_n)) / 3.0

        return loss_infonce + (self.mse_weight * loss_mse), loss_infonce, loss_mse


# ── LR Scheduler ────────────────────────────────────────────────
def get_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps: return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Training Loop ────────────────────────────────────────────────
def train_compressor(cfg, target_dim, loader):
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPRESSOR: 768 → {target_dim}")
    print(f"{'='*60}")

    model = EmbeddingAutoencoder(cfg["input_dim"], cfg["hidden_dim"], target_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    total_steps  = len(loader) * cfg["epochs"]
    warmup_steps = int(cfg["warmup_frac"] * total_steps)
    scheduler    = get_scheduler(optimizer, warmup_steps, total_steps)
    criterion    = InfoNCEWithHardNegAndMSE(cfg["temperature"], cfg["mse_weight"])

    best_loss = float('inf')

    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1:02d}/{cfg['epochs']}", leave=False)

        for anchor_768, pos_768, neg_768 in pbar:
            anchor_768, pos_768, neg_768 = anchor_768.to(DEVICE), pos_768.to(DEVICE), neg_768.to(DEVICE)

            z_a, rec_a = model(anchor_768)
            z_p, rec_p = model(pos_768)
            z_n, rec_n = model(neg_768)

            loss, l_nce, l_mse = criterion(z_a, z_p, z_n, rec_a, rec_p, rec_n, anchor_768, pos_768, neg_768)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        avg_loss = epoch_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(cfg["save_dir"], f"compressor_{target_dim}_best.pt"))

    print(f"  ✅ Finished 768 → {target_dim} (Best Loss: {best_loss:.4f})")
    return model


# ── Main ─────────────────────────────────────────────────────────
print("Loading 768-dim triplet embeddings...")
dataset = TripletEmbDataset(CFG["emb_path"])
loader  = DataLoader(dataset, batch_size=CFG["batch_size"], shuffle=True,
                     num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)

for dim in CFG["target_dims"]:
    train_compressor(CFG, dim, loader)

json.dump(CFG, open(os.path.join(CFG["save_dir"], "config.json"), "w"), indent=2)
print(f"\n✅ All compressors saved successfully to {CFG['save_dir']}!")

# ================================================================
# ULTIMATE EVALUATION BENCHMARK
# Evaluates Native, PCA, and Ours across 768, 512, 256, and 128 dims
# ================================================================

import time
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
from collections import defaultdict
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
MAX_LENGTH = 128

MODELS = {
    768: "bert-base-uncased",
    512: "google/bert_uncased_L-4_H-512_A-8",
    256: "google/bert_uncased_L-4_H-256_A-4",
    128: "google/bert_uncased_L-2_H-128_A-2"
}
COMPRESSOR_DIR = "/kaggle/working/multi_compressors"
DIMS = [512, 256, 128]

class EmbeddingAutoencoder(nn.Module):
    def __init__(self, in_dim=768, hid_dim=512, out_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hid_dim), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(hid_dim, out_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, hid_dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(hid_dim, in_dim)
        )
    def forward(self, x):
        return F.normalize(self.encoder(x), p=2, dim=-1, eps=1e-6)

print("\n[1/3] Loading Native Models...")
native_models = {}
native_tokenizers = {}
for dim, m_name in MODELS.items():
    native_tokenizers[dim] = AutoTokenizer.from_pretrained(m_name)
    native_models[dim] = AutoModel.from_pretrained(m_name).eval().to(DEVICE)

print("\n[2/3] Loading Compressors...")
compressors = {}
for dim in DIMS:
    pt_path = os.path.join(COMPRESSOR_DIR, f"compressor_{dim}_best.pt")
    comp = EmbeddingAutoencoder(768, 512, dim).to(DEVICE)
    try:
        comp.load_state_dict(torch.load(pt_path, map_location=DEVICE))
    except Exception:
        print(f"  ⚠️ Warning: Could not load compressor for {dim}-dim")
    comp.eval()
    compressors[dim] = comp


# ── Helpers ──────────────────────────────────────────────────────
@torch.no_grad()
def encode_native(sentences, dim):
    model, tokenizer = native_models[dim], native_tokenizers[dim]
    embs = []
    for i in tqdm(range(0, len(sentences), BATCH_SIZE), desc=f"  Native-{dim}", leave=False):
        b = sentences[i : i+BATCH_SIZE]
        toks = tokenizer(b, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
        out = model(**toks)
        mask = toks["attention_mask"].unsqueeze(-1).float()
        e = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        embs.append(F.normalize(e, p=2, dim=-1).cpu())
    return torch.cat(embs, dim=0)

@torch.no_grad()
def compress(embs_768, dim):
    comp = compressors[dim]
    parts = []
    for i in range(0, len(embs_768), BATCH_SIZE):
        b = embs_768[i : i+BATCH_SIZE].to(DEVICE)
        parts.append(comp(b).cpu())
    return torch.cat(parts, dim=0)

def fit_pca(embs, dim):
    pca = PCA(n_components=dim, random_state=42)
    pca.fit(embs)
    return pca

def apply_pca(pca, embs):
    res = pca.transform(embs.numpy())
    norms = np.maximum(np.linalg.norm(res, axis=1, keepdims=True), 1e-6)
    return torch.from_numpy(res / norms).float()

def pair_features(e1, e2):
    return np.concatenate([np.abs(e1 - e2), e1 * e2], axis=1)

# Global results dictionary
results = defaultdict(dict)


# ── 1. STS-B ─────────────────────────────────────────────────────
print("\n[3/3] Evaluating STS-B...")
ds = load_dataset("glue", "stsb", split="validation")
sents1, sents2, scores = ds["sentence1"], ds["sentence2"], np.array(ds["label"])

e1_nat = {d: encode_native(sents1, d) for d in [768] + DIMS}
e2_nat = {d: encode_native(sents2, d) for d in [768] + DIMS}

pcas = {d: fit_pca(torch.cat([e1_nat[768], e2_nat[768]]), d) for d in DIMS}

def get_spearman(e1, e2):
    cos = F.cosine_similarity(e1, e2).numpy()
    sp, _ = spearmanr(cos, scores)
    return sp

results["768"]["sts"] = get_spearman(e1_nat[768], e2_nat[768])
for d in DIMS:
    results[f"nat_{d}"]["sts"]  = get_spearman(e1_nat[d], e2_nat[d])
    results[f"pca_{d}"]["sts"]  = get_spearman(apply_pca(pcas[d], e1_nat[768]), apply_pca(pcas[d], e2_nat[768]))
    results[f"ours_{d}"]["sts"] = get_spearman(compress(e1_nat[768], d), compress(e2_nat[768], d))


# ── 2. SST-2 ─────────────────────────────────────────────────────
print("Evaluating SST-2...")
train = load_dataset("glue", "sst2", split="train")
val   = load_dataset("glue", "sst2", split="validation")
tr_sents, tr_lbls = train["sentence"], train["label"]
va_sents, va_lbls = val["sentence"], val["label"]

tr_nat = {d: encode_native(tr_sents, d) for d in [768] + DIMS}
va_nat = {d: encode_native(va_sents, d) for d in [768] + DIMS}

pcas = {d: fit_pca(tr_nat[768], d) for d in DIMS}

def get_acc(tr_x, va_x):
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(tr_x.numpy(), tr_lbls)
    return accuracy_score(va_lbls, clf.predict(va_x.numpy()))

results["768"]["sst2"] = get_acc(tr_nat[768], va_nat[768])
for d in DIMS:
    results[f"nat_{d}"]["sst2"]  = get_acc(tr_nat[d], va_nat[d])
    results[f"pca_{d}"]["sst2"]  = get_acc(apply_pca(pcas[d], tr_nat[768]), apply_pca(pcas[d], va_nat[768]))
    results[f"ours_{d}"]["sst2"] = get_acc(compress(tr_nat[768], d), compress(va_nat[768], d))


# ── 3. QNLI ──────────────────────────────────────────────────────
print("Evaluating QNLI...")
train = load_dataset("glue", "qnli", split="train")
val   = load_dataset("glue", "qnli", split="validation")
tr_q, tr_s, tr_lbls = train["question"], train["sentence"], train["label"]
va_q, va_s, va_lbls = val["question"], val["sentence"], val["label"]

trq_nat = {d: encode_native(tr_q, d) for d in [768] + DIMS}
trs_nat = {d: encode_native(tr_s, d) for d in [768] + DIMS}
vaq_nat = {d: encode_native(va_q, d) for d in [768] + DIMS}
vas_nat = {d: encode_native(va_s, d) for d in [768] + DIMS}

pcas = {d: fit_pca(torch.cat([trq_nat[768], trs_nat[768]]), d) for d in DIMS}

def get_acc_pairs(trq, trs, vaq, vas):
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(pair_features(trq.numpy(), trs.numpy()), tr_lbls)
    return accuracy_score(va_lbls, clf.predict(pair_features(vaq.numpy(), vas.numpy())))

results["768"]["qnli"] = get_acc_pairs(trq_nat[768], trs_nat[768], vaq_nat[768], vas_nat[768])
for d in DIMS:
    results[f"nat_{d}"]["qnli"]  = get_acc_pairs(trq_nat[d], trs_nat[d], vaq_nat[d], vas_nat[d])
    results[f"pca_{d}"]["qnli"]  = get_acc_pairs(apply_pca(pcas[d], trq_nat[768]), apply_pca(pcas[d], trs_nat[768]),
                                                 apply_pca(pcas[d], vaq_nat[768]), apply_pca(pcas[d], vas_nat[768]))
    results[f"ours_{d}"]["qnli"] = get_acc_pairs(compress(trq_nat[768], d), compress(trs_nat[768], d),
                                                 compress(vaq_nat[768], d), compress(vas_nat[768], d))


# ── 4. Quora Retrieval (Recall/MRR) ──────────────────────────────
print("Evaluating Quora Retrieval...")
ds = load_dataset("glue", "qqp", split="validation")
pos_pairs = [r for r in ds if r["label"] == 1][:5000]
queries, docs = [r["question1"] for r in pos_pairs], [r["question2"] for r in pos_pairs]

eq_nat = {d: encode_native(queries, d) for d in [768] + DIMS}
ed_nat = {d: encode_native(docs, d) for d in [768] + DIMS}
pcas = {d: fit_pca(ed_nat[768], d) for d in DIMS}

def get_retrieval(eq, ed):
    eq, ed = eq.to(DEVICE), ed.to(DEVICE)
    sim = torch.mm(eq, ed.T)
    _, top10 = sim.topk(10, dim=1)
    target = torch.arange(len(queries), device=DEVICE).unsqueeze(1)
    matches = (top10 == target)
    r10 = matches.any(dim=1).float().mean().item()
    mrr = (1.0 / (matches.nonzero()[:, 1] + 1).float()).sum().item() / len(queries)
    return r10, mrr

r10, mrr = get_retrieval(eq_nat[768], ed_nat[768])
results["768"]["r10"], results["768"]["mrr"] = r10, mrr
for d in DIMS:
    r10, mrr = get_retrieval(eq_nat[d], ed_nat[d])
    results[f"nat_{d}"]["r10"], results[f"nat_{d}"]["mrr"] = r10, mrr
    r10, mrr = get_retrieval(apply_pca(pcas[d], eq_nat[768]), apply_pca(pcas[d], ed_nat[768]))
    results[f"pca_{d}"]["r10"], results[f"pca_{d}"]["mrr"] = r10, mrr
    r10, mrr = get_retrieval(compress(eq_nat[768], d), compress(ed_nat[768], d))
    results[f"ours_{d}"]["r10"], results[f"ours_{d}"]["mrr"] = r10, mrr


# ── 5. Efficiency ────────────────────────────────────────────────
print("Evaluating Efficiency...")
N_DOCS, N_QUERIES = 100_000, 1_000
embs_768 = F.normalize(torch.randn(N_DOCS, 768), p=2, dim=1)
queries_768 = F.normalize(torch.randn(N_QUERIES, 768), p=2, dim=1)

def bench_search(d_cpu, q_cpu):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    d_gpu, q_gpu = d_cpu.to(DEVICE), q_cpu.to(DEVICE)
    _ = torch.mm(q_gpu[:10], d_gpu.T); torch.cuda.synchronize() # Warmup
    t0 = time.perf_counter()
    _ = torch.mm(q_gpu, d_gpu.T).topk(10, dim=1)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000, torch.cuda.max_memory_allocated() / (1024*1024)

# 768 Efficiency
results["768"]["stor"] = N_DOCS * 768 * 4 / (1024*1024)
s, mem = bench_search(embs_768, queries_768)
results["768"]["s_ms"], results["768"]["mem"] = s, mem

for d in DIMS:
    stor = N_DOCS * d * 4 / (1024*1024)
    # Native
    results[f"nat_{d}"]["stor"] = stor
    s, mem = bench_search(F.normalize(torch.randn(N_DOCS, d), p=2, dim=1), F.normalize(torch.randn(N_QUERIES, d), p=2, dim=1))
    results[f"nat_{d}"]["s_ms"], results[f"nat_{d}"]["mem"] = s, mem
    # PCA
    t0 = time.perf_counter(); pca_docs = apply_pca(pcas[d], embs_768); t_pca = (time.perf_counter() - t0) * 1000
    results[f"pca_{d}"]["stor"] = stor; results[f"pca_{d}"]["enc"] = t_pca
    s, mem = bench_search(pca_docs, apply_pca(pcas[d], queries_768))
    results[f"pca_{d}"]["s_ms"], results[f"pca_{d}"]["mem"] = s, mem
    # Ours
    t0 = time.perf_counter(); comp_docs = compress(embs_768, d); t_comp = (time.perf_counter() - t0) * 1000
    results[f"ours_{d}"]["stor"] = stor; results[f"ours_{d}"]["enc"] = t_comp
    s, mem = bench_search(comp_docs, compress(queries_768, d))
    results[f"ours_{d}"]["s_ms"], results[f"ours_{d}"]["mem"] = s, mem


# ── FINAL OUTPUT TABLE ───────────────────────────────────────────
print("\n" + "="*115)
print(f"  FINAL BENCHMARK: 10 Representations across 8 Metrics")
print("="*115)
header = f"{'Model':<12} | {'STS-B':>7} | {'SST-2':>7} | {'QNLI':>7} | {'Rcl@10':>7} | {'MRR@10':>7} || {'Stor(MB)':>8} | {'Srch(ms)':>8} | {'Enc(ms)':>8} | {'GPU(MB)':>7}"
print(header)
print("-" * 115)

def pt(k, label):
    r = results[k]
    enc = f"{r.get('enc', 0):8.1f}" if "enc" in r else "     N/A"
    print(f"{label:<12} | {r['sts']:7.4f} | {r['sst2']:7.4f} | {r['qnli']:7.4f} | {r['r10']:7.4f} | {r['mrr']:7.4f} || {r['stor']:8.1f} | {r['s_ms']:8.1f} | {enc} | {r['mem']:7.1f}")

pt("768", "Native-768")
print("-" * 115)
for d in DIMS:
    pt(f"nat_{d}", f"Native-{d}")
    pt(f"pca_{d}", f"PCA-{d}")
    pt(f"ours_{d}", f"Ours-{d}")
    print("-" * 115)