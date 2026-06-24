# ================================================================
# Generate Triplet Embeddings for Gemma Models
# Trains MRL and Autoencoder Compressor
# Evaluates Native, INT8, MRL, and Ours
# ================================================================

import os
import gc
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
from collections import defaultdict
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")

# Configuration
MODELS_CONFIG = {
    "gemma-2b": {"path": "google/gemma-2b", "native_dim": 2048},
    "gemma2-2b": {"path": "google/gemma-2-2b", "native_dim": 2304}
}

SELECTED_MODEL = "gemma-2b"
NATIVE_DIM = MODELS_CONFIG[SELECTED_MODEL]["native_dim"]
MODEL_PATH = MODELS_CONFIG[SELECTED_MODEL]["path"]

BATCH_SIZE_ENCODE = 64
BATCH_SIZE_EVAL = 128
MAX_LENGTH = 128
USE_FP16 = True
MAX_TRIPLETS = None

SAVE_DIR = "./output/contrastive_embs"
COMPRESSOR_DIR = "./output/multi_compressors"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(COMPRESSOR_DIR, exist_ok=True)

TARGET_DIMS = [1024, 512, 256, 128]

# 1. Build Triplets
def build_triplets(dataset) -> list[tuple[str, str, str]]:
    prem2hyp = {}
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

print("\n[1/3] Loading Data...")
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

print(f"      Total triplets: {len(all_triplets):,}")
del all_triplets, snli_triplets, mnli_triplets; gc.collect()

# 2. Helper Functions
def last_token_pool(last_hidden_state, attention_mask):
    last_idx = attention_mask.cumsum(dim=1).argmax(dim=1)
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), last_idx]

@torch.no_grad()
def encode(sentences: list[str], model, tokenizer) -> np.ndarray:
    all_embs = []
    for i in tqdm(range(0, len(sentences), BATCH_SIZE_ENCODE), desc="  encoding", leave=False):
        batch = sentences[i : i + BATCH_SIZE_ENCODE]
        enc   = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
        out   = model(**enc)
        emb   = last_token_pool(out.last_hidden_state, enc["attention_mask"])
        emb   = F.normalize(emb, p=2, dim=1)
        all_embs.append(emb.cpu().float().numpy())
    return np.vstack(all_embs)

# 3. Encode
print(f"\n[2/3] Encoding Native Dimension ({NATIVE_DIM}) using {MODEL_PATH}...")
metadata = {"max_length": MAX_LENGTH, "pooling": "last_token", "total_triplets": len(anchors), "model": MODEL_PATH}

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

dtype = torch.float16 if USE_FP16 else torch.float32
model = AutoModel.from_pretrained(MODEL_PATH, torch_dtype=dtype, trust_remote_code=True).eval().to(DEVICE)

t0 = time.time()
anchor_embs = encode(anchors, model, tokenizer)
gc.collect(); torch.cuda.empty_cache()

pos_embs    = encode(positives, model, tokenizer)
gc.collect(); torch.cuda.empty_cache()

neg_embs    = encode(negatives, model, tokenizer)
gc.collect(); torch.cuda.empty_cache()

elapsed = time.time() - t0
print(f"  Done in {elapsed/60:.1f} min")

npz_path = os.path.join(SAVE_DIR, "triplet_embeddings_native.npz")
np.savez_compressed(npz_path, anchors=anchor_embs, positives=pos_embs, negatives=neg_embs)
print(f"  Saved -> {npz_path} ({os.path.getsize(npz_path)/1e6:.1f} MB)")

del model, tokenizer, anchor_embs, pos_embs, neg_embs
gc.collect(); torch.cuda.empty_cache()

text_path = os.path.join(SAVE_DIR, "triplet_sentences.npz")
np.savez_compressed(text_path, anchors=np.array(anchors, dtype=object),
                    positives=np.array(positives, dtype=object), negatives=np.array(negatives, dtype=object))

with open(os.path.join(SAVE_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nAll embeddings saved successfully to {SAVE_DIR}")

# ================================================================
# Train Compressors and MRL Head
# ================================================================

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

print("\nLoading Native-dim triplet embeddings for training...")
dataset = TripletEmbDataset(os.path.join(SAVE_DIR, "triplet_embeddings_native.npz"))
loader  = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

class EmbeddingAutoencoder(nn.Module):
    def __init__(self, in_dim=2048, hid_dim=1024, out_dim=128):
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

class InfoNCEWithHardNegAndMSE(nn.Module):
    def __init__(self, temperature=0.05, mse_weight=1.0):
        super().__init__()
        self.tau = temperature
        self.mse_weight = mse_weight
        self.mse_loss_fn = nn.MSELoss()

    def forward(self, z_a, z_p, z_n, rec_a, rec_p, rec_n, orig_a, orig_p, orig_n):
        B = z_a.size(0)

        pos_sim = (z_a * z_p).sum(dim=1) / self.tau
        hard_neg_sim = (z_a * z_n).sum(dim=1) / self.tau
        sim_ap = torch.mm(z_a, z_p.T) / self.tau
        sim_an = torch.mm(z_a, z_n.T) / self.tau

        eye = torch.eye(B, device=z_a.device).bool()
        sim_ap = sim_ap.masked_fill(eye, float('-inf'))

        all_logits = torch.cat([pos_sim.unsqueeze(1), hard_neg_sim.unsqueeze(1), sim_ap, sim_an], dim=1)
        log_denom = torch.logsumexp(all_logits, dim=1)
        loss_infonce = -(pos_sim - log_denom).mean()

        loss_mse = (self.mse_loss_fn(rec_a, orig_a) +
                    self.mse_loss_fn(rec_p, orig_p) +
                    self.mse_loss_fn(rec_n, orig_n)) / 3.0

        return loss_infonce + (self.mse_weight * loss_mse)

def get_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps: return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_autoencoder(target_dim, loader):
    print(f"\nTraining Autoencoder: {NATIVE_DIM} -> {target_dim}")

    hid_dim = 1024 if NATIVE_DIM >= 2048 else 512
    model = EmbeddingAutoencoder(in_dim=NATIVE_DIM, hid_dim=hid_dim, out_dim=target_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 20
    total_steps  = len(loader) * epochs
    warmup_steps = int(0.05 * total_steps)
    scheduler    = get_scheduler(optimizer, warmup_steps, total_steps)
    criterion    = InfoNCEWithHardNegAndMSE(temperature=0.05, mse_weight=1.0)

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1:02d}/{epochs}", leave=False)

        for a, p, n in pbar:
            a, p, n = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
            z_a, rec_a = model(a)
            z_p, rec_p = model(p)
            z_n, rec_n = model(n)

            loss = criterion(z_a, z_p, z_n, rec_a, rec_p, rec_n, a, p, n)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        avg_loss = epoch_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(COMPRESSOR_DIR, f"autoencoder_{target_dim}_best.pt"))

    print(f"  Finished Autoencoder {NATIVE_DIM} -> {target_dim} (Best Loss: {best_loss:.4f})")

for dim in TARGET_DIMS:
    train_autoencoder(dim, loader)

class MRLProjectionHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x):
        return self.proj(x)

class MRLNestedInfoNCE(nn.Module):
    def __init__(self, target_dims, temperature=0.05):
        super().__init__()
        self.target_dims = target_dims
        self.tau = temperature

    def forward(self, z_a, z_p, z_n):
        B = z_a.size(0)
        eye = torch.eye(B, device=z_a.device).bool()
        loss = 0.0

        for dim in self.target_dims:
            a_d = F.normalize(z_a[:, :dim], p=2, dim=-1)
            p_d = F.normalize(z_p[:, :dim], p=2, dim=-1)
            n_d = F.normalize(z_n[:, :dim], p=2, dim=-1)

            pos_sim = (a_d * p_d).sum(dim=1) / self.tau
            hard_neg_sim = (a_d * n_d).sum(dim=1) / self.tau
            sim_ap = torch.mm(a_d, p_d.T) / self.tau
            sim_an = torch.mm(a_d, n_d.T) / self.tau

            sim_ap = sim_ap.masked_fill(eye, float('-inf'))

            all_logits = torch.cat([pos_sim.unsqueeze(1), hard_neg_sim.unsqueeze(1), sim_ap, sim_an], dim=1)
            log_denom = torch.logsumexp(all_logits, dim=1)
            loss += -(pos_sim - log_denom).mean()

        return loss / len(self.target_dims)

def train_mrl_head(loader):
    print(f"\nTraining MRL Projection Head ({NATIVE_DIM} -> {NATIVE_DIM})")

    model = MRLProjectionHead(dim=NATIVE_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 20
    total_steps  = len(loader) * epochs
    warmup_steps = int(0.05 * total_steps)
    scheduler    = get_scheduler(optimizer, warmup_steps, total_steps)
    criterion    = MRLNestedInfoNCE(TARGET_DIMS, temperature=0.05)

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1:02d}/{epochs}", leave=False)

        for a, p, n in pbar:
            a, p, n = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
            z_a = model(a)
            z_p = model(p)
            z_n = model(n)

            loss = criterion(z_a, z_p, z_n)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        avg_loss = epoch_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(COMPRESSOR_DIR, "mrl_head_best.pt"))

    print(f"  Finished MRL Head (Best Loss: {best_loss:.4f})")

train_mrl_head(loader)

# ================================================================
# Evaluation Benchmark
# ================================================================

print("\n[3/3] Running Evaluation...")

print("  Loading Compressors and MRL Head...")
autoencoders = {}
hid_dim = 1024 if NATIVE_DIM >= 2048 else 512
for dim in TARGET_DIMS:
    comp = EmbeddingAutoencoder(in_dim=NATIVE_DIM, hid_dim=hid_dim, out_dim=dim).to(DEVICE)
    try:
        comp.load_state_dict(torch.load(os.path.join(COMPRESSOR_DIR, f"autoencoder_{dim}_best.pt"), map_location=DEVICE))
    except Exception:
        pass
    comp.eval()
    autoencoders[dim] = comp

mrl_head = MRLProjectionHead(dim=NATIVE_DIM).to(DEVICE)
try:
    mrl_head.load_state_dict(torch.load(os.path.join(COMPRESSOR_DIR, "mrl_head_best.pt"), map_location=DEVICE))
except Exception:
    pass
mrl_head.eval()

@torch.no_grad()
def encode_native_eval(sentences):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
        
    dtype = torch.float16 if USE_FP16 else torch.float32
    model = AutoModel.from_pretrained(MODEL_PATH, torch_dtype=dtype, trust_remote_code=True).eval().to(DEVICE)
    
    embs = []
    for i in tqdm(range(0, len(sentences), BATCH_SIZE_EVAL), desc=f"  Eval-Native", leave=False):
        b = sentences[i : i+BATCH_SIZE_EVAL]
        toks = tokenizer(b, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
        out = model(**toks)
        e = last_token_pool(out.last_hidden_state, toks["attention_mask"])
        embs.append(F.normalize(e, p=2, dim=-1).cpu())
        
    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()
    return torch.cat(embs, dim=0)

def quantize_int8(embs):
    abs_max = torch.max(torch.abs(embs), dim=1, keepdim=True).values
    scale = 127.0 / torch.clamp(abs_max, min=1e-5)
    return torch.round(embs * scale).to(torch.int8), scale

def dequantize_int8(int8_embs, scale):
    return F.normalize((int8_embs.float() / scale).float(), p=2, dim=-1)

@torch.no_grad()
def compress_ours(embs_native, dim):
    comp = autoencoders[dim]
    parts = []
    for i in range(0, len(embs_native), BATCH_SIZE_EVAL * 4):
        b = embs_native[i : i+BATCH_SIZE_EVAL * 4].to(DEVICE)
        parts.append(comp.encoder(b).cpu())
    return F.normalize(torch.cat(parts, dim=0), p=2, dim=-1)

@torch.no_grad()
def compress_mrl(embs_native, dim):
    parts = []
    for i in range(0, len(embs_native), BATCH_SIZE_EVAL * 4):
        b = embs_native[i : i+BATCH_SIZE_EVAL * 4].to(DEVICE)
        proj = mrl_head(b)
        parts.append(proj[:, :dim].cpu())
    return F.normalize(torch.cat(parts, dim=0), p=2, dim=-1)

def pair_features(e1, e2):
    return np.concatenate([np.abs(e1 - e2), e1 * e2], axis=1)

results = defaultdict(dict)

print("\nEvaluating STS-B...")
ds = load_dataset("glue", "stsb", split="validation")
sents1, sents2, scores = ds["sentence1"], ds["sentence2"], np.array(ds["label"])

e1_nat = encode_native_eval(sents1)
e2_nat = encode_native_eval(sents2)

e1_int8, scale1 = quantize_int8(e1_nat)
e2_int8, scale2 = quantize_int8(e2_nat)

def get_spearman(e1, e2):
    cos = F.cosine_similarity(e1, e2).numpy()
    sp, _ = spearmanr(cos, scores)
    return sp

results["Native"]["sts"] = get_spearman(e1_nat, e2_nat)
results["INT8"]["sts"]   = get_spearman(dequantize_int8(e1_int8, scale1), dequantize_int8(e2_int8, scale2))

for d in TARGET_DIMS:
    results[f"MRL_{d}"]["sts"]  = get_spearman(compress_mrl(e1_nat, d), compress_mrl(e2_nat, d))
    results[f"Ours_{d}"]["sts"] = get_spearman(compress_ours(e1_nat, d), compress_ours(e2_nat, d))

print("Evaluating SST-2...")
train = load_dataset("glue", "sst2", split="train")
val   = load_dataset("glue", "sst2", split="validation")
tr_sents, tr_lbls = train["sentence"][:10000], train["label"][:10000]
va_sents, va_lbls = val["sentence"], val["label"]

tr_nat = encode_native_eval(tr_sents)
va_nat = encode_native_eval(va_sents)

tri, scale_tri = quantize_int8(tr_nat)
vai, scale_vai = quantize_int8(va_nat)

def get_acc(tr_x, va_x):
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(tr_x.numpy(), tr_lbls)
    return accuracy_score(va_lbls, clf.predict(va_x.numpy()))

results["Native"]["sst2"] = get_acc(tr_nat, va_nat)
results["INT8"]["sst2"]   = get_acc(dequantize_int8(tri, scale_tri), dequantize_int8(vai, scale_vai))

for d in TARGET_DIMS:
    results[f"MRL_{d}"]["sst2"]  = get_acc(compress_mrl(tr_nat, d), compress_mrl(va_nat, d))
    results[f"Ours_{d}"]["sst2"] = get_acc(compress_ours(tr_nat, d), compress_ours(va_nat, d))

print("Evaluating QNLI...")
train = load_dataset("glue", "qnli", split="train")
val   = load_dataset("glue", "qnli", split="validation")
tr_q, tr_s, tr_lbls = train["question"][:10000], train["sentence"][:10000], train["label"][:10000]
va_q, va_s, va_lbls = val["question"], val["sentence"], val["label"]

trq_nat = encode_native_eval(tr_q)
trs_nat = encode_native_eval(tr_s)
vaq_nat = encode_native_eval(va_q)
vas_nat = encode_native_eval(va_s)

trqi, scale_trqi = quantize_int8(trq_nat); trsi, scale_trsi = quantize_int8(trs_nat)
vaqi, scale_vaqi = quantize_int8(vaq_nat); vasi, scale_vasi = quantize_int8(vas_nat)

def get_acc_pairs(trq, trs, vaq, vas):
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(pair_features(trq.numpy(), trs.numpy()), tr_lbls)
    return accuracy_score(va_lbls, clf.predict(pair_features(vaq.numpy(), vas.numpy())))

results["Native"]["qnli"] = get_acc_pairs(trq_nat, trs_nat, vaq_nat, vas_nat)
results["INT8"]["qnli"]   = get_acc_pairs(dequantize_int8(trqi, scale_trqi), dequantize_int8(trsi, scale_trsi),
                                          dequantize_int8(vaqi, scale_vaqi), dequantize_int8(vasi, scale_vasi))

for d in TARGET_DIMS:
    results[f"MRL_{d}"]["qnli"]  = get_acc_pairs(compress_mrl(trq_nat, d), compress_mrl(trs_nat, d),
                                                 compress_mrl(vaq_nat, d), compress_mrl(vas_nat, d))
    results[f"Ours_{d}"]["qnli"] = get_acc_pairs(compress_ours(trq_nat, d), compress_ours(trs_nat, d),
                                                 compress_ours(vaq_nat, d), compress_ours(vas_nat, d))

print("Evaluating Quora Retrieval...")
ds = load_dataset("glue", "qqp", split="validation")
pos_pairs = [r for r in ds if r["label"] == 1][:5000]
queries, docs = [r["question1"] for r in pos_pairs], [r["question2"] for r in pos_pairs]

eq_nat = encode_native_eval(queries)
ed_nat = encode_native_eval(docs)

eqi, scale_eqi = quantize_int8(eq_nat)
edi, scale_edi = quantize_int8(ed_nat)

def get_retrieval(eq, ed):
    eq, ed = eq.to(DEVICE), ed.to(DEVICE)
    sim = torch.mm(eq, ed.T)
    _, top10 = sim.topk(10, dim=1)
    target = torch.arange(len(queries), device=DEVICE).unsqueeze(1)
    matches = (top10 == target)
    r10 = matches.any(dim=1).float().mean().item()
    mrr = (1.0 / (matches.nonzero()[:, 1] + 1).float()).sum().item() / len(queries)
    return r10, mrr

results["Native"]["r10"], results["Native"]["mrr"] = get_retrieval(eq_nat, ed_nat)
results["INT8"]["r10"], results["INT8"]["mrr"]     = get_retrieval(dequantize_int8(eqi, scale_eqi), dequantize_int8(edi, scale_edi))

for d in TARGET_DIMS:
    r10, mrr = get_retrieval(compress_mrl(eq_nat, d), compress_mrl(ed_nat, d))
    results[f"MRL_{d}"]["r10"], results[f"MRL_{d}"]["mrr"] = r10, mrr
    
    r10, mrr = get_retrieval(compress_ours(eq_nat, d), compress_ours(ed_nat, d))
    results[f"Ours_{d}"]["r10"], results[f"Ours_{d}"]["mrr"] = r10, mrr

print("Evaluating Efficiency...")
N_DOCS, N_QUERIES = 100_000, 1_000
embs_native_eff = F.normalize(torch.randn(N_DOCS, NATIVE_DIM), p=2, dim=1)
queries_native_eff = F.normalize(torch.randn(N_QUERIES, NATIVE_DIM), p=2, dim=1)

def bench_search(d_cpu, q_cpu):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    d_gpu, q_gpu = d_cpu.to(DEVICE), q_cpu.to(DEVICE)
    _ = torch.mm(q_gpu[:10], d_gpu.T); torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = torch.mm(q_gpu, d_gpu.T).topk(10, dim=1)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000, torch.cuda.max_memory_allocated() / (1024*1024)

results["Native"]["stor"] = N_DOCS * NATIVE_DIM * 4 / (1024*1024)
s, mem = bench_search(embs_native_eff, queries_native_eff)
results["Native"]["s_ms"], results["Native"]["mem"] = s, mem

results["INT8"]["stor"] = N_DOCS * NATIVE_DIM * 1 / (1024*1024)
results["INT8"]["s_ms"], results["INT8"]["mem"] = s, mem

for d in TARGET_DIMS:
    stor = N_DOCS * d * 4 / (1024*1024)
    results[f"MRL_{d}"]["stor"] = stor
    s, mem = bench_search(F.normalize(torch.randn(N_DOCS, d), p=2, dim=1), F.normalize(torch.randn(N_QUERIES, d), p=2, dim=1))
    results[f"MRL_{d}"]["s_ms"], results[f"MRL_{d}"]["mem"] = s, mem

    results[f"Ours_{d}"]["stor"] = stor
    results[f"Ours_{d}"]["s_ms"], results[f"Ours_{d}"]["mem"] = s, mem

print("\n" + "="*145)
print(f"  Final Benchmark")
print("="*145)
header = f"{'Model':<12} | {'STS-B':>7} | {'SST-2':>7} | {'QNLI':>7} | {'Rcl@10':>7} | {'MRR@10':>7} || {'Stor(MB)':>8} | {'Srch(ms)':>8} | {'GPU(MB)':>7} || {'Retain %':>8}"
print(header)
print("-" * 145)

native = results["Native"]

def pt(k, label):
    r = results[k]
    
    avg_retain = 0.0
    for metric in ["sts", "sst2", "qnli", "r10", "mrr"]:
        val = r[metric]
        nat_val = native[metric]
        if nat_val > 0:
            avg_retain += (val / nat_val) * 100
    avg_retain /= 5.0
    
    print(f"{label:<12} | {r['sts']:7.4f} | {r['sst2']:7.4f} | {r['qnli']:7.4f} | {r['r10']:7.4f} | {r['mrr']:7.4f} || {r.get('stor', 0.0):8.1f} | {r.get('s_ms', 0.0):8.1f} | {r.get('mem', 0.0):7.1f} || {avg_retain:7.2f}%")

pt("Native", f"Native-{NATIVE_DIM}")
pt("INT8", "INT8-Quant")
print("-" * 145)
for d in TARGET_DIMS:
    pt(f"MRL_{d}", f"MRL-{d}")
    pt(f"Ours_{d}", f"Ours-{d}")
    print("-" * 145)