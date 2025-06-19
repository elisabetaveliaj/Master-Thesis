# main_train.py

import os
import random
import json
from pathlib import Path
import numpy as np
import torch
from torch.amp import GradScaler, autocast

# reusable helpers
from ltn_core import (
    load_rules, load_triples, BilinearLTN,
    ltn_loss, validate, sample_batch, RuleBank
)

# ─────────────────── Reproducibility ───────────────────
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)

device   = 'cuda' if torch.cuda.is_available() else 'cpu'
scaler = GradScaler() if device == 'cuda' else None
DATA_DIR = Path("/home3/s5792010/LTNs/ltn_project/data/")
METRICS_PATH = DATA_DIR / "ltn_metrics.json"
TEST_METRICS_PATH = DATA_DIR / "ltn_test_metrics.json"

# ─────────────────── 1.  Load entity / relation maps ───────────────────
ent2id = json.load(open(DATA_DIR/"entities.json"))
rel2id = json.load(open(DATA_DIR/"relations.json"))
next_ent = max(ent2id.values()) + 1
next_rel = max(rel2id.values()) + 1
for special in ("LE", "GE"):
    if special not in rel2id:
        rel2id[special] = next_rel; next_rel += 1


# ─────────────────── 2.  Patient facts  ────────────────────
pat_rows = []
literal_value = {}                           # entity → float  (for comparators)

with open(DATA_DIR/"patient_facts.jsonl") as f:
    for line in f:
        h, r, t = item = json.loads(line).values()  # preserves key order
        # ---- entity bookkeeping ----
        for e in (h, t):
            if e not in ent2id:
                ent2id[e] = next_ent; next_ent += 1
        if r not in rel2id:
            rel2id[r] = next_rel; next_rel += 1
        pat_rows.append((ent2id[h], rel2id[r], ent2id[t]))

        # keep the *float* value for every literal node
        if str(t).startswith("num::"):
            literal_value[ent2id[t]] = float(t.split("::")[1])

PAT   = torch.as_tensor(pat_rows, dtype=torch.long)

# ─────────────────── 3.  KG triples ───────────────────
train_triples = load_triples(DATA_DIR/"train_triples.json", ent2id, rel2id)
valid_triples = load_triples(DATA_DIR/"valid_triples.json", ent2id, rel2id)
test_triples  = load_triples(DATA_DIR/"test_triples.json" , ent2id, rel2id)
KG = torch.as_tensor(train_triples, dtype=torch.long)       # (N,3)

# ─────────────────── 4.  Rules (multi‑var) ─────────────
rules, next_ent, next_rel = load_rules(DATA_DIR, ent2id, rel2id,
                                       next_ent, next_rel)

# give every rule a learnable log‑λ
for r in rules:
    r.log_lam = torch.nn.Parameter(torch.log(torch.tensor(r.lam, device=device)))
rule_bank = RuleBank(rules)     # just a container for those parameters

# ─────────────────── 5.  Build model & optimiser ───────
model = BilinearLTN(len(ent2id), len(rel2id), dim=256, literal_value=literal_value).to(device)

model.register_numeric_relations(rel2id)

optim = torch.optim.AdamW(
    list(model.parameters()) + list(rule_bank.parameters()),
    lr=1e-3, weight_decay=1e-5
)
sched  = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.5)
scaler = GradScaler() if device == 'cuda' else None

# helpers
entity_ids = torch.arange(len(ent2id), device=device)
rel_ids    = torch.arange(len(rel2id), device=device)
PAT, KG = PAT.to(device), KG.to(device)

# ─────────────────── 6.  Training loop ────────────────
import json
from pathlib import Path

# --------------------------------------------------------------------------
# Where to store the aggregated validation metrics
# --------------------------------------------------------------------------
METRICS_PATH = DATA_DIR / "ltn_metrics.json"


def train(alphas=(0.1, 0.5, 1.0, 1.5), *, epochs: int = 150, batch_size: int = 128):
    """Train the LTN model and persist dev‑set metrics to a JSON file.

    The function now keeps a running ``metrics`` list.  Every time we run a
    validation pass (every 10 epochs for each α) we append a log‑entry of the
    form ::

        {
            "alpha": 0.5,
            "epoch": 40,
            "loss": 0.2473,
            "AUC": 0.862,
            "Accuracy": 0.791,
            "MRR": 0.721,
            "Hits@1": 0.511,
            "Hits@3": 0.734,
            "Hits@10": 0.890
        }

    The list is written back to ``ltn_metrics.json`` after *every* validation
    so the logfile stays up‑to‑date even if training is interrupted.
    """

    metrics = [] 

    for α in alphas:
        num_batches = max(1, KG.size(0) // batch_size)
        print(f"\n=== α = {α} ===")

        for epoch in range(epochs):
            tot = 0.0
            for _ in range(num_batches):
                triples, labels = sample_batch(
                    batch_size, KG, PAT, entity_ids, device, neg_per_pos=1
                )
                print(labels.min().item(), labels.max().item())

                optim.zero_grad()
                bce, regularization = ltn_loss(triples, labels, model, rules, entity_ids)
                loss = bce + α * regularization

                if scaler is not None:
                    with autocast(device_type="cuda"):
                        scaler.scale(loss).backward()
                        scaler.unscale_(optim)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optim)
                        scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()

                tot += loss.item()
                
            torch.cuda.empty_cache()

            # ---- end of one epoch ---------------------------------------
            sched.step()
            avg_loss = tot / num_batches
            lr = optim.param_groups[0]['lr']
            print(f"Epoch {epoch:03d}   loss={avg_loss:.4f}  lr={lr:.1e}")

            # quick dev‑set check every 10 epochs
            if epoch % 10 == 0:
                stats = validate(
                    model,
                    valid_triples,
                    entity_ids,
                    device,
                    num_samples=50,
                    hits_ks=(1, 3, 10),
                )
                print("  → VAL:", {k: f"{v:.3f}" for k, v in stats.items()})

                # ---------------------------------------------------------
                # Persist the metrics snapshot
                # ---------------------------------------------------------
                log = {
                    "alpha": α,
                    "epoch": epoch,
                    "loss": round(avg_loss, 4),
                    **{k: round(v, 3) for k, v in stats.items()},
                }
                metrics.append(log)
                
                with open(METRICS_PATH, "w") as f:
                    json.dump(metrics, f, indent=2)

    # ----------------------------------------------------------------------
    # All alphas done  →  final checkpoint & metrics flush
    # ----------------------------------------------------------------------
    #torch.save(model.state_dict(), DATA_DIR / "ltn_final.pt")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics



# ─────────────────── 7.  Entry point ──────────────────
if __name__ == "__main__":
    train(alphas=(0.1,0.5,1.0,1.5))

    print("\n=== TEST RESULTS ===")
    test_stats = validate(
        model,
        test_triples,
        entity_ids,
        device,
        num_samples=50,
        hits_ks=(1, 3, 10),
    )
    test_stats_rounded = {k: round(v, 4) for k, v in test_stats.items()}

    # Persist to separate JSON for clarity -------------------------------
    with open(TEST_METRICS_PATH, "w") as f:
        json.dump(test_stats_rounded, f, indent=2)

    # Pretty console printout --------------------------------------------
    print("\n=== TEST RESULTS ===")
    for k, v in test_stats_rounded.items():
        print(f"{k:8s}: {v:.4f}")
