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
    ltn_loss, validate, sample_batch, RuleBank, rule_metrics, Rule
)

import collections
try:
    import wandb                                 
    wandb.init(project="ltn_alpha_sweep", reinit=True)
except ImportError:
    wandb = None


# ─────────────────── Reproducibility ───────────────────
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)

device   = 'cuda' if torch.cuda.is_available() else 'cpu'
#scaler = GradScaler() if device == 'cuda' else None
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
    r.log_lam = torch.nn.Parameter(torch.log(r.lam.clone().to(device)))
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

METRICS_PATH = DATA_DIR / "ltn_metrics.json"
TEST_METRICS_PATH = DATA_DIR / "ltn_test_metrics.json"

# ─────────────────── 6.  Training loop ────────────────

def train(
    alphas=(0.1, 0.5, 1.0, 2.0),
    *,
    epochs: int = 90,
    batch_size: int = 128,
    λ_kg: float = 50.0,                 # strength of KG hard constraint
):
    """
    Train the LTN model for each α in *alphas*.
    Writes incremental snapshots to ``ltn_metrics.json`` so you never
    lose progress on an interrupted run.
    """
    metrics: list[dict] = []
    initial_model_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
    initial_rule_state  = {k: v.clone().detach() for k, v in rule_bank.state_dict().items()}


    for α in alphas:
        print(f"\n=== α = {α} ===")
            # --- reset model & rules ------------------------------------------
        model.load_state_dict(initial_model_state, strict=True)
        rule_bank.load_state_dict(initial_rule_state, strict=True)

        num_batches = max(1, KG.size(0) // batch_size)
        
        optim = torch.optim.AdamW(
            list(model.parameters()) + list(rule_bank.parameters()),
            lr=1e-3,
            weight_decay=1e-5,
        )
        sched = torch.optim.lr_scheduler.StepLR(
            optim,
            step_size=50,  
            gamma=0.5        
        )

        for epoch in range(epochs):
            epoch_bce, epoch_reg, epoch_kg = [], [], []
            tot = 0.0

            for batch_idx in range(num_batches):
                triples, labels, kg_mask = sample_batch(
                    batch_size, KG, PAT, entity_ids, device, neg_per_pos=1, rules=rules
                )

                optim.zero_grad()

                # ~~~ forward pass – three raw loss parts ~~~
                bce, reg, kg_pen = ltn_loss(
                    triples,
                    labels,
                    model,
                    rules,
                    entity_ids,
                    kg_mask=kg_mask,
                    λ_kg=λ_kg,           
                )

                loss = bce + λ_kg * kg_pen + α * reg

                if scaler is not None:
                    with autocast(device_type="cuda", dtype=torch.float16):
                        scaler.scale(loss).backward()
                        scaler.unscale_(optim)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optim)
                        scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
                    


                # -------- bookkeeping per batch ----------
                epoch_bce.append(bce.item())
                epoch_reg.append(reg.item())
                epoch_kg.append(kg_pen.item())
                
            tot += loss.item()
            
        
            # ---- end epoch -------------------------------------------
            sched.step()
            mean_bce = float(np.mean(epoch_bce))
            mean_reg = float(np.mean(epoch_reg))
            mean_kg  = float(np.mean(epoch_kg))
            rule_ratio = α * mean_reg / (mean_bce + λ_kg * mean_kg + 1e-9)

            print(
                f"Epoch {epoch:03d}  "
                f"loss={loss:.4f}  "
                f"Rule={α*mean_reg:.4f}  "
                f"KG={λ_kg*mean_kg:.4f}  "
                f"(rule/total={rule_ratio:.2%})"
            )

            if wandb is not None:
                wandb.log(
                    {
                        "alpha": α,
                        "epoch": epoch,
                        "loss_bce": mean_bce,
                        "loss_rule": α * mean_reg,
                        "loss_kg": λ_kg * mean_kg,
                        "ratio_rule_to_total": rule_ratio,
                    }
                )

            # --- quick dev-set check every 10 epochs ------------------
            if epoch % 10 == 0:
                stats = validate(
                    model,
                    valid_triples,
                    entity_ids,
                    device,
                    num_samples=50,
                    hits_ks=(1, 3, 10),
                )
                
                mean_sat, wr_cov = rule_metrics(
                    model,
                    torch.tensor(valid_triples, device=device),
                    rules,
                    entity_ids,
                    device,
                )
                stats["MeanSatisfaction"]     = mean_sat
                stats["WeightedRuleCoverage"] = wr_cov
                print("  → VAL:", {k: f"{v:.3f}" for k, v in stats.items()})

                snapshot = {
                    "alpha": α,
                    "epoch": epoch,
                    "loss": round(mean_bce + α * mean_reg + λ_kg * mean_kg, 4),
                    **{k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()},
                }
                metrics.append(snapshot)

                # persist after every validation step
                with open(METRICS_PATH, "w") as f:
                    json.dump(metrics, f, indent=2)
                    
    torch.cuda.empty_cache()

    # ------------- after all α runs: save final state -------------------
    def serialise_rule(rule: rules) -> dict:
      return {
          "num_vars": rule.num_vars,
          "head": vars(rule.head),                    # -> dict with 5 fields
          "body": [vars(a) for a in rule.body],
          "lambda": rule.lam.item(),                 # positive weight
      }

    rules_serialised = [serialise_rule(r) for r in rules]
    
    ckpt = {
        "model": model.state_dict(),
        "rule_bank": rule_bank.state_dict(),          # λ in log space (kept for gradients)
        "rules_serialised": rules_serialised,         #  <<< NEW
        "literal_value": literal_value,
        "hyper": {"dim": model.dim},
    }
    torch.save(ckpt, DATA_DIR / "ltn_final.pt")
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics



# ─────────────────── 7.  Entry point ──────────────────
if __name__ == "__main__":
    combined = np.vstack([train_triples, valid_triples])
    KG = torch.as_tensor(combined, dtype=torch.long).to(device)


    train(alphas=(2.0,))
    with open(DATA_DIR / "entities_after.json", "w") as f:
        json.dump(ent2id, f, indent=2)

    with open(DATA_DIR / "relations_after.json", "w") as f:
        json.dump(rel2id, f, indent=2)

    print("Saved regenerated entities.json and relations.json")


    print("\n=== TEST RESULTS ===")
    test_stats = validate(
        model,
        test_triples,
        entity_ids,
        device,
        num_samples=50,
        hits_ks=(1, 3, 10),
    )
    mean_sat, wr_cov = rule_metrics(
                    model,
                    torch.tensor(test_triples, device=device),
                    rules,
                    entity_ids,
                    device,
                )
    test_stats["MeanSatisfaction"]     = mean_sat
    test_stats["WeightedRuleCoverage"] = wr_cov
    test_stats_rounded = {k: round(v, 4) for k, v in test_stats.items()}

    # Persist to separate JSON for clarity -------------------------------
    with open(TEST_METRICS_PATH, "w") as f:
        json.dump(test_stats_rounded, f, indent=2)

    # Pretty console printout --------------------------------------------
    print("\n=== TEST RESULTS ===")
    for k, v in test_stats_rounded.items():
        print(f"{k:8s}: {v:.4f}")
