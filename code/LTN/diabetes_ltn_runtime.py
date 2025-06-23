# ────────────────────────────────────────────────────────────────────
#  diabetes_ltn_runtime.py
# ────────────────────────────────────────────────────────────────────
import json
import warnings
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

import torch
import torch.nn as nn

from ltn_core import load_rules, load_triples, BilinearLTN, evaluate_rules, sample_batch, RuleBank, Rule, Atom

# -------------------------------------------------------------------
#  DEVICE SELECTION + WARNING
# -------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu" and torch.cuda.is_available():
    warnings.warn(
        "CUDA is available but the Diabetes-LTN runtime is on CPU.\n"
        "Set CUDA_VISIBLE_DEVICES or pass device='cuda' for faster inference."
    )
    


class DiabetesLTNRuntime:
    """
    Loads the trained Bilinear-LTN plus its fuzzy-rule bank and exposes:

        • triple_confidence(h, r, t) -> float
        • explain(h, r, t, top_k=3)  -> list[dict]
    """

    # ────────────────────────────────────────────────────────────────
    def __init__(self, checkpoint_dir: Path):
        # 1) entity / relation maps
        self.ent2id = json.load(open(checkpoint_dir / "entities_after.json"))
        self.rel2id = json.load(open(checkpoint_dir / "relations_after.json"))

        # 2) checkpoint
        ckpt_path = checkpoint_dir / "ltn_final.pt"
        #print("Runtime loading checkpoint:", ckpt_path.resolve())
        ckpt = torch.load(ckpt_path, map_location=DEVICE)

        model_state       = ckpt["model"]
        rule_bank_state   = ckpt["rule_bank"]        
        dim               = ckpt.get("hyper", {}).get("dim", 256)

        # 3) model skeleton
        self.model = BilinearLTN(
            n_ent=len(self.ent2id),
            n_rel=len(self.rel2id),
            dim=dim,
            literal_value=ckpt.get("literal_value", {}),
        )
        self.model.register_numeric_relations(self.rel2id)
        self.model.load_state_dict(model_state, strict=False)
        self.model.to(DEVICE).eval()

        # 4) rules
        def _reconstruct_rules(raw: list[dict]) -> list[Rule]:
            rules = []
            for r in raw:
                head = Atom(**r["head"])
                body = [Atom(**b) for b in r["body"]]
                rule = Rule(r["num_vars"], head, body)
                rule.log_lam = torch.nn.Parameter(
                    torch.log(torch.tensor(r["lambda"], dtype=torch.float32))
                )
                rule.log_lam.requires_grad_(False)
                rules.append(rule)
            return rules

        self.rules = _reconstruct_rules(ckpt["rules_serialised"])
        self.rule_bank = RuleBank(self.rules)
        self.rule_bank.load_state_dict(rule_bank_state, strict=False)
        for p in self.rule_bank.parameters():
            p.requires_grad_(False)

        self.rules_by_rel = defaultdict(list)
        for rule in self.rules:
            self.rules_by_rel[rule.head.r_id].append(rule)

        self.entity_ids = torch.arange(len(self.ent2id), device=DEVICE)

    # ────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def triple_confidence(self, h: str, r: str, t: str) -> float | None:
        h_id = self.ent2id.get(h)
        r_id = self.rel2id.get(r)
        t_id = self.ent2id.get(t)
        if None in (h_id, r_id, t_id):
            return None

        logit = self.model.score(
            torch.tensor([h_id], device=DEVICE),
            torch.tensor([r_id], device=DEVICE),
            torch.tensor([t_id], device=DEVICE),
        )
        return torch.sigmoid(logit).item()


    # ----------------------------------------------------------------
    @torch.no_grad()
    def explain(self, h: str, r: str, t: str, top_k: int = 3):
        h_id = self.ent2id.get(h)          
        r_id = self.rel2id.get(r)          
        t_id = self.ent2id.get(t) 
        if None in (h_id, r_id, t_id):
            return []

        triples = torch.tensor([(h_id, r_id, t_id)], device=DEVICE)
        subset = self.rules_by_rel.get(r_id, [])
        if not subset:
            return []

        μ = evaluate_rules(self.model, triples, subset, self.entity_ids)  # (|subset|,1)

        scored = sorted(
            [(rule.lam.item(), μ[i, 0].item(), rule) for i, rule in enumerate(subset)],
            key=lambda x: x[0] * x[1],
            reverse=True,
        )[:top_k]

        return [
            {"rule": rule, "lambda": lam, "mu": mu, "support": lam * mu}
            for lam, mu, rule in scored
        ]
