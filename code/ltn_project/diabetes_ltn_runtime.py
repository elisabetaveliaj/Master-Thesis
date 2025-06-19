# ────────────────────────────────────────────────────────────────────
#  diabetes_ltn_runtime.py
# ────────────────────────────────────────────────────────────────────
import torch, json
from pathlib import Path
from ltn_core import BilinearLTN, load_rules   # ← from training code
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class DiabetesLTNRuntime:
    """
    Thin wrapper that loads the trained BilinearLTN + rule bank and
    offers an easy `score(head, rel, tail)` + `explain()` interface.
    """

    def __init__(self, checkpoint_dir: Path):
        # ----- 1. load entity / relation maps exactly as in training -----
        self.ent2id = json.load(open(checkpoint_dir / "entities.json"))
        self.rel2id = json.load(open(checkpoint_dir / "relations.json"))

        # ----- 2. rebuild model & rule objects --------------------------
        self.model  = BilinearLTN(len(self.ent2id), len(self.rel2id))
        self.model.load_state_dict(torch.load(checkpoint_dir / "ltn_final.pt",
                                              map_location=DEVICE))
        self.model.to(DEVICE).eval()

        # rules + learned log‑λ parameters
        rules, *_ = load_rules(checkpoint_dir, self.ent2id, self.rel2id,
                               next_ent=max(self.ent2id.values()) + 1,
                               next_rel=max(self.rel2id.values()) + 1)
        for r, log_lam in zip(rules, self.model.rule_bank.parameters()):
            r.log_lam = log_lam        # restore λ
        self.rules = rules

        # cache tensors reused for each query
        self.entity_ids = torch.arange(len(self.ent2id), device=DEVICE)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def triple_confidence(self, h:str, r:str, t:str) -> float:
        h_id = self.ent2id.get(h); r_id = self.rel2id.get(r); t_id = self.ent2id.get(t)
        if None in (h_id, r_id, t_id):
            return 0.0
        logit = self.model.score(torch.tensor([h_id], device=DEVICE),
                                 torch.tensor([r_id], device=DEVICE),
                                 torch.tensor([t_id], device=DEVICE))
        return torch.sigmoid(logit).item()

    def explain(self, h:str, r:str, t:str, top_k:int=3):
        """
        Compute μ_rule for all rules whose head matches (h,r,t) constants
        and return the K most supportive ones with their λ·μ values.
        """
        from ltn_core import evaluate_rules
        target = (self.ent2id[h], self.rel2id[r], self.ent2id[t])
        triples = torch.tensor([target], device=DEVICE)

        μ = evaluate_rules(self.model, triples, self.rules, self.entity_ids)  # (R,1)
        weighted = [(r.lam.item(), μ[i,0].item(), r) for i,r in enumerate(self.rules)
                    if (r.head.r_id == target[1])]      # head‑rel match
        top = sorted(weighted, key=lambda x: x[0]*x[1], reverse=True)[:top_k]
        return [
            {
                "rule": rule, "lambda": lam, "mu": mu,
                "support": lam * mu
            } for lam, mu, rule in top
        ]
