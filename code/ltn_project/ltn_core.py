# ltn_core.py

import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import List, Dict, Tuple
from dataclasses import dataclass
import json, csv, math, statistics, random
from pathlib import Path

# ---------- dataclasses ----------
@dataclass
class Atom:
    s_kind: str
    s_id: int
    r_id: int
    o_kind: str
    o_id: int


@dataclass
class Rule:
    lam: float
    num_vars: int
    head: Atom
    body: List[Atom]

# ---------- parsing / loading ----------
def _tok_to_kind_id(tok: str,
                    var2idx: Dict[str, int],
                    ent2id: Dict[str, int],
                    next_ent: int
                    ) -> Tuple[str, int, int]:
    """
    Returns (kind,id,next_ent).
      kind ∈ {'var','const'}
      id   = var‑idx  or  entity‑id
    """
    if tok.startswith('?'):
        if tok not in var2idx:
            var2idx[tok] = len(var2idx)
        return 'var', var2idx[tok], next_ent
    else:
        if tok not in ent2id:
            ent2id[tok] = next_ent
            next_ent += 1
        return 'const', ent2id[tok], next_ent



def load_rules(data_dir: str,
               ent2id: Dict[str, int],
               rel2id: Dict[str, int],
               next_ent: int,
               next_rel: int
               ) -> Tuple[List[Rule], int, int]:
    """Parse empirical + AMIE rules → List[Rule]."""
    rules: List[Rule] = []

    # 3‑A empirical JSON
    for R in json.load(open(Path(data_dir) / "empirical_rules_compiled.json")):
        lam = R['weight']
        var2idx = {}
        # — head —
        rel_s, _, const = R['head'].split()
        if rel_s not in rel2id: rel2id[rel_s], next_rel = next_rel, next_rel + 1
        head_rel = rel2id[rel_s]

        kind_s, idx_s, next_ent = _tok_to_kind_id('?p', var2idx, ent2id, next_ent)
        kind_o, idx_o, next_ent = _tok_to_kind_id(const, var2idx, ent2id, next_ent)
        head = Atom(kind_s, idx_s, head_rel, kind_o, idx_o)

        # — body —
        body_atoms = []
        for atom_s in R['body']:
            rel_s, _, const = atom_s.split()
            if rel_s not in rel2id: rel2id[rel_s], next_rel = next_rel, next_rel + 1
            kind_s, idx_s, next_ent = _tok_to_kind_id('?p', var2idx, ent2id, next_ent)
            kind_o, idx_o, next_ent = _tok_to_kind_id(const, var2idx, ent2id, next_ent)
            body_atoms.append(Atom(kind_s, idx_s, rel2id[rel_s], kind_o, idx_o))

        rules.append(Rule(lam, len(var2idx), head, body_atoms))

    # 3‑B AMIE CSV
    amie_path = Path(data_dir) / "AMIE-rules.csv"
    with open(amie_path) as f:
        rdr, hdr = csv.reader(f), None
        hdr = next(rdr)
        pca_idx = hdr.index('PCA Confidence')
        for row in rdr:
            rule_str = row[0].strip()
            conf = float(row[pca_idx])
            lam = -math.log(1 - conf + 1e-2)

            body_s, head_s = [x.strip() for x in rule_str.split('=>')]
            s_tok, r_tok, o_tok = head_s.split()

            if r_tok not in rel2id: rel2id[r_tok], next_rel = next_rel, next_rel + 1
            head_rel = rel2id[r_tok]

            var2idx = {}
            kind_s, idx_s, next_ent = _tok_to_kind_id(s_tok, var2idx, ent2id, next_ent)
            kind_o, idx_o, next_ent = _tok_to_kind_id(o_tok, var2idx, ent2id, next_ent)
            head = Atom(kind_s, idx_s, head_rel, kind_o, idx_o)

            body_atoms = []
            for atom in body_s.split('  '):  # AMIE uses double‑space sep.
                parts = atom.strip().split()
                if len(parts) != 3: continue
                s_tok, r_tok, o_tok = parts
                if r_tok not in rel2id: rel2id[r_tok], next_rel = next_rel, next_rel + 1
                kind_s, idx_s, next_ent = _tok_to_kind_id(s_tok, var2idx, ent2id, next_ent)
                kind_o, idx_o, next_ent = _tok_to_kind_id(o_tok, var2idx, ent2id, next_ent)
                body_atoms.append(Atom(kind_s, idx_s, rel2id[r_tok], kind_o, idx_o))

            rules.append(Rule(lam, len(var2idx), head, body_atoms))

    # OPTIONAL: rescale empirical rules (keep your original heuristic)
    med_amie = statistics.median([r.lam for r in rules if len(r.body) > 1])
    for r in rules:
        if r.num_vars == 1:  # empirical single‑var rules
            r.lam = 2 * med_amie

    return rules, next_ent, next_rel

def load_triples(path, ent2id, rel2id):
    """
    Load a JSON array of either
      • dicts with keys 'h','r','t'
      • lists/tuples [h,r,t]
    and return a list of (h_id, r_id, t_id) tuples.
    """
    raw = json.load(open(path))
    out = []
    for item in raw:
        if isinstance(item, dict):
            h, r, t = item['h'], item['r'], item['t']
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            h, r, t = item
        else:
            raise ValueError(f"Unrecognized triple format: {item!r}")
        out.append((ent2id[h], rel2id[r], ent2id[t]))
    return out

# ---------- model ----------
class BilinearLTN(nn.Module):
    def __init__(self, n_ent, n_rel, dim=256, *, literal_value=None):
        super().__init__()
        self.dim = dim
        self.ent = nn.Embedding(n_ent, dim)
        self.rel = nn.Embedding(n_rel, dim * dim)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight.view(-1, dim, dim))

        # ---- NEW: literals ------------------------------------------------
        self.literal_value = literal_value or {}      # id → float

        # Relation ids for the two hard‑coded comparators
        self.rel_le, self.rel_ge = None, None   # set later by helper

    # helper: call once *after* you know rel2id
    def register_numeric_relations(self, rel2id, le_name="LE", ge_name="GE"):
        self.rel_le = rel2id.setdefault(le_name, max(rel2id.values()) + 1)
        self.rel_ge = rel2id.setdefault(ge_name, max(rel2id.values()) + 1)
        
        
    def bilinear_score(self, h, r, t):
        """
        Pure bilinear score for (h, r, t):
          score_i = e_h[i]ᵀ · W_r[i] · e_t[i]
        """
        # (B, dim)
        e_h = self.ent(h)
        e_t = self.ent(t)
        # (B, dim*dim) → (B, dim, dim)
        W   = self.rel(r).view(-1, self.dim, self.dim)
        # einsum sums over the middle dim: for each batch i, bi·bij·bj → score_i
        return torch.einsum("bi,bij,bj->b", e_h, W, e_t)

    def score(self, h, r, t, *, temp=5.0):
        """Return logits. Two special relations implement ≤ / ≥ on literals."""
        if r.dim() == 0:                    # allow scalar ids
            r = r.view(1)

        if self.rel_le is not None:
            mask_le = (r == self.rel_le)
            mask_ge = (r == self.rel_ge)
        else:
            mask_le = mask_ge = torch.zeros_like(r, dtype=torch.bool)

        if mask_le.any() or mask_ge.any():      # at least one numeric comp.
            logits = torch.empty_like(r, dtype=torch.float)
            # ––– standard bilinear for the rest –––
            std_mask = ~(mask_le | mask_ge)
            if std_mask.any():
                logits[std_mask] = self.bilinear_score(
                    h[std_mask], r[std_mask], t[std_mask])
                    

            # ––– numeric le / ge –––
            for m, sign in ((mask_le, +1), (mask_ge, -1)):   # ≤ is +1, ≥ is −1
                if m.any():
                    hv = torch.tensor([self.literal_value.get(int(i), 0.0)
                                       for i in h[m]], device=h.device)
                    tv = torch.tensor([self.literal_value.get(int(i), 0.0)
                                       for i in t[m]], device=h.device)
                    logits[m] =  temp * (sign * (tv - hv))   # ↑ if condition holds
            return logits

        # fallback: pure bilinear
        return self.bilinear_score(h, r, t)

# ---------- rule evaluation ----------

def evaluate_rules(model, triples, rules, entity_ids, *, chunk=2048):
    """
    Compute μ_rule for each rule (no λ yet).

    Args
    ----
    model        : BilinearLTN with score(h,r,t) → logits
    triples      : (B,3) batch of (h,r,t)   — used to bind head-vars
    rules        : list[Rule]
    entity_ids   : (|E|,) tensor with all entity ids
    chunk        : how many entities to materialise per GPU block

    Returns
    -------
    μ : (len(rules), B) tensor with fuzzy truth-values of every rule
    """
    device = triples.device
    h, _, t = triples.t()
    B = h.size(0)
    E_ids = entity_ids

    outs = []
    for rule in rules:
        # ---------- HEAD ----------
        bound = {}
        def bind(kind, idx, default_vec):
            if kind == 'var':
                bound[idx] = default_vec
                return default_vec
            return torch.full_like(default_vec, idx)

        Hh = bind(rule.head.s_kind, rule.head.s_id, h)
        Rh = torch.full_like(Hh, rule.head.r_id)
        Th = bind(rule.head.o_kind, rule.head.o_id, t)
        μ_head = torch.sigmoid(model.score(Hh, Rh, Th))

        # ---------- BODY ----------
        μ_body = torch.ones_like(μ_head)
        for atom in rule.body:

            def eval_existential(var_on_subject, var_on_object, sub_fixed, obj_fixed):
                best = torch.full((B,), -1e9, device=device)
                subj_range = [None] if not var_on_subject else [E_ids[i:i+chunk] for i in range(0, E_ids.size(0), chunk)]
                obj_range = [None] if not var_on_object else [E_ids[i:i+chunk] for i in range(0, E_ids.size(0), chunk)]

                for Sblk in subj_range:
                    # Build H
                    if Sblk is not None:
                        H = Sblk.unsqueeze(0).expand(B, -1).reshape(-1)
                    else:
                        H = sub_fixed

                    for Oblk in obj_range:
                        # Build T
                        if Oblk is not None:
                            T = Oblk.unsqueeze(0).expand(B, -1).reshape(-1)
                        else:
                            T = obj_fixed

                        # Broadcast to match lengths
                        nH, nT = H.size(0), T.size(0)
                        if nH != nT:
                            if nH == B and nT > B:
                                reps = nT // B
                                H = H.repeat_interleave(reps)
                            elif nT == B and nH > B:
                                reps = nH // B
                                T = T.repeat_interleave(reps)
                            else:
                                raise RuntimeError(f"Unexpected shapes H={nH},T={nT}")

                        Rr = torch.full_like(H, atom.r_id)
                        logits = model.score(H, Rr, T)
                        if H.size(0) > B:
                            logits = logits.view(B, -1).max(dim=1).values
                        best = torch.maximum(best, logits)
                return torch.sigmoid(best)

            # Subject binding
            if atom.s_kind == 'var' and atom.s_id in bound:
                Hs = bound[atom.s_id]
                subj_exist = False
            elif atom.s_kind == 'var':
                Hs = None
                subj_exist = True
            else:
                Hs = torch.full_like(h, atom.s_id)
                subj_exist = False

            # Object binding
            if atom.o_kind == 'var' and atom.o_id in bound:
                To = bound[atom.o_id]
                obj_exist = False
            elif atom.o_kind == 'var':
                To = None
                obj_exist = True
            else:
                To = torch.full_like(t, atom.o_id)
                obj_exist = False

            if subj_exist or obj_exist:
                μ_atom = eval_existential(subj_exist, obj_exist, Hs, To)
            else:
                Rr = torch.full_like(Hs, atom.r_id)
                μ_atom = torch.sigmoid(model.score(Hs, Rr, To))

            μ_body = torch.clamp(μ_body + μ_atom - 1.0, min=0.)

        μ_rule = torch.clamp(1.0 - μ_body + μ_head, max=1.0)
        outs.append(μ_rule)

    return torch.stack(outs)

    


class RuleBank(nn.Module):
    """
    Thin container so torch.optim can see rule.log_lam parameters.
    """
    def __init__(self, rules: List[Rule]):
        super().__init__()
        # register each log_lam as a sub‑parameter
        for i, r in enumerate(rules):
            self.register_parameter(f"log_lam_{i}", r.log_lam)

    def forward(self):
        raise RuntimeError("RuleBank is a parameter holder only.")

def validate(model,
             valid_triples,
             ent_ids,
             device,
             num_samples: int,
             hits_ks):
    """
    Link-prediction evaluation (binary).

    Returns
    -------
    stats : dict with keys
        'AUC', 'Accuracy', 'MRR', 'Hits@k' for every k in hits_ks.
    """
    model.eval()
    with torch.no_grad():

        # -------- positives ----------
        pos = torch.tensor(valid_triples, device=device)        # (N,3)
        N   = pos.size(0)
        pos_logit = model.score(pos[:, 0], pos[:, 1], pos[:, 2])  # (N,)

        # -------- negatives ----------
        K = num_samples
        tails = torch.randint(0, len(ent_ids), (N, K), device=device)
        neg = pos.unsqueeze(1).expand(N, K, 3).clone()
        neg[:, :, 2] = tails
        neg_logit = model.score(neg[:, :, 0].flatten(),
                                neg[:, :, 1].flatten(),
                                neg[:, :, 2].flatten()).view(N, K)        # (N,K)

        # -------- ranks for MRR / Hits@k ----------
        all_scores = torch.cat([pos_logit.unsqueeze(1), neg_logit], 1)    # (N,K+1)
        ranks = (all_scores > pos_logit.unsqueeze(1)).sum(1) + 1          # (N,)

        # -------- flat vectors for AUC / Accuracy ----------
        pos_sig = torch.sigmoid(pos_logit)           # (N,)
        neg_sig = torch.sigmoid(neg_logit.flatten()) # (N·K,)

        y_scores = torch.cat([pos_sig, neg_sig])                         # 1-D
        y_true   = torch.cat([torch.ones_like(pos_sig),
                              torch.zeros_like(neg_sig)])               # 1-D

        auc  = roc_auc_score(y_true.cpu(), y_scores.cpu())
        acc  = accuracy_score(y_true.cpu(), (y_scores > 0.5).cpu())

        stats = {
            'AUC'      : auc,
            'Accuracy' : acc,
            'MRR'      : (1.0 / ranks.float()).mean().item(),
            **{f'Hits@{k}': (ranks <= k).float().mean().item() for k in hits_ks}
        }

    model.train()
    return stats

# ---------- rule evaluation ----------
def ltn_loss(triples, labels, model, rules, entity_ids, rule_sample_frac=0.3):
    """
    Compute the Logical Tensor Network (LTN) loss for a batch of triples,
    combining supervised binary cross-entropy with fuzzy-rule regularization.

    Args:
        triples (torch.LongTensor of shape (B, 3)):
            Batch of triples, each (head_id, relation_id, tail_id).
        labels (torch.FloatTensor of shape (B,)):
            Ground-truth labels: 1.0 for positive triples, 0.0 for negatives.
        model (BilinearLTN):
            The LTN model providing a `score(h, r, t)` method that outputs logits.
        rules (list):
            Collection of fuzzy-logic rule objects to use for regularization.
        entity_ids (torch.LongTensor of shape (num_entities,)):
            All valid entity IDs for grounding rule evaluation.
        rule_sample_frac (float, optional):
            Fraction ∈ (0,1] of rules to randomly sample each batch (default=0.3).

    Returns:
        bce_loss (torch.Tensor):
            Supervised binary cross-entropy loss between `logits` and `labels`.
        reg_loss (torch.Tensor):
            Fuzzy-rule regularization loss computed as the mean squared
            deviation of rule satisfaction from 1.0.

    Behavior:
        1. Unpacks heads, relations, and tails from `triples`.
        2. Computes logits via `model.score(h, r, t)` and BCE loss.
        3. Samples `k = max(1, int(rule_sample_frac * len(rules)))` rules.
        4. Evaluates the sampled rules to obtain weighted satisfaction scores μ (shape: k × B).
        5. Averages μ across batch for each rule, then computes
           regularization as mean((1 − μ_mean)²) across the k rules.
    """
    # unpack triples
    h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]

    # supervised BCE loss
    logits = model.score(h, r, t)
    bce = F.binary_cross_entropy_with_logits(logits, labels)

    # sample and evaluate fuzzy rules
    k = max(1, int(rule_sample_frac * len(rules)))
    sampled = random.sample(rules, k)
    μ = evaluate_rules(model, triples, sampled, entity_ids)  # (k,B)
    mu_mean = μ.mean(dim=1)  # universal ≈ mean
    regularization = torch.stack([
        r.lam * (1.0 - m) ** 2 for r, m in zip(sampled, mu_mean)  # λ outside square
    ]).mean()

    return bce, regularization


# ---------- batching -------------------------------------------------------
def sample_batch(batch_size: int,
                 KG: torch.Tensor,
                 PAT: torch.Tensor,
                 entity_ids: torch.Tensor,
                 device: torch.device,
                 neg_per_pos: int = 1):
    """
    Build a batch of size `batch_size` with a fixed ratio of
        50 % KG-positives, 30 % PAT-positives, 20 % negatives
    while still honouring `neg_per_pos` when possible.
    """

    # -------- how many positives & negatives? -----------------------------
    n_pos = batch_size // (1 + neg_per_pos)           # e.g. 64 for B=128,k=1
    n_neg = batch_size - n_pos                        # fill up the rest

    # Split positives: 50 % KG, 30 % PAT  →  5:3 within the *positives*
    # 5 / 8  = 0.625   and   3 / 8 = 0.375
    n_kg  = int(round(n_pos * 0.625))
    n_pat = n_pos - n_kg                              # whatever is left

    # -------- sample positives --------------------------------------------
    idx_kg  = torch.randint(0, KG.size(0),  (n_kg,),  device=device)
    idx_pat = torch.randint(0, PAT.size(0), (n_pat,), device=device)
    pos     = torch.cat([KG[idx_kg], PAT[idx_pat]], dim=0)      # (n_pos, 3)

    # -------- make negatives ----------------------------------------------
    # repeat-and-trim so we end up with *exactly* n_neg rows
    repeats = math.ceil(n_neg / n_pos)
    neg     = pos.repeat_interleave(repeats, dim=0)[:n_neg].clone()

    rand_tails = entity_ids[torch.randint(0, entity_ids.size(0),
                                          (n_neg,), device=device)]
    neg[:, 2] = rand_tails                           # corrupt the tails

    # -------- concatenate & shuffle ---------------------------------------
    triples = torch.cat([pos, neg], dim=0)           # (batch_size, 3)
    labels  = torch.cat([torch.ones(n_pos, device=device),
                         torch.zeros(n_neg, device=device)])
    perm = torch.randperm(batch_size, device=device)
    return triples[perm], labels[perm]