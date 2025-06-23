# ltn_core.py

import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import List, Dict, Tuple
from dataclasses import dataclass
from typing import List
import json, csv, math, statistics, random, re
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
    num_vars: int
    head: Atom
    body: List[Atom]

    # one learnable parameter per rule, initialised later
    log_lam: torch.nn.Parameter = None       # to be filled in after construction

    @property
    def lam(self) -> torch.Tensor:
        """Positive rule weight λ = exp(log_lam)."""
        return self.log_lam.exp()
    
    

# ---------- parsing / loading ----------
import re

def _split_fields(s: str) -> list[str]:
    """Return AMIE fields separated by ≥2 whitespace (or tabs)."""
    return re.split(r'\s{2,}|\t', s.strip())

def _fields_to_triples(fields: list[str]) -> list[tuple[str, str, str]]:
    if len(fields) % 3:
        raise ValueError(f"Bad AMIE rule – #fields ≠ 3 × n: {fields}")
    return [
        (fields[i], fields[i + 1], fields[i + 2])
        for i in range(0, len(fields), 3)
    ]


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

        r = Rule(len(var2idx), head, body_atoms)              
        r.log_lam = torch.nn.Parameter(torch.log(torch.tensor(lam, dtype=torch.float32))) # attach weight
        rules.append(r)

    # --- 3‑B  AMIE CSV ------------------------------------------------------
    amie_path = Path(data_dir) / "AMIE-rules.csv"
    
    with open(amie_path) as f:
        rdr = csv.reader(f)
        hdr = next(rdr)
        pca_idx = hdr.index("PCA Confidence")
    
        for row in rdr:
            rule_txt   = row[0].strip()
            confidence = float(row[pca_idx])
            lam = -math.log(1 - confidence + 1e-6)
    
            body_txt, head_txt = (x.strip() for x in rule_txt.split("=>"))
    
            # ---- HEAD ------------------------------------------------------
            head_fields = _split_fields(head_txt)
            head_s, head_r, head_o = _fields_to_triples(head_fields)[0]
    
            var2idx = {}
            kind_s, idx_s, next_ent = _tok_to_kind_id(head_s, var2idx, ent2id, next_ent)
            kind_o, idx_o, next_ent = _tok_to_kind_id(head_o, var2idx, ent2id, next_ent)
    
            if head_r not in rel2id:
                rel2id[head_r], next_rel = next_rel, next_rel + 1
            head = Atom(kind_s, idx_s, rel2id[head_r], kind_o, idx_o)
    
            # ---- BODY ------------------------------------------------------
            body_fields = _split_fields(body_txt)
            body_atoms  = []
            for s_tok, r_tok, o_tok in _fields_to_triples(body_fields):
                if r_tok not in rel2id:
                    rel2id[r_tok], next_rel = next_rel, next_rel + 1
                kind_s, idx_s, next_ent = _tok_to_kind_id(s_tok, var2idx, ent2id, next_ent)
                kind_o, idx_o, next_ent = _tok_to_kind_id(o_tok, var2idx, ent2id, next_ent)
                body_atoms.append(Atom(kind_s, idx_s, rel2id[r_tok], kind_o, idx_o))
    
            rule = Rule(len(var2idx), head, body_atoms)
            rule.log_lam = torch.nn.Parameter(torch.log(torch.tensor(lam, dtype=torch.float32)))
            rules.append(rule)
        

    med_amie = statistics.median([r.lam for r in rules if len(r.body) > 1])
    for r in rules:
        if r.num_vars == 1:
            r.log_lam.data = torch.log((2 * med_amie).clone())
            r.log_lam.requires_grad_(False)

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
                    H = (
                        Sblk.unsqueeze(0).expand(B, -1)
                        if Sblk is not None else sub_fixed
                    )
                
                    for Oblk in obj_range:
                        T = (
                            Oblk.unsqueeze(0).expand(B, -1)
                            if Oblk is not None else obj_fixed
                        )
                
                        # --- broadcast & scoring stays exactly as you had ---
                        nH, nT = H.size(0), T.size(0)
                        if nH != nT:
                            if nH == B and nT > B:
                                H = H.repeat_interleave(nT // B)
                            elif nT == B and nH > B:
                                T = T.repeat_interleave(nH // B)
                            else:
                                raise RuntimeError(f"Unexpected shapes H={nH},T={nT}")
                
                        Rr = torch.full_like(H, atom.r_id)
                        logits = model.score(H, Rr, T)
                        if H.size(0) > B:
                            logits = logits.view(B, -1).max(1).values
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
    
    


#  RULE-AWARE METRICS  (mean satisfaction & weighted rule coverage)
# ----------------------------------------------------------------------
def rule_metrics(model,
                 triples: torch.Tensor,     # (N,3)  validation positives
                 rules:   List[Rule],
                 entity_ids: torch.Tensor,
                 device: torch.device):
    """
    Returns a pair:
        mean_sat   : average sigmoid(score) over *triples*
        wr_cov     : Σ_g λ_g μ̄_g / Σ_g λ_g   (λ truncated to match μ̄ len)

    If evaluate_rules() returns fewer rows than `rules` (because some rules
    cannot be grounded) we truncate λ to the common prefix to avoid a size
    mismatch.
    """
    model.eval()
    with torch.no_grad():
        # -- mean satisfaction of plain KG triples -----------------------
        h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
        mean_sat = torch.sigmoid(model.score(h, r, t)).mean().item()

        # -- rule coverage ----------------------------------------------
        μ = evaluate_rules(model, triples, rules, entity_ids)  # (R′, N)
        μ_bar = μ.mean(dim=1)                                 # (R′,)

        λ_all = torch.stack([
            torch.exp(r.log_lam) if r.log_lam is not None else
            torch.tensor(r.lam, device=device)
            for r in rules
        ])

        R_eff = min(len(λ_all), len(μ_bar))       # common prefix length
        if R_eff == 0:
            return mean_sat, 0.0                  # no evaluable rules

        λ = λ_all[:R_eff]
        μ_bar = μ_bar[:R_eff]

        wr_cov = (λ * μ_bar).sum() / λ.sum()

    model.train()
    return mean_sat, wr_cov.item()

    
    
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
    def register_numeric_relations(
        self,
        rel2id: dict,
        le_name: str = "LE",
        ge_name: str = "GE",
    ) -> None:
        """
        Guarantee that the two numeric-comparison relations exist *and*
        have embedding rows.  Afterwards:
    
            self.rel_le, self.rel_ge     # valid indices into self.rel.weight
        """
        def _append_row() -> None:
            extra = torch.empty(
                1,
                self.dim * self.dim,
                device=self.rel.weight.device,
                dtype=self.rel.weight.dtype,
            )
            nn.init.xavier_uniform_(extra.view(-1, self.dim, self.dim))
            self.rel.weight = nn.Parameter(torch.cat([self.rel.weight, extra], 0))
    
        # --- make sure the names are in rel2id ---------------------------------
        for name in (le_name, ge_name):
            if name not in rel2id:
                rel2id[name] = self.rel.num_embeddings        # next id
                _append_row()                                 # 1-row grow
    
        # --- make sure table is long enough even if ids pre-existed ------------
        for name in (le_name, ge_name):
            rid = rel2id[name]
            while rid >= self.rel.weight.size(0):             # still missing row
                _append_row()
    
        # --- store ids & final assertion ---------------------------------------
        self.rel_le, self.rel_ge = rel2id[le_name], rel2id[ge_name]
        assert self.rel_le < self.rel.weight.size(0) and \
               self.rel_ge < self.rel.weight.size(0), \
               "Numeric comparator relations are missing embeddings."


        
        
   
    def score(self, h, r, t):
      """
      Vectorised triple scoring.
  
      Args
      ----
      h, r, t : LongTensor of shape (B,)
          Indices of head entity, relation and tail entity.
  
      Returns
      -------
      Tensor of shape (B,) – higher means the triple is considered true.
      """
  
      # ---------- 1) bilinear score for *all* triples ----------
      W      = self.rel(r).view(-1, self.dim, self.dim)      # (B,D,D)
      h_emb  = self.ent(h).unsqueeze(1)                      # (B,1,D)
      t_emb  = self.ent(t).unsqueeze(2)                      # (B,D,1)
      scores = (h_emb @ W @ t_emb).squeeze(-1).squeeze(-1)   # (B,)
  
      # ---------- 2) overwrite numeric-comparison triples ----------
      # make sure the two special IDs exist
      if hasattr(self, "rel_le") and hasattr(self, "rel_ge"):
          le_mask = (r == self.rel_le)
          ge_mask = (r == self.rel_ge)
  
          if le_mask.any():
              h_val = self.literal_value[h[le_mask]]
              t_val = self.literal_value[t[le_mask]]
              # sigmoid for fuzziness
              τ = 50.0
              scores[le_mask] = torch.sigmoid((t_val - h_val) / τ)
  
          if ge_mask.any():
              h_val = self.literal_value[h[ge_mask]]
              t_val = self.literal_value[t[ge_mask]]
              τ = 50.0
              scores[ge_mask] = torch.sigmoid((h_val - t_val) / τ)

      return scores
    

# ---------- ltn loss -----------------------------------------------------
def ltn_loss(triples, labels, model, rules, entity_ids,
             rule_sample_frac: float = 0.3,
             kg_mask: torch.Tensor | None = None,
             λ_kg: float = 100.0):
    """
    Return
        bce, regularization, kg_penalty
    (three tensors, each carries .grad_fn so they stay in the graph)
    """
    h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]

    # --- 1) supervised BCE ----------------------------------------------
    logits = model.score(h, r, t)
    bce = F.binary_cross_entropy_with_logits(logits, labels)

    # --- 2) KG hard constraint  (not added to bce – handled in train loop)
    if kg_mask is not None and kg_mask.any():
        pos_logits = torch.sigmoid(logits[kg_mask])
        kg_pen = ((1.0 - pos_logits) ** 2).mean()
    else:
        kg_pen = torch.tensor(0.0, device=bce.device)

    # --- 3) fuzzy-rule regulariser  (no α scaling here) ------------------
    k = max(1, int(rule_sample_frac * len(rules)))
    sampled = random.sample(rules, k)
    μ = evaluate_rules(model, triples, sampled, entity_ids)      # (k,B)
    μ = μ.mean(dim=1)                                           # universal ≈ mean

    reg_terms = []
    for r_obj, mu_val in zip(sampled, μ):
        lam = torch.exp(r_obj.log_lam) if r_obj.log_lam is not None else r_obj.lam
        reg_terms.append(lam * (1.0 - mu_val) ** 2)
    regularization = torch.stack(reg_terms).mean()

    return bce, regularization, kg_pen



#------------------------SAMPLES--------------------------

def sample_rule_triples(rules, entity_ids, device, k_pos=4, k_neg=4):
    """
    For a random subset of rules:
        • choose concrete entity IDs for every variable
        • emit the HEAD triple as a *positive*
        • flip the tail entity to a random different one → *negative*
    Returns two tensors: rule_pos, rule_neg  (may be empty)
    """
    if len(rules) == 0:
        return None, None

    chosen = random.sample(rules, min(len(rules), k_pos))
    pos_rows, neg_rows = [], []
    E = entity_ids

    for rule in chosen:
        # 1) pick random entities for each variable
        var_bind = {i: int(E[torch.randint(0, len(E), (1,))]) for i in range(rule.num_vars)}
        def _get_id(kind, idx):
            return var_bind[idx] if kind == "var" else idx

        # 2) build HEAD triple
        h = _get_id(rule.head.s_kind, rule.head.s_id)
        r = rule.head.r_id
        t = _get_id(rule.head.o_kind, rule.head.o_id)
        pos_rows.append((h, r, t))

        # 3) negative: corrupt tail (ensure different)
        t_neg = t
        while t_neg == t:
            t_neg = int(E[torch.randint(0, len(E), (1,))])
        neg_rows.append((h, r, t_neg))

    if not pos_rows:
        return None, None
    return (torch.tensor(pos_rows, device=device, dtype=torch.long),
            torch.tensor(neg_rows, device=device, dtype=torch.long))


def sample_batch(batch_size: int,
                 KG: torch.Tensor,
                 PAT: torch.Tensor,
                 entity_ids: torch.Tensor,
                 device: torch.device,
                 rules: List[Rule] | None = None,   # <- default first!
                 neg_per_pos: int = 1):
    """
    Returns
        triples : (B,3) LongTensor
        labels  : (B,)  FloatTensor  (1 = positive, 0 = negative)
        kg_mask : (B,)  BoolTensor   True where row is a KG positive
    """

    # -------- 1. how many positives / negatives ------------------------
    n_pos = batch_size // (1 + neg_per_pos)      # e.g. 64 for B=128,k=1
    n_neg = batch_size - n_pos

    # split positives: 62.5 % KG, 37.5 % PAT
    n_kg  = int(round(n_pos * 0.625))
    n_pat = n_pos - n_kg

    # -------- 2. sample KG / PAT positives -----------------------------
    idx_kg  = torch.randint(0, KG.size(0),  (n_kg,),  device=device)
    idx_pat = torch.randint(0, PAT.size(0), (n_pat,), device=device)
    pos = torch.cat([KG[idx_kg], PAT[idx_pat]], dim=0)        # (n_pos,3)

    # -------- 3. sample random negatives -------------------------------
    repeats = math.ceil(n_neg / n_pos)
    neg = pos.repeat_interleave(repeats, dim=0)[:n_neg].clone()
    neg[:, 2] = entity_ids[torch.randint(0, len(entity_ids),
                                         (n_neg,), device=device)]

    # -------- 4. add rule-derived examples (optional) ------------------
    if rules is not None:
        rule_pos, rule_neg = sample_rule_triples(rules, entity_ids, device)
        if rule_pos is not None:                     # at least one grounded
            pos = torch.cat([pos, rule_pos], dim=0)
            neg = torch.cat([neg, rule_neg], dim=0)
            n_pos += rule_pos.size(0)
            n_neg += rule_neg.size(0)

    # -------- 5. concat & label ----------------------------------------
    triples = torch.cat([pos, neg], dim=0)                      # (B,3)
    labels  = torch.cat([torch.ones(n_pos, device=device),
                         torch.zeros(n_neg, device=device)])

    # -------- 6. KG mask before shuffle -------------------------------
    kg_mask = torch.cat([
        torch.ones(n_kg, dtype=torch.bool, device=device),
        torch.zeros(triples.size(0) - n_kg, dtype=torch.bool, device=device)
    ])

    # -------- 7. random shuffle ----------------------------------------
    perm = torch.randperm(triples.size(0), device=device)
    return triples[perm], labels[perm], kg_mask[perm]