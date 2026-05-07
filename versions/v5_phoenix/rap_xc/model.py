"""model.py — RAP-XC policy architecture.

Retrieval-Augmented Policy with Crisis-Conditioned Cross-Attention.
~4.3M params, fits in 12GB VRAM with batch=256, bf16.

Architecture per pass-7 subagent design:

    state_feats (64)         crisis_embeds (k=8, 1024)         dag_feats (80)
        │                            │                              │
        ▼                            ▼                              ▼
    Linear(64→256)              Linear(1024→256)             Linear(80→256)
    + GELU + Linear(256→256)        │                       + GELU + Linear(256→256)
        │                            │                              │
        │  query token               │  k=8 keys/values             │
        └─────────────► MHA cross-attn (4 layers, 4 heads, d=256) ◄─┘
                                     │
                                     ▼
                       fusion: concat(state, xattn, dag) (768)
                       → Linear(768→512) + GELU → Linear(512→256)
                                     │
                            ┌────────┴────────┐
                            ▼                 ▼
                     action_head           value_head
                     Linear(256→280)       Linear(256→1)
                     + judge_prior_bias
                       (frozen, additive)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RAPXCConfig:
    state_dim: int = 64
    crisis_embed_dim: int = 1024
    dag_dim: int = 80
    n_actions: int = 280
    d_model: int = 256
    n_heads: int = 4
    n_xattn_layers: int = 4
    fusion_hidden: int = 512
    dropout: float = 0.1
    retrieved_k: int = 8
    judge_prior_strength: float = 1.0
    use_value_head: bool = True
    target_modules_to_freeze: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------
# Submodules
# ---------------------------------------------------------------------

class _StateEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _CrisisProjector(nn.Module):
    """Project FAISS-retrieved crisis embeddings (k × 1024) -> (k × d_model)."""
    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, k, in_dim) -> (B, k, d_model)
        return self.proj(x)


class _DAGEncoder(nn.Module):
    """Encode cascade-distance + node-status features."""
    def __init__(self, in_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _CrossAttnBlock(nn.Module):
    """One layer: query attends to retrieved crisis keys/values + FFN residual."""
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q: (B, 1, d), kv: (B, k, d)
        attn_out, _ = self.attn(self.norm1(q), kv, kv)
        q = q + attn_out
        q = q + self.ffn(self.norm2(q))
        return q


# ---------------------------------------------------------------------
# Main policy
# ---------------------------------------------------------------------

class RAPXCPolicy(nn.Module):
    """Retrieval-Augmented Policy with Crisis-Conditioned Cross-Attention.

    Forward inputs (all batched, leading dim = batch_size):
      state_feats:    (B, state_dim=64)         engineered numeric state vector
      crisis_embeds:  (B, k=8, embed_dim=1024)  FAISS-retrieved EMDAT events
      dag_feats:      (B, dag_dim=80)            cascade-distance + node-status
      judge_prior:    (B, n_actions=280) | None  optional pre-distilled judge bias
                                                 (additive on logits)
      action_mask:    (B, n_actions=280) | None  invalid-action mask (-inf)

    Returns:
      logits: (B, n_actions)  — raw, post-mask, post-judge-bias
      value:  (B,)             — scalar state value (V-head)
    """

    def __init__(self, cfg: RAPXCConfig | None = None):
        super().__init__()
        self.cfg = cfg or RAPXCConfig()
        d = self.cfg.d_model

        self.state_enc = _StateEncoder(self.cfg.state_dim, d, self.cfg.dropout)
        self.crisis_proj = _CrisisProjector(self.cfg.crisis_embed_dim, d)
        self.dag_enc = _DAGEncoder(self.cfg.dag_dim, d, self.cfg.dropout)
        self.xattn_layers = nn.ModuleList([
            _CrossAttnBlock(d, self.cfg.n_heads, self.cfg.dropout)
            for _ in range(self.cfg.n_xattn_layers)
        ])
        self.fusion = nn.Sequential(
            nn.Linear(d * 3, self.cfg.fusion_hidden),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.fusion_hidden, d),
        )
        self.action_head = nn.Linear(d, self.cfg.n_actions)
        self.value_head = nn.Linear(d, 1) if self.cfg.use_value_head else None

    def forward(
        self,
        state_feats: torch.Tensor,
        crisis_embeds: torch.Tensor,
        dag_feats: torch.Tensor,
        judge_prior: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Encode
        s = self.state_enc(state_feats)               # (B, d)
        c = self.crisis_proj(crisis_embeds)           # (B, k, d)
        g = self.dag_enc(dag_feats)                    # (B, d)

        # Cross-attention: state token queries the k crisis keys/values
        q = s.unsqueeze(1)                             # (B, 1, d)
        for layer in self.xattn_layers:
            q = layer(q, c)
        x = q.squeeze(1)                               # (B, d)

        # Fuse and head
        fused = self.fusion(torch.cat([s, x, g], dim=-1))   # (B, d)
        logits = self.action_head(fused)                     # (B, n_actions)

        # Add (frozen) judge prior bias if provided
        if judge_prior is not None:
            logits = logits + self.cfg.judge_prior_strength * judge_prior

        # Mask invalid actions
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))

        if self.value_head is not None:
            value = self.value_head(fused).squeeze(-1)       # (B,)
        else:
            value = torch.zeros(logits.size(0), device=logits.device)

        return logits, value

    @torch.no_grad()
    def select_action(
        self,
        state_feats: torch.Tensor,
        crisis_embeds: torch.Tensor,
        dag_feats: torch.Tensor,
        judge_prior: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        logits, _ = self.forward(state_feats, crisis_embeds, dag_feats,
                                  judge_prior, action_mask)
        if temperature == 0.0:
            return logits.argmax(dim=-1)
        return torch.distributions.Categorical(logits=logits / temperature).sample()

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def smoke_test() -> dict:
    """Forward pass on a fake batch — verify shapes + parameter count."""
    cfg = RAPXCConfig()
    model = RAPXCPolicy(cfg)
    B = 8
    state = torch.randn(B, cfg.state_dim)
    crisis = torch.randn(B, cfg.retrieved_k, cfg.crisis_embed_dim)
    dag = torch.randn(B, cfg.dag_dim)
    mask = torch.ones(B, cfg.n_actions, dtype=torch.bool)
    mask[:, :10] = False  # arbitrary illegal actions
    logits, value = model(state, crisis, dag, action_mask=mask)
    return {
        "n_parameters": model.n_parameters(),
        "logits_shape": tuple(logits.shape),
        "value_shape": tuple(value.shape),
        "logits_min": float(logits.min()),
        "logits_max": float(logits.max()),
        "logits_mask_inf_count": int((logits == float("-inf")).sum()),
        "expected_inf_per_batch": 10,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(smoke_test(), indent=2))
