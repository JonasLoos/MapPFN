"""Muon optimizer (Newton-Schulz orthogonalized momentum) with an Adam fallback.

Muon is applied only to 2-D weight matrices; all other parameters (norms,
biases, embeddings, scalars / 1-D and >2-D tensors) use AdamW. This matches the
standard Muon recipe and sidesteps the optax-muon/equinox masking incompatibility
(`optax.contrib.muon` chokes on equinox's ``None``-filtered param tree), because
we route per-leaf by ndim inside a single ``jax.tree.map`` that naturally skips
``None`` leaves.

Reference: Jordan et al., "Muon" (Newton-Schulz quintic iteration).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax


def newton_schulz(g: jax.Array, steps: int = 5) -> jax.Array:
    """Quintic Newton-Schulz iteration to (approximately) orthogonalize a matrix."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(jnp.float32)
    x = x / (jnp.linalg.norm(x) + 1e-7)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        aa = x @ x.T
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x
    if transposed:
        x = x.T
    return x.astype(g.dtype)


class MuonState(NamedTuple):
    count: jax.Array
    mu: Any  # muon momentum (2-D) / adam first moment (other)
    nu: Any  # adam second moment (other); unused for 2-D


def muon_adamw(
    learning_rate,
    muon_momentum: float = 0.95,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    eps: float = 1e-8,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> optax.GradientTransformation:
    """Muon for 2-D params, AdamW for the rest. Weight decay handled in the outer chain.

    ``learning_rate`` is the base LR; the Muon branch additionally scales each
    update by ``sqrt(max(1, rows/cols))`` so its RMS matches across shapes.
    """
    lr_fn = learning_rate if callable(learning_rate) else (lambda _: learning_rate)

    def init(params):
        mu = jax.tree.map(jnp.zeros_like, params)
        nu = jax.tree.map(jnp.zeros_like, params)
        return MuonState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update(grads, state, params=None):  # noqa: ARG001
        count = state.count + 1
        lr = lr_fn(state.count)
        b1c = 1.0 - adam_b1**count
        b2c = 1.0 - adam_b2**count

        def per_leaf(g, m, v):
            if g.ndim == 2:
                m_new = muon_momentum * m + g
                buf = g + muon_momentum * m_new if nesterov else m_new
                o = newton_schulz(buf, ns_steps)
                scale = jnp.sqrt(jnp.maximum(1.0, g.shape[0] / g.shape[1]))
                upd = scale * o
                v_new = v
            else:
                m_new = adam_b1 * m + (1.0 - adam_b1) * g
                v_new = adam_b2 * v + (1.0 - adam_b2) * (g * g)
                upd = (m_new / b1c) / (jnp.sqrt(v_new / b2c) + eps)
            return -lr * upd, m_new, v_new

        out = jax.tree.map(per_leaf, grads, state.mu, state.nu)
        is_tup = lambda x: isinstance(x, tuple)  # noqa: E731
        updates = jax.tree.map(lambda x: x[0], out, is_leaf=is_tup)
        mu_new = jax.tree.map(lambda x: x[1], out, is_leaf=is_tup)
        nu_new = jax.tree.map(lambda x: x[2], out, is_leaf=is_tup)
        return updates, MuonState(count=count, mu=mu_new, nu=nu_new)

    return optax.GradientTransformation(init, update)
