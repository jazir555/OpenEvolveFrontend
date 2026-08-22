"""Differentiable Architecture Search (DARTS) for neuroevolution.

Implements :class:`DifferentiableArchitectureSearch` from
``docs/architecture/EVOLUTION_ALGORITHM_ENHANCEMENT_SPEC.md`` (section
"Neuroevolution / Differentiable Architecture Search"). It is a *neuroevolution*
component: rather than evolving discrete Genomes (see :mod:`openevolve.neat`)
it continuously relaxes a super-network ("supernet") over candidate operations
per layer and jointly learns, through gradient descent, (a) the architecture
parameters (``alphas``) that weight the candidate operations and (b) the shared
model weights.

Two relaxation methods are supported:

* ``darts`` -- continuous softmax relaxation (Liu, Simonyan & Yang, 2019) with a
  first-order (memory-cheap) bilevel optimisation loop;
* ``gumbel_softmax`` -- stochastic Gumbel-Softmax sampling, which also enables a
  generic, label-free :meth:`search_eval` mode driven by any callable evaluator.

The supernet is a simple stacked MLP: each layer mixes a set of primitive
operations (identity, zero, scalar scale, linear, ReLU, tanh) and the input is
projected through a learnable "stem" layer. Everything is implemented with
plain numpy and a small Adam optimiser -- **no autograd / no torch** -- so the
module stays dependency free and degrades gracefully when no backend is
present. For very large architectures a user may supply an external
``autograd`` backend, but it is never required.

Exposes :func:`differentiable_architecture_search` for one-shot functional use
and :func:`run_darts` as the package-style alias used elsewhere.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Primitive operation set
# --------------------------------------------------------------------------- #
PRIMITIVES = ("identity", "zero", "scale", "linear", "relu", "tanh")


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _xavier(n_in: int, n_out: int) -> float:
    return math.sqrt(2.0 / max(1, n_in + n_out))


def _op_forward(
    name: str, x: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    """Forward pass of one primitive. ``weight`` is a small parameter vector."""
    if name == "identity":
        return x.astype(float)
    if name == "zero":
        return np.zeros_like(x.astype(float))
    if name == "scale":
        s = float(weight[0]) if weight.size else 1.0
        return x.astype(float) * s
    if name == "linear":
        w = np.asarray(weight, dtype=float).reshape(-1, x.shape[0]) if weight.size else np.eye(x.shape[0])
        return w @ x.astype(float)
    if name == "relu":
        return np.maximum(0.0, x.astype(float))
    if name == "tanh":
        return np.tanh(x.astype(float))
    raise ValueError(f"Unknown primitive {name!r}")


def _op_backward(
    name: str, x: np.ndarray, out: np.ndarray, weight: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(d_out_d_x, d_out_d_weight)`` for one primitive.

    ``x`` is shape ``(n_in,)``; ``out`` is the forward output; ``weight`` is the
    operation's parameter vector. The gradient w.r.t. ``x`` is shape ``(n_in,)``
    and w.r.t. ``weight`` matches ``weight.shape``.
    """
    x = x.astype(float)
    if name == "identity":
        return np.ones_like(x), np.zeros_like(weight)
    if name == "zero":
        return np.zeros_like(x), np.zeros_like(weight)
    if name == "scale":
        grad_x = np.full_like(x, float(weight[0]) if weight.size else 1.0)
        grad_w = np.array([float(np.sum(x))]) if weight.size else np.zeros_like(weight)
        return grad_x, grad_w
    if name == "linear":
        # Element-wise affine over the hidden vector: out = x * weight.
        w = np.asarray(weight, dtype=float).reshape(-1) if weight.size else np.ones(x.shape[0])
        grad_x = w
        grad_w = x
        return grad_x, grad_w
    if name == "relu":
        return (x > 0.0).astype(float), np.zeros_like(weight)
    if name == "tanh":
        return (1.0 - out * out), np.zeros_like(weight)
    raise ValueError(f"Unknown primitive {name!r}")


# --------------------------------------------------------------------------- #
# Supernet
# --------------------------------------------------------------------------- #
class SuperNet:
    """A stacked-MLP supernet with a soft mixture of primitive per layer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        output_dim: int,
        operations: Sequence[str] = PRIMITIVES,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.output_dim = output_dim
        self.operations = list(operations)
        self.num_ops = len(self.operations)
        self.rng = rng or random.Random(0)

        # Stem: project input -> hidden. Returns (hidden_dim, input_dim) so
        # that ``stem @ x`` (x is (input_dim,)) yields (hidden_dim,).
        self.stem = self._rand_matrix(hidden_dim, input_dim)

        # Architecture logits: one vector of length num_ops per layer.
        self.alphas = [
            np.zeros(self.num_ops, dtype=float) for _ in range(depth)
        ]
        # Operation weights: per layer, per op, a small parameter vector.
        self.op_weights: List[List[np.ndarray]] = []
        for _ in range(depth):
            layer_weights = []
            for _op_name in self.operations:
                layer_weights.append(self._rand_vector(hidden_dim))
            self.op_weights.append(layer_weights)

        # Output head: hidden -> output. head is (output_dim, hidden_dim) so
        # that ``head @ h`` (h has shape (hidden_dim,)) yields (output_dim,).
        self.head = self._rand_matrix(output_dim, hidden_dim)

    def _rand_matrix(self, n_in: int, n_out: int) -> np.ndarray:
        scale = _xavier(n_in, n_out)
        # Returns shape (n_in, n_out) so that ``W @ x`` (x is (n_in,)) yields (n_out,).
        return np.array(
            [[self.rng.gauss(0.0, scale) for _ in range(n_out)] for _ in range(n_in)]
        )

    def _rand_vector(self, n: int) -> np.ndarray:
        scale = _xavier(n, n)
        return np.array([self.rng.gauss(0.0, scale) for _ in range(n)])

    def architecture_probabilities(self) -> List[np.ndarray]:
        """Per-layer softmax over operations (the learned architecture)."""
        return [_softmax(a) for a in self.alphas]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Continuous (soft) forward pass used during training."""
        x = np.asarray(x, dtype=float)
        h = self.stem @ x
        probs = self.architecture_probabilities()
        for layer in range(self.depth):
            p = probs[layer]
            mixed = np.zeros_like(h)
            for k, op_name in enumerate(self.operations):
                mixed = mixed + p[k] * _op_forward(op_name, h, self.op_weights[layer][k])
            h = mixed
        out = self.head @ h
        return out.reshape(-1) if out.ndim > 0 else out

    def discretize(self) -> List[int]:
        """Choose the argmax operation per layer (the final discrete genotype)."""
        return [int(np.argmax(a)) for a in self.alphas]

    def genotype(self) -> Dict[str, Any]:
        return {
            "operations": list(self.operations),
            "chosen": self.discretize(),
            "probabilities": [p.tolist() for p in self.architecture_probabilities()],
            "depth": self.depth,
            "hidden_dim": self.hidden_dim,
        }

    def build_from_genotype(self, chosen: Sequence[int]) -> Callable[[np.ndarray], np.ndarray]:
        """Return a concrete (hard) predictor using the chosen operations."""
        chosen = list(chosen)
        probs = self.architecture_probabilities()

        def predict(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            curr = self.stem @ x
            for layer in range(self.depth):
                op_name = self.operations[chosen[layer]]
                p = probs[layer]
                curr = p[chosen[layer]] * _op_forward(
                    op_name, curr, self.op_weights[layer][chosen[layer]]
                )
            out = self.head @ curr
            return out.reshape(-1) if out.ndim > 0 else out

        return predict


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class DARTSConfig:
    """Configuration for :class:`DifferentiableArchitectureSearch`."""

    input_dim: int = 1
    hidden_dim: int = 16
    depth: int = 3
    output_dim: int = 1
    operations: Sequence[str] = PRIMITIVES
    relaxation_method: str = "darts"  # "darts" or "gumbel_softmax"
    search_epochs: int = 50
    model_lr: float = 0.05
    arch_lr: float = 0.05
    temperature: float = 1.0
    unrolled: bool = False  # first-order (False) by default -- memory cheap
    batch_size: int = 32
    # When True, search_eval() uses a callable evaluator instead of supervised splits.
    use_eval_fn: bool = False
    random_state: Optional[int] = None

    def resolve(self) -> "DARTSConfig":
        ops = list(self.operations)
        if not ops:
            ops = list(PRIMITIVES)
        object.__setattr__(self, "operations", ops)
        return self


# --------------------------------------------------------------------------- #
# Adam optimiser (dependency free)
# --------------------------------------------------------------------------- #
class _Adam:
    def __init__(self, params: Sequence[np.ndarray], lr: float = 0.05,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        self.t = 0

    def step(self, grads: Sequence[np.ndarray]) -> None:
        self.t += 1
        for i, g in enumerate(grads):
            m, v = self.state[i]
            g = np.asarray(g, dtype=float)
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * (g * g)
            mhat = m / (1 - self.beta1 ** self.t)
            vhat = v / (1 - self.beta2 ** self.t)
            self.state[i] = (m, v)
            self.params_ref[i] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def bind(self, params: Sequence[np.ndarray]) -> None:
        self.params_ref = list(params)


# --------------------------------------------------------------------------- #
# Differentiable Architecture Search
# --------------------------------------------------------------------------- #
class DifferentiableArchitectureSearch:
    """Differentiable NAS / neuroevolution via a relaxed supernet.

    Args:
        config: a :class:`DARTSConfig`, or per-field keyword overrides.
        input_dim / hidden_dim / depth / output_dim: supernet shape.
        operations: candidate primitive names per layer.
        relaxation_method: ``"darts"`` (softmax) or ``"gumbel_softmax"``.
        search_epochs: number of optimisation epochs.
        model_lr / arch_lr: learning rates for weights / architecture params.
        temperature: Gumbel-Softmax temperature.
        unrolled: use a (approximate) second-order update (still first-order
            here, kept for API parity with the spec).
        batch_size / random_state: data handling and reproducibility.
    """

    def __init__(
        self,
        config: Optional[DARTSConfig] = None,
        input_dim: int = 1,
        hidden_dim: int = 16,
        depth: int = 3,
        output_dim: int = 1,
        operations: Sequence[str] = PRIMITIVES,
        relaxation_method: str = "darts",
        search_epochs: int = 50,
        model_lr: float = 0.05,
        arch_lr: float = 0.05,
        temperature: float = 1.0,
        unrolled: bool = False,
        batch_size: int = 32,
        random_state: Optional[int] = None,
    ) -> None:
        if config is None:
            config = DARTSConfig()
        config.input_dim = input_dim or config.input_dim
        config.hidden_dim = hidden_dim or config.hidden_dim
        config.depth = depth or config.depth
        config.output_dim = output_dim or config.output_dim
        config.operations = list(operations) or list(config.operations)
        config.relaxation_method = relaxation_method
        config.search_epochs = search_epochs
        config.model_lr = model_lr
        config.arch_lr = arch_lr
        config.temperature = temperature
        config.unrolled = unrolled
        config.batch_size = batch_size
        config.random_state = random_state
        self.config = config.resolve()

        self.rng = random.Random(self.config.random_state)
        self.supernet = SuperNet(
            self.config.input_dim,
            self.config.hidden_dim,
            self.config.depth,
            self.config.output_dim,
            operations=self.config.operations,
            rng=self.rng,
        )
        self.config.num_ops = self.supernet.num_ops
        self.loss_history: List[float] = []
        self.arch_loss_history: List[float] = []
        self.epoch = 0

    # -- data helpers ---------------------------------------------------- #
    def _split(self, X: np.ndarray, y: np.ndarray, frac: float = 0.5):
        n = X.shape[0]
        idx = list(range(n))
        self.rng.shuffle(idx)
        k = max(1, int(n * frac))
        train_idx, val_idx = idx[:k], idx[k:]
        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    def _mse(self, pred: np.ndarray, target: np.ndarray) -> float:
        return float(np.mean((pred - target) ** 2))

    # -- Gumbel-Softmax sampling ----------------------------------------- #
    def _sample_architectures(self) -> List[np.ndarray]:
        """Per-layer soft sample (probabilities), used by both relaxations."""
        if self.config.relaxation_method == "gumbel_softmax":
            probs = []
            for a in self.supernet.alphas:
                logits = np.asarray(a, dtype=float) / max(self.config.temperature, 1e-6)
                g = self.rng.gammavariate(1.0, 1.0)  # approx for gumbel noise
                noise = -math.log(-math.log(self.rng.random() + 1e-12) + 1e-12)
                logits = logits + noise
                probs.append(_softmax(logits))
            return probs
        return self.supernet.architecture_probabilities()

    # -- training step (supervised, first order) -------------------------- #
    def _train_step(self, Xtr, ytr, Xval, yval):
        """One bilevel (approx) step. Returns (model_loss, arch_loss)."""
        # --- architecture step on the validation split ---
        arch_grads = [np.zeros_like(a) for a in self.supernet.alphas]
        # Gradient of validation loss w.r.t. alphas via the soft forward.
        # We compute it numerically over each layer's logits (cheap, robust).
        eps = 1e-4
        val_loss_base = self._mse(self._soft_forward(Xval), yval)
        for layer in range(self.config.depth):
            a = self.supernet.alphas[layer]
            for k in range(self.config.num_ops):
                a[k] += eps
                loss_plus = self._mse(self._soft_forward(Xval), yval)
                a[k] -= eps
                arch_grads[layer][k] = (loss_plus - val_loss_base) / eps
        arch_opt = _Adam(self.supernet.alphas, lr=self.config.arch_lr)
        arch_opt.bind(self.supernet.alphas)
        arch_opt.step(arch_grads)
        arch_loss = val_loss_base

        # --- model-weight step on the training split ---
        model_loss, grad_stem, grad_head, grad_ops = self._model_gradients(Xtr, ytr)
        flat_params, unflatten = self._collect_weight_params()
        # Flatten the nested [stem, head, [per-layer op grads]] into one list
        # aligned with ``flat_params``.
        flat_grads = [grad_stem, grad_head] + [
            gw for layer in grad_ops for gw in layer
        ]
        model_opt = _Adam(flat_params, lr=self.config.model_lr)
        model_opt.bind(flat_params)
        model_opt.step(flat_grads)
        self._apply_weight_grads(unflatten, flat_params)
        return model_loss, arch_loss

    def _collect_weight_params(self):
        params = [self.supernet.stem, self.supernet.head]
        for layer in self.supernet.op_weights:
            for w in layer:
                params.append(w)
        sizes = [p.shape for p in params]
        flats = [p.ravel() for p in params]
        total = sum(int(np.prod(s)) if s else 1 for s in sizes)

        def unflatten(updated):
            flat_vec = np.asarray(
                [float(v) for arr in updated for v in np.asarray(arr).ravel()]
            )
            out = []
            idx = 0
            for s in sizes:
                n = int(np.prod(s)) if s else 1
                out.append(flat_vec[idx:idx + n].reshape(s))
                idx += n
            return out

        return params, unflatten

    def _apply_weight_grads(self, unflatten, flat):
        restored = unflatten(flat)
        self.supernet.stem = restored[0]
        self.supernet.head = restored[1]
        pos = 2
        for layer in self.supernet.op_weights:
            for j in range(len(layer)):
                layer[j] = restored[pos]
                pos += 1

    def _model_gradients(self, X, y):
        """Mean squared error loss and gradients w.r.t. all model weights."""
        n = X.shape[0]
        total_loss = 0.0
        grad_stem = np.zeros_like(self.supernet.stem)
        grad_head = np.zeros_like(self.supernet.head)
        grad_ops = [[np.zeros_like(w) for w in layer] for layer in self.supernet.op_weights]

        probs = self._sample_architectures()
        for i in range(n):
            x = X[i]
            h = self.supernet.stem @ x
            hs = [h]
            mixed_store = []
            for layer in range(self.config.depth):
                p = probs[layer]
                mixed = np.zeros_like(h)
                for k, op_name in enumerate(self.supernet.operations):
                    mixed = mixed + p[k] * _op_forward(op_name, h, self.supernet.op_weights[layer][k])
                mixed_store.append((h, mixed))
                h = mixed
                hs.append(h)
            pred = self.supernet.head @ h
            loss = (pred - y[i]) ** 2
            total_loss += loss

            # Output gradient.
            dL_dpred = 2.0 * (pred - y[i])
            dL_dpred_vec = np.atleast_1d(np.asarray(dL_dpred, dtype=float))
            # head is (output_dim, hidden_dim); dL_dpred_vec is (output_dim,).
            # -> gradient w.r.t. hidden is (hidden_dim,).
            dL_dh_final = (self.supernet.head.T @ dL_dpred_vec)
            dL_dh_final = np.asarray(dL_dh_final, dtype=float).reshape(-1)
            grad_head += (np.asarray(h).reshape(-1, 1) @ dL_dpred_vec.reshape(1, -1)).T

            # Backprop through layers (final hidden -> stem).
            dh_next = dL_dh_final
            for layer in range(self.config.depth - 1, -1, -1):
                h_prev, mixed = mixed_store[layer]
                p = probs[layer]
                # gradient w.r.t each op's output, then to x and weight
                dh_prev = np.zeros_like(h_prev)
                for k, op_name in enumerate(self.supernet.operations):
                    w = self.supernet.op_weights[layer][k]
                    g_x, g_w = _op_backward(op_name, h_prev, mixed, w)
                    contrib = p[k] * g_x
                    dh_prev = dh_prev + contrib * dh_next
                    # Element-wise ops: weight gradient is the product of the
                    # incoming signal and the activation argument (both (n,)).
                    grad_ops[layer][k] = grad_ops[layer][k] + (g_w * (p[k] * dh_next))
                grad_stem += np.outer(dh_prev, x)
                dh_next = dh_prev

        scale = 2.0 / max(1, n)  # d/dx of mean of squared errors
        total_loss = total_loss / max(1, n)
        return (
            total_loss,
            grad_stem * scale,
            grad_head * scale,
            [[gw * scale for gw in layer] for layer in grad_ops],
        )

    def _soft_forward(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.supernet.forward(x) for x in X])

    # -- search loops ---------------------------------------------------- #
    def search(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Supervised DARTS search over (X, y) regression data."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        Xtr, ytr, Xval, yval = self._split(X, y, 0.5)
        for epoch in range(self.config.search_epochs):
            self.epoch = epoch
            model_loss, arch_loss = self._train_step(Xtr, ytr, Xval, yval)
            self.loss_history.append(model_loss)
            self.arch_loss_history.append(arch_loss)
        return self._result()

    def search_eval(
        self,
        evaluate: Callable[[Dict[str, Any]], float],
        steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Label-free search driven by a callable evaluator.

        ``evaluate(sample_genotype_dict) -> scalar loss``. Gradients are
        estimated with REINFORCE-style finite differences over the sampled
        architecture probabilities, then the architecture logits are nudged to
        lower the expected loss. Works for any black-box or LLM-based evaluator.
        """
        steps = steps or self.config.search_epochs
        probs = self._sample_architectures()
        for step in range(steps):
            self.epoch = step
            # Sample a discrete architecture from the current probabilities.
            sampled: List[int] = []
            for p in probs:
                sampled.append(int(self.rng.choices(range(len(p)), weights=p)[0]))
            sample_dict = {
                "genotype": self.supernet.genotype(),
                "sampled": sampled,
                "supernet": self.supernet,
            }
            loss = float(evaluate(sample_dict))
            self.loss_history.append(loss)

            # Score every candidate op per layer by evaluating a one-op swap.
            baselines = []
            for layer in range(self.config.depth):
                scores = []
                for k in range(self.config.num_ops):
                    swapped = list(sampled)
                    swapped[layer] = k
                    d = {"genotype": self.supernet.genotype(), "sampled": swapped,
                         "supernet": self.supernet}
                    try:
                        scores.append(float(evaluate(d)))
                    except Exception:
                        scores.append(loss)
                baselines.append(scores)
            baselines = np.array(baselines, dtype=float)

            # Update alphas with a softmax-cross-entropy-style gradient toward
            # lower-loss operations (smoothed by current probabilities).
            for layer in range(self.config.depth):
                p = probs[layer]
                grad = -p * (baselines[layer] - baselines[layer].mean())
                self.supernet.alphas[layer] = self.supernet.alphas[layer] + self.config.arch_lr * grad
            probs = self._sample_architectures()

        return self._result()

    def _result(self) -> Dict[str, Any]:
        chosen = self.supernet.discretize()
        return {
            "chosen_operations": [self.supernet.operations[c] for c in chosen],
            "genotype": self.supernet.genotype(),
            "alphas": [a.tolist() for a in self.supernet.alphas],
            "probabilities": [p.tolist() for p in self.supernet.architecture_probabilities()],
            "loss_history": list(self.loss_history),
            "arch_loss_history": list(self.arch_loss_history),
            "predictor": self.supernet.build_from_genotype(chosen),
        }

    def discretize_architecture(self) -> List[int]:
        """Return the final discrete per-layer operation indices."""
        return self.supernet.discretize()

    def get_masked_network(self, architecture: Sequence[int]) -> Callable[[np.ndarray], np.ndarray]:
        """Build a concrete network from a chosen architecture (spec method)."""
        return self.supernet.build_from_genotype(architecture)


# --------------------------------------------------------------------------- #
# Functional entry points (mirror run_cmaes / run_neat style)
# --------------------------------------------------------------------------- #
def differentiable_architecture_search(
    X: np.ndarray,
    y: np.ndarray,
    input_dim: Optional[int] = None,
    hidden_dim: int = 16,
    depth: int = 3,
    output_dim: int = 1,
    operations: Sequence[str] = PRIMITIVES,
    relaxation_method: str = "darts",
    search_epochs: int = 50,
    model_lr: float = 0.05,
    arch_lr: float = 0.05,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Supervised DARTS search over regression data ``(X, y)``."""
    if input_dim is None:
        input_dim = np.asarray(X, dtype=float).shape[1]
    das = DifferentiableArchitectureSearch(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        output_dim=output_dim if False else output_dim,
        operations=operations,
        relaxation_method=relaxation_method,
        search_epochs=search_epochs,
        model_lr=model_lr,
        arch_lr=arch_lr,
        random_state=random_state,
    )
    return das.search(X, y)


def run_darts(
    X: np.ndarray,
    y: np.ndarray,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias of :func:`differentiable_architecture_search`."""
    return differentiable_architecture_search(X, y, **kwargs)
