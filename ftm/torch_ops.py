"""Optional PyTorch helpers for pathwise gradients and differentiable PGD.

PyTorch is an optional dependency. Import this module only for the deep-learning
extension; the simulator and tabular solver use the Python standard library.
"""

from __future__ import annotations


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("install the 'learning' extra to use PyTorch helpers") from exc
    return torch


def gaussian_reparameterize(mu, scale_tril, epsilon):
    """Sample x=mu+L epsilon, where L L^T is the covariance."""
    torch = _torch()
    return mu + torch.matmul(scale_tril, epsilon.unsqueeze(-1)).squeeze(-1)


def enumerated_discrete_expectation(probabilities, values):
    """Exact weighted expectation over enumerated discrete next-state outcomes."""
    return (probabilities * values).sum(dim=-1).mean()


def value_parameter_sensitivity(raw_parameters, value_from_parameters,
                                epsilon=None):
    """Differentiate expected value through continuous reparameterized states.

    ``value_from_parameters(raw_parameters, epsilon)`` must return a scalar tensor.
    For discrete events, enumerate their outcomes and include differentiable event
    probabilities inside that callable instead of sampling a hard category.
    """
    torch = _torch()
    raw = raw_parameters.detach().clone().requires_grad_(True)
    value = value_from_parameters(raw, epsilon)
    gradient, = torch.autograd.grad(value, raw, retain_graph=False)
    return value.detach(), gradient.detach()


def projected_adversarial_search(center, axes, alpha, objective,
                                 steps: int = 8, learning_rate: float = 0.05,
                                 starts: int = 3):
    """PGD over an L2 unit ball in normalized ellipsoid coordinates."""
    torch = _torch()
    center = center.detach()
    axes = axes.detach().clamp_min(1e-8)
    best_point = center
    best_value = objective(center).detach()
    dim = center.numel()
    initial_vectors = [torch.zeros_like(center)]
    for start in range(max(0, starts - 1)):
        initial_vectors.append(torch.randn_like(center) / max(1.0, dim ** 0.5))
    for z in initial_vectors:
        z = z.detach()
        for _ in range(steps):
            z = z.detach().requires_grad_(True)
            point = center + alpha * axes * z
            value = objective(point)
            grad, = torch.autograd.grad(value, z)
            with torch.no_grad():
                z_next = z - learning_rate * grad
                norm = torch.linalg.vector_norm(z_next).clamp_min(1e-12)
                z = z_next / torch.maximum(norm, torch.ones_like(norm))
        point = center + alpha * axes * z.detach()
        value = objective(point).detach()
        if value < best_value:
            best_point, best_value = point, value
    return best_point, best_value
