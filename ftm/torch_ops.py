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
                                 starts: int = 3, lower_bounds=None,
                                 upper_bounds=None):
    """PGD over an ellipsoid, optionally intersected with a raw-space box."""
    torch = _torch()
    if alpha < 0 or steps < 0 or learning_rate <= 0:
        raise ValueError("alpha/steps must be non-negative and learning_rate positive")
    center = center.detach()
    axes = axes.detach().clamp_min(1e-8)
    if (lower_bounds is None) != (upper_bounds is None):
        raise ValueError("lower_bounds and upper_bounds must be provided together")
    z_lower = z_upper = None
    if lower_bounds is not None:
        lower = torch.as_tensor(lower_bounds, dtype=center.dtype, device=center.device)
        upper = torch.as_tensor(upper_bounds, dtype=center.dtype, device=center.device)
        if lower.shape != center.shape or upper.shape != center.shape:
            raise ValueError("box bounds must match the ellipsoid dimension")
        if bool(torch.any(lower > upper)):
            raise ValueError("lower_bounds must not exceed upper_bounds")
        if bool(torch.any(center < lower - 1e-7)) or bool(torch.any(center > upper + 1e-7)):
            raise ValueError("ellipsoid center must lie inside the box for a feasible intersection")
        if alpha > 0:
            radius = alpha * axes
            z_lower = (lower - center) / radius
            z_upper = (upper - center) / radius
            box_contains_ball = (bool(torch.all(z_lower <= -1.0).item())
                                 and bool(torch.all(z_upper >= 1.0).item()))
        else:
            box_contains_ball = True
    else:
        box_contains_ball = True

    def project_to_intersection(z):
        norm = torch.linalg.vector_norm(z).clamp_min(1e-12)
        if z_lower is None or box_contains_ball:
            return z / torch.maximum(norm, torch.ones_like(norm))

        box_point = torch.maximum(torch.minimum(z, z_upper), z_lower)
        if torch.linalg.vector_norm(box_point) <= 1.0:
            return box_point

        # KKT projection onto the ball-box intersection: find lambda >= 0
        # such that clip(z / (1 + lambda), box) has unit norm.
        low = 0.0
        high = 1.0
        while torch.linalg.vector_norm(torch.maximum(
                torch.minimum(z / (1.0 + high), z_upper), z_lower)) > 1.0:
            high *= 2.0
        for _ in range(20):
            middle = (low + high) / 2.0
            candidate = torch.maximum(
                torch.minimum(z / (1.0 + middle), z_upper), z_lower
            )
            if torch.linalg.vector_norm(candidate) > 1.0:
                low = middle
            else:
                high = middle
        return torch.maximum(torch.minimum(z / (1.0 + high), z_upper), z_lower)

    if alpha == 0:
        return center, objective(center).detach()
    best_point = center
    best_value = objective(center).detach()
    dim = center.numel()
    initial_vectors = [torch.zeros_like(center)]
    for start in range(max(0, starts - 1)):
        initial_vectors.append(project_to_intersection(
            torch.randn_like(center) / max(1.0, dim ** 0.5)
        ))
    for z in initial_vectors:
        z = project_to_intersection(z.detach())
        for _ in range(steps):
            z = z.detach().requires_grad_(True)
            point = center + alpha * axes * z
            value = objective(point)
            grad, = torch.autograd.grad(value, z)
            with torch.no_grad():
                z_next = z - learning_rate * grad
                z = project_to_intersection(z_next)
        point = center + alpha * axes * z.detach()
        value = objective(point).detach()
        if value < best_value:
            best_point, best_value = point, value
    return best_point, best_value
