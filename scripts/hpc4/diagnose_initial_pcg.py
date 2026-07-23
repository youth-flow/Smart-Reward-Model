#!/usr/bin/env python3
"""Diagnose the zero-head ProRM+ Fisher solve on a materialized artifact."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from smart_reward.artifacts import load_controlled_feature_artifact
from smart_reward.config import config_hash, load_config
from smart_reward.linear import DampedEmpiricalFisher, resolve_fisher_solve_dtype
from smart_reward.objective import empirical_moment
from smart_reward.pcg import pcg


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-iterations", type=int, required=True)
    parser.add_argument("--residual-recompute-interval", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    config = load_config(arguments.config)
    digest = config_hash(config)
    experiment = load_controlled_feature_artifact(
        arguments.artifact,
        expected_config_hash=digest,
        expected_seed=arguments.seed,
    )
    solve_dtype = resolve_fisher_solve_dtype(config["objective"]["pcg_dtype"])
    stored_scores = experiment.train.policy_scores.to(arguments.device)
    targets = experiment.train.h.to(arguments.device, dtype=solve_dtype)
    # Match the formal byte path: the canonical edge difference is formed in
    # artifact FP32 and only then promoted, while Fisher nodes are promoted
    # directly from their stored values.
    flat_scores = stored_scores.reshape(-1, stored_scores.shape[-1]).to(dtype=solve_dtype)
    edge_scores = (stored_scores[:, 0] - stored_scores[:, 1]).to(dtype=solve_dtype)
    margins = torch.zeros_like(targets)
    moment = empirical_moment(edge_scores, margins, targets)

    relative_damping = float(config["objective"]["damping_relative_to_mean_fisher_diagonal"])
    mean_fisher_diagonal = float(flat_scores.square().mean(dim=0).mean().item())
    damping = relative_damping * mean_fisher_diagonal
    if not math.isfinite(damping) or damping <= 0.0:
        raise ValueError("diagnostic damping must be finite and positive")
    operator = DampedEmpiricalFisher(flat_scores, damping)

    if flat_scores.is_cuda:
        torch.cuda.synchronize(flat_scores.device)
        torch.cuda.reset_peak_memory_stats(flat_scores.device)
    started = time.perf_counter()
    result = pcg(
        operator.matvec,
        moment,
        inverse_diagonal=None,
        max_iterations=arguments.max_iterations,
        tolerance=float(config["objective"]["pcg_tolerance"]),
        absolute_tolerance=0.0,
        residual_recompute_interval=arguments.residual_recompute_interval,
    )
    if flat_scores.is_cuda:
        torch.cuda.synchronize(flat_scores.device)
    elapsed = time.perf_counter() - started

    true_residual = moment - operator.matvec(result.solution)
    rhs_norm = float(torch.linalg.vector_norm(moment).item())
    residual_norm = float(torch.linalg.vector_norm(true_residual).item())
    payload = {
        "artifact": arguments.artifact.name,
        "config_hash": digest,
        "converged": result.converged,
        "damping": damping,
        "device": str(flat_scores.device),
        "dtype": str(flat_scores.dtype),
        "elapsed_seconds": elapsed,
        "iterations": result.iterations,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "max_iterations": arguments.max_iterations,
        "mean_fisher_diagonal": mean_fisher_diagonal,
        "num_parameters": flat_scores.shape[1],
        "num_samples": flat_scores.shape[0],
        "peak_memory_allocated_bytes": (
            torch.cuda.max_memory_allocated(flat_scores.device) if flat_scores.is_cuda else None
        ),
        "peak_memory_reserved_bytes": (
            torch.cuda.max_memory_reserved(flat_scores.device) if flat_scores.is_cuda else None
        ),
        "reason": result.reason,
        "relative_residual": residual_norm / rhs_norm,
        "reported_relative_residual": result.relative_residual,
        "residual_recompute_interval": arguments.residual_recompute_interval,
        "seed": arguments.seed,
        "tolerance": float(config["objective"]["pcg_tolerance"]),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
