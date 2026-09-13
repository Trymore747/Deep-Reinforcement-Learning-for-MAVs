#!/usr/bin/env python3
"""
Tunnel TD3 — Inference Benchmark
=================================

Measures inference latency of the Actor network to verify it meets
real-world deployment constraints (target: <100 µs on CPU,
<50 µs on GPU / Jetson Nano).

No ROS2 or Gazebo required — pure PyTorch benchmark.

Usage:
  python3 benchmark_inference.py
  python3 benchmark_inference.py --model src/tunnel_drl/results/tunnel_drl_best.pth
"""

from __future__ import annotations

import argparse
import time
import numpy as np
import torch

# Reuse the same Actor class
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tunnel_td3_nav import Actor, DEVICE


def benchmark(model_path: str = None, n_warmup: int = 500,
              n_iters: int = 5000):
    """Run inference benchmark on the Actor network."""

    actor = Actor(state_dim=12, action_dim=3).to(DEVICE)

    if model_path:
        ckpt = torch.load(model_path, map_location=DEVICE)
        actor.load_state_dict(ckpt['actor'])
        print(f"📂 Loaded model: {model_path}")
    else:
        print("📦 Using randomly initialised weights")

    actor.eval()

    total_params = sum(p.numel() for p in actor.parameters())
    print(f"🧠 Actor: {total_params:,} parameters")
    print(f"⚡ Device: {DEVICE}")
    print(f"🔢 Warmup: {n_warmup}   Benchmark: {n_iters} iterations\n")

    # Random state batch (single inference — real-world scenario)
    dummy = torch.randn(1, 12, device=DEVICE)

    # ── Warmup ──
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = actor(dummy)

    # ── Timed run ──
    latencies = []
    with torch.no_grad():
        for _ in range(n_iters):
            state = torch.randn(1, 12, device=DEVICE)
            t0 = time.perf_counter()
            action = actor(state)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1e6)  # µs

    lat = np.array(latencies)

    print("=" * 55)
    print("  TUNNEL TD3 — INFERENCE BENCHMARK")
    print("=" * 55)
    print(f"  Mean latency   : {lat.mean():.1f} µs")
    print(f"  Median latency : {np.median(lat):.1f} µs")
    print(f"  P95 latency    : {np.percentile(lat, 95):.1f} µs")
    print(f"  P99 latency    : {np.percentile(lat, 99):.1f} µs")
    print(f"  Min latency    : {lat.min():.1f} µs")
    print(f"  Max latency    : {lat.max():.1f} µs")
    print(f"  Throughput     : {1e6 / lat.mean():.0f} Hz")
    print(f"  Parameters     : {total_params:,}")
    print(f"  Model size     : {total_params * 4 / 1024:.1f} KB (FP32)")
    print("=" * 55)

    # ── Real-world readiness check ──
    mean_us = lat.mean()
    if mean_us < 50:
        print("  ✅ EXCELLENT — ready for Jetson Nano / real-time control")
    elif mean_us < 100:
        print("  ✅ GOOD — suitable for real-time 10 Hz control loop")
    elif mean_us < 500:
        print("  ⚠️  ACCEPTABLE — may need optimisation for embedded")
    else:
        print("  ❌ TOO SLOW — needs model compression or hardware upgrade")

    return lat.mean()


def main():
    parser = argparse.ArgumentParser(
        description="Tunnel TD3 Inference Benchmark")
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model (.pth)')
    parser.add_argument('--iters', type=int, default=5000,
                        help='Number of benchmark iterations')
    args = parser.parse_args()

    benchmark(model_path=args.model, n_iters=args.iters)


if __name__ == '__main__':
    main()
