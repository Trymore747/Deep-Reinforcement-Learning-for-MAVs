# 03 — Simulation Testing

This guide covers running the R54 policy in the simulation test environments — including the real-world-scenario test (DARPA Tunnel Tile 5 geometry) and cross-validation.

---

## Test Scripts

All scripts are in `Final Results/testing/`.

| Script | Purpose |
|---|---|
| `run_test_r54_realworld.py` | 10-episode test in real-world tunnel geometry |
| `run_test_r54.py` | Test in the training world (`tunnel_100m_dynamic_25`) |
| `run_crossval_r54.py` | 5-level noise cross-validation |
| `run_demo_10flights.py` | Visual demonstration run (publishes to RViz) |

---

## Real-World Scenario Test (Primary)

This is the main test — it evaluates the policy in a tunnel geometry matching the DARPA Subterranean Challenge Tunnel Tile 5 (5 m wide corridor, 100 m long).

```bash
# Terminal 1 — launch Gazebo (real-world world)
ros2 launch uav_simulator start.launch world:=tunnel_basic_static

# Terminal 2 — run 10-episode test
source .venv/bin/activate
python3 "Final Results/testing/run_test_r54_realworld.py" \
    --policy src/tunnel_drl/results_r54/r54_best.pth \
    --episodes 10
```

Results are saved to `Final Results/testing/R54_RealWorld_TunnelTile5_20260905/`.

### Results (2026-09-05)

| Metric | Value |
|---|---|
| Success rate | **6 / 10 = 60%** |
| Average speed | 1.56 m/s |
| Min obstacle clearance | 0.06 m |
| Near-miss events | 38 total across 10 episodes |

| Episode | Result | Duration (s) | Speed (m/s) |
|---|---|---|---|
| A01 | SUCCESS | 101.2 | 1.61 |
| A02 | SUCCESS | 98.4  | 1.72 |
| A03 | TIMEOUT | 150.0 | 1.43 |
| A04 | SUCCESS | 103.7 | 1.58 |
| A05 | SUCCESS | 99.1  | 1.69 |
| A06 | TIMEOUT | 150.0 | 1.38 |
| A07 | SUCCESS | 105.6 | 1.52 |
| A08 | TIMEOUT | 150.0 | 1.41 |
| A09 | TIMEOUT | 150.0 | 1.45 |
| A10 | SUCCESS | 97.3  | 1.78 |

Result plots are in `Final Results/testing/R54_RealWorld_TunnelTile5_20260905/`:

![Summary dashboard](../Final%20Results/testing/R54_RealWorld_TunnelTile5_20260905/09_summary_dashboard.png)

---

## Cross-Validation (5 Noise Levels)

Tests robustness to domain-randomisation noise (σ in {0.00, 0.02, 0.04, 0.06, 0.08}).

```bash
python3 "Final Results/testing/run_crossval_r54.py" \
    --policy src/tunnel_drl/results_r54/r54_best.pth
```

Results: `Final Results/testing/cross_validation/`

---

## Analysing Results

After a test run, generate plots with:

```bash
python3 "Final Results/build_final_results.py"
```

This reads the CSV episode logs and produces the 6 thesis-quality plots in `Final Results/`.

---

## Notes for Reuse

- The test scripts expect a ROS 2 node to be running (`ros2 run ...` or `ros2 launch ...`).
- Set `ROS_DOMAIN_ID=42` to match the simulator.
- The policy is loaded from `--policy` path; default is `src/tunnel_drl/results_r54/r54_best.pth`.
- Each episode writes a CSV log to the output directory. The summary JSON is written at the end.
