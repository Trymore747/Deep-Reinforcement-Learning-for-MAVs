#!/usr/bin/env python3
"""
Tunnel TD3 v6 Deploy — Deterministic policy for any environment
===============================================================

Loads a v6 checkpoint and runs the trained policy in any environment.
Works in: training tunnel, corridor, any other environment.

Position-free 12-D state means NO distribution shift regardless of
environment length, obstacle layout, or turn-point distance.

Usage:
  # Tunnel (training environment):
  python3 tunnel_td3_v6_deploy.py --model results_v6/tunnel_td3_v6_best.pth --missions 3

  # Corridor (unseen environment, 40 m):
  python3 tunnel_td3_v6_deploy.py --model results_v6/tunnel_td3_v6_best.pth \
      --missions 3 --turn-point 35 --speed 4.0

  # Custom altitude (e.g. above short obstacles):
  python3 tunnel_td3_v6_deploy.py --model results_v6/tunnel_td3_v6_best.pth \
      --missions 3 --turn-point 35 --altitude 2.2
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time

import numpy as np
import torch

import rclpy

from tunnel_td3_v6 import (
    Actor,
    CERLABDroneInterface,
    Mission,
    SAFE_ALT, MAX_ALT, SPEED_STAGES,
    STATE_DIM, ACTION_DIM, DEVICE,
)
import tunnel_td3_v6


def recovery(drone: CERLABDroneInterface, spawn_x: float, spawn_y: float,
             cruise_alt: float = SAFE_ALT) -> bool:
    """Teleport back to spawn and re-takeoff."""
    print("\n  🔄 Recovery — teleporting to spawn …")
    ok = drone.teleport_to_spawn(spawn_x, spawn_y, 0.2)
    if not ok:
        print("  ❌ Teleport failed")
        return False
    drone.takeoff()
    t0 = time.time()
    while time.time() - t0 < 6.0:
        drone.publish_setpoint(spawn_x, spawn_y, cruise_alt, 0.0)
        time.sleep(0.08)
    if drone.pz < 0.5:
        t0 = time.time()
        while time.time() - t0 < 4.0:
            drone.publish_setpoint(spawn_x, spawn_y, cruise_alt, 0.0)
            time.sleep(0.08)
    print(f"  ✅ Recovered  alt={drone.pz:.2f}")
    return True


def run_deploy(args):
    # Altitude override (patches module constants so Mission uses new altitude)
    if args.altitude is not None:
        tunnel_td3_v6.SAFE_ALT = args.altitude
        safe_ceiling = args.altitude + 0.25
        if tunnel_td3_v6.MAX_ALT > safe_ceiling:
            tunnel_td3_v6.MAX_ALT = safe_ceiling
        print(f"⚙️  Altitude override: SAFE_ALT → {args.altitude:.2f} m  "
              f"MAX_ALT → {tunnel_td3_v6.MAX_ALT:.2f} m")
    cruise_alt = tunnel_td3_v6.SAFE_ALT

    rclpy.init()
    drone = CERLABDroneInterface()
    drone.start_spinning()

    print("⏳ Waiting for pose …")
    for _ in range(200):
        time.sleep(0.1)
        if drone.pose_received:
            break
    if not drone.pose_received:
        print("❌ No pose — is Gazebo running?")
        return

    print(f"📍 Pose: ({drone.px:.2f}, {drone.py:.2f}, {drone.pz:.2f})")
    print(f"⚙️  Cruise altitude: {cruise_alt:.2f} m")

    if math.hypot(drone.px, drone.py) > 1.0:
        print(f"  ⚠️  Not at origin — teleporting …")
        drone.teleport_to_spawn(0.0, 0.0, 0.2)
        for _ in range(50):
            drone.publish_setpoint(0.0, 0.0, cruise_alt, 0.0)
            time.sleep(0.08)

    SPAWN_X, SPAWN_Y = 0.0, 0.0

    # Load model
    print(f"📂 Loading: {args.model}")
    ckpt = torch.load(args.model, map_location=DEVICE, weights_only=False)
    saved_sd = ckpt.get('state_dim', STATE_DIM)
    if saved_sd != STATE_DIM:
        print(f"  ⚠️  Checkpoint state_dim={saved_sd} ≠ {STATE_DIM} — wrong checkpoint version?")
        return
    actor = Actor(state_dim=STATE_DIM, action_dim=ACTION_DIM).to(DEVICE)
    actor.load_state_dict(ckpt['actor'])
    actor.eval()
    total_params = sum(p.numel() for p in actor.parameters())
    print(f"🧠 Actor: {total_params:,} parameters  (state_dim={saved_sd})")

    if 'episode' in ckpt:
        print(f"   Trained: ep={ckpt['episode']}  "
              f"speed_stage={ckpt.get('speed_stage', '?')}  "
              f"best_reward={ckpt.get('best_reward', '?'):.0f}"
              if isinstance(ckpt.get('best_reward'), float) else
              f"   Trained: ep={ckpt['episode']}")

    # Takeoff
    print("🚁 Takeoff …")
    drone.takeoff()
    t0 = time.time()
    while time.time() - t0 < 6.0:
        drone.publish_setpoint(SPAWN_X, SPAWN_Y, cruise_alt, 0.0)
        time.sleep(0.08)
    if drone.pz < 0.5:
        t0 = time.time()
        while time.time() - t0 < 4.0:
            drone.publish_setpoint(SPAWN_X, SPAWN_Y, cruise_alt, 0.0)
            time.sleep(0.08)
    for _ in range(30):
        drone.publish_setpoint(SPAWN_X, SPAWN_Y, cruise_alt, 0.0)
        time.sleep(0.08)

    speed      = args.speed if args.speed else SPEED_STAGES[-1]
    turn_point = args.turn_point

    print(f"📍 Ready: ({drone.px:.2f}, {drone.py:.2f}, {drone.pz:.2f})")
    print(f"⚙️  speed={speed:.1f} m/s  turn={turn_point:.0f} m")

    for mi in range(args.missions):
        # Reset position if needed
        if drone.pz < 0.5:
            ok = recovery(drone, SPAWN_X, SPAWN_Y, cruise_alt)
            if not ok:
                break
        else:
            dist = math.hypot(drone.px - SPAWN_X, drone.py - SPAWN_Y)
            if dist > 2.0:
                ok = recovery(drone, SPAWN_X, SPAWN_Y, cruise_alt)
                if not ok:
                    break
            else:
                for _ in range(25):
                    drone.publish_setpoint(SPAWN_X, SPAWN_Y, cruise_alt, 0.0)
                    time.sleep(0.08)

        mission = Mission(drone, max_speed=speed, turn_x=turn_point)
        mission.start()

        # Wait for clear sensor reading
        t0 = time.time()
        while time.time() - t0 < 15.0:
            if float(np.min(drone.depth_sectors)) > 1.0:
                break
            drone.publish_setpoint(SPAWN_X, SPAWN_Y, cruise_alt, 0.0)
            time.sleep(0.1)

        state = mission.get_state()
        ep_start = time.time()
        done = False
        info = {}

        # RETURN-phase stall recovery counters
        return_blocked_steps = 0
        return_stall_steps   = 0
        RETURN_BLOCKED_LIMIT = 25
        RETURN_STALL_LIMIT   = 20

        print(f"\n{'─' * 60}")
        print(f"  🎯 DEPLOY MISSION {mi + 1}/{args.missions}   "
              f"speed={speed:.1f} m/s  alt={cruise_alt:.1f}m  turn={turn_point:.0f}m")
        print(f"{'─' * 60}")

        while not done:
            if mission.phase == "RETURN":
                min_d = float(np.min(drone.depth_sectors))

                if min_d < 0.4:
                    return_blocked_steps += 1
                    return_stall_steps    = 0
                    if return_blocked_steps >= RETURN_BLOCKED_LIMIT:
                        print(f"\n  ⏸  RETURN blocked at x={mission.fwd_progress:.1f}m — waiting …")
                        t_wait = time.time()
                        cleared = False
                        yaw_ret = math.pi
                        while time.time() - t_wait < 6.0:
                            drone.publish_setpoint(drone.px, drone.py, cruise_alt, yaw_ret)
                            time.sleep(0.1)
                            if float(np.min(drone.depth_sectors)) > 1.0:
                                cleared = True
                                break
                        if not cleared:
                            backup_x = drone.px - 3.0
                            t_b = time.time()
                            while time.time() - t_b < 2.5:
                                drone.publish_setpoint(backup_x, drone.py, cruise_alt, yaw_ret)
                                time.sleep(0.08)
                        return_blocked_steps = 0
                        state = mission.get_state()
                        continue

                elif min_d > 1.5 and abs(drone.vx) < 0.3:
                    return_stall_steps   += 1
                    return_blocked_steps  = 0
                    if return_stall_steps >= RETURN_STALL_LIMIT:
                        print(f"\n  ↩️  RETURN stalled at x={mission.fwd_progress:.1f}m — forcing home …")
                        t_fly = time.time()
                        while time.time() - t_fly < 8.0:
                            tgt_x = max(float(SPAWN_X), drone.px - 12.0)
                            drone.publish_setpoint(tgt_x, drone.py, cruise_alt, math.pi)
                            time.sleep(0.1)
                            if drone.px < SPAWN_X + 1.5:
                                break
                            if float(np.min(drone.depth_sectors)) < 0.5:
                                break
                        print(f"  ✅ Forced to x={drone.px:.1f}m — resuming policy")
                        return_stall_steps = 0
                        state = mission.get_state()
                        continue
                else:
                    return_blocked_steps = 0
                    return_stall_steps   = 0

            # Deterministic action (no noise)
            with torch.no_grad():
                s      = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                action = actor(s).cpu().numpy().flatten()

            next_state, rew, done, info = mission.step(action)

            if mission.steps % 25 == 0:
                ds = drone.depth_sectors
                print(f"  [{mission.phase:8s}] {mission.steps:4d} │ "
                      f"X:{mission.fwd_progress:+7.1f}/{turn_point:.0f}m  "
                      f"Z:{drone.pz:.2f}m │ "
                      f"V:{mission.speed:+5.2f} │ "
                      f"D:[{ds[0]:.1f},{ds[2]:.1f},{ds[4]:.1f}]")

            state = next_state

        dur    = time.time() - ep_start
        evt    = info.get('event', '?')
        ok_str = '✅ SUCCESS' if evt == 'success' else f'❌ {evt}'
        print(f"\n  📊 Mission {mi + 1}: dist={mission.fwd_progress:.1f}m  "
              f"max_spd={mission.max_vel:.2f} m/s  "
              f"col={mission.collision_count}  avd={mission.obstacles_avoided}  "
              f"dur={dur:.1f}s  {ok_str}")

    print("🛬 Landing …")
    drone.land()
    time.sleep(5.0)
    drone.stop_spinning()
    drone.destroy_node()
    rclpy.shutdown()
    print("🏁 Deploy complete.")


def main():
    parser = argparse.ArgumentParser(description="Tunnel TD3 v6 Deploy")
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained v6 model (.pth)')
    parser.add_argument('--missions', type=int, default=3,
                        help='Number of missions to run')
    parser.add_argument('--speed', type=float, default=None,
                        help='Override speed (m/s). Default: max trained stage')
    parser.add_argument('--turn-point', type=float, default=95.0,
                        help='Distance (m) at which to reverse. '
                             '95 for tunnel, 35 for 40m corridor.')
    parser.add_argument('--altitude', type=float, default=None,
                        help='Override cruise altitude (m). Default: SAFE_ALT=1.5')
    args = parser.parse_args()

    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

    try:
        run_deploy(args)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n❌ Fatal: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
