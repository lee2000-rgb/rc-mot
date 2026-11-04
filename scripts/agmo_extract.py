#!/usr/bin/env python3
"""Stream an agmo rig bag once (read-only) and dump compact time series to an .npz:
GNSS/odom/IMU/heading series, /tf map->base_link, and per-scan radar ego-velocity fits
(rcmot ScanStatistician, same call as pipeline._radar) for radar_{left,mid,right}.

usage: PYTHONPATH=src python3 scripts/agmo_extract.py BAG.mcap OUT.npz [--max-s S]
"""
import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

from rcmot.radar.scan_stats import ScanStatistician, ScanStatsParams
from rcmot.sensors.bag_source import pc2_to_struct, radar_frame
from rcmot.utils.calib import RigCalib

RADARS = ["/radar_left/driver/locations", "/radar_mid/driver/locations", "/radar_right/driver/locations"]


def hs(m):
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("out")
    ap.add_argument("--max-s", type=float, default=None)
    a = ap.parse_args()
    ident = RigCalib(np.eye(3), np.zeros(5), (640, 400), np.eye(3), np.zeros(3), np.eye(3), np.zeros(3))
    # same parameters as the HDX config (ego_velocity.planar etc. are ScanStatsParams defaults)
    stats = {t: ScanStatistician(ScanStatsParams(), seed=0) for t in RADARS}
    S = defaultdict(list)
    strings = defaultdict(list)
    want = set(RADARS) | {"/agmo/gnss/fix", "/agmo/gnss/raw_fix", "/agmo/gnss/vel", "/agmo/odom", "/agmo/imu",
                          "/agmo/yaw_deg", "/agmo/heading_track_deg", "/agmo/gnss/raw_heading_deg",
                          "/agmo/gnss/status", "/agmo/gnss/raw_status", "/agmo/heading_track_status",
                          "/ouster/imu", "/oak/imu/data", "/realsense/imu", "/tf"}
    t_wall = time.time()
    with AnyReader([Path(a.bag)]) as r:
        conns = [c for c in r.connections if c.topic in want]
        t_end = r.start_time + (a.max_s * 1e9 if a.max_s else 1e30)
        nmsg = 0
        for c, t_ns, raw in r.messages(connections=conns):
            if t_ns > t_end:
                break
            nmsg += 1
            t = t_ns * 1e-9
            tp = c.topic
            m = r.deserialize(raw, c.msgtype)
            if tp in RADARS:
                arr = pc2_to_struct(m)
                rf = radar_frame(arr, hs(m), ident)
                st = stats[tp].compute(rf.stamp, rf.xyz_radar, rf.v_r, rf.snr, rf.rcs, rf.peer)
                az = np.degrees(np.arctan2(rf.xyz_radar[:, 1], rf.xyz_radar[:, 0])) if len(rf.v_r) else np.zeros(1)
                rng = np.linalg.norm(rf.xyz_radar, axis=1) if len(rf.v_r) else np.zeros(1)
                k = tp.split("/")[1]
                S[k].append([t, hs(m), *st.v_sensor, float(st.ego_valid), st.inlier_ratio, st.sigma_v, st.n,
                             float(st.dynamic.sum()), np.percentile(az, 2), np.percentile(az, 98),
                             np.percentile(rng, 50), float(np.percentile(np.abs(rf.v_r), 70)) if len(rf.v_r) else 0.0])
                # azimuth histogram of points for FOV (accumulate)
                h, _ = np.histogram(az, bins=72, range=(-90, 90))
                S[k + "_azh"].append(h)
            elif tp in ("/agmo/gnss/fix", "/agmo/gnss/raw_fix"):
                S[tp].append([t, hs(m), m.latitude, m.longitude, m.altitude, m.status.status,
                              m.position_covariance[0], m.position_covariance[4], m.position_covariance[8]])
            elif tp == "/agmo/gnss/vel":
                l, w = m.twist.linear, m.twist.angular
                S[tp].append([t, hs(m), l.x, l.y, l.z, w.z])
            elif tp == "/agmo/odom":
                p, q, l, w = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist.linear, m.twist.twist.angular
                S[tp].append([t, hs(m), p.x, p.y, p.z, q.x, q.y, q.z, q.w, l.x, l.y, l.z, w.x, w.y, w.z])
                if len(S[tp]) == 1:
                    strings["odom_frames"].append(f"{m.header.frame_id}->{m.child_frame_id} cov_pose={list(m.pose.covariance[::7])} cov_tw={list(m.twist.covariance[::7])}")
            elif tp in ("/agmo/imu", "/ouster/imu", "/realsense/imu"):
                g, acc, q = m.angular_velocity, m.linear_acceleration, m.orientation
                S[tp].append([t, hs(m), g.x, g.y, g.z, acc.x, acc.y, acc.z, q.x, q.y, q.z, q.w])
            elif tp == "/oak/imu/data":
                g, acc = m.imu.angular_velocity, m.imu.linear_acceleration
                S[tp].append([t, hs(m), g.x, g.y, g.z, acc.x, acc.y, acc.z, 0, 0, 0, 1])
            elif tp in ("/agmo/yaw_deg", "/agmo/heading_track_deg", "/agmo/gnss/raw_heading_deg"):
                S[tp].append([t, m.data])
            elif tp in ("/agmo/gnss/status", "/agmo/gnss/raw_status", "/agmo/heading_track_status"):
                strings[tp].append((t, m.data))
            elif tp == "/tf":
                for tf in m.transforms:
                    key = f"tf:{tf.header.frame_id}->{tf.child_frame_id}"
                    tr, q = tf.transform.translation, tf.transform.rotation
                    S[key].append([t, hs(tf), tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w])
            if nmsg % 50000 == 0:
                print(f"{nmsg} msgs, bag t+{(t_ns - r.start_time) * 1e-9:.0f}s, wall {time.time() - t_wall:.0f}s", flush=True)
    out = {k: np.asarray(v, float) for k, v in S.items()}
    for k, v in strings.items():
        out["str:" + k] = np.array([f"{tt:.3f}|{s}" for tt, s in v] if isinstance(v[0], tuple) else v)
    np.savez_compressed(a.out, **out)
    print("saved", a.out, {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
