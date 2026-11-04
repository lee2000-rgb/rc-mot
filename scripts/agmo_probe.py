#!/usr/bin/env python3
"""Probe an agmo rig bag (read-only, streaming): topics, /tf_static (rpy deg), /tf content, camera_info,
first message of selected topics, radar PointCloud2 fields."""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation as Rot


def q2rpy(q):
    return Rot.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz", degrees=True)


def main(bag):
    with AnyReader([Path(bag)]) as r:
        print("duration", (r.end_time - r.start_time) * 1e-9)
        for c in sorted(r.connections, key=lambda c: c.topic):
            print(f"  {c.topic:55s} {c.msgtype:60s} {c.msgcount}")
        conns = {c.topic: c for c in r.connections}
        # tf_static
        print("\n== /tf_static ==")
        seen = {}
        for c, t, raw in r.messages(connections=[x for x in r.connections if x.topic == "/tf_static"]):
            m = r.deserialize(raw, c.msgtype)
            for tf in m.transforms:
                tr = tf.transform.translation
                seen[(tf.header.frame_id, tf.child_frame_id)] = (tr.x, tr.y, tr.z, *q2rpy(tf.transform.rotation))
        for (p, ch), v in sorted(seen.items()):
            print(f"  {p:25s} -> {ch:35s} t=({v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f}) rpy=({v[3]:+.2f},{v[4]:+.2f},{v[5]:+.2f})")
        # /tf content (first 30 s)
        print("\n== /tf (first 30 s) ==")
        cnt = Counter()
        first = {}
        t0 = r.start_time
        for c, t, raw in r.messages(connections=[conns["/tf"]]):
            if t > t0 + 30e9:
                break
            m = r.deserialize(raw, c.msgtype)
            for tf in m.transforms:
                k = (tf.header.frame_id, tf.child_frame_id)
                cnt[k] += 1
                first.setdefault(k, tf)
        for k, n in cnt.items():
            tf = first[k]
            tr = tf.transform.translation
            print(f"  {k} n={n} ({n/30:.1f} Hz) first t=({tr.x:.2f},{tr.y:.2f},{tr.z:.2f}) rpy={np.round(q2rpy(tf.transform.rotation),2)}")
        # first message of interesting topics
        want = [t for t in conns if "camera_info" in t or t.startswith("/agmo") or t.endswith("imu") or
                t.endswith("imu/data") or t.endswith("locations") or t.endswith("locations_header")
                or "image" in t]
        got = set()
        print("\n== first messages ==")
        for c, t, raw in r.messages(connections=[conns[x] for x in want]):
            if c.topic in got:
                if len(got) == len(want):
                    break
                continue
            got.add(c.topic)
            m = r.deserialize(raw, c.msgtype)
            print(f"\n--- {c.topic} ({c.msgtype}) bag_t-hdr={(t*1e-9 - (m.header.stamp.sec+m.header.stamp.nanosec*1e-9)) if hasattr(m,'header') else 'n/a'}")
            if c.msgtype == "sensor_msgs/msg/PointCloud2":
                print("   frame", m.header.frame_id, "n", m.width * m.height, "fields", [(f.name, f.datatype, f.offset) for f in m.fields])
            elif c.msgtype == "sensor_msgs/msg/CameraInfo":
                print("   frame", m.header.frame_id, m.width, m.height, m.distortion_model, "K", list(m.k), "D", list(m.d))
            elif "Image" in c.msgtype:
                print("   frame", m.header.frame_id, getattr(m, "format", getattr(m, "encoding", "")))
            else:
                s = repr(m)
                print("  ", s[:1500])


if __name__ == "__main__":
    main(sys.argv[1])
