"""One-bag overview for the 10-26 HDX dataset doc: sensors (rate, recv-hdr delay, gaps), rig motion (Ouster IMU gyro),
GNSS (fix status, travelled distance), people (yolo11s at 1 Hz), thumbnails. Writes <out>/<bag>.json and <out>/<bag>.jpg."""
import json, struct, sys, datetime as D
from pathlib import Path
import cv2, numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

TS = get_typestore(Stores.ROS2_HUMBLE)
SENS = ["/radar_mid/driver/locations", "/radar_left/driver/locations", "/radar_right/driver/locations",
        "/oak/rgb/image_raw/compressed", "/oak/stereo/image_raw/compressedDepth", "/ouster/points", "/ouster/imu",
        "/oak/imu/data", "/bosch_imu/imu", "/fix", "/rdd/points_diffused"]
IMG = "/oak/rgb/image_raw/compressed"


def main(bag, out, model):
    bag = Path(bag); out = Path(out); name = bag.name
    hdr, gyro, fix, frames = {}, [], [], []
    with Reader(bag) as r:
        t0, t1 = r.start_time, r.end_time; dur = (t1 - t0) * 1e-9
        topics = {c.topic: c.msgcount for c in r.connections}
        cons = [c for c in r.connections if c.topic in SENS]
        next_img = t0
        for c, t, raw in r.messages(connections=cons):
            s, ns = struct.unpack_from("<iI", raw, 4)
            hdr.setdefault(c.topic, []).append((t * 1e-9, s + ns * 1e-9))
            if c.topic == "/ouster/imu":
                m = TS.deserialize_cdr(raw, c.msgtype); w = m.angular_velocity; a = m.linear_acceleration
                gyro.append((t * 1e-9, w.x, w.y, w.z, a.x, a.y, a.z))
            elif c.topic == "/fix":
                m = TS.deserialize_cdr(raw, c.msgtype)
                fix.append((t * 1e-9, m.status.status, m.latitude, m.longitude, m.altitude, m.position_covariance[0]))
            elif c.topic == IMG and t >= next_img:
                m = TS.deserialize_cdr(raw, c.msgtype)
                frames.append(((t - t0) * 1e-9, cv2.imdecode(np.frombuffer(m.data, np.uint8), 1))); next_img = t + 1_000_000_000
    rep = {"bag": name, "start_epoch": t0 * 1e-9,
           "start_kst": D.datetime.fromtimestamp(t0 * 1e-9, D.timezone(D.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S"),
           "duration_s": round(dur, 1), "size_gb": round(sum(f.stat().st_size for f in bag.glob("*.mcap")) / 1e9, 2),
           "topics": topics, "sensors": {}}
    for k, v in hdr.items():
        if len(v) < 3: continue
        a = np.array(v); lag = (a[:, 0] - a[:, 1]) * 1e3; dr = np.diff(a[:, 0])
        own = bool(np.median(np.abs(lag)) > 1e5)
        rep["sensors"][k] = {"hz": round(len(a) / dur, 1), "own_clock": own,
                             "lag_p50_ms": None if own else round(float(np.median(lag)), 1),
                             "lag_p95_ms": None if own else round(float(np.percentile(lag, 95)), 1),
                             "lag_max_ms": None if own else round(float(lag.max()), 1),
                             "gaps": int((dr > 3 * np.median(dr)).sum()), "max_gap_ms": round(float(dr.max() * 1e3))}
    if gyro:
        g = np.array(gyro); hz = len(g) / dur; k = max(1, int(hz * 0.5))
        lp = np.c_[[np.convolve(np.degrees(g[:, i]), np.ones(k) / k, "same") for i in (1, 2, 3)]].T   # 0.5 s mean per axis
        yaw = np.abs(lp[:, 2]); tilt = np.linalg.norm(lp[:, :2], axis=1)
        acc = np.linalg.norm(g[:, 4:7], axis=1); hp = acc - np.convolve(acc, np.ones(50) / 50, "same")
        rep["motion"] = {"yaw_p50_dps": round(float(np.median(yaw)), 2), "yaw_p95_dps": round(float(np.percentile(yaw, 95)), 1),
                         "yaw_max_dps": round(float(yaw.max()), 1), "turning_frac": round(float((yaw > 5).mean()), 3),
                         "yaw_total_deg": round(float(yaw.sum() / hz)), "tilt_p95_dps": round(float(np.percentile(tilt, 95)), 1),
                         "vib_acc_std": round(float(np.std(hp[50:-50])), 3), "yaw_series": np.round(yaw[::max(1, int(hz))], 1).tolist()}
    if fix:
        f = np.array(fix); ok = f[:, 1] >= 0
        rep["gnss"] = {"n": len(f), "fix_frac": round(float(ok.mean()), 2)}
        f = f[ok]
    if fix and len(f) > 1:
        lat0 = np.radians(f[0, 2])
        xy = np.c_[(f[:, 3] - f[0, 3]) * 111320 * np.cos(lat0), (f[:, 2] - f[0, 2]) * 110540]
        rep["gnss"].update({"status_mode": int(np.bincount(f[:, 1].astype(int)).argmax()),
                       "path_m": round(float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()), 1),
                       "span_m": round(float(np.ptp(xy, axis=0).max()), 1), "cov_p50": float(np.median(f[:, 5])),
                       "lat": round(float(f[0, 2]), 6), "lon": round(float(f[0, 3]), 6)})
    if frames and model:
        from ultralytics import YOLO
        y = YOLO(model); np_, hmax, veh = [], [], []
        for _, im in frames:
            b = y.predict(im, imgsz=960, conf=0.35, classes=[0, 2, 7], verbose=False)[0].boxes
            cls = b.cls.cpu().numpy(); h = (b.xyxy[:, 3] - b.xyxy[:, 1]).cpu().numpy() / im.shape[0]
            np_.append(int((cls == 0).sum())); hmax.append(float(h[cls == 0].max()) if (cls == 0).any() else 0.0); veh.append(int((cls != 0).sum()))
        np_ = np.array(np_)
        rep["people"] = {"frames": len(np_), "p50": int(np.median(np_)), "max": int(np_.max()), "frac_any": round(float((np_ > 0).mean()), 2),
                         "max_box_h_frac": round(max(hmax), 2), "per_s": np_.tolist(), "veh_p50": int(np.median(veh)),
                         "img_wh": [frames[0][1].shape[1], frames[0][1].shape[0]]}
    N = 6; pick = [frames[int((i + 0.5) * len(frames) / N)] for i in range(N)] if len(frames) >= N else frames
    tiles = []
    for ts_, im in pick:
        im = cv2.resize(im, (400, int(400 * im.shape[0] / im.shape[1])))
        cv2.putText(im, f"t={ts_:.0f}s", (6, 24), 0, 0.7, (0, 255, 255), 2); tiles.append(im)
    cv2.imwrite(str(out / f"{name}.jpg"), np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:6])]) if len(tiles) == 6 else np.hstack(tiles),
                [cv2.IMWRITE_JPEG_QUALITY, 80])
    json.dump(rep, open(out / f"{name}.json", "w"), indent=1)
    print(name, json.dumps({k: rep[k] for k in rep if k in ("duration_s", "motion", "gnss")}), rep.get("people", {}).get("p50"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
