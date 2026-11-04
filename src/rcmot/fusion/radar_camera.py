"""Late fusion: 2D detections + 4D radar scan (with ScanStats) -> 3D detections with absolute velocity.

Per box (docs/08 §3[C], docs/09 §4.5):
  candidates   radar points projected inside the box (slightly shrunk) and within a generous gate around
               the ground-ray range of the box bottom; each point is given to ONE box only (the box whose
               ground-ray range it agrees with best) so overlapping boxes do not share returns
  weights      w = alpha (range-conditioned SNR reliability) x keep (clutter filter)
  range mode   weighted kernel density over range, bandwidth = radar range sigma -> densest mode
  Doppler      points of the mode whose ego-compensated Doppler is > dop_z robust sigmas (scan sigma_v)
               away from the mode's weighted median are dropped (person walking in front of a parked machine)
  position     bearing from the CAMERA (box-centre ray, ~0.1 deg) x range from the RADAR (cm-level);
               covariance built along / across the ray. No radar support -> ground-ray point
  velocity     full LSQ when the cluster spans enough azimuth, else ONE radial measurement z = u.v with
               variance sigma_v(scan)^2 / n_eff (the tracker fuses it as a scalar)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from rcmot.types import Det2D, Det3D, RadarFrame
from rcmot.utils.calib import RigCalib

MAD_K = 1.4826


@dataclass
class FusionParams:
    shrink: float = 0.08              # box shrink (fraction) for point association
    mask_out_weight: float = 0.0      # instance mask (seg detector): weight of in-box points outside the mask
                                      # (0 = hard gate; GT/LiDAR: wrong range picks -30 % day, docs/14)
    dyn_static_weight: float = 1.0    # <1: when the box holds moving returns, static ones (background) are down-weighted
    dyn_min_weight: float = 0.6       # summed alpha of moving in-box returns needed to apply it
    motion_evidence: str = "track"    # point weight 1 - (1 - alpha_SNR)(1 - p_motion): "track" = only for returns
                                      # whose Doppler matches the box's track, "all", "off" (docs/14 §9)
    track_doppler: bool = True        # weight in-box points by the track's predicted radial velocity (docs/14 §9)
    track_doppler_min_speed: float = 0.3   # [m/s] only for moving tracks (a static track matches static clutter)
    track_doppler_sigma: float = 0.3  # [m/s] limb micro-Doppler / velocity lag allowance
    track_doppler_floor: float = 0.02 # minimum weight factor (never fully discard)
    radar_occluder: bool = True       # a static return well in front of the target, above ground, in the lower box
                                      # -> lower body hidden: no height prior / ground contact (docs/14 §9)
    occluder_gap: float = 2.0         # [m] in front of the reference range
    occluder_min_z: float = 0.2       # [m] above ground (a nearer ground return would project below the box)
    occluder_min_w: float = 0.3       # point weight needed
    camera_track_prior: bool = True   # camera-only boxes: combine the camera range with the box's track range
    occluded_ub_margin: float = 0.1   # occluded box: true range <= height-prior range * (1 + margin)
    occluded_prior_sigma: float = 1.0 # [m] added to the track range when it is the only range left
    track_height: bool = True         # height prior uses the learned height of the track in the box (docs/14 §8)
    ground_contact: bool = True       # box-bottom ground range as a camera range source (docs/14 §6)
    # segmentation uses (docs/14 §7), each switchable for ablation
    # ownership / feet: no gain on the day dev segments, and together they caused one ID transfer (docs/14 §7) -> off
    mask_ownership: bool = False      # shared in-box points go to the box whose mask they hit (front one if several)
    mask_feet: bool = False           # another instance's mask over this box's bottom band -> no ground contact
    feet_band: float = 0.15           # bottom fraction of the box checked for occlusion
    feet_occl: float = 0.05           # covered fraction above this -> feet hidden
    mask_posture: bool = False        # mask mass centre low + wide box -> no height prior (off: -0.036 IDF1 halt-mid)
    posture_cy: float = 0.5
    posture_aspect: float = 0.45
    reid_masked: bool = True          # ReID crop with the background filled (ImageNet mean): AUC 0.977 -> 0.994 at 5-30 s
    mask_dilate_px: int = 0           # tolerance for radar elevation / calibration error
    track_prior: bool = True          # range prior from the camera-anchored track already in the box (docs/14)
    track_prior_iou: float = 0.3
    track_prior_centre: bool = True   # or: the box centre inside the track's projected box (unique) (docs/16 §9)
    track_prior_cover: float = 0.8    # or: this fraction of the (occluded, shorter) box inside the track's full box
    track_prior_short: float = 0.85   # ... and the box shorter than this fraction of the track's projected height
    track_prior_max_tcam_s: float = 1.0
    track_prior_radar_anchor: bool = True   # prior range from the last radar-backed position (docs/16 §9)
    track_prior_max_tradar_s: float = 3.0
    track_prior_anchor_rate: float = 0.3    # [m/s] sigma growth of the anchored range with time since radar
    track_prior_sigma_floor: float = 0.3
    ground_gate_rel: float = 0.5      # |range - ground-ray range| <= rel * range + abs
    ground_gate_abs: float = 3.0
    sigma_range: float = 0.15         # radar range sigma [m] (KDE bandwidth floor)
    kde_halfwidth: float = 1.2        # points within +- this of the mode belong to the object [m]
    dop_z: float = 3.0                # Doppler outlier cut in scan-sigma units
    min_pts: int = 1                  # radar points needed for a radar-supported position
    min_weight: float = 0.6           # summed alpha needed
    bearing_sigma_deg: float = 0.3    # camera bearing sigma incl. box-centre jitter
    full_vel_min_pts: int = 3
    full_vel_min_spread_deg: float = 2.0
    sigma_v_default: float = 0.05
    max_range: float = 60.0           # beyond this a ground-ray-only det gets no 3D position
    max_ground_sigma: float = 8.0     # ... nor when its along-ray sigma exceeds this [m]
    max_tan_sigma: float = 0.5        # full velocity only if its tangential sigma is below this [m/s]
    max_speed: float = 10.0           # plausibility of a full velocity [m/s]
    velocity_mode: str = "median"     # "median": robust radial (weighted median Doppler) | "lsq": full 2D when observable
                                      # | "none": no radar velocity (camera / position only, for comparisons)
    class_height: dict = field(default_factory=lambda: {"worker": 1.8})   # known heights -> monocular range
    height_sigma: float = 0.12        # person height spread [m]
    box_px_sigma: float = 3.0         # box top / bottom pixel error
    crouch_aspect: float = 0.75       # w/h above this: not standing -> no height prior
    border_px: int = 4                # box touching the image top/bottom: truncated -> no height prior
    mono_gate: float = 3.0            # radar vs monocular range consistency [sigma]
    # docs/21 E6-4 (16a recommendation 4), both off by default:
    px_sigma_rel: float = 0.0         # > 0: box pixel errors grow with the box (UCMC: sigma_u = k w, sigma_v = k h, k ~ 0.05);
                                      #      the fixed box_px_sigma / bearing_sigma_deg stay as floors
    shake_gain: float = 0.0           # > 0: per-frame high-frequency image shake (px, Pipeline) added to the bearing
                                      #      sigma (horizontal) and, as pitch, to the ground-ray range sigma (vertical)
    rewind_image_yaw: float = 0.0     # docs/21 E2-c (1 / true = on, -1 = flipped sign for checking, 0 = off): fuse in the image's time (radar points rotated by the yaw swept
                                      # between image and scan, yaw_rate x img_dt), results rotated back to the scan
    # docs/30: cluster-level candidate scoring x 2-D track gate. The OLD path (KDE x camera prior x track prior on all
    # in-box points, box-centre bearing, ground-ray gate) = cluster_score "off", point_bearing false,
    # ground_gate_mono_fallback false (+ recovery.fuse_track_prior false).
    cluster_score: str = "lr"         # "lr": range clusters of the in-box points scored by the layout-descriptor logistic
                                      # regression (configs/fusion_cluster_lr.json), the best one gives the range | "off"
    cluster_score_model: str = "configs/fusion_cluster_lr.json"   # relative to the project root (or absolute)
    cluster_classes: list = field(default_factory=lambda: ["worker"])   # scorer trained on workers; other classes: KDE path
    cluster_link_m: float = 1.0       # single-linkage range gap between clusters [m]
    cluster_range: str = "kde"        # range inside the chosen cluster: "kde" (mode x prior, members within kde_halfwidth)
                                      # | "median" (cluster median, all members)
    cluster_score_no_track: str = "lr"  # box without a track prior: "lr" = best cluster by score | "kde" = old path
    cluster_min_score: float = 0.0    # clusters below this probability are not used (0 = off)
    cluster_gate_chi2: float = 9.21   # chi2_2 99 %: cluster centroid vs the track's predicted BEV position
    cluster_gate_sigma: float = 0.5   # [m] cluster position noise added to the track covariance in the gate
    cluster_gate_speed: dict = field(default_factory=lambda: {"worker": 2.5, "default": 8.0})   # [m/s] reach gate
    cluster_gate_margin: float = 1.0  # [m] |centroid - prediction| <= speed * dt + margin (0.5: 29 % of the static
                                      # conti_halt_mid boxes lost their true cluster to the gate; 1.0: 2 %)
    track_prior_ego_lag_s: float = 0.0   # [s] 2-D prior widened by the ego motion over this lag (|v| and yaw_rate x r
                                      # across the line of sight): image / radar / IMU timing on a moving base
    cluster_gate_mono_override: float = 0.1   # gate rejects every cluster but the best-scoring one agrees with the
                                      # height-prior range within this fraction: take it (the track prediction is the
                                      # suspect; a rejected true cluster otherwise drags the track further off, xpanner05)
    override_static_only: bool = True # ... only while the ego is static (moving ego: the box lags the scan and the
                                      # height prior agrees with the wrong cluster; conti_side_swing_a HOTA -0.07)
    ego_static_speed: float = 0.2     # [m/s] ego speed below this and ...
    ego_static_yaw_deg: float = 8.0   # [deg/s] yaw rate below this = static ego (Pipeline passes ego_static to fuse);
                                      # a swing is 20-60 deg/s, the xpanner bags carry a 4-6 deg/s gyro bias at rest
    cluster_gate_fail: str = "camera" # no cluster passes the gate: "camera" = camera / track range as when the radar
                                      # has nothing (mono+track) | "kde" = the old KDE x prior path on all points
    point_bearing: bool = False       # chosen cluster: bearing from the points' BEV centroid instead of the box-centre ray
                                      # (off: -0.05 HOTA on conti_side_swing_a, 1 FN on day_halt_short; range unchanged)
    point_bearing_sigma: float = 0.15 # [m] lateral sigma floor of a point-derived bearing (LiDAR: centroid within 0.15 m)
    ground_gate_mono_fallback: bool = True    # candidate gate centred on the height-prior range when the ground ray
                                      # disagrees with it by more than ground_gate_mono_ratio (xpanner level-frame height,
                                      # docs/30 §4); the ground-contact range is then not used for that box either
    ground_gate_mono_ratio: float = 2.0
    # docs/30 §6 (impl2): moving ego. Everything below leaves a static ego (Pipeline's ego_static) bit-identical
    # when the *_moving_only keys are true. "image time" = the stamp of the paired image, t_img = t_scan + img_dt.
    prior_img_time: bool = True       # 2-D prior / projected track box propagated with the ego motion to the IMAGE time
                                      # (yaw_rate x (dt + img_dt) rotation, (v_ego - v_track) x (dt + img_dt) translation)
                                      # instead of to the scan time only
    prior_sigma_w_deg: float = 1.2    # [deg/s] yaw-rate noise (10-sample gyro mean, design §4.1): the 2-D prior is widened
                                      # across the line of sight by (horizon x sigma_w x r)^2, horizon = since the last
                                      # position measurement until the image
    prior_sigma_v_ego: float = 0.1    # [m/s] ego-velocity uncertainty: along the line of sight by (horizon x sigma_v)^2
    prior_moving_only: bool = True    # the two items above only while the ego moves (static ego: old prior, bit-identical)
    box_time: bool = True             # fuse in the image's time: the radar points are rotated by the yaw swept between the
                                      # time the scan describes and the image (yaw_rate x (img_dt + calib offset - radar_lag_s)),
                                      # the fused position is rotated / translated back to the scan frame of the tracker
                                      # (generalises rewind_image_yaw: that key, when non-zero, is the sign multiplier)
    box_time_moving_only: bool = True
    radar_lag_s: dict = field(default_factory=lambda: {"bosch": -0.13, "conti": -0.067})
                                      # [s] the scan describes the state at header + lag, per radar type (Pipeline: cfg
                                      # radar_type or the radar topic). Only the DIFFERENCE to the calib's
                                      # radar_time_offset_s matters here: rf.stamp already carries the calib value
                                      # (bag_source), so the box-time angle is yaw_rate x (img_dt + calib - lag); an entry
                                      # "calib" (or a missing type) means the calib value wins, angle = yaw_rate x img_dt.
                                      # Conti: calib -0.067 = lag -0.067 -> identical (online scan on conti_swing_065: the
                                      # optimum is -0.067..-0.10). Bosch: online scan on the hdx swings (62 deg/s, impl2):
                                      # within 1 m 50 % without rotation, 47 % at the calib's -0.11, 74-76 % at -0.125..-0.14,
                                      # 17-21 % at exp2's cache value -0.03 (that value included the cache's own pairing);
                                      # the optimum is sharp (+-0.02 s = 30 pp).
    gate_turn_yaw_deg: float = 8.0    # [deg/s] |yaw_rate| above this = turning: the 2-D gate is widened across the line
                                      # of sight by sigma_az(r) x r instead of rejecting (design §4.3)
    gate_sigma_c_deg: float = 0.3     # sigma_az^2 = sigma_c^2 + sigma_radar^2 + (yaw_rate sigma_dt)^2 + (sigma_w horizon)^2
    gate_sigma_radar_deg: float = 1.0
    gate_sigma_dt_s: float = 0.015    # [s] residual image / radar timing uncertainty after the lag correction
    cluster_gate_safety_m: float = 1.0   # [m] a cluster whose range is within this of the OLD path's pick (KDE mode x camera
                                      # prior x track range prior over all in-box points) is never rejected by the gate
                                      # (it keeps at least the gate-edge weight); 0 = off
    cluster_gate_safety_moving_only: bool = True
    cluster_gate_safety_ref: str = "camera"   # reference pick of the net: "camera" = KDE x camera prior only | "old" = KDE x
                                      # camera prior x track range prior (exactly the old path's pick)
    cluster_gate_safety_strict: bool = False  # true: the net only acts when NO cluster passes the real gate (pure fallback);
                                      # false: a safe cluster keeps the gate-edge weight and may still win on its score
    cluster_gate_safety_sigma: bool = True    # a safety pick (cluster the real gate rejected) is reported with its range sigma
                                      # widened by |range - track range prior| (after the camera fusion; the range itself is
                                      # unchanged) so the tracker smooths the frame-to-frame alternation of such picks
                                      # instead of following it (impl2: conti drive IDsw 3 -> 25 without this)
    turn_box_window: bool = False     # while turning: in-box window widened by turn_box_margin_lat / _v (fractions of the
                                      # box size) instead of shrunk by `shrink` (exp2 lat10v5; off: no gain online, impl2)
    turn_box_margin_lat: float = 0.1
    turn_box_margin_v: float = 0.05
    # docs/33 wave 3, Track A (research/fusion_wave/trackA/report.md §5): soft instance-mask weights. "hard" = the
    # code above (mask_out_weight outside, cluster features unweighted). The other modes weight each in-box point by
    # a mask factor and compute the cluster features with it (count -> sum w, w-weighted u/v statistics, BEV centroid
    # and spread, |v_r|, weighted median range); points whose factor is below mask_min_weight leave the box.
    mask_weight: str = "hard"         # hard | const (mask_out_weight outside) | gauss_px | gauss_m
    mask_sigma_ang_deg: float = 0.3   # gauss_px: sigma_px = max(floor, fx sigma_ang) + fx |yaw_rate| sigma_dt, factor
    mask_sigma_floor_px: float = 1.5  # exp(-0.5 (d / sigma_px)^2), d = pixel distance outside the mask boundary
    mask_sigma_dt_s: float = 0.015
    mask_sigma_m: float = 0.2         # gauss_m: lateral metre offset d r / fx against this sigma
    mask_min_weight: float = 0.05
    mask_soft_min_p: float = 0.2      # soft modes: a cluster with no point ON the mask is used only when its cluster
                                      # probability is at least this, else its points leave the box as under hard
                                      # (diag_A 11-03: lone off-mask returns at p 0.005-0.08 birthed wrong-range tracks;
                                      # 0 = wave-3 behaviour)
    mask_hard_below_m: float = 15.0   # box range reference (track range prior, else height prior, else ground ray)
                                      # below this: the hard path above (Track A S4); no reference: soft
    # docs/33 wave 3, Track B (research/fusion_wave/trackB/report.md §10): metric depth map (Pipeline depth stage,
    # rcmot.fusion.depth_stage) per box: r_dmap with a fit-quality flag, offered only when the fit is good and
    # r_dmap <= depth_map.max_range (25 m)
    cluster_dmap: bool = False        # B-b: boxes with r_dmap are scored with cluster_score_model_dmap (the 11 features
                                      # + |ln(r_cluster / r_dmap)|); boxes without r_dmap keep cluster_score_model
    cluster_score_model_dmap: str = "configs/fusion_cluster_lr_dmap.json"
    dmap_fallback: bool = False       # B-a: box without a usable radar cluster, dmap_min_range <= r_dmap <= max: range
                                      # from the depth map x the per-track bias (track's radar / dmap range, last 10 s),
                                      # source "dmap", combined with the track range prior like the camera fallback
    dmap_min_range: float = 15.0      # [m] below this the camera fallback stays (Track B §5: 0-10 m camera 97 % vs
                                      # map 68-87 %, 10-15 m map 34-37 %)
    dmap_fallback_bias_only: bool = False   # B-a only for boxes whose track has a bias (radar history); others keep
                                      # the camera fallback (wave 3: the raw map is worse than the camera range there)
    # docs/38: range as a distribution (docs/37 model online). "legacy" = everything above. "lik" (person boxes of
    # cluster_classes): fuse() takes NO track input for them (range / BEV / height / velocity priors are dropped, as in
    # the docs/37 no-prior protocol the model was fitted on); the camera cues and every radar cluster of the box are
    # multiplied into a posterior over range (rcmot.fusion.range_lik); the output position is the dominant mode and
    # Det3D.rlik carries the whole likelihood for the tracker (tracker.range_lik multiplies its own prediction).
    range_model: str = "lik"          # lik (default since 2026-11-03, docs/38) | legacy (the point-range fusion with the
                                      # track-range feedback paths, docs/35 §6-§9)
    range_lik_params: str = "configs/range_lik/full.json"   # docs/37 fit (evaluation: configs/range_lik/loso/<seg>.json)
    lik_radar_backed: float = 3.0     # radar likelihood ratio at the dominant mode that makes a box "radar-backed"
    lik_bg_slope: float = -4.14       # docs/41 §10: a radar cluster's score is multiplied by clip(exp(slope * mean
    lik_bg_fmin: float = 0.05         # background score of its returns), fmin, 1) when the cell memory gives scores
    lik_wedge_mode: str = "off"       # docs/38 §13.6 range-only BEV wedge evidence: "off" | "fallback" (boxes without an
                                      # in-box cluster) | "prod" (every box, lik_wedge_beta x wedge term); unlike
                                      # lik_wedge_fallback it never makes a box radar-backed nor gives it a velocity
    lik_wedge_beta: float = 0.5       # "prod": power of the wedge likelihood
    lik_wedge_mirror: bool = True     # below-ground (mirror) returns are wedge cluster members (else only counted)
    lik_wedge_fallback: bool = False  # docs/41: a box without any in-box radar cluster takes the range clusters of the
                                      # returns inside its BEV azimuth wedge (box left / right rays +- margin, any
                                      # height), scored by the wedge LR of the range_lik parameters ("wedge" block)


def turn_az_sigma(p: FusionParams, yaw_rate: float, horizon: float) -> float:
    """Azimuth sigma [rad] of a box / radar association on a turning base (design §4.3): camera bearing, radar
    azimuth, residual timing x yaw rate, gyro noise x prediction horizon."""
    return float(np.sqrt(np.radians(p.gate_sigma_c_deg) ** 2 + np.radians(p.gate_sigma_radar_deg) ** 2
                         + (yaw_rate * p.gate_sigma_dt_s) ** 2 + (np.radians(p.prior_sigma_w_deg) * horizon) ** 2))


def _ray_basis(d: np.ndarray):
    """Orthonormal basis (along ray, horizontal-lateral, up-ish)."""
    a = d / np.linalg.norm(d)
    lat = np.cross([0.0, 0.0, 1.0], a)
    lat = lat / max(np.linalg.norm(lat), 1e-9)
    up = np.cross(a, lat)
    return np.stack([a, lat, up], 1)


def _ground_sigma(gr: float) -> float:
    """Along-ray sigma of a ground-ray range (box-bottom pixel error, ground slope) ~ r^2."""
    return max(0.5, 0.05 * gr + 0.004 * gr ** 2)


def mono_range(box, H: float, calib: RigCalib, p) -> tuple[float, float] | None:
    """Horizontal range from the camera to an upright object of known height H spanning the box:
    rho = H / (tan(elev_top) - tan(elev_bottom)) (exact for any camera pitch); sigma from pixel / height errors."""
    x1, y1, x2, y2 = box
    h_px = y2 - y1
    if h_px < 12 or (x2 - x1) / h_px > p.crouch_aspect or y1 <= p.border_px or y2 >= calib.image_size[1] - p.border_px:
        return None
    uc = 0.5 * (x1 + x2)
    dt, db = calib.pixel_ray(uc, y1), calib.pixel_ray(uc, y2)
    dtan = dt[2] / np.linalg.norm(dt[:2]) - db[2] / np.linalg.norm(db[:2])
    if dtan <= 1e-4:
        return None
    rho = H / dtan
    px = max(p.box_px_sigma, p.px_sigma_rel * h_px)
    sig = rho * np.hypot(np.sqrt(2) * px / h_px, p.height_sigma / H)
    return float(rho), float(sig)


def bearing_sigma_rad(p, box, calib: RigCalib, shake_px: tuple | None = None) -> float:
    """Camera bearing sigma [rad]: the fixed floor, box-size term (px_sigma_rel * width) and image shake."""
    fx = calib.K[0, 0]
    s = np.radians(p.bearing_sigma_deg)
    if p.px_sigma_rel > 0:
        s = max(s, p.px_sigma_rel * (box[2] - box[0]) / fx)
    if p.shake_gain > 0 and shake_px is not None:
        s = float(np.hypot(s, p.shake_gain * shake_px[0] / fx))
    return s


def pitch_shake_range_sigma(p, r: float, calib: RigCalib, shake_px: tuple | None = None) -> float:
    """Extra along-ray sigma of a ground-contact range from vertical image shake: dr = r^2 / h_cam * d(pitch)."""
    if p.shake_gain <= 0 or shake_px is None:
        return 0.0
    h_cam = max(float(calib.camera_center()[2]), 0.5)
    return r * r / h_cam * p.shake_gain * shake_px[1] / calib.K[1, 1]


def _kde_mode(r: np.ndarray, w: np.ndarray, bw: float, prior: tuple | None = None) -> float:
    """Weighted kernel-density mode; with prior (mean, sigma) from the camera ground ray, the density is
    multiplied by that Gaussian so the box's own object wins over a denser background behind it."""
    grid = np.unique(r)
    dens = (w[None, :] * np.exp(-0.5 * ((grid[:, None] - r[None, :]) / bw) ** 2)).sum(1)
    if prior is not None:
        dens = dens * np.exp(-0.5 * ((grid - prior[0]) / prior[1]) ** 2)
    return float(grid[np.argmax(dens)])


_CLUSTER_MODELS: dict = {}


def load_cluster_model(path: str) -> dict:
    """Cluster-score logistic regression (configs/fusion_cluster_lr.json): feature names, standardisation, weights."""
    if path not in _CLUSTER_MODELS:
        import json
        from pathlib import Path

        pth = Path(path)
        if not pth.is_absolute():
            pth = Path(__file__).resolve().parents[3] / pth
        m = json.load(open(pth))
        _CLUSTER_MODELS[path] = {"features": list(m["features"]), "mean": np.asarray(m["mean"], float),
                                 "std": np.asarray(m["std"], float), "coef": np.asarray(m["coef"], float),
                                 "intercept": float(m["intercept"]), "height_m": float(m.get("height_m", 1.7))}
    return _CLUSTER_MODELS[path]


def mask_factors(mask: np.ndarray, u: np.ndarray, v: np.ndarray, x0: int, y0: int, mode: str, sigma_px: float,
                 out_weight: float = 0.0, sigma_m: float = 0.2, rng: np.ndarray | None = None, fx: float = 1.0) -> np.ndarray:
    """Per-point instance-mask factor (docs/33 Track A) of image points (u, v) against a box-cropped mask whose top-left
    pixel is (x0, y0): 1 on the mask; outside, by `mode`: "const" out_weight, "gauss_px" exp(-0.5 (d / sigma_px)^2),
    "gauss_m" exp(-0.5 (d rng / fx / sigma_m)^2), with d = Euclidean pixel distance to the nearest mask pixel - 0.5
    (points beyond the crop: + their distance to the crop)."""
    m = mask > 0
    H, W = m.shape
    uu = u.astype(int) - x0
    vv = v.astype(int) - y0
    uc, vc = np.clip(uu, 0, W - 1), np.clip(vv, 0, H - 1)
    extra = np.hypot(uu - uc, vv - vc)
    on = (extra == 0) & m[vc, uc]
    f = np.ones(len(u))
    off = ~on
    if not off.any():
        return f
    if mode == "const":
        f[off] = out_weight
        return f
    D = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)   # 0 on the mask
    d = np.maximum(D[vc[off], uc[off]] + extra[off] - 0.5, 0.0)
    if mode == "gauss_m":
        d = d * np.asarray(rng, float)[off] / fx / max(sigma_m, 1e-6)
    else:
        d = d / max(sigma_px, 1e-6)
    f[off] = np.exp(-0.5 * d ** 2)
    return f


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    """Weighted median (= np.median for equal weights)."""
    if len(x) == 0:
        return float("nan")
    if np.ptp(w) == 0:
        return float(np.median(x))
    o = np.argsort(x)
    xs, cw = x[o], np.cumsum(w[o])
    half = 0.5 * cw[-1]
    i = int(np.searchsorted(cw, half))
    if i < len(xs) - 1 and abs(cw[i] - half) <= 1e-12 * cw[-1]:
        return float(0.5 * (xs[i] + xs[i + 1]))
    return float(xs[min(i, len(xs) - 1)])


def range_clusters(r: np.ndarray, link: float) -> np.ndarray:
    """1-D single-linkage clustering on range: label (0..k) per point, nearest cluster first."""
    o = np.argsort(r)
    lab = np.zeros(len(r), int)
    if len(r) > 1:
        lab[o[1:]] = np.cumsum(np.diff(r[o]) > link)
    return lab


def cluster_candidates(idx: np.ndarray, xyz: np.ndarray, uv: np.ndarray, v_r: np.ndarray | None, box, fx: float,
                       model: dict, link: float, w: np.ndarray | None = None, r_dmap: float | None = None,
                       feats_out: list | None = None) -> list:
    """Range clusters of a box's radar points with the layout-descriptor probability (docs/30 §3): per cluster
    (member indices, median range from the level origin, BEV centroid xy, p). Features as in the experiment:
    log1p(n), normalised image position inside the box (u_n, v_n: mean / std / |u_n - 0.5|), BEV spread relative to
    the box width at that range, ln(r / r_h) with r_h = fx * H / box_h_px (and its magnitude), mean |raw v_r|, r.
    w (per idx point, docs/33 Track A): soft instance-mask weights -> n becomes sum(w) and every statistic is
    w-weighted (median range = weighted median); None = unweighted (the original features).
    r_dmap (docs/33 Track B, B-b): depth-map range of the box (level origin) -> features logdm = ln(r / r_dmap) and
    abs_logdm for a model that uses them.
    feats_out (evaluation dumps, docs/37): when a list, the feature dict of every cluster is appended to it."""
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
    xy = xyz[idx, :2]
    r = np.hypot(xy[:, 0], xy[:, 1])
    un = (uv[idx, 0] - x1) / bw
    vn = (uv[idx, 1] - y1) / bh
    lab = range_clusters(r, link)
    r_h = fx * model["height_m"] / bh
    feats, mean, std, coef = model["features"], model["mean"], model["std"], model["coef"]
    out = []
    for L in range(int(lab.max()) + 1):
        m = lab == L
        if w is None:
            rk = float(np.median(r[m]))
            um = float(un[m].mean())
            f = {"logn": float(np.log1p(m.sum())), "u_mean": um, "u_std": float(un[m].std()), "u_dev": abs(um - 0.5),
                 "v_mean": float(vn[m].mean()), "v_std": float(vn[m].std()),
                 "spread_rel": float(np.sqrt(xy[m, 0].var() + xy[m, 1].var())) / (rk * bw / fx),
                 "rv_abs": float(np.abs(v_r[idx][m]).mean()) if v_r is not None else None}
            cxy = xy[m].mean(0)
        else:
            wm = np.asarray(w, float)[m]
            sw = max(float(wm.sum()), 1e-9)
            rk = weighted_median(r[m], wm)
            um, vm = float(wm @ un[m]) / sw, float(wm @ vn[m]) / sw
            cxy = (wm[:, None] * xy[m]).sum(0) / sw
            var = float(wm @ ((xy[m] - cxy) ** 2).sum(1)) / sw
            f = {"logn": float(np.log1p(sw)), "u_mean": um, "u_std": float(np.sqrt(wm @ (un[m] - um) ** 2 / sw)),
                 "u_dev": abs(um - 0.5), "v_mean": vm, "v_std": float(np.sqrt(wm @ (vn[m] - vm) ** 2 / sw)),
                 "spread_rel": float(np.sqrt(var)) / (rk * bw / fx),
                 "rv_abs": float(wm @ np.abs(v_r[idx][m])) / sw if v_r is not None else None}
        logrh = float(np.log(rk / r_h))
        f.update(logrh=logrh, abs_logrh=abs(logrh), r=rk)
        if r_dmap is not None and r_dmap > 0:
            ld = float(np.log(rk / r_dmap))
            f.update(logdm=ld, abs_logdm=abs(ld))
        x = np.array([f[k] if f.get(k) is not None else mean[i] for i, k in enumerate(feats)])
        z = float(((x - mean) / std) @ coef + model["intercept"])
        out.append((idx[m], rk, cxy, 1.0 / (1.0 + np.exp(-z))))
        if feats_out is not None:
            feats_out.append(f)
    return out


def gate_clusters(cands: list, prior: tuple, chi2: float, sigma: float, vmax: float, margin: float,
                  lat_sigma: float = 0.0, origin: np.ndarray | None = None) -> np.ndarray:
    """2-D track gate (docs/30 §3.4): prior = (predicted BEV xy, 2x2 covariance, seconds since the last position
    measurement). Per cluster exp(-d_M^2 / 2) when the centroid passes the chi2 gate on (P + sigma^2 I) and the
    class reach |d| <= vmax * dt + margin, else 0.
    lat_sigma > 0 (turning base, impl2): the covariance is widened across the line of sight from `origin` (camera
    centre) by lat_sigma^2 and the reach test ignores the first 3 lat_sigma of lateral offset."""
    mu, P, dt = prior
    S = np.asarray(P, float) + np.eye(2) * sigma ** 2
    u = perp = None
    if lat_sigma > 0:
        d0 = mu - (np.zeros(2) if origin is None else np.asarray(origin, float)[:2])
        u = d0 / max(float(np.hypot(d0[0], d0[1])), 1e-6)
        perp = np.array([-u[1], u[0]])
        S = S + np.outer(perp, perp) * lat_sigma ** 2
    Si = np.linalg.inv(S)
    reach = vmax * max(float(dt), 0.066) + margin
    w = np.zeros(len(cands))
    for i, (_, _, c, _) in enumerate(cands):
        d = c - mu
        d2 = float(d @ Si @ d)
        dist = np.hypot(d[0], d[1]) if perp is None else np.hypot(d @ u, max(abs(d @ perp) - 3.0 * lat_sigma, 0.0))
        if d2 <= chi2 and dist <= reach:
            w[i] = np.exp(-0.5 * d2)
    return w


class RangeReliability:
    """Online reliability of the two camera-only ranges (docs/14 §6), learned from boxes the radar confirms:
    height prior (posture-dependent: crouching / bending workers look farther) and box-bottom ground contact
    (pitch / terrain-dependent: worse while driving). Keeps a window of relative residuals against the radar
    range (robust sigma = 1.4826 MAD) and, for boxes where the two disagree, which one the radar sided with.

    Bias (docs/14 §8): the ground contact carries a range-proportional bias rel = k * r (camera pitch or terrain
    slope error grows ~ r^2 in absolute terms; +6..+11 % at 15-20 m on halt-mid, ~0 at 7 m), so k is fitted
    robustly (median of rel / r) and the ground range is corrected before use; its sigma is the spread of the
    corrected residuals. The height prior's sigma is a robust RMS around zero (bias included), because a
    person-mix bias is not removable globally (per-track heights handle it, see Pipeline)."""

    def __init__(self, window: int = 150, min_n: int = 20, prior_rel: dict | None = None, floor_rel: float = 0.03):
        from collections import deque

        self.res = {"mono": deque(maxlen=window), "ground": deque(maxlen=window)}
        self.ground_r = deque(maxlen=window)             # reference ranges of the ground residuals
        self.wins = deque(maxlen=window // 3)            # 1 = ground right in a mono/ground conflict, 0 = mono
        self.min_n, self.floor = min_n, floor_rel
        self.max_ground_rel = 0.15                       # learned ground-contact sigma above this: not used
        self.prior = prior_rel or {"mono": 0.10, "ground": 0.08}

    def ground_k(self) -> float:
        """Range-proportional ground-contact bias k [1/m] (rel = k * r); 0 until enough samples."""
        if len(self.res["ground"]) < self.min_n:
            return 0.0
        return float(np.median(np.asarray(self.res["ground"]) / np.asarray(self.ground_r)))

    def correct_ground(self, g: float) -> float:
        """Invert g = r (1 + k r) for the true range r."""
        k = self.ground_k()
        if abs(k) < 1e-6:
            return g
        disc = 1.0 + 4.0 * k * g
        return float((-1.0 + np.sqrt(disc)) / (2.0 * k)) if disc > 0 else g

    def sigma_rel(self, name: str) -> float:
        r = self.res[name]
        if len(r) < self.min_n:
            return self.prior[name]
        a = np.asarray(r)
        if name == "ground":                             # spread after removing the fitted bias
            a = a - self.ground_k() * np.asarray(self.ground_r)
            return max(MAD_K * float(np.median(np.abs(a - np.median(a)))), self.floor)
        return max(MAD_K * float(np.median(np.abs(a))), self.floor)       # robust RMS: bias counts as error

    def ground_win_rate(self) -> float:
        return float(np.mean(self.wins)) if len(self.wins) >= 5 else 0.5

    def observe(self, ref: float, mono: float | None, ground: float | None, conflict: bool) -> None:
        if mono is not None:
            self.res["mono"].append((mono - ref) / ref)
        if ground is not None:
            self.res["ground"].append((ground - ref) / ref)
            self.ground_r.append(ref)
        if conflict and mono is not None and ground is not None:
            self.wins.append(float(abs(ground - ref) < abs(mono - ref)))


def camera_range(mono: tuple | None, ground: float | None, rel: "RangeReliability | None",
                 gate: float = 3.0) -> tuple[tuple | None, str, bool]:
    """((range, sigma) or None, source, conflict) from the height-prior and ground-contact ranges."""
    rel = rel or RangeReliability()
    if ground is not None:
        ground = rel.correct_ground(ground)
    if ground is not None and rel.sigma_rel("ground") > rel.max_ground_rel:
        ground = None                                    # e.g. pitching while driving: ground contact unusable
    m = None if mono is None else (mono[0], max(mono[1], rel.sigma_rel("mono") * mono[0]))
    g = None if ground is None else (ground, max(0.3, rel.sigma_rel("ground") * ground))
    if m is None or g is None:
        return (m, "mono", False) if m is not None else ((g, "ground_ray", False) if g is not None else (None, "", False))
    if abs(m[0] - g[0]) <= gate * np.hypot(m[1], g[1]):
        return _combine(m, g), "mono", False
    # conflict (posture or pitch): side with the source the radar has backed in recent conflicts, and keep
    # the discrepancy in the sigma so the tracker's gate stays permissive along the ray
    pick, src = (g, "ground_ray") if rel.ground_win_rate() >= 0.5 else (m, "mono")
    return (pick[0], float(np.hypot(pick[1], 0.25 * abs(m[0] - g[0])))), src, True


def _mask_full(d: Det2D, shape: tuple) -> np.ndarray | None:
    """Box-cropped instance mask placed in a full-image boolean array."""
    if d.mask is None:
        return None
    full = np.zeros(shape, bool)
    x1, y1 = max(int(d.box[0]), 0), max(int(d.box[1]), 0)
    m = d.mask[:shape[0] - y1, :shape[1] - x1]
    full[y1:y1 + m.shape[0], x1:x1 + m.shape[1]] = m
    return full


def mask_hits(full: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Which projected points fall on the mask (full-image boolean array)."""
    u = np.clip(uv[:, 0].astype(int), 0, full.shape[1] - 1)
    v = np.clip(uv[:, 1].astype(int), 0, full.shape[0] - 1)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < full.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < full.shape[0])
    return inside & full[v, u]


def bottom_occlusion(box: np.ndarray, own: np.ndarray, others: list, band: float) -> float:
    """Fraction of the box's bottom band covered by other instances' masks (feet hidden behind someone)."""
    x1, y1, x2, y2 = [int(round(c)) for c in box]
    H, W = own.shape
    yb, x1, x2, y2 = max(int(y2 - band * (y2 - y1)), 0), max(x1, 0), min(x2, W), min(y2, H)
    if x2 <= x1 or y2 <= yb or not others:
        return 0.0
    cover = np.zeros((y2 - yb, x2 - x1), bool)
    for o in others:
        cover |= o[yb:y2, x1:x2]
    return float((cover & ~own[yb:y2, x1:x2]).mean())


def mask_mass_centre(mask: np.ndarray) -> float:
    """Height of the mask's mass centre as a fraction of the box height from the top (upright person ~0.45)."""
    ys = np.nonzero(mask)[0]
    return float(ys.mean() / max(mask.shape[0], 1)) if len(ys) else 0.5


def occluder_points(box, uv, front, rng, z, w, ref: float, p) -> np.ndarray:
    """Radar returns in the lower half of the box, clearly in front of the box's object (ref range) and above the
    ground: something standing between the camera and the object's legs (docs/14 §9). A ground return nearer than
    the feet would project below the box, so these are objects."""
    x1, y1, x2, y2 = box
    m = front & (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= 0.5 * (y1 + y2)) & (uv[:, 1] <= y2)
    return m & (rng < ref - p.occluder_gap) & (z > p.occluder_min_z) & (w >= p.occluder_min_w)


def occluded_bottom(box, uv, front, rng, z, w, ref: float, p) -> bool:
    return bool(occluder_points(box, uv, front, rng, z, w, ref, p).any())


def _combine(a: tuple | None, b: tuple | None) -> tuple | None:
    """Inverse-variance product of two Gaussian range priors (mean, sigma)."""
    if a is None or b is None:
        return a or b
    wa, wb = a[1] ** -2, b[1] ** -2
    return ((wa * a[0] + wb * b[0]) / (wa + wb), (wa + wb) ** -0.5)


def fuse(dets: list[Det2D], radar: RadarFrame, calib: RigCalib, *, image: np.ndarray | None = None,
         params: FusionParams | None = None, range_priors: list | None = None,
         reliability: RangeReliability | None = None, heights: list | None = None,
         vel_priors: list | None = None, shake_px: tuple | None = None,
         xy_priors: list | None = None, ego_static: bool | None = None, ego_yaw_rate: float = 0.0,
         dmap: list | None = None, cue_log: list | None = None, ego_speed: float = 0.0) -> list[Det3D]:
    """range_priors: per detection (range, sigma) of the track already seen in this box (docs/14), or None.
    It joins the monocular prior in the range-mode search, so the box's own object wins over the points of
    whatever stands behind (or in front of) it.
    xy_priors: per detection (predicted BEV xy, 2x2 covariance, seconds since the track's last position
    measurement) of that track, or None; used by the cluster-score path (docs/30) as a 2-D gate.
    ego_static: whether the base is at rest this frame (None = unknown, treated as static).
    ego_yaw_rate: base yaw rate [rad/s] this frame; above gate_turn_yaw_deg the 2-D gate is widened across the
    line of sight (and the in-box window with turn_box_window). The radar points are expected in the frame of the
    box (image time): the caller (Pipeline) rotates them when box_time is on.
    dmap (docs/33 wave 3): per detection the depth stage's box record (rcmot.fusion.depth_stage.DepthStage.boxes:
    r = range from the camera, r_org = from the level origin, sig, ok = fit quality and r <= max range, bias = the box's
    track's radar / dmap ratio or None, sig_rel) or None; used by cluster_dmap (B-b) and dmap_fallback (B-a).
    cue_log (evaluation dumps, docs/37 range uncertainty): when a list, one dict per detection (also those that get no
    3D output) with every range cue fuse() saw -- box geometry, height-prior / ground-contact / width terms, the
    RangeReliability state, the track priors, the candidate radar points and clusters (LR features, gate weights), the
    chosen range and the output covariance. Read-only: the detections are identical with or without it.
    ego_speed: |base velocity| [m/s] (range_model "lik": ground-contact scale term).
    range_model "lik" (docs/38): person boxes get the range distribution of docs/37 and no track input (see FusionParams)."""
    p = params or FusionParams()
    lik = None
    if p.range_model == "lik":
        from rcmot.fusion.range_lik import load_range_lik
        lik = load_range_lik(p.range_lik_params)
        use = [d.cls in p.cluster_classes for d in dets]

        def _strip(L):
            return None if L is None else [None if u else x for x, u in zip(L, use)]
        range_priors, xy_priors, heights, vel_priors = map(_strip, (range_priors, xy_priors, heights, vel_priors))
    model = load_cluster_model(p.cluster_score_model) if p.cluster_score == "lr" else None
    model_dm = load_cluster_model(p.cluster_score_model_dmap) if (model is not None and p.cluster_dmap and dmap is not None) else None
    turning = abs(float(ego_yaw_rate)) > np.radians(p.gate_turn_yaw_deg)
    fx = float(calib.K[0, 0])
    st = radar.stats
    n = len(radar.v_r)
    w_pt = np.ones(n) if st is None else st.alpha * st.keep
    if st is not None and p.motion_evidence == "all" and getattr(st, "p_dyn", None) is not None:
        # a statistically significant Doppler (|z_v| above this scan's noise) is evidence of a real return on
        # its own: weak-SNR returns of a walking person must not be discarded (docs/14 §9)
        w_pt = (1.0 - (1.0 - st.alpha) * (1.0 - st.p_dyn)) * st.keep
    v_comp = radar.v_r_comp if radar.v_r_comp is not None else (st.v_comp if st is not None else None)
    sigma_v = st.sigma_v if st is not None and np.isfinite(st.sigma_v) else p.sigma_v_default
    uv, front = calib.project(radar.xyz) if n else (np.zeros((0, 2)), np.zeros(0, bool))
    C = calib.camera_center()
    rng_c = np.linalg.norm(radar.xyz[:, :2] - C[:2], axis=1) if n else np.zeros(0)   # horizontal range from camera

    # ground-ray range per box, candidate sets, then exclusive assignment of shared points
    g_pts, g_rng, gate_rng, ground_bad, cand = [], [], [], [], []
    for d in dets:
        x1, y1, x2, y2 = d.box
        bw, bh = x2 - x1, y2 - y1
        g = calib.pixel_ray_to_ground(0.5 * (x1 + x2), y2)
        gr = float(np.linalg.norm(g[:2] - C[:2])) if g is not None else None
        gate_r, bad = gr, False
        if p.ground_gate_mono_fallback and gr is not None:
            # docs/30 §4: a ground ray that contradicts the height-prior range by a factor (level-frame camera
            # height wrong, xpanner) would reject every true return; gate on the height prior instead
            H0 = p.class_height.get(d.cls)
            m0 = mono_range(d.box, H0, calib, p) if H0 is not None else None
            if m0 is not None and max(gr / m0[0], m0[0] / gr) > p.ground_gate_mono_ratio:
                gate_r, bad = m0[0], True
        if turning and p.turn_box_window:                  # impl2: widened window while turning (exp2 lat10v5)
            ml, top, bot = -p.turn_box_margin_lat, y1 - p.turn_box_margin_v * bh, y2 + p.turn_box_margin_v * bh
        else:
            ml, top, bot = p.shrink, y1 + p.shrink * bh, y2
        inside = front & (uv[:, 0] >= x1 + ml * bw) & (uv[:, 0] <= x2 - ml * bw) \
            & (uv[:, 1] >= top) & (uv[:, 1] <= bot) & (w_pt > 0)
        idx = np.flatnonzero(inside)
        if gate_r is not None and len(idx):
            idx = idx[np.abs(rng_c[idx] - gate_r) <= p.ground_gate_rel * gate_r + p.ground_gate_abs]
        g_pts.append(g)
        g_rng.append(gr)
        gate_rng.append(gate_r)
        ground_bad.append(bad)
        cand.append(idx)
    shape = (calib.image_size[1], calib.image_size[0])
    full = [_mask_full(d, shape) if (p.mask_ownership or p.mask_feet) else None for d in dets]
    hits = {}
    if p.mask_ownership:
        for k, idx in enumerate(cand):
            if full[k] is not None and len(idx):
                hits[k] = set(idx[mask_hits(full[k], uv[idx])].tolist())
    owner = {}
    for k, idx in enumerate(cand):                                     # default: closest to the box's ground range
        for i in idx:
            err = abs(rng_c[i] - gate_rng[k]) if gate_rng[k] is not None else 0.0
            if i not in owner or err < owner[i][1]:
                owner[i] = (k, err)
    for i in owner:                                                    # on a mask beats off-mask; front mask first
        on = [k for k in hits if i in hits[k]]
        if on:
            owner[i] = (min(on, key=lambda k: g_rng[k] if g_rng[k] is not None else np.inf), 0.0)
    cand = [np.array([i for i in idx if owner[i][0] == k], int) for k, idx in enumerate(cand)]

    out = []
    for k, d in enumerate(dets):
        x1, y1, x2, y2 = d.box
        idx = cand[k]
        det = None
        wk = w_pt
        dm_k = dmap[k] if dmap is not None else None
        _cq = {} if cue_log is not None else None                      # docs/37 cue log of this box (dumps only)
        mw = None                                                      # docs/33 Track A: per-point mask factors
        soft = p.mask_weight != "hard" and d.mask is not None and len(idx) > 0 and bool(d.mask.any())
        idx0 = idx.copy()
        if _cq is not None:
            _cq["idx0"] = idx0
        if soft and p.mask_hard_below_m > 0:                           # S4: near boxes keep the hard path
            if range_priors is not None and range_priors[k] is not None:
                r_ref = range_priors[k][0]
            else:
                H0 = heights[k] if heights is not None and heights[k] is not None else p.class_height.get(d.cls)
                m0 = mono_range(d.box, H0, calib, p) if H0 is not None else None
                r_ref = m0[0] if m0 is not None else g_rng[k]
            soft = r_ref is None or r_ref >= p.mask_hard_below_m
        if soft:
            m = d.mask.astype(np.uint8)
            if p.mask_dilate_px > 0:
                m = cv2.dilate(m, np.ones((2 * p.mask_dilate_px + 1,) * 2, np.uint8))
            sig_px = max(p.mask_sigma_floor_px, fx * np.radians(p.mask_sigma_ang_deg)) \
                + fx * abs(float(ego_yaw_rate)) * p.mask_sigma_dt_s
            fac = mask_factors(m, uv[idx, 0], uv[idx, 1], max(int(x1), 0), max(int(y1), 0), p.mask_weight, sig_px,
                               p.mask_out_weight, p.mask_sigma_m, rng_c[idx], fx)
            wk = w_pt.copy()
            wk[idx] *= fac
            keep = fac >= p.mask_min_weight
            idx = idx[keep]
            mw = np.zeros(n)
            mw[idx] = fac[keep]
        elif d.mask is not None and len(idx) and p.mask_out_weight < 1.0:   # soft instance-mask gating
            m = d.mask.astype(np.uint8)
            if p.mask_dilate_px > 0:
                m = cv2.dilate(m, np.ones((2 * p.mask_dilate_px + 1,) * 2, np.uint8))
            u = np.clip(uv[idx, 0].astype(int) - max(int(x1), 0), 0, m.shape[1] - 1)
            v = np.clip(uv[idx, 1].astype(int) - max(int(y1), 0), 0, m.shape[0] - 1)
            wk = w_pt.copy()
            wk[idx] *= np.where(m[v, u] > 0, 1.0, p.mask_out_weight)
        dyn = getattr(st, "dynamic", None) if st is not None else None
        if dyn is not None and len(idx) and p.dyn_static_weight < 1.0:   # Doppler: the box's mover vs static background
            moving = dyn[idx]
            if (wk[idx] * moving).sum() >= p.dyn_min_weight:
                wk = wk.copy() if wk is w_pt else wk
                wk[idx] *= np.where(moving, 1.0, p.dyn_static_weight)
        has_track = range_priors is not None and range_priors[k] is not None
        if p.motion_evidence == "track" and not has_track and getattr(st, "p_dyn", None) is not None and len(idx):
            wk = wk.copy() if wk is w_pt else wk                       # a new object: any significant mover counts
            wk[idx] = 1.0 - (1.0 - wk[idx]) * (1.0 - st.p_dyn[idx] * (wk[idx] > 0))
        if p.track_doppler and vel_priors is not None and vel_priors[k] is not None and v_comp is not None and len(idx):
            v_t, Pv = vel_priors[k]                                     # predicted absolute velocity of the box's track
            if np.linalg.norm(v_t[:2]) >= p.track_doppler_min_speed:
                u3 = radar.xyz_radar[idx] / np.linalg.norm(radar.xyz_radar[idx], axis=1, keepdims=True) @ calib.R_lr.T
                vr_exp = u3 @ v_t
                var = sigma_v ** 2 + p.track_doppler_sigma ** 2 + np.einsum("ij,jk,ik->i", u3[:, :2], Pv[:2, :2], u3[:, :2])
                match = np.exp(-0.5 * (v_comp[idx] - vr_exp) ** 2 / var)
                wk = wk.copy() if wk is w_pt else wk
                if p.motion_evidence == "track" and getattr(st, "p_dyn", None) is not None:
                    pm = st.p_dyn[idx] * match                         # moving AND moving like this track
                    wk[idx] = 1.0 - (1.0 - wk[idx]) * (1.0 - pm * (wk[idx] > 0))
                wk[idx] *= np.maximum(match, p.track_doppler_floor)   # the mover's own returns, not the static scene
        if wk is not w_pt:
            idx = idx[wk[idx] > 0]                                    # hard-gated points leave the box entirely
        H = p.class_height.get(d.cls)
        if H is not None and heights is not None and heights[k] is not None:
            H = heights[k]                                             # learned height of the track in this box
        mono = mono_range(d.box, H, calib, p) if H is not None else None
        if p.mask_posture and mono is not None and d.mask is not None and \
                mask_mass_centre(d.mask) > p.posture_cy and (x2 - x1) / max(y2 - y1, 1) > p.posture_aspect:
            mono = None                                                # bent / crouching: height prior invalid
        if _cq is not None:
            _cq.update(H=H, mono0=mono)
        bottom_ok = y2 < calib.image_size[1] - p.border_px                 # feet visible -> ground contact usable
        if p.mask_feet and bottom_ok and full[k] is not None:
            others = [f for j, f in enumerate(full) if j != k and f is not None]
            bottom_ok = bottom_occlusion(d.box, full[k], others, p.feet_band) <= p.feet_occl
        occluded, r_ub = False, None
        if p.radar_occluder and n:                                     # radar sees what hides the lower body
            ref = range_priors[k][0] if range_priors is not None and range_priors[k] is not None else \
                (mono[0] if mono is not None else g_rng[k])
            occ = occluder_points(d.box, uv, front, rng_c, radar.xyz[:, 2], w_pt, ref, p) if ref is not None else None
            if occ is not None and occ.any():
                occluded = True
                # the visible box is shorter than the body: the height-prior range is an UPPER bound
                r_ub = mono[0] * (1.0 + p.occluded_ub_margin) if mono is not None else None
                mono, bottom_ok = None, False                          # box height and box bottom both cut
                idx = idx[~occ[idx]]                                   # the occluder is not the object
                if r_ub is not None:
                    idx = idx[rng_c[idx] <= r_ub]
        ground = g_rng[k] if (p.ground_contact and g_rng[k] is not None and bottom_ok and g_rng[k] <= p.max_range
                              and not ground_bad[k]) else None
        if _cq is not None:
            _rl = reliability or RangeReliability()
            _cq.update(rel={"sm": _rl.sigma_rel("mono"), "sg": _rl.sigma_rel("ground"), "k": _rl.ground_k(),
                            "win": _rl.ground_win_rate(), "nm": len(_rl.res["mono"]), "ng": len(_rl.res["ground"])},
                       g_corr=None if ground is None else _rl.correct_ground(ground))
        cam, cam_src, conflict = camera_range(mono, ground, reliability)
        cam_only = None                                                # range used when the radar has nothing
        if cam is None and range_priors is not None and range_priors[k] is not None and mono is None and not bottom_ok:
            cam, cam_src = (range_priors[k][0], float(np.hypot(range_priors[k][1], p.occluded_prior_sigma))), "track_prior"
        elif cam is not None and p.camera_track_prior and range_priors is not None and range_priors[k] is not None:
            # camera-only range of a box the track already occupies: the camera range is fragile (truncated box,
            # posture), the track range came from radar -> combine, widening by the discrepancy (docs/14 §10)
            tr = range_priors[k]
            gap = abs(cam[0] - tr[0])
            cam_c = _combine((cam[0], float(np.hypot(cam[1], gap / 3.0))), (tr[0], float(np.hypot(tr[1], p.occluded_prior_sigma))))
            cam_only = cam_c
        dm_fb = None                                                   # docs/33 B-a: depth-map fallback range
        if p.dmap_fallback and dm_k is not None and dm_k["ok"] and not occluded and d.cls in p.cluster_classes \
                and dm_k["r"] >= p.dmap_min_range and (dm_k.get("bias") is not None or not p.dmap_fallback_bias_only):
            r_d = dm_k["r"] * (dm_k["bias"] if dm_k.get("bias") is not None else 1.0)
            s_d = max(float(dm_k["sig"]), float(dm_k.get("sig_rel", 0.06)) * r_d)
            if p.camera_track_prior and range_priors is not None and range_priors[k] is not None:
                tr = range_priors[k]                                   # as the camera fallback (mono+track)
                dm_fb = _combine((r_d, float(np.hypot(s_d, abs(r_d - tr[0]) / 3.0))),
                                 (tr[0], float(np.hypot(tr[1], p.occluded_prior_sigma))))
            else:
                dm_fb = (r_d, s_d)
        ray = calib.pixel_ray(0.5 * (x1 + x2), 0.5 * (y1 + y2))
        B = _ray_basis(ray)
        hor = max(np.linalg.norm(ray[:2]), 1e-6)

        b_sig = bearing_sigma_rad(p, d.box, calib, shake_px)

        def at_range(rho, sig_along, src, ray_=None, lat_floor=0.0, **kw):
            ray_ = ray if ray_ is None else ray_
            s_ = rho / hor
            sig_lat = max(s_ * b_sig, 0.05, lat_floor)
            sig_up = max(0.25 * (y2 - y1) / calib.K[1, 1] * s_, 0.2)
            B_ = B if ray_ is ray else _ray_basis(ray_)
            cov = B_ @ np.diag([sig_along ** 2, sig_lat ** 2, sig_up ** 2]) @ B_.T
            return Det3D(d.cls, d.score, d.box, C + s_ * ray_, cov, source=src, **kw)

        if lik is not None and d.cls in p.cluster_classes:              # docs/38: range as a distribution
            det, dbg, rl_cues = _lik_detection(lik, p, d, k, idx, idx0, wk, mw, rng_c, uv, radar, calib, fx, model, dm_k,
                                               g_rng[k], occluded, mono, ground, conflict, reliability, v_comp, sigma_v,
                                               ray, b_sig, ego_yaw_rate, ego_speed, C, _cq)
            if _cq is not None:
                rec = _cue_record(_cq, k, d, det, calib, p, radar, uv, rng_c, w_pt, wk, v_comp, mw, C, fx, g_rng[k],
                                  gate_rng[k], ground_bad[k], ground, bottom_ok, mono, cam, cam_src, conflict, None, None,
                                  dm_k, occluded, r_ub, None, None, None, None, None, None, False)
                if det is not None and det.rlik is not None:
                    sm = det.rlik.summ
                    rec["lik"] = {"r": round(sm["r"], 3), "sig": round(sm["sig"], 3), "sig_full": round(sm["sig_full"], 3),
                                  "p_alt": round(sm["p_alt"], 4), "n_modes": int(sm["n_modes"]),
                                  "modes": [[round(m[0], 2), round(m[1], 3), round(m[2], 4)] for m in sm["modes"]],
                                  "rb": int(det.rlik.radar_backed)}
                cue_log.append(rec)
            _finish_det(det, dbg, d, dm_k, image, p, out)
            continue

        # range-mode prior: camera range (height prior and / or ground contact, reliability-weighted) x track range
        prior_cam = cam if cam is not None else ((g_rng[k], _ground_sigma(g_rng[k])) if g_rng[k] is not None and not occluded else None)
        prior = _combine(prior_cam, range_priors[k]) if range_priors is not None and range_priors[k] is not None else prior_cam

        # docs/30: range clusters scored by the layout descriptor, gated by the track's predicted BEV position
        chosen, skip_radar, dbg, chosen_safe = None, False, None, False
        xy_prior = xy_priors[k] if xy_priors is not None else None
        if model is not None and d.cls in p.cluster_classes and len(idx) >= p.min_pts and wk[idx].sum() >= p.min_weight \
                and not (xy_prior is None and p.cluster_score_no_track == "kde"):
            use_dm = model_dm is not None and dm_k is not None and dm_k["ok"]          # B-b: depth-map feature
            _cf = [] if _cq is not None else None
            cc = cluster_candidates(idx, radar.xyz, uv, radar.v_r, d.box, fx, model_dm if use_dm else model,
                                    p.cluster_link_m, w=None if mw is None else mw[idx],
                                    r_dmap=dm_k["r_org"] if use_dm else None, feats_out=_cf)
            if _cq is not None:
                _cq.update(cc=cc, cf=_cf)
            lr = sc = np.array([c[3] for c in cc])
            ok = sc >= p.cluster_min_score
            dbg = {"nc": len(cc), "p_max": round(float(sc.max()), 3), "r_c": [round(c[1], 1) for c in cc]}
            if use_dm:
                dbg["dm_bb"] = 1
            if mw is not None:
                dbg["mask_soft"] = 1
                if p.mask_soft_min_p > 0 and len(cc):              # diag_A: no on-mask member and a weak score ->
                    onm = np.array([bool((mw[c[0]] >= 1.0 - 1e-9).any()) for c in cc])   # as under the hard mask
                    weak = ~onm & (lr < p.mask_soft_min_p)
                    if weak.any():
                        ok &= ~weak
                        lr = np.where(weak, -1.0, lr)
                        sc = np.where(weak, -1.0, sc)
                        idx = idx[~np.isin(idx, np.concatenate([cc[i][0] for i in np.flatnonzero(weak)]))]
                        dbg["soft_weak"] = int(weak.sum())
            if xy_prior is not None:
                vmax = p.cluster_gate_speed.get(d.cls, p.cluster_gate_speed["default"])
                lat_sig = 0.0
                if turning:                                            # impl2: widen across the line of sight
                    r_pr = float(np.hypot(*(xy_prior[0] - C[:2])))
                    lat_sig = turn_az_sigma(p, float(ego_yaw_rate), float(xy_prior[2])) * r_pr
                gw = gate_clusters(cc, xy_prior, p.cluster_gate_chi2, p.cluster_gate_sigma, vmax, p.cluster_gate_margin,
                                   lat_sigma=lat_sig, origin=C[:2])
                if p.cluster_gate_safety_m > 0 and len(idx) and (ego_static is False or not p.cluster_gate_safety_moving_only):
                    # impl2 safety net: the KDE mode x CAMERA prior over all in-box points (the old path's pick without
                    # the track range prior -- the track prediction is what is suspect on a moving ego) is never gated
                    # out; it keeps the gate-edge weight, i.e. it is the fallback when nothing else passes
                    r_old = _kde_mode(rng_c[idx], wk[idx], max(p.sigma_range, 0.25),
                                      prior if p.cluster_gate_safety_ref == "old" else prior_cam)
                    safe = np.array([abs(float(np.median(rng_c[c[0]])) - r_old) <= p.cluster_gate_safety_m for c in cc])
                    gw_real = gw.copy()
                    if not (p.cluster_gate_safety_strict and (gw > 0).any()):
                        gw = np.where(safe, np.maximum(gw, np.exp(-0.5 * p.cluster_gate_chi2)), gw)
                    dbg["safe"] = int(safe.sum())
                ok &= gw > 0
                sc = sc * gw
                if _cq is not None:
                    _cq.update(gw=gw.copy(), gw_real=gw_real.copy() if "safe" in dbg else gw.copy())
                dd = [float(np.hypot(*(c[2] - xy_prior[0]))) for c in cc]
                dbg.update(npass=int((gw > 0).sum()), d_min=round(min(dd), 2), prior_sig=round(float(np.sqrt(np.trace(xy_prior[1]) / 2)), 2),
                           prior_r=round(float(np.hypot(*(xy_prior[0] - C[:2]))), 1), dt=round(float(xy_prior[2]), 3),
                           lat_sig=round(lat_sig, 2))
            if ok.any():
                i_ch = int(np.argmax(np.where(ok, sc, -1.0)))
                chosen = cc[i_ch]
                if _cq is not None:
                    _cq["ch"] = i_ch
                chosen_safe = "safe" in dbg and gw_real[i_ch] == 0.0
            elif xy_prior is not None and mono is not None and p.cluster_gate_mono_override > 0 and (lr >= p.cluster_min_score).any() \
                    and (ego_static is not False or not p.override_static_only) \
                    and abs(cc[int(np.argmax(lr))][1] - mono[0]) <= p.cluster_gate_mono_override * mono[0]:
                chosen = cc[int(np.argmax(lr))]                        # radar and camera agree against the track
                dbg["override"] = True
                if _cq is not None:
                    _cq["ch"] = int(np.argmax(lr))
            elif p.cluster_gate_fail == "camera" or xy_prior is None:
                skip_radar = True                                      # no plausible cluster: camera / track range

        if not skip_radar and len(idx) >= p.min_pts and wk[idx].sum() >= p.min_weight:
            if chosen is not None:                                     # the scored cluster's own points only
                mem = chosen[0]
                if p.cluster_range == "median":
                    r_mode, sel = float(np.median(rng_c[mem])), mem
                else:
                    r_mode = _kde_mode(rng_c[mem], wk[mem], max(p.sigma_range, 0.25), prior)
                    sel = mem[np.abs(rng_c[mem] - r_mode) <= p.kde_halfwidth]
            elif prior is None and occluded:                           # the seen person stands in front of the
                strong = idx[wk[idx] >= p.occluder_min_w]              # background behind it: front-most return
                r_mode = float(rng_c[strong].min()) if len(strong) else float(rng_c[idx][np.argmax(wk[idx])])
                sel = idx[np.abs(rng_c[idx] - r_mode) <= p.kde_halfwidth]
            else:
                r_mode = _kde_mode(rng_c[idx], wk[idx], max(p.sigma_range, 0.25), prior)
                sel = idx[np.abs(rng_c[idx] - r_mode) <= p.kde_halfwidth]
            if v_comp is not None and len(sel) >= 3:                  # Doppler consistency in scan-sigma units
                vs, ws = v_comp[sel], wk[sel]
                order = np.argsort(vs)
                med = vs[order][np.searchsorted(np.cumsum(ws[order]), 0.5 * ws.sum())]
                sel = sel[np.abs(vs - med) <= p.dop_z * sigma_v + 0.05]
            ws = wk[sel]
            rho = float(np.average(rng_c[sel], weights=ws))
            rho_radar = rho
            n_eff = ws.sum() ** 2 / max((ws ** 2).sum(), 1e-9)
            spread = MAD_K * np.median(np.abs(rng_c[sel] - np.median(rng_c[sel]))) if len(sel) > 2 else 0.3
            sig_r = max(np.sqrt(p.sigma_range ** 2 + spread ** 2 / max(n_eff, 1)), 0.15)
            src = "radar"
            if _cq is not None:
                _cq.update(rad=(r_mode, rho_radar, sig_r, n_eff, float(spread), int(len(sel))), sel=sel.copy())
            h_obs = None
            if reliability is not None and len(sel) >= 2 and n_eff >= 1.5:   # radar-backed box: teach the camera model
                reliability.observe(rho, None if mono is None else mono[0], ground, conflict)
                if mono is not None and abs(rho / mono[0] - 1.0) < 0.3:
                    h_obs = H * rho / mono[0]                          # this person's height seen by the radar
            if cam is not None:
                gap = abs(rho - cam[0]) / np.hypot(sig_r, cam[1])
                if _cq is not None:
                    _cq["gap"] = float(gap)
                if gap <= p.mono_gate:                                 # consistent: inverse-variance fusion
                    w_r, w_m = 1 / sig_r ** 2, 1 / cam[1] ** 2
                    rho, sig_r, src = (w_r * rho + w_m * cam[0]) / (w_r + w_m), (w_r + w_m) ** -0.5, "radar+mono"
                else:                                                  # disagree: keep radar, cover the discrepancy
                    sig_r = np.hypot(sig_r, rho - cam[0])
            if chosen_safe and p.cluster_gate_safety_sigma and range_priors is not None and range_priors[k] is not None:
                sig_r = float(np.hypot(sig_r, rho - range_priors[k][0]))   # impl2: a gate-rejected pick, covariance covers the
                dbg["safe_pick"] = True                                    # disagreement with the track (tracker smooths)
            ray_use, lat_floor = None, 0.0
            if chosen is not None and p.point_bearing:                # bearing from the cluster's BEV centroid
                cxy = np.average(radar.xyz[sel, :2], axis=0, weights=ws) - C[:2]
                az = np.arctan2(cxy[1], cxy[0])
                ray_use = np.array([hor * np.cos(az), hor * np.sin(az), ray[2]])   # same elevation as the box ray
                lat = (radar.xyz[sel, :2] - C[:2]) @ np.array([-np.sin(az), np.cos(az)])
                lat_sp = MAD_K * float(np.median(np.abs(lat - np.median(lat)))) if len(sel) > 2 else 0.0
                lat_floor = max(p.point_bearing_sigma, lat_sp / np.sqrt(max(n_eff, 1.0)))
            det = at_range(rho, sig_r, src, ray_=ray_use, lat_floor=lat_floor, n_radar=len(sel), radar_idx=sel)
            det.h_obs = h_obs
            if dm_k is not None and dm_k["ok"] and n_eff >= 1.5 and not chosen_safe and d.cls in p.cluster_classes:
                det.dmap_ratio = rho_radar / dm_k["r"]                 # teaches the depth stage this track's bias
            if v_comp is not None and len(sel):
                _velocity(det, radar, sel, ws, v_comp, sigma_v, calib, p)
        elif dm_fb is not None:                                        # docs/33 B-a
            det = at_range(dm_fb[0], max(dm_fb[1], 0.2), "dmap")
        elif cam_only is not None and cam_only[0] <= p.max_range:          # camera range tied to the box's track
            det = at_range(cam_only[0], max(cam_only[1], 0.2), "mono+track")
        elif cam is not None and cam[0] <= p.max_range and cam_src in ("mono", "track_prior"):
            det = at_range(cam[0], max(cam[1], 0.2), cam_src)
        elif cam is not None and cam[0] <= p.max_range:                  # ground contact chosen / only
            gray = calib.pixel_ray(0.5 * (x1 + x2), y2)
            s_ = cam[0] / max(np.linalg.norm(gray[:2]), 1e-6)
            Bg = _ray_basis(gray)
            s_along = np.hypot(cam[1], pitch_shake_range_sigma(p, cam[0], calib, shake_px)) if cam_src == "ground_ray" else cam[1]
            cov = Bg @ np.diag([s_along ** 2, max(cam[0] * b_sig, 0.05) ** 2, 1.0]) @ Bg.T
            det = Det3D(d.cls, d.score, d.box, C + s_ * gray, cov, source="ground_ray")
        elif g_pts[k] is not None and g_rng[k] <= p.max_range and _ground_sigma(g_rng[k]) <= p.max_ground_sigma:
            gr = g_rng[k]
            gray = calib.pixel_ray(0.5 * (x1 + x2), y2)
            Bg = _ray_basis(gray)
            s_along = np.hypot(_ground_sigma(gr), pitch_shake_range_sigma(p, gr, calib, shake_px))
            cov = Bg @ np.diag([s_along ** 2, max(gr * b_sig, 0.05) ** 2, 1.0]) @ Bg.T
            det = Det3D(d.cls, d.score, d.box, g_pts[k], cov, source="ground_ray")
        if _cq is not None:
            cue_log.append(_cue_record(_cq, k, d, det, calib, p, radar, uv, rng_c, w_pt, wk, v_comp, mw, C, fx,
                                       g_rng[k], gate_rng[k], ground_bad[k], ground, bottom_ok, mono, cam, cam_src,
                                       conflict, cam_only, dm_fb, dm_k, occluded, r_ub, prior_cam, prior,
                                       None if range_priors is None else range_priors[k],
                                       None if xy_priors is None else xy_priors[k],
                                       None if vel_priors is None else vel_priors[k],
                                       None if heights is None else heights[k], skip_radar))
        _finish_det(det, dbg, d, dm_k, image, p, out)
    return out


def _finish_det(det, dbg, d: Det2D, dm_k, image, p: FusionParams, out: list) -> None:
    """Common tail of fuse() per box: diagnostics, GUI / ReID crops, append."""
    if det is None:
        return
    x1, y1, x2, y2 = d.box
    if dm_k is not None:                                           # depth-stage diagnostics (evaluation dumps)
        dbg = dict(dbg or {})
        dbg.update(dm=round(float(dm_k["r"]), 2), dm_ok=int(bool(dm_k["ok"])), dm_age=dm_k.get("age"),
                   dm_bias=None if dm_k.get("bias") is None else round(float(dm_k["bias"]), 3))
    det.dbg = dbg
    if image is not None:
        xi1, yi1 = max(int(x1), 0), max(int(y1), 0)
        det.crop = image[yi1:int(y2), xi1:int(x2)].copy()
        if d.mask is not None:
            cm = np.zeros(det.crop.shape[:2], bool)
            m = d.mask[:cm.shape[0], :cm.shape[1]]
            cm[:m.shape[0], :m.shape[1]] = m
            det.crop_mask = cm
        if p.reid_masked and d.mask is not None:
            rc = det.crop.copy()
            m = d.mask[:rc.shape[0], :rc.shape[1]]
            rc[:m.shape[0], :m.shape[1]][~m] = (104, 116, 124)        # BGR ImageNet mean
            det.reid_crop = rc
    out.append(det)


def _cluster_sel(mem: np.ndarray, rng_c, wk, v_comp, sigma_v, p: FusionParams) -> np.ndarray:
    """Members of a range cluster used for its range / velocity: within kde_halfwidth of the cluster median, then the
    Doppler consistency cut of the legacy path (scan-sigma units)."""
    r0 = float(np.median(rng_c[mem]))
    sel = mem[np.abs(rng_c[mem] - r0) <= p.kde_halfwidth]
    if v_comp is not None and len(sel) >= 3:
        vs, ws = v_comp[sel], wk[sel]
        order = np.argsort(vs)
        med = vs[order][np.searchsorted(np.cumsum(ws[order]), 0.5 * ws.sum())]
        sel = sel[np.abs(vs - med) <= p.dop_z * sigma_v + 0.05]
    return sel


def _wedge_clusters(W: dict, radar: RadarFrame, uv, rng_c, C, box, a_left: float, a_right: float,
                    mirror_members: bool = False) -> list:
    """docs/41: range clusters (1-D, gap W["gap_m"]) of the kept returns whose azimuth from the camera centre lies within
    the box's left / right rays +- W["margin_deg"] (any height; image row not used), with the wedge-LR features: count,
    in-box count, lateral spread, azimuth offset, moving share, whole-scan RCS z, mirror returns (below ground / under
    W["mirror_z"] m) within 1 m of the cluster range, returns projecting below the box."""
    st = radar.stats
    n = len(rng_c)
    if st is None or n == 0:
        return []
    x1, y1, x2, y2 = box
    m = np.radians(W["margin_deg"])
    lo, hi = min(a_left, a_right) - m, max(a_left, a_right) + m
    half, mid = 0.5 * (hi - lo), 0.5 * (hi + lo)
    xy = radar.xyz[:, :2] - np.asarray(C, float)[:2]
    d_az = np.angle(np.exp(1j * (np.arctan2(xy[:, 1], xy[:, 0]) - mid)))
    wedge = (np.abs(d_az) <= half) & (rng_c <= W["max_range"])
    below = st.below_ground if st.below_ground is not None else np.zeros(n, bool)
    mirror = wedge & (below | (radar.xyz[:, 2] < W["mirror_z"]))
    base = st.keep
    if mirror_members and st.keep_pre_ground is not None:          # mirrors: right BEV place, wrong height (docs/38 §13)
        base = st.keep | (st.keep_pre_ground & below)
    idx = np.flatnonzero(wedge & base)
    if len(idx) == 0:
        return []
    zrs = st.z_rcs_scan if st.z_rcs_scan is not None else np.zeros(n)
    o = idx[np.argsort(rng_c[idx])]
    cuts = np.flatnonzero(np.diff(rng_c[o]) > W["gap_m"]) + 1
    out = []
    for ii in np.split(o, cuts):
        rm = float(np.median(rng_c[ii]))
        u, v = uv[ii, 0], uv[ii, 1]
        inb = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        f = dict(n_log=float(np.log1p(len(ii))), n_in=float(inb.sum()), lat_sd=float(np.std(d_az[ii] * rm)),
                 az_c=float(np.median(np.abs(d_az[ii])) / half), dyn=float(np.mean(st.dynamic[ii])),
                 rcs=float(np.nanmedian(zrs[ii])) if np.isfinite(zrs[ii]).any() else 0.0,
                 n_mirror=float(np.sum(np.abs(rng_c[mirror] - rm) <= 1.0)), n_out_low=float(np.sum(~inb & (v > y2))))
        out.append({"r": rm, "f": f, "mem": ii, "wedge": True})
    return out


def _lik_detection(lik, p: FusionParams, d: Det2D, k: int, idx, idx0, wk, mw, rng_c, uv, radar: RadarFrame, calib: RigCalib,
                   fx: float, model, dm_k, g_raw, occluded: bool, mono, ground, conflict, reliability, v_comp, sigma_v,
                   ray, b_sig: float, yaw: float, speed: float, C, _cq):
    """docs/38: the range distribution of one person box (rcmot.fusion.range_lik) and its Det3D.
    Cue inputs exactly as the docs/37 fit read them from the cue dumps (research/range_unc/cues.py, model.box_inputs):
    height prior from the box-centre column rays with H 1.8 m (no learned height), ground contact from the raw box-bottom
    ground range corrected by the online RangeReliability slope, width from the centre-row azimuth span, the range
    clusters of the scored path (layout features) or, when that path does not run (too little weight), single-linkage
    clusters of the positive-weight candidate points (unscored)."""
    from rcmot.fusion.range_lik import BoxCues, GRID, H_CLASS, W_PERSON, RangeLik, posterior, summarize

    x1, y1, x2, y2 = [float(v) for v in d.box]
    W, Himg = calib.image_size
    uc, vc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    dt_, db_ = calib.pixel_ray(uc, y1), calib.pixel_ray(uc, y2)
    dtan = dt_[2] / np.linalg.norm(dt_[:2]) - db_[2] / np.linalg.norm(db_[:2])
    dl, dr = calib.pixel_ray(x1, vc), calib.pixel_ray(x2, vc)
    daz = float(np.arctan2(dl[1], dl[0]) - np.arctan2(dr[1], dr[0]))
    h_px, w_px = y2 - y1, x2 - x1
    trunc_bot = y2 >= Himg - p.border_px
    cut = (y1 <= p.border_px) or trunc_bot or occluded
    side = (x1 <= p.border_px) or (x2 >= W - p.border_px)
    rel = reliability or RangeReliability()
    g_ok = (not trunc_bot) and (not occluded) and g_raw is not None and np.isfinite(g_raw) and g_raw < 150
    r_g = rel.correct_ground(float(g_raw)) if g_ok else np.nan
    g_ok = bool(g_ok and np.isfinite(r_g) and r_g > 0.5)
    r_w = W_PERSON / daz if daz > 1e-4 else np.nan
    dm_ok = bool(dm_k is not None and dm_k.get("ok") and dm_k.get("r") is not None and np.isfinite(dm_k["r"]))
    clusters = []
    if model is not None and len(idx) >= p.min_pts and wk[idx].sum() >= p.min_weight:
        feats = []
        cc = cluster_candidates(idx, radar.xyz, uv, radar.v_r, d.box, fx, model, p.cluster_link_m,
                                w=None if mw is None else mw[idx], feats_out=feats)
        for (mem, _, _, _), f in zip(cc, feats):
            clusters.append({"r": float(np.median(rng_c[mem])), "f": f, "mem": mem})
        if _cq is not None:
            _cq.update(cc=cc, cf=feats)
    else:
        pos = idx0[wk[idx0] > 0]
        if len(pos):
            rr = rng_c[pos]
            lab = range_clusters(rr, 1.0)
            for L in range(int(lab.max()) + 1):
                m = lab == L
                clusters.append({"r": float(np.median(rr[m])), "f": None, "mem": pos[m]})
    cues = BoxCues(r_mono=H_CLASS / dtan if dtan > 1e-4 else np.nan, h_px=h_px, w_px=w_px, aspect=w_px / max(h_px, 1),
                   cut=bool(cut), r_g=r_g, g_ok=g_ok, h_cam=float(C[2]), yaw=abs(float(yaw)), speed=float(speed),
                   r_w=r_w, w_ok=bool(not side and np.isfinite(r_w) and w_px >= 4),
                   r_dm=float(dm_k["r"]) if dm_ok else np.nan, dm_ok=dm_ok, clusters=clusters)
    terms = lik.camera_terms(cues)
    wedge_used = False
    if not clusters and p.lik_wedge_fallback and lik.wedge is not None:
        clusters = _wedge_clusters(lik.wedge, radar, uv, rng_c, C, (x1, y1, x2, y2),
                                   float(np.arctan2(dl[1], dl[0])), float(np.arctan2(dr[1], dr[0])))
        wedge_used = bool(clusters)
    bgs = getattr(radar, "bg_score", None)
    if bgs is not None:                                  # docs/41 §10: background clusters of the cell memory count less
        for c in clusters:
            mem = np.asarray(c.get("mem", []), int)
            if len(mem):
                c["bgf"] = float(np.clip(np.exp(p.lik_bg_slope * float(np.mean(bgs[mem]))), p.lik_bg_fmin, 1.0))
    gws = getattr(radar.stats, "ground_w", None) if radar.stats is not None else None
    if gws is not None:                                  # research/ground: soft below-ground weight (scan_stats ground_mode soft)
        for c in clusters:
            mem = np.asarray(c.get("mem", []), int)
            if len(mem):
                c["gwf"] = float(np.mean(gws[mem]))
    if wedge_used:
        log_rad, parts = lik.wedge_term(clusters)
    else:
        parts = lik.radar_parts(clusters)
        log_rad = lik.radar_term(parts)
    log_bev, n_bev = None, 0                             # docs/38 §13.6: range-only wedge evidence
    if p.lik_wedge_mode != "off" and lik.wedge is not None and not wedge_used and (p.lik_wedge_mode == "prod" or not clusters):
        wcl = _wedge_clusters(lik.wedge, radar, uv, rng_c, C, (x1, y1, x2, y2), float(np.arctan2(dl[1], dl[0])),
                              float(np.arctan2(dr[1], dr[0])), mirror_members=p.lik_wedge_mirror)
        if wcl:
            lw, _ = lik.wedge_term(wcl)
            if lw is not None:
                log_bev, n_bev = (p.lik_wedge_beta if p.lik_wedge_mode == "prod" else 1.0) * lw, len(wcl)
    log_cam = sum(terms.values()) if terms else None
    if log_cam is None and log_rad is None and log_bev is None:
        return None, None, cues
    post = posterior([t for t in (log_cam, log_rad, log_bev) if t is not None])
    summ = summarize(post)
    r_hat = summ["r"]
    j = int(np.clip(np.searchsorted(GRID, r_hat), 0, len(GRID) - 1))
    radar_backed = log_rad is not None and log_rad[j] >= np.log(p.lik_radar_backed)
    cl_out = []
    # wedge clusters (docs/41) have members outside the box, where this box's point weights wk are 0: kept returns
    # weigh 1 there
    w_wedge = radar.stats.keep.astype(float) if (wedge_used and radar.stats is not None) else None
    for c, (S, l) in zip(clusters, parts):
        wkc = w_wedge if c.get("wedge") else wk
        sel = _cluster_sel(c["mem"], rng_c, wkc, v_comp, sigma_v, p)
        e = {"r": c["r"], "idx": sel, "n": int(len(sel)), "l": float(l), "S": float(S[j]), "vel": None, "vel_cov": None,
             "vr": None, "vr_u": None, "vr_var": None, "wedge": bool(c.get("wedge"))}
        if v_comp is not None and len(sel):
            tmp = Det3D(d.cls, d.score, d.box, np.zeros(3), np.eye(3))
            _velocity(tmp, radar, sel, wkc[sel], v_comp, sigma_v, calib, p)
            e.update(vel=tmp.vel, vel_cov=tmp.vel_cov, vr=tmp.vr, vr_u=tmp.vr_u, vr_var=tmp.vr_var)
        cl_out.append(e)
    hor = max(float(np.linalg.norm(ray[:2])), 1e-6)
    s_ = r_hat / hor
    sig_up = max(0.25 * h_px / calib.K[1, 1] * s_, 0.2)
    B = _ray_basis(ray)
    cov = B @ np.diag([max(summ["sig"], 0.05) ** 2, max(s_ * b_sig, 0.05) ** 2, sig_up ** 2]) @ B.T
    det = Det3D(d.cls, d.score, d.box, C + s_ * ray, cov, source="lik+radar" if radar_backed else "lik")
    if radar_backed and cl_out:
        ch = max(cl_out, key=lambda e: e["S"])
        sel = ch["idx"]
        det.n_radar, det.radar_idx = int(len(sel)), sel
        det.vel, det.vel_cov, det.vr, det.vr_u, det.vr_var = ch["vel"], ch["vel_cov"], ch["vr"], ch["vr_u"], ch["vr_var"]
        ws = (w_wedge if ch.get("wedge") else wk)[sel]
        n_eff = ws.sum() ** 2 / max((ws ** 2).sum(), 1e-9) if len(sel) else 0.0
        if reliability is not None and len(sel) >= 2 and n_eff >= 1.5:  # radar-backed box: teach the camera model
            rho = float(np.average(rng_c[sel], weights=ws))
            reliability.observe(rho, None if mono is None else mono[0], ground, conflict)
    det.rlik = RangeLik(C=np.asarray(C, float).copy(), ray=np.asarray(ray, float).copy(), log_cam=log_cam, log_rad=log_rad,
                        summ=summ, radar_backed=bool(radar_backed), b_sig=float(b_sig), sig_up=float(sig_up), clusters=cl_out,
                        cut=bool(cues.cut), cues=cues, log_bev=log_bev)
    dbg = {"lik": {"r": round(r_hat, 2), "sig": round(summ["sig"], 3), "sig_full": round(summ["sig_full"], 3),
                   "p_alt": round(summ["p_alt"], 3), "n_modes": int(summ["n_modes"]), "rb": int(radar_backed),
                   "cues": sorted(terms) + (["radar"] if log_rad is not None else []), "nc": len(clusters),
                   "wedge": int(wedge_used), "bev": n_bev,
                   "bgf": [round(c.get("bgf", 1.0), 3) for c in clusters] if bgs is not None else None,
                   "modes": [[round(m[0], 2), round(m[1], 2), round(m[2], 3)] for m in summ["modes"]],
                   "cl": [[round(c["r"], 2), round(c["l"], 2), c["n"]] for c in cl_out],
                   "cam": {k: (round(float(v), 2) if v is not None and np.isfinite(v) else None) for k, v in
                           (("mono", cues.r_mono if cues.m_ok else None), ("ground", cues.r_g if cues.g_ok else None),
                            ("width", cues.r_w if cues.w_ok else None))}, "cut": int(cues.cut)}}
    return det, dbg, cues


def _r2(x, n: int = 3):
    return None if x is None else round(float(x), n)


def _cue_record(cq: dict, k: int, d: Det2D, det, calib: RigCalib, p: FusionParams, radar: RadarFrame, uv, rng_c, w_pt,
                wk, v_comp, mw, C, fx, g_raw, gate_r, g_bad, ground, bottom_ok, mono, cam, cam_src, conflict, cam_only,
                dm_fb, dm_k, occluded, r_ub, prior_cam, prior, tp, xyp, vp, h_learn, skip_radar) -> dict:
    """docs/37: JSON-ready record of every range cue of one box (fuse(cue_log=...)); read-only. Angles from the
    level-frame pixel rays: tan_top / tan_bot = tan(elevation) of the box top / bottom at the centre column (height
    prior: rho = H / (tan_top - tan_bot)), daz = azimuth span of the box at its centre row (width prior), az = azimuth
    of the box-centre ray; pts = the box's candidate radar points before the mask / weight gates (r from the camera
    centre, w0 = SNR x clutter weight, w = final weight, 0 = gated out, cl = cluster index or -1); cc = scored range
    clusters (p = LR probability, gw / gwr = 2-D track gate weight with / without the safety net, f = LR features);
    rad = the chosen radar range before the camera fusion; out = the 3-D output (r, sigma along the horizontal ray)."""
    x1, y1, x2, y2 = [float(v) for v in d.box]
    W, Himg = calib.image_size
    uc, vc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    dt_, db_ = calib.pixel_ray(uc, y1), calib.pixel_ray(uc, y2)
    tt, tb = dt_[2] / np.linalg.norm(dt_[:2]), db_[2] / np.linalg.norm(db_[:2])
    dl, dr = calib.pixel_ray(x1, vc), calib.pixel_ray(x2, vc)
    daz = float(np.arctan2(dl[1], dl[0]) - np.arctan2(dr[1], dr[0]))
    dcen = calib.pixel_ray(uc, vc)

    def pair(t):
        return None if t is None else [round(float(t[0]), 3), round(float(t[1]), 3)]

    rec = {"k": k, "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
           "trunc": [int(y1 <= p.border_px), int(y2 >= Himg - p.border_px), int(x1 <= p.border_px),
                     int(x2 >= W - p.border_px)],
           "tan_top": round(float(tt), 5), "tan_bot": round(float(tb), 5), "daz": round(daz, 5),
           "az": round(float(np.arctan2(dcen[1], dcen[0])), 5), "h_cam": round(float(C[2]), 3), "fx": round(fx, 1),
           "H": _r2(cq.get("H")), "H_learn": _r2(h_learn), "mono0": pair(cq.get("mono0")), "mono": pair(mono),
           "g_raw": _r2(g_raw), "gate_r": _r2(gate_r), "g_bad": int(bool(g_bad)), "g_used": _r2(ground),
           "g_corr": _r2(cq.get("g_corr")), "bottom_ok": int(bool(bottom_ok)),
           "rel": {kk: round(float(vv), 4) for kk, vv in cq["rel"].items()} if "rel" in cq else None,
           "cam": pair(cam), "cam_src": cam_src, "conflict": int(bool(conflict)), "cam_only": pair(cam_only),
           "dm_fb": pair(dm_fb), "occ": int(bool(occluded)), "r_ub": _r2(r_ub), "prior_cam": pair(prior_cam),
           "prior": pair(prior), "tp": pair(tp), "skip_radar": int(bool(skip_radar))}
    if xyp is not None:
        P2 = np.asarray(xyp[1], float)
        rec["xyp"] = [round(float(xyp[0][0]), 3), round(float(xyp[0][1]), 3), round(float(P2[0, 0]), 4),
                      round(float(P2[0, 1]), 4), round(float(P2[1, 1]), 4), round(float(xyp[2]), 3)]
    if vp is not None:
        Pv = np.asarray(vp[1], float)
        rec["vp"] = [round(float(vp[0][0]), 3), round(float(vp[0][1]), 3), round(float(np.sqrt(max(Pv[0, 0], 0.0))), 3),
                     round(float(np.sqrt(max(Pv[1, 1], 0.0))), 3)]
    if d.mask is not None:
        rec["mask_frac"] = round(float(np.asarray(d.mask, bool).mean()), 3)
        ys = np.nonzero(d.mask)[0]
        rec["mask_cy"] = round(float(ys.mean() / max(d.mask.shape[0], 1)), 3) if len(ys) else None
    idx0 = cq.get("idx0", np.zeros(0, int))
    if len(idx0):
        bw, bh = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
        onm = None
        if d.mask is not None and np.asarray(d.mask).size:
            m = np.asarray(d.mask, bool)
            u = np.clip(uv[idx0, 0].astype(int) - max(int(x1), 0), 0, m.shape[1] - 1)
            v = np.clip(uv[idx0, 1].astype(int) - max(int(y1), 0), 0, m.shape[0] - 1)
            onm = m[v, u]
        lab = np.full(len(idx0), -1)
        for ci, c in enumerate(cq.get("cc") or []):
            lab[np.isin(idx0, c[0])] = ci
        sel = cq.get("sel")
        rec["pts"] = {"r": [round(float(v), 2) for v in rng_c[idx0]],
                      "w0": [round(float(v), 3) for v in w_pt[idx0]],
                      "w": [round(float(v), 3) for v in wk[idx0]],
                      "un": [round(float(v), 3) for v in (uv[idx0, 0] - x1) / bw],
                      "vn": [round(float(v), 3) for v in (uv[idx0, 1] - y1) / bh],
                      "z": [round(float(v), 2) for v in radar.xyz[idx0, 2]],
                      "vc": None if v_comp is None else [round(float(v), 3) for v in v_comp[idx0]],
                      "on": None if onm is None else [int(v) for v in onm],
                      "mf": None if mw is None else [round(float(v), 3) for v in mw[idx0]],
                      "cl": [int(v) for v in lab],
                      "sel": None if sel is None else [int(v) for v in np.isin(idx0, sel)]}
    if cq.get("cc") is not None:
        gw, gwr = cq.get("gw"), cq.get("gw_real")
        cl = []
        for ci, (mem, rk, cxy, pr) in enumerate(cq["cc"]):
            f = cq["cf"][ci]
            cl.append({"r": round(float(rk), 3), "n": int(len(mem)), "p": round(float(pr), 4),
                       "w": round(float(np.sum(wk[mem])), 3), "rmin": round(float(rng_c[mem].min()), 2),
                       "rmax": round(float(rng_c[mem].max()), 2), "cxy": [round(float(cxy[0]), 2), round(float(cxy[1]), 2)],
                       "gw": None if gw is None else round(float(gw[ci]), 4),
                       "gwr": None if gwr is None else round(float(gwr[ci]), 4),
                       "f": {kk: (None if vv is None else round(float(vv), 4)) for kk, vv in f.items()}})
        rec["cc"] = cl
        rec["ch"] = cq.get("ch")
    if "rad" in cq:
        r_mode, rho_r, sig_r, n_eff, spread, n_sel = cq["rad"]
        rec["rad"] = {"mode": _r2(r_mode), "rho": _r2(rho_r), "sig": _r2(sig_r, 4), "n_eff": _r2(n_eff),
                      "spread": _r2(spread), "n": n_sel}
    if "gap" in cq:
        rec["gap"] = round(cq["gap"], 3)
    if dm_k is not None:
        dmr = {}
        for kk in ("r", "r_org", "sig", "bias", "sig_rel", "age"):
            vv = dm_k.get(kk)
            dmr[kk] = None if vv is None else round(float(vv), 4)
        dmr["ok"] = int(bool(dm_k.get("ok")))
        rec["dm"] = dmr
    if det is not None:
        rel_xy = det.pos[:2] - C[:2]
        rr = float(np.linalg.norm(rel_xy))
        u = rel_xy / max(rr, 1e-9)
        cov = np.asarray(det.pos_cov, float)[:2, :2]
        rec["out"] = {"r": round(rr, 3), "sig": round(float(np.sqrt(max(u @ cov @ u, 0.0))), 4), "src": det.source,
                      "n_radar": int(det.n_radar), "xy": [round(float(det.pos[0]), 3), round(float(det.pos[1]), 3)]}
    return rec


def _velocity(det: Det3D, radar: RadarFrame, sel, ws, v_comp, sigma_v, calib: RigCalib, p: FusionParams):
    """Full velocity (LSQ over the cluster's lines of sight) or a single radial measurement."""
    if p.velocity_mode == "none":
        return
    D = radar.xyz_radar[sel] / np.linalg.norm(radar.xyz_radar[sel], axis=1, keepdims=True)
    az = np.degrees(np.arctan2(D[:, 1], D[:, 0]))
    R = calib.R_lr
    if p.velocity_mode == "lsq" and len(sel) >= p.full_vel_min_pts and np.ptp(az) >= p.full_vel_min_spread_deg:
        A = D[:, :2] * np.sqrt(ws)[:, None]
        info = A.T @ A / sigma_v ** 2
        if np.linalg.cond(info) < 1e4:
            v2 = np.linalg.solve(info, A.T @ (v_comp[sel] * np.sqrt(ws)) / sigma_v ** 2)
            res = (v_comp[sel] - D[:, :2] @ v2) * np.sqrt(ws)
            infl = max(1.0, float(res @ res) / sigma_v ** 2 / max(len(sel) - 2, 1))   # residual-based inflation
            cov2 = np.linalg.inv(info) * infl
            if np.sqrt(np.linalg.eigvalsh(cov2).max()) <= p.max_tan_sigma and np.linalg.norm(v2) <= p.max_speed:
                cov = np.diag([0.0, 0.0, 0.0])
                cov[:2, :2] = cov2
                det.vel = R @ np.r_[v2, 0.0]
                cov = R @ cov @ R.T
                cov[2, 2] += 25.0                                   # vertical velocity unobserved
                det.vel_cov = cov
                return
    u_r = (D * ws[:, None]).sum(0)
    if not np.isfinite(u_r).all() or np.linalg.norm(u_r) < 1e-9 or ws.sum() <= 0:
        return                                                  # no weighted member: no velocity
    u_r /= np.linalg.norm(u_r)
    order = np.argsort(v_comp[sel])
    med = v_comp[sel][order][np.searchsorted(np.cumsum(ws[order]), 0.5 * ws.sum())]
    n_eff = ws.sum() ** 2 / max((ws ** 2).sum(), 1e-9)
    vs = v_comp[sel]
    local = MAD_K * np.median(np.abs(vs - med)) if len(sel) > 2 else sigma_v       # spread inside this object
    sig = max(sigma_v, local)
    det.vr, det.vr_u, det.vr_var = float(med), R @ u_r, float(1.57 * sig ** 2 / max(n_eff, 1) + 0.02 ** 2)
