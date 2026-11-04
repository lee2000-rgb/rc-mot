"""Radar-camera 3D multi-object tracker (3D ByteTrack + MOTDD modules mapped to radar), docs/08 §3[D].

Per frame:
  0. ego compensation (MOTDD CMC -> 3D): tracks live in the current base_link_level frame; the ego displacement
     (radar ego velocity + IMU yaw rate) is removed, so the KF velocity is the ABSOLUTE velocity.
  1. KF predict (class-specific acceleration noise).
  2. stage-1 association (score >= high_thr), all terms in chi^2-normalised units (docs/09 §4 "cost"):
       C = d2_pos / g_pos + d2_dop / g_dop + w2d * [a(s)(1-IoU) + (1-a(s)) d2_pos/g_pos] + w_app * z_app
     d2_pos  3D Mahalanobis (gate chi^2_3), d2_dop radial or 2D-velocity Mahalanobis (gate chi^2_1 / chi^2_2),
     a(s)    distance-adaptive IoU weight (MOTDD SAAC analogue), z_app frame-standardised appearance (ReID).
  3. stage-2 association (low-score dets, ByteTrack) with the remaining confirmed / lost tracks.
  4. recovery hook (Radar-Gated Recovery, tracking/recovery.py) for still-unmatched tracks.
  5. updates (position, then velocity: full 2D or radial scalar), births (re-linked to a recently deleted
     track by appearance + reachability when possible), lifecycle.

Lifecycle against duplicates / ghosts / ID inflation (docs/12):
  S1 radar-only measurements only EXTEND a track: inflated noise and a position-covariance floor that grows
     with the time since the last camera update (the radar may be following someone else's returns).
  S2 camera re-acquisition: an unmatched confident detection at the place of a lost / radar-only track
     (inflated gate or image overlap, unique, appearance not contradicting) takes over that track's ID and
     re-anchors its state (OC-SORT ORU / TrackTrack TAI analogue) instead of starting a new ID.
  S3 duplicate merge keeps the older ID when its identity is still camera-backed.
  S4 existence score (CBMOT-style): up on camera updates, decays while radar-only / coasting; only tracks
     above `exist_show` are output.

Standstill mode (docs/16): rig vibration, box jitter and radar range picks put 0.1-0.5 m of noise on every
position, which the constant-velocity filter turns into 0.3-1 m/s of phantom velocity on standing workers.
A track enters "static" when, over the last `static_window_s`, the line fit of its camera-backed positions is
not a significant motion (speed < max(static_v_enter, 2 standard errors), and below static_v_cap) and its
Doppler (radial, absolute) stays small; while static every update is followed by a zero-velocity pseudo-
measurement (object ZUPT). It leaves static as soon as the radar sees radial motion (|v_r| > static_vr_exit
and > 3 sigma, `static_vr_n` updates in a row) or the recent mean position has moved from the anchor by more
than max(static_d_exit, 3 x its standard error).

Re-entry memory (`reentry_memory`, off by default; docs/33 Track C, research/fusion_wave/trackC/report.md §7 and
impl_report.md): on a swing everybody leaves the image for 2-5 s and comes back. A confirmed track without a
camera fix for `reentry_coast_s` is OUT_OF_VIEW: no longer a candidate of the association, recovery, birth suppression
or merge (its coasting covariance would let it grab anybody), only a memory -- alive (lost / radar-only) or deleted
(the graveyard entry IS the memory). A memory carries a world anchor (median of its camera-backed positions over the
last 1 s, in the pipeline's world frame), an LSQ velocity from non-turning positions, a motion mode (walking = not
static, significant LSQ speed and filter speed; else standing), its range and an appearance gallery. A new track does
not take a relinked ID at birth; it is decided against the memories (Hungarian, position x appearance) on its first
detection (inherit only when decisive) and again when it reaches confirmation (median of its positions so far):
Mahalanobis gate on the world displacement (P_w = floor^2 + (sigma_v gap)^2 + R_anchor + R_det, R along the line of
sight widened while turning), cost m_w / 9.21 + d_app / bound, accepted when appearance is decisive, position is
decisive, or the memory is alone in the gate with a good match; an ambiguous decision waits (track not output) up to
`reentry_decide_s`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from rcmot.reid.appearance import DEFAULT_SCALES, ema_hist, fused_distance
from rcmot.tracking.kalman import KalmanCV3D
from rcmot.tracking.range_select import camera_only_range, select
from rcmot.types import Det3D, TrackOut, TrackState

CHI2_3_99, CHI2_2_99, CHI2_1_99 = 11.34, 9.21, 6.63
MAD_K = 1.4826


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU, a (N,4), b (M,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


@dataclass
class TrackerParams:
    high_thr: float = 0.5
    low_thr: float = 0.1
    new_thr: float = 0.6
    gate_pos: float = CHI2_3_99
    gate_dop: float = CHI2_1_99
    w3d: float = 1.0
    w2d: float = 0.5
    wdop: float = 1.0
    w_app: float = 0.5
    dac: bool = True                  # distance-adaptive IoU weight (SAAC analogue)
    dac_s_lo: float = 16.0            # px: below -> pure 3D
    dac_s_hi: float = 64.0            # px: above -> IoU at full weight
    min_hits: int = 3
    min_hits_radar: int = 2           # confirm faster when the detections are radar-supported
    max_lost_s: float = 1.5           # no measurement for this long -> delete
    max_lost_static_vehicle_s: float = 10.0   # parked machines are kept much longer (ImmortalTracker idea)
    static_speed: float = 0.3         # m/s
    max_radar_only_s: float = 3.0     # no camera confirmation for this long -> delete even with radar
    accel_std: dict = field(default_factory=lambda: {"worker": 1.5, "heavy_equipment": 0.8, "default": 1.0})
    max_speed: dict = field(default_factory=lambda: {"worker": 3.0, "heavy_equipment": 8.0, "default": 8.0})
    reach_sigma: float = 3.0          # reachability gate: |det - last update| <= v_max*dt + reach_sigma*sqrt(s_det^2+s_trk^2) + margin
    reach_margin: float = 0.5         # [m]
    extent_sigma: dict = field(default_factory=lambda: {"worker": 0.25, "heavy_equipment": 1.0, "default": 0.6})
    # adaptive radial-velocity noise per track: robust spread of its own innovations (micro-Doppler, few points)
    doppler_prior: dict = field(default_factory=lambda: {"worker": 0.15, "heavy_equipment": 0.05, "default": 0.1})
    doppler_window: int = 30
    doppler_min_innov: int = 5
    birth_suppress_chi2: float = CHI2_3_99   # no new track inside an existing track's gate
    merge_chi2: float = CHI2_2_99     # duplicate tracks: same class, positions within this (P1 + P2 + extent)
    merge_speed: float = 0.6          # ... and velocities closer than this [m/s]
    merge_skip_both_cam: bool = True  # two tracks updated by two different camera boxes this frame are two objects
                                      # (side-by-side workers at 20-30 m merged every few s: xpanner, docs/16)
    class_agnostic: bool = False
    # appearance (ReID)
    emb_momentum: float = 0.9
    emb_update_iou: float = 0.3       # update the gallery only when the box overlaps no other box above this
    app_margin0: float = 1.0          # appearance weight = sigmoid((margin - m0)/tau), margin in robust sigmas
    app_tau: float = 0.5
    relink_window_s: float = 30.0     # re-identify recently deleted tracks
    relink_ratio: float = 0.8         # Lowe ratio test d1/d2
    relink_max_dist: float = 0.35     # cosine distance (day GT, masked crops, gap >= 2 s: accepts 94 % same / 0.15 % other)
    app_reacq: bool = True            # appearance re-acquisition of alive tracks the camera lost (docs/14 §7)
    app_reacq_min_tcam_s: float = 0.5
    history_s: float = 6.0            # trajectory kept for the GUI tail
    # S1 radar-only = extend-only. Off by default: on the GT segments (docs/14) the inflated gates let tracks
    # grab other workers' detections (IDF1 0.81 -> 0.70 day)
    ro_extra_sigma: dict = field(default_factory=lambda: {"worker": 0.0, "heavy_equipment": 0.0, "default": 0.0})
    ro_floor0: float = 0.0            # [m] position sigma floor while radar-only ...
    ro_floor_rate: dict = field(default_factory=lambda: {"worker": 0.0, "heavy_equipment": 0.0, "default": 0.0})  # ... + rate * t_cam
    # S2 camera re-acquisition. Off by default: image-overlap inheritance swapped the IDs of crossing workers
    # (docs/14, IDF1 0.81 -> 0.55 day)
    reacq: bool = False
    reacq_floor_rate: dict = field(default_factory=lambda: {"worker": 0.6, "heavy_equipment": 0.5, "default": 0.6})
    reacq_biou: float = 0.2           # buffered IoU with the track's last camera box ...
    reacq_buffer: float = 0.3
    reacq_max_tcam_s: float = 3.0
    reacq_ratio: float = 0.6          # best d2 < ratio * second d2, else ambiguous -> no re-acquisition
    reacq_app_veto: float = 0.5       # cosine distance above which appearance contradicts
    # same-uniform crews (docs/32): the pipeline sets app_mode from reid.crowd_mean / reid.part_colour
    app_mode: str = "cos"             # cos = raw OSNet cosine | crowd = cosine of crowd-mean embeddings |
                                      # fused = crowd-mean embedding + pants / sleeve / shoe / whole colour (z units)
    app_w_emb: float = 3.0            # fused: embedding weight (each colour part weighs 1); 3.0 = docs/32 §5.5 (all configs)
    app_scales: dict = field(default_factory=lambda: dict(DEFAULT_SCALES))
    relink_max_crowd: float = 0.90    # relink / appearance re-acquisition threshold per mode (cos: relink_max_dist):
    relink_max_fused: float = 1.22    # same different-person accept rate as cos < 0.35 (bank pairs, gap >= 2 s)
    app_norm_crowd: float = 1.26      # cost normaliser per mode (cos: 0.5) = 0.5 * median(mode) / median(cos)
    app_norm_fused: float = 1.72
    relink_fragment_aware: bool = True    # (on since 88063b6, docs/32 §5.5) ratio test rivals: drop candidates that may be the SAME person as the best
                                      # (never alive at the same time AND relinkable to it): after one failed relink the
                                      # person's own fragments filled the graveyard and blocked every later relink
    # S3
    merge_keep_old_id: bool = True
    merge_id_tcam_s: float = 1.5      # the older track's identity counts if camera-backed this recently
    # S4 existence score
    exist_gain: float = 0.5           # camera update: e += gain * score * (1 - e)
    exist_tau_radar_s: float = 2.0    # radar-only update: e *= exp(-dt / tau)
    exist_tau_coast_s: float = 1.0    # no measurement: e *= exp(-dt / tau)
    exist_show: float = 0.3
    # standstill mode (docs/16)
    static_mode: bool = True
    static_window_s: float = 1.0      # positions used for the entry test
    static_min_n: int = 8
    static_v_enter: float = 0.3       # [m/s] fit speed below this (or below 2 standard errors) ...
    static_v_cap: float = 0.6         # ... but never above this
    static_vr_enter: float = 0.25     # [m/s] median |radial velocity| in the window below this
    static_vr_exit: float = 0.35      # [m/s] radial velocity that ends standstill ...
    static_vr_n: int = 2              # ... in this many updates in a row
    static_d_exit: float = 0.6        # [m] displacement of the recent mean from the anchor that ends standstill
    static_recent_n: int = 5
    static_zupt_sigma: float = 0.1    # [m/s] zero-velocity pseudo-measurement
    static_exit_fit_s: float = 0.5    # [s] short line fit that also ends standstill ...
    static_v_exit: float = 0.5        # ... when its speed exceeds this and is > static_exit_z standard errors
    static_exit_z: float = 3.0
    static_exit_vel_sigma: float = 0.6  # [m/s] velocity sigma restored on exit (the ZUPTs shrank it)
    max_lost_static_s: float = 5.0    # a standing worker (static mode) survives an occlusion this long (docs/16 §9)
    # position innovation limiter (docs/16 §9): a measurement farther from the prediction than the object could
    # have moved since its last measurement is damped (R scaled by (gap/cap)^2) and the posterior step is capped,
    # so a wrong range pick / neighbour's box cannot throw the position (and hence the velocity) 3 m in one frame
    innov_limit: bool = True
    same_ray_suppress: bool = True    # no birth for a box whose centre lies inside the projected box of a confirmed
                                      # same-class track that was NOT matched this frame: it is that track's box with a
                                      # wrong range (docs/16 §9); the track picks it up next frame via the range prior
    innov_margin: float = 0.4         # [m] cap = v_max * dt_since_measurement + margin + 2 sigma(P)
    # re-entry memory (docs/33 Track C step 2; offline spec research/fusion_wave/trackC/report.md §7). False = the
    # tracker is unchanged (no relink-at-birth replacement, no world history, no deferred confirmation)
    reentry_memory: bool = False
    reentry_world: bool = True        # anchors in the pipeline's world frame (WorldState, gyro-bias corrected, yaw held
                                      # at standstill); False or no world given: frame dead-reckoned from the tracker's
                                      # own ego compensation (= memories ego-compensated like lost tracks)
    reentry_coast_s: float = 1.5      # camera gap after which a confirmed track is OUT_OF_VIEW: no longer an association /
                                      # recovery / birth-suppression / merge candidate, only a memory
    reentry_min_tcam_s: float = 0.5   # alive tracks without a camera fix for this long are memory candidates too (a lost
                                      # track whose person came back as a new track the association missed: take over)
    reentry_hist_s: float = 5.0       # camera-backed world positions kept per track
    reentry_anchor_s: float = 1.0     # anchor = median of the camera-backed positions over the last this many s
    reentry_vel_s: float = 1.5        # LSQ velocity over this window of positions measured while NOT turning (|yaw rate|
                                      # <= reentry_turn_yaw_dps; turning adds apparent motion), >= 3 points over >= 0.6 s ...
    reentry_vmax: float = 2.5         # ... clamped [m/s]
    reentry_walk_speed: float = 0.3   # walking (CV) = not static, |v| > max(this, 2 standard errors) and the median filter
                                      # speed in the window > this; else standing (a position jump alone is not walking)
    reentry_sigma_v_stand: float = 0.2   # [m/s] random-walk sigma of a standing memory
    reentry_sigma_v_walk: float = 1.0    # ... of a walking one (constant velocity mean)
    reentry_floor: float = 0.7        # P_w floor sigma = max(floor, floor_rate x range) [m]
    reentry_floor_rate: float = 0.08
    reentry_r_los0: float = 0.3       # detection sigma along the line of sight = los0 + los_rate x r, at least
    reentry_r_los_rate: float = 0.05  # turn_rate x r while |yaw rate| > turn_yaw_dps; across the line of sight r_across
    reentry_r_turn_rate: float = 0.15
    reentry_turn_yaw_dps: float = 10.0
    reentry_r_across: float = 0.3
    reentry_margin: float = 0.5       # [m] displacement shortened by this before the Mahalanobis test
    reentry_gate: float = 13.8        # m_w gate (chi2_2 0.999)
    reentry_m_norm: float = 9.21      # cost = m_w / m_norm + w_app x d / app_relink_max (d missing: 1)
    reentry_w_app: float = 1.0
    reentry_nc_ratio: float = 1.15    # feasible only if d <= nc_ratio x app_relink_max (fused 1.22 -> 1.40, cos 0.35 -> 0.40)
    reentry_ratio: float = 0.9        # appearance decisive: d <= app_relink_max and d < ratio x min d of the gated rivals
    reentry_pos_ratio: float = 2.0    # position decisive: m2 >= pos_ratio x m1 and m1 <= pos_max
    reentry_pos_max: float = 4.6
    reentry_alone: str = "either"     # memory alone in the gate is accepted when: "any" (offline spec: appearance does not
                                      # contradict) | "either" (also m_w <= pos_max or d <= app_relink_max) | "both" (m_w <=
                                      # pos_max and d <= app_relink_max or missing); online the turn-widened gate often
                                      # holds one wrong memory after the right one was missed / expired
    reentry_decide_s: float = 0.4     # ambiguous at confirmation: keep the track unconfirmed (not output) up to this age;
                                      # the query position is the median of its camera-backed positions so far
    reentry_life_stand_s: float = 30.0   # memory lifetime (since the last camera fix): standing
    reentry_life_walk_s: float = 20.0    # walking with an appearance gallery
    reentry_life_walk_noapp_s: float = 10.0   # walking without
    reentry_gallery: int = 5          # appearance gallery: last N descriptors ...
    reentry_gallery_step_s: float = 0.2   # ... at least this far apart (clean boxes only)
    reentry_app_relink: bool = False  # optional 2nd stage: appearance-only relink to a memory OUTSIDE the gate
    reentry_app_relink_scale: float = 0.82   # d <= scale x app_relink_max (fused 1.22 -> 1.0 ~ A2 FA 0.1) ...
    reentry_app_relink_ratio: float = 0.7    # ... and d < ratio x the next memory
    # docs/38: range as a distribution. True (the Pipeline sets it when fusion.range_model is "lik"): person boxes that
    # carry Det3D.rlik are associated and updated through rcmot.tracking.range_select -- the track's prediction
    # multiplies the box's range likelihood, the measurement is the box likelihood inside the mode that wins -- and the
    # birth rules below replace the point-position ones for them. False: the tracker is unchanged.
    range_lik: bool = False
    lik_eps: float = 0.05             # robust prior: the track itself may be wrong (weight of a uniform over 1-90 m)
    lik_radar_backed: float = 3.0     # radar likelihood ratio at the measured range that counts as radar support
    lik_birth_rel_sigma: float = 0.10 # a single-mode box starts a track only when its range sigma / range is below this
    lik_birth_alt: float = 0.10       # p_alt above this (and >= 2 modes): the box starts a multi-hypothesis tentative track
    lik_hyp_max_s: float = 1.0        # hypotheses are decided by then (the most likely one wins) ...
    lik_hyp_decide: float = 0.9       # ... or as soon as one holds this posterior weight; no confirmation before that
    lik_hyp_speed: float = 1.5        # [m/s] random-walk speed of a hypothesis (a person may walk while it is open)
    lik_bias: bool = False            # per-track camera bias b (docs/38 §4.4). Off since 11-03 (docs/38 §10.6, r39): with it
                                      # the alarm view is slightly worse (danger missed +0.18 %p, zone too low +0.13 %p),
                                      # BEV IDF1 -0.014; it only saves ~8 ID switches (alarm priority: user 11-03).
                                      # When on: learned from radar-backed updates, applied to the camera part of every
                                      # later box of the track
    lik_bias_n0: float = 1.5          # shrinkage of b towards 0 (prior samples; sigma_b0 = se / sqrt(n0) = 0.05)
    lik_bias_se: float = 0.06         # per-sample sigma of ln(camera range / radar range)
    lik_bias_window: int = 30
    lik_bias_max: float = 0.3         # |ln(camera / radar)| above this is not a bias sample (posture, other object)
    lik_bias_sd_min: float = 0.03     # floor of the bias sigma (a person's camera bias also changes with posture / box)
    lik_birth_box_iou: float = 0.3    # birth gate: a box overlapping the camera box that updated a track this frame
                                      # less than this is another person, whatever the track's gate says (docs/36 §8.2)
    lik_birth_rules: bool = True      # False: the old birth gate / same-ray rule on the box's point position (ablation)
    lik_min_mass: float = 0.05        # a track may only take box modes holding at least this posterior mass
    lik_mask: str = "basin"           # "basin": every valley splits (range_select.basin_mask, docs/38 §7.4); "mode": only
                                      # the summary modes (peaks >= 1 % of the highest; the r38 server run)
    lik_reanchor: bool = True         # docs/38 §7.4: an unmatched confident one-peak RADAR-BACKED box (p_alt <
                                      # lik_reanchor_alt, sigma <= max(lik_reanchor_sigma, lik_reanchor_rel r)) that covers
                                      # the last camera box of a track left without a measurement this frame (IoU >=
                                      # lik_reanchor_iou, camera fix within lik_reanchor_tcam s) moves that track to the
                                      # box's range (velocity 0) instead of starting a new track next to it
    lik_reanchor_iou: float = 0.5
    lik_reanchor_tcam: float = 1.0
    lik_reanchor_alt: float = 0.05
    lik_reanchor_sigma: float = 0.4   # the box's sigma must be <= max(this, lik_reanchor_rel * r) [m]
    lik_reanchor_rel: float = 0.05
    lik_reanchor_n: int = 3           # ... and lik_reanchor_n of its boxes in a row (within lik_reanchor_win s) must say the
    lik_reanchor_win: float = 0.5     # same range (spread <= max(1 m, 10 %)); meanwhile the boxes start no track. One
                                      # wrong box (a radar return of something else) does not move a right track
    lik_dop_soft: bool = True         # range-likelihood boxes: a Doppler (or full-velocity) mismatch adds at most wdop to
                                      # the cost instead of rejecting the pair. The box's Doppler is the median of its whole
                                      # cluster (the old path weighted the points by the TRACK's predicted Doppler, another
                                      # feedback path); a limb / neighbour return must not veto a position match
                                      # (cf_vib_100: two people 0.8 m apart swapped 44 times). The velocity update keeps its
                                      # own chi² outlier test.
    debug_pairs: bool = False         # evaluation: record every (track, box) cost term / rejection reason in dbg_pairs


class Track:
    _next_id = 1                      # display IDs, handed out on confirmation
    _next_tmp = -1                    # tentative tracks carry a negative temporary ID

    def __init__(self, det: Det3D, p: TrackerParams, t: float, tid: int | None = None):
        if tid is None:
            tid = Track._next_tmp
            Track._next_tmp -= 1
        self.tid = tid
        self.cls = det.cls
        self.kf = KalmanCV3D(det.pos, det.pos_cov, accel_std=p.accel_std.get(det.cls, p.accel_std["default"]))
        self.state = TrackState.TENTATIVE
        self.hits, self.score = 1, det.score
        self.exist = 0.5 * det.score  # existence score (S4)
        self.h_obs: list = []         # radar-seen heights of this object (docs/14 §8)
        self.cam_pos = det.pos.copy() # position at the last camera-backed update
        self.tsu = 0.0                # time since any measurement (camera or radar)
        self.t_cam = 0.0              # time since a camera-backed measurement
        self.t_born, self.age = t, 0.0
        self.box2d, self.thumb = det.box2d, det.crop
        self.hist_t, self.hist_p = [t], [det.pos.copy()]
        self.emb = None if getattr(det, "emb", None) is None else det.emb.copy()
        self.hist = None if getattr(det, "hist", None) is None else det.hist.copy()   # part colour (docs/32)
        self.spans: list = []         # alive intervals of earlier incarnations of this ID (relink inherits them)
        self.t_dead: float | None = None
        self.radar_hits = int(det.n_radar > 0)
        self.last_pos, self.since_pos = det.pos.copy(), 0.0           # filtered position at the last update, time since
        self.innov: list = []                                          # radial-velocity innovations
        self.last_source, self.n_radar = det.source, det.n_radar
        self.innov = []
        self.static = False           # standstill mode (docs/16)
        self.radar_pos = det.pos.copy() if det.n_radar > 0 else None   # position at the last radar-backed update
        self.t_radar = 0.0 if det.n_radar > 0 else 1e9                 # time since it
        self.anchor = None            # mean position when standstill was entered
        self.mhist: list = []         # (t, xy, |radial velocity| or nan) of camera-backed updates, current frame
        self.vr_run = 0               # consecutive updates with significant radial motion
        # re-entry memory (p.reentry_memory only; docs/33 Track C)
        self.ready = False            # confirmation reached: the tracker decides the ID before confirming
        self.wh: list = []            # (t, world xy, range, world line of sight, |yaw rate| deg/s, KF speed) of camera-
                                      # backed updates
        self.gal: list = []           # appearance gallery: SimpleNamespace(emb, hist) per kept crop
        self.gal_t = -1e9
        self._cam_xy = None           # this frame's camera-backed detection xy (level frame), recorded by the tracker
        self._mem = None              # cached memory statistics
        self.cam_e: list = []         # docs/38: samples of ln(camera range / radar range) of this track (bias b)
        self.ra_votes: list = []      # docs/38 §7.4: (time, range) of its own boxes that disagreed with it (re-anchor)
        self.hyps = None              # docs/38: range hypotheses of a tentative track born from a multi-mode box:
                                      # list of dict(xy, P, lw); None = single hypothesis (the KF)
        self.hyp_cur = 0
        if p.reentry_memory:
            self._cam_xy = det.pos[:2].copy()
            self._gallery_add(det, p, t)
        self._apply_velocity(det, p)

    @property
    def tracklet_len(self) -> int:
        return self.hits

    def _gallery_add(self, det: Det3D, p: TrackerParams, t: float) -> None:
        if getattr(det, "emb", None) is None or t - self.gal_t < p.reentry_gallery_step_s - 1e-9:
            return
        h = getattr(det, "hist", None)
        self.gal = (self.gal + [SimpleNamespace(emb=det.emb.copy(), hist=None if h is None else h.copy())])[-p.reentry_gallery:]
        self.gal_t = t

    def height(self, min_n: int = 5) -> float | None:
        """Robust (median) learned object height, clipped to a plausible range; None until min_n samples."""
        if len(self.h_obs) < min_n:
            return None
        return float(np.clip(np.median(self.h_obs), 1.45, 2.05))

    def bias(self, p: TrackerParams) -> tuple[float, float]:
        """docs/38: camera range bias b of this track and its variance (shrunk towards 0)."""
        if not p.lik_bias:
            return 0.0, 0.0
        n = len(self.cam_e)
        return float(sum(self.cam_e)) / (n + p.lik_bias_n0), max(p.lik_bias_se ** 2 / (n + p.lik_bias_n0), p.lik_bias_sd_min ** 2)

    def speed(self) -> float:
        return float(np.linalg.norm(self.kf.vel[:2]))

    def predict_xy(self, dt: float, v_ego: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Horizontal position and 2x2 covariance of the CV prediction dt ahead (state untouched); v_ego: base
        velocity subtracted when the tracks live in the moving base frame (docs/16). Used as the 2-D prior of the
        fusion's cluster gate (docs/30)."""
        vel = self.kf.vel - (v_ego if v_ego is not None else 0.0)
        P, q = self.kf.P, self.kf.accel_std ** 2
        P2 = P[:2, :2] + dt * (P[:2, 3:5] + P[3:5, :2]) + dt * dt * P[3:5, 3:5] + np.eye(2) * (0.5 * dt * dt) ** 2 * q
        return self.kf.pos[:2] + vel[:2] * dt, 0.5 * (P2 + P2.T)

    def pos_R(self, det: Det3D, p: TrackerParams) -> np.ndarray:
        """Measurement covariance + object-extent ambiguity (which part of the body / machine was measured)."""
        e = p.extent_sigma.get(self.cls, p.extent_sigma["default"])
        return det.pos_cov + np.diag([e ** 2, e ** 2, (0.5 * e) ** 2])

    def doppler_var(self, p: TrackerParams) -> float:
        """Target-intrinsic radial-velocity variance: robust spread of this track's own innovations
        (class prior until enough samples)."""
        if len(self.innov) >= p.doppler_min_innov:
            a = np.asarray(self.innov)
            return float((MAD_K * np.median(np.abs(a - np.median(a)))) ** 2)
        return p.doppler_prior.get(self.cls, p.doppler_prior["default"]) ** 2

    def reachable(self, det: Det3D, p: TrackerParams) -> bool:
        """Class-specific short-term motion limit (max speed x elapsed time) around the last measured position."""
        vmax = p.max_speed.get(self.cls, p.max_speed["default"])
        s_det = max(np.linalg.eigvalsh(self.pos_R(det, p)[:2, :2]).max(), 0.0)
        s_trk = float(np.clip(np.linalg.eigvalsh(self.last_P), 0, None).max()) if hasattr(self, "last_P") else 0.0
        reach = vmax * max(self.since_pos, 0.066) + p.reach_sigma * np.sqrt(s_det + s_trk) + p.reach_margin
        return float(np.linalg.norm(det.pos[:2] - self.last_pos[:2])) <= reach

    def clamp_speed(self, p: TrackerParams) -> None:
        """Project the velocity onto the class's feasible set |v| <= v_max."""
        vmax = p.max_speed.get(self.cls, p.max_speed["default"])
        sp = self.speed()
        if sp > vmax:
            self.kf.x[3:5] *= vmax / sp

    def ego_compensate(self, dp: np.ndarray, dyaw: float) -> None:
        """Express the track in the new base frame after the base moved by dp (old frame) and rotated dyaw."""
        c, s = np.cos(dyaw), np.sin(dyaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        T = np.zeros((6, 6))
        T[:3, :3] = T[3:, 3:] = R.T
        self.kf.x = T @ (self.kf.x - np.concatenate([dp, np.zeros(3)]))
        self.kf.P = T @ self.kf.P @ T.T
        self.hist_p = [R.T @ (h - dp) for h in self.hist_p]
        self.last_pos = R.T @ (self.last_pos - dp)
        self.cam_pos = R.T @ (self.cam_pos - dp)
        if self.radar_pos is not None:
            self.radar_pos = R.T @ (self.radar_pos - dp)
        self.mhist = [(t, (R.T @ np.r_[xy - dp[:2], 0.0])[:2], vr) for t, xy, vr in self.mhist]
        if self.anchor is not None:
            self.anchor = (R.T @ np.r_[self.anchor - dp[:2], 0.0])[:2]
        if self.hyps:
            for h in self.hyps:
                h["xy"] = (R.T @ np.r_[h["xy"] - dp[:2], 0.0])[:2]
                h["P"] = R[:2, :2].T @ h["P"] @ R[:2, :2]

    def _apply_velocity(self, det: Det3D, p: TrackerParams) -> None:
        """Velocity measurement with a chi^2 outlier test (robust update); radial noise = measurement variance
        + this track's adaptive intrinsic Doppler variance."""
        if det.vel is not None:
            if self.hits < 2 or self.kf.velocity_d2(det.vel, det.vel_cov) <= CHI2_2_99:
                self.kf.update_velocity(det.vel, det.vel_cov)
        elif det.vr is not None:
            var = det.vr_var + self.doppler_var(p)
            if self.hits >= 2:
                self.innov = (self.innov + [det.vr - det.vr_u @ self.kf.vel])[-p.doppler_window:]
            if self.hits < 2 or self.kf.radial_d2(det.vr_u, det.vr, var) <= CHI2_1_99:
                self.kf.update_radial(det.vr_u, det.vr, var)

    def _floor_pos_cov(self, p: TrackerParams) -> None:
        """S1: while only radar sees the track, its horizontal position sigma stays >= floor0 + rate * t_cam
        (adds a PSD term, so P stays a valid covariance)."""
        s = p.ro_floor0 + p.ro_floor_rate.get(self.cls, p.ro_floor_rate["default"]) * self.t_cam
        w, V = np.linalg.eigh(self.kf.P[:2, :2])
        self.kf.P[:2, :2] += V @ np.diag(np.clip(s ** 2 - w, 0, None)) @ V.T

    def reacquire(self, det: Det3D, p: TrackerParams, t: float, boxes_others: np.ndarray | None = None,
                  zero_vel: bool = False) -> None:
        """S2: re-anchor the state on a camera detection after a lost / radar-only stretch (OC-SORT ORU idea:
        velocity from the last camera-backed position, not from what the radar may have followed).
        zero_vel (docs/38, range-likelihood boxes): velocity 0 -- a range jump over the gap is not a velocity."""
        dt = max(self.t_cam, 1e-3)
        v = np.zeros(3) if zero_vel else ((det.pos - self.cam_pos) / dt if dt > 0.3 else self.kf.vel.copy())
        v[2] = 0.0
        self.kf.x[:3], self.kf.x[3:] = det.pos, 0.5 * (v + self.kf.vel)
        P = np.zeros((6, 6))
        P[:3, :3] = self.pos_R(det, p)
        P[3:, 3:] = np.eye(3) * 0.5 ** 2
        self.kf.P = P
        self.last_pos, self.since_pos = det.pos.copy(), 0.0
        self.clamp_speed(p)
        self.update(det, p, t, boxes_others, position=False)

    def _standstill(self, det: Det3D, p: TrackerParams, t: float, camera: bool) -> None:
        """Standstill mode switch + zero-velocity pseudo-measurement (docs/16)."""
        vr = abs(det.vr) if det.vr is not None else (abs(float(det.vel[:2] @ det.pos[:2]) / max(np.linalg.norm(det.pos[:2]), 1e-6))
                                                    if det.vel is not None else np.nan)
        sig_r = np.sqrt(det.vr_var + self.doppler_var(p)) if det.vr is not None else 0.1
        moving_r = np.isfinite(vr) and vr > max(p.static_vr_exit, 3.0 * sig_r)
        self.vr_run = self.vr_run + 1 if moving_r else 0
        if camera:
            self.mhist = [h for h in self.mhist if h[0] >= t - max(p.static_window_s, 1.5)] + [(t, det.pos[:2].copy(), vr)]
        if self.static:
            recent = np.array([h[1] for h in self.mhist[-p.static_recent_n:]])
            moved = False
            if self.anchor is not None and len(recent) >= p.static_recent_n:
                d = float(np.linalg.norm(recent.mean(0) - self.anchor))
                se = float(np.sqrt(np.mean(np.sum((recent - recent.mean(0)) ** 2, 1)) / len(recent)))
                moved = d > max(p.static_d_exit, 3.0 * se)
            fast = False
            win = [h for h in self.mhist if h[0] >= t - p.static_exit_fit_s]
            if len(win) >= 4:
                ts = np.array([h[0] for h in win]) - t
                xy = np.array([h[1] for h in win])
                A = np.c_[np.ones(len(ts)), ts]
                coef, *_ = np.linalg.lstsq(A, xy, rcond=None)
                s2 = float(np.sum((xy - A @ coef) ** 2) / max(2 * (len(ts) - 2), 1))
                se = np.sqrt(s2 / max(np.sum((ts - ts.mean()) ** 2), 1e-6))
                v_fit = float(np.linalg.norm(coef[1]))
                fast = v_fit > max(p.static_v_exit, p.static_exit_z * se)
            if moved or fast or self.vr_run >= p.static_vr_n:
                self.static, self.anchor = False, None
                self.kf.P[3:5, 3:5] += np.eye(2) * p.static_exit_vel_sigma ** 2   # let the velocity re-converge fast
        else:
            win = [h for h in self.mhist if h[0] >= t - p.static_window_s]
            if len(win) >= p.static_min_n and win[-1][0] - win[0][0] >= 0.7 * p.static_window_s:
                ts = np.array([h[0] for h in win]) - t
                xy = np.array([h[1] for h in win])
                A = np.c_[np.ones(len(ts)), ts]
                coef, *_ = np.linalg.lstsq(A, xy, rcond=None)
                res = xy - A @ coef
                s2 = float(np.sum(res ** 2) / max(2 * (len(ts) - 2), 1))          # per-axis residual variance
                se = np.sqrt(s2 / max(np.sum((ts - ts.mean()) ** 2), 1e-6))       # slope standard error
                v_fit = float(np.linalg.norm(coef[1]))
                vrs = np.array([h[2] for h in win if np.isfinite(h[2])])
                quiet_r = len(vrs) == 0 or float(np.median(vrs)) < p.static_vr_enter
                if v_fit < min(max(p.static_v_enter, 2.0 * se), p.static_v_cap) and quiet_r and self.vr_run == 0:
                    self.static, self.anchor = True, xy.mean(0)
        if self.static:
            R = np.diag([p.static_zupt_sigma ** 2, p.static_zupt_sigma ** 2, 1.0])
            self.kf.update_velocity(np.zeros(3), R)

    def update(self, det: Det3D, p: TrackerParams, t: float, boxes_others: np.ndarray | None = None,
               position: bool = True) -> None:
        dt_e = max(self.tsu, 0.0)
        R = self.pos_R(det, p)
        if det.source == "radar_only":
            e = p.ro_extra_sigma.get(self.cls, p.ro_extra_sigma["default"])
            R = R + np.diag([e ** 2, e ** 2, 0.0])
        if position:
            if p.innov_limit:
                vmax = p.max_speed.get(self.cls, p.max_speed["default"])
                sig_p = float(np.sqrt(max(np.linalg.eigvalsh(self.kf.P[:2, :2]).max(), 0.0)))
                cap = vmax * max(self.since_pos, 0.066) + p.innov_margin + 2.0 * sig_p   # uncertain track: bigger step
                y = det.pos[:2] - self.kf.pos[:2]
                gap = float(np.linalg.norm(y))
                if gap > cap:
                    R = R * (gap / cap) ** 2
                pred = self.kf.pos[:2].copy()
                self.kf.update_position(det.pos, R)
                step = self.kf.pos[:2] - pred
                sn = float(np.linalg.norm(step))
                if sn > cap:                                           # hard physical limit on the posterior step
                    self.kf.x[:2] = pred + step * (cap / sn)
            else:
                self.kf.update_position(det.pos, R)
        if det.n_radar > 0 and det.source != "radar_only":
            self.radar_pos, self.t_radar = self.kf.pos.copy(), 0.0
        if det.source == "radar_only":
            self._floor_pos_cov(p)
        self._apply_velocity(det, p)
        self.clamp_speed(p)
        self.last_pos, self.since_pos, self.last_P = self.kf.pos.copy(), 0.0, self.kf.P[:2, :2].copy()
        self.tsu = 0.0
        self.last_source, self.n_radar = det.source, det.n_radar
        self.hist_t.append(t)
        self.hist_p.append(self.kf.pos.copy())
        while self.hist_t and self.hist_t[0] < t - p.history_s:
            self.hist_t.pop(0)
            self.hist_p.pop(0)
        if p.static_mode:
            self._standstill(det, p, t, camera=det.source != "radar_only")
        if det.source == "radar_only":            # radar keeps the state alive, but does not confirm it
            self.exist *= np.exp(-dt_e / p.exist_tau_radar_s)
            if self.state != TrackState.TENTATIVE:
                self.state = TrackState.RADAR_ONLY
            return
        self.exist += p.exist_gain * det.score * (1.0 - self.exist)
        if getattr(det, "h_obs", None) is not None:
            self.h_obs = (self.h_obs + [det.h_obs])[-40:]
        self.cam_pos = self.kf.pos.copy()
        self.hits += 1
        self.radar_hits += int(det.n_radar > 0)
        self.t_cam = 0.0
        self.score = 0.7 * self.score + 0.3 * det.score
        self.box2d = det.box2d
        clean = boxes_others is None or len(boxes_others) == 0 or \
            iou_xyxy(det.box2d[None], boxes_others).max() < p.emb_update_iou
        if det.crop is not None and det.crop.size and det.score >= self.score and clean:
            self.thumb = det.crop                                   # best-looking unoccluded crop for the gallery
        if getattr(det, "emb", None) is not None and clean:
            self.emb = det.emb.copy() if self.emb is None else p.emb_momentum * self.emb + (1 - p.emb_momentum) * det.emb
            self.emb /= np.linalg.norm(self.emb) + 1e-9
            if getattr(det, "hist", None) is not None:
                self.hist = ema_hist(self.hist, det.hist, p.emb_momentum)
        if p.reentry_memory:
            self._cam_xy = det.pos[:2].copy()
            if clean:
                self._gallery_add(det, p, t)
        if self.state == TrackState.TENTATIVE:
            need = p.min_hits_radar if self.radar_hits >= p.min_hits_radar else p.min_hits
            if self.hyps is not None:
                pass                                 # docs/38: range hypotheses still open -- not confirmed yet
            elif self.hits >= need and p.reentry_memory and self.tid < 0:
                self.ready = True                    # re-entry memory: the tracker decides the ID (_reentry_resolve)
            elif self.hits >= need:
                self.state = TrackState.CONFIRMED
                if self.tid < 0:                     # unconfirmed births do not consume visible ID numbers
                    self.tid = Track._next_id
                    Track._next_id += 1
        else:
            self.state = TrackState.CONFIRMED

    def out(self, t: float) -> TrackOut:
        o = TrackOut(self.tid, self.cls, self.state, self.kf.pos.copy(), self.kf.vel.copy(), self.box2d,
                     self.score, self.hits, self.tsu, self.thumb, list(self.hist_p))
        o.history_t = list(self.hist_t)
        o.vel_cov = self.kf.P[3:5, 3:5].copy()
        o.pos_cov = self.kf.P[:3, :3].copy()
        o.age = t - self.t_born
        o.source, o.n_radar = self.last_source, self.n_radar
        o.exist = self.exist
        o.static = self.static
        o.time_since_camera = self.t_cam
        return o


RecoverFn = Callable[[list[Track]], "dict[int, Det3D]"]


def _rot2(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def buffered_iou_xyxy(a: np.ndarray, b: np.ndarray, ratio: float) -> float:
    """IoU of two boxes after enlarging each by `ratio` of its size on every side (C-BIoU)."""
    def grow(x):
        w, h = x[2] - x[0], x[3] - x[1]
        return np.array([x[0] - ratio * w, x[1] - ratio * h, x[2] + ratio * w, x[3] + ratio * h])
    return float(iou_xyxy(grow(a)[None], grow(b)[None])[0, 0])


def app_distance(ta, tb, p: TrackerParams) -> float:
    """Appearance distance between two objects carrying .emb (and .hist): raw / crowd-mean cosine, or the fused
    crowd-mean embedding + part-colour distance (docs/32) when p.app_mode == 'fused'."""
    if p.app_mode == "fused":
        return fused_distance(ta.emb, getattr(ta, "hist", None), tb.emb, getattr(tb, "hist", None),
                              p.app_scales, p.app_w_emb)
    return 1.0 - float(ta.emb @ tb.emb)


def app_relink_max(p: TrackerParams) -> float:
    return {"crowd": p.relink_max_crowd, "fused": p.relink_max_fused}.get(p.app_mode, p.relink_max_dist)


def app_norm(p: TrackerParams) -> float:
    return {"crowd": p.app_norm_crowd, "fused": p.app_norm_fused}.get(p.app_mode, 0.5)


def appearance_terms(tracks: list[Track], dets: list[Det3D], p: TrackerParams):
    """Appearance distance and a per-track-row weight from the frame-relative margin (docs/09 §5.2).
    Rows whose candidates all look alike (same vests) get weight ~0, so motion / Doppler decide."""
    n, m = len(tracks), len(dets)
    D, W = np.zeros((n, m)), np.zeros(n)
    if n == 0 or m == 0:
        return D, W
    te = [t.emb for t in tracks]
    de = [getattr(d, "emb", None) for d in dets]
    if all(e is None for e in te) or all(e is None for e in de):
        return D, W
    for i, a in enumerate(te):
        if a is None:
            continue
        row = np.array([app_distance(tracks[i], dets[j], p) if b is not None else np.nan for j, b in enumerate(de)])
        ok = np.isfinite(row)
        if ok.sum() == 0:
            continue
        D[i, ok] = row[ok]
        if ok.sum() >= 2:
            srt = np.sort(row[ok])
            spread = max(MAD_K * np.median(np.abs(row[ok] - np.median(row[ok]))), 0.02)
            margin = (srt[1] - srt[0]) / spread
            W[i] = 1.0 / (1.0 + np.exp(-(margin - p.app_margin0) / p.app_tau))
        else:
            W[i] = 0.5
    return D, W


class RadarCameraTracker:
    def __init__(self, params: TrackerParams | None = None, project_box=None):
        self.p = params or TrackerParams()
        self.project_box = project_box          # (cls, pos) -> image box of the object's size, or None (docs/16 §9)
        self.tracks: list[Track] = []
        self.graveyard: list[tuple[float, Track]] = []     # (time of deletion, track) for re-identification
        self.t = 0.0
        self.relinked = 0
        self.relink_log: list | None = None       # diagnosis: set to [] to record every relink attempt
        self._log_new = False
        self._relinked_from = None
        self.reacquired = 0
        self.app_reacquired = 0
        self.reentered = 0                        # re-entry memory: IDs inherited at confirmation
        self.reentry_new = 0                      # ... confirmations decided "new" with memories in the gate
        self.reentry_deferred = 0                 # ... frames a confirmation was deferred (ambiguous)
        self.reentry_s = 0.0                      # ... CPU time spent in the re-entry code [s]
        self._pose = (0.0, np.zeros(2))           # memory frame: world pose (theta, origin) of the current base frame
        self._yaw_dps = 0.0
        self._sel_cache: dict = {}                # docs/38: (id(track), id(det)) -> range_select.Sel of this frame
        self.dbg_pairs: list = []                 # debug_pairs: this frame's (track, box) cost records
        self.lik_counts = {"birth_wide": 0, "birth_hyp": 0, "hyp_switch": 0, "gate_sep_box": 0, "bias_samples": 0,
                           "reanchor": 0}

    # ------------------------------------------------------------------------------------ docs/38 range likelihood
    def _lik(self, d: Det3D) -> bool:
        return self.p.range_lik and getattr(d, "rlik", None) is not None

    def _ext(self, cls: str) -> float:
        return self.p.extent_sigma.get(cls, self.p.extent_sigma["default"])

    def _sel(self, t: Track, d: Det3D, cache: bool = True):
        """range_select.Sel of box d for track t (its KF state now; a track with open range hypotheses: the best of
        them). cache: reuse / store within this frame's association (same track state)."""
        return self._sel_full(t, d, cache)[0]

    def _sel_full(self, t: Track, d: Det3D, cache: bool = True):
        """(Sel, per-hypothesis Sels or None)."""
        key = (id(t), id(d))
        if cache and key in self._sel_cache:
            return self._sel_cache[key]
        p = self.p
        lat_gate = 4.0 * p.gate_pos
        sels = None
        if t.hyps:
            sels = [select(d, h["xy"], h["P"], self._ext(t.cls), t.bias(p), p.lik_eps, p.lik_radar_backed, lat_gate,
                           p.lik_min_mass, p.lik_mask) for h in t.hyps]
            ok = [s_ for s_ in sels if s_ is not None]
            sel = min(ok, key=lambda s_: s_.d2) if ok else None
        else:
            sel = select(d, t.kf.pos[:2], t.kf.P[:2, :2], self._ext(t.cls), t.bias(p), p.lik_eps, p.lik_radar_backed,
                         lat_gate, p.lik_min_mass, p.lik_mask)
        if cache:
            self._sel_cache[key] = (sel, sels)
        return sel, sels

    def _hyp_update(self, t: Track, d: Det3D, sel, sels):
        """Accumulate the evidence of box d for each open hypothesis of t, move each to its own measurement, follow the
        best one with the KF, close the hypotheses when one dominates or time is up. Returns the measurement to use."""
        p = self.p
        sels = sels or []
        for h, s_ in zip(t.hyps, sels):
            if s_ is None:
                h["lw"] += np.log(1e-6)
                continue
            h["lw"] += s_.lm
            P = np.asarray(s_.det.pos_cov, float)[:2, :2]
            h["xy"], h["P"] = s_.det.pos[:2].copy(), P + np.eye(2) * self._ext(t.cls) ** 2
        lw = np.array([h["lw"] for h in t.hyps])
        w = np.exp(lw - lw.max())
        w /= w.sum()
        for h, wi in zip(t.hyps, w):
            h["lw"] = float(np.log(max(wi, 1e-300)))
        best = int(np.argmax(w))
        out = sels[best] if best < len(sels) and sels[best] is not None else sel
        if best != t.hyp_cur and out is not None:                      # the KF follows the best hypothesis
            t.kf.x[:3] = out.det.pos
            t.kf.x[3:] = 0.0
            t.kf.P = np.zeros((6, 6))
            t.kf.P[:3, :3] = t.pos_R(out.det, p)
            t.kf.P[3:, 3:] = np.eye(3) * 3.0 ** 2
            t.last_pos, t.cam_pos = out.det.pos.copy(), out.det.pos.copy()
            t.hyp_cur = best
            self.lik_counts["hyp_switch"] += 1
        if w.max() >= p.lik_hyp_decide or self.t - t.t_born >= p.lik_hyp_max_s:
            t.hyps = None
        return out

    def _meas(self, t: Track, d: Det3D, sel=None) -> Det3D:
        """The measurement box d gives track t: d itself (no range likelihood) or its Sel det (docs/38), after the
        hypothesis bookkeeping and the camera-bias sample of a radar-backed update."""
        if not self._lik(d):
            return d
        p = self.p
        if sel is None:
            sel, sels = self._sel_full(t, d)
        else:
            sels = self._sel_full(t, d, cache=False)[1] if t.hyps else None
        if t.hyps:
            sel = self._hyp_update(t, d, sel, sels)
        self._sel_cache.pop((id(t), id(d)), None)
        if sel is None:
            return d
        if p.lik_bias and sel.radar and not d.rlik.cut:          # full boxes only: a cut box's camera-only range is
            c = camera_only_range(d)                               # one-sided, not this person's camera bias
            if c is not None and c[2] < 0.1:
                e = float(np.log(c[0] / sel.r))
                if abs(e) <= p.lik_bias_max:
                    t.cam_e = (t.cam_e + [e])[-p.lik_bias_window:]
                    self.lik_counts["bias_samples"] += 1
        return sel.det

    def _reanchor_target(self, d: Det3D) -> Track | None:
        """docs/38 §7.4: the track an unmatched confident one-peak box re-anchors (see TrackerParams.lik_reanchor), or
        None. The track must have been left without a measurement this frame and its last camera box must be this one
        (IoU); two such tracks -> ambiguous, none."""
        p = self.p
        sm = d.rlik.summ
        if not d.rlik.radar_backed:
            return None          # a camera-only range can be wrong the same way frame after frame (a box cut at the image
                                 # edge: conti_halt_short_060 GT 3 at 8.0 m said 13.9 m) -- persistence does not catch it
        if sm["p_alt"] >= p.lik_reanchor_alt or sm["sig"] > max(p.lik_reanchor_sigma, p.lik_reanchor_rel * sm["r"]):
            return None
        cand = []
        for t in self.tracks:
            if t.cls != d.cls or t.state == TrackState.TENTATIVE or t.hyps or t.tsu == 0.0 or t.box2d is None \
                    or t.t_cam > p.lik_reanchor_tcam or self._oov(t):
                continue
            o = float(iou_xyxy(np.asarray(t.box2d, float)[None], np.asarray(d.box2d, float)[None])[0, 0])
            if o >= p.lik_reanchor_iou:
                cand.append((o, t))
        if len(cand) != 1:
            return None
        return cand[0][1]

    def _reacq_meas(self, t: Track, d: Det3D):
        """(measurement, zero_vel) for a re-acquisition: a range-likelihood box gives the range selected against the
        track's coasting prediction (its significant modes only) and no velocity from the range jump (docs/38)."""
        if not self._lik(d):
            return d, False
        sel = self._sel(t, d, cache=False)
        return (sel.det if sel is not None else d), True

    def _birth_suppressed_lik(self, d: Det3D) -> bool:
        """Birth gate of a box with a range likelihood: inside an existing track's gate (range_select d²), unless that
        track was updated this frame by its own, different camera box (another person)."""
        p = self.p
        for t in self.tracks:
            if t.cls != d.cls or not ((t.state != TrackState.TENTATIVE or t.ready) and not self._oov(t)):
                continue
            if t.tsu == 0.0 and t.t_cam == 0.0 and t.box2d is not None and \
                    iou_xyxy(np.asarray(t.box2d, float)[None], np.asarray(d.box2d, float)[None])[0, 0] < p.lik_birth_box_iou:
                sel = self._sel(t, d, cache=False)
                if sel is not None and sel.d2 <= p.birth_suppress_chi2:
                    self.lik_counts["gate_sep_box"] += 1
                continue
            sel = self._sel(t, d, cache=False)
            if sel is not None and sel.d2 <= p.birth_suppress_chi2:
                return True
        return False

    def _same_ray_lik(self, d: Det3D) -> bool:
        """Same-ray rule with a range likelihood: the box lies inside the projected box of a confirmed track that got no
        CAMERA update this frame (lost, coasting, or kept alive by radar only), and either
          - the box's likelihood supports the track's range (along-ray d² within the gate): it is that track's box, or
          - the track has radar support (a radar-backed update within the last 0.5 s) and the box has none at its own
            range: a camera-only range (a box whose bottom is hidden, docs/38) does not outvote the radar.
        Otherwise (the box is radar-backed at another range, or neither has radar) it may start a track: another person
        on the same ray, or a wrong track."""
        cx, cy = 0.5 * (d.box2d[0] + d.box2d[2]), 0.5 * (d.box2d[1] + d.box2d[3])
        for t in self.tracks:
            if t.state == TrackState.TENTATIVE or t.cls != d.cls or t.t_cam == 0.0 or self._oov(t):
                continue
            b = self.project_box(t.cls, t.kf.pos)
            if b is not None and b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                sel = self._sel(t, d, cache=False)
                if sel is not None and sel.d2_along <= self.p.gate_pos:
                    return True
                if t.t_radar <= 0.5 and not d.rlik.radar_backed:
                    return True
        return False

    def _birth_range_ok(self, d: Det3D) -> bool:
        """docs/38: a single-mode box starts a track only when its range is sharp enough (sigma / range <
        lik_birth_rel_sigma); a multi-mode box (p_alt >= lik_birth_alt, >= 2 modes) always may (range hypotheses)."""
        p = self.p
        sm = d.rlik.summ
        if sm["p_alt"] >= p.lik_birth_alt and len(sm["modes"]) >= 2:
            return True
        if sm["sig"] / max(sm["r"], 1.0) >= p.lik_birth_rel_sigma:
            self.lik_counts["birth_wide"] += 1
            return False
        return True

    def _new_track(self, d: Det3D, tid) -> Track:
        """Track from an unmatched box; a multi-mode box (docs/38) opens range hypotheses (the KF starts at the dominant
        mode and follows the best hypothesis until they are decided)."""
        p = self.p
        hyps = None
        if self._lik(d):
            sm = d.rlik.summ
            if sm["p_alt"] >= p.lik_birth_alt and len(sm["modes"]) >= 2:
                rl = d.rlik
                hyps = []
                for r_k, s_k, m_k in sm["modes"][:3]:
                    s_ = r_k / rl.hor
                    u = rl.u
                    perp = np.array([-u[1], u[0]])
                    P = np.outer(u, u) * max(s_k, 0.05) ** 2 + np.outer(perp, perp) * max(s_ * rl.b_sig, 0.05) ** 2
                    hyps.append({"xy": rl.point(r_k)[:2].copy(), "P": P + np.eye(2) * self._ext(d.cls) ** 2,
                                 "lw": float(np.log(max(m_k, 1e-6)))})
                self.lik_counts["birth_hyp"] += 1
        tr = Track(d, p, self.t, tid)
        if hyps is not None and tid is None:
            tr.hyps = hyps
            tr.hyp_cur = int(np.argmax([h["lw"] for h in hyps]))
        return tr

    def _cost(self, tracks: list[Track], dets: list[Det3D]) -> np.ndarray:
        p = self.p
        C = np.full((len(tracks), len(dets)), 1e6)
        if not tracks or not dets:
            return C
        tb = np.array([t.box2d for t in tracks])
        iou = iou_xyxy(tb, np.array([d.box2d for d in dets]))
        s = np.sqrt(np.clip((tb[:, 2] - tb[:, 0]) * (tb[:, 3] - tb[:, 1]), 0, None))
        alpha = np.clip((s - p.dac_s_lo) / (p.dac_s_hi - p.dac_s_lo), 0, 1) if p.dac else np.ones(len(tracks))
        A, Wa = appearance_terms(tracks, dets, p)
        norm = app_norm(p)
        dbg = self.dbg_pairs if p.debug_pairs else None
        for i, t in enumerate(tracks):
            for j, d in enumerate(dets):
                if not p.class_agnostic and t.cls != d.cls:
                    continue
                d0 = d
                if self._lik(d):                                     # docs/38: range likelihood x track prediction
                    sel = self._sel(t, d)
                    if dbg is not None:
                        dbg.append(dict(tid=int(t.tid), oid=id(d0), d2=None if sel is None else round(sel.d2, 2),
                                        al=None if sel is None else round(sel.d2_along, 2),
                                        lat=None if sel is None else round(sel.d2_lat, 2),
                                        r=None if sel is None else round(sel.r, 2), s=None if sel is None else round(sel.s, 3),
                                        rb=None if sel is None else int(sel.radar), why="ok"))
                    if sel is None or sel.d2 > p.gate_pos:
                        if dbg is not None:
                            dbg[-1]["why"] = "gate"
                        continue
                    m, d = sel.d2, sel.det                           # this track's measurement of the box
                else:
                    m = t.kf.mahalanobis_pos(d.pos, t.pos_R(d, p))
                    if dbg is not None:
                        dbg.append(dict(tid=int(t.tid), oid=id(d0), d2=round(m, 2), why="ok"))
                    if m > p.gate_pos:
                        if dbg is not None:
                            dbg[-1]["why"] = "gate"
                        continue
                if not t.reachable(d, p):
                    if dbg is not None:
                        dbg[-1]["why"] = "reach"
                    continue
                m_n = m / p.gate_pos
                c = p.w3d * m_n + p.w2d * (alpha[i] * (1.0 - iou[i, j]) + (1 - alpha[i]) * m_n)
                soft = p.lik_dop_soft and d0 is not d                # a range-likelihood measurement (docs/38)
                if t.hits >= 2:                                      # velocity of a 1-hit track is unknown
                    if d.vel is not None:
                        dd = t.kf.velocity_d2(d.vel, d.vel_cov)
                        if dd > CHI2_2_99 and not soft:
                            if dbg is not None:
                                dbg[-1]["why"] = f"vel {dd:.1f}"
                            continue
                        c += p.wdop * min(dd / CHI2_2_99, 1.0)
                    elif d.vr is not None:
                        dd = t.kf.radial_d2(d.vr_u, d.vr, d.vr_var + t.doppler_var(p))
                        if dd > p.gate_dop and not soft:
                            if dbg is not None:
                                dbg[-1]["why"] = f"dop {dd:.1f}"
                            continue
                        c += p.wdop * min(dd / p.gate_dop, 1.0)
                if Wa[i] > 0:
                    c += p.w_app * Wa[i] * A[i, j] / norm
                C[i, j] = c
        return C

    def _match(self, tracks, dets):
        C = self._cost(tracks, dets)
        if C.size == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))
        r, c = linear_sum_assignment(C)
        pairs = [(i, j) for i, j in zip(r, c) if C[i, j] < 1e5]
        mt = {i for i, _ in pairs}
        md = {j for _, j in pairs}
        return pairs, [i for i in range(len(tracks)) if i not in mt], [j for j in range(len(dets)) if j not in md]

    def _spans(self, tr: Track) -> list:
        return tr.spans + [(tr.t_born, self.t if tr.t_dead is None else tr.t_dead)]

    def _maybe_same(self, a: Track, b: Track) -> bool:
        """relink_fragment_aware: b cannot serve as a rival of a in the ratio test when it may be the same person:
        never alive at the same time (no overlap > 0.2 s of their alive intervals) and close enough in appearance to
        be relinked to it."""
        if not self.p.relink_fragment_aware or a is b or a.emb is None or b.emb is None:
            return False                                  # no appearance (ReID off / not yet seen): cannot be judged
        if any(min(a1, b1) - max(a0, b0) > 0.2 for a0, a1 in self._spans(a) for b0, b1 in self._spans(b)):
            return False
        return app_distance(a, b, self.p) < app_relink_max(self.p)

    def _relink(self, det: Det3D) -> int | None:
        """Reuse the ID of a recently deleted track: appearance ratio test + reachable by max speed."""
        p = self.p
        e = getattr(det, "emb", None)
        if e is None:
            return None
        cands = []
        log = [] if self.relink_log is not None else None
        for td, tr in self.graveyard:
            if tr.cls != det.cls or tr.emb is None:
                continue
            gap = self.t - td + tr.t_cam                             # since its last camera fix
            reach = p.max_speed.get(tr.cls, p.max_speed["default"]) * gap + 2.0
            dxy = float(np.linalg.norm(det.pos[:2] - tr.cam_pos[:2]))
            if log is not None:
                log.append(dict(tid=tr.tid, gap=round(gap, 2), dxy=round(dxy, 2), reach=round(reach, 2),
                                d=round(app_distance(tr, det, p), 4)))
            if dxy > reach:
                continue
            cands.append((app_distance(tr, det, p), tr.tid, tr))
        out = None
        if cands:
            cands.sort(key=lambda c: c[0])
            d1, best = cands[0][0], cands[0][2]
            d2 = min([c[0] for c in cands[1:] if not self._maybe_same(best, c[2])] +
                     [1.0 if p.app_mode == "cos" else np.inf])
            if d1 < app_relink_max(p) and d1 < p.relink_ratio * d2:
                self.graveyard = [(td, tr) for td, tr in self.graveyard if tr.tid != cands[0][1]]
                self.relinked += 1
                out = cands[0][1]
                self._relinked_from = best
        if log is not None:
            self.relink_log.append(dict(t=round(self.t, 3), pos=det.pos[:2].round(2).tolist(), box=det.box2d.round(1).tolist(),
                                        cands=log, relinked=out, max=app_relink_max(p)))
            self._log_new = True
        return out

    def _app_reacquire(self, det: Det3D) -> Track | None:
        """docs/14 §7: an unmatched confident detection that LOOKS like an alive track the camera lost (lost /
        radar-only for >= app_reacq_min_tcam_s) and is reachable from that track's last camera fix takes it over.
        Appearance decides (masked-crop ReID, ratio test against every other known identity of the class, incl.
        tracks matched this frame and the graveyard); position only rules out the impossible."""
        p = self.p
        e = getattr(det, "emb", None)
        if e is None:
            return None
        cands, rivals = [], []
        for t in self.tracks:
            if t.state == TrackState.TENTATIVE or t.cls != det.cls or t.emb is None:
                continue
            dist = app_distance(t, det, p)
            vmax = p.max_speed.get(t.cls, p.max_speed["default"])
            if t.t_cam >= p.app_reacq_min_tcam_s and np.linalg.norm(det.pos[:2] - t.cam_pos[:2]) <= vmax * t.t_cam + 2.0 \
                    and not self._oov(t):                            # out of view: the re-entry memory decides
                cands.append((dist, t))
            else:
                rivals.append((dist, t))
        rivals += [(app_distance(tr, det, p), tr) for _, tr in self.graveyard if tr.cls == det.cls and tr.emb is not None]
        if not cands:
            return None
        cands.sort(key=lambda c: c[0])
        d1, t = cands[0]
        d2 = min([c[0] for c in cands[1:] + rivals if not self._maybe_same(t, c[1])] +
                 [1.0 if p.app_mode == "cos" else np.inf])
        if d1 < app_relink_max(p) and d1 < p.relink_ratio * d2:
            self.app_reacquired += 1
            return t
        return None

    def _reacquire(self, det: Det3D, boxes_others: np.ndarray) -> Track | None:
        """S2: the lost / radar-only track whose place this unmatched camera detection takes, if unambiguous."""
        p = self.p
        cands = []
        for t in self.tracks:
            if t.state == TrackState.TENTATIVE or t.cls != det.cls or t.t_cam <= 0 or t.t_cam > p.reacq_max_tcam_s:
                continue
            vmax = p.max_speed.get(t.cls, p.max_speed["default"])
            if np.linalg.norm(det.pos[:2] - t.cam_pos[:2]) > vmax * t.t_cam + 1.5:
                continue                                             # not reachable from the last camera fix
            if getattr(det, "emb", None) is not None and t.emb is not None and \
                    app_distance(t, det, p) > p.reacq_app_veto * app_relink_max(p) / p.relink_max_dist:
                continue                                             # appearance clearly contradicts
            g = p.reacq_floor_rate.get(t.cls, p.reacq_floor_rate["default"]) * t.t_cam
            S = t.kf.P[:3, :3] + t.pos_R(det, p) + np.diag([g ** 2, g ** 2, 0.0])
            y = det.pos - t.kf.pos
            d2 = float(y @ np.linalg.solve(S, y))
            biou = buffered_iou_xyxy(t.box2d, det.box2d, p.reacq_buffer) if t.box2d is not None else 0.0
            if d2 <= p.gate_pos or biou >= p.reacq_biou:
                cands.append((min(d2, p.gate_pos * (1.0 - biou)), t))
        if not cands:
            return None
        cands.sort(key=lambda c: c[0])
        if len(cands) > 1 and cands[0][0] >= p.reacq_ratio * cands[1][0]:
            return None                                              # two plausible owners: do not guess
        self.reacquired += 1
        return cands[0][1]

    def _same_ray(self, d: Det3D) -> bool:
        cx, cy = 0.5 * (d.box2d[0] + d.box2d[2]), 0.5 * (d.box2d[1] + d.box2d[3])
        for t in self.tracks:
            if t.state == TrackState.TENTATIVE or t.cls != d.cls or t.tsu == 0.0 or self._oov(t):
                continue
            b = self.project_box(t.cls, t.kf.pos)
            if b is not None and b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                return True
        return False

    # ------------------------------------------------------------------------------------ re-entry memory (Track C)
    def _oov(self, t: Track) -> bool:
        """OUT_OF_VIEW (re-entry memory only): confirmed, no camera fix for reentry_coast_s."""
        return self.p.reentry_memory and t.state != TrackState.TENTATIVE and t.t_cam >= self.p.reentry_coast_s

    def _reentry_pose(self, world, ego_dp, ego_dyaw: float, dt: float) -> None:
        """World pose of the current base frame: the pipeline's WorldState when given (reentry_world), else dead-reckoned
        from the tracker's own ego compensation (x_world = R(theta) x + origin, updated with the old heading)."""
        t0 = time.perf_counter()
        self._yaw_dps = abs(float(np.degrees(ego_dyaw / dt))) if dt > 0 else 0.0
        if world is not None and self.p.reentry_world:
            self._pose = (float(world.theta), np.asarray(world.origin, float)[:2].copy())
        else:
            th, o = self._pose
            if ego_dp is not None:
                o = o + _rot2(th) @ np.asarray(ego_dp, float)[:2]
            self._pose = (th + float(ego_dyaw), o)
        self.reentry_s += time.perf_counter() - t0

    def _R_det(self, r: float, los: np.ndarray, yaw_dps: float) -> np.ndarray:
        """Detection position covariance (world, 2x2): along the line of sight los0 + los_rate r (turn_rate r while
        turning), across r_across."""
        p = self.p
        sa = p.reentry_r_los0 + p.reentry_r_los_rate * r
        if yaw_dps > p.reentry_turn_yaw_dps:
            sa = max(sa, p.reentry_r_turn_rate * r)
        perp = np.array([-los[1], los[0]])
        return sa * sa * np.outer(los, los) + p.reentry_r_across ** 2 * np.outer(perp, perp)

    def _mem_stats(self, tr: Track) -> dict | None:
        """Memory of a track from its camera-backed world history: anchor, velocity, motion mode, range, R_anchor,
        lifetime. Cached until the next camera update (or standstill change)."""
        if not tr.wh:
            return None
        t_last = tr.wh[-1][0]
        if tr._mem is not None and tr._mem["t_last"] == t_last and tr._mem["static"] == tr.static and \
                tr._mem["n_gal"] == len(tr.gal):
            return tr._mem
        p = self.p
        W = np.array([h[1] for h in tr.wh if t_last - h[0] <= p.reentry_anchor_s + 1e-9])
        anchor = np.median(W, axis=0)
        _, _, r, los, yaw, _ = tr.wh[-1]
        Q = [h for h in tr.wh if h[4] <= p.reentry_turn_yaw_dps]       # quiet (not turning) positions
        V = [h for h in Q if Q[-1][0] - h[0] <= p.reentry_vel_s + 1e-9] if Q else []
        v, se = np.zeros(2), np.inf
        if len(V) >= 3 and V[-1][0] - V[0][0] >= 0.6:
            ts = np.array([h[0] for h in V]) - V[-1][0]
            X = np.array([h[1] for h in V])
            A = np.c_[ts, np.ones_like(ts)]
            coef, *_ = np.linalg.lstsq(A, X, rcond=None)
            v = coef[0]
            s2 = float(np.sum((X - A @ coef) ** 2) / max(2 * (len(ts) - 2), 1))
            se = float(np.sqrt(s2 / max(np.sum((ts - ts.mean()) ** 2), 1e-6)))
            sp = float(np.linalg.norm(v))
            if sp > p.reentry_vmax:
                v = v * (p.reentry_vmax / sp)
        kf_v = float(np.median([h[5] for h in V])) if V else 0.0       # filter speed (Doppler-aided) in the same window
        walking = (not tr.static) and float(np.linalg.norm(v)) > max(p.reentry_walk_speed, 2.0 * se) and kf_v > p.reentry_walk_speed
        life = p.reentry_life_stand_s if not walking else (p.reentry_life_walk_s if tr.gal else p.reentry_life_walk_noapp_s)
        tr._mem = dict(t_last=t_last, static=tr.static, n_gal=len(tr.gal), anchor=anchor, kf_v=kf_v, se=se,
                       v=v if walking else np.zeros(2), walking=walking, r=float(r), R=self._R_det(r, los, yaw), life=life)
        return tr._mem

    def _mem_alive(self, tr: Track) -> bool:
        s = self._mem_stats(tr)
        return s is not None and self.t - s["t_last"] <= s["life"]

    def _reentry_step(self) -> None:
        """Record this frame's camera-backed positions in the world frame, then decide the IDs of the tracks that
        reached confirmation."""
        t0 = time.perf_counter()
        th, org = self._pose
        R = _rot2(th)
        for t in self.tracks:
            if t._cam_xy is None:
                continue
            xy, t._cam_xy = t._cam_xy, None
            r = float(np.linalg.norm(xy))
            t.wh.append((self.t, R @ xy + org, r, R @ (xy / max(r, 1e-6)), self._yaw_dps, t.speed()))
            while len(t.wh) > 1 and t.wh[0][0] < self.t - self.p.reentry_hist_s:
                t.wh.pop(0)
        pend = [t for t in self.tracks if t.state == TrackState.TENTATIVE and (t.ready or t.t_born == self.t)]
        if pend:
            self._reentry_resolve(pend)
        self.reentry_s += time.perf_counter() - t0

    def _confirm_new(self, q: Track) -> None:
        q.state, q.ready = TrackState.CONFIRMED, False
        if q.tid < 0:
            q.tid = Track._next_id
            Track._next_id += 1

    def _inherit(self, q: Track, m: Track) -> None:
        """q (a track reaching confirmation) continues memory m: m's ID, alive intervals and gallery; m is removed
        (from the graveyard, or from the alive tracks when it was a lost / radar-only track out of view)."""
        p = self.p
        if any(t is m for t in self.tracks):
            m.t_dead = self.t - m.tsu
            self.tracks = [t for t in self.tracks if t is not m]
        else:
            self.graveyard = [(td, tr) for td, tr in self.graveyard if tr is not m]
        q.tid, q.spans = m.tid, self._spans(m)
        q.gal = (m.gal + q.gal)[-p.reentry_gallery:]
        q.state, q.ready, q.hits = TrackState.CONFIRMED, False, max(q.hits, p.min_hits)
        self.reentered += 1

    def _reentry_resolve(self, pend: list[Track]) -> None:
        """Joint decision (Hungarian) of the pending tracks against the OUT_OF_VIEW memories (report §7 items 2-7).
        pend = tracks born this frame (decided on their first detection: inherit when decisive, else they stay
        tentative) and tracks that reached confirmation (median of their positions so far; ambiguous -> deferred up
        to reentry_decide_s, then new)."""
        p = self.p
        mems = []
        for m in [tr for _, tr in self.graveyard] + [t for t in self.tracks if t.state != TrackState.TENTATIVE
                                                     and t.t_cam >= p.reentry_min_tcam_s]:
            s = self._mem_stats(m)
            if s is not None and self.t - s["t_last"] <= s["life"]:
                mems.append((m, s))
        nq, nm = len(pend), len(mems)
        M, D = np.full((nq, nm), np.inf), np.full((nq, nm), np.nan)
        qpos = [None] * nq
        acc = app_relink_max(p)
        for qi, q in enumerate(pend):
            if not q.wh:
                continue
            xy = np.median(np.array([h[1] for h in q.wh]), axis=0)
            qpos[qi] = xy
            Rq = self._R_det(q.wh[-1][2], q.wh[-1][3], max(h[4] for h in q.wh))
            for mi, (m, s) in enumerate(mems):
                if not p.class_agnostic and m.cls != q.cls:
                    continue
                gap = self.t - s["t_last"]
                floor = max(p.reentry_floor, p.reentry_floor_rate * s["r"])
                sv = p.reentry_sigma_v_walk if s["walking"] else p.reentry_sigma_v_stand
                S = (floor ** 2 + (sv * gap) ** 2) * np.eye(2) + s["R"] + Rq
                d = xy - (s["anchor"] + s["v"] * gap)
                n = float(np.linalg.norm(d))
                if n > 0:
                    d = d * max(0.0, 1.0 - p.reentry_margin / n)
                M[qi, mi] = float(d @ np.linalg.solve(S, d))
                if q.emb is not None:
                    gal = m.gal or ([m] if m.emb is not None else [])
                    if gal:
                        D[qi, mi] = min(app_distance(g, q, p) for g in gal)
        ingate = M <= p.reentry_gate
        nc = p.reentry_nc_ratio * acc
        feas = ingate & (np.isnan(D) | (D <= nc))
        assigned = {}
        if feas.any():
            C = np.where(feas, M / p.reentry_m_norm + p.reentry_w_app * np.where(np.isnan(D), 1.0, D / acc), 1e6)
            for qi, mi in zip(*linear_sum_assignment(C)):
                if not feas[qi, mi]:
                    continue
                m = mems[mi][0]
                others = [k for k in range(nm) if k != mi and ingate[qi, k] and not self._maybe_same(m, mems[k][0])]
                d1, m1 = D[qi, mi], M[qi, mi]
                dO = [D[qi, k] for k in others if not np.isnan(D[qi, k])]
                app_ok = not np.isnan(d1) and d1 <= acc and d1 < p.reentry_ratio * (min(dO) if dO else np.inf)
                m2 = min((M[qi, k] for k in others), default=np.inf)
                pos_ok = m2 >= p.reentry_pos_ratio * m1 and m1 <= p.reentry_pos_max
                # alone in the gate (no rival): accepted when the match itself is good (reentry_alone) -- the turn-
                # widened gate alone is no evidence online, where the right memory may be missing / expired
                pos_good, app_good = m1 <= p.reentry_pos_max, not np.isnan(d1) and d1 <= acc
                alone_ok = {"any": True, "either": pos_good or app_good,
                            "both": pos_good and (app_good or np.isnan(d1))}[p.reentry_alone]
                if (alone_ok if not others else (app_ok or pos_ok)):
                    assigned[qi] = mi
        taken = set(assigned.values())
        log = self.relink_log
        for qi, q in enumerate(pend):
            if qi in assigned:
                m = mems[assigned[qi]][0]
                if log is not None:
                    log.append(dict(kind="reentry", t=round(self.t, 3), decision="inherit" if q.ready else "inherit_birth", tid=int(m.tid),
                                    m_w=round(float(M[qi, assigned[qi]]), 2), d=None if np.isnan(D[qi, assigned[qi]])
                                    else round(float(D[qi, assigned[qi]]), 3), n_gate=int(ingate[qi].sum()),
                                    alive=any(t is m for t in self.tracks), pos=None if qpos[qi] is None else qpos[qi].round(2).tolist()))
                self._inherit(q, m)
                continue
            if not q.ready:
                continue                                             # birth frame, not decisive: normal confirmation
            if feas[qi].any() and q.tsu == 0 and self.t - q.t_born < p.reentry_decide_s - 1e-9:
                self.reentry_deferred += 1                           # ambiguous: wait for more positions / crops
                continue
            mi = self._reentry_app_relink(qi, q, qpos[qi], mems, M, D, acc, taken) if p.reentry_app_relink else None
            if mi is not None:
                taken.add(mi)
                if log is not None:
                    log.append(dict(kind="reentry", t=round(self.t, 3), decision="app_relink", tid=int(mems[mi][0].tid),
                                    d=round(float(D[qi, mi]), 3)))
                self._inherit(q, mems[mi][0])
                continue
            self.reentry_new += int(ingate[qi].any())
            self._confirm_new(q)
            if log is not None:
                near = np.argsort(M[qi])[:4] if nm else []
                log.append(dict(kind="reentry", t=round(self.t, 3), decision="new", tid=int(q.tid), n_gate=int(ingate[qi].sum()),
                                pos=None if qpos[qi] is None else qpos[qi].round(2).tolist(), n_mem=nm,
                                cands=[dict(tid=int(mems[k][0].tid), m_w=round(float(M[qi, k]), 2), walk=bool(mems[k][1]["walking"]),
                                            gap=round(self.t - mems[k][1]["t_last"], 2), anchor=mems[k][1]["anchor"].round(2).tolist(),
                                            v=mems[k][1]["v"].round(2).tolist(),
                                            d=None if np.isnan(D[qi, k]) else round(float(D[qi, k]), 3)) for k in near]))

    def _reentry_app_relink(self, qi, q, xy, mems, M, D, acc, taken) -> int | None:
        """Optional strict appearance-only relink to a memory outside the position gate (report §7 item 9)."""
        p = self.p
        if xy is None:
            return None
        cand = []
        for k, (m, s) in enumerate(mems):
            if k in taken or np.isnan(D[qi, k]) or M[qi, k] <= p.reentry_gate:
                continue
            gap = self.t - s["t_last"]
            if np.linalg.norm(xy - s["anchor"]) <= p.max_speed.get(m.cls, p.max_speed["default"]) * gap + 2.0:
                cand.append((D[qi, k], k))
        if not cand:
            return None
        d1, k1 = min(cand)
        m = mems[k1][0]
        d2 = min([D[qi, k] for k in range(len(mems)) if k != k1 and not np.isnan(D[qi, k])
                  and not self._maybe_same(m, mems[k][0])] + [np.inf])
        if d1 <= p.reentry_app_relink_scale * acc and d1 < p.reentry_app_relink_ratio * d2:
            return k1
        return None

    def _merge_duplicates(self, tracks: list[Track]) -> list[Track]:
        """Two tracks of one object (e.g. a lost / radar-only copy next to the re-acquired track): keep the
        one with the most camera support (then the older one)."""
        p = self.p
        conf = [t for t in tracks if t.state != TrackState.TENTATIVE]
        drop: set[int] = set()                                       # id() of dropped Track objects
        for i in range(len(conf)):
            for j in range(i + 1, len(conf)):
                a, b = conf[i], conf[j]
                if a.cls != b.cls or id(a) in drop or id(b) in drop:
                    continue
                if p.merge_skip_both_cam and a.tsu == 0.0 and b.tsu == 0.0 and a.t_cam == 0.0 and b.t_cam == 0.0:
                    continue
                if p.reentry_memory and max(a.t_cam, b.t_cam) >= p.reentry_coast_s:
                    continue                                         # an OUT_OF_VIEW track is a memory: its coasting
                                                                     # covariance overlaps anybody after a few s; it is
                                                                     # taken over by the re-entry decision or expires
                e = p.extent_sigma.get(a.cls, p.extent_sigma["default"])
                S = a.kf.P[:2, :2] + b.kf.P[:2, :2] + 2 * e ** 2 * np.eye(2)
                d = a.kf.pos[:2] - b.kf.pos[:2]
                if d @ np.linalg.solve(S, d) > p.merge_chi2 or np.linalg.norm(a.kf.vel[:2] - b.kf.vel[:2]) > p.merge_speed:
                    continue
                key = lambda t: (t.t_cam == 0.0, t.hits, -t.tid)
                lose = min((a, b), key=key)
                keep = b if lose is a else a
                drop.add(id(lose))
                if p.merge_keep_old_id and lose.t_born < keep.t_born and lose.t_cam <= p.merge_id_tcam_s:
                    keep.tid, keep.t_born = lose.tid, lose.t_born            # S3: identity of the older track
                    keep.hits = max(keep.hits, lose.hits)
        return [t for t in tracks if id(t) not in drop]

    def update(self, dets: list[Det3D], dt: float, *, ego_dp: np.ndarray | None = None, ego_dyaw: float = 0.0,
               recover: RecoverFn | None = None, world=None) -> list[TrackOut]:
        """world: the pipeline's WorldState of this scan (theta, origin), used only by the re-entry memory."""
        p = self.p
        self.t += dt
        self._sel_cache = {}
        self.dbg_pairs = []
        if p.reentry_memory:
            self._reentry_pose(world, ego_dp, ego_dyaw, dt)
        for t in self.tracks:
            if ego_dp is not None:
                t.ego_compensate(ego_dp, ego_dyaw)
            t.kf.predict(dt)
            t.clamp_speed(p)
            if t.hyps:                                              # docs/38: open range hypotheses random-walk
                for h in t.hyps:
                    h["P"] = h["P"] + np.eye(2) * (p.lik_hyp_speed * dt) ** 2
            t.tsu += dt
            t.t_cam += dt
            t.t_radar += dt
            t.since_pos += dt
        for _, tr in self.graveyard:
            if ego_dp is not None:
                tr.ego_compensate(ego_dp, ego_dyaw)
        boxes = np.array([d.box2d for d in dets]) if dets else np.zeros((0, 4))

        high = [d for d in dets if d.score >= p.high_thr]
        low = [d for d in dets if p.low_thr <= d.score < p.high_thr]

        def others(d):
            return boxes[[not np.array_equal(b, d.box2d) for b in boxes]] if len(boxes) else boxes

        # re-entry memory: a track out of view (no camera fix for reentry_coast_s) is a memory, not a candidate of the
        # association stages or of the recovery hook -- its coasting covariance / reach would let it grab anybody's
        # box or radar returns; a re-entry is decided by the memory
        act = [t for t in self.tracks if not self._oov(t)] if p.reentry_memory else self.tracks
        pairs, ut, ud = self._match(act, high)
        for i, j in pairs:
            act[i].update(self._meas(act[i], high[j]), p, self.t, others(high[j]))

        rest = [act[i] for i in ut if act[i].state != TrackState.TENTATIVE]
        pairs2, ut2, _ = self._match(rest, low)
        for i, j in pairs2:
            rest[i].update(self._meas(rest[i], low[j]), p, self.t, others(low[j]))

        if recover is not None and ut2:
            unmatched = [rest[i] for i in ut2]
            occupied = [(t.kf.pos.copy(), t.cls, t.box2d) for t in self.tracks if t.tsu == 0 and t.state != TrackState.TENTATIVE]
            got = recover(unmatched, occupied=occupied)
            for t in unmatched:
                if t.tid not in got:
                    continue
                g = got[t.tid]
                if self._lik(g):                                     # docs/38: the recovered box's range likelihood
                    sel = self._sel(t, g, cache=False)
                    if sel is None or sel.d2 > p.gate_pos:
                        continue
                    g = self._meas(t, g, sel)
                if t.reachable(g, p):
                    t.update(g, p, self.t)

        if p.range_lik and p.lik_reanchor:
            for t in self.tracks:
                if t.tsu == 0.0 or self.t - (t.ra_votes[-1][0] if t.ra_votes else -1e9) > p.lik_reanchor_win:
                    t.ra_votes = []                          # measured this frame, or the disagreement stopped
        for j in ud:
            d = high[j]
            if self._lik(d) and p.lik_reanchor and d.score >= p.high_thr:
                t = self._reanchor_target(d)
                if t is not None:                            # docs/38 §7.4: its own box disagrees with the track
                    r = d.rlik.summ["r"]
                    t.ra_votes = [v for v in t.ra_votes if self.t - v[0] <= p.lik_reanchor_win] + [(self.t, r)]
                    rs = [v[1] for v in t.ra_votes]
                    if len(rs) >= p.lik_reanchor_n and max(rs) - min(rs) <= max(1.0, 0.1 * r):
                        t.reacquire(d, p, self.t, others(d), zero_vel=True)   # the track jumps to that range
                        t.ra_votes = []
                        self.lik_counts["reanchor"] += 1
                    else:
                        self.lik_counts["reanchor_vote"] = self.lik_counts.get("reanchor_vote", 0) + 1
                    continue
            if p.app_reacq and d.score >= p.new_thr:
                t = self._app_reacquire(d)
                if t is not None:
                    m, zv = self._reacq_meas(t, d)
                    t.reacquire(m, p, self.t, others(d), zero_vel=zv)
                    continue
            if p.reacq and d.score >= p.high_thr:
                t = self._reacquire(d, others(d))
                if t is not None:
                    m, zv = self._reacq_meas(t, d)
                    t.reacquire(m, p, self.t, others(d), zero_vel=zv)
                    continue
            if self._lik(d) and p.lik_birth_rules:                   # docs/38 birth gate / same-ray rule
                if self._birth_suppressed_lik(d):
                    continue
                if p.same_ray_suppress and self.project_box is not None and self._same_ray_lik(d):
                    continue
            else:
                if any(t.cls == d.cls and t.kf.mahalanobis_pos(d.pos, t.pos_R(d, p)) <= p.birth_suppress_chi2
                       for t in self.tracks if (t.state != TrackState.TENTATIVE or t.ready) and not self._oov(t)):
                    continue                                         # inside an existing track's gate: not new
                if p.same_ray_suppress and self.project_box is not None and self._same_ray(d):
                    continue
            if high[j].score >= p.new_thr:
                if self._lik(d) and not self._birth_range_ok(d):     # docs/38: range too vague to start a track
                    continue
                self._relinked_from = None
                tid = None if p.reentry_memory else self._relink(high[j])   # re-entry memory: decided at confirmation
                tr = self._new_track(high[j], tid) if self._lik(d) else Track(high[j], p, self.t, tid)
                if self._relinked_from is not None:
                    tr.spans = self._spans(self._relinked_from)
                if self._log_new:
                    self.relink_log[-1]["track"] = tr
                    self._log_new = False
                if tid is not None:
                    tr.state, tr.hits = TrackState.CONFIRMED, p.min_hits
                self.tracks.append(tr)

        if p.reentry_memory:
            self._reentry_step()

        alive = []
        for t in self.tracks:
            if t.tsu > 0:                                           # no measurement this frame
                t.exist *= np.exp(-dt / p.exist_tau_coast_s)
                if t.state == TrackState.TENTATIVE:
                    continue
                t.state = TrackState.LOST
            static_vehicle = t.cls != "worker" and t.speed() < p.static_speed and t.hits >= 5
            max_lost = p.max_lost_static_vehicle_s if static_vehicle else (p.max_lost_static_s if t.static else p.max_lost_s)
            if t.tsu > max_lost or (t.state != TrackState.TENTATIVE and t.t_cam > max(p.max_radar_only_s, max_lost)):
                if t.state != TrackState.TENTATIVE and (t.emb is not None or p.reentry_memory):
                    t.t_dead = self.t - t.tsu
                    self.graveyard.append((self.t, t))
                continue
            alive.append(t)
        self.tracks = self._merge_duplicates(alive)
        if p.reentry_memory:
            t0 = time.perf_counter()
            keep = [(td, tr) for td, tr in self.graveyard if self._mem_alive(tr)]
            if self.relink_log is not None and len(keep) < len(self.graveyard):
                for td, tr in self.graveyard:
                    if not any(tr is k for _, k in keep):
                        s = self._mem_stats(tr)
                        self.relink_log.append(dict(kind="reentry", t=round(self.t, 3), decision="expired", tid=int(tr.tid),
                                                    walk=None if s is None else bool(s["walking"]), static=bool(tr.static),
                                                    life=None if s is None else s["life"], n_wh=len(tr.wh), n_gal=len(tr.gal),
                                                    kf_v=None if s is None else round(s["kf_v"], 2),
                                                    v=None if s is None else s["v"].round(2).tolist()))
            self.graveyard = keep
            self.reentry_s += time.perf_counter() - t0
        else:
            self.graveyard = [(td, tr) for td, tr in self.graveyard if self.t - td < p.relink_window_s]
        return [t.out(self.t) for t in self.tracks if t.state != TrackState.TENTATIVE and t.exist >= p.exist_show]
