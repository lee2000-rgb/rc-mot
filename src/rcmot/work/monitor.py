"""Soil-pile / workload monitor (docs/24): a 2.5D height map in the world frame of rcmot.world (swing-axis model),
filled from the radar's static points and, when the source provides it, the OAK stereo depth scaled by the radar
(stereo reads 3-8 % short, one robust radar/stereo ratio fixes the height bias; docs/23 §4.2).

Per frame: points -> world xy -> per-cell median of this frame -> exponential update of the cell height (so a dug
cell goes down within a few frames). Outputs:
  pile volume   per separate pile: material above that pile's own ground (a plane fitted to the observed ground
                ring around it), over its footprint, empty cells inside the footprint linearly interpolated
                (volume_method "filled", docs/34 §4.1); "threshold" = cells > pile_thr above the base, empty = 0
  cut / fill    volume removed / added against the reference surface (each cell's height once it has settled, or
                reset_reference()): the earthwork done, independent of the ground model
  cycles        swing cycles from the world heading (out + back swings > swing_min_deg), cycle time, cut per cycle
and a height-map snapshot every snapshot_s for the GUI."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NAN_Q = 255                                            # quantised snapshot: no data


@dataclass
class WorkParams:
    enable: bool = False
    cell: float = 0.2                  # [m]
    half_extent: float = 25.0          # grid +-half_extent around the world origin (start of the segment)
    z_min: float = -1.0                # [m] level-frame heights kept
    z_max: float = 2.5                 # bucket / boom hang above this (docs/23 §4.2)
    r_min: float = 2.5                 # [m] horizontal range from the sensor kept (machine body nearer)
    r_max: float = 16.0
    alpha_stereo: float = 0.3          # weight of one frame's cell median
    alpha_radar: float = 0.1
    radar_filter: bool = True          # only radar points the scan statistics keep (clutter / below-ground multipath out,
    radar_min_alpha: float = 0.5       # rcmot.radar.scan_stats) with reliability >= this
    stereo_every: int = 2              # stereo frames used (every n-th radar frame)
    stereo_step: int = 2               # pixel subsampling of the 640x400 depth image
    stereo_max_m: float = 9.0          # stereo error grows ~quadratically (6 % at 9 m); farther: radar only
    stereo_border: float = 0.15        # fraction of the depth image dropped at each side (wide-angle edges read
                                       # 0.3-0.8 m high vs LiDAR on 155530)
    depth: str = "stereo"              # "stereo" (OAK, needs topics.depth) or "vggt_omega" (rcmot.work.depth_engine;
                                       # falls back to stereo when the model is missing / fails)
    vggt_ckpt: str = "models/vggt/vggt_omega_1b_512.pt"
    vggt_views: int = 4                # keyframes per inference (current + earlier ones >= vggt_gap_s apart)
    vggt_gap_s: float = 2.0
    vggt_period_s: float = 0.5         # keyframe interval while the cab is still [s of data]
    vggt_sync: bool = False            # True: infer inline (reproducible offline evaluation), False: background thread
    alpha_engine: float = 0.5
    engine_step: int = 4               # pixel subsampling of the 1280x720 depth map
    engine_max_m: float = 20.0
    mask_machine: bool = True          # boom / bucket colour mask (orange / rust, hanging from the image top edge)
    mask_s_min: int = 55               # HSV saturation of the paint / rust (soil 5-36, 155530)
    mask_dilate: int = 12              # [px, full resolution] margin around the boom / bucket (flying depth at edges)
    mask_drop_px: int = 80             # [px, full res] the mask also covers this far below the bucket (soil falling out
                                       # of it after a dump made a "horn" on the dumped heap)
    pause_bucket_in_pile: bool = True  # no map update while the bucket is down at the pile: the camera ray through the
    bucket_zone_m: float = 8.5         # bucket's lowest pixel meets the ground closer than this [m, from the sensor].
                                       # 155530: held up 9.3-11 m (rows 204-232), digging 5-7.7 m (rows 268-368)
    pause_on_motion: bool = True       # also pause while the bucket / boom moves elsewhere (weak signal, strict thresholds)
    motion_diff: int = 18              # grey-level change counted as motion (1/4 resolution frames)
    motion_frac: float = 0.005         # changed share AFTER a 3x3 opening (scattered noise removed): 0 % false triggers
                                       # on held-bucket stretches (raw 2 % triggered on 3.6 % of quiet frames)
    motion_radar_pts: int = 5          # moving radar points within the reach (people boxes excluded) = bucket moving
    motion_hold_s: float = 0.3         # keep paused this long after the bucket left the pile / the motion stopped
                                       # (1.0 s hid the dump spot: paused 98 % at the dump heading on 155530)
    event_keyframes: bool = True       # bucket leaves the pile / dump spot -> forced keyframe, cut / fill of that event
    event_settle_s: float = 0.4        # wait after the bucket left (falling soil)
    event_max_rise: float = 1.2        # [m] a cell cannot rise more than about a bucket height in one event (volume cap)
    event_confirm_raise: bool = False  # event keyframe raises need the next keyframe: hid the dumped heap (the cab swings
                                       # away, no second look); horns are stopped by horn_thr + waiting for the pour to end
    event_window_s: float = 2.0        # give up if no usable frame within this (the swing usually starts ~2 s later)
    event_max_rate_dps: float = 6.0    # cab may already turn this fast at the forced keyframe (calm image required)
    event_sweep_dps: float = 30.0      # ... or up to this fast once event_sweep_wait_s passed: the swing back has started,
    event_sweep_wait_s: float = 0.3    # the pour is over and the heap slides out from under the bucket (the bucket turns
                                       # with the camera); pose corrected to the image time
    event_sweep_window_s: float = 20.0 # after the first event keyframe, a second one when the cab next starts to turn
                                       # (155530: pour seen at 136.9 s as calm, swing back at 145 s)
    event_thr: float = 0.1             # [m] cell change counted in an event's cut / fill
    event_min_down_s: float = 1.0      # the bucket must have been down this long (0.3 s flickers made fake events) ...
    event_min_out_s: float = 0.5       # ... and out this long before the event keyframe
    event_radius: float = 4.0          # [m] cut / fill counted around where the bucket was (its lowest point's ground;
                                       # a bucket held over the dump spot puts that point a few m behind the heap)
    event_ring: tuple = (5.0, 9.0)     # [m] ring around it whose median change (should be 0) is the keyframe offset
    event_local_m: float = 4.0         # forced keyframe writes at full weight only within this distance of the event site;
                                       # elsewhere its points take the regular keyframe path (alpha_engine, jumps confirmed).
                                       # 0 = whole frame at full weight (before 11-03: forced keyframes at the dig site
                                       # lifted flat ground 0.1-0.3 m, pile volume +2..+7 m3, docs/39 §4.1)
    kf_ring_check: bool = False        # forced keyframe vs the map on the kf_ring_m ring around the event site (should
                                       # not have changed): fit a plane to (keyframe - map) there and subtract it from the
                                       # keyframe (tilted depth); drop the keyframe if the ring still disagrees.
                                       # Off by default: on 155530 vs LiDAR it changed nothing on volume (+32 % either
                                       # way), cut the >2 m3 jumps 12 -> 9 but dropped 4 of 24 event keyframes (docs/39 §8)
    kf_ring_m: tuple = (5.0, 9.0)      # [m] that ring (distinct from ground_ring, the per-pile ground ring)
    kf_ring_min: int = 40              # ring cells needed for the plane (fewer: no correction, no drop)
    kf_ring_max_deg: float = 1.5       # plane steeper than this -> drop the keyframe (an arc-only ring makes a steep fit
                                       # that extrapolates badly: 2 deg fits at the 155530 dump spot added 6 m3)
    kf_ring_max_res: float = 0.12      # [m] 1.4826 MAD of the ring residual after the plane above this -> drop
    flying_grad: float = 0.02          # drop depth pixels whose relative depth jump per pixel exceeds this (object
                                       # edges: height error p90 0.5-1.5 m, ~1 % of pixels)
    jump_thr: float = 0.3              # [m] a keyframe change larger than this is applied only when the next keyframe
                                       # confirms it (within jump_thr); single-frame jumps are dropped
    spike_thr: float = 0.4             # [m] a keyframe cell this far from its 3x3 neighbours' mean is dropped
    scale_alpha: float = 0.1           # EMA of the radar / stereo depth ratio
    pile_thr: float = 0.2              # [m] a cell counts as pile this far above the ground plane (pile cores)
    volume_method: str = "filled"      # "filled": per pile, local ground plane from the observed ring around it, footprint
                                       # grown over the toe (> pile_edge), closed, empty cells interpolated, no threshold;
                                       # "threshold": sum of cells > pile_thr above the base, empty cells = 0 (before
                                       # 11-02; on a real LiDAR pile 7.1 vs 9.0 m3: 30 % empty cells, calib ground 7-8 cm
                                       # low and 1.5-1.8 deg tilted, the thin toe; docs/34 §4.1)
    pile_edge: float = 0.05            # [m] the footprint grows from the cores over cells this high above the local ground
    pile_close: int = 3                # [cells] footprint closing: bridges empty cells (LiDAR rings, boom shadow)
    ground_ring: tuple = (3, 10)       # [cells] ring around the core footprint whose observed cells give the pile's ground
    ground_ring_min: int = 30          # fewer ring cells (or a fit steeper than ground_max_deg) -> the global base
    ground_max_deg: float = 6.0
    partial_in_pile: bool = False      # while the bucket rests in the pile (cab still, no image / radar motion) keep taking
                                       # regular keyframes, without their points within partial_m of the bucket's ground
                                       # point (155530: the bucket-in-pile gate froze the whole map 52 % of the frames)
    partial_m: float = 3.0
    ground_radar: bool = False         # shift each pile's ring plane by the robust median height of the radar static points
                                       # that fell on its ring (radar heights are unbiased on the ground; the camera ring
                                       # plane sat ~5-10 cm low at the 155530 pile, +25 % volume; docs/40)
    ground_radar_win_s: float = 240.0  # [s] radar points of this long back (only those fused into the map: cab still,
                                       # map not paused)
    ground_radar_min: int = 200        # ring radar points needed for the shift
    ground_radar_clip: float = 0.2     # [m] the shift is limited to +-this
    ground_radar_tau_s: float = 5.6    # [s] the shift of a pile (identity: core centre within 2 m) follows its new values
                                       # with this time constant and is held while its ring has too few radar points
                                       # (unsmoothed: 2 m3 volume jumps 12 -> 22 on 155530 when the shift came and went)
    volume_every_s: float = 0.5        # pile analysis at most this often (labels, plane fits, interpolation ~5-20 ms)
    ground: str = "calib"              # "calib": z = 0 of the calibration (the install's LiDAR ground, field_calib reground);
                                       # "fit": dominant plane of the map (RANSAC; stereo noise makes it less reliable)
    ground_local_m: float = 7.0        # where the terrain is LOWER than that plane (dump spot on 155530: -0.15 m), the
    ground_local_min: float = 0.05     # ground is lowered to the median of the ground-like cells (within 0.3 m of the
                                       # plane) in this window, if that is more than ground_local_min lower (never raised:
                                       # the dug pile keeps its volume; a low percentile sat ~0.1 m under noisy ground); 0 = off
    ref_min_obs: int = 15              # cell observations before its reference height is frozen
    min_obs: int = 1                   # a cell is drawn / counted only after this many observations (2 hid 43 % of cells)
    horn_thr: float = 0.6              # [m] a cell this far above the median of its 1.4 m neighbourhood is not soil (steeper
                                       # than the angle of repose ~35-40 deg): bucket pouring soil made a 2.2 m "horn" on the
                                       # dump spot (155530 136.8 s); excluded from display and volume
    change_thr: float = 0.2            # [m] cut / fill dead band (stereo banding ~0.2 m at 9 m)
    update_max_rate_dps: float = 3.0   # map updated only while the cab turns slower than this (digging / dumping);
                                       # during swings timing and motion blur misplace points (docs/24)
    zone: list | None = None           # [x0, x1, y0, y1] world volume zone; None = the reach circle below
    reach_m: float = 9.0               # volume / cut / fill counted within this distance of the swing axis (the bucket's
                                       # reach; farther cells come from sparse radar only, height MAE 0.4-0.6 m); 0 = all
    snapshot_s: float = 1.0
    pile_min_m3: float = 0.3           # separate piles (8-connected pile cells) smaller than this get no GUI label
    swing_rate_dps: float = 8.0        # |yaw rate| above this = swinging
    swing_min_deg: float = 30.0        # a swing shorter than this is not a cycle half
    q_step: float = 0.02               # [m] snapshot quantisation (uint8, -1 m offset)
    axis_tilt: bool = True             # turn the points about the swing axis by the calib's swing_axis_level (the level
                                       # frame is the ground in front, ~2 deg off the axis on 155530: the same ground read
                                       # 0.25 m low / 0.3 m high 73 deg left / right; scripts/swing_axis_calib.py, docs/25 §3.1)


@dataclass
class HeightSnapshot:
    t: float
    q: np.ndarray                      # (n, n) uint8 heights ((h + 1) / q_step), NAN_Q = no data
    dq: np.ndarray                     # (n, n) int8 height change vs the reference (q_step units), -128 = no reference
    cell: float
    half_extent: float
    q_step: float
    series_t: np.ndarray               # history up to t (float64 stamps): pile volume, cut, fill [m3]
    series_v: np.ndarray
    series_cut: np.ndarray
    series_fill: np.ndarray
    cycles_t: np.ndarray               # cycle completion times
    plane: tuple                       # ground plane z = a x + b y + c (world xy)
    rgb: np.ndarray | None = None      # (n, n, 3) uint8 camera colour per cell (RGB), 0 = none (3D view)
    piles: tuple = ()                  # per separate pile, largest first: (x, y, z_top, volume m3, area m2, height m,
                                       # interpolated share); x, y = volume-weighted centre (world), z_top = highest cell

    def heights(self) -> np.ndarray:
        h = self.q.astype(np.float32) * self.q_step - 1.0
        h[self.q == NAN_Q] = np.nan
        return h

    def ground(self) -> np.ndarray:
        n = self.q.shape[0]
        c = (np.arange(n) + 0.5) * self.cell - self.half_extent
        X, Y = np.meshgrid(c, c, indexing="ij")
        a, b, c0 = self.plane
        return (a * X + b * Y + c0).astype(np.float32)

    def change(self) -> np.ndarray:
        d = self.dq.astype(np.float32) * self.q_step
        d[self.dq == -128] = np.nan
        return d


@dataclass
class WorkState:
    t: float
    volume: float                      # [m3] pile above the fitted ground plane
    area: float                        # [m2] pile cells
    observed: float                    # [m2] cells with any height
    cut: float                         # [m3] removed against the reference surface
    fill: float                        # [m3] added
    n_cycles: int
    cycle_s: float                     # mean cycle time [s] (nan before two cycles)
    last_dcut: float                   # cut during the last cycle [m3] (~ one bucket when digging)
    scale: float                       # radar / stereo depth ratio in use
    swinging: bool
    snapshot: HeightSnapshot | None    # latest snapshot at or before t (shared reference)
    n_stereo_pts: int = 0              # dense-depth points fused this frame (stereo or engine)
    n_radar_pts: int = 0
    source: str = "stereo"             # dense depth in use: vggt_omega / stereo / radar (radar only)
    n_engine: int = 0                  # engine keyframes fused so far
    engine_ms: float = 0.0
    active: bool = False               # bucket in the pile or bucket / boom moving -> map updates paused
    bucket_in_pile: bool = False
    paused_frac: float = 0.0           # share of the cab-still frames paused for bucket motion
    mask_frac: float = 0.0             # share of the image masked as boom / bucket
    events: tuple = ()                 # per bucket event so far: (t, kind "dig"/"dump"[+sweep], cut m3, fill m3, heading deg,
                                       # t_down, x, y, keyframe offset m, site pile volume before m3, after m3)
    last_event: tuple | None = None
    ledger: tuple = ()                 # per work cycle (scoop -> put down), WorkMonitor.work_cycles()
    t_map: float = float("nan")        # time of the last dense-depth map update (keyframe / stereo); GUI shows its age
    fill_frac: float = 0.0             # share of the pile footprints interpolated (not observed), volume_method "filled"


class WorkMonitor:
    def __init__(self, params: WorkParams, calib, engine=None):
        """engine: rcmot.work.depth_engine.VGGTOmegaEngine (shared across resets, it keeps the model loaded)."""
        self.p, self.calib, self.engine = params, calib, engine
        self.n = int(round(2 * params.half_extent / params.cell))
        a = getattr(calib, "swing_axis_level", None)
        self.R_axis = None
        if params.axis_tilt and a is not None:                     # minimal rotation taking the swing axis onto z
            a = np.asarray(a, float) / np.linalg.norm(a)
            v, c = np.cross(a, [0.0, 0.0, 1.0]), float(a[2])
            K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            self.R_axis = np.eye(3) + K + K @ K / (1.0 + c)
        self.reset()

    def reset(self):
        n = self.n
        self.H = np.full((n, n), np.nan, np.float32)
        self.last_obs = np.full((n, n), -np.inf)
        self.scale = 1.0
        self.k = 0
        self.snap: HeightSnapshot | None = None
        self.t_snap = -np.inf
        self.series_t: list[float] = []
        self.series_v: list[float] = []
        self.swing = None                  # [t0, angle] of the swing in progress
        self.calm_t = 0.0
        self.swings: list[tuple[float, float]] = []      # (t_end, signed angle)
        self.cycles: list[tuple[float, float]] = []      # (t_done, volume at completion)
        self.center = None                                          # swing axis (or sensor) in the world, for reach_m
        self.n_obs = np.zeros((self.n, self.n), np.int32)
        self.base = np.zeros((self.n, self.n), np.float32)          # ground under each cell (plane, lowered locally)
        self.spike = np.zeros((self.n, self.n), bool)               # isolated cells (map level), out of volume / display
        self.C = np.full((self.n, self.n, 3), np.nan, np.float32)    # camera colour per cell (BGR), for the 3D view
        self.H_ref = np.full((self.n, self.n), np.nan, np.float32)
        self.plane = (0.0, 0.0, 0.0)                                # ground z = a x + b y + c (world xy)
        self.plane_set = self.R_axis is None
        c = (np.arange(self.n) + 0.5) * self.p.cell - self.p.half_extent
        self.X, self.Y = np.meshgrid(c, c, indexing="ij")
        self.series_cut: list[float] = []
        self.series_fill: list[float] = []
        self.t_submit = -np.inf
        self.n_engine = 0
        self.prev_small = None                                      # 1/4-resolution grey of the previous frame
        self.motion_outside = 0.0
        self.mask_core = None
        self.pend_h = np.full((self.n, self.n), np.nan, np.float32)  # unconfirmed large changes (jump_thr)
        self.in_pile = False
        self.was_down = False
        self.armed = None                                           # event waiting for a forced keyframe
        self.t_down = None                                          # bucket down since
        self.t_out = None                                           # bucket out since (after a long-enough down)
        self.bucket_ground = None
        self.down_pts: list = []                                    # world ground points of the bucket while down
        self.events: list = []
        self.active_until = -np.inf
        self.n_still = self.n_paused = 0
        self.mask_small = None
        self.t_map = float("nan")
        self.n_kf_dropped = 0                                       # forced keyframes dropped by kf_ring_check
        self.kf_ring_last = None                                    # (a, b, c, deg, mad, n) of the last ring fit
        self.pa = None                                              # last pile analysis (_pile_analysis)
        self.n_partial = 0                                          # partial keyframes submitted (partial_in_pile)
        self.t_moved = -np.inf                                      # last image / radar motion (_motion, without holds)
        self.rad_buf = []                                           # (t, world xy, z) of fused radar points (ground_radar)
        self.g_off_state = []                                       # [(x, y, shift, t)] per pile (ground_radar smoothing)
        self.t_pa = -np.inf
        self.map_ver = 0                                            # bumped by every map update (re-analyse when changed)
        self.pa_ver = -1
        if self.engine is not None:
            self.engine.reset()

    def reset_reference(self):
        """Take the current surface as the reference for cut / fill (e.g. at the start of a shift)."""
        self.H_ref = np.where(np.isfinite(self.H), self.H, np.nan).astype(np.float32)
        self.cycles = []

    # ------------------------------------------------------------------ inputs
    def _axis(self, P: np.ndarray, world) -> np.ndarray:
        """Level-frame points -> frame whose z is the swing axis (turned about the axis), so that turning the cab
        about z (the world pose) keeps the heights of the same ground."""
        if self.R_axis is None or not len(P):
            return P
        c = np.zeros(3)
        if world.lever is not None:
            c[:2] = -world.lever                                    # swing axis in the level frame (z = ground)
        return (P - c) @ self.R_axis.T + c

    def _set_plane(self, world):
        """The calib ground (level z = 0, seen at the segment start) in the axis frame, as a world plane."""
        n = self.R_axis @ np.array([0.0, 0.0, 1.0])
        k = -n[:2] / n[2]                                           # z = k . (p - c), p level xy
        c = -world.lever if world.lever is not None else np.zeros(2)
        kw = world.R() @ k                                          # level -> world (p = R^T (w - origin))
        self.plane = (float(kw[0]), float(kw[1]), float(-kw @ world.origin - k @ c))
        self.plane_set = True

    def _cells(self, xy: np.ndarray):
        p = self.p
        ij = np.floor((xy + p.half_extent) / p.cell).astype(int)
        ok = (ij >= 0).all(1) & (ij < self.n).all(1)
        return ij[ok], ok

    def _fuse(self, xy: np.ndarray, z: np.ndarray, alpha: float, t: float, robust: bool = False, confirm: bool = True) -> int:
        """Per-cell median of this frame's points, then H <- (1 - alpha) H + alpha median (new cells: median)."""
        if not len(z):
            return 0
        ij, ok = self._cells(xy)
        z = z[ok]
        if not len(z):
            return 0
        key = ij[:, 0] * self.n + ij[:, 1]
        order = np.lexsort((z, key))                       # by cell, then height: the middle element = cell median
        key, z = key[order], z[order]
        uk, start, cnt = np.unique(key, return_index=True, return_counts=True)
        med = z[start + (cnt - 1) // 2].astype(np.float32)  # (lower) median, vectorised
        if robust:
            uk, med = self._robust(uk, med, confirm)
            if not len(uk):
                return 0
        old = self.H.flat[uk]
        self.H.flat[uk] = np.where(np.isfinite(old), (1 - alpha) * old + alpha * med, med)
        self.map_ver += 1
        self.last_obs.flat[uk] = t
        self.n_obs.flat[uk] += 1
        new_ref = np.isnan(self.H_ref.flat[uk]) & (self.n_obs.flat[uk] >= self.p.ref_min_obs)
        self.H_ref.flat[uk[new_ref]] = self.H.flat[uk[new_ref]]
        return len(z)

    def _frame_grid(self, xy: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Cell medians of one set of points (NaN elsewhere), without touching the map."""
        G = np.full((self.n, self.n), np.nan, np.float32)
        ij, ok = self._cells(xy)
        if len(ij):
            key = ij[:, 0] * self.n + ij[:, 1]
            zz = z[ok]
            o = np.lexsort((zz, key))
            key, zz = key[o], zz[o]
            uk, st, cnt = np.unique(key, return_index=True, return_counts=True)
            G.flat[uk] = zz[st + (cnt - 1) // 2]
        return G

    def _robust(self, uk: np.ndarray, med: np.ndarray, confirm=True):
        """Keyframe cells: (1) drop cells farther than spike_thr from the mean of their observed 3x3 neighbours in this
        keyframe; (2) a change larger than jump_thr is applied only when the next keyframe confirms it."""
        import cv2

        p, n = self.p, self.n
        G = np.zeros((n, n), np.float32)
        M = np.zeros((n, n), np.float32)
        G.flat[uk], M.flat[uk] = med, 1.0
        S = cv2.boxFilter(G, -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT).flat[uk] - med
        C = cv2.boxFilter(M, -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT).flat[uk] - 1.0
        ok = (C < 3) | (np.abs(med - S / np.maximum(C, 1.0)) <= p.spike_thr)
        uk, med = uk[ok], med[ok]
        if confirm is False:
            self.pend_h.flat[uk] = np.nan
            return uk, med
        old = self.H.flat[uk]
        big = np.isfinite(old) & (np.abs(med - old) > p.jump_thr)
        if confirm == "down":                                       # event keyframe: dug cells at once, raises confirmed
            big &= med > old
        pend = self.pend_h.flat[uk]
        confirmed = big & np.isfinite(pend) & (np.abs(med - pend) <= p.jump_thr)
        self.pend_h.flat[uk[big & ~confirmed]] = med[big & ~confirmed]        # remember, apply when seen again
        self.pend_h.flat[uk[~big | confirmed]] = np.nan
        take = ~big | confirmed
        med = med.copy()
        med[confirmed] = 0.5 * (med[confirmed] + pend[confirmed])
        # confirmed jumps replace the cell (alpha is applied by the caller; a 2-keyframe agreement is trusted)
        self.H.flat[uk[confirmed]] = med[confirmed]
        return uk[take & ~confirmed], med[take & ~confirmed]

    def _fuse_color(self, xy: np.ndarray, bgr: np.ndarray, alpha: float):
        """Per-cell mean colour of this frame -> exponential update (like the heights)."""
        ij, ok = self._cells(xy)
        if not len(ij):
            return
        key = ij[:, 0] * self.n + ij[:, 1]
        col = bgr[ok].astype(np.float32)
        cnt = np.bincount(key, minlength=self.n * self.n)
        has = np.flatnonzero(cnt)
        mean = np.stack([np.bincount(key, weights=col[:, c], minlength=self.n * self.n)[has] / cnt[has] for c in range(3)], 1)
        C = self.C.reshape(-1, 3)
        old = C[has]
        C[has] = np.where(np.isfinite(old), (1 - alpha) * old + alpha * mean, mean)

    MASK_F = 4                                                      # mask / motion at 1/4 resolution

    def machine_mask(self, bgr: np.ndarray, small_given: bool = False) -> np.ndarray:
        """Boom / bucket at 1/MASK_F resolution: orange / rust pixels (H <= 22 or >= 172, S >= mask_s_min), closed,
        components touching the image top edge (the camera is on the cab: the boom always enters from the top),
        convex hull (the dark bucket inside), dilated."""
        import cv2

        f, p = self.MASK_F, self.p
        small = bgr if small_given else cv2.resize(bgr, (bgr.shape[1] // f, bgr.shape[0] // f), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        m = (((H <= 22) | (H >= 172)) & (S >= p.mask_s_min) & (V >= 50)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31 // f | 1, 31 // f | 1)))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        out = np.zeros_like(m)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_TOP] <= 2 and st[i, cv2.CC_STAT_AREA] >= 800 // (f * f):
                cnt, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(out, [cv2.convexHull(c) for c in cnt], -1, 1, -1)
        d = max(p.mask_dilate // f, 1)
        out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1)))
        self.mask_core = out.astype(bool)                           # boom / bucket only (event keyframes, motion)
        drop = p.mask_drop_px // f
        if drop > 0 and out.any():                                  # falling soil / shadow below the bucket
            ext = out.copy()
            for k in range(1, drop + 1):
                ext[k:] |= out[:-k]
            out = ext
        return out.astype(bool)

    def _in_mask(self, mask, u, v) -> np.ndarray:
        """Full-resolution pixel coordinates -> inside the 1/MASK_F mask."""
        shape = np.shape(u)
        if mask is None:
            return np.zeros(shape, bool)
        f = self.MASK_F
        uu, vv = (np.ravel(u) // f).astype(int), (np.ravel(v) // f).astype(int)
        ok = (uu >= 0) & (uu < mask.shape[1]) & (vv >= 0) & (vv < mask.shape[0])
        out = np.zeros(uu.shape, bool)
        out[ok] = mask[vv[ok], uu[ok]]
        return out.reshape(shape)

    def _project(self, P_level: np.ndarray):
        cal = self.calib
        pc = P_level @ cal.R_cl.T + cal.t_cl
        z = np.maximum(pc[:, 2], 1e-3)
        return cal.K[0, 0] * pc[:, 0] / z + cal.K[0, 2], cal.K[1, 1] * pc[:, 1] / z + cal.K[1, 2], pc[:, 2] > 0.3

    def _motion(self, t: float, image, dyn_level: np.ndarray, boxes: list, mask=None) -> bool:
        """Bucket / boom moving: image change between consecutive frames or moving radar points in the reach, people
        excluded. Returns True while paused (motion now or within motion_hold_s)."""
        import cv2

        p, f = self.p, self.MASK_F
        moving = False
        if image is not None:                                       # the shared 1/MASK_F frame
            g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.int16)
            if self.prev_small is not None and self.prev_small.shape == g.shape:
                ch = (np.abs(g - self.prev_small) > p.motion_diff).astype(np.uint8)
                for b in boxes:
                    ch[int(b[1]) // f:int(b[3]) // f + 1, int(b[0]) // f:int(b[2]) // f + 1] = 0
                ch = cv2.morphologyEx(ch, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))   # scattered noise out
                moving |= ch.mean() > p.motion_frac
                # outside the boom / bucket: the bucket lifting away does not count, soil still pouring below it does
                self.motion_outside = float(ch[~mask].mean()) if mask is not None and mask.shape == ch.shape else float(ch.mean())
            self.prev_small = g
        if len(dyn_level):
            r = np.linalg.norm(dyn_level[:, :2], axis=1)
            k = (r <= p.reach_m + 2.0) & (dyn_level[:, 2] > 0.3)
            if k.any() and boxes:
                u, v, fr = self._project(dyn_level[k])
                inside = np.zeros(len(u), bool)
                for b in boxes:
                    inside |= (u >= b[0]) & (u <= b[2]) & (v >= b[1]) & (v <= b[3])
                kk = np.flatnonzero(k)
                k[kk[inside & fr]] = False
            moving |= int(k.sum()) >= p.motion_radar_pts
        if moving:
            self.active_until = t + p.motion_hold_s
            self.t_moved = t                                        # real motion (the hold below also covers in_pile)
        return t <= self.active_until

    def _bucket_down(self, mask: np.ndarray) -> bool:
        """Bucket at the pile: the ground point of the ray through the mask's lowest pixel is within bucket_zone_m
        (a raised bucket's ray reaches the ground far behind it; the arm cannot reach that far)."""
        rows = np.flatnonzero(mask.any(1))
        if not len(rows):
            return False
        r = rows.max()
        u = (np.flatnonzero(mask[r]).mean() + 0.5) * self.MASK_F
        g = self.calib.pixel_ray_to_ground(u, (r + 1) * self.MASK_F)
        self.bucket_ground = None if g is None else np.asarray(g[:2], float)       # level frame
        return g is not None and float(np.hypot(g[0], g[1])) < self.p.bucket_zone_m

    def _update_base(self):
        """Ground per cell: the plane, lowered (never raised) to the local low percentile of the map within
        ground_local_m (coarse 1 m grid, percentile filter); plus the map-level spike mask (cells farther than
        spike_thr from the mean of >= 4 observed 3x3 neighbours)."""
        import cv2
        from scipy.ndimage import zoom

        p = self.p
        plane = self._ground_grid().astype(np.float32)
        base = plane
        if p.ground_local_m > 0:
            from scipy.ndimage import generic_filter

            f = max(int(round(1.0 / p.cell)), 1)
            m = self.n // f
            dev = np.where(np.abs(self.H - plane) < 0.3, self.H - plane, np.nan)[:m * f, :m * f]
            blk = dev.reshape(m, f, m, f).transpose(0, 2, 1, 3).reshape(m, m, f * f)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                c = np.nanmedian(blk, axis=2)                       # ground-like offset per 1 m block
                w = max(int(round(p.ground_local_m)) | 1, 3)
                g = generic_filter(c, lambda v: np.nanmedian(v) if np.isfinite(v).sum() >= 5 else 0.0, size=w, mode="nearest")
            g = np.where(np.isfinite(g) & (g < -p.ground_local_min), g, 0.0).astype(np.float32)
            loc = np.zeros_like(plane)
            z = zoom(g, f, order=1, mode="nearest", grid_mode=True)
            loc[:z.shape[0], :z.shape[1]] = z[:self.n, :self.n]
            base = plane + np.minimum(loc, 0.0)
        self.base = base
        self.map_ver += 1                                           # the pile analysis depends on the base too
        fin = np.isfinite(self.H)
        Hz = np.where(fin, self.H, 0).astype(np.float32)
        S = cv2.boxFilter(Hz, -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT) - Hz
        C = cv2.boxFilter(fin.astype(np.float32), -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT) - fin
        self.spike = fin & (C >= 4) & (np.abs(self.H - S / np.maximum(C, 1)) > p.spike_thr)
        if p.horn_thr > 0 and fin.any():                            # steeper than soil can stand -> not soil
            from scipy.ndimage import median_filter

            med = median_filter(np.where(fin, self.H, np.nanmedian(self.H[fin])).astype(np.float32), size=7)
            self.spike |= fin & ((self.H - med) > p.horn_thr)

    @staticmethod
    def _pose_at(world, dt: float):
        """World pose dt seconds later (constant yaw rate; the swing axis stays, the sensor turns about it)."""
        from dataclasses import replace

        from rcmot.world import rot2
        th = world.theta + world.w * dt
        org = world.axis + rot2(th) @ world.lever if world.axis is not None and world.lever is not None else world.origin
        return replace(world, theta=th, origin=org)

    def _ground_grid(self) -> np.ndarray:
        a_, b_, c_ = self.plane
        return a_ * self.X + b_ * self.Y + c_

    def _keep(self, P_level: np.ndarray) -> np.ndarray:
        p = self.p
        r = np.linalg.norm(P_level[:, :2], axis=1)
        return (r >= p.r_min) & (r <= p.r_max) & (P_level[:, 2] >= p.z_min) & (P_level[:, 2] <= p.z_max)

    def _stereo(self, depth, radar_static: np.ndarray, boxes: list, world, t: float, mask=None) -> int:
        """depth = (D (h, w) metres, Kd 3x3) aligned to the RGB optical frame."""
        p, cal = self.p, self.calib
        D, Kd = depth
        h, w = D.shape
        # radar / stereo scale from the radar's static points projected into the depth image
        if len(radar_static):
            pc = radar_static @ cal.R_cl.T + cal.t_cl
            pc = pc[(pc[:, 2] > 1.0) & (pc[:, 2] < p.stereo_max_m)]
            u = np.round(Kd[0, 0] * pc[:, 0] / pc[:, 2] + Kd[0, 2]).astype(int)
            v = np.round(Kd[1, 1] * pc[:, 1] / pc[:, 2] + Kd[1, 2]).astype(int)
            m = (u >= 1) & (u < w - 1) & (v >= 1) & (v < h - 1)
            rat = []
            for uu, vv, zz in zip(u[m], v[m], pc[m, 2]):
                win = D[vv - 1:vv + 2, uu - 1:uu + 2]
                win = win[win > 0.3]
                if len(win):
                    rat.append(zz / np.median(win))
            if len(rat) >= 10:
                s = float(np.median(rat))
                if 0.8 < s < 1.25:
                    self.scale += p.scale_alpha * (s - self.scale)
        st = max(p.stereo_step, 1)
        bh, bw = int(h * p.stereo_border), int(w * p.stereo_border)
        vv, uu = np.mgrid[bh:h - bh:st, bw:w - bw:st]
        z = D[vv, uu] * self.scale
        k = (z > 0.3) & (z < p.stereo_max_m)
        z, uu, vv = z[k], uu[k], vv[k]
        pc = np.stack([(uu - Kd[0, 2]) / Kd[0, 0] * z, (vv - Kd[1, 2]) / Kd[1, 1] * z, z], 1)
        if boxes or mask is not None:                     # drop pixels inside person boxes / boom-bucket mask (RGB image)
            K = cal.K
            ur, vr = K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]
            keep = ~self._in_mask(mask, ur, vr)
            for b in boxes:
                keep &= ~((ur >= b[0]) & (ur <= b[2]) & (vr >= b[1]) & (vr <= b[3]))
            pc = pc[keep]
        P = (pc - cal.t_cl) @ cal.R_cl                   # camera optical -> level
        P = self._axis(P[self._keep(P)], world)
        return self._fuse(world.to_world(P), P[:, 2], p.alpha_stereo, t)

    def _engine_depth(self, job) -> int:
        """Relative depth of a keyframe -> metric with the radar (median ratio at the radar points, 3x3 window),
        back-projected with the calibrated K, people removed, fused with the keyframe's own world pose."""
        p, cal = self.p, self.calib
        D, K = job.depth, cal.K
        h, w = D.shape
        pc = job.radar_level @ cal.R_cl.T + cal.t_cl if len(job.radar_level) else np.zeros((0, 3))
        pc = pc[(pc[:, 2] > 1.0) & (pc[:, 2] < p.engine_max_m)]
        u = np.round(K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]).astype(int)
        m = (u >= 1) & (u < w - 1) & (v >= 1) & (v < h - 1)
        rat = [zz / np.median(D[vv - 1:vv + 2, uu - 1:uu + 2]) for uu, vv, zz in zip(u[m], v[m], pc[m, 2])]
        rat = [r for r in rat if np.isfinite(r) and r > 0]
        if len(rat) < 10:
            return 0
        s = float(np.median(rat))
        st = max(p.engine_step, 1)
        vv, uu = np.mgrid[0:h:st, 0:w:st]
        z = D[vv, uu] * s
        keep = (z > 0.5) & (z < p.engine_max_m)
        if p.flying_grad > 0:                                       # object edges: in-between depths ("flying pixels")
            gy, gx = np.gradient(D)
            keep &= (np.hypot(gx, gy) / np.maximum(D, 1e-3))[vv, uu] < p.flying_grad
        for b in job.boxes:
            keep &= ~((uu >= b[0]) & (uu <= b[2]) & (vv >= b[1]) & (vv <= b[3]))
        if p.mask_machine:                                          # boom / bucket of THIS keyframe
            m_ = self.machine_mask(job.image)
            if job.forced and job.event and job.event.get("mask") is not None:
                m_ = job.event["mask"]                              # pour over: the heap right under the bucket is seen
            keep &= ~self._in_mask(m_, uu, vv)
        z, uu, vv = z[keep], uu[keep], vv[keep]
        pcam = np.stack([(uu - K[0, 2]) / K[0, 0] * z, (vv - K[1, 2]) / K[1, 1] * z, z], 1)
        P = (pcam - cal.t_cl) @ cal.R_cl
        k2 = self._keep_engine(P)
        P = self._axis(P[k2], job.world)
        self.engine_scale = s
        xy = job.world.to_world(P)
        cu, cv = uu[k2], vv[k2]
        ex = None if job.forced else (job.event or {}).get("exclude")
        if ex is not None:                                          # partial keyframe: nothing near the bucket in the pile
            far = np.hypot(xy[:, 0] - ex[0][0], xy[:, 1] - ex[0][1]) > ex[1]
            xy, P, cu, cv = xy[far], P[far], cu[far], cv[far]
        if job.image.shape[:2] == D.shape:
            self._fuse_color(xy, job.image[cv, cu], 1.0 if job.forced else p.alpha_engine)
        if not job.forced:
            return self._fuse(xy, P[:, 2], p.alpha_engine, job.t, robust=True)
        before = self.H.copy()                                      # forced: dug cells at once, raises confirmed later
        ev = job.event or {}
        where = ev.get("where")
        z = P[:, 2]
        if where is not None and p.kf_ring_check:                   # keyframe vs map on the should-not-change ring
            corr = self._ring_plane(xy, z, before, where)
            if corr is None:
                self.n_kf_dropped += 1
                return 0
            z = z - corr
        frame = self._frame_grid(xy, z)                             # this keyframe's own cell medians (event volume)
        confirm = "down" if p.event_confirm_raise else False
        if where is not None and p.event_local_m > 0:              # full weight only around the event site
            near = np.hypot(xy[:, 0] - where[0], xy[:, 1] - where[1]) <= p.event_local_m
            n = self._fuse(xy[near], z[near], 1.0, job.t, robust=True, confirm=confirm)
            n += self._fuse(xy[~near], z[~near], p.alpha_engine, job.t, robust=True)
        else:
            n = self._fuse(xy, z, 1.0, job.t, robust=True, confirm=confirm)
        if where is None:
            return n
        d = np.minimum(frame - before, p.event_max_rise)
        rr = np.hypot(self.X - where[0], self.Y - where[1])
        ring = np.isfinite(d) & (rr >= p.event_ring[0]) & (rr <= p.event_ring[1])
        off = float(np.median(d[ring])) if ring.sum() >= 20 else 0.0      # keyframe-to-map height offset
        d = d - off
        ok = np.isfinite(d) & (rr <= p.event_radius)
        a = p.cell ** 2
        cut = float(-d[ok & (d < -p.event_thr)].sum() * a)
        fill = float(d[ok & (d > p.event_thr)].sum() * a)
        kind = ("dig" if cut >= fill else "dump") + ("+sweep" if ev.get("phase") == 2 else "")
        v0, v1 = self._site_volume(before, where), self._site_volume(self.H, where, before)
        self.events.append((job.t, kind, cut, fill, float(np.degrees(job.world.theta)), ev.get("t_down", np.nan),
                            float(where[0]), float(where[1]), off, v0, v1))
        return n

    def _ring_plane(self, xy: np.ndarray, z: np.ndarray, H: np.ndarray, where):
        """Forced keyframe check (kf_ring_check): cell medians of the keyframe minus the map H on the kf_ring_m ring around
        the event site, where nothing should have changed. A plane d = a x + b y + c is fitted (trimmed least squares);
        returns its value at every point (to subtract from z), 0 when the ring is too sparse to judge, or None when the
        keyframe should be dropped (plane steeper than kf_ring_max_deg or the residual 1.4826 MAD above kf_ring_max_res)."""
        p = self.p
        d = self._frame_grid(xy, z) - H
        rr = np.hypot(self.X - where[0], self.Y - where[1])
        ring = np.isfinite(d) & (rr >= p.kf_ring_m[0]) & (rr <= p.kf_ring_m[1])
        self.kf_ring_last = None
        if ring.sum() < p.kf_ring_min:
            return np.zeros(len(z), np.float32)
        A = np.c_[self.X[ring], self.Y[ring], np.ones(ring.sum())]
        v = d[ring].astype(np.float64)
        w = np.ones(len(v), bool)
        for _ in range(3):
            sol = np.linalg.lstsq(A[w], v[w], rcond=None)[0]
            res = v - A @ sol
            w = np.abs(res) < max(3 * 1.4826 * np.median(np.abs(res[w])), 0.03)
        res = v[w] - A[w] @ sol
        mad = float(1.4826 * np.median(np.abs(res - np.median(res))))
        deg = float(np.degrees(np.arctan(np.hypot(sol[0], sol[1]))))
        self.kf_ring_last = (float(sol[0]), float(sol[1]), float(sol[2]), deg, mad, int(ring.sum()))
        if deg > p.kf_ring_max_deg or mad > p.kf_ring_max_res:
            return None
        return (sol[0] * xy[:, 0] + sol[1] * xy[:, 1] + sol[2]).astype(np.float32)

    def _site_volume(self, H: np.ndarray, where, other: np.ndarray | None = None) -> float:
        """Volume of the pile(s) at an event site in height map H: the piles whose footprint touches the event disc
        (event_radius), by the same definition as the pile volume (volume_method)."""
        p = self.p
        if p.volume_method == "filled":
            pa = self._pile_analysis(H)
            near = np.hypot(self.X - where[0], self.Y - where[1]) <= p.event_radius
            ids = set(np.unique(pa["label"][near & (pa["label"] > 0)]).tolist())
            return float(sum(q["volume"] for q in pa["piles"] if q["id"] in ids))
        from scipy import ndimage
        def pile_of(M):
            ph = M - self.base
            return np.isfinite(M) & ~self.spike & (ph > p.pile_thr), ph
        pm, ph = pile_of(H)
        union = pm | pile_of(other)[0] if other is not None else pm
        lab, n = ndimage.label(union, structure=np.ones((3, 3), int))
        if not n:
            return 0.0
        near = np.hypot(self.X - where[0], self.Y - where[1]) <= p.event_radius
        ids = np.unique(lab[near & (lab > 0)])
        sel = np.isin(lab, ids) & pm
        return float(ph[sel].sum() * p.cell ** 2)

    def work_cycles(self, same_site_m: float = 2.0) -> tuple:
        """Bucket events grouped into work cycles: a scoop ('dig', with its '+sweep' second look at the same site)
        followed by the put-down ('dump', likewise) before the next scoop. One dict per cycle:
        t_dig, dig_before / dig_after (pile volume at the scoop site), cut (scooped), t_dump, dump_before / dump_after
        (pile volume at the put-down site), fill (put down); nan where that half was not seen (e.g. dumped out of view)."""
        nan = float("nan")
        out: list[dict] = []
        for e in self.events:
            t, kind, cut, fill, xy = e[0], e[1], e[2], e[3], (e[6], e[7])
            v0, v1 = (e[9], e[10]) if len(e) > 10 else (nan, nan)
            dig, sweep = kind.startswith("dig"), kind.endswith("+sweep")
            c = out[-1] if out else None
            half = "dig" if dig else "dump"
            if c is not None and sweep and c.get(half + "_xy") is not None \
                    and np.hypot(xy[0] - c[half + "_xy"][0], xy[1] - c[half + "_xy"][1]) <= same_site_m:
                c[half + "_after"] = v1                              # second look at the same site: totals add up
                c["cut" if dig else "fill"] += cut if dig else fill
                continue
            if c is None or dig or not np.isnan(c["t_dump"]):
                c = dict(t_dig=nan, dig_before=nan, dig_after=nan, cut=nan, dig_xy=None,
                         t_dump=nan, dump_before=nan, dump_after=nan, fill=nan, dump_xy=None)
                out.append(c)
            c["t_" + half], c[half + "_before"], c[half + "_after"], c[half + "_xy"] = t, v0, v1, xy
            c["cut" if dig else "fill"] = cut if dig else fill
        return tuple(out)

    def _keep_engine(self, P_level: np.ndarray) -> np.ndarray:
        p = self.p
        r = np.linalg.norm(P_level[:, :2], axis=1)
        return (r >= p.r_min) & (r <= p.engine_max_m) & (P_level[:, 2] >= p.z_min) & (P_level[:, 2] <= p.z_max)

    # ------------------------------------------------------------------ cycles
    def _cycles(self, t: float, world, dt: float):
        p = self.p
        w = world.w
        if self.swing is None:
            if abs(w) > np.radians(p.swing_rate_dps):
                self.swing = [t, 0.0]
                self.calm_t = 0.0
        if self.swing is not None:
            self.swing[1] += w * dt
            self.calm_t = self.calm_t + dt if abs(w) < np.radians(p.swing_rate_dps) else 0.0
            if self.calm_t > 0.4:                          # swing over
                ang = np.degrees(self.swing[1])
                if abs(ang) >= p.swing_min_deg:
                    self.swings.append((t, ang))
                    if len(self.swings) >= 2 and np.sign(self.swings[-1][1]) != np.sign(self.swings[-2][1]) \
                            and (not self.cycles or self.cycles[-1][0] < self.swings[-2][0]):
                        self.cycles.append((t, self._volume()[3]))   # out + back = one work cycle (cut at its end)
                self.swing = None

    # ------------------------------------------------------------------ outputs
    def _zone_mask(self) -> np.ndarray | None:
        z, X, Y = self.p.zone, self.X, self.Y
        if z is not None:
            return (X >= z[0]) & (X <= z[1]) & (Y >= z[2]) & (Y <= z[3])
        if self.p.reach_m > 0 and self.center is not None:
            return (X - self.center[0]) ** 2 + (Y - self.center[1]) ** 2 <= self.p.reach_m ** 2
        return None

    def _fit_ground(self, rng=np.random.default_rng(0)):
        """ground == "fit": dominant plane of the observed cells (RANSAC, 0.1 m), least-squares refit on its inliers."""
        if self.p.ground != "fit":
            return
        fin = np.isfinite(self.H)
        if fin.sum() < 50:
            return
        P = np.c_[self.X[fin], self.Y[fin], self.H[fin]]
        best, best_n = None, 0
        for _ in range(200):
            q = P[rng.choice(len(P), 3, replace=False)]
            n = np.cross(q[1] - q[0], q[2] - q[0])
            if abs(n[2]) < 0.9 * np.linalg.norm(n):                 # steeper than ~25 deg: not ground
                continue
            inl = np.abs((P - q[0]) @ n / np.linalg.norm(n)) < 0.1
            if inl.sum() > best_n:
                best, best_n = inl, int(inl.sum())
        if best is not None:
            sol = np.linalg.lstsq(np.c_[P[best, :2], np.ones(best_n)], P[best, 2], rcond=None)[0]
            self.plane = tuple(float(v) for v in sol)

    def _volume(self, t: float | None = None) -> tuple[float, float, float, float, float]:
        """(pile volume, pile area, observed area, cut, fill) inside the zone. The pile part (volume_method "filled")
        comes from _pile_analysis, redone at most every volume_every_s while the map changes (t None: reuse)."""
        H = self.H
        fin = np.isfinite(H)
        zm = self._zone_mask()
        if zm is not None:
            fin &= zm
        fin &= ~self.spike & (self.n_obs >= self.p.min_obs)         # isolated / once-seen cells do not count
        d = H - self.H_ref
        ok = fin & np.isfinite(d)
        thr = self.p.change_thr
        cut = ok & (d < -thr)
        fill = ok & (d > thr)
        a = self.p.cell ** 2
        if t is not None and self.map_ver != self.pa_ver and t - self.t_pa >= self.p.volume_every_s or self.pa is None:
            self.pa = self._pile_analysis(H, t)
            self.pa_ver, self.t_pa = self.map_ver, (t if t is not None else self.t_pa)
        return (self.pa["volume"], self.pa["area"], float(fin.sum() * a), float(-d[cut].sum() * a), float(d[fill].sum() * a))

    def _pile_analysis(self, H: np.ndarray, t: float | None = None) -> dict:
        """Piles of height map H inside the zone. volume_method "filled" (docs/34 §4.1), per pile:
          1. cores: observed cells > pile_thr above the base (calib plane, lowered locally), 8-connected
          2. its own ground: robust plane through the observed ground-like cells in a ring (ground_ring cells) around
             the closed core footprint (the calib plane sat 7-8 cm low and 1.5-1.8 deg tilted at the 155530 pile)
          3. footprint: cells > pile_edge above that plane connected to the core, closed (pile_close) and hole-filled,
             kept within the ring's outer edge
          4. empty footprint cells: linear interpolation of the observed footprint + ring cells (as a triangle mesh
             would bridge them), nearest where outside their hull
          5. volume = sum over the footprint of max(height above the plane, 0) x cell area (no threshold)
        "threshold": cells > pile_thr above the base, empty cells 0. Returns dict(volume, area, fill_frac, piles=[...],
        label = footprint id per cell)."""
        from scipy import ndimage
        p = self.p
        a = p.cell ** 2
        fin = np.isfinite(H) & ~self.spike & (self.n_obs >= p.min_obs)
        zm = self._zone_mask()
        zone = zm if zm is not None else np.ones(H.shape, bool)
        h0 = H - self.base
        core = fin & zone & (h0 > p.pile_thr)
        label = np.zeros(H.shape, np.int32)
        out = dict(volume=0.0, area=0.0, fill_frac=0.0, piles=[], label=label)
        if not core.any():
            return out
        k8 = np.ones((3, 3), bool)
        if p.volume_method != "filled":
            lab, n = ndimage.label(core, structure=k8)
            for i in range(1, n + 1):
                m = lab == i
                w = h0[m]
                out["piles"].append(dict(id=i, volume=float(w.sum() * a), area=float(m.sum() * a), fill=0.0,
                                         x=float((self.X[m] * w).sum() / w.sum()), y=float((self.Y[m] * w).sum() / w.sum()),
                                         z_top=float(H[m].max()), h_max=float(w.max())))
            label[:] = lab
        else:
            lab, n = ndimage.label(core, structure=k8)
            r_in, r_out = p.ground_ring
            pad = r_out + p.pile_close + 2
            taken = np.zeros(H.shape, bool)
            objs = ndimage.find_objects(lab)
            rad_ij = rad_z = None
            new_state = []
            if p.ground_radar:                                          # fused radar points of the last ground_radar_win_s
                rxy = np.concatenate([b[1] for b in self.rad_buf]) if self.rad_buf else np.zeros((0, 2), np.float32)
                rz = np.concatenate([b[2] for b in self.rad_buf]) if self.rad_buf else np.zeros(0, np.float32)
                rij = np.floor((rxy + p.half_extent) / p.cell).astype(int)
                okr = (rij >= 0).all(1) & (rij < self.n).all(1)
                rad_ij, rad_z = rij[okr], rz[okr]
            for i in np.argsort(-np.bincount(lab.ravel(), minlength=n + 1)[1:], kind="stable") + 1:   # largest core first
                sl = objs[i - 1]
                i0, i1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, H.shape[0])
                j0, j1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, H.shape[1])
                S = (slice(i0, i1), slice(j0, j1))
                c = lab[S] == i
                if c.sum() < 3 or taken[S][c].all():                    # single cells: noise; merged into a previous pile
                    continue
                Hs, fs, Xs, Ys = H[S], fin[S] & zone[S], self.X[S], self.Y[S]
                F1 = ndimage.binary_fill_holes(ndimage.binary_closing(c, structure=k8, iterations=p.pile_close))
                outer = ndimage.binary_dilation(F1, structure=k8, iterations=r_out)
                ring_geo = outer & ~ndimage.binary_dilation(F1, structure=k8, iterations=r_in)
                ring = ring_geo & fs & (h0[S] < p.pile_thr)
                G = self.base[S]
                g_off = 0.0
                if ring.sum() >= p.ground_ring_min:
                    A = np.c_[Xs[ring], Ys[ring], np.ones(ring.sum())]
                    z = Hs[ring]
                    w = np.ones(len(z), bool)
                    for _ in range(3):                                   # trimmed least squares (toe, clutter)
                        sol = np.linalg.lstsq(A[w], z[w], rcond=None)[0]
                        res = z - A @ sol
                        w = np.abs(res) < max(3 * 1.48 * np.median(np.abs(res[w])), 0.03)
                    if np.degrees(np.arctan(np.hypot(sol[0], sol[1]))) <= p.ground_max_deg and w.sum() >= p.ground_ring_min:
                        G = (sol[0] * Xs + sol[1] * Ys + sol[2]).astype(np.float32)
                        if rad_ij is not None:                       # radar heights on the ring: one robust shift
                            ii, jj = rad_ij[:, 0] - i0, rad_ij[:, 1] - j0
                            inb = (ii >= 0) & (ii < i1 - i0) & (jj >= 0) & (jj < j1 - j0)
                            sel = np.zeros(len(rad_z), bool)
                            sel[inb] = ring_geo[ii[inb], jj[inb]] & zone[S][ii[inb], jj[inb]]
                            if sel.sum() >= p.ground_radar_min:
                                d = rad_z[sel] - G[ii[sel], jj[sel]]
                                for _ in range(3):
                                    m_ = np.median(d)
                                    d = d[np.abs(d - m_) < 2.5 * 1.4826 * np.median(np.abs(d - m_)) + 0.02]
                                g_off = float(np.clip(np.median(d), -p.ground_radar_clip, p.ground_radar_clip))
                            else:
                                g_off = np.nan
                            g_off = self._smooth_shift(Xs[c].mean(), Ys[c].mean(), g_off, t, new_state)
                            G = G + np.float32(g_off)
                h = Hs - G
                grow = fs & (h > p.pile_edge) & outer
                gl, _ = ndimage.label(grow | c, structure=k8)
                seed = np.isin(gl, np.unique(gl[c]))
                F = ndimage.binary_fill_holes(ndimage.binary_closing(seed, structure=k8, iterations=p.pile_close)) & outer & zone[S]
                F &= ~taken[S]
                if F.sum() < 3:
                    continue
                obs = F & fs
                empty = F & ~fs
                hf = np.where(obs, h, 0.0).astype(np.float64)
                if empty.any():
                    src = (obs | ring) & np.isfinite(h)
                    if src.sum() >= 3:
                        from scipy.interpolate import griddata
                        pts = np.c_[Xs[src], Ys[src]]
                        q = (Xs[empty], Ys[empty])
                        try:
                            v = griddata(pts, h[src], q, method="linear")
                        except Exception:                            # degenerate (collinear) support
                            v = np.full(len(q[0]), np.nan)
                        bad = ~np.isfinite(v)
                        if bad.any():
                            v[bad] = griddata(pts, h[src], (q[0][bad], q[1][bad]), method="nearest")
                        hf[empty] = v
                hp = np.clip(hf, 0.0, None)
                vol = float(hp[F].sum() * a)
                if vol <= 0:
                    continue
                taken[S] |= F
                label[S][F] = i
                wsum = hp[F].sum()
                out["piles"].append(dict(id=i, volume=vol, area=float(F.sum() * a), fill=float(empty.sum() / F.sum()), g_off=g_off,
                                         x=float((Xs[F] * hp[F]).sum() / wsum), y=float((Ys[F] * hp[F]).sum() / wsum),
                                         z_top=float(np.nanmax(np.where(obs, Hs, np.nan))) if obs.any() else float(G[F].max()),
                                         h_max=float(hp[F].max())))
        if p.volume_method == "filled" and p.ground_radar and t is not None:
            self.g_off_state = new_state                            # analysis of the live map: keep the smoothed shifts
        out["piles"].sort(key=lambda q: -q["volume"])
        out["volume"] = float(sum(q["volume"] for q in out["piles"]))
        out["area"] = float(sum(q["area"] for q in out["piles"]))
        tot = sum(q["area"] for q in out["piles"])
        out["fill_frac"] = float(sum(q["fill"] * q["area"] for q in out["piles"]) / tot) if tot > 0 else 0.0
        return out

    def _smooth_shift(self, x: float, y: float, raw: float, t: float | None, new_state: list) -> float:
        """ground_radar: the pile's shift after the time smoothing (state of the previous live analysis); raw nan = no
        new value (too few ring points). t None (event before / after maps): previous shift, else raw, no update."""
        p = self.p
        prev = [q for q in self.g_off_state if np.hypot(q[0] - x, q[1] - y) < 2.0]
        po, tp = (prev[0][2], prev[0][3]) if prev else (np.nan, np.nan)
        if t is None:
            return float(po if np.isfinite(po) else (raw if np.isfinite(raw) else 0.0))
        if np.isfinite(raw):
            if np.isfinite(po) and p.ground_radar_tau_s > 0:
                b = 1.0 - np.exp(-max(t - tp, 0.0) / p.ground_radar_tau_s)
                o = po + b * (raw - po)
            else:
                o = raw
        else:
            o = po if np.isfinite(po) else 0.0
        new_state.append((float(x), float(y), float(o), float(t)))
        return float(o)

    def _piles(self) -> tuple:
        """Separate piles of the last pile analysis, largest first, those under pile_min_m3 dropped (GUI labels):
        (x, y, z_top, volume, area, height above its ground, interpolated share)."""
        if self.pa is None:
            return ()
        return tuple((q["x"], q["y"], q["z_top"], q["volume"], q["area"], q["h_max"], q["fill"])
                     for q in self.pa["piles"] if q["volume"] >= self.p.pile_min_m3)

    def update(self, t: float, dt: float, world, radar_xyz: np.ndarray, radar_dynamic, boxes: list, depth=None,
               image=None, radar_keep=None, radar_alpha=None, img_dt: float = 0.0) -> WorkState:
        p = self.p
        self.k += 1
        self.center = world.axis if world.axis is not None else world.origin
        if not self.plane_set:
            self._set_plane(world)
        good = ~np.asarray(radar_dynamic, bool) if radar_dynamic is not None else np.ones(len(radar_xyz), bool)
        if p.radar_filter and radar_keep is not None and len(radar_keep) == len(good):
            good &= np.asarray(radar_keep, bool)
            if radar_alpha is not None and len(radar_alpha) == len(good):
                good &= np.asarray(radar_alpha) >= p.radar_min_alpha
        stat = radar_xyz[good]
        n_r = n_s = 0
        use_engine = self.engine is not None and self.engine.error is None
        if use_engine:
            for job in self.engine.poll():                   # keyframes finished since the last frame
                n_s += self._engine_depth(job)
                self.n_engine += 1
        still = abs(world.w) < np.radians(p.update_max_rate_dps)    # cab (nearly) still
        dyn = radar_xyz[np.asarray(radar_dynamic, bool)] if radar_dynamic is not None and len(radar_xyz) else np.zeros((0, 3))
        small = None
        if image is not None and (p.pause_on_motion or p.mask_machine):      # one 1/4 resize for motion + mask (~1 ms)
            import cv2
            f = self.MASK_F
            small = cv2.resize(image, (image.shape[1] // f, image.shape[0] // f), interpolation=cv2.INTER_AREA)
        slow = abs(world.w) < np.radians(max(p.update_max_rate_dps, p.event_max_rate_dps))
        sweep = abs(world.w) < np.radians(p.event_sweep_dps)
        mask = self.machine_mask(small, small_given=True) if p.mask_machine and small is not None and slow else None
        moving = p.pause_on_motion and self._motion(t, small, dyn, boxes, self.mask_core if mask is not None else None)
        self.in_pile = False
        if p.pause_bucket_in_pile and mask is not None and mask.any():
            self.in_pile = self._bucket_down(mask)
            if self.in_pile and still:
                self.active_until = max(self.active_until, t + p.motion_hold_s)
        if mask is not None and p.event_keyframes:                  # bucket event: down (>= min_down_s) -> out (>= min_out_s)
            if self.in_pile:
                if self.t_down is None or self.t_out is not None:   # a new down period (a short out flicker continues it)
                    if self.t_out is None or t - self.t_out >= p.event_min_out_s:
                        self.t_down, self.down_pts = t, []
                    self.t_out = None
                if self.bucket_ground is not None:
                    self.down_pts.append(world.to_world(self.bucket_ground[None])[0])
            elif self.t_down is not None:
                if self.t_out is None:
                    self.t_out = t
                if t - self.t_out >= p.event_min_out_s and self.armed is None:
                    if self.t_out - self.t_down >= p.event_min_down_s and self.down_pts:
                        self.armed = (t, self.t_down, np.median(np.array(self.down_pts), axis=0))
                    self.t_down, self.t_out, self.down_pts = None, None, []
            self.was_down = self.in_pile
        active = moving or (self.in_pile and still) or t <= self.active_until
        if self.armed is not None and use_engine and image is not None:
            t_ready, t_down, where = self.armed[:3]
            phase = self.armed[3] if len(self.armed) > 3 else 1
            window = p.event_window_s if phase == 1 else p.event_sweep_window_s
            fire = (phase == 1 and ((slow and self.motion_outside <= p.motion_frac)
                                    or (not slow and sweep and t - t_ready >= p.event_sweep_wait_s))) \
                or (phase == 2 and not slow and sweep)
            if t - t_ready > window or (phase == 1 and self.in_pile):
                self.armed = None                                   # missed (bucket went back in / no swing); phase 2
                                                                    # waits for the swing even if the bucket hovers low
            elif fire:
                from rcmot.work.depth_engine import DepthJob
                wj = world if abs(img_dt) < 1e-3 else self._pose_at(world, img_dt)   # the image's own time (turning)
                if not slow and mask is None:                       # turning: the bucket mask of this image
                    mask = self.machine_mask(small, small_given=True) if small is not None else None
                job = DepthJob(t + img_dt, image, wj, stat, list(boxes), forced=True,
                               event={"t_down": t_down, "where": where, "mask": None if self.mask_core is None else self.mask_core.copy()})
                job.event["phase"] = phase
                if self.engine.submit(job):
                    # phase 1 taken while still: keep a second look for the start of the next swing (heap uncovered)
                    self.armed = (t, t_down, where, 2) if phase == 1 and slow else None
                    self.t_submit = t
                    if p.vggt_sync:
                        for jb in self.engine.poll():
                            n_s += self._engine_depth(jb)
                            self.n_engine += 1
        self.mask_small = mask
        if still:
            self.n_still += 1
            self.n_paused += int(active)
        if still and not active:
            R = stat[self._keep(stat)] if len(stat) else stat
            if mask is not None and len(R):                          # radar returns from the boom / bucket
                u, v, fr = self._project(R)
                R = R[~(self._in_mask(mask, u, v) & fr)]
            R = self._axis(R, world)
            n_r = self._fuse(world.to_world(R), R[:, 2], p.alpha_radar, t) if len(R) else 0
            if p.ground_radar and len(R):
                self.rad_buf.append((t, world.to_world(R).astype(np.float32), R[:, 2].astype(np.float32)))
                while self.rad_buf and self.rad_buf[0][0] < t - p.ground_radar_win_s:
                    self.rad_buf.pop(0)
            if use_engine:
                if image is not None and t - self.t_submit >= p.vggt_period_s:
                    from rcmot.work.depth_engine import DepthJob
                    if self.engine.submit(DepthJob(t, image, world, stat, list(boxes))):
                        self.t_submit = t
                        if p.vggt_sync:
                            for job in self.engine.poll():
                                n_s += self._engine_depth(job)
                                self.n_engine += 1
            elif depth is not None and self.k % max(p.stereo_every, 1) == 0:
                n_s = self._stereo(depth, stat, boxes, world, t, mask)
        elif (p.partial_in_pile and still and self.in_pile and t - self.t_moved > p.motion_hold_s and self.bucket_ground is not None
              and use_engine
              and image is not None and self.armed is None and t - self.t_submit >= p.vggt_period_s):
            from rcmot.work.depth_engine import DepthJob
            bxy = world.to_world(self.bucket_ground[None])[0]
            if self.engine.submit(DepthJob(t, image, world, stat, list(boxes), event={"exclude": (bxy, p.partial_m)})):
                self.t_submit = t
                self.n_partial += 1
                if p.vggt_sync:
                    for job in self.engine.poll():
                        n_s += self._engine_depth(job)
                        self.n_engine += 1
        if n_s > 0:
            self.t_map = t
        if t - self.t_snap >= p.snapshot_s:
            self._fit_ground()
            self._update_base()
        self._cycles(t, world, dt)
        vol, area, obs, cut, fill = self._volume(t)
        self.series_t.append(t)
        self.series_v.append(vol)
        self.series_cut.append(cut)
        self.series_fill.append(fill)
        if t - self.t_snap >= p.snapshot_s:
            q = np.full(self.H.shape, NAN_Q, np.uint8)
            fin = np.isfinite(self.H) & ~self.spike & (self.n_obs >= p.min_obs)
            q[fin] = np.clip(np.round((self.H[fin] + 1.0) / p.q_step), 0, NAN_Q - 1).astype(np.uint8)
            d = self.H - self.H_ref
            dq = np.full(self.H.shape, -128, np.int8)
            okd = np.isfinite(d)
            dq[okd] = np.clip(np.round(d[okd] / p.q_step), -127, 127).astype(np.int8)
            self.snap = HeightSnapshot(t, q, dq, p.cell, p.half_extent, p.q_step, np.array(self.series_t, np.float64),
                                       np.array(self.series_v, np.float32), np.array(self.series_cut, np.float32),
                                       np.array(self.series_fill, np.float32), np.array([c[0] for c in self.cycles]), self.plane,
                                       None if not np.isfinite(self.C).any() else
                                       np.nan_to_num(self.C[..., ::-1], nan=0.0).clip(0, 255).astype(np.uint8),
                                       self._piles())
            self.t_snap = t
        ct = [c[0] for c in self.cycles]
        cyc_s = float(np.mean(np.diff(ct))) if len(ct) >= 2 else float("nan")
        dcut = self.cycles[-1][1] - self.cycles[-2][1] if len(self.cycles) >= 2 else float("nan")
        eng = "vggt_omega" if use_engine else ("stereo" if depth is not None else "radar")
        return WorkState(t=t, volume=vol, area=area, observed=obs, cut=cut, fill=fill, n_cycles=len(self.cycles), cycle_s=cyc_s,
                         last_dcut=dcut, scale=getattr(self, "engine_scale", self.scale) if use_engine else self.scale,
                         swinging=self.swing is not None, snapshot=self.snap, n_stereo_pts=n_s, n_radar_pts=n_r, source=eng,
                         n_engine=self.n_engine, engine_ms=self.engine.last_ms if self.engine is not None else 0.0,
                         active=bool(active), bucket_in_pile=bool(self.in_pile), paused_frac=self.n_paused / max(self.n_still, 1),
                         mask_frac=float(mask.mean()) if mask is not None else 0.0,
                         events=tuple(self.events), last_event=self.events[-1] if self.events else None, t_map=self.t_map,
                         ledger=self.work_cycles() if self.events else (),
                         fill_frac=self.pa["fill_frac"] if self.pa is not None else 0.0)
