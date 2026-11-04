"""MOT metrics on 2D boxes: CLEAR (MOTA, IDsw, Frag, MT/ML), identity (IDF1/IDP/IDR) and HOTA (DetA, AssA, LocA),
following the definitions of Bernardin & Stiefelhagen 2008, Ristani et al. 2016 and Luiten et al. 2021 (TrackEval).

Input per frame: gt_ids, gt_boxes, gt_ignore (bool), pr_ids, pr_boxes. Predictions matched (IoU >= thr) to
ignore-flagged GT boxes are removed before scoring, as are the ignore GT boxes themselves.

v2 (docs/13 "평가 지표 v2", gt_design §D.7): the optional 3D columns of a Frame (gt_range / gt_xy / gt_quality from the
GT box ref3d, pr_range / pr_xy from the track position) feed
  * rates():       IDSWR = IDsw / n_gt_boxes, FNR = FN / GT, FPR_gt = FP / GT, FPR_pred = FP / Pred
  * motp_range():  MOTP[m] = |pred horizontal range - ref3d range| over the 2D-IoU-matched TP pairs whose GT has a
                   reference (quality != none), mean / median / p90, by range bin and by reference quality
  * evaluate_bev(): auxiliary matching by BEV centre distance <= 2 m (nuScenes convention) on the GT with a reference.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from rcmot.labeling.draft import iou


@dataclass
class Frame:
    gt_ids: np.ndarray
    gt_boxes: np.ndarray
    pr_ids: np.ndarray
    pr_boxes: np.ndarray
    gt_ignore: np.ndarray | None = None
    gt_range: np.ndarray | None = None      # ref3d.range_m per GT (nan = no reference)
    gt_xy: np.ndarray | None = None         # (N,2) ref3d.xy_level (nan rows = no reference)
    gt_quality: np.ndarray | None = None    # "dense" | "sparse" | "none" per GT
    pr_range: np.ndarray | None = None      # horizontal range of the prediction (nan = unknown)
    pr_xy: np.ndarray | None = None         # (M,2) BEV position of the prediction (level frame)

    def gt_sub(self, keep: np.ndarray) -> dict:
        """kwargs of the GT columns restricted to a boolean / index selection."""
        return {"gt_range": None if self.gt_range is None else self.gt_range[keep],
                "gt_xy": None if self.gt_xy is None else self.gt_xy[keep],
                "gt_quality": None if self.gt_quality is None else self.gt_quality[keep]}

    def pr_sub(self, keep: np.ndarray) -> dict:
        return {"pr_range": None if self.pr_range is None else self.pr_range[keep],
                "pr_xy": None if self.pr_xy is None else self.pr_xy[keep]}


RANGE_BINS = (0.0, 10.0, 20.0, 30.0, np.inf)
BEV_MATCH_M = 2.0


def range_bin_name(bins, q: int) -> str:
    return f"{int(bins[q])}-{int(bins[q + 1]) if np.isfinite(bins[q + 1]) else 'inf'}"


def bev_sim(gt_xy: np.ndarray, pr_xy: np.ndarray, radius: float = BEV_MATCH_M) -> np.ndarray:
    """Similarity 1 - d / radius (0 outside the radius) between GT and prediction BEV centres; a threshold > 0 on it
    is the nuScenes-style "centre distance < radius" rule. Rows without a reference (nan) never match."""
    if len(gt_xy) == 0 or len(pr_xy) == 0:
        return np.zeros((len(gt_xy), len(pr_xy)))
    d = np.linalg.norm(gt_xy[:, None, :] - pr_xy[None, :, :], axis=2)
    return np.where(np.isfinite(d), np.clip(1.0 - d / radius, 0.0, 1.0), 0.0)


def drop_ignored(f: Frame, thr: float = 0.5) -> tuple[Frame, int]:
    """Remove ignore GT and the predictions they absorb. Returns the cleaned frame and #absorbed predictions."""
    ign = np.zeros(len(f.gt_ids), bool) if f.gt_ignore is None else np.asarray(f.gt_ignore, bool)
    if not ign.any():
        return f, 0
    keep_pr = np.ones(len(f.pr_ids), bool)
    if len(f.pr_ids):
        S = iou(f.gt_boxes[ign], f.pr_boxes)
        # predictions first matched to the regular GT keep their chance; the rest may fall into ignore regions
        reg = ~ign
        taken = set()
        if reg.any():
            Sr = iou(f.gt_boxes[reg], f.pr_boxes)
            r, c = linear_sum_assignment(-Sr)
            taken = {j for i, j in zip(r, c) if Sr[i, j] >= thr}
        for j in range(len(f.pr_ids)):
            if j not in taken and S[:, j].max() >= thr:
                keep_pr[j] = False
    g = ~ign
    return Frame(f.gt_ids[g], f.gt_boxes[g], f.pr_ids[keep_pr], f.pr_boxes[keep_pr],
                 **f.gt_sub(g), **f.pr_sub(keep_pr)), int((~keep_pr).sum())


def frame_sim(f: Frame, sim: str = "iou") -> np.ndarray:
    """GT x prediction similarity of one frame: 2D box IoU (default) or the BEV centre rule ("bev")."""
    if sim == "bev":
        return bev_sim(f.gt_xy, f.pr_xy)
    return iou(f.gt_boxes, f.pr_boxes)


def clear_matches(frames: list[Frame], thr: float = 0.5, sim: str = "iou"):
    """CLEAR matching frame by frame: previous correspondences are kept while still valid (similarity >= thr),
    the rest is Hungarian. Yields (frame, [(gt index, pred index)]) in order."""
    last_match: dict = {}                    # gt id -> last matched pred id
    for f in frames:
        if len(f.gt_ids) and len(f.pr_ids):
            S = frame_sim(f, sim)
            bonus = np.zeros_like(S)
            for i, g in enumerate(f.gt_ids):
                if g in last_match:
                    bonus[i, f.pr_ids == last_match[g]] = 1.0
            C = -(S + bonus * (S >= thr))
            r, c = linear_sum_assignment(C)
            m = [(i, j) for i, j in zip(r, c) if S[i, j] >= thr]
        else:
            m = []
        yield f, m
        for i, j in m:
            last_match[f.gt_ids[i]] = f.pr_ids[j]


def clear_identity(frames: list[Frame], thr: float = 0.5, sim: str = "iou") -> dict:
    tp = fp = fn = idsw = 0
    last_match: dict = {}                    # gt id -> last matched pred id
    tracked_frames = defaultdict(int)
    gt_frames = defaultdict(int)
    status = defaultdict(list)               # gt id -> [matched? per frame where present]
    pairs = defaultdict(int)                 # (gt, pr) -> #frames matched (CLEAR matching)
    gt_count, pr_count = defaultdict(int), defaultdict(int)
    for f, m in clear_matches(frames, thr, sim):
        for g in f.gt_ids:
            gt_count[g] += 1
        for p in f.pr_ids:
            pr_count[p] += 1
        mg = {i for i, _ in m}
        tp += len(m)
        fp += len(f.pr_ids) - len(m)
        fn += len(f.gt_ids) - len(m)
        for i, j in m:
            g, p = f.gt_ids[i], f.pr_ids[j]
            if g in last_match and last_match[g] != p:
                idsw += 1
            last_match[g] = p
            tracked_frames[g] += 1
            pairs[(g, p)] += 1
        for i, g in enumerate(f.gt_ids):
            gt_frames[g] += 1
            status[g].append(i in mg)
    n_gt = sum(gt_count.values())
    frag = 0
    for s in status.values():
        seen = False
        for a, b in zip(s[:-1], s[1:]):
            seen = seen or a
            if a and not b:
                frag += 1
        if s and not s[-1] and seen:
            frag -= 1                         # ending untracked is not a fragmentation
    ratio = {g: tracked_frames[g] / gt_frames[g] for g in gt_frames}
    # IDF1: global one-to-one gt-id <-> pred-id assignment maximising matched detections (IoU >= thr per frame)
    idtp_pair = defaultdict(int)
    for f in frames:
        if len(f.gt_ids) and len(f.pr_ids):
            S = frame_sim(f, sim)
            for i, j in zip(*np.nonzero(S >= thr)):
                idtp_pair[(f.gt_ids[i], f.pr_ids[j])] += 1
    G, P = sorted(gt_count), sorted(pr_count)
    idtp = 0
    if G and P:
        M = np.zeros((len(G), len(P)))
        gi, pj = {g: k for k, g in enumerate(G)}, {p: k for k, p in enumerate(P)}
        for (g, p), v in idtp_pair.items():
            M[gi[g], pj[p]] = v
        r, c = linear_sum_assignment(-M)
        idtp = int(M[r, c].sum())
    n_pr = sum(pr_count.values())
    preds_per_gt = {int(g): sorted({int(p) for (gg, p) in pairs if gg == g}) for g in G}
    return {"MOTA": 1 - (fn + fp + idsw) / max(n_gt, 1), "TP": tp, "FP": fp, "FN": fn, "IDsw": idsw, "Frag": frag,
            "IDF1": 2 * idtp / max(n_gt + n_pr, 1), "IDP": idtp / max(n_pr, 1), "IDR": idtp / max(n_gt, 1),
            "MT": sum(v > 0.8 for v in ratio.values()), "ML": sum(v < 0.2 for v in ratio.values()),
            "n_gt_ids": len(G), "n_pred_ids": len(P), "n_gt_boxes": n_gt, "n_pred_boxes": n_pr,
            "pred_ids_per_gt": preds_per_gt}


def hota(frames: list[Frame], alphas=np.arange(0.05, 0.99, 0.05), sim: str = "iou") -> dict:
    """HOTA as in TrackEval: global alignment score weights the per-frame Hungarian, then DetA / AssA per alpha.
    sim "bev" (docs/36 BEV HOTA): similarity 1 - d / 2 m between the BEV centres instead of the box IoU, i.e. the
    alphas 0.05 .. 0.95 are centre distances 1.9 .. 0.1 m."""
    G = sorted({g for f in frames for g in f.gt_ids})
    P = sorted({p for f in frames for p in f.pr_ids})
    gi, pj = {g: k for k, g in enumerate(G)}, {p: k for k, p in enumerate(P)}
    pot = np.zeros((len(G), len(P)))
    gt_cnt, pr_cnt = np.zeros((len(G), 1)), np.zeros((1, len(P)))
    sims = []
    for f in frames:
        S = frame_sim(f, sim) if len(f.gt_ids) and len(f.pr_ids) else np.zeros((len(f.gt_ids), len(f.pr_ids)))
        sims.append(S)
        if S.size:
            den = S.sum(0, keepdims=True) + S.sum(1, keepdims=True) - S
            sim_iou = np.where(den > 1e-9, S / np.maximum(den, 1e-9), 0)
            pot[np.ix_([gi[g] for g in f.gt_ids], [pj[p] for p in f.pr_ids])] += sim_iou
        for g in f.gt_ids:
            gt_cnt[gi[g], 0] += 1
        for p in f.pr_ids:
            pr_cnt[0, pj[p]] += 1
    gas = pot / np.maximum(gt_cnt + pr_cnt - pot, 1e-9)
    na = len(alphas)
    TP, FN, FP, LocA = np.zeros(na), np.zeros(na), np.zeros(na), np.zeros(na)
    mc = np.zeros((na, len(G), len(P)))
    for f, S in zip(frames, sims):
        ng, npr = len(f.gt_ids), len(f.pr_ids)
        if ng == 0 or npr == 0:
            FN += ng
            FP += npr
            continue
        gidx, pidx = [gi[g] for g in f.gt_ids], [pj[p] for p in f.pr_ids]
        score = gas[np.ix_(gidx, pidx)] * S
        r, c = linear_sum_assignment(-score)
        for a, al in enumerate(alphas):
            ok = S[r, c] >= al - 1e-9
            n = int(ok.sum())
            TP[a] += n
            FN[a] += ng - n
            FP[a] += npr - n
            LocA[a] += S[r[ok], c[ok]].sum()
            mc[a, [gidx[i] for i in r[ok]], [pidx[j] for j in c[ok]]] += 1
    AssA = np.zeros(na)
    for a in range(na):
        A = mc[a] / np.maximum(gt_cnt + pr_cnt - mc[a], 1e-9)
        AssA[a] = (mc[a] * A).sum() / max(TP[a], 1)
    DetA = TP / np.maximum(TP + FN + FP, 1)
    H = np.sqrt(DetA * AssA)
    LocA = LocA / np.maximum(TP, 1)
    return {"HOTA": float(H.mean()), "DetA": float(DetA.mean()), "AssA": float(AssA.mean()),
            "LocA": float(LocA.mean()), "HOTA(0.5)": float(H[np.argmin(abs(alphas - 0.5))])}


def fp_breakdown(frames: list[Frame], thr: float = 0.5, near: float = 0.1) -> dict:
    """Unmatched predictions: 'duplicate' if it overlaps (IoU >= near) a GT that is matched to another
    prediction, 'mislocated' if it overlaps an unmatched GT only, else 'ghost' (nothing there)."""
    out = {"duplicate": 0, "mislocated": 0, "ghost": 0}
    for f in frames:
        if not len(f.pr_ids):
            continue
        if not len(f.gt_ids):
            out["ghost"] += len(f.pr_ids)
            continue
        S = iou(f.gt_boxes, f.pr_boxes)
        r, c = linear_sum_assignment(-S)
        m = {j: i for i, j in zip(r, c) if S[i, j] >= thr}
        for j in range(len(f.pr_ids)):
            if j in m:
                continue
            hits = np.nonzero(S[:, j] >= near)[0]
            if any(i in m.values() for i in hits):
                out["duplicate"] += 1
            elif len(hits):
                out["mislocated"] += 1
            else:
                out["ghost"] += 1
    return out


def evaluate(frames: list[Frame], thr: float = 0.5) -> dict:
    clean, absorbed = [], 0
    for f in frames:
        c, n = drop_ignored(f, thr)
        clean.append(c)
        absorbed += n
    res = clear_identity(clean, thr)
    res.update(hota(clean))
    res["FP_breakdown"] = fp_breakdown(clean, thr)
    res["pred_in_ignore"] = absorbed
    return res


# ----------------------------------------------------------------------------------------------- v2 additions
def rates(res: dict) -> dict:
    """IDSWR (IDsw / GT boxes), FNR (FN / GT), FPR_gt (FP / GT) and FPR_pred (FP / predicted boxes) from an
    evaluate() / clear_identity() result."""
    n_gt, n_pr = res.get("n_gt_boxes", 0), res.get("n_pred_boxes", 0)
    g = (lambda v: v / n_gt) if n_gt else (lambda v: None)          # no regular GT of this class: ratio undefined
    return {"IDSWR": g(res["IDsw"]), "FNR": g(res["FN"]), "FPR_gt": g(res["FP"]),
            "FPR_pred": res["FP"] / n_pr if n_pr else None}


def _stats(e: np.ndarray) -> dict:
    e = np.asarray(e, float)
    if e.size == 0:
        return {"n": 0, "mean": None, "median": None, "p90": None}
    return {"n": int(e.size), "mean": round(float(np.mean(e)), 3), "median": round(float(np.median(e)), 3),
            "p90": round(float(np.percentile(e, 90)), 3)}


def motp_range(frames: list[Frame], thr: float = 0.5, sim: str = "iou", bins=RANGE_BINS) -> dict:
    """MOTP in metres: |pred horizontal range - ref3d range| over the matched TP pairs (CLEAR matching with `sim`)
    whose GT has a reference (quality != none) and whose prediction has a finite range. Also by range bin of the
    reference and by reference quality; "bias" = mean(pred - ref). Frames without the 3D columns contribute nothing."""
    err, rng, qual = [], [], []
    n_tp = n_noref = 0
    for f, m in clear_matches(frames, thr, sim):
        n_tp += len(m)
        if f.gt_range is None or f.pr_range is None:
            n_noref += len(m)
            continue
        q = f.gt_quality if f.gt_quality is not None else np.full(len(f.gt_ids), "dense", object)
        for i, j in m:
            g, p = float(f.gt_range[i]), float(f.pr_range[j])
            if not np.isfinite(g) or q[i] == "none" or not np.isfinite(p):
                n_noref += 1
                continue
            err.append(p - g)
            rng.append(g)
            qual.append(str(q[i]))
    e, r, q = np.asarray(err, float), np.asarray(rng, float), np.asarray(qual, object)
    out = {**_stats(np.abs(e)), "bias": round(float(e.mean()), 3) if e.size else None, "n_tp": int(n_tp),
           "n_tp_without_ref": int(n_noref), "by_range": {}, "by_quality": {}}
    for k in range(len(bins) - 1):
        sel = (r >= bins[k]) & (r < bins[k + 1])
        out["by_range"][range_bin_name(bins, k)] = _stats(np.abs(e[sel]))
    for name in ("dense", "sparse"):
        out["by_quality"][name] = _stats(np.abs(e[q == name]))
    return out


def with_ref(frames: list[Frame], thr_iou: float = 0.5) -> tuple[list[Frame], int]:
    """Frames reduced to the GT boxes that carry a usable reference (BEV matching needs gt_xy). GT without one is
    treated like an ignore region: the predictions it absorbs by 2D IoU (>= thr_iou) are removed as well, so a
    missing reference costs neither FN nor FP. Returns (frames, #predictions removed that way)."""
    out, absorbed = [], 0
    for f in frames:
        if f.gt_xy is None:
            noref = np.ones(len(f.gt_ids), bool)
        else:
            noref = ~np.isfinite(np.asarray(f.gt_xy, float)).all(1) if len(f.gt_ids) else np.zeros(0, bool)
            if f.gt_quality is not None:
                noref |= np.asarray(f.gt_quality, object) == "none"
        g, n = drop_ignored(Frame(f.gt_ids, f.gt_boxes, f.pr_ids, f.pr_boxes, noref, **f.gt_sub(slice(None)),
                                  **f.pr_sub(slice(None))), thr_iou)
        out.append(g)
        absorbed += n
    return out, absorbed


def evaluate_bev(frames: list[Frame], thr_iou: float = 0.5) -> dict:
    """Auxiliary scoring with the BEV centre-distance rule (<= 2 m, gt_design §D.7): ignore GT still absorbs
    predictions by 2D IoU (thr_iou), GT without a reference is dropped together with the predictions on it
    ("pred_on_noref"), and the rest is matched by bev_sim > 0.
    Returns CLEAR / identity counts, the rates, MOTP[m] under that matching and BEV HOTA / DetA / AssA / LocA (hota()
    with the BEV similarity 1 - d / 2 m: alphas = centre distances 1.9 .. 0.1 m; docs/36 §1)."""
    cleaned = [drop_ignored(f, thr_iou) for f in frames]
    clean, on_noref = with_ref([c for c, _ in cleaned], thr_iou)
    res = clear_identity(clean, 1e-9, sim="bev")
    res.update(hota(clean, sim="bev"))
    res["pred_in_ignore"] = int(sum(n for _, n in cleaned))
    res["pred_on_noref"] = int(on_noref)
    res.update(rates(res))
    res["MOTP_m"] = motp_range(clean, 1e-9, sim="bev")
    res["match"] = f"bev{BEV_MATCH_M:g}m"
    return res


def near_zone(f: Frame, max_range: float, margin: float = BEV_MATCH_M, thr: float = 0.5) -> Frame:
    """docs/38 near-zone view: only people within max_range [m] of the camera (ref3d range) count.
      GT beyond max_range           -> ignore (absorbs the predictions on it by 2-D IoU: neither TP, FN nor FP)
      GT without a reference        -> kept as it is (its range is unknown)
      prediction beyond max_range + margin (its own 3-D range) that overlaps no in-zone regular GT box at IoU >= thr
                                    -> removed (an output outside the zone is not a false alarm of the zone)
    A prediction on an in-zone person stays whatever its range: a near person drawn at a wrong far range is still a 2-D
    TP and still a BEV miss + false box."""
    ign = np.zeros(len(f.gt_ids), bool) if f.gt_ignore is None else np.asarray(f.gt_ignore, bool).copy()
    far = np.zeros(len(f.gt_ids), bool)
    if f.gt_range is not None and len(f.gt_ids):
        gr = np.asarray(f.gt_range, float)
        far = np.isfinite(gr) & (gr > max_range)
    ign_n = ign | far
    keep_pr = np.ones(len(f.pr_ids), bool)
    if f.pr_range is not None and len(f.pr_ids):
        pr = np.asarray(f.pr_range, float)
        out = np.isfinite(pr) & (pr > max_range + margin)
        if out.any():
            reg = ~ign_n
            ov = iou(f.gt_boxes[reg], f.pr_boxes).max(0) if reg.any() else np.zeros(len(f.pr_ids))
            keep_pr = ~(out & (ov < thr))
    return Frame(f.gt_ids, f.gt_boxes, f.pr_ids[keep_pr], f.pr_boxes[keep_pr], ign_n, **f.gt_sub(slice(None)),
                 **f.pr_sub(keep_pr))


ALARM_ZONES = (10.0, 20.0, 30.0)


def alarm_zone(r: float, zones=ALARM_ZONES) -> int:
    """3 danger (<= zones[0]) .. 1 caution (<= zones[-1]), 0 none / unknown (rcmot.alarm.zone_of on sorted zones)."""
    if r is None or not np.isfinite(r):
        return 0
    zs = sorted(zones)
    for k, z in enumerate(zs):
        if r <= z:
            return len(zs) - k
    return 0


def _assign(sim: np.ndarray, thr: float) -> dict:
    """{row: col} maximising the summed similarity over pairs with sim >= thr (Hungarian)."""
    if sim.size == 0:
        return {}
    s = np.where(sim >= thr, sim, 0.0)
    r, c = linear_sum_assignment(-s)
    return {int(i): int(j) for i, j in zip(r, c) if s[i, j] >= thr and s[i, j] > 0}


def _band(r: float, edges, zs) -> str:
    return f"b{zs[max(int(np.searchsorted(edges, r, side='left')), 1) - 1]:g}"


def alarm_metrics(frames: list[Frame], stamps=None, zones=ALARM_ZONES, thr: float = 0.5,
                  bev_m: float = BEV_MATCH_M) -> dict:
    """docs/38 §10 alarm view: what the distance alarm needs is every person inside the zones found, at the right range,
    and no object in a zone where there is none. Counted over regular (not ignored) GT people with a LiDAR reference,
    in range bands (0, z0], (z0, z1], (z1, z2] of the alarm zones. The predictions should be the tracks the alarm
    counts (filter the states before building the frames).

    Per band (key "b<z>"):
      n         GT person boxes in the band
      found     2-D matched (IoU >= thr, Hungarian against every GT incl. ignored): the person is on screen
      bev       a prediction within bev_m of the reference position (Hungarian on distance): found at the right place
      zone_ok   found and the prediction's range is in the person's alarm zone
      under     found but the prediction's range is in a LESS severe zone (or beyond all zones): the dangerous error
      err_n / err_le1 / err_le2 / err_abs / err_sum   |range error| counts / sums of the found ones (range error =
                prediction horizontal range - reference range)
      false     predictions with range in the band that match no GT box in 2-D and are farther than bev_m from every
                referenced GT (incl. ignored ones): an object where there is none
    Frame level (only frames whose regular GT all have a reference): true level = zone of the nearest regular person,
    predicted level = zone of the nearest prediction not absorbed by an ignored GT (2-D IoU >= thr) ->
      lvl_n, lvl_exact, lvl_under (predicted less severe), lvl_over (more severe), lvl_miss_danger (true danger,
      predicted lower), lvl_conf (4x4 counts [true][pred], 0 none .. 3 danger).
    Delay (stamps given): per GT id, from its first frame inside the outer zone to its first frame "found right"
    (bev) inside the zone: delay_s (list), never (ids never found right while inside)."""
    zs = tuple(sorted(float(z) for z in zones))
    edges = (0.0,) + zs
    out = {f"b{z:g}": defaultdict(float) for z in zs}
    conf = np.zeros((len(zs) + 1, len(zs) + 1), int)
    first_in, first_ok = {}, {}
    for k, f in enumerate(frames):
        ng, npd = len(f.gt_ids), len(f.pr_ids)
        ign = np.zeros(ng, bool) if f.gt_ignore is None else np.asarray(f.gt_ignore, bool)
        gr = np.full(ng, np.nan) if f.gt_range is None else np.asarray(f.gt_range, float)
        gxy = np.full((ng, 2), np.nan) if f.gt_xy is None else np.asarray(f.gt_xy, float).reshape(-1, 2)
        pr = np.full(npd, np.nan) if f.pr_range is None else np.asarray(f.pr_range, float)
        pxy = np.full((npd, 2), np.nan) if f.pr_xy is None else np.asarray(f.pr_xy, float).reshape(-1, 2)
        ious = iou(f.gt_boxes, f.pr_boxes) if ng and npd else np.zeros((ng, npd))
        m2 = _assign(ious, thr)
        pr_on_gt = np.zeros(npd, bool)
        pr_on_ign = np.zeros(npd, bool)
        for i, j in m2.items():
            pr_on_gt[j] = True
            pr_on_ign[j] = ign[i]
        reg = ~ign & np.isfinite(gr)
        ri = np.flatnonzero(reg)
        dist = (np.linalg.norm(gxy[ri][:, None, :] - pxy[None, :, :], axis=2) if len(ri) and npd
                else np.zeros((len(ri), npd)))
        mb = _assign(np.where(np.isfinite(dist), np.clip(1.0 - dist / bev_m, 0.0, 1.0), 0.0), 1e-9) if len(ri) else {}
        bev_ok = {int(ri[a]) for a in mb}
        for i in ri:
            r = gr[i]
            if r > zs[-1]:
                continue
            b = out[_band(r, edges, zs)]
            b["n"] += 1
            if i in bev_ok:
                b["bev"] += 1
            j = m2.get(int(i))
            if j is not None:
                b["found"] += 1
                e = pr[j] - r
                if np.isfinite(e):
                    b["err_n"] += 1
                    b["err_le1"] += abs(e) <= 1.0
                    b["err_le2"] += abs(e) <= 2.0
                    b["err_abs"] += abs(e)
                    b["err_sum"] += e
                zt, zp = alarm_zone(r, zs), alarm_zone(pr[j], zs)
                b["zone_ok"] += zt == zp
                b["under"] += zp < zt
            gid = int(f.gt_ids[i])
            first_in.setdefault(gid, k)
            if i in bev_ok and gid not in first_ok:
                first_ok[gid] = k
        # false objects: no 2-D match, no referenced GT (incl. ignored) within bev_m
        anyref = np.isfinite(gxy).all(1)
        for j in range(npd):
            if pr_on_gt[j] or not np.isfinite(pr[j]) or pr[j] > zs[-1]:
                continue
            if anyref.any() and np.isfinite(pxy[j]).all() and \
                    np.min(np.linalg.norm(gxy[anyref] - pxy[j], axis=1)) <= bev_m:
                continue
            out[_band(pr[j], edges, zs)]["false"] += 1
        # frame alarm level
        if (~ign).sum() == reg.sum():
            lt = alarm_zone(np.min(gr[reg]) if reg.any() else np.nan, zs)
            cand = pr[~pr_on_ign & np.isfinite(pr)]
            lp = alarm_zone(np.min(cand) if cand.size else np.nan, zs)
            conf[lt, lp] += 1
    res = {k: dict(v) for k, v in out.items()}
    res["frames"] = len(frames)
    res["lvl_conf"] = conf.tolist()
    n = int(conf.sum())
    res["lvl_n"] = n
    res["lvl_exact"] = int(np.trace(conf))
    res["lvl_under"] = int(np.tril(conf, -1).sum())
    res["lvl_over"] = int(np.triu(conf, 1).sum())
    res["lvl_miss_danger"] = int(conf[len(zs), :len(zs)].sum())
    if stamps is not None:
        st = np.asarray(stamps, float)
        res["delay_s"] = [float(st[first_ok[g]] - st[first_in[g]]) for g in first_in if g in first_ok]
        res["never"] = int(sum(g not in first_ok for g in first_in))
    return res


def alarm_summary(parts: list[dict], zones=ALARM_ZONES) -> dict:
    """Pool alarm_metrics results (segments): per band recall (found / bev / zone_ok / under over n), range error
    (mean |e|, bias, share within 1 / 2 m), false objects per 100 frames; frame-level rates; delay median / p90."""
    zs = tuple(sorted(float(z) for z in zones))
    frames = sum(p["frames"] for p in parts)
    res = {"frames": frames}
    for z in zs:
        k = f"b{z:g}"
        s = defaultdict(float)
        for p in parts:
            for a, v in p.get(k, {}).items():
                s[a] += v
        n, en = max(s["n"], 1), max(s["err_n"], 1)
        res[k] = dict(n=int(s["n"]), found=s["found"] / n, bev=s["bev"] / n, zone_ok=s["zone_ok"] / n,
                      under=s["under"] / n, err_abs=s["err_abs"] / en, err_bias=s["err_sum"] / en,
                      err_le1=s["err_le1"] / en, err_le2=s["err_le2"] / en,
                      false_per100=100.0 * s["false"] / max(frames, 1))
    ln = sum(p["lvl_n"] for p in parts)
    res["lvl"] = {k: sum(p[f"lvl_{k}"] for p in parts) / max(ln, 1) for k in ("exact", "under", "over")}
    nd = sum(np.sum(np.asarray(p["lvl_conf"])[len(zs)]) for p in parts)
    res["lvl"]["miss_danger"] = sum(p["lvl_miss_danger"] for p in parts) / max(nd, 1)
    res["lvl"]["n"] = ln
    d = [x for p in parts for x in p.get("delay_s", [])]
    res["delay"] = dict(n=len(d), median=float(np.median(d)) if d else np.nan,
                        p90=float(np.percentile(d, 90)) if d else np.nan,
                        never=int(sum(p.get("never", 0) for p in parts)))
    return res
