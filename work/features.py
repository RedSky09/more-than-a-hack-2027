"""Per-frame features shared by the experiments."""
import numpy as np

L, W = 105.3, 68.0


def player_speed(players, half_window=12):
    """Mean over visible players of displacement over +/- half_window frames, in m/s (NaN: none)."""
    px, py = players[:, :, 0] * L, players[:, :, 1] * W
    h = half_window
    dx = np.full_like(px, np.nan)
    dy = np.full_like(py, np.nan)
    dx[h:-h] = px[2 * h:] - px[:-2 * h]
    dy[h:-h] = py[2 * h:] - py[:-2 * h]
    speed = np.hypot(dx, dy) / (2 * h / 25.0)
    valid = ~np.isnan(speed)
    total = np.where(valid, speed, 0).sum(axis=1)
    count = valid.sum(axis=1)
    out = np.full(len(px), np.nan)
    ok = count > 0
    out[ok] = total[ok] / count[ok]
    return out


def rolling_mean(x, window):
    """Centred rolling mean ignoring NaN (window = total width in frames)."""
    x = np.asarray(x, float)
    valid = ~np.isnan(x)
    v = np.where(valid, x, 0.0)
    cs = np.concatenate(([0], np.cumsum(v)))
    cn = np.concatenate(([0], np.cumsum(valid)))
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - window // 2)
    hi = np.minimum(len(x), idx + window // 2 + 1)
    num = cs[hi] - cs[lo]
    den = cn[hi] - cn[lo]
    out = np.full(len(x), np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def min_run(mask, min_len):
    """Drop True-runs shorter than min_len."""
    mask = np.asarray(mask, bool)
    out = mask.copy()
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    for s, e in zip(edges[::2], edges[1::2]):
        if e - s < min_len:
            out[s:e] = False
    return out


def fill_short_gaps(mask, max_gap):
    """Merge True-runs separated by False-runs shorter than max_gap."""
    return ~min_run(~np.asarray(mask, bool), max_gap)
