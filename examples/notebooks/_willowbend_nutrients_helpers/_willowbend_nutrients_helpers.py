"""Helper functions for the Willowbend Creek Nutrients notebook.

These helpers classify mesh regions, build face polygons, and produce the
figures that appear in `03_willowbend_nutrients.ipynb`. Each `fig_*` function
returns a `matplotlib.figure.Figure` so the notebook can display it inline.

Adapted from `Publication-ClearWater-Riverine-02-Nutrients/notebooks/figures/generate_figures.py`.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection


def load_dataset(path):
    """Open the precomputed NSM1 simulation output as an xarray.Dataset."""
    p = Path(path)
    if p.suffix == ".zarr" or p.is_dir():
        return xr.open_zarr(str(p))
    return xr.open_dataset(p)


# ===================== REGION CLASSIFICATION =====================


def _largest_connected_component(ds, mask):
    if not mask.any():
        return mask.copy()
    n = mask.shape[0]
    efc = ds["edge_face_connectivity"].values
    adj = [[] for _ in range(n)]
    for e in range(efc.shape[0]):
        a, b = efc[e]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        a, b = int(a), int(b)
        if 0 <= a < n and 0 <= b < n:
            adj[a].append(b)
            adj[b].append(a)

    visited = np.zeros(n, dtype=bool)
    best = np.zeros(n, dtype=bool)
    for start in np.where(mask)[0]:
        if visited[start]:
            continue
        stack = [start]
        comp = []
        while stack:
            u = stack.pop()
            if visited[u] or not mask[u]:
                continue
            visited[u] = True
            comp.append(u)
            for v in adj[u]:
                if not visited[v] and mask[v]:
                    stack.append(v)
        if len(comp) > best.sum():
            best = np.zeros(n, dtype=bool)
            best[comp] = True
    return best


def classify_regions(ds):
    """Return masks for active cells, main channel, and oxbow, plus representative cell indices."""
    wet = ds["wetted_surface_area"].values
    vmag = ds["face_velocity_magnitude"].values
    depth = ds["face_hydraulic_depth"].values
    fx = ds["face_x"].values
    fy = ds["face_y"].values

    wet_frac = (wet > 0).sum(axis=0) / wet.shape[0]
    mean_wet = np.nanmean(wet, axis=0)
    mean_vmag = np.nanmean(vmag, axis=0)
    mean_depth = np.nanmean(depth, axis=0)

    active = _largest_connected_component(ds, wet_frac >= 0.95)

    # Geographic bounding box for the oxbow backwater
    ox_box = (fx >= 500450) & (fx <= 501300) & (fy >= 700) & (fy <= 1450)

    oxbow_candidate = active & ox_box & (mean_vmag < 0.08)

    # Four cells on the eastern margin of the oxbow have moderate velocity but
    # belong geographically to the oxbow.
    oxbow_annex = np.zeros_like(oxbow_candidate)
    for i in (131, 141, 142, 155):
        if i < oxbow_annex.size:
            oxbow_annex[i] = True
    oxbow_candidate = oxbow_candidate | oxbow_annex

    oxbow = _largest_connected_component(ds, oxbow_candidate)
    channel = active & (mean_vmag > 0.10) & ~oxbow

    chan_idx = np.where(channel)[0]
    upper_chan = chan_idx[fy[chan_idx] > 1500]
    upstream_idx = upper_chan[np.argmin(fx[upper_chan])] if len(upper_chan) else chan_idx[0]
    lower_chan = chan_idx[fy[chan_idx] < 1000]
    downstream_idx = lower_chan[np.argmax(fx[lower_chan])] if len(lower_chan) else chan_idx[-1]
    chan_x_mid = fx[chan_idx].mean()
    chan_y_mid = fy[chan_idx].mean()
    d2 = (fx[chan_idx] - chan_x_mid) ** 2 + (fy[chan_idx] - chan_y_mid) ** 2
    midchannel_idx = chan_idx[np.argmin(d2)]
    if oxbow.any():
        ox_idx = np.where(oxbow)[0]
        ox_cx = fx[ox_idx].mean()
        ox_cy = fy[ox_idx].mean()
        d2 = (fx[ox_idx] - ox_cx) ** 2 + (fy[ox_idx] - ox_cy) ** 2
        oxbow_center_idx = ox_idx[np.argmin(d2)]
    else:
        oxbow_center_idx = chan_idx[0]

    return {
        "active": active,
        "oxbow": oxbow,
        "channel": channel,
        "mean_wet": mean_wet,
        "mean_vmag": mean_vmag,
        "mean_depth": mean_depth,
        "upstream_idx": int(upstream_idx),
        "midchannel_idx": int(midchannel_idx),
        "downstream_idx": int(downstream_idx),
        "oxbow_center_idx": int(oxbow_center_idx),
    }


# ===================== MESH GEOMETRY =====================


def face_polygons(ds):
    """Return a list of face polygons as (Nx2) arrays and the face index for each."""
    nx = ds["node_x"].values
    ny = ds["node_y"].values
    face_nodes = ds["face_nodes"].values
    n_nodes = len(nx)
    polys = []
    face_ids = []
    for i in range(face_nodes.shape[0]):
        row = face_nodes[i]
        mask = np.isfinite(row) & (row >= 0) & (row < n_nodes)
        nodes = row[mask].astype(int)
        if len(nodes) < 3:
            continue
        polys.append(np.column_stack([nx[nodes], ny[nodes]]))
        face_ids.append(i)
    return polys, np.array(face_ids)


# ===================== STYLE CONSTANTS =====================


DRY_FILL = (0.94, 0.94, 0.94, 1.0)
DRY_EDGE = (0.55, 0.55, 0.55, 1.0)


def plot_polygons(ax, polys, face_ids, values, cmap="viridis", norm=None,
                  show_edges=False, edgecolor="k", lw=0.2, vmin=None, vmax=None,
                  mask=None):
    """Shade each face polygon by its cell-center value."""
    vals = values[face_ids]
    if mask is not None:
        cell_mask = mask[face_ids]
    else:
        cell_mask = np.ones(len(face_ids), dtype=bool)

    finite_vals = vals[cell_mask & np.isfinite(vals)]
    if norm is None:
        if vmin is None:
            vmin = float(np.nanmin(finite_vals)) if finite_vals.size else 0.0
        if vmax is None:
            vmax = float(np.nanmax(finite_vals)) if finite_vals.size else 1.0
        if vmin == vmax:
            vmax = vmin + max(1e-9, abs(vmin) * 1e-3)
        norm = Normalize(vmin=vmin, vmax=vmax)

    cmap_obj = plt.get_cmap(cmap)
    data_colors = cmap_obj(norm(np.clip(vals, norm.vmin, norm.vmax)))

    dry_polys = [polys[i] for i in range(len(polys)) if not cell_mask[i]]
    if dry_polys:
        coll_dry = PatchCollection(
            [MplPolygon(p, closed=True) for p in dry_polys],
            facecolors=DRY_FILL, edgecolors=DRY_EDGE, linewidths=0.15,
        )
        ax.add_collection(coll_dry)

    wet_polys = [polys[i] for i in range(len(polys)) if cell_mask[i]]
    wet_colors = np.array([data_colors[i] for i in range(len(polys)) if cell_mask[i]])
    if wet_polys:
        coll_wet = PatchCollection(
            [MplPolygon(p, closed=True) for p in wet_polys],
            facecolors=wet_colors,
            edgecolors=DRY_EDGE,
            linewidths=0.15,
        )
        ax.add_collection(coll_wet)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    return sm


def set_common_axis(ax, ds):
    ax.set_aspect("equal")
    ax.set_xlim(ds["face_x"].min() - 50, ds["face_x"].max() + 50)
    ax.set_ylim(ds["face_y"].min() - 50, ds["face_y"].max() + 50)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")


# ===================== FIGURE GENERATORS =====================


def fig_mesh_overview(ds, regions, polys, face_ids):
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    ax_mesh, ax_depth, ax_vel, ax_cls = axes.ravel()
    fx = ds["face_x"].values
    fy = ds["face_y"].values

    patches = [MplPolygon(p, closed=True) for p in polys]
    coll = PatchCollection(patches, facecolors="#e8f1f8", edgecolors="#2060a0", linewidths=0.4)
    ax_mesh.add_collection(coll)
    active = regions["active"]
    ax_mesh.scatter(fx[active], fy[active], c="#1f4e8c", s=5, zorder=3)
    set_common_axis(ax_mesh, ds)
    ax_mesh.set_title(f"(a) Computational mesh -- {len(polys)} cells, {active.sum()} active")

    depth_mask = _data_mask(regions["mean_depth"], active)
    depth_wet = regions["mean_depth"][depth_mask]
    sm = plot_polygons(ax_depth, polys, face_ids, regions["mean_depth"],
                       cmap="viridis",
                       vmin=float(depth_wet.min()) if depth_wet.size else 0.0,
                       vmax=float(depth_wet.max()) if depth_wet.size else 1.0,
                       mask=depth_mask)
    set_common_axis(ax_depth, ds)
    ax_depth.set_title("(b) Time-averaged hydraulic depth (m)")
    plt.colorbar(sm, ax=ax_depth, shrink=0.85, label="depth (m)")

    vmag_mask = _data_mask(regions["mean_vmag"], active, threshold=1e-3)
    vmag_wet = regions["mean_vmag"][vmag_mask]
    sm = plot_polygons(ax_vel, polys, face_ids, regions["mean_vmag"],
                       cmap="turbo",
                       vmin=float(vmag_wet.min()) if vmag_wet.size else 0.0,
                       vmax=float(vmag_wet.max()) if vmag_wet.size else 0.5,
                       mask=vmag_mask)
    set_common_axis(ax_vel, ds)
    ax_vel.set_title("(c) Time-averaged velocity magnitude (m/s)")
    plt.colorbar(sm, ax=ax_vel, shrink=0.85, label="|v| (m/s)")

    patches = [MplPolygon(p, closed=True) for p in polys]
    face_colors = np.full((len(polys), 4), (0.90, 0.90, 0.90, 1.0))
    for i, fid in enumerate(face_ids):
        if regions["oxbow"][fid]:
            face_colors[i] = (0.20, 0.63, 0.17, 1.0)
        elif regions["channel"][fid]:
            face_colors[i] = (0.12, 0.31, 0.55, 1.0)
        elif regions["active"][fid]:
            face_colors[i] = (0.70, 0.84, 0.92, 1.0)
    coll = PatchCollection(patches, facecolors=face_colors, edgecolors="#666", linewidths=0.2)
    ax_cls.add_collection(coll)
    set_common_axis(ax_cls, ds)
    ax_cls.set_title("(d) Regions: main channel, oxbow, floodplain margin")
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=(0.12, 0.31, 0.55), label=f"main channel ({regions['channel'].sum()})"),
        Patch(facecolor=(0.20, 0.63, 0.17), label=f"oxbow ({regions['oxbow'].sum()})"),
        Patch(facecolor=(0.70, 0.84, 0.92), label="floodplain margin"),
        Patch(facecolor=(0.90, 0.90, 0.90), label="inactive"),
    ]
    ax_cls.legend(handles=legend_elems, loc="center right",
                  bbox_to_anchor=(0.99, 0.38), fontsize=8, framealpha=0.95)
    fig.suptitle(
        "Willowbend Creek -- mesh and flow field "
        "(24-hour NSM1 simulation, 2022-05-13 08:00 to 2022-05-14 08:00)",
        fontsize=11, y=0.995,
    )
    fig.tight_layout()
    return fig


def select_time_index(ds, target_hour=12):
    t = ds["time"].values
    t0 = t[0]
    target = t0 + np.timedelta64(target_hour, "h")
    idx = int(np.argmin(np.abs(t - target)))
    return idx, t[idx]


def _data_mask(values, active, threshold=0.0):
    return active & np.isfinite(values) & (values > threshold)


def _spatial_panel(ax, ds, polys, face_ids, values, title, cmap, regions,
                   vmin=None, vmax=None, label="", data_threshold=0.0):
    active = regions["active"]
    mask = _data_mask(values, active, threshold=data_threshold)
    sm = plot_polygons(ax, polys, face_ids, values, cmap=cmap,
                       vmin=vmin, vmax=vmax, mask=mask)
    set_common_axis(ax, ds)
    ax.set_title(title)
    ox = regions["oxbow"]
    if ox.any():
        ox_polys = [polys[i] for i, fid in enumerate(face_ids) if ox[fid]]
        coll = PatchCollection(
            [MplPolygon(p, closed=True) for p in ox_polys],
            facecolors="none", edgecolors="#333333", linewidths=0.8,
        )
        ax.add_collection(coll)
    plt.colorbar(sm, ax=ax, shrink=0.82, label=label)


def _active_range_capped(val, active, threshold=0.0):
    mask = active & np.isfinite(val) & (val > threshold)
    v = val[mask]
    if v.size == 0:
        return 0.0, 1.0
    if v.size >= 2:
        sv = np.sort(v)
        vmax = float(sv[-2])
    else:
        vmax = float(v[0])
    vmin = float(v.min())
    if vmin == vmax:
        vmax = vmin + max(1e-9, abs(vmin) * 1e-3)
    return vmin, vmax


def _active_range(val, active, pad_frac=0.01):
    v = val[active]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo = float(v.min())
    hi = float(v.max())
    if hi == lo:
        pad = max(1e-9, abs(lo) * pad_frac)
        return lo - pad, hi + pad
    gap_thresh = 0.50
    lower_thresh = 0.35
    sv = np.sort(v)
    total_range = sv[-1] - sv[0]
    if total_range > 0 and sv.size > 4:
        gaps = np.diff(sv)
        k = int(np.argmax(gaps))
        gap = float(gaps[k])
        below_frac = (k + 1) / sv.size
        if (gap / total_range) > gap_thresh and below_frac <= lower_thresh:
            lo = float(sv[k + 1])
    return lo, hi


def fig_velocity_field(ds, regions, polys, face_ids):
    tidx, tstamp = select_time_index(ds, target_hour=12)
    vx = ds["face_velocity_x"].isel(time=tidx).values
    vy = ds["face_velocity_y"].isel(time=tidx).values
    vm = np.hypot(vx, vy)
    act = regions["active"]

    fig, ax = plt.subplots(figsize=(10, 8))
    vm_mask = _data_mask(vm, act, threshold=1e-3)
    vm_wet = vm[vm_mask]
    sm = plot_polygons(ax, polys, face_ids, vm, cmap="turbo",
                       vmin=float(vm_wet.min()) if vm_wet.size else 0.0,
                       vmax=float(vm_wet.max()) if vm_wet.size else 0.5,
                       mask=vm_mask)
    fx = ds["face_x"].values; fy = ds["face_y"].values
    sub = act.copy()
    idx_sub = np.where(sub)[0][::2]
    ax.quiver(fx[idx_sub], fy[idx_sub], vx[idx_sub], vy[idx_sub],
              scale=6, width=0.0025, color="white")
    set_common_axis(ax, ds)
    ts = pd.Timestamp(tstamp).strftime("%Y-%m-%d %H:%M UTC")
    ax.set_title(f"Velocity field at {ts}")
    plt.colorbar(sm, ax=ax, shrink=0.85, label="|v| (m/s)")
    fig.tight_layout()
    return fig


def fig_nitrogen_spatial(ds, regions, polys, face_ids, tidx, tstamp):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    nh4 = ds["NH4"].isel(time=tidx).values
    no3 = ds["NO3"].isel(time=tidx).values
    orgn = ds["OrgN"].isel(time=tidx).values
    act = regions["active"]
    for ax, val, name, cmap, units in [
        (axes[0], nh4, r"NH$_4^+$ (ammonium)", "YlOrBr", "mg-N/L"),
        (axes[1], no3, r"NO$_3^-$ (nitrate)", "YlOrBr", "mg-N/L"),
        (axes[2], orgn, "OrgN (organic nitrogen)", "YlOrBr", "mg-N/L"),
    ]:
        vmin, vmax = _active_range(val, act)
        _spatial_panel(ax, ds, polys, face_ids, val,
                       f"{name}", cmap, regions, vmin=vmin, vmax=vmax, label=units)
    ts = pd.Timestamp(tstamp).strftime("%Y-%m-%d %H:%M UTC")
    fig.suptitle(f"Simulated nitrogen species distributions at {ts}", fontsize=11)
    fig.tight_layout()
    return fig


def fig_phosphorus_spatial(ds, regions, polys, face_ids, tidx, tstamp):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    tip = ds["TIP"].isel(time=tidx).values
    orgp = ds["OrgP"].isel(time=tidx).values
    act = regions["active"]
    for ax, val, name, cmap, cap_outlier in [
        (axes[0], tip, "TIP (total inorganic P)", "viridis", False),
        (axes[1], orgp, "OrgP (organic P)", "viridis", True),
    ]:
        if cap_outlier:
            vmin, vmax = _active_range_capped(val, act)
        else:
            vmin, vmax = _active_range(val, act)
        _spatial_panel(ax, ds, polys, face_ids, val,
                       f"{name}", cmap, regions, vmin=vmin, vmax=vmax, label="mg-P/L")
    ts = pd.Timestamp(tstamp).strftime("%Y-%m-%d %H:%M UTC")
    fig.suptitle(f"Simulated phosphorus distribution at {ts}", fontsize=11)
    fig.tight_layout()
    return fig


def fig_do_spatial(ds, regions, polys, face_ids):
    t = ds["time"].values
    act = regions["active"]
    do_all = ds["DOX"].values
    ox = regions["oxbow"]
    ox_do = do_all[:, ox].mean(axis=1)
    idx_peak = int(np.argmax(ox_do))
    idx_trough = int(np.argmin(ox_do))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    active_vals = np.concatenate([do_all[idx_peak][act], do_all[idx_trough][act]])
    active_vals = active_vals[np.isfinite(active_vals)]
    vmin = float(active_vals.min())
    vmax = float(active_vals.max())
    for ax, idx, label in [
        (axes[0], idx_peak, "oxbow DO peak"),
        (axes[1], idx_trough, "oxbow DO trough"),
    ]:
        val = ds["DOX"].isel(time=idx).values
        ts = pd.Timestamp(t[idx]).strftime("%Y-%m-%d %H:%M UTC")
        _spatial_panel(ax, ds, polys, face_ids, val,
                       f"DO at {ts} ({label})",
                       "turbo", regions, vmin=vmin, vmax=vmax,
                       label=r"mg-O$_2$/L")
    fig.suptitle("Simulated dissolved oxygen: within-simulation extremes", fontsize=11)
    fig.tight_layout()
    return fig


def fig_algae_spatial(ds, regions, polys, face_ids, tidx, tstamp):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    ap = ds["Ap"].isel(time=tidx).values
    ab = ds["Ab"].isel(time=tidx).values
    act = regions["active"]
    vmin_p, vmax_p = _active_range(ap, act)
    _spatial_panel(axes[0], ds, polys, face_ids, ap,
                   r"A$_p$ -- phytoplankton",
                   "YlGn", regions, vmin=vmin_p, vmax=vmax_p,
                   label=r"$\mu$g-Chl-a/L")
    vmin_b, vmax_b = _active_range_capped(ab, act)
    _spatial_panel(axes[1], ds, polys, face_ids, ab,
                   r"A$_b$ -- benthic algae",
                   "YlGn", regions, vmin=vmin_b, vmax=vmax_b,
                   label="g-D/m$^2$")
    ts = pd.Timestamp(tstamp).strftime("%Y-%m-%d %H:%M UTC")
    fig.suptitle(f"Simulated algal biomass distribution at {ts}", fontsize=11)
    fig.tight_layout()
    return fig


def fig_carbon_spatial(ds, regions, polys, face_ids, tidx, tstamp):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    act = regions["active"]
    for ax, varname, cmap, label in [
        (axes[0], "DOC", "YlOrBr", "mg-C/L"),
        (axes[1], "POC", "YlOrBr", "mg-C/L"),
        (axes[2], "CBOD", "YlOrBr", r"mg-O$_2$/L"),
    ]:
        val = ds[varname].isel(time=tidx).values
        vmin, vmax = _active_range_capped(val, act)
        _spatial_panel(ax, ds, polys, face_ids, val, varname, cmap, regions,
                       vmin=vmin, vmax=vmax, label=label)
    ts = pd.Timestamp(tstamp).strftime("%Y-%m-%d %H:%M UTC")
    fig.suptitle(f"Simulated dissolved/particulate organic carbon and CBOD at {ts}", fontsize=11)
    fig.tight_layout()
    return fig


def fig_limitation_spatial(ds, regions, polys, face_ids, tidx, tstamp):
    KsN = 0.04
    KsP = 0.0012
    nh4 = ds["NH4"].isel(time=tidx).values
    no3 = ds["NO3"].isel(time=tidx).values
    tip = ds["TIP"].isel(time=tidx).values
    fn = (nh4 + no3) / (KsN + nh4 + no3)
    fp = tip / (KsP + tip)
    min_lim = np.minimum(fn, fp)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    for ax, val, name in [
        (axes[0], fn, r"F$_N$ (nitrogen)"),
        (axes[1], fp, r"F$_P$ (phosphorus)"),
        (axes[2], min_lim, r"min(F$_N$, F$_P$) -- limiting nutrient"),
    ]:
        _spatial_panel(ax, ds, polys, face_ids, val, name,
                       "viridis", regions, vmin=0.8, vmax=1.0, label="limitation factor")
    ts = pd.Timestamp(tstamp).strftime("%Y-%m-%d %H:%M UTC")
    fig.suptitle(f"Algal nutrient-limitation factors (Michaelis-Menten, derived) at {ts}", fontsize=11)
    fig.tight_layout()
    return fig


def _timeseries_panel(ax, ds, varname, cell_indices, labels, ylabel, colors=None):
    t = ds["time"].values
    th = (t - t[0]) / np.timedelta64(1, "h")
    if colors is None:
        colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    for idx, lab, color in zip(cell_indices, labels, colors):
        series = ds[varname].isel(nface=idx).values
        ax.plot(th, series, lw=1.4, label=lab, color=color)
    ax.set_xlabel("hours from start of simulation")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def fig_do_timeseries(ds, regions):
    idxs = [regions["upstream_idx"], regions["midchannel_idx"],
            regions["downstream_idx"], regions["oxbow_center_idx"]]
    labels = ["upstream channel", "mid channel", "downstream channel", "oxbow centroid"]
    fig, ax = plt.subplots(figsize=(11, 5))
    _timeseries_panel(ax, ds, "DOX", idxs, labels, r"DO (mg-O$_2$/L)")
    ax.set_title("Dissolved oxygen time series at representative cells")
    fig.tight_layout()
    return fig


def fig_algae_timeseries(ds, regions):
    idxs = [regions["upstream_idx"], regions["midchannel_idx"],
            regions["downstream_idx"], regions["oxbow_center_idx"]]
    labels = ["upstream channel", "mid channel", "downstream channel", "oxbow centroid"]
    fig, ax = plt.subplots(figsize=(11, 5))
    _timeseries_panel(ax, ds, "Ap", idxs, labels, r"A$_p$ ($\mu$g-Chl-a/L)")
    ax.set_title(r"Phytoplankton (A$_p$) time series at representative cells")
    fig.tight_layout()
    return fig


def fig_nutrient_timeseries(ds, regions):
    idx = regions["midchannel_idx"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    t = ds["time"].values
    th = (t - t[0]) / np.timedelta64(1, "h")
    for name, color in [("NH4", "#d62728"), ("NO3", "#1f77b4"), ("OrgN", "#9467bd")]:
        axes[0].plot(th, ds[name].isel(nface=idx).values, lw=1.4, label=name, color=color)
    axes[0].set_xlabel("hours from start"); axes[0].set_ylabel("mg-N/L")
    axes[0].set_title("(a) Nitrogen species at mid-channel")
    axes[0].grid(alpha=0.3); axes[0].legend()
    for name, color in [("TIP", "#2ca02c"), ("OrgP", "#ff7f0e")]:
        axes[1].plot(th, ds[name].isel(nface=idx).values, lw=1.4, label=name, color=color)
    axes[1].set_xlabel("hours from start"); axes[1].set_ylabel("mg-P/L")
    axes[1].set_title("(b) Phosphorus species at mid-channel")
    axes[1].grid(alpha=0.3); axes[1].legend()
    fig.suptitle("Nutrient time series at a representative mid-channel cell", fontsize=11)
    fig.tight_layout()
    return fig


def fig_oxbow_averaged_timeseries(ds, regions):
    ox = regions["oxbow"]
    t = ds["time"].values
    th = (t - t[0]) / np.timedelta64(1, "h")

    def ox_mean(varname):
        da = ds[varname].isel(nface=np.where(ox)[0])
        return da.mean(dim="nface").values

    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
    panels = [
        ("DOX", r"DO (mg-O$_2$/L)", "Dissolved oxygen", "#1f77b4"),
        ("Ap", r"A$_p$ ($\mu$g-Chl-a/L)", "Phytoplankton", "#2ca02c"),
        ("CBOD", r"CBOD (mg-O$_2$/L)", "Carbonaceous BOD", "#8c564b"),
        ("NH4", r"NH$_4$ (mg-N/L)", "Ammonium", "#d62728"),
        ("NO3", r"NO$_3$ (mg-N/L)", "Nitrate", "#1f77b4"),
        ("OrgN", "OrgN (mg-N/L)", "Organic nitrogen", "#9467bd"),
        ("TIP", "TIP (mg-P/L)", "Total inorganic P", "#2ca02c"),
        ("OrgP", "OrgP (mg-P/L)", "Organic phosphorus", "#ff7f0e"),
        ("DOC", "DOC (mg-C/L)", "Dissolved organic C", "#17becf"),
    ]
    for ax, (var, ylab, title, color) in zip(axes.ravel(), panels):
        series = ox_mean(var)
        ax.plot(th, series, lw=1.6, color=color)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    for ax in axes[-1, :]:
        ax.set_xlabel("hours from start of simulation")
    fig.suptitle(
        f"Oxbow-averaged water quality time series (mean over {int(ox.sum())} oxbow cells)",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


def fig_algae_do_coupling(ds, regions):
    active = regions["active"]
    do = ds["DOX"].values
    ap = ds["Ap"].values
    do_amp = np.nanmax(do, axis=0) - np.nanmin(do, axis=0)
    ap_mean = np.nanmean(ap, axis=0)
    t = ds["time"].values
    th = (t - t[0]) / np.timedelta64(1, "h")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    ax = axes[0]
    ch_mask = regions["channel"]
    ox_mask = regions["oxbow"]
    other_mask = active & ~ch_mask & ~ox_mask
    ax.scatter(ap_mean[other_mask], do_amp[other_mask], s=18, c="#9ec9e0",
               label="floodplain margin", edgecolors="none")
    ax.scatter(ap_mean[ch_mask], do_amp[ch_mask], s=22, c="#d62728",
               label="main channel", edgecolors="none")
    ax.scatter(ap_mean[ox_mask], do_amp[ox_mask], s=28, c="#2ca02c",
               label="oxbow", edgecolors="none")
    ax.set_xlabel(r"mean A$_p$ ($\mu$g-Chl-a/L)")
    ax.set_ylabel(r"DO diurnal amplitude (mg-O$_2$/L)")
    ax.set_title("(a) Chlorophyll-a vs. DO diurnal amplitude")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    ox_ap = ds["Ap"].isel(nface=np.where(ox_mask)[0]).mean(dim="nface").values
    ox_do = ds["DOX"].isel(nface=np.where(ox_mask)[0]).mean(dim="nface").values
    l1, = ax.plot(th, ox_do, lw=1.6, color="#1f77b4", label="DO (left axis)")
    ax.set_xlabel("hours from start")
    ax.set_ylabel(r"DO (mg-O$_2$/L)", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    l2, = ax2.plot(th, ox_ap, lw=1.6, color="#2ca02c", label=r"A$_p$ (right axis)")
    ax2.set_ylabel(r"A$_p$ ($\mu$g-Chl-a/L)", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax.set_title(r"(b) Oxbow-averaged DO and A$_p$")
    ax.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="best")

    fig.suptitle("Algae-DO coupling", fontsize=11)
    fig.tight_layout()
    return fig
