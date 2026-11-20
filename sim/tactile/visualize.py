"""触觉可视化（Stage 0 / 0-2 时序图）。"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_timeseries(log: dict, out_path: str, title: str = "tactile sanity") -> str:
    """log 为各字段的 (T, ...) 数组：
       cheap (T,2), normal (T,2), shear (T,2), gripper (T,), [phase (T,) 可选]。"""
    cheap = np.asarray(log["cheap"])
    normal = np.asarray(log["normal"])
    shear = np.asarray(log["shear"])
    gripper = np.asarray(log["gripper"])
    T = len(cheap)
    t = np.arange(T)

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)


    axes[0].step(t, cheap[:, 0], where="post", label="left contact", color="tab:blue")
    axes[0].step(t, cheap[:, 1] + 1.2, where="post", label="right contact", color="tab:orange")
    axes[0].set_yticks([0, 1, 1.2, 2.2])
    axes[0].set_yticklabels(["L=0", "L=1", "R=0", "R=1"])
    axes[0].set_ylabel("cheap (0/1)")
    axes[0].legend(loc="upper right", fontsize=8)


    axes[1].plot(t, normal[:, 0], label="left normal", color="tab:blue")
    axes[1].plot(t, normal[:, 1], label="right normal", color="tab:orange")
    axes[1].set_ylabel("normal force")
    axes[1].legend(loc="upper right", fontsize=8)


    axes[2].plot(t, shear[:, 0], label="left shear", color="tab:blue")
    axes[2].plot(t, shear[:, 1], label="right shear", color="tab:orange")
    axes[2].set_ylabel("shear force")
    axes[2].legend(loc="upper right", fontsize=8)


    axes[3].plot(t, gripper, label="gripper qpos sum", color="tab:green")
    axes[3].set_ylabel("gripper open/close")
    axes[3].set_xlabel("step")
    axes[3].legend(loc="upper right", fontsize=8)


    any_contact = (cheap.sum(axis=1) > 0).astype(float)
    for ax in axes:
        ax.fill_between(t, *ax.get_ylim(), where=any_contact > 0, color="red", alpha=0.06, step="post")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


import mujoco  # noqa: E402

_FORCE_FLAG = int(mujoco.mjtVisFlag.mjVIS_CONTACTFORCE)


def fingerpad_force_arrows(env, pads: dict) -> dict:
    """每个指垫聚合成一个 (接触点中心, 世界力向量)。无接触则为 None。"""
    sim = env.sim
    m, d = sim.model._model, sim.data._data

    def agg(geom_names):
        ids = {sim.model.geom_name2id(n) for n in geom_names}
        pts, force, w6 = [], np.zeros(3), np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 in ids or c.geom2 in ids:
                mujoco.mj_contactForce(m, d, i, w6)
                fr = np.array(c.frame).reshape(3, 3)
                force = force + w6[0] * fr[0] + w6[1] * fr[1] + w6[2] * fr[2]
                pts.append(np.array(c.pos))
        if not pts:
            return None
        return np.mean(pts, axis=0), force

    return {"left": agg(pads["left"]), "right": agg(pads["right"])}


def render_fingerpad_arrows(env, pads: dict, cam: str = "agentview", w: int = 384, h: int = 384,
                            scale: float = 0.02, width: float = 0.008):
    """渲染一帧，**只在两个指垫**画力箭头（左红/右蓝），场景其它接触一律不画。"""
    sim = env.sim
    sim.render(camera_name=cam, width=w, height=h)
    rc = sim._render_context_offscreen
    m, d = sim.model._model, sim.data._data

    arrows = fingerpad_force_arrows(env, pads)
    rc.vopt.flags[_FORCE_FLAG] = 0
    rc.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    rc.cam.fixedcamid = sim.model.camera_name2id(cam)
    mujoco.mjv_updateScene(m, d, rc.vopt, rc.pert, rc.cam, mujoco.mjtCatBit.mjCAT_ALL, rc.scn)

    colors = {"left": (1.0, 0.0, 0.0, 1.0), "right": (0.0, 0.4, 1.0, 1.0)}
    for side, a in arrows.items():
        if a is not None and rc.scn.ngeom < rc.scn.maxgeom:
            p0 = a[0]
            p1 = a[0] + scale * a[1]
            gm = rc.scn.geoms[rc.scn.ngeom]
            mujoco.mjv_initGeom(gm, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                                np.zeros(9), np.array(colors[side], np.float32))
            mujoco.mjv_connector(gm, mujoco.mjtGeom.mjGEOM_ARROW, width, p0, p1)
            rc.scn.ngeom += 1

    vp = mujoco.MjrRect(0, 0, w, h)
    mujoco.mjr_render(vp, rc.scn, rc.con)
    rgb = np.empty((h, w, 3), np.uint8)
    mujoco.mjr_readPixels(rgb, None, vp, rc.con)
    return rgb[::-1]


def render_frame(env, cam: str = "agentview", w: int = 256, h: int = 256):
    """渲染一帧（已翻正：MuJoCo 离屏渲染默认上下颠倒）。"""
    img = env.sim.render(camera_name=cam, width=w, height=h)
    return np.ascontiguousarray(img[::-1])


def save_video(frames, out_path: str, fps: int = 20) -> str:
    import imageio

    imageio.mimsave(out_path, [np.asarray(f) for f in frames], fps=fps)
    return out_path


import matplotlib.patheffects as _pe  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SHEAR_ARROW_MIN = 0.1

_FORCE_CMAP = LinearSegmentedColormap.from_list(
    "white_orange_red", ["#ffffff", "#ffe0b0", "#ff9a30", "#ff5a00", "#d92600"])
_TXT_FX = [_pe.withStroke(linewidth=2.0, foreground="white")]


def _draw_force_region(ax, region, title, vmax, force_decimals: int = 0):
    """画一个传感区（指垫 n×n 或爪尖 1×1）：法向力热力图底 + 每格标 N/S 数值 + 绿箭头指切向方向。"""
    region = np.asarray(region)
    n = region.shape[0]
    step = 16.0 / n
    normal_map = region[..., 0].T
    ax.imshow(normal_map, origin="lower", extent=[-8, 8, -8, 8], cmap=_FORCE_CMAP, vmin=0, vmax=vmax)
    ax.set_xticks([-8, 0, 8])
    ax.set_yticks([-8, 0, 8])
    ax.set_xticklabels(["-8", "0", "+8"], fontsize=6)
    ax.set_yticklabels(["-8", "0", "+8"], fontsize=6)
    ax.tick_params(length=2, pad=1)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-8, 8)
    ax.set_ylim(8, -8)
    ax.autoscale(False)
    ax.set_xlabel("pad x (mm),  +x ->", fontsize=8)
    ax.set_ylabel("pad z (mm)", fontsize=8)
    ax.text(0, 7.2, "TIP", ha="center", va="center", fontsize=7, color="black",
            fontweight="bold", path_effects=_TXT_FX)
    ax.text(0, -7.2, "base", ha="center", va="center", fontsize=6, color="0.35", path_effects=_TXT_FX)
    fs = max(7.0, min(16.0, step * 1.2))
    for ix in range(n):
        for iz in range(n):
            nf = float(region[ix, iz, 0])
            sf = float(region[ix, iz, 1])
            if nf <= 0 and sf <= 0:
                continue
            cx = -8 + (ix + 0.5) * step
            cz = -8 + (iz + 0.5) * step
            ax.text(cx, cz - 0.27 * step,
                    f"Fn={nf:.{force_decimals}f}\nFs={sf:.{force_decimals}f}", ha="center", va="center",
                    fontsize=fs, color="black", fontweight="bold", path_effects=_TXT_FX)
            if region.shape[2] >= 4:
                su, sv = float(region[ix, iz, 2]), float(region[ix, iz, 3])
                mag = (su * su + sv * sv) ** 0.5
                if mag > SHEAR_ARROW_MIN:


                    L = min(0.45 * step, mag * 0.45 * step)
                    hw = max(0.04 * step, 0.35 * L)
                    ax.arrow(cx, cz + 0.05 * step, su / mag * L, sv / mag * L,
                             head_width=hw, head_length=hw,
                             fc="#0050d0", ec="#0050d0", length_includes_head=True, linewidth=1.4, zorder=5)


def _region_totals(g, shear_curve="magnitude_sum"):
    """该帧 (总法向力, 总切向力模长)，对 4 个传感区求和；g 为 None 返回 (0,0)。"""
    if g is None:
        return 0.0, 0.0
    keys = ("left_pad", "right_pad", "left_tip", "right_tip")
    fn = sum(float(np.asarray(g[k])[..., 0].sum()) for k in keys)
    if shear_curve == "pad_vector_resultant":
        pads = (np.asarray(g["left_pad"]), np.asarray(g["right_pad"]))
        if any(pad.shape[-1] < 4 for pad in pads):
            raise ValueError("pad_vector_resultant requires with_shear_dir=True grids")
        su = sum(float(pad[..., 2].sum()) for pad in pads)
        sv = sum(float(pad[..., 3].sum()) for pad in pads)
        fs = float(np.hypot(su, sv))
    elif shear_curve == "magnitude_sum":
        fs = sum(float(np.asarray(g[k])[..., 1].sum()) for k in keys)
    else:
        raise ValueError(f"unknown shear_curve: {shear_curve}")
    return fn, fs


def save_tactile_force_video(scene_frames, grids, out_path, fps: int = 10, scale: int = 2,
                             force_decimals: int = 0, title=None, shear_curve="magnitude_sum"):
    """逐帧合成 mp4：左=抓取画面 | 中=2×2 受力面板 | 右=Fn/Fs 随时间曲线(双Y轴+当前帧游标)。"""
    import imageio

    if not scene_frames:
        return out_path
    sc0 = np.asarray(scene_frames[0])
    scale = max(1, int(scale))
    h_px = sc0.shape[0] * scale


    Fn_s, Fs_s = [], []
    for g in grids:
        fn, fs = _region_totals(g, shear_curve=shear_curve)
        Fn_s.append(fn)
        Fs_s.append(fs)
    T = len(grids)
    fn_max = max(Fn_s + [1e-6]) * 1.1
    fs_max = max(Fs_s + [1e-6]) * 1.1


    fig = plt.figure(figsize=(1.75 * h_px / 100.0, h_px / 100.0), dpi=100)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.5])
    ax_p = [[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]]
    ax_c = fig.add_subplot(gs[:, 2])
    ax_c2 = ax_c.twinx()

    _blank = {"left_pad": np.zeros((1, 1, 4)), "right_pad": np.zeros((1, 1, 4)),
              "left_tip": np.zeros(4), "right_tip": np.zeros(4)}
    up = np.ones((scale, scale, 1), dtype=sc0.dtype)

    out = []
    for t, (scene, grid) in enumerate(zip(scene_frames, grids)):
        g = grid if grid is not None else _blank
        regions = {
            "left_pad": np.asarray(g["left_pad"]),
            "right_pad": np.asarray(g["right_pad"]),
            "left_tip": np.asarray(g["left_tip"]).reshape(1, 1, -1),
            "right_tip": np.asarray(g["right_tip"]).reshape(1, 1, -1),
        }
        vmax = max([float(r[..., 0].max()) for r in regions.values()] + [1e-6])
        for row in ax_p:
            for ax in row:
                ax.clear()
        _draw_force_region(ax_p[0][0], regions["left_pad"], "LEFT PAD", vmax, force_decimals)
        _draw_force_region(ax_p[0][1], regions["right_pad"], "RIGHT PAD", vmax, force_decimals)
        _draw_force_region(ax_p[1][0], regions["left_tip"], "LEFT TIP", vmax, force_decimals)
        _draw_force_region(ax_p[1][1], regions["right_tip"], "RIGHT TIP", vmax, force_decimals)


        ax_c.clear()
        ax_c2.clear()
        xs = list(range(t + 1))
        ax_c.plot(xs, Fn_s[:t + 1], color="tab:red", lw=1.4)
        ax_c2.plot(xs, Fs_s[:t + 1], color="tab:blue", lw=1.4)
        ax_c.axvline(t, color="0.5", ls="--", lw=0.8)
        ax_c.set_xlim(0, max(T - 1, 1))
        ax_c.set_ylim(0, fn_max)
        ax_c2.set_ylim(0, fs_max)
        ax_c.set_xlabel("step", fontsize=8)
        ax_c.set_ylabel("Fn total normal (N)", color="tab:red", fontsize=8)
        shear_label = "|sum Ft| pad resultant (N)" if shear_curve == "pad_vector_resultant" else "Fs total shear (N)"
        ax_c2.set_ylabel(shear_label, color="tab:blue", fontsize=8)
        ax_c.tick_params(axis="y", labelcolor="tab:red", labelsize=6)
        ax_c2.tick_params(axis="y", labelcolor="tab:blue", labelsize=6)
        ax_c.tick_params(axis="x", labelsize=6)
        ax_c.set_title(f"force vs time (total)   t={t}", fontsize=9)

        prefix = f"{title}   |   " if title else ""
        fig.suptitle(prefix + "Fn=normal(N)  Fs=shear(N)  green arrow = shear ON pad   +x right, TIP bottom",
                     fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        panel = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3]
        sc = np.kron(np.asarray(scene), up)
        hh = min(sc.shape[0], panel.shape[0])
        out.append(np.hstack([sc[:hh], panel[:hh]]))
    plt.close(fig)
    imageio.mimwrite(out_path, out, fps=fps)
    return out_path
