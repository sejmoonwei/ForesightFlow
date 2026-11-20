"""触觉读取纯函数（Stage 0 / 0-1 + 0-4 读取部分）。"""

from __future__ import annotations

import re

import numpy as np
import mujoco
from robosuite.utils import sim_utils


def _unwrap_robosuite_env(env):
    """从 LIBERO 包装 env 取底层 robosuite env（持有 robots / sim）。"""
    return getattr(env, "env", env)


def resolve_pads(env) -> dict:
    """一次性解析左右指垫的 geom 全名（运行时带前缀）。"""
    rs_env = _unwrap_robosuite_env(env)
    gripper = rs_env.robots[0].gripper
    return {
        "left": list(gripper.important_geoms["left_fingerpad"]),
        "right": list(gripper.important_geoms["right_fingerpad"]),
    }


def _normal_shear_for(sim, geom_ids: set) -> tuple[float, float]:
    """该指垫所有相关接触的 (法向力之和, 剪切力模长之和)。"""
    m, d = sim.model._model, sim.data._data
    w = np.zeros(6, dtype=np.float64)
    fn_sum = 0.0
    fs_sum = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if c.geom1 in geom_ids or c.geom2 in geom_ids:
            mujoco.mj_contactForce(m, d, i, w)
            fn_sum += abs(float(w[0]))
            fs_sum += float(np.linalg.norm(w[1:3]))
    return fn_sum, fs_sum


def read_tactile(sim, pads: dict) -> dict:
    """每步纯函数：给定 sim 状态 + 指垫 geom 名，返回触觉读数。"""
    left_geoms = pads["left"]
    right_geoms = pads["right"]

    cheap = np.array(
        [
            float(sim_utils.check_contact(sim, left_geoms)),
            float(sim_utils.check_contact(sim, right_geoms)),
        ],
        dtype=np.float32,
    )

    left_ids = {sim.model.geom_name2id(n) for n in left_geoms}
    right_ids = {sim.model.geom_name2id(n) for n in right_geoms}
    fn_l, fs_l = _normal_shear_for(sim, left_ids)
    fn_r, fs_r = _normal_shear_for(sim, right_ids)

    return {
        "cheap": cheap,
        "normal": np.array([fn_l, fn_r], dtype=np.float32),
        "shear": np.array([fs_l, fs_r], dtype=np.float32),
    }


def tactile(env, pads: dict | None = None) -> dict:
    """便捷在线入口：自动解析 pads（若未给）并读当前 sim 状态。"""
    if pads is None:
        pads = resolve_pads(env)
    return read_tactile(env.sim, pads)


def _subgeom_cell(name):
    """从子 geom 名 `..._x{i}_z{j}` 解析格 (i_x, j_z)；非子 geom 返回 None。"""
    mo = re.search(r"_x(\d+)_z(\d+)$", name)
    return (int(mo.group(1)), int(mo.group(2))) if mo else None


def _subgeom_pad_grid(sim, geom_names, n_pad, nch):
    """子 geom=一格读法（0-6 拆 geom 时用）：接触按 **geom 归属**落格，力归属确定（不靠投影舍入）。"""
    m, d = sim.model._model, sim.data._data
    cell = {}
    maxidx = 0
    for nm in geom_names:
        ij = _subgeom_cell(nm)
        if ij is None:
            continue
        gid = sim.model.geom_name2id(nm)
        cell[gid] = (ij[0], ij[1], np.array(d.geom_xmat[gid]).reshape(3, 3))
        maxidx = max(maxidx, ij[0], ij[1])
    n_sub = maxidx + 1
    g = np.zeros((n_pad, n_pad, nch), np.float32)
    w6 = np.zeros(6)
    for k in range(d.ncon):
        c = d.contact[k]
        gid = c.geom1 if c.geom1 in cell else (c.geom2 if c.geom2 in cell else None)
        if gid is None:
            continue
        i, j, R = cell[gid]
        a = min(n_pad - 1, i * n_pad // n_sub)
        b = min(n_pad - 1, j * n_pad // n_sub)
        mujoco.mj_contactForce(m, d, k, w6)
        g[a, b, 0] += abs(w6[0])
        g[a, b, 1] += np.linalg.norm(w6[1:3])
        if nch >= 4:
            fr = np.array(c.frame).reshape(3, 3)
            shear_w = -(w6[1] * fr[1] + w6[2] * fr[2])
            g[a, b, 2] += float(shear_w @ R[:, 0])
            g[a, b, 3] += float(shear_w @ R[:, 2])
    return g


def read_tactile_grid(env, pads, n_pad=4, with_shear_dir=False):
    """Stage 0.5 / Step 1：34 格触觉表征（指垫 2×(4×4) + 爪尖 2×(1×1)），binning 读出、不改模型。"""
    sim = env.sim
    m, d = sim.model._model, sim.data._data
    bR = np.array(d.xmat[m.body("gripper0_right_gripper").id]).reshape(3, 3)
    nch = 4 if with_shear_dir else 2
    out = {}
    w6 = np.zeros(6)

    for side in ("left", "right"):
        if len(pads[side]) > 1:
            out[f"{side}_pad"] = _subgeom_pad_grid(sim, pads[side], n_pad, nch)
            continue
        gid = sim.model.geom_name2id(pads[side][0])
        R = np.array(d.geom_xmat[gid]).reshape(3, 3)
        o = np.array(d.geom_xpos[gid])
        g = np.zeros((n_pad, n_pad, nch), np.float32)
        for i in range(d.ncon):
            c = d.contact[i]
            if gid in (c.geom1, c.geom2):
                mujoco.mj_contactForce(m, d, i, w6)
                loc = R.T @ (np.array(c.pos) - o) * 1000.0
                a = min(n_pad - 1, max(0, int((loc[0] + 8) / 16 * n_pad)))
                b = min(n_pad - 1, max(0, int((loc[2] + 8) / 16 * n_pad)))
                g[a, b, 0] += abs(w6[0])
                g[a, b, 1] += np.linalg.norm(w6[1:3])
                if with_shear_dir:
                    fr = np.array(c.frame).reshape(3, 3)


                    shear_w = -(w6[1] * fr[1] + w6[2] * fr[2])
                    g[a, b, 2] += float(shear_w @ R[:, 0])
                    g[a, b, 3] += float(shear_w @ R[:, 2])
        out[f"{side}_pad"] = g

    fg = {"left": "gripper0_finger1_collision", "right": "gripper0_finger2_collision"}
    for side, name in fg.items():
        gid = sim.model.geom_name2id(name)
        Rf = np.array(d.geom_xmat[gid]).reshape(3, 3)
        t = np.zeros(nch, np.float32)
        for i in range(d.ncon):
            c = d.contact[i]
            if gid in (c.geom1, c.geom2):
                if abs((bR.T @ np.array(c.frame[:3]))[2]) < 0.6:
                    continue
                mujoco.mj_contactForce(m, d, i, w6)
                t[0] += abs(w6[0])
                t[1] += np.linalg.norm(w6[1:3])
                if with_shear_dir:
                    fr = np.array(c.frame).reshape(3, 3)
                    shear_w = -(w6[1] * fr[1] + w6[2] * fr[2])
                    t[2] += float(shear_w @ Rf[:, 0])
                    t[3] += float(shear_w @ Rf[:, 1])
        out[f"{side}_tip"] = t
    return out


TIER_REGIONS = ("left_pad", "right_pad", "left_tip", "right_tip")


def tactile_tier2(grid: dict) -> np.ndarray:
    """档2：4 个区域各自的**总法向力**（切向丢弃）。grid=read_tactile_grid 输出。返回 (4,) float32。"""
    return np.array(
        [float(np.asarray(grid[k])[..., 0].sum()) for k in TIER_REGIONS], dtype=np.float32
    )


def tactile_tier1(grid: dict, thresh: float = 0.0) -> np.ndarray:
    """档1：4 个区域**是否接触**（总法向力 > thresh → 1，否则 0）。返回 (4,) float32。顺序 = TIER_REGIONS。"""
    return (tactile_tier2(grid) > thresh).astype(np.float32)


def tactile_ladder(grid: dict, thresh: float = 0.0) -> dict:
    """一次从档3 grid 派生全部三档（同源、离线降级）。"""
    return {"tier1": tactile_tier1(grid, thresh), "tier2": tactile_tier2(grid), "tier3": grid}


def read_wrist_wrench(env) -> np.ndarray:
    """腕部六轴力/力矩 = ForceVLA 那 6 维。返回 (6,) float32 = [Fx, Fy, Fz, Tx, Ty, Tz]。"""
    rs_env = _unwrap_robosuite_env(env)
    robot = rs_env.robots[0]
    return np.concatenate(
        [np.asarray(robot.ee_force, dtype=np.float32),
         np.asarray(robot.ee_torque, dtype=np.float32)]
    )
