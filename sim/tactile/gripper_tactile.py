"""Stage 0 / 0-6：指垫拆 N×N 子 box 的夹爪类 + 零污染注册（不改 site-packages）。"""
from __future__ import annotations

import pathlib

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.grippers.panda_gripper import PandaGripper

N = 4
_XML = str(pathlib.Path(__file__).resolve().parent / "assets" / "panda_gripper_tactileN.xml")


class PandaGripperTactileN(PandaGripper):
    """指垫=N×N 子 box 的 Panda 夹爪（仍单动作，动作逻辑同 PandaGripper）。"""

    def __init__(self, idn=0):


        GripperModel.__init__(self, _XML, idn=idn)

    @property
    def _important_geoms(self):
        subs1 = [f"finger1_pad_collision_x{i}_z{j}" for i in range(N) for j in range(N)]
        subs2 = [f"finger2_pad_collision_x{i}_z{j}" for i in range(N) for j in range(N)]
        return {
            "left_finger": ["finger1_collision", *subs1],
            "right_finger": ["finger2_collision", *subs2],
            "left_fingerpad": subs1,
            "right_fingerpad": subs2,
        }


def subgeom_names(side: str):
    """返回某指全部 N² 子 geom 的**裸名**（顺序 = x 外层 i、z 内层 j）。side='left'→finger1。"""
    f = "finger1" if side == "left" else "finger2"
    return [f"{f}_pad_collision_x{i}_z{j}" for i in range(N) for j in range(N)]


def register_tactile_gripper(override_default: bool = False):
    """进程内注册（不改 robosuite 文件）。**推荐默认 override_default=False**："""
    import robosuite.models.grippers as G

    G.GRIPPER_MAPPING["PandaGripperTactileN"] = PandaGripperTactileN
    if override_default:
        G.GRIPPER_MAPPING["PandaGripper"] = PandaGripperTactileN
    return PandaGripperTactileN
