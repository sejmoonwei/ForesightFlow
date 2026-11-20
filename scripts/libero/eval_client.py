"""Evaluate a policy in LIBERO using explicit per-decision identities."""

import atexit
import collections
import dataclasses
import datetime as _datetime
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
from trajectory_writer import LeRobotTrajectoryWriter
import tyro


try:
    from tactile.sensor import read_tactile, read_tactile_grid, read_wrist_wrench, resolve_pads, tactile_tier1, tactile_tier2
    from tactile.visualize import save_tactile_force_video

    _TACTILE_OK = True
except Exception as _tac_e:  # noqa: BLE001
    logging.warning(f"[tactile] import failed, tactile logging disabled: {_tac_e}")
    _TACTILE_OK = False

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:


    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5


    task_suite_name: str = (
        "libero_spatial"
    )
    num_steps_wait: int = 10
    num_trials_per_task: int = 50


    success_settle_steps: int = 60


    task_id_start: int = 0
    task_id_end: int = -1


    video_out_path: str = "outputs/rollouts"

    seed: int = 7


    save_trajectory: bool = False
    trajectory_out_path: str = "outputs/rollouts"
    trajectory_run_id: str = ""


    tactile_video: bool = False


    tactile_tier: int = 3


    tactile_subgeom: bool = True


    tactile_wrench: bool = False


    send_wrench: bool = False


def eval_libero(args: Args) -> None:

    np.random.seed(args.seed)


    _tactile_gripper_types = "default"


    _tac_video_npad = 4
    if getattr(args, "tactile_subgeom", True) and _TACTILE_OK:
        try:
            from tactile.gripper_tactile import register_tactile_gripper

            register_tactile_gripper()
            _tactile_gripper_types = "PandaGripperTactileN"
            logging.info("[tactile] subgeom 夹爪启用 (gripper_types=PandaGripperTactileN, 指垫拆 4×4 子box)")
        except Exception as _tac_e:  # noqa: BLE001
            logging.warning(f"[tactile] subgeom gripper 注册失败, 用原夹爪: {_tac_e}")


    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")


    if args.trajectory_run_id:
        run_folder_name = args.trajectory_run_id
    else:
        run_folder_name = f"{_datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.task_suite_name}"
    run_root = pathlib.Path(args.video_out_path) / run_folder_name
    video_dir = run_root / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Rollout outputs go under: {run_root}")


    writer = None
    if args.save_trajectory:
        writer = LeRobotTrajectoryWriter(
            out_dir=args.trajectory_out_path, run_id=run_folder_name
        )
        logging.info(f"Saving trajectories to {writer.root}")

        atexit.register(writer.finalize)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)


    task_id_end = args.task_id_end if args.task_id_end != -1 else num_tasks_in_suite
    task_id_start = args.task_id_start
    if not (0 <= task_id_start < task_id_end <= num_tasks_in_suite):
        raise ValueError(
            f"Invalid task-id slice [{task_id_start}, {task_id_end}) for suite of size {num_tasks_in_suite}"
        )
    if task_id_start != 0 or task_id_end != num_tasks_in_suite:
        logging.info(f"Sharded eval: task_ids [{task_id_start}, {task_id_end}) of {num_tasks_in_suite}")


    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(task_id_start, task_id_end)):

        task = task_suite.get_task(task_id)


        initial_states = task_suite.get_task_init_states(task_id)


        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed,
                                                gripper_types=_tactile_gripper_types)


        tactile_pads = None
        tactile_grid_ok = True
        wrench_ok = True
        if _TACTILE_OK:
            try:
                tactile_pads = resolve_pads(env)
            except Exception as _tac_e:  # noqa: BLE001
                logging.warning(f"[tactile] resolve_pads failed, disabled: {_tac_e}")


        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")


            env.reset()
            action_plan = collections.deque()
            v31_replan_idx = 0


            obs = env.set_init_state(initial_states[episode_idx])


            t = 0
            replay_images = []

            replay_wrist_images = [] if args.save_trajectory else None

            tactile_grids = [] if (args.tactile_video and tactile_pads is not None) else None
            done = False
            episode_success = False
            settle_remaining = -1


            current_episode_index = total_episodes
            if writer is not None:

                _ = writer.start_episode(str(task_description))

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:


                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue


                    img_raw = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_raw = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img_raw, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_raw, args.resize_size, args.resize_size)
                    )


                    replay_images.append(img_raw)
                    if replay_wrist_images is not None:
                        replay_wrist_images.append(wrist_raw)


                    if tactile_grids is not None:
                        try:
                            tactile_grids.append(read_tactile_grid(env, tactile_pads, n_pad=_tac_video_npad, with_shear_dir=True))
                        except Exception:  # noqa: BLE001
                            tactile_grids.append(None)

                    state_vec = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    )

                    if not action_plan:


                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": state_vec,
                            "prompt": str(task_description),
                        }


                        if args.send_wrench:
                            if not _TACTILE_OK:
                                raise RuntimeError("--args.send_wrench 需要 tactile.sensor 可用（read_wrist_wrench 未导入）")
                            element["observation/ft_wrench"] = np.asarray(read_wrist_wrench(env), dtype=np.float32)


                        element["__v31_seed_identity"] = {
                            "eval_seed": int(args.seed),
                            "suite": str(args.task_suite_name),
                            "task_id": int(task_id),
                            "episode_idx": int(episode_idx),
                            "replan_idx": int(v31_replan_idx),
                        }
                        reply = client.infer(element)
                        action_chunk = reply["actions"]
                        if v31_replan_idx == 0:
                            logging.info(
                                "[V31 seed] task=%s episode=%d derived=%s",
                                task_id,
                                episode_idx,
                                reply.get("v31_seed", {}).get("derived_seed", "missing"),
                            )
                        v31_replan_idx += 1
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()


                    if writer is not None:
                        writer.add_step(
                            image=img_raw,
                            wrist_image=wrist_raw,
                            state=state_vec,
                            action=np.asarray(action),
                        )


                    obs, reward, done, info = env.step(action.tolist())


                    if tactile_pads is not None and tactile_grid_ok:
                        try:
                            _g = read_tactile_grid(env, tactile_pads)
                            if args.tactile_tier == 1:
                                _v = tactile_tier1(_g)
                                if float(_v.sum()) > 0:
                                    logging.info(f"[tactile t1] t={t} contact[Lpad,Rpad,Ltip,Rtip]={_v.tolist()}")
                            elif args.tactile_tier == 2:
                                _v = tactile_tier2(_g)
                                if float(_v.sum()) > 0:
                                    logging.info(f"[tactile t2] t={t} normalN[Lpad,Rpad,Ltip,Rtip]={np.round(_v, 2).tolist()}")
                            else:
                                _lp, _rp = _g["left_pad"][..., 0], _g["right_pad"][..., 0]
                                if float(_lp.sum() + _rp.sum() + _g["left_tip"][0] + _g["right_tip"][0]) > 0:
                                    logging.info(
                                        f"[tactile t3] t={t} "
                                        f"L_pad(nz={int((_lp > 0).sum())}/{_lp.size},pk={_lp.max():.1f}) "
                                        f"R_pad(nz={int((_rp > 0).sum())}/{_rp.size},pk={_rp.max():.1f}) "
                                        f"L_tip={np.round(_g['left_tip'], 1).tolist()} "
                                        f"R_tip={np.round(_g['right_tip'], 1).tolist()}"
                                    )
                        except Exception as _tac_e:  # noqa: BLE001
                            logging.warning(f"[tactile] read failed at t={t}, disabled: {_tac_e}")
                            tactile_grid_ok = False


                    if args.tactile_wrench and _TACTILE_OK and wrench_ok:
                        try:
                            _w = read_wrist_wrench(env)
                            logging.info(
                                f"[tactile wrench] t={t} F={np.round(_w[:3], 2).tolist()} "
                                f"T={np.round(_w[3:], 3).tolist()} |F|={float(np.linalg.norm(_w[:3])):.2f}"
                            )
                        except Exception as _tac_e:  # noqa: BLE001
                            logging.warning(f"[tactile wrench] read failed at t={t}, disabled: {_tac_e}")
                            wrench_ok = False

                    if done and not episode_success:

                        episode_success = True
                        task_successes += 1
                        total_successes += 1


                        settle_remaining = args.success_settle_steps
                    if episode_success:


                        if settle_remaining <= 0:
                            break
                        settle_remaining -= 1
                    else:
                        t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1


            suffix = "success" if episode_success else "failure"
            task_segment = task_description.replace(" ", "_")
            video_name = f"episode_{current_episode_index:06d}_{suffix}_{task_segment}.mp4"
            imageio.mimwrite(
                video_dir / video_name,
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            if replay_wrist_images is not None:
                wrist_video_name = (
                    f"episode_{current_episode_index:06d}_{suffix}_{task_segment}_wrist.mp4"
                )
                imageio.mimwrite(
                    video_dir / wrist_video_name,
                    [np.asarray(x) for x in replay_wrist_images],
                    fps=10,
                )


            if tactile_grids is not None and replay_images and len(tactile_grids) == len(replay_images):
                try:
                    tac_video_name = f"episode_{current_episode_index:06d}_{suffix}_{task_segment}_tactile.mp4"
                    save_tactile_force_video(
                        replay_images, tactile_grids, video_dir / tac_video_name, fps=10
                    )
                    logging.info(f"[tactile-video] saved {tac_video_name}")
                except Exception as _tv_e:  # noqa: BLE001
                    logging.warning(f"[tactile-video] compose failed: {_tv_e}")


            if writer is not None:
                writer.end_episode(success=bool(episode_success))


            logging.info(f"Success: {episode_success}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")


        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


    if writer is not None:
        writer.finalize(
            extra_summary={
                "task_suite_name": args.task_suite_name,
                "num_trials_per_task": args.num_trials_per_task,
                "total_success_rate": (
                    float(total_successes) / float(total_episodes)
                    if total_episodes
                    else 0.0
                ),
            }
        )


def _get_libero_env(task, resolution, seed, gripper_types="default"):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution,
                "gripper_types": gripper_types}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    """Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55"""

    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):

        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
