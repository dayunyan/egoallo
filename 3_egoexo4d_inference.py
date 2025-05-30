from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import jax.tree
import numpy as np
import torch
import viser
import yaml
from addict import Dict
from tqdm.auto import tqdm

from egoallo import fncsmpl, fncsmpl_extensions
from egoallo.data.aria_mps import load_point_cloud_and_find_ground
from egoallo.guidance_optimizer_jax import GuidanceMode
from egoallo.hand_detection_structs import (
    CorrespondedAriaHandWristPoseDetections,
    CorrespondedHamerDetections,
)
from egoallo.inference_utils import (
    InferenceInputTransforms,
    InferenceTrajectoryPaths,
    load_denoiser,
)
from egoallo.sampling import run_sampling_with_stitching
from egoallo.transforms import SE3, SO3
from egoallo.vis_helpers import visualize_traj_and_hand_detections
from egoallo.data.egoexo4d import get_val_loader, BodyPoseDataset
from egoallo.utils import (
    get_config,
    get_logger_and_tb_writer
)
from egoallo.fncsmpl_extensions import get_T_world_root_from_cpf_pose
from egoallo.metrics_helpers import (
    compute_foot_contact,
    compute_foot_skate,
    compute_head_trans,
    compute_mpjpe,
)


@dataclasses.dataclass
class Args:
    config_path: Path = Path("./configs/egoallo.yaml")
    data_dir: Path = Path("/dfs/dataset/91-1743583282702/data/egobodypose/EgoExoBodyPose")
    save_dir: Path = Path("./egoexo4d_output")

    checkpoint_dir: Path = Path("./egoallo_checkpoint_april13/checkpoints_3000000/")
    smplh_npz_path: Path = Path("./data/smplh/neutral/model.npz")

    glasses_x_angle_offset: float = 0.0
    """Rotate the CPF poses by some X angle."""
    start_index: int = 0
    """Index within the downsampled trajectory to start inference at."""
    traj_length: int = 128
    """How many timesteps to estimate body motion for."""
    num_samples: int = 1
    """Number of samples to take."""
    guidance_mode: GuidanceMode = 'no_hands' #"aria_hamer"
    """Which guidance mode to use."""
    guidance_inner: bool = True
    """Whether to apply guidance optimizer between denoising steps. This is
    important if we're doing anything with hands. It can be turned off to speed
    up debugging/experiments, or if we only care about foot skating losses."""
    guidance_post: bool = True
    """Whether to apply guidance optimizer after diffusion sampling."""
    save_traj: bool = True
    """Whether to save the output trajectory, which will be placed under `traj_dir/egoallo_outputs/some_name.npz`."""
    visualize_traj: bool = False
    """Whether to visualize the trajectory after sampling."""


def main(args: Args) -> None:
    config = get_config(args)
    device = torch.device("cuda")
    logger, _ = get_logger_and_tb_writer(config, split="val")
    logger.info(f"config: {config}")
    logger.info(f"args: {args}")

    val_dataset = BodyPoseDataset(config, split="val", logger=logger)

    # traj_paths = Dict({
    #     "takes_root": args.takes_root,
    #     "cam_pose_root": args.cam_pose_root
    # })
    # with open(traj_paths.cam_pose_root / args.uid) as f:
    #     camera_pose_json = json.load(f)
    # take_name = camera_pose_json["metadata"]["take_name"]
    # traj_paths.takes_root = traj_paths.takes_root / take_name
    # if traj_paths.splat_path is not None:
    #     print("Found splat at", traj_paths.splat_path)
    # else:
    #     print("No scene splat found.")
    # Get point cloud + floor.
    points_data = np.zeros((10, 3))
    floor_z = -1.45

    # # Read transforms from VRS / MPS, downsampled.
    # Ts_world_cpf  #TODO: 这里需要修改egoexo4d的dataset，将aria的外参视为Ts_world_cpf，并用SE3转换
    # transforms = InferenceInputTransforms.load(
    #     traj_paths.vrs_file, traj_paths.slam_root_dir, fps=30
    # ).to(device=device)

    # Note the off-by-one for Ts_world_cpf, which we need for relative transform computation.
    # Ts_world_cpf = (
    #     SE3(
    #         transforms.Ts_world_cpf[
    #             args.start_index : args.start_index + args.traj_length + 1
    #         ]
    #     )
    #     @ SE3.from_rotation(
    #         SO3.from_x_radians(
    #             transforms.Ts_world_cpf.new_tensor(args.glasses_x_angle_offset)
    #         )
    #     )
    # ).parameters()
    # pose_timestamps_sec = transforms.pose_timesteps[
    #     args.start_index + 1 : args.start_index + args.traj_length + 1
    # ]
    # Ts_world_device = transforms.Ts_world_device[
    #     args.start_index + 1 : args.start_index + args.traj_length + 1
    # ]
    # del transforms

    # Get temporally corresponded HaMeR detections.
    # if traj_paths.hamer_outputs is not None:
    #     hamer_detections = CorrespondedHamerDetections.load(
    #         traj_paths.hamer_outputs,
    #         pose_timestamps_sec,
    #     ).to(device)
    # else:
    #     print("No hand detections found.")
    #     hamer_detections = None

    # # Get temporally corresponded Aria wrist and palm estimates.
    # if traj_paths.wrist_and_palm_poses_csv is not None:
    #     aria_detections = CorrespondedAriaHandWristPoseDetections.load(
    #         traj_paths.wrist_and_palm_poses_csv,
    #         pose_timestamps_sec,
    #         Ts_world_device=Ts_world_device.numpy(force=True),
    #     ).to(device)
    # else:
    #     print("No Aria hand detections found.")
    #     aria_detections = None


    server = None
    if args.visualize_traj:
        server = viser.ViserServer()
        server.gui.configure_theme(dark_mode=True)

    denoiser_network = load_denoiser(args.checkpoint_dir).to(device)
    body_model = fncsmpl.SmplhModel.load(args.smplh_npz_path).to(device)

    coco_joint_vertex_indices = [
        332,   # COCO 0: nose
        2800,  # COCO 1: left-eye
        6260,  # COCO 2: right-eye
        329,   # COCO 3: left-ear
        661,   # COCO 4: right-ear
        # 1276,  # COCO 5: left-shoulder
        # 1630,  # COCO 6: right-shoulder
        # 412,  # COCO 7: left-elbow
        # 756,  # COCO 8: right-elbow
        # 4203,  # COCO 9: left-wrist
        # 4567,  # COCO 10: right-wrist
        # 292,   # COCO 11: left-hip
        # 728,   # COCO 12: right-hip
        # 1855,   # COCO 13: left-knee
        # 2327,   # COCO 14: right-knee
        # 2078,   # COCO 15: left-ankle
        # 2549    # COCO 16: right-ankle
    ]

    smpl_to_coco_joint_indices = [
        # 53,  # COCO 0: nose
        # 55,  # COCO 1: left-eye
        # 54,  # COCO 2: right-eye
        # 57,  # COCO 3: left-ear
        # 56,  # COCO 4: right-ear
        16,  # COCO 5: left-shoulder
        17,  # COCO 6: right-shoulder
        18,  # COCO 7: left-elbow
        19,  # COCO 8: right-elbow
        20,  # COCO 9: left-wrist
        21,  # COCO 10: right-wrist
        1,   # COCO 11: left-hip
        2,   # COCO 12: right-hip
        4,   # COCO 13: left-knee
        5,   # COCO 14: right-knee
        7,   # COCO 15: left-ankle
        8    # COCO 16: right-ankle
    ]
    smpl_to_coco_joint_indices = [ _ - 1 for _ in smpl_to_coco_joint_indices]

    metrics = list[dict[str, np.ndarray]]()

    pbar = tqdm(
        range(len(val_dataset)),
        position=0,
        desc="Inference",
    )
    for i in pbar:
        data = val_dataset[i]
        # Get the data.
        take_uid = data["take_uid"]
        take_name = data["take_name"]
        frame_id_str = data["frame_id_str"]
        Ts_world_cpf = torch.from_numpy(data["inputs_IMU"]).to(device)
        print(f"{Ts_world_cpf[-1, :]=}, {SE3(Ts_world_cpf[-1, :]).as_matrix()=}")
        target = torch.from_numpy(data["target"]).to(device)
        traj_length = Ts_world_cpf.shape[0]
        Ts_world_cpf = (
            SE3(
                Ts_world_cpf
            )
            @ SE3.from_rotation(
                SO3.from_x_radians(
                    Ts_world_cpf.new_tensor(args.glasses_x_angle_offset)
                )
            )
        ).parameters()
        # Ts_world_cpf[:, 4] -= torch.mean(Ts_world_cpf[:, 4])
        # Ts_world_cpf[:, 5] -= torch.mean(Ts_world_cpf[:, 5])
        # Ts_world_cpf[:, 6] -= torch.mean(Ts_world_cpf[:, 6])
        print(f"{Ts_world_cpf.shape=}")
        print(f"{Ts_world_cpf[-1, :]=}, {SE3(Ts_world_cpf[-1, :]).as_matrix()=}")
        print(f"{Ts_world_cpf[:, 4:]=}")

        traj = run_sampling_with_stitching(
            denoiser_network,
            body_model=body_model,
            guidance_mode=args.guidance_mode,
            guidance_inner=args.guidance_inner,
            guidance_post=args.guidance_post,
            Ts_world_cpf=Ts_world_cpf,
            hamer_detections=None,
            aria_detections=None,
            num_samples=args.num_samples,
            device=device,
            floor_z=floor_z,
            # guidance_verbose=False,
        )
        print(f"{traj_length=}")
        print(f"{traj.betas.shape=}, {traj.body_rotmats.shape=}, {traj.hand_rotmats.shape=}")
        assert traj.hand_rotmats is not None
        assert traj.betas.shape == (args.num_samples, traj_length-1, 16)
        assert traj.body_rotmats.shape == (args.num_samples, traj_length-1, 21, 3, 3)
        assert traj.hand_rotmats.shape == (args.num_samples, traj_length-1, 30, 3, 3)

        # We'll only use the body joint rotations.
        pred_posed = body_model.with_shape(traj.betas).with_pose(
            T_world_root=SE3.identity(device, torch.float32).wxyz_xyz,
            local_quats=SO3.from_matrix(
                torch.cat([traj.body_rotmats, traj.hand_rotmats], dim=2)
            ).wxyz,
        )
        pred_posed = pred_posed.with_new_T_world_root(
            get_T_world_root_from_cpf_pose(pred_posed, Ts_world_cpf[1:, ...])
        )
        print(f"{pred_posed.T_world_root.shape=}, {pred_posed.Ts_world_joint.shape=}, {pred_posed.shaped_model.verts_zero.shape=}")
        print(f"{pred_posed.Ts_world_joint[0, -1, :21, :]=}")

        pred_face_cpf = pred_posed.shaped_model.verts_zero[..., coco_joint_vertex_indices, :]
        pred_cpf_face = SE3(torch.cat([SO3.identity(device, torch.float32).wxyz[None, None, ...].repeat(*pred_face_cpf.shape[:-1], 1), pred_face_cpf], dim=-1)).inverse().parameters()
        pred_world_face = torch.cat([(SE3(Ts_world_cpf[1:, ...]) @ SE3(pred_cpf_face[..., i, :])).parameters().unsqueeze(-2) for i in range(pred_cpf_face.shape[-2])], dim=-2)

        pred_body = pred_posed.Ts_world_joint[..., smpl_to_coco_joint_indices, :]

        pred = torch.cat([pred_world_face, pred_body], dim=-2)
        # pred = pred_world_face
        # pred[..., -1] = pred[..., -1] - 0.6
        print(f"{pred.shape=}, {pred_face_cpf.shape=}, {pred_cpf_face.shape=}, {pred_world_face.shape=}")

        label_posed = torch.cat([SO3.identity(device, torch.float32).wxyz.repeat(target.shape[0], 1), target], dim=-1)  # [N, 3]
        # label_posed[:, 4] -= torch.mean(label_posed[:, 4])
        # label_posed[:, 5] -= torch.mean(label_posed[:, 5])
        # label_posed[:, 6] -= torch.mean(label_posed[:, 6])
        print(f"{label_posed.shape=}")
        print(f"{pred[..., -1, :, 4:]=}")
        print(f"{label_posed[:, 4:]=}")
        print(f"pred - label:\n{pred[0, -1, :, 4:] - label_posed[:, 4:]}")
        print(f"{pred_posed.T_world_root[..., -1:, :]=}")

        # pred -= torch.mean(pred[0, -1, :, :] - label_posed[:, :], dim=-2)

        metrics.append(
            {
                "mpjpe": compute_mpjpe(
                    label_T_world_root=label_posed[None, 0, :],
                    label_Ts_world_joint=label_posed[None, 1:, :],
                    pred_T_world_root=pred[..., -1:, 0, :],  # pred_posed.T_world_root[..., -1:, :],
                    pred_Ts_world_joint=pred[..., -1:, 1:, :],  #pred_posed.Ts_world_joint[:, :, smpl_to_coco_joint_indices, :],
                    per_frame_procrustes_align=False,
                ),
            }
        )
        print(metrics)

        print("=" * 80)
        print("=" * 80)
        print("=" * 80)
        print(f"Metrics ({i}/{len(val_dataset)} processed)")
        for k, v in jax.tree.map(
            lambda *x: f"{np.mean(x):.3f} +/- {np.std(x) / np.sqrt(len(metrics) * args.num_samples):.3f}",
            *metrics,
        ).items():
            print("\t", k, v)
        print("=" * 80)
        print("=" * 80)
        print("=" * 80)

        # Save outputs in case we want to visualize later.
        if args.save_traj:
            save_name = (
                time.strftime("%Y%m%d-%H%M%S")
                + f"_{take_uid}-{take_name}-{frame_id_str}"
            )
            out_path = args.save_dir / (save_name + ".npz")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            assert not out_path.exists()
            (args.save_dir / (save_name + "_args.yaml")).write_text(
                yaml.dump(config)
            )

            posed = traj.apply_to_body(body_model)
            Ts_world_root = fncsmpl_extensions.get_T_world_root_from_cpf_pose(
                posed, Ts_world_cpf[..., 1:, :]
            )
            print(f"Saving to {out_path}...", end="")
            np.savez(
                out_path,
                Ts_world_cpf=Ts_world_cpf[1:, :].numpy(force=True),
                Ts_world_root=Ts_world_root.numpy(force=True),
                body_quats=posed.local_quats[..., :21, :].numpy(force=True),
                left_hand_quats=posed.local_quats[..., 21:36, :].numpy(force=True),
                right_hand_quats=posed.local_quats[..., 36:51, :].numpy(force=True),
                contacts=traj.contacts.numpy(force=True),  # Sometimes we forgot this...
                betas=traj.betas.numpy(force=True),
                frame_nums=np.arange(args.start_index, args.start_index + args.traj_length),
                # timestamps_ns=(np.array(pose_timestamps_sec) * 1e9).astype(np.int64),
            )
            print("saved!")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
