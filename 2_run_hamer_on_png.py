"""Script to run HaMeR on image sequence and save outputs to a pickle file."""

import pickle
import shutil
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import tyro
from egoallo.hand_detection_structs import (
    SavedHamerOutputs,
    SingleHandHamerOutputWrtCamera,
)
from hamer_helper import HamerHelper
from tqdm.auto import tqdm
import random
import time

# Configuration parameters
DEFAULT_CAMERA_CONFIG = {
    "focal_length": 450,
    "sensor_width": 1408,
    "sensor_height": 1408,
    "undistort": False
}

def main(traj_root: Path, overwrite: bool = False, take_group: bool = False) -> None:
    """Run HaMeR for on image sequence. Save outputs to
    `traj_root/hamer_outputs.pkl` and `traj_root/hamer_outputs_render".

    Arguments:
        traj_root: A directory containing image sequences
        overwrite: Whether to overwrite existing output
        take_group: Whether there are many groups of take data
    """
    if take_group:
        # Shuffle takes with current time as seed
        takes = list(traj_root.iterdir())
        random.seed(int(time.time()))
        random.shuffle(takes)
        for take in takes:
            # Check if it's a directory and doesn't have hamer_outputs_render
            if take.is_dir() and not (take / "hamer_outputs_render").exists():
                print(f"Processing take: {take.name}, the number of images: {len(list(take.glob('**/*.png')))}")
                image_dir = take
                assert image_dir.exists(), f"The image directory does not exist: {image_dir}"
                
                pickle_out = take / "hamer_outputs.pkl"
                hamer_render_out = take / "hamer_outputs_render"
                
                run_hamer_and_save(image_dir, pickle_out, hamer_render_out, overwrite)
    else:
        image_dir = traj_root
        assert image_dir.exists(), f"The image directory does not exist: {image_dir}"
        
        pickle_out = traj_root / "hamer_outputs.pkl"
        hamer_render_out = traj_root / "hamer_outputs_render"
        
        run_hamer_and_save(image_dir, pickle_out, hamer_render_out, overwrite)

def run_hamer_and_save(
    image_dir: Path,
    pickle_out: Path,
    hamer_render_out: Path,
    overwrite: bool
) -> None:
    # Clear out old files
    if not overwrite:
        assert not pickle_out.exists()
        assert not hamer_render_out.exists()
    else:
        pickle_out.unlink(missing_ok=True)
        shutil.rmtree(hamer_render_out, ignore_errors=True)
    
    hamer_render_out.mkdir(exist_ok=True)
    hamer_helper = HamerHelper()

    # Get the sorted list of image files.
    take_name = image_dir.name
    image_paths = sorted(image_dir.glob("**/*.png"))
    num_images = len(image_paths)
    print(f"Found {num_images} images")

    # Initialize virtual camera parameters (⚠️ modify according to actual situation)
    T_device_cam = np.zeros(7)  # Virtual extrinsic parameters
    T_cpf_cam = np.zeros(7)     # Virtual coordinate system transformation

    # Store the detection results.
    detections_left_wrt_cam = {}
    detections_right_wrt_cam = {}

    pbar = tqdm(image_paths)
    for idx, img_path in enumerate(pbar):
        img_name = img_path.name.split('.')[0]
        # Load images
        image = iio.imread(img_path)
        if image.shape[-1] == 4:  # remove the alpha channel
            image = image[..., :3]
    
        undistorted_image = image.copy()

        # Run HaMeR detectation
        hamer_out_left, hamer_out_right = hamer_helper.look_for_hands(
            undistorted_image,
            focal_length=DEFAULT_CAMERA_CONFIG["focal_length"],
        )
        
        # Store the detection results
        if hamer_out_left is None:
            detections_left_wrt_cam[img_name] = None
        else:
            detections_left_wrt_cam[img_name] = {
                "verts": hamer_out_left["verts"],
                "keypoints_3d": hamer_out_left["keypoints_3d"],
                "mano_hand_pose": hamer_out_left["mano_hand_pose"],
                "mano_hand_betas": hamer_out_left["mano_hand_betas"],
                "mano_hand_global_orient": hamer_out_left["mano_hand_global_orient"],
            }

        if hamer_out_right is None:
            detections_right_wrt_cam[img_name] = None
        else:
            detections_right_wrt_cam[img_name] = {
                "verts": hamer_out_right["verts"],
                "keypoints_3d": hamer_out_right["keypoints_3d"],
                "mano_hand_pose": hamer_out_right["mano_hand_pose"],
                "mano_hand_betas": hamer_out_right["mano_hand_betas"],
                "mano_hand_global_orient": hamer_out_right["mano_hand_global_orient"],
            }

        # Rendering
        composited = undistorted_image.copy()
        composited = hamer_helper.composite_detections(
            composited,
            hamer_out_left,
            border_color=(255, 100, 100),
            focal_length=DEFAULT_CAMERA_CONFIG["focal_length"],
        )
        composited = hamer_helper.composite_detections(
            composited,
            hamer_out_right,
            border_color=(100, 100, 255),
            focal_length=DEFAULT_CAMERA_CONFIG["focal_length"],
        )
        
        # Add annotated text
        composited = put_text(
            composited,
            f"Frame: {img_name}",
            0,
            color=(255, 255, 255),
            font_scale=10.0 / 2880.0 * composited.shape[0],
        )
        
        # Save the rendering result
        output_path = hamer_render_out / f"{img_name}.jpeg"
        iio.imwrite(output_path, composited, quality=90)
        pbar.set_description(f"Save the rendering result: {output_path.name}")

    # Save the final result
    outputs = SavedHamerOutputs(
        mano_faces_right=hamer_helper.get_mano_faces("right"),
        mano_faces_left=hamer_helper.get_mano_faces("left"),
        detections_right_wrt_cam=detections_right_wrt_cam,
        detections_left_wrt_cam=detections_left_wrt_cam,
        T_device_cam=T_device_cam,
        T_cpf_cam=T_cpf_cam,
    )
    
    with open(pickle_out, "wb") as f:
        pickle.dump(outputs, f)

def put_text(
    image: np.ndarray,
    text: str,
    line_number: int,
    color: tuple[int, int, int],
    font_scale: float,
) -> np.ndarray:
    """Put some text on the top-left corner of an image."""
    image = image.copy()
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(
        image,
        text=text,
        org=(2, 1 + int(15 * font_scale * (line_number + 1))),
        fontFace=font,
        fontScale=font_scale,
        color=(0, 0, 0),
        thickness=max(int(font_scale), 1),
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text=text,
        org=(2, 1 + int(15 * font_scale * (line_number + 1))),
        fontFace=font,
        fontScale=font_scale,
        color=color,
        thickness=max(int(font_scale), 1),
        lineType=cv2.LINE_AA,
    )
    return image

if __name__ == "__main__":
    tyro.cli(main)