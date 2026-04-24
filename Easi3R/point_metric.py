import os
import numpy as np
import cv2
from tqdm import tqdm
import glob
import json
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from dust3r.depth_eval import depth_evaluation, group_by_directory

import roma
import torch
from dust3r.utils.vo_eval import load_traj
import numpy as np
from scipy.spatial import cKDTree as KDTree
import torch


def accuracy(gt_points, rec_points):
    gt_points_kd_tree = KDTree(gt_points)
    distances, idx = gt_points_kd_tree.query(rec_points, workers=-1)
    acc = np.mean(distances)
    acc_median = np.median(distances)
    return acc, acc_median


def completion(gt_points, rec_points):
    gt_points_kd_tree = KDTree(rec_points)
    distances, idx = gt_points_kd_tree.query(gt_points, workers=-1)
    comp = np.mean(distances)
    comp_median = np.median(distances)
    return comp, comp_median

def compute_3d_metrics(pred_xyz, gt_xyz):
    pred_xyz_aligned = pred_xyz
    gt_xyz_np = gt_xyz.cpu().numpy()
    pred_xyz_aligned_np = pred_xyz_aligned.cpu().numpy()
    acc, acc_med = accuracy(gt_xyz_np, pred_xyz_aligned_np)
    comp, comp_med = completion(gt_xyz_np, pred_xyz_aligned_np)
    dist = torch.mean(torch.norm(gt_xyz - pred_xyz_aligned, dim=-1)).cpu().numpy()
    dist_med = torch.median(torch.norm(gt_xyz - pred_xyz_aligned, dim=-1)).cpu().numpy()
        
    metrics = {
        'Accuracy': float(acc),
        'Acc_med': float(acc_med),
        'Completion': float(comp), 
        'Comp_med': float(comp_med),
        'Distance': float(dist),
        'Dist_med': float(dist_med)
    }
    return metrics

def depth_to_3d(depth, intrinsics, pose):
    """
    transform batch of depth maps to 3D point clouds

    Args:
        depth: depth maps (B, H, W)
        intrinsics: camera intrinsics (B, 3, 3)
        pose: camera pose (c2w) (B, 4, 4)

    Returns:
        3D point clouds (B, N, 3), where N is the number of points in each depth map
    """
    B, H, W = depth.shape
    # create grid coordinates
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    
    # flatten grid coordinates to match batch size
    x = x.flatten()  # (H*W,)
    y = y.flatten()  # (H*W,)
    
    # convert pixel coordinates to camera coordinates
    z = depth.reshape(B, -1)  # (B, H*W)
    
    # compute camera coordinates
    x_camera = (x - intrinsics[:, 0, 2].reshape(B, 1)) * z / intrinsics[:, 0, 0].reshape(B, 1)  # (B, H*W)
    y_camera = (y - intrinsics[:, 1, 2].reshape(B, 1)) * z / intrinsics[:, 1, 1].reshape(B, 1)  # (B, H*W)
    
    # combine to 3D points
    points_camera = np.stack((x_camera, y_camera, z), axis=-1)  # (B, H*W, 3)

    # apply camera pose (use c2w matrix directly, no transpose)
    points_camera_homogeneous = np.concatenate((points_camera, np.ones((B, points_camera.shape[1], 1))), axis=-1)  # (B, H*W, 4)
    
    # directly use c2w matrix to multiply camera coordinates
    points_world = np.matmul(pose, points_camera_homogeneous.transpose(0, 2, 1)).transpose(0, 2, 1)  # (B, H*W, 4)
    
    return points_world[:, :, :3]  # return (B, H*W, 3)


def visualize_point_cloud_aligned(gt_points, pred_aligned_points, pred_original_points=None, filename='aligned_point_clouds.png'):
    """
    visualize three point clouds with different colors and add transparency so that all point clouds can be seen when overlapping
    
    Args:
        gt_points: ground truth point cloud data (N, 3)
        pred_aligned_points: aligned predicted point cloud data (N, 3)
        pred_original_points: original predicted point cloud data (N, 3), optional
        filename: save file name
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # plot ground truth point cloud and aligned predicted point cloud, add transparency
    ax.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], 
               s=1, c='blue', label='GT', alpha=0.05)
    ax.scatter(pred_aligned_points[:, 0], pred_aligned_points[:, 1], pred_aligned_points[:, 2], 
               s=1, c='green', label='Aligned Pred', alpha=0.05)
    
    # if original predicted point cloud is provided, also plot it
    if pred_original_points is not None:
        ax.scatter(pred_original_points[:, 0], pred_original_points[:, 1], pred_original_points[:, 2], 
                   s=1, c='red', label='Pred', alpha=0.05)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # save figure
    plt.savefig(filename, dpi=300)
    plt.close()  # close figure to release memory

def depth_read_iphone(filename):
    """read iPhone depth data"""
    depth = np.load(filename)
    return depth

def load_iphone_traj(pose_file):
    with open(pose_file, 'r') as f:
        camera_data = json.load(f)

    gt_pose = np.array(camera_data['w2c'])
    gt_pose_inv = np.linalg.inv(gt_pose)
    return gt_pose_inv

def load_iphone_intrinsics(pose_file):
    with open(pose_file, 'r') as f:
        camera_data = json.load(f)

    intrinsics = np.array(camera_data['K'])
    return intrinsics

def tum_to_4x4(tum_pose, xyzw=True):
    # tum format to 4x4
    from scipy.spatial.transform import Rotation
    pred_pose = np.zeros((tum_pose.shape[0], 4, 4))
    pred_pose[:, :3, 3] = tum_pose[:, 1:4]
    
    if xyzw:
        pred_pose[:, :3, :3] = Rotation.from_quat(tum_pose[:, 4:]).as_matrix()
    else:
        # if wxyz format, need to rearrange to xyzw format
        pred_pose[:, :3, :3] = Rotation.from_quat(
            np.concatenate([tum_pose[:, 5:], tum_pose[:, 4:5]], -1)
        ).as_matrix()

    pred_pose[:, 3, 3] = 1.0  # set the last element of homogeneous coordinates to 1
    return pred_pose

def align_trajectory(pred_pose, gt_pose):
    """
    align predicted trajectory and ground truth trajectory using SE(3) Umeyama algorithm
    
    Args:
        pred_pose: predicted pose (B, 4, 4)
        gt_pose: ground truth pose (B, 4, 4)
        
    Returns:
        aligned_pose: aligned predicted pose (B, 4, 4)
        scale: scale factor
    """
    from evo.core.trajectory import PoseTrajectory3D
    from evo.core import sync
    from scipy.spatial.transform import Rotation
    import numpy as np
    
    # convert pose to PoseTrajectory3D object
    def pose_to_traj(poses):
        # extract positions and quaternions
        positions = poses[:, :3, 3]
        rotations = [Rotation.from_matrix(pose[:3, :3]).as_quat() for pose in poses] # xyzw
        # convert to wxyz format
        quats = np.array([[q[3], q[0], q[1], q[2]] for q in rotations]) # wxyz
        # create trajectory object
        timestamps = np.arange(len(poses)).astype(float)
        return PoseTrajectory3D(
            positions_xyz=positions,
            orientations_quat_wxyz=quats,
            timestamps=timestamps
        )
    
    gt_traj = pose_to_traj(gt_pose)
    pred_traj = pose_to_traj(pred_pose)
    
    # synchronize trajectories
    gt_aligned, pred_aligned = sync.associate_trajectories(gt_traj, pred_traj)
    
    # align trajectories and get transformation parameters
    R, t, s = pred_aligned.align(gt_aligned, correct_scale=True)
    
    # get aligned pose
    aligned_positions = pred_aligned.positions_xyz
    aligned_orientations = pred_aligned.orientations_quat_wxyz
    
    # convert back to 4x4 matrix
    aligned_pose = np.zeros_like(pred_pose)
    for i in range(len(aligned_positions)):
        # convert quaternions to rotation matrix (wxyz format)
        w, x, y, z = aligned_orientations[i]
        quat = np.array([x, y, z, w])  # convert to xyzw format for scipy
        rot_matrix = Rotation.from_quat(quat).as_matrix()
        
        # build transformation matrix
        aligned_pose[i, :3, :3] = rot_matrix
        aligned_pose[i, :3, 3] = aligned_positions[i]
        aligned_pose[i, 3, 3] = 1.0

    return aligned_pose, s

def visualize_trajectory_alignment(pred_pose, gt_pose, aligned_pred_pose, filename='trajectory_alignment.pdf'):
    """
    visualize original predicted trajectory, aligned predicted trajectory and ground truth trajectory
    
    Args:
        pred_pose: original predicted pose (B, 4, 4)
        gt_pose: ground truth pose (B, 4, 4)
        aligned_pred_pose: aligned predicted pose (B, 4, 4)
        filename: save file name
    """
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import plot
    from scipy.spatial.transform import Rotation
    import matplotlib.pyplot as plt
    from dust3r.utils.vo_eval import best_plotmode
    import os
    
    # create save directory
    save_dir = os.path.dirname(filename)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # convert pose to PoseTrajectory3D object
    def pose_to_traj(poses):
        # extract positions and quaternions
        positions = poses[:, :3, 3]
        rotations = [Rotation.from_matrix(pose[:3, :3]).as_quat() for pose in poses]
        # convert to wxyz format
        quats = np.array([[q[3], q[0], q[1], q[2]] for q in rotations])
        # create trajectory object
        timestamps = np.arange(len(poses)).astype(float)
        return PoseTrajectory3D(
            positions_xyz=positions,
            orientations_quat_wxyz=quats,
            timestamps=timestamps
        )
    
    gt_traj = pose_to_traj(gt_pose)
    pred_traj = pose_to_traj(pred_pose)
    aligned_traj = pose_to_traj(aligned_pred_pose)
    
    # create plot collection
    plot_collection = plot.PlotCollection("PlotCol")
    
    # create figure
    fig = plt.figure(figsize=(12, 10))
    plot_mode = best_plotmode(gt_traj)
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title("Trajectory Alignment Result", fontsize=18)
    
    # set line style
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['lines.markersize'] = 8
    
    # draw ground truth trajectory - bold
    plt.rcParams['lines.linewidth'] = 4.0
    plot.traj(ax, plot_mode, gt_traj, "--", "gray", "GT")
    
    # draw original predicted trajectory
    plt.rcParams['lines.linewidth'] = 2.0
    plot.traj(ax, plot_mode, pred_traj, ":", "#1f77b4", "Pred")
    
    # draw aligned predicted trajectory
    plt.rcParams['lines.linewidth'] = 2.0
    plot.traj(ax, plot_mode, aligned_traj, "-", "#d62728", "Aligned Pred")
    
    # set coordinate axis labels and font size
    ax.set_xlabel("X (m)", fontsize=16)
    ax.set_ylabel("Y (m)", fontsize=16)
    if plot_mode == plot.PlotMode.xyz:
        ax.set_zlabel("Z (m)", fontsize=16)
    
    # optimize legend position and style
    ax.legend(loc='best', fontsize=20, framealpha=0.8)
    
    # adjust coordinate axis font size
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    
    # save figure
    plot_collection.add_figure("traj_error", fig)
    plot_collection.export(filename, confirm_overwrite=False)
    plt.close(fig=fig)
    print(f"saved trajectory alignment visualization to {filename.replace('.pdf','')}_traj_error.pdf")

def eval_iphone(base_dir):
    """evaluate iPhone depth prediction results"""
    methods = os.listdir(base_dir)
    
    for method in tqdm(methods, desc="Processing iphone"):

        method_dir = os.path.join(base_dir, method)
        if not os.path.isdir(method_dir):
            continue
            
        pred_pathes = glob.glob(os.path.join(method_dir, "*/frame_*.npy"))
        pred_pathes = sorted(pred_pathes)
        
        if len(pred_pathes) == 0:
            print(f"warning: {method} no prediction results found")
            continue
            
        # get corresponding ground truth depth data path
        gt_pathes = []
        gt_pose_pathes = []
        for pred_path in pred_pathes:
            frame_name = os.path.basename(pred_path).replace('frame_', '').replace('.npy', '')
            frame_number = int(frame_name)
            scene_name = os.path.basename(os.path.dirname(pred_path))
            gt_path = os.path.join('data/iphone', scene_name, 'prepare_all_stride10/depth_npy', f"0_{frame_number*10:05d}.npy")
            gt_pose_path = os.path.join('data/iphone', scene_name, 'prepare_all_stride10/camera', f"0_{frame_number*10:05d}.json")
            
            if os.path.exists(gt_path):
                gt_pathes.append(gt_path)
            if os.path.exists(gt_pose_path):
                gt_pose_pathes.append(gt_pose_path)
        
        if len(gt_pathes) == 0:
            print("error: no corresponding ground truth depth data file found")
            continue
            
        grouped_pred_depth = group_by_directory(pred_pathes)
        grouped_gt_depth = group_by_directory(gt_pathes, idx=-3)  # use idx=-3 to match scene name
        grouped_gt_pose = group_by_directory(gt_pose_pathes, idx=-3)  # use idx=-3 to match scene name
        gathered_point_metrics = []
        
        for key in tqdm(grouped_pred_depth.keys(), desc=f"Processing {method}"):
            pd_pathes = grouped_pred_depth[key]
            gt_pathes = grouped_gt_depth[key]
            gt_pose_pathes = grouped_gt_pose[key]
            gt_pose = np.stack([load_iphone_traj(gt_pose_path) for gt_pose_path in gt_pose_pathes], axis=0) # [B, 4, 4]
            gt_intrinsics = np.stack([load_iphone_intrinsics(gt_pose_path) for gt_pose_path in gt_pose_pathes], axis=0) # [B, 3, 3]

            pr_depth = []
            pr_conf = []
            for pd_path in pd_pathes:
                depth_image = np.load(pd_path)  # load depth map
                frame_name = os.path.basename(pd_path).replace('frame_', '').replace('.npy', '')
                frame_number = int(frame_name)
                conf_path = '/'.join(pd_path.split('/')[:-1]) + f'/conf_{frame_number}.npy'
                conf_image = np.load(conf_path)
                pr_depth.append(depth_image)
                pr_conf.append(conf_image)
            pr_depth = np.stack(pr_depth, axis=0)
            pr_conf = np.stack(pr_conf, axis=0)

            gt_depth = []
            for gt_path in gt_pathes:
                depth_image = depth_read_iphone(gt_path)
                original_shape = depth_image.shape 
                resized_depth = cv2.resize(depth_image, (pr_depth.shape[2], pr_depth.shape[1]), interpolation=cv2.INTER_CUBIC)
                gt_depth.append(resized_depth)
            gt_depth = np.stack(gt_depth, axis=0)


            # load predicted trajectory
            pred_traj_path = os.path.join(method_dir, key, 'pred_traj.txt')
            pred_traj = np.loadtxt(pred_traj_path) # [B, 8]
            pred_pose = tum_to_4x4(pred_traj, xyzw=False) 

            aligned_pred_pose, scale = align_trajectory(pred_pose, gt_pose)
            # visualize_trajectory_alignment(pred_pose, gt_pose, aligned_pred_pose)

            # load predicted intrinsics
            pred_intrinsics_path = os.path.join(method_dir, key, 'pred_intrinsics.txt')
            pred_intrinsics = np.loadtxt(pred_intrinsics_path) # [B, 9]
            pred_intrinsics = pred_intrinsics.reshape(pred_intrinsics.shape[0], 3, 3)

            new_width, new_height = pr_depth.shape[2], pr_depth.shape[1]  # new width and height

            # calculate scale factor
            fx_scale = new_width / original_shape[1]
            fy_scale = new_height / original_shape[0]

            # update intrinsics
            gt_intrinsics[:, 0, 0] *= fx_scale  # fx
            gt_intrinsics[:, 1, 1] *= fy_scale  # fy
            gt_intrinsics[:, 0, 2] *= fx_scale  # cx
            gt_intrinsics[:, 1, 2] *= fy_scale  # cy

            
            # generate point cloud
            gt_points = depth_to_3d(gt_depth, gt_intrinsics, gt_pose)
            gt_points = gt_points.reshape(-1, 3)[(gt_depth>0).reshape(-1)]

            pred_points = depth_to_3d(pr_depth*scale, pred_intrinsics, aligned_pred_pose)
            pred_points = pred_points.reshape(-1, 3)[(gt_depth>0).reshape(-1)]

            # visualize_point_cloud_aligned(gt_points.reshape(-1, 3)[::1000], pred_points.reshape(-1, 3)[::1000])

            # use compute_3d_metrics to evaluate
            pred_points_tensor = torch.tensor(pred_points, dtype=torch.float32).cuda().reshape(-1, 3)
            gt_points_tensor = torch.tensor(gt_points, dtype=torch.float32).cuda().reshape(-1, 3)
            with torch.no_grad():
                metrics = compute_3d_metrics(pred_points_tensor, gt_points_tensor)
            gathered_point_metrics.append(metrics)

        if not gathered_point_metrics:
            print(f"warning: {method} no valid evaluation results")
            continue

        point_log_path = os.path.join(method_dir, 'point_metrics.json')
        average_metrics = {
            key: np.average(
                [metrics[key] for metrics in gathered_point_metrics]
            )
            for key in gathered_point_metrics[0].keys()
        }
        print(f'{method} average point cloud evaluation metrics:', average_metrics)
        with open(point_log_path, 'w') as f:
            f.write(json.dumps(average_metrics, indent=4))
            
        # add per scene metrics output
        per_scene_metrics = {}
        for i, key in enumerate(grouped_pred_depth.keys()):
            scene_metrics = gathered_point_metrics[i]
            per_scene_metrics[key] = {k: v for k, v in scene_metrics.items()}
            
        per_scene_log_path = os.path.join(method_dir, 'point_metrics_perscene.json')
        with open(per_scene_log_path, 'w') as f:
            f.write(json.dumps(per_scene_metrics, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='iPhone point cloud evaluation tool')
    parser.add_argument('--result_path', type=str, required=True,
                        help='The directory path containing iPhone prediction results')
    args = parser.parse_args()
    
    eval_iphone(args.result_path)