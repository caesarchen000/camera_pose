import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
import glob
from tqdm import tqdm
import os
from multiprocessing import Pool
from functools import partial
import argparse

def load_image(path):
    if path is None or not os.path.exists(str(path)):
        return np.zeros((100, 100, 3))
    img = cv2.imread(str(path))
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    return np.zeros((100, 100, 3))


def create_video(video_info):
    """ create video for single video sequence"""
    video_name, frames_dir, videos_dir = video_info
    frames_output_dir = os.path.join(frames_dir, video_name)
    video_output_path = os.path.join(videos_dir, f"{video_name}.mp4")
    os.system(f'/usr/bin/ffmpeg -y -framerate 24 -i "{frames_output_dir}/frame_%04d.png" '
             f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
             f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
             f'-movflags +faststart -b:v 5000k "{video_output_path}"')

def visualize_attns(image_path, a_mu_oc_path, a_mu_co_path, a_sigma_oc_path, a_sigma_co_path, a_fuse_path, save_path=None):
    """ create attention visualization layout"""
    plt.ioff()  # close interactive mode
    fig = plt.figure(figsize=(20, 5))  # adjust figure size to fit one line
    gs = plt.GridSpec(1, 6, figure=fig, wspace=0.05)  # 6 columns layout
    
    # image and attention map path and title
    images_row = [
        (image_path, "Image", gs[0, 0]),
        (a_mu_oc_path, r"$1-A_{\mu}^{o}$", gs[0, 1]),
        (a_sigma_oc_path, r"$A_{\sigma}^{o}$", gs[0, 2]),
        (a_mu_co_path, r"$1-A_{\mu}^{c}$", gs[0, 3]),
        (a_sigma_co_path, r"$A_{\sigma}^{c}$", gs[0, 4]),
        (a_fuse_path, r"$A_{fuse}$", gs[0, 5]),
    ]

    # show all images
    for path, title, pos in images_row:
        ax = fig.add_subplot(pos)
        img = load_image(path)
        ax.imshow(img)
        ax.axis('off')
        ax.text(0.5, -0.2, title, fontsize=16, ha='center', transform=ax.transAxes)

    # save or show image
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.05)
        plt.close()
    else:
        plt.show()

def visualize_cluster(image_path, a_fuse_path, a_cluster_path, a_temporal_fuse_path, mask_path, refined_mask_path, save_path=None):
    """ create cluster result visualization layout"""
    plt.ioff()  # close interactive mode
    fig = plt.figure(figsize=(20, 5))  # adjust figure size to fit one line
    gs = plt.GridSpec(1, 6, figure=fig, wspace=0.05)  # 6 columns layout
    
    # image and visualization path and title
    images_row = [
        (image_path, "Image", gs[0, 0]),
        (a_fuse_path, r"$A_{fuse}$", gs[0, 1]),
        (a_cluster_path, r"$Feature_{cluster}$", gs[0, 2]),
        (a_temporal_fuse_path, r"$A_{temporal\_fuse}$", gs[0, 3]),
        (mask_path, r"$Mask$", gs[0, 4]),
        (refined_mask_path, r"$Refined Mask$", gs[0, 5]),

    ]

    # show all images
    for path, title, pos in images_row:
        ax = fig.add_subplot(pos)
        img = load_image(path)
        ax.imshow(img)
        ax.axis('off')
        ax.text(0.5, -0.2, title, fontsize=16, ha='center', transform=ax.transAxes)

    # save or show image
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.05)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize attention and cluster results')
    parser.add_argument('--method_name', type=str, default='easi3r_monst3r',
                        help='Method name for the results directory')
    parser.add_argument('--base_output_dir', type=str, default='results/visualization',
                        help='Base output directory for visualization results')
    args = parser.parse_args()

    base_output_dir = args.base_output_dir
    method_name = args.method_name

    frames_attns_dir = os.path.join(base_output_dir, "frames_attns")
    frames_cluster_dir = os.path.join(base_output_dir, "frames_cluster")
    videos_attns_dir = os.path.join(base_output_dir, "videos_attns")
    videos_cluster_dir = os.path.join(base_output_dir, "videos_cluster")

    os.makedirs(frames_attns_dir, exist_ok=True)
    os.makedirs(frames_cluster_dir, exist_ok=True)
    os.makedirs(videos_attns_dir, exist_ok=True)
    os.makedirs(videos_cluster_dir, exist_ok=True)

    # get all video sequence image files
    image_files = sorted(glob.glob("data/davis/DAVIS/JPEGImages/480p/*/*.jpg"))
    
    # group video sequence by video name
    video_groups = {}
    for image_path in image_files:
        video_name = image_path.split('/')[-2]
        if video_name not in video_groups:
            video_groups[video_name] = []
        video_groups[video_name].append(image_path)
    
    def process_frame(args):
        video_name, i, image_path = args

        # call visualize_masks_attns to generate attention visualization
        frames_output_dir = os.path.join(frames_attns_dir, video_name)
        os.makedirs(frames_output_dir, exist_ok=True)
        visualize_attns(
            image_path=image_path,
            a_mu_oc_path=f"results/davis/{method_name}/{video_name}/0_cross_att_k_i_mean_fused/frames_att/frame_{i:04d}.png",
            a_mu_co_path=f"results/davis/{method_name}/{video_name}/0_cross_att_k_j_mean_fused/frames_att/frame_{i:04d}.png",
            a_sigma_oc_path=f"results/davis/{method_name}/{video_name}/0_cross_att_k_i_var_fused/frames_att/frame_{i:04d}.png",
            a_sigma_co_path=f"results/davis/{method_name}/{video_name}/0_cross_att_k_j_var_fused/frames_att/frame_{i:04d}.png",
            a_fuse_path=f"results/davis/{method_name}/{video_name}/0_dynamic_map_fused/frames_att/frame_{i:04d}.png",
            save_path=f"{frames_output_dir}/frame_{i:04d}.png"
        )

        # create output directory and generate visualization for cluster result
        frames_output_dir_cluster = os.path.join(frames_cluster_dir, video_name)
        os.makedirs(frames_output_dir_cluster, exist_ok=True)
        visualize_cluster(
            image_path=image_path,
            a_fuse_path=f"results/davis/{method_name}/{video_name}/0_dynamic_map_fused/frames_att/frame_{i:04d}.png",
            a_cluster_path=f"results/davis/{method_name}/{video_name}/0_refined_dynamic_map_labels_fused/frames_mask/frame_{i:04d}.png",
            a_temporal_fuse_path=f"results/davis/{method_name}/{video_name}/0_refined_dynamic_map_fused/frames_att/frame_{i:04d}.png",
            mask_path=f"results/davis/{method_name}/{video_name}/0_refined_dynamic_map_fused/frames_mask/frame_{i:04d}.png",
            refined_mask_path=f"results/davis/{method_name}/{video_name}/dynamic_mask_{i}.png",
            save_path=f"{frames_output_dir_cluster}/frame_{i:04d}.png",
        )
    
    # create processing tasks for each video sequence
    tasks = []
    for video_name, video_frames in video_groups.items():
        for i, image_path in enumerate(video_frames):
            tasks.append((video_name, i, image_path))
    
    # use process pool to process frames in parallel
    with Pool() as pool:
        list(tqdm(pool.imap(process_frame, tasks), total=len(tasks)))
    
    # generate attention video
    video_tasks = [(video_name, frames_attns_dir, videos_attns_dir) for video_name in video_groups.keys()]
    with Pool() as pool:
        list(tqdm(pool.imap(create_video, video_tasks), total=len(video_tasks), 
                 desc="generate attention video files"))

    # generate cluster video
    video_tasks_cluster = [(video_name, frames_cluster_dir, videos_cluster_dir) for video_name in video_groups.keys()]
    with Pool() as pool:
        list(tqdm(pool.imap(create_video, video_tasks_cluster), total=len(video_tasks_cluster), 
                 desc="generate cluster video files"))