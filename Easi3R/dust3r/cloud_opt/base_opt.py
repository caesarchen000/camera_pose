# --------------------------------------------------------
# Base class for the global alignement procedure
# --------------------------------------------------------
from copy import deepcopy
import cv2

import numpy as np
import torch
import torch.nn as nn
import roma
from copy import deepcopy
import tqdm

from dust3r.utils.geometry import inv, geotrf
from dust3r.utils.device import to_numpy
from dust3r.utils.image import rgb
from dust3r.viz import SceneViz, segment_sky, auto_cam_size
from dust3r.optim_factory import adjust_learning_rate_by_lr

from dust3r.cloud_opt.commons import (edge_str, ALL_DISTS, NoGradParamDict, get_imshapes, signed_expm1, signed_log1p,
                                      cosine_schedule, linear_schedule, cycled_linear_schedule, get_conf_trf)
import dust3r.cloud_opt.init_im_poses as init_fun
from scipy.spatial.transform import Rotation
from dust3r.utils.vo_eval import save_trajectory_tum_format
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu, threshold_multiotsu
import math
import torchvision
import cv2
    

def c2w_to_tumpose(c2w):
    """
    Convert a camera-to-world matrix to a tuple of translation and rotation
    
    input: c2w: 4x4 matrix
    output: tuple of translation and rotation (x y z qw qx qy qz)
    """
    # convert input to numpy
    c2w = to_numpy(c2w)
    xyz = c2w[:3, -1]
    rot = Rotation.from_matrix(c2w[:3, :3])
    qx, qy, qz, qw = rot.as_quat()
    tum_pose = np.concatenate([xyz, [qw, qx, qy, qz]])
    return tum_pose

class BasePCOptimizer (nn.Module):
    """ Optimize a global scene, given a list of pairwise observations.
    Graph node: images
    Graph edges: observations = (pred1, pred2)
    """

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            other = deepcopy(args[0])
            attrs = '''edges is_symmetrized dist n_imgs pred_i pred_j imshapes 
                        min_conf_thr conf_thr conf_i conf_j im_conf
                        base_scale norm_pw_scale POSE_DIM pw_poses 
                        pw_adaptors pw_adaptors has_im_poses rand_pose imgs verbose'''.split()
            self.__dict__.update({k: other[k] for k in attrs})
        else:
            self._init_from_views(*args, **kwargs)

    def _init_from_views(self, view1, view2, pred1, pred2,
                         dist='l1',
                         conf='log',
                         min_conf_thr=2.5,
                         thr_for_init_conf=False,
                         base_scale=0.5,
                         allow_pw_adaptors=False,
                         pw_break=20,
                         rand_pose=torch.randn,
                         empty_cache=False,
                         verbose=True,
                         use_atten_mask=False):
        super().__init__()
        if not isinstance(view1['idx'], list):
            view1['idx'] = view1['idx'].tolist()
        if not isinstance(view2['idx'], list):
            view2['idx'] = view2['idx'].tolist()
        self.edges = [(int(i), int(j)) for i, j in zip(view1['idx'], view2['idx'])]
        self.is_symmetrized = set(self.edges) == {(j, i) for i, j in self.edges}
        self.dist = ALL_DISTS[dist]
        self.verbose = verbose
        self.empty_cache = empty_cache
        self.n_imgs = self._check_edges()

        # input data
        pred1_pts = pred1['pts3d']
        pred2_pts = pred2['pts3d_in_other_view']
        self.pred_i = NoGradParamDict({ij: pred1_pts[n] for n, ij in enumerate(self.str_edges)})
        self.pred_j = NoGradParamDict({ij: pred2_pts[n] for n, ij in enumerate(self.str_edges)})
        self.imshapes = get_imshapes(self.edges, pred1_pts, pred2_pts)

        # work in log-scale with conf
        pred1_conf = pred1['conf']  # (Number of image_pairs, H, W)
        pred2_conf = pred2['conf']  # (Number of image_pairs, H, W)
        self.min_conf_thr = min_conf_thr
        self.thr_for_init_conf = thr_for_init_conf
        self.conf_trf = get_conf_trf(conf)

        self.conf_i = NoGradParamDict({ij: pred1_conf[n] for n, ij in enumerate(self.str_edges)})
        self.conf_j = NoGradParamDict({ij: pred2_conf[n] for n, ij in enumerate(self.str_edges)})
        self.im_conf = self._compute_img_conf(pred1_conf, pred2_conf)
        for i in range(len(self.im_conf)):
            self.im_conf[i].requires_grad = False

        self.init_conf_maps = [c.clone() for c in self.im_conf]

        # pairwise pose parameters
        self.base_scale = base_scale
        self.norm_pw_scale = True
        self.pw_break = pw_break
        self.POSE_DIM = 7
        self.pw_poses = nn.Parameter(rand_pose((self.n_edges, 1+self.POSE_DIM)))  # pairwise poses
        self.pw_adaptors = nn.Parameter(torch.zeros((self.n_edges, 2)))  # slight xy/z adaptation
        self.pw_adaptors.requires_grad_(allow_pw_adaptors)
        self.has_im_poses = False
        self.rand_pose = rand_pose

        # possibly store images, camera_pose, instance for show_pointcloud
        self.imgs = None
        if 'img' in view1 and 'img' in view2:
            imgs = [torch.zeros((3,)+hw) for hw in self.imshapes]
            for v in range(len(self.edges)):
                idx = view1['idx'][v]
                imgs[idx] = view1['img'][v]
                idx = view2['idx'][v]
                imgs[idx] = view2['img'][v]
            self.imgs = rgb(imgs)

        self.dynamic_masks = None
        if 'dynamic_mask' in view1 and 'dynamic_mask' in view2:
            dynamic_masks = [torch.zeros(hw) for hw in self.imshapes]
            for v in range(len(self.edges)):
                idx = view1['idx'][v]
                dynamic_masks[idx] = view1['dynamic_mask'][v]
                idx = view2['idx'][v]
                dynamic_masks[idx] = view2['dynamic_mask'][v]
            self.dynamic_masks = dynamic_masks

        self.camera_poses = None
        if 'camera_pose' in view1 and 'camera_pose' in view2:
            camera_poses = [torch.zeros((4, 4)) for _ in range(self.n_imgs)]
            for v in range(len(self.edges)):
                idx = view1['idx'][v]
                camera_poses[idx] = view1['camera_pose'][v]
                idx = view2['idx'][v]
                camera_poses[idx] = view2['camera_pose'][v]
            self.camera_poses = camera_poses

        self.img_pathes = None
        if 'instance' in view1 and 'instance' in view2:
            img_pathes = ['' for _ in range(self.n_imgs)]
            for v in range(len(self.edges)):
                idx = view1['idx'][v]
                img_pathes[idx] = view1['instance'][v]
                idx = view2['idx'][v]
                img_pathes[idx] = view2['instance'][v]
            self.img_pathes = img_pathes


        if use_atten_mask:
            # attention map
            cross_att_k_i_mean, cross_att_k_i_var, cross_att_k_j_mean, cross_att_k_j_var = self.aggregate_attention_maps(pred1, pred2)

            def fuse_attention_channels(att_maps):
                # att_maps: B, H, W, C
                # normalize
                att_maps_min = att_maps.min()
                att_maps_max = att_maps.max()
                att_maps_normalized = (att_maps - att_maps_min) / (att_maps_max - att_maps_min + 1e-6)
                # average channel
                att_maps_fused = att_maps_normalized.mean(dim=-1) # B, H, W
                # normalize
                att_maps_fused_min = att_maps_fused.min()
                att_maps_fused_max = att_maps_fused.max()
                att_maps_fused = (att_maps_fused - att_maps_fused_min) / (att_maps_fused_max - att_maps_fused_min + 1e-6)
                return att_maps_normalized, att_maps_fused
            
            self.cross_att_k_i_mean, self.cross_att_k_i_mean_fused = fuse_attention_channels(cross_att_k_i_mean)
            self.cross_att_k_i_var, self.cross_att_k_i_var_fused = fuse_attention_channels(cross_att_k_i_var)
            self.cross_att_k_j_mean, self.cross_att_k_j_mean_fused = fuse_attention_channels(cross_att_k_j_mean)
            self.cross_att_k_j_var, self.cross_att_k_j_var_fused = fuse_attention_channels(cross_att_k_j_var)

            # create dynamic mask
            dynamic_map = (1-self.cross_att_k_i_mean_fused) * self.cross_att_k_i_var_fused * self.cross_att_k_j_mean_fused * (1-self.cross_att_k_j_var_fused)
            dynamic_map_min = dynamic_map.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0] # B, 1, 1
            dynamic_map_max = dynamic_map.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0] # B, 1, 1
            self.dynamic_map = (dynamic_map - dynamic_map_min) / (dynamic_map_max - dynamic_map_min + 1e-6)

            # feature
            pred1_feat = pred1['match_feature']
            feat_i = NoGradParamDict({ij: nn.Parameter(pred1_feat[n], requires_grad=False) for n, ij in enumerate(self.str_edges)})
            stacked_feat_i = [feat_i[k] for k in self.str_edges]
            stacked_feat = [None] * len(self.imshapes)
            for i, ei in enumerate(torch.tensor([i for i, j in self.edges])):
                stacked_feat[ei]=stacked_feat_i[i]
            self.stacked_feat = torch.stack(stacked_feat).float().detach()

            self.refined_dynamic_map, self.dynamic_map_labels = cluster_attention_maps(self.stacked_feat, self.dynamic_map, n_clusters=64)


    def aggregate_attention_maps(self, pred1, pred2):
        
        def aggregate_attention(attention_maps, aggregate_j=True):
            attention_maps = NoGradParamDict({ij: nn.Parameter(attention_maps[n], requires_grad=False) 
                                            for n, ij in enumerate(self.str_edges)})
            aggregated_maps = {}
            for edge, attention_map in attention_maps.items():
                idx = edge.split('_')[1 if aggregate_j else 0]
                att = attention_map.clone()
                if idx not in aggregated_maps: 
                    aggregated_maps[idx] = [att]
                else:
                    aggregated_maps[idx].append(att)
            stacked_att_mean = [None] * len(self.imshapes)
            stacked_att_var = [None] * len(self.imshapes)
            for i, aggregated_map in aggregated_maps.items():
                att = torch.stack(aggregated_map, dim=-1)
                att[0,0] = (att[0,1] + att[1,0])/2
                stacked_att_mean[int(i)] = att.mean(dim=-1)
                stacked_att_var[int(i)] = att.std(dim=-1)
            return torch.stack(stacked_att_mean).float().detach(), torch.stack(stacked_att_var).float().detach()
        
        cross_att_k_i_mean, cross_att_k_i_var = aggregate_attention(pred1['cross_atten_maps_k'], aggregate_j=True)
        cross_att_k_j_mean, cross_att_k_j_var = aggregate_attention(pred2['cross_atten_maps_k'], aggregate_j=False)
        return cross_att_k_i_mean, cross_att_k_i_var, cross_att_k_j_mean, cross_att_k_j_var

    def save_attention_maps(self, save_folder='demo_tmp/attention_vis'):
        self.vis_attention_masks(1-self.cross_att_k_i_mean_fused, save_folder=save_folder, save_name='cross_att_k_i_mean')
        self.vis_attention_masks(self.cross_att_k_i_var_fused, save_folder=save_folder, save_name='cross_att_k_i_var')
        self.vis_attention_masks(1-self.cross_att_k_j_mean_fused, save_folder=save_folder, save_name='cross_att_k_j_mean')
        self.vis_attention_masks(self.cross_att_k_j_var_fused, save_folder=save_folder, save_name='cross_att_k_j_var')
        self.vis_attention_masks(self.dynamic_map, save_folder=save_folder, save_name='dynamic_map')
        self.vis_attention_masks(self.refined_dynamic_map, save_folder=save_folder, save_name='refined_dynamic_map')
        self.vis_attention_masks(self.refined_dynamic_map, save_folder=save_folder, save_name='refined_dynamic_map_labels', \
                            cluster_labels=self.dynamic_map_labels)

    @property
    def n_edges(self):
        return len(self.edges)

    @property
    def str_edges(self):
        return [edge_str(i, j) for i, j in self.edges]

    @property
    def imsizes(self):
        return [(w, h) for h, w in self.imshapes]

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def state_dict(self, trainable=True):
        all_params = super().state_dict()
        return {k: v for k, v in all_params.items() if k.startswith(('_', 'pred_i.', 'pred_j.', 'conf_i.', 'conf_j.')) != trainable}

    def load_state_dict(self, data):
        return super().load_state_dict(self.state_dict(trainable=False) | data)

    def _check_edges(self):
        indices = sorted({i for edge in self.edges for i in edge})
        assert indices == list(range(len(indices))), 'bad pair indices: missing values '
        return len(indices)

    @torch.no_grad()
    def _compute_img_conf(self, pred1_conf, pred2_conf):
        im_conf = nn.ParameterList([torch.zeros(hw, device=self.device) for hw in self.imshapes])
        for e, (i, j) in enumerate(self.edges):
            im_conf[i] = torch.maximum(im_conf[i], pred1_conf[e])
            im_conf[j] = torch.maximum(im_conf[j], pred2_conf[e])
        return im_conf

    def get_adaptors(self):
        adapt = self.pw_adaptors
        adapt = torch.cat((adapt[:, 0:1], adapt), dim=-1)  # (scale_xy, scale_xy, scale_z)
        if self.norm_pw_scale:  # normalize so that the product == 1
            adapt = adapt - adapt.mean(dim=1, keepdim=True)
        return (adapt / self.pw_break).exp()

    def _get_poses(self, poses):
        # normalize rotation
        Q = poses[:, :4]
        T = signed_expm1(poses[:, 4:7])
        RT = roma.RigidUnitQuat(Q, T).normalize().to_homogeneous()
        return RT

    def _set_pose(self, poses, idx, R, T=None, scale=None, force=False):
        # all poses == cam-to-world
        pose = poses[idx]
        if not (pose.requires_grad or force):
            return pose

        if R.shape == (4, 4):
            assert T is None
            T = R[:3, 3]
            R = R[:3, :3]

        if R is not None:
            pose.data[0:4] = roma.rotmat_to_unitquat(R)
        if T is not None:
            pose.data[4:7] = signed_log1p(T / (scale or 1))  # translation is function of scale

        if scale is not None:
            assert poses.shape[-1] in (8, 13)
            pose.data[-1] = np.log(float(scale))
        return pose

    def get_pw_norm_scale_factor(self):
        if self.norm_pw_scale:
            # normalize scales so that things cannot go south
            # we want that exp(scale) ~= self.base_scale
            return (np.log(self.base_scale) - self.pw_poses[:, -1].mean()).exp()
        else:
            return 1  # don't norm scale for known poses

    def get_pw_scale(self):
        scale = self.pw_poses[:, -1].exp()  # (n_edges,)
        scale = scale * self.get_pw_norm_scale_factor()
        return scale

    def get_pw_poses(self):  # cam to world
        RT = self._get_poses(self.pw_poses)
        scaled_RT = RT.clone()
        scaled_RT[:, :3] *= self.get_pw_scale().view(-1, 1, 1)  # scale the rotation AND translation
        return scaled_RT

    def get_masks(self):
        if self.thr_for_init_conf:
            return [(conf > self.min_conf_thr) for conf in self.init_conf_maps]
        else:
            return [(conf > self.min_conf_thr) for conf in self.im_conf]

    def get_feats(self):
        return self.stacked_feat

    def get_atts(self):
        return self.refined_dynamic_map

    def depth_to_pts3d(self):
        raise NotImplementedError()

    def get_pts3d(self, raw=False, **kwargs):
        res = self.depth_to_pts3d(**kwargs)
        if not raw:
            res = [dm[:h*w].view(h, w, 3) for dm, (h, w) in zip(res, self.imshapes)]
        return res

    def _set_focal(self, idx, focal, force=False):
        raise NotImplementedError()

    def get_focals(self):
        raise NotImplementedError()

    def get_known_focal_mask(self):
        raise NotImplementedError()

    def get_principal_points(self):
        raise NotImplementedError()

    def get_conf(self, mode=None):
        trf = self.conf_trf if mode is None else get_conf_trf(mode)
        return [trf(c) for c in self.im_conf]
    
    def get_init_conf(self, mode=None):
        trf = self.conf_trf if mode is None else get_conf_trf(mode)
        return [trf(c) for c in self.init_conf_maps]

    def get_im_poses(self):
        raise NotImplementedError()

    def _set_depthmap(self, idx, depth, force=False):
        raise NotImplementedError()

    def get_depthmaps(self, raw=False):
        raise NotImplementedError()

    def clean_pointcloud(self, **kw):
        cams = inv(self.get_im_poses())
        K = self.get_intrinsics()
        depthmaps = self.get_depthmaps()
        all_pts3d = self.get_pts3d()

        new_im_confs = clean_pointcloud(self.im_conf, K, cams, depthmaps, all_pts3d, **kw)

        for i, new_conf in enumerate(new_im_confs):
            self.im_conf[i].data[:] = new_conf
        return self

    def get_tum_poses(self):
        poses = self.get_im_poses()
        tt = np.arange(len(poses)).astype(float)
        tum_poses = [c2w_to_tumpose(p) for p in poses]
        tum_poses = np.stack(tum_poses, 0)
        return [tum_poses, tt]

    def save_tum_poses(self, path):
        traj = self.get_tum_poses()
        save_trajectory_tum_format(traj, path)
        return traj[0] # return the poses
    
    def save_focals(self, path):
        # convert focal to txt
        focals = self.get_focals()
        np.savetxt(path, focals.detach().cpu().numpy(), fmt='%.6f')
        return focals

    def save_intrinsics(self, path):
        K_raw = self.get_intrinsics()
        K = K_raw.reshape(-1, 9)
        np.savetxt(path, K.detach().cpu().numpy(), fmt='%.6f')
        return K_raw

    def save_conf_maps(self, path):
        conf = self.get_conf()
        for i, c in enumerate(conf):
            np.save(f'{path}/conf_{i}.npy', c.detach().cpu().numpy())
        return conf
    
    def save_init_conf_maps(self, path):
        conf = self.get_init_conf()
        for i, c in enumerate(conf):
            np.save(f'{path}/init_conf_{i}.npy', c.detach().cpu().numpy())
        return conf

    def save_rgb_imgs(self, path):
        imgs = self.imgs
        for i, img in enumerate(imgs):
            # convert from rgb to bgr
            img = img[..., ::-1]
            cv2.imwrite(f'{path}/frame_{i:04d}.png', img*255)
        return imgs

    def save_dynamic_masks(self, path):
        dynamic_masks = self.dynamic_masks if getattr(self, 'sam2_dynamic_masks', None) is None else self.sam2_dynamic_masks
        for i, dynamic_mask in enumerate(dynamic_masks):
            cv2.imwrite(f'{path}/dynamic_mask_{i}.png', (dynamic_mask * 255).detach().cpu().numpy().astype(np.uint8))

        # save video - use ffmpeg instead of OpenCV
        if len(dynamic_masks) > 0:
            h, w = dynamic_masks[0].shape
            # save all frames first
            frames_dir = os.path.join(path, 'frames_dynamic_masks')
            os.makedirs(frames_dir, exist_ok=True)
            for i, mask in enumerate(dynamic_masks):
                frame = (mask * 255).detach().cpu().numpy().astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(f'{frames_dir}/frame_{i:04d}.png', frame)
            
            # use ffmpeg to generate video, frame rate set to 24
            video_output_path = os.path.join(path, '0_dynamic_masks.mp4')
            os.system(f'/usr/bin/ffmpeg -y -framerate 24 -i "{frames_dir}/frame_%04d.png" '
                     f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                     f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                     f'-movflags +faststart -b:v 5000k "{video_output_path}"')

        return dynamic_masks

    def save_init_fused_dynamic_masks(self, path):
        # save init_dynamic_masks video
        if len(self.init_dynamic_masks) > 0:
            h, w = self.init_dynamic_masks[0].shape
            # save all frames first
            frames_dir = os.path.join(path, 'frames_init_dynamic_masks')
            os.makedirs(frames_dir, exist_ok=True)
            for i, mask in enumerate(self.init_dynamic_masks):
                frame = (mask * 255).detach().cpu().numpy().astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(f'{frames_dir}/frame_{i:04d}.png', frame)
            
            # use ffmpeg to generate video, frame rate set to 24
            video_output_path = os.path.join(path, '0_init_dynamic_masks.mp4')
            os.system(f'/usr/bin/ffmpeg -y -framerate 24 -i "{frames_dir}/frame_%04d.png" '
                     f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                     f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                     f'-movflags +faststart -b:v 5000k "{video_output_path}"')

        # save dynamic_masks video
        if len(self.dynamic_masks) > 0:
            h, w = self.dynamic_masks[0].shape
            # save all frames first
            frames_dir = os.path.join(path, 'frames_fused_dynamic_masks')
            os.makedirs(frames_dir, exist_ok=True)
            for i, mask in enumerate(self.dynamic_masks):
                frame = (mask * 255).detach().cpu().numpy().astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(f'{frames_dir}/frame_{i:04d}.png', frame)
            
            # use ffmpeg to generate video, frame rate set to 24
            video_output_path = os.path.join(path, '0_fused_dynamic_masks.mp4')
            os.system(f'/usr/bin/ffmpeg -y -framerate 24 -i "{frames_dir}/frame_%04d.png" '
                     f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                     f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                     f'-movflags +faststart -b:v 5000k "{video_output_path}"')

    def save_depth_maps(self, path):
        depth_maps = self.get_depthmaps()
        images = []
        
        for i, depth_map in enumerate(depth_maps):
            # Apply color map to depth map
            depth_map_colored = cv2.applyColorMap((depth_map * 255).detach().cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET)
            img_path = f'{path}/frame_{(i):04d}.png'
            cv2.imwrite(img_path, depth_map_colored)
            images.append(Image.open(img_path))
            np.save(f'{path}/frame_{(i):04d}.npy', depth_map.detach().cpu().numpy())
        
        images[0].save(f'{path}/_depth_maps.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
        
        return depth_maps

    def forward(self, ret_details=False):
        pw_poses = self.get_pw_poses()  # cam-to-world
        pw_adapt = self.get_adaptors()
        proj_pts3d = self.get_pts3d()
        # pre-compute pixel weights
        weight_i = {i_j: self.conf_trf(c) for i_j, c in self.conf_i.items()}
        weight_j = {i_j: self.conf_trf(c) for i_j, c in self.conf_j.items()}

        loss = 0
        if ret_details:
            details = -torch.ones((self.n_imgs, self.n_imgs))

        for e, (i, j) in enumerate(self.edges):
            i_j = edge_str(i, j)
            # distance in image i and j
            aligned_pred_i = geotrf(pw_poses[e], pw_adapt[e] * self.pred_i[i_j])
            aligned_pred_j = geotrf(pw_poses[e], pw_adapt[e] * self.pred_j[i_j])
            li = self.dist(proj_pts3d[i], aligned_pred_i, weight=weight_i[i_j]).mean()
            lj = self.dist(proj_pts3d[j], aligned_pred_j, weight=weight_j[i_j]).mean()
            loss = loss + li + lj

            if ret_details:
                details[i, j] = li + lj
        loss /= self.n_edges  # average over all pairs

        if ret_details:
            return loss, details
        return loss

    @torch.cuda.amp.autocast(enabled=False)
    def compute_global_alignment(self, init=None, save_score_path=None, save_score_only=False, niter_PnP=10, **kw):
        if init is None:
            pass
        elif init == 'msp' or init == 'mst':
            init_fun.init_minimum_spanning_tree(self, save_score_path=save_score_path, save_score_only=save_score_only, niter_PnP=niter_PnP)
            if save_score_only: # if only want the score map
                return None
        elif init == 'known_poses':
            self.preset_pose(known_poses=self.camera_poses, requires_grad=True)
            init_fun.init_from_known_poses(self, min_conf_thr=self.min_conf_thr,
                                           niter_PnP=niter_PnP)
        else:
            raise ValueError(f'bad value for {init=}')

        return global_alignment_loop(self, **kw)

    @torch.no_grad()
    def mask_sky(self):
        res = deepcopy(self)
        for i in range(self.n_imgs):
            sky = segment_sky(self.imgs[i])
            res.im_conf[i][sky] = 0
        return res

    def show(self, show_pw_cams=False, show_pw_pts3d=False, cam_size=None, **kw):
        viz = SceneViz()
        if self.imgs is None:
            colors = np.random.randint(0, 256, size=(self.n_imgs, 3))
            colors = list(map(tuple, colors.tolist()))
            for n in range(self.n_imgs):
                viz.add_pointcloud(self.get_pts3d()[n], colors[n], self.get_masks()[n])
        else:
            viz.add_pointcloud(self.get_pts3d(), self.imgs, self.get_masks())
            colors = np.random.randint(256, size=(self.n_imgs, 3))

        # camera poses
        im_poses = to_numpy(self.get_im_poses())
        if cam_size is None:
            cam_size = auto_cam_size(im_poses)
        viz.add_cameras(im_poses, self.get_focals(), colors=colors,
                        images=self.imgs, imsizes=self.imsizes, cam_size=cam_size)
        if show_pw_cams:
            pw_poses = self.get_pw_poses()
            viz.add_cameras(pw_poses, color=(192, 0, 192), cam_size=cam_size)

            if show_pw_pts3d:
                pts = [geotrf(pw_poses[e], self.pred_i[edge_str(i, j)]) for e, (i, j) in enumerate(self.edges)]
                viz.add_pointcloud(pts, (128, 0, 128))

        viz.show(**kw)
        return viz


    @torch.no_grad()
    def vis_attention_masks(self, attns_fused, save_folder='demo_tmp/attention_vis', save_name='attention_channels_all_frames', cluster_labels=None):
        B, H, W = attns_fused.shape

        # ensure self.imshape exists, otherwise use the original size
        target_size = getattr(self, 'imshape', (H, W))
        
        # upsample the attention maps
        upsampled_attns = torch.nn.functional.interpolate(
            attns_fused.unsqueeze(1),  # [B, 1, H, W]
            size=target_size, 
            mode='nearest'
        ).squeeze(1)  # [B, H', W']
        
        # if there is cluster_labels, also upsample it
        if cluster_labels is not None:
            upsampled_labels = torch.nn.functional.interpolate(
                cluster_labels.float().unsqueeze(1),  # [B, 1, H, W]
                size=target_size,
                mode='nearest'
            ).squeeze(1).long()  # [B, H', W']
        
        # use matplotlib's Spectral_r color map
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap('Spectral_r')
        
        # apply color map to each attention map
        H_up, W_up = upsampled_attns.shape[1:]
        stacked_att_img = torch.zeros((B, 3, H_up, W_up), device=upsampled_attns.device)
        for i in range(B):
            att_np = upsampled_attns[i].cpu().numpy()
            colored_att = cmap(att_np)[:, :, :3]  # remove alpha channel
            colored_att_torch = torch.from_numpy(colored_att).float().permute(2, 0, 1).to(upsampled_attns.device)
            stacked_att_img[i] = colored_att_torch

        # calculate mask
        stacked_mask = (upsampled_attns > adaptive_multiotsu_variance(upsampled_attns.cpu().numpy()))

        if cluster_labels is not None:
            import matplotlib.pyplot as plt
            num_clusters = upsampled_labels.max().item() + 1
            colors = plt.cm.tab20(np.linspace(0, 1, num_clusters))[:, :3]
            colors = torch.from_numpy(colors).float().to(upsampled_labels.device)
            
            stacked_mask_img = torch.zeros((B, 3, H_up, W_up), device=upsampled_labels.device)
            for i in range(num_clusters):
                mask = (upsampled_labels == i) & stacked_mask 
                mask = mask.unsqueeze(1)  # [B, 1, H', W']
                stacked_mask_img += mask * colors[i].view(1, 3, 1, 1)
        else:
            stacked_mask_img = stacked_mask.unsqueeze(1).expand(-1, 3, -1, -1)  # [B, 3, H', W']

        # create grid layout  
        grid_size = int(math.ceil(math.sqrt(B)))
        # for stacked_att and cluster_map create grid
        grid_att = torchvision.utils.make_grid(stacked_att_img, nrow=grid_size, padding=2, normalize=False)
        grid_cluster = torchvision.utils.make_grid(stacked_mask_img, nrow=grid_size, padding=2, normalize=False)
        # concatenate two grids in vertical direction
        final_grid = torch.cat([grid_att, grid_cluster], dim=1)
        torchvision.utils.save_image(final_grid, os.path.join(save_folder, f'0_{save_name}_fused.png'))

        # vis
        fused_save_folder = os.path.join(save_folder, f'0_{save_name}_fused')
        os.makedirs(fused_save_folder, exist_ok=True)

        # save video
        if B > 0:
            # create frames directory for stacked_att_img
            frames_att_dir = os.path.join(fused_save_folder, 'frames_att')
            os.makedirs(frames_att_dir, exist_ok=True)
            
            for i in range(B):
                att_frame = stacked_att_img[i].cpu().numpy().transpose(1, 2, 0)  # convert to HWC format
                frame = (att_frame * 255).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # convert to BGR format
                cv2.imwrite(f'{frames_att_dir}/frame_{i:04d}.png', frame)
            
            # use ffmpeg to generate video, frame rate set to 24
            video_att_path = os.path.join(fused_save_folder, f'0_{save_name}_att_video.mp4')
            os.system(f'/usr/bin/ffmpeg -y -framerate 24 -i "{frames_att_dir}/frame_%04d.png" '
                    f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                    f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                    f'-movflags +faststart -b:v 5000k "{video_att_path}"')

            # create frames directory for stacked_mask_img
            frames_mask_dir = os.path.join(fused_save_folder, 'frames_mask')
            os.makedirs(frames_mask_dir, exist_ok=True)
            
            for i in range(B):
                mask_frame = stacked_mask_img[i].cpu().numpy().transpose(1, 2, 0)  # convert to HWC format
                frame = (mask_frame * 255).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # convert to BGR format
                cv2.imwrite(f'{frames_mask_dir}/frame_{i:04d}.png', frame)
            
            # use ffmpeg to generate video, frame rate set to 24
            video_mask_path = os.path.join(fused_save_folder, f'0_{save_name}_mask_video.mp4')
            os.system(f'/usr/bin/ffmpeg -y -framerate 24 -i "{frames_mask_dir}/frame_%04d.png" '
                    f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                    f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                    f'-movflags +faststart -b:v 5000k "{video_mask_path}"')


def global_alignment_loop(net, lr=0.01, niter=300, schedule='cosine', lr_min=1e-3, temporal_smoothing_weight=0, depth_map_save_dir=None):
    params = [p for p in net.parameters() if p.requires_grad]
    if not params:
        return net

    verbose = net.verbose
    if verbose:
        print('Global alignement - optimizing for:')
        print([name for name, value in net.named_parameters() if value.requires_grad])

    lr_base = lr
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.9))

    loss = float('inf')
    if verbose:
        with tqdm.tqdm(total=niter) as bar:
            while bar.n < bar.total:
                if bar.n % 500 == 0 and depth_map_save_dir is not None:
                    if not os.path.exists(depth_map_save_dir):
                        os.makedirs(depth_map_save_dir)
                    # visualize the depthmaps
                    depth_maps = net.get_depthmaps()
                    for i, depth_map in enumerate(depth_maps):
                        depth_map_save_path = os.path.join(depth_map_save_dir, f'depthmaps_{i}_iter_{bar.n}.png')
                        plt.imsave(depth_map_save_path, depth_map.detach().cpu().numpy(), cmap='jet')
                    print(f"Saved depthmaps at iteration {bar.n} to {depth_map_save_dir}")
                loss, lr = global_alignment_iter(net, bar.n, niter, lr_base, lr_min, optimizer, schedule, 
                                                 temporal_smoothing_weight=temporal_smoothing_weight)
                bar.set_postfix_str(f'{lr=:g} loss={loss:g}')
                bar.update()
    else:
        for n in range(niter):
            loss, _ = global_alignment_iter(net, n, niter, lr_base, lr_min, optimizer, schedule, 
                                            temporal_smoothing_weight=temporal_smoothing_weight)
    return loss


def global_alignment_iter(net, cur_iter, niter, lr_base, lr_min, optimizer, schedule, temporal_smoothing_weight=0):
    t = cur_iter / niter
    if schedule == 'cosine':
        lr = cosine_schedule(t, lr_base, lr_min)
    elif schedule == 'linear':
        lr = linear_schedule(t, lr_base, lr_min)
    elif schedule.startswith('cycle'):
        try:
            num_cycles = int(schedule[5:])
        except ValueError:
            num_cycles = 2
        lr = cycled_linear_schedule(t, lr_base, lr_min, num_cycles=num_cycles)
    else:
        raise ValueError(f'bad lr {schedule=}')
    
    adjust_learning_rate_by_lr(optimizer, lr)
    optimizer.zero_grad()

    if net.empty_cache:
        torch.cuda.empty_cache()
    
    loss = net(epoch=cur_iter)
    
    if net.empty_cache:
        torch.cuda.empty_cache()
    
    loss.backward()
    
    if net.empty_cache:
        torch.cuda.empty_cache()
    
    optimizer.step()
    
    return float(loss), lr



@torch.no_grad()
def clean_pointcloud( im_confs, K, cams, depthmaps, all_pts3d, 
                      tol=0.001, bad_conf=0, dbg=()):
    """ Method: 
    1) express all 3d points in each camera coordinate frame
    2) if they're in front of a depthmap --> then lower their confidence
    """
    assert len(im_confs) == len(cams) == len(K) == len(depthmaps) == len(all_pts3d)
    assert 0 <= tol < 1
    res = [c.clone() for c in im_confs]

    # reshape appropriately
    all_pts3d = [p.view(*c.shape,3) for p,c in zip(all_pts3d, im_confs)]
    depthmaps = [d.view(*c.shape) for d,c in zip(depthmaps, im_confs)]
    
    for i, pts3d in enumerate(all_pts3d):
        for j in range(len(all_pts3d)):
            if i == j: continue

            # project 3dpts in other view
            proj = geotrf(cams[j], pts3d)
            proj_depth = proj[:,:,2]
            u,v = geotrf(K[j], proj, norm=1, ncol=2).round().long().unbind(-1)

            # check which points are actually in the visible cone
            H, W = im_confs[j].shape
            msk_i = (proj_depth > 0) & (0 <= u) & (u < W) & (0 <= v) & (v < H)
            msk_j = v[msk_i], u[msk_i]

            # find bad points = those in front but less confident
            bad_points = (proj_depth[msk_i] < (1-tol) * depthmaps[j][msk_j]) & (res[i][msk_i] < res[j][msk_j])

            bad_msk_i = msk_i.clone()
            bad_msk_i[msk_i] = bad_points
            res[i][bad_msk_i] = res[i][bad_msk_i].clip_(max=bad_conf)

    return res


@torch.no_grad()
def cluster_attention_maps(feature, dynamic_map, n_clusters=64):
    """use KMeans to cluster the attention maps using feature
    
    Args:
        feature: encoder feature [B,H,W,C]
        dynamic_map: dynamic_map feature [B,H,W]
        n_clusters: number of clusters
        
    Returns:
        normalized_map: normalized cluster map [B,H,W]
        cluster_labels: reshaped cluster labels [B,H,W]
    """
    # data preprocessing
    B, H, W, C = feature.shape
    feature_np = feature.cpu().numpy()
    flattened_feature = feature_np.reshape(-1, C)
    
    # KMeans clustering
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(flattened_feature)
    
    # calculate the average dynamic score for each cluster
    dynamic_map_np = dynamic_map.cpu().numpy()
    flattened_dynamic = dynamic_map_np.reshape(-1)
    cluster_dynamic_scores = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_mask = (cluster_labels == i)
        cluster_dynamic_scores[i] = np.mean(flattened_dynamic[cluster_mask])
    
    # map the cluster labels to the dynamic score
    cluster_map = cluster_dynamic_scores[cluster_labels]
    normalized_map = cluster_map.reshape(B, H, W)

    # reshape cluster_labels
    reshaped_labels = cluster_labels.reshape(B, H, W)
    
    # convert to torch tensor
    normalized_map = torch.from_numpy(normalized_map).float()
    cluster_labels = torch.from_numpy(reshaped_labels).long()
    
    normalized_map_min = normalized_map.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    normalized_map_max = normalized_map.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    normalized_map = (normalized_map - normalized_map_min) / (normalized_map_max - normalized_map_min + 1e-6)

    return normalized_map, cluster_labels

def adaptive_multiotsu_variance(img, verbose=False):
    """adaptive multi-threshold Otsu algorithm based on inter-class variance maximization
    
    Args:
        img: input image array
        verbose: whether to print detailed information
        
    Returns:
        tuple: (best threshold, best number of classes)
    """
    max_classes = 4
    best_score = -float('inf')
    best_threshold = None
    best_n_classes = None
    scores = {}
    
    for n_classes in range(2, max_classes + 1):
        thresholds = threshold_multiotsu(img, classes=n_classes)
        
        regions = np.digitize(img, bins=thresholds)
        var_between = np.var([img[regions == i].mean() for i in range(n_classes)])
        
        score = var_between / np.sqrt(n_classes)
        scores[n_classes] = score
        
        if score > best_score:
            best_score = score
            best_threshold = thresholds[-1]
            best_n_classes = n_classes
    
    if verbose:
        print("number of classes score:")
        for n_classes, score in scores.items():
            print(f"number of classes {n_classes}: score {score:.4f}" + 
                  (" (best)" if n_classes == best_n_classes else ""))
        print(f"final selected number of classes: {best_n_classes}")
    
    return best_threshold