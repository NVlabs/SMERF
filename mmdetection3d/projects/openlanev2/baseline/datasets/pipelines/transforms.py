# ==============================================================================
# Binaries and/or source for the following packages or projects 
# are presented under one or more of the following open source licenses:
# transforms.py    The OpenLane-V2 Dataset Authors    Apache License, Version 2.0
#
# Contact wanghuijie@pjlab.org.cn if you have any issue.
#
# Copyright (c) 2023 The OpenLane-v2 Dataset Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Any
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from numpy import random
from math import factorial

import mmcv
from mmdet.datasets import PIPELINES
from shapely.geometry import LineString
from geopandas import GeoSeries


@PIPELINES.register_module()
class ResizeFrontView:

    def __init__(self):
        pass

    def __call__(self, results):
        assert 'ring_front_center' in results['img_paths'][0], \
            'the first image should be the front view'

        #image
        front_view = results['img'][0]
        h, w, _ = front_view.shape
        resiezed_front_view, w_scale, h_scale = mmcv.imresize(
            front_view,
            (h, w),
            return_scale=True,
        )
        results['img'][0] = resiezed_front_view
        results['img_shape'][0] = resiezed_front_view.shape

        # gt
        scale_factor = np.array(
            [w_scale, h_scale, w_scale, h_scale],
            dtype=np.float32,
        )
        # results['scale_factor'] = scale_factor
        results['scale_factor'] = scale_factor * results['scale_factor'] if 'scale_factor' in results else scale_factor
        if 'gt_te' in results:
            results['gt_te'] = results['gt_te'] * scale_factor

        # intrinsic
        lidar2cam_r = results['rots'][0]
        lidar2cam_t = (-results['trans'][0]) @ lidar2cam_r.T
        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t

        intrinsic = results['cam2imgs'][0]
        viewpad = np.eye(4)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

        cam_s = np.eye(4)
        cam_s[0, 0] *= w_scale
        cam_s[1, 1] *= h_scale

        viewpad = cam_s @ viewpad 
        intrinsic = viewpad[:intrinsic.shape[0], :intrinsic.shape[1]]
        lidar2img_rt = (viewpad @ lidar2cam_rt.T)

        results['cam_intrinsic'][0] = viewpad
        results['lidar2img'][0] = lidar2img_rt
        results['cam2imgs'][0] = intrinsic

        return results
    

@PIPELINES.register_module()
class ResizeMultiviewImage:
    r"""
    Notes
    -----
    Resize the image.
    Added key is "img_norm_cfg".
    Args:
        width (int): width of img in pxl.
        height (int): height of img in pxl.
    """

    def __init__(self, width, height, skip_frontal_view=False):
        self.width = width
        self.height = height
        self.skip_frontal_view = skip_frontal_view

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        resized_imgs = []
        viewpads = []
        intrinsics = []
        lidar2imgs = []
        for idx, img in enumerate(results['img']):
            if self.skip_frontal_view and idx == 0:
                resized_imgs.append(img)
                viewpads.append(results['cam_intrinsic'][idx])
                intrinsics.append(results['cam2imgs'][idx])
                lidar2imgs.append(results['lidar2img'][idx])
                continue
            resized_img, w_scale, h_scale = mmcv.imresize(img, (self.width, self.height), return_scale=True)
            resized_imgs.append(resized_img)
            # gt
            scale_factor = np.array(
                [w_scale, h_scale, w_scale, h_scale],
                dtype=np.float32,
            )
            results['scale_factor'] = scale_factor * results['scale_factor'] if 'scale_factor' in results else scale_factor
            if 'gt_te' in results and idx == 0:
                results['gt_te'] = results['gt_te'] * scale_factor

            # intrinsic
            lidar2cam_r = results['rots'][idx]
            lidar2cam_t = (-results['trans'][idx]) @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t

            intrinsic = results['cam2imgs'][idx]
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

            cam_s = np.eye(4)
            cam_s[0, 0] *= w_scale
            cam_s[1, 1] *= h_scale

            viewpad = cam_s @ viewpad 
            intrinsic = viewpad[:intrinsic.shape[0], :intrinsic.shape[1]]
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            viewpads.append(viewpad)
            intrinsics.append(intrinsic)
            lidar2imgs.append(lidar2img_rt)

        results['img'] = resized_imgs
        results['cam_intrinsic'] = viewpads
        results['lidar2img'] = lidar2imgs
        results['cam2imgs'] = intrinsics
        results['img_shape'] = [img.shape for img in resized_imgs]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(width={self.width}, height={self.height})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage:
    r"""
    Notes
    -----
    Adapted from https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py#L62.

    Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb


    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        results['img'] = [mmcv.imnormalize(img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str

    
@PIPELINES.register_module()
class NormalizeSDMapRaster:
    r"""
    Notes
    -----
    Adapted from https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py#L62.

    Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb


    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results['map_raster'] = mmcv.imnormalize(results['map_raster'], self.mean, self.std, self.to_rgb)
        results['map_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class AffineTransformSDMap:

    def __init__(self, rot_angle, translate_ratio):
        self.rot_angle = rot_angle
        self.translate_ratio = translate_ratio
        self.affine_transformer = T.RandomAffine(degrees=self.rot_angle, 
                                                translate=self.translate_ratio, 
                                                interpolation=InterpolationMode.NEAREST,
                                                fill=191)

    def __call__(self, results):
        # results['map_raster'] = mmcv.imnormalize(results['map_raster'], self.mean, self.std, self.to_rgb)
        map_raster_rand_affine_ = self.affine_transformer(torch.from_numpy(results['map_raster'].transpose(2, 0, 1)))
        results['map_raster'] = map_raster_rand_affine_.numpy().transpose(1, 2, 0)
        results['map_affine_cfg'] = dict(
            rot_angle=self.rot_angle, translate_ratio=self.translate_ratio)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(rot={self.rot_angle}, transl={self.translate_ratio})'
        return repr_str


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    r"""
    Notes
    -----
    Adapted from https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py#L99.
    
    Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str

@PIPELINES.register_module()
class CustomPadMultiViewImage:

    def __init__(self, size_divisor=None, pad_val=0):
        self.size_divisor = size_divisor
        self.pad_val = pad_val

    def __call__(self, results):
        max_h = max([img.shape[0] for img in results['img']])
        max_w = max([img.shape[1] for img in results['img']])
        padded_img = [mmcv.impad(img, shape=(max_h, max_w), pad_val=self.pad_val) for img in results['img']]
        if self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in padded_img]
        
        results['img'] = padded_img
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = None
        results['pad_size_divisor'] = self.size_divisor

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str
    
@PIPELINES.register_module()
class CustomResizeCropRaster:

    def __init__(self, size_divisor=None, pad_val=0):
        self.size_divisor = size_divisor
        self.pad_val = pad_val

    def __call__(self, results):
        max_h = max([img.shape[0] for img in results['img']])
        max_w = max([img.shape[1] for img in results['img']])
        padded_img = [mmcv.impad(img, shape=(max_h, max_w), pad_val=self.pad_val) for img in results['img']]
        if self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in padded_img]
        
        results['img'] = padded_img
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = None
        results['pad_size_divisor'] = self.size_divisor

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str

@PIPELINES.register_module()
class CustomParameterizeLane:

    def __init__(self, method, method_para):
        method_list = ['bezier', 'polygon', 'point_subsample', 'bezier_Direction_attribute', 'bezier_Endpointfixed']
        self.method = method
        if not self.method in method_list:
            raise Exception("Not implemented!")
        self.method_para = method_para

    def __call__(self, results):
        centerlines = results['gt_lc']
        para_centerlines = getattr(self, self.method)(centerlines, **self.method_para)
        results['gt_lc'] = para_centerlines
        return results

    def comb(self, n, k):
        return factorial(n) // (factorial(k) * factorial(n - k))

    def fit_bezier(self, points, n_control):
        n_points = len(points)
        A = np.zeros((n_points, n_control))
        t = np.arange(n_points) / (n_points - 1)
        for i in range(n_points):
            for j in range(n_control):
                A[i, j] = self.comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
        conts = np.linalg.lstsq(A, points, rcond=None)
        return conts

    def fit_bezier_Endpointfixed(self, points, n_control):
        n_points = len(points)
        A = np.zeros((n_points, n_control))
        t = np.arange(n_points) / (n_points - 1)
        for i in range(n_points):
            for j in range(n_control):
                A[i, j] = self.comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
        A_BE = A[1:-1, 1:-1]
        _points = points[1:-1]
        _points = _points - A[1:-1, 0].reshape(-1, 1) @ points[0].reshape(1, -1) - A[1:-1, -1].reshape(-1, 1) @ points[-1].reshape(1, -1)

        conts = np.linalg.lstsq(A_BE, _points, rcond=None)

        control_points = np.zeros((n_control, points.shape[1]))
        control_points[0] = points[0]
        control_points[-1] = points[-1]
        control_points[1:-1] = conts[0]

        return control_points
    
    def point_subsample(self, input_data, n_points=11):
        subsampled_list = []
        for idx, centerline in enumerate(input_data):
            points_subsample = centerline[::int((centerline.shape[0] - 1) / (n_points - 1)), :]
            subsampled_list.append(points_subsample.flatten())
        return np.array(subsampled_list, dtype=np.float32)

    def bezier(self, input_data, n_control=2):

        coeffs_list = []
        for idx, centerline in enumerate(input_data):
            sorted_x = np.array(centerline[:, 1])
            sorted_y = np.array(centerline[:, 0])
            points = np.array(list(zip(sorted_x, sorted_y)))
            res = self.fit_bezier(points, n_control)[0]
            start_res = res[0]
            end_res = res[-1]
            first_diff = (np.sum(np.square(start_res - points[0]))) + (np.sum(np.square(end_res - points[-1])))
            second_diff = (np.sum(np.square(start_res - points[-1]))) + (np.sum(np.square(end_res - points[0])))

            if first_diff <= second_diff:
                fin_res = res
            else:
                fin_res = np.zeros_like(res)
                for m in range(len(res)):
                    fin_res[len(res) - m - 1] = res[m]

            fin_res = np.clip(fin_res, 0, 1)
            coeffs_list.append(np.reshape(np.float32(fin_res), (-1)))

        return np.array(coeffs_list)

    def bezier_Direction_attribute(self, input_data, n_control=3):
        coeffs_list = []
        for idx, centerline in enumerate(input_data):
            centerline[:, 1] = centerline[:, 1]
            centerline[:, 0] = centerline[:, 0]
            sorted_x = np.array(centerline[:, 1])
            sorted_y = np.array(centerline[:, 0])
            points = np.array(list(zip(sorted_x, sorted_y)))
            res = self.fit_bezier(points, n_control)[0]
            fin_res = np.clip(res, 0, 1)
            start_res = res[0]
            end_res = res[-1]
            first_diff = (np.sum(np.square(start_res - points[0]))) + (np.sum(np.square(end_res - points[-1])))
            second_diff = (np.sum(np.square(start_res - points[-1]))) + (np.sum(np.square(end_res - points[0])))
            if first_diff <= second_diff:
                da = 0
            else:
                da = 1
            fin_res = np.append(fin_res, da)
            coeffs_list.append(np.reshape(np.float32(fin_res), (-1)))
        return np.array(coeffs_list)

    def bezier_Endpointfixed(self, input_data, n_control=2):
        coeffs_list = []
        for idx, centerline in enumerate(input_data):
            res = self.fit_bezier_Endpointfixed(centerline, n_control)
            coeffs = res.flatten()
            coeffs_list.append(coeffs)
        return np.array(coeffs_list, dtype=np.float32)

    def polygon(self, input_data, key_rep='Bounding Box'):
        keypoints = []
        for idx, centerline in enumerate(input_data):
            centerline[:, 1] = centerline[:, 1]
            centerline[:, 0] = centerline[:, 0]
            sorted_x = np.array(centerline[:, 1])
            sorted_y = np.array(centerline[:, 0])
            points = np.array(list(zip(sorted_x, sorted_y)))
            if key_rep not in ['Bounding Box', 'SME', 'Extreme Points']:
                raise Exception(f"{key_rep} not existed!")
            elif key_rep == 'Bounding Box':
                res = np.array(
                    [points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()]).reshape((2, 2))
                keypoints.append(np.reshape(np.float32(res), (-1)))
            elif key_rep == 'SME':
                res = np.array([points[0], points[-1], points[int(len(points) / 2)]])
                keypoints.append(np.reshape(np.float32(res), (-1)))
            else:
                min_x = np.min([points[:, 0] for p in points])
                ind_left = np.where(points[:, 0] == min_x)
                max_x = np.max([points[:, 0] for p in points])
                ind_right = np.where(points[:, 0] == max_x)
                max_y = np.max([points[:, 1] for p in points])
                ind_top = np.where(points[:, 1] == max_y)
                min_y = np.min([points[:, 1] for p in points])
                ind_botton = np.where(points[:, 1] == min_y)
                res = np.array(
                    [points[ind_left[0][0]], points[ind_right[0][0]], points[ind_top[0][0]], points[ind_botton[0][0]]])
                keypoints.append(np.reshape(np.float32(res), (-1)))
        return np.array(keypoints)


@PIPELINES.register_module()
class CustomParametrizeSDMapGraph:
    def __init__(self, method, method_para):
        self.method = method
        self.method_para = method_para

        # define some category mappings
        self.category2id = {
            'road': 0,        # from openlanev2
            'cross_walk': 1,  # from openlanev2
            'side_walk': 1,   # from openlanev2
            'pedestrian': 1,  # set to be sidewalk
            'truck_road': 2, 
            'highway': 3, 
            'residential': 4, 
            'service': 5, 
            'bus_way': 6, 
            'other': 7,
        }

    def __call__(self, results):
        sd_map = results['sd_map']
        sd_map_graph, map_meta = getattr(self, self.method)(sd_map, **self.method_para)
        
        # sd_map_graph: num_polylines x max_num_points x 3
        results["map_graph"] = sd_map_graph
        for key, value in map_meta.items():
            results[key] = value
        # results["map_num_poly_pnts"] = max_num_vec_per_polyline
        return results
    
    def fit_bezier_Endpointfixed(self, points, n_control):
        n_points = len(points)
        A = np.zeros((n_points, n_control))
        t = np.arange(n_points) / (n_points - 1)
        for i in range(n_points):
            for j in range(n_control):
                A[i, j] = self.comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
        A_BE = A[1:-1, 1:-1]
        _points = points[1:-1]
        _points = _points - A[1:-1, 0].reshape(-1, 1) @ points[0].reshape(1, -1) - A[1:-1, -1].reshape(-1, 1) @ points[-1].reshape(1, -1)

        conts = np.linalg.lstsq(A_BE, _points, rcond=None)

        control_points = np.zeros((n_control, points.shape[1]))
        control_points[0] = points[0]
        control_points[-1] = points[-1]
        control_points[1:-1] = conts[0]

        return control_points
    
    def bezier_Endpointfixed(self, sd_map, n_control=2):
        sd_map_graph = []
        for category, polylines in sd_map.items():
            for polyline in polylines:
                res = self.fit_bezier_Endpointfixed(polyline, n_control)
                coeffs = res.flatten()  # n_control * 2, 
                sd_map_graph.append(np.concatenate([coeffs, 
                                                    [self.category2id[category]]], 
                                                    axis=0))
        sd_map_graph = np.concatenate(sd_map_graph, axis=0)
        return sd_map_graph, None
    
    def points_all_with_type(self, sd_map):
        sd_map_graph = []
        num_pnt_per_segment = []
        for category, polylines in sd_map.items():
            # polyline: list npoints, 2
            # category: str
            if len(polylines) > 0:
                for segment in polylines:
                    num_pnt_per_segment.append(segment.shape[0])
                    # segment: npoints, 2
                    sd_map_graph.append(np.concatenate([segment.astype(np.float32), 
                                                        np.ones([segment.shape[0], 1], dtype=np.float32) * self.category2id[category]], 
                                                        axis=1))
        if len(num_pnt_per_segment) == 0:
            num_pnt_per_segment = np.empty([0], dtype=np.int64)
        else:
            num_pnt_per_segment = np.array(num_pnt_per_segment)  # num_polylines, 
        max_num_points = num_pnt_per_segment.max() if len(num_pnt_per_segment) > 0 else 0

        # list of segments: num_polylines, num_points x 3
        # pad at the end
        sd_map_graph = [np.pad(map_graph, ((0, max_num_points - map_graph.shape[0]), (0, 0)), 'constant') for map_graph in sd_map_graph]
        # list of segments: num_polylines, max_num_points x 3

        # then stack: num_polylines x max_num_points x 3
        sd_map_graph = np.stack(sd_map_graph, axis=0) if len(sd_map_graph) > 0 else np.zeros([0, max_num_points, 3], dtype=np.float32)
        map_meta = dict(map_num_poly_pnts=num_pnt_per_segment)
        return sd_map_graph, map_meta
    
    def points_onehot_type(self, sd_map):
        num_categories = max(self.category2id.values()) + 1

        sd_map_graph = []
        num_pnt_per_segment = []
        # TODO: polylines can potentially belong to 2 categories, need to re-map it back to the same polyline
        for category, polylines in sd_map.items():
            # polyline: list npoints, 2
            # category: str
            if len(polylines) > 0:
                for segment in polylines:
                    # segment: npoints, 2
                    num_pnt_per_segment.append(segment.shape[0])
                    
                    lane_category_onehot = np.zeros([num_categories], dtype=np.float32)
                    lane_category_onehot[self.category2id[category]] = 1.0
                    sd_map_graph.append(np.concatenate([segment.astype(np.float32),
                                                        lane_category_onehot.reshape(1, -1).repeat(segment.shape[0], axis=0)],
                                                        axis=1))
        if len(num_pnt_per_segment) == 0:
            num_pnt_per_segment = np.empty([0], dtype=np.int64)
        else:
            num_pnt_per_segment = np.array(num_pnt_per_segment)  # num_polylines, 
        max_num_points = num_pnt_per_segment.max() if len(num_pnt_per_segment) > 0 else 0

        # list of segments: num_polylines, num_points x (2 + num_categories)
        # pad at the end
        sd_map_graph = [np.pad(map_graph, ((0, max_num_points - map_graph.shape[0]), (0, 0)), 'constant') for map_graph in sd_map_graph]
        # list of segments: num_polylines, max_num_points x (2 + num_categories)

        # then stack: num_polylines x max_num_points x (2 + num_categories)
        sd_map_graph = np.stack(sd_map_graph, axis=0) if len(sd_map_graph) > 0 else np.zeros([0, max_num_points, num_categories + 2], dtype=np.float32)
        map_meta = dict(map_num_poly_pnts=num_pnt_per_segment)
        return sd_map_graph, map_meta
    
    @staticmethod
    def interpolate_line(line, n_points):
        # interpolates a shapely line to n_points
        distances = np.linspace(0, line.length, n_points)
        points = [line.interpolate(distance) for distance in distances]
        return np.stack([point.coords.xy for point in points]).squeeze(-1)

    
    def even_points_onehot_type(self, sd_map, n_points=11):
        num_categories = max(self.category2id.values()) + 1

        sd_map_graph = []
        onehot_category = []

        for category, polylines in sd_map.items():
            # polyline: list, npoints * 2
            # category: str
            if len(polylines) > 0:
                lane_category_onehot = np.zeros([num_categories], dtype=np.float32)
                lane_category_onehot[self.category2id[category]] = 1.0
                # onehot_category: num_polylines * num_categories
                onehot_category.append(lane_category_onehot.reshape(1, -1).repeat(len(polylines), axis=0))

                # interpolate the lines
                lines = GeoSeries(map(LineString, polylines))

                np_lines = [CustomParametrizeSDMapGraph.interpolate_line(line, n_points=n_points).astype(np.float32) \
                             for line in list(lines)]
                sd_map_graph.extend(np_lines)

        # then stack: num_polylines x n_points x 2
        sd_map_graph = np.stack(sd_map_graph, axis=0) if len(sd_map_graph) > 0 \
            else np.zeros([0, n_points, 2], dtype=np.float32)
        onehot_category = np.concatenate(onehot_category, axis=0) if len(onehot_category) > 0 \
            else np.zeros([0, num_categories], dtype=np.float32)
        map_meta = dict(onehot_category=onehot_category)
        return sd_map_graph, map_meta
    
    def even_points_by_type(self, sd_map, n_points=11, lane_types=[0, 1], include_category=True):
        # num_categories = max(self.category2id.values()) + 1
        num_categories = len(lane_types)
        lane_old_to_new = {old: new for new, old in enumerate(lane_types)}        

        sd_map_graph = []
        onehot_category = []

        for category, polylines in sd_map.items():
            # polyline: list, npoints * 2
            # category: str
            lanetype_num = self.category2id[category]

            if len(polylines) > 0 and (lanetype_num in lane_types):

                if include_category:
                    lane_category_onehot = np.zeros([num_categories], dtype=np.float32)
                    # lane_category_onehot[lanetype_num] = 1.0
                    lane_category_onehot[lane_old_to_new[lanetype_num]] = 1.0
                    # onehot_category: num_polylines * num_categories
                    onehot_category.append(lane_category_onehot.reshape(1, -1).repeat(len(polylines), axis=0))

                # interpolate the lines
                lines = GeoSeries(map(LineString, polylines))

                np_lines = [CustomParametrizeSDMapGraph.interpolate_line(line, n_points=n_points).astype(np.float32) \
                             for line in list(lines)]
                sd_map_graph.extend(np_lines)

        # then stack: num_polylines x n_points x 2
        sd_map_graph = np.stack(sd_map_graph, axis=0) if len(sd_map_graph) > 0 \
            else np.zeros([0, n_points, 2], dtype=np.float32)
        
        if include_category:
            onehot_category = np.concatenate(onehot_category, axis=0) if len(onehot_category) > 0 \
                else np.zeros([0, num_categories], dtype=np.float32)
        else:
            onehot_category = np.zeros([0, 0], dtype=np.float32)
        map_meta = dict(onehot_category=onehot_category)
        return sd_map_graph, map_meta
