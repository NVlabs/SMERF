# ==============================================================================
# Binaries and/or source for the following packages or projects 
# are presented under one or more of the following open source licenses:
# loading.py    The OpenLane-V2 Dataset Authors    Apache License, Version 2.0
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
import os
import numpy as np
import sparse

import mmcv
from mmdet.datasets import PIPELINES
from mmdet3d.datasets.pipelines import LoadMultiViewImageFromFiles


@PIPELINES.register_module()
class CustomLoadMultiViewImageFromFiles(LoadMultiViewImageFromFiles):

    def __call__(self, results):
        filename = results['img_paths']
        img = [mmcv.imread(name, self.color_type) for name in filename]
        if self.to_float32:
            img = [i.astype(np.float32) for i in img]
        results['img'] = img
        results['img_shape'] = [i.shape for i in results['img']]
        return results

@PIPELINES.register_module()
class CustomLoadSDMapRasterFromFiles(LoadMultiViewImageFromFiles):

    def load_map_raster(self, filename):
        file_ext = os.path.splitext(filename)[-1]
        if file_ext == '.png':
            map_raster = mmcv.imread(filename, self.color_type)
        elif file_ext == '.npy':
            map_raster = np.load(filename)
        elif file_ext == '.npz':
            map_raster_s_ = sparse.load_npz(filename)
            map_raster = map_raster_s_.todense()
        else:
            raise NotImplementedError()
        return map_raster

    def __call__(self, results):
        filename = results['sd_map_path']
        if os.path.exists(filename):
            # map_raster = mmcv.imread(filename, self.color_type)
            map_raster = self.load_map_raster(filename)
            if self.to_float32:
                map_raster = map_raster.astype(np.float32)
            
            # TODO: remove!
            # if the channel is incorrect, pad
            if map_raster.shape[-1] <= 8:
                map_raster = np.pad(map_raster, ((0, 0), (0, 0), (0, 8 - map_raster.shape[-1])), mode='constant')

            results['map_raster'] = map_raster
            results['map_raster_shape'] = map_raster.shape
        else:
            raise FileNotFoundError("Cannot find file:", filename)
            results['map_raster'] = np.ones((800, 400, 9), dtype=np.float32) * 0.
            results['map_raster_shape'] = results['map_raster']
        return results
