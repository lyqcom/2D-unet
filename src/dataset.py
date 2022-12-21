# Copyright 2021 Pengcheng Laboratory
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
# ============================================================================

import os
from glob import glob

import suwen.data.dataset as ds
from suwen.transforms.vision import AddChannel, ScaleIntensity, RandRotate90, RandGaussianNoise, LoadImage

class Dataset:
    """
    A generic dataset with a length property and an optional callable data transform
    when fetching a data sample.
    For example, typical input data can be a list of dictionaries::

        [{                            {                            {
             'img': 'image1.nii.gz',      'img': 'image2.nii.gz',      'img': 'image3.nii.gz',
             'seg': 'label1.nii.gz',      'seg': 'label2.nii.gz',      'seg': 'label3.nii.gz',
             'extra': 123                 'extra': 456                 'extra': 789
         },                           },                           }]
    """

    def __init__(self, data, column_names):
        """
        Args:
            data: input data to load and transform to generate dataset for model.
        """
        self.data = data
        self.column_names = column_names

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        res = []
        for col in self.column_names:
            res.append(self.data[index][col])
        return tuple(res)

def create_dataset(data_dir_path='', train_batch_size=16, eval_batch_size=8):
    val_ratio = 0.2
    dataset_path = data_dir_path
    images = sorted(glob(os.path.join(dataset_path, 'images', '*.png')))
    segs = sorted(glob(os.path.join(dataset_path, 'masks', '*.png')))
    num_val = int(val_ratio * len(images))
    train_files = [{"image": img, "seg": seg} for img, seg in zip(images[:-num_val], segs[:-num_val])]
    val_files = [{"image": img, "seg": seg} for img, seg in zip(images[-num_val:], segs[-num_val:])]

    train_transforms_img = [
        LoadImage(),
        AddChannel(),
        ScaleIntensity(),
        RandRotate90(prob = 0.5, spatial_axes = [0, 1]),
        RandGaussianNoise(prob = 0.5, std = 0.02),
    ]

    val_transforms_img = [
        LoadImage(),
        AddChannel(),
        ScaleIntensity(),
    ]

    train_ds = Dataset(data = train_files, column_names = ["image", "seg"])
    train_loader = ds.GeneratorDataset(train_ds, column_names = ["image", "seg"], num_parallel_workers = 1,
                                       python_multiprocessing = False)
    train_loader = train_loader.map(operations = train_transforms_img, input_columns = ["image", "seg"],
                                    num_parallel_workers = 12, python_multiprocessing = True)
    train_loader = train_loader.batch(train_batch_size, drop_remainder = True)

    val_ds = Dataset(data = val_files, column_names = ["image", "seg"])
    val_loader = ds.GeneratorDataset(val_ds, column_names = ["image", "seg"],
                                     num_parallel_workers = 1, python_multiprocessing = False)
    val_loader = val_loader.map(operations = val_transforms_img, input_columns = ["image", "seg"],
                                num_parallel_workers = 12, python_multiprocessing = True)
    val_loader = val_loader.batch(eval_batch_size, drop_remainder = True)

    return train_loader, val_loader
