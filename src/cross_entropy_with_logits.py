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

import mindspore.nn as nn
import mindspore.ops.operations as F
from mindspore.ops import operations as P

from suwen.losses import Loss
from suwen.data import Tensor, swtype

class CrossEntropyWithLogits(Loss):
    """
    Cross Entropy loss for unet
    """

    def __init__(self, num_classes):
        super(CrossEntropyWithLogits, self).__init__()
        self.transpose_fn = F.Transpose()
        self.reshape_fn = F.Reshape()
        self.softmax_cross_entropy_loss = nn.SoftmaxCrossEntropyWithLogits()
        self.cast = F.Cast()
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, swtype.float32)
        self.off_value = Tensor(0.0, swtype.float32)
        self.transpose = F.Transpose()
        self.reshape = F.Reshape()
        self.cast = F.Cast()
        self.num_classes = num_classes

    def construct(self, logits, label):
        N, C, H, W = logits.shape
        logits = self.transpose_fn(logits, (0, 2, 3, 1))
        logits = self.cast(logits, swtype.float32)
        label = self.transpose_fn(label, (0, 2, 3, 1))
        label = self.cast(label, swtype.int32)
        one_hot_label = self.one_hot(label, self.num_classes, self.on_value, self.off_value)
        one_hot_label = self.reshape(one_hot_label, (N, H, W, C))

        loss = self.reduce_mean(
            self.softmax_cross_entropy_loss(self.reshape_fn(logits, (-1, self.num_classes)),
                                            self.reshape_fn(one_hot_label, (-1, self.num_classes))))
        return loss
