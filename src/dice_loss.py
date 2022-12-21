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

from mindspore import nn
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P

from suwen.data import swtype, Tensor
from suwen.losses import DiceLoss as DiceHelper

class DiceLoss(Cell):
    r"""
    The Dice coefficient loss is a set similarity loss. It is used to calculate the similarity between two samples. The
    value of the Dice coefficient is 1 when the segmentation result is the best and 0 when the segmentation result
    is the worst. The Dice coefficient indicates the ratio of the area between two objects to the total area.
    The function is shown as follows:

    .. math::
        dice = 1 - \frac{2 * (pred \bigcap true)}{pred \bigcup true}

    Args:
        num_classes: Number of label classes.
        sparse: Whether one-hot the label, bool type.
    Inputs:
        - **y_pred** (Tensor) - Tensor of shape (N, ...). The data type must be float16 or float32.
        - **y** (Tensor) - Tensor of shape (N, ...). The data type must be float16 or float32.

    Outputs:
        Tensor, a tensor of shape with the per-example sampled Dice losses.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> loss = nn.DiceLoss(num_classes=2, sparse = True)
        >>> y_pred = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mstype.float32)
        >>> y = Tensor(np.array([[0, 1], [1, 0], [0, 1]]), mstype.float32)
        >>> output = loss(y_pred, y)
        >>> print(output)
        [0.7953220862819745]

    Raises:
        ValueError: If the dimensions are different.
        TypeError: If the type of inputs are not Tensor.
    """

    def __init__(self, num_classes=2, sparse=True):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.softmax = nn.Softmax(axis = 1)
        self.dice_loss = DiceHelper()
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, swtype.float32)
        self.off_value = Tensor(0.0, swtype.float32)
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.sparse = sparse

    def construct(self, logits, label):

        one_hot_label = None
        N, C, H, W = logits.shape
        logits = self.reshape(logits, (N, C, -1))
        out = self.softmax(logits)
        logits = self.reshape(out, (N, C, H, W))

        pred = self.transpose(logits, (0, 2, 3, 1))
        label = self.cast(label, swtype.int32)
        label = self.transpose(label, (0, 2, 3, 1))
        if self.sparse:
            one_hot_label = self.one_hot(label, self.num_classes, self.on_value, self.off_value)
            one_hot_label = self.reshape(one_hot_label, (N, H, W, C))
        else:
            one_hot_label = label
        dice_loss = self.dice_loss(pred, one_hot_label)

        return dice_loss
