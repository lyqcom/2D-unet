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

import argparse

import suwen.utils as su
from cross_entropy_with_logits import CrossEntropyWithLogits
from dataset import create_dataset
from dice_loss import DiceLoss
from dice_metric import DiceMetric
from suwen.algorithm.nets.unet2d import UNet2D
from suwen.engine import Engine

def eval(args):
    su.initial_context(su.GRAPH_MODE, device_id = args.device_id)
    net = UNet2D(n_channels = 1, n_classes = args.number_class)

    if args.loss_function.lower() is 'Dice_Loss':
        criterion = DiceLoss(args.number_class)
    else:
        criterion = CrossEntropyWithLogits(args.number_class)

    _, eval_dataset = create_dataset(args.data_path, train_batch_size = args.train_batch_size,
                                     eval_batch_size = args.eval_batch_size)
    dataset_steps = eval_dataset.get_dataset_size()
    print('eval data steps', dataset_steps)

    mine_train = Engine(net,
                        loss_fn = criterion,
                        amp_level = "O0",
                        metrics = {'dice_loss': DiceMetric(args.number_class)})

    mine_train.eval(valid_dataset = eval_dataset, load_ckpt_path = args.ckpt_path)

def get_args():
    parser = argparse.ArgumentParser(description = "eval Unet on images and target mask",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device_id', type = int, default = 0, help = 'device id')
    parser.add_argument('--data_path', type = str, default = "", help = 'data directory')
    parser.add_argument('--train_batch_size', type = int, default = 16, help = 'train batch size, default: 8')
    parser.add_argument('--eval_batch_size', type = int, default = 8, help = 'eval batch size, default: 8')
    parser.add_argument('--number_class', type = int, default = 2, help = 'classification number, default: 6')
    parser.add_argument('--loss_function', type = str, choices = ['CE', 'Dice_Loss'], default = 'CE')
    parser.add_argument('--ckpt_path', type = str, default = '', help = 'load checkpoint path')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    print(args)
    eval(args)
