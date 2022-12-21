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

import mindspore.nn as nn
from mindspore.train.loss_scale_manager import FixedLossScaleManager

import suwen.utils as su
from suwen.engine import Engine
from dice_loss import DiceLoss
from suwen.algorithm.nets.unet2d import UNet2D
from dice_metric import DiceMetric
from dataset import create_dataset
from cross_entropy_with_logits import CrossEntropyWithLogits

def train_and_eval(args):
    su.initial_context(su.GRAPH_MODE, device_id = args.device_id)
    net = UNet2D(n_channels = 1, n_classes = args.number_class)

    if args.loss_function.lower() is 'Dice_Loss':
        criterion = DiceLoss(args.number_class)
    else:
        criterion = CrossEntropyWithLogits(args.number_class)

    train_dataset, eval_dataset = create_dataset(args.data_path, train_batch_size = args.train_batch_size,
                                                 eval_batch_size = args.eval_batch_size)
    dataset_steps = train_dataset.get_dataset_size()
    new_epochs = dataset_steps * args.epochs // args.sink_steps
    print('train data steps', dataset_steps)
    print('train data epochs', new_epochs)

    optimizer = nn.Adam(params = net.trainable_params(), learning_rate = args.lr,
                        weight_decay = args.weight_decay, loss_scale = args.loss_scale)

    loss_scale_manager = FixedLossScaleManager(args.loss_scale, False)

    mine_train = Engine(net,
                        loss_fn = criterion,
                        optimizer = optimizer,
                        amp_level = "O3",
                        metrics = {'dice_loss': DiceMetric(args.number_class)},
                        loss_scale_manager = loss_scale_manager)

    save_checkpoint_config = {'max_num': args.keep_ckpt_max,
                              'prefix': 'unet2d',
                              "root": args.save_ckpt_path}

    mine_train.train_and_eval(new_epochs,
                              train_dataset = train_dataset,
                              eval_dataset = eval_dataset,
                              sink_size = dataset_steps,
                              save_checkpoints = save_checkpoint_config)

def get_args():
    parser = argparse.ArgumentParser(description = "Train Unet on images and target mask",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device_id', type = int, default = 0, help = 'device id')
    parser.add_argument('--data_path', type = str, default = "", help = 'data directory')
    parser.add_argument('--lr', type = float, default = 0.0001, help = 'learning rate, default: 0.0001')
    parser.add_argument('--epochs', type = int, default = 100, help = 'epoch number, default: 100')
    parser.add_argument('--train_batch_size', type = int, default = 16, help = 'train batch size, default: 16')
    parser.add_argument('--eval_batch_size', type = int, default = 8, help = 'eval batch size, default: 8')
    parser.add_argument('--sink_steps', type = int, default = 100, help = 'data sink steps, default: 100')
    parser.add_argument('--number_class', type = int, default = 2, help = 'classification number, default: 6')
    parser.add_argument('--keep_ckpt_max', type = int, default = 1,
                        help = 'max number of saving checkpoint, default: 5')
    parser.add_argument('--weight_decay', type = float, default = 0.0005, help = 'weight decay')
    parser.add_argument('--loss_scale', type = float, default = 1024.0, help = 'loss scale value, default: 1024.0')
    parser.add_argument('--loss_function', type = str, choices = ['CE', 'Dice_Loss'], default = 'CE')
    parser.add_argument('--save_ckpt_path', type = str, default = './ckpt', help = 'save checkpoint path')
    parser.add_argument('--load_ckpt_path', type = str, default = '', help = 'load checkpoint path')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    print(args)
    train_and_eval(args)
