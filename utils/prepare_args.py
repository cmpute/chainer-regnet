import argparse
import os, sys, time

import chainer
from chainer import optimizers

def create_args(phase='train'):
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group('Dataset settings')
    if phase == 'train':
        group = parser.add_argument_group('Train settings')\
                if phase == 'train' else argparse.ArgumentParser()
        group.add_argument(
            '-r', '--resume_from',
            help='Resume the training from snapshot')
        group.add_argument(
            '--epoch', type=int, default=100,
            help='When the trianing will finish')
        group.add_argument(
            '--batchsize', type=int, default=4,
            help='minibatch size')
        group.add_argument(
            '--snapshot_iter', type=int, default=1000,
            help='The current learnt parameters in the model is saved every'
                'this iteration')
        group.add_argument(
            '--valid_freq', type=int, default=1,
            help='Perform test every this epoch (0 means no test)')
        group.add_argument(
            '--valid_batchsize', type=int, default=1,
            help='The mini-batch size during validation loop')
        group.add_argument(
            '--show_log_iter', type=int, default=10,
            help='Show loss value per this iterations')
        group.add_argument(
            '--epsilon', type=float, default=1,
            help='The weight of location loss')
        group.add_argument(
            '--gpus', type=str, default='0',
            help='GPU Ids to be used')
        group.add_argument(
            '--kitti_path', type=str,
            help='The path to the root of KITTI (object) dataset')
        group.add_argument(
            '--valid_proportion', type=float, default=0.1,
            help='The proportion of valid data in the whole dataset'
        )
    elif phase == 'test':
        group.add_argument(
            'net_weight_path', type=str,
            help='The path to net model (in HDF5 format)')
        group.add_argument(
            '--iter', type=int, default=1,
            help='How many times do recalibrate')
        group.add_argument(
            '--input_pointcloud', type=str,
            help='The path to input point cloud')
        group.add_argument(
            '--input_image', type=str,
            help='The path to input image file')
        group.add_argument(
            '--input_calibration', type=str,
            help='The path to input calibration file')
    
    group = parser.add_argument_group('Model parameters')
    group.add_argument(
        '--model_name', type=str, default='RegNet',
        help='The model type name')
    group.add_argument(
        '--init_pose', type=float, default=[1,-1,1,0,0,0], nargs='*',
        help='Initial calibration quaternion')

    group = parser.add_argument_group('Optimization settings')\
            if phase == 'train' else argparse.ArgumentParser()
    group.add_argument(
        '--opt', type=str, default='Adam',
        choices=['MomentumSGD', 'Adam', 'AdaGrad', 'RMSprop'],
        help='Optimization method')
    group.add_argument('--lr', type=float, default=0.01)
    group.add_argument('--weight_decay', type=float, default=0.0005)
    group.add_argument('--adam_alpha', type=float, default=0.001)
    group.add_argument('--adam_beta1', type=float, default=0.9)
    group.add_argument('--adam_beta2', type=float, default=0.999)
    group.add_argument('--adam_eps', type=float, default=1e-8)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    return args

def create_result_dir(model_name):
    result_dir = 'results/{}_{}'.format(
        model_name, time.strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(result_dir):
        result_dir += '_{}'.format(time.clock())
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir

def create_result_file(model_name):
    result_dir = 'results/{}_{}.h5'.format(
        model_name, time.strftime('%Y-%m-%d_%H-%M-%S'))
    return result_dir

def get_optimizer(model, opt, lr, adam_alpha, adam_beta1,
                  adam_beta2, adam_eps, weight_decay, **discard):
    if opt == 'MomentumSGD':
        optimizer = optimizers.MomentumSGD(lr=lr, momentum=0.9)
    elif opt == 'Adam':
        optimizer = optimizers.Adam(
            alpha=adam_alpha, beta1=adam_beta1,
            beta2=adam_beta2, eps=adam_eps)
    elif opt == 'AdaGrad':
        optimizer = optimizers.AdaGrad(lr=lr)
    elif opt == 'RMSprop':
        optimizer = optimizers.RMSprop(lr=lr)
    else:
        raise Exception('No optimizer is selected')

    # The first model as the master model
    optimizer.setup(model)
    if opt == 'MomentumSGD':
        optimizer.add_hook(
            chainer.optimizer.WeightDecay(weight_decay))

    return optimizer

def get_gpu_dict(gpu_string):
    devices = {}
    for gid in [int(i) for i in gpu_string]:
        if 'main' not in devices:
            devices['main'] = gid
        else:
            devices['gpu{}'.format(gid)] = gid
    return devices
