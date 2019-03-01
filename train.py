from model import RegNet
from utils.prepare_args import *
from utils.prepare_data import CalibPrepare
from kitti.kitti_object import Kitti3DObjectDetectionDataset as KITTI

from chainer.datasets import TransformDataset, split_dataset
from chainer.iterators import MultiprocessIterator
from chainer.training import Trainer, ParallelUpdater
from chainer.training import extensions

def main():
    args = create_args('train')
    result_dir = create_result_dir(args.model_name)

    # Prepare devices
    devices = {}
    for gid in [int(i) for i in args.gpus.split(',')]:
        if 'main' not in devices:
            devices['main'] = gid
        else:
            devices['gpu{}'.format(gid)] = gid
           
    # Instantiate a model
    model = RegNet()

    # Instantiate a optimizer
    optimizer = get_optimizer(model, **vars(args))
    
    # Setting up datasets
    prep = TransformDataset(KITTI(args.kitti_path, 'train'), CalibPrepare())
    train, valid = split_dataset(prep, round(len(prep)*0.8))
    print('train: {}, valid: {}'.format(len(train), len(valid)))

    # Iterator
    train_iter = MultiprocessIterator(train, args.batchsize)
    valid_iter = MultiprocessIterator(valid, args.valid_batchsize, repeat=False, shuffle=False)

    # Updater
    updater = ParallelUpdater(train_iter, optimizer, devices=devices)
    trainer = Trainer(updater, (args.epoch, 'epoch'), out=result_dir)

    # Extentions
    trainer.extend(extensions.Evaluator(valid_iter, model, device=devices['main']),
        trigger=(args.valid_freq, 'epoch'))
    trainer.extend(extensions.snapshot(),
        trigger=(args.snapshot_iter, 'iteration'))
    trainer.extend(extensions.LogReport(),
        trigger=(args.show_log_iter, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=20))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss']))

    # Resume from snapshot
    if args.resume_from:
        chainer.serializers.load_npz(args.resume_from, trainer)

    # Train and save
    trainer.run()
    model.save(create_result_file(args.model_name))

if __name__ == '__main__':
    main()