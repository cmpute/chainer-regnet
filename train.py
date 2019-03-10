from model import RegNet
from utils.prepare_args import *
from utils.prepare_data import CalibPrepare
from kitti.kitti_object import Kitti3DObjectDetectionDataset as KITTI

from chainer.datasets import TransformDataset, split_dataset
from chainer.iterators import SerialIterator, MultiprocessIterator
from chainer.training import Trainer, StandardUpdater, ParallelUpdater
from chainer.training import extensions
from chainer.function_hooks import CupyMemoryProfileHook

DEBUG = False

def main():
    args = create_args('train')
    result_dir = create_result_dir(args.model_name)

    # Prepare devices
    devices = get_gpu_dict(args.gpus)
           
    # Instantiate a model
    model = RegNet(epsilon=args.epsilon)

    # Instantiate a optimizer
    optimizer = get_optimizer(model, **vars(args))
    
    # Setting up datasets
    prep = TransformDataset(KITTI(args.kitti_path, 'train'), CalibPrepare())
    train, valid = split_dataset(prep, round(len(prep)*0.5))
    print('train: {}, valid: {}'.format(len(train), len(valid)))

    # Iterator
    if DEBUG: Iterator = SerialIterator
    else: Iterator = MultiprocessIterator
    train_iter = Iterator(train, args.batchsize)
    valid_iter = Iterator(valid, args.valid_batchsize, repeat=False, shuffle=False)

    # Updater
    if DEBUG: Updater = StandardUpdater(train_iter, optimizer, device=0)
    else: Updater = ParallelUpdater(train_iter, optimizer, devices=devices)
    trainer = Trainer(Updater, (args.epoch, 'epoch'), out=result_dir)

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
    hook = CupyMemoryProfileHook()
    with hook: trainer.run()

    print("========== Saving ==========")
    chainer.serializers.save_hdf5(create_result_file(args.model_name), model)
    print("========== Memory Profiling ==========")
    hook.print_report()

if __name__ == '__main__':
    main()