from model import RegNet
from utils.prepare_args import *
from utils.prepare_data import CalibPrepare, h_quat2mat, trace_method
from kitti.utils import *

import numpy as np
from chainer import global_config
from chainer.serializers import load_hdf5
import chainer.cuda as cuda

def main():
    args = create_args('test')
    global_config.train = False

    points = load_kitti_points(args.input_pointcloud)
    image = load_kitti_image(args.input_image)
    calib = load_kitti_calib(args.input_calibration)
    model = RegNet(epsilon=args.epsilon)
    load_hdf5(args.net_weight_path, model)

    # Prepare input
    qinit = np.array([1,-1,1,0,0,0])
    Hinit = h_quat2mat(qinit)
    prep = CalibPrepare(qinit)
    img, dimg = prep((points, image, calib))

    # Prepare devices
    devices = get_gpu_dict(args.gpus)
    if cuda.available and args.gpus:
        gpu = devices['main']
        model.to_gpu(gpu)
        img = cuda.to_gpu(img, gpu)
        dimg = cuda.to_gpu(dimg, gpu)

    xp = cuda.get_array_module(img)
    img = xp.expand_dims(img, 0)
    dimg = xp.expand_dims(dimg, 0)
    H_decalib = h_quat2mat(cuda.to_cpu(model(img, dimg).array)[0])
    H_result = Hinit.dot(H_decalib)

    print("Generated result")
    print(H_result[:3,:])
    print("Groundtruth result")
    print(calib['Tr_velo_to_cam'].reshape(3,4))

if __name__ == '__main__':
    main()
