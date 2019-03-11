from model import RegNet
from utils.prepare_args import *
from utils.prepare_data import CalibPrepare, h_quat2mat, h_mat2quat
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
    qinit = np.array(args.init_pose)
    H_init = h_quat2mat(qinit)
    prep = CalibPrepare(qinit, rot_disturb=0, trans_disturb=0)
    img, dimg, q_decalib_gt = prep((points, image, calib, None))

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
    q_decalib = cuda.to_cpu(model(img, dimg).array)[0]
    H_decalib = h_quat2mat(q_decalib)

    # Compare calibration result
    H_result = H_init.dot(H_decalib)
    print("Generated matrix")
    print(H_result[:3,:])
    print("Groundtruth matrix")
    print(calib['Tr_velo_to_cam'].reshape(3,4))

    q_result = h_mat2quat(H_result)
    q_gt = h_mat2quat(calib['Tr_velo_to_cam'].reshape(3,4))
    print("Generated quaternion", q_result)
    print("Groundtruth quaternion", q_gt)
    print("Quaternion loss", np.linalg.norm(q_result - q_gt))

    ## Compare decalibration
    # H_decalib_gt = h_quat2mat(q_decalib_gt)
    # print("Generated matrix")
    # print(H_decalib)
    # print("Groundtruth matrix")
    # print(H_decalib_gt)

    # print("Generated quaternion", q_decalib)
    # print("Groundtruth quaternion", q_decalib_gt)
    # print("Quaternion loss", np.linalg.norm(q_decalib - q_decalib_gt))


if __name__ == '__main__':
    main()
