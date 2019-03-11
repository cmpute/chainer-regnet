from model import RegNet
from utils.prepare_args import *
from utils.prepare_data import CalibPrepare, h_quat2mat, h_mat2quat
from kitti.utils import *

import time
import numpy as np
from chainer import global_config
from chainer.serializers import load_hdf5
import chainer.cuda as cuda

import matplotlib.pyplot as plt

def main():
    args = create_args('test')
    global_config.train = False
    t_start = time.time()

    points = load_kitti_points(args.input_pointcloud)
    image = load_kitti_image(args.input_image)
    calib = load_kitti_calib(args.input_calibration)
    model = RegNet(epsilon=1)
    load_hdf5(args.net_weight_path, model)
    if cuda.available:
        model.to_gpu(0)
    qinit = np.array(args.init_pose)

    # Prepare input
    for i in range(args.iter):
        print("========== Iteration %d ==========" % i)
        H_init = h_quat2mat(qinit)
        prep = CalibPrepare(qinit, rot_disturb=0, trans_disturb=0)
        img, dimg, q_decalib_gt = prep((points, image, calib, None))
        if cuda.available:
            img = cuda.to_gpu(img, 0)
            dimg = cuda.to_gpu(dimg, 0)

        xp = cuda.get_array_module(img)
        img = xp.expand_dims(img, 0)
        dimg = xp.expand_dims(dimg, 0)
        q_decalib = cuda.to_cpu(model(img, dimg).array)[0]
        H_decalib = h_quat2mat(q_decalib)
        H_result = H_init.dot(H_decalib)
        q_result = h_mat2quat(H_result)
        qinit = q_result

        # Compare calibration result
        t_span = time.time() - t_start
        np.set_printoptions(precision=4, suppress=True)

        H_gt = np.eye(4)
        H_gt[:3,:] = calib['Tr_velo_to_cam'].reshape(3,4)
        print("Generated matrix")
        print(H_result[:3,:])
        print("Groundtruth matrix")
        print(H_gt[:3,:])

        q_gt = h_mat2quat(H_gt)
        print("Generated quaternion", q_result)
        print("Groundtruth quaternion", q_gt)
        print("Quaternion loss", np.linalg.norm(q_result - q_gt))
        print("Execution time", t_span)

    # Make projection plots
    intrinsic = np.eye(4)
    intrinsic[:3,:3] = calib['R0_rect'].reshape(3,3)
    intrinsic = calib['P2'].reshape(3,4).dot(intrinsic)

    _, height, width = image.shape
    pimage = image.transpose(1,2,0).astype(int)
    plt.figure()
    plt.subplot(2,1,1)
    plt.title("Regnet calibration")
    x,y,z = project_lidar_to_image(points, H_result, intrinsic)
    mask = (x > 0) & (x < width) & (y > 0) & (y < height) & (z > 0)
    plt.imshow(pimage)
    plt.scatter(x[mask], y[mask], c=z[mask], s=1, cmap='gist_ncar')
    
    plt.subplot(2,1,2)
    plt.title("Kitti calibration")
    x,y,z = project_lidar_to_image(points, H_gt, intrinsic)
    mask = (x > 0) & (x < width) & (y > 0) & (y < height) & (z > 0)
    plt.imshow(pimage)
    plt.scatter(x[mask], y[mask], c=z[mask], s=1, cmap='gist_ncar')

    plt.show()

if __name__ == '__main__':
    main()
