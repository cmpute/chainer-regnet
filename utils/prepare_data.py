import numpy as np
import pyquaternion as pq

class CalibPrepare:
    def __init__(self, init_pose=None):
        if init_pose is not None and init_pose.shape != (4,4):
            raise ValueError("The initial H matrix should be 4x4")
        self.init_pose = init_pose

    def __call__(self, args):
        # Parse input
        lidar, image, calib, labels = args
        if self.init_pose is None:
            H_init = pq.Quaternion([1] + np.random.rand(3).tolist()).transformation_matrix # rotation
            H_init[:3, 3] = np.random.rand(3) # translation
        else:
            H_init = init_pose

        # Calculate decalibration coeffs
        H_gt = np.vstack((calib['Tr_velo_to_cam'].reshape(3,4), [0,0,0,1]))
        H_decalib = np.linalg.solve(H_init, H_gt)

        # Project point cloud to image
        xyz1 = np.insert(lidar[:,:3], 3, values=1, axis=1)
        rect = np.eye(4)
        rect[:3,:3] = calib['R0_rect'].reshape(3,3)
        proj = H_gt.dot(xyz1.T)
        zloc = proj[2,:]
        proj = calib['P2'].reshape(3,4).dot(rect).dot(proj)
        xloc = proj[0,:] / proj[2,:]
        yloc = proj[1,:] / proj[2,:]

        height, width = image.shape[1:]
        dimg = np.zeros((height, width, 2))
        mask = (xloc > 0) & (xloc < width) & (yloc > 0) & (yloc < height) & (zloc > 0)
        ymask, xmask = yloc[mask].astype(int), xloc[mask].astype(int)
        dimg[ymask, xmask, 0] = zloc[mask]
        dimg[ymask, xmask, 1] = lidar[mask,3]

        return image, dimg, H_decalib
