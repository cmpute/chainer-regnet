import numpy as np
import scipy.linalg as spl
from scipy.spatial.transform import Rotation

def h_mat2quat(h):
    q = Rotation.from_dcm(h[:3,:3]).as_quat()
    return np.concatenate((q[:3]/q[3], h[:3,3]))
    
def h_quat2mat(h):
    q = np.concatenate((h[:3], [1]))
    Hmat = np.eye(4)
    Hmat[:3,:3] = Rotation.from_quat(q).as_dcm() # rotation
    Hmat[:3, 3] = h[3:] # translation
    return Hmat

class CalibPrepare:
    def __init__(self, init_pose, rot_disturb=0.2, trans_disturb=0.2):
        if init_pose is not None:
            init_pose = np.asarray(init_pose)
            if init_pose.shape != (6,):
                raise ValueError("The initial H vector should have exactly 6 elements")
        elif init_pose is None:
            raise ValueError("Please specify initial pose")
        self.init_pose = init_pose
        self._drot = rot_disturb
        self._dtrans = trans_disturb

    def __call__(self, args):
        # Parse input
        if len(args) == 4:
            lidar, image, calib, labels = args
            do_train = True
        elif len(args) == 3:
            lidar, image, calib = args
            do_train = False
        else: raise ValueError("Unrecognized input")

        q_rand = np.random.rand(6)
        q_rand[:3] = (q_rand[:3] - 0.5) * self._drot
        q_rand[3:] = (q_rand[3:] - 0.5) * self._dtrans
        H_init = h_quat2mat(q_rand + self.init_pose)

        # Project point cloud to image
        xyz1 = np.insert(lidar[:,:3], 3, values=1, axis=1)
        rect = np.eye(4)
        rect[:3,:3] = calib['R0_rect'].reshape(3,3)
        proj = H_init.dot(xyz1.T)
        zloc = proj[2,:]
        proj = calib['P2'].reshape(3,4).dot(rect).dot(proj)
        xloc = proj[0,:] / proj[2,:]
        yloc = proj[1,:] / proj[2,:]

        height, width = image.shape[1:]
        dimg = np.zeros((2, height, width))
        mask = (xloc > 0) & (xloc < width) & (yloc > 0) & (yloc < height) & (zloc > 0)
        ymask, xmask = yloc[mask].astype(int), xloc[mask].astype(int)
        dimg[0, ymask, xmask] = zloc[mask]
        dimg[1, ymask, xmask] = lidar[mask,3]

        # Crop image to same size
        cwidth, cheight = 1220, 370
        image = image[:, :cheight, :cwidth].astype('f4')
        dimg = dimg[:, :cheight, :cwidth].astype('f4')

        if do_train:
            # Calculate decalibration coeffs
            H_gt = np.vstack((calib['Tr_velo_to_cam'].reshape(3,4), [0,0,0,1]))
            decalib = h_mat2quat(np.linalg.solve(H_init, H_gt)).astype('f4')
            return image, dimg, decalib
        else:
            return image, dimg
