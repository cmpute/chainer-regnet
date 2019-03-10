import numpy as np
import scipy.linalg as spl
import pyquaternion as pq

def h_quat2mat(h):
    Hmat = pq.Quaternion([1, h[0], h[1], h[2]]).transformation_matrix # rotation
    Hmat[0, 3] = h[3] # translation
    Hmat[1, 3] = h[4]
    Hmat[2, 3] = h[5]
    return Hmat

def trace_method(matrix):
    """
    from pyquaternion... the precision cannot be achieved to make R orthogonal
    """
    m = matrix.conj().transpose() # This method assumes row-vector and postmultiplication of that vector
    if m[2, 2] < 0:
        if m[0, 0] > m[1, 1]:
            t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
            q = [m[1, 2]-m[2, 1],  t,  m[0, 1]+m[1, 0],  m[2, 0]+m[0, 2]]
        else:
            t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
            q = [m[2, 0]-m[0, 2],  m[0, 1]+m[1, 0],  t,  m[1, 2]+m[2, 1]]
    else:
        if m[0, 0] < -m[1, 1]:
            t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
            q = [m[0, 1]-m[1, 0],  m[2, 0]+m[0, 2],  m[1, 2]+m[2, 1],  t]
        else:
            t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
            q = [t,  m[1, 2]-m[2, 1],  m[2, 0]-m[0, 2],  m[0, 1]-m[1, 0]]

    q = np.array(q)
    # q *= 0.5 / np.sqrt(t)
    q /= q[0] # make w = 1
    return q

class CalibPrepare:
    # FIXME: implement randomize option and init_pose specified by argument
    def __init__(self, init_pose=None, randomize=True):
        if init_pose is not None and init_pose.shape != (6,):
            raise ValueError("The initial H vector should have exactly 6 elements")
        self.init_pose = init_pose

    def __call__(self, args):
        # Parse input
        if len(args) == 4:
            lidar, image, calib, labels = args
            do_train = True
        elif len(args) == 3:
            lidar, image, calib = args
            do_train = False
        else: raise ValueError("Unrecognized input")

        if self.init_pose is None:
            h_rand = np.random.rand(6)
            H_init = h_quat2mat(h_rand + [1,-1,1,0,0,0])
        else:
            H_init = h_quat2mat(self.init_pose)

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
            H_decalib = np.linalg.solve(H_init, H_gt)
            ########## FIXME: precision error ##########
            # rot = pq.Quaternion(matrix=H_decalib[:3,:3])
            # rot = [rot.x, rot.y, rot.z]
            #####################################
            rot = trace_method(H_decalib[:3,:3])
            decalib = np.concatenate((rot[1:], H_decalib[:3,3])).astype('f4')
            return image, dimg, decalib
        else:
            return image, dimg
