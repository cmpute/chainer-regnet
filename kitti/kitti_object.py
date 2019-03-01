
import os
import os.path as osp
from chainer.dataset import dataset_mixin
from .utils import *

class Kitti3DObjectDetectionDataset(dataset_mixin.DatasetMixin):
    '''
    Should be used along with TransformDataset
    transform input: point_cloud (ndarray), labels (list of tuple), calibration (dict of ndarray)
    '''
    def __init__(self, root_path, split='train', train_proportion=0.9, valid_proportion=0.1):
        if split is 'test': mid = 'testing'
        elif split in 'train': mid = 'training'
        else: raise ValueError('Invalid dataset split')
        self.is_test = split is 'test'

        point_path = osp.join(root_path, mid, 'velodyne')
        image_path = osp.join(root_path, mid, 'image_2')
        label_path = osp.join(root_path, mid, 'label_2')
        calib_path = osp.join(root_path, mid, 'calib')

        self.data = []
        for pfile in filter(lambda fname: fname.endswith('.bin'), os.listdir(point_path)):
            pname = pfile[:-4]
            pfile = osp.join(point_path, pfile)
            ifile = osp.join(image_path, pname + '.png')
            cfile = osp.join(calib_path, pname + '.txt')
            lfile = osp.join(label_path, pname + '.txt')
            if self.is_test:
                self.data.append((pfile, ifile, cfile, pname))
            else:
                self.data.append((pfile, ifile, cfile, lfile))

    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        pfile, ifile, cfile, lfile = self.data[i]
        if self.is_test:
            return load_kitti_points(pfile), load_kitti_image(ifile),\
                load_kitti_calib(cfile)
        else:
            return load_kitti_points(pfile), load_kitti_image(ifile),\
                load_kitti_calib(cfile), load_kitti_object_labels(lfile)
