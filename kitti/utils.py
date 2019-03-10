import numpy as np
import chainer
from chainercv.utils import read_image
chainer.global_config.cv_read_image_backend = "PIL"

def load_kitti_points(path, intensity=True):
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return points if intensity else points[:, :3]

def load_kitti_object_labels(path):
    records = []
    with open(path, 'r') as fin:
        for line in fin:
            class_name, trunc, occlusion, alpha,\
                x1, y1, x2, y2,\
                h, w, l, x, y, z, yaw = line.strip().split(' ')
            records.append((class_name, float(trunc), int(occlusion), float(alpha),
                            float(x1), float(y1), float(x2), float(y2),
                            float(h), float(w), float(l),
                            float(x), float(y), float(z), float(yaw)))

    return records

def load_kitti_calib(path):
    float_chars = set("0123456789.e+- ")

    data = {}
    with open(path, 'r') as f:
        for line in [l for l in f.read().split('\n') if len(l) > 0]:
            key, value = line.split(' ', 1)
            if key.endswith(':'):
                key = key[:-1]
            value = value.strip()
            if float_chars.issuperset(value):
                data[key] = np.array([float(v) for v in value.split(' ')])
            else:
                print('warning: unknown value!')

    return data
    
def load_kitti_image(path):
    # image = Image.open(path)
    # image.load()
    # return np.asarray(image, dtype="i4")
    return read_image(path)
