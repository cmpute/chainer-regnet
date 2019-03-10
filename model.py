from chainer import Chain
from chainer import initializers, reporter
from chainercv.links.model.resnet.resblock import ResBlock
from chainercv.links.connection import Conv2DActiv
import chainer.links as L
import chainer.functions as F

class RgbFeature(Chain):
    def __init__(self):
        super(RgbFeature, self).__init__()
        kwargs = {'initialW': initializers.HeNormal(scale=1., fan_option='fan_out'),
                  'ksize': 3}
        with self.init_scope():
            self.pool = lambda x: F.max_pooling_2d(x, ksize=3, stride=2)
            self.nin1 = Conv2DActiv(None, 96, **kwargs)
            self.nin2 = Conv2DActiv(None, 256, **kwargs)
            self.nin3 = Conv2DActiv(None, 384, **kwargs)

    def __call__(self, x):
        x = self.pool(self.nin1(x))
        x = self.pool(self.nin2(x))
        x = self.pool(self.nin3(x))
        return x

class DepthFeature(Chain):
    def __init__(self):
        super(DepthFeature, self).__init__()
        kwargs = {'initialW': initializers.HeNormal(scale=1., fan_option='fan_out'),
                  'ksize': 3}
        with self.init_scope():
            self.pool = lambda x: F.max_pooling_2d(x, ksize=3, stride=2)
            self.nin1 = Conv2DActiv(None, 48, **kwargs)
            self.nin2 = Conv2DActiv(None, 128, **kwargs)
            self.nin3 = Conv2DActiv(None, 192, **kwargs)
            
    def __call__(self, x):
        x = self.pool(self.nin1(x))
        x = self.pool(self.nin2(x))
        x = self.pool(self.nin3(x))
        return x

class Matching(Chain):
    def __init__(self):
        super(Matching, self).__init__()
        with self.init_scope():
            self.pool = lambda x: F.average(x, axis=(2, 3))
            self.nin1 = Conv2DActiv(None, 512, 3)
            self.nin2 = Conv2DActiv(None, 512, 3)
            self.fc1 = L.Linear(None, 256)
            self.fc2 = L.Linear(None, 6)

    def __call__(self, x):
        x = self.nin1(x)
        x = self.nin2(x)
        x = self.pool(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class RegNet(Chain):
    def __init__(self, epsilon):
        super(RegNet, self).__init__()
        self.epsilon = epsilon
        with self.init_scope():
            self.feat_rgb = RgbFeature()
            self.feat_depth = DepthFeature()
            self.match = Matching()

    def __call__(self, rgb, depth, gt_decalib=None):
        feat = F.concat((self.feat_rgb(rgb), self.feat_depth(depth)))
        feat = self.match(feat)

        if gt_decalib is not None:
            rot_loss = F.mean_squared_error(feat[:, :3], gt_decalib[:, :3])
            loc_loss = F.mean_squared_error(feat[:, 3:], gt_decalib[:, 3:])
            loss = rot_loss + self.epsilon * loc_loss
            reporter.report({'loss': loss}, self)
            return loss
        else:
            return feat
