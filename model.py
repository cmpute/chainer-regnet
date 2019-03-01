from chainer import Chain
from chainer import initializers, reporter
from chainercv.links.model.resnet.resblock import ResBlock
import chainer.links as L
import chainer.functions as F

class RgbFeature(Chain):
    def __init__(self):
        super(RgbFeature, self).__init__()
        initialW = initializers.HeNormal(scale=1., fan_option='fan_out')
        kwargs = {'initialW': initialW}
        with self.init_scope():
            self.nin1 = ResBlock(3, None, 96, 96, 1, **kwargs)
            self.nin2 = ResBlock(3, None, 96, 256, 1, **kwargs)
            self.nin3 = ResBlock(3, None, 256, 384, 1, **kwargs)

    def forward(self, x):
        x = self.nin1(x)
        x = self.nin2(x)
        x = self.nin3(x)
        return x

class DepthFeature(Chain):
    def __init__(self):
        super(DepthFeature, self).__init__()
        initialW = initializers.HeNormal(scale=1., fan_option='fan_out')
        kwargs = {'initialW': initialW}
        with self.init_scope():
            self.nin1 = ResBlock(3, None, 48, 48, 1, **kwargs)
            self.nin2 = ResBlock(3, None, 48, 128, 1, **kwargs)
            self.nin3 = ResBlock(3, None, 128, 192, 1, **kwargs)
            
    def forward(self, x):
        x = self.nin1(x)
        x = self.nin2(x)
        x = self.nin3(x)
        return x

class Matching(Chain):
    def __init__(self):
        super(Matching, self).__init__()
        with self.init_scope():
            self.nin1 = ResBlock(3, None, 512, 512, 1)
            self.nin2 = ResBlock(3, None, 512, 512, 1)
            self.fc1 = L.Linear(None, 256)
            self.fc2 = L.Linear(None, 6)

class RegNet(Chain):
    def __init__(self):
        super(RegNet, self).__init__()
        with self.init_scope():
            self.feat_rgb = RgbFeature()
            self.feat_depth = DepthFeature()
            self.match = Matching()

    def forward(self, rgb, depth, gt_decalib):
        feat = F.concat(self.feat_rgb(rgb), self.feat_depth(depth))
        feat = self.math(feat)
        loss = F.mean_squared_error(feat)
        reporter.report({'loss': loss}, self)
        return loss
