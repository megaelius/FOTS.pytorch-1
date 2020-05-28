import torch.nn as nn
from ..functions.rroi_align import RRoiAlignFunction


class _RRoiAlign(nn.Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(_RRoiAlign, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RRoiAlignFunction.apply(
            self.pooled_height, self.pooled_width, self.spatial_scale,
            features, rois
        )
