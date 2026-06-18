# ---------------------------------------------------------------
# Modified from 'Pointnet_Pointnet2_pytorch ' 
# Reference: https://github.com/yanx27/Pointnet_Pointnet2_pytorch 
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()
        self.k = num_classes

        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 9 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

        #####################################
        self.attentpoint = nn.Conv1d(9, 3, 1)

        self.attent1 = nn.Conv1d(64, 1, 1)
        self.attent2 = nn.Conv1d(128, 1, 1)
        self.attent3 = nn.Conv1d(256, 1, 1)

        self.upsample1 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=16, mode='nearest')
        self.upsample3 = nn.Upsample(scale_factor=64, mode='nearest')

        self.sigmoid = nn.Sigmoid()
        #####################################

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        ###############################
        W = self.attentpoint(l0_points)
        W = F.softmax(W, dim=1)
        w1, w2, w3 = torch.split(W, split_size_or_sections=1, dim=1)
        w1 = w1.transpose(2,1).contiguous().view(-1)
        w2 = w2.transpose(2,1).contiguous().view(-1)
        w3 = w3.transpose(2,1).contiguous().view(-1)

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        attention1 = self.attent1(l1_points)
        attention1 = self.upsample1(attention1)
        attention1 = attention1.transpose(2,1).contiguous().view(-1)
        attention1 = self.sigmoid(attention1)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        attention2 = self.attent2(l2_points)
        attention2 = self.upsample2(attention2)
        attention2 = attention2.transpose(2,1).contiguous().view(-1)
        attention2 = self.sigmoid(attention2)

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        attention3 = self.attent3(l3_points)
        attention3 = self.upsample3(attention3)
        attention3 = attention3.transpose(2,1).contiguous().view(-1)
        attention3 = self.sigmoid(attention3)

        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)

        x_tmp = x
        x_tmp.permute(0, 2, 1)        
        x_msp = F.softmax(x_tmp.view(-1, self.k), dim=-1)
        x_msp, idx_msp           = torch.max(x_msp, dim=-1)
        x_maxlogit, idx_maxlogit = torch.max(x_tmp.view(-1, self.k), dim=-1)
        ####################################################################

        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points, x_msp, idx_msp, x_maxlogit, idx_maxlogit, attention1, attention2, attention3, w1, w2, w3


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss


if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))
