import copy
import math
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torchvision.models.resnet import resnet50, Bottleneck, resnet18, resnet34
from opt import opt
from torch.nn import init


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.C = EncoderPRCCVer()
        self.G = Generator()
        self.D = Discriminator()
        self.DC = ft_net()


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
        nn.init.constant_(m.bias.data, 0.)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, mean=1., std=0.02)
        nn.init.constant_(m.bias.data, 0.0)


def init_weights(net):
    net.apply(weights_init_normal)


def conv3x3(in_places, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_places, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SPTNet(nn.Module):
    def __init__(self, channel=3, stripe=8, class_num=int(opt.num_cls)):  # channel=1 for default.
        block = BasicBlock
        layers = [2, 2, 2, 2]

        self.inplanes = 64
        self.stripe = stripe
        super(SPTNet, self).__init__()

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.mask = nn.ModuleList([self._make_maks() for _ in range(self.stripe)])

        self.down = nn.ModuleList([nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
            for _ in range(self.stripe)])

        self.fc_list = nn.ModuleList([nn.Linear(256, class_num) for _ in range(stripe)])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_maks(self):

        return nn.Sequential(nn.Linear(256, 128),
                             nn.ReLU(inplace=True),
                             nn.Linear(128, 256),
                             nn.Sigmoid())

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        assert x.size(2) % self.stripe == 0
        stripe_h = x.size(2) // self.stripe
        local_feat_list = []
        logits_list = []

        for i in range(self.stripe):
            # shape [N, C, 1, 1]
            local_feat = F.avg_pool2d(
                x[:, :, i * stripe_h: (i + 1) * stripe_h, :], (stripe_h, x.size(-1)))
            # shape [N, c, 1, 1]

            mask = self.mask[i](local_feat.view(local_feat.size(0), -1))
            local_feat = local_feat * mask.view(mask.size(0), mask.size(1), 1, 1) + local_feat

            local_feat = self.down[i](local_feat)  # independent conv

            # shape [N, c]
            local_feat = local_feat.view(local_feat.size(0), -1)
            local_feat_list.append(local_feat)

            logits_list.append(self.fc_list[i](local_feat))

        return logits_list, local_feat_list


class SPTNetForNid(nn.Module):
    def __init__(self, channel=3, stripe=8, feat_nid=int(opt.feat_nid)):  # channel=1 for default.
        block = BasicBlock
        layers = [2, 2, 2, 2]

        self.inplanes = 64
        self.stripe = stripe
        super(SPTNetForNid, self).__init__()

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.mask = nn.ModuleList([self._make_maks() for _ in range(self.stripe)])

        self.down = nn.ModuleList([nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
            for _ in range(self.stripe)])

        self.fc_mu_list = nn.ModuleList([nn.Linear(256, feat_nid) for _ in range(stripe)])
        self.fc_lv_list = nn.ModuleList([nn.Linear(256, feat_nid) for _ in range(stripe)])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_maks(self):

        return nn.Sequential(nn.Linear(256, 128),
                             nn.ReLU(inplace=True),
                             nn.Linear(128, 256),
                             nn.Sigmoid())

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        assert x.size(2) % self.stripe == 0
        stripe_h = x.size(2) // self.stripe
        local_feat_list = []
        mu_list = []
        lv_list = []

        for i in range(self.stripe):
            # shape [N, C, 1, 1]
            local_feat = F.avg_pool2d(
                x[:, :, i * stripe_h: (i + 1) * stripe_h, :], (stripe_h, x.size(-1)))
            # shape [N, c, 1, 1]

            mask = self.mask[i](local_feat.view(local_feat.size(0), -1))
            local_feat = local_feat * mask.view(mask.size(0), mask.size(1), 1, 1) + local_feat

            local_feat = self.down[i](local_feat)  # independent conv

            # shape [N, c]
            local_feat = local_feat.view(local_feat.size(0), -1)
            local_feat_list.append(local_feat)

            mu_list.append(self.fc_mu_list[i](local_feat))
            lv_list.append(self.fc_lv_list[i](local_feat))
            # logits_list.append(self.fc_list[i](local_feat))

        return mu_list, lv_list


class EncoderPRCCVer(nn.Module):
    def __init__(self):
        super(EncoderPRCCVer, self).__init__()
        self.id_encoder = SPTNet(class_num=int(opt.num_cls))
        self.cloth_encoder = SPTNet(class_num=int(opt.num_cls) * 2)
        self.nid_encoder = SPTNetForNid()


    def reparameterization(self, mu, lv):
        std = torch.exp(lv / 2)
        sampled_z = torch.FloatTensor(np.random.normal(0, 1, mu.size())).to(opt.device)
        return sampled_z * std + mu

    def forward(self, x):
        lg_id_list, feat_id_list = self.id_encoder(x)
        id = torch.cat(feat_id_list, dim=1)
        lg_cloth_list, feat_cloth_list = self.cloth_encoder(x)
        cloth_cls = lg_cloth_list
        cloth_feature = feat_cloth_list
        cloth = torch.cat(feat_cloth_list, dim=1)
        list_mu, list_lv = self.nid_encoder(x)
        nid = [self.reparameterization(list_mu[i], list_lv[i]) for i in range(len(list_mu))]
        nid = torch.cat(nid, dim=1)
        return id\
            , lg_id_list[0], lg_id_list[1], lg_id_list[2], lg_id_list[3], lg_id_list[4], \
               lg_id_list[5], lg_id_list[6], lg_id_list[7],\
               cloth_cls, cloth_feature, cloth, list_mu, list_lv, nid


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        resnet = resnet50(pretrained=True)

        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3[0], )  # conv4_1
        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        res_g_conv5 = resnet.layer4
        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        #########identity-related#########

        self.p0_id = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p1_id = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p2_id = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        self.maxpool_zg_p0 = nn.MaxPool2d(kernel_size=(12, 4))
        self.maxpool_zg_p1 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zp1 = nn.MaxPool2d(kernel_size=(12, 8))
        self.maxpool_zg_p2 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zp2 = nn.MaxPool2d(kernel_size=(8, 8))

        self.reduction_zg_p0_id = nn.Sequential(
            nn.Conv2d(2048, opt.feat_id, 1, bias=False), nn.BatchNorm2d(opt.feat_id))
        self.reduction_zg_p1_id = nn.Sequential(
            nn.Conv2d(2048, opt.feat_id, 1, bias=False), nn.BatchNorm2d(opt.feat_id))
        self.reduction_z0_p1_id = nn.Sequential(
            nn.Conv2d(2048, opt.feat_id, 1, bias=False), nn.BatchNorm2d(opt.feat_id))
        self.reduction_z1_p1_id = nn.Sequential(
            nn.Conv2d(2048, opt.feat_id, 1, bias=False), nn.BatchNorm2d(opt.feat_id))
        self.reduction_zg_p2_id = nn.Sequential(
            nn.Conv2d(2048, opt.feat_id, 1, bias=False), nn.BatchNorm2d(opt.feat_id))
        self.reduction_z0_p2_id = nn.Sequential(
            nn.Conv2d(2048, opt.feat_id, 1, bias=False), nn.BatchNorm2d(opt.feat_id))
        self.reduction_z1_p2_id = nn.Sequential(
            nn.Conv2d(2048, opt.feat_id, 1, bias=False), nn.BatchNorm2d(opt.feat_id))
        self.reduction_z2_p2_id = nn.Sequential(
            nn.Conv2d(2048, opt.feat_id, 1, bias=False), nn.BatchNorm2d(opt.feat_id))

        self.fc_fg_p0_id = nn.Linear(opt.feat_id, int(opt.num_cls))
        self.fc_fg_p1_id = nn.Linear(opt.feat_id, int(opt.num_cls))
        self.fc_f0_p1_id = nn.Linear(opt.feat_id, int(opt.num_cls))
        self.fc_f1_p1_id = nn.Linear(opt.feat_id, int(opt.num_cls))
        self.fc_fg_p2_id = nn.Linear(opt.feat_id, int(opt.num_cls))
        self.fc_f0_p2_id = nn.Linear(opt.feat_id, int(opt.num_cls))
        self.fc_f1_p2_id = nn.Linear(opt.feat_id, int(opt.num_cls))
        self.fc_f2_p2_id = nn.Linear(opt.feat_id, int(opt.num_cls))

        #########cloth_related#########
        self.p0_cloth = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p1_cloth = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p2_cloth = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        self.reduction_zg_p0_cloth = nn.Sequential(
            nn.Conv2d(2048, opt.feat_id, 1, bias=False), nn.BatchNorm2d(opt.feat_id))
        self.reduction_zg_p1_cloth = nn.Sequential(
            nn.Conv2d(2048, opt.feat_id, 1, bias=False), nn.BatchNorm2d(opt.feat_id))
        self.reduction_z0_p1_cloth = nn.Sequential(
            nn.Conv2d(2048, opt.feat_id, 1, bias=False), nn.BatchNorm2d(opt.feat_id))
        self.reduction_z1_p1_cloth = nn.Sequential(
            nn.Conv2d(2048, opt.feat_id, 1, bias=False), nn.BatchNorm2d(opt.feat_id))
        self.reduction_zg_p2_cloth = nn.Sequential(
            nn.Conv2d(2048, opt.feat_id, 1, bias=False), nn.BatchNorm2d(opt.feat_id))
        self.reduction_z0_p2_cloth = nn.Sequential(
            nn.Conv2d(2048, opt.feat_id, 1, bias=False), nn.BatchNorm2d(opt.feat_id))
        self.reduction_z1_p2_cloth = nn.Sequential(
            nn.Conv2d(2048, opt.feat_id, 1, bias=False), nn.BatchNorm2d(opt.feat_id))
        self.reduction_z2_p2_cloth = nn.Sequential(
            nn.Conv2d(2048, opt.feat_id, 1, bias=False), nn.BatchNorm2d(opt.feat_id))

        self.fc_cloth_g0 = nn.Linear(opt.feat_id, int(opt.num_cls) * 2)
        self.fc_cloth_g1 = nn.Linear(opt.feat_id, int(opt.num_cls) * 2)
        self.fc_cloth_g2 = nn.Linear(opt.feat_id, int(opt.num_cls) * 2)

        #########identity_unrelated#########
        self.p0_nid = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p1_nid = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p2_nid = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        self.avgpool_zg_p0 = nn.AvgPool2d(kernel_size=(12, 4))
        self.avgpool_zg_p1 = nn.AvgPool2d(kernel_size=(24, 8))
        self.avgpool_zp1 = nn.AvgPool2d(kernel_size=(12, 8))
        self.avgpool_zg_p2 = nn.AvgPool2d(kernel_size=(24, 8))
        self.avgpool_zp2 = nn.AvgPool2d(kernel_size=(8, 8))

        self.fc_zg_p0_nid_mu = nn.Linear(2048, int(opt.feat_nid))
        self.fc_zg_p0_nid_lv = nn.Linear(2048, int(opt.feat_nid))
        self.fc_zg_p1_nid_mu = nn.Linear(2048, int(opt.feat_nid))
        self.fc_zg_p1_nid_lv = nn.Linear(2048, int(opt.feat_nid))
        self.fc_z0_p1_nid_mu = nn.Linear(2048, int(opt.feat_nid))
        self.fc_z0_p1_nid_lv = nn.Linear(2048, int(opt.feat_nid))
        self.fc_z1_p1_nid_mu = nn.Linear(2048, int(opt.feat_nid))
        self.fc_z1_p1_nid_lv = nn.Linear(2048, int(opt.feat_nid))
        self.fc_zg_p2_nid_mu = nn.Linear(2048, int(opt.feat_nid))
        self.fc_zg_p2_nid_lv = nn.Linear(2048, int(opt.feat_nid))
        self.fc_z0_p2_nid_mu = nn.Linear(2048, int(opt.feat_nid))
        self.fc_z0_p2_nid_lv = nn.Linear(2048, int(opt.feat_nid))
        self.fc_z1_p2_nid_mu = nn.Linear(2048, int(opt.feat_nid))
        self.fc_z1_p2_nid_lv = nn.Linear(2048, int(opt.feat_nid))
        self.fc_z2_p2_nid_mu = nn.Linear(2048, int(opt.feat_nid))
        self.fc_z2_p2_nid_lv = nn.Linear(2048, int(opt.feat_nid))

        # self.fc_nid = nn.Linear(512, int(opt.num_cls) * 2)

        id_dict2 = self.get_modules(self.id_dict2())
        for i in range(np.size(id_dict2)):
            init_weights(id_dict2[i])

    def reparameterization(self, mu, lv):
        std = torch.exp(lv / 2)
        sampled_z = torch.FloatTensor(np.random.normal(0, 1, mu.size())).to(opt.device)
        return sampled_z * std + mu

    def id_dict1(self):
        return ['p0_id', 'p1_id', 'p2_id']

    def id_dict2(self):
        return ['reduction_zg_p0_id', 'reduction_zg_p1_id', 'reduction_zg_p2_id',
                'reduction_z0_p1_id', 'reduction_z1_p1_id',
                'reduction_z0_p2_id', 'reduction_z1_p2_id', 'reduction_z2_p2_id',
                'fc_fg_p0_id', 'fc_fg_p1_id', 'fc_fg_p2_id',
                'fc_f0_p1_id', 'fc_f1_p1_id',
                'fc_f0_p2_id', 'fc_f1_p2_id', 'fc_f2_p2_id']

    def cloth_dict1(self):
        return ['p0_cloth', 'p1_cloth', 'p2_cloth']

    def cloth_dict2(self):
        return ['reduction_zg_p0_cloth', 'reduction_zg_p1_cloth', 'reduction_zg_p2_cloth',
                'reduction_z0_p1_cloth', 'reduction_z1_p1_cloth',
                'reduction_z0_p2_cloth', 'reduction_z1_p2_cloth', 'reduction_z2_p2_cloth',
                'fc_cloth_g0', 'fc_cloth_g1', 'fc_cloth_g2']

    def nid_dict1(self):
        return ['p0_nid', 'p1_nid', 'p2_nid']

    def nid_dict2(self):
        return ['fc_zg_p0_nid_mu', 'fc_zg_p0_nid_lv',
                'fc_zg_p1_nid_mu', 'fc_zg_p1_nid_lv', 'fc_zg_p2_nid_mu', 'fc_zg_p2_nid_lv',
                'fc_z0_p1_nid_mu', 'fc_z0_p1_nid_lv', 'fc_z1_p1_nid_mu', 'fc_z1_p1_nid_lv',
                'fc_z0_p2_nid_mu', 'fc_z0_p2_nid_lv', 'fc_z1_p2_nid_mu', 'fc_z1_p2_nid_lv',
                'fc_z2_p2_nid_mu', 'fc_z2_p2_nid_lv']

    def get_modules(self, list):
        modules = []
        for name, module in self.named_children():
            if name in list:
                modules.append(module)
        return modules

    def forward(self, x):
        x = self.backbone(x)
        ##################################### identity-related #########################################
        p0_id = self.p0_id(x)
        p1_id = self.p1_id(x)
        p2_id = self.p2_id(x)

        zg_p0_id = self.maxpool_zg_p0(p0_id)
        zg_p1_id = self.maxpool_zg_p1(p1_id)
        zp1_id = self.maxpool_zp1(p1_id)
        z0_p1_id = zp1_id[:, :, 0:1, :]
        z1_p1_id = zp1_id[:, :, 1:2, :]
        zg_p2_id = self.maxpool_zg_p2(p2_id)
        zp2_id = self.maxpool_zp2(p2_id)
        z0_p2_id = zp2_id[:, :, 0:1, :]
        z1_p2_id = zp2_id[:, :, 1:2, :]
        z2_p2_id = zp2_id[:, :, 2:3, :]

        fg_p0_id = self.reduction_zg_p0_id(zg_p0_id).squeeze(dim=3).squeeze(dim=2)
        fg_p1_id = self.reduction_zg_p1_id(zg_p1_id).squeeze(dim=3).squeeze(dim=2)
        f0_p1_id = self.reduction_z0_p1_id(z0_p1_id).squeeze(dim=3).squeeze(dim=2)
        f1_p1_id = self.reduction_z1_p1_id(z1_p1_id).squeeze(dim=3).squeeze(dim=2)
        fg_p2_id = self.reduction_zg_p2_id(zg_p2_id).squeeze(dim=3).squeeze(dim=2)
        f0_p2_id = self.reduction_z0_p2_id(z0_p2_id).squeeze(dim=3).squeeze(dim=2)
        f1_p2_id = self.reduction_z1_p2_id(z1_p2_id).squeeze(dim=3).squeeze(dim=2)
        f2_p2_id = self.reduction_z2_p2_id(z2_p2_id).squeeze(dim=3).squeeze(dim=2)

        lg_p0 = self.fc_fg_p0_id(fg_p0_id)
        lg_p1 = self.fc_fg_p1_id(fg_p1_id)
        l0_p1 = self.fc_f0_p1_id(f0_p1_id)
        l1_p1 = self.fc_f1_p1_id(f1_p1_id)
        lg_p2 = self.fc_fg_p2_id(fg_p2_id)
        l0_p2 = self.fc_f0_p2_id(f0_p2_id)
        l1_p2 = self.fc_f1_p2_id(f1_p2_id)
        l2_p2 = self.fc_f2_p2_id(f2_p2_id)

        ###################################### cloth-related ########################################
        p0_cloth = self.p0_cloth(x)
        p1_cloth = self.p1_cloth(x)
        p2_cloth = self.p2_cloth(x)

        zg_p0_cloth = self.maxpool_zg_p0(p0_cloth)
        zg_p1_cloth = self.maxpool_zg_p1(p1_cloth)
        zp1_cloth = self.maxpool_zp1(p1_cloth)
        z0_p1_cloth = zp1_cloth[:, :, 0:1, :]
        z1_p1_cloth = zp1_cloth[:, :, 1:2, :]
        zg_p2_cloth = self.maxpool_zg_p2(p2_cloth)
        zp2_cloth = self.maxpool_zp2(p2_cloth)
        z0_p2_cloth = zp2_cloth[:, :, 0:1, :]
        z1_p2_cloth = zp2_cloth[:, :, 1:2, :]
        z2_p2_cloth = zp2_cloth[:, :, 2:3, :]

        fg_p0_cloth = self.reduction_zg_p0_cloth(zg_p0_cloth).squeeze(dim=3).squeeze(dim=2)
        fg_p1_cloth = self.reduction_zg_p1_cloth(zg_p1_cloth).squeeze(dim=3).squeeze(dim=2)
        f0_p1_cloth = self.reduction_z0_p1_cloth(z0_p1_cloth).squeeze(dim=3).squeeze(dim=2)
        f1_p1_cloth = self.reduction_z1_p1_cloth(z1_p1_cloth).squeeze(dim=3).squeeze(dim=2)
        fg_p2_cloth = self.reduction_zg_p2_cloth(zg_p2_cloth).squeeze(dim=3).squeeze(dim=2)
        f0_p2_cloth = self.reduction_z0_p2_cloth(z0_p2_cloth).squeeze(dim=3).squeeze(dim=2)
        f1_p2_cloth = self.reduction_z1_p2_cloth(z1_p2_cloth).squeeze(dim=3).squeeze(dim=2)
        f2_p2_cloth = self.reduction_z2_p2_cloth(z2_p2_cloth).squeeze(dim=3).squeeze(dim=2)

        ###################################### identity-unrelated ########################################
        p0_nid = self.p0_nid(x)
        p1_nid = self.p1_nid(x)
        p2_nid = self.p2_nid(x)

        zg_p0_nid = self.avgpool_zg_p0(p0_nid)
        zg_p1_nid = self.avgpool_zg_p1(p1_nid)
        zp1_nid = self.avgpool_zp1(p1_nid)
        z0_p1_nid = zp1_nid[:, :, 0:1, :]
        z1_p1_nid = zp1_nid[:, :, 1:2, :]
        zg_p2_nid = self.avgpool_zg_p2(p2_nid)
        zp2_nid = self.avgpool_zp2(p2_nid)
        z0_p2_nid = zp2_nid[:, :, 0:1, :]
        z1_p2_nid = zp2_nid[:, :, 1:2, :]
        z2_p2_nid = zp2_nid[:, :, 2:3, :]

        fc_zg_p0_nid_mu = self.fc_zg_p0_nid_mu(zg_p0_nid.squeeze(dim=3).squeeze(dim=2))
        fc_zg_p0_nid_lv = self.fc_zg_p0_nid_lv(zg_p0_nid.squeeze(dim=3).squeeze(dim=2))
        fc_zg_p1_nid_mu = self.fc_zg_p1_nid_mu(zg_p1_nid.squeeze(dim=3).squeeze(dim=2))
        fc_zg_p1_nid_lv = self.fc_zg_p1_nid_lv(zg_p1_nid.squeeze(dim=3).squeeze(dim=2))
        fc_z0_p1_nid_mu = self.fc_z0_p1_nid_mu(z0_p1_nid.squeeze(dim=3).squeeze(dim=2))
        fc_z0_p1_nid_lv = self.fc_z0_p1_nid_lv(z0_p1_nid.squeeze(dim=3).squeeze(dim=2))
        fc_z1_p1_nid_mu = self.fc_z1_p1_nid_mu(z1_p1_nid.squeeze(dim=3).squeeze(dim=2))
        fc_z1_p1_nid_lv = self.fc_z1_p1_nid_lv(z1_p1_nid.squeeze(dim=3).squeeze(dim=2))
        fc_zg_p2_nid_mu = self.fc_zg_p2_nid_mu(zg_p2_nid.squeeze(dim=3).squeeze(dim=2))
        fc_zg_p2_nid_lv = self.fc_zg_p2_nid_lv(zg_p2_nid.squeeze(dim=3).squeeze(dim=2))
        fc_z0_p2_nid_mu = self.fc_z0_p2_nid_mu(z0_p2_nid.squeeze(dim=3).squeeze(dim=2))
        fc_z0_p2_nid_lv = self.fc_z0_p2_nid_lv(z0_p2_nid.squeeze(dim=3).squeeze(dim=2))
        fc_z1_p2_nid_mu = self.fc_z1_p2_nid_mu(z1_p2_nid.squeeze(dim=3).squeeze(dim=2))
        fc_z1_p2_nid_lv = self.fc_z1_p2_nid_lv(z1_p2_nid.squeeze(dim=3).squeeze(dim=2))
        fc_z2_p2_nid_mu = self.fc_z2_p2_nid_mu(z2_p2_nid.squeeze(dim=3).squeeze(dim=2))
        fc_z2_p2_nid_lv = self.fc_z2_p2_nid_lv(z2_p2_nid.squeeze(dim=3).squeeze(dim=2))

        fc_zg_p0_nid = self.reparameterization(fc_zg_p0_nid_mu, fc_zg_p0_nid_lv)
        fc_zg_p1_nid = self.reparameterization(fc_zg_p1_nid_mu, fc_zg_p1_nid_lv)
        fc_z0_p1_nid = self.reparameterization(fc_z0_p1_nid_mu, fc_z0_p1_nid_lv)
        fc_z1_p1_nid = self.reparameterization(fc_z1_p1_nid_mu, fc_z1_p1_nid_lv)
        fc_zg_p2_nid = self.reparameterization(fc_zg_p2_nid_mu, fc_zg_p2_nid_lv)
        fc_z0_p2_nid = self.reparameterization(fc_z0_p2_nid_mu, fc_z0_p2_nid_lv)
        fc_z1_p2_nid = self.reparameterization(fc_z1_p2_nid_mu, fc_z1_p2_nid_lv)
        fc_z2_p2_nid = self.reparameterization(fc_z2_p2_nid_mu, fc_z2_p2_nid_lv)

        list_mu = [fc_zg_p0_nid_mu, fc_zg_p1_nid_mu, fc_z0_p1_nid_mu, fc_z1_p1_nid_mu,
                   fc_zg_p2_nid_mu, fc_z0_p2_nid_mu, fc_z1_p2_nid_mu, fc_z2_p2_nid_mu]
        list_lv = [fc_zg_p0_nid_lv, fc_zg_p1_nid_lv, fc_z0_p1_nid_lv, fc_z1_p1_nid_lv,
                   fc_zg_p2_nid_lv, fc_z0_p2_nid_lv, fc_z1_p2_nid_lv, fc_z2_p2_nid_lv]

        id = torch.cat(
            [fg_p0_id, fg_p1_id, f0_p1_id, f1_p1_id, fg_p2_id, f0_p2_id, f1_p2_id, f2_p2_id], dim=1)
        nid = torch.cat(
            [fc_zg_p0_nid, fc_zg_p1_nid, fc_z0_p1_nid, fc_z1_p1_nid,
             fc_zg_p2_nid, fc_z0_p2_nid, fc_z1_p2_nid, fc_z2_p2_nid], dim=1)
        cloth_feature = [fg_p0_cloth, fg_p1_cloth, fg_p2_cloth]
        # cloth = self.fc_nid(nid)
        cloth = torch.cat(
            [fg_p0_cloth, fg_p1_cloth, f0_p1_cloth, f1_p1_cloth, fg_p2_cloth, f0_p2_cloth, f1_p2_cloth, f2_p2_cloth],
            dim=1)

        cloth_cls = [self.fc_cloth_g0(fg_p0_cloth), self.fc_cloth_g1(fg_p1_cloth), self.fc_cloth_g2(fg_p2_cloth)]
        return id, lg_p0, lg_p1, l0_p1, l1_p1, lg_p2, l0_p2, l1_p2, l2_p2, cloth_cls, cloth_feature, cloth, list_mu, list_lv, nid


class Generator(nn.Module):
    def __init__(self, output_dim=3):
        super(Generator, self).__init__()

        self.G_fc = nn.Sequential(
            nn.Linear(opt.feat_id * 8 * 2 + opt.feat_nid * 8 + opt.feat_niz + opt.num_cls, opt.feat_G * 8),
            nn.BatchNorm1d(opt.feat_G * 8),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(opt.dropout))

        '''
        self.G_fc_feature = nn.Sequential(
            nn.Linear(opt.feat_id * 8 * 2 + opt.feat_nid * 8, opt.feat_G * 8),
            nn.BatchNorm1d(opt.feat_G * 8),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(opt.dropout))
        self.G_fc_class = nn.Linear(opt.feat_G * 8, opt.num_cls)
        '''

        self.G_deconv = nn.Sequential(
            # 1st block
            nn.ConvTranspose2d(opt.feat_G * 8, opt.feat_G * 8, kernel_size=(6, 2), bias=False),
            nn.BatchNorm2d(opt.feat_G * 8),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(opt.dropout),
            # 2nd block
            nn.ConvTranspose2d(opt.feat_G * 8, opt.feat_G * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(opt.feat_G * 8),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(opt.dropout),
            # 3rd block
            nn.ConvTranspose2d(opt.feat_G * 8, opt.feat_G * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(opt.feat_G * 8),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(opt.dropout),
            # 4th block
            nn.ConvTranspose2d(opt.feat_G * 8, opt.feat_G * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(opt.feat_G * 4),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(opt.dropout),
            # 5th block
            nn.ConvTranspose2d(opt.feat_G * 4, opt.feat_G * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(opt.feat_G * 2),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(opt.dropout),
            # 6th block
            nn.ConvTranspose2d(opt.feat_G * 2, opt.feat_G * 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(opt.feat_G * 1),
            nn.LeakyReLU(0.2, True),
            # 7th block
            nn.ConvTranspose2d(opt.feat_G * 1, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh())
        init_weights(self.G_fc)
        init_weights(self.G_deconv)

    def forward(self, inputs, labels):
        combine = inputs[:, :2560]
        # feature = self.G_fc_feature(combine)
        # cls = self.G_fc_class(feature)
        x = torch.cat([inputs, labels], 1)
        x = self.G_fc(x).view(-1, opt.feat_G * 8, 1, 1)
        x = self.G_deconv(x)
        return x  # feature, cls

    '''
    def eva_forward(self, inputs):
        combine = inputs[:, :2560]
        feature = self.G_fc_feature(combine)
        cls = self.G_fc_class(feature)
        return feature, cls
    '''


class Discriminator(nn.Module):
    def __init__(self, norm_layer=nn.InstanceNorm2d):
        super(Discriminator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        n_layers = 5  # 5
        kw = 4
        padw = 1
        backbone = [
            nn.Tanh(),
            nn.Conv2d(3, opt.feat_D, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            backbone += [
                nn.Conv2d(
                    opt.feat_D * nf_mult_prev, opt.feat_D * nf_mult, kernel_size=kw,
                    stride=2, padding=padw, bias=use_bias),
                norm_layer(opt.feat_D * nf_mult),
                nn.LeakyReLU(0.2, True), ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        image_D = [
            nn.Conv2d(
                opt.feat_D * nf_mult_prev, opt.feat_D * nf_mult, kernel_size=kw,
                stride=1, padding=padw, bias=use_bias),
            norm_layer(opt.feat_D * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(opt.feat_D * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
            nn.Sigmoid()]

        label_D1 = [
            nn.Conv2d(
                opt.feat_D * nf_mult_prev, opt.feat_D * nf_mult, kernel_size=kw,
                stride=1, padding=padw, bias=use_bias),
            norm_layer(opt.feat_D * nf_mult), ]
        self.avgp = nn.AvgPool2d(kernel_size=(11, 3))  # 11, 3  #95, 31
        label_D2 = [nn.Linear(opt.feat_D * nf_mult, int(opt.num_cls))]
        label_D3 = [nn.Linear(opt.feat_D * nf_mult, int(opt.num_cls * 2))]

        self.backbone = nn.Sequential(*backbone)
        self.image_D = nn.Sequential(*image_D)
        self.label_D1 = nn.Sequential(*label_D1)
        self.label_D2 = nn.Sequential(*label_D2)
        self.label_D1_cloth = nn.Sequential(*label_D1)
        self.label_D2_cloth = nn.Sequential(*label_D3)

    def forward(self, input):
        backbone = self.backbone(input)
        image_D = self.image_D(backbone)
        # label_D1 = self.label_D1(backbone)
        # avgp = self.avgp(label_D1)
        # avgp = avgp.squeeze(dim=3).squeeze(dim=2)
        # label_D2 = self.label_D2(avgp)
        # label_D1_cloth = self.label_D1_cloth(backbone)
        # avgp2 = self.avgp(label_D1_cloth).squeeze(dim=3).squeeze(dim=2)
        # label_D2_cloth = self.label_D2_cloth(avgp2)
        return image_D  # , label_D2, label_D2_cloth


######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=False, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        if dropout > 0:
            classifier += [nn.Dropout(p=dropout)]
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        f = x
        f_norm = f.norm(p=2, dim=1, keepdim=True) + 1e-8
        f = f.div(f_norm)
        x = self.classifier(x)
        return x, f


# Define the ResNet-based Model
class ft_net(nn.Module):

    def __init__(self, class_num=150):
        super(ft_net, self).__init__()
        model_ft = resnet18(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.backbone = nn.Sequential(model_ft.conv1, model_ft.bn1, model_ft.relu, model_ft.maxpool)
        self.id = nn.Sequential(model_ft.layer1, model_ft.layer2, model_ft.layer3, model_ft.layer4, model_ft.avgpool)
        self.cloth = nn.Sequential(model_ft.layer1, model_ft.layer2, model_ft.layer3, model_ft.layer4, model_ft.avgpool)
        self.classifier_id = ClassBlock(2048, class_num, dropout=0.5, relu=False)
        self.classifier_cloth = ClassBlock(2048, class_num * 2, dropout=0.5, relu=False)
        # remove the final downsample
        # self.model.layer4[0].downsample[0].stride = (1,1)
        # self.model.layer4[0].conv2.stride = (1,1)

    def forward(self, x):
        x = self.backbone(x)
        x1 = self.id(x)
        x2 = self.cloth(x)
        x1 = x1.view(x1.size(0), x1.size(1))
        x1, f = self.classifier_id(x1)
        x2 = x2.view(x2.size(0), x2.size(1))
        x2, f = self.classifier_cloth(x2)
        return x1, x2


if __name__ == '__main__':
    print(Model().DC)
