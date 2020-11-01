from torch.nn import CrossEntropyLoss, BCELoss, L1Loss, Tanh
from torch.nn.modules import loss
from utils.get_optimizer import get_optimizer
from utils.TripletLoss import TripletLoss
import torch
from torch.distributions import normal
import numpy as np
import copy
from utils.tensor2img import tensor2im
from opt import opt

from torchvision import transforms
from PIL import Image

class Loss(loss._Loss):
    def __init__(self, model):
        super(Loss, self).__init__()
        self.batch_size = opt.batchid * opt.batchimage
        self.num_gran = 8
        self.tanh = Tanh()
        self.l1_loss = L1Loss()
        self.bce_loss = BCELoss()
        self.cross_entropy_loss = CrossEntropyLoss()

        self.model = model
        self.optimizer, self.optimizer_D, self.optimizer_DC = get_optimizer(model)

    def get_positive_pairs(self):
        idx = []
        for i in range(self.batch_size):
            r = i
            while r == i:
                r = int(torch.randint(
                    low=opt.batchimage * (i // opt.batchimage), high=opt.batchimage * (i // opt.batchimage + 1),
                    size=(1,)).item())
            idx.append(r)
        return idx

    def region_wise_shuffle(self, id, ps_idx):
        sep_id = id.clone()
        idx = torch.tensor([0] * (self.num_gran))
        while (torch.sum(idx) == 0) and (torch.sum(idx) == self.num_gran):
            idx = torch.randint(high=2, size=(self.num_gran,))

        for i in range(self.num_gran):
            if idx[i]:
                sep_id[:, opt.feat_id * i:opt.feat_id * (i + 1)] = id[ps_idx][:, opt.feat_id * i:opt.feat_id * (i + 1)]
        return sep_id

    def get_noise(self):
        return torch.randn(self.batch_size, opt.feat_niz, device=opt.device)

    def make_onehot(self, label):
        onehot_vec = torch.zeros(self.batch_size, opt.num_cls)
        for i in range(label.size()[0]):
            onehot_vec[i, label[i]] = 1
        return onehot_vec

    def set_parameter(self, m, train=True):
        if train:
            for param in m.parameters():
                param.requires_grad = True
            m.apply(self.set_bn_to_train)
        else:
            for param in m.parameters():
                param.requires_grad = False
            m.apply(self.set_bn_to_eval)

    def set_bn_to_eval(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            m.eval()

    def set_bn_to_train(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            m.train()

    def set_model(self, batch=None):
        self.model.C.zero_grad()
        self.model.G.zero_grad()
        self.model.D.zero_grad()

        if opt.stage == 0:
            self.set_parameter(self.model.C, train=True)
            self.set_parameter(self.model.G, train=False)
            self.set_parameter(self.model.D, train=False)
            self.set_parameter(self.model.DC, train=False)

        elif opt.stage == 1:
            self.set_parameter(self.model.C, train=False)
            # cloth_dict1 = self.model.C.get_modules(self.model.C.cloth_dict1())
            # cloth_dict2 = self.model.C.get_modules(self.model.C.cloth_dict2())
            # for i in range(np.shape(cloth_dict1)[0]):
            #     self.set_parameter(cloth_dict1[i], train=True)
            # for i in range(np.shape(cloth_dict2)[0]):
            #     self.set_parameter(cloth_dict2[i], train=True)
            self.set_parameter(self.model.C.cloth_encoder, train=True)
            self.set_parameter(self.model.G, train=False)
            self.set_parameter(self.model.D, train=False)
            self.set_parameter(self.model.DC, train=True)

        elif opt.stage == 2:
            self.set_parameter(self.model.C, train=False)
            # nid_dict1 = self.model.C.get_modules(self.model.C.nid_dict1())
            # nid_dict2 = self.model.C.get_modules(self.model.C.nid_dict2())
            # for i in range(np.shape(nid_dict1)[0]):
            #     self.set_parameter(nid_dict1[i], train=True)
            # for i in range(np.shape(nid_dict2)[0]):
            #     self.set_parameter(nid_dict2[i], train=True)
            self.set_parameter(self.model.C.nid_encoder, train=True)
            self.set_parameter(self.model.G, train=True)
            self.set_parameter(self.model.D, train=True)
            self.set_parameter(self.model.DC, train=False)



    def id_related_loss(self, labels, outputs):
        CrossEntropy_Loss = [self.cross_entropy_loss(output, labels) for output in outputs[1:1 + self.num_gran]]  # outputs[1:1 + self.num_gran]
        return sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

    def cloth_related_loss(self, labels, outputs):
        CrossEntropy_Loss = [self.cross_entropy_loss(output, labels) for output in outputs[-6]]
        return sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

    def rgb_gray_distance(self, rgb_outputs, gray_outputs):
        return torch.sum(torch.pairwise_distance(rgb_outputs[0], gray_outputs[0]))

    def KL_loss(self, outputs):
        list_mu = outputs[-3]
        list_lv = outputs[-2]
        loss_KL = 0.
        for i in range(np.size(list_mu)):
            loss_KL += torch.sum(0.5 * (list_mu[i] ** 2 + torch.exp(list_lv[i]) - list_lv[i] - 1))
        return loss_KL / np.size(list_mu)

    def GAN_loss(self, inputs, outputs, labels, cloth_labels, epoch, batch):
        id = outputs[0]
        nid = outputs[-1]
        cloth = outputs[-4]
        one_hot_labels = self.make_onehot(labels).to(opt.device)
        ps_idx = self.get_positive_pairs()
        neg_idx = ps_idx[::-1]

        if epoch <= 100:
            remain_identity = ['aaa', 'paa']
            remain_cloth = ['aaa', 'nnn']
        else:
            remain_identity = ['aaa', 'paa', 'naa', 'aap', 'pap', 'nap', 'aan', 'pan', 'nan']
            remain_cloth = ['aaa', 'apa', 'ana', 'nan', 'npn', 'nnn']
        id_cls_label = []
        cloth_cls_label = []
        identity_G = []
        cloth_G = []
        construction_list = [id, cloth, nid]
        G_loss = 0
        D_loss = 0
        G_out_buffer = None

        ############################################## G_in_for_identity ############################################
        for construction in remain_identity:
            feature_list = []
            construction_label = one_hot_labels if construction[0] == 'a' or construction[0] == 'p' else one_hot_labels[
                neg_idx]
            id_cls_label.append(labels if construction[0] == 'a' or construction[0] == 'p' else labels[neg_idx])

            for i in range(len(construction)):
                if construction[i] == 'a':
                    feature_list.append(construction_list[i])
                elif construction[i] == 'p':
                    feature_list.append(construction_list[i][ps_idx])
                elif construction[i] == 'n':
                    feature_list.append(construction_list[i][neg_idx])
            feature_list.append(self.get_noise())

            G_in = torch.cat(feature_list, dim=1)
            G_out = self.model.G.forward(G_in, construction_label)
            identity_G.append(G_out)

        for construction in remain_cloth:
            feature_list = []
            construction_label = one_hot_labels if construction[0] == 'a' or construction[0] == 'p' else \
                one_hot_labels[neg_idx]
            classification_label = cloth_labels
            if construction[1] == 'p':
                classification_label = cloth_labels[ps_idx]
            elif construction[1] == 'n':
                classification_label = cloth_labels[neg_idx]
            cloth_cls_label.append(classification_label)

            for i in range(len(construction)):
                if construction[i] == 'a':
                    feature_list.append(construction_list[i])
                elif construction[i] == 'p':
                    feature_list.append(construction_list[i][ps_idx])
                elif construction[i] == 'n':
                    feature_list.append(construction_list[i][neg_idx])
            feature_list.append(self.get_noise())

            G_in = torch.cat(feature_list, dim=1)
            G_out = self.model.G.forward(G_in, construction_label)
            cloth_G.append(G_out)

        ############################################## D_loss for_identity############################################
        for i in range(len(remain_identity)):
            D_real = self.model.D(inputs)
            # I_real, C_real = self.model.DC(inputs)
            REAL_LABEL = torch.FloatTensor(D_real.size()).uniform_(0.7, 1.0).to(opt.device)
            D_real_loss = self.bce_loss(D_real, REAL_LABEL)


            D_fake = self.model.D(identity_G[i].detach())
            # I_fake, C_fake = self.model.DC(identity_G[i].detach())
            FAKE_LABEL = torch.FloatTensor(D_fake.size()).uniform_(0.0, 0.3).to(opt.device)
            D_fake_loss = self.bce_loss(D_fake, FAKE_LABEL)
            # I_fake_loss = 0
            # if epoch > 150:
                # I_fake_loss = self.cross_entropy_loss(I_fake, id_cls_label[i])

            D_loss += (D_real_loss + D_fake_loss)

        ############################################## D_loss for_cloth############################################
        for i in range(len(remain_cloth)):
            D_fake = self.model.D(cloth_G[i].detach())
            # I_fake, C_fake = self.model.DC(cloth_G[i].detach())
            FAKE_LABEL = torch.FloatTensor(D_fake.size()).uniform_(0.0, 0.3).to(opt.device)
            D_fake_loss = self.bce_loss(D_fake, FAKE_LABEL)
            # C_fake_loss = 0
            # if epoch > 150:
               #  C_fake_loss = self.cross_entropy_loss(C_fake, cloth_cls_label[i])

            D_loss += (D_fake_loss)

        D_loss.backward()
        self.optimizer_D.step()

        ############################################## G_loss for_identity##############################################
        for i in range(len(remain_identity)):
            D_fake = self.model.D(identity_G[i])
            I_fake, C_fake = self.model.DC(identity_G[i])
            REAL_LABEL = torch.ones_like(D_fake)
            D_fake_loss = self.bce_loss(D_fake, REAL_LABEL)
            I_fake_loss = 0
            if epoch > 100:
                I_fake_loss = self.cross_entropy_loss(I_fake, id_cls_label[i])

            G_loss += D_fake_loss + I_fake_loss * 2

        imgr_loss = 0
        imgr_loss += self.l1_loss(identity_G[0], self.tanh(inputs))
        imgr_loss += self.l1_loss(identity_G[1], self.tanh(inputs))
        if epoch > 100:
            imgr_loss += self.l1_loss(identity_G[3], identity_G[4])
            imgr_loss += self.l1_loss(identity_G[6], identity_G[7])

        G_loss += imgr_loss * 10

        ############################################## G_loss_for_cloth ############################################
        for i in range(len(remain_cloth)):
            D_fake = self.model.D(cloth_G[i])
            I_fale, C_fake = self.model.DC(cloth_G[i])
            REAL_LABEL = torch.ones_like(D_fake)
            D_fake_loss = self.bce_loss(D_fake, REAL_LABEL)
            C_fake_loss = 0
            if epoch > 50:
                C_fake_loss = self.cross_entropy_loss(C_fake, cloth_cls_label[i])

            G_loss += D_fake_loss + C_fake_loss * 2

        imgr_loss = 0
        if epoch > 100:
            imgr_loss += self.l1_loss(cloth_G[0], self.tanh(inputs))
            imgr_loss += self.l1_loss(cloth_G[-1], self.tanh(inputs[neg_idx]))

        G_loss += imgr_loss * 10

        ############################################################################################

        return D_loss, G_loss

    def GAN_loss_stage3(self, inputs, outputs, labels, cloth_labels, batch):
        id = outputs[0]
        nid = outputs[-1]
        cloth = outputs[-4]
        one_hot_labels = self.make_onehot(labels).to(opt.device)
        ps_idx = self.get_positive_pairs()
        neg_idx = ps_idx[::-1]
        remain_identity = []
        remain_cloth = []

        if batch % 9 == 0:
            remain_identity = ['aaa']
            remain_cloth = ['aaa']
        elif batch % 9 == 1:
            remain_identity = ['paa']
            remain_cloth = ['apa']
        elif batch % 9 == 2:
            remain_identity = ['naa']
            remain_cloth = ['ana']
        elif batch % 9 == 3:
            remain_identity = ['aap', 'pap']
            remain_cloth = []
        elif batch % 9 == 4:
            remain_identity = ['nap']
            remain_cloth = ['pap']
        elif batch % 9 == 5:
            remain_identity = ['aan', 'pan']
            remain_cloth = []
        elif batch % 9 == 6:
            remain_identity = ['nan']
            remain_cloth = ['nan']
        elif batch % 9 == 7:
            remain_identity = []
            remain_cloth = ['ppp', 'pnp']
        elif batch % 9 == 8:
            remain_identity = []
            remain_cloth = ['npn', 'nnn']
        id_cls_label = []
        cloth_cls_label = []
        identity_G = []
        cloth_G = []
        construction_list = [id, cloth, nid]
        G_loss = 0
        D_loss = 0
        G_out_buffer = None

        ############################################## G_in_for_identity ############################################
        for construction in remain_identity:
            feature_list = []
            construction_label = one_hot_labels if construction[0] == 'a' or construction[0] == 'p' else one_hot_labels[neg_idx]
            id_cls_label.append(labels if construction[0] == 'a' or construction[0] == 'p' else labels[neg_idx])

            for i in range(len(construction)):
                if construction[i] == 'a':
                    feature_list.append(construction_list[i])
                elif construction[i] == 'p':
                    feature_list.append(construction_list[i][ps_idx])
                elif construction[i] == 'n':
                    feature_list.append(construction_list[i][neg_idx])
            feature_list.append(self.get_noise())

            G_in = torch.cat(feature_list, dim=1)
            G_out = self.model.G.forward(G_in, construction_label)
            identity_G.append(G_out)

        for construction in remain_cloth:
            feature_list = []
            construction_label = one_hot_labels if construction[0] == 'a' or construction[0] == 'p' else \
            one_hot_labels[neg_idx]
            classification_label = cloth_labels
            if construction[1] == 'p':
                classification_label = cloth_labels[ps_idx]
            elif construction[1] == 'n':
                classification_label = cloth_labels[neg_idx]
            cloth_cls_label.append(classification_label)

            for i in range(len(construction)):
                if construction[i] == 'a':
                    feature_list.append(construction_list[i])
                elif construction[i] == 'p':
                    feature_list.append(construction_list[i][ps_idx])
                elif construction[i] == 'n':
                    feature_list.append(construction_list[i][neg_idx])
            feature_list.append(self.get_noise())

            G_in = torch.cat(feature_list, dim=1)
            G_out = self.model.G.forward(G_in, construction_label)
            cloth_G.append(G_out)

        ############################################## D_loss for_identity############################################
        for i in range(len(remain_identity)):
            D_real = self.model.D(inputs)
            # I_real, C_real = self.model.DC(inputs)
            REAL_LABEL = torch.FloatTensor(D_real.size()).uniform_(0.7, 1.0).to(opt.device)
            D_real_loss = self.bce_loss(D_real, REAL_LABEL)
            # I_real_loss = self.cross_entropy_loss(I_real, labels)
            # C_real_loss = self.cross_entropy_loss(C_real, cloth_labels)

            D_fake = self.model.D(identity_G[i].detach())
            # I_fake, C_fake = self.model.DC(identity_G[i].detach())
            FAKE_LABEL = torch.FloatTensor(D_fake.size()).uniform_(0.0, 0.3).to(opt.device)
            D_fake_loss = self.bce_loss(D_fake, FAKE_LABEL)
            # I_fake_loss = self.cross_entropy_loss(I_fake, id_cls_label[i])

            D_loss += (D_real_loss + D_fake_loss)

        ############################################## D_loss for_cloth############################################
        for i in range(len(remain_cloth)):
            D_fake = self.model.D(cloth_G[i].detach())
            # I_fake, C_fake = self.model.DC(cloth_G[i].detach())
            FAKE_LABEL = torch.FloatTensor(D_fake.size()).uniform_(0.0, 0.3).to(opt.device)
            D_fake_loss = self.bce_loss(D_fake, FAKE_LABEL)
            # C_fake_loss = self.cross_entropy_loss(C_fake, cloth_cls_label[i])

            D_loss += (D_fake_loss)# + (C_fake_loss) / 2

        D_loss.backward()
        self.optimizer_D.step()

        ############################################## G_loss for_identity##############################################
        for i in range(len(remain_identity)):
            D_fake = self.model.D(identity_G[i])
            I_fake = self.model.DC(identity_G[i], 0)
            REAL_LABEL = torch.ones_like(D_fake)
            D_fake_loss = self.bce_loss(D_fake, REAL_LABEL)
            I_fake_loss = self.cross_entropy_loss(I_fake, id_cls_label[i])
            if batch % 400 == 0 and i == 0:
                print('loss:', I_fake_loss.data.cpu().numpy())

            G_loss += D_fake_loss + I_fake_loss * 2

        imgr_loss = 0
        if batch % 9 == 0 or batch % 9 == 1:
            imgr_loss += self.l1_loss(identity_G[0], self.tanh(inputs))
        elif batch % 9 == 3 or batch % 9 == 5:
            imgr_loss += self.l1_loss(identity_G[0], identity_G[1])

        G_loss += imgr_loss * 10

        ############################################## G_loss_for_cloth ############################################
        for i in range(len(remain_cloth)):
            D_fake = self.model.D(cloth_G[i])
            C_fake = self.model.DC(cloth_G[i], 1)
            REAL_LABEL = torch.ones_like(D_fake)
            D_fake_loss = self.bce_loss(D_fake, REAL_LABEL)
            C_fake_loss = self.cross_entropy_loss(C_fake, cloth_cls_label[i])
            if batch % 400 == 0 and i == 0:
                print('loss:', C_fake_loss.data.cpu().numpy())

            G_loss += D_fake_loss + C_fake_loss * 2

        imgr_loss = 0
        if batch % 9 == 0 or batch % 9 == 7:
            imgr_loss += self.l1_loss(cloth_G[0], self.tanh(inputs))
        elif batch % 9 == 8:
            imgr_loss += self.l1_loss(cloth_G[0], self.tanh(inputs[neg_idx]))

        G_loss += imgr_loss * 10

        ############################################################################################

        return D_loss, G_loss

    def forward(self, rgb, labels, cloth_labels, batch, epoch):
        self.set_model(batch)

        if opt.stage == 0:
            rgb_outputs = self.model.C(rgb)
            Rgb_CE = self.id_related_loss(labels, rgb_outputs)
            # ID_Error画图依赖
            IDcnt = 0
            IDtotal = opt.batchid * opt.batchimage * self.num_gran
            for classifyprobabilities in rgb_outputs[1:1 + self.num_gran]:
                for i in range(opt.batchid * opt.batchimage):
                    class_ = torch.argmax(classifyprobabilities[i])
                    label = labels[i]
                    if class_ == label:
                        IDcnt += 1

            loss_sum = Rgb_CE

            print('\rRgb_CE:%.2f' % (
                Rgb_CE.data.cpu().numpy()
            ), end=' ')

            return loss_sum, [Rgb_CE.data.cpu().numpy()], [[IDcnt, IDtotal], [1, 1]]


        elif opt.stage == 1:
            rgb_outputs = self.model.C(rgb)
            D_outputs_id, D_outputs_cloth = self.model.DC(rgb)
            Cloth_CE = self.cloth_related_loss(cloth_labels, rgb_outputs)

            #Cloth画图依赖
            Clothcnt = 0
            Clothtotal = opt.batchid * opt.batchimage * 3
            for classifyprobabilities in rgb_outputs[-6]:
                for i in range(opt.batchid * opt.batchimage):
                    class_ = torch.argmax(classifyprobabilities[i])
                    cloth_label = cloth_labels[i]
                    if class_ == cloth_label:
                        Clothcnt += 1

            D_I = self.cross_entropy_loss(D_outputs_id, labels)
            D_C = self.cross_entropy_loss(D_outputs_cloth, cloth_labels)
            DC_loss = D_I + D_C
            DC_loss.backward()
            self.optimizer_DC.step()
            # Mix_CE = self.id_related_loss(labels, mix_outputs)
            loss_sum = Cloth_CE

            print('\rCloth_CE:%.2f D_I:%.2f D_C:%.2f' % (
                Cloth_CE.data.cpu().numpy(),
                D_I.data.cpu().numpy(),
                D_C.data.cpu().numpy()
                ), end=' ')

            return loss_sum, \
                   [Cloth_CE.data.cpu().numpy(), D_I.data.cpu().numpy(), D_C.data.cpu().numpy()],\
                   [[1, 1], [Clothcnt, Clothtotal]]

        elif opt.stage == 2:
            rgb_outputs = self.model.C(rgb)
            if epoch <= 100:
                D_loss, G_loss = self.GAN_loss(rgb, rgb_outputs, labels, cloth_labels, epoch, batch)
            else:
                D_loss, G_loss = self.GAN_loss_stage3(rgb, rgb_outputs, labels, cloth_labels, batch)
            KL_loss = self.KL_loss(rgb_outputs)

            loss_sum = G_loss + KL_loss / 500

            print('\rD_loss:%.2f  G_loss:%.2f KL:%.2f' % (
                D_loss.data.cpu().numpy(),
                G_loss.data.cpu().numpy(),
                KL_loss.data.cpu().numpy()), end=' ')

            return loss_sum,\
                   [D_loss.data.cpu().numpy(), G_loss.data.cpu().numpy(), KL_loss.data.cpu().numpy()],\
                   [[1, 1], [1, 1]]

        elif opt.stage == 3:
            D_outputs_id, D_outputs_cloth = self.model.DC(rgb)
            D_I = self.cross_entropy_loss(D_outputs_id, labels)
            D_C = self.cross_entropy_loss(D_outputs_cloth, cloth_labels)
            DC_loss = D_I + D_C
            DC_loss.backward()
            self.optimizer_DC.step()


            rgb_outputs = self.model.C(rgb)
            Rgb_CE = self.id_related_loss(labels, rgb_outputs)
            cloth_loss = self.cloth_related_loss(cloth_labels, rgb_outputs)

            # ID_Error画图依赖
            IDcnt = 0
            IDtotal = opt.batchid * opt.batchimage * self.num_gran
            for classifyprobabilities in rgb_outputs[1:1 + self.num_gran]:
                for i in range(opt.batchid * opt.batchimage):
                    class_ = torch.argmax(classifyprobabilities[i])
                    label = labels[i]
                    if class_ == label:
                        IDcnt += 1

            # Cloth画图依赖
            Clothcnt = 0
            Clothtotal = opt.batchid * opt.batchimage * 3
            for classifyprobabilities in rgb_outputs[-6]:
                for i in range(opt.batchid * opt.batchimage):
                    class_ = torch.argmax(classifyprobabilities[i])
                    cloth_label = cloth_labels[i]
                    if class_ == cloth_label:
                        Clothcnt += 1

            D_loss, G_loss = self.GAN_loss_stage3(rgb, rgb_outputs, labels, cloth_labels, batch)
            KL_loss = self.KL_loss(rgb_outputs)

            loss_sum = (Rgb_CE) * 20 + cloth_loss * 10 + G_loss / 2 + KL_loss / 100

            print('\rRgb_CE:%.2f Cloth:%.2f  D_loss:%.2f  G_loss:%.2f D_I:%.2f D_C:%.2f '
                  ' KL:%.2f' % (
                      Rgb_CE.data.cpu().numpy(),
                      cloth_loss.data.cpu().numpy(),
                      D_loss.data.cpu().numpy(),
                      G_loss.data.cpu().numpy(),
                      D_I.data.cpu().numpy(),
                      D_C.data.cpu().numpy(),
                      KL_loss.data.cpu().numpy()), end=' ')
            return loss_sum, \
                   [Rgb_CE.data.cpu().numpy(), cloth_loss.data.cpu().numpy(), D_loss.data.cpu().numpy(),
                        G_loss.data.cpu().numpy(), D_I.data.cpu().numpy(), D_C.data.cpu().numpy(),
                        KL_loss.data.cpu().numpy()], \
                   [[IDcnt, IDtotal], [Clothcnt, Clothtotal]]
