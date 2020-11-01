import os
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import torch
import torchvision.utils as vutils
from torch.optim import lr_scheduler
from matplotlib import pyplot as plt
from opt import opt
from data import Data
from network import Model
# from PCB import Model
from loss import Loss
from utils.extract_feature import extract_feature
from utils.metrics import mean_ap, cmc, re_ranking


class Main():
    def __init__(self, model, loss, data):
        if opt.stage == 1 or opt.stage == 0:
            self.train_loader = data.train_loader
        else:
            self.train_loader = data.train_loader_woEr
        self.test_loader = data.test_loader
        self.query_loader = data.query_loader
        self.testset = data.testset
        self.queryset = data.queryset

        self.model = model.to(opt.device)
        self.loss = loss
        self.data = data

        self.x = []
        self.y = [[], [], [], [], [], [], []]
        self.errors = [[], []]  # IDError/ClothError

        self.scheduler = lr_scheduler.MultiStepLR(loss.optimizer, milestones=opt.lr_scheduler, gamma=0.1)
        self.scheduler_D = lr_scheduler.MultiStepLR(loss.optimizer_D, milestones=opt.lr_scheduler, gamma=0.1)
        self.scheduler_DC = lr_scheduler.MultiStepLR(loss.optimizer_DC, milestones=opt.lr_scheduler, gamma=0.1)

    def train(self, epoch):

        self.scheduler.step()
        self.scheduler_D.step()
        self.scheduler_DC.step()
        self.model.train()

        self.x.append(epoch)
        self.y2batch = [[], [], [], [], [], [], []]
        self.errorcnt = [[0, 0], [0, 0]]

        for batch, (rgb, cloth, labels) in enumerate(self.train_loader):
            if rgb.size()[0] != opt.batchid * opt.batchimage: continue
            rgb = rgb.to(opt.device)
            labels = labels.to(opt.device)
            cloth = cloth.to(opt.device)

            if opt.stage == 0:
                self.loss.optimizer.zero_grad()
                loss, loss_values, errors = self.loss(rgb, labels, cloth, batch, epoch)
                loss.backward()
                self.loss.optimizer.step()

            elif opt.stage == 1:
                self.loss.optimizer.zero_grad()
                self.loss.optimizer_DC.zero_grad()
                loss, loss_values, errors = self.loss(rgb, labels, cloth, batch, epoch)
                loss.backward()
                self.loss.optimizer.step()

            elif (opt.stage == 2) or (opt.stage == 3):
                self.loss.optimizer_D.zero_grad()
                self.loss.optimizer.zero_grad()
                self.loss.optimizer_DC.zero_grad()
                loss, loss_values, errors = self.loss(rgb, labels, cloth, batch, epoch)
                loss.backward()
                self.loss.optimizer.step()

            for i in range(len(loss_values)):
                self.y2batch[i].append(loss_values[i])
            for i in range(len(errors)):
                self.errorcnt[i][0] += errors[i][0]
                self.errorcnt[i][1] += errors[i][1]
        for i in range(len(loss_values)):
            self.y[i].append(sum(self.y2batch[i]) / len(self.y2batch[i]))
        for i in range(len(self.errorcnt)):
            self.errors[i].append((self.errorcnt[i][1] - self.errorcnt[i][0]) / self.errorcnt[i][1])

    def evaluate(self, save_path, epoch=None):

        self.model.eval()

        print('extract features, this may take a few minutes')
        qf = extract_feature(self.model, tqdm(self.query_loader)).numpy()
        gf = extract_feature(self.model, tqdm(self.test_loader)).numpy()

        def rank(dist):
            r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                    separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
            m_ap = mean_ap(
                dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

            return r, m_ap

        #         #########################   re rank##########################
        #         q_g_dist = np.dot(qf, np.transpose(gf))
        #         q_q_dist = np.dot(qf, np.transpose(qf))
        #         g_g_dist = np.dot(gf, np.transpose(gf))
        #         dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

        #         r, m_ap = rank(dist)

        #         print('[With    Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
        #               .format(m_ap, r[0], r[2], r[4], r[9]))

        #         #########################no re rank##########################
        dist = cdist(qf, gf)
        # from utils.combine_feature import combine_feature
        # dist = combine_feature(self.model, self.query_loader, self.test_loader)

        r, m_ap = rank(dist)

        '''added by Haorui'''
        if opt.stage == 0 and epoch is not None:
            if r[0] >= opt.s0_best_r1:
                opt.s0_best_r1 = r[0]
                opt.s0_best_epoch = epoch
        elif opt.stage == 1 and epoch is not None:
            if r[0] >= opt.s1_best_r1:
                opt.s1_best_r1 = r[0]
                opt.s1_best_epoch = epoch
        elif opt.stage == 2 and epoch is not None:
            if r[0] >= opt.s2_best_r1:
                opt.s2_best_r1 = r[0]
                opt.s2_best_epoch = epoch
        elif opt.stage == 3 and epoch is not None:
            if r[0] >= opt.s3_best_r1:
                opt.s3_best_r1 = r[0]
                opt.s3_best_epoch = epoch
        '''added by Haorui'''


        print('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f} rank20:{:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9], r[19]))

        with open(save_path, 'a') as f:
            if epoch is not None:
                f.write(
                    '[Without Re-Ranking] epoch: {:} mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f} rank20:{:.4f}\n'
                        .format(epoch, m_ap, r[0], r[2], r[4], r[9], r[19]))
            else:
                f.write(
                    '[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f} rank20:{:.4f}\n'
                        .format(m_ap, r[0], r[2], r[4], r[9], r[19]))


    def evaluate_multi_test(self, qf, save_path, test):

        self.model.eval()

        print('extract features, this may take a few minutes')
        gf = extract_feature(self.model, tqdm(self.test_loader)).numpy()

        def rank(dist):
            r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                    separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
            m_ap = mean_ap(
                dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

            return r, m_ap

        #         #########################   re rank##########################
        #         q_g_dist = np.dot(qf, np.transpose(gf))
        #         q_q_dist = np.dot(qf, np.transpose(qf))
        #         g_g_dist = np.dot(gf, np.transpose(gf))
        #         dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

        #         r, m_ap = rank(dist)

        #         print('[With    Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
        #               .format(m_ap, r[0], r[2], r[4], r[9]))

        #         #########################no re rank##########################
        dist = cdist(qf, gf)
        # from utils.combine_feature import combine_feature
        # dist = combine_feature(self.model, self.query_loader, self.test_loader)

        r, m_ap = rank(dist)

        print('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f} rank20:{:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9], r[19]))

        with open(save_path, 'a') as f:
            f.write(
                '[Without Re-Ranking] epoch: {:} mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f} rank20:{:.4f}\n'
                    .format(test, m_ap, r[0], r[2], r[4], r[9], r[19]))
        return r[0], r[2], r[4], r[9], r[19]

    def save_model(self, save_path):
        torch.save({
            'model_C': self.model.C.state_dict(),
            'model_G': self.model.G.state_dict(),
            'model_D': self.model.D.state_dict(),
            'model_DC': self.model.DC.state_dict(),
            'optimizer': self.loss.optimizer.state_dict(),
            'optimizer_D': self.loss.optimizer_D.state_dict(),
            'optimizer_DC': self.loss.optimizer_DC.state_dict()
        }, save_path)

    def load_model(self, load_path, last_epoch):
        checkpoint = torch.load(load_path)
        pretrained_dict = checkpoint['model_C']
        model_dict = self.model.C.state_dict()
        state_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.model.C.load_state_dict(model_dict, strict=False)
        if opt.stage != 1:
            pretrained_dict = checkpoint['model_DC']
            model_dict = self.model.DC.state_dict()
            state_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.model.DC.load_state_dict(model_dict, strict=False)
        if opt.stage == 2:
            checkpoint = torch.load('weights/isgan_prcc_stage0_best.pt')
            pretrained_dict = checkpoint['model_C']
            model_dict = self.model.C.id_encoder.state_dict()
            state_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.model.C.id_encoder.load_state_dict(model_dict, strict=False)
        if opt.stage == 3:
            self.model.G.load_state_dict(checkpoint['model_G'])
            self.model.D.load_state_dict(checkpoint['model_D'])
            self.loss.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        self.scheduler.last_epoch = last_epoch
        self.scheduler_D.last_epoch = last_epoch


def start():
    data = Data()
    model = Model()
    loss = Loss(model)
    main = Main(model, loss, data)

    if opt.mode == 'train':
        os.makedirs(opt.save_path, exist_ok=True)

        if opt.stage == 0:
            opt.start = 0
            opt.epoch = 300

        if opt.stage == 1:
            main.load_model(opt.save_path + opt.name + '_stage0_best.pt', 0)
            opt.start = 0
            opt.epoch = 300

        if opt.stage == 2:
            main.load_model(opt.save_path + opt.name + '_stage1_best.pt', 200)
            opt.start = 0
            opt.epoch = 200

        if opt.stage == 3:
            main.load_model(opt.save_path + opt.name + '_stage2_best.pt', 400)
            opt.start = 400
            opt.epoch = 600


        for epoch in range(opt.start + 1, opt.epoch + 1):

            print('\nepoch', epoch)
            main.train(epoch)

            if opt.stage == 0:
                rgb_CE_fig = plt.figure()
                plt.ylabel("RgbCE")
                plt.xlim(0, 400)
                plt.ylim(0, 7)
                plt.plot(main.x, main.y[0])
                rgb_CE_fig.savefig("./weights/Stage0_rgbCEloss.jpg")
                plt.close()

                ID_Error = plt.figure()
                plt.ylabel("ID Error")
                plt.xlim(0, 400)
                plt.ylim(0, 1)
                plt.plot(main.x, main.errors[0])
                ID_Error.savefig("./weights/Stage0_IDError.jpg")
                plt.close()

            if opt.stage == 1:

                cloth_CE_fig = plt.figure()
                plt.ylabel("Cloth_CE")
                plt.xlim(0, 300)
                plt.ylim(0, 7)
                plt.plot(main.x, main.y[0])
                cloth_CE_fig.savefig("./weights/Stage1_cloth_CE_loss.jpg")
                plt.close()

                Cloth_Error = plt.figure()
                plt.ylabel("Cloth_Error")
                plt.xlim(0, 300)
                plt.ylim(0, 1)
                plt.plot(main.x, main.errors[1])
                Cloth_Error.savefig("./weights/Stage1_ClothError.jpg")
                plt.close()

                D_I_fig = plt.figure()
                plt.ylabel("D_I")
                plt.xlim(0, 300)
                plt.ylim(0, 7)
                plt.plot(main.x, main.y[1])
                D_I_fig.savefig("./weights/Stage1_D_I_loss.jpg")
                plt.close()

                D_C_fig = plt.figure()
                plt.ylabel("D_C")
                plt.xlim(0, 300)
                plt.ylim(0, 7)
                plt.plot(main.x, main.y[2])
                D_C_fig.savefig("./weights/Stage1_D_C_loss.jpg")
                plt.close()

            if opt.stage == 2:

                D_Loss_fig = plt.figure()
                plt.plot(main.x, main.y[0])
                plt.ylabel("D_Loss")
                plt.xlim(0, 200)
                D_Loss_fig.savefig("./weights/Stage2_D_Loss.jpg")
                plt.close()

                G_Loss_fig = plt.figure()
                plt.plot(main.x, main.y[1])
                plt.ylabel("G_Loss")
                plt.xlim(0, 200)
                G_Loss_fig.savefig("./weights/Stage2_G_Loss.jpg")
                plt.close()

                KL_Loss_fig = plt.figure()
                plt.plot(main.x, main.y[2])
                plt.ylabel("KL_Loss")
                plt.xlim(0, 200)
                KL_Loss_fig.savefig("./weights/Stage2_KL_Loss.jpg")
                plt.close()

            if opt.stage == 3:

                rgb_CE_fig = plt.figure()
                plt.ylabel("RgbCE")
                plt.xlim(400, 600)
                plt.ylim(0, 7)
                plt.plot(main.x, main.y[0])
                rgb_CE_fig.savefig("./weights/Stage3_rgbCE_loss.jpg")
                plt.close()

                ID_Error = plt.figure()
                plt.ylabel("ID Error")
                plt.xlim(400, 600)
                plt.ylim(0, 1)
                plt.plot(main.x, main.errors[0])
                ID_Error.savefig("./weights/Stage3_IDError.jpg")
                plt.close()

                Cloth_CE_fig = plt.figure()
                plt.ylabel("ClothCE")
                plt.xlim(400, 600)
                plt.ylim(0, 7)
                plt.plot(main.x, main.y[1])
                Cloth_CE_fig.savefig("./weights/Stage3_ClothCE_loss.jpg")
                plt.close()

                Cloth_Error = plt.figure()
                plt.ylabel("Cloth Error")
                plt.xlim(400, 600)
                plt.ylim(0, 1)
                plt.plot(main.x, main.errors[1])
                Cloth_Error.savefig("./weights/Stage3_ClothError,jpg")
                plt.close()


                D_Loss_fig = plt.figure()
                plt.plot(main.x, main.y[2])
                plt.ylabel("D_Loss")
                plt.xlim(400, 600)
                D_Loss_fig.savefig("./weights/Stage3_D_Loss.jpg")
                plt.close()

                G_Loss_fig = plt.figure()
                plt.plot(main.x, main.y[3])
                plt.ylabel("G_Loss")
                plt.xlim(400, 600)
                G_Loss_fig.savefig("./weights/Stage3_G_Loss.jpg")
                plt.close()

                D_I_fig = plt.figure()
                plt.ylabel("D_I")
                plt.xlim(400, 600)
                plt.ylim(0, 7)
                plt.plot(main.x, main.y[4])
                D_I_fig.savefig("./weights/Stage3_D_I_loss.jpg")
                plt.close()

                D_C_fig = plt.figure()
                plt.ylabel("D_C")
                plt.xlim(400, 600)
                plt.ylim(0, 7)
                plt.plot(main.x, main.y[5])
                D_C_fig.savefig("./weights/Stage3_D_C_loss.jpg")
                plt.close()

                KL_Loss_fig = plt.figure()
                plt.plot(main.x, main.y[6])
                plt.ylabel("KL_Loss")
                plt.xlim(400, 600)
                KL_Loss_fig.savefig("./weights/Stage3_KL_Loss.jpg")
                plt.close()


            # if epoch % 100 == 0:
            #     os.makedirs(opt.save_path, exist_ok=True)
            #     weight_save_path = opt.save_path + opt.name + \
            #                        '_stage{}_{:03d}.pt'.format(opt.stage, epoch)
            #     main.save_model(weight_save_path)
            #     main.evaluate(opt.save_path + opt.name + '_accr.txt', epoch)
            # if opt.stage == 3 and epoch % 25 == 0:
            #     weight_save_path = opt.save_path + opt.name + \
            #                        '_stage{}_{:03d}.pt'.format(opt.stage, epoch)
            #     main.save_model(weight_save_path)
            #     main.evaluate(opt.save_path + opt.name + '_accr.txt', epoch)

            '''modified by Haorui'''
            if epoch % 25 == 0:
                if opt.stage == 0 or opt.stage == 3:
                    main.evaluate(opt.save_path + opt.name + '_accr.txt', epoch)
                if  opt.stage == 0 or \
                    opt.stage == 1 or opt.stage == 2 or \
                   (opt.stage == 3 and opt.s3_best_epoch == epoch):
                    os.makedirs(opt.save_path, exist_ok=True)
                    weight_save_path = opt.save_path + opt.name + \
                                            '_stage{}_best.pt'.format(opt.stage)
                    if os.path.exists(weight_save_path):
                        os.remove(weight_save_path)
                    main.save_model(weight_save_path)
            '''modified by Haorui'''



    if opt.mode == 'evaluate':
        print('start evaluate')
        main.load_model(opt.weight, 0)
        main.evaluate(opt.save_path + opt.name + '_accr.txt')

def multi_test():
    model = Model()
    loss = Loss(model)
    data = Data()
    main = Main(model, loss, data)
    print('start evaluate')
    main.load_model(opt.weight, 0)
    main.model.eval()
    qf = extract_feature(main.model, tqdm(main.query_loader)).numpy()
    rank1 = []
    rank3 = []
    rank5 = []
    rank10 = []
    rank20 = []
    for i in range(10):
        data = Data(i)
        main = Main(model, loss, data)
        print('start evaluate', i)
        main.load_model(opt.weight, 0)
        r1, r3, r5, r10, r20 = main.evaluate_multi_test(qf, opt.save_path + opt.name + '_accr.txt', i)
        rank1.append(r1)
        rank3.append(r3)
        rank5.append(r5)
        rank10.append(r10)
        rank20.append(r20)
    r1 = np.mean(rank1)
    r3 = np.mean(rank3)
    r5 = np.mean(rank5)
    r10 = np.mean(rank10)
    r20 = np.mean(rank20)

    print('[Average] rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f} rank20:{:.4f}'
          .format(r1, r3, r5, r10, r20))

    with open(opt.save_path + opt.name + '_accr.txt', 'a') as f:
        f.write(
            '[Average] rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f} rank20:{:.4f}'
          .format(r1, r3, r5, r10, r20))


if __name__ == '__main__':
    opt.mode = 'train'
    # print(opt.device)
    # opt.mode = 'evaluate'
    # opt.weight = 'weights/isgan_stage3_400.pt'
    # start()
    # opt.mode = 'train'
    # opt.stage = 0
    # start()
    # opt.stage = 1
    # start()
    opt.stage = 2
    start()
    opt.stage = 3
    start()
    # opt.mode = 'evaluate'
    # opt.weight = 'weights/isgan_stage11_600.pt'
    # start()
    # multi_test()
