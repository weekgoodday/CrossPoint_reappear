from __future__ import print_function
import os
import random
import argparse
import torch
import math
import numpy as np
import wandb
from lightly.loss.ntx_ent_loss import NTXentLoss
import time
from sklearn.svm import SVC

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet18
from torch.utils.data import DataLoader

from datasets.data import ShapeNetRender, ModelNet40SVM, ShapeNetRender_Update
from models.dgcnn import DGCNN, ResNet, DGCNN_partseg
from util import IOStream, AverageMeter
from models.intactnn import HNet, dgdtNet, IntactNet
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')


def train(args, io):
    wandb.init(project="CrossPoint", name=args.exp_name)

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # xlin: original code
    # train_loader = DataLoader(ShapeNetRender(transform, n_imgs=2), num_workers=0,
    #                           batch_size=args.batch_size, shuffle=True, drop_last=True)
    dataset = ShapeNetRender_Update(transform, n_imgs=2)
    train_loader = DataLoader(dataset, num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda:1" if args.cuda else "cpu")

    # Try to load models
    if args.model == 'dgcnn':
        point_model = DGCNN(args).to(device)
    elif args.model == 'dgcnn_seg':  # DGCNN_seg模型分pretrain和正式工作两部分，pretrain只到
        point_model = DGCNN_partseg(args).to(device)
    else:
        raise Exception("Not implemented")

    img_model = ResNet(resnet50(), feat_dim=2048)
    img_model = img_model.to(device)

    intact_model = IntactNet(len(dataset))

    wandb.watch(point_model)

    if args.resume:  # 输入是False model不会报错
        # xlin: change the model to point_model to avoid warning at xiao's local computer
        point_model.load_state_dict(torch.load(args.model_path))
        print("Model Loaded !!")

    parameters = list(point_model.parameters()) + list(img_model.parameters())

    if args.use_sgd:  # 输入是False model不会报错
        # xlin: change the model to point_model to avoid warning at xiao's local computer
        print("Use SGD")
        opt = optim.SGD(point_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-6)
    else:  # 默认是这个 parameters是有定义的
        print("Use Adam")
        opt = optim.Adam(parameters, lr=args.lr, weight_decay=1e-6)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0, last_epoch=-1)
    criterion = NTXentLoss(temperature=0.1).to(device)

    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        lr_scheduler.step()

        ####################
        # Train 这里的train_loader还是ShapeNet数据集
        ####################
        train_losses = AverageMeter()
        train_imid_losses = AverageMeter()
        train_cmid_losses = AverageMeter()
        train_imid1_losses = AverageMeter()

        point_model.train()
        img_model.train()
        wandb_log = {}
        print(f'Start training epoch: ({epoch}/{args.epochs})')
        for i, ((data_t1, data_t2), (imgs0, imgs1), idxs) in enumerate(train_loader):
            data_t1, data_t2, imgs0, imgs1 = data_t1.to(device), data_t2.to(device), imgs0.to(device), imgs1.to(device)
            batch_size = data_t1.size()[0]

            opt.zero_grad()
            data = torch.cat((data_t1, data_t2))
            data = data.transpose(2, 1).contiguous()
            _, point_feats, _ = point_model(data)

            # print('imgs0', imgs0.shape)
            # print('imgs1', imgs1.shape)
            # exit()
            img_feats0 = img_model(imgs0)
            img_feats1 = img_model(imgs1)

            point_t1_feats = point_feats[:batch_size, :]
            point_t2_feats = point_feats[batch_size:, :]

            # # xlin: purposely added, do not delete
            # # # add new code here
            # # # Q1. how to update
            # # # Q2. data augmentation for img_feat fixed
            # # # meeting with haotian
            # # # probably the following code
            # #
            # # obtain intact feature
            # aver_pnt_feat = (point_t1_feats + point_t2_feats) / 2
            # aver_img_feat = (img_feats0 + img_feats1) / 2
            # itct_pnt_feat, itct_img_feat = intact_model(idxs)
            #
            # # calculate similarity
            # intact_sim = ...
            # loss_crsmodal1 = intact_sim(aver_pnt_feat, itct_pnt_feat)
            # loss_crsmodal2 = intact_sim(aver_img_feat, itct_img_feat)
            # loss_cmid = loss_crsmodal1 + loss_crsmodal2
            # #
            # # # add new code here

            loss_imid = criterion(point_t1_feats, point_t2_feats)
            point_feats = torch.stack([point_t1_feats, point_t2_feats]).mean(dim=0)

            loss_imid1 = criterion(img_feats0, img_feats1)
            img_feats = torch.stack([img_feats0, img_feats1]).mean(dim=0)
            loss_cmid = criterion(point_feats, img_feats)

            total_loss = loss_imid + loss_cmid + loss_imid1
            total_loss.backward()
            opt.step()

            train_losses.update(total_loss.item(), batch_size)
            train_imid_losses.update(loss_imid.item(), batch_size)
            train_cmid_losses.update(loss_cmid.item(), batch_size)
            train_imid1_losses.update(loss_imid1.item(), batch_size)


            if i % args.print_freq == 0:  # 这是batch数
                print('Epoch (%d), Batch(%d/%d), loss: %.6f, imid loss: %.6f, imid1 loss: %.6f, cmid0 loss: %.6f ' % (
                    epoch, i, len(train_loader), train_losses.avg, train_imid_losses.avg, train_imid1_losses.avg, train_cmid_losses.avg))

        wandb_log['Train Loss'] = train_losses.avg
        wandb_log['Train IMIDp Loss'] = train_imid_losses.avg
        wandb_log['Train IMIDi Loss'] = train_imid1_losses.avg
        wandb_log['Train CMID Loss'] = train_cmid_losses.avg
        

        outstr = 'Train %d, loss: %.6f' % (epoch, train_losses.avg)
        io.cprint(outstr)

        # Testing 这里的DataLoader是ModelNet40SVM分类数据集

        train_val_loader = DataLoader(ModelNet40SVM(partition='train', num_points=1024), batch_size=128, shuffle=True)
        test_val_loader = DataLoader(ModelNet40SVM(partition='test', num_points=1024), batch_size=128, shuffle=True)

        feats_train = []
        labels_train = []
        point_model.eval()

        for i, (data, label) in enumerate(train_val_loader):
            labels = list(map(lambda x: x[0], label.numpy().tolist()))
            data = data.permute(0, 2, 1).to(device)
            with torch.no_grad():
                feats = point_model(data)[2]
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_train.append(feat)
            labels_train += labels

        feats_train = np.array(feats_train)
        labels_train = np.array(labels_train)

        feats_test = []
        labels_test = []

        for i, (data, label) in enumerate(test_val_loader):
            labels = list(map(lambda x: x[0], label.numpy().tolist()))
            data = data.permute(0, 2, 1).to(device)
            with torch.no_grad():
                feats = point_model(data)[2]
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_test.append(feat)
            labels_test += labels

        feats_test = np.array(feats_test)
        labels_test = np.array(labels_test)

        model_tl = SVC(C=0.1, kernel='linear')  # 只是对提的特征进行线性分类
        model_tl.fit(feats_train, labels_train)
        test_accuracy = model_tl.score(feats_test, labels_test)  # 可以得到经过训练train之后的model_tl的分数，看提的特征好不好
        wandb_log['Linear Accuracy'] = test_accuracy
        print(f"Linear Accuracy : {test_accuracy}")

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            print('==> Saving Best Model...')
            save_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                                     'best_model_epoch{epoch}.pth'.format(epoch=epoch))
            torch.save(point_model.state_dict(), save_file)

            save_img_model_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                                               'img_model_best_epoch{epoch}.pth'.format(epoch=epoch))
            torch.save(img_model.state_dict(), save_img_model_file)

        if epoch % args.save_freq == 0:
            print('==> Saving...')
            save_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                                     'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(point_model.state_dict(), save_file)

        wandb.log(wandb_log)

    print('==> Saving Last Model...')
    save_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                             'ckpt_epoch{epoch}_last.pth'.format(epoch=epoch))
    torch.save(point_model.state_dict(), save_file)
    save_img_model_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                                       'img_model_epoch{epoch}_last.pth'.format(epoch=epoch))
    torch.save(img_model.state_dict(), save_img_model_file)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',  # 默认exp会和文件夹名有关
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn', 'dgcnn_seg'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')  # 输入是100
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action="store_true", help='Use SGD')  # store_true代表有这个参数就True 不过和resume一样 都没有
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')  # 输入是15
    parser.add_argument('--resume', action="store_true", help='resume from checkpoint')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')  # 改成了5
    parser.add_argument('--print_freq', type=int, default=1, help='print frequency')  # 200指的是每200个batch打印一次
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:  # evaluate是False 只会运行train
        train(args, io)
    else:  # test在别的地方运行
        test(args, io)
