'''
Target: an Encoder-Decoder structure to generate new density maps with MSE Loss and Set Loss
'''

import sys, os, warnings
from utils import save_checkpoint
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import argparse, json, cv2, time
import dataset_xie as dataset
from model import * 
from set_loss_compute import *
from PIL import Image
from set_matcher import build_matcher
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

parser = argparse.ArgumentParser(description='EncoDeco_MSE_SET')
parser.add_argument('train_json', metavar='TRAIN', help='path to train json')
parser.add_argument('test_json', metavar='TEST', help='path to test json')
parser.add_argument('--pre_g', '-p_g', metavar='PRETRAINED_NET', default=None, type=str, help='path to the pretrained EncoDeco model')
parser.add_argument('gpu',metavar='GPU', type=str, help='GPU id to use.')
parser.add_argument('task',metavar='TASK', type=str, help='task id to use.')

def main():
    global args, best_prec1
    best_dist1 = 1e6
    best_prec1 = 1e6
    args = parser.parse_args()
    args.original_lr = 1e-4
    args.lr = 1e-4
    args.batch_size    = 1
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 200
    args.steps         = [-1,200,400,450] # adjust learning rate
    args.scales        = [1, 0.1, 0.01, 0.001]
    args.workers = 8
    args.seed = 0 
    args.print_freq = 20

    img_shape = (256, 256) # for the VGG_dataset only

    args.arch = args.task + 'EnDe100_mse_set_'
    with open(args.train_json, 'r') as outfile:
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:
        val_list = json.load(outfile)

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.empty_cache()

    model = Generator().cuda()
    
    criterion1 = torch.nn.MSELoss(size_average=False, reduction='none').cuda()
    matcher = build_matcher()
    criterion2 = SetCriterion(matcher=matcher).cuda()  
    criterion3 = torch.nn.L1Loss(size_average=False, reduction='none').cuda() 

    g_opti = torch.optim.Adam(model.parameters(), lr = args.lr)

    #For pretrained model setting
    if args.pre_g:
        if os.path.isfile(args.pre_g):
            print("=> loading the model checkpoint '{}'".format(args.pre_g))
            checkpoint_g = torch.load(args.pre_g)
            args.start_epoch = checkpoint_g['epoch']
            best_prec1 = checkpoint_g['best_prec1']
            model.load_state_dict(checkpoint_g['g_state_dict'])
            g_opti.load_state_dict(checkpoint_g['g_optimizer'])
            print("==> Pre-trained model loaded checkpoint '{}' (epoch {})".format(args.pre_g, checkpoint_g['epoch']))
        else:
            print("==> no pre-trained model checkpoint found at '{}'".format(args.pre_g))
    
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(g_opti, epoch)
        train(train_list, model, criterion1, criterion2, criterion3, g_opti, epoch)
        prec1, dist1 = validate(val_list, model)

        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        line1 = '*** Xie Best _MAE_ {mae:.3f} ***'.format(mae=best_prec1)


        with open('logs/Log-{}_{}.txt'.format(time_stp, args.arch), 'a+') as flog:
            print(line1)
            flog.write('{}\n'.format(line1))
        
        save_checkpoint({
            'epoch': epoch + 1,
            'g_arch': args.pre_g,
            'g_state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'g_optimizer' : g_opti.state_dict()
        }, is_best, args.task)


def train(train_list, model, criterion1, criterion2, criterion3, g_opti, epoch): 
    # rest all the parameters
    g_losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]),
                       train=True),
        batch_size=args.batch_size,
        num_workers=8)
    print('epoch %d, processed %d samples, lr %.10f, dsf_%s' % (epoch, epoch * len(train_loader.dataset), args.lr, args.task.split('dsf')[-1][:3]))
    model.train()
    end = time.time()

    for i, (img, target, dots, f_name) in enumerate(train_loader):
        data_time.update(time.time() - end)
        img = img.cuda()
        img = Variable(img)
        target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
        target = Variable(target)
        dots = dots.type(torch.FloatTensor).unsqueeze(0).cuda()
        dots = Variable(dots)

        #reset grads 
        g_opti.zero_grad()
        g_dMap, g_dots = model(img)
        g_dots = g_dots.view(1, 1, -1, 2)
        mse_loss = criterion1(g_dMap, target)
        set_loss = criterion2(g_dots, dots)
        c_loss = criterion3(target.sum(), g_dMap.sum())

        # these weights can be changed during training.
        par1 = 1.0 # mse loss
        par2 = 0.0001 # set loss
        par3 = 0.01 # count loss
         
        g_loss = par1*mse_loss + par2*set_loss + par3*c_loss 
        g_losses.update(g_loss.item(), img.size(0)) 
        g_loss.backward()
        g_opti.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            with open('logs/Log-{}_{}.txt'.format(time_stp, args.arch), 'a+') as flog:
                line = 'Xie Epoch: [{0}][{1}/{2}]  ' \
                       'Loss: {g_loss.val:.3f} ({g_loss.avg:.3f})  ' 'MSELoss: {mse_loss:.3f}  ' 'SetLoss: {set_loss:.3f}  ' 'wMSE: {par1:.2f}  ' 'wSET: {par2:.4f} '.format(epoch, i, len(train_loader), g_loss=g_losses, mse_loss=par1*mse_loss, set_loss=par2*set_loss, par1=par1, par2=par2)
                print(line)
                flog.write('{}\n'.format(line))


def validate(val_list, model):
    print ('\n|||---------begin test---------|||')
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(val_list,
                    shuffle=False,
                    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]),        
                    train=False), 
        batch_size=args.batch_size,
        num_workers=8)

    model.eval()
    val_mae = 0.0
    val_dist = 0.0

    for i, (img, target, dots, f_name) in enumerate(test_loader):
        img = img.cuda()
        img = Variable(img)
        v_dMap, v_dots = model(img)
        v_dots = v_dots.view(1, 1, -1, 2).data.cpu().clone().detach().numpy()
        v_dots = v_dots.squeeze()
        dots = dots.squeeze()
        cost_m = cdist(v_dots, dots, 'minkowski', p=1)
        src_ind, tgt_ind = linear_sum_assignment(cost_m)

        d_sum = v_dMap.data.cpu().clone().detach().numpy() 
        val_mae += abs(d_sum.sum() - target.sum()) 
        val_dist += cost_m[src_ind, tgt_ind].sum() #/ len(target)

    val_mae = val_mae/len(test_loader)
    val_dist = val_dist/len(test_loader)  
    line = '**** Xie MAE {val_mae:.3f} **** MEAN_DIST {val_dist:.3f}'.format(val_mae=val_mae, val_dist=val_dist)
    with open('logs/Log-{}_{}.txt'.format(time_stp, args.arch), 'a+') as flog:
        print(line)
        flog.write('{}\n'.format(line))

    return val_mae, val_dist

def adjust_learning_rate(optimizer, epoch):
    args.lr = args.original_lr 
    for i in range(len(args.steps)):
        scale = args.scales[i] if i < len(args.scales) else 1 
        if epoch >= args.steps[i]: 
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    

if __name__ == '__main__':
    time_stp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    main()