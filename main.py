import argparse
import os
import random
import time
import datetime
import math
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from carla_net import CarlaNet, FinalNet
from carla_loader import CarlaH5Data
from helper import AverageMeter, save_checkpoint


parser = argparse.ArgumentParser(description='Carla CIL')
parser.add_argument('-j', '--worker', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--speed_weight', default=0.1, type=float)
parser.add_argument('--branch_weight', default=0.1, type=float)
parser.add_argument('--id', default="test", type=str)
parser.add_argument('--train_dir', default="", type=str, metavar='PATH', help='training dataset')
parser.add_argument('--eval_dir', default="", type=str, metavar='PATH', help='evaluation dataset')
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number(useful on restarts')
parser.add_argument('-b', '--batch_size', default=1, type=int, metavar='N', help='mini_batch size(default:256)')
parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, metavar="LR", help='initial learning rate')
parser.add_argument('--lr-gamma', default=0.5, type=float, help='learning rate gamma')
parser.add_argument('--lr-step', default=10, type=int, help='learning rate step size')
parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W')
parser.add_argument('--print_frequency', default=10, type=int, metavar='N')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint(default: NONE')
parser.add_argument('--evaluate', '-e', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--evaluation_log', default='', type=str, metavar='PATH', help='path to log evaluation results')
parser.add_argument('--world_size', default=1, type=int, help='number of distributed process')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use')
parser.add_argument('--net_structure', default=2, type=int, help='Network structure 1/2/3/4')


def output_log(output_str, logger=None):
    print ("[{}]:{}".format(datetime.datetime.now(), output_str))
    if logger is not None:
        logger.critical("[{}]:{}".format(datetime.datetime.now(), output_str))


def log_args(logger):
    pass


def main():
    global args
    args = parser.parse_args()
    log_dir = os.path.join("./", "logs", args.id)
    run_dir = os.path.join("./", "runs", args.id)
    save_weight_dir = os.path.join("./save_models", args.id)
    os.makedirs(log_dir)
    os.makedirs(save_weight_dir)

    logging.basicConfig(filename=os.path.join(log_dir, "carla_training.log"), level=logging.ERROR)
    tsbd = SummaryWriter(log_dir=run_dir)
    log_args(logging)

    # the setting of seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        output_log('You have chosen to seed training.'
                   'This will turn on the CUDNN deterministic setting'
                   'which can slow down your training considerably'
                   'you may see unexpected behavior when restarting'
                   'from checkpoints', logger=logging)

    # the setting of gpu
    if args.gpu is not None:
        output_log('You have chosen a specific GPU which will completely disable data parallelism', logger=logging)

    # the setting of parallelism
    args.distributed = args.world_size > 1  # judgement
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=0)

    model = FinalNet(args.net_structure)
    criterion = nn.MSELoss()

    # load the carla_net parameters
    model.carla_net.load_state_dict(torch.load("./save_models/new_structure_best.pth")['state_dict'])

    # tensorboard usage
    tsbd.add_graph(model, (torch.zeros(1, 3, 88, 200), torch.zeros(1, 1)))

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    optimizer = optim.Adam(model.uncertain_net.parameters(), args.lr, betas=(0.7, 0.85))
    # the adjustment of the lr
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    best_prec = math.inf

    # optionally resume from a checkpoint
    if args.resume:
        args.resume = os.path.join(save_weight_dir, args.resume)
        if os.path.isfile(args.resume):
            output_log("=> loading checkpoint'{}'".format(args.resume), logging)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
            best_prec = checkpoint['best_prec']
            output_log("=> loaded checkpoint'{}'".format(args.resume), logging)
        else:
            output_log("=> no checkpoint found at '{}'".format(args.resume), logging)

    cudnn.benchmark = True   # where to add this line of code

    carla_data = CarlaH5Data()      #  unfinished

    train_loader = carla_data.loaders["train"]
    eval_loader = carla_data.loaders["eval"]

    # setting of evaluation
    if args.evaluate:
        args.id = args.id + "_test"
        if not os.path.isfile(args.resume):
            output_log("=> no checkpoint found at '{}'".format(args.resume), logging)
            return
        if args.evaluate_log =="":
            output_log("=> please set evaluate log path with --evaluate-log<log-path>")

        evaluate(eval_loader, model, criterion, 0, tsbd)
        return

    for epoch in range(args.start_epoch, args.epochs):   # (0,90)
        lr_scheduler.step()         # lr_scheduler needs to step when using
        branch_losses, speed_losses, losses = train(train_loader, model, criterion, optimizer, epoch, tsbd)

        prec = evaluate(eval_loader, model, criterion, epoch, tsbd)

        # rememeber best prec@1 and save checkpoint
        is_best = prec < best_prec
        best_prec = min(prec, best_prec)
        save_checkpoint(
            {'epoch': epoch+1,
             'state_dict': model.state_dict(),
             'best_prec': best_prec,
             'scheduler': lr_scheduler.state_dict(),
             'optimizer': optimizer.state_dict()},
            args.id,
            is_best,
            os.path.join(save_weight_dir, "{}_{}.pth".format(epoch+1, args.id))
        )


def train(loader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    uncertain_losses = AverageMeter()
    ori_losses = AverageMeter()
    branch_losses = AverageMeter()
    speed_losses = AverageMeter()
    uncertain_control_means = AverageMeter()
    uncertain_speed_means = AverageMeter()

    # train to train mode
    model.train()
    end = time.time()
    step = epoch*len(loader)
    for i, (img, speed, target, mask) in enumerate(loader):
        data_time.update(time.time()-end)

        # if args.gpu is not None
        img = img.cuda(args.gpu, non_blocking=True)
        speed = speed.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        mask = mask.cuda(args.gpu, non_blocking=True)

        if args.net_structure != 1:
            branches_out, pred_speed, log_var_control, log_var_speed = model(img, speed)

            branch_square = torch.pow((branches_out-target), 2)
            branch_loss = torch.mean((torch.exp(-log_var_control)*branch_square+log_var_speed)*0.5*mask)*4

            speed_square = torch.pow((pred_speed-speed), 2)
            speed_loss = torch.mean((torch.exp(-log_var_speed)*speed_square+log_var_speed)*0.5)

            uncertain_loss = args.branch_weight*branch_loss+args.speed_weight*speed_loss

            with torch.no_grad():           # the wrapped section's gradient won't be tracked when in operation
                ori_loss = args.branch_weight * torch.mean(branch_square*mask*4)\
                    + args.speed_weight * torch.mean(speed_square)
                uncertain_control_mean = torch.mean(torch.exp(log_var_control) * mask * 4)
                uncertain_speed_mean = torch.mean(torch.exp(log_var_speed))

                ori_losses.update(ori_loss.item(), args.batch_size)
                uncertain_control_means.update(uncertain_control_mean.item(), args.bacth_size)
                uncertain_speed_means.update(uncertain_speed_mean.item(), args.bacth_size)

        else:
            branches_out, pred_speed = model(img, speed)
            branch_loss = criterion(branches_out*mask, target) * 4
            speed_loss = criterion(pred_speed, speed)
            uncertain_loss = args.branch_weight * branch_loss + args.speed_weight * speed_loss

        uncertain_losses.update(uncertain_loss.item(), args.batch_size)         # the usage of the update function
        branch_losses.update(branch_loss.item(), args.batch_size)
        speed_losses.update(speed_loss.item(), args.batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        model.zero_grad()
        uncertain_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i+1 ==len(loader):
            writer.add_scalar('train/branch_loss', branch_losses.val, step + i)
            writer.add_scalar('train/speed_loss', speed_losses.val, step + i)
            writer.add_scalar('train/uncertain_loss', uncertain_losses.val, step + i)
            writer.add_scalar('train/ori_loss', ori_losses.val, step + i)
            writer.add_scalar('train/control_uncertain', uncertain_control_means.val, step + i)
            writer.add_scalar('train/speed_uncertain', uncertain_speed_means.val, step + i)

            output_log(
                'Epoch:[{0}][{1}/{2}]\t'
                'Time {bacth_time.val:.3f}({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f}({data_time.avg:.3f})\t'
                'Branch loss {branch_loss.val:.3f}({branch_loss.avg:.3f})\t'
                'Speed loss {Speed_loss.val:.3f}({Speed_loss.avg:.3f})\t'
                'Uncertain loss {uncertain_loss.val:.4f}({uncertain_loss.avg:.4f})\t'
                'Ori Loss {ori_loss.val:.4f}({ori_loss.avg:.4f})\t'
                'Control Uncertain {control_uncertain.val:.4f}({control_uncertain.avg:.4f})\t'
                'Speed Uncertain {speed_uncertain.val:.4f}({speed_uncertain.avg:.4f})\t'
                .format(
                    epoch+1, i, len(loader), batch_time=batch_time,
                    data_time=data_time,
                    branch_loss=branch_losses,
                    speed_loss=speed_losses,
                    uncertain_loss=uncertain_losses,
                    ori_loss=ori_losses,
                    control_uncertain=uncertain_control_means,
                    speed_uncertain=uncertain_control_means,
                    ), logging)

    return branch_losses.avg, speed_losses.avg, uncertain_losses.avg


def evaluate(loader, model, criterion, epoch, writer):
    batch_time = AverageMeter()
    uncertain_losses = AverageMeter()
    ori_losses = AverageMeter()
    uncertain_control_means = AverageMeter()
    uncertain_speed_means = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (img, speed, target, mask) in enumerate(loader):
            img = img.cuda(args.cpu, non_blocking=True)
            speed = speed.cuda(args.cpu, non_blocking=True)
            target = target.cuda(args.cpu, non_blocking=True)
            mask = mask.cuda(args.cpu, non_blocking=True)

            branches_out, pred_speed, log_var_control, log_var_speed = model(img, speed)

            mask_out = branches_out * mask
            ori_branch_loss = criterion(mask_out, target) * 4
            ori_speed_loss = criterion(pred_speed, speed)

            branch_loss = torch.mean((torch.exp(-log_var_control)
                                      * torch.pow((branches_out - target), 2)
                                      + log_var_control) * 0.5 * mask) * 4

            speed_loss = torch.mean((torch.exp(-log_var_speed)
                                     * torch.pow((pred_speed - speed), 2)
                                     + log_var_speed) * 0.5)

            uncertain_loss = args.branch_weight*branch_loss + \
                    args.speed_weight*speed_loss
            ori_loss = args.branch_weight*ori_branch_loss + \
                    args.speed_weight*ori_speed_loss

            uncertain_control_mean = torch.mean(torch.exp(log_var_control) * mask * 4)
            uncertain_speed_mean = torch.mean(torch.exp(log_var_speed))

            uncertain_losses.update(uncertain_loss.item(), args.batch_size)
            ori_losses.update(ori_loss.item(), args.batch_size)
            uncertain_control_means.update(uncertain_control_mean.item(),
                                           args.batch_size)
            uncertain_speed_means.update(uncertain_speed_mean.item(),
                                         args.batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        writer.add_scalar('eval/uncertain_loss', uncertain_losses.avg, epoch+1)
        writer.add_scalar('eval/origin_loss', ori_losses.avg, epoch+1)
        writer.add_scalar('eval/control_uncertain', uncertain_control_means.avg, epoch+1)
        writer.add_scalar('eval/speed_uncertain', uncertain_speed_means.avg, epoch+1)

        output_log(
            'Epoch Test: [{0}]\t'
            'Time {batch_time.avg:.3f}\t'
            'Uncertain Loss {uncertain_loss.avg:.4f}\t'
            'Original Loss {ori_loss.avg:.4f}\t'
            'Control Uncertain {control_uncertain.avg:.4f}\t'
            'Speed Uncertain {speed_uncertain.avg:.4f}\t'
            .format(
                epoch + 1, batch_time=batch_time,
                uncertain_loss=uncertain_losses,
                ori_loss=ori_losses,
                control_uncertain=uncertain_control_means,
                speed_uncertain=uncertain_speed_means,
                ), logging)

    return uncertain_losses.avg


if __name__ == '__main__':
    main()






