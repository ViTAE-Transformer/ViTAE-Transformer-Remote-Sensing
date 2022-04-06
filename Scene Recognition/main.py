# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('Remote sensing training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=False, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--data-path', default=None, type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # batch
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    # epoch
    parser.add_argument('--epochs', type=int, help="epochs")
    # distributed training
    parser.add_argument("--local_rank", default=0, type=int, help='local rank for DistributedDataParallel')
    # dataset
    parser.add_argument('--dataset', default=None, type=str, choices=['millionAID','ucm','aid','nwpuresisc'], help='type of dataset')
    # ratio
    parser.add_argument('--ratio', default=None, type=int, help='trn tes ratio')
    # model
    parser.add_argument('--model', default=None, type=str, choices=['resnet','vit','swin','vitae_win'], help='type of model')
    # input size
    parser.add_argument("--img_size", default=None, type=int, help='size of input')
    # exp_num
    parser.add_argument("--exp_num", default=0, type=int, help='number of experiment times')
    # tag
    parser.add_argument("--split", default=None, type=int, help='id of split')
    # lr
    parser.add_argument("--lr", default=None, type=float, help='learning rate')
    # wd
    parser.add_argument("--weight_decay", default=None, type=float, help='learning rate')
    # gpu_num
    parser.add_argument("--gpu_num", default=None, type=int, help='id of split')
    # optimizer
    parser.add_argument('--optimizer', default='adamw', type=str, choices=['adamw','sgd'], help='type of optimizer')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(args,config):

    config.defrost() # 释放cfg

    if config.MODEL.TYPE == 'resnet':
        config.MODEL.NAME = 'resnet_50_224'
    elif config.MODEL.TYPE == 'swin':
        config.MODEL.NAME = 'swin_tiny_patch4_window7_224'
    elif config.MODEL.TYPE == 'vitae_win':
        config.MODEL.NAME = 'ViTAE_Window_NoShift_12_basic_stages4_14_224'

    if config.DATA.DATASET == 'millionAID':
        config.DATA.DATA_PATH = '../Dataset/millionaid/'
        config.MODEL.NUM_CLASSES = 51
    elif config.DATA.DATASET == 'ucm':
        config.DATA.DATA_PATH = '../Dataset/ucm/'
        config.MODEL.NUM_CLASSES = 21
    elif config.DATA.DATASET == 'aid':
        config.DATA.DATA_PATH = '../Dataset/aid/'
        config.MODEL.NUM_CLASSES = 30
    elif config.DATA.DATASET == 'nwpuresisc':
        config.DATA.DATA_PATH = '../Dataset/nwpu_resisc45/'
        config.MODEL.NUM_CLASSES = 45

    config.freeze()

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=True)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    exp_record = np.zeros([3,args.exp_num + 2])

    for i in range(args.exp_num):
        
        seed = i+2022
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.benchmark = True

        dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config, args.ratio, logger, args.split)

        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

        max_accuracy = 0.0

        # if config.TRAIN.AUTO_RESUME:
        #     resume_file = auto_resume_helper(config.OUTPUT)
        #     if resume_file:
        #         if config.MODEL.RESUME:
        #             logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
        #         config.defrost()
        #         config.MODEL.RESUME = resume_file
        #         config.freeze()
        #         logger.info(f'auto resuming from {resume_file}')
        #     else:
        #         logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
        
        if config.MODEL.RESUME:
            max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
            acc1, acc5, loss = validate(config, data_loader_val, model)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            if config.EVAL_MODE:
                return
                
        if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
            load_pretrained(config, model_without_ddp, logger)
            acc1, acc5, loss = validate(config, data_loader_val, model)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")

        if config.THROUGHPUT_MODE:
            throughput(data_loader_val, model, logger)
            return

        logger.info("Start training")
        start_time = time.time()
        best_acc1 = 0
        for epoch in range(config.TRAIN.START_EPOCH, args.epochs):
            data_loader_train.sampler.set_epoch(epoch)

            train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler)
            if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

            acc1, acc5, loss = validate(config, data_loader_val, model)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            max_accuracy = max(max_accuracy, acc1)
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')

            if acc1 > best_acc1:
                save_state = {'model': model.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'max_accuracy': max_accuracy,
                            'epoch': epoch,
                            'config': config}
                if config.AMP_OPT_LEVEL != "O0":
                    save_state['amp'] = amp.state_dict()

                save_path = os.path.join(config.OUTPUT, 'best_ckpt.pth')
                logger.info(f"{save_path} saving......")
                torch.save(save_state, save_path)
                logger.info(f"{save_path} saved !!!")
                best_acc1 = acc1

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Onetime training time {}'.format(total_time_str))

        exp_record[0,i] = acc1
        exp_record[1,i] = max_accuracy
        exp_record[2,i] = int(total_time)

    exp_record[0,-2] = np.mean(exp_record[0,:args.exp_num])
    exp_record[0,-1] = np.std(exp_record[0,:args.exp_num])
    exp_record[1,-2] = np.mean(exp_record[1,:args.exp_num])
    exp_record[1,-1] = np.std(exp_record[1,:args.exp_num])
    exp_record[2,-2] = np.mean(exp_record[2,:args.exp_num])
    exp_record[2,-1] = np.std(exp_record[2,:args.exp_num])

    logger.info(exp_record)
    logger.info('Last acc1 of {} model on {} dataset: {:.2f} ± {:.2f}'.format(args.model, args.dataset, exp_record[0,-2], exp_record[0,-1]))
    logger.info('Max acc1 of {} model on {} dataset: {:.2f} ± {:.2f}'.format(args.model, args.dataset, exp_record[1,-2], exp_record[1,-1]))
    logger.info('Average training time on {} epoch: {} ± {}'.format(args.epochs, str(datetime.timedelta(seconds=int(exp_record[2,-2]))), \
                                                                                        str(datetime.timedelta(seconds=int(exp_record[2,-1])))))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        outputs = model(samples)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = criterion(outputs, targets)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    #seed = config.SEED + dist.get_rank()

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()

    if opt_lower == 'adamw':
        # linear scale the learning rate according to total batch size, may not be optimal
        linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
        linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
        linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
        # gradient accumulation also need to scale the learning rate
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
        
        config.defrost()
        config.TRAIN.BASE_LR = linear_scaled_lr
        config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
        config.TRAIN.MIN_LR = linear_scaled_min_lr
        config.freeze()

    elif opt_lower == 'sgd':
        config.defrost()
        config.TRAIN.LR_SCHEDULER.NAME = 'step'
        config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = config.TRAIN.EPOCHS // 3 - config.TRAIN.EPOCHS //10
        config.freeze()


    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(args,config)
