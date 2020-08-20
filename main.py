import os
import json
import socket
from datetime import datetime

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn

from opts import parse_opts
from dataloaders.dataset import get_training_set, get_test_set

from model import generate_model
from utils import Logger, get_fine_tuning_parameters
from train import train_epoch
from test import test_epoch

import torchvision.transforms as transforms
from temporal_transforms import TemporalSegmentRandomCrop, TemporalSegmentCenterCrop
from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop, MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor, GroupScale, GroupCenterCrop, GroupOverSample, GroupNormalize, Stack, ToTorchFormatTensor)

from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


if __name__ == '__main__':

    # set parameters
    timestamp = datetime.now().strftime('%b%d_%H-%M-%S')
    opt = parse_opts()
    torch.manual_seed(opt.manual_seed)
    opt.timestamp = timestamp
    if opt.dataset == 'FCVID':
        opt.event_classes = 238
    elif opt.dataset == 'ActivityNet':
        opt.event_classes = 200

    # set path
    if opt.data_root_path != '':
        opt.video_path = os.path.join(
            opt.data_root_path, opt.dataset, opt.video_path)
        opt.annotation_path = os.path.join(
            opt.data_root_path,
            opt.dataset,
            opt.annotation_path +
            opt.dataset +
            '.json')
    if opt.result_path != '':
        opt.result_path = os.path.join(
            opt.result_path,
            opt.dataset,
            opt.model_name)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.result_path, opt.resume_path)
        opt.save_path = os.path.join(opt.result_path, timestamp)
        os.makedirs(opt.save_path)

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)

    # save parameters
    with open(os.path.join(opt.save_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)
    print(opt)

    # Logging Tensorboard
    log_dir = os.path.join(
        opt.save_path,
        'tensorboard',
        'runs',
        socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir, comment='-params')
    
    # model and criterion
    model, parameters = generate_model(opt)
    criterion = nn.MultiLabelSoftMarginLoss()

    # set cuda
    if not opt.no_cuda:
       os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
       torch.cuda.manual_seed(opt.manual_seed)
    
    opt.device = torch.device("cuda" if not opt.no_cuda else "cpu")
    criterion = criterion.to(opt.device)
    model = model.to(opt.device)
    
    if opt.ngpus > 1:
        model = torch.nn.DataParallel(model)

    ## optimizer
    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening

    optimizer = optim.SGD(
            parameters,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)

    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ## prepare train
    if not opt.no_train:
        temporal_transform = TemporalSegmentRandomCrop(opt.segment_number, opt.sample_duration)

        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            spatial_crop_method = MultiScaleRandomCrop(opt.scales, opt.frame_size)
        elif opt.train_crop == 'corner':
            spatial_crop_method = MultiScaleCornerCrop(opt.scales, opt.frame_size)
        elif opt.train_crop == 'center':
            spatial_crop_method = MultiScaleCornerCrop(opt.scales, opt.frame_size, crop_positions=['c'])
        spatial_transform = Compose([
            spatial_crop_method,
            RandomHorizontalFlip(),
            ToTensor(opt.norm_value),
            normalize
        ])
        training_data = get_training_set(opt, spatial_transform, temporal_transform)

        train_loader = DataLoaderX(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True,
            drop_last=True)
        train_logger = Logger(
            os.path.join(opt.save_path, 'train.log'),
            ['epoch', 'loss', 'final_mAP', 'lr'])

        temporal_transform = TemporalSegmentCenterCrop(opt.segment_number, opt.sample_duration)

        if opt.test_crop_number == 1:
            cropping = transforms.Compose([
                GroupScale(opt.reshape_size),
                GroupCenterCrop(opt.frame_size),
            ])
        elif opt.test_crop_number == 10:
            cropping = transforms.Compose([
                GroupOverSample(opt.frame_size, opt.reshape_size)
            ])
        else:
            raise ValueError("Only 1 and 10 crops are supported while we got {}".format(opt.test_crop_number))

        spatial_transform = transforms.Compose([
                cropping,
                Stack(roll=False),
                ToTorchFormatTensor(div=True),
                GroupNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
               ])

        test_data = get_test_set(opt, spatial_transform, temporal_transform)

        test_loader = DataLoaderX(
            test_data,
            batch_size=opt.batch_size if opt.test_crop_number == 1 else 1,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True,
            drop_last=True)

        test_logger = Logger(
            os.path.join(opt.save_path, 'test.log'), ['epoch', 'final_mAP'])


    #t# prepare test
    if not opt.no_test:
        temporal_transform = TemporalSegmentCenterCrop(opt.segment_number, opt.sample_duration)

        if opt.test_crop_number == 1:
            cropping = transforms.Compose([
                GroupScale(opt.reshape_size),
                GroupCenterCrop(opt.frame_size),
            ])
        elif opt.test_crop_number == 10:
            cropping = transforms.Compose([
                GroupOverSample(opt.frame_size, opt.reshape_size)
            ])
        else:
            raise ValueError("Only 1 and 10 crops are supported while we got {}".format(opt.test_crop_number))

        spatial_transform = transforms.Compose([
                cropping,
                Stack(roll=False),
                ToTorchFormatTensor(div=True),
                GroupNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
               ])

        test_data = get_test_set(opt, spatial_transform, temporal_transform)

        test_loader = DataLoaderX(
            test_data,
            batch_size=opt.batch_size if opt.test_crop_number == 1 else 1,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True,
            drop_last=True)

        test_logger = Logger(
            os.path.join(opt.save_path, 'test.log'), ['epoch', 'final_mAP'])

    ## train process
    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        del checkpoint
    
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.lr_decay)

    if not opt.no_train:
        for _ in range(1, opt.begin_epoch):
            scheduler.step()
        for i in range(opt.begin_epoch, opt.n_epochs + 1):
            scheduler.step()
            cudnn.benchmark = True
            train_epoch(i, train_loader, model, criterion, optimizer, opt, train_logger, writer)

            if i % opt.checkpoint == 0:
                save_file_path = os.path.join(opt.save_path, 'train_' + str(i+1) + '_model.pth')
                states = {
                    'epoch': i + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),}
                torch.save(states, save_file_path)

            if i % opt.test_per_epoches == 0:
                test_epoch(i, test_loader, model, opt, test_logger, writer)
     
    elif not opt.no_test:
        test_epoch(0, test_loader, model, opt, test_logger, writer)

    writer.close()
