import torch
import torch.nn.functional as F
from torch.autograd import Variable

import time
import os
import ipdb

from utils import AverageMeter, calculate_mAP_sklearn


def performance(prediction, target):
    prediction = F.softmax(prediction, dim=1)
    mAP = calculate_mAP_sklearn(prediction, target)
    print('Final_mAP:', mAP)

    return mAP


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, writer):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    classification_results_final = []
    all_targets = []

    end_time = time.time()
    for i, data in enumerate(data_loader):

        data_time.update(time.time() - end_time)

        inputs = data[0]
        targets = data[1].float()

        all_targets.append(targets)

        inputs = inputs.to(opt.device)
        targets = targets.to(opt.device)

        optimizer.zero_grad()

        outputs = model(inputs)

        classification_results_final.append(outputs.cpu().data)

        loss = criterion(outputs, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        writer.add_scalar('train/loss_iter', losses.val, i + 1 + len(data_loader) * (epoch - 1))

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss ({loss.avg:.4f})'.format(
                      epoch,
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses))

    classification_results_final = torch.cat(classification_results_final, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    final_mAP = performance(classification_results_final, all_targets)

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'final_mAP': final_mAP,
        'lr': optimizer.param_groups[-1]['lr']
    })

    writer.add_scalar('train/loss_epoch', losses.avg, epoch)
    writer.add_scalar('train/learning_rate_epoch', opt.learning_rate, epoch)
