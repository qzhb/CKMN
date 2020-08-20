import torch
import torch.nn.functional as F
from torch.autograd import Variable

import os
import time
import numpy as np

from utils import AverageMeter, calculate_mAP_sklearn


def performance(prediction_s, target, test_crop_number):
    
    prediction = F.softmax(prediction_s, dim=-1)
    if test_crop_number != 1:
        prediction = prediction.mean(0)
    mAP = calculate_mAP_sklearn(prediction, target)
    print('mAP performance:', mAP)

    return mAP


def test_epoch(epoch, data_loader, model, opt, logger, writer):
    print('validation at epoch {}'.format(epoch))
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    classification_results_final = []
    all_targets = []

    end_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            if opt.test_crop_number == 1:
                inputs = data[0]
            else:
                inputs = data[0].reshape(-1, data[0].size()[2], data[0].size()[3], data[0].size()[4], data[0].size()[5], data[0].size()[6])
            targets = data[1].float()
            inputs = inputs.to(opt.device)
            targets = targets.to(opt.device)
            
            outputs = model(inputs)
            
            all_targets.append(targets)
            if opt.test_crop_number == 1:
                classification_results_final.append(outputs.cpu().data)
            else:
                classification_results_final.append(outputs.unsqueeze(1).cpu().data)

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if i % 100 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    epoch,
                    i + 1,
                    len(data_loader),
                    batch_time=batch_time))

    if opt.test_crop_number == 1:
        classification_results_final = torch.cat(classification_results_final, dim=1)
    else:
        classification_results_final = torch.cat(classification_results_final, dim=0)

    all_targets = torch.cat(all_targets, dim=0)
    final_mAP = performance(classification_results_final, all_targets)

    logger.log({
        'epoch': epoch,
        'final_mAP': final_mAP
    })
