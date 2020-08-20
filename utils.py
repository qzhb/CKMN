import csv
import numpy as np
from sklearn.metrics import average_precision_score
import ipdb

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


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def load_list_file(file_path):
    with open(file_path, 'r') as input_file:
        lists = input_file.readlines()

    return lists


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().data[0]

    return n_correct_elems / batch_size


def calculate_mAP_sklearn(outputs, targets):
    class_num = np.size(targets, 1)
    mAP = []

    for idx in range(class_num):
        mAP.append(average_precision_score(targets[:, idx], outputs[:, idx]))

    mAP = np.mean(mAP)

    return mAP


def get_fine_tuning_parameters(model, ft_module_names):
    # for k, v in model.named_parameters():
    #     print(k)

    parameters = []
    #parameters_name_one = []
    #parameters_name_two = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                #parameters_name_one.append(k)
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})
            #parameters_name_two.append(k)

    return parameters
