import os
import random
import shutil

import torch


class AverageMeter(object):
    '''
        Compute and store the average and current value
    '''
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

    def save_checkpoint(state, id_, is_best, filename='checkpoint.pth'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(
                filename,
                os.path.join("save_models", "{}_best.pth".format(id_))
                )