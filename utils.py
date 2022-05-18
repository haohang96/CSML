import os
import math
import pdb
import absl
import datetime
import logging
import torch
import torch.nn as nn
import numpy as np
import shutil


from PIL import Image
from absl import flags
from absl import app

FLAGS = flags.FLAGS
if FLAGS.moxing:
    import moxing as mox

class Log():
    def __init__(self, exp_path):
        self.logger = logging.getLogger(__name__)
        formatter = logging.Formatter('%(asctime)s - %(name)-8s - %(levelname)-6s - %(message)s')

        if FLAGS.rank == 0: # only rank 0 output is visible
            # str handler
            strHandler = logging.StreamHandler()
            strHandler.setFormatter(formatter)
            self.logger.addHandler(strHandler)
            self.logger.setLevel(logging.INFO)

        # file handler
        self.log_path = os.path.join(exp_path, 'logs')
        if FLAGS.moxing:
            if (not mox.file.exists(self.log_path)):
                mox.file.make_dirs(self.log_path)
        else:
            if (not os.path.exists(self.log_path)):
                os.makedirs(self.log_path, exist_ok=True) # handle FileExistsError in multiprocessing mode

        now_str = datetime.datetime.now().__str__().replace(' ','_')
        self.file_name = 'LOG_INFO_'+now_str+'_rank'+str(FLAGS.rank)+'.txt'
        self.log_file = os.path.join(self.log_path, self.file_name)
        self.log_fileHandler = logging.FileHandler(self.log_file)
        self.log_fileHandler.setFormatter(formatter)
        self.logger.addHandler(self.log_fileHandler)

        # remove root hanlder intro by absl
        logging.root.removeHandler(absl.logging._absl_handler)



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, log):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        log.logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_ckpt(state, epoch, save_freq, is_best=None):
    if FLAGS.rank == 0:
        filename = os.path.join(FLAGS.train_url, 'ckpt.pth.tar')
        if FLAGS.moxing:
            filename = os.path.join(FLAGS.cache_ckpt_folder, 'ckpt.pth.tar')
        torch.save(state, filename)
        if FLAGS.moxing:
            mox.file.copy(filename, os.path.join(FLAGS.train_url, 'ckpt.pth.tar'))

        if epoch % save_freq == 0:
            filename = os.path.join(FLAGS.train_url, 'ckpt_%s.pth.tar'%(epoch))
            if FLAGS.moxing:
                filename = os.path.join(FLAGS.cache_ckpt_folder, 'ckpt_%s.pth.tar'%(epoch))
            torch.save(state, filename)
            if FLAGS.moxing:
                mox.file.copy(filename, os.path.join(FLAGS.train_url, 'ckpt_%s.pth.tar'%(epoch)))
        if is_best:
            if FLAGS.moxing:
                mox.file.copy(filename, os.path.join(FLAGS.train_url, 'ckpt_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(FLAGS.train_url, 'ckpt_best.pth.tar'))

    else:
        pass

def adjust_learning_rate(optimizer, epoch, log):
    """Decay the learning rate based on schedule"""
    lr = FLAGS.init_lr
    if FLAGS.decay_method == 'cos':  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / FLAGS.end_epoch))
    else:  # stepwise lr schedule
        for milestone in FLAGS.schedule:
            # lr *= 0.1 if epoch >= milestone else 1.
            lr *= FLAGS.lr_decay if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    log.logger.info('==> Setting model optimizer lr = %.6f'%(lr))
