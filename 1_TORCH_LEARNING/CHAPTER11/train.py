#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :train.py
# @Time        :2024/10/9 下午12:05
# @Author      :InubashiriLix
import torch.cuda
import torch.nn as nn
from torch.optim import SGD
from utils.util import logging
import argparse

from torch.utils.data import DataLoader, Dataset

from CHAPTER10.dsets import LunaDataset

log = logging.getLogger(__name__)


class LunaTrainingApp:

    def __init__(self, sys_argv=None):
        # initial the parser
        parser = argparse.ArgumentParser()

        parser.add_argument("--num-workers",
                            help="Number of worker process for background data loading",
                            default=8,
                            type=int,
                            )
        parser.add_argument('--batch-size',
                            help="Batch size to use for training",
                            default=32,
                            type=int)

        parser.add_argument('--epoches',
                            help="Number of epochs to train for",
                            default=1,
                            type=int)

        parser.add_argument('--tb-prefix',
                            default='CHAPTER11',
                            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
                            )

        parser.add_argument('comment',
                            help="Comment suffix for Tensorboard run.",
                            nargs='?',
                            default='dwlpt',
                            )

        if sys_argv is not None:
            self.cli_args = parser.parse_args(sys_argv)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.learning_rate = 0.001
        self.momentum = 0.90

        self.model = self.initModel()
        self.initOptimizer = self.initOptimizer()

    def initModel(self) -> nn.Module:
        model = LunaModel()
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model.to(self.device)
        return model

    # we can use the DistributedDataParallel too.
    def initOptimizer(self) -> torch.optim:
        return SGD(
            self.model,
            lr=self.learning_rate,
            momentum=self.momentum
        )

    def initTrainDl(self):
        # init the training dataset
        train_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=False,
        )
        # batch_size should be the batch size per GPU
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        # initialize the DL
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda
        )
        return train_dl

    def initValDl(self):
        val_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=True
        )
        batch_size = self.cli_args.batch_size

        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda
        )
        return train_dl

    def main(self):
        train_dl = self.initTrainDl()
        val_dl = self.initValDl()
