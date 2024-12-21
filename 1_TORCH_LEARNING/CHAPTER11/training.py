#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :train.py
# @Time        :2024/10/8 下午6:52
# @Author      :InubashiriLix
import argparse
import logging
import datetime
import sys
import os

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset

from utils.util import logging, enumerateWithEstimate

from model import LunaModel
from CHAPTER10.dsets import LunaDataset

log = logging.getLogger(__name__)

METRIC_LABEL_NDX = 0
METRIC_PRED_NDX = 1
METRIC_LOSS_NDX = 2
METRIC_SIZE = 3


class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
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

        parser.add_argument('--epochs',
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

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.initModel()
        self.optimzer = self.optimizer()

    def initModel(self):
        # TODO: add the LunaModel
        model = LunaModel()
        if self.use_cuda:
            log.info("Using Cuda; {} device".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def optimizer(self):
        # TODO : MIGHT NEED MULTI OPTIMIZER
        return SGD(self.model, lr=0.001, momentum=0.99)

    def initTrainDl(self):
        train_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=False
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size * torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda
        )

        return train_dl

    def initValDl(self):
        val_ds = LunaDataset(
            isValSet_bool=True,
            val_stride=10
        )
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda
        )

        return val_dl

    # TODO : WHAT IS TENSORBOARD???
    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '-trn_cls-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls-' + self.cli_args.comment)

    def computeBatchLoss(self,
                         batch_ndx,
                         batch_tup,
                         batch_size,
                         metrics_g: torch.tensor
                         ):
        input_t, label_t, _series_list, _center_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, probability_g = self.model(input_g)

        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_g = loss_func(
            logits_g,
            label_g[:, 1]
        )
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        metrics_g[METRIC_LABEL_NDX, start_ndx: end_ndx] = label_g[:1].detach()
        metrics_g[METRIC_PRED_NDX, start_ndx: end_ndx] = probability_g[:1].detach()
        metrics_g[METRIC_LOSS_NDX, start_ndx: end_ndx] = loss_g.detach()

        return loss_g.mean()

    def doTraining(self, epochs_ndx, train_dl):
        # start training
        self.model.train()

        trnMetrics = torch.zeros(
            # TODO: WTF IS THIS???
            # METRIC_SIZE,
            len(train_dl.data),
            device=self.device
        )

        # create batch training
        # can be replace with enumerate directly and
        # it is use to show the progress with estimated time
        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} training".format(epochs_ndx),
            start_ndx=train_dl.num_workers
        )

        # clean the weights and bias
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer().zero_grad()
            # TODO : computeBatchLoss
            loss_var = self.computeBatchLoss(
                batch_ndx,
                batch_ndx,
                train_dl.batch_size,
                trnMetrics_g
            )
            loss_var.backward()
            self.optimzer.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            trnMetrics_t = self.doTraining()


if __name__ == "__main__":
    LunaTrainingApp().main()
