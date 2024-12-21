#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :run_everything.py
# @Time        :2024/10/8 下午6:58
# @Author      :InubashiriLix

import datetime
import argparse
from utils.util import logging
from utils.util import importstr

import os
import shutil

log = logging.getLogger(__name__)


def run(app, *argv):
    argv = list(argv)
    argv.insert(0, '--num-workers=4')
    log.info("Running: {}({!r}).main()".format(app, argv))

    app_cls = importstr(*app.rsplit('.', 1))
    app_cls(argv).main()

    log.info("Finished: {}.{!r}).main()".format(app, argv))


# TODO: THE CACHE PATH MIGHT BE NOT CORRECT, GO CHECK IT
def cleanCache():
    shutil.rmtree('G:/LUNA16')
    os.mkdir('G:/LUNA16/cache')



training_epochs = 20
experiment_epochs = 10
final_epochs = 50

training_epochs = 2
experiment_epochs = 2
final_epochs = 5
seg_epochs = 10
run('training.LunaTrainingApp', '--epoch=1')