#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :logconf.py
# @Time        :2024/10/8 下午7:08
# @Author      :InubashiriLix

import logging
import logging.handlers

# initialize the logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# some libs try to the loggers of their own, we fuck them all there
for handler in list(root_logger.handlers):
    root_logger.removeHandler(handler)

# setup our own logger
logfmt_str = "%(asctime)s %(levelname)-8s pid:%(process)d %(name)s:%(lineno)03d:%(funcName)s %(message)s"
formatter = logging.Formatter(logfmt_str)

# the handler has two kings: StreamHandler and FileHandler
streamHandler = logging.StreamHandler()
# we use the console to output the info now
streamHandler.setFormatter(formatter)
streamHandler.setLevel(logging.DEBUG)

root_logger.addHandler(streamHandler)
