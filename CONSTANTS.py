#! /usr/bin/env python
# coding: utf-8
# --------------------------------------------------------------------------------
#     File Name           :     CONSTANTS.py
#     Created By          :     hao
#     Description         :
# --------------------------------------------------------------------------------

SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

# OUTPUT_ROOT_DIR = "out"
LOG_FILE_NAME = "log.txt"
CHECKPOINT_DIR_NAME = "params"
EVAL_DIR_NAME = "eval"
STREAM_LOGGER_FORMATTER = "[%(levelname)s]%(asctime)s - [{}] %(message)s"
FILE_LOGGER_FORMATTER = "[%(levelname)s]%(asctime)s - [%(filename)s:%(lineno)s - %(funcName)10s() ] [{}] %(message)s"
LOGGER_MODE = "a"
MAX_LENGTH = 100
MAX_SENT_LENGTH = 100
