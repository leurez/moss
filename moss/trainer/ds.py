#!/usr/bin/env python
#-*- coding:utf-8 -*-


import math
from threading import local
import yaml
import torch
import deepspeed as ds

from transformers import(
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.deepspeed import HfDeepSpeedConfig

from moss.trainer.base import Config
from moss.trainer.base import Trainer
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam


class DSTrainer(Trainer):

    def __init__(self, tokenizer, model, config, local_rank):
        super(self, DSTrainer).__init__()
        self.model = model
        self.tokenizer = tokenizer
        # config deepspeed trainer
        with open(config) as fp:
            self.config = yaml.safe_load(fp)
        # init the environment
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)
        ds.init_distributed()
    
    def get_dataset_loader(self):

    def run(self):

