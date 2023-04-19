#!/usr/bin/env python
#-*- coding:utf-8 -*-


import torch
import random
import numpy as np

from types import Any
from transformers import set_seed


MICRO_BATCH_SIZE = 4
GLOBAL_BATCH_SIZE = 32


class Trainer:

    def __init__(self, **kw):
        # set the seed
        seed = 1751
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    def run(self):
        raise NotImplementedError("please define your own run!")


class Config(dict):

    def __init__(self, *args, **kw):
        super(self, Config).__init__(*args, **kw)
    
    def __getattr__(self, __name: str) -> Any:
        return self[__name]

    def __setattr__(self, __name: str, __value: Any) -> None:
        self[__name] = __value
