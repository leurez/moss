#!/usr/bin/env python
#-*- coding:utf-8 -*-


import torch


#precision

MASK = 150000
gMASK = 150001
NUM_LAYERS = 28
PRECISION_TYPE = torch.half

# ATTENTION 
ROTARY_EMB_BASE = 10000
MASKED_FILL_VALUE = -10000.0
ROTARY_EMB_LEARNABLE = False

#OPENAI GELU CONST
OPENAI_GELU_TANH_P2 = 0.044715
OPENAI_GELU_TANH_P1 = 0.7978845608028654