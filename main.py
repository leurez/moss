#!/usr/bin/env python
#-*- coding:utf-8 -*-


import os
import pdb
import sys
import torch
import platform
from transformers import AutoTokenizer, AutoModel
from moss.model.glm import ChatGLMForConditionalGeneration


class GLMConfig:

    def __init__(self, device):
        self.device = device
        self.bos_token_id = 150004
        self.eos_token_id = 150005
        self.hidden_size = 4096
        self.inner_hidden_size = 16384
        self.layernorm_epsilon = 1e-05
        self.max_sequence_length = 2048
        self.model_type = "chatglm"
        self.num_attention_heads = 32
        self.num_layers = 28
        self.position_encoding_2d = True
        self.torch_dtype = torch.float16
        self.transformers_version = "4.23.1"
        self.use_cache = True
        self.vocab_size = 150528

        self.is_encoder_decoder = False


def load_model(config):
    model = ChatGLMForConditionalGeneration(config)
    state_dict = None
    import config as cfg
    for f in cfg.shard_model_files:
        st = torch.load(f, map_location='cpu')
        if state_dict is None:
            state_dict = st
        else:
            state_dict.update(**st)
    model.load_state_dict(state_dict)
    return model


which = int(sys.argv[1])
# model_path = "THUDM/chatglm-6b"
model_path = "/home/ec2-user/.cache/huggingface/hub/models--THUDM--chatglm-6b/snapshots/d2bbc82a2cdd04522ad340bdd464379808676950"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
if which:
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, local_files_only=True).half().cuda()
else:
    confi = GLMConfig(0)
    model = load_model(confi).half().cuda()
    #model = ChatGLMForConditionalGeneration(confi).half().cuda()
model = model.eval()

# pdb.set_trace()
os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'


def build_prompt(history):
    prompt = "欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt


def debug():
    count, history = 0, []
    for response, history in model.stream_chat(tokenizer, 'hello', history=history):
        if count > 1:
            break
        count += 1


def main():
    history = []
    print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query == "stop":
            break
        if query == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        count = 0
        for response, history in model.stream_chat(tokenizer, query, history=history):
            count += 1
            if count % 8 == 0:
                os.system(clear_command)
                print(build_prompt(history), flush=True)
        os.system(clear_command)
        print(build_prompt(history), flush=True)


if __name__ == "__main__":
    if sys.argv[2] == '0':
        main()
    else:
        debug()
