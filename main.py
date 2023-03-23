#!/usr/bin/env python
#-*- coding:utf-8 -*-


import os
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

which = int(sys.argv[1])
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
if which:
    model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
else:
    model = ChatGLMForConditionalGeneration(GLMConfig(0))
model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'


def build_prompt(history):
    prompt = "欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt


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
            with open("temp.cfg","w") as fp:
                print(model.generation_config, file=fp)
            count += 1
            if count % 8 == 0:
                os.system(clear_command)
                print(build_prompt(history), flush=True)
        # os.system(clear_command)
        print(build_prompt(history), flush=True)


if __name__ == "__main__":
    main()
