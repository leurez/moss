#!/usr/bin/env python
#-*- coding:utf-8 -*-


import sys
import abc

from typing import Optional
from moss.loader.model import load_model

from moss.utils.log import logger
from moss.utils.generator import generate
from moss.utils.conversation import SeparatorStyle
from moss.utils.conversation import conv_templates
from moss.utils.conversation import get_default_conv_template


class ChatIO(abc.ABC):
    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str):
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream, skip_echo_len: int):
        """Stream output."""


def compute_skip_echo_len(model_name, conv, prompt):
    model_name = model_name.lower()
    if "chatglm" in model_name:
        skip_echo_len = len(conv.messages[-2][1]) + 1
    elif "dolly" in model_name:
        special_toks = ["### Instruction:", "### Response:", "### End"]
        prompt_tmp = prompt
        for tok in special_toks:
            prompt_tmp = prompt_tmp.replace(tok, "")
        skip_echo_len = len(prompt_tmp)
    else:
        skip_echo_len = len(prompt) + 1 - prompt.count("</s>") * 3
    return skip_echo_len


def chat_loop(
    model_path: str, device: str, num_gpus: str,
    max_gpu_memory: str, load_8bit: bool,
    conv_template: Optional[str], temperature: float,
    max_new_tokens: int, chatio: ChatIO,
    debug: bool):
    # Model
    model, tokenizer = load_model(model_path, device,
        num_gpus, max_gpu_memory, load_8bit, debug)
    # Chat
    if conv_template:
        conv = conv_templates[conv_template].copy()
    else:
        conv = get_default_conv_template(model_path).copy()
    while True:
        try:
            inp = chatio.prompt_for_input(conv.roles[0])
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        generate_stream_func = generate
        prompt = conv.get_prompt()

        skip_echo_len = compute_skip_echo_len(model_path, conv, prompt)

        params = {
            "model": model_path,
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
        }

        chatio.prompt_for_output(conv.roles[1])
        outputs = generate_stream_func(model, tokenizer, params, device)
        # outputs = chatio.stream_output(outputs, skip_echo_len)
        logger.typewriter_log(outputs)
        # NOTE: strip is important to align with the training data.
        conv.messages[-1][-1] = outputs.strip()

        if debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


class SimpleChatIO(ChatIO):
    def prompt_for_input(self, role) -> str:
        return input(f"{role}: ")

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream, skip_echo_len: int):
        pre = 0

        for outputs in output_stream:
            outputs = outputs[skip_echo_len:].strip()
            outputs = outputs.split(" ")
            now = len(outputs) - 1
            if now > pre:
                # print(" ".join(outputs[pre:now]), end=" ", flush=True)
                logger.typewriter_log()
                pre = now
        print(" ".join(outputs[pre:]), flush=True)
        return " ".join(outputs)


if __name__ == "__main__":
    chat_loop(
        model_path = "",
        device = "cuda",
        num_gpus = int(sys.argv[1]),
        max_gpu_memory = '13GiB',
        load_8bit = False,
        conv_template = None,
        temperature = 1.0,
        max_new_tokens = 256,
        chatio = SimpleChatIO(),
    )