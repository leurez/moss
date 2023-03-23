#!/usr/bin/env python
#-*- coding:utf-8 -*-


import openai

if __name__ == "__main__":
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        message=[{"role":"user", "content": "how to train a chatgpt"}]
    )
    print(completion)