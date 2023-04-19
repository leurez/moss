import os
import openai
import datetime
import pyperclip
import pandas as pd
openai.api_key = os.getenv("OPENAI_API_KEY")
#os.environ['NO_PROXY']='api.openai.com'

#chatGPT3.5支持连续对话
MAX_TEXT_LENGTH = 1024

print("max_text_length:",MAX_TEXT_LENGTH)
print('输入1复制上次回答内容')
print('输入0导出本次所有对话内容')

#创建一个DataFrame用于存储每次翻译结果
df = pd.DataFrame(columns=['发送的内容', '收到的回复'])

conversation=[{"role": "system", "content": "You are a helpful assistant."}]

while True:
    prompt = (input("请输入您的内容："))

    if prompt=='1':
        pyperclip.copy(result.replace('\n', ''))  # 复制到剪切板并删除换行符
        print("已复制到剪切板！")
        continue

    #如果输入0则将过去的所的内容导出
    if prompt =='0':
        # 获取当前日期和时间
        now = datetime.datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        date_time=date_time.replace(':', '-')
        # 定义文件名
        filename = "对话导出" + date_time + ".xlsx"
        # 将DataFrame导出为Excel文件
        df.to_excel(filename, sheet_name='Sheet1', index=False)
        print("Excel文件已保存为: ", filename)
        continue

    conversation.append({"role": "user","content": prompt})
    print("debug msg-----001")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = conversation,
        temperature=1,
        max_tokens=MAX_TEXT_LENGTH,
        top_p=0.9
    )
    print("debug msg-----002")
    conversation.append({"role": "assistant", "content": response['choices'][0]['message']['content']})

    answer = response['choices'][0]['message']['content']
    if len(answer)<200:
        print("answer <200")
    else:
        print("answer >200.is:",len(answer))

    print("\n" + answer + "\n")

    result=answer

    print("debug msg-----003")

    # 创建一个列表，存储本次原文与译句
    qa = []
    q = prompt # 发送的内容
    a = answer  # 收到的回答
    qa.append(q)
    qa.append(a)

    # 向DataFrame中添加本次QA
    df.loc[len(df)] = qa