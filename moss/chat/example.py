#!/usr/bin/env python
#-*- coding:utf-8 -*-


import os
import sqlite3
import readline
import requests

# 连接到 SQLite 数据库
conn = sqlite3.connect('chatbot.db')
c = conn.cursor()

# 创建会话表
c.execute('''CREATE TABLE IF NOT EXISTS sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
    )'''
)

# 创建消息表
c.execute('''CREATE TABLE IF NOT EXISTS messages (
    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER, message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
    )'''
)

# 提交更改
conn.commit()
ctx = None


# 开始聊天
def get_promot(id, role):
    if role == 0:
        return f"\033[32m[session {id}]> "
    else:
        return f"\033[0m[session {id}]>"


def get_user_input():
    msg = input(get_promot(session_id, 0))
    readline.add_history(msg)
    return msg


def get_bot_output(content):
    readline.add_history(content)
    return get_promot(session_id, 1) + content


def create_seesion(ctx, c):
    c.execute('''INSERT INTO sessions DEFAULT VALUES''')
    session_id = c.lastrowid
    conn.commit()
    os.system('clear')
    return session_id
    

def get_session(ctx, c):
    # 查询所有会话 ID
    c.execute('SELECT session_id, description FROM sessions DESC LIMIT 20')
    sessions = c.fetchall()

    if len(sessions):
        # 输出会话列表供用户选择
        print('\033[33m从历史开始（输入数字）或新建会话（回车）？:\033[0m')
        for i, ss in enumerate(sessions):
            print(f'\033[33m    {i+1}. session {ss[0]:06d} {ss[1]}\033[0m')

        choice = input('\033[32myour choice[None]> ')
        choice = int(choice) if choice else 0
        if choice and 1 <= choice <= len(sessions):
            session = sessions[choice-1]
            session_id = session[0]
            os.system('clear')
            print(f'\033[31m已选择会话 {session_id}\033[0m')
        else:
            session_id = create_seesion(ctx, c)
            print(f'\033[31m没有选择任何历史会话，创建新会话：{session_id}\033[0m')
    else:
        session_id = create_seesion(ctx, c)
    return session_id


session_id = get_session(ctx, c)
while True:
    try:
        user_input = get_user_input()
        if not user_input.strip():
            print('\033[31m请输入一些内容\033[0m')
            continue

        # 新建会话
        if user_input.strip() == '/new':
            session_id = create_seesion(ctx, c)
            print(f'新建会话成功，会话 ID 为 {session_id}')
            continue
        elif user_input.strip() == '/switch':
            session_id = get_session(ctx, c)
        elif user_input.strip() == '/clear':
            os.system('clear')
        # 向服务端发送 POST 请求
        data = {'message': user_input, 'session_id': session_id}
        # response = requests.post(API_ENDPOINT, data=data).json()
        response = {"response":f"I recived:{user_input}"}

        # 保存用户输入到数据库
        c.execute('''INSERT INTO messages (session_id, message)
                     VALUES (?, ?)''', (session_id, user_input))
        conn.commit()

        # 打印服务端返回的响应并保存到数据库
        response_text = response['response']
        c.execute('''INSERT INTO messages (session_id, message)
                     VALUES (?, ?)''', (session_id, response_text))
        conn.commit()
        print(get_bot_output(response_text))
    except KeyboardInterrupt:
        print('再见！')
        break

# 关闭数据库连接
c.close()
conn.close()

