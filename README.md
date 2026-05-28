# Emotion Recognition and Intervention Agent

这是 Project 3 的可运行原型，代码全部放在 `EID` 文件夹中。

## 功能

- User Authentication：注册、登录、密码哈希存储、用户数据隔离
- 半结构式一对一对话
- 情绪状态识别
- 社交关系状态识别
- 识别结果到反馈策略的映射
- DeepSeek API 深度参与单轮规划：结合历史对话、信息槽、当前输入，同时生成识别结果、策略和自然回复
- API 不可用时自动使用本地规则 fallback
- SQLite 保存消息、识别结果、策略和会话摘要
- Emotion Dashboard：展示当前会话指标、历史情绪统计、社交状态统计、强度趋势和历史记录

## 启动方式

在 PowerShell 中进入 `EID` 文件夹：

```powershell
cd "C:\Users\Sansuf\Desktop\school information\智能信息系统\project 3\EID"
```

设置 DeepSeek API Key：

```powershell
$env:DEEPSEEK_API_KEY="你的 DeepSeek API Key"
```

启动系统：

```powershell
python emotion_agent_app.py
```

浏览器打开：

```text
http://127.0.0.1:8501
```

## 使用流程

1. 在右上角输入用户名和密码。
2. 第一次使用点击“注册”。
3. 已经注册过点击“登录”。
4. 登录后可以开始聊天。
5. 点击“新会话”可以结束当前会话并开始新的演示。
6. 右侧 Dashboard 会显示当前用户自己的情绪历史和分析结果。
7. 可以在右上角的 `DeepSeek API Key` 输入框临时粘贴 API key；它只随请求发送，不会保存到数据库。
8. 勾选“使用 DeepSeek API”时，系统会让 DeepSeek 深度参与整轮对话规划；取消勾选时，系统强制使用本地规则 fallback。

默认页面会填入：

```text
username: demo_user
password: demo123
```

第一次使用这个默认账号时，点击“注册”即可。

## 演示输入

积极状态：

```text
I finished my part of the group work and my teammates said it was helpful.
```

焦虑状态：

```text
I have a presentation tomorrow and I keep worrying that I will forget everything.
```

挫败感和弱支持：

```text
I tried to fix the same bug for two hours. I feel stuck, and I do not know who I can ask.
```

中文也可以：

```text
我明天要汇报，一直很担心自己会忘词，而且最近睡得也不太好。
```

```text
我这个 bug 改了两个小时还是不行，也不知道可以问谁。
```

## 说明

当前版本是课堂展示原型，不是最终产品。

如果设置了 `DEEPSEEK_API_KEY`，系统会优先调用 DeepSeek。也可以直接在页面右上角临时输入 DeepSeek API Key。

如果没有设置 API key，或者网络/API 调用失败，系统会使用本地规则进行识别和回复，保证演示不中断。

密码不会以明文保存到数据库，系统使用 PBKDF2 哈希存储密码。
