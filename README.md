# monikavoice

Windows 离线中文语音听写工具：按热键说话，文字自动打进当前焦点输入框。
全程本地推理，**不联网、无广告、无第三方输入法**。命名与 [monikasearch](https://github.com/Efold499e/monikasearch) 同风格。

## 功能

- **全局热键开关**：Alt+R 开/停听写（被 NVIDIA Overlay 等占用时自动退回 Alt+T / Alt+W），Ctrl+Alt+Q 退出，托盘右键菜单兜底
- **实时跟随浮窗**：听写时弹出置顶小浮窗，流式显示识别中的文字，不抢目标应用焦点
- **双模型管线**：
  1. 流式 zipformer 实时出字（浮窗跟随）
  2. 停顿约 0.4 s 断句后，整段音频交给离线 Paraformer 重识别 + **同音替换器纠错**（专治"语音识别同音错别字"）
  3. ct-transformer 自动补标点
  4. 文字以 Unicode 按键打进光标处（记事本、聊天框、IDE 通吃）
- **托盘常驻图标**：左键开关听写，右键退出
- **历史记录**：全部识别内容（含关闭听写时未上屏的部分）持久化到 `history.jsonl`，内置网页查看页，并提供本机 HTTP API 供其他程序调用

## 工作原理

```
麦克风(16k) ──► 流式 zipformer ──► 浮窗实时跟随
                   │ 停顿 0.4s 断句
                   ▼
        离线 Paraformer 重识别
                   ▼
        同音替换器纠错（jieba + replace.fst）
                   ▼
        ct-transformer 自动标点
                   ▼
        SendInput(UNICODE) 打进当前输入框
```

## 基于的开源项目

| 组件 | 用途 | 许可证 |
|------|------|--------|
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | 语音识别/标点/同音替换推理框架 | Apache-2.0 |
| [streaming zipformer 中英双语模型](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) | 流式实时识别 | Apache-2.0 |
| [Paraformer zh（阿里 FunASR）](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) | 离线整句精修 | Apache-2.0 |
| [ct-transformer 标点模型（FunASR）](https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models) | 自动加标点 | Apache-2.0 |
| [hr-files（dict / lexicon / replace.fst）](https://github.com/k2-fsa/sherpa-onnx/releases/tag/hr-files) | 同音替换资源 | Apache-2.0 |
| [jieba 词典](https://github.com/fxsjy/jieba) | 同音替换分词 | MIT |
| [python-sounddevice / PortAudio](https://python-sounddevice.readthedocs.io/) | 麦克风采集 | MIT |
| [pystray](https://pystray.readthedocs.io/) | 托盘图标 | LGPL-3.0 |
| tkinter | 实时浮窗 | PSF |

感谢 k2-fsa / Next-gen Kaldi 社区与 FunASR 团队的模型工作。

## 安装

1. Python ≥ 3.9，安装依赖：

   ```
   pip install sherpa-onnx sounddevice pystray pillow numpy
   ```

2. 下载模型（Windows 可把 wget 换成浏览器下载，解压到脚本同目录）：

   ```
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2

   mkdir hr
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/dict.tar.bz2
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/lexicon.txt
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/replace.fst
   ```

   解压后目录结构：

   ```
   monikavoice/
   ├── monikavoice.py
   ├── icon64.png
   ├── sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/
   ├── sherpa-onnx-paraformer-zh-2023-09-14/
   ├── sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/
   └── hr/
       ├── dict/
       ├── lexicon.txt
       └── replace.fst
   ```

3. 启动：`pythonw monikavoice.py`（无窗口后台运行），或双击 `start_voice_input.bat`。

## 使用

| 操作 | 方式 |
|------|------|
| 开/停听写 | Alt+R（或自动回退的 Alt+T / Alt+W，或点托盘图标） |
| 退出 | Ctrl+Alt+Q，或托盘右键 → 退出 |
| 上屏 | 说完一句停顿约 0.4 s，文字自动打进光标处 |

## 历史记录与 API

程序启动后在本机起一个只监听 `127.0.0.1` 的 HTTP 服务（默认端口 `8397`，被占用时自动停用并在日志说明）：

- 网页查看：浏览器打开 <http://127.0.0.1:8397/>，或托盘右键 → 历史记录（网页）。已上屏/未上屏、是否精修、时间一目了然，每 3 秒自动刷新。
- 接口（供其他程序调用）：

| 接口 | 说明 |
|------|------|
| `GET /api/history?limit=100&since_id=0` | 按新到旧返回记录，`since_id` 用于增量拉取 |
| `GET /api/status` | 返回 `{running, hotkey, total}` |
| `POST /api/clear` | 清空历史（内存与 history.jsonl） |

记录同时追加写入 `monikavoice.py` 同目录的 `history.jsonl`（每行一条 JSON），重启后自动载入尾部继续。`history.jsonl` 含语音内容，已在 `.gitignore` 中排除。

## 已知限制

- Alt+R 若被 NVIDIA App 的 Overlay 性能显示占用，会自动退回备用热键；要夺回 Alt+R 需先在 NVIDIA App 设置中关闭该快捷键
- 断句灵敏度、热键、模型路径集中在 `monikavoice.py` 顶部常量，可自行调整
- `ctypes.SendInput` 的 INPUT 结构体在 64 位下必须按 40 字节对齐（ unions 按 MOUSEINPUT 取最大成员），否则按键会被系统静默丢弃——二次开发时注意

## License

Apache-2.0
