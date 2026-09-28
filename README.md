# monikavoice

Windows 离线中文语音听写与环境记录工具：常态录音自动过滤环境噪音，按热键说话文字打进当前焦点输入框。
全程本地推理，**不联网、无广告、无第三方输入法**。命名与 [monikasearch](https://github.com/Efold499e/monikasearch) 同风格。

## 功能

- **全局热键开关**：Alt+R 开/停听写（被 NVIDIA Overlay 等占用时自动退回 Alt+T / Alt+W），Ctrl+Alt+Q 退出，托盘右键菜单兜底
- **实时跟随浮窗**：听写时弹出置顶小浮窗，流式显示识别中的文字，不抢目标应用焦点
- **四模型管线**：
  1. 流式 zipformer 实时出字（浮窗跟随），停顿约 0.4 s 断句
  2. **Silero VAD 人声门控**：段内人声占比过低（默认 <25%）判定为环境噪音，整段丢弃——不精修、不落日志
  3. **未完句合并**：断句碎片若不含句末标点，自动与下一段合并后整段重精修（解决"输入"被切碎成"初"+"呼入"、标点按碎片硬加句号的问题），句末标点（或 30 s 超时/静音 6 s）才落盘
  4. 离线 Paraformer 重识别 + **同音替换纠错**（专治"语音识别同音错别字"）+ ct-transformer 自动补标点
- **托盘常驻图标**：左键开关听写，右键退出
- **声纹识别**：逐句标注说话人来源，可选 `speaker_allowlist_only` 白名单模式（只保留已注册说话人）；自助录入/管理走网页表单
- **人声增强**：GT-CRN 降噪模型接入精修与转写路径，嘈杂环境的人声更干净、识别更准
- **上课/长录音会话**（宿曜适配）：`/api/session/start` 开启后，所有落日志的语音同步写入 `sessions/session-<时间戳>.md`，结束后可一键 DeepSeek 提炼要点追加到文末；`/api/transcribe` 支持数小时录音文件（VAD 按人声切分，非 wav 格式自动 ffmpeg 转换）
- **历史记录**：全部识别内容（含未上屏部分）持久化到 `history.jsonl`，内置网页查看页，并提供本机 HTTP API 供其他程序调用

## 工作原理

```
麦克风(16k，常态录音) ──► 流式 zipformer ──► 浮窗实时跟随（仅上屏模式）
                   │ 停顿 0.4s 断句
                   ▼
          Silero VAD 人声门控 ── 人声占比过低 ──► 整段丢弃（不落日志）
                   ▼
            GT-CRN 人声增强降噪
                   ▼
        未完句合并（无句末标点则与下一段合并重精修）
                   ▼
     离线 Paraformer 重识别 + 同音替换纠错
                   ▼
          ct-transformer 自动标点
                   ▼
     CampPlus 声纹标注来源（我 / 他人N / 未知）
                   ▼
   ┌─ 记录模式(默认) ─► 按天日志 logs/voice-日期.log + 会话文件（不上屏）
   │
   └─ 上屏模式(唤出后) ─► SendInput(UNICODE) 打进当前输入框
```

- 切换模式（热键/托盘/接口）瞬间的半句会被截断、只进日志不上屏。
- 每天 02:00 用 DeepSeek 总结前一天 02:01 起的全部内容到 `summaries/日期.md`；
  同时清理 6 个月前的语音日志与历史。电脑 2 点关机错过时，开机后自动补跑。
  API key 等配置在 `config.json`（不入库），模型名默认 `deepseek-chat`。
- 断句静音、VAD 阈值、合并超时等参数集中在 `monikavoice.py` 顶部 `DEFAULT_CONFIG`，
  可在 `config.json` 里覆盖（`end_silence_s` / `vad_enable` / `vad_speech_ratio` /
  `merge_max_seconds` / `merge_gap_seconds` / `speaker_allowlist_only`）。

## 基于的开源项目

| 组件 | 用途 | 许可证 |
|------|------|--------|
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | 语音识别/标点/同音替换推理框架 | Apache-2.0 |
| [streaming zipformer 中英双语模型](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) | 流式实时识别 | Apache-2.0 |
| [Paraformer zh（阿里 FunASR）](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) | 离线整句精修 | Apache-2.0 |
| [ct-transformer 标点模型（FunASR）](https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models) | 自动加标点 | Apache-2.0 |
| [hr-files（dict / lexicon / replace.fst）](https://github.com/k2-fsa/sherpa-onnx/releases/tag/hr-files) | 同音替换资源 | Apache-2.0 |
| [3D-Speaker CampPlus 声纹模型](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models) | 说话人来源标注 | Apache-2.0 |
| [GT-CRN 降噪模型](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models) | 人声增强 | Apache-2.0 |
| [Silero VAD](https://github.com/snakers4/silero-vad) | 人声门控（环境噪音不入识别） | MIT |
| [jieba 词典](https://github.com/fxsjy/jieba) | 同音替换分词 | MIT |
| [python-sounddevice / PortAudio](https://python-sounddevice.readthedocs.io/) | 麦克风采集 | MIT |
| [pystray](https://pystray.readthedocs.io/) | 托盘图标 | LGPL-3.0 |
| tkinter | 实时浮窗 | PSF |

感谢 k2-fsa / Next-gen Kaldi 社区与 FunASR 团队的模型工作。

## 安装

1. Python ≥ 3.9，安装依赖（仓库含 `pyproject.toml`，也可 `pip install .`）：

   ```
   pip install sherpa-onnx sounddevice pystray pillow numpy
   ```

2. 下载模型（Windows 可把 wget 换成浏览器下载，解压到脚本同目录）：

   ```
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

   mkdir hr
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/dict.tar.bz2
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/lexicon.txt
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/replace.fst
   ```

   声纹与降噪模型（可选）：`campplus.onnx`、`gtcrn_simple.onnx` 同在 sherpa-onnx
   releases 的 speaker-recongition-models / speech-enhancement-models 标签下。

   解压后目录结构：

   ```
   monikavoice/
   ├── monikavoice.py
   ├── pyproject.toml
   ├── icon64.png
   ├── silero_vad.onnx
   ├── sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/
   ├── sherpa-onnx-paraformer-zh-2023-09-14/
   ├── sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/
   └── hr/
       ├── dict/
       ├── lexicon.txt
       └── replace.fst
   ```

3. 启动：`pythonw monikavoice.py`（无窗口后台运行），或双击 `start_monikavoice.bat`；
   开机自启把 `start_monikavoice_autostart.bat` 的快捷方式放进 `shell:startup`。

## 使用

| 操作 | 方式 |
|------|------|
| 开/停听写 | Alt+R（或自动回退的 Alt+T / Alt+W，或点托盘图标） |
| 退出 | Ctrl+Alt+Q，或托盘右键 → 退出 |
| 上屏 | 说完一句停顿约 0.4 s，文字自动打进光标处 |
| 上课录制 | 调 `POST /api/session/start`，全程自动转写入 `sessions/`，结束时 `stop` 可顺带出要点 |

## 历史记录与 API

程序启动后在本机起一个只监听 `127.0.0.1` 的 HTTP 服务（默认端口 `8397`，被占用时自动停用并在日志说明）：

- 网页查看：浏览器打开 <http://127.0.0.1:8397/>，或托盘右键 → 历史记录（网页）。已上屏/未上屏、是否精修、时间一目了然，每 3 秒自动刷新。
- 接口（供其他程序调用）：

| 接口 | 说明 |
|------|------|
| `GET /api/history?limit=100&since_id=0` | 按新到旧返回记录，`since_id` 用于增量拉取 |
| `GET /api/status` | 返回 `{running, mode, hotkey, total}` |
| `GET /api/summary?date=YYYY-MM-DD` | 取某天的 DeepSeek 总结（`/api/summaries` 列出全部日期） |
| `POST /api/mode` | `{"mode":"commit"/"ambient"}` 或 `{"toggle":true}` 切换上屏/记录 |
| `POST /api/summarize` | `{"date":"YYYY-MM-DD"}` 立即生成某天总结 |
| `POST /api/transcribe` | `{"path":"D:\\x.wav","log":true}` 转写音频文件，支持 mp3/m4a/flac/mp4 等（非 wav 自动 ffmpeg 转换），长录音按 VAD 人声段切分 |
| `POST /api/session/start` | `{"title":"宿曜第x讲"}` 开启上课/长录音会话，此后全部语音同步转录到 `sessions/` |
| `POST /api/session/stop` | `{"summarize":true}` 结束会话；true 时 DeepSeek 提炼要点追加到转录文末并随响应返回 |
| `GET /api/session/status` · `GET /api/sessions` | 会话状态 / 历史会话列表 |
| `GET /api/speakers` | 列出已注册声纹 |
| `POST /api/enroll` | `{"name":"张三","path":"D:\\x.wav"}` 自助录入声纹 |
| `POST /api/who` | `{"path":"D:\\x.wav"}` 判断音频最像哪个已注册说话人 |
| `POST /api/forget` | `{"name":"张三"}` 删除声纹 |
| `POST /api/clear` | 清空历史（内存与 history.jsonl） |

服务只监听 `127.0.0.1`；在 `config.json` 配置 `api_token` 后，所有 POST 需带 `X-Token` 头。配套的 AI 调用说明见 `~/.agents/monikavoice/SKILL.md`。

记录同时追加写入 `monikavoice.py` 同目录的 `history.jsonl`（每行一条 JSON），重启后自动载入尾部继续；语音原文按天写入 `logs/voice-YYYY-MM-DD.log`。`history.jsonl`、`logs/`、`config.json` 均含隐私内容，已在 `.gitignore` 中排除。

## 已知限制

- Alt+R 若被 NVIDIA App 的 Overlay 性能显示占用，会自动退回备用热键；要夺回 Alt+R 需先在 NVIDIA App 设置中关闭该快捷键
- 断句静音、VAD 阈值、合并超时、热键、模型路径集中在 `monikavoice.py` 顶部（`DEFAULT_CONFIG` 与常量区），可自行调整
- 记录模式下，不含句末标点的半句会持有最多 `merge_max_seconds`（默认 45 s）才落盘，属预期行为；落盘精修在独立线程完成，不阻塞流式识别
- `ctypes.SendInput` 的 INPUT 结构体在 64 位下必须按 40 字节对齐（ unions 按 MOUSEINPUT 取最大成员），否则按键会被系统静默丢弃——二次开发时注意

## License

Apache-2.0
