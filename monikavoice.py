# -*- coding: utf-8 -*-
"""
monikavoice —— Windows 离线中文语音听写（本地运行，无广告，不联网）

- 常态录音：程序运行即持续监听麦克风，Silero VAD 判定为人声的句子经
  "离线 Paraformer 精修 + 同音替换纠错 + ct-transformer 标点"后按时间写入
  logs/voice-日期.log 与 history.jsonl（记录模式，不上屏）；未完句自动与
  下一段合并再精修，避免固定断句把半句话切错、标点打碎
- 唤出上屏：Alt+R/Alt+T（被占用自动回退）切换到上屏模式，切换瞬间会把正在录的
  半句截断、只上 log 不上屏；之后断句的内容经同一管线打进当前焦点输入框；
  再次按热键收回记录模式
- 实时浮窗：仅上屏模式弹出，流式跟随识别内容，不抢目标应用焦点
- 历史与 API：http://127.0.0.1:8397/ 网页查看；/api/history /api/status
  /api/mode /api/transcribe /api/session（宿曜上课长录音会话）等接口供 AI/程序调用
- 每日任务：02:00 用 DeepSeek 总结前一天 02:01 起的全部内容到 summaries/；
  同时清理 6 个月前的历史与日志
"""
import ctypes
import ctypes.wintypes as wt
import glob
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import webbrowser
from collections import deque
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from urllib import request as urlrequest

import numpy as np
import sherpa_onnx
import sounddevice as sd
import pystray
from PIL import Image as PILImage

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

VK_R, VK_T, VK_W, VK_Q = 0x52, 0x54, 0x57, 0x51
MOD_ALT, MOD_CONTROL = 0x0001, 0x0002
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
HOTKEY_TOGGLE, HOTKEY_QUIT = 1, 2
SAMPLE_RATE = 16000

BASE = os.path.dirname(os.path.abspath(__file__))
VERSION = "0.2.1"
MODEL_DIR = os.path.join(BASE, "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20")
PUNCT_DIR = os.path.join(BASE, "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12")
PARAFORMER_DIR = os.path.join(BASE, "sherpa-onnx-paraformer-zh-2023-09-14")
HR_DIR = os.path.join(BASE, "hr")
LOGS_DIR = os.path.join(BASE, "logs")
SUMMARIES_DIR = os.path.join(BASE, "summaries")
SESSIONS_DIR = os.path.join(BASE, "sessions")
LOG_PATH = os.path.join(BASE, "monikavoice.log")
HISTORY_PATH = os.path.join(BASE, "history.jsonl")
CONFIG_PATH = os.path.join(BASE, "config.json")
ICON64 = os.path.join(BASE, "icon64.png")
SPEAKER_MODEL = os.path.join(BASE, "campplus.onnx")
DENOISER_MODEL = os.path.join(BASE, "gtcrn_simple.onnx")
VAD_MODEL = os.path.join(BASE, "silero_vad.onnx")
SPEAKERS_JSON = os.path.join(BASE, "speakers.json")
HTTP_PORT = 8397
HISTORY_MAX = 2000
RETAIN_DAYS = 183  # 六个月
VAD_WINDOW = 512  # silero VAD 固定窗口（16kHz 下 32ms）

# 句子终结标点：ambient 未完句会继续与下一段合并，直到出现终结标点才落盘
TERMINAL_RE = re.compile(r"[。！？；…!?;.；]\s*[」”’》\]）)]*\s*$")

# 转写接口接受的媒体格式（非 wav 走 ffmpeg 转 16k 单声道）
AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus",
              ".wma", ".mp4", ".mov", ".mkv", ".webm")

# 推理模型互斥锁：dictation 线程与 HTTP 线程共享 refiner/punct/denoiser，
# sherpa-onnx 对象不支持并发解码，必须串行
model_lock = threading.Lock()

# 会话（宿曜上课长录音）：全局状态，voice_log 钩子自动写入，HTTP 线程启停。
# 文件名永远由字面量 "session-" + 纯时间戳拼出（用户标题只写进文件内容），
# 每次写入都走 open/append/close，不保留跨线程的持久句柄
SESSION = {"active": False, "title": "", "start": "", "stamp": "", "lines": 0, "parts": []}
SESSION_LOCK = threading.Lock()


def _session_append(line):
    """会话转录追加一行（文件名不含任何用户输入）"""
    name = os.path.basename("session-" + SESSION["stamp"] + ".md")
    with open(os.path.join(SESSIONS_DIR, name), "a", encoding="utf-8") as f:
        f.write(line)


def _session_summary_path():
    return os.path.join(SESSIONS_DIR, os.path.basename("session-" + SESSION["stamp"] + ".summary.md"))

# 固定的 DeepSeek 开放接口：协议与主机白名单硬编码，请求前再校验一次
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_API_HOSTS = {"api.deepseek.com"}

DEFAULT_CONFIG = {
    "deepseek_api_key": "",
    "deepseek_model": "deepseek-chat",
    "summary_hour": 2,
    "api_token": "",
    "denoise_enable": True,
    "speaker_enable": True,
    "speaker_threshold": 0.5,
    "auto_enroll_user": False,  # 上屏语音是否自动注册"我"（默认关，避免录错人；手动录入更可控）
    "end_silence_s": 0.4,       # 断句静音时长；ambient 未完句会自动合并，此值只影响断句粒度与上屏延迟
    "vad_enable": True,         # Silero VAD 人声门控：非人声段不精修、不落日志
    "vad_speech_ratio": 0.25,   # 段内人声窗口占比低于此值判定为环境噪音
    "merge_max_seconds": 45.0,  # ambient 未完句最长持有音频（超时强制落盘）
    "merge_gap_seconds": 12.0,  # ambient 静音超过该秒数，未完句先落盘
    "speaker_allowlist_only": False,  # True 时 ambient 只保留命中已注册声纹的句子
}


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    except Exception:
        pass
    return cfg


def log(*args):
    line = " ".join(str(a) for a in args)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(time.strftime("[%m-%d %H:%M:%S] ") + line + "\n")
    except OSError:
        pass


def voice_log(text, committed, speaker=""):
    """语音内容按天落盘：logs/voice-YYYY-MM-DD.log
    每行 [时间] [说话人] 标记 文本。会话（上课录音）开启时同步写入会话转录"""
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        name = os.path.basename("voice-" + time.strftime("%Y-%m-%d") + ".log")
        tag = "[" + speaker + "] " if speaker else ""
        line = time.strftime("[%H:%M:%S] ") + tag + ("上屏 " if committed else "记录 ") + text + "\n"
        with open(os.path.join(LOGS_DIR, name), "a", encoding="utf-8") as f:
            f.write(line)
    except OSError as exc:
        log("语音日志写入失败:", exc)
        return
    if SESSION["active"]:
        try:
            tag = "[" + speaker + "] " if speaker else ""
            _session_append(time.strftime("[%H:%M:%S] ") + tag + text + "\n")
            SESSION["lines"] += 1
            SESSION["parts"].append(text)
        except OSError as exc:
            log("会话转录写入失败:", exc)


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def valid_date(s):
    """严格校验 YYYY-MM-DD，防止路径拼接注入"""
    if not DATE_RE.match(s or ""):
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


# ---------------- 历史存储 ----------------

class HistoryStore:
    """识别历史：内存 deque + JSONL 追加落盘，重启后载入尾部继续"""

    def __init__(self, path, maxlen=HISTORY_MAX):
        self.path = path
        self.lock = threading.Lock()
        self.items = deque(maxlen=maxlen)
        self.next_id = 1
        self._load_tail()

    def _load_tail(self):
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                tail = deque(f, maxlen=HISTORY_MAX)
            for line in tail:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            if self.items:
                self.next_id = self.items[-1].get("id", 0) + 1
        except OSError as exc:
            log("历史文件读取失败:", exc)

    def add(self, kind, text, committed, refined, speaker=""):
        entry = {
            "id": self.next_id,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "type": kind,          # final=上屏, flush=切换截断, ambient=常态记录, transcribe=文件转写
            "committed": bool(committed),
            "refined": bool(refined),
            "speaker": speaker,
            "text": text,
        }
        with self.lock:
            self.items.append(entry)
            self.next_id += 1
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except OSError as exc:
                log("历史写入失败:", exc)

    def query(self, limit=100, since_id=0):
        with self.lock:
            rows = [e for e in self.items if e["id"] > since_id]
        rows.sort(key=lambda e: e["id"], reverse=True)
        return rows[:max(1, min(limit, HISTORY_MAX))]

    def clear(self):
        with self.lock:
            self.items.clear()
            try:
                os.remove(self.path)
            except OSError:
                pass

    def reload(self):
        with self.lock:
            self.items.clear()
        self.next_id = 1
        self._load_tail()


# ---------------- 打字 ----------------

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


class _INPUT(ctypes.Structure):
    # union 必须按最大成员 MOUSEINPUT（32 字节）对齐，否则整个结构体
    # 小于系统要求的 40 字节，SendInput 会静默拒绝全部按键
    class _U(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("pad", ctypes.c_ulonglong * 4)]
    _anonymous_ = ("u",)
    _fields_ = [("type", ctypes.c_ulong), ("u", _U)]


assert ctypes.sizeof(_INPUT) == 40, ctypes.sizeof(_INPUT)

KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_KEYUP = 0x0002

user32.SendInput.restype = ctypes.c_uint
user32.SendInput.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_int]
user32.GetForegroundWindow.restype = ctypes.c_void_p
user32.SetForegroundWindow.restype = ctypes.c_bool
user32.SetForegroundWindow.argtypes = [ctypes.c_void_p]
user32.RegisterHotKey.restype = ctypes.c_bool
user32.RegisterHotKey.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_uint]


def foreground_title():
    hwnd = user32.GetForegroundWindow()
    buf = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(hwnd, buf, 256)
    return buf.value


def send_text(text):
    """把文本以 Unicode 按键打进当前焦点窗口"""
    log("[打字目标窗口]", foreground_title(), "文本:", text)
    arr = []
    for ch in text:
        scan = ord(ch)
        if scan > 0xFFFF:
            continue
        for flags in (KEYEVENTF_UNICODE, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP):
            inp = _INPUT()
            inp.type = 1
            inp.ki = KEYBDINPUT(0, scan, flags, 0, None)
            arr.append(inp)
    n = len(arr)
    if not n:
        return
    buf = (_INPUT * n)(*arr)
    for i in range(0, n, 40):
        group = ((_INPUT * min(40, n - i)).from_buffer_copy(buf, i * ctypes.sizeof(_INPUT)))
        sent = user32.SendInput(len(group), group, ctypes.sizeof(_INPUT))
        if sent != len(group):
            log("[SendInput 只成功]", sent, "/", len(group))
        time.sleep(0.005)


# ---------------- 模型加载 ----------------

def build_recognizer(cfg):
    files = {}
    for key, pat in (("encoder", "*encoder*.onnx"), ("decoder", "*decoder*.onnx"),
                     ("joiner", "*joiner*.onnx"), ("tokens", "*tokens.txt")):
        hits = sorted(glob.glob(os.path.join(MODEL_DIR, pat)), reverse=True)  # int8 优先
        if not hits:
            log("缺少模型文件:", pat)
            sys.exit(1)
        files[key] = hits[0]

    return sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=files["encoder"],
        decoder=files["decoder"],
        joiner=files["joiner"],
        tokens=files["tokens"],
        num_threads=4,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        decoding_method="modified_beam_search",
        enable_endpoint_detection=True,
        rule2_min_trailing_silence=float(cfg.get("end_silence_s", 0.4)),
        rule3_min_utterance_length=20.0,
    )


def load_offline_refiner():
    """离线 Paraformer + 同音替换：断句后对整段音频精修，纠正同音错别字"""
    model = os.path.join(PARAFORMER_DIR, "model.int8.onnx")
    tokens = os.path.join(PARAFORMER_DIR, "tokens.txt")
    lexicon = os.path.join(HR_DIR, "lexicon.txt")
    fst = os.path.join(HR_DIR, "replace.fst")
    ddir = os.path.join(HR_DIR, "dict")
    if not (os.path.isfile(model) and os.path.isfile(tokens)):
        log("离线精修模型缺失，最终文本改用流式结果")
        return None
    try:
        rec = sherpa_onnx.OfflineRecognizer.from_paraformer(
            model, tokens,
            hr_lexicon=lexicon if os.path.isfile(lexicon) else "",
            hr_rule_fsts=fst if os.path.isfile(fst) else "",
            hr_dict_dir=ddir if os.path.isdir(ddir) else "",
            num_threads=4, provider="cpu",
        )
        log("离线精修(Paraformer+同音替换)已加载")
        return rec
    except Exception as exc:
        log("离线精修加载失败，最终文本改用流式结果:", exc)
        return None


def load_punctuation():
    """离线标点模型（中英文），加载失败则降级为不加标点"""
    hits = sorted(glob.glob(os.path.join(PUNCT_DIR, "*model*.onnx")), reverse=True)
    if not hits:
        log("未找到标点模型，标点功能关闭")
        return None
    try:
        cfg = sherpa_onnx.OfflinePunctuationConfig(
            model=sherpa_onnx.OfflinePunctuationModelConfig(ct_transformer=hits[0], num_threads=2),
        )
        punct = sherpa_onnx.OfflinePunctuation(cfg)
        log("标点模型已加载:", os.path.basename(hits[0]))
        return punct
    except Exception as exc:
        log("标点模型加载失败，标点功能关闭:", exc)
        return None


def load_denoiser():
    """GT-CRN 人声增强/降噪，加载失败则原样透传"""
    if not os.path.isfile(DENOISER_MODEL):
        log("降噪模型缺失，人声增强关闭")
        return None
    try:
        den = sherpa_onnx.OfflineSpeechDenoiser(
            sherpa_onnx.OfflineSpeechDenoiserConfig(
                model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                    gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(model=DENOISER_MODEL),
                    num_threads=2)))
        log("人声增强(GT-CRN)已加载")
        return den
    except Exception as exc:
        log("降噪模型加载失败，人声增强关闭:", exc)
        return None


def load_vad(cfg):
    """Silero VAD 人声检测：只用作"该段是否人声"的门槛（清噪不辨义 → VAD 辨声不辨义），
    不改变流式断句逻辑。加载失败则退回无门控（行为同 0.1.x）"""
    if not cfg.get("vad_enable", True):
        log("VAD 人声门控未启用（vad_enable=false）")
        return None
    if not os.path.isfile(VAD_MODEL):
        log("VAD 模型缺失(silero_vad.onnx)，人声门控关闭，环境音会照常进入识别")
        return None
    try:
        vcfg = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model=VAD_MODEL, threshold=0.5, min_silence_duration=0.25,
                min_speech_duration=0.25, max_speech_duration=20.0,
                window_size=VAD_WINDOW),
            sample_rate=SAMPLE_RATE, num_threads=1)
        vad = sherpa_onnx.VoiceActivityDetector(vcfg, buffer_size_in_seconds=30)
        log("VAD 人声门控(Silero)已加载，人声占比阈值:",
            cfg.get("vad_speech_ratio", 0.25))
        return vad
    except Exception as exc:
        log("VAD 加载失败，人声门控关闭:", exc)
        return None


class SpeakerRegistry:
    """声纹识别：注册过的说话人（自动从上屏模式采集"我"）优先匹配，
    未匹配的按余弦相似度聚成"他人N"。speakers.json 存声纹特征（含生物特征，勿外传）"""

    MIN_SAMPLES = 1.0  # 至少 1 秒音频才计算嵌入

    @staticmethod
    def _speakers_file():
        """声纹档案路径：显式校验必须落在程序目录内，杜绝路径拼接逃逸"""
        base = os.path.realpath(BASE)
        p = os.path.realpath(os.path.join(base, "speakers.json"))
        if os.path.dirname(p) != base:
            raise ValueError("非法声纹档案路径")
        return p

    def __init__(self, cfg):
        self.enabled = bool(cfg.get("speaker_enable", True)) and os.path.isfile(SPEAKER_MODEL)
        self.threshold = float(cfg.get("speaker_threshold", 0.55))
        self.extractor = None
        self.manager = None
        self.user_centroid = None
        self.user_count = 0
        self.user_samples = []
        self.named = {}    # name -> {"samples": [[float]], "count": int, "centroid": [float]} 自助录入
        self.others = {}   # name -> centroid(list[float])
        self.next_other = 1
        self.lock = threading.Lock()
        self.embed_lock = threading.Lock()  # 声纹 extractor 不支持并发，HTTP 与听写线程共用
        if not self.enabled:
            log("声纹识别关闭（缺模型或配置禁用）")
            return
        try:
            self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
                sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=SPEAKER_MODEL, num_threads=2))
            self.manager = sherpa_onnx.SpeakerEmbeddingManager(self.extractor.dim)
            self._load()
            log("声纹识别已加载（注册说话人: " +
                (", ".join([n for n in ["我"] if self.user_centroid] + list(self.others)) or "无") + "）")
        except Exception as exc:
            log("声纹识别加载失败，来源标注关闭:", exc)
            self.enabled = False

    def _load(self):
        try:
            with open(SPEAKERS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            u = data.get("我")
            if u and u.get("centroid"):
                self.user_centroid = u["centroid"]
                self.user_count = u.get("count", len(u.get("samples", [])))
                self.user_samples = u.get("samples", [])
                self.manager.add("我", [self.user_centroid])
            for name, rec in (data.get("named") or {}).items():
                if rec and rec.get("centroid"):
                    self.named[name] = rec
                    self.manager.add(name, [rec["centroid"]])
            for name, c in (data.get("others") or {}).items():
                self.others[name] = c
                self.manager.add(name, [c])
                idx = int(re.sub(r"\D", "", name) or 0)
                self.next_other = max(self.next_other, idx + 1)
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    def _save(self):
        try:
            data = {}
            if self.user_centroid:
                data["我"] = {"centroid": self.user_centroid, "count": self.user_count,
                              "samples": self.user_samples[-8:]}
            data["others"] = self.others
            data["named"] = self.named
            with open(SPEAKERS_JSON, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except OSError as exc:
            log("声纹档案保存失败:", exc)

    def _embed(self, samples):
        if self.extractor is None or len(samples) < SAMPLE_RATE * self.MIN_SAMPLES:
            return None
        with self.embed_lock:
            st = self.extractor.create_stream()
            st.accept_waveform(SAMPLE_RATE, np.asarray(samples, dtype=np.float32))
            st.input_finished()
            emb = []
            while self.extractor.is_ready(st):
                emb = self.extractor.compute(st)
        return emb if emb else None

    @staticmethod
    def _cos(u, v):
        u, v = np.asarray(u), np.asarray(v)
        return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))

    def label(self, samples):
        """返回 (说话人名, 相似度)：只匹配已注册声纹（自助录入或"我"），
        匹配不上的一律"未知"——环境碎音不会自动创建新聚类。
        命中时质心轻微向本次嵌入漂移（EMA 0.08），自适应不同距离/姿态"""
        if not self.enabled:
            return "", 0.0
        emb = self._embed(samples)
        if emb is None:
            return "未知", 0.0
        with self.lock:
            if self.manager is not None:
                name = self.manager.search(emb, self.threshold)
                if name:
                    changed = False
                    if name == "我" and self.user_centroid:
                        self.user_centroid = [a + 0.08 * (b - a) for a, b in
                                              zip(self.user_centroid, emb)]
                        changed = True
                    elif name in self.named and self.named[name].get("centroid"):
                        c = self.named[name]["centroid"]
                        self.named[name]["centroid"] = [a + 0.08 * (b - a) for a, b in
                                                        zip(c, emb)]
                        changed = True
                    if changed:
                        self._rebuild()
                        now = time.time()
                        if now - getattr(self, "_last_save", 0) > 60:
                            self._last_save = now
                            self._save()
                    return name, 1.0
            return "未知", 0.0

    def enroll_user(self, samples):
        """上屏模式的话音 = 用户本人：采集嵌入，累计 >=2 条即注册/更新"我"质心"""
        if not self.enabled:
            return
        emb = self._embed(samples)
        if emb is None:
            return
        with self.lock:
            self.user_samples.append(list(emb))
            if len(self.user_samples) > 8:
                self.user_samples = self.user_samples[-8:]
            if len(self.user_samples) < 2:
                return
            self.user_centroid = list(np.mean(np.asarray(self.user_samples), axis=0))
            self.user_count = len(self.user_samples)
            # 清理注册前被误标成"他人N"的本人质心（与"我"几乎相同的都是早期误标）
            mislabeled = [n for n, c in self.others.items()
                          if self._cos(self.user_centroid, c) >= 0.85]
            for n in mislabeled:
                del self.others[n]
            # 覆盖式重建 manager 中"我"的质心
            self.manager = sherpa_onnx.SpeakerEmbeddingManager(self.extractor.dim)
            self.manager.add("我", [self.user_centroid])
            for name, c in self.others.items():
                self.manager.add(name, [c])
            if mislabeled:
                log("[声纹] 已纠正误标的早期说话人:", ", ".join(mislabeled))
            self._save()

    def enroll_speaker(self, name, samples):
        """自助录入：命名声纹，1 条即可注册；重复录入累积质心（最多 8 条）。
        返回 (样本数, 是否新注册)"""
        name = (name or "").strip()[:32] or "未命名"
        emb = self._embed(samples)
        if emb is None:
            return 0, False
        with self.lock:
            is_new = name not in self.named
            rec = self.named.setdefault(name, {"samples": [], "count": 0, "centroid": None})
            rec["samples"].append(list(emb))
            if len(rec["samples"]) > 8:
                rec["samples"] = rec["samples"][-8:]
            rec["count"] = len(rec["samples"])
            rec["centroid"] = list(np.mean(np.asarray(rec["samples"]), axis=0))
            self.manager.add(name, [rec["centroid"]])
            self._save()
            return rec["count"], is_new

    def forget(self, name):
        """删除一个命名声纹（"我"的自动档案也可删，会重新自动采集）"""
        removed = self.named.pop(name, None) is not None
        removed = (self.others.pop(name, None) is not None) or removed
        if name == "我":
            self.user_centroid = None
            self.user_count = 0
            self.user_samples = []
        if removed:
            self._rebuild_manager()
            self._save()
        return removed

    def _rebuild_manager(self):
        self.manager = sherpa_onnx.SpeakerEmbeddingManager(self.extractor.dim)
        if self.user_centroid:
            self.manager.add("我", [self.user_centroid])
        for name, rec in self.named.items():
            if rec.get("centroid"):
                self.manager.add(name, [rec["centroid"]])
        for name, c in self.others.items():
            self.manager.add(name, [c])

    def _rebuild(self):
        self._rebuild_manager()

    def list_speakers(self):
        with self.lock:
            named = {n: r.get("count", 0) for n, r in self.named.items()}
            if self.user_centroid:
                named["我"] = self.user_count or named.get("我", 1)
            return {"named": named, "clustered": sorted(self.others.keys())}

    def who(self, samples):
        """判断一段音频最像哪个已注册说话人。返回 (名字, 相似度, 全部分数)"""
        if not self.enabled:
            return "未知", 0.0, {}
        emb = self._embed(samples)
        if emb is None:
            return "未知", 0.0, {}
        with self.lock:
            scores = {}
            if self.user_centroid:
                scores["我"] = round(self._cos(emb, self.user_centroid), 3)
            for name, rec in self.named.items():
                if rec.get("centroid"):
                    scores[name] = round(self._cos(emb, rec["centroid"]), 3)
            for name, c in self.others.items():
                scores[name] = round(self._cos(emb, c), 3)
        if not scores:
            return "未知", 0.0, scores
        best = max(scores, key=scores.get)
        if scores[best] >= self.threshold:
            return best, scores[best], scores
        return "未知", scores[best], scores


# ---------------- 文件转写（宿曜预留） ----------------

def safe_media_path(path):
    """转写文件路径校验：真实路径、必须是常见音视频格式、禁止系统目录。
    非 wav 的格式由 _ensure_wav 借助 ffmpeg 转换"""
    if not path or not os.path.isabs(path):
        raise ValueError("需要绝对路径")
    p = os.path.realpath(os.path.abspath(path))
    if not p.lower().endswith(AUDIO_EXTS):
        raise ValueError("不支持的格式，可选: " + " ".join(AUDIO_EXTS))
    if not os.path.isfile(p):
        raise ValueError("文件不存在")
    blocked_roots = [os.environ.get("WINDIR", r"C:\Windows"),
                     os.environ.get("ProgramFiles", r"C:\Program Files"),
                     os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")]
    for root in blocked_roots:
        if root and p.lower().startswith(os.path.realpath(root).lower() + os.sep):
            raise ValueError("不允许访问系统目录")
    return p


def _ensure_wav(path):
    """非 wav 输入用 ffmpeg 转 16k 单声道 pcm 到临时目录；wav 原样返回"""
    if path.lower().endswith(".wav"):
        return path, None
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("文件不是 .wav 且系统未安装 ffmpeg，无法转换")
    stamp = str(int(time.time() * 1000))
    out = os.path.join(tempfile.gettempdir(), "monikavoice-tc-" + stamp + ".wav")
    proc = subprocess.run(
        [ffmpeg, "-y", "-i", path, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-c:a", "pcm_s16le", out],
        capture_output=True, timeout=1800)
    if proc.returncode != 0 or not os.path.isfile(out):
        raise ValueError("ffmpeg 转换失败: " + proc.stderr.decode("utf-8", "ignore")[-400:])
    return out, out  # (使用路径, 待清理路径)


def _read_wav_mono16k(path):
    """读 wav（PCM 8/16/32bit），混单声道、线性重采样到 16k。不依赖 scipy"""
    import wave
    w = wave.open(path, "rb")
    try:
        nch, sw, sr, nf = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(nf)
    finally:
        w.close()
    if sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError("不支持的位宽: %d 字节（支持 8/16/32bit PCM）" % sw)
    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)
    if sr != SAMPLE_RATE and len(data) > 1:
        x_old = np.arange(len(data), dtype=np.float64)
        n_new = int(round(len(data) * SAMPLE_RATE / sr))
        x_new = np.linspace(0.0, len(data) - 1, n_new)
        data = np.interp(x_new, x_old, data).astype(np.float32)
    return data


def _vad_split(data, vad_cfg):
    """用独立 VAD 实例把整段音频切成人声片段（避开长静音与纯噪音），
    返回 [float32 数组]。失败返回空列表（调用方回退固定窗口）"""
    try:
        vad = sherpa_onnx.VoiceActivityDetector(vad_cfg, buffer_size_in_seconds=120)
    except Exception as exc:
        log("[转写] VAD 初始化失败，退回固定窗口:", exc)
        return []
    segs = []
    pos = 0
    n = len(data)
    while pos < n:
        chunk = np.asarray(data[pos:pos + SAMPLE_RATE // 2], dtype=np.float32)  # 0.5s 一喂
        vad.accept_waveform(chunk)
        pos += len(chunk)
        while not vad.empty():
            seg = vad.front
            vad.pop()
            if seg.samples is not None and len(seg.samples) > SAMPLE_RATE // 4:
                segs.append(np.asarray(seg.samples, dtype=np.float32))
    try:
        vad.flush()
        while not vad.empty():
            seg = vad.front
            vad.pop()
            if seg.samples is not None and len(seg.samples) > SAMPLE_RATE // 4:
                segs.append(np.asarray(seg.samples, dtype=np.float32))
    except Exception:
        pass
    return segs


def transcribe_file(path, refiner, punct, denoiser=None, registry=None, log_it=True,
                    vad_cfg=None):
    """转写音频文件（宿曜上课录音用，支持数小时长文件）：
    有 VAD 时按人声段切分（识别边界干净），否则固定 30 秒窗口；
    每段降噪 → Paraformer+同音替换 → 标点 → 声纹标注。返回 (拼接文本, 分段列表)"""
    if refiner is None:
        raise RuntimeError("离线精修模型未加载")
    src, cleanup = _ensure_wav(path)
    try:
        data = _read_wav_mono16k(src)
    finally:
        if cleanup:
            try:
                os.remove(cleanup)
            except OSError:
                pass
    if vad_cfg is not None:
        chunks = _vad_split(data, vad_cfg)
        log("[转写] VAD 切分:", len(chunks), "个人声段 / 共",
            round(len(data) / SAMPLE_RATE / 60.0, 1), "分钟")
    else:
        chunks = []
    if not chunks:
        step = SAMPLE_RATE * 30
        chunks = [np.asarray(data[i:i + step], dtype=np.float32)
                  for i in range(0, max(1, len(data)), step)]
        chunks = [c for c in chunks if len(c) > SAMPLE_RATE // 10]
    texts, segs = [], []
    for seg in chunks:
        with model_lock:
            if denoiser is not None:
                try:
                    seg = denoiser.run(seg, SAMPLE_RATE).samples
                except Exception:
                    pass
            rs = refiner.create_stream()
            rs.accept_waveform(SAMPLE_RATE, np.asarray(seg, dtype=np.float32))
            refiner.decode_stream(rs)
            t = rs.result.text.strip()
            if t and punct is not None:
                try:
                    t = punct.add_punctuation(t)
                except Exception:
                    pass
        if t:
            spk = registry.label(seg) if registry else ("", 0)
            spk_name = spk[0] if spk and spk[0] else "未知"
            texts.append("[" + spk_name + "] " + t)
            segs.append({"speaker": spk_name, "text": t})
    if log_it and texts:
        base = os.path.basename(path)
        voice_log("[转写] " + base + " " + " ".join(texts), committed=False)
    return "".join(texts), segs


# ---------------- 会话：宿曜上课长录音（启停接口 + 会话总结） ----------------

def session_start(title):
    """开启一个会话：此后所有落日志的语音（ambient/上屏/转写）同步写入
    sessions/session-<时间戳>.md，标题写在文件首行。返回会话信息 dict"""
    with SESSION_LOCK:
        if SESSION["active"]:
            raise RuntimeError("已有会话进行中: " + SESSION["title"])
        SESSION.update({"active": True, "title": (title or "").strip()[:60] or "未命名",
                        "stamp": time.strftime("%Y%m%d-%H%M%S"),
                        "start": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "lines": 0, "parts": []})
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        _session_append("# 会话转录：" + SESSION["title"] +
                        "\n\n开始：" + SESSION["start"] + "\n\n")
        log("[会话] 开始:", SESSION["title"], "->", "session-" + SESSION["stamp"] + ".md")
        return session_status()


def session_stop(cfg=None, summarize=False):
    """结束会话。summarize=True 时由 HTTP 层再调 session_summarize 提炼要点"""
    with SESSION_LOCK:
        if not SESSION["active"]:
            raise RuntimeError("没有进行中的会话")
        title, lines = SESSION["title"], SESSION["lines"]
        SESSION["active"] = False
        _session_append("\n结束：" + time.strftime("%Y-%m-%d %H:%M:%S") +
                        " · 共 " + str(lines) + " 条\n")
        parts = list(SESSION.get("parts") or [])
        log("[会话] 结束:", title, "共", lines, "条")
    if summarize and parts:
        return {"summarize_pending_parts": len(parts)}
    return {}

LINE_RE = re.compile(r"\[(\d\d:\d\d:\d\d)\] (?:\[(?P<spk>[^\]]+)\] )?(?P<mark>上屏 |记录 |截断 )?(?P<text>.*)$")

FILLER_ONLY_RE = re.compile(
    r"^(?:嗯+|呃+|啊+|哎+|哦+|噢+|对+|好+|是+|行|去|切|喂|嘘|没有|不是|可以|谢谢|"
    r"ok|yeah+|hello|哈+|好酷啊|我不对|这个|那个)+$", re.I)


def is_noise_line(text):
    """确定性噪音过滤：超短碎片/单字母数字/解码复读/纯语气词。
    只用于总结前的输入净化，原始日志永不改动"""
    core = re.sub(r"[，。！？、…\s,.!?：:\"'（）()\[\]—\-]", "", text)
    if len(core) <= 4:
        return True
    if re.fullmatch(r"[A-Za-z0-9]+", core):
        return True
    if re.search(r"(.)\1{3,}", core):             # 据据据据（单字复读）
        return True
    if re.search(r"(\S{1,6})(\s?\1){3,}", text):  # the the the the（词组复读）
        return True
    if FILLER_ONLY_RE.match(core):
        return True
    return False


def voice_log_path(date):
    """某天的语音日志文件名（basename 消化任何路径成分）"""
    return os.path.join(LOGS_DIR, os.path.basename("voice-" + date.strftime("%Y-%m-%d") + ".log"))


def collect_window(start_dt, end_dt):
    """收集 [start_dt, end_dt] 内的语音日志，返回 [(显示行, 纯文本)]"""
    lines = []
    d = start_dt.date()
    while d <= end_dt.date():
        p = voice_log_path(d)
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        m = LINE_RE.match(line.rstrip("\n"))
                        if not m:
                            continue
                        t = datetime.combine(d, datetime.strptime(m.group(1), "%H:%M:%S").time())
                        if start_dt <= t <= end_dt:
                            spk = "[" + m.group("spk") + "] " if m.group("spk") else ""
                            disp = "[" + d.strftime("%m-%d ") + m.group(1) + "] " + spk + m.group("text")
                            lines.append((disp, m.group("text")))
            except OSError:
                pass
        d += timedelta(days=1)
    return lines


def _guarded_urlopen(req, timeout):
    """外发请求守卫：仅允许白名单域名的 https 请求"""
    u = urlparse(req.full_url)
    if u.scheme != "https" or u.hostname not in DEEPSEEK_API_HOSTS:
        raise ValueError("blocked api host: " + str(u.hostname))
    return urlrequest.urlopen(req, timeout=timeout)


def deepseek_summarize(cfg, range_label, transcript, mode="full"):
    """mode: full=对已过滤内容出完整总结；points=长日志分段提取要点；combine=合并各段要点"""
    if mode == "points":
        system = ("你是语音日志整理助手。以下是一天中某一时段的语音识别内容（已预过滤噪音）。"
                  "请提取其中有效信息和要点：有信息量的句子按主题归组；无意义的忽略。"
                  "简体中文，300 字以内，只输出要点。")
        user = "时段：" + range_label + "\n语音内容：\n" + transcript
    elif mode == "combine":
        system = ("你是个人语音日志整理助手。以下是同一天各时段的要点摘录，请合并为一份"
                  "最终总结：按主题分组给出小标题和要点，明显待办单独列出；重复内容合并；"
                  "过滤后没有有效内容就直说\"该时段无有效记录\"。简体中文，600 字以内。")
        user = "时间范围：" + range_label + "\n各时段要点：\n" + transcript
    else:
        system = ("你是个人语音日志整理助手。输入是用户通过环境常开录音记录的原始内容，"
                  "已经过程序预过滤，但可能仍有残留杂音，请再甄别一次：\n"
                  "1) 剔除无意义碎片、语气词、没说完的半句；\n"
                  "2) 剔除媒体/视频/游戏声音和与用户无关的对话；\n"
                  "3) 剩余内容按主题分组，给出小标题和要点，明显待办单独列出；\n"
                  "4) 过滤后没有有效内容就直说\"该时段无有效记录\"，不要硬凑；\n"
                  "5) 简体中文，600 字以内。")
        user = "时间范围：" + range_label + "\n语音内容：\n" + transcript
    body = json.dumps({
        "model": cfg.get("deepseek_model", "deepseek-chat"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
        "max_tokens": 2000,
    }, ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(
        DEEPSEEK_API_URL, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + cfg.get("deepseek_api_key", "")})
    with _guarded_urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def session_status():
    return {"active": SESSION["active"], "title": SESSION["title"],
            "start": SESSION["start"], "lines": SESSION["lines"],
            "file": ("session-" + SESSION["stamp"] + ".md") if SESSION["active"] else ""}


def list_sessions():
    out = []
    if os.path.isdir(SESSIONS_DIR):
        for p in sorted(glob.glob(os.path.join(SESSIONS_DIR, "*.md"))):
            try:
                out.append({"file": os.path.basename(p),
                            "size": os.path.getsize(p),
                            "mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                                                   time.localtime(os.path.getmtime(p)))})
            except OSError:
                continue
    return out


def session_summarize(cfg):
    """把最近结束的会话内容交给 DeepSeek 提炼要点，追加写入会话转录文件尾部。
    返回要点文本（失败时返回错误说明）"""
    parts = list(SESSION.get("parts") or [])
    if not parts:
        return "（会话无有效内容）"
    label = "会话「" + SESSION["title"] + "」 " + SESSION["start"]
    try:
        if len("".join(parts)) <= 20000:
            text = deepseek_summarize(cfg, label, "\n".join(parts), mode="points")
        else:
            chunks, cur, cur_len = [], [], 0
            for p in parts:
                cur.append(p)
                cur_len += len(p) + 1
                if cur_len >= 20000:
                    chunks.append(cur)
                    cur, cur_len = [], 0
            if cur:
                chunks.append(cur)
            log("[会话] 内容较长，分", len(chunks), "段提炼")
            points = [deepseek_summarize(cfg, label, "\n".join(ch), mode="points")
                      for ch in chunks]
            text = deepseek_summarize(cfg, label, "\n\n".join(points), mode="combine")
    except Exception as exc:
        log("[会话] 总结失败:", exc)
        text = "> 总结失败：" + str(exc)
    _session_append("\n## 要点\n\n" + text + "\n")
    log("[会话] 要点已追加至 session-" + SESSION["stamp"] + ".md")
    return text


def summarize_for_date(date_str, cfg, store):
    """总结 date_str 当天 summary_hour+1 分到次日 summary_hour 点的内容"""
    day = valid_date(date_str)
    if day is None:
        raise ValueError("非法日期: " + str(date_str))
    hour = int(cfg.get("summary_hour", 2))
    start_dt = datetime(day.year, day.month, day.day, hour) + timedelta(minutes=1)
    end_dt = start_dt + timedelta(days=1) - timedelta(minutes=1)
    raw = collect_window(start_dt, end_dt)
    kept, dropped = [], 0
    prev_text = None
    for disp, text in raw:
        if text == prev_text or is_noise_line(text):
            dropped += 1
            continue
        prev_text = text
        kept.append(disp)
    log("[每日总结] 原始", len(raw), "条，过滤", dropped, "条噪音，有效", len(kept), "条")
    os.makedirs(SUMMARIES_DIR, exist_ok=True)
    out_path = os.path.join(SUMMARIES_DIR, os.path.basename(date_str + ".md"))
    if not kept:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# " + date_str + " 语音日志总结\n\n（该时段无记录）\n")
        return out_path
    chunks, cur, cur_len = [], [], 0
    for disp in kept:
        cur.append(disp)
        cur_len += len(disp) + 1
        if cur_len >= 20000:
            chunks.append(cur)
            cur, cur_len = [], 0
    if cur:
        chunks.append(cur)
    range_label = start_dt.strftime("%Y-%m-%d %H:%M") + " 至 " + end_dt.strftime("%Y-%m-%d %H:%M")
    try:
        if len(chunks) == 1:
            text = deepseek_summarize(cfg, range_label, "\n".join(chunks[0]))
        else:
            log("[每日总结] 有效内容较长，分", len(chunks), "段处理")
            points = []
            for i, ch in enumerate(chunks):
                try:
                    points.append(deepseek_summarize(cfg, range_label, "\n".join(ch), mode="points"))
                except Exception as exc:
                    log("[每日总结] 段", i + 1, "失败:", exc)
            if not points:
                raise RuntimeError("所有分段均失败")
            text = deepseek_summarize(cfg, range_label, "\n\n".join(points), mode="combine")
    except Exception as exc:
        log("[每日总结] DeepSeek 调用失败:", exc)
        text = "> 总结失败：" + str(exc)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# " + date_str + " 语音日志总结\n\n范围：" + range_label +
                " · 原始 " + str(len(raw)) + " 条 · 过滤 " + str(dropped) +
                " 条 · 有效 " + str(len(kept)) + " 条\n\n" + text + "\n")
    log("[每日总结] 已写入", out_path)
    return out_path


def cleanup_old(days=RETAIN_DAYS, store=None):
    """删除六个月前的语音日志与历史条目"""
    cutoff = datetime.now() - timedelta(days=days)
    n_logs = 0
    if os.path.isdir(LOGS_DIR):
        for p in glob.glob(os.path.join(LOGS_DIR, "voice-*.log")):
            m = re.search(r"(\d{4}-\d{2}-\d{2})\.log$", p)
            if m and datetime.strptime(m.group(1), "%Y-%m-%d") < cutoff:
                try:
                    os.remove(p)
                    n_logs += 1
                except OSError:
                    pass
    n_hist = 0
    if os.path.isfile(HISTORY_PATH):
        kept = []
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        e = json.loads(line.strip())
                        if datetime.strptime(e.get("ts", ""), "%Y-%m-%d %H:%M:%S") < cutoff:
                            n_hist += 1
                            continue
                        kept.append(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
            tmp = HISTORY_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.writelines(kept)
            os.replace(tmp, HISTORY_PATH)
        except OSError as exc:
            log("历史清理失败:", exc)
    if store is not None:
        store.reload()
    log("[清理] 删除", days, "天前日志", n_logs, "个、历史", n_hist, "条")


def daily_worker(cfg, store):
    """每天 summary_hour 点：总结昨天 + 清理旧数据"""
    hour = int(cfg.get("summary_hour", 2))
    while True:
        now = datetime.now()
        target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        wait = (target - now).total_seconds()
        log("[每日任务] 下次运行:", target.strftime("%Y-%m-%d %H:%M"))
        if wait > 0:
            time.sleep(wait)
        try:
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            summarize_for_date(yesterday, cfg, store)
        except Exception as exc:
            log("[每日任务] 总结异常:", exc)
        try:
            cleanup_old(store=store)
        except Exception as exc:
            log("[每日任务] 清理异常:", exc)
        time.sleep(60)


def catchup_worker(cfg, store):
    """开机补跑：昨天的总结缺失则补一次（电脑 2 点关机的情况）"""
    time.sleep(90)
    if not cfg.get("deepseek_api_key"):
        return
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    if not os.path.isfile(os.path.join(SUMMARIES_DIR, os.path.basename(yesterday + ".md"))):
        try:
            summarize_for_date(yesterday, cfg, store)
        except Exception as exc:
            log("[补跑总结] 失败:", exc)


# ---------------- 浮窗 ----------------

class FloatWindow:
    """上屏模式弹出的置顶小浮窗：实时显示识别内容，不抢目标应用焦点"""

    def __init__(self, ui_q, state, audio_q):
        self.q = ui_q
        self.state = state
        self.audio_q = audio_q
        self.hotkey_name = "…"
        self.last_partial = ""
        self.tray = None

        self.root = tk.Tk()
        self.root.title("monikavoice")
        self.root.configure(bg="#1f1f23")
        self.root.attributes("-topmost", True)
        self.root.withdraw()  # 常态隐藏，仅上屏模式弹出

        w, h = 380, 150
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{sw - w - 16}+{sh - h - 80}")

        self.header = tk.Label(self.root, text="", fg="#8ab4f8", bg="#1f1f23",
                               font=("Microsoft YaHei UI", 10, "bold"), anchor="w")
        self.header.pack(fill="x", padx=12, pady=(8, 2))
        self.final_lbl = tk.Label(self.root, text="", fg="#9aa0a6", bg="#1f1f23",
                                  font=("Microsoft YaHei UI", 9), anchor="w", wraplength=w - 24)
        self.final_lbl.pack(fill="x", padx=12)
        self.partial_lbl = tk.Label(self.root, text="", fg="#e8eaed", bg="#1f1f23",
                                    font=("Microsoft YaHei UI", 13), anchor="nw", wraplength=w - 24,
                                    justify="left")
        self.partial_lbl.pack(fill="both", expand=True, padx=12, pady=(2, 8))

        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        try:
            self._icon_img = tk.PhotoImage(file=ICON64)
            self.root.iconphoto(True, self._icon_img)
        except Exception:
            pass
        self.root.after(50, self.poll)

    def poll(self):
        try:
            while True:
                cmd, payload = self.q.get_nowait()
                if cmd == "mode":
                    self.set_mode(payload)
                elif cmd == "partial":
                    self.show_partial(payload)
                elif cmd == "final":
                    self.show_final(payload)
                elif cmd == "hotkey":
                    self.hotkey_name = payload or "无"
                    self.refresh_header()
                elif cmd == "quit":
                    # 停止前让识别线程把未上屏内容截断上 log，约 1 秒后退出
                    self.state["flush"].set()
                    self.root.after(1000, self.root.destroy)
                    return
        except queue.Empty:
            pass
        self.root.after(50, self.poll)

    def set_mode(self, mode):
        if mode == "toggle":
            mode = "ambient" if self.state["mode"] == "commit" else "commit"
        self.state["mode"] = mode
        self.state["flush"].set()  # 模式切换瞬间：截断半句，只上 log 不上屏
        if mode == "commit":
            with self.audio_q.mutex:
                self.audio_q.queue.clear()
            self.final_lbl.config(text="")
            self.partial_lbl.config(text="（请说话…）")
            self.refresh_header()
            hwnd = user32.GetForegroundWindow()
            self.root.deiconify()
            self.root.lift()
            self.root.after(120, lambda: user32.SetForegroundWindow(hwnd))
            log("[模式] 上屏")
        else:
            self.root.withdraw()
            log("[模式] 记录（不上屏）")
        self.update_tray_title()

    def update_tray_title(self):
        try:
            if self.tray is not None:
                self.tray.title = "monikavoice（" + ("上屏模式" if self.state["mode"] == "commit" else "记录中") + "）"
        except Exception:
            pass

    def refresh_header(self):
        if self.state["mode"] == "commit":
            self.header.config(text="● 上屏模式 · " + self.hotkey_name + " 收起", fg="#81c995")
        else:
            self.header.config(text="○ 记录中", fg="#9aa0a6")

    def show_partial(self, text):
        if text != self.last_partial:
            self.last_partial = text
            self.partial_lbl.config(text=text if text else "（请说话…）")

    def show_final(self, text):
        tail = text if len(text) <= 26 else "…" + text[-26:]
        self.final_lbl.config(text="已上屏: " + tail)
        self.last_partial = ""
        self.partial_lbl.config(text="（说话中…）")


# ---------------- 本机 HTTP 服务（页面 + API，宿曜接口预留） ----------------

PAGE_HTML = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>monikavoice 历史记录</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body{background:#141416;color:#e8eaed;font-family:"Microsoft YaHei UI",sans-serif;margin:0;padding:24px}
 h1{font-size:18px;font-weight:600;margin:0 0 4px}
 .sub{color:#9aa0a6;font-size:12px;margin-bottom:16px}
 .item{background:#1f1f23;border-radius:8px;padding:10px 14px;margin-bottom:8px}
 .meta{font-size:11px;color:#9aa0a6;margin-bottom:4px}
 .badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;margin-right:8px}
 .committed{background:#1e3a29;color:#81c995}
 .dropped{background:#3a2e1e;color:#fdd663}
 .spk-other{background:#26314a;color:#8ab4f8}
 .spk-me{background:#2a2540;color:#c58af9}
 .spk-unknown{background:#2a2a2e;color:#9aa0a6}
 .text{font-size:14px;line-height:1.6;white-space:pre-wrap;word-break:break-word}
 #empty{color:#9aa0a6;text-align:center;padding:40px 0}
 .api{color:#9aa0a6;font-size:12px;margin-top:18px;line-height:1.8}
 code{background:#1f1f23;padding:1px 6px;border-radius:4px}
</style></head><body>
<h1>monikavoice 历史记录</h1>
<div class="sub" id="status">加载中…</div>
<div id="list"></div>
<div style="margin-top:24px;padding:14px;background:#1f1f23;border-radius:8px">
 <b style="font-size:13px">声纹管理</b>
 <div style="margin-top:8px;font-size:13px">
  录入 <input id="en-name" placeholder="名字" style="width:90px;background:#141416;color:#e8eaed;border:1px solid #3c4043;border-radius:4px;padding:3px 6px">
  <button onclick="svLive('/api/record_enroll','en')">● 录 5 秒录入</button>
  或用文件
  <input id="en-path" placeholder="wav 绝对路径" style="width:250px;background:#141416;color:#e8eaed;border:1px solid #3c4043;border-radius:4px;padding:3px 6px">
  <button onclick="svPost('/api/enroll','en',this)">文件录入</button>
  <span id="en-out" class="meta"></span>
 </div>
 <div style="margin-top:6px;font-size:13px">
  判断 <button onclick="svLive('/api/record_who','who')">● 录 5 秒判断</button>
  或用文件
  <input id="who-path" placeholder="wav 绝对路径" style="width:250px;background:#141416;color:#e8eaed;border:1px solid #3c4043;border-radius:4px;padding:3px 6px">
  <button onclick="svPost('/api/who','who',this)">文件判断</button>
  <span id="who-out" class="meta"></span>
 </div>
 <div id="spk-list" class="meta" style="margin-top:8px"></div>
</div>
<div class="api">
 API：<code>GET /api/history?limit=100&amp;since_id=0</code> ·
 <code>GET /api/status</code> · <code>GET /api/summary?date=YYYY-MM-DD</code> ·
 <code>POST /api/mode /api/clear /api/summarize /api/transcribe</code> ·
 <code>POST /api/session/start</code>{"title"} · <code>POST /api/session/stop</code>{"summarize":true} ·
 <code>GET /api/sessions</code>
 （仅本机 127.0.0.1 可访问；配置 api_token 后 POST 需带 X-Token 头）
</div>
<script>
async function svLive(url, outId){
  const out = document.getElementById(outId + '-out');
  out.textContent = '● 录音中，请对着麦克风说话 5 秒…';
  try{
    const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({seconds: 5, name: document.getElementById('en-name').value})});
    const d = await r.json();
    if(url === '/api/record_enroll'){ out.textContent = d.ok ? ('已注册 '+d.name+'（样本 '+d.samples+' 条）') : ('失败: '+(d.error||'未知')); }
    else{
      const top = Object.entries(d.scores||{}).sort((a,b)=>b[1]-a[1]).slice(0,3)
        .map(([k,v])=>k+' '+v).join('，');
      out.textContent = d.ok ? ('最像: '+d.speaker+'（相似度 '+d.similarity+'）｜'+top) : ('失败: '+(d.error||'未知'));
    }
    load();
  }catch(e){ out.textContent = '请求失败'; }
}
async function svPost(url, outId, btn){
  const out = document.getElementById(outId + '-out');
  out.textContent = '处理中…';
  const body = {};
  if(outId === 'en'){ body.name = document.getElementById('en-name').value; body.path = document.getElementById('en-path').value; }
  else { body.path = document.getElementById('who-path').value; }
  try{
    const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
    const d = await r.json();
    if(url === '/api/enroll'){ out.textContent = d.ok ? ('已注册 '+d.name+'（样本 '+d.samples+' 条）') : ('失败: '+(d.error||'未知')); }
    else{
      const top = Object.entries(d.scores||{}).sort((a,b)=>b[1]-a[1]).slice(0,3)
        .map(([k,v])=>k+' '+v).join('，');
      out.textContent = d.ok ? ('最像: '+d.speaker+'（相似度 '+d.similarity+'）｜'+top) : ('失败: '+(d.error||'未知'));
    }
    load();
  }catch(e){ out.textContent = '请求失败'; }
}
async function load(){
  try{
    const r = await fetch('/api/history?limit=200');
    const d = await r.json();
    const list = document.getElementById('list');
    if(!d.items.length){ list.innerHTML = '<div id="empty">还没有记录</div>'; }
    else{
      list.innerHTML = d.items.map(e =>
        `<div class="item"><div class="meta">` +
        `<span class="badge ${e.committed?'committed':'dropped'}">${e.committed?'已上屏':'未上屏'}</span>` +
        `<span class="badge ${e.speaker==='我'?'spk-me':(e.speaker&&e.speaker!=='未知'?'spk-other':'spk-unknown')}" data-spk></span>` +
        `#${e.id} · ${e.ts} · ${e.type}${e.refined?' · 已精修':''}</div>` +
        `<div class="text"></div></div>`).join('');
      list.querySelectorAll('.text').forEach((el,i)=>{ el.textContent = d.items[i].text; });
      list.querySelectorAll('[data-spk]').forEach((el,i)=>{ el.textContent = d.items[i].speaker || '未知'; });
    }
    const s = await (await fetch('/api/status')).json();
    document.getElementById('status').textContent =
      `状态：${s.mode==='commit'?'上屏模式':'记录模式（不上屏）'} · 开关热键 ${s.hotkey||'无（用托盘图标）'} · 共 ${s.total} 条 · 每 3 秒自动刷新`;
    const sp = await (await fetch('/api/speakers')).json();
    const parts = [];
    for(const [k,v] of Object.entries(sp.named||{})) parts.push(k+'('+v+'条)');
    for(const n of (sp.clustered||[])) parts.push(n);
    document.getElementById('spk-list').textContent = '已注册声纹：' + (parts.join('，') || '无');
  }catch(e){}
}
load(); setInterval(load, 3000);
</script></body></html>"""


class HistoryServer:
    """仅绑定 127.0.0.1 的历史查询/模式控制/文件转写/总结接口"""

    def __init__(self, store, state, hotkeys, ui_q, refiner, punct, cfg,
                 denoiser=None, registry=None, capture=None, vad_cfg=None):
        self.store = store
        self.state = state
        self.hotkeys = hotkeys
        self.ui_q = ui_q
        self.refiner = refiner
        self.punct = punct
        self.cfg = cfg
        self.denoiser = denoiser
        self.registry = registry
        self.capture = capture or {}
        self.vad_cfg = vad_cfg
        self.token = (cfg.get("api_token") or "").strip()
        hs = self  # Handler 内通过闭包访问服务端状态（self 已被 Handler 实例占用）

        def _auth_ok(hdr_value):
            return (not self.token) or hdr_value == self.token

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code, body, ctype):
                data = body.encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", ctype + "; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _json(self, code, obj):
                self._send(code, json.dumps(obj, ensure_ascii=False), "application/json")

            def do_GET(self):
                u = urlparse(self.path)
                if u.path == "/":
                    self._send(200, PAGE_HTML, "text/html")
                elif u.path == "/api/history":
                    q = parse_qs(u.query)
                    limit = int(q.get("limit", ["100"])[0])
                    since = int(q.get("since_id", ["0"])[0])
                    rows = store.query(limit=limit, since_id=since)
                    self._json(200, {"items": rows,
                                     "last_id": rows[0]["id"] if rows else since})
                elif u.path == "/api/status":
                    with store.lock:
                        total = len(store.items)
                    self._json(200, {"running": state["mode"] == "commit",
                                     "mode": state["mode"],
                                     "hotkey": hotkeys.get("name"),
                                     "total": total,
                                     "denoise": denoiser is not None,
                                     "speaker": bool(registry and registry.enabled),
                                     "vad": vad_cfg is not None,
                                     "version": VERSION,
                                     "stats": dict(state.get("stats") or {}),
                                     "session": session_status()})
                elif u.path == "/api/session/status":
                    self._json(200, session_status())
                elif u.path == "/api/sessions":
                    self._json(200, {"items": list_sessions()})
                elif u.path == "/api/summary":
                    q = parse_qs(u.query)
                    date = q.get("date", [time.strftime("%Y-%m-%d")])[0]
                    if valid_date(date) is None:
                        self._json(400, {"ok": False, "error": "date must be YYYY-MM-DD"})
                        return
                    p = os.path.join(SUMMARIES_DIR, os.path.basename(date + ".md"))
                    if os.path.isfile(p):
                        with open(p, "r", encoding="utf-8") as f:
                            self._send(200, f.read(), "text/plain")
                    else:
                        self._send(404, "no summary for " + date, "text/plain")
                elif u.path == "/api/summaries":
                    files = sorted(glob.glob(os.path.join(SUMMARIES_DIR, "*.md")))
                    self._json(200, {"dates": [os.path.basename(x)[:-3] for x in files]})
                elif u.path == "/api/speakers":
                    self._json(200, registry.list_speakers() if registry else {"named": {}, "clustered": []})
                else:
                    self._send(404, "not found", "text/plain")

            def _record_seconds(self, seconds):
                """从麦克风采集指定秒数，返回拼接的 float32 音频（独占采集通道）"""
                seconds = max(2, min(30, int(seconds or 5)))
                cap = hs.capture
                if cap.get("active"):
                    if not cap["done"].wait(timeout=2.0):
                        raise RuntimeError("已有录音在进行")
                cap["buf"] = []
                cap["need"] = SAMPLE_RATE * seconds
                cap["done"].clear()
                cap["active"] = True
                if not cap["done"].wait(timeout=seconds + 3):
                    cap["active"] = False
                    raise RuntimeError("录音超时")
                return np.concatenate(cap["buf"]).reshape(-1)

            def do_POST(self):
                u = urlparse(self.path)
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n).decode("utf-8") if n else ""
                try:
                    payload = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    # Windows 路径的反斜杠常被客户端漏转义，退回宽松提取
                    payload = {}
                    for key in ("path", "name", "date", "mode"):
                        m = re.search(r'"%s"\s*:\s*"([^"]*)"' % key, raw)
                        if m:
                            payload[key] = m.group(1).replace("\\\\", "\\")
                if not _auth_ok(self.headers.get("X-Token")):
                    self._json(401, {"ok": False, "error": "bad token"})
                    return
                if u.path == "/api/clear":
                    store.clear()
                    self._json(200, {"ok": True})
                elif u.path == "/api/enroll":
                    try:
                        p = safe_media_path(payload.get("path", ""))
                    except ValueError as exc:
                        self._json(400, {"ok": False, "error": str(exc)})
                        return
                    name = str(payload.get("name", "")).strip()
                    if not name:
                        self._json(400, {"ok": False, "error": "name required"})
                        return
                    try:
                        samples = _read_wav_mono16k(p)
                        if denoiser is not None:
                            with model_lock:
                                try:
                                    samples = denoiser.run(samples, SAMPLE_RATE).samples
                                except Exception:
                                    pass
                        count, is_new = registry.enroll_speaker(name, samples) if registry \
                            else (0, False)
                        self._json(200, {"ok": bool(count), "name": name,
                                         "samples": count, "new": is_new})
                    except Exception as exc:
                        self._json(500, {"ok": False, "error": str(exc)})
                elif u.path == "/api/who":
                    try:
                        p = safe_media_path(payload.get("path", ""))
                    except ValueError as exc:
                        self._json(400, {"ok": False, "error": str(exc)})
                        return
                    try:
                        samples = _read_wav_mono16k(p)
                        if denoiser is not None:
                            with model_lock:
                                try:
                                    samples = denoiser.run(samples, SAMPLE_RATE).samples
                                except Exception:
                                    pass
                        name, sim, scores = registry.who(samples) if registry \
                            else ("未知", 0.0, {})
                        self._json(200, {"ok": True, "speaker": name,
                                         "similarity": sim, "scores": scores})
                    except Exception as exc:
                        self._json(500, {"ok": False, "error": str(exc)})
                elif u.path == "/api/record_enroll":
                    name = str(payload.get("name", "")).strip()
                    if not name:
                        self._json(400, {"ok": False, "error": "name required"})
                        return
                    try:
                        samples = self._record_seconds(payload.get("seconds", 5))
                    except RuntimeError as exc:
                        self._json(409, {"ok": False, "error": str(exc)})
                        return
                    if denoiser is not None:
                        with model_lock:
                            try:
                                samples = denoiser.run(samples, SAMPLE_RATE).samples
                            except Exception:
                                pass
                    count, is_new = registry.enroll_speaker(name, samples) if registry \
                        else (0, False)
                    self._json(200, {"ok": bool(count), "name": name,
                                     "samples": count, "new": is_new})
                elif u.path == "/api/record_who":
                    try:
                        samples = self._record_seconds(payload.get("seconds", 5))
                    except RuntimeError as exc:
                        self._json(409, {"ok": False, "error": str(exc)})
                        return
                    if denoiser is not None:
                        with model_lock:
                            try:
                                samples = denoiser.run(samples, SAMPLE_RATE).samples
                            except Exception:
                                pass
                    name, sim, scores = registry.who(samples) if registry \
                        else ("未知", 0.0, {})
                    self._json(200, {"ok": True, "speaker": name,
                                     "similarity": sim, "scores": scores})
                elif u.path == "/api/forget":
                    name = str(payload.get("name", "")).strip()
                    removed = registry.forget(name) if registry else False
                    self._json(200, {"ok": removed, "name": name})
                elif u.path == "/api/mode":
                    mode = payload.get("mode")
                    if payload.get("toggle"):
                        mode = "ambient" if state["mode"] == "commit" else "commit"
                    if mode in ("commit", "ambient"):
                        ui_q.put(("mode", mode))
                        self._json(200, {"ok": True, "mode": mode})
                    else:
                        self._json(400, {"ok": False,
                                         "error": "mode must be commit|ambient, or toggle:true"})
                elif u.path == "/api/summarize":
                    date = payload.get("date") or (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                    if valid_date(date) is None:
                        self._json(400, {"ok": False, "error": "date must be YYYY-MM-DD"})
                        return
                    try:
                        p = summarize_for_date(date, cfg, store)
                        self._json(200, {"ok": True, "file": p})
                    except Exception as exc:
                        self._json(500, {"ok": False, "error": str(exc)})
                elif u.path == "/api/transcribe":
                    try:
                        p = safe_media_path(payload.get("path", ""))
                    except ValueError as exc:
                        self._json(400, {"ok": False, "error": str(exc)})
                        return
                    try:
                        text, segs = transcribe_file(
                            p, refiner, punct, denoiser=denoiser,
                            registry=registry,
                            log_it=bool(payload.get("log", True)),
                            vad_cfg=hs.vad_cfg)
                        store.add("transcribe", text, committed=False,
                                  refined=True, speaker="多段" if len(segs) > 1 else (segs[0]["speaker"] if segs else "未知"))
                        self._json(200, {"ok": True, "text": text,
                                         "chars": len(text), "segments": segs})
                    except Exception as exc:
                        self._json(500, {"ok": False, "error": str(exc)})
                elif u.path == "/api/session/start":
                    try:
                        info = session_start(payload.get("title", ""))
                        self._json(200, {"ok": True, **info})
                    except RuntimeError as exc:
                        self._json(409, {"ok": False, "error": str(exc)})
                elif u.path == "/api/session/stop":
                    try:
                        info = session_stop(cfg)
                        summary = ""
                        if payload.get("summarize") and cfg.get("deepseek_api_key"):
                            summary = session_summarize(cfg)
                        self._json(200, {"ok": True, **info, "summary": summary})
                    except RuntimeError as exc:
                        self._json(409, {"ok": False, "error": str(exc)})
                    except Exception as exc:
                        self._json(500, {"ok": False, "error": str(exc)})
                else:
                    self._send(404, "not found", "text/plain")

            def log_message(self, *args):  # 静默 access log
                pass

        self.Handler = Handler

    def start(self):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), self.Handler)
        except OSError as exc:
            log(f"历史页面端口 {HTTP_PORT} 被占用，接口与页面停用:", exc)
            return
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        log(f"历史页面: http://127.0.0.1:{HTTP_PORT}/")


# ---------------- 主流程 ----------------

def main():
    log("=== 启动 monikavoice ===")
    cfg = load_config()
    if not cfg.get("deepseek_api_key"):
        log("[每日总结] 未配置 deepseek_api_key（config.json），总结功能停用")
    recognizer = build_recognizer(cfg)
    refiner = load_offline_refiner()
    punct = load_punctuation()
    denoiser = load_denoiser() if cfg.get("denoise_enable", True) else None
    if denoiser is None:
        log("人声增强未启用")
    vad = load_vad(cfg)
    registry = SpeakerRegistry(cfg)
    if not cfg.get("auto_enroll_user", False):
        log("声纹自动注册已关闭（上屏语音不再自动采集；手动录入不受影响）")

    audio_q = queue.Queue()
    ui_q = queue.Queue()
    state = {"mode": "ambient", "flush": threading.Event(),  # 常态=记录模式
             "stats": {"noise_dropped": 0, "unknown_dropped": 0}}
    hotkeys = {}
    history = HistoryStore(HISTORY_PATH)

    capture = {"active": False, "buf": [], "need": 0, "done": threading.Event()}

    # ambient 断句段 → finalize 工作线程：精修/合并/落盘不再阻塞流式解码，
    # 停顿处写日志的同时新的语音照常识别
    finalize_q = queue.Queue()
    flush_done = threading.Event()  # 工作线程处理完 FLUSH 哨兵后置位

    def refine_block(block):
        """降噪 → 离线精修 → 标点，返回 (文本, 是否精修)。
        模型对象与 HTTP 线程共享（transcribe/enroll），sherpa 不支持并发，须串行"""
        text, refined = "", False
        with model_lock:
            if denoiser is not None:
                try:
                    block = denoiser.run(block, SAMPLE_RATE).samples
                except Exception as exc:
                    log("[降噪异常]", exc)
            if refiner is not None and len(block) > SAMPLE_RATE // 2:
                try:
                    t0 = time.time()
                    rs = refiner.create_stream()
                    rs.accept_waveform(SAMPLE_RATE, np.asarray(block, dtype=np.float32))
                    refiner.decode_stream(rs)
                    refined_text = rs.result.text.strip()
                    if refined_text:
                        text = refined_text
                        refined = True
                    log("[精修]", round((time.time() - t0) * 1000), "ms:", text)
                except Exception as exc:
                    log("[精修异常，用流式结果]", exc)
            if text and punct is not None:
                try:
                    text = punct.add_punctuation(text)
                except Exception as exc:
                    log("[标点异常]", exc)
        return text, refined

    def mic_callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())
        if capture["active"]:
            capture["buf"].append(indata.copy())
            if sum(len(x) for x in capture["buf"]) >= capture["need"]:
                capture["active"] = False
                capture["done"].set()

    mic = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                         blocksize=SAMPLE_RATE // 10, callback=mic_callback)
    mic.start()

    def finalize_worker():
        """ambient 断句段的精修/合并/落盘专用线程：
        VAD 门槛 → 未完句合并整段重精修 → 出现终结标点才落日志。
        独立成线程后，停顿处写日志（重精修可达 1-2 秒）期间 dictation 线程的
        流式解码照常跑，新输入不会被卡住；写日志顺序与断句顺序一致（单线程消费）"""
        merge_gap_s = float(cfg.get("merge_gap_seconds", 12.0))
        merge_max_s = float(cfg.get("merge_max_seconds", 45.0))
        vad_ratio_min = float(cfg.get("vad_speech_ratio", 0.25))
        allowlist_only = bool(cfg.get("speaker_allowlist_only", False))
        # 未完句持有状态：blocks[0]=已合并音频，text=最近一次精修+标点结果
        pending = {"blocks": [], "samples": 0, "text": "", "updated": 0.0}

        def pending_flush():
            """把持有中的未完句落日志（用最近一次合并精修的文本，不再重算）"""
            nonlocal pending
            if not pending["blocks"]:
                pending.update(samples=0, text="", updated=0.0)
                return
            audio = pending["blocks"][0]
            spk = registry.label(audio)
            speaker = spk[0] if spk and spk[0] else "未知"
            if allowlist_only and speaker == "未知":
                state["stats"]["unknown_dropped"] += 1
                log("[白名单] 丢弃未注册来源句子:", pending["text"][:30])
            elif pending["text"]:
                voice_log(pending["text"], committed=False, speaker=speaker)
                history.add("ambient", pending["text"], committed=False,
                            refined=True, speaker=speaker)
            pending = {"blocks": [], "samples": 0, "text": "", "updated": 0.0}

        while True:
            try:
                job = finalize_q.get(timeout=0.5)
            except queue.Empty:
                # 静默期：未完句超过 merge_gap 秒先落盘，避免一直挂起
                if pending["samples"] and time.time() - pending["updated"] > merge_gap_s:
                    pending_flush()
                continue
            if job == "FLUSH":
                pending_flush()
                flush_done.set()
                continue
            try:
                block = job["block"]
                # VAD 人声门槛：段内人声占比过低 = 环境噪音，整段丢弃
                if vad is not None and job["ratio"] < vad_ratio_min:
                    state["stats"]["noise_dropped"] += 1
                    continue
                # 持有中的未完句已到上限：先落盘，避免合并音频无限增长
                if pending["blocks"] and pending["samples"] >= merge_max_s * SAMPLE_RATE:
                    pending_flush()
                # 与持有中的未完句合并后整段重精修（纠正跨段切分造成的同音错字）
                audio = np.concatenate(pending["blocks"] + [block]) if pending["blocks"] else block
                text, refined = refine_block(audio)
                if not text and pending["text"]:
                    text = pending["text"]
                if not text:
                    text = job["partial"]
                if not text:
                    continue
                pending = {"blocks": [audio], "samples": len(audio),
                           "text": text, "updated": time.time()}
                # 出现句末标点（或音频超长）才落日志；半句继续持有等下一段合并
                if TERMINAL_RE.search(text) or pending["samples"] >= merge_max_s * SAMPLE_RATE:
                    pending_flush()
            except Exception as exc:
                log("[ambient 落盘异常]", exc)
                time.sleep(0.2)

    def dictation():
        s = recognizer.create_stream()
        last_partial = ""
        buf, buf_samples = [], 0  # 距上次断句的原始音频，供精修
        vad_carry = []            # 不足一个 VAD 窗口的余样
        seg_speech_windows = 0    # 当前段内检出人声的窗口数

        def vad_feed(samples):
            """喂 VAD 并累计本段人声窗口数；返回本 chunk 是否检出人声"""
            nonlocal vad_carry, seg_speech_windows
            if vad is None:
                return True
            vad_carry.append(samples)
            total = sum(len(x) for x in vad_carry)
            if total < VAD_WINDOW:
                return bool(seg_speech_windows)
            data = np.concatenate(vad_carry)
            n_win = len(data) // VAD_WINDOW
            speech = False
            for i in range(n_win):
                vad.accept_waveform(np.asarray(data[i * VAD_WINDOW:(i + 1) * VAD_WINDOW],
                                               dtype=np.float32))
                while not vad.empty():
                    vad.pop()
                if vad.is_speech_detected():
                    seg_speech_windows += 1
                    speech = True
            rest = data[n_win * VAD_WINDOW:]
            vad_carry = [rest] if len(rest) else []
            return speech

        def flush_inflight():
            """模式切换/退出时：正在录的半句截断，未完句一并落日志（只上 log 不上屏）"""
            nonlocal buf, buf_samples, last_partial, seg_speech_windows, vad_carry
            if last_partial:
                text = last_partial
                with model_lock:
                    if punct is not None:
                        try:
                            text = punct.add_punctuation(text)
                        except Exception:
                            pass
                voice_log(text, committed=False, speaker="截断")
                history.add("flush", text, committed=False, refined=False, speaker="截断")
                ui_q.put(("partial", ""))
            buf, buf_samples, last_partial = [], 0, ""
            seg_speech_windows = 0
            vad_carry = []
            # 让 finalize 工作线程把持有中的未完句落盘，等它确认（最多 3s）
            flush_done.clear()
            finalize_q.put("FLUSH")
            flush_done.wait(timeout=3.0)
            recognizer.reset(s)

        def ambient_finalize(partial):
            """记录模式断句：只截取音频和 VAD 统计，精修/合并/落盘交给 finalize
            工作线程——停顿处的重精修不再阻塞流式解码，新的输入照常识别"""
            nonlocal buf, buf_samples, seg_speech_windows, vad_carry
            block = np.concatenate(buf) if buf else None
            ratio = min(1.0, seg_speech_windows * VAD_WINDOW / max(buf_samples, 1))
            buf, buf_samples = [], 0
            seg_speech_windows = 0
            vad_carry = []
            recognizer.reset(s)
            if block is not None:
                finalize_q.put({"block": block, "partial": partial, "ratio": ratio})

        def commit(stream_text):
            """上屏模式断句：降噪+精修+同音替换+标点 → 声纹标注 → 上屏+log+历史"""
            nonlocal buf, buf_samples, last_partial
            block = np.concatenate(buf) if buf else np.zeros(SAMPLE_RATE // 10, dtype=np.float32)
            text, refined = refine_block(block)
            if not text:
                text = stream_text
            spk = registry.label(block)
            speaker = spk[0] if spk and spk[0] else "未知"
            if text:
                voice_log(text, committed=True, speaker=speaker)
                history.add("final", text, committed=True, refined=refined, speaker=speaker)
                ui_q.put(("final", text))
                send_text(text)
            # 上屏语音是否自动注册"我"（auto_enroll_user 开启时）
            if cfg.get("auto_enroll_user", False):
                registry.enroll_user(block)
            buf, buf_samples, last_partial = [], 0, ""
            recognizer.reset(s)

        while True:
            if state["flush"].is_set():
                state["flush"].clear()
                flush_inflight()
            try:
                chunk = audio_q.get(timeout=0.1)
            except queue.Empty:
                continue  # 静默期未完句的落盘由 finalize 工作线程按 merge_gap 处理
            try:
                samples = chunk.reshape(-1)
                vad_feed(samples)
                buf.append(samples)
                buf_samples += len(samples)
                s.accept_waveform(SAMPLE_RATE, samples)
                while recognizer.is_ready(s):
                    recognizer.decode_stream(s)
                partial = recognizer.get_result(s)
                if partial != last_partial:
                    last_partial = partial
                    if state["mode"] == "commit":
                        ui_q.put(("partial", partial))
                if recognizer.is_endpoint(s):
                    if state["mode"] == "commit":
                        commit(partial)
                    else:
                        ambient_finalize(partial)
            except Exception as exc:
                log("[识别/上屏异常]", exc)
                time.sleep(0.1)
                try:
                    recognizer.reset(s)
                except Exception:
                    s = recognizer.create_stream()

    threading.Thread(target=finalize_worker, daemon=True).start()
    threading.Thread(target=dictation, daemon=True).start()

    def hotkey_worker():
        hotkeys["tid"] = threading.get_ident()
        name = None
        for mod, vk, label in ((MOD_ALT, VK_R, "Alt+R"), (MOD_ALT, VK_T, "Alt+T"), (MOD_ALT, VK_W, "Alt+W")):
            if user32.RegisterHotKey(None, HOTKEY_TOGGLE, mod, vk):
                name = label
                break
            log("热键注册失败（被占用）:", label)
        if not name:
            log("所有备选热键均被占用，只能用托盘图标/接口切换模式")
        hotkeys["name"] = name
        if not user32.RegisterHotKey(None, HOTKEY_QUIT, MOD_CONTROL | MOD_ALT, VK_Q):
            log("热键注册失败（被占用）: Ctrl+Alt+Q，可用托盘右键退出")
        log("开关热键:", name or "无", "| 退出热键: Ctrl+Alt+Q")
        ui_q.put(("hotkey", name))
        msg = wt.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY:
                if msg.wParam == HOTKEY_TOGGLE:
                    ui_q.put(("mode", "toggle"))
                elif msg.wParam == HOTKEY_QUIT:
                    ui_q.put(("quit", None))
                    break

    threading.Thread(target=hotkey_worker, daemon=True).start()

    HistoryServer(history, state, hotkeys, ui_q, refiner, punct, cfg,
                  denoiser, registry, capture, vad_cfg=None if vad is None else vad.config).start()

    # 托盘图标：左键=唤出/收起上屏，右键=菜单
    def _tray_toggle(icon, item):
        ui_q.put(("mode", "toggle"))

    def _tray_history(icon, item):
        webbrowser.open_new(f"http://127.0.0.1:{HTTP_PORT}/")

    def _tray_quit(icon, item):
        ui_q.put(("quit", None))

    tray = pystray.Icon(
        "monikavoice", PILImage.open(ICON64), "monikavoice（记录中）",
        menu=pystray.Menu(
            pystray.MenuItem("唤出/收起上屏", _tray_toggle, default=True),
            pystray.MenuItem("历史记录（网页）", _tray_history),
            pystray.MenuItem("退出", _tray_quit),
        ),
    )
    tray.run_detached()

    win = FloatWindow(ui_q, state, audio_q)
    win.tray = tray

    threading.Thread(target=catchup_worker, args=(cfg, history), daemon=True).start()
    threading.Thread(target=daily_worker, args=(cfg, history), daemon=True).start()

    try:
        win.root.mainloop()
    finally:
        state["flush"].set()
        time.sleep(3.0)  # 给识别线程截断 + finalize 线程落盘未完句留时间
        tid = hotkeys.get("tid")
        if tid:
            user32.PostThreadMessageW(tid, WM_QUIT, 0, 0)
        try:
            tray.stop()
        except Exception:
            pass
        try:
            mic.stop()
            mic.close()
        except Exception:
            pass
    log("已退出")


if __name__ == "__main__":
    main()
