# -*- coding: utf-8 -*-
"""
monikavoice —— Windows 离线中文语音听写（本地运行，无广告，不联网）

- 常态录音：程序运行即持续监听麦克风，所有说到的话按时间写入 logs/voice-日期.log
  与 history.jsonl（记录模式，不上屏）
- 唤出上屏：Alt+R/Alt+T（被占用自动回退）切换到上屏模式，切换瞬间会把正在录的
  半句截断、只上 log 不上屏；之后断句的内容经"离线 Paraformer 精修 + 同音替换
  纠错 + ct-transformer 标点"打进当前焦点输入框；再次按热键收回记录模式
- 实时浮窗：仅上屏模式弹出，流式跟随识别内容，不抢目标应用焦点
- 历史与 API：http://127.0.0.1:8397/ 网页查看；/api/history /api/status
  /api/mode /api/transcribe（预留宿曜整理上课录音）等接口供 AI/程序调用
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
import sys
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
MODEL_DIR = os.path.join(BASE, "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20")
PUNCT_DIR = os.path.join(BASE, "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12")
PARAFORMER_DIR = os.path.join(BASE, "sherpa-onnx-paraformer-zh-2023-09-14")
HR_DIR = os.path.join(BASE, "hr")
LOGS_DIR = os.path.join(BASE, "logs")
SUMMARIES_DIR = os.path.join(BASE, "summaries")
LOG_PATH = os.path.join(BASE, "monikavoice.log")
HISTORY_PATH = os.path.join(BASE, "history.jsonl")
CONFIG_PATH = os.path.join(BASE, "config.json")
ICON64 = os.path.join(BASE, "icon64.png")
SPEAKER_MODEL = os.path.join(BASE, "campplus.onnx")
DENOISER_MODEL = os.path.join(BASE, "gtcrn_simple.onnx")
SPEAKERS_JSON = os.path.join(BASE, "speakers.json")
HTTP_PORT = 8397
HISTORY_MAX = 2000
RETAIN_DAYS = 183  # 六个月

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
    "speaker_threshold": 0.55,
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
    每行 [时间] [说话人] 标记 文本"""
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        name = os.path.basename("voice-" + time.strftime("%Y-%m-%d") + ".log")
        tag = "[" + speaker + "] " if speaker else ""
        with open(os.path.join(LOGS_DIR, name), "a", encoding="utf-8") as f:
            f.write(time.strftime("[%H:%M:%S] ") + tag + ("上屏 " if committed else "记录 ") + text + "\n")
    except OSError as exc:
        log("语音日志写入失败:", exc)


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

def build_recognizer():
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
        rule2_min_trailing_silence=0.4,
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
        匹配不上的一律"未知"——环境碎音不会自动创建新聚类"""
        if not self.enabled:
            return "", 0.0
        emb = self._embed(samples)
        if emb is None:
            return "未知", 0.0
        with self.lock:
            if self.manager is not None:
                name = self.manager.search(emb, self.threshold)
                if name:
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
    """转写文件路径校验：真实路径、必须是 .wav、禁止系统目录"""
    if not path or not os.path.isabs(path):
        raise ValueError("需要绝对路径")
    p = os.path.realpath(os.path.abspath(path))
    if not p.lower().endswith(".wav"):
        raise ValueError("仅支持 .wav 文件")
    if not os.path.isfile(p):
        raise ValueError("文件不存在")
    blocked_roots = [os.environ.get("WINDIR", r"C:\Windows"),
                     os.environ.get("ProgramFiles", r"C:\Program Files"),
                     os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")]
    for root in blocked_roots:
        if root and p.lower().startswith(os.path.realpath(root).lower() + os.sep):
            raise ValueError("不允许访问系统目录")
    return p


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


def transcribe_file(path, refiner, punct, denoiser=None, registry=None, log_it=True):
    """转写音频文件：降噪 → 30 秒分段 → Paraformer+同音替换+标点，
    每段带声纹来源标注。返回 (拼接文本, 分段列表)"""
    if refiner is None:
        raise RuntimeError("离线精修模型未加载")
    data = _read_wav_mono16k(path)
    step = SAMPLE_RATE * 30
    texts, segs = [], []
    for i in range(0, max(1, len(data)), step):
        seg = data[i:i + step]
        if len(seg) < SAMPLE_RATE // 10:
            break
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


# ---------------- 每日任务：DeepSeek 总结 + 六个月清理 ----------------

LINE_RE = re.compile(r"\[(\d\d:\d\d:\d\d)\] (.*)$")


def voice_log_path(date):
    """某天的语音日志文件名（basename 消化任何路径成分）"""
    return os.path.join(LOGS_DIR, os.path.basename("voice-" + date.strftime("%Y-%m-%d") + ".log"))


def collect_window(start_dt, end_dt):
    """收集 [start_dt, end_dt] 内的语音日志行"""
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
                            lines.append("[" + d.strftime("%m-%d ") + m.group(1) + "] " + m.group(2))
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


def deepseek_summarize(cfg, range_label, transcript):
    body = json.dumps({
        "model": cfg.get("deepseek_model", "deepseek-chat"),
        "messages": [
            {"role": "system", "content":
                "你是个人语音日志整理助手。输入是用户通过语音听写记录的原始内容，"
                "可能混有环境杂音误识别的碎片。请：1) 忽略无意义碎片、重复和杂音；"
                "2) 把有意义的内容按主题分组，每组给出小标题和要点；"
                "3) 明显的待办事项单独列出；4) 简体中文，600 字以内。"},
            {"role": "user", "content": "时间范围：" + range_label + "\n语音内容：\n" + transcript},
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


def summarize_for_date(date_str, cfg, store):
    """总结 date_str 当天 summary_hour+1 分到次日 summary_hour 点的内容"""
    day = valid_date(date_str)
    if day is None:
        raise ValueError("非法日期: " + str(date_str))
    hour = int(cfg.get("summary_hour", 2))
    start_dt = datetime(day.year, day.month, day.day, hour) + timedelta(minutes=1)
    end_dt = start_dt + timedelta(days=1) - timedelta(minutes=1)
    lines = collect_window(start_dt, end_dt)
    os.makedirs(SUMMARIES_DIR, exist_ok=True)
    out_path = os.path.join(SUMMARIES_DIR, os.path.basename(date_str + ".md"))
    if not lines:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# " + date_str + " 语音日志总结\n\n（该时段无记录）\n")
        return out_path
    transcript = "\n".join(lines)
    if len(transcript) > 24000:
        transcript = "…（更早内容已截断）\n" + transcript[-24000:]
    range_label = start_dt.strftime("%Y-%m-%d %H:%M") + " 至 " + end_dt.strftime("%Y-%m-%d %H:%M")
    try:
        text = deepseek_summarize(cfg, range_label, transcript)
    except Exception as exc:
        log("[每日总结] DeepSeek 调用失败:", exc)
        text = "> 总结失败：" + str(exc) + "\n\n原始条数：" + str(len(lines))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# " + date_str + " 语音日志总结\n\n范围：" + range_label +
                " · 共 " + str(len(lines)) + " 条\n\n" + text + "\n")
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
  <input id="en-path" placeholder="该人说话的 wav 绝对路径" style="width:340px;background:#141416;color:#e8eaed;border:1px solid #3c4043;border-radius:4px;padding:3px 6px">
  <button onclick="svPost('/api/enroll','en',this)">录入</button>
  <span id="en-out" class="meta"></span>
 </div>
 <div style="margin-top:6px;font-size:13px">
  判断 <input id="who-path" placeholder="wav 绝对路径" style="width:440px;background:#141416;color:#e8eaed;border:1px solid #3c4043;border-radius:4px;padding:3px 6px">
  <button onclick="svPost('/api/who','who',this)">判断</button>
  <span id="who-out" class="meta"></span>
 </div>
 <div id="spk-list" class="meta" style="margin-top:8px"></div>
</div>
<div class="api">
 API：<code>GET /api/history?limit=100&amp;since_id=0</code> ·
 <code>GET /api/status</code> · <code>GET /api/summary?date=YYYY-MM-DD</code> ·
 <code>POST /api/mode /api/clear /api/summarize /api/transcribe</code>
 （仅本机 127.0.0.1 可访问；配置 api_token 后 POST 需带 X-Token 头）
</div>
<script>
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
        `#${e.id} · ${e.ts} · ${e.type}${e.refined?' · 已精修':''}</div>` +
        `<div class="text"></div></div>`).join('');
      list.querySelectorAll('.text').forEach((el,i)=>{ el.textContent = d.items[i].text; });
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
                 denoiser=None, registry=None):
        self.store = store
        self.state = state
        self.hotkeys = hotkeys
        self.ui_q = ui_q
        self.refiner = refiner
        self.punct = punct
        self.cfg = cfg
        self.denoiser = denoiser
        self.registry = registry
        self.token = (cfg.get("api_token") or "").strip()

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
                                     "speaker": bool(registry and registry.enabled)})
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
                            log_it=bool(payload.get("log", True)))
                        store.add("transcribe", text, committed=False,
                                  refined=True, speaker="多段" if len(segs) > 1 else (segs[0]["speaker"] if segs else "未知"))
                        self._json(200, {"ok": True, "text": text,
                                         "chars": len(text), "segments": segs})
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
    recognizer = build_recognizer()
    refiner = load_offline_refiner()
    punct = load_punctuation()
    denoiser = load_denoiser() if cfg.get("denoise_enable", True) else None
    if denoiser is None:
        log("人声增强未启用")
    registry = SpeakerRegistry(cfg)

    audio_q = queue.Queue()
    ui_q = queue.Queue()
    state = {"mode": "ambient", "flush": threading.Event()}  # 常态=记录模式
    hotkeys = {}
    history = HistoryStore(HISTORY_PATH)

    def mic_callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    mic = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                         blocksize=SAMPLE_RATE // 10, callback=mic_callback)
    mic.start()

    def dictation():
        s = recognizer.create_stream()
        last_partial = ""
        buf, buf_samples = [], 0  # 距上次断句的原始音频，供精修

        def refine_block(block):
            """降噪 → 离线精修 → 标点，返回 (文本, 是否精修)"""
            text, refined = "", False
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

        def flush_inflight():
            """模式切换/退出时：正在录的半句截断，加标点后只上 log 不上屏"""
            nonlocal buf, buf_samples, last_partial
            if last_partial:
                text = last_partial
                if punct is not None:
                    try:
                        text = punct.add_punctuation(text)
                    except Exception:
                        pass
                voice_log(text, committed=False, speaker="截断")
                history.add("flush", text, committed=False, refined=False, speaker="截断")
                ui_q.put(("partial", ""))
            buf, buf_samples, last_partial = [], 0, ""
            recognizer.reset(s)

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
            # 上屏模式默认是用户在打字，采集声纹自动注册/强化"我"
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
                continue
            try:
                samples = chunk.reshape(-1)
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
                        # 记录模式：降噪+精修+标点+声纹标注，只进日志不上屏
                        block = np.concatenate(buf) if buf else None
                        if block is not None or partial:
                            text, refined = ("", False)
                            if block is not None:
                                text, refined = refine_block(block)
                            if not text:
                                text = partial
                            spk = registry.label(block) if block is not None else ("未知", 0)
                            speaker = spk[0] if spk and spk[0] else "未知"
                            if text:
                                voice_log(text, committed=False, speaker=speaker)
                                history.add("ambient", text, committed=False,
                                            refined=refined, speaker=speaker)
                        buf, buf_samples, last_partial = [], 0, ""
                        recognizer.reset(s)
            except Exception as exc:
                log("[识别/上屏异常]", exc)
                time.sleep(0.1)
                try:
                    recognizer.reset(s)
                except Exception:
                    s = recognizer.create_stream()

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
                  denoiser, registry).start()

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
        time.sleep(1.0)  # 给识别线程留出截断上 log 的时间
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
