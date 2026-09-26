# -*- coding: utf-8 -*-
"""
monikavoice —— Windows 离线中文语音听写（本地运行，无广告，不联网）

- Alt+R 开关听写（若被 NVIDIA 等程序占用则自动退回 Alt+T / Alt+W，浮窗会显示当前热键）
- Ctrl+Alt+Q 退出（被占用时可用托盘右键菜单退出）
- 听写时弹出置顶小浮窗，实时显示识别内容；停止后浮窗隐藏
- 双模型管线：流式 zipformer 实时跟随显示；断句后由离线 Paraformer +
  同音替换器（jieba）整句精修纠正同音错别字，再经 ct-transformer 加标点，
  最后以 Unicode 按键打进当前焦点输入框
- 托盘图标常驻：左键开关听写，右键菜单
"""
import ctypes
import ctypes.wintypes as wt
import glob
import os
import queue
import sys
import threading
import time
import tkinter as tk

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
LOG_PATH = os.path.join(BASE, "monikavoice.log")
ICON64 = os.path.join(BASE, "icon64.png")


def log(*args):
    line = " ".join(str(a) for a in args)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(time.strftime("[%m-%d %H:%M:%S] ") + line + "\n")
    except OSError:
        pass


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


class FloatWindow:
    """听写时弹出的置顶小浮窗：实时显示识别内容，不抢目标应用焦点"""

    def __init__(self, ui_q, state, audio_q):
        self.q = ui_q
        self.state = state
        self.audio_q = audio_q
        self.hotkey_name = "…"
        self.last_partial = ""

        self.root = tk.Tk()
        self.root.title("语音听写")
        self.root.configure(bg="#1f1f23")
        self.root.attributes("-topmost", True)
        self.root.withdraw()  # 初始隐藏，听写开启时才弹出

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
                if cmd == "toggle":
                    self.on_toggle()
                elif cmd == "partial":
                    self.show_partial(payload)
                elif cmd == "final":
                    self.show_final(payload)
                elif cmd == "hotkey":
                    self.hotkey_name = payload or "无"
                    self.refresh_header()
                elif cmd == "quit":
                    self.root.destroy()
                    return
        except queue.Empty:
            pass
        self.root.after(50, self.poll)

    def on_toggle(self):
        self.state["running"] = not self.state["running"]
        if self.state["running"]:
            with self.audio_q.mutex:
                self.audio_q.queue.clear()
            self.final_lbl.config(text="")
            self.partial_lbl.config(text="（请说话…）")
            self.refresh_header()
            hwnd = user32.GetForegroundWindow()
            self.root.deiconify()
            self.root.lift()
            # 弹窗不抢目标应用的键盘焦点
            self.root.after(120, lambda: user32.SetForegroundWindow(hwnd))
            log("[听写 开]")
        else:
            self.root.withdraw()
            log("[听写 关]")

    def refresh_header(self):
        if self.state["running"]:
            self.header.config(text=f"● 听写中 · {self.hotkey_name} 停止", fg="#81c995")
        else:
            self.header.config(text="○ 已暂停", fg="#9aa0a6")

    def show_partial(self, text):
        if text != self.last_partial:
            self.last_partial = text
            self.partial_lbl.config(text=text if text else "（请说话…）")

    def show_final(self, text):
        tail = text if len(text) <= 26 else "…" + text[-26:]
        self.final_lbl.config(text="已上屏: " + tail)
        self.last_partial = ""
        self.partial_lbl.config(text="（说话中…）")


def main():
    log("=== 启动 v3 ===")
    recognizer = build_recognizer()
    refiner = load_offline_refiner()
    punct = load_punctuation()

    audio_q = queue.Queue()
    ui_q = queue.Queue()
    state = {"running": False}
    hotkeys = {}

    def mic_callback(indata, frames, time_info, status):
        if state["running"]:
            audio_q.put(indata.copy())

    mic = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                         blocksize=SAMPLE_RATE // 10, callback=mic_callback)
    mic.start()

    def dictation():
        s = recognizer.create_stream()
        last_partial = ""
        buf, buf_samples = [], 0  # 距上次断句的原始音频，供离线精修

        def commit(stream_text):
            nonlocal buf, buf_samples, last_partial
            text = stream_text
            if refiner is not None and buf_samples > SAMPLE_RATE // 2:
                try:
                    t0 = time.time()
                    rs = refiner.create_stream()
                    rs.accept_waveform(SAMPLE_RATE, np.concatenate(buf))
                    refiner.decode_stream(rs)
                    refined = rs.result.text.strip()
                    if refined:
                        text = refined
                        log("[精修]", round((time.time() - t0) * 1000), "ms:", refined)
                except Exception as exc:
                    log("[精修异常，用流式结果]", exc)
            if text and punct is not None:
                try:
                    text = punct.add_punctuation(text)
                except Exception as exc:
                    log("[标点异常]", exc)
            if text:
                ui_q.put(("final", text))
                send_text(text)
            buf, buf_samples, last_partial = [], 0, ""
            recognizer.reset(s)

        while True:
            try:
                chunk = audio_q.get(timeout=0.1)
            except queue.Empty:
                if not state["running"]:
                    time.sleep(0.02)
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
                    ui_q.put(("partial", partial))
                if recognizer.is_endpoint(s):
                    commit(partial)
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
            log("所有备选热键均被占用，只能用托盘图标开关听写")
        hotkeys["name"] = name
        if not user32.RegisterHotKey(None, HOTKEY_QUIT, MOD_CONTROL | MOD_ALT, VK_Q):
            log("热键注册失败（被占用）: Ctrl+Alt+Q，可用托盘右键退出")
        log("开关热键:", name or "无", "| 退出热键: Ctrl+Alt+Q")
        ui_q.put(("hotkey", name))
        msg = wt.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY:
                if msg.wParam == HOTKEY_TOGGLE:
                    ui_q.put(("toggle", None))
                elif msg.wParam == HOTKEY_QUIT:
                    ui_q.put(("quit", None))
                    break

    threading.Thread(target=hotkey_worker, daemon=True).start()

    # 托盘小图标：程序上线的常驻标志；左键=开关听写，右键=菜单
    def _tray_toggle(icon, item):
        ui_q.put(("toggle", None))

    def _tray_quit(icon, item):
        ui_q.put(("quit", None))

    tray = pystray.Icon(
        "monikavoice", PILImage.open(ICON64), "monikavoice 语音听写（运行中）",
        menu=pystray.Menu(
            pystray.MenuItem("开始/停止听写", _tray_toggle, default=True),
            pystray.MenuItem("退出", _tray_quit),
        ),
    )
    tray.run_detached()

    win = FloatWindow(ui_q, state, audio_q)
    try:
        win.root.mainloop()
    finally:
        state["running"] = False
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
