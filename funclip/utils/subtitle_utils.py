#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunClip). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)
import re

from .subtitle_segmentation import split_long_chinese_sentence
from funclip.subtitle_style import SubtitleFormattingError, format_subtitle_text

def time_convert(ms):
    ms = int(ms)
    tail = ms % 1000
    s = ms // 1000
    mi = s // 60
    s = s % 60
    h = mi // 60
    mi = mi % 60
    h = "00" if h == 0 else str(h)
    mi = "00" if mi == 0 else str(mi)
    s = "00" if s == 0 else str(s)
    tail = str(tail).zfill(3)
    if len(h) == 1: h = '0' + h
    if len(mi) == 1: mi = '0' + mi
    if len(s) == 1: s = '0' + s
    return "{}:{}:{},{}".format(h, mi, s, tail)

def str2list(text):
    pattern = re.compile(r'[\u4e00-\u9fff]|[\w-]+', re.UNICODE)
    elements = pattern.findall(text)
    return elements

class Text2SRT():
    def __init__(self, text, timestamp, offset=0, format_output=True):
        self.token_list = text
        self.timestamp = timestamp
        self.format_output = format_output
        start, end = timestamp[0][0] - offset, timestamp[-1][1] - offset
        self.start_sec, self.end_sec = start, end
        self.start_time = time_convert(start)
        self.end_time = time_convert(end)
    def text(self):
        if isinstance(self.token_list, str):
            res = self.token_list
        else:
            res = ""
            for word in self.token_list:
                if '\u4e00' <= word <= '\u9fff':
                    res += word
                else:
                    res += " " + word
            res = res.lstrip()
        return format_subtitle_text(res) if self.format_output else res
    def srt(self, acc_ost=0.0):
        return "{} --> {}\n{}\n".format(
            time_convert(self.start_sec+acc_ost*1000),
            time_convert(self.end_sec+acc_ost*1000), 
            self.text())
    def time(self, acc_ost=0.0):
        return (self.start_sec/1000+acc_ost, self.end_sec/1000+acc_ost)


def generate_srt(sentence_list, *, split_long_chinese=False, format_output=True):
    """Render cues without changing the original ASR sentences or timestamps."""
    sentences_to_render = sentence_list
    if split_long_chinese:
        sentences_to_render = (
            cue for sentence in sentence_list
            for cue in split_long_chinese_sentence(sentence)
        )
    srt_total = ''
    for i, sent in enumerate(sentences_to_render):
        t2s = Text2SRT(
            sent['text'], sent['timestamp'],
            format_output=format_output,
        )
        try:
            rendered = t2s.srt()
        except SubtitleFormattingError as exc:
            raise SubtitleFormattingError(f"Cue {i + 1} requires review: {exc}") from exc
        if 'spk' in sent:
            srt_total += "{}  spk{}\n{}".format(i + 1, sent['spk'], rendered)
        else:
            srt_total += "{}\n{}\n".format(i + 1, rendered)
    return srt_total

def generate_srt_clip(sentence_list, start, end, begin_index=0, time_acc_ost=0.0):
    start, end = int(start * 1000), int(end * 1000)
    srt_total = ''
    cc = 1 + begin_index
    subs = []
    for _, original_sentence in enumerate(sentence_list):
        # Clipping must not replace the original ASR text in recognition state.
        sent = dict(original_sentence)
        if sent['timestamp'][-1][1] <= start:
            # print("CASE0")
            continue
        if sent['timestamp'][0][0] >= end:
            # print("CASE4")
            break
        # parts in between
        if start <= sent['timestamp'][0][0] and sent['timestamp'][-1][1] <= end:
            # print("CASE1"); import pdb; pdb.set_trace()
            t2s = Text2SRT(sent['text'], sent['timestamp'], offset=start)
            srt_total += "{}\n{}".format(cc, t2s.srt(time_acc_ost))
            subs.append((t2s.time(time_acc_ost), t2s.text()))
            cc += 1
            continue
        if isinstance(sent['text'], str):
            format_subtitle_text(sent['text'])
            # Legacy partial clipping indexes punctuation-free tokens. Do not
            # silently lose decimals, negation, percentages or contractions.
            if re.search(r"['’‘ʼ%％‰]|[+−-]\s*\d|\d[.,，．:/]\d", sent['text']):
                raise SubtitleFormattingError(
                    "Partial subtitle clipping of numeric or apostrophized text requires review"
                )
            tokens = str2list(sent['text'])
            if len(tokens) != len(sent['timestamp']):
                raise SubtitleFormattingError("Partial subtitle clipping needs aligned token timestamps")
            sent['text'] = tokens
        if sent['timestamp'][0][0] <= start:
            # print("CASE2"); import pdb; pdb.set_trace()
            if not sent['timestamp'][-1][1] > end:
                for j, ts in enumerate(sent['timestamp']):
                    if ts[1] > start:
                        break
                _text = sent['text'][j:]
                _ts = sent['timestamp'][j:]
            else:
                for j, ts in enumerate(sent['timestamp']):
                    if ts[1] > start:
                        _start = j
                        break
                for j, ts in enumerate(sent['timestamp']):
                    if ts[1] > end:
                        _end = j
                        break
                # _text = " ".join(sent['text'][_start:_end])
                _text = sent['text'][_start:_end]
                _ts = sent['timestamp'][_start:_end]
            if len(ts):
                t2s = Text2SRT(_text, _ts, offset=start)
                srt_total += "{}\n{}".format(cc, t2s.srt(time_acc_ost))
                subs.append((t2s.time(time_acc_ost), t2s.text()))
                cc += 1
            continue
        if sent['timestamp'][-1][1] > end:
            # print("CASE3"); import pdb; pdb.set_trace()
            for j, ts in enumerate(sent['timestamp']):
                if ts[1] > end:
                    break
            _text = sent['text'][:j]
            _ts = sent['timestamp'][:j]
            if len(_ts):
                t2s = Text2SRT(_text, _ts, offset=start)
                srt_total += "{}\n{}".format(cc, t2s.srt(time_acc_ost))
                subs.append(
                    (t2s.time(time_acc_ost), t2s.text())
                    )
                cc += 1
            continue
    return srt_total, subs, cc
