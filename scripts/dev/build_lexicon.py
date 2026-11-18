# -*- coding: utf-8 -*-
"""扩充 `conditioning/g2p/lexicon.json`：高频多音字词组 → 整词音素 id 串。

输出格式（与 `GptSoVitsOnnxEngine.loadG2p()` 的读取方式一致）：

    {"table": {"银行": [11, 250, 85, 190], "觉得": [...]}}

引擎匹配规则：最长 4 字优先；每字切几个音素由 `char2phone.json` 决定，要求
「逐字切分数之和 == 整词 id 数」，否则该词条被忽略。这里生成时就按同一口径
校验，保证写进去的每一条都能被引擎用上。
"""

import json
import sys
from pathlib import Path

from pypinyin import Style, pinyin
from pypinyin.phrases_dict import phrases_dict

PUBLISH = Path(r"D:\爱弥斯\_work\publish")
G2P = PUBLISH / "conditioning" / "g2p"

INITIALS = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l",
            "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w"]

REQUIRED = [
    "觉得", "睡着", "看着", "笑着", "走着", "音乐", "快乐", "长大", "长短", "了解",
    "只要", "只有", "还要", "归还", "发现", "头发", "一行", "行动", "银行", "重要",
    "重新", "因为", "为了", "都行", "首都", "好客", "好奇", "背包", "后背",
    "漂泊者", "爱弥斯", "什么", "行走", "行李", "着急", "还是", "得到", "的确",
    "地方", "地壳", "子弹", "弹琴", "干净", "干练", "差不多", "出差", "恶心", "厌恶",
    "禁止", "情不自禁", "便宜", "方便", "处理", "到处", "答应", "回答", "积累", "劳累",
    "参加", "人参", "抢救", "强迫", "重量", "重复", "睡觉", "自觉", "角色", "角度",
    "露面", "暴露", "结实", "结束", "模糊", "模样", "宁愿", "宁静", "炮火", "炮制",
    "银行", "行业", "行长", "弯曲", "歌曲", "调节", "调查", "空调", "空气",
    "种子", "种地", "难过", "灾难", "为难", "稍微", "单薄", "薄荷", "薄弱",
    "朝气", "朝霞", "传单", "传记", "呼应", "答应", "血型", "流血", "系统", "系鞋带",
    "降落", "投降", "数落", "数字", "小说", "说服", "游说", "弹力", "弹幕",
]


def split_syllable(syl: str):
    for ini in INITIALS:
        if syl.startswith(ini) and len(syl) > len(ini):
            return ini, syl[len(ini):]
    return "", syl


def normalize_final(ini: str, fin: str) -> str:
    """GPT-SoVITS 的符号表用 `v` 表示 ü：jue2 -> j + ve2，qu4 -> q + v4，yu2 -> y + v2。"""
    fin = fin.replace("ü", "v")
    if ini in ("j", "q", "x", "y") and fin.startswith("u"):
        fin = "v" + fin[1:]
    return fin


def build_polyphone_set(char_table: dict) -> set:
    """对字表里每个字取 heteronym 读音，>1 种即视为多音字（用于筛选词组）。"""
    poly = set()
    for ch in char_table:
        try:
            readings = pinyin(ch, style=Style.TONE3, heteronym=True, neutral_tone_with_five=True)
        except Exception:
            continue
        if readings and len({r for r in readings[0]}) > 1:
            poly.add(ch)
    return poly


def build_entry(word: str, index: dict, char_table: dict):
    syls = pinyin(word, style=Style.TONE3, neutral_tone_with_five=True, errors="ignore")
    if len(syls) != len(word):
        return None
    ids, counts = [], []
    for ch, (syl,) in zip(word, syls):
        ini, fin = split_syllable(syl)
        fin = normalize_final(ini, fin)
        piece = []
        if ini:
            if ini not in index:
                return None
            piece.append(index[ini])
        else:
            # 零声母字（爱/安/恶…）在 GPT-SoVITS 里用占位声母 `AA`，与 char2phone 口径一致
            if "AA" not in index:
                return None
            piece.append(index["AA"])
        if fin not in index:
            return None
        piece.append(index[fin])
        default = char_table.get(ch) or []
        if len(default) != len(piece):  # 与引擎的切分口径对齐
            return None
        ids.extend(piece)
        counts.append(len(piece))
    if not ids or sum(counts) != len(ids):
        return None
    return ids


def main() -> int:
    symbols = json.loads((G2P / "symbols.json").read_text(encoding="utf-8"))
    index = {s: i for i, s in enumerate(symbols)}
    raw = json.loads((G2P / "char2phone.json").read_text(encoding="utf-8"))
    char_table = raw.get("table", raw)

    poly = build_polyphone_set(char_table)
    print(f"[poly] 多音字 {len(poly)} 个")

    candidates = dict.fromkeys(REQUIRED, True)
    for word in phrases_dict:
        if not (2 <= len(word) <= 4):
            continue
        if not all(ch in char_table for ch in word):
            continue
        if not any(ch in poly for ch in word):
            continue
        candidates.setdefault(word, True)
    print(f"[candidates] {len(candidates)} 个词组")

    table, skipped = {}, 0
    for word in candidates:
        ids = build_entry(word, index, char_table)
        if ids is None:
            skipped += 1
            continue
        table[word] = ids

    # 体型控制：保留全部 REQUIRED，再按「2 字 > 3 字 > 4 字」补常见词，总量 800 条左右。
    MAX_WORDS = 800
    if len(table) > MAX_WORDS:
        required = [w for w in REQUIRED if w in table]
        rest = [w for w in table if w not in set(required)]
        rest.sort(key=lambda w: (len(w), w))
        keep = required + rest[: max(0, MAX_WORDS - len(required))]
        table = {w: table[w] for w in keep}

    out = G2P / "lexicon.json"
    out.write_text(
        json.dumps({"table": table}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"[lexicon] {len(table)} 词 / 跳过 {skipped} / {out.stat().st_size} B")
    for probe in ("觉得", "睡着", "音乐", "银行", "漂泊者", "长发", "重要"):
        print("  ", probe, table.get(probe))
    return 0


if __name__ == "__main__":
    sys.exit(main())
