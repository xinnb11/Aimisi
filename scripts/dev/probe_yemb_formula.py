"""探针：判断 t2s_sdec 的 y_emb 末行 = audio_embedding[token] 还是 = 后者 + 位置编码。

用法：
  python scripts/dev/probe_yemb_formula.py --model-dir D:\爱弥斯\_work\publish

输出：
  - 末行与 audio_embedding[token] 的余弦 / 最大绝对差
  - 差分向量与 sin/cos 基频的相关性（粗判是否叠了位置编码）
"""
import argparse
import json
import os

import numpy as np
import onnxruntime as ort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    args = ap.parse_args()
    root = args.model_dir
    cond = os.path.join(root, 'conditioning')

    manifest = json.load(open(os.path.join(cond, 'conditioning.json'), 'r', encoding='utf-8'))
    emo = manifest['emotions']['平静']

    def load_blob(entry, key, dtype=np.float32):
        info = entry[key]
        p = os.path.join(cond, '平静', info['file'])
        return np.fromfile(p, dtype=dtype).reshape(info['shape'])

    ref_seq = load_blob(emo, 'ref_seq', np.int64)
    ref_bert = load_blob(emo, 'ref_bert')
    ssl = load_blob(emo, 'ssl_content')

    char2phone = json.load(open(os.path.join(cond, 'g2p', 'char2phone.json'), 'r', encoding='utf-8'))
    if 'table' in char2phone:
        char2phone = char2phone['table']
    text = '今天的任务就做到这里吧'
    ids, counts, chars = [], [], []
    for ch in text:
        ph = char2phone.get(ch) or char2phone.get('*')
        if not ph:
            continue
        ids.extend(ph)
        counts.append(len(ph))
        chars.append(ch)
    text_seq = np.array([ids], dtype=np.int64)

    # text_bert：用 BERT 模型算第 21 层，按音素数 repeat
    bert_path = os.path.join(root, 'bert', 'model.int8.onnx')
    vocab_path = os.path.join(root, 'bert', 'vocab.txt')
    vocab = {w: i for i, w in enumerate(open(vocab_path, 'r', encoding='utf-8').read().split('\n'))}
    seq = [101]
    for ch in chars:
        seq.extend(vocab.get(c, 100) for c in ch)
    seq.append(102)
    ids_arr = np.array([seq], dtype=np.int64)
    bert_sess = ort.InferenceSession(bert_path, providers=['CPUExecutionProvider'])
    out = bert_sess.run(None, {
        'input_ids': ids_arr,
        'attention_mask': np.ones_like(ids_arr),
        'token_type_ids': np.zeros_like(ids_arr),
    })[0][0][1:-1]
    bert_feat = np.repeat(out, counts, axis=0)
    text_bert = bert_feat

    enc = ort.InferenceSession(os.path.join(root, 't2s_encoder.onnx'), providers=['CPUExecutionProvider'])
    x, prompts = enc.run(None, {
        'ref_seq': ref_seq,
        'text_seq': text_seq,
        'ref_bert': ref_bert,
        'text_bert': text_bert.astype(np.float32),
        'ssl_content': ssl,
    })
    fsdec = ort.InferenceSession(os.path.join(root, 't2s_fsdec.onnx'), providers=['CPUExecutionProvider'])
    y, k, v, y_emb, x_ex = fsdec.run(None, {'x': x, 'prompts': prompts})
    print('encoder x', x.shape, 'prompts', prompts.shape, 'fsdec y', y.shape, 'y_emb', y_emb.shape, 'k', k.shape)

    sdec = ort.InferenceSession(os.path.join(root, 't2s_sdec.onnx'), providers=['CPUExecutionProvider'])
    outs = sdec.run(None, {
        'iy': y, 'ik': k, 'iv': v, 'iy_emb': y_emb, 'ix_example': x_ex
    })
    names = [o.name for o in sdec.get_outputs()]
    print('sdec outputs:', names)
    pack = dict(zip(names, outs))
    logits = pack['logits']
    y_emb2 = pack['y_emb']
    y2 = pack['y']
    token = int(np.argmax(logits[0]))
    print('logits', logits.shape, 'sampled(argmax)', token, 'y_out', y2.shape, 'y_emb_out', y_emb2.shape)

    table = np.fromfile(os.path.join(cond, 'audio_embedding.f32'), dtype=np.float32).reshape(1025, 512)
    emb = table[token]

    last = y_emb2[0][-1]
    diff = last - emb
    cos = float(np.dot(last, emb) / (np.linalg.norm(last) * np.linalg.norm(emb) + 1e-9))
    print('cos(y_emb_last, embedding[token]) =', round(cos, 5))
    print('|diff| =', round(float(np.linalg.norm(diff)), 5), ' max|diff| =', round(float(np.abs(diff).max()), 5))
    print('y_emb_last[:6] =', np.round(last[:6], 4).tolist())
    print('embedding[:6]  =', np.round(emb[:6], 4).tolist())
    print('diff[:6]       =', np.round(diff[:6], 4).tolist())

    # 粗判位置编码：diff 与前 64 维的 sin/cos 基频相关性
    pos = y_emb2.shape[1] - 1
    idx = np.arange(0, 64)
    freq = 1.0 / (10000 ** (idx / 512.0))
    sin_v = np.sin(pos * freq)
    cos_v = np.cos(pos * freq)
    d = diff[:64]
    print('corr(diff, sin) =', round(float(abs(np.corrcoef(d, sin_v)[0, 1])), 4),
          'corr(diff, cos) =', round(float(abs(np.corrcoef(d, cos_v)[0, 1])), 4))


if __name__ == '__main__':
    main()
