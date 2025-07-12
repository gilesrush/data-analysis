# -*- coding: utf-8 -*-
import os
import re
import jieba
from gensim.models import Word2Vec
from sudachipy import dictionary, tokenizer
from tqdm import tqdm  # ✅ 動的進捗表示用

# === 設定 ===
CORPUS_PATH = "corpus"
MODEL_PATH_CH = "model_chinese.model"
MODEL_PATH_JP = "model_japanese.model"
MIN_COUNT = 3
VECTOR_SIZE = 100
WINDOW = 5
EPOCHS = 10
SUDACHI_LIMIT = 49149  # ✅ SudachiPy の最大入力制限（バイト）

# === SudachiPy 初期化 ===
sudachi = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C

# === テキストクリーニング（中日漢字と仮名のみ残す）===
def clean_text(text):
    text = re.sub(r"[^\u4e00-\u9fa5\u3040-\u30ff\u3400-\u9fff]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# === 分かち書き関数（長文対応）===
def tokenize_text(text, lang):
    text = clean_text(text)
    if lang == "chinese":
        return list(jieba.cut(text))
    else:
        tokens = []
        text_bytes = text.encode("utf-8")
        start = 0
        while start < len(text_bytes):
            end = min(start + SUDACHI_LIMIT, len(text_bytes))
            # UTF-8の文字の途中で切らないように調整
            while end < len(text_bytes) and (text_bytes[end] & 0b11000000) == 0b10000000:
                end -= 1
            chunk = text_bytes[start:end].decode("utf-8", errors="ignore")
            for m in sudachi.tokenize(chunk, mode):
                word = m.dictionary_form()
                if word and word != " ":
                    tokens.append(word)
            start = end
        return tokens

# === ファイル名から言語判定 ===
def detect_language(filename):
    base = os.path.splitext(filename)[0]
    match = re.match(r"(\d+)([A-Za-z]*)", base)
    if not match:
        return "chinese"
    num, suffix = match.groups()
    return "japanese" if int(num) >= 2025 and not suffix else "chinese"

# === 文書処理とデータ準備 ===
sentences_ch = []
sentences_jp = []

print("📂 corpus フォルダのスキャンを開始...")

file_list = [f for f in os.listdir(CORPUS_PATH) if f.endswith(".txt")]

for fname in tqdm(file_list, desc="文書処理中", ncols=100):
    tqdm.write(f"📄 処理中：{fname}")
    path = os.path.join(CORPUS_PATH, fname)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    lang = detect_language(fname)
    tokens = tokenize_text(text, lang)
    if len(tokens) >= MIN_COUNT:
        if lang == "chinese":
            sentences_ch.append(tokens)
        else:
            sentences_jp.append(tokens)

print("\n📊 文献統計：")
print(f" - 中国語文献数：{len(sentences_ch)} 件")
print(f" - 日本語文献数：{len(sentences_jp)} 件")

# === Word2Vec モデル訓練 ===
if sentences_ch:
    print("🧠 中国語 Word2Vec モデルを訓練中...")
    model_ch = Word2Vec(
        sentences=sentences_ch,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=4,
        epochs=EPOCHS
    )
    model_ch.save(MODEL_PATH_CH)
    print(f"✅ 中国語モデルを保存しました：{MODEL_PATH_CH}")
else:
    print("⚠ 中国語文献が見つかりませんでした。")

if sentences_jp:
    print("🧠 日本語 Word2Vec モデルを訓練中...")
    model_jp = Word2Vec(
        sentences=sentences_jp,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=4,
        epochs=EPOCHS
    )
    model_jp.save(MODEL_PATH_JP)
    print(f"✅ 日本語モデルを保存しました：{MODEL_PATH_JP}")
else:
    print("⚠ 日本語文献が見つかりませんでした。")
