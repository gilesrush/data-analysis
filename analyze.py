# -*- coding: utf-8 -*-
import os
import re
import jieba
import pandas as pd
from sudachipy import tokenizer
from sudachipy import dictionary
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 設定 ---
CORPUS_PATH = "corpus"
SKIP_PHRASE_FILE = "skip_phrases.txt"
STOPWORDS_PATH = "stopwords.txt"
USER_DICT_PATH = "user_dict.txt"
NGRAM_RANGE = (1, 2)
MIN_DF = 2
SUDACHI_LIMIT = 49149

# --- 定型句読み込み ---
def load_skip_phrases(filepath):
    """指定されたパスから定型句を読み込んでセットとして返す。"""
    if not os.path.exists(filepath):
        return set()
    with open(filepath, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

SKIP_PHRASES = load_skip_phrases(SKIP_PHRASE_FILE)

def remove_skip_phrases(text):
    """テキストから定型句を削除する。"""
    for phrase in SKIP_PHRASES:
        text = text.replace(phrase, "")
    return text

# --- ストップワード読み込み ---
def load_stopwords(filepath):
    """指定されたパスからストップワードを読み込んでセットとして返す。"""
    if not os.path.exists(filepath):
        print(f"警告: ストップワードファイル '{filepath}' が見つかりません。")
        return set()
    with open(filepath, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f)

stopwords = load_stopwords(STOPWORDS_PATH)

# --- SudachiPy 初期化 ---
tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C

# --- Jieba ユーザー辞書読み込み ---
try:
    jieba.load_userdict(USER_DICT_PATH)
except FileNotFoundError:
    print(f"警告: ユーザー辞書 '{USER_DICT_PATH}' が見つかりません。")

# --- 言語判定 ---
def detect_language(filename):
    """ファイル名に基づいて言語を判定する（'chinese' または 'japanese'）。"""
    base = filename.split('.')[0]
    match = re.match(r'^(\d+)', base)
    # ファイル名が数字で始まり、その数字が2000以上なら日本語と判定
    if match:
        year = int(match.group(1))
        return 'japanese' if year >= 2000 else 'chinese'
    # それ以外は中国語と判定
    return 'chinese'

# --- SudachiPy 分割トークナイズ ---
def tokenize_japanese_safely(text):
    """SudachiPyの文字数制限を回避するため、テキストを分割してトークナイズする。"""
    byte_text = text.encode("utf-8")
    if len(byte_text) <= SUDACHI_LIMIT:
        return [m.surface() for m in tokenizer_obj.tokenize(text, mode) if m.surface() not in stopwords]

    tokens = []
    print("⚠️ SudachiPyの文字数制限を超過したため、テキストを分割して処理しています...")
    start = 0
    while start < len(byte_text):
        end = min(start + SUDACHI_LIMIT - 1000, len(byte_text))
        chunk = byte_text[start:end]
        # UTF-8の文字境界で分割を確実にする
        while True:
            try:
                chunk_text = chunk.decode("utf-8")
                break
            except UnicodeDecodeError:
                end -= 1
                chunk = byte_text[start:end]

        tokens.extend(m.surface() for m in tokenizer_obj.tokenize(chunk_text, mode) if m.surface() not in stopwords)
        start = end
    return tokens

# --- 前処理・分詞 ---
def preprocess_and_tokenize(text, language):
    """テキストの前処理（記号削除など）と分詞処理を行う。"""
    text = re.sub(r"[0-9a-zA-Zａ-ｚＡ-Ｚ。、．・「」（）『』【】［］〈〉《》“”‘’！？，、：；（）…\s]", "", text)
    text = remove_skip_phrases(text)
    if language == 'chinese':
        return [w for w in jieba.cut(text) if w and w not in stopwords]
    elif language == 'japanese':
        return tokenize_japanese_safely(text)
    return []

# --- メイン処理 ---
def main():
    """
    コーパス内の文書を処理し、TF-IDFに基づいたコサイン類似度行列を計算してCSVファイルとして保存する。
    """
    print("🧘 文書の処理を開始します...")
    documents = []
    raw_docs = []
    filenames = []

    try:
        corpus_files = sorted([f for f in os.listdir(CORPUS_PATH) if f.endswith(".txt")])
        if not corpus_files:
            print(f"🛑 エラー: '{CORPUS_PATH}' ディレクトリに.txtファイルが見つかりません。")
            return
    except FileNotFoundError:
        print(f"🛑 エラー: コーパスディレクトリ '{CORPUS_PATH}' が見つかりません。")
        return

    for filename in corpus_files:
        path = os.path.join(CORPUS_PATH, filename)
        with open(path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        lang = detect_language(filename)
        tokens = preprocess_and_tokenize(raw_text, lang)

        if len(tokens) < 5:
            print(f"⏩ {filename} は内容が少ないためスキップしました。")
            continue

        # 分詞結果をスペースで連結して文書リストに追加
        raw_docs.append(" ".join(tokens))
        filenames.append(filename)
        print(f"📄 {filename} を処理しました（言語: {lang}, トークン数: {len(tokens)}）")

    if not raw_docs:
        print("🛑 処理できる有効な文書がありませんでした。処理を終了します。")
        return

    print("\n📊 TF-IDFベクトル化とコサイン類似度の計算を実行中...")
    # TF-IDFベクトライザの初期化
    vectorizer = TfidfVectorizer(ngram_range=NGRAM_RANGE, min_df=MIN_DF)
    # TF-IDF行列の計算
    tfidf_matrix = vectorizer.fit_transform(raw_docs)
    # コサイン類似度行列の計算
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # 結果をDataFrameに変換
    df_sim = pd.DataFrame(similarity_matrix, index=filenames, columns=filenames)
    # CSVファイルとして保存
    output_filename = "intertextual_proximity_matrix.csv"
    df_sim.to_csv(output_filename, encoding='utf-8-sig')
    print(f"✔ TF-IDF類似度行列をファイルに出力しました: {output_filename}")

    # --- ネットワーク構築機能は削除済み ---

    print("\n✨ 処理完了。")


if __name__ == "__main__":
    main()