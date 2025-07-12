import numpy as np
from gensim.models import Word2Vec
from scipy.linalg import orthogonal_procrustes

# === 模型加载 ===
model_ch = Word2Vec.load("model_chinese.model")
model_jp = Word2Vec.load("model_japanese.model")

# === 设置关键词：佛教相关词部件 ===
import numpy as np
from gensim.models import Word2Vec
from scipy.linalg import orthogonal_procrustes

# === 模型加载 ===
model_ch = Word2Vec.load("model_chinese.model")
model_jp = Word2Vec.load("model_japanese.model")

# === 设置关键词：佛教相关词部件 ===
# keywords 只包含词汇列表（用于匹配 anchor）
keywords = [
    "仏", "佛", "如来", "法", "空", "無", "心", "色", "性", "悟", "道", "僧", "戒", "涅槃", "菩薩",
    "因", "果", "真", "観", "禅", "経", "輪", "音", "念", "苦", "集", "滅",
    "坐禅", "無心", "公案", "見性", "悟り", "只管打坐", "道元", "臘八", "禅堂", "警策", "頭陀行",
    "本来無一物", "参禅", "語録", "念仏", "布薩", "袈裟", "十牛図", "仏性", "悟後の修行", "看話禅",
    "黙照禅", "五山文学", "四弘誓願", "因果応報", "菩提", "三界", "五蘊", "阿弥陀", "観音",
    "八正道", "四諦", "般若", "中道", "色即是空", "無我", "正念", "修行", "解脱", "三寶", "煩悩",
    "慈悲", "輪廻", "三昧", "一如", "随喜", "信心", "自性", "因縁", "涅槃寂静", "仏教", "真理",
    "方便", "衆生", "法身", "報身", "化身", "発菩提心", "転法輪", "十界", "六道", "三毒", "八苦",
    "六波羅蜜", "般若波羅蜜多", "不立文字", "見仏", "仏陀", "仏教経典", "毘盧遮那仏", "華厳",
    "法華", "阿含", "涅槃経", "摩訶般若", "真如", "無明", "受", "想", "行", "識", "因果",
    "縁起", "空性", "大乗", "小乗", "三乗", "円教", "蔵教", "通教", "別教", "即身成仏",
    "生死即涅槃", "諸法無我", "諸行無常", "一切皆苦", "有為法", "仏心", "仏道", "如実知見",
    "真如平等", "菩薩", "地蔵", "文殊", "普賢", "弥勒", "中陰", "往生", "浄土",
    "西方極楽浄土", "法界", "阿閦仏", "薬師如来", "三身", "慈", "悲", "喜", "捨", "五戒",
    "十善", "三学", "戒定慧", "自利", "利他", "清浄心", "遍知", "方便力", "弘誓",
    "般若心経", "金剛経", "大日経", "理趣経", "華厳経", "維摩経", "法華経", "観音経", "三部経"
]
buddhist_terms_extra = {
    "仏": "ぶつ", "法": "ほう", "僧": "そう", "空": "くう", "無": "む", "色": "しき",
    "苦": "く", "集": "じゅう", "滅": "めつ", "道": "どう", "智": "ち", "慧": "え",
    "心": "しん", "性": "しょう", "体": "たい", "中": "ちゅう", "明": "みょう",
    "光": "こう", "音": "おん", "身": "しん", "界": "かい", "有": "う"
}

# === 提取中日模型词汇交集中包含关键词的词 ===
common_vocab = set(model_ch.wv.index_to_key).intersection(set(model_jp.wv.index_to_key))
anchor_words = [w for w in common_vocab if any(k in w for k in keywords)]
print(f"✅ 筛选出可对齐的佛教 anchor 词数量：{len(anchor_words)}")

if len(anchor_words) < 2:
    print("❌ 锚定词数量不足，请检查模型词表或训练语料。")
    exit()

# === 执行 Procrustes 对齐 ===
X = np.array([model_ch.wv[w] for w in anchor_words])
Y = np.array([model_jp.wv[w] for w in anchor_words])
R, _ = orthogonal_procrustes(X, Y)

# === 映射中文词向量到日文空间 ===
aligned_ch_vectors = {
    word: model_ch.wv[word] @ R for word in model_ch.wv.index_to_key
}

# === 查询词设置（你可以修改）===
query_word = "空"
top_n = 15

# === 输出：中文模型原空间中相似词 ===
print(f"\n【中文模型】「{query_word}」的相似词（原空间）:")
if query_word in model_ch.wv:
    for i, (w, score) in enumerate(model_ch.wv.most_similar(query_word, topn=top_n), 1):
        print(f"{i:2d}. {w:<10} 相似度: {score:.4f}")
else:
    print("  詞不在中文模型中")

# === 输出：日文模型原空间中相似词 ===
print(f"\n【日本語モデル】「{query_word}」の近接語（日本語語義空間）:")
if query_word in model_jp.wv:
    for i, (w, score) in enumerate(model_jp.wv.most_similar(query_word, topn=top_n), 1):
        print(f"{i:2d}. {w:<10} 類似度: {score:.4f}")
else:
    print("  該当語が日本語モデルに存在しません。")

# === 输出：中文 query 映射到日文空间后的相似词 ===
print(f"\n【語義空間整列後】中文「{query_word}」對齊後 → 在日文空間中的近似詞：")
if query_word in aligned_ch_vectors:
    aligned_vec = aligned_ch_vectors[query_word]
    sims = []
    for word in model_jp.wv.index_to_key:
        v = model_jp.wv[word]
        sim = np.dot(aligned_vec, v) / (np.linalg.norm(aligned_vec) * np.linalg.norm(v))
        sims.append((word, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    for i, (w, score) in enumerate(sims[:top_n], 1):
        print(f"{i:2d}. {w:<10} 相似度: {score:.4f}")
else:
    print("  詞不在中文模型中")

buddhist_terms_extra = {
    "仏": "ぶつ", "法": "ほう", "僧": "そう", "空": "くう", "無": "む", "色": "しき",
    "苦": "く", "集": "じゅう", "滅": "めつ", "道": "どう", "智": "ち", "慧": "え",
    "心": "しん", "性": "しょう", "体": "たい", "中": "ちゅう", "明": "みょう",
    "光": "こう", "音": "おん", "身": "しん", "界": "かい", "有": "う"
}

# === 提取中日模型词汇交集中包含关键词的词 ===
common_vocab = set(model_ch.wv.index_to_key).intersection(set(model_jp.wv.index_to_key))
anchor_words = [w for w in common_vocab if any(k in w for k in keywords)]
print(f"✅ 筛选出可对齐的佛教 anchor 词数量：{len(anchor_words)}")

if len(anchor_words) < 2:
    print("❌ 锚定词数量不足，请检查模型词表或训练语料。")
    exit()

# === 执行 Procrustes 对齐 ===
X = np.array([model_ch.wv[w] for w in anchor_words])
Y = np.array([model_jp.wv[w] for w in anchor_words])
R, _ = orthogonal_procrustes(X, Y)

# === 映射中文词向量到日文空间 ===
aligned_ch_vectors = {
    word: model_ch.wv[word] @ R for word in model_ch.wv.index_to_key
}

# === 查询词设置（你可以修改）===
query_word = "空"
top_n = 15

# === 输出：中文模型原空间中相似词 ===
print(f"\n【中文模型】「{query_word}」的相似词（原空间）:")
if query_word in model_ch.wv:
    for i, (w, score) in enumerate(model_ch.wv.most_similar(query_word, topn=top_n), 1):
        print(f"{i:2d}. {w:<10} 相似度: {score:.4f}")
else:
    print("  詞不在中文模型中")

# === 输出：日文模型原空间中相似词 ===
print(f"\n【日本語モデル】「{query_word}」の近接語（日本語語義空間）:")
if query_word in model_jp.wv:
    for i, (w, score) in enumerate(model_jp.wv.most_similar(query_word, topn=top_n), 1):
        print(f"{i:2d}. {w:<10} 類似度: {score:.4f}")
else:
    print("  該当語が日本語モデルに存在しません。")

# === 输出：中文 query 映射到日文空间后的相似词 ===
print(f"\n【語義空間整列後】中文「{query_word}」對齊後 → 在日文空間中的近似詞：")
if query_word in aligned_ch_vectors:
    aligned_vec = aligned_ch_vectors[query_word]
    sims = []
    for word in model_jp.wv.index_to_key:
        v = model_jp.wv[word]
        sim = np.dot(aligned_vec, v) / (np.linalg.norm(aligned_vec) * np.linalg.norm(v))
        sims.append((word, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    for i, (w, score) in enumerate(sims[:top_n], 1):
        print(f"{i:2d}. {w:<10} 相似度: {score:.4f}")
else:
    print("  詞不在中文模型中")
