# -*- coding: utf-8 -*import os
import os
import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
from scipy.linalg import orthogonal_procrustes
import numpy as np



# 设置路径
CH_MODEL_PATH = "model_chinese.model"
JP_MODEL_PATH = "model_japanese.model"
CUSTOM_DIC_PATH = "custom.dic"  # 本地词典CSV文件，格式：word,definition
OUTPUT_CSV_FILE = "semantic_neighbors_offline.csv"

# 1. 加载模型
model_ch = Word2Vec.load(CH_MODEL_PATH)
model_jp = Word2Vec.load(JP_MODEL_PATH)

# 2. 找公共词汇锚点并正交对齐
vocab_ch = set(model_ch.wv.index_to_key)
vocab_jp = set(model_jp.wv.index_to_key)
shared_vocab = sorted(list(vocab_ch & vocab_jp),
                      key=lambda w: model_ch.wv.get_vecattr(w, "count") + model_jp.wv.get_vecattr(w, "count"),
                      reverse=True)
anchors = shared_vocab[:2000]

X = np.array([model_ch.wv[w] for w in anchors])
Y = np.array([model_jp.wv[w] for w in anchors])
R, _ = orthogonal_procrustes(X, Y)

aligned_ch = KeyedVectors(vector_size=model_ch.vector_size)
aligned_ch.add_vectors(model_ch.wv.index_to_key, model_ch.wv.vectors @ R)

# 3. 读取本地词典
custom_dic_df = pd.read_csv(CUSTOM_DIC_PATH, encoding='utf-8')
custom_dic_dict = dict(zip(custom_dic_df['word'], custom_dic_df['definition']))

def get_definition(word):
    if not word or word == "-":
        return "无定义"
    return custom_dic_dict.get(word, "无定义")

# 4. 找词近邻并生成简单差异说明（模板化）
keywords = shared_vocab[:200]
rows = []
for word in tqdm(keywords, desc="生成语义邻居及说明"):
    ch_neighbor = model_ch.wv.most_similar(word, topn=1)[0] if word in model_ch.wv else ("-", 0.0)
    jp_neighbor = model_jp.wv.most_similar(word, topn=1)[0] if word in model_jp.wv else ("-", 0.0)

    ch_def_word = get_definition(word)
    ch_def_neighbor = get_definition(ch_neighbor[0])
    jp_def_neighbor = get_definition(jp_neighbor[0])

    # 简单差异说明模板
    explanation = (
        f"核心词「{word}」的中文近邻为「{ch_neighbor[0]}」，其定义为：{ch_def_neighbor}。"
        f"日本语近邻为「{jp_neighbor[0]}」，定义为：{jp_def_neighbor}。"
        f"根据词义来看，中文近邻强调……（此处需要人工补充），"
        f"日语近邻更侧重……（此处需要人工补充）。"
    )

    rows.append({
        "核心词": word,
        "中文模型近邻": ch_neighbor[0],
        "中文相似度": round(ch_neighbor[1], 4),
        "日本语模型近邻": jp_neighbor[0],
        "日本语相似度": round(jp_neighbor[1], 4),
        "语义差异说明": explanation
    })

# 5. 保存结果
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')
print(f"处理完成，结果保存到 {OUTPUT_CSV_FILE}")
