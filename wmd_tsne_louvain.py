# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.font_manager as fm
import networkx as nx
import community  # pip install python-louvain
from matplotlib.lines import Line2D

# ===== ファイルパスの設定 =====
WMD_RESULTS_PATH = "wmd_results.csv"

# ===== クロスプラットフォーム対応の日本語/中国語フォント設定 =====
def get_chinese_font():
    font_candidates = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for path in font_candidates:
        if os.path.exists(path):
            return fm.FontProperties(fname=path)
    return fm.FontProperties()

font = get_chinese_font()
plt.rcParams["font.sans-serif"] = [font.get_name()]
plt.rcParams["axes.unicode_minus"] = False

# ===== Louvainクラスタリングと描画 =====
def compute_and_plot_communities(distance_matrix, embedding, all_docs, font):
    print("🔗 グラフ構築 & Louvain 社群検出中...")
    G = nx.Graph()
    for i in range(len(all_docs)):
        for j in range(i + 1, len(all_docs)):
            a, b = all_docs[i], all_docs[j]
            dist = distance_matrix.loc[a, b]
            if dist > 0:
                G.add_edge(a, b, weight=1 / dist)

    partition = community.best_partition(G, weight='weight')
    communities = sorted(set(partition.values()))
    color_map = plt.get_cmap("tab10")
    comm_color_dict = {comm: color_map(i % 10) for i, comm in enumerate(communities)}
    node_colors = [comm_color_dict[partition[node]] for node in all_docs]

    plt.figure(figsize=(16, 16))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=node_colors, alpha=0.8)
    for i, name in enumerate(all_docs):
        plt.text(embedding[i, 0], embedding[i, 1], name, fontsize=8, fontproperties=font)

    plt.title("WMD + Louvainによる禅宗文献の意味空間クラスタリング", fontproperties=font, fontsize=20)
    plt.xlabel("t-SNE次元1", fontproperties=font)
    plt.ylabel("t-SNE次元2", fontproperties=font)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label=f"社群 {comm}", markersize=10, markerfacecolor=color)
        for comm, color in comm_color_dict.items()
    ]
    plt.legend(handles=legend_elements, prop=font, loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("semantic_space_modularity.png")
    print("✅ 社群図保存完了：semantic_space_modularity.png")
    plt.show()

# ===== WMD距離を使ってt-SNEで意味空間を可視化する関数 =====
def visualize_semantic_space():
    print("📥 WMD距離結果ファイルを読み込み中...")
    wmd_df = pd.read_csv(WMD_RESULTS_PATH)

    doc_set = set(wmd_df["Document_A"]).union(set(wmd_df["Document_B"]))
    all_docs = sorted(doc_set)
    distance_matrix = pd.DataFrame(np.nan, index=all_docs, columns=all_docs)

    for _, row in wmd_df.iterrows():
        a, b, dist = row["Document_A"], row["Document_B"], row["Distance"]
        distance_matrix.loc[a, b] = dist
        distance_matrix.loc[b, a] = dist
    distance_matrix.fillna(0, inplace=True)

    print("✅ 距離行列の構築完了。文書数：", len(all_docs))

    lang_colors = {}
    for fname in all_docs:
        base_name = os.path.splitext(fname)[0]
        match = re.match(r"(\d+)([A-Za-z]*)", base_name)
        if match:
            base_num, suffix = match.groups()
            lang = "japanese" if int(base_num) >= 2025 and not suffix else "chinese"
        else:
            lang = "chinese"
        lang_colors[fname] = "red" if lang == "japanese" else "blue"
    colors = [lang_colors[f] for f in all_docs]

    print("🌀 t-SNEを実行中（次元削減）...")
    tsne = TSNE(n_components=2, metric="precomputed", random_state=42, perplexity=30, init="random")
    embedding = tsne.fit_transform(distance_matrix)

    print("📊 意味空間マップを描画中...")
    plt.figure(figsize=(16, 16))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.7)
    for i, name in enumerate(all_docs):
        plt.text(embedding[i, 0], embedding[i, 1], name, fontsize=8, fontproperties=font)

    plt.title("WMDに基づく禅宗文献の意味空間", fontproperties=font, fontsize=20)
    plt.xlabel("t-SNE次元1", fontproperties=font)
    plt.ylabel("t-SNE次元2", fontproperties=font)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="中国語文献", markersize=10, markerfacecolor="blue"),
        Line2D([0], [0], marker="o", color="w", label="日本語文献", markersize=10, markerfacecolor="red"),
    ]
    plt.legend(handles=legend_elements, prop=font)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("semantic_space_map.png")
    print("✅ 可視化完了。ファイル保存：semantic_space_map.png")
    plt.show()

    # Louvain 社群分析を追加
    compute_and_plot_communities(distance_matrix, embedding, all_docs, font)

# ===== メイン実行 =====
if __name__ == "__main__":
    visualize_semantic_space()
