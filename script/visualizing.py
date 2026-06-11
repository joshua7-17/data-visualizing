# ============================================================================
# CCKS 2021 中文地址文本相关性任务可视化
# 饼图，柱状图，散点图，热力图，小提琴图，箱型图
# 百分比分布直方，气泡图，kdeplot，误差棒图，词云图
# 折线图+面积图，河流图subplots，雷达图，network，节点图，地理分布图
# ============================================================================

import json
import os
import re
from collections import Counter
from pathlib import Path

import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager as fm
from wordcloud import WordCloud

# ============================================================================
# 0. 全局配置 —— 参照作业1~4的模板风格
# ============================================================================

# ---- 输出目录 ----
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---- 中文字体配置（参照作业2 homework2_solution.py 的字体探测函数）----
def setup_matplotlib_font() -> str | None:
    """探测 Windows 系统中文字体并配置 matplotlib，返回字体文件路径。"""
    candidate_fonts = [
        "C:/Windows/Fonts/msyh.ttc",     # 微软雅黑
        "C:/Windows/Fonts/msyh.ttf",
        "C:/Windows/Fonts/simhei.ttf",   # 黑体
        "C:/Windows/Fonts/simsun.ttc",   # 宋体
    ]
    for font_path in candidate_fonts:
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            font_name = fm.FontProperties(fname=font_path).get_name()
            plt.rcParams["font.family"] = font_name
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            return font_path
    # fallback
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei",
                                        "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return None

# ---- Seaborn 全局主题（参照作业2、3）—— 必须先调用，否则会覆盖字体设置 ----
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300

# ---- 中文字体设置（在 set_theme 之后执行，防止被覆盖）----
FONT_PATH = setup_matplotlib_font()

# ---- 调色板 ----
LABEL_COLORS = {
    "exact_match":   "#2ecc71",   # 绿色
    "partial_match": "#f39c12",   # 橙色
    "not_match":     "#e74c3c",   # 红色
}
LABEL_NAMES_CN = {
    "exact_match":   "完全匹配",
    "partial_match": "部分匹配",
    "not_match":     "不匹配",
}
PALETTE = [LABEL_COLORS["exact_match"], LABEL_COLORS["partial_match"],
           LABEL_COLORS["not_match"]]
LABEL_ORDER = ["exact_match", "partial_match", "not_match"]

# ============================================================================
# 1. 数据加载与特征工程
# ============================================================================

def load_jsonl(path: Path) -> list[dict]:
    """读取 JSONL 文件。"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def jaccard_similarity(s1: str, s2: str) -> float:
    """字符级 Jaccard 相似度。"""
    set1, set2 = set(s1), set(s2)
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)


def edit_distance(s1: str, s2: str) -> int:
    """Levenshtein 编辑距离（动态规划）。"""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def overlap_ratio(s1: str, s2: str) -> float:
    """公共字符占较短句子长度的比例。"""
    set1, set2 = set(s1), set(s2)
    shorter = min(len(set1), len(set2))
    if shorter == 0:
        return 0.0
    return len(set1 & set2) / shorter


# ---- 省份提取 ----
PROVINCES = [
    "北京", "天津", "上海", "重庆",
    "河北", "山西", "辽宁", "吉林", "黑龙江",
    "江苏", "浙江", "安徽", "福建", "江西", "山东",
    "河南", "湖北", "湖南", "广东", "海南",
    "四川", "贵州", "云南", "陕西", "甘肃", "青海",
    "台湾", "内蒙古", "广西", "西藏", "宁夏", "新疆",
]
# 部分城市 -> 省份映射（处理没有省份前缀的情况）
CITY_TO_PROVINCE = {
    "杭州": "浙江", "宁波": "浙江", "温州": "浙江", "嘉兴": "浙江", "绍兴": "浙江",
    "南京": "江苏", "苏州": "江苏", "无锡": "江苏", "常州": "江苏", "南通": "江苏",
    "合肥": "安徽", "芜湖": "安徽", "福州": "福建", "厦门": "福建", "泉州": "福建",
    "广州": "广东", "深圳": "广东", "东莞": "广东", "佛山": "广东", "珠海": "广东",
    "成都": "四川", "绵阳": "四川", "武汉": "湖北", "宜昌": "湖北",
    "长沙": "湖南", "株洲": "湖南", "郑州": "河南", "洛阳": "河南",
    "济南": "山东", "青岛": "山东", "烟台": "山东", "潍坊": "山东",
    "石家庄": "河北", "唐山": "河北", "保定": "河北",
    "太原": "山西", "大同": "山西",
    "沈阳": "辽宁", "大连": "辽宁", "长春": "吉林", "哈尔滨": "黑龙江",
    "南昌": "江西", "昆明": "云南", "贵阳": "贵州", "西安": "陕西",
    "兰州": "甘肃", "西宁": "青海", "海口": "海南", "拉萨": "西藏",
    "呼和浩特": "内蒙古", "南宁": "广西", "银川": "宁夏", "乌鲁木齐": "新疆",
    "思明": "福建", "湖里": "福建", "集美": "福建",  # 厦门下辖区
    "朝阳": "北京", "海淀": "北京", "丰台": "北京", "通州": "北京",
    "浦东": "上海", "黄浦": "上海", "静安": "上海", "徐汇": "上海",
    "渝北": "重庆", "江北": "重庆", "沙坪坝": "重庆",
    "和平": "天津", "河西": "天津", "南开": "天津",
    "苏家屯": "辽宁", "铁西": "辽宁",
    "秦淮": "江苏", "鼓楼": "江苏", "建邺": "江苏", "玄武": "江苏",
    "余杭": "浙江", "萧山": "浙江", "滨江": "浙江", "西湖": "浙江",
}


def extract_province(text: str) -> str:
    """从地址文本中提取省份。"""
    # 先尝试直接匹配省份名
    for prov in PROVINCES:
        if prov in text:
            return prov
    # 再尝试城市名映射
    for city, prov in CITY_TO_PROVINCE.items():
        if city in text:
            return prov
    return "未知"


def build_features(data: list[dict]) -> pd.DataFrame:
    """从原始数据构建特征 DataFrame。"""
    print("  正在提取文本特征...")
    records = []
    total = len(data)
    for i, item in enumerate(data):
        if (i + 1) % 5000 == 0:
            print(f"    已处理 {i + 1}/{total} 条...")
        s1 = item.get("sentence1", "")
        s2 = item.get("sentence2", "")
        label = item.get("label", "")
        len1 = len(s1)
        len2 = len(s2)

        records.append({
            "sentence1": s1,
            "sentence2": s2,
            "label": label,
            "len1": len1,
            "len2": len2,
            "len_diff": abs(len1 - len2),
            "len_ratio": len1 / len2 if len2 > 0 else 0,
            "overlap_ratio": overlap_ratio(s1, s2),
            "jaccard": jaccard_similarity(s1, s2),
            "edit_dist": edit_distance(s1, s2),
            "province": extract_province(s1 + s2),
        })
    df = pd.DataFrame(records)
    print(f"  特征构建完成，共 {len(df)} 条数据。")
    return df


# ============================================================================
# 2. 绘图函数 —— 不依赖模型的图表
# ============================================================================

def plot_01_pie_chart(df: pd.DataFrame):
    """图1：饼图 — 三类标签分布占比。"""
    counts = df["label"].value_counts().reindex(LABEL_ORDER)
    labels_cn = [LABEL_NAMES_CN[k] for k in LABEL_ORDER]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=labels_cn,
        colors=PALETTE,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": 14},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        pctdistance=0.75,
    )
    for autotext in autotexts:
        autotext.set_fontsize(13)
        autotext.set_fontweight("bold")

    # 中央空心环效果
    centre_circle = plt.Circle((0, 0), 0.50, fc="white")
    ax.add_artist(centre_circle)
    ax.set_title("地址文本匹配标签分布", fontsize=18, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_饼图_标签分布.png")
    plt.close()
    print("  ✅ 01_饼图_标签分布.png")


def plot_02_bar_chart(df: pd.DataFrame):
    """图2：柱状图 — 不同标签下平均文本长度。"""
    avg_data = df.groupby("label")[["len1", "len2"]].mean().reindex(LABEL_ORDER)
    labels_cn = [LABEL_NAMES_CN[k] for k in LABEL_ORDER]

    x = np.arange(len(LABEL_ORDER))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, avg_data["len1"], width, label="地址1 (sentence1)",
                   color="#3498db", edgecolor="white", linewidth=1.2)
    bars2 = ax.bar(x + width / 2, avg_data["len2"], width, label="地址2 (sentence2)",
                   color="#e67e22", edgecolor="white", linewidth=1.2)

    # 数值标注
    for bar in bars1:
        ax.annotate(f"{bar.get_height():.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=11)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_cn, fontsize=13)
    ax.set_ylabel("平均字符长度", fontsize=13)
    ax.set_title("不同匹配类别下地址文本的平均长度", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_ylim(0, avg_data.values.max() * 1.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_柱状图_平均文本长度.png")
    plt.close()
    print("  ✅ 02_柱状图_平均文本长度.png")


def plot_03_histogram(df: pd.DataFrame):
    """图3：百分比分布直方图 — 文本长度分布。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (col, title) in enumerate([("len1", "地址1 (sentence1) 长度分布"),
                                         ("len2", "地址2 (sentence2) 长度分布")]):
        ax = axes[idx]
        for label in LABEL_ORDER:
            subset = df[df["label"] == label][col]
            ax.hist(subset, bins=30, alpha=0.6, label=LABEL_NAMES_CN[label],
                    color=LABEL_COLORS[label], edgecolor="white", linewidth=0.5,
                    density=True)
        ax.set_xlabel("字符长度", fontsize=12)
        ax.set_ylabel("密度", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)

    plt.suptitle("地址文本长度百分比分布直方图", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_直方图_文本长度分布.png", bbox_inches="tight")
    plt.close()
    print("  ✅ 03_直方图_文本长度分布.png")


def plot_04_boxplot(df: pd.DataFrame):
    """图4：箱型图 — 不同标签下文本长度差分布。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    order_cn = [LABEL_NAMES_CN[k] for k in LABEL_ORDER]
    df_plot = df.copy()
    df_plot["label_cn"] = df_plot["label"].map(LABEL_NAMES_CN)

    sns.boxplot(data=df_plot, x="label_cn", y="len_diff", order=order_cn,
                palette=PALETTE, width=0.5, fliersize=3, ax=ax)
    ax.set_xlabel("匹配类别", fontsize=13)
    ax.set_ylabel("文本长度差 (|len1 - len2|)", fontsize=13)
    ax.set_title("不同匹配类别下两段地址的长度差异分布", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_箱型图_长度差分布.png")
    plt.close()
    print("  ✅ 04_箱型图_长度差分布.png")


def plot_05_violin(df: pd.DataFrame):
    """图5：小提琴图 — 不同标签下 Jaccard 相似度分布。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    order_cn = [LABEL_NAMES_CN[k] for k in LABEL_ORDER]
    df_plot = df.copy()
    df_plot["label_cn"] = df_plot["label"].map(LABEL_NAMES_CN)

    sns.violinplot(data=df_plot, x="label_cn", y="jaccard", order=order_cn,
                   palette=PALETTE, inner="quartile", cut=0, ax=ax)
    ax.set_xlabel("匹配类别", fontsize=13)
    ax.set_ylabel("Jaccard 相似度", fontsize=13)
    ax.set_title("不同匹配类别下字符级 Jaccard 相似度分布",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_小提琴图_Jaccard相似度.png")
    plt.close()
    print("  ✅ 05_小提琴图_Jaccard相似度.png")


def plot_06_kdeplot(df: pd.DataFrame):
    """图6：KDE 密度分布图 — 不同标签下文本长度概率密度。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    for label in LABEL_ORDER:
        subset = df[df["label"] == label]["len1"]
        sns.kdeplot(subset, ax=ax, label=LABEL_NAMES_CN[label],
                    color=LABEL_COLORS[label], linewidth=2.5, fill=True, alpha=0.15)
    ax.set_xlabel("地址1 字符长度", fontsize=13)
    ax.set_ylabel("概率密度", fontsize=13)
    ax.set_title("不同匹配类别下地址文本长度的核密度估计 (KDE)",
                 fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_KDE_文本长度密度.png")
    plt.close()
    print("  ✅ 06_KDE_文本长度密度.png")


def plot_07_scatter(df: pd.DataFrame):
    """图7：散点图 — sentence1 长度 vs sentence2 长度。"""
    fig, ax = plt.subplots(figsize=(10, 8))
    for label in LABEL_ORDER:
        subset = df[df["label"] == label]
        ax.scatter(subset["len1"], subset["len2"], alpha=0.35, s=15,
                   color=LABEL_COLORS[label], label=LABEL_NAMES_CN[label],
                   edgecolors="none")
    # 对角参考线
    max_val = max(df["len1"].max(), df["len2"].max())
    ax.plot([0, max_val], [0, max_val], "--", color="gray", alpha=0.5, linewidth=1)

    ax.set_xlabel("地址1 字符长度 (sentence1)", fontsize=13)
    ax.set_ylabel("地址2 字符长度 (sentence2)", fontsize=13)
    ax.set_title("地址对文本长度散点图", fontsize=16, fontweight="bold")
    ax.legend(fontsize=11, markerscale=3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_散点图_长度对比.png")
    plt.close()
    print("  ✅ 07_散点图_长度对比.png")


def plot_08_bubble(df: pd.DataFrame):
    """图8：气泡图 — X=len1, Y=len2, 气泡大小=编辑距离。"""
    # 为性能考虑，抽样 2000 条
    sample = df.sample(n=min(2000, len(df)), random_state=42).copy()

    fig, ax = plt.subplots(figsize=(12, 8))
    for label in LABEL_ORDER:
        subset = sample[sample["label"] == label]
        # 气泡大小：编辑距离归一化后 * 缩放因子
        sizes = subset["edit_dist"] / sample["edit_dist"].max() * 200 + 10
        ax.scatter(subset["len1"], subset["len2"], s=sizes, alpha=0.45,
                   color=LABEL_COLORS[label], label=LABEL_NAMES_CN[label],
                   edgecolors="white", linewidth=0.3)
    ax.set_xlabel("地址1 字符长度", fontsize=13)
    ax.set_ylabel("地址2 字符长度", fontsize=13)
    ax.set_title("气泡图：文本长度与编辑距离的多维关系\n(气泡大小 = 编辑距离)",
                 fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, markerscale=0.8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_气泡图_多维关系.png")
    plt.close()
    print("  ✅ 08_气泡图_多维关系.png")


def plot_09_heatmap(df: pd.DataFrame):
    """图9：热力图 — 数值特征相关系数矩阵。"""
    numeric_cols = ["len1", "len2", "len_diff", "len_ratio",
                    "overlap_ratio", "jaccard", "edit_dist"]
    col_names_cn = ["地址1长度", "地址2长度", "长度差", "长度比",
                    "公共字符占比", "Jaccard相似度", "编辑距离"]
    corr = df[numeric_cols].corr()
    corr.index = col_names_cn
    corr.columns = col_names_cn

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=1, linecolor="white",
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("地址文本特征相关系数矩阵", fontsize=16, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "09_热力图_特征相关矩阵.png")
    plt.close()
    print("  ✅ 09_热力图_特征相关矩阵.png")


def plot_10_wordcloud(df: pd.DataFrame):
    """图10：词云图 — 地址文本高频词（分面：完全匹配 vs 不匹配）。"""
    # 加载停用词（通用地理词汇）
    stop_words = {"省", "市", "区", "县", "镇", "乡", "街道", "路", "号",
                  "号楼", "栋", "单元", "室", "层", "幢", "座", "期",
                  "小区", "大厦", "广场", "花园", "公寓", "中心",
                  "东", "西", "南", "北", "中", "门", "口", "旁", "内", "外",
                  "组", "村", "弄", "巷", "里", "楼", "的"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    plot_labels = ["exact_match", "not_match"]
    plot_titles = ["完全匹配地址的高频词", "不匹配地址的高频词"]

    for idx, (label, title) in enumerate(zip(plot_labels, plot_titles)):
        subset = df[df["label"] == label]
        all_text = " ".join(subset["sentence1"].tolist() + subset["sentence2"].tolist())
        words = jieba.lcut(all_text)
        words = [w for w in words if len(w) > 1 and w not in stop_words]
        word_freq = Counter(words)

        wc_kwargs = {
            "width": 800,
            "height": 500,
            "background_color": "white",
            "max_words": 120,
            "colormap": "tab20",
            "max_font_size": 100,
            "min_font_size": 10,
            "random_state": 42,
            "collocations": False,
        }
        if FONT_PATH:
            wc_kwargs["font_path"] = FONT_PATH

        wc = WordCloud(**wc_kwargs).generate_from_frequencies(word_freq)
        axes[idx].imshow(wc, interpolation="bilinear")
        axes[idx].axis("off")
        axes[idx].set_title(title, fontsize=15, fontweight="bold", pad=10)

    plt.suptitle("地址文本词云分析", fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "10_词云图_高频词.png", bbox_inches="tight")
    plt.close()
    print("  ✅ 10_词云图_高频词.png")


def plot_11_province_bar(df: pd.DataFrame):
    """图11：地理分布条形图 — 各省份样本分布。"""
    prov_counts = df[df["province"] != "未知"]["province"].value_counts()
    top_n = min(20, len(prov_counts))
    prov_top = prov_counts.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))[::-1]
    bars = ax.barh(range(top_n), prov_top.values, color=colors,
                   edgecolor="white", linewidth=1)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(prov_top.index, fontsize=12,
                       fontfamily="Microsoft YaHei")
    ax.invert_yaxis()
    ax.set_xlabel("样本数量", fontsize=13, fontfamily="Microsoft YaHei")
    ax.set_title(f"地址数据全国省份分布 (Top {top_n})",
                 fontsize=16, fontweight="bold", fontfamily="Microsoft YaHei")

    # 数值标注
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f"{int(width)}",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=10, color="#333")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "11_条形图_省份分布.png")
    plt.close()
    print("  ✅ 11_条形图_省份分布.png")


# ============================================================================
# 3. 主函数入口
# ============================================================================

def main():
    print("=" * 60)
    print("  CCKS 2021 中文地址文本相关性 — 数据可视化")
    print("=" * 60)

    # ---- 加载数据 ----
    dev_path = DATA_DIR / "dev.json"
    print(f"\n📂 加载数据：{dev_path}")
    raw_data = load_jsonl(dev_path)
    print(f"   原始数据量：{len(raw_data)} 条")

    # ---- 构建特征 ----
    print("\n🔧 特征工程：")
    df = build_features(raw_data)

    # ---- 保存特征表 ----
    feature_path = DATA_DIR / "processed_features.csv"
    df.to_csv(feature_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 特征数据已保存：{feature_path}")

    # ---- 数据概览 ----
    print("\n📊 数据概览：")
    print(f"   样本总数：{len(df)}")
    for label in LABEL_ORDER:
        count = (df["label"] == label).sum()
        pct = count / len(df) * 100
        print(f"   {LABEL_NAMES_CN[label]}: {count} ({pct:.1f}%)")

    # ---- 绘制图表 ----
    print(f"\n🎨 开始绘制图表（输出目录：{OUTPUT_DIR}）\n")

    plot_01_pie_chart(df)
    plot_02_bar_chart(df)
    plot_03_histogram(df)
    plot_04_boxplot(df)
    plot_05_violin(df)
    plot_06_kdeplot(df)
    plot_07_scatter(df)
    plot_08_bubble(df)
    plot_09_heatmap(df)
    plot_10_wordcloud(df)
    plot_11_province_bar(df)

    print(f"\n✨ 全部完成！共生成 11 张图表，保存在：{OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()