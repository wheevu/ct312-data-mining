#!/usr/bin/env python3
"""Build the categorized Lab 4 turn-in folder in ~/Downloads/lab4."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TURN_IN = Path.home() / "Downloads" / "lab4" / "B2113348_NguyenHuyVu_Lab4_Clustering"

EXERCISES = [
    ("01", "kmeans_toy", [], "01_kmeans_toy.py"),
    ("02", "abc_customers_kmeans", ["ABC_Customers.csv"], "02_abc_customers_kmeans.py"),
    ("03", "abc_customers_hierarchical_compare", ["ABC_Customers.csv"], "03_abc_customers_hierarchical_compare.py"),
    ("04", "student_single_linkage", [], "04_student_single_linkage.py"),
    ("05", "meanshift_3d", ["MeanShift-3D.csv"], "05_meanshift_3d.py"),
    ("06", "eurojobs", ["Eurojobs.csv"], "06_eurojobs.py"),
    ("07", "flowers", ["flowers.csv"], "07_flowers.py"),
    ("08", "usarrests", ["USArrests.csv"], "08_usarrests.py"),
    ("09", "rfm", ["dataCustomerRFM.csv"], "09_rfm.py"),
    ("10", "bank", ["bank-data.csv"], "10_bank.py"),
    ("11", "ecommerce_spending", ["ABC_customerSpending.csv"], "11_ecommerce_spending.py"),
    ("12", "moon", ["moon_dataset.csv"], "12_moon.py"),
]

NOTES = {
    "01": "Ví dụ KMeans với dữ liệu mẫu 6 điểm và tâm cụm khởi tạo cố định, kiểm tra dự đoán trên điểm mới.",
    "02": "Phân khúc khách hàng ABC bằng KMeans: phương pháp elbow, phân cụm theo thu nhập/chi tiêu, theo tuổi/chi tiêu và kết hợp 3 đặc trưng.",
    "03": "So sánh KMeans, Agglomerative, DBSCAN và MeanShift trên cùng dữ liệu ABC Customers.",
    "04": "Phân cụm đơn liên kết (single linkage) thủ công cho 8 sinh viên theo điểm Toán và Văn, kiểm chứng với SciPy.",
    "05": "MeanShift trên dữ liệu 3D tổng hợp; ước lượng bandwidth, xác định số cụm và trực quan hóa.",
    "06": "Phân cụm các nước châu Âu theo cơ cấu ngành nghề; dendrogram và giải thích cụm.",
    "07": "Phân cụm dữ liệu hoa (giống Iris) với KMeans, đánh giá silhouette cho các k khác nhau.",
    "08": "Phân cụm 50 bang Hoa Kỳ theo tỷ lệ tội phạm (Murder, Assault, UrbanPop, Rape) sau khi chuẩn hóa.",
    "09": "Phân khúc khách hàng RFM (Recency-Frequency-Monetary) từ dữ liệu đơn hàng lớn.",
    "10": "Phân cụm khách hàng ngân hàng với mã hóa biến phân loại và chuẩn hóa đặc trưng.",
    "11": "Phân khúc khách hàng thương mại điện tử theo tổng chi tiêu, tần suất và đa dạng danh mục.",
    "12": "So sánh DBSCAN và KMeans trên dữ liệu mặt trăng (two moons) — KMeans thất bại do cụm không lồi.",
}

COMMON_BLOCK = '''from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PLOTS_DIR = ROOT / "plots"
RESULTS_DIR = ROOT / "results"

def ensure_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def prepare_data(df, feature_cols, *, scale=True):
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    if scale:
        X_scaled = StandardScaler().fit_transform(X_imp)
    else:
        X_scaled = X_imp
    n = X_scaled.shape[1]
    if n == 2:
        coords = X_scaled
    elif n == 1:
        coords = np.column_stack([X_scaled[:, 0], np.zeros(len(X_scaled))])
    else:
        coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    return X_scaled, coords

def labeled_points(frame, labels, coords):
    out = frame.copy()
    out["cluster"] = labels.astype(int)
    out["plot_x"] = coords[:, 0]
    out["plot_y"] = coords[:, 1]
    return out

def cluster_summary(labels, *, true_labels=None):
    rows = []
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        row = {"cluster": int(label), "n_rows": int(mask.sum()), "share_pct": round(float(mask.mean() * 100), 2)}
        if true_labels is not None:
            vc = true_labels[mask].astype(str).value_counts()
            if len(vc):
                row["dominant_true"] = vc.index[0]
                row["dominant_count"] = int(vc.iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)

def safe_silhouette(X, labels):
    if len(set(labels.tolist())) < 2:
        return None
    if len(labels) < 3:
        return None
    return float(silhouette_score(X, labels))

def save_metrics(rows, filename):
    ensure_dirs()
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / filename, index=False)
    print(df.to_string(index=False))
    print(f"Saved {RESULTS_DIR / filename}")
    return df

def savefig(filename):
    ensure_dirs()
    path = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

def elbow_data(X, k_max=10, random_state=42):
    from sklearn.cluster import KMeans
    rows = []
    for k in range(1, k_max + 1):
        m = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        m.fit(X)
        rows.append({"k": k, "inertia": float(m.inertia_)})
    return pd.DataFrame(rows)
'''


def standalone_script(content: str) -> str:
    """Inline the COMMON_BLOCK and adapt cache_data calls to local paths."""
    lines = content.splitlines()
    output: list[str] = []
    inserted = False

    skip_prefixes = (
        "from common import",
        "import matplotlib.pyplot as plt",
        "import numpy as np",
        "import pandas as pd",
        "from sklearn.decomposition import PCA",
        "from sklearn.impute import SimpleImputer",
        "from sklearn.metrics import silhouette_score",
        "from sklearn.preprocessing import StandardScaler",
    )
    # Extra modules brought in via COMMON_BLOCK
    extra_skip = {
        "from sklearn.metrics import silhouette_score",
    }

    for line in lines:
        # Replace `from common import ...` with COMMON_BLOCK
        if line.startswith("from common import"):
            if not inserted:
                output.append(COMMON_BLOCK.rstrip())
                inserted = True
            continue

        # Replace cache_data("...") with DATA_DIR / "..."
        if "cache_data(" in line:
            line = line.replace("cache_data(", "(DATA_DIR / ").replace(")", ")")

        # Skip duplicate imports already in COMMON_BLOCK
        if inserted and any(line.startswith(p) for p in skip_prefixes):
            # Check if this is a non-duplicate import (e.g. other sklearn submodules)
            if line.startswith("from sklearn."):
                # Keep imports not already covered
                pass
            continue

        output.append(line)

    result = "\n".join(output)
    # Remove triple blank lines
    while "\n\n\n" in result:
        result = result.replace("\n\n\n", "\n\n")
    return result + "\n"


def copy_matching(src_dir: Path, dst_dir: Path, prefix: str) -> None:
    for path in src_dir.glob(f"bai_{prefix}_*"):
        shutil.copy2(path, dst_dir / path.name)


def extract_data_filenames(line: str) -> list[str]:
    """Generate data filenames needed for an exercise."""
    # Already provided in EXERCISES, this is a fallback
    return []


def main() -> None:
    scripts_dir = ROOT / "scripts" / "lab4"
    data_dir = ROOT / "data" / "lab4"
    outputs_dir = ROOT / "outputs" / "lab4"
    plots_dir = outputs_dir / "plots"
    results_dir = outputs_dir / "results"

    if TURN_IN.exists():
        shutil.rmtree(TURN_IN)
    TURN_IN.mkdir(parents=True)

    (TURN_IN / "requirements.txt").write_text(
        "numpy\npandas\nmatplotlib\nseaborn\nscikit-learn\nscipy\n"
    )
    (TURN_IN / "README.md").write_text(
        "# B2113348_NguyenHuyVu_Lab4_Clustering\n\n"
        "Mỗi thư mục `Bai_XX/` chứa script, dữ liệu, biểu đồ và kết quả riêng cho bài đó.\n"
        "Chạy từng bài độc lập, ví dụ:\n\n"
        "```bash\ncd Bai_05\npython bai_05_meanshift_3d.py\n```\n"
    )

    report_parts = ["# Báo cáo Lab 4 — Phân cụm\n",
                    "Sinh viên: **B2113348 - Nguyen Huy Vu**\n"]

    for num, slug, data_files, script_file in EXERCISES:
        folder = TURN_IN / f"Bai_{num}"
        data_dir_local = folder / "data"
        plots_dir_local = folder / "plots"
        results_dir_local = folder / "results"
        for d in (data_dir_local, plots_dir_local, results_dir_local):
            d.mkdir(parents=True, exist_ok=True)

        # Copy data files
        for df_name in data_files:
            src = data_dir / df_name
            if src.exists():
                shutil.copy2(src, data_dir_local / df_name)
            else:
                print(f"WARNING: data file not found: {src}")

        # Inline and write standalone script
        src_script = scripts_dir / script_file
        script_text = standalone_script(src_script.read_text())
        (folder / f"bai_{num}_{slug}.py").write_text(script_text)

        # Copy plots and results
        copy_matching(plots_dir, plots_dir_local, num)
        copy_matching(results_dir, results_dir_local, num)

        # Build report section
        report_parts.append(
            f"\n## Bài {int(num)}\n\n"
            f"**Nhận xét.** {NOTES[num]}\n\n"
            f"- Thư mục: `Bai_{num}/`\n"
            f"- Script: `Bai_{num}/bai_{num}_{slug}.py`\n"
            f"- Dữ liệu: `Bai_{num}/data/` ({', '.join(data_files) if data_files else 'không có, dữ liệu tự sinh'})\n"
            f"- Biểu đồ: `Bai_{num}/plots/`\n"
            f"- Kết quả: `Bai_{num}/results/`\n"
            f"{_metrics_markdown(num, results_dir)}\n"
        )

    report_parts.append(
        "\n## Nhận xét chung\n\n"
        "Các bài thực hành phân cụm cho thấy:\n\n"
        "- **KMeans** phù hợp với cụm hình cầu, lồi, cần xác định k trước và nhạy với khởi tạo.\n"
        "- **Hierarchical clustering** cung cấp cái nhìn trực quan qua dendrogram và không cần k trước.\n"
        "- **DBSCAN** phát hiện cụm không lồi và điểm nhiễu, nhưng nhạy với tham số eps.\n"
        "- **MeanShift** tự động tìm số cụm nhưng chậm trên dữ liệu lớn.\n"
        "- Chuẩn hóa dữ liệu (StandardScaler) là bước bắt buộc trước khi phân cụm.\n"
        "- Phân khúc RFM và khách hàng cho thấy ứng dụng thực tế của phân cụm trong kinh doanh.\n"
    )
    (TURN_IN / "Bao_cao_Lab4_Clustering.md").write_text("\n".join(report_parts))
    print(f"Built {TURN_IN}")


def _metrics_markdown(prefix: str, results_dir: Path) -> str:
    """Read the metrics CSV for a given bài and return a markdown table."""
    import glob as glob_mod
    pattern = str(results_dir / f"bai_{prefix}_*metrics.csv")
    matches = glob_mod.glob(pattern)
    if not matches:
        return "\n*Kết quả chi tiết trong thư mục `results/`.*\n"
    path = Path(matches[0])
    df = pd.read_csv(path)
    # Keep relevant columns
    skip_cols = {"Unnamed: 0"}
    keep = [c for c in df.columns if c not in skip_cols]
    if not keep:
        return "\n*Kết quả chi tiết trong thư mục `results/`.*\n"
    lines = ["", "| " + " | ".join(keep) + " |",
             "| " + " | ".join(["---"] * len(keep)) + " |"]
    for _, row in df[keep].iterrows():
        values = []
        for col in keep:
            val = row[col]
            if pd.isna(val):
                values.append("—")
            elif isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    # pd needed for _metrics_markdown
    import pandas as pd  # noqa: F401
    main()
