"""클래스별 대표 웨이퍼맵을 그려 figures/에 저장한다.

실행:
    python src/explore.py

목적:
    숫자를 보기 전에 눈으로 본다. 패턴이 실제로 어떻게 생겼는지 모르면
    "왜 헷갈리는가"를 설명할 수 없다.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 창 없이 파일로만 저장
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import config

# 0=영역 밖(흰색), 1=정상(연회색), 2=불량(빨강)
CMAP = ListedColormap(["#ffffff", "#dcdcdc", "#d32f2f"])


def load_class(cls):
    path = config.DATA_PROCESSED / f"{cls}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def main():
    rng = np.random.default_rng(config.SEED)
    config.FIGURES.mkdir(parents=True, exist_ok=True)

    n_cols = 6
    classes = [c for c in config.ALL_CLASSES
               if (config.DATA_PROCESSED / f"{c}.npz").exists()]
    if not classes:
        raise SystemExit("[중단] data/processed/ 가 비었다. 먼저 load_data.py를 실행하라.")

    fig, axes = plt.subplots(
        len(classes), n_cols,
        figsize=(n_cols * 1.6, len(classes) * 1.7),
    )
    axes = np.atleast_2d(axes)

    for r, cls in enumerate(classes):
        data = load_class(cls)
        maps = data["wafer_maps"]
        pick = rng.choice(len(maps), size=min(n_cols, len(maps)), replace=False)
        for c in range(n_cols):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            if c < len(pick):
                ax.imshow(maps[pick[c]], cmap=CMAP, vmin=0, vmax=2,
                          interpolation="nearest")
            else:
                ax.axis("off")
            if c == 0:
                ax.set_ylabel(f"{cls}\n(n={len(maps):,})",
                              rotation=0, ha="right", va="center", fontsize=8)

    fig.suptitle("WM-811K — class samples", fontsize=11)
    fig.tight_layout()
    out = config.FIGURES / "class_samples.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"저장: {out}")

    # 목표 혼동 쌍만 크게 — 왜 헷갈리는지 눈으로 확인
    a, b = config.TARGET_PAIR
    if all((config.DATA_PROCESSED / f"{c}.npz").exists() for c in (a, b)):
        fig, axes = plt.subplots(2, 8, figsize=(14, 4))
        for r, cls in enumerate((a, b)):
            maps = load_class(cls)["wafer_maps"]
            pick = rng.choice(len(maps), size=min(8, len(maps)), replace=False)
            for c in range(8):
                ax = axes[r, c]
                ax.set_xticks([]); ax.set_yticks([])
                if c < len(pick):
                    ax.imshow(maps[pick[c]], cmap=CMAP, vmin=0, vmax=2,
                              interpolation="nearest")
                else:
                    ax.axis("off")
                if c == 0:
                    ax.set_ylabel(cls, rotation=0, ha="right",
                                  va="center", fontsize=10)
        fig.suptitle(f"{a} vs {b} — 방위각 분포가 실제로 다른가?", fontsize=11)
        fig.tight_layout()
        out = config.FIGURES / "target_pair.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"저장: {out}")

    print("\n그림을 열어서 직접 봐라. 남길 그림은 figures/keep/ 으로 옮긴다.")


if __name__ == "__main__":
    main()
