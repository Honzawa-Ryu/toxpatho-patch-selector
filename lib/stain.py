"""Macenko stain normalization for H&E patches.

0004で、病変が一切無いcontrolスライド同士が由来する実験で有意に分離することが分かった
（silhouette 0.175, z=31.9）。染色・スキャン条件のバッチ署名が埋め込みに乗っている。
クラスタ＝視覚トークンがバッチ不変でないと語彙が染色条件で汚染されるため、
エンコーダに入れる前の画像段階で染色を揃える。

実装はMacenko et al. (2009) の標準手順。ポイントは2つ:

1. 染色ベクトルの推定はスライド単位で1回だけ行う（patchごとに推定しない）。
   patchごとに推定すると、組織が薄い・核が少ないpatchで推定が暴れる上に遅い。
2. 推定さえ済めば、正規化はOD空間の 3x3 線形写像に畳める（`macenko_matrix`）。
   patchごとの処理は行列積1回で済むので、抽出のボトルネックにならない。
"""

from __future__ import annotations

import numpy as np

# Macenko法で広く使われている参照染色ベクトル（列がH, E）と参照濃度。
# torchstain/staintools と同じ値。
STAIN_REF = np.array(
    [
        [0.5626, 0.2159],
        [0.7201, 0.8012],
        [0.4062, 0.5581],
    ]
)
MAXC_REF = np.array([1.9705, 1.0308])

IO = 240.0  # 透過光の基準強度


def optical_density(rgb: np.ndarray, io: float = IO) -> np.ndarray:
    """Convert RGB (..., 3) uint8 to optical density (N, 3) float."""
    # float32で十分（画素は8bit）。float64だとlog/expが倍かかり、
    # 300万patchを通すこの用途では無視できない差になる。
    flat = rgb.reshape(-1, 3).astype(np.float32)
    return -np.log((flat + 1.0) / np.float32(io))


def fit_macenko(
    rgb_samples: np.ndarray, *, io: float = IO, alpha: float = 1.0, beta: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate this slide's stain vectors and concentration scale.

    Args:
        rgb_samples: Pixels to fit on, any shape ending in 3 (uint8 RGB).
            Pass a sample of patches from one slide, not a single patch.
        alpha: Percentile used for the extreme stain angles. 1 means the
            1st and 99th percentile directions are taken as the two stains,
            which is what makes the estimate robust to a few odd pixels.
        beta: Optical-density floor. Pixels below it are nearly transparent
            (background glass) and carry no stain direction, so they are
            dropped before the SVD.

    Returns:
        (he, max_c): he is (3, 2) with the H and E stain vectors as columns;
        max_c is the 99th percentile concentration of each stain.
    """
    od = optical_density(rgb_samples, io)
    od_hat = od[~np.any(od < beta, axis=1)]
    if len(od_hat) < 100:
        raise ValueError(
            f"染色ベクトルを推定するには組織画素が足りません（{len(od_hat)}画素）。"
            "背景ばかりのpatchを引いていないか確認してください。"
        )

    # 共分散の上位2固有ベクトルが張る平面に射影し、その中での角度の両端を染色方向とする
    eigvals, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
    plane = eigvecs[:, [2, 1]]  # eighは昇順なので大きい方から2本

    # 固有ベクトルの符号は不定。第1軸が平均OD方向と逆を向くと、射影角が全体として
    # ±π 付近に来て arctan2 の切れ目をまたぐ。するとパーセンタイルが分布の両端では
    # なく切れ目の両側を拾い、染色ベクトル2本がほぼ平行に潰れる（max_cが発散する）。
    # 第1軸を平均OD方向に揃えて、角度が0付近に集まるようにしておく。
    mean_direction = od_hat.mean(axis=0)
    if plane[:, 0] @ mean_direction < 0:
        plane = plane * np.array([-1.0, 1.0])

    projected = od_hat @ plane
    angles = np.arctan2(projected[:, 1], projected[:, 0])

    lo, hi = np.percentile(angles, alpha), np.percentile(angles, 100 - alpha)
    v_lo = plane @ np.array([np.cos(lo), np.sin(lo)])
    v_hi = plane @ np.array([np.cos(hi), np.sin(hi)])

    # ヘマトキシリンは赤成分が小さい側。列の順序が入れ替わるとH/Eが逆転するので固定する。
    he = np.stack([v_lo, v_hi], axis=1) if v_lo[0] > v_hi[0] else np.stack([v_hi, v_lo], axis=1)

    # 2本がほぼ平行だと濃度分解が病的になる（lstsqの係数が発散し、正規化が壊れる）。
    # 静かに通すと後段の埋め込みが全部おかしくなるので、ここで落とす。
    cosine = float(abs(he[:, 0] @ he[:, 1]))
    if cosine > 0.99:
        raise ValueError(
            f"推定した染色ベクトル2本がほぼ平行です (cos={cosine:.4f})。"
            "サンプルしたpatchに核が少ないか、角度推定が破綻しています。"
        )

    concentrations = np.linalg.lstsq(he, od_hat.T, rcond=None)[0]
    max_c = np.percentile(concentrations, 99, axis=1)
    return he, max_c


def macenko_matrix(
    he: np.ndarray,
    max_c: np.ndarray,
    he_ref: np.ndarray | None = None,
    maxc_ref: np.ndarray | None = None,
) -> np.ndarray:
    """Collapse the whole normalization into one 3x3 map in OD space.

    Macenko normalization is: decompose OD into stain concentrations with this
    slide's stain matrix, rescale each concentration to the reference level,
    then recompose with the reference stain matrix. Every step is linear, so
    the composition is a single 3x3 matrix and each patch costs one matmul.

    Args:
        he, max_c: this slide's fit, from fit_macenko().
        he_ref, maxc_ref: the target to map onto. Defaults to the published
            reference (STAIN_REF / MAXC_REF), but prefer a target fitted on
            your own slides: the published concentrations are weaker than the
            TG-GATEs liver sections here, so normalizing to them washes the
            images out. Eosin intensity is itself part of what separates these
            lesions, so bleaching everything toward a paler reference throws
            away signal along with the batch difference.
    """
    he_ref = STAIN_REF if he_ref is None else he_ref
    maxc_ref = MAXC_REF if maxc_ref is None else maxc_ref
    scale = maxc_ref / max_c
    return he_ref @ np.diag(scale) @ np.linalg.pinv(he)


def apply_macenko(rgb: np.ndarray, matrix: np.ndarray, *, io: float = IO) -> np.ndarray:
    """Normalize an RGB patch (H, W, 3) uint8 with a precomputed 3x3 map."""
    shape = rgb.shape
    od = optical_density(rgb, io)
    od_norm = od @ matrix.T
    out = io * np.exp(-od_norm)
    return np.clip(out, 0, 255).reshape(shape).astype(np.uint8)
