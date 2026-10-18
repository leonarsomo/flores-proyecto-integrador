"""Grupos de fuga y partición estratificada por grupos de tf_flowers.

Grupo de fuga = componente conexa del grafo cuyas aristas unen dos imágenes que
comparten autor de Flickr, píxeles idénticos o pHash a distancia de Hamming <= 6.
Uso: python code/b_particion.py out
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

SEED = 2026
PHASH_MAX_BITS = 6          # umbral de casi duplicado (U2A4)
N_FOLDS = 20                # 3 pliegues prueba, 3 validación, 14 entrenamiento
TEST_FOLDS, VAL_FOLDS = {0, 1, 2}, {3, 4, 5}


def phash_bits(hex_hashes: pd.Series) -> np.ndarray:
    """Convierte hashes hexadecimales de 64 bits en una matriz booleana (n, 64)."""
    ints = np.array([int(h, 16) for h in hex_hashes], dtype=np.uint64)
    return ((ints[:, None] >> np.arange(64, dtype=np.uint64)) & np.uint64(1)).astype(bool)


def near_duplicate_pairs(bits: np.ndarray, max_bits: int) -> np.ndarray:
    """Pares (i, j), i < j, con distancia de Hamming <= max_bits."""
    b = bits.astype(np.int16)
    # Hamming = unos_i + unos_j - 2 * coincidencias de unos
    ones = b.sum(1)
    dist = ones[:, None] + ones[None, :] - 2 * (b @ b.T)
    i, j = np.where(np.triu(dist <= max_bits, k=1))
    return np.stack([i, j], 1)


def leakage_groups(inv: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Componentes conexas por autor, píxeles idénticos y casi duplicados."""
    n = len(inv)
    edges = [near_duplicate_pairs(phash_bits(inv["phash"]), PHASH_MAX_BITS)]
    for key in ("owner", "md5_pixels"):
        codes = inv[key].fillna(pd.Series(inv.index.astype(str), index=inv.index))
        for _, idx in inv.groupby(codes).indices.items():
            if len(idx) > 1:
                edges.append(np.stack([np.repeat(idx[0], len(idx) - 1), idx[1:]], 1))
    e = np.concatenate(edges)
    adj = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, comp = connected_components(adj, directed=False)
    pairs = pd.DataFrame(edges[0], columns=["i", "j"])
    return comp, pairs


def contamination(inv: pd.DataFrame, split: pd.Series, pairs: pd.DataFrame) -> dict:
    """Imágenes de validación/prueba con autor o casi duplicado presente en entrenamiento."""
    res = {}
    train_owners = set(inv.loc[split == "train", "owner"].dropna())
    for part in ("val", "test"):
        mask = split == part
        same_owner = inv.loc[mask, "owner"].isin(train_owners).sum()
        s = split.to_numpy()
        cross = ((s[pairs.i] == part) & (s[pairs.j] == "train")) | (
            (s[pairs.j] == part) & (s[pairs.i] == "train"))
        idx = np.unique(np.where(s[pairs.i] == part, pairs.i, pairs.j)[cross])
        res[part] = {"n": int(mask.sum()), "autor_en_train": int(same_owner),
                     "casi_duplicado_en_train": int(len(idx))}
    return res


def main(out: Path) -> None:
    inv = pd.read_csv(out / "inventario.csv")
    # Exclusiones documentadas: sin licencia verificable o etiqueta contradictoria.
    conflict = inv.groupby("md5_pixels").label.transform("nunique") > 1
    inv["exclusion"] = np.select(
        [inv.owner.isna(), conflict], ["sin_licencia", "etiqueta_contradictoria"], "")
    comp, pairs_all = leakage_groups(inv)
    inv["grupo"] = comp
    pairs_all.to_csv(out / "pares_casi_duplicados.csv", index=False)

    kept = inv[inv.exclusion == ""].reset_index(drop=True)
    _, pairs = leakage_groups(kept)          # pares sobre el conjunto depurado
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = np.empty(len(kept), int)
    for k, (_, te) in enumerate(sgkf.split(kept, kept.label, kept.grupo)):
        fold[te] = k
    kept["split"] = np.where(np.isin(fold, list(TEST_FOLDS)), "test",
                             np.where(np.isin(fold, list(VAL_FOLDS)), "val", "train"))

    # Contraste: partición aleatoria estratificada ingenua con las mismas proporciones.
    tr, rest = train_test_split(kept.index, test_size=0.30, stratify=kept.label,
                                random_state=SEED)
    va, te = train_test_split(rest, test_size=0.50, stratify=kept.label[rest],
                              random_state=SEED)
    naive = pd.Series("train", index=kept.index)
    naive[va], naive[te] = "val", "test"

    report = {
        "n_total": len(inv),
        "exclusiones": inv.exclusion.value_counts().to_dict(),
        "n_usadas": len(kept),
        "grupos": int(kept.grupo.nunique()),
        "tam_grupo_max": int(kept.groupby("grupo").size().max()),
        "pares_casi_duplicados": int(len(pairs)),
        "imagenes_en_pares": int(len(np.unique(pairs.to_numpy()))),
        "conteos": pd.crosstab(kept.label, kept.split).to_dict(),
        "contaminacion_por_grupos": contamination(kept, kept.split, pairs),
        "contaminacion_aleatoria": contamination(kept, naive, pairs),
        "grupos_compartidos_entre_particiones": int(
            (kept.groupby("grupo").split.nunique() > 1).sum()),
    }
    kept.to_csv(out / "particion.csv", index=False)
    inv.to_csv(out / "inventario_grupos.csv", index=False)
    (out / "particion_reporte.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
