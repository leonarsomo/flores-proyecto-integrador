"""Pruebas de la partición: sin fuga por grupos, proporciones y exclusiones."""
from pathlib import Path

import pandas as pd

P = pd.read_csv(Path(__file__).resolve().parents[1] / "outputs" / "particion.csv")


def test_grupos_no_cruzan_particiones():
    assert (P.groupby("grupo").split.nunique() == 1).all()


def test_autores_no_cruzan_particiones():
    assert (P.groupby("owner").split.nunique() == 1).all()


def test_pixeles_identicos_en_una_sola_particion():
    assert (P.groupby("md5_pixels").split.nunique() == 1).all()


def test_proporciones_y_estratos():
    frac = P.split.value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.02 and abs(frac["test"] - 0.15) < 0.02
    tab = pd.crosstab(P.label, P.split, normalize="columns")
    assert (tab.max(axis=1) - tab.min(axis=1)).max() < 0.02   # misma mezcla de clases


def test_sin_imagenes_excluidas():
    assert P.exclusion.isna().all() or (P.exclusion.fillna("") == "").all()
    assert P.owner.notna().all()
