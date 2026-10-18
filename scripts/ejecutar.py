"""Punto de entrada: python scripts/ejecutar.py configs/e01_hog_color_lr.yaml [...]"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flores.experimentos import REGISTRO, Datos  # noqa: E402

comun = yaml.safe_load(Path("configs/comun.yaml").read_text())
datos = Datos(Path(comun["raiz_datos"]), Path(comun["particion"]))
for ruta in sys.argv[1:]:
    cfg = {**comun, **yaml.safe_load(Path(ruta).read_text())}
    res = REGISTRO[cfg["familia"]](cfg, datos, Path(cfg["salida"])).ejecutar()
    t = res["test"]
    print(f"{cfg['id']}: macro-F1 prueba {t['macro_f1']:.4f} "
          f"IC95 {t['macro_f1_ic95'][0]:.3f}-{t['macro_f1_ic95'][1]:.3f} | "
          f"acc {t['accuracy']:.4f} | ECE {t['ece']:.4f}", flush=True)
