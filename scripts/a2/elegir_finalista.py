"""Regla de adopción del finalista (Google Developers, s. f.): se parte de E04b_01 y solo se adopta una
variante de E09 si reduce la pérdida de validación en más de 5 % relativo; si no, se conserva la
configuración más simple. Escribe la configuración elegida en configs/a2/cv_finalista.yaml y final.yaml."""
import json
import pandas as pd
import yaml
from pathlib import Path

A2 = Path("outputs/a2")
runs = pd.concat([pd.read_csv(f) for f in A2.glob("E0*_estudio.csv")], ignore_index=True)
base = runs.set_index("run").loc["E04b_01"]
cand = runs[runs.run.str.startswith("E09")].sort_values("val_loss")
elegido = base
if len(cand) and cand.iloc[0].val_loss < 0.95 * base.val_loss:
    elegido = cand.iloc[0]
claves = ["ajustar", "lora_r", "aumentacion", "optimizador", "lr", "decaimiento", "drop_path", "epocas"]
cfg = {k: (elegido[k].item() if hasattr(elegido[k], "item") else elegido[k]) for k in claves}
cfg["aumentacion"] = bool(cfg["aumentacion"])
nombre = elegido.name if elegido is base else elegido.run
for ruta, extra in (("configs/a2/cv_finalista.yaml", {"epocas": int(cfg["epocas"])}), ("configs/a2/final.yaml", {"semillas": [11, 22, 33]})):
    d = yaml.safe_load(Path(ruta).read_text())
    d["base"] = {**d["base"], **cfg}
    d.update(extra)
    d["origen"] = nombre
    Path(ruta).write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False))
(A2 / "tablas").mkdir(exist_ok=True)
(A2 / "tablas" / "eleccion_finalista.json").write_text(json.dumps(
    {"base": "E04b_01", "val_loss_base": float(base.val_loss), "mejor_E09": cand.iloc[0].run if len(cand) else None,
     "val_loss_mejor_E09": float(cand.iloc[0].val_loss) if len(cand) else None, "elegido": nombre,
     "regla": "adoptar variante solo si reduce la pérdida de validación más de 5 % relativo"}, indent=2))
print("finalista:", nombre, cfg)
