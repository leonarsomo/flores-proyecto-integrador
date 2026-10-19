#!/usr/bin/env bash
# Verificación rápida (unos 2 min en CPU): integridad de los pesos publicados, pruebas e inferencia.
# Uso:  bash verificar.sh [foto1.jpg foto2.jpg ...]
# Sin fotos, usa una por clase de data/flower_photos si existe.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src

echo "== Integridad de modelos/finalista.pt =="
python - <<'PY'
import hashlib, json
meta = json.load(open("modelos/finalista.json"))
real = hashlib.sha256(open("modelos/finalista.pt", "rb").read()).hexdigest()
assert real == meta["sha256"], f"SHA-256 distinto: {real}"
print("SHA-256 correcto:", real)
print("Temperatura:", round(meta["temperatura"], 4), "| umbral de abstención:", meta["umbral"])
PY

echo "== Pruebas unitarias =="
python -m pytest -q tests

echo "== Inferencia con el modelo publicado =="
fotos=("$@")
if [ ${#fotos[@]} -eq 0 ] && [ -d data/flower_photos ]; then
  for c in daisy dandelion roses sunflowers tulips; do f=(data/flower_photos/$c/*.jpg); fotos+=("${f[0]}"); done
fi
if [ ${#fotos[@]} -gt 0 ]; then
  python -m flores.predecir modelos/finalista.pt "${fotos[@]}"
else
  echo "Sin fotos: pase rutas de imágenes o descargue los datos con reproducir.sh --a1."
fi
echo "Verificación terminada."
