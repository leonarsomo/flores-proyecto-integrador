#!/usr/bin/env bash
# Reproduce el proyecto de principio a fin (Actividades 1 y 2) en CPU.
# Uso:  bash reproducir.sh            todo el ciclo (varias horas en CPU de 2 hilos)
#       bash reproducir.sh --a1       solo datos, partición y líneas base (~35 min)
# Requiere: Python 3.13 y las dependencias de requirements.txt instaladas.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src HILOS="${HILOS:-2}"

echo "== 1. Datos =="
mkdir -p data
if [ ! -d data/flower_photos ]; then
  (cd data && curl -sSO https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz \
   && python -c "import hashlib,sys;h=hashlib.sha256(open('flower_photos.tgz','rb').read()).hexdigest();sys.exit(h!='4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c')" \
   && tar xzf flower_photos.tgz)
fi

echo "== 2. Auditoría, partición sin fuga y pruebas =="
python src/flores/auditoria.py data/flower_photos outputs
python src/flores/particion.py outputs
python -m pytest -q tests

echo "== 3. Líneas base (Actividad 1) =="
python scripts/ejecutar.py configs/e00_mayoritaria.yaml configs/e01_hog_color_lr.yaml \
       configs/e03_dinov2_sonda.yaml configs/e02_cnn_s11.yaml configs/e02_cnn_s22.yaml configs/e02_cnn_s33.yaml
python src/flores/diagnostico.py data/flower_photos outputs
python src/flores/seleccion.py outputs
python src/flores/tiempos.py data/flower_photos outputs
python scripts/figuras.py outputs
[ "${1:-}" = "--a1" ] && { echo "Actividad 1 reproducida."; exit 0; }

echo "== 4. Ajuste fino y contraste CNN (Actividad 2) =="
python -m flores.ajuste_fino cache data/flower_photos outputs --desde 8 --vistas 3
for e in e04_lpft_k2 e04b_lpft_k2_corto e09a_sgd e09b_decaimiento e09b2_decaimiento_alto e09c_drop_path; do
  python -m flores.experimentos_a2 estudio "configs/a2/$e.yaml"
done
python -m flores.cnn_contraste configs/a2/e10_cnn_effnet.yaml
python -m flores.cnn_contraste configs/a2/e10b_cnn_effnet_borde.yaml

echo "== 5. Finalista, validación cruzada y prueba única =="
python scripts/a2/elegir_finalista.py
python -m flores.experimentos_a2 cv configs/a2/cv_e03.yaml
python -m flores.experimentos_a2 cv configs/a2/cv_finalista.yaml
python -m flores.experimentos_a2 final configs/a2/final.yaml

echo "== 6. Análisis, figuras y fichas =="
python -m flores.analisis_a2
python scripts/a2/diagramas.py
python scripts/a2/rellenar_fichas.py
bash verificar.sh
echo "Ciclo completo reproducido. Compare outputs/a2/tablas con las tablas del informe."
