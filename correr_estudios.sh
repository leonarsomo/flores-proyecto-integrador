#!/bin/bash
# Cadena de estudios de la A2 (CPU, 2 hilos). Registro en outputs/a2/log_*.txt
export MLFLOW_DISABLE_AGENT_HINT=1 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONPATH=src
mkdir -p outputs/a2
for c in "$@"; do
  n=$(basename $c .yaml)
  echo "INICIO $n $(date +%H:%M)"
  python3 -m flores.experimentos_a2 estudio $c > outputs/a2/log_$n.txt 2>&1 || echo "FALLO $n"
  grep '^{"run"' outputs/a2/log_$n.txt | python3 -c "import sys,json;[print('RES',d['run'],'lr',d['lr'],'val_loss',round(d['val_loss'],4),'f1',round(d['val_macro_f1'],4),'ep',d['mejor_epoca'],'/',d['epocas_hechas'],'s',round(d['segundos_ft'])) for d in map(json.loads,sys.stdin)]"
  echo "FIN $n $(date +%H:%M)"
done
