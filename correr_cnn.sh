#!/bin/bash
# E10 (CNN EfficientNet-B0) tras esperar a que termine la cadena anterior
export MLFLOW_DISABLE_AGENT_HINT=1 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONPATH=src
while pgrep -f "flores.experimentos_a2" >/dev/null; do sleep 20; done
echo "INICIO E10 $(date +%H:%M)"
python3 -m flores.cnn_contraste configs/a2/e10_cnn_effnet.yaml > outputs/a2/log_e10_cnn_effnet.txt 2>&1 || echo FALLO
grep '^{"run"' outputs/a2/log_e10_cnn_effnet.txt | cut -c1-300
echo "FIN E10 $(date +%H:%M)"
