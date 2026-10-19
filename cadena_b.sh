#!/bin/bash
export MLFLOW_DISABLE_AGENT_HINT=1 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONPATH=src
echo "INICIO E10b $(date +%H:%M)"
python3 -m flores.cnn_contraste configs/a2/e10b_cnn_effnet_borde.yaml > outputs/a2/log_e10b.txt 2>&1 || echo FALLO
grep '^{"run"' outputs/a2/log_e10b.txt | python3 -c "import sys,json;[print('RES',d['run'],d['lr'],d['epocas'],round(d['val_loss'],4),round(d['val_macro_f1'],4)) for d in map(json.loads,sys.stdin)]"
./correr_estudios.sh configs/a2/e04b_lpft_k2_corto.yaml configs/a2/e05_lpft_k4.yaml
