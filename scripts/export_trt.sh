#!/usr/bin/env bash
# Export a YOLO .pt to a TensorRT engine. RUN ON THE JETSON (engines are not portable across GPUs / TRT versions).
#   bash scripts/export_trt.sh models/yolo11s.pt 960 fp16      -> models/yolo11s.engine
#   bash scripts/export_trt.sh models/yolo11s.pt 960 int8      (needs a calibration yaml: data=...)
set -euo pipefail
W=${1:-models/yolo11s.pt}; IMGSZ=${2:-960}; PREC=${3:-fp16}
case $PREC in
  fp16) python3 -c "from ultralytics import YOLO; YOLO('$W').export(format='engine', imgsz=$IMGSZ, half=True, device=0, workspace=4, dynamic=True, batch=${BATCH:-4})";;   # dynamic batch: RGR detects K crops in one call
  int8) python3 -c "from ultralytics import YOLO; YOLO('$W').export(format='engine', imgsz=$IMGSZ, int8=True, data='${DATA:-coco8.yaml}', device=0)";;
  *) echo "precision: fp16 | int8"; exit 1;;
esac
# Orin power mode for benchmarking:  sudo nvpmodel -m 0 && sudo jetson_clocks
