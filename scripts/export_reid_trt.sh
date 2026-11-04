#!/usr/bin/env bash
# OSNet ReID -> ONNX -> TensorRT FP16 engine (run ON the Jetson; docs/18: 42 ms -> 10 ms for 4 crops).
#   bash scripts/export_reid_trt.sh models/reid/osnet_x0_25_msmt17.pth
set -euo pipefail
W=${1:-models/reid/osnet_x0_25_msmt17.pth}
python3 scripts/export_reid_onnx.py "$W" "${W%.pth}.onnx"
python3 scripts/trt_build_onnx.py "${W%.pth}.onnx" "${W%.pth}.engine"
