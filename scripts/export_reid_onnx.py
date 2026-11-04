"""OSNet ReID .pth -> ONNX (dynamic batch) for scripts/trt_build_onnx.py (docs/18)."""
import sys, torch; sys.path.insert(0,'src')
from rcmot.reid.extractor import ReID
w_in, w_out = sys.argv[1], sys.argv[2]
r=ReID(w_in, device='cpu')
x=torch.randn(1,3,256,128)
torch.onnx.export(r.model, x, w_out, input_names=['input'], output_names=['feat'],
                  dynamic_axes={'input':{0:'batch'},'feat':{0:'batch'}}, opset_version=17, dynamo=False)
print('ONNX-OK')
