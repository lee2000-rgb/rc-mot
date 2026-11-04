"""Build a TensorRT FP16 engine from an ONNX file with a dynamic batch (python API; trtexec is not installed)."""
import sys, tensorrt as trt
onnx, out = sys.argv[1], sys.argv[2]
logger = trt.Logger(trt.Logger.WARNING)
b = trt.Builder(logger); net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
p = trt.OnnxParser(net, logger)
assert p.parse(open(onnx, 'rb').read()), [p.get_error(i) for i in range(p.num_errors)]
cfg = b.create_builder_config(); cfg.set_flag(trt.BuilderFlag.FP16)
cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
prof = b.create_optimization_profile(); name = net.get_input(0).name
prof.set_shape(name, (1, 3, 256, 128), (4, 3, 256, 128), (16, 3, 256, 128)); cfg.add_optimization_profile(prof)
eng = b.build_serialized_network(net, cfg); assert eng is not None
open(out, "wb").write(eng); print("ENGINE-OK", out)
