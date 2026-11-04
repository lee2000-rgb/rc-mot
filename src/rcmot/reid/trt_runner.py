"""Minimal TensorRT runner for the ReID network (docs/18): the OSNet x0.25 forward costs 32 ms in PyTorch on the
Orin (launch-bound, independent of batch / precision) and ~1-2 ms as a TensorRT engine.

Engine: trtexec --onnx=osnet.onnx --saveEngine=osnet.engine --fp16 --minShapes=input:1x3x256x128
        --optShapes=input:4x3x256x128 --maxShapes=input:16x3x256x128
Inputs / outputs are torch CUDA tensors (device pointers handed to TensorRT), so no extra copies.
"""
from __future__ import annotations

import numpy as np
import torch


class TrtRunner:
    def __init__(self, engine_path: str):
        import tensorrt as trt

        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inp = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT][0]
        self.out = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT][0]
        self.in_dtype = torch.float16 if self.engine.get_tensor_dtype(self.inp) == trt.DataType.HALF else torch.float32
        self.out_dtype = torch.float16 if self.engine.get_tensor_dtype(self.out) == trt.DataType.HALF else torch.float32
        self.max_batch = self.engine.get_tensor_profile_shape(self.inp, 0)[2][0]
        self.stream = torch.cuda.Stream()

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N,3,H,W) CUDA tensor -> (N,F) float32 CUDA tensor (N <= max batch, else chunked)."""
        if len(x) > self.max_batch:
            return torch.cat([self(x[i:i + self.max_batch]) for i in range(0, len(x), self.max_batch)])
        x = x.to(self.in_dtype).contiguous()
        self.ctx.set_input_shape(self.inp, tuple(x.shape))
        out_shape = tuple(self.ctx.get_tensor_shape(self.out))
        y = torch.empty(out_shape, dtype=self.out_dtype, device=x.device)
        self.ctx.set_tensor_address(self.inp, x.data_ptr())
        self.ctx.set_tensor_address(self.out, y.data_ptr())
        with torch.cuda.stream(self.stream):
            ok = self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        if not ok:
            raise RuntimeError("TensorRT execute failed")
        return y.float()
