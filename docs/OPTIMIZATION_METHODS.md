# Optimization methods and supported transitions

| Pipeline | Backend/device | Actual transformation |
|---|---|---|
| fp16 | PyTorch CUDA | Half parameters, buffers and input |
| quantization | PyTorch CPU | Dynamic qint8 nn.Linear; other layers remain FP32 |
| onnx | ONNX Runtime CPU | Opset-17 export, dynamic batch, ORT_ENABLE_ALL |
| compile | Inductor CPU/CUDA | Lazy compilation, requiring platform compiler/Triton |
| fp16+compile | Inductor CUDA | Half conversion before compilation |
| onnx+quantization | ONNX Runtime CPU | Dynamic MatMul/Gemm INT8 after export |

ONNX INT8 uses per-channel weights and reduced range to avoid U8S8 saturation on
older x86 vector instructions. It does not quantize every convolution. Runtime
operation support and numerical policy are still checked; smaller weights do not
imply acceptance. Calibration-based static CNN quantization is not implemented.
See [ORT quantization documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

The compatibility table checks device, precision transitions, optional imports,
eligible Linear modules and compiler prerequisites before copying the source
model. Operator/export support is finally established by real execution, not a
claim of universal static analysis. Unsupported compositions (for example
ONNX→compile, INT8→FP16) have no adapter and are rejected. CLI choices expose only
implemented chains. Programmatic preflight returns an explicit incompatible
exception. Reports use `status=skipped`, `status_detail=SKIPPED — INCOMPATIBLE`.
Actual export/compile/execution exceptions produce `failed` with their reason.

The original CPU model is preserved; one independent model is owned by each
candidate. Dynamic quantization uses in-place conversion only on that owned copy.
The public quantize function still defaults to preserving caller-owned models.

`torch.compile` is not proof of fusion. `fusion_confirmed` stays false without
compiler evidence. Custom Triton RMSNorm/LayerNorm/softmax/activation kernels are
not delivered: no isolated bottleneck and standard-library comparison has
justified them, and compatible Triton is absent on the inspected Windows host.
Ordinary inference recomputation was not introduced. Detection/transformer task
adapters remain future work rather than empty classes.

## Export and reload

```bash
python -m kernelixx export --run-dir results/pipeline/latest --output results/exported-model
```

Exports the selected result, which may correctly be baseline. ONNX is a saved
graph; other PyTorch paths save an original FP32 state_dict plus an explicit
execution recipe that reapplies the transforms. Compiled code is recreated at
startup, not advertised as a portable binary. Model factories remain trusted
local Python and must be available when reloading. State loading uses
`weights_only=True`, strict keys, and a source tensor hash.

The manifest stores config, transforms, versions, hardware, original report,
checksums and reload validation. Export succeeds only after a new runner reloads
the artifact and runs the complete validation stream. Existing destinations,
changed weights/datasets, checksum errors or validation failure prevent success.
Failed artifacts retain a manifest marked unverified for inspection. ONNX
quantized recipes record actual reduced-range settings. The Python reload entry
point is `kernelixx.core.export.reload_artifact(directory)`.

New reports also fingerprint the Kernelixx Python package and direct factory/model
class source modules. Export rejects changed recorded source, even if weights are
identical; PyTorch recipe reload checks it again. Source-file hashes do not prove
arbitrary dependency/native/runtime-patch equivalence. Legacy reports remain usable
with `source_code_validation.status=NOT VERIFIED` for historical code provenance.
See the [supplied SoupLite archive follow-up](SOUPLITE_REFERENCE_AUDIT.md).

GGUF export saves a hashed local-weight runtime recipe and prompts, then creates
a fresh native process and verifies outputs against the source run. Weights are
not copied or transformed. The first validated configuration is exported; no
unconfirmed faster configuration is silently promoted.
