#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_and_quantize_onnx.py
===========================
爱弥斯 (Aemeath) 端侧语音模型「ONNX 导出 → Int8/FP16 混合量化 → 情绪特征固化 →
魔搭清单生成」全流程工具。

核心工程铁律（防电音）
----------------------
* **前端语义大胆压 Int8**：GPT-SoVITS v2 的 ``t2s_encoder.onnx`` /
  ``t2s_fsdec.onnx`` / ``t2s_sdec.onnx``（以及 Bert-VITS2 轻量版的文本流
  ``text_encoder.onnx``）统一走 ``onnxruntime.quantization.quantize_dynamic``
  → ``QUInt8`` 动态权重量化。自回归矩阵小、对单点误差不敏感，体积可砍 70%+。
* **后端声码器严守 FP16**：``vits_vocoder.onnx`` 用
  ``onnxconverter_common.float16.convert_float_to_float16(..., keep_io_types=True)``
  转半精度，且保持输入输出接口为 Float32。声码器最后几层是逐样本波形反卷积，
  Int8 权重会把量化噪声直接刻进波形 → 金属电流声/电音，必须 FP16。
* **数值敏感算子 FP32 白名单**：Exp / Log / Div / Sqrt / Softmax / ReduceMean
  等算子在 FP16 下极易溢出或丢精度，转换时列入 ``op_block_list`` 保持 FP32，
  仅让大矩阵 GEMM / Conv 走半精度。
* **采样永不退化**：清单与原生层统一使用 temperature=0.85、top_k=15、
  top_p=0.90、repetition_penalty=1.35 的概率核采样，导出阶段不做任何
  argmax / 贪心近似（见 ``sampling_params``）。

子命令
------
``export``    调用 GPT-SoVITS 官方导出脚本产出 FP32 ONNX（或校验已导出模型）。
``quantize``  批量执行 Int8 动态量化 / FP16 半精度转换，并写 ``quant_report.json``。
``emotions``  从 4 条代表性参考原声提取固化 256 维情绪向量（``平静/开心/温柔/调侃.bin``），
              手机端因此无需运行 CNHubert/Wav2Vec2 前处理大模型。
``manifest``  生成上传魔搭社区（ModelScope）的 ``manifest.json``（文件列表/大小/SHA256）。
``verify``    ONNX Runtime 实测加载 + 形状推断 + 与 FP32 数值一致性比对，
              并输出 ``model_io.json`` 供 Android 原生层按名绑定张量。
``all``       一键串联 quantize → emotions → manifest → verify（含 export 可选）。

典型用法
--------
::

    # 1) 从 GPT-SoVITS 仓库权重导出 FP32 ONNX（自动调用官方 onnx_export.py）
    python scripts/export_and_quantize_onnx.py export \
        --gpt-sovits-repo /root/GPT-SoVITS \
        --ckpt-dir /root/GPT-SoVITS/GPT_SoVITS/pretrained_models/aemeath \
        --out-dir /root/autodl-tmp/aemeath_pro/fp32

    # 2) 混合量化 + 情绪向量固化 + 魔搭清单（一条命令出包）
    python scripts/export_and_quantize_onnx.py all \
        --fp32-dir /root/autodl-tmp/aemeath_pro/fp32 \
        --out-dir /root/autodl-tmp/aemeath_pro/onnx \
        --reference-dir /root/autodl-tmp/aemeath_ref \
        --arch "GPT-SoVITS v2" --sample-rate 32000

    # 3) 只做量化（已有 FP32 ONNX，跳过导出）
    python scripts/export_and_quantize_onnx.py quantize \
        --fp32-dir fp32 --out-dir onnx --print-plan

    # 4) 单独重跑情绪向量（例如换了参考音）
    python scripts/export_and_quantize_onnx.py emotions \
        --reference-dir refs --out-dir onnx/emotions

依赖
----
``onnx``、``onnxruntime``、``onnxconverter-common``、``numpy`` 为必需；
``soundfile``/``scipy`` 用于读参考音；``torch``+``transformers`` 仅在需要
CNHubert 真特征时使用（缺失时自动降级为谱域确定性嵌入，仍严格产出 1024 字节
的 256 维 float32 小端 .bin）。

作者：DeepSeek Harness（端侧 AI 推理引擎与云端训练量化工程师）
许可：AGPL-3.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

LOG = logging.getLogger("aemeath.onnx")

# --------------------------------------------------------------------------- #
# 一、全局常量与契约
# --------------------------------------------------------------------------- #
TOOL_VERSION = "1.0.0"

EMOTIONS: Tuple[str, ...] = ("平静", "开心", "温柔", "调侃")
EMOTION_DIM = 256
EMOTION_BYTES = EMOTION_DIM * 4  # float32 小端 → 严格 1024 字节

#: 与 ARCHITECTURE.md / 原生 AemeathTtsPlugin 保持一致的采样契约。
#: top_k=15 为端侧出厂默认；前端可通过 mountModel.samplingConfig 覆盖（如 25）。
SAMPLING_PARAMS: Dict[str, Any] = {
    "temperature": 0.85,
    "top_k": 15,
    "top_p": 0.90,
    "repetition_penalty": 1.35,
    "stop_token": 1024,
    "max_semantic_length": 512,
    "sampling_method": "multinomial_nucleus",
    "greedy_fallback": False,
}

#: 走 Int8 动态量化的模块（前端语义 / 文本流 / 编码器）。
DEFAULT_INT8_PATTERNS: Tuple[str, ...] = (
    r"t2s_encoder",
    r"t2s_fsdec",
    r"t2s_sdec",
    r"t2s\.int8",
    r"^t2s$",
    r"s1bert",
    r"text_encoder",
    r"text_dur",
    r"bert",
    r"cnhubert",
    r"hubert",
    r"phone_embed",
    r"speaker_encoder",
)

#: 走 FP16 半精度的模块（声码器 / 波形反卷积）。
DEFAULT_FP16_PATTERNS: Tuple[str, ...] = (
    r"vits_vocoder",
    r"vocoder",
    r"^vits",
    r"s2g",
    r"s2d",
    r"decoder_flow",
    r"flow",
    r"generator",
)

#: FP16 转换时保持 FP32 的数值敏感算子（防溢出 / 防金属电流声）。
DEFAULT_FP16_BLOCK_OPS: Tuple[str, ...] = (
    "Div",
    "Exp",
    "Log",
    "Pow",
    "Reciprocal",
    "Sqrt",
    "Softmax",
    "ReduceMean",
    "ReduceMax",
    "ReduceMin",
    "Clip",
    "Erf",
    "Tanh",
    "Sigmoid",
    "LayerNormalization",
    "MeanVarianceNormalization",
    "InstanceNormalization",
)

#: 量化产物命名后缀，用于跳过已处理文件。
QUANT_SUFFIXES: Tuple[str, ...] = (".int8.onnx", ".fp16.onnx", ".quant.onnx")


class PipelineError(RuntimeError):
    """流水线致命错误。"""


# --------------------------------------------------------------------------- #
# 二、通用工具
# --------------------------------------------------------------------------- #
def setup_logging(level: str) -> None:
    LOG.setLevel(getattr(logging, level.upper(), logging.INFO))
    LOG.handlers.clear()
    LOG.propagate = False  # 避免 onnxruntime 内部 basicConfig 造成日志重复
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    )
    LOG.addHandler(handler)


def human_bytes(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0 or unit == "TB":
            return f"{num:.1f} {unit}" if unit != "B" else f"{int(num)} B"
        num /= 1024.0
    return f"{num:.1f} TB"


def sha256_of_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def relative_posix(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.name


def _lazy_import_onnx() -> Any:
    try:
        import onnx  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise PipelineError("缺少 onnx 依赖：pip install onnx onnxconverter-common") from exc
    return onnx


def _lazy_import_ort() -> Any:
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise PipelineError("缺少 onnxruntime 依赖：pip install onnxruntime") from exc
    try:
        # 3 = ERROR，压掉 ORT 形状推断的刷屏告警（不影响真实报错）
        ort.set_default_logger_severity(3)
    except Exception:
        pass
    return ort


# --------------------------------------------------------------------------- #
# 三、模型角色判定与量化计划
# --------------------------------------------------------------------------- #
def _matches_any(name: str, patterns: Sequence[str]) -> Optional[str]:
    lowered = name.lower()
    for pattern in patterns:
        try:
            if re.search(pattern, lowered):
                return pattern
        except re.error:
            if pattern.lower() in lowered:
                return pattern
    return None


@dataclass
class QuantPlanItem:
    """单个 ONNX 文件的目标精度计划。"""

    source: str
    role: str
    precision: str          # "int8_dynamic" | "fp16"
    matched_pattern: Optional[str] = None
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_quant_plan(
    files: Sequence[Path],
    int8_patterns: Sequence[str],
    fp16_patterns: Sequence[str],
    default_precision: str = "fp16",
) -> List[QuantPlanItem]:
    """按文件名规则判定每个模块走 Int8 还是 FP16。

    优先级：显式 FP16 规则（声码器）> 显式 Int8 规则（语义前端）> 兜底策略。
    声码器优先，确保任何一个既含 ``vits`` 又含 ``encoder`` 的命名都不会被误压。
    """
    plan: List[QuantPlanItem] = []
    for path in files:
        name = path.name.lower()
        fp16_hit = _matches_any(name, fp16_patterns)
        int8_hit = _matches_any(name, int8_patterns)

        if fp16_hit is not None:
            plan.append(
                QuantPlanItem(
                    source=str(path),
                    role=_guess_role(name),
                    precision="fp16",
                    matched_pattern=fp16_hit,
                    reason="声码器/波形生成模块，FP16 半精度以杜绝电音",
                )
            )
        elif int8_hit is not None:
            plan.append(
                QuantPlanItem(
                    source=str(path),
                    role=_guess_role(name),
                    precision="int8_dynamic",
                    matched_pattern=int8_hit,
                    reason="语义/文本流前端，QUInt8 动态权重量化",
                )
            )
        else:
            plan.append(
                QuantPlanItem(
                    source=str(path),
                    role=_guess_role(name),
                    precision=default_precision,
                    matched_pattern=None,
                    reason=f"未命中规则，按兜底精度 {default_precision} 处理",
                )
            )
    return plan


def _guess_role(name: str) -> str:
    lowered = name.lower()
    if "t2s_encoder" in lowered or "text_encoder" in lowered:
        return "semantic_encoder"
    if "t2s_fsdec" in lowered:
        return "semantic_decoder_first"
    if "t2s_sdec" in lowered:
        return "semantic_decoder_second"
    if "vocoder" in lowered or "vits" in lowered:
        return "vocoder"
    if "bert" in lowered:
        return "bert_feature"
    if "hubert" in lowered:
        return "ssl_feature"
    return "auxiliary"


def output_name_for(source: Path, precision: str) -> str:
    suffix = ".int8.onnx" if precision == "int8_dynamic" else ".fp16.onnx"
    return source.stem + suffix


# --------------------------------------------------------------------------- #
# 四、量化实现（Int8 动态 / FP16 半精度）
# --------------------------------------------------------------------------- #
@dataclass
class QuantConfig:
    weight_type: str = "QUInt8"          # QUInt8 | QInt8
    per_channel: bool = True
    reduce_range: bool = False
    extra_options: Dict[str, Any] = field(default_factory=lambda: {"MatMulConstBOnly": True})
    op_types_to_quantize: Optional[List[str]] = None
    nodes_to_exclude: List[str] = field(default_factory=list)
    # FP16 相关
    keep_io_types: bool = True
    fp16_block_ops: Tuple[str, ...] = DEFAULT_FP16_BLOCK_OPS
    # 形状推断必须开启：blocklist 场景下转换器依赖形状信息插入 Cast，
    # 关闭后会出现 Div/Mul 输入 float16 与 float32 混绑的非法图。
    fp16_shapes_infer: bool = True


class Quantizer:
    """封装 ONNX Runtime 动态量化与 onnxconverter-common 半精度转换。"""

    def __init__(self, cfg: QuantConfig) -> None:
        self.cfg = cfg

    # -- Int8 动态量化 ------------------------------------------------------ #
    def quantize_dynamic_int8(self, source: Path, target: Path) -> Dict[str, Any]:
        """QUInt8 动态权重量化（无需校准数据，激活值运行时动态量化）。

        **中文/非 ASCII 路径兼容**：`onnxruntime.quantization` 内部会写出
        `<model>-inferred.onnx` 中间文件，其上层的 C++ 路径处理在 Windows 遇到
        非 ASCII 目录时会直接抛 `No such file or directory`（实测
        `D:\\爱弥斯\\...` 必然失败）。因此先把源模型复制到 ASCII 临时目录、
        在临时目录里量化，再把结果搬回目标路径，并清理中间残留文件。
        """
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise PipelineError("缺少 onnxruntime.quantization，请安装 onnxruntime") from exc

        weight_type = QuantType.QUInt8 if self.cfg.weight_type.upper() == "QUINT8" else QuantType.QInt8
        ensure_dir(target.parent)

        kwargs: Dict[str, Any] = {
            "weight_type": weight_type,
            "per_channel": self.cfg.per_channel,
            "reduce_range": self.cfg.reduce_range,
            "extra_options": dict(self.cfg.extra_options),
        }
        if self.cfg.op_types_to_quantize:
            kwargs["op_types_to_quantize"] = list(self.cfg.op_types_to_quantize)
        if self.cfg.nodes_to_exclude:
            kwargs["nodes_to_exclude"] = list(self.cfg.nodes_to_exclude)

        LOG.info("  ├─ Int8 动态量化：%s -> %s", source.name, target.name)

        import shutil
        import tempfile

        needs_sandbox = not str(source.resolve()).isascii() or not str(target.resolve()).isascii()
        sandbox: Optional[Path] = None
        try:
            if needs_sandbox:
                sandbox = Path(tempfile.mkdtemp(prefix="aemeath_quant_"))
                local_source = sandbox / "model.onnx"
                local_target = sandbox / "model.int8.onnx"
                shutil.copy2(source, local_source)
                LOG.debug("  ├─ 检测到非 ASCII 路径，改在临时目录量化：%s", sandbox)
            else:
                local_source = source
                local_target = target

            quantize_dynamic(str(local_source), str(local_target), **kwargs)

            if sandbox is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(local_target), str(target))
        finally:
            if sandbox is not None:
                shutil.rmtree(sandbox, ignore_errors=True)

        # 清理与源模型同目录的 -inferred.onnx 中间文件
        inferred_sibling = source.with_name(source.stem + "-inferred.onnx")
        if inferred_sibling.exists() and inferred_sibling != target:
            try:
                inferred_sibling.unlink()
            except OSError:
                pass

        return {"precision": "int8_dynamic", "weight_type": self.cfg.weight_type}

    # -- FP16 半精度 -------------------------------------------------------- #
    def convert_fp16(self, source: Path, target: Path) -> Dict[str, Any]:
        """FP32 -> FP16，``keep_io_types=True`` 保证对外仍收/发 Float32。

        加固策略：``op_block_list``（数值敏感算子保持 FP32）在不同转换器上
        表现不一致——``onnxconverter_common`` 对部分图会生成非法节点定义
        （Duplicate definition / Div 类型不一致）。因此这里按「先规范实现、
        后工程加固」的顺序逐级尝试，每一步都用 ONNX Runtime 真实加载验证，
        只有能跑通的产物才会落到目标路径：

        1. ``onnxconverter_common.float16`` + blocklist（任务书指定 API）；
        2. ``onnxruntime.transformers.float16`` + blocklist（cast 插入更可靠）；
        3. ``onnxconverter_common`` 纯 FP16（无 blocklist，最后一档兜底）。
        """
        block_ops = list(self.cfg.fp16_block_ops)
        attempts: List[Tuple[str, str, List[str]]] = [
            ("onnxconverter_common+blocklist", "occ", block_ops),
            ("onnxruntime.transformers+blocklist", "ort", block_ops),
            ("onnxconverter_common(no-blocklist)", "occ", []),
        ]
        errors: List[str] = []
        ensure_dir(target.parent)
        LOG.info("  ├─ FP16 半精度转换：%s -> %s", source.name, target.name)

        # onnxconverter_common 会为每个越界权重各 warn 一次（大模型动辄数百条），
        # 而这些截断正是 FP16 转换的预期行为，安静掉以免淹没真正的错误日志。
        import warnings

        warnings.filterwarnings("ignore", message=r".*will be truncated.*")

        for label, backend, blocked in attempts:
            try:
                model = _lazy_import_onnx().load(str(source))
                converted = self._run_fp16_conversion(model, backend, blocked)
                converted = toposort_model(converted)
                _lazy_import_onnx().save(converted, str(target))
                self._assert_fp16_loadable(target)
                io_types = _io_dtype_summary(converted)
                if self.cfg.keep_io_types and any(v == "float16" for v in io_types.values()):
                    raise PipelineError("I/O 被转成 float16，前端 Float32 契约将被破坏")
                LOG.info("  ├─ FP16 转换器：%s（保持 FP32 的算子 %d 类）", label, len(blocked))
                return {
                    "precision": "fp16",
                    "converter": label,
                    "keep_io_types": self.cfg.keep_io_types,
                    "blocked_ops": blocked,
                    "io_types": io_types,
                    "failed_attempts": errors,
                }
            except Exception as exc:
                errors.append(f"{label}: {exc}")
                LOG.debug("  ├─ FP16 尝试「%s」未通过：%s", label, str(exc)[:180])

        # 全部尝试失败：删除可能残留的半成品，绝不把坏模型留在产物目录
        try:
            if target.exists():
                target.unlink()
        except Exception:
            pass
        raise PipelineError("FP16 转换全部失败：" + " || ".join(errors))

    def _run_fp16_conversion(self, model: Any, backend: str, blocked: Sequence[str]) -> Any:
        keep_io = self.cfg.keep_io_types
        disable_shape_infer = not self.cfg.fp16_shapes_infer
        if backend == "occ":
            try:
                from onnxconverter_common import float16 as occ_float16  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise PipelineError("缺少 onnxconverter-common") from exc
            return occ_float16.convert_float_to_float16(
                model,
                keep_io_types=keep_io,
                disable_shape_infer=disable_shape_infer,
                op_block_list=list(blocked) or None,
            )

        try:
            from onnxruntime.transformers import float16 as ort_float16  # type: ignore
        except Exception as exc:
            raise PipelineError("当前 onnxruntime 未内置 transformers.float16 模块") from exc
        return ort_float16.convert_float_to_float16(
            model,
            keep_io_types=keep_io,
            disable_shape_infer=disable_shape_infer,
            op_block_list=list(blocked) or None,
        )

    @staticmethod
    def _assert_fp16_loadable(path: Path) -> None:
        """用 ORT 真实加载一次，杜绝「转换成功但无法运行」的假产物。"""
        ort = _lazy_import_ort()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        try:
            session = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise PipelineError(f"ORT 加载失败：{exc}") from exc
        if not session.get_inputs():
            raise PipelineError("模型没有输入，疑似转换损坏")


def toposort_model(model: Any) -> Any:
    """对计算图做拓扑排序（Kahn）。

    ``onnxconverter_common`` / ``onnxruntime.transformers`` 在插入 Cast 节点后
    并不会重排 node 顺序，导致 ``onnx.checker`` 报 “must be topologically
    sorted”。这里统一在落盘前修复，保证产物既能在 ORT 上跑，也能通过官方
    checker（魔搭上传 / 端侧加载都会做校验）。
    """
    graph = model.graph
    available = {item.name for item in graph.input}
    available |= {item.name for item in graph.initializer}
    pending = list(graph.node)
    ordered: List[Any] = []

    while pending:
        progressed = False
        blocked: List[Any] = []
        for node in pending:
            inputs = [name for name in node.input if name]
            if all(name in available for name in inputs):
                ordered.append(node)
                available.update(name for name in node.output if name)
                progressed = True
            else:
                blocked.append(node)
        if not progressed:
            # 存在环或图本身缺输入：原样保留剩余节点，交给 checker 报真实错误
            ordered.extend(blocked)
            break
        pending = blocked

    del graph.node[:]
    graph.node.extend(ordered)
    return model


def _io_dtype_summary(model: Any) -> Dict[str, str]:
    """读取 ONNX 模型 I/O 的 elem_type 文本描述。"""
    try:
        from onnx import TensorProto  # type: ignore

        names = {v: k for k, v in TensorProto.DataType.items()}
    except Exception:  # pragma: no cover
        return {}

    summary: Dict[str, str] = {}
    try:
        for value_info in list(model.graph.input) + list(model.graph.output):
            tensor_type = value_info.type.tensor_type
            summary[value_info.name] = names.get(tensor_type.elem_type, str(tensor_type.elem_type))
    except Exception:
        pass
    return summary


# --------------------------------------------------------------------------- #
# 五、ONNX 校验与 model_io.json（原生绑定契约）
# --------------------------------------------------------------------------- #
DTYPE_TO_NUMPY = {
    "tensor(float)": "float32",
    "tensor(float16)": "float16",
    "tensor(double)": "float64",
    "tensor(int64)": "int64",
    "tensor(int32)": "int32",
    "tensor(int16)": "int16",
    "tensor(int8)": "int8",
    "tensor(uint8)": "uint8",
    "tensor(bool)": "bool",
}


def describe_model(session: Any, model_path: Path) -> Dict[str, Any]:
    """把 ORT session 的输入输出整理为可序列化的契约描述。"""
    def _shape(value: Any) -> List[Any]:
        out: List[Any] = []
        for dim in value.shape:
            out.append(dim if isinstance(dim, int) and dim > 0 else "dynamic")
        return out

    def _dtype(value: Any) -> str:
        text = str(value.type)
        return DTYPE_TO_NUMPY.get(text, text)

    return {
        "path": model_path.name,
        "size_bytes": model_path.stat().st_size if model_path.exists() else 0,
        "sha256": sha256_of_file(model_path) if model_path.exists() else "",
        "inputs": [
            {"name": item.name, "dtype": _dtype(item), "shape": _shape(item)}
            for item in session.get_inputs()
        ],
        "outputs": [
            {"name": item.name, "dtype": _dtype(item), "shape": _shape(item)}
            for item in session.get_outputs()
        ],
    }


def _dummy_feed(inputs: Sequence[Any], seed: int = 20241115) -> Dict[str, Any]:
    """按 session 输入签名构造可跑通的最小 dummy 张量。

    动态维默认取 1，但波形类输入（`ref_audio` / `audio` / `wav`）取 16000 点：
    声码器前置的 reflect padding 要求轴长 ≥ 2，取 1 会让 dummy 前向假失败。
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    feeds: Dict[str, Any] = {}
    for item in inputs:
        dtype = DTYPE_TO_NUMPY.get(str(item.type), "float32")
        name = item.name.lower()
        is_waveform = any(key in name for key in ("audio", "wave", "wav"))
        shape: List[int] = []
        for dim in item.shape:
            if isinstance(dim, int) and dim > 0:
                shape.append(dim)
            else:
                shape.append(16000 if is_waveform else 1)
        if not shape:
            shape = [1]
        if dtype.startswith("float"):
            feeds[item.name] = rng.standard_normal(shape).astype(dtype) * 0.5
        elif dtype in ("int64", "int32", "int16", "int8"):
            feeds[item.name] = np.zeros(shape, dtype=dtype)
        elif dtype == "bool":
            feeds[item.name] = np.zeros(shape, dtype=bool)
        else:
            feeds[item.name] = np.zeros(shape, dtype="float32")
    return feeds


def verify_model(
    model_path: Path,
    threads: int = 4,
    run_dummy: bool = True,
    fp32_counterpart: Optional[Path] = None,
) -> Dict[str, Any]:
    """加载模型，可选跑一次 dummy 前向，并可与 FP32 版本做数值一致性比对。"""
    import numpy as np

    ort = _lazy_import_ort()
    onnx = _lazy_import_onnx()

    try:
        onnx.checker.check_model(str(model_path))
    except Exception as exc:
        return {"ok": False, "error": f"onnx.checker 失败：{exc}", "path": model_path.name}

    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, int(threads))
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    except Exception as exc:
        return {"ok": False, "error": f"ORT 加载失败：{exc}", "path": model_path.name}

    info = describe_model(session, model_path)
    info["ok"] = True

    if not run_dummy:
        return info

    try:
        feeds = _dummy_feed(session.get_inputs())
        started = time.perf_counter()
        outputs = session.run(None, feeds)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        info["dummy_run_ms"] = round(elapsed_ms, 2)
        info["dummy_output_shapes"] = [list(np.shape(out)) for out in outputs]
        for out in outputs:
            if not np.all(np.isfinite(out)):
                info["numeric_warning"] = "dummy 输出包含 NaN/Inf，疑似量化溢出"
                break
    except Exception as exc:
        info["dummy_run_error"] = str(exc)
        LOG.warning("  ├─ dummy 前向未跑通（形状推断可能依赖真实输入）：%s", exc)

    if fp32_counterpart is not None and fp32_counterpart.exists():
        try:
            ref_session = ort.InferenceSession(
                str(fp32_counterpart), sess_options=options, providers=["CPUExecutionProvider"]
            )
            feeds = _dummy_feed(ref_session.get_inputs())
            ref_outputs = ref_session.run(None, feeds)
            got_outputs = session.run(None, feeds)
            sims: List[float] = []
            for ref, got in zip(ref_outputs, got_outputs):
                a = np.asarray(ref, dtype=np.float64).ravel()
                b = np.asarray(got, dtype=np.float64).ravel()
                if a.size == 0 or b.size == 0:
                    continue
                denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                sims.append(float(np.dot(a, b) / denom) if denom > 1e-12 else 1.0)
            if sims:
                info["cosine_similarity_vs_fp32"] = [round(v, 5) for v in sims]
                info["min_cosine_similarity"] = round(min(sims), 5)
                if min(sims) < 0.95:
                    info["numeric_warning"] = (
                        f"与 FP32 相似度偏低（{min(sims):.4f}），请复核该模块量化强度"
                    )
        except Exception as exc:
            info["compare_error"] = str(exc)
    return info


# --------------------------------------------------------------------------- #
# 六、情绪特征向量（.bin）固化
# --------------------------------------------------------------------------- #
@dataclass
class EmotionConfig:
    model_id: str = "TencentGameMate/chinese-hubert-base"
    device: str = "cuda"
    projection_seed: int = 20241115
    projection_matrix: Optional[str] = None
    sample_rate: int = 16000
    decorrelate: bool = True
    dsp_fallback: bool = True


def _read_mono(path: Path, target_sr: int) -> Any:
    """读取参考音为单声道 float32 并重采样。"""
    import numpy as np

    try:
        import soundfile as sf  # type: ignore

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    except Exception:
        import wave  # 标准库兜底

        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
        if width != 2:
            raise PipelineError(f"参考音 {path} 位宽不受支持（仅支持 16bit），请先转 wav")
        pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        mono = pcm.reshape(-1, channels).mean(axis=1) if channels > 1 else pcm

    mono = np.ascontiguousarray(mono, dtype=np.float32)
    if sr != target_sr and mono.size:
        try:
            from scipy.signal import resample_poly  # type: ignore
            from math import gcd

            g = gcd(int(sr), int(target_sr))
            mono = np.asarray(
                resample_poly(mono, target_sr // g, sr // g), dtype=np.float32
            )
        except Exception:
            n_out = int(round(mono.size * target_sr / float(sr)))
            xp = np.linspace(0.0, 1.0, num=mono.size, endpoint=False)
            xq = np.linspace(0.0, 1.0, num=max(n_out, 2), endpoint=False)
            mono = np.interp(xq, xp, mono).astype(np.float32)
    return mono


def _mel_filterbank(n_fft: int, sr: int, n_mels: int, fmin: float = 60.0, fmax: float = 7800.0) -> Any:
    """轻量 mel 三角滤波器组（numpy 实现，避免 librosa 依赖）。"""
    import numpy as np

    def hz_to_mel(f: float) -> float:
        return 2595.0 * math.log10(1.0 + f / 700.0)

    def mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(min(fmax, sr / 2.0)), n_mels + 2)
    hz = mel_to_hz(mels)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    bank = np.zeros((n_mels, freqs.size), dtype=np.float32)
    for i in range(n_mels):
        lo, center, hi = hz[i], hz[i + 1], hz[i + 2]
        left = (freqs - lo) / max(center - lo, 1e-9)
        right = (hi - freqs) / max(hi - center, 1e-9)
        bank[i] = np.clip(np.minimum(left, right), 0.0, None)
    return bank


def _dsp_embedding(samples: Any, sr: int) -> Any:
    """谱域确定性嵌入：4 种窗长 × 32 mel 带 × (均值, 标准差) = 256 维。

    在无法加载 CNHubert（无 GPU / 无 transformers）时作为降级方案使用，
    依然能区分音高、音色与能量分布，保证端侧情绪切换有实际区分度。
    """
    import numpy as np

    if samples.size < sr // 4:
        samples = np.pad(samples, (0, sr // 4 - samples.size + 1))

    features: List[float] = []
    for n_fft in (1024, 2048, 4096, 8192):
        hop = n_fft // 4
        if samples.size < n_fft:
            frame = np.pad(samples, (0, n_fft - samples.size))
            frames = frame[None, :]
        else:
            count = 1 + (samples.size - n_fft) // hop
            indices = np.arange(n_fft)[None, :] + hop * np.arange(count)[:, None]
            frames = samples[indices] * np.hanning(n_fft)[None, :]
        spectrum = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
        bank = _mel_filterbank(n_fft, sr, 32)
        mel = np.log(spectrum @ bank.T + 1e-8)
        features.extend(mel.mean(axis=0).tolist())
        features.extend(mel.std(axis=0).tolist())

    vector = np.asarray(features, dtype=np.float32)
    if vector.size != EMOTION_DIM:  # 兜底对齐维度
        vector = np.resize(vector, EMOTION_DIM).astype(np.float32)
    return vector


class EmotionExtractor:
    """抽取 4 条情绪参考音的 256 维嵌入并固化为 .bin。"""

    def __init__(self, cfg: EmotionConfig) -> None:
        self.cfg = cfg
        self.method = "unknown"
        self.projection: Optional[Any] = None

    # -- 投影矩阵（SSL 1536 维 -> 256 维，确定性可复现） -------------------- #
    def _get_projection(self, in_dim: int) -> Any:
        import numpy as np

        if self.projection is not None:
            return self.projection

        if self.cfg.projection_matrix:
            matrix_path = Path(self.cfg.projection_matrix)
            if matrix_path.suffix == ".npz":
                self.projection = np.load(str(matrix_path))["matrix"].astype(np.float32)
            else:
                self.projection = np.load(str(matrix_path)).astype(np.float32)
            LOG.info("已加载外部投影矩阵：%s %s", matrix_path.name, self.projection.shape)
            return self.projection

        rng = np.random.default_rng(self.cfg.projection_seed)
        raw = rng.standard_normal((in_dim, EMOTION_DIM)).astype(np.float64)
        # QR 正交化，保证投影保持能量、不放大噪声
        q, _ = np.linalg.qr(raw)
        self.projection = q[:, :EMOTION_DIM].astype(np.float32)
        return self.projection

    # -- CNHubert 嵌入 ------------------------------------------------------ #
    def _ssl_embedding(self, samples: Any, sr: int) -> Any:
        import numpy as np

        import torch  # type: ignore
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model  # type: ignore

        device = self.cfg.device if torch.cuda.is_available() and "cuda" in self.cfg.device else "cpu"
        processor = Wav2Vec2FeatureExtractor.from_pretrained(self.cfg.model_id)
        model = Wav2Vec2Model.from_pretrained(self.cfg.model_id).to(device).eval()

        inputs = processor(samples, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state  # (1, T, 768)
            pooled = torch.cat([hidden.mean(dim=1), hidden.std(dim=1)], dim=-1)  # 1536
        return pooled.squeeze(0).cpu().numpy().astype(np.float32)

    # -- 单条参考音 -> 256 维 ---------------------------------------------- #
    def embed(self, wav_path: Path) -> Tuple[Any, str]:
        import numpy as np

        samples = _read_mono(wav_path, self.cfg.sample_rate)
        if samples.size == 0:
            raise PipelineError(f"参考音为空：{wav_path}")

        try:
            ssl = self._ssl_embedding(samples, self.cfg.sample_rate)
            vector = ssl @ self._get_projection(ssl.size)
            return vector.astype(np.float32), "cnhubert_mean_std+orthogonal_projection"
        except Exception as exc:
            if not self.cfg.dsp_fallback:
                raise
            LOG.warning("  ├─ CNHubert 不可用（%s），降级谱域嵌入", str(exc)[:120])
            return _dsp_embedding(samples, self.cfg.sample_rate), "dsp_mel_statistics"

    # -- 全流程 ------------------------------------------------------------- #
    def extract_all(self, mapping: Mapping[str, Path]) -> Dict[str, Any]:
        import numpy as np

        vectors: Dict[str, Any] = {}
        methods: Dict[str, str] = {}
        for emotion, path in mapping.items():
            if not path.exists():
                raise PipelineError(f"缺少「{emotion}」参考音：{path}")
            vector, method = self.embed(path)
            vectors[emotion] = vector
            methods[emotion] = method
            LOG.info("  ├─ %s <- %s (%s, 原始范数 %.3f)",
                     emotion, path.name, method, float(np.linalg.norm(vector)))

        stacked = np.stack([vectors[e] for e in EMOTIONS if e in vectors])
        if self.cfg.decorrelate and stacked.shape[0] == len(EMOTIONS):
            mean = stacked.mean(axis=0, keepdims=True)
            center = stacked - mean
            for idx, emotion in enumerate([e for e in EMOTIONS if e in vectors]):
                norm = float(np.linalg.norm(center[idx]))
                if norm > 1e-6:
                    vectors[emotion] = center[idx] / norm
                else:
                    # 差值退化（参考音几乎同质）时保留原向量，避免全零
                    LOG.warning("  ├─ 「%s」去相关后趋近零向量，回退为归一化原向量", emotion)
                    original = stacked[idx]
                    vectors[emotion] = original / max(float(np.linalg.norm(original)), 1e-6)

        return {
            "vectors": vectors,
            "methods": methods,
            "method": "cnhubert" if all(m.startswith("cnhubert") for m in methods.values()) else "mixed",
        }


def validate_emotion_vectors(vectors: Mapping[str, Any]) -> Dict[str, Any]:
    """校验维度 / 有限性 / 两两余弦相似度（过高说明情绪区分度不足）。"""
    import numpy as np

    report: Dict[str, Any] = {"ok": True, "warnings": [], "pairwise_cosine": {}}
    names = [e for e in EMOTIONS if e in vectors]
    for name in names:
        vec = np.asarray(vectors[name], dtype=np.float32)
        if vec.size != EMOTION_DIM:
            report["ok"] = False
            report["warnings"].append(f"{name} 维度 {vec.size} != {EMOTION_DIM}")
        if not np.all(np.isfinite(vec)):
            report["ok"] = False
            report["warnings"].append(f"{name} 含 NaN/Inf")
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            report["ok"] = False
            report["warnings"].append(f"{name} 为零向量")
        report.setdefault("norms", {})[name] = round(norm, 5)

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            va = np.asarray(vectors[a], dtype=np.float64).ravel()
            vb = np.asarray(vectors[b], dtype=np.float64).ravel()
            denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
            cos = float(np.dot(va, vb) / denom) if denom > 1e-12 else 1.0
            report["pairwise_cosine"][f"{a}-{b}"] = round(cos, 4)
            if abs(cos) > 0.985:
                report["warnings"].append(f"{a} 与 {b} 相似度过高（{cos:.3f}），区分度不足")
    return report


def write_emotion_bin(path: Path, vector: Any) -> int:
    """写出 float32 小端裸二进制（严格 1024 字节）。"""
    import numpy as np

    array = np.asarray(vector, dtype="<f4").ravel()
    if array.size != EMOTION_DIM:
        raise PipelineError(f"情绪向量维度错误：{array.size} != {EMOTION_DIM}")
    ensure_dir(path.parent)
    payload = array.astype("<f4").tobytes(order="C")
    if len(payload) != EMOTION_BYTES:
        raise PipelineError(f"情绪向量字节数错误：{len(payload)} != {EMOTION_BYTES}")
    path.write_bytes(payload)
    return len(payload)


# --------------------------------------------------------------------------- #
# 七、魔搭清单（manifest.json）
# --------------------------------------------------------------------------- #
@dataclass
class ManifestMeta:
    name: str = "爱弥斯 · GPT-SoVITS 2 端侧混合量化语音包"
    character: str = "爱弥斯 (Aemeath)"
    version: str = "2.1.0-int8-fp16"
    arch: str = "GPT-SoVITS v2"
    engine: str = "onnxruntime"
    sample_rate: int = 32000
    channels: int = 1
    bit_depth: int = 16
    license: str = "AGPL-3.0"
    author: str = "Aemeath Terminal Architecture Team"
    repo_id: str = "aemeath/gpt-sovits-aemeath-pro-onnx"
    language: str = "ZH"
    speaker: str = "aemeath"


def build_manifest(
    out_dir: Path,
    meta: ManifestMeta,
    plan: Sequence[QuantPlanItem],
    io_descriptions: Mapping[str, Any],
    include_globs: Sequence[str] = (
        "*.onnx",
        "emotions/*.bin",
        # GPT-SoVITS 参考条件包（prepare_ref_conditioning.py 产出）
        "conditioning/*.json",
        "conditioning/**/*.json",
        "conditioning/**/*.f32",
        "conditioning/**/*.i64",
        "tokens.txt",
        "lexicon.txt",
    ),
) -> Dict[str, Any]:
    """扫描输出目录，生成含 SHA256 与体积的魔搭上传清单。"""
    files: List[Dict[str, Any]] = []
    seen: set = set()

    for pattern in include_globs:
        for path in sorted(out_dir.glob(pattern)):
            if not path.is_file():
                continue
            rel = relative_posix(path, out_dir)
            if rel in seen or rel == "manifest.json":
                continue
            seen.add(rel)
            digest = sha256_of_file(path)
            entry: Dict[str, Any] = {
                "path": rel,
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
            io_info = io_descriptions.get(path.name) or io_descriptions.get(rel)
            if io_info:
                entry["precision"] = io_info.get("precision", "")
                entry["inputs"] = io_info.get("inputs", [])
                entry["outputs"] = io_info.get("outputs", [])
            elif rel.endswith(".onnx"):
                # 无 model_io.json 时按命名后缀回推精度与角色
                lowered = path.name.lower()
                if lowered.endswith(".int8.onnx"):
                    entry["precision"] = "int8_dynamic"
                elif lowered.endswith(".fp16.onnx"):
                    entry["precision"] = "fp16"
                entry["role"] = _guess_role(lowered)
            if rel.startswith("emotions/") and rel.endswith(".bin"):
                entry["role"] = "emotion_embedding"
                entry["dim"] = EMOTION_DIM
            files.append(entry)

    total_bytes = sum(item["bytes"] for item in files)

    models: Dict[str, str] = {}
    for item in plan:
        rel = relative_posix(Path(item.source), out_dir)
        if item.role in models and models[item.role] != rel:
            LOG.warning("  角色 %s 出现多个文件（%s / %s），清单以最新为准", item.role, models[item.role], rel)
        models[item.role] = rel

    manifest: Dict[str, Any] = {
        "schema_version": "1.1",
        "name": meta.name,
        "character": meta.character,
        "version": meta.version,
        "license": meta.license,
        "author": meta.author,
        "language": meta.language,
        "speaker": meta.speaker,
        "repo_id": meta.repo_id,
        "architecture": meta.arch,
        "arch": meta.arch,
        "engine": meta.engine,
        "sample_rate": meta.sample_rate,
        "channels": meta.channels,
        "bit_depth": meta.bit_depth,
        "quantization": {
            "strategy": "int8_dynamic_frontend + fp16_vocoder",
            "int8_backend": "onnxruntime.quantization.quantize_dynamic",
            "int8_weight_type": "QUInt8",
            "fp16_backend": "onnxconverter_common.float16.convert_float_to_float16",
            "fp16_keep_io_types": True,
            "fp16_blocked_ops": list(DEFAULT_FP16_BLOCK_OPS),
            "rationale": "前端语义压 Int8 提速省内存，声码器守 FP16 杜绝金属电流声",
        },
        "sampling_params": dict(SAMPLING_PARAMS),
        "supported_emotions": list(EMOTIONS),
        "emotion_vector_dim": EMOTION_DIM,
        "emotions": {
            emotion: f"emotions/{emotion}.bin" for emotion in EMOTIONS
        },
        "models": models,
        "files": files,
        "total_bytes": total_bytes,
        "total_human": human_bytes(total_bytes),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "generator": f"export_and_quantize_onnx.py/{TOOL_VERSION}",
    }
    return manifest


# --------------------------------------------------------------------------- #
# 八、子命令实现
# --------------------------------------------------------------------------- #
def collect_onnx_files(root: Path, exclude_quantized: bool = True) -> List[Path]:
    if not root.exists():
        raise PipelineError(f"目录不存在：{root}")
    files: List[Path] = []
    for path in sorted(root.rglob("*.onnx")):
        if not path.is_file():
            continue
        if exclude_quantized and any(path.name.lower().endswith(s) for s in QUANT_SUFFIXES):
            continue
        files.append(path)
    return files


def cmd_export(args: argparse.Namespace) -> int:
    """调用 GPT-SoVITS 官方 onnx_export.py 产出 FP32 ONNX。"""
    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve())
    repo = Path(args.gpt_sovits_repo).expanduser().resolve() if args.gpt_sovits_repo else None

    if args.skip_export:
        LOG.info("--skip-export：跳过导出，仅校验 %s 中的既有 FP32 ONNX。", out_dir)
        files = collect_onnx_files(Path(args.fp32_dir).expanduser().resolve(), exclude_quantized=False)
        LOG.info("发现 %d 个 ONNX 文件。", len(files))
        return 0 if files else 3

    if repo is None:
        raise PipelineError(
            "export 需要 --gpt-sovits-repo 指向 GPT-SoVITS 源码根目录，"
            "或使用 --skip-export 复用已导出的 FP32 ONNX。"
        )

    script = repo / "GPT_SoVITS" / "onnx_export.py"
    if not script.exists():
        candidates = list(repo.rglob("onnx_export.py"))
        if not candidates:
            raise PipelineError(f"在 {repo} 中找不到 GPT_SoVITS/onnx_export.py")
        script = candidates[0]

    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve() if args.ckpt_dir else repo / "GPT_SoVITS" / "pretrained_models"
    cmd = [
        sys.executable,
        str(script),
        "--s2G_path", str(args.s2g_path or (ckpt_dir / "s2G.pth")),
        "--s1_path", str(args.s1_path or (ckpt_dir / "s1bert.pth")),
        "--cnhubert_base_path", str(args.cnhubert_path or (ckpt_dir / "chinese-hubert-base")),
        "--bert_path", str(args.bert_path or (ckpt_dir / "chinese-roberta-wwm-ext-large")),
        "--output_path", str(out_dir),
    ]
    if args.repo_version:
        cmd.extend(["--repo_version", args.repo_version])
    if args.extra_export_args:
        cmd.extend(args.extra_export_args)

    LOG.info("执行官方导出命令：\n  %s", " ".join(cmd))
    if args.dry_run:
        return 0

    started = time.perf_counter()
    completed = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
    if completed.stdout:
        LOG.info("标准输出：\n%s", completed.stdout.strip()[-4000:])
    if completed.returncode != 0:
        LOG.error("导出失败（exit=%d）：\n%s", completed.returncode, completed.stderr.strip()[-4000:])
        return 1

    produced = collect_onnx_files(out_dir, exclude_quantized=False)
    LOG.info("导出完成，用时 %.1fs，产物 %d 个：", time.perf_counter() - started, len(produced))
    for path in produced:
        LOG.info("  ├─ %-32s %s", path.name, human_bytes(path.stat().st_size))
    return 0 if produced else 3


def cmd_quantize(args: argparse.Namespace) -> int:
    fp32_dir = Path(args.fp32_dir).expanduser().resolve()
    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve())
    files = collect_onnx_files(fp32_dir)
    if not files:
        LOG.error("目录中没有待量化的 FP32 ONNX：%s", fp32_dir)
        return 3

    int8_patterns = tuple(args.int8_pattern or DEFAULT_INT8_PATTERNS)
    fp16_patterns = tuple(args.fp16_pattern or DEFAULT_FP16_PATTERNS)
    plan = build_quant_plan(files, int8_patterns, fp16_patterns, args.default_precision)

    if args.only:
        wanted = {token.strip().lower() for token in args.only.split(",") if token.strip()}
        plan = [item for item in plan if any(w in Path(item.source).name.lower() for w in wanted)]

    LOG.info("量化计划（%d 个模块）：", len(plan))
    for item in plan:
        LOG.info(
            "  ├─ %-34s %-15s [%s]",
            Path(item.source).name,
            item.precision,
            item.matched_pattern or "default",
        )
    if args.print_plan:
        print(json.dumps([item.as_dict() for item in plan], ensure_ascii=False, indent=2))

    if args.dry_run:
        LOG.info("--dry-run：仅展示计划，不执行量化。")
        return 0

    cfg = QuantConfig(
        weight_type=args.weight_type,
        per_channel=not args.no_per_channel,
        reduce_range=args.reduce_range,
        keep_io_types=not args.no_keep_io_types,
        fp16_block_ops=() if args.fp16_blocklist == "none" else tuple(args.fp16_block_ops or DEFAULT_FP16_BLOCK_OPS),
        fp16_shapes_infer=not args.no_fp16_shapes_infer,
    )
    quantizer = Quantizer(cfg)

    report: Dict[str, Any] = {
        "tool": f"export_and_quantize_onnx.py/{TOOL_VERSION}",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fp32_dir": str(fp32_dir),
        "out_dir": str(out_dir),
        "config": {
            "weight_type": cfg.weight_type,
            "per_channel": cfg.per_channel,
            "reduce_range": cfg.reduce_range,
            "keep_io_types": cfg.keep_io_types,
            "fp16_blocked_ops": list(cfg.fp16_block_ops),
        },
        "items": [],
        "failures": 0,
    }

    for item in plan:
        source = Path(item.source)
        target = out_dir / output_name_for(source, item.precision)
        entry: Dict[str, Any] = {
            "source": str(source),
            "target": str(target),
            "role": item.role,
            "precision": item.precision,
            "matched_pattern": item.matched_pattern,
        }
        try:
            if target.exists() and not args.force:
                LOG.info("  ├─ 跳过已存在：%s（--force 可覆盖）", target.name)
                entry["skipped"] = True
            else:
                base_info = quantizer.quantize_dynamic_int8(source, target) if item.precision == "int8_dynamic" else quantizer.convert_fp16(source, target)
                entry.update(base_info)
            entry["source_bytes"] = source.stat().st_size
            entry["target_bytes"] = target.stat().st_size if target.exists() else 0
            if entry["source_bytes"] and entry["target_bytes"]:
                entry["ratio"] = round(entry["target_bytes"] / entry["source_bytes"], 4)
                LOG.info(
                    "  ├─ %s：%s -> %s（压缩至 %.1f%%）",
                    target.name,
                    human_bytes(entry["source_bytes"]),
                    human_bytes(entry["target_bytes"]),
                    entry["ratio"] * 100.0,
                )
        except Exception as exc:
            entry["error"] = str(exc)
            report["failures"] += 1
            LOG.error("  ├─ 量化失败 %s：%s", source.name, exc)

        if not args.no_verify and target.exists() and "error" not in entry:
            try:
                verification = verify_model(
                    target,
                    threads=args.threads,
                    run_dummy=not args.no_dummy_run,
                    fp32_counterpart=source if args.compare_fp32 else None,
                )
                entry["verification"] = verification
                if not verification.get("ok", False):
                    report["failures"] += 1
                    LOG.error("  ├─ 校验失败 %s：%s", target.name, verification.get("error"))
                else:
                    sims = verification.get("cosine_similarity_vs_fp32")
                    LOG.info(
                        "  ├─ 校验通过 %s%s",
                        target.name,
                        f"（vs FP32 最小余弦 {min(sims):.4f}）" if sims else "",
                    )
            except Exception as exc:
                entry["verification"] = {"ok": False, "error": str(exc)}
                report["failures"] += 1
                LOG.error("  ├─ 校验异常 %s：%s", target.name, exc)

        report["items"].append(entry)

    report_path = out_dir / "quant_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    total_before = sum(i.get("source_bytes", 0) for i in report["items"])
    total_after = sum(i.get("target_bytes", 0) for i in report["items"])
    LOG.info("量化完成：%s -> %s（%.1f%%），报告 %s",
             human_bytes(total_before), human_bytes(total_after),
             (total_after / total_before * 100.0) if total_before else 0.0, report_path)
    return 0 if report["failures"] == 0 else 1


def _resolve_emotion_refs(args: argparse.Namespace) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    if args.emotion_map:
        with Path(args.emotion_map).expanduser().open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        for emotion in EMOTIONS:
            if emotion in raw:
                mapping[emotion] = Path(raw[emotion]).expanduser().resolve()
    else:
        root = Path(args.reference_dir).expanduser().resolve() if args.reference_dir else None
        if root is None:
            raise PipelineError("emotions 需要 --reference-dir 或 --emotion-map")
        for emotion in EMOTIONS:
            candidates = [
                root / f"{emotion}.wav",
                root / f"{emotion}.WAV",
                root / f"ref_{emotion}.wav",
            ]
            for candidate in candidates:
                if candidate.exists():
                    mapping[emotion] = candidate
                    break
            else:
                # 允许任意前缀，例如 aimisi_ref_平静_01.wav
                matches = sorted(root.glob(f"*{emotion}*.wav"))
                if matches:
                    mapping[emotion] = matches[0]
    missing = [e for e in EMOTIONS if e not in mapping]
    if missing:
        raise PipelineError(f"缺少情绪参考音：{', '.join(missing)}")
    return mapping


def cmd_emotions(args: argparse.Namespace) -> int:
    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve())
    mapping = _resolve_emotion_refs(args)
    cfg = EmotionConfig(
        model_id=args.ssl_model,
        device=args.device,
        projection_seed=args.projection_seed,
        projection_matrix=args.projection_matrix,
        decorrelate=not args.no_decorrelate,
        dsp_fallback=not args.no_dsp_fallback,
    )
    extractor = EmotionExtractor(cfg)
    LOG.info("开始提取情绪特征（SSL 模型：%s，设备：%s）", cfg.model_id, cfg.device)
    result = extractor.extract_all(mapping)

    if args.dry_run:
        LOG.info("--dry-run：仅计算不落盘。")
        return 0

    written: Dict[str, Any] = {}
    for emotion in EMOTIONS:
        vector = result["vectors"][emotion]
        target = out_dir / f"{emotion}.bin"
        size = write_emotion_bin(target, vector)
        written[emotion] = {
            "path": str(target),
            "bytes": size,
            "sha256": sha256_of_file(target),
            "method": result["methods"][emotion],
        }
        LOG.info("  ├─ 写出 %s（%d 字节，SHA256 %s…）", target.name, size, written[emotion]["sha256"][:12])

    validation = validate_emotion_vectors(result["vectors"])
    report = {
        "tool": f"export_and_quantize_onnx.py/{TOOL_VERSION}",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": result["method"],
        "ssl_model": cfg.model_id,
        "projection_seed": cfg.projection_seed,
        "projection_matrix": cfg.projection_matrix,
        "dim": EMOTION_DIM,
        "bytes_per_file": EMOTION_BYTES,
        "decorrelated": cfg.decorrelate,
        "files": written,
        "validation": validation,
    }
    report_path = out_dir / "emotion_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for warning in validation.get("warnings", []):
        LOG.warning("  ├─ %s", warning)
    LOG.info("情绪向量固化完成：%s（校验 %s）", out_dir, "通过" if validation["ok"] else "有问题")
    return 0 if validation["ok"] else 1


def cmd_manifest(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not out_dir.exists():
        raise PipelineError(f"目录不存在：{out_dir}")
    # 清单阶段需要包含已量化产物（*.int8.onnx / *.fp16.onnx）
    files = collect_onnx_files(out_dir, exclude_quantized=False)
    plan = build_quant_plan(files, DEFAULT_INT8_PATTERNS, DEFAULT_FP16_PATTERNS, "fp16")

    io_path = out_dir / "model_io.json"
    io_descriptions: Dict[str, Any] = {}
    if io_path.exists():
        try:
            io_descriptions = json.loads(io_path.read_text(encoding="utf-8")).get("files", {})
        except Exception as exc:
            LOG.warning("读取 model_io.json 失败：%s", exc)

    meta = ManifestMeta(
        name=args.name,
        version=args.version,
        arch=args.arch,
        engine=args.engine,
        sample_rate=args.sample_rate,
        repo_id=args.repo_id,
        license=args.license,
    )
    manifest = build_manifest(out_dir, meta, plan, io_descriptions)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    LOG.info("清单已生成：%s", manifest_path)
    LOG.info("  文件 %d 个，合计 %s", len(manifest["files"]), manifest["total_human"])
    for item in manifest["files"]:
        LOG.info("  ├─ %-34s %-10s %s…", item["path"], human_bytes(item["bytes"]), item["sha256"][:12])

    if args.bundle:
        bundle_path = out_dir / f"{args.bundle_name or args.repo_id.split('/')[-1]}-{args.version}.zip"
        with shutil_zipfile(bundle_path) as archive:
            for item in manifest["files"]:
                archive.write(out_dir / item["path"], arcname=item["path"])
            archive.write(manifest_path, arcname="manifest.json")
        LOG.info("语音包 DLC 已打包：%s（%s）", bundle_path, human_bytes(bundle_path.stat().st_size))

    if args.upload_cmd:
        print(
            f"modelscope upload {args.repo_id} {out_dir} . "
            f"--commit-message 'aemeath onnx int8+fp16 {args.version}'"
        )
    return 0


def shutil_zipfile(path: Path) -> Any:
    import zipfile

    ensure_dir(path.parent)
    return zipfile.ZipFile(str(path), "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6)


def cmd_verify(args: argparse.Namespace) -> int:
    target = Path(args.target).expanduser().resolve()
    # 校验对象本身就是量化产物，不能过滤掉 *.int8.onnx / *.fp16.onnx
    files = collect_onnx_files(target, exclude_quantized=False) if target.is_dir() else [target]
    if not files:
        LOG.error("没有可校验的 ONNX：%s", target)
        return 3

    descriptions: Dict[str, Any] = {}
    failures = 0
    for path in files:
        LOG.info("校验 %s …", path.name)
        fp32_counterpart = None
        if args.compare_fp32:
            candidate = Path(args.fp32_dir).expanduser().resolve() / path.name.replace(".int8", "").replace(".fp16", "")
            if candidate.exists():
                fp32_counterpart = candidate
        result = verify_model(
            path,
            threads=args.threads,
            run_dummy=not args.no_dummy_run,
            fp32_counterpart=fp32_counterpart,
        )
        descriptions[path.name] = result
        # 把精度写进 model_io.json，Android 原生层据此决定 INT8/FP16 会话装配方式
        if path.name.endswith(".int8.onnx"):
            result["precision"] = "int8_dynamic"
        elif path.name.endswith(".fp16.onnx"):
            result["precision"] = "fp16"
        elif path.name.endswith(".onnx"):
            result["precision"] = "fp32"
        if not result.get("ok"):
            failures += 1
            LOG.error("  ├─ ✗ %s", result.get("error"))
            continue
        LOG.info(
            "  ├─ ✓ %d 入 / %d 出，dummy %.1fms%s",
            len(result.get("inputs", [])),
            len(result.get("outputs", [])),
            result.get("dummy_run_ms", 0.0) or 0.0,
            f"，vs FP32 最小余弦 {result['min_cosine_similarity']:.4f}"
            if "min_cosine_similarity" in result
            else "",
        )
        if result.get("numeric_warning"):
            LOG.warning("  ├─ ⚠ %s", result["numeric_warning"])
        if args.strict and result.get("dummy_run_error"):
            failures += 1

    io_payload = {
        "schema_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "arch": args.arch,
        "files": descriptions,
    }
    io_path = (target if target.is_dir() else target.parent) / "model_io.json"
    io_path.write_text(json.dumps(io_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("张量 I/O 契约已写出：%s", io_path)
    return 0 if failures == 0 else 1


def cmd_all(args: argparse.Namespace) -> int:
    """quantize → emotions → verify(model_io) → manifest 的一键串联。"""
    fp32_dir = Path(args.fp32_dir).expanduser().resolve()
    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve())

    quant_args = argparse.Namespace(
        fp32_dir=str(fp32_dir),
        out_dir=str(out_dir),
        int8_pattern=None,
        fp16_pattern=None,
        default_precision=args.default_precision,
        only=None,
        print_plan=False,
        dry_run=args.dry_run,
        weight_type=args.weight_type,
        no_per_channel=False,
        reduce_range=False,
        no_keep_io_types=False,
        fp16_blocklist="default",
        fp16_block_ops=None,
        no_fp16_shapes_infer=False,
        force=args.force,
        no_verify=True,          # 串联模式统一在 verify 阶段做一次
        no_dummy_run=False,
        compare_fp32=False,
        threads=args.threads,
    )
    code = cmd_quantize(quant_args)
    if code not in (0,):
        LOG.error("量化阶段失败（exit=%d），中止后续步骤。", code)
        return code

    if args.reference_dir or args.emotion_map:
        emotion_args = argparse.Namespace(
            reference_dir=args.reference_dir,
            emotion_map=args.emotion_map,
            out_dir=str(out_dir / "emotions"),
            ssl_model=args.ssl_model,
            device=args.device,
            projection_seed=args.projection_seed,
            projection_matrix=None,
            no_decorrelate=False,
            no_dsp_fallback=False,
            dry_run=args.dry_run,
        )
        code = cmd_emotions(emotion_args)
        if code != 0:
            LOG.warning("情绪向量阶段返回 %d，请检查 emotion_report.json", code)
    else:
        LOG.warning("未提供 --reference-dir / --emotion-map，跳过情绪向量固化。")

    verify_args = argparse.Namespace(
        target=str(out_dir),
        fp32_dir=str(fp32_dir),
        compare_fp32=True,
        no_dummy_run=args.no_dummy_run,
        threads=args.threads,
        strict=False,
        arch=args.arch,
    )
    verify_code = cmd_verify(verify_args)

    manifest_args = argparse.Namespace(
        out_dir=str(out_dir),
        name=args.name,
        version=args.version,
        arch=args.arch,
        engine=args.engine,
        sample_rate=args.sample_rate,
        repo_id=args.repo_id,
        license="AGPL-3.0",
        bundle=args.bundle,
        bundle_name=None,
        upload_cmd=True,
    )
    manifest_code = cmd_manifest(manifest_args)

    if args.dry_run:
        LOG.info("--dry-run：全流程演练完成，未产生真实模型文件。")
    return max(verify_code, manifest_code)


# --------------------------------------------------------------------------- #
# 九、命令行入口
# --------------------------------------------------------------------------- #
def add_common_quant_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--weight-type", choices=("QUInt8", "QInt8"), default="QUInt8",
                   help="Int8 权重类型（端侧默认 QUInt8，ARM NEON 利用率更高）")
    p.add_argument("--no-per-channel", action="store_true", help="关闭逐通道量化（体积更小、精度略降）")
    p.add_argument("--reduce-range", action="store_true", help="使用 7bit 量化范围（兼容老旧 CPU）")
    p.add_argument("--no-keep-io-types", action="store_true",
                   help="⚠ 关闭 FP16 I/O 保持（会破坏前端 Float32 契约，不推荐）")
    p.add_argument("--fp16-blocklist", choices=("default", "none"), default="default",
                   help="FP16 数值敏感算子白名单策略")
    p.add_argument("--fp16-block-ops", nargs="*", default=None, help="自定义 FP16 保持 FP32 的算子列表")
    p.add_argument("--no-fp16-shapes-infer", action="store_true",
                   help="关闭 FP16 转换前的形状推断（不推荐：blocklist 时会生成非法类型图）")
    p.add_argument("--threads", type=int, default=4, help="校验阶段 ORT 线程数")
    p.add_argument("--no-dummy-run", action="store_true", help="校验时跳过 dummy 前向")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_and_quantize_onnx.py",
        description="爱弥斯端侧语音模型 ONNX 导出 / Int8+FP16 混合量化 / 情绪向量固化 / 魔搭清单生成",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"aemeath-onnx {TOOL_VERSION}")
    parser.add_argument("--log-level", default="INFO", help="日志级别")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # export
    p_export = subparsers.add_parser("export", help="导出 FP32 ONNX（调用 GPT-SoVITS 官方脚本）")
    p_export.add_argument("--gpt-sovits-repo", default=None, help="GPT-SoVITS 源码根目录")
    p_export.add_argument("--ckpt-dir", default=None, help="权重目录（s1bert.pth / s2G.pth 所在处）")
    p_export.add_argument("--s1-path", default=None, help="自定义 s1 语义模型权重路径")
    p_export.add_argument("--s2g-path", default=None, help="自定义 s2G 声码器权重路径")
    p_export.add_argument("--cnhubert-path", default=None, help="chinese-hubert-base 目录")
    p_export.add_argument("--bert-path", default=None, help="chinese-roberta-wwm-ext-large 目录")
    p_export.add_argument("--repo-version", default=None, help="GPT-SoVITS 版本标识（如 v2）")
    p_export.add_argument("--out-dir", "--output-path", dest="out_dir", required=True, help="FP32 ONNX 输出目录")
    p_export.add_argument("--fp32-dir", default=None, help="--skip-export 时用于扫描既有 ONNX 的目录")
    p_export.add_argument("--skip-export", action="store_true", help="不执行导出，仅校验既有产物")
    p_export.add_argument("--extra-export-args", nargs=argparse.REMAINDER, default=None,
                          help="透传给官方脚本的额外参数")
    p_export.add_argument("--dry-run", action="store_true", help="仅打印将执行的命令")
    p_export.set_defaults(func=cmd_export)

    # quantize
    p_quant = subparsers.add_parser("quantize", help="Int8 动态量化 + FP16 声码器转换")
    p_quant.add_argument("--fp32-dir", required=True, help="FP32 ONNX 输入目录")
    p_quant.add_argument("--out-dir", required=True, help="量化产物输出目录")
    p_quant.add_argument("--int8-pattern", action="append", default=None,
                         help="自定义 Int8 匹配正则（可多次指定，覆盖默认）")
    p_quant.add_argument("--fp16-pattern", action="append", default=None,
                         help="自定义 FP16 匹配正则（可多次指定，覆盖默认）")
    p_quant.add_argument("--default-precision", choices=("fp16", "int8_dynamic"), default="fp16",
                         help="未命中任何规则时的兜底精度（保守选 fp16）")
    p_quant.add_argument("--only", default=None, help="仅处理文件名包含该子串的模块（逗号分隔）")
    p_quant.add_argument("--print-plan", action="store_true", help="打印并输出量化计划 JSON")
    p_quant.add_argument("--force", action="store_true", help="覆盖已存在的量化产物")
    p_quant.add_argument("--compare-fp32", action="store_true", help="与同目录 FP32 版本比对余弦相似度")
    p_quant.add_argument("--no-verify", action="store_true", help="跳过量化后校验")
    p_quant.add_argument("--dry-run", action="store_true", help="只展示计划，不写文件")
    add_common_quant_args(p_quant)
    p_quant.set_defaults(func=cmd_quantize)

    # emotions
    p_emo = subparsers.add_parser("emotions", help="提取 4 组情绪特征向量并写出 .bin")
    p_emo.add_argument("--reference-dir", default=None, help="含 平静/开心/温柔/调侃 参考音的目录")
    p_emo.add_argument("--emotion-map", default=None, help="JSON 映射文件：{\"平静\": \"/path/a.wav\", ...}")
    p_emo.add_argument("--out-dir", required=True, help=".bin 输出目录")
    p_emo.add_argument("--ssl-model", default="TencentGameMate/chinese-hubert-base",
                       help="CNHubert / Wav2Vec2 特征模型（魔搭或 HF 模型 ID）")
    p_emo.add_argument("--device", default="cuda", help="特征提取设备：cuda / cpu")
    p_emo.add_argument("--projection-seed", type=int, default=20241115,
                       help="1536 -> 256 维正交投影随机种子（固定即可跨机复现）")
    p_emo.add_argument("--projection-matrix", default=None, help="外部投影矩阵 .npy/.npz（优先于随机种子）")
    p_emo.add_argument("--no-decorrelate", action="store_true", help="不做跨情绪均值去相关")
    p_emo.add_argument("--no-dsp-fallback", action="store_true", help="禁止降级谱域嵌入（无 GPU 时直接报错）")
    p_emo.add_argument("--dry-run", action="store_true", help="仅计算不落盘")
    p_emo.set_defaults(func=cmd_emotions)

    # manifest
    p_man = subparsers.add_parser("manifest", help="生成魔搭上传清单 manifest.json")
    p_man.add_argument("--out-dir", required=True, help="待打包的模型目录")
    p_man.add_argument("--name", default=ManifestMeta.name, help="模型包名称")
    p_man.add_argument("--version", default=ManifestMeta.version, help="版本号")
    p_man.add_argument("--arch", default=ManifestMeta.arch, help="模型架构标识")
    p_man.add_argument("--engine", default=ManifestMeta.engine, help="推理引擎标识")
    p_man.add_argument("--sample-rate", type=int, default=ManifestMeta.sample_rate, help="输出采样率")
    p_man.add_argument("--repo-id", default=ManifestMeta.repo_id, help="魔搭仓库 ID")
    p_man.add_argument("--license", default=ManifestMeta.license, help="开源许可")
    p_man.add_argument("--bundle", action="store_true", help="同时打 zip DLC 语音包")
    p_man.add_argument("--bundle-name", default=None, help="zip 包名（默认取仓库名+版本）")
    p_man.add_argument("--upload-cmd", action="store_true", help="打印 modelscope upload 命令")
    p_man.set_defaults(func=cmd_manifest)

    # verify
    p_ver = subparsers.add_parser("verify", help="ORT 加载校验 + 数值比对 + 生成 model_io.json")
    p_ver.add_argument("--target", required=True, help="ONNX 文件或目录")
    p_ver.add_argument("--fp32-dir", default=None, help="FP32 对照目录（--compare-fp32 时使用）")
    p_ver.add_argument("--compare-fp32", action="store_true", help="与 FP32 版本比对余弦相似度")
    p_ver.add_argument("--no-dummy-run", action="store_true", help="跳过 dummy 前向")
    p_ver.add_argument("--threads", type=int, default=4, help="ORT 线程数")
    p_ver.add_argument("--strict", action="store_true", help="dummy 前向失败也判为失败")
    p_ver.add_argument("--arch", default="GPT-SoVITS v2", help="写入 model_io.json 的架构标识")
    p_ver.set_defaults(func=cmd_verify)

    # all
    p_all = subparsers.add_parser("all", help="quantize + emotions + verify + manifest 一键出包")
    p_all.add_argument("--fp32-dir", required=True, help="FP32 ONNX 输入目录")
    p_all.add_argument("--out-dir", required=True, help="最终产物目录")
    p_all.add_argument("--reference-dir", default=None, help="4 条情绪参考音目录")
    p_all.add_argument("--emotion-map", default=None, help="情绪参考音 JSON 映射")
    p_all.add_argument("--arch", default=ManifestMeta.arch, help="架构标识")
    p_all.add_argument("--engine", default=ManifestMeta.engine, help="引擎标识")
    p_all.add_argument("--sample-rate", type=int, default=ManifestMeta.sample_rate, help="采样率")
    p_all.add_argument("--name", default=ManifestMeta.name, help="模型包名称")
    p_all.add_argument("--version", default=ManifestMeta.version, help="版本号")
    p_all.add_argument("--repo-id", default=ManifestMeta.repo_id, help="魔搭仓库 ID")
    p_all.add_argument("--ssl-model", default="TencentGameMate/chinese-hubert-base", help="情绪特征模型")
    p_all.add_argument("--device", default="cuda", help="情绪特征设备")
    p_all.add_argument("--projection-seed", type=int, default=20241115, help="投影种子")
    p_all.add_argument("--default-precision", choices=("fp16", "int8_dynamic"), default="fp16",
                       help="未命中规则的兜底精度")
    p_all.add_argument("--force", action="store_true", help="覆盖已有产物")
    p_all.add_argument("--bundle", action="store_true", help="打包 zip DLC")
    p_all.add_argument("--dry-run", action="store_true", help="演练模式")
    add_common_quant_args(p_all)
    p_all.set_defaults(func=cmd_all)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    try:
        return int(args.func(args))
    except PipelineError as exc:
        LOG.error("流水线中断：%s", exc)
        return 4
    except KeyboardInterrupt:
        LOG.warning("用户中断。")
        return 130


if __name__ == "__main__":
    sys.exit(main())
