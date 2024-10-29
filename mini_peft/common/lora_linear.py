import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import is_bitsandbytes_available

from moe_peft.executors import executor

from .abstracts import LLMMoeBlock
from .config import LLMModelInput, LoraConfig

if is_bitsandbytes_available():
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit, Linear8bitLt
else:
    from moe_peft.utils import Linear8bitLt, Linear4bit

from typing import Any, Dict, List, Tuple


def dequantize_bnb_weight(weight: torch.nn.Parameter, state=None):
    # BNB requires CUDA weights
    device = weight.device
    is_cpu = device.type == torch.device("cpu").type
    if is_cpu:
        weight = weight.to(torch.device("cuda"))

    cls_name = weight.__class__.__name__
    if cls_name == "Params4bit":
        dequantized = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
        if is_cpu:
            dequantized = dequantized.to(device)
        return dequantized

    if state.SCB is None:
        state.SCB = weight.SCB

    im = torch.eye(weight.data.shape[-1]).contiguous().half().to(weight.device)
    im, imt, SCim, SCimt, coo_tensorim = bnb.functional.double_quant(im)
    im, Sim = bnb.functional.transform(im, "col32")
    if state.CxB is None:
        state.CxB, state.SB = bnb.functional.transform(
            weight.data, to_order=state.formatB
        )
    out32, Sout32 = bnb.functional.igemmlt(im, state.CxB, Sim, state.SB)
    dequantized = bnb.functional.mm_dequant(
        out32, Sout32, SCim, state.SCB, bias=None
    ).t()
    if is_cpu:
        dequantized = dequantized.to(device)
    return dequantized


def dequantize_module_weight(module: torch.nn.Module) -> torch.nn.Parameter:
    if hasattr(module, "W_q"):  # For handling HQQ quantized weight
        weight = module.dequantize()
        return weight

    weight = module.weight
    if not isinstance(weight, torch.nn.Parameter):
        raise TypeError(
            f"Input weight should be of type nn.Parameter, got {type(weight)} instead"
        )

    cls_name = weight.__class__.__name__
    if cls_name not in ("Params4bit", "Int8Params"):
        return weight

    quant_state = getattr(module, "state", None)
    device = weight.device
    is_cpu = device.type == torch.device("cpu").type
    weight = dequantize_bnb_weight(weight, state=quant_state)  # no-op if not bnb
    if is_cpu:
        # dequantize_bnb_weight for 8bit moves the device in-place, thus we need to move it back to CPU if necessary
        module.weight = module.weight.to(device)
    return weight


g_cached_range_tensor: Dict[torch.device, torch.Tensor] = {}
# also max batch size
g_max_range = 128


def get_range_tensor(device: torch.device, batch_size: int = 1024):
    global g_cached_range_tensor
    global g_max_range
    if device not in g_cached_range_tensor or batch_size > g_max_range:
        g_max_range = g_max_range if g_max_range > batch_size else batch_size
        g_cached_range_tensor[device] = torch.arange(
            0, g_max_range, step=1, device=device
        )
    return g_cached_range_tensor[device]


class LoraFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        result: torch.Tensor,
        data: torch.Tensor,
        input_args: LLMModelInput,
        dropouts: List[float],
        scalings: List[float],
        *args,
    ):
        # the lora module is f32 precision
        data = data.to(torch.float32)

        save_inputs: Tuple[torch.Tensor | None, ...] = (data,)

        lora_range = get_range_tensor(data.device, data.shape[0])
        for lora_a, lora_b, lora_config, dropout, scaling in zip(
            args[::2],
            args[1::2],
            input_args.batch_configs_,
            dropouts,
            scalings,
        ):
            assert not ((lora_a is None) ^ (lora_b is None))
            if lora_a is None and lora_b is None:
                save_inputs += (None, None, None)
                continue

            assert not ((lora_a.requires_grad) ^ (lora_b.requires_grad))
            if not lora_a.requires_grad and not lora_b.requires_grad:
                save_inputs += (None, None, None)
                continue

            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_

            # must ensure the dropout is not zero
            # is dropout == 0, dropdata is a data's referece, so the data will be changed
            assert dropout > 0.0

            drop_data = F.dropout(data[start_idx:end_idx], p=dropout)
            drop_data.mul_(scaling)
            drop_data = drop_data @ lora_a.transpose(0, 1)
            lora_data = drop_data @ lora_b.transpose(0, 1)

            lora_data = lora_data.to(result.dtype)

            result.index_add_(0, lora_range[start_idx:end_idx], lora_data)

            save_inputs += (lora_a, lora_b, drop_data)

        ctx.input_args = input_args
        ctx.dropouts = dropouts
        ctx.scalings = scalings
        ctx.save_for_backward(*save_inputs)

        return result

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_output: torch.Tensor = grad_outputs[0]
        grad_result = None
        grad_data: torch.Tensor | None = None
        grad_input_args = None
        grad_dropouts = None
        grad_scalings = None
        grad_loras: Tuple[torch.Tensor | None, ...] = ()

        data, *loras = ctx.saved_tensors

        if ctx.needs_input_grad[0]:
            grad_result = grad_output
        if ctx.needs_input_grad[1]:
            grad_data = executor.init_tensor(data)

        # the lora module is fp32 precision
        grad_output = grad_output.to(torch.float32)
        lora_range = get_range_tensor(
            grad_output.device, batch_size=grad_output.shape[0]
        )
        for lora_a, lora_b, drop_data, dropout, scaling, lora_config in zip(
            loras[::3],
            loras[1::3],
            loras[2::3],
            ctx.dropouts,
            ctx.scalings,
            ctx.input_args.batch_configs_,
        ):
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_
            assert not ((lora_a is None) ^ (lora_b is None))
            if lora_a is None and lora_b is None:
                grad_loras += (None, None)
                if grad_data is not None:
                    executor.index_fill(grad_data, 0, lora_range[start_idx:end_idx], 0)
                continue

            # lora_data shape is batch_size * seq_len * in_dim
            lora_data = data[start_idx:end_idx]
            # grad_y shape is batch_size * seq_len * out_dim
            grad_y = grad_output[start_idx:end_idx]

            # drop_data shape is batch_size * seq_len * r

            # bstage shape is batch_size * seq_len * r
            bstage = grad_y @ lora_b
            bstage *= scaling / (1 - dropout)

            grad_a = torch.sum(bstage.transpose(1, 2) @ lora_data, dim=0)
            grad_b = torch.sum(grad_y.transpose(1, 2) @ drop_data, dim=0)
            grad_loras += (grad_a, grad_b)

            # grad_data shape is batch_size * seq_len * in_dim
            if grad_data is not None:
                grad_x = bstage @ lora_a
                executor.index_copy(grad_data, 0, lora_range[start_idx:end_idx], grad_x)

        return (
            grad_result,
            grad_data,
            grad_input_args,
            grad_dropouts,
            grad_scalings,
            *grad_loras,
        )
    

class Lora(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        shape: Tuple[int, int],
        config: LoraConfig,
        device: str,
    ):

        super().__init__()

        self.base_layer_ = base_layer
        self.device_ = torch.device(device)

        self.initializer_ = config.lora_init_
        self.r_ = config.lora_r_
        self.alpha_ = config.lora_alpha_

        if config.use_rslora_:
            self.scaling_ = self.alpha_ / math.sqrt(self.r_)
        else:
            self.scaling_ = self.alpha_ / self.r_

        self.in_features_, self.out_features_ = shape

        assert config.lora_dropout_ > 0.0
        self.dropout_ = nn.Dropout(p=config.lora_dropout_)

        self.lora_a_ = nn.Linear(
            self.in_features_,
            self.r_,
            bias=False,
            dtype=torch.float32,
            device=self.device_,
        )
        self.lora_b_ = nn.Linear(
            self.r_,
            self.out_features_,
            bias=False,
            dtype=torch.float32,
            device=self.device_,
        )

        self.magnitude_vector_: nn.Parameter = None

    def reset_parameters(self, lora_tensor=(None, None)) -> None:
        # if the lora_tensor is not (None, None), use it to init the lora weight
        assert isinstance(lora_tensor, Tuple)
        assert len(lora_tensor) == 2
        assert ((lora_tensor[0] is None) and (lora_tensor[1] is None)) or (
            isinstance(lora_tensor[0], torch.Tensor)
            and isinstance(lora_tensor[1], torch.Tensor)
        )

        if lora_tensor == (None, None):
            if self.initializer_ == "original":
                nn.init.kaiming_uniform_(self.lora_a_.weight, a=math.sqrt(5))
            elif self.initializer_ == "gaussian":
                nn.init.normal_(self.lora_a_.weight, std=1 / self.r_)
            else:
                raise ValueError(f"Unknown initialization {self.initializer_}")
            nn.init.zeros_(self.lora_b_.weight)
        else:
            with torch.no_grad():
                self.lora_a_.weight.copy_(lora_tensor[0])
                self.lora_b_.weight.copy_(lora_tensor[1])

    def lora_forward(self, hidden_states: torch.Tensor):
        return (
            self.lora_b_(self.lora_a_(self.dropout_(hidden_states.to(torch.float32))))
            * self.scaling_
        )

    def forward(
        self,
        residual: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        result_lora = self.lora_forward(hidden_states)
        return residual + result_lora.to(residual.dtype)