import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from .linear import LRU


@dataclass
class DLRUConfig:
    d_model: int = 10
    d_state: int = 64
    n_layers: int = 6
    dropout: float = 0.0
    bias: bool = True
    rmin: float = 0.0
    rmax: float = 1.0
    max_phase: float = math.pi
    ff: str = "GLU"


class MLP(nn.Module):
    """ Standard Transformer MLP """
    def __init__(self, config: DLRUConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GLU(nn.Module):
    """ The static nonlinearity used in the S4 paper"""
    def __init__(self, config: DLRUConfig):
        super().__init__()
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.output_linear = nn.Sequential(
            nn.Linear(config.d_model, 2 * config.d_model),#nn.Conv1d(config.d_model, 2 * config.d_model, kernel_size=1),
            nn.GLU(dim=-1),
        )

    def forward(self, x):
        x = self.dropout(self.activation(x))
        x = self.output_linear(x)
        return x
    
class DWNBlock(nn.Module):
    def __init__(self, config: DLRUConfig):
        super().__init__()
        self.ln = nn.LayerNorm(config.d_model, bias=config.bias)
        self.lru = LRU(config.d_model, config.d_model, config.d_state,
                        rmin=config.rmin, rmax=config.rmax, max_phase=config.max_phase)
        #self.ff = MLP(config)  # feedforward layer, or GLU
        match config.ff:
            case "GLU":
                self.ff = GLU(config)
            case "MLP":
                self.ff = MLP(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, state=None, mode="scan"):

        z = x
        z = self.ln(z)  # prenorm

        z = self.lru(z, state, mode) 

        z = self.ff(z) # non-linearity: MLP or GLU
        z = self.dropout(z)

        x = z + x # residual connection

        return x


class DLRU(nn.Module):
    def __init__(self, n_u, n_y, config: DLRUConfig):
        super().__init__()
        self.encoder = nn.Linear(n_u, config.d_model)
        self.blocks = nn.ModuleList(
            [DWNBlock(config) for _ in range(config.n_layers)]
        )
        self.decoder = nn.Linear(config.d_model, n_y)

    def forward(self, u, state=None, mode="scan"):
        x = self.encoder(u)

        for layer, block in enumerate(self.blocks):
            state_block = state[layer] if state is not None else None
            x = block(x, state=state_block, mode=mode)
        x = self.decoder(x)
        return x

    def configure_optimizer(self, weight_decay, learning_rate):#, betas, device_type):

        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer