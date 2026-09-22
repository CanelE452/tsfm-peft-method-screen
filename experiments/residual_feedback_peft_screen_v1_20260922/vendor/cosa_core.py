# Extracted unmodified SimpleOutputAdapter class from bigbases/COSA_ICLR2026.
# Revision 527c0feb9e997dd85af485ee027616b446e4ae77, blob 09c0aafbbcc69ea0c65b49e869b575bcf5809214.
# CC BY-NC-SA 4.0; see COSA_LICENSE.txt. Imports isolated; trailing whitespace stripped; class logic unchanged.
import torch
from torch import nn
ADAPTER_TYPES = ('Linear', 'MLP')

class SimpleOutputAdapter(nn.Module):
    def __init__(self, pred_len: int, buffer_context_size: int = 5, n_vars: int = 1,
                 var_wise_gating: bool = False, adapter_type: str = 'Linear',
                 hidden_dim: int = 64):
        super().__init__()
        self.pred_len = pred_len
        self.buffer_context_size = buffer_context_size
        self.n_vars = n_vars
        self.var_wise = var_wise_gating
        self.adapter_type = adapter_type
        self.hidden_dim = hidden_dim

        assert adapter_type in ADAPTER_TYPES, \
            f"ADAPTER_TYPE must be one of {list(ADAPTER_TYPES)}, got {adapter_type!r}"

        input_dim = self.pred_len + self.buffer_context_size
        output_dim = self.pred_len

        if self.var_wise:
            self.fc_layers = nn.ModuleList([
                self._build_stack(input_dim, output_dim) for _ in range(n_vars)
            ])
            self.gate = nn.Parameter(torch.zeros(n_vars))
        else:
            self.fc = self._build_stack(input_dim, output_dim)
            self.gate = nn.Parameter(torch.zeros(1))
        self._initialize_parameters()

    def _build_stack(self, input_dim: int, output_dim: int):

        if self.adapter_type == 'Linear':
            return nn.Linear(input_dim, output_dim)

        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, output_dim),
        )

    def _initialize_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)

    def forward(self, y: torch.Tensor, context_data: torch.Tensor = None):
        if context_data is None:
            return y

        batch_size, pred_len, n_vars = y.shape

        if self.var_wise:
            corrections = []
            for var_idx in range(n_vars):
                y_var = y[:, :, var_idx]

                combined_input = torch.cat([y_var, context_data], dim=-1)


                correction_var = self.fc_layers[var_idx](combined_input)
                corrections.append(correction_var.unsqueeze(-1))

            correction = torch.cat(corrections, dim=-1)

            gating_factor = torch.tanh(self.gate).unsqueeze(0).unsqueeze(0)

        else:
            y_flattened = y.transpose(1, 2).contiguous().view(batch_size * n_vars, pred_len)
            context_repeated = context_data.unsqueeze(1).repeat(1, n_vars, 1).view(batch_size * n_vars, -1)
            combined_input = torch.cat([y_flattened, context_repeated], dim=-1)
            correction = self.fc(combined_input)
            correction = correction.view(batch_size, n_vars, pred_len).transpose(1, 2)
            gating_factor = torch.tanh(self.gate)

        return y + gating_factor * correction
