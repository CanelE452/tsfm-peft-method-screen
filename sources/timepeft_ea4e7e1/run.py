from __future__ import annotations

import argparse
import gc
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from data_provider.data_factory import data_provider
from momentfm import MOMENTPipeline
from momentfm.data.base import TimeseriesOutputs
from momentfm.utils.masking import Masking


PROJECT_ROOT = Path(__file__).resolve().parent


class FrequencyAdapter(nn.Module):
    def __init__(self, top_k: int, d_model: int) -> None:
        super().__init__()
        if top_k < 1:
            raise ValueError(f"top_k_freq must be at least 1, got {top_k}")
        self.top_k = top_k
        self.d_model = d_model
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 4:
            raise ValueError("Expected backbone embeddings with shape [B, C, N, D].")
        batch, channels, patches, hidden = embeddings.shape
        if hidden != self.d_model:
            raise ValueError(f"Expected hidden size {self.d_model}, got {hidden}.")

        flattened = embeddings.permute(0, 1, 3, 2).contiguous().view(batch * channels * hidden, patches)
        spectrum = torch.fft.rfft(flattened, dim=1)
        keep = min(self.top_k, spectrum.shape[1])
        indices = torch.topk(spectrum.abs(), keep, dim=1).indices
        filtered = torch.zeros_like(spectrum)
        filtered.scatter_(1, indices, spectrum.gather(1, indices))
        reconstructed = torch.fft.irfft(filtered, n=patches, dim=1)
        output = reconstructed.view(batch, channels, hidden, patches).permute(0, 1, 3, 2).contiguous()
        return self.projection(output)


class ChannelAdapter(nn.Module):
    def __init__(self, d_model: int, n_channels: int, hidden_dim_ratio: int) -> None:
        super().__init__()
        if hidden_dim_ratio < 1:
            raise ValueError("hidden_dim_ratio must be a positive integer.")
        self.d_model = d_model
        self.n_channels = n_channels
        hidden_dim = d_model // hidden_dim_ratio
        if hidden_dim < 1:
            raise ValueError("hidden_dim_ratio is too large for the selected model.")

        self.down_projection = nn.Linear(d_model * 2, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.up_projections = nn.ModuleList(nn.Linear(hidden_dim, d_model) for _ in range(n_channels))
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, backbone_out: torch.Tensor, frequency_out: torch.Tensor) -> torch.Tensor:
        if backbone_out.shape != frequency_out.shape:
            raise ValueError("Backbone and frequency-adapter embeddings must have the same shape.")
        if backbone_out.shape[1] != self.n_channels or backbone_out.shape[-1] != self.d_model:
            raise ValueError("Input embedding shape does not match the initialized channel adapter.")

        combined = torch.cat([backbone_out, frequency_out], dim=-1)
        features = self.dropout(self.activation(self.down_projection(combined)))
        output = [projection(features[:, channel]) for channel, projection in enumerate(self.up_projections)]
        return self.layer_norm(torch.stack(output, dim=1))


class TimePEFTPipeline(nn.Module):
    """MOMENT forecasting pipeline specialized to the timepeft method."""

    def __init__(
        self,
        base_model: MOMENTPipeline,
        frequency_adapter: FrequencyAdapter,
        channel_adapter: ChannelAdapter,
    ) -> None:
        super().__init__()
        self.normalizer = base_model.normalizer
        self.tokenizer = base_model.tokenizer
        self.patch_embedding = base_model.patch_embedding
        self.encoder = base_model.encoder
        self.head = base_model.head
        self.frequency_adapter = frequency_adapter
        self.channel_adapter = channel_adapter

        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in (self.frequency_adapter, self.channel_adapter, self.head):
            for parameter in module.parameters():
                parameter.requires_grad = True
        for name, parameter in self.encoder.named_parameters():
            if "lora_" in name:
                parameter.requires_grad = True

    def forward(self, x_enc: torch.Tensor, input_mask: torch.Tensor | None = None) -> TimeseriesOutputs:
        batch, channels, time_steps = x_enc.shape
        if input_mask is None:
            input_mask = torch.ones(batch, time_steps, device=x_enc.device, dtype=torch.bool)

        normalized = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        patches = self.patch_embedding(self.tokenizer(normalized), mask=input_mask)
        batch, channels, n_patches, hidden = patches.shape
        encoder_input = patches.reshape(batch * channels, n_patches, hidden)
        attention_mask = Masking.convert_seq_to_patch_view(
            input_mask, self.patch_embedding.patch_len
        ).repeat_interleave(channels, dim=0)
        backbone_out = self.encoder(inputs_embeds=encoder_input, attention_mask=attention_mask)
        backbone_embeddings = backbone_out.last_hidden_state.view(batch, channels, n_patches, hidden)

        frequency_embeddings = self.frequency_adapter(backbone_embeddings)
        adapted_embeddings = self.channel_adapter(backbone_embeddings, frequency_embeddings)
        forecast = self.head(adapted_embeddings)
        return TimeseriesOutputs(input_mask=input_mask, forecast=self.normalizer(x=forecast, mode="denorm"))


def build_model(args: argparse.Namespace, n_channels: int, prediction_length: int, device: torch.device) -> nn.Module:
    model_specs = {
        "small": ("AutonLab/MOMENT-1-small", 512),
        "base": ("AutonLab/MOMENT-1-base", 768),
    }
    model_name, d_model = model_specs[args.modelsize]
    base_model = MOMENTPipeline.from_pretrained(
        model_name,
        model_kwargs={
            "task_name": "forecasting",
            "forecast_horizon": prediction_length,
            "head_dropout": 0.1,
            "weight_decay": 0,
            "freeze_encoder": True,
            "freeze_embedder": True,
            "freeze_head": False,
            "seq_len": args.seq_len,
        },
    ).to(device)
    base_model.init()

    target_modules = [name for name in ("q", "k", "v", "o") if name in args.target_modules]
    if not target_modules:
        raise ValueError("target_modules must contain at least one of q, k, v, or o.")
    base_model.encoder = get_peft_model(
        base_model.encoder,
        LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=args.rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
        ),
    )
    model = TimePEFTPipeline(
        base_model=base_model,
        frequency_adapter=FrequencyAdapter(args.top_k_freq, d_model),
        channel_adapter=ChannelAdapter(
            d_model=d_model,
            n_channels=n_channels,
            hidden_dim_ratio=args.hidden_dim_ratio,
        ),
    ).to(device)
    return model


def input_mask(time_features: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.ones_like(time_features[:, :, 0], dtype=torch.bool, device=device)


def autocast_context(device: torch.device):
    return torch.autocast(device_type="cuda", enabled=True) if device.type == "cuda" else nullcontext()


def validation_loss(model: nn.Module, loader, device: torch.device, loss_fn: nn.Module) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for seq_x, seq_y, seq_x_mark, _ in loader:
            values = seq_x.permute(0, 2, 1).float().to(device)
            targets = seq_y.float().to(device)
            with autocast_context(device):
                predictions = model(x_enc=values, input_mask=input_mask(seq_x_mark, device)).forecast.permute(0, 2, 1)
                losses.append(loss_fn(predictions, targets).item())
    return float(np.mean(losses))


def train(model: nn.Module, train_loader, val_loader, args: argparse.Namespace, device: torch.device) -> nn.Module:
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    loss_fn = nn.MSELoss().to(device)
    best_loss = float("inf")
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for seq_x, seq_y, seq_x_mark, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            values = seq_x.permute(0, 2, 1).float().to(device)
            targets = seq_y.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                predictions = model(x_enc=values, input_mask=input_mask(seq_x_mark, device)).forecast.permute(0, 2, 1)
                loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        current_val_loss = validation_loss(model, val_loader, device, loss_fn)
        print(f"Epoch {epoch}: train_mse={np.mean(losses):.6f}, val_mse={current_val_loss:.6f}")
        if current_val_loss < best_loss - args.early_stopping_delta: # For stability
            best_loss = current_val_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break
        scheduler.step()
    return model


def evaluate(model: nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    truths, predictions = [], []
    with torch.no_grad():
        for seq_x, seq_y, seq_x_mark, _ in loader:
            values = seq_x.permute(0, 2, 1).float().to(device)
            with autocast_context(device):
                forecast = model(x_enc=values, input_mask=input_mask(seq_x_mark, device)).forecast.permute(0, 2, 1)
            truths.append(seq_y.float().cpu().numpy())
            predictions.append(forecast.float().cpu().numpy())
    y_true = np.concatenate(truths)
    y_pred = np.concatenate(predictions)
    metrics = {"mse": float(np.mean((y_true - y_pred) ** 2)), "mae": float(np.mean(np.abs(y_true - y_pred)))}
    print(f"MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TimePEFT forecasting experiment.")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--modelsize", choices=("small", "base"), default="base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--early-stopping-delta", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora-alpha", "--lora_alpha", dest="lora_alpha", type=int, default=32)
    parser.add_argument("--target-modules", "--target_modules", dest="target_modules", default="qkv", help="Combination of q, k, v, o.")
    parser.add_argument("--top-k-freq", "--top_k_freq", dest="top_k_freq", type=int, default=3)
    parser.add_argument("--hidden-dim-ratio", "--hidden_dim_ratio", dest="hidden_dim_ratio", type=int, default=2)

    parser.add_argument(
        "--data-root", "--forecast-base-path", "--forecast_base_path",
        dest="forecast_base_path", type=Path, required=True,
        help="Directory containing the forecasting CSV file.",
    )
    parser.add_argument("--data-path", "--data_path", dest="data_path", default="ETTh1.csv", help="CSV filename inside --data-root.")
    parser.add_argument("--dataset-name", "--dataset_name", dest="dataset_name", choices=("ETTh1", "ETTh2", "ETTm1", "ETTm2", "custom"), default="ETTh1")
    parser.add_argument("--model-id", "--model_id", dest="model_id", default="ETTh1")
    parser.add_argument("--seq-len", "--seq_len", dest="seq_len", type=int, default=96)
    parser.add_argument("--label-len", "--label_len", dest="label_len", type=int, default=0)
    parser.add_argument("--pred-len", "--pred_len", dest="pred_len", type=int, default=96)
    parser.add_argument("--features", choices=("S", "M", "MS"), default="M")
    parser.add_argument("--target", default="OT")
    parser.add_argument("--freq", default="h")
    parser.add_argument("--embed", default="timeF")
    parser.add_argument("--seasonal-patterns", dest="seasonal_patterns", default="Yearly")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=128)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=0)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, default=PROJECT_ROOT / "results-forecasting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rank < 1 or args.lora_alpha < 1:
        raise ValueError("rank and lora_alpha must both be positive.")
    if args.seq_len % 8 != 0:
        raise ValueError("seq_len must be a multiple of the MOMENT patch length (8).")
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda":
        print("CUDA is unavailable; using CPU.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.task_name = "forecasting"
    args.data = args.dataset_name
    args.root_path = str(args.forecast_base_path.expanduser().resolve())
    data_file = Path(args.root_path) / args.data_path
    if not data_file.is_file():
        raise FileNotFoundError(f"Dataset file not found: {data_file}")
    args.augmentation_ratio = 0
    train_data, train_loader = data_provider(args, "train")
    _, val_loader = data_provider(args, "val")
    _, test_loader = data_provider(args, "test")
    first_batch = next(iter(train_loader))
    prediction_length = first_batch[1].shape[1]
    n_channels = first_batch[0].shape[2]
    del train_data, first_batch

    model = build_model(args, n_channels, prediction_length, device)
    model = train(model, train_loader, val_loader, args, device)
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"sl{args.seq_len}_pl{prediction_length}.json"
    results = {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved results: {result_path}")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
