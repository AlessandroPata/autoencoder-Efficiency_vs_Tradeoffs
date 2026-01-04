#!/usr/bin/env python3
"""
Main Script - Auto-Encoder Anomaly Detection Study
Paper: "A comprehensive study of auto-encoders for anomaly detection"
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import DEVICE, MODEL_CONFIGS, EXPECTED_RESULTS
from utils.data_loader import get_anomaly_detection_loaders

# Import all models
from models.dae import DAE
from models.sae import SAE
from models.cae import CAE
from models.vae import VAE
from models.beta_vae import BetaVAE
from models.advae import AdVAE
from models.cvae import CVAE
from models.vqvae import VQVAE
from models.others import IWAE, PAE, RDA


MODEL_CLASSES = {
    "dae": DAE,
    "sae": SAE,
    "cae": CAE,
    "vae": VAE,
    "beta_vae": BetaVAE,
    "advae": AdVAE,
    "cvae": CVAE,
    "vqvae": VQVAE,
    "iwae": IWAE,
    "pae": PAE,
    "rda": RDA,
}
ALL_MODELS = list(MODEL_CLASSES.keys())


# -------------------------
# Model construction
# -------------------------
def create_model(model_name: str) -> nn.Module:
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model: {model_name}. Available: {ALL_MODELS}")

    cfg = MODEL_CONFIGS.get(model_name, {}).copy()
    cls = MODEL_CLASSES[model_name]

    # Keep your explicit defaults (paper-ish), but allow config override
    if model_name in ("dae", "sae", "cae", "vae", "beta_vae", "advae", "cvae", "iwae", "pae", "rda"):
        input_dim = cfg.get("input_dim", 784)
        hidden_dims = cfg.get("hidden_dims", [512, 256, 128])

    if model_name == "dae":
        return cls(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=cfg.get("latent_dim", 32),
            noise_factor=cfg.get("noise_factor", 0.20),  # start at 20%, schedule will raise to 52%
        )

    if model_name == "sae":
        return cls(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=cfg.get("latent_dim", 32),
            sparsity_weight=cfg.get("sparsity_weight", 1e-3),
            sparsity_target=cfg.get("sparsity_target", 0.45),
        )

    if model_name == "cae":
        return cls(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=cfg.get("latent_dim", 32),
            lambda_=cfg.get("lambda_", 1e-4),
        )

    if model_name == "vae":
        return cls(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=cfg.get("latent_dim", 2),
            kl_weight=cfg.get("kl_weight", 1.0),
        )

    if model_name == "beta_vae":
        return cls(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=cfg.get("latent_dim", 2),
            beta=cfg.get("beta", 1.5),
        )

    if model_name == "advae":
        # NOTE: matching paper requires the *training procedure* (two-step + T) in the trainer/model
        return cls(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=cfg.get("latent_dim", 2),
            **{k: v for k, v in cfg.items() if k not in {"input_dim", "hidden_dims", "latent_dim"}}
        )

    if model_name == "cvae":
        return cls(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=cfg.get("latent_dim", 2),
            num_classes=cfg.get("num_classes", 10),
        )

    if model_name == "vqvae":
        return cls(
            in_channels=cfg.get("in_channels", 1),
            hidden_dims=cfg.get("hidden_dims", [128, 256]),
            num_embeddings=cfg.get("num_embeddings", 512),
            embedding_dim=cfg.get("embedding_dim", 64),
            commitment_cost=cfg.get("commitment_cost", 0.25),
        )

    if model_name == "iwae":
        return cls(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=cfg.get("latent_dim", 2),
            num_samples=cfg.get("num_samples", 50),
        )

    if model_name == "pae":
        return cls(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=cfg.get("latent_dim", 2),
            beta=cfg.get("beta", 1.0),
            C=cfg.get("C", 0.5),
        )

    if model_name == "rda":
        return cls(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=cfg.get("latent_dim", 32),
            lambda_=cfg.get("lambda_", 1e-3),
        )

    return cls(**cfg)


# -------------------------
# Helpers: score aligned to paper
# -------------------------
def _recon_per_sample(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """
    Per-sample MSE recon error. Handles cases where recon batch != x batch (e.g., IWAE K*B).
    Returns: tensor [B]
    """
    b = x.size(0)
    x_flat = x.view(b, -1)

    recon_flat = recon.view(recon.size(0), -1)

    # If recon is (K*B, D), fold back to (B, K, D) and average recon error over K
    if recon_flat.size(0) != b:
        if recon_flat.size(0) % b != 0:
            raise RuntimeError(f"Recon batch {recon_flat.size(0)} not compatible with B={b}")
        k = recon_flat.size(0) // b
        recon_flat = recon_flat.view(b, k, -1)
        x_rep = x_flat.unsqueeze(1).expand(b, k, x_flat.size(1))
        mse = torch.mean((x_rep - recon_flat) ** 2, dim=2)  # [B, K]
        return mse.mean(dim=1)  # [B]

    return torch.mean((x_flat - recon_flat) ** 2, dim=1)


def _kl_from_mu_logvar(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # KL(q(z|x) || N(0,1)) per-sample
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=1)


def compute_score_per_sample(model: nn.Module, model_name: str, x: torch.Tensor, outputs) -> torch.Tensor:
    """
    Paper-aligned score: higher => more normal (normal is positive class).
    Default: score = -(recon error).
    VAE-like: score = -(recon + beta*KL).
    SAE: score = -(recon + lambda*L1(h)) if h exists.
    CAE: score = -(recon + lambda*penalty) if penalty exists (otherwise just recon).
    """
    # Extract recon
    if isinstance(outputs, dict):
        recon = outputs.get("recon", outputs.get("reconstruction", None))
    elif isinstance(outputs, tuple):
        # common patterns: (loss, recon, ...)
        recon = outputs[1] if len(outputs) > 1 else outputs[0]
    else:
        recon = outputs

    recon_err = _recon_per_sample(x, recon)
    score = -recon_err

    # VAE family: recon + beta*KL
    if model_name in ("vae", "beta_vae", "cvae", "advae", "iwae"):
        kl = None
        beta = float(getattr(model, "beta", 1.0))

        if isinstance(outputs, dict):
            kl = outputs.get("kl_per_sample", None)
            mu = outputs.get("mu", None)
            logvar = outputs.get("logvar", None)
        else:
            mu = logvar = None

        if kl is None and (mu is not None) and (logvar is not None):
            kl = _kl_from_mu_logvar(mu, logvar)
        if kl is None:
            kl = torch.zeros_like(recon_err)

        total = recon_err + beta * kl
        score = -total

    elif model_name == "sae":
        lam = float(getattr(model, "lambda_sparsity", getattr(model, "sparsity_weight", 1e-3)))
        l1 = torch.zeros_like(recon_err)
        if isinstance(outputs, dict):
            h = outputs.get("h", None)
            if h is not None:
                l1 = h.abs().view(h.size(0), -1).mean(dim=1)
        score = -(recon_err + lam * l1)

    elif model_name == "cae":
        lam = float(getattr(model, "lambda_contractive", getattr(model, "lambda_", 1e-4)))
        penalty = torch.zeros_like(recon_err)
        if isinstance(outputs, dict):
            penalty = outputs.get("contractive_per_sample", outputs.get("penalty_per_sample", penalty))
        score = -(recon_err + lam * penalty)

    # others (dae, vqvae, pae, rda) -> recon only (paper table uses recon-style scoring)
    return score


# -------------------------
# Train / Evaluate
# -------------------------
def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    verbose: bool,
) -> Dict:
    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    history = {"loss": [], "epoch_times": []}
    t0 = time.time()

    # DAE noise schedule: 20% -> 52% (linear over epochs)
    dae_noise_start = 0.20
    dae_noise_end = 0.52

    for epoch in range(epochs):
        ep_t0 = time.time()
        total_loss = 0.0
        n = 0

        # apply DAE noise schedule (if model supports it)
        if model_name == "dae":
            noise = dae_noise_start + (dae_noise_end - dae_noise_start) * (epoch / max(1, epochs - 1))
            if hasattr(model, "noise_factor"):
                model.noise_factor = float(noise)
            if hasattr(model, "set_noise_factor"):
                model.set_noise_factor(float(noise))

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=not verbose)
        for data, labels in pbar:
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # forward
            if model_name == "cvae":
                outputs = model(data, labels.long())
            else:
                outputs = model(data)

            # loss
            if isinstance(outputs, dict) and hasattr(model, "loss_function"):
                loss_dict = model.loss_function(data, outputs)
                loss = loss_dict["loss"] if isinstance(loss_dict, dict) else loss_dict
            elif isinstance(outputs, tuple):
                loss = outputs[0]
            else:
                # fallback: plain AE recon
                recon = outputs
                loss = F.mse_loss(recon, data.view(data.size(0), -1))

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n += 1
            if verbose:
                pbar.set_postfix(loss=(total_loss / n))

        history["loss"].append(total_loss / max(1, n))
        history["epoch_times"].append(time.time() - ep_t0)

    history["total_time"] = time.time() - t0
    return history


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    model_name: str,
    test_loader: DataLoader,
    normal_class: int,
    device: torch.device,
) -> Dict:
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

    model = model.to(device)
    model.eval()

    all_scores = []
    all_ytrue = []

    for data, labels in tqdm(test_loader, desc="Evaluating"):
        data = data.to(device)
        labels = labels.to(device)

        # forward
        if model_name == "cvae":
            outputs = model(data, labels.long())
        else:
            outputs = model(data)

        score = compute_score_per_sample(model, model_name, data, outputs)  # higher = more normal
        y_true = (labels == normal_class).long()  # 1 normal, 0 anomaly

        all_scores.append(score.detach().cpu().numpy())
        all_ytrue.append(y_true.detach().cpu().numpy())

    scores = np.concatenate(all_scores, axis=0)
    y_true = np.concatenate(all_ytrue, axis=0)

    roc_auc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)

    # threshold on "normal score": pred_normal = 1 if score >= thr
    best_f1 = 0.0
    for thr in np.percentile(scores, np.linspace(0, 100, 100)):
        pred = (scores >= thr).astype(int)
        best_f1 = max(best_f1, f1_score(y_true, pred, zero_division=0))

    return {"roc_auc": float(roc_auc), "ap": float(ap), "f1": float(best_f1)}


def run_experiment(
    model_name: str,
    dataset: str,
    epochs: int,
    batch_size: int,
    normal_class: int,
    checkpoint_dir: str,
    device: torch.device,
    verbose: bool,
) -> Dict:
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} on {dataset.upper()} | normal_class={normal_class}")
    print(f"{'='*60}")

    model = create_model(model_name)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_loader, test_loader = get_anomaly_detection_loaders(
        dataset_name=dataset,
        normal_class=normal_class,
        batch_size=batch_size,
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    lr = float(MODEL_CONFIGS.get(model_name, {}).get("learning_rate", 1e-3))
    history = train_model(model, model_name, train_loader, epochs, lr, device, verbose)
    print(f"\nTraining completed in {history['total_time']:.1f} seconds")

    results = evaluate_model(model, model_name, test_loader, normal_class, device)

    print("\nResults:")
    print(f"  ROC-AUC: {results['roc_auc']:.4f}")
    print(f"  AP:      {results['ap']:.4f}")
    print(f"  F1:      {results['f1']:.4f}")

    expected = EXPECTED_RESULTS.get(dataset, {}).get(model_name, None)
    if isinstance(expected, (int, float)):
        diff = results["roc_auc"] - float(expected)
        print(f"  Paper ROC-AUC: {expected:.2f} (diff: {diff:+.4f})")

    # save checkpoint
    save_dir = os.path.join(checkpoint_dir, dataset, f"nc{normal_class}")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"{model_name}_{dataset}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "dataset": dataset,
            "normal_class": normal_class,
            "history": history,
            "results": results,
        },
        ckpt_path,
    )
    print(f"\nCheckpoint saved to {ckpt_path}")

    return {"model_name": model_name, "dataset": dataset, "history": history, "results": results}


def run_all_experiments(
    dataset: str,
    models: List[str],
    epochs: int,
    batch_size: int,
    normal_class: int,
    checkpoint_dir: str,
    results_dir: str,
    device: torch.device,
    verbose: bool,
) -> Dict:
    print("\n" + "=" * 70)
    print("AUTO-ENCODER ANOMALY DETECTION - FULL EXPERIMENT")
    print("=" * 70)
    print(f"Dataset: {dataset}")
    print(f"Models:  {', '.join(models)}")
    print(f"Epochs:  {epochs}")
    print(f"Device:  {device}")
    print("=" * 70)

    all_results = {}

    for m in models:
        try:
            all_results[m] = run_experiment(
                m, dataset, epochs, batch_size, normal_class,
                checkpoint_dir, device, verbose
            )
        except Exception as e:
            print(f"\nError training/evaluating {m}: {e}")
            import traceback
            traceback.print_exc()

    # summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<12} {'ROC-AUC':<10} {'AP':<10} {'F1':<10} {'Time(s)':<10}")
    print("-" * 70)
    for m, r in all_results.items():
        res = r["results"]
        t = r["history"]["total_time"]
        print(f"{m:<12} {res['roc_auc']:<10.4f} {res['ap']:<10.4f} {res['f1']:<10.4f} {t:<10.1f}")
    print("=" * 70)

    # compare with paper
    print("\nCOMPARISON WITH PAPER (Table 3)")
    print("-" * 70)
    print(f"{'Model':<12} {'Our AUC':<10} {'Paper':<10} {'Diff':<10}")
    print("-" * 70)
    exp = EXPECTED_RESULTS.get(dataset, {})
    for m, r in all_results.items():
        our = r["results"]["roc_auc"]
        paper = exp.get(m, None)
        if isinstance(paper, (int, float)):
            diff = our - float(paper)
            print(f"{m:<12} {our:<10.4f} {paper:<10.2f} {diff:<+10.4f}")
        else:
            print(f"{m:<12} {our:<10.4f} {'N/A':<10} {'N/A':<10}")
    print("-" * 70)

    # save json
    out_dir = os.path.join(results_dir, dataset, f"nc{normal_class}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"results_{dataset}.json")
    json_out = {
        m: {
            "roc_auc": all_results[m]["results"]["roc_auc"],
            "ap": all_results[m]["results"]["ap"],
            "f1": all_results[m]["results"]["f1"],
            "training_time": float(all_results[m]["history"]["total_time"]),
        }
        for m in all_results.keys()
    }
    with open(out_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Auto-Encoder Anomaly Detection Study")
    parser.add_argument("--dataset", choices=["mnist", "fashion_mnist"], default="mnist")
    parser.add_argument("--model", choices=ALL_MODELS)
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--normal_class", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu") if args.no_cuda else DEVICE
    epochs = 5 if args.quick else args.epochs
    verbose = not args.quiet

    if args.all:
        models = ALL_MODELS
    elif args.models:
        models = args.models
    elif args.model:
        models = [args.model]
    else:
        raise SystemExit("Specify --model or --models or --all")

    run_all_experiments(
        dataset=args.dataset,
        models=models,
        epochs=epochs,
        batch_size=args.batch_size,
        normal_class=args.normal_class,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        device=device,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
