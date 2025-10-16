# optim.py
import torch
from torch import nn, optim

def make_loss(loss_cfg: dict):
    if loss_cfg is None:
        loss_cfg = {}
    name = str(loss_cfg.get("name", "mse")).lower()
    reduction = str(loss_cfg.get("reduction", "mean")).lower()
    if name in ["mse", "mse_loss"]:
        return nn.MSELoss(reduction=reduction)
    if name in ["mae", "l1", "l1_loss"]:
        return nn.L1Loss(reduction=reduction)
    if name in ["huber", "smooth_l1"]:
        beta = float(loss_cfg.get("beta", 1.0))
        return nn.SmoothL1Loss(beta=beta, reduction=reduction)
    if name in ["bce", "binary_cross_entropy"]:
        return nn.BCEWithLogitsLoss(reduction=reduction)
    if name in ["ce", "crossentropy", "cross_entropy"]:
        return nn.CrossEntropyLoss(reduction=reduction)
    raise ValueError(f"Loss '{name}' non supportée.")

def make_optimizer(model, opt_cfg: dict):
    if opt_cfg is None:
        opt_cfg = {}
    name = str(opt_cfg.get("name", "adam")).lower()
    lr = float(opt_cfg.get("lr", 1e-3))
    wd = float(opt_cfg.get("weight_decay", 0.0))
    if name == "adam":
        betas = opt_cfg.get("betas", (0.9, 0.999))
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd, betas=tuple(betas))
    if name == "adamw":
        betas = opt_cfg.get("betas", (0.9, 0.999))
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=tuple(betas))
    if name == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.0))
        nesterov = bool(opt_cfg.get("nesterov", False))
        return optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum, nesterov=nesterov)
    if name == "rmsprop":
        momentum = float(opt_cfg.get("momentum", 0.0))
        alpha = float(opt_cfg.get("alpha", 0.99))
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum, alpha=alpha)
    if name == "adagrad":
        lr_decay = float(opt_cfg.get("lr_decay", 0.0))
        return optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd, lr_decay=lr_decay)
    raise ValueError(f"Optimiseur '{name}' non supporté.")