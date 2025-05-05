#!/usr/bin/env python3

import json
import logging
import warnings
import os
import yaml
import math
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from jsonargparse import ActionConfigFile, ArgumentParser, namespace_to_dict
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger

from comet.models import RankingMetric, ReferencelessRegression, RegressionMetric, UnifiedMetric

torch.set_float32_matmul_precision('high')
logger = logging.getLogger(__name__)

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=1e-8):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        super().__init__(optimizer)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            warmup_factor = self.warmup_start_lr * (1 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return [base_lr * cosine_decay for base_lr in self.base_lrs]

class SparseLinear(nn.Module):
    def __init__(self, linear, sparsity=0.9, mask_init="random"):
        super().__init__()
        self.linear = linear
        self.weight_mask = nn.Parameter(torch.ones_like(linear.weight))
        self.sparsity = sparsity
        
        if mask_init == "random":
            with torch.no_grad():
                mask = torch.rand_like(self.weight_mask)
                threshold = torch.quantile(mask, self.sparsity)
                self.weight_mask.data = (mask > threshold).float()
    
    def forward(self, x):
        weight = self.linear.weight * self.weight_mask
        return nn.functional.linear(x, weight, self.linear.bias)

def apply_smt(model, sparsity=0.9):
    replaced_layers = 0
    
    for name, module in list(model.encoder.model.named_modules()):
        if isinstance(module, nn.Linear):
            parts = name.split('.')
            if len(parts) < 2:
                continue
                
            parent_module = model.encoder.model
            for i in range(len(parts) - 1):
                try:
                    parent_module = getattr(parent_module, parts[i])
                except (AttributeError, IndexError):
                    break
                    
            try:
                leaf_name = parts[-1]
                original_linear = getattr(parent_module, leaf_name)
                setattr(parent_module, leaf_name, SparseLinear(original_linear, sparsity=sparsity))
                replaced_layers += 1
                
                if replaced_layers >= 8:
                    break
            except (AttributeError, IndexError):
                continue
    
    if replaced_layers == 0:
        logger.warning("No layers replaced with SMT. Using model as is.")
    else:
        logger.info(f"Applied SMT to {replaced_layers} linear layers with sparsity {sparsity}")
    
    return model

def read_arguments():
    p = ArgumentParser(description="Train COMET models with SMT")
    p.add_argument("--seed_everything", type=int, default=12)
    p.add_argument("--cfg", action=ActionConfigFile)
    p.add_subclass_arguments(RegressionMetric, "regression_metric")
    p.add_subclass_arguments(ReferencelessRegression, "referenceless_regression_metric")
    p.add_subclass_arguments(RankingMetric, "ranking_metric")
    p.add_subclass_arguments(UnifiedMetric, "unified_metric")
    p.add_subclass_arguments(EarlyStopping, "early_stopping")
    p.add_subclass_arguments(ModelCheckpoint, "model_checkpoint")
    p.add_subclass_arguments(Trainer, "trainer")
    p.add_argument("--load_from_checkpoint", default=None)
    p.add_argument("--strict_load", action="store_true")
    
    p.add_argument("--smt_sparsity", type=float, default=0.9)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--test_file", type=str, default="test.csv")
    
    return p

def initialize_trainer(cfg):
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_dir = f"checkpoints/{date}"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    cb_args = namespace_to_dict(cfg.model_checkpoint.init_args)
    cb_args.pop('dirpath', None)
    cb = ModelCheckpoint(dirpath=ckpt_dir, **cb_args)
    es = EarlyStopping(**namespace_to_dict(cfg.early_stopping.init_args))
    tr_args = namespace_to_dict(cfg.trainer.init_args)
    tr_args['callbacks'] = [es, cb, LearningRateMonitor(logging_interval="step")]
    
    if cfg.referenceless_regression_metric is not None:
        import wandb
        wandb.init(project="cometh", id=date, name=date, 
                  config=cfg.referenceless_regression_metric.init_args, mode='offline')
    
    wl = WandbLogger(project="cometh", name=date, version=date, checkpoint_name=date, offline=True)
    tr_args['logger'] = [wl]
    
    return Trainer(**tr_args), ckpt_dir, date

def save_hparams(model, cfg, path, date):
    name, args = None, None
    if hasattr(cfg, 'regression_metric') and cfg.regression_metric: 
        name, args = "regression_metric", cfg.regression_metric.init_args
    elif hasattr(cfg, 'referenceless_regression_metric') and cfg.referenceless_regression_metric:
        name, args = "referenceless_regression_metric", cfg.referenceless_regression_metric.init_args
    elif hasattr(cfg, 'ranking_metric') and cfg.ranking_metric:
        name, args = "ranking_metric", cfg.ranking_metric.init_args
    elif hasattr(cfg, 'unified_metric') and cfg.unified_metric:
        name, args = "unified_metric", cfg.unified_metric.init_args
    
    if name and args:
        h = {
            'model_type': name,
            'model_config': namespace_to_dict(args),
            'smt': {
                'sparsity': getattr(cfg, 'smt_sparsity', 0.9),
            },
            'training': {
                'warmup_ratio': getattr(cfg, 'warmup_ratio', 0.1),
                'seed': cfg.seed_everything,
                'test_file': getattr(cfg, 'test_file', '')
            },
            'trainer': namespace_to_dict(cfg.trainer.init_args)
        }
        
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'hparams.yaml'), 'w') as f:
            yaml.dump(h, f)
        
        cd = Path(f"cometh/{date}")
        cd.mkdir(parents=True, exist_ok=True)
        with open(cd / 'hparams.yaml', 'w') as f:
            yaml.dump(h, f)
        
        if hasattr(model, 'hparams'):
            for k, v in h.items():
                if k not in model.hparams:
                    model.hparams[k] = v
                    
        return h
    return {}

def initialize_model(cfg):
    if hasattr(cfg, 'regression_metric') and cfg.regression_metric:
        cls, args = RegressionMetric, cfg.regression_metric
    elif hasattr(cfg, 'referenceless_regression_metric') and cfg.referenceless_regression_metric:
        cls, args = ReferencelessRegression, cfg.referenceless_regression_metric
    elif hasattr(cfg, 'ranking_metric') and cfg.ranking_metric:
        cls, args = RankingMetric, cfg.ranking_metric
    elif hasattr(cfg, 'unified_metric') and cfg.unified_metric:
        cls, args = UnifiedMetric, cfg.unified_metric
    else:
        raise ValueError("No model configuration provided")
    
    if cfg.load_from_checkpoint:
        logger.info(f"Loading from checkpoint: {cfg.load_from_checkpoint}")
        model = cls.load_from_checkpoint(
            cfg.load_from_checkpoint, 
            strict=cfg.strict_load, 
            **namespace_to_dict(args.init_args)
        )
    else:
        model = cls(**namespace_to_dict(args.init_args))
    
    sparsity = getattr(cfg, 'smt_sparsity', 0.9)
    model = apply_smt(model, sparsity=sparsity)
    
    if getattr(cfg, 'warmup_ratio', 0) > 0:
        orig_opt = model.configure_optimizers
        
        def custom_opt(*args, **kwargs):
            opt, sch = orig_opt(*args, **kwargs)
            me = getattr(cfg.trainer.init_args, 'max_epochs', 5)
            we = int(me * cfg.warmup_ratio)
            return opt, [CosineWarmupScheduler(opt[0], we, me)]
        
        model.configure_optimizers = custom_opt
    
    return model

def train_command():
    parser = read_arguments()
    cfg = parser.parse_args()
    seed_everything(cfg.seed_everything)
    
    trainer, ckpt, date = initialize_trainer(cfg)
    model = initialize_model(cfg)
    save_hparams(model, cfg, ckpt, date)
    
    warnings.filterwarnings(
        "ignore", 
        category=UserWarning, 
        message=".*Consider increasing the value of the `num_workers` argument.*"
    )
    
    trainer.fit(model)
    save_hparams(model, cfg, ckpt, date)
    
    tf = getattr(cfg, 'test_file', None)
    if tf and os.path.exists(tf):
        import pandas as pd
        
        df = pd.read_csv(tf)
        inp = df.drop(columns=['score']).to_dict('records')
        res = model.predict(inp, batch_size=8, gpus=1)['scores']
        
        pearson = df['score'].corr(torch.tensor(res), method='pearson')
        spearman = df['score'].corr(torch.tensor(res), method='spearman')
        kendall = df['score'].corr(torch.tensor(res), method='kendall')
        
        print(f"Test Results:")
        print(f"  Pearson: {pearson:.4f}")
        print(f"  Spearman: {spearman:.4f}")
        print(f"  Kendall: {kendall:.4f}")
        
        with open(os.path.join(ckpt, 'evaluation.json'), 'w') as f:
            json.dump({
                "pearson_correlation": float(pearson),
                "spearman_correlation": float(spearman),
                "kendall_correlation": float(kendall)
            }, f, indent=4)

if __name__ == "__main__":
    train_command()