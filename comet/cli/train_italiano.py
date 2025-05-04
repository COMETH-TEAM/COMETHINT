#!/usr/bin/env python3

import json
import logging
import warnings
import os
import yaml
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

def compute_pissa_init(model, dataloader, rank=8):
    model_layers, target_layers = [], []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'output' in name.lower():
            target_layers.append((name, module))
        elif isinstance(module, nn.Linear) and module.weight.requires_grad:
            model_layers.append((name, module))
    if not target_layers:
        target_layers = model_layers[:10]

    activations = {}
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            device = next(model.parameters()).device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            handles = []
            for name, module in target_layers:
                def get_activation(name):
                    def hook(module, input, output):
                        x = input[0].detach()
                        x = x.reshape(-1, x.size(-1))
                        activations.setdefault(name, []).append(x)
                    return hook
                handles.append(module.register_forward_hook(get_activation(name)))

            if "input_ids" in batch:
                model.encoder(batch["input_ids"], batch.get("attention_mask"))
            elif "src_input_ids" in batch:
                model.encoder(batch["src_input_ids"], batch.get("src_attention_mask"))

            for h in handles:
                h.remove()
            if all(len(v) >= 10 for v in activations.values()):
                break

    init_weights = {}
    for name, acts in activations.items():
        acts = torch.cat(acts, dim=0)
        if acts.size(0) > 1000:
            idx = torch.randperm(acts.size(0))[:1000]
            acts = acts[idx]
        acts -= acts.mean(dim=0, keepdim=True)
        if acts.size(0) > acts.size(1):
            cov = acts.t() @ acts / (acts.size(0) - 1)
            U, S, V = torch.svd(cov)
            comps = U[:, :rank]
        else:
            U, S, V = torch.svd(acts)
            comps = U[:, :rank]
        init_weights[name] = comps
    return init_weights

def initialize_weights_with_pissa(model, pissa_weights):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, 'weight'):
            for key, comps in pissa_weights.items():
                if key in name and comps.size(0) == module.weight.size(1):
                    with torch.no_grad():
                        norm = module.weight.norm(dim=1, keepdim=True)
                        new_w = torch.randn_like(module.weight)
                        for i in range(min(new_w.size(0), comps.size(1))):
                            new_w[i] = comps[:, i].t()
                        new_w *= norm / new_w.norm(dim=1, keepdim=True)
                        module.weight.copy_(new_w)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
    return model

def read_arguments():
    p = ArgumentParser()
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
    p.add_argument("--use_pissa", action="store_true")
    p.add_argument("--pissa_rank", type=int, default=8)
    p.add_argument("--pissa_sample_path", type=str, default="test.csv")
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
        import wandb; wandb.init(project="cometh", id=date, name=date, config=cfg.referenceless_regression_metric.init_args, mode='offline')
    wl = WandbLogger(project="cometh", name=date, version=date, checkpoint_name=date, offline=True)
    tr_args['logger'] = [wl]
    return Trainer(**tr_args), ckpt_dir, date

def save_hparams(model, cfg, path, date):
    name, args = None, None
    if cfg.regression_metric: name, args = "regression_metric", cfg.regression_metric.init_args
    elif cfg.referenceless_regression_metric: name, args = "referenceless_regression_metric", cfg.referenceless_regression_metric.init_args
    elif cfg.ranking_metric: name, args = "ranking_metric", cfg.ranking_metric.init_args
    elif cfg.unified_metric: name, args = "unified_metric", cfg.unified_metric.init_args
    h = {
        'model_type': name,
        'model_config': namespace_to_dict(args),
        'pissa': {'enabled': getattr(cfg, 'use_pissa', True), 'rank': cfg.pissa_rank, 'sample_path': cfg.pissa_sample_path},
        'training': {'warmup_ratio': cfg.warmup_ratio, 'seed': cfg.seed_everything, 'test_file': cfg.test_file},
        'trainer': namespace_to_dict(cfg.trainer.init_args)
    }
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'hparams.yaml'), 'w') as f: yaml.dump(h, f)
    cd = Path(f"cometh/{date}"); cd.mkdir(parents=True, exist_ok=True)
    with open(cd / 'hparams.yaml', 'w') as f: yaml.dump(h, f)
    if hasattr(model, 'hparams'):
        for k, v in h.items():
            if k not in model.hparams: model.hparams[k] = v
    return h

def initialize_model(cfg):
    if cfg.regression_metric:
        cls, args = RegressionMetric, cfg.regression_metric
    elif cfg.referenceless_regression_metric:
        cls, args = ReferencelessRegression, cfg.referenceless_regression_metric
    elif cfg.ranking_metric:
        cls, args = RankingMetric, cfg.ranking_metric
    elif cfg.unified_metric:
        cls, args = UnifiedMetric, cfg.unified_metric
    else:
        raise
    if cfg.load_from_checkpoint:
        model = cls.load_from_checkpoint(cfg.load_from_checkpoint, strict=cfg.strict_load, **namespace_to_dict(args.init_args))
    else:
        model = cls(**namespace_to_dict(args.init_args))
    model.pissa_enabled = True; model.pissa_rank = cfg.pissa_rank
    sp = cfg.pissa_sample_path
    if os.path.exists(sp):
        import pandas as pd
        data = pd.read_csv(sp); recs = data.drop(columns=['score']).to_dict('records')
        dl = DataLoader(dataset=recs, batch_size=8, collate_fn=lambda x: model.prepare_sample(x, stage="predict"), num_workers=0, pin_memory=True)
        pw = compute_pissa_init(model, dl, cfg.pissa_rank)
        model = initialize_weights_with_pissa(model, pw)
    if cfg.warmup_ratio > 0:
        orig = model.configure_optimizers
        def custom_opt(*a, **k):
            opt, sch = orig(*a, **k)
            me = getattr(cfg.trainer.init_args, 'max_epochs', 5)
            we = int(me * cfg.warmup_ratio)
            return opt, [CosineWarmupScheduler(opt[0], we, me)]
        model.configure_optimizers = custom_opt
    return model

def train_command():
    import pandas as pd
    parser = read_arguments(); cfg = parser.parse_args(); seed_everything(cfg.seed_everything)
    trainer, ckpt, date = initialize_trainer(cfg)
    model = initialize_model(cfg)
    save_hparams(model, cfg, ckpt, date)
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Consider increasing the value of the `num_workers` argument")
    trainer.fit(model)
    save_hparams(model, cfg, ckpt, date)
    tf = cfg.test_file
    if os.path.exists(tf):
        df = pd.read_csv(tf)
        inp = df.drop(columns=['score']).to_dict('records')
        res = model.predict(inp, batch_size=8, gpus=1)['scores']
        corr = df['score'].corr(torch.tensor(res), method='spearman')
        print(f"OUT {corr:.4f}")
        with open(os.path.join(ckpt, 'evaluation.json'), 'w') as f:
            json.dump({"spearman_correlation": float(corr)}, f, indent=4)

if __name__ == "__main__":
    train_command()