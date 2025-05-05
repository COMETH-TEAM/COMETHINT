#!/usr/bin/env python3

import json
import logging
import warnings
import os
import yaml
from datetime import datetime
from pathlib import Path

import torch
from jsonargparse import ActionConfigFile, ArgumentParser, namespace_to_dict
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from peft import LoraConfig, get_peft_model
from comet.models import RankingMetric, ReferencelessRegression, RegressionMetric, UnifiedMetric

torch.set_float32_matmul_precision('high')

logger = logging.getLogger(__name__)

def find_target_modules(model):
    target_modules = []
    module_names = [name for name, _ in model.encoder.model.named_modules()]
    attention_modules = [name for name in module_names if any(pattern in name.lower() for pattern in ['attention', 'self', 'attn'])]
    if attention_modules:
        logger.info(f"Found potential attention modules: {attention_modules[:5]}...")
        for name in module_names:
            if any(proj in name.lower() for proj in ['query', 'q_proj']):
                target_modules.append(name)
            if any(proj in name.lower() for proj in ['value', 'v_proj']):
                target_modules.append(name)
    if not target_modules:
        logger.info("No projection modules found, looking for transformer layers...")
        layers = [name for name in module_names if 'layer' in name.lower() and name.count('.') <= 3]
        if layers:
            target_modules = layers[:4]
    if not target_modules:
        logger.info("No suitable layers found, using general modules with parameters...")
        for name, module in model.encoder.model.named_modules():
            if hasattr(module, 'weight') and isinstance(module.weight, torch.nn.Parameter):
                if 'embed' not in name.lower():
                    target_modules.append(name)
                    if len(target_modules) >= 4:
                        break
    return target_modules


def apply_dora(model, alpha=8.0, target_modules=None, rank=8):
    logger.info(f"Applying DoRA with alpha={alpha}, rank={rank}")
    if target_modules is None:
        target_modules = find_target_modules(model)
    if not target_modules:
        logger.warning("No target modules found, DoRA cannot be applied")
        return model
    logger.info(f"Using target modules: {target_modules}")
    dora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    base_model = model.encoder.model
    model.encoder.model = get_peft_model(base_model, dora_config)
    return model


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(description="Command for training COMET models.")
    parser.add_argument(
        "--seed_everything",
        type=int,
        default=12,
        help="Training Seed.",
    )
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_subclass_arguments(RegressionMetric, "regression_metric")
    parser.add_subclass_arguments(ReferencelessRegression, "referenceless_regression_metric")
    parser.add_subclass_arguments(RankingMetric, "ranking_metric")
    parser.add_subclass_arguments(UnifiedMetric, "unified_metric")
    parser.add_subclass_arguments(EarlyStopping, "early_stopping")
    parser.add_subclass_arguments(ModelCheckpoint, "model_checkpoint")
    parser.add_subclass_arguments(Trainer, "trainer")
    parser.add_argument(
        "--load_from_checkpoint",
        help="Loads a model checkpoint for fine-tuning",
        default=None,
    )
    parser.add_argument(
        "--strict_load",
        action="store_true",
        help="Strictly enforce checkpoint key matching.",
    )
    parser.add_argument(
        "--use_dora",
        action="store_true",
        help="Use DoRA for parameter-efficient fine-tuning."
    )
    parser.add_argument(
        "--dora_alpha",
        type=float,
        default=8.0,
        help="DoRA alpha parameter"
    )
    parser.add_argument(
        "--dora_rank",
        type=int,
        default=8,
        help="DoRA rank parameter"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/test_ref.csv",
        help="Path to test file for evaluation"
    )
    return parser


def initialize_trainer(configs) -> tuple:
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_path = f"checkpoints/{date}"
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_args = namespace_to_dict(configs.model_checkpoint.init_args)
    if 'dirpath' in checkpoint_args:
        del checkpoint_args['dirpath']
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        **checkpoint_args
    )
    early_stop_callback = EarlyStopping(
        **namespace_to_dict(configs.early_stopping.init_args)
    )
    trainer_args = namespace_to_dict(configs.trainer.init_args)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer_args["callbacks"] = [early_stop_callback, checkpoint_callback, lr_monitor]
    wandb_configs = None
    if configs.referenceless_regression_metric is not None:
        wandb_configs = configs.referenceless_regression_metric
        import wandb
        wandb.init(
            project="cometh",
            id=date,
            name=date,
            config=wandb_configs.init_args,
            mode='offline',
        )
    wandb_logger = WandbLogger(
        project="cometh",
        name=date,
        version=date,
        checkpoint_name=date,
        offline=True
    )
    trainer_args["logger"] = [wandb_logger]
    print("TRAINER ARGUMENTS: ")
    print(json.dumps(trainer_args, indent=4, default=lambda x: x.__dict__))
    trainer = Trainer(**trainer_args)
    return trainer, checkpoint_path, date


def save_hparams(model, configs, dir_path, date):
    dora_used = getattr(configs, 'use_dora', False)
    dora_alpha = getattr(configs, 'dora_alpha', 8.0)
    dora_rank = getattr(configs, 'dora_rank', 8)
    test_file = getattr(configs, 'test_file', "data/test_ref.csv")
    model_name = None
    model_args = None
    if configs.regression_metric is not None:
        model_name = "regression_metric"
        model_args = configs.regression_metric.init_args
    elif configs.referenceless_regression_metric is not None:
        model_name = "referenceless_regression_metric"
        model_args = configs.referenceless_regression_metric.init_args
    elif configs.ranking_metric is not None:
        model_name = "ranking_metric"
        model_args = configs.ranking_metric.init_args
    elif configs.unified_metric is not None:
        model_name = "unified_metric"
        model_args = configs.unified_metric.init_args
    hparams = {
        "model_type": model_name,
        "model_config": namespace_to_dict(model_args),
        "dora": {"enabled": dora_used, "alpha": dora_alpha, "rank": dora_rank},
        "training": {"seed": configs.seed_everything, "test_file": test_file},
        "trainer": namespace_to_dict(configs.trainer.init_args)
    }
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, "hparams.yaml"), "w") as f:
        yaml.dump(hparams, f, default_flow_style=False)
    cometh_dir = Path(f"cometh/{date}")
    cometh_dir.mkdir(parents=True, exist_ok=True)
    with open(cometh_dir / "hparams.yaml", "w") as f:
        yaml.dump(hparams, f, default_flow_style=False)
    if hasattr(model, 'hparams'):
        for key, value in hparams.items():
            if key not in model.hparams:
                model.hparams[key] = value
    logger.info(f"Saved hyperparameters to {dir_path}/hparams.yaml and {cometh_dir}/hparams.yaml")
    return hparams


def initialize_model(configs):
    print("MODEL ARGUMENTS: ")
    if configs.regression_metric is not None:
        print(json.dumps(configs.regression_metric.init_args, indent=4, default=lambda x: x.__dict__))
        if configs.load_from_checkpoint is not None:
            logger.info(f"Loading weights from {configs.load_from_checkpoint}.")
            model = RegressionMetric.load_from_checkpoint(
                checkpoint_path=configs.load_from_checkpoint,
                strict=configs.strict_load,
                **namespace_to_dict(configs.regression_metric.init_args),
            )
        else:
            model = RegressionMetric(**namespace_to_dict(configs.regression_metric.init_args))
    elif configs.referenceless_regression_metric is not None:
        print(json.dumps(configs.referenceless_regression_metric.init_args, indent=4, default=lambda x: x.__dict__))
        if configs.load_from_checkpoint is not None:
            logger.info(f"Loading weights from {configs.load_from_checkpoint}.")
            model = ReferencelessRegression.load_from_checkpoint(
                checkpoint_path=configs.load_from_checkpoint,
                strict=configs.strict_load,
                **namespace_to_dict(configs.referenceless_regression_metric.init_args),
            )
        else:
            model = ReferencelessRegression(**namespace_to_dict(configs.referenceless_regression_metric.init_args))
    elif configs.ranking_metric is not None:
        print(json.dumps(configs.ranking_metric.init_args, indent=4, default=lambda x: x.__dict__))
        if configs.load_from_checkpoint is not None:
            logger.info(f"Loading weights from {configs.load_from_checkpoint}.")
            model = RankingMetric.load_from_checkpoint(
                checkpoint_path=configs.load_from_checkpoint,
                strict=configs.strict_load,
                **namespace_to_dict(configs.ranking_metric.init_args),
            )
        else:
            model = RankingMetric(**namespace_to_dict(configs.ranking_metric.init_args))
    elif configs.unified_metric is not None:
        print(json.dumps(configs.unified_metric.init_args, indent=4, default=lambda x: x.__dict__))
        if configs.load_from_checkpoint is not None:
            logger.info(f"Loading weights from {configs.load_from_checkpoint}.")
            model = UnifiedMetric.load_from_checkpoint(
                checkpoint_path=configs.load_from_checkpoint,
                strict=configs.strict_load,
                **namespace_to_dict(configs.unified_metric.init_args),
            )
        else:
            model = UnifiedMetric(**namespace_to_dict(configs.unified_metric.init_args))
    else:
        raise Exception("Model configurations missing!")
    if getattr(configs, 'use_dora', False):
        model.dora_enabled = True
        model.dora_alpha = getattr(configs, 'dora_alpha', 8.0)
        model.dora_rank = getattr(configs, 'dora_rank', 8)
        model = apply_dora(model, alpha=configs.dora_alpha, rank=configs.dora_rank)
        logger.info("DoRA applied successfully!")
    else:
        model.dora_enabled = False
        model.dora_alpha = None
        model.dora_rank = None
    return model


def train_command() -> None:
    import pandas as pd
    parser = read_arguments()
    cfg = parser.parse_args()
    seed_everything(cfg.seed_everything)
    trainer, checkpoint_path, date = initialize_trainer(cfg)
    model = initialize_model(cfg)
    hparams = save_hparams(model, cfg, checkpoint_path, date)
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*Consider increasing the value of the `num_workers` argument` .*",
    )
    trainer.fit(model)
    save_hparams(model, cfg, checkpoint_path, date)
    test_file = getattr(cfg, 'test_file', "data/test_ref.csv")
    if os.path.exists(test_file):
        data = pd.read_csv(test_file)
        model_input = data.drop(columns=['score']).to_dict('records')
        result_ours = model.predict(model_input, batch_size=8, gpus=1)['scores']
        correlation = data['score'].corr(pd.Series(result_ours), method='spearman')
        print(f"OUT {correlation:.4f}")
        with open(os.path.join(checkpoint_path, "evaluation.json"), "w") as f:
            json.dump({"spearman_correlation": float(correlation)}, f, indent=4)

if __name__ == "__main__":
    train_command()