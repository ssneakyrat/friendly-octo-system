#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from pathlib import Path
import datetime
from omegaconf import OmegaConf

from models.futurevox import FutureVoxModel
from data.datamodule import FutureVoxDataModule


def load_config(config_path):
    """
    Load configuration file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration object
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert to OmegaConf for easier access
    config = OmegaConf.create(config)
    
    return config


def setup_output_dirs(config):
    """
    Create output directories
    
    Args:
        config: Configuration object
        
    Returns:
        Updated configuration with resolved paths
    """
    # Create timestamp for run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up output directory
    output_dir = Path(config.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update config with resolved paths
    config.output_dir = str(output_dir)
    config.checkpoint_dir = str(output_dir / "checkpoints")
    config.log_dir = str(output_dir / "logs")
    
    # Create directories
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Save config for reproducibility
    config_save_path = output_dir / "config.yaml"
    OmegaConf.save(config, config_save_path)
    
    return config


def train(config, args):
    """
    Train FutureVox+ model
    
    Args:
        config: Configuration object
        args: Command-line arguments
        
    Returns:
        Trained model
    """
    # Set random seed for reproducibility
    pl.seed_everything(args.seed)
    
    # Create data module
    data_module = FutureVoxDataModule(config)
    
    # Create model
    model = FutureVoxModel(config)
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=config.log_dir,
        name="futurevox",
        log_graph=config.logger.tensorboard.log_graph,
        default_hp_metric=False
    )
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename="futurevox-{epoch:02d}-{val/total_loss:.4f}",
        monitor="val/total_loss",
        save_top_k=3,
        mode="min",
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    callbacks = [checkpoint_callback, lr_monitor]
    
    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=config.training.precision,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        gradient_clip_val=config.training.grad_clip_val,
        logger=logger,
        callbacks=callbacks,
        val_check_interval=config.validation.val_check_interval,
        num_sanity_val_steps=2
    )
    
    # Resume from checkpoint if provided
    if args.resume_from_checkpoint:
        trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_from_checkpoint)
    else:
        trainer.fit(model, datamodule=data_module)
    
    return model


def main():
    """
    Main function for training FutureVox+ model
    """
    parser = argparse.ArgumentParser(description="Train FutureVox+ model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up output directories
    config = setup_output_dirs(config)
    
    # Train model
    model = train(config, args)
    
    print(f"Training completed. Model checkpoints saved to {config.checkpoint_dir}")


if __name__ == "__main__":
    main()