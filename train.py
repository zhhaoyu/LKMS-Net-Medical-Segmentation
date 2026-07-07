import os
import argparse
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, CSVLogger, EarlyStopping
)

from src.model import MF2_Net
from data import get_dataset


def set_seed(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train MF2-Net on Multiple Medical Segmentation Datasets'
    )
    parser.add_argument(
        '--dataset', type=str, required=True,
        choices=['ISIC2017', 'ISIC2018', 'Kvasir-SEG'],
        help='Dataset name to train on'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to config file (default: configs/{dataset}.yaml)'
    )
    parser.add_argument(
        '--data_dir', type=str, default=None,
        help='Override data root directory'
    )
    parser.add_argument(
        '--batch_size', type=int, default=None,
        help='Override batch size'
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Override number of epochs'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.config is None:
        config_path = f'configs/{args.dataset}.yaml'
    else:
        config_path = args.config

    print(f"Loading config from: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if args.data_dir:
        config['dataset']['image_dir'] = os.path.join(args.data_dir, 'images')
        config['dataset']['mask_dir'] = os.path.join(args.data_dir, 'masks')

    if args.batch_size:
        config['training']['batch_size'] = args.batch_size

    if args.epochs:
        config['training']['epochs'] = args.epochs

    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Image dir: {config['dataset']['image_dir']}")
    print(f"Mask dir: {config['dataset']['mask_dir']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Epochs: {config['training']['epochs']}")
    print("=" * 60)

    dataset = get_dataset(args.dataset, config)
    images, masks = dataset.load_data()

    save_config = config['save']
    os.makedirs(save_config['checkpoint_dir'], exist_ok=True)
    os.makedirs(save_config['log_dir'], exist_ok=True)

    model = MF2_Net(input_shape=(
        config['model']['input_height'],
        config['model']['input_width'],
        config['model']['input_channels']
    ))
    model.summary()

    checkpoint_path = os.path.join(
        save_config['checkpoint_dir'],
        save_config['best_model_name']
    )
    log_path = os.path.join(
        save_config['log_dir'],
        save_config['log_name']
    )

    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True
        ),
        CSVLogger(log_path, append=False, separator=','),
        EarlyStopping(
            monitor='val_accuracy',
            patience=config['training']['patience'],
            restore_best_weights=True
        )
    ]

    history = model.fit(
        images, masks,
        batch_size=config['training']['batch_size'],
        epochs=config['training']['epochs'],
        verbose=1,
        validation_split=config['training']['validation_split'],
        shuffle=config['training']['shuffle'],
        callbacks=callbacks
    )

    print(f"\nTraining completed for {args.dataset}")
    print(f"Best model saved to: {checkpoint_path}")
    print(f"Log saved to: {log_path}")


if __name__ == '__main__':
    main()
