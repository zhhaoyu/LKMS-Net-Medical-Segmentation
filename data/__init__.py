from data.base_dataset import BaseDataset
from data.isic_dataset import ISICDataset
from data.kvasir_dataset import KvasirDataset

DATASET_REGISTRY = {
    'ISIC2017': ISICDataset,
    'ISIC2018': ISICDataset,
    'Kvasir-SEG': KvasirDataset,
}

def get_dataset(dataset_name, config):
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[dataset_name](config)