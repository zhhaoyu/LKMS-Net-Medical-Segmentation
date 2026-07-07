import os
import numpy as np
from abc import ABC, abstractmethod
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class BaseDataset(ABC):
    def __init__(self, config):
        self.config = config
        self.dataset_config = config['dataset']
        self.img_size = tuple(self.dataset_config['input_size'])
        self.image_dir = self.dataset_config['image_dir']
        self.mask_dir = self.dataset_config['mask_dir']
        self.image_ext = self.dataset_config.get('image_extension', '.jpg')
        self.mask_ext = self.dataset_config.get('mask_extension', '.png')

        self._validate_paths()
        self.file_list = self._get_file_list()

        print(f"[{self.dataset_config['name']}] Found {len(self.file_list)} samples")

    def _validate_paths(self):
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}\n"
                "Please download dataset and place in data/ directory."
            )
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(
                f"Mask directory not found: {self.mask_dir}\n"
                "Please download dataset and place in data/ directory."
            )

    def _get_file_list(self):
        files = [f for f in os.listdir(self.image_dir)
                 if f.endswith(self.image_ext)]
        return sorted(files)

    def _load_image(self, file_name):
        img_path = os.path.join(self.image_dir, file_name)
        img = load_img(img_path)
        img = img.resize(self.img_size)
        img = img_to_array(img).astype('float32')
        img = (img - img.mean()) / (img.std() + 1e-8)
        return img

    def _load_mask(self, file_name):
        mask_name = file_name.replace(self.image_ext, self.mask_ext)
        mask_path = os.path.join(self.mask_dir, mask_name)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        mask = load_img(mask_path, color_mode='grayscale')
        mask = mask.resize(self.img_size)
        mask = img_to_array(mask).astype('float32')
        mask = mask / 255.0
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0
        return mask

    @abstractmethod
    def load_data(self):
        pass

    def get_sample_count(self):
        return len(self.file_list)