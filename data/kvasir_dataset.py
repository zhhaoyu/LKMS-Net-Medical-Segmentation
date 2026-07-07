import numpy as np
from data.base_dataset import BaseDataset


class KvasirDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)

    def load_data(self):
        n_samples = len(self.file_list)
        h, w = self.img_size

        images = np.ndarray((n_samples, h, w, 3), dtype='float32')
        masks = np.ndarray((n_samples, h, w, 1), dtype='float32')

        for i, fname in enumerate(self.file_list):
            images[i] = self._load_image(fname)
            masks[i] = self._load_mask(fname)

            if (i + 1) % 100 == 0:
                print(f'[{self.dataset_config["name"]}] Loaded {i + 1}/{n_samples}')

        return images, masks
