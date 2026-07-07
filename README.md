# LKMS-Net for Medical Image Segmentation

LKMS-Net for medical image segmentation on ISIC2017, ISIC2018, and Kvasir-SEG datasets.

## Dataset

Download datasets from the official sources:

- ISIC 2017:https://challenge.isic-archive.com/data/#2017
- ISIC 2018: https://challenge.isic-archive.com/data/#2018
- Kvasir-SEG: https://www.kaggle.com/datasets/debeshjha1/kvasirseg

Place them in `data/` folder as:

data/
├── ISIC2017/
│   ├── images/    # .jpg
│   └── masks/     # .png
├── ISIC2018/
│   ├── images/    # .jpg
│   └── masks/     # .png
└── Kvasir-SEG/
    ├── images/    # .jpg
    └── masks/     # .jpg

## Environment

- Python 3.8.19
- TensorFlow 2.9.1 (GPU version)

conda activate tensorflow-gpu

pip install -r requirements.txt

## Training

python train.py --dataset ISIC2017
python train.py --dataset ISIC2018
python train.py --dataset Kvasir-SEG

## Test

python test.py --model_path checkpoints/isic2017/isic2017_best.hdf5 --image_path test.jpg

## Citation

@article{zhhaoyu2026lkmsnet,
  title={LKMS-Net for Medical Image Segmentation},
  author={Zhhao Yu},
  year={2026}
}

## License

MIT
