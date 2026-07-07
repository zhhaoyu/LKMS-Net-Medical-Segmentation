\# LKMS-Net for Medical Image Segmentation



LKMS-Net for medical image segmentation on ISIC2017, ISIC2018, and Kvasir-SEG datasets.



\## Paper Link



The original paper is available at:

https://www.sciencedirect.com/science/article/pii/S1746809426014813?dgcid=author



\## Dataset



Download datasets from the official sources:



\- ISIC 2017: https://challenge.isic-archive.com/data/

\- ISIC 2018: https://challenge2018.isic-archive.com/task1/training

\- Kvasir-SEG: https://www.kaggle.com/datasets/debeshjha1/kvasirseg



Place them in `data/` folder as:



data/

├── ISIC2017/

│   ├── images/    # .jpg

│   └── masks/     # .png

├── ISIC2018/

│   ├── images/    # .jpg

│   └── masks/     # .png

└── Kvasir-SEG/

&#x20;   ├── images/    # .jpg

&#x20;   └── masks/     # .jpg



\## Environment



conda activate tensorflow-gpu

pip install -r requirements.txt



\## Training



python train.py --dataset ISIC2017

python train.py --dataset ISIC2018

python train.py --dataset Kvasir-SEG



\## Test



python test.py --model\_path checkpoints/isic2017/isic2017\_best.hdf5 --image\_path test.jpg



\## Citation



@article{zhhaoyu2026mf2net,

&#x20; title={LKMS-Net for Medical Image Segmentation},

&#x20; author={Zhhao Yu},

&#x20; year={2026}

}



\## License



MIT

