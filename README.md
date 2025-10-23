# SegDINO

[SegDINO: An Efficient Design for Medical and Natural Image Segmentation with DINO-V3](https://arxiv.org/abs/2509.00833)

We propose SegDINO, an efficient image segmentation framework that couples a frozen DINOv3 backbone with a lightweight MLP decoder, achieving state-of-the-art performance on both medical and natural image segmentation tasks while maintaining minimal parameter and computational cost.

![](src/segdino_pic.png)


## Environment Setup

```bash
conda create -n segdino python=3.10.16

conda activate segdino

pip install -r requirements.txt
````

Clone the DINOv3 repository:

```bash
git clone https://github.com/facebookresearch/dinov3.git
```

Download [DINOv3](https://github.com/facebookresearch/dinov3)  pretrained weights and place them in:

```
./web_pth
```

## Dataset Preparation


- **[TN3K](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation)**  
  A large-scale thyroid nodule segmentation dataset~\citep{gong2023thyroid}, containing **3,493 ultrasound images** with pixel-level annotations collected from multiple hospitals.  

- **[Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)**  
  A polyp segmentation dataset~\citep{jha2019kvasir} derived from colonoscopy examinations, consisting of **1,000 images** with high-quality expert annotations.  

- **[ISIC](https://challenge.isic-archive.com/data/#2017)**  
  A skin lesion segmentation benchmark~\citep{codella2018skin}, providing **2,750 dermoscopic images** annotated for lesion boundaries and covering diverse lesion types and acquisition conditions.  

Organize datasets in the following structure:

```
./segdata/tn3k
./segdata/kvasir
./segdata/isic
```

Each dataset folder should contain an `image` directory and a `mask` directory.

## Training

Example training command:

```bash
python train_segdino.py \
  --data_dir ./segdata \
  --dataset tn3k \
  --input_h 256 --input_w 256 \
  --dino_size s \
  --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --repo_dir ./dinov3 \
  --img_dir_name image \
  --label_dir_name mask \
  --mask_ext '.jpg' \
  --epochs 50 \
  --batch_size 4 \
  --lr 1e-4
```

## Testing

Example testing command:

```bash
python test_segdino.py \
  --data_dir ./segdata \
  --dataset tn3k \
  --input_h 256 --input_w 256 \
  --dino_size s \
  --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --ckpt ./tn3k.pth \
  --repo_dir ./dinov3 \
  --img_dir_name image \
  --label_dir_name mask \
  --mask_ext '.jpg'
```

## Notes

* Make sure the pretrained DINO weights (`.pth` file) are correctly downloaded and placed under `./web_pth`.
* Modify paths as needed for your environment.
* Training and testing configurations (e.g., image size, batch size, learning rate) can be adjusted via command-line arguments.

## Acknowledgements

We would like to thank the open-source community for their invaluable contributions.  
In particular, we acknowledge the following repositories that made this work possible:

- [DINOv3](https://github.com/facebookresearch/dinov3)   

- [DPT](https://github.com/isl-org/DPT)

- [Unimatch](https://github.com/LiheYoung/UniMatch-V2)


## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{yang2025segdino,
  title={SegDINO: An Efficient Design for Medical and Natural Image Segmentation with DINO-V3},
  author={Yang, Sicheng and Wang, Hongqiu and Xing, Zhaohu and Chen, Sixiang and Zhu, Lei},
  journal={arXiv preprint arXiv:2509.00833},
  year={2025},
  url={https://arxiv.org/abs/2509.00833}
}
```


