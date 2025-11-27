## Using SAT-493M Pretrained ViT Models

We have received and addressed the feedback from issue [#13](https://github.com/script-Yang/segdino/issues/13).  
Support for ViT models pretrained on the **SAT-493M** satellite dataset is now included.

### 1. Setup
Place **`train_sat.py`** in the project’s root directory.

### 2. Pretrained Checkpoints
Ensure that the required `.pth` checkpoint file is available, for example:

dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth


The checkpoint filename must match the **model identifier** you specify (e.g., `"large"`).

### 3. Example Usage

```bash
python train_sat.py \
  --data_dir ./data \
  --dataset xxx \
  --input_h 256 --input_w 256 \
  --repo_dir ./dinov3 \
  --img_dir_name image \
  --label_dir_name mask \
  --mask_ext '.jpg'
  --epochs 50 \
  --batch_size 4 \
  --lr 1e-4
```