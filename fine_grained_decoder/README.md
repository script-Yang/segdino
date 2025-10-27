# Fine-Grained Decoder

This module provides a redesigned decoder inspired by discussions in [segdino issue #9](https://github.com/script-Yang/segdino/issues/9).  
Building upon insights from [UniMatch](https://github.com/LiheYoung/UniMatch-V2) and [DPT](https://github.com/isl-org/DPT),  
we developed a more expressive decoding architecture optimized for **fine-grained tasks** such as **remote sensing** and **detailed segmentation**.

## Highlights
- Redesigned decoder structure for improved representation of subtle details.  
- Demonstrates strong performance on **fine-grained and high-resolution datasets**.  
- Slight increase in parameter count compared to the baseline, balanced by better visual and quantitative results.

## Reference
This work was motivated by prior architectures from UniMatch and DPT, and refined based on practical issues raised in the segdino repository.
