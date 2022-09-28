# MAN
### Codes for paper "Multi-scale Attention Network for Image Super-Resolution".

 
## Implementary Details
**Network architecture**: MAB number (n_resblocks): **5/24/36**, channel width (n_feats): **48/60/180** for **tiny/light/base MAN**.
<p align="center">
    <img src="images/MAN_arch.png" width="960"> <br /></p>
    <em> Overview of the proposed MAN constituted of three components: the shallow feature extraction module (SF), the deep feature extraction module (DF) based on
    multiple multi-scale attention blocks (MAB), and the high-quality image reconstruction module. </em>
 
 &nbsp;

**Component details:** three multi-scale decomposition modes are utilized in MLKA. The 7×7 depth-wise convolution is used in the GSAU.
<p align="center">
    <img src="images/MAN_details.png" width="480"> <br /></p>
    <em> Details of Multi-scale Large Kernel Attention (MLKA), Gated Spatial Attention Unit (GSAU), and Large Kernel Attention Tail (LKAT). </em>

&nbsp;
 
## Results and Models

Pretrained models avialable at [Google Drive](https://drive.google.com/drive/folders/1sARYFkVeTIFVCa2EnZg9TjZvirDvUNOL?usp=sharing) and [Baidu Pan](https://pan.baidu.com/s/15CTY-mgdTuOc1I8mzIA4Ug?pwd=mans) (pwd: **mans** for all links)


Results of our MAN-tiny/light/base models. Set5 validation set is used below to show the general performance. The visual results of five testsets are provided in the last column.

| Methods  |  Params   |  Madds   |PSNR/SSIM (x2)|PSNR/SSIM (x3)|PSNR/SSIM (x4)|Results|
|:---------|:---------:|:--------:|:------:|:------:|:------:|:--------:|
| MAN-tiny |      150K |     8.4G | 37.91/0.9603  |        -      | 32.07/0.8930  | [x2](https://pan.baidu.com/s/1mYkGvAlz0bSZuCVubkpsmg?pwd=mans)/[x4](https://pan.baidu.com/s/1u22su2bT4Pq_idVxAnqWdw?pwd=mans)    |
| MAN-light|      840K |    47.1G | 38.18/0.9612  | 34.65/0.9292  | 32.50/0.8988  | [x2](https://pan.baidu.com/s/1AVuPa7bsbb3qMQqMSM-IJQ?pwd=mans)/[x3](https://pan.baidu.com/s/1TRL7-Y23JddVOpEhH0ObEQ?pwd=mans)/[x4](https://pan.baidu.com/s/1T2bPZcjFRxAgMxGWtPv-Lw?pwd=mans) |
| MAN+     |     8712K |     495G | 38.44/0.9623  | 34.97/0.9315  | 32.87/0.9030  | [x2](https://pan.baidu.com/s/1pTb3Fob_7MOxMKIdopI0hQ?pwd=mans)/[x3](https://pan.baidu.com/s/1L3HEtcraU8Y9VY-HpCZdfg?pwd=mans)/[x4](https://pan.baidu.com/s/1FCNqht9zi9HecG3ExRdeWQ?pwd=mans) |

## Training and Testing

The [BasicSR](https://github.com/XPixelGroup/BasicSR) framework is utilized to train our MAN, also testing. 

## Acknowledgements

We would thank [VAN](https://github.com/Visual-Attention-Network/VAN-Classification) and [BasicSR](https://github.com/XPixelGroup/BasicSR) for their enlightening work!
