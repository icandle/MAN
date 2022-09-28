# MAN
Codes for paper "Multi-scale Attention Network for Image Super-Resolution".

 
## Details
**Network architecture**:
<p align="center">
    <img src="images/MAN_arch.png" width="960"> <br /></p>
    <em> Overview of the proposed MAN constituted of three components: the shallow feature extraction module (SF), the deep feature extraction module (DF) based on
    multiple multi-scale attention blocks (MAB), and the high-quality image reconstruction module. </em>



**Component details:**

MAB number (n_resblocks): **5/24/36**, channel width (n_feats): **48/60/180** for **tiny/light/base MAN**.

## Results and Models

Models download at [Google Drive](https://drive.google.com/drive/folders/1sARYFkVeTIFVCa2EnZg9TjZvirDvUNOL?usp=sharing) and [Baidu Pan](https://pan.baidu.com/s/15CTY-mgdTuOc1I8mzIA4Ug?pwd=mans) (pwd: **mans** for all links)


Results of our MAN-tiny/light/base models. Set5 validation set is used below to show the general performance.

| Methods  |  Params   |  Madds   |PSNR(x2)|PSNR(x3)|PSNR(x4)|Download Results|
|:---------|:---------:|:--------:|:------:|:------:|:------:|:--------:|
| MAN-tiny |      150K |     8.4G | 37.91  | -      | 32.07  | x2/[x4](https://pan.baidu.com/s/1u22su2bT4Pq_idVxAnqWdw?pwd=mans)    |
| MAN-light|      840K |    47.1G | 38.18  | 34.65  | 32.50  | [x2](https://pan.baidu.com/s/1AVuPa7bsbb3qMQqMSM-IJQ?pwd=mans)/x3/[x4](https://pan.baidu.com/s/1T2bPZcjFRxAgMxGWtPv-Lw?pwd=mans) |
| MAN+     |     8712K |     495G | 38.44  | 34.97  | 32.87  | [x2](https://pan.baidu.com/s/1pTb3Fob_7MOxMKIdopI0hQ?pwd=mans)/[x3](https://pan.baidu.com/s/1L3HEtcraU8Y9VY-HpCZdfg?pwd=mans)/[x4](https://pan.baidu.com/s/1FCNqht9zi9HecG3ExRdeWQ?pwd=mans) |

##

We would thank [VAN](https://github.com/Visual-Attention-Network/VAN-Classification) and [BasicSR]() for their enlightening work!
