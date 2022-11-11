# -*- coding: utf-8 -*-
import torch, cv2, os, sys, numpy as np, matplotlib.pyplot as plt
from PIL import Image
from ModelZoo.utils import load_as_tensor, Tensor2PIL, PIL2Tensor, _add_batch_one
from ModelZoo import get_model, load_model, print_network
from SaliencyModel.utils import vis_saliency, vis_saliency_kde, click_select_position, grad_abs_norm, grad_norm, prepare_images, make_pil_grid, blend_input
from SaliencyModel.utils import cv2_to_pil, pil_to_cv2, gini
from SaliencyModel.attributes import attr_grad
from SaliencyModel.BackProp import I_gradient, attribution_objective, Path_gradient
from SaliencyModel.BackProp import saliency_map_PG as saliency_map
from SaliencyModel.BackProp import GaussianBlurPath
from SaliencyModel.utils import grad_norm, IG_baseline, interpolation, isotropic_gaussian_kernel
from cal_metrix import calculate_psnr, calculate_ssim, bgr2ycbcr, tensor2img

#LAM: Interpreting Super-Resolution Networks with Local Attribution Maps

def LAM(model_names='MAN@Base', image_path='test_images/3.png',w=90, h=120):
    model = load_model(model_names)
    window_size = 16  # Define windoes_size of D
    img_lr, img_hr = prepare_images(image_path)  # Change this image name
    tensor_lr = PIL2Tensor(img_lr)[:3] ; tensor_hr = PIL2Tensor(img_hr)[:3]
    cv2_lr = np.moveaxis(tensor_lr.numpy(), 0, 2) ; cv2_hr = np.moveaxis(tensor_hr.numpy(), 0, 2)
    
    plt.imshow(cv2_hr)

    draw_img = pil_to_cv2(img_hr)
    cv2.rectangle(draw_img, (w, h), (w + window_size, h + window_size), (0, 0, 255), 2)
    position_pil = cv2_to_pil(draw_img)
    plt.imshow(position_pil) 

    sigma = 1.2 ; fold = 50 ; l = 9 ; alpha = 0.5
    attr_objective = attribution_objective(attr_grad, h, w, window=window_size)
    gaus_blur_path_func = GaussianBlurPath(sigma, fold, l)
    interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(tensor_lr.numpy(), model, attr_objective, gaus_blur_path_func, cuda=True)
    grad_numpy, result = saliency_map(interpolated_grad_numpy, result_numpy)
    abs_normed_grad_numpy = grad_abs_norm(grad_numpy)
    saliency_image_abs = vis_saliency(abs_normed_grad_numpy, zoomin=4)
    saliency_image_kde = vis_saliency_kde(abs_normed_grad_numpy)
    blend_abs_and_input = cv2_to_pil(pil_to_cv2(saliency_image_abs) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
    blend_kde_and_input = cv2_to_pil(pil_to_cv2(saliency_image_kde) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
    pil = make_pil_grid(
        [position_pil,
         saliency_image_abs,
         blend_abs_and_input,
         blend_kde_and_input,
         Tensor2PIL(torch.clamp(result, min=0., max=1.))]
        )
    
    plt.axis('off')
    
    gini_index = gini(abs_normed_grad_numpy)
    diffusion_index = (1 - gini_index) * 100
    
    plt.imshow(img_lr.resize(img_hr.size))
    plt.savefig('./lam_results/{}/0lr.png'.format(image_path[-5]), dpi=300, bbox_inches = 'tight',pad_inches=0.0)
     
    plt.imshow(position_pil) 
    plt.savefig('./lam_results/{}/1hr.png'.format(image_path[-5]), dpi=300, bbox_inches = 'tight',pad_inches=0.0)
    plt.imshow(saliency_image_abs) 
    plt.savefig('./lam_results/{}/2abs_{}.png'.format(image_path[-5],model_names), dpi=300, bbox_inches = 'tight',pad_inches=0.0)
    plt.imshow(blend_kde_and_input) 
    plt.savefig('./lam_results/{}/3kde_{}.png'.format(image_path[-5],model_names), dpi=300, bbox_inches = 'tight',pad_inches=0.0)
    plt.imshow(Tensor2PIL(torch.clamp(result, min=0., max=1.))) 
    plt.savefig('./lam_results/{}/4sr_{}.png'.format(image_path[-5],model_names), dpi=300, bbox_inches = 'tight',pad_inches=0.0)
    
    plt.imshow(pil)

    im_GT = tensor2img(tensor_hr)
    im_Gen = tensor2img(result)
    im_GT_in = bgr2ycbcr(im_GT)
    im_Gen_in = bgr2ycbcr(im_Gen)
    
    crop_border = 4
    
    if im_GT_in.ndim == 3:
        cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border, :]
        cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border, :]
    elif im_GT_in.ndim == 2:
        cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border]
        cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border]    
        
    PSNR = calculate_psnr(cropped_GT * 255, cropped_Gen * 255)
    SSIM = calculate_ssim(cropped_GT * 255, cropped_Gen * 255)        
    
    print('DI: {:.5f} PSNR: {:.3f}dB  SSIM: {:.5f}'.format(diffusion_index,PSNR,SSIM))
        
model_names='EDSR@Large'
image_path='test_images/e.png'
w=120
h=100  
LAM(model_names,image_path,w,h)    