import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
from urllib.request import urlretrieve

sys.path.append("/devel/dinov2")


BACKBONE_SIZE = "large" # in ("small", "base", "large" or "giant")


backbone_archs = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
    "reg_small" : "vits14_reg"
}
backbone_arch = backbone_archs[BACKBONE_SIZE]
backbone_name = f"dinov2_{backbone_arch}"
from dinov2.models.vision_transformer import vit_small, vit_large

import urllib

import mmcv
from mmcv.runner import load_checkpoint
from math import ceil, sqrt
def load_image_from_url(url: str) -> Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")
    
def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    
    heatmap = np.float32(heatmap) / 255
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    cam = heatmap_rgb + np.float32(img)
    
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

if __name__ == '__main__':
    image_size = (448, 448)
    output_dir = './images/result'
    patch_size = 14

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # init model
    model = vit_large(
            patch_size=14,
            img_size=526,
            init_values=1.0,
            #ffn_layer="mlp",
            block_chunks=0
    )
    # load model\
    # model.load_state_dict(torch.load('dinov2_vitl14_pretrain.pth'))
    model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    model.eval()

    # load image
    # img = Image.open('/devel/dinov2/images/test/input.png')
    # img = Image.open('cow-on-the-beach2.jpg')
    # EXAMPLE_IMAGE_URL = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"
    # EXAMPLE_IMAGE_URL = "https://images.mypetlife.co.kr/content/uploads/2019/04/09192811/welsh-corgi-1581119_960_720.jpg"
    # img_url = "https://images.mypetlife.co.kr/content/uploads/2019/04/09192811/welsh-corgi-1581119_960_720.jpg"
    # urlretrieve(img_url, "attention_data/img.jpg")
    # img = load_image_from_url(EXAMPLE_IMAGE_URL)
    # img = img.convert('RGB')
    img = Image.open("/devel/dinov2/notebooks/asd1.jpg")

    # resize image image_size = (952, 952)
    transform = pth_transforms.Compose([
        pth_transforms.Resize(image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)
    print("img.shape",img.shape)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    print("w_featmap",w_featmap,"h_featmap",h_featmap)
    print("img.shape",img.shape)
    print(model.__dict__)
    attentions = model.get_last_self_attention(img.to(device))

    nh = attentions.shape[1] # number of head

    # batch x head x (patch +1) x (patch +1)?
    # attentions.shape torch.Size([1, 16, 4625, 4625])
    print("attentions.shape",attentions.shape)
    # attentions = 1 x 16 x 4625 x 4625
    # we keep only the output patch attention
    # for every patch
    attention = attentions[0, :, 0, 1:].reshape(nh, -1)
    print("attention.shape",attention.shape)
    # attention = 1 x 16 x 4625 x 4625

    attention_mean = attentions.mean(dim=1)[0,0,1:]

    # attentions.shape torch.Size([16, 4624])
    print("attentions.shape",attentions.shape)
    # weird: one pixel gets high attention over all heads?
    # print(torch.max(attentions, dim=1)) 
    # attentions[:, 283] = 0 

    attention = attention.reshape(nh, w_featmap, h_featmap)
    print("nh, w_featmap, h_featmap",nh, w_featmap, h_featmap)
    
    attention_mean = attention_mean.reshape(1, w_featmap, h_featmap)
    print("attention_map",attentions.shape)
    attention = nn.functional.interpolate(attention.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    print("attention_map after interpolate",attention.shape)

    attention_mean = nn.functional.interpolate(attention_mean.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    # save attentions heatmaps
    os.makedirs(output_dir, exist_ok=True)



    # for j in range(nh):
    #     fname = os.path.join(output_dir, "attn-head" + str(j) + ".png")
    #     plt.imsave(fname=fname, arr=attentions[j], format='png')
    #     print(f"{fname} saved.")
    N = int(ceil(sqrt(nh)))  # Calculate N x N grid size for nh images

    fig, axes = plt.subplots(N+1, N, figsize=(20, 20))  # Adjust figsize as needed
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    for j in range(nh):
        axes[j].imshow(attention[j],cmap="jet")
        axes[j].axis('off')  # Hide axis
        axes[j].set_title(f"Head {j}")  # Optional: add titles

    axes[0].imshow(attention_mean[0]/attention_mean.max(),cmap="jet")
    axes[0].axis('off')  # Hide axis
    axes[0].set_title(f"rollout")  # Optional: add titles

    ## draw attention on image
    

    # Hide any unused subplots
    for k in range(j + 2, N * (N+1)):
        axes[k].axis('off')

    plt.tight_layout()
    # plt.show()  # Close the figure to free memory
    print("end")


    img = Image.open("/devel/dinov2/notebooks/asd1.jpg")
    image = np.array(img)
    rollout_mask = attention_mean[0]
    print(rollout_mask.shape)
    resized_rollout_mask = cv2.resize(rollout_mask/rollout_mask.max(), (image.shape[1],image.shape[0]) )
    print("resized_rollout_mask",resized_rollout_mask.shape)

    heatmap = show_mask_on_image(image,resized_rollout_mask)
    axes[j+1].imshow(heatmap)
    plt.imshow(heatmap)
    output_fname = os.path.join(output_dir, "combined_attentions.png")  # Define your output file name
    plt.savefig(output_fname)  # Save the figure as a single image
    cv2.imshow('heatmap only',heatmap)
    # cv2.waitKey(0)