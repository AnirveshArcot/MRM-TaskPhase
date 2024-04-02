import os
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import to_pil_image
from PIL import Image

def inferencer(generator, test_dataset, device):
    print()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_weights_path = os.path.normpath(os.path.join(script_dir, "../../51246.pth"))
    weights = torch.load(absolute_weights_path, map_location=torch.device(device))
    generator.load_state_dict(weights)
    generator.eval()
    psnr_values = []
    with torch.no_grad():
        for hr_imgs, lr_imgs in test_dataset:
            hr_imgs, lr_imgs = hr_imgs.to(device), lr_imgs.to(device)
            gen_hr_imgs = generator(lr_imgs)
            lr_image_test = lr_imgs[0].squeeze().permute(1, 2, 0)
            hr_image_test = gen_hr_imgs[0].squeeze().permute(1, 2, 0)
            hr_image_test = (hr_image_test - hr_image_test.min()) / (hr_image_test.max() - hr_image_test.min())
            mse = F.mse_loss(hr_imgs, gen_hr_imgs)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            psnr_values.append(psnr.item())
            hr_image_ground_truth = hr_imgs[0].squeeze().permute(1, 2, 0)
            hr_image_ground_truth = (hr_image_ground_truth - hr_image_ground_truth.min()) / (hr_image_ground_truth.max() - hr_image_ground_truth.min())
            
            # Bicubic interpolation for LR image
            lr_image_bicubic = to_pil_image(lr_imgs[0].cpu().detach().squeeze())
            lr_image_bicubic = lr_image_bicubic.resize(hr_image_ground_truth.shape[:2][::-1], resample=Image.BICUBIC)
            
            plt.figure()
            plt.subplot(1, 4, 1)
            plt.imshow(lr_image_test)
            plt.title('Low Resolution')
            plt.subplot(1, 4, 2)
            plt.imshow(hr_image_test)
            plt.title('Generated High Resolution - PSNR: {:.2f} dB'.format(psnr))
            plt.subplot(1, 4, 3)
            plt.imshow(lr_image_bicubic)
            plt.title('Bicubic Interpolation')
            plt.subplot(1, 4, 4)
            plt.imshow(hr_image_ground_truth)
            plt.title('Ground Truth High Resolution')
            plt.show()
            
            break  # Remove this line if you want to visualize all images
    avg_psnr = np.mean(psnr_values)
    print("Average PSNR:", avg_psnr)
 