import os
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F
import numpy as np

def inference(generator, test_dataset, device):
    print()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_weights_path = os.path.normpath(os.path.join(script_dir, "./best_model_weights.pth"))
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
            
            # Display ground truth high-res image
            hr_image_ground_truth = hr_imgs[0].squeeze().permute(1, 2, 0)
            hr_image_ground_truth = (hr_image_ground_truth - hr_image_ground_truth.min()) / (hr_image_ground_truth.max() - hr_image_ground_truth.min())
            
            plt.figure()
            plt.subplot(1, 3, 1)
            plt.imshow(lr_image_test)
            plt.title('Low Resolution')
            plt.subplot(1, 3, 2)
            plt.imshow(hr_image_test)
            plt.title('Generated High Resolution - PSNR: {:.2f} dB'.format(psnr))
            plt.subplot(1, 3, 3)
            plt.imshow(hr_image_ground_truth)
            plt.title('Ground Truth High Resolution')
            plt.show()
            
            break  # Remove this line if you want to visualize all images
    avg_psnr = np.mean(psnr_values)
    print("Average PSNR:", avg_psnr)
