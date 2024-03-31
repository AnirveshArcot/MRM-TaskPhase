import os
from matplotlib import pyplot as plt
import torch


def inference(generator,test_dataset,device):
    print()
    script_dir=os.path.dirname(os.path.abspath(__file__))
    absolute_weights_path = os.path.normpath(os.path.join(script_dir,"./last_model_weights.pth"))
    weights = torch.load(absolute_weights_path, map_location=torch.device(device))
    generator.load_state_dict(weights)
    generator.eval()
    with torch.no_grad():
        for hr_imgs, lr_imgs in test_dataset:
            hr_imgs, lr_imgs = hr_imgs.to(device), lr_imgs.to(device)
            gen_hr_imgs = generator(lr_imgs)
            lr_image_test = lr_imgs[0].squeeze().permute(1, 2, 0)
            hr_image_test = gen_hr_imgs[0].squeeze().permute(1, 2, 0)
            hr_image_test = (hr_image_test - hr_image_test.min()) / (hr_image_test.max() - hr_image_test.min())
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(lr_image_test)
            plt.title('Low Resolution')
            plt.subplot(1, 2, 2)
            plt.imshow(hr_image_test)
            plt.title('High Resolution')
            plt.show()
            break