import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def validate(generator, pixelwise_loss, val_dataset, device):
    generator.eval()  # Set the generator to evaluation mode
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for hr_imgs, lr_imgs in val_dataset:
            hr_imgs, lr_imgs = hr_imgs.to(device), lr_imgs.to(device)
            gen_hr_imgs = generator(lr_imgs)
            loss = pixelwise_loss(gen_hr_imgs, hr_imgs)
            total_loss += loss.item()
            num_batches += 1
    
    average_loss = total_loss / num_batches
    print(f"Validation Loss: {average_loss:.4f}")
    return average_loss




def test_output(generator,val_dataset,device):
    generator.eval()  # Set the generator to evaluation mode
    with torch.no_grad():
        for hr_imgs, lr_imgs in val_dataset:
            hr_imgs, lr_imgs = hr_imgs.to(device), lr_imgs.to(device)
            gen_hr_imgs = generator(lr_imgs)
            lr_image_test=lr_image_test.cpu()
            lr_image_test=lr_image_test.numpy()
            lr_image_test = lr_imgs[0].squeeze().permute(1, 2, 0)
            hr_image_test=hr_image_test.cpu()
            hr_image_test=hr_image_test.numpy()
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