from datasets import load_dataset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class HighLowResDatasetTest(Dataset):
    def __init__(self, root_dir, resize_dim=(512, 512),upscale_factor=8):
        self.root_dir = root_dir
        self.resize_dim = resize_dim
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.image_filenames = os.listdir(self.root_dir)
        self.upscale_factor=upscale_factor

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_filenames[idx])
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_high_res = cv2.resize(img, self.resize_dim)
        img_low_res = cv2.resize(img, (self.resize_dim[0] // self.upscale_factor, self.resize_dim[1] // self.upscale_factor))
        img_high_res = self.transform(img_high_res)
        img_low_res = self.transform(img_low_res)
        return img_high_res,img_low_res
    



class HighLowResDatasetTrain(Dataset):
    def __init__(self, root_dir, resize_dim=(512, 512),upscale_factor=8,prob=0.5):
        self.root_dir = root_dir
        self.resize_dim = resize_dim
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.image_filenames = os.listdir(self.root_dir)
        self.upscale_factor=upscale_factor
        self.transform_high_res = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
            ], p=prob),
            transforms.ToTensor()
        ])
        self.transform_low_res = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
            ], p=prob),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_filenames[idx])
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_high_res = cv2.resize(img, self.resize_dim)
        img_low_res = cv2.resize(img, (self.resize_dim[0] // self.upscale_factor, self.resize_dim[1] // self.upscale_factor))
        img_high_res = self.transform(img_high_res)
        img_low_res = self.transform(img_low_res)
        img_low_res=self.transform_low_res(img_low_res)
        img_high_res=self.transform_high_res(img_high_res)
        return img_high_res,img_low_res
    

def getTrainDatasets(resize_dim=(512, 512),batch_size = 16,validation_split = 0.2,root_dir = "",upscale_factor=8):
    dataset = HighLowResDatasetTrain(root_dir, resize_dim,upscale_factor)
    train_dataset, val_dataset = train_test_split(dataset, test_size=validation_split, random_state=42)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    for lr_image, hr_image in train_dataset:
        lr_image = lr_image.squeeze().permute(1, 2, 0)  # Assuming channels-last format
        hr_image = hr_image.squeeze().permute(1, 2, 0)  # Assuming channels-last format
        
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(lr_image)
        plt.title('Low Resolution')
        plt.subplot(1, 2, 2)
        plt.imshow(hr_image)
        plt.title('High Resolution')
        plt.show()
        break
    return train_loader,val_loader


def getTestDataset(resize_dim=(512, 512),batch_size = 16,root_dir = "",upscale_factor=8):
    test_dataset = HighLowResDatasetTest(root_dir, resize_dim,upscale_factor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return test_loader