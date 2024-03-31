import torch
import torch.nn as nn
import torch.optim as optim
from inference import inference
from model import Generator,Discriminator
from dataset import getTrainDatasets ,getTestDataset
from tqdm import tqdm
from train_utils import test_output, validate
import os


def train(train_dataset,val_dataset,num_epochs,device,lr):
    adversarial_loss = nn.BCELoss()
    pixelwise_loss = nn.MSELoss()
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
    best_val_loss=float('inf')
    for epoch in range(num_epochs):
        for i, (hr_imgs, lr_imgs) in enumerate(tqdm(train_dataset, desc=f"Epoch {epoch+1}/{num_epochs}")):
            valid = torch.ones(hr_imgs.size(0), 1).to(device)
            fake = torch.zeros(hr_imgs.size(0), 1).to(device)
            hr_imgs, lr_imgs = hr_imgs.to(device), lr_imgs.to(device)
            optimizer_G.zero_grad()
            gen_hr_imgs = generator(lr_imgs)
            g_loss = adversarial_loss(discriminator(gen_hr_imgs), valid)
            g_loss += pixelwise_loss(gen_hr_imgs, hr_imgs)
            g_loss.backward()
            optimizer_G.step()
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(hr_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_hr_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
        val_loss = validate(generator, pixelwise_loss, val_dataset,device=device)
        test_output(generator, val_dataset,device=device)
        if best_val_loss >= val_loss:
            best_val_loss = val_loss
            torch.save(generator.state_dict(),'best_model_weights.pth')
            print("Model weights saved.")
        torch.save(generator.state_dict(), 'last_model_weights.pth')






def train_model(config):
    num_epochs=config['epochs']
    lr=config['learning_rate']
    batch_size=config['batch_size']
    resize_dim=(config['resize_dim'],config['resize_dim'])
    upscale_factor=config['upscale_factor']
    mode=config['mode']
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_train_path = os.path.normpath(os.path.join(script_dir,config['train_path']))
    absolute_valid_path = os.path.normpath(os.path.join(script_dir,config['val_path']))
    if(mode=='train'):
        train_dataset, val_dataset = getTrainDatasets(root_dir=absolute_train_path,batch_size=batch_size,resize_dim=resize_dim,upscale_factor=upscale_factor)
        train(train_dataset,val_dataset,num_epochs,device,lr)
    if(mode=='test'):
        test_dataset= getTestDataset(root_dir=absolute_valid_path,batch_size=batch_size,resize_dim=resize_dim,upscale_factor=upscale_factor)
        generator = Generator().to(device)
        inference(generator,test_dataset,device)