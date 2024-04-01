import torch
import torch.nn as nn
import torch.optim as optim
from inference import inference
from model import Generator
from dataset import getTrainDatasets, getTestDataset
from tqdm import tqdm
from train_utils import test_output, validate
import os


def train(train_dataset, val_dataset, num_epochs, device, lr, num_residual_blocks):
    pixelwise_loss = nn.MSELoss()
    generator = Generator(num_residual_blocks=num_residual_blocks).to(device)
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        for i, (hr_imgs, lr_imgs) in enumerate(tqdm(train_dataset, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            generator.train()
            hr_imgs, lr_imgs = hr_imgs.to(device), lr_imgs.to(device)
            optimizer_G.zero_grad()
            gen_hr_imgs = generator(lr_imgs)
            g_loss = pixelwise_loss(gen_hr_imgs, hr_imgs)
            g_loss.backward()
            optimizer_G.step()
        val_loss = validate(generator, pixelwise_loss, val_dataset, device=device)
        test_output(generator, val_dataset, device=device)
        if best_val_loss >= val_loss:
            best_val_loss = val_loss
            torch.save(generator.state_dict(), 'best_model_weights.pth')
            print("Model weights saved.")
        torch.save(generator.state_dict(), 'last_model_weights.pth')


def train_model(config):
    num_epochs = config['epochs']
    lr = config['learning_rate']
    batch_size = config['batch_size']
    resize_dim = (config['resize_dim'], config['resize_dim'])
    upscale_factor = config['upscale_factor']
    mode = config['mode']
    num_residual_blocks = config['res_block']
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_train_path = os.path.normpath(os.path.join(script_dir, config['train_path']))
    absolute_valid_path = os.path.normpath(os.path.join(script_dir, config['val_path']))
    if mode == 'train':
        train_dataset, val_dataset = getTrainDatasets(root_dir=absolute_train_path, batch_size=batch_size,
                                                      resize_dim=resize_dim, upscale_factor=upscale_factor)
        train(train_dataset, val_dataset, num_epochs, device, lr, num_residual_blocks=num_residual_blocks)
    if mode == 'test':
        test_dataset = getTestDataset(root_dir=absolute_valid_path, batch_size=batch_size, resize_dim=resize_dim,
                                      upscale_factor=upscale_factor)
        generator = Generator(num_residual_blocks=num_residual_blocks).to(device)
        inference(generator, test_dataset, device)
