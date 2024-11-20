import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
import numpy as np
from PIL import Image

class TrainingManager:
    def __init__(self, generator, discriminator, device='cuda'):
        self.device = device
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Initialize lists to store losses
        self.g_losses = []
        self.d_losses = []

    def train_step(self, real_images, batch_size):
        # Create labels
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)

        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        # Real images
        d_output_real = self.discriminator(real_images)
        d_loss_real = self.criterion(d_output_real, real_labels)
        
        # Fake images
        z = torch.randn(batch_size, 100).to(self.device)
        fake_images = self.generator(z)
        d_output_fake = self.discriminator(fake_images.detach())
        d_loss_fake = self.criterion(d_output_fake, fake_labels)
        
        # Combined discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()

        # Train Generator
        self.g_optimizer.zero_grad()
        
        # Generate fake images again (since we detached them before)
        g_output = self.discriminator(fake_images)
        g_loss = self.criterion(g_output, real_labels)
        
        g_loss.backward()
        self.g_optimizer.step()

        return d_loss.item(), g_loss.item(), fake_images

    def save_checkpoint(self, epoch, path):
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses
        }, path)

class DataProcessor:
    def __init__(self, image_size=64):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_dataset(self, path, batch_size=64):
        dataset = ImageFolder(root=path, transform=self.transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    @staticmethod
    def save_image(tensor, path):
        image = tensor.cpu().data.numpy()
        image = (image + 1) / 2.0
        image = image.transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)
        Image.fromarray(image).save(path)
