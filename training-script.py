import torch
from models import Generator, Discriminator, StyleTransferNet
from training_utils import TrainingManager, DataProcessor
import os
from tqdm import tqdm

def train_gan(data_path, num_epochs=200, batch_size=64, save_interval=10):
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Initialize training manager and data processor
    trainer = TrainingManager(generator, discriminator, device)
    processor = DataProcessor()
    
    # Load dataset
    dataloader = processor.load_dataset(data_path, batch_size)
    
    # Training loop
    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for i, (images, _) in enumerate(progress_bar):
            real_images = images.to(device)
            
            # Training step
            d_loss, g_loss, fake_images = trainer.train_step(real_images, len(images))
            
            # Update progress bar
            progress_bar.set_postfix({
                'D Loss': f'{d_loss:.4f}',
                'G Loss': f'{g_loss:.4f}'
            })
            
            # Save sample images periodically
            if i % 100 == 0:
                processor.save_image(
                    fake_images[0],
                    f'samples/epoch_{epoch+1}_batch_{i}.png'
                )
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            trainer.save_checkpoint(
                epoch,
                f'checkpoints/checkpoint_epoch_{epoch+1}.pth'
            )

def train_style_transfer(content_path, style_path, num_epochs=100, batch_size=16):
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    style_transfer = StyleTransferNet().to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(style_transfer.parameters(), lr=0.0001)
    
    # Initialize data processor
    processor = DataProcessor()
    
    # Load datasets
    content_loader = processor.load_dataset(content_path, batch_size)
    style_loader = processor.load_dataset(style_path, batch_size)
    
    # Training loop
    for epoch in range(num_epochs):
        progress_bar = tqdm(zip