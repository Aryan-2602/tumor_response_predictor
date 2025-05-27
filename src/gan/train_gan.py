import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from src.gan.models import Generator, Discriminator
from src.gan.dataset import get_gan_dataloader

# Configuration
image_size = 64
batch_size = 128
latent_dim = 100
num_epochs = 50
lr = 0.0002
beta1 = 0.5
save_dir = "generated"
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(save_dir, exist_ok=True)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def main():
    # Data
    dataloader = get_gan_dataloader("data/gan_train", image_size=image_size, batch_size=batch_size)

    # Models
    netG = Generator(latent_dim=latent_dim).to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Loss + optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

    for epoch in range(num_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            real = imgs.to(device)
            b_size = real.size(0)
            real_labels = torch.full((b_size,), 1., device=device)
            fake_labels = torch.full((b_size,), 0., device=device)

            # Train Discriminator
            netD.zero_grad()
            output_real = netD(real)
            loss_real = criterion(output_real, real_labels)

            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake = netG(noise)
            output_fake = netD(fake.detach())
            loss_fake = criterion(output_fake, fake_labels)

            d_loss = loss_real + loss_fake
            d_loss.backward()
            optimizerD.step()

            # Train Generator
            netG.zero_grad()
            output = netD(fake)
            g_loss = criterion(output, real_labels)  # Trick discriminator
            g_loss.backward()
            optimizerG.step()

            if i % 50 == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

        # Save generated images
        with torch.no_grad():
            fake_samples = netG(fixed_noise).detach().cpu()
        save_image(fake_samples, f"{save_dir}/epoch_{epoch+1:03d}.png", normalize=True)

        # Save models
        torch.save(netG.state_dict(), f"{save_dir}/netG_epoch{epoch+1}.pth")
        torch.save(netD.state_dict(), f"{save_dir}/netD_epoch{epoch+1}.pth")

if __name__ == "__main__":
    main()
