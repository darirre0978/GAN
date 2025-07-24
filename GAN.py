import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets, utils
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

latent_dim = 100
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
num_epochs = 30

os.makedirs("generated_images", exist_ok=True)

class Generator(nn.Module):
  def __init__(self, latent_dim):
    super(Generator, self).__init__()

    self.model = nn.Sequential(
        nn.Linear(latent_dim, 128*8*8),
        nn.ReLU(),
        nn.Unflatten(1, (128, 8, 8)),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(128, 128, kernel_size = 3, padding=1),
        nn.BatchNorm2d(128, momentum=0.78),
        nn.ReLU(),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(128, 64, kernel_size = 3, padding=1),
        nn.BatchNorm2d(64, momentum=0.78),
        nn.ReLU(),
        nn.Conv2d(64, 3, kernel_size=3, padding=1),
        nn.Tanh()
    )

  def forward(self, z):
    img = self.model(z)
    return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(256 * 5 * 5, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)
dev_loss = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
        real_images = batch[0].to(device)

        valid = torch.ones(real_images.size(0), 1, device=device)
        fake = torch.zeros(real_images.size(0), 1, device=device)

        optimizer_D.zero_grad()

        z = torch.randn(real_images.size(0), latent_dim).to(device)
        fake_images = generator(z)

        real_loss = dev_loss(discriminator(real_images), valid)
        fake_loss = dev_loss(discriminator(fake_images.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        gen_images = generator(z)
        g_loss = dev_loss(discriminator(gen_images), valid)
        g_loss.backward()
        optimizer_G.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], "
                  f"d_loss: {d_loss.item():.3f}, g_loss: {g_loss.item():.3f}")

    if (epoch + 1) % 1 == 0:
        with torch.no_grad():
            z = torch.randn(16, latent_dim, device=device)
            generated = generator(z).cpu()
            utils.save_image(generated, f"generated_images/epoch_{epoch+1}.png", nrow=4, normalize=True)

        grid = utils.make_grid(generated, nrow=4, normalize=True)
        plt.figure(figsize=(6, 6))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis("off")
        plt.title(f"Epoch {epoch+1}")
        plt.show()
