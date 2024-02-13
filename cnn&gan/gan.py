import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils

# Generator network
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output

# Discriminator network
class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)

# Hyperparameters
batch_size = 128
nz = 100
ngf = 64
ndf = 64
lr = 0.0002
beta1 = 0.5
num_epochs = 50

# Load the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.ImageFolder(root='train',transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

# Initialize the generator and discriminator networks
generator = Generator(nz, ngf, 1).cuda()
discriminator = Discriminator(1, ndf).cuda()

# Initialize the optimizer for each network
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Define the loss function for the generator
criterion_G = nn.BCELoss()

# Define the loss function for the discriminator
criterion_D = nn.BCELoss()

# Training loop
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(train_loader):
        # Train discriminator with real images
        real_images = real_images.cuda()
        batch_size = real_images.size(0)
        labels_real = torch.ones(batch_size).cuda()
        output_real = discriminator(real_images)
        loss_D_real = criterion_D(output_real, labels_real)

        # Train discriminator with fake images
        noise = torch.randn(batch_size, nz, 1, 1).cuda()
        fake_images = generator(noise)
        labels_fake = torch.zeros(batch_size).cuda()
        output_fake = discriminator(fake_images.detach())
        loss_D_fake = criterion_D(output_fake, labels_fake)

        # Total discriminator loss
        loss_D = loss_D_real + loss_D_fake

        # Backpropagation and update discriminator parameters
        discriminator.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # Train generator
        labels = torch.ones(batch_size).cuda()
        output = discriminator(fake_images)
        loss_G = criterion_G(output, labels)

        # Backpropagation and update generator parameters
        generator.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # Print training progress
        if i % 100 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f' % (epoch+1, num_epochs, i+1, len(train_loader), loss_D.item(), loss_G.item()))

    # Fixed noise for generating images throughout training
    fixed_noise = torch.randn(64, nz, 1, 1).cuda()
    # Save generated images
    fake_images = generator(fixed_noise)
    vutils.save_image(fake_images.detach(), '/generated_images/epoch_%03d.png' % (epoch+1), normalize=True)


