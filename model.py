import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import config

PRINT = False
class PrintLayer(nn.Module):
    def __init__(self, text=None,):
        super(PrintLayer, self).__init__()
        self.text = text
        self.print = PRINT
    def forward(self, x):
        # Do your print / debug stuff here
        if self.print:
            print(f"{self.text} - {x.shape}")
        return x
    
# Definir a arquitetura do VAE-GAN
class VAE_GAN(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE_GAN, self).__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            PrintLayer("encoder_1"),
            nn.Conv2d(3,3,kernel_size=2,padding=0,stride=2),
            nn.ReLU(),
            PrintLayer("econder_2"),
            nn.Conv2d(3, 3, kernel_size=3,padding=1,stride=2),
            nn.ReLU(),
            PrintLayer("encoder_3"),
            nn.Flatten(2,3),
            PrintLayer("flatten_4"),
        )
        self.fc_mean = nn.Linear(self.latent_size*self.latent_size*4, latent_size*latent_size)
        self.fc_logvar = nn.Linear(self.latent_size*self.latent_size*4, latent_size*latent_size)
        self.decoder = nn.Sequential(
            nn.Unflatten(2,(32,32)),
            PrintLayer("unflatten_1"),
            nn.ConvTranspose2d(3, 3,kernel_size=5, padding=0,stride=2),
            nn.ReLU(),
            PrintLayer("decoder_2"),
            nn.ConvTranspose2d(3, 3, kernel_size=1, padding=2,stride=2),
            nn.ReLU(),
            PrintLayer("decoder_3"),
            nn.Upsample(scale_factor=2),
            PrintLayer("upsample_4"),
            nn.Conv2d(3, 3,kernel_size=5, padding=1,stride=1),
            nn.Sigmoid(),
            PrintLayer("decoder_5"),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        if PRINT:
                print(f"mu_1 - {mu.shape}")
                print(f"logvar_1 - {logvar.shape}")
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        if PRINT:
            print(f"std_1 - {std.shape}")
            print(f"eps_1 - {eps.shape}")
        
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        if PRINT:
            print(f"reparam_1 - {z.shape}")
        recon_batch = self.decode(z)
        return recon_batch, mu, logvar

# Definir a função de perda e otimizadores
def vae_loss(recon_x, x, mu, logvar, kld_weight):
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL-divergence regularization
    #kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    # return recon_loss + KLD

    # Total VAE loss
    total_loss = recon_loss + kld_loss * kld_weight
    return total_loss

def gan_loss(D_real, D_fake):
    real_loss = F.binary_cross_entropy_with_logits(D_real, torch.ones_like(D_real))
    fake_loss = F.binary_cross_entropy_with_logits(D_fake, torch.zeros_like(D_fake))
    total_loss = real_loss + fake_loss
    return total_loss


if __name__ == '__main__':
    model = VAE_GAN(config.IN_CHANNELS,config.Z_DIM,config.LATENT_DIM).to(config.DEVICE)
    x = torch.randn(16,3,256,256, device=config.DEVICE)
    try:
        response = model(x)
    except Exception as exp:
        print (exp)
        pass