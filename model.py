# model.py

import torch.nn.functional as F
import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, activation_func, dropout_rate=0.25, in_channel=3, kernel_size=7, stride=2, padding=3, latent_dim=16):
        super(Encoder, self).__init__()
        self.activation_func = activation_func

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=256, kernel_size=kernel_size, stride=stride,padding=padding), # shape : (3, 101) -> (256, 51)
            self.activation_func,
            nn.BatchNorm1d(256)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256,kernel_size=kernel_size, stride=stride, padding=padding), # shape : (256, 51) -> (256, 26)
            self.activation_func,
            nn.BatchNorm1d(256)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=kernel_size, stride=stride, padding=padding), # shape : (256, 26) -> (512, 13)
            self.activation_func,
            nn.BatchNorm1d(512)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=kernel_size, stride=stride, padding=padding), # shape : (512, 13) ->(512, 7) ## 너무 작진 않겠지..##
            self.activation_func,
            nn.BatchNorm1d(512)
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(512*7, 512) # convolution layer하나 없애는 경우 수정요망
        self.fc2 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(dropout_rate)

        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        conv1_passed_x = self.conv1(x)
        conv2_passed_x = self.conv2(conv1_passed_x)
        conv3_passed_x = self.conv3(conv2_passed_x)
        conv4_passed_x = self.conv4(conv3_passed_x)

        flattened_x = self.flatten( conv4_passed_x )

        x = self.dropout( self.activation_func( self.fc1(flattened_x) ) )
        x = self.dropout( self.activation_func( self.fc2(x) ) )
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_latent_z(self, x):
        mu, logvar = self.forward(x)
        return self. reparameterize(mu, logvar)



class Decoder(nn.Module):
    def __init__(self, activation_func, dropout_rate=0.25, out_channel=3, kernel_size=7, stride=2, padding=3, latent_dim=16):
        super(Decoder, self).__init__()
        self.activation_func = activation_func

        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc1 = nn.Linear(512, 512*7)
        self.dropout = nn.Dropout(dropout_rate)

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose1d(512, 512, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=0),
            self.activation_func,
            nn.BatchNorm1d(512)
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=1),
            self.activation_func,
            nn.BatchNorm1d(256)
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(256, 256, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=0),
            self.activation_func,
            nn.BatchNorm1d(256)
        )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(256, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=0)
        )

    def forward(self, z):
        x = self.activation_func( self.fc3(z) )
        x = self.dropout(x)
        x = self.activation_func( self.fc2(x) )
        x = self.dropout(x)
        x = self.activation_func( self.fc1(x) )
        x = self.dropout(x)
        x = x.view(-1, 512, 7) # (B, Channels, Seq_Length), deconvolution layer하나 없애는 경우 수정요망
        x = self.deconv4(x)
        x = self.deconv3(x)
        x = self.deconv2(x)
        x = self.deconv1(x)
        return x


class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=4):
        super().__init__()
        self.encoder = Encoder(in_channel=in_channels, latent_dim=latent_dim, activation_func=nn.Tanh())
        self.decoder = Decoder(out_channel=in_channels, latent_dim=latent_dim, activation_func=nn.Tanh())

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        recons = self.decoder(z)
        return recons, mu, logvar