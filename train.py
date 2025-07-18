# train.py

from tqdm.auto import tqdm
from loss import modified_elbo_loss
from dataset import load_dataloader
from model import VAE, Encoder, Decoder
import torch.nn as nn
import numpy as np
import torch

def train(model, dataloader, optimizer, device, epochs=50):
    """
    Train VAE model for one patient and print its variation of original & reconstructed signal

    Args:
        model (VAE): initialized weight of VAE
        dataloader (DataLoader): DataLoader of single beat datas(1 patient)
        optimizer : optimizer(Adam)
        device : 'cuda:0' or 'cpu'
        epochs (int, optional): epochs. Defaults to 50.

    Returns:
        VAE: trained VAE model. This model will provide trainer the holter data's latent avg & latent log variation 
    """
    model.to(device)
    model.train()

    mse_loss = nn.MSELoss()

    for epoch in range(1, epochs+1):
        total_elbo_loss = 0
        total_recons = 0
        total_samples = 0
        total_t_loss = 0
        total_qrs_loss = 0
        total_p_loss = 0

        for _, batch in enumerate(tqdm(dataloader, total=len(dataloader), desc=f'epoch: {epoch}/{epochs}')):
            batch = batch.to(device)
            optimizer.zero_grad()

            recons, mu, logvar = model(batch)

            # reconstruction losses
            reconstruction_error = mse_loss(batch, recons)
            t_loss = mse_loss(batch[:, :, :35], recons[:, :, :35])
            qrs_loss = mse_loss(batch[:, :, 35:66], recons[:, :, 35:66])
            p_loss = mse_loss(batch[:, :, 66:], recons[:, :, 66:])

            total_loss = modified_elbo_loss(mu, logvar, beta=0, t_loss=t_loss, qrs_loss=qrs_loss, p_loss=p_loss, t_theta=20.0, p_theta=25.0)

            current_batch_size = batch.size(0)
            total_loss.backward()
            optimizer.step()

            total_elbo_loss += total_loss * current_batch_size
            total_recons += reconstruction_error * current_batch_size
            total_samples += current_batch_size
            total_t_loss += t_loss * current_batch_size
            total_qrs_loss += qrs_loss * current_batch_size
            total_p_loss += p_loss * current_batch_size

            # print(f'recons error : {reconstruction_error:.4f}')

        avg_elbo_loss = total_elbo_loss / total_samples
        avg_recons_loss = total_recons / total_samples

        print(f"Epoch [{epoch}/{epochs}]\n",
              f"T_Loss: {total_t_loss / total_samples:.4f}\n",
              f"QRS_Loss: {total_qrs_loss / total_samples:.4f}\n",
              f"P_Loss: {total_p_loss / total_samples:.4f}\n",
              f"Loss: {avg_elbo_loss:.4f}\n",
              f"Recon Error: {avg_recons_loss:.4f}\n")

    # print variation of original & reconstructed signal
    model.eval()

    input_list = []
    recon_list = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, tuple):
                inputs = batch[0].to(device)
            else:
                inputs = batch.to(device)

            outputs = model(inputs)
            recons = outputs[0]  # (recons, mu, logvar)

            input_list.append(inputs.cpu())
            recon_list.append(recons.cpu())
    
    # concat all batches
    inputs_all = torch.cat(input_list, dim=0).numpy() # shape: (N, C, L)
    recon_all = torch.cat(recon_list, dim=0).numpy()  # shape: (N, C, L)

    # calculate std across samples
    input_std = np.std(inputs_all, axis=0) # shape: (C, L)
    recon_std = np.std(recon_all, axis=0)  # shape: (C, L)

    # calculate mean std to summarize overall variation
    input_std_mean = np.mean(input_std)
    recon_std_mean = np.mean(recon_std)

    # print log
    print(f"Input variation (mean std): {input_std_mean:.6f}")
    print(f"Reconstruction variation (mean std): {recon_std_mean:.6f}")

    return model



if __name__ == "__main__":
    file_directory = input('file_directory')
    dataloader = load_dataloader(file_directory=file_directory,
                                 batch_size=16,
                                 shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = VAE(in_channels=3, 
                latent_dim=4)
    
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-6,
                                 weight_decay=0.01)
    
    trained_model = train(model, dataloader, optimizer, device, epochs=20)

    torch.save(trained_model.state_dict(), '/content/drive/MyDrive/BOAZ/ADV_project/model_weight/vae_weights_250716_3.pth')
    # path 정리해서 수정하기