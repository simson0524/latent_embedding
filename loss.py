# loss.py

import torch

def kl_divergence(mu, logvar, eps=1e-10):
    """
    KL divergence with average(mu) and log variation(logvar)

    Args:
        mu : average value
        logvar : variation value with ln -> ln(var)
        eps : epsilon. Defaults to 1e-10.

    Returns:
        mean of KL divergence
    """
    kl = (-0.5) * torch.sum(1+logvar-mu.pow(2)-torch.exp(logvar), dim=1)
    return kl.mean()

def modified_elbo_loss(mu, logvar, t_loss, qrs_loss, p_loss, beta=3.0, t_theta=15.0, qrs_theta=10.0, p_theta=20.0):
    """
    Loss function that proposed in paper.

    => weighted MSE + 3 * KL

    Args:
        mu : average value
        logvar : variation value with ln -> ln(var)
        t_loss : MSE loss of "T" range
        qrs_loss : MSE loss of "QRS complex" range
        p_loss : MSE loss of "P" range
        beta : hyperparameter for "KL divergence term". Defaults to 3.0.
        t_theta : hyperparameter for "t_loss term". Defaults to 15.0.
        qrs_theta : hyperparameter for "qrs_loss term". Defaults to 10.0.
        p_theta : hyperparameter for "p_loss term". Defaults to 20.0.

    Returns:
        Modified ELBO loss
    """
    weighted_mse_loss_term = (t_theta*t_loss) + (qrs_theta*qrs_loss) + (p_theta*t_loss)
    kl_divergence_term = kl_divergence(mu, logvar)
    return weighted_mse_loss_term + (beta*kl_divergence_term)