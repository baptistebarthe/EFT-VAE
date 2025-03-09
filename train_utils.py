import torch
import torch.nn as nn
from typing import Tuple, List, Dict
import numpy as np


class VAE_EFT(nn.Module):

    def __init__(
        self,
        input_dim=13,
        intermediate_dim=28,
        latent_dim=3,
        relu_slope=1e-1,
    ):
        super(VAE_EFT, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.LeakyReLU(relu_slope),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.LeakyReLU(relu_slope),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.LeakyReLU(relu_slope),
            nn.Linear(intermediate_dim, 2 * latent_dim),
            nn.LeakyReLU(relu_slope),
        )

        # latent mu and variance
        self.mu_layer = nn.Linear(2 * latent_dim, latent_dim)
        self.logvar_layer = nn.Linear(2 * latent_dim, latent_dim)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2 * latent_dim),
            nn.LeakyReLU(relu_slope),
            nn.Linear(2 * latent_dim, intermediate_dim),
            nn.LeakyReLU(relu_slope),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.LeakyReLU(relu_slope),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.LeakyReLU(relu_slope),
            nn.Linear(intermediate_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        mu, logvar = self.mu_layer(x), self.logvar_layer(x)
        return mu, logvar

    def reparameterization(self, mu, std):
        device = mu.device
        epsilon = torch.randn_like(std).to(device)
        z = mu + std * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = (0.5 * logvar).exp()
        z = self.reparameterization(mu, std)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def loss_function(x, x_hat, mu, logvar, beta=1e-6):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    # reproduction_loss = nn.functional.mse_loss(x_hat, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return reproduction_loss, beta * KLD


def train_with_early_stopping(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    patience: int = 5,
    max_epochs: int = 100,
    min_delta: float = 1e-4,
) -> Tuple[int, float, List[float]]:
    """
    Train a model with early stopping to find optimal number of epochs.

    Args:
        model: The VAE model to train
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        optimizer: Optimizer for training
        device: Device to train on (cuda/cpu)
        patience: Number of epochs to wait for improvement before stopping
        max_epochs: Maximum number of epochs to train
        min_delta: Minimum change in loss to qualify as an improvement

    Returns:
        Tuple containing:
        - best_epoch: The epoch with lowest validation loss
        - best_loss: The lowest validation loss achieved
        - losses: List of validation losses per epoch
    """
    best_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    losses = []

    for epoch in range(max_epochs):
        print(f"=== EPOCH {epoch} ===")

        # Training phase
        model.train()
        train_repro_loss = 0.0
        train_kld_loss = 0.0

        for batch_idx, x in enumerate(train_loader):
            x = x.float().to(device)
            optimizer.zero_grad()

            x_hat, mu, logvar = model.forward(x)
            repro_loss, kld_loss = loss_function(x, x_hat, mu, logvar)

            loss = repro_loss + kld_loss

            train_repro_loss += repro_loss.item()
            train_kld_loss += kld_loss.item()

            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        valid_repro_loss = 0.0
        valid_kld_loss = 0.0

        with torch.no_grad():
            for batch_idx, x in enumerate(valid_loader):
                x = torch.clamp(x, 0, 1).float().to(device)
                x_hat, mu, logvar = model.forward(x)
                repro_loss, kld_loss = loss_function(x, x_hat, mu, logvar)

                valid_repro_loss += repro_loss.item()
                valid_kld_loss += kld_loss.item()

        # Calculate average losses
        current_loss = (valid_repro_loss + valid_kld_loss) / len(valid_loader.dataset)
        losses.append(current_loss)

        print(
            f"Training loss: {train_repro_loss/len(train_loader.dataset):.2e}, "
            f"{train_kld_loss/len(train_loader.dataset):.2e}\n"
            f"Validation loss: {valid_repro_loss/len(valid_loader.dataset):.2e}, "
            f"{valid_kld_loss/len(valid_loader.dataset):.2e}"
        )

        # Early stopping logic
        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"\nEarly stopping triggered! Best epoch was {best_epoch} "
                f"with loss {best_loss:.4e}"
            )
            break

    return best_epoch, best_loss, losses


def compute_errors(X, X_hat, loss_fn="bce"):
    n_events = X.shape[0]
    error = np.zeros(n_events)
    if loss_fn == "bce":
        for i in range(n_events):
            error[i] = nn.functional.binary_cross_entropy(
                torch.clamp(X[i, :], 0, 1),
                torch.clamp(X_hat[i, :], 0, 1),
                reduction="sum",
            )

    elif loss_fn == "mse":
        for i in range(n_events):
            error[i] = nn.functional.mse_loss(X[i, :], X_hat[i, :])
    return error


def ROC_curve(err_SM, err_BSM):
    K = np.linspace(0, max(np.max(err_SM), np.max(err_BSM)), 1000)
    FP = [np.sum(err_SM > k) / err_SM.size for k in K]
    TP = [np.sum(err_BSM > k) / err_BSM.size for k in K]
    AUC = np.trapz(np.sort(TP), np.sort(FP))

    return TP, FP, AUC


def find_optimal_latent_dim(
    model_class: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_latent_dim: int,
    **kwargs,
) -> Dict[int, Dict]:
    """
    Find the optimal latent dimension by testing multiple dimensions.

    Args:
        model_class: The VAE model class to instantiate
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        device: Device to train on (cuda/cpu)
        latent_dims: List of latent dimensions to test
        **kwargs: Additional arguments to pass to train_with_early_stopping

    Returns:
        Dictionary containing results for each latent dimension
    """
    results = {}

    for dim in range(1, max_latent_dim + 1):
        print(f"\nTesting latent dimension: {dim}")
        model = model_class(latent_dim=dim).to(device)
        optimizer = torch.optim.Adam(model.parameters())

        best_epoch, best_loss, losses = train_with_early_stopping(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            device=device,
            **kwargs,
        )

        results[dim] = {
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "loss_history": losses,
        }

        print(
            f"Latent dim {dim}: Best epoch = {best_epoch}, "
            f"Best loss = {best_loss:.4e}"
        )

    return results
