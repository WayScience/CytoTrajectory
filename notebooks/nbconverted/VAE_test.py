#!/usr/bin/env python
# coding: utf-8

# # Image-based profiles VAEs

# In[1]:
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

warnings.simplefilter("ignore")
sys.path.append("../")
from cytotraj.utils.io import load_data  # noqa

# ## Building VAE class for Image-based profiles and Loss Function
#

# In[2]:


class ImageProfileVAE(nn.Module):
    """
    VAE architecture for Image-based profiles
    """

    def __init__(self, input_dim, latent_dim):
        """Initialize the Variational Autoencoder (VAE).

        Parameters:
        ----------
        input_dim : int
            The number of input features (dimensions) in the data.
        latent_dim : int
            The dimension of the latent space.

        Returns:
        -------
        None
        """
        super(ImageProfileVAE, self).__init__()

        # building the enconder sequence
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()
        )

        # define the layers responsible for mapping the output of the encoder neural network
        self.fc_mean = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        # building the decoder sequence
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    # methods for the VAE
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input data into the latent space.

        Parameters:
        ----------
        x : torch.Tensor
            Input data.

        Returns:
        -------
        mean : torch.Tensor
            Mean of the latent space.
        logvar : torch.Tensor
            Logarithm of the variance of the latent space.
        """
        x = self.encoder(x)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterize the latent space for sampling.

        Parameters:
        ----------
        mean : torch.Tensor
            Mean of the latent space.
        logvar : torch.Tensor
            Logarithm of the variance of the latent space.

        Returns:
        -------
        z : torch.Tensor
            Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mean + epsilon * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the latent vector back into data space.

        Parameters:
        ----------
        z : torch.Tensor
            Latent vector.

        Returns:
        -------
        x_recon : torch.Tensor
            Reconstructed data.
        """

        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the VAE.

        Parameters:
        ----------
        x : torch.Tensor
            Input data.

        Returns:
        -------
        x_recon : torch.Tensor
            Reconstructed data.
        mean : torch.Tensor
            Mean of the latent space.
        logvar : torch.Tensor
            Logarithm of the variance of the latent space.
        """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decode(z)
        return x_recon, mean, logvar


# loss function
def vae_loss(
    x: np.ndarray, x_recon: np.ndarray, mean: np.ndarray, logvar: np.ndarray
) -> float:
    """
    Calculate the loss for a Variational Autoencoder (VAE) model.

    This function computes the VAE loss, which comprises two main components:
    - Reconstruction loss: It quantifies the dissimilarity between the input
    data and its reconstructed version.
    - KL divergence loss: This loss measures the divergence between the
    learned latent space and a standard Gaussian distribution.

    Parameters:
    x : np.ndarray
        The input data.
    x_recon : np.ndarray
        The reconstructed data.
    mean : np.ndarray
        The mean of the learned latent space.
    logvar : np.ndarray
        The log-variance of the learned latent space.

    Returns
    -------
    float
        The total VAE loss, which is the summation of the reconstruction loss and
        the KL divergence loss.
    """
    # MSE (Reconstruction loss)
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction="sum")

    # KL divergence eq.
    kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    # total loss
    return recon_loss + kl_divergence


# ## Creating CustomData class for VAE's
#
# The `CustomDataset` class is designed to facilitate the integration of Pandas DataFrames with PyTorch for deep learning tasks.
# It converts the data from a Pandas DataFrame into a PyTorch Tensor with the appropriate data type, allowing for efficient usage in PyTorch data loaders.
# The class includes methods for determining the dataset's length and retrieving data samples by index, making it suitable for structured data applications.
# A typical use case is to create a `CustomDataset` instance from a Pandas DataFrame, enabling data loading for training and inference in PyTorch deep learning models.
#

# In[3]:


class ImageProfileDataset(Dataset):
    """A custom dataset class for working with PyTorch and Pandas DataFrames.

    This class allows you to create a PyTorch Dataset from a Pandas DataFrame.
    It's designed to provide an easy way to load and use your data in PyTorch's
    data loading utilities.

    Parameters:
    ----------
    data : pd.DataFrame
        A Pandas DataFrame containing your data.

    Attributes:
    ----------
    data : torch.Tensor
        The data from the DataFrame, converted to a PyTorch Tensor with dtype float32.

    Methods:
    ----------
    __len__()
        Get the number of samples in the dataset.

    __getitem__(idx)
        Retrieve a sample from the dataset by its index.

    Example:
    --------
    df = pd.read_csv('your_data.csv')
    dataset = CustomDataset(df)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    """

    def __init__(self, data: pd.DataFrame):
        """Initialize the CustomDataset with the provided data.

        Parameters:
        ----------
        data : pd.DataFrame
            A Pandas DataFrame containing your data.

        Returns:
        -------
        None
        """
        self.data = torch.tensor(data.values, dtype=torch.float32)

    def __len__(self):
        """Get the number of samples in the dataset.

        Returns:
        -------
        int
            The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieve a sample from the dataset by its index.

        Parameters:
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns:
        -------
        torch.Tensor
            The data sample as a PyTorch Tensor.
        """
        return self.data[idx]


# ## Load CFReT Data

# In[4]:


sc_profile = load_data(
    "../data/localhost220512140003_KK22-05-198_sc_normalized.parquet"
)
sc_profile.head()


# In[5]:


print("removing features that do not contain real numerical values")
object_columns = sc_profile.select_dtypes(include=["object"])
print(f"columns removed {list(object_columns)}")
sc_profile = sc_profile.drop(columns=object_columns.columns)

# making all values into float32
print("making all values into float32 and drop NaN's")
sc_profile = sc_profile.astype("float32").dropna()


# In[6]:


sc_profile.info()


# ## Train VAE with CFReT Data

# In[7]:


# Hyper parameters
input_dim = len(sc_profile.columns)
latent_dim = 10
batch_size = 64
num_epochs = 5


# In[8]:


# Add Dataframe into Dataset class allowing easy integration to VAE
dataloader = ImageProfileDataset(sc_profile)


# In[9]:


# Initialize VAE and optimizer
vae = ImageProfileVAE(input_dim, latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=0.001)


# In[10]:


# training oop
for epoch in range(num_epochs):
    for data in dataloader:
        optimizer.zero_grad()
        x = data
        x_recon, mean, logvar = vae(x)
        loss = vae_loss(x, x_recon, mean, logvar)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# In[ ]:
