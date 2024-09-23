import torch
import torch.multiprocessing as multiprocessing
from torch import nn
import torchvision.models as models
from torchvision.models.vgg import VGG11_Weights

# Increase the shared memory limit
multiprocessing.set_sharing_strategy("file_system")


class VGG11Module(nn.Module):
    """
    Adapted VGG11 model for binary classification on 16x16x6 input data.

    Returns:
        torch.Tensor: The output tensor containing the probabilities
        for each class.
    """

    def __init__(self, nbands=6):
        super().__init__()
        # Charger VGG11 pré-entraîné
        self.model = models.vgg11(weights=VGG11_Weights.DEFAULT)

        # Adapter la première couche pour accepter 6 canaux
        self.model.features[0] = nn.Conv2d(
            nbands,
            self.model.features[0].out_channels,
            kernel_size=self.model.features[0].kernel_size,
            stride=self.model.features[0].stride,
            padding=self.model.features[0].padding,
        )

        # Retirer les couches de MaxPool à la fin, car l'entrée est petite
        self.model.features = nn.Sequential(
            self.model.features[0],
            self.model.features[1],
            nn.Identity(),  # MaxPool
            self.model.features[3],
            self.model.features[4],
            nn.Identity(),  # MaxPool
            self.model.features[6],
            self.model.features[7],
            self.model.features[8],
            self.model.features[9],
            nn.Identity(),  # MaxPool
            self.model.features[11],
            self.model.features[12],
            self.model.features[13],
            self.model.features[14],
            nn.Identity(),  # MaxPool
            self.model.features[16],
            self.model.features[17],
            self.model.features[18],
            self.model.features[19],
            nn.Identity(),  # MaxPool
        )

        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, 2)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        """
        Performs the forward pass of the model.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output probabilities after applying the softmax activation.
        """
        output = self.model(input)
        probabilities = self.softmax(output)

        return probabilities
