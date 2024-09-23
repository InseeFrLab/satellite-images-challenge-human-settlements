import torch
import torch.multiprocessing as multiprocessing
from torch import nn
import torchvision.models as models
from torchvision.models.vgg import VGG11_BN_Weights

# Increase the shared memory limit
multiprocessing.set_sharing_strategy("file_system")


class VGG11BNModule(nn.Module):
    """
    Finetuned VGG11_BN model for binary classification.

    The model is based on the VGG11_BN architecture and has been trained
    to classify inputs into two labels.

    Returns:
        torch.Tensor: The output tensor containing the probabilities
        for each class.
    """

    def __init__(self, nbands=6):
        super().__init__()
        # Charger VGG11_BN pré-entraîné
        self.model = models.vgg11_bn(weights=VGG11_BN_Weights.DEFAULT)

        # Adapter la première couche pour accepter 6 canaux
        self.model.features[0] = nn.Conv2d(
            nbands,
            self.model.features[0].out_channels,
            kernel_size=self.model.features[0].kernel_size,
            stride=self.model.features[0].stride,
            padding=self.model.features[0].padding,
        )

        # Retirer les couches de MaxPool
        self.model.features = nn.Sequential(
            self.model.features[0],
            self.model.features[1],
            self.model.features[2],
            self.model.features[4],
            self.model.features[5],
            self.model.features[6],
            self.model.features[8],
            self.model.features[9],
            self.model.features[10],
            self.model.features[12],
            self.model.features[13],
            self.model.features[14],
            self.model.features[16],
            self.model.features[17],
            self.model.features[18],
            self.model.features[20],
            self.model.features[21],
            self.model.features[22],
            self.model.features[24],
            self.model.features[25],
            self.model.features[26],
            self.model.features[28],
        )

        # Adapter la dernière couche fully connected pour une sortie binaire (2 classes)
        self.model.classifier = nn.Sequential(
            nn.Linear(512, 256),  # Ajustement pour correspondre à la taille de la sortie
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2),  # Sortie pour 2 classes
        )

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
