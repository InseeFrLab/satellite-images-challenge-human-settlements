import torch
import torch.multiprocessing as multiprocessing
from torch import nn
import torchvision.models as models
from torchvision.models.vgg import VGG11_Weights

# Increase the shared memory limit
multiprocessing.set_sharing_strategy("file_system")


class VGG11Module(nn.Module):
    """
    Finetuned VGG11 model for binary classification.

    The model is based on the VGG11 architecture and has been trained
    to classify inputs into two labels.

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
            stride=(1, 1),  # Réduire le stride pour limiter la réduction spatiale
            padding=self.model.features[0].padding,
            bias=self.model.features[0].bias,
        )

        # Remplacer la dernière couche fully connected pour une sortie binaire (2 classes)
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
        probabilities = torch.softmax(output, dim=1)

        return probabilities
