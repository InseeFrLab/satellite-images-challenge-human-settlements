import torch
import torch.multiprocessing as multiprocessing
from torch import nn
import torchvision.models as models
from torchvision.models.resnet import ResNet34_Weights

# Increase the shared memory limit
multiprocessing.set_sharing_strategy("file_system")


class ResNet34Module(nn.Module):
    """
    Finetuned ResNet34 model for binary classification.

    The model is based on the ResNet34 architecture and has been trained on a
    specific task to classify inputs into two labels.

    Returns:
        torch.Tensor: The output tensor containing the probabilities
        for each class.
    """

    def __init__(self, nbands=6):
        super().__init__()
        # Charger ResNet34 pré-entraîné
        self.model = models.resnet34(weights=ResNet34_Weights.DEFAULT)

        # Adapter la première couche pour accepter 6 canaux
        self.model.conv1 = nn.Conv2d(
            nbands,
            self.model.conv1.out_channels,
            kernel_size=self.model.conv1.kernel_size,
            stride=(1, 1),  # Réduire le stride pour limiter la réduction spatiale
            padding=self.model.conv1.padding,
            bias=False,
        )

        # Supprimer MaxPool pour éviter une réduction de taille trop rapide
        self.model.maxpool = nn.Identity()

        # Remplacer la dernière couche fully connected pour une sortie binaire (2 classes)
        self.model.fc = nn.Linear(512, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        """
        Performs the forward pass of the model.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output probabilities after applying the
            softmax activation.
        """
        output = self.model(input)
        probabilities = torch.softmax(output, dim=1)

        return probabilities
