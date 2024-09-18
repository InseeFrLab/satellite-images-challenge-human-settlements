import torch
import torch.multiprocessing as multiprocessing
from torch import nn
import torchvision.models as models
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights

# Increase the shared memory limit
multiprocessing.set_sharing_strategy("file_system")


class MobileNetV3SmallModule(nn.Module):
    """
    Finetuned MobileNetV3 Small model for binary classification.

    The model is based on the MobileNetV3 Small architecture and has been trained
    to classify inputs into two labels.

    Returns:
        torch.Tensor: The output tensor containing the probabilities for each class.
    """

    def __init__(self, nbands=6):
        super().__init__()
        # Charger MobileNetV3 Small pré-entraîné
        self.model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

        # Adapter la première couche pour accepter 6 canaux
        self.model.features[0][0] = nn.Conv2d(
            nbands,
            self.model.features[0][0].out_channels,
            kernel_size=self.model.features[0][0].kernel_size,
            stride=(1, 1),  # Réduire le stride pour limiter la réduction spatiale
            padding=self.model.features[0][0].padding,
            bias=False,
        )

        # Remplacer la dernière couche fully connected pour une sortie binaire (2 classes)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, 2)
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
