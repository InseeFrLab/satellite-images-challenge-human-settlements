import torch
import torch.multiprocessing as multiprocessing
from torch import nn
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights

# Increase the shared memory limit
multiprocessing.set_sharing_strategy("file_system")


class MobileNetV2Module(nn.Module):
    """
    Finetuned MobileNetV2 model for binary classification.

    The model is based on the MobileNetV2 architecture and has been trained on a
    specific task to classify inputs into two labels.

    Returns:
        torch.Tensor: The output tensor containing the probabilities
        for each class.
    """

    def __init__(self, nbands=6):
        super().__init__()
        # Charger MobileNetV2 pré-entraîné
        self.model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        # Adapter la première couche pour accepter 6 canaux
        self.model.features[0][0] = nn.Conv2d(
            nbands,
            self.model.features[0][0].out_channels,
            kernel_size=self.model.features[0][0].kernel_size,
            stride=(1, 1),  # Réduire le stride pour limiter la réduction spatiale
            padding=self.model.features[0][0].padding,
            bias=False
        )

        # Remplacer la dernière couche fully connected pour une sortie binaire (2 classes)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 2)
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
