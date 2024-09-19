import torch
from torch import nn
import torchvision.models as models
from torchvision.models.efficientnet import EfficientNet_B0_Weights
import torch.multiprocessing as multiprocessing

# Increase the shared memory limit
multiprocessing.set_sharing_strategy("file_system")


class EfficientNetB0Module(nn.Module):
    """
    Finetuned EfficientNetB0 model for binary classification.
    """

    def __init__(self, nbands=6):
        super().__init__()
        # Charger EfficientNet-B0 pré-entraîné
        self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # Adapter la première couche pour accepter 6 canaux
        self.model.features[0][0] = nn.Conv2d(
            nbands,
            self.model.features[0][0].out_channels,
            kernel_size=self.model.features[0][0].kernel_size,
            stride=self.model.features[0][0].stride,
            padding=self.model.features[0][0].padding,
            bias=self.model.features[0][0].bias,
        )

        # Remplacer la dernière couche fully connected pour une sortie binaire (2 classes)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        output = self.model(input)
        probabilities = torch.softmax(output, dim=1)
        return probabilities
