import torch
import torch.multiprocessing as multiprocessing
from torch import nn
import torchvision.models as models
from torchvision.models.densenet import DenseNet121_Weights

# Increase the shared memory limit
multiprocessing.set_sharing_strategy("file_system")


class DenseNet121Module(nn.Module):
    """
    Finetuned DenseNet121 model for binary classification.
    """

    def __init__(self, nbands=6):
        super().__init__()
        # Charger DenseNet121 pré-entraîné
        self.model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)

        # Adapter la première couche pour accepter 6 canaux
        self.model.features.conv0 = nn.Conv2d(
            nbands,
            self.model.features.conv0.out_channels,
            kernel_size=self.model.features.conv0.kernel_size,
            stride=self.model.features.conv0.stride,
            padding=self.model.features.conv0.padding,
            bias=self.model.features.conv0.bias,
        )

        # Remplacer la dernière couche fully connected pour une sortie binaire (2 classes)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        output = self.model(input)
        probabilities = torch.softmax(output, dim=1)
        return probabilities
