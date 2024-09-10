import torch

def calculate_pourcentage_loss(output, labels):
    """
    Calculate the pourcentage of wrong predicted classes
    based on output classification predictions of a model
    and the true classes.

    Args:
        output: the output of the classification
        labels: the true classes

    """
    probability_class_1 = output[:, 1]

    # Set a threshold for class prediction
    threshold = 0.50

    # Make predictions based on the threshold
    predictions = torch.where(probability_class_1 > threshold, torch.tensor([1]), torch.tensor([0]))

    predicted_classes = predictions.type(torch.float)

    misclassified_percentage = (predicted_classes != labels).float().mean()

    return misclassified_percentage

def proportion_ones(labels):
    """
    Calculate the proportion of ones in the validation dataloader.

    Args:
        labels: the true classes

    """

    # Count the number of zeros
    num_zeros = int(torch.sum(labels == 0))

    # Count the number of ones
    num_ones = int(torch.sum(labels == 1))

    prop_ones = num_ones / (num_zeros + num_ones)

    # Rounded to two digits after the decimal point
    prop_ones = round(prop_ones, 2)

    return prop_ones