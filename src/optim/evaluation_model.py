import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def metrics_quality(test_dl, model):
    model.eval()

    y_true_list, y_pred_list = []

    for idx, batch in enumerate(test_dl):

        images, labels = batch

        model = model.to("cuda:0")
        images = images.to("cuda:0")
        labels = labels.to("cuda:0")
        labels = labels.numpy()

        y_true_list.append(labels)

        output_model = model(images)
        output_model = output_model.to("cuda:0")
        probability_class_1 = output_model[:, 1]

        threshold = 0.50

        predictions = torch.where(
            probability_class_1 > threshold,
            torch.tensor([1]),
            torch.tensor([0]),
        )
        predicted_classes = predictions.type(torch.float)
        predicted_classes = predicted_classes.numpy()

        y_pred_list.append(predicted_classes)

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, precision, recall, f1


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
