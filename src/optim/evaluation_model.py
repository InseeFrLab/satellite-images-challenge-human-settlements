import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data.download_data import load_data


def metrics_quality(test_dl, model):
    print("***** Calcul de métriques de qualité sur le jeu de test *****")
    model.eval()

    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for idx, batch in enumerate(test_dl):

            images, labels, __ = batch

            model = model.to("cpu")
            images = images.to("cpu")
            labels = labels.to("cpu")
            labels = labels.numpy()

            y_true_list.append(labels)

            output_model = model(images)
            output_model = output_model.to("cpu")
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


def run_eval_data(eval_dl, model):
    print("***** Prédiction du jeu de test à soumettre *****")
    model.eval()

    eval_submission = {}

    with torch.no_grad():
        for idx, batch in enumerate(eval_dl):

            images, __, metadata = batch

            model = model.to("cpu")
            images = images.to("cpu")

            output_model = model(images)
            output_model = output_model.to("cpu")
            probability_class_1 = output_model[:, 1]

            threshold = 0.50

            predictions = torch.where(
                probability_class_1 > threshold,
                torch.tensor([1]),
                torch.tensor([0]),
            )
            predicted_classes = predictions.type(torch.float)
            predicted_classes = predicted_classes.numpy()

            ids_list = metadata['ID'].tolist()
            id_strings = metadata['id']
            metadata_dict = [{'ID': id_val, 'id': id_str} for id_val, id_str in zip(ids_list, id_strings)]

            for i, ids_dict in enumerate(metadata_dict):
                image_id = ids_dict['id']
                eval_submission[image_id] = int(predicted_classes[i])

    return eval_submission
