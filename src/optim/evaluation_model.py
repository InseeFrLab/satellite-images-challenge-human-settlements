import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def reality_prediction(
    test_dl, model
):
    with torch.no_grad():
        model.eval()
        y_true = []
        y_prob = []

        for idx, batch in enumerate(test_dl):

            images, labels, __ = batch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            images = images.to(device)

            output_model = model(images)
            output_model = output_model.to("cpu")
            y_prob_idx = output_model[:, 1].tolist()
            y_prob.append(y_prob_idx)

            y_true.append(labels.tolist())

            del images, labels

        y_true = np.array(y_true).flatten().tolist()
        y_prob = np.array(y_prob).flatten().tolist()

        return y_true, y_prob


def find_best_threshold(
    y_true, y_prob
):
    thresholds = np.linspace(0, 1, num=100)
    auc_list = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        auc = roc_auc_score(y_true, y_pred)
        auc_list.append(auc)

    best_threshold_idx = np.argmax(auc_list)
    best_threshold = thresholds[best_threshold_idx]

    return best_threshold


def metrics_quality(test_dl, model):
    print("***** Calcul de métriques de qualité sur le jeu de test *****")
    model.eval()

    y_true, y_prob = reality_prediction(
        test_dl, model
    )

    best_threshold = find_best_threshold(y_true, y_prob)

    predictions_best = np.where(
            y_prob > best_threshold,
            np.array([1]),
            np.array([0]),
        )

    predicted_classes_best = predictions_best.tolist()
    accuracy_best = accuracy_score(y_true, predicted_classes_best)
    precision_best = precision_score(y_true, predicted_classes_best)
    recall_best = recall_score(y_true, predicted_classes_best)
    f1_best = f1_score(y_true, predicted_classes_best)
    auc_best = roc_auc_score(y_true, predicted_classes_best)

    predictions = np.where(
        y_prob > np.array([0.5]),
        np.array([1]),
        np.array([0]),
    )
    predicted_classes = predictions.tolist()
    accuracy = accuracy_score(y_true, predicted_classes)
    precision = precision_score(y_true, predicted_classes)
    recall = recall_score(y_true, predicted_classes)
    f1 = f1_score(y_true, predicted_classes)
    auc = roc_auc_score(y_true, predicted_classes)

    return (accuracy_best, precision_best, recall_best, f1_best, auc_best), (accuracy, precision, recall, f1, auc), best_threshold


def run_eval_data(eval_dl, model, best_threshold):
    print("***** Prédiction du jeu de test à soumettre *****")
    model.eval()

    eval_submission = {}
    eval_submission_best = {}

    with torch.no_grad():
        for idx, batch in enumerate(eval_dl):

            images, __, metadata = batch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            images = images.to(device)

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

            predictions_best = torch.where(
                probability_class_1 > best_threshold,
                torch.tensor([1]),
                torch.tensor([0]),
            )
            predicted_classes_best = predictions_best.type(torch.float)
            predicted_classes_best = predicted_classes_best.numpy()

            for i, ids_dict in enumerate(metadata_dict):
                image_id = ids_dict['id']
                eval_submission_best[image_id] = int(predicted_classes_best[i])

    return eval_submission_best, eval_submission
