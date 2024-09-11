import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def metrics_quality(test_dl, model):
    model.eval()

    y_true_list, y_pred_list = []

    for idx, batch in enumerate(test_dl):

        images, labels = batch

        model = model.to("cuda:0")
        images = images.to("cuda:0")
        labels = labels.to("cuda:0")
        labels = predicted_classes.numpy()

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

