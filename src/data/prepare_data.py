from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle
import os
import json
from data.download_data import upload_json_to_s3


def crop_and_balance_data(X, y, sample_size=50000, prop_of_zeros=0.5):
    # Step 1: Ensure sample_size does not exceed the available data

    if sample_size == len(y):
        X_sampled, y_sampled = X, y
        sampled_indices = np.arange(len(y))  # Keep all indices
    elif sample_size > len(y):
        raise ValueError("Sample size exceeds the number of available data points")
    else:
        # Step 2: Randomly sample the data
        sampled_indices = np.random.choice(len(y), size=sample_size, replace=False)
        X_sampled = X[sampled_indices]
        y_sampled = y[sampled_indices]

    # Step 3: Count the number of 1's in the sampled y
    num_ones = np.sum(y_sampled == 1)

    # Step 4: Get indices of 0's and 1's in sampled y
    ones_indices = np.where(y_sampled == 1)[0]
    zeros_indices = np.where(y_sampled == 0)[0]

    # Step 5: Randomly sample the same number of 0's as there are 1's
    balanced_zero_indices = np.random.choice(zeros_indices, int(num_ones * prop_of_zeros / (1 - prop_of_zeros)), replace=False)

    # Step 6: Combine indices of 0's and 1's
    balanced_indices = np.concatenate([ones_indices, balanced_zero_indices])

    # Step 7: Create balanced X and y
    X_balanced = X_sampled[balanced_indices]
    y_balanced = y_sampled[balanced_indices]

    # Create a list of original indices that are retained
    retained_indices = sampled_indices[balanced_indices]

    # Display the number of 0's and 1's in the balanced y
    print(f"Number of 1's in balanced y: {np.sum(y_balanced == 1)}")
    print(f"Number of 0's in balanced y: {np.sum(y_balanced == 0)}")

    # Shuffle both X_balanced and y_balanced together
    X_balanced, y_balanced = shuffle(X_balanced, y_balanced, random_state=1)

    return X_balanced, y_balanced, retained_indices


def split_data(X, y, indices, train_size=0.6, val_size=0.2, test_size=0.2):
    # Vérifier que les tailles des splits sont cohérentes
    assert train_size + val_size + test_size == 1, "Les tailles des splits doivent être égales à 1"

    # Séparer les données par classe
    X_class0 = X[y == 0]
    X_class1 = X[y == 1]

    y_class0 = y[y == 0]
    y_class1 = y[y == 1]

    indices_class0 = indices[y == 0]
    indices_class1 = indices[y == 1]

    # Diviser chaque classe en train, validation, test
    # Classe 0
    (X_train_0, X_temp_0, y_train_0, y_temp_0, indices_train_0, indices_temp_0) = train_test_split(
        X_class0, y_class0, indices_class0, train_size=train_size, stratify=y_class0
    )
    (X_val_0, X_test_0, y_val_0, y_test_0, indices_val_0, indices_test_0) = train_test_split(
        X_temp_0, y_temp_0, indices_temp_0, test_size=test_size/(test_size + val_size), stratify=y_temp_0
    )

    # Classe 1
    (X_train_1, X_temp_1, y_train_1, y_temp_1, indices_train_1, indices_temp_1) = train_test_split(
        X_class1, y_class1, indices_class1, train_size=train_size, stratify=y_class1
    )
    (X_val_1, X_test_1, y_val_1, y_test_1, indices_val_1, indices_test_1) = train_test_split(
        X_temp_1, y_temp_1, indices_temp_1, test_size=test_size/(test_size + val_size), stratify=y_temp_1
    )

    # Combiner les données de classe 0 et classe 1
    X_train = np.concatenate([X_train_0, X_train_1], axis=0)
    y_train = np.concatenate([y_train_0, y_train_1], axis=0)
    indices_train = np.concatenate([indices_train_0, indices_train_1], axis=0)

    X_val = np.concatenate([X_val_0, X_val_1], axis=0)
    y_val = np.concatenate([y_val_0, y_val_1], axis=0)
    indices_val = np.concatenate([indices_val_0, indices_val_1], axis=0)

    X_test = np.concatenate([X_test_0, X_test_1], axis=0)
    y_test = np.concatenate([y_test_0, y_test_1], axis=0)
    indices_test = np.concatenate([indices_test_0, indices_test_1], axis=0)

    # Shuffle les ensembles pour mélanger les classes
    train_indices = np.random.permutation(X_train.shape[0])
    val_indices = np.random.permutation(X_val.shape[0])
    test_indices = np.random.permutation(X_test.shape[0])

    X_train, y_train, indices_train = X_train[train_indices], y_train[train_indices], indices_train[train_indices]
    X_val, y_val, indices_val = X_val[val_indices], y_val[val_indices], indices_val[val_indices]
    X_test, y_test, indices_test = X_test[test_indices], y_test[test_indices], indices_test[test_indices]

    return (X_train, y_train, indices_train), (X_val, y_val, indices_val), (X_test, y_test, indices_test)


def generate_data_for_train(X, y, config):
    print("*****Generate data for train*****")
    if config['len data limit'] in [None, "None", 1100000]:
        len_images = len(y)
    else:
        len_images = config['len data limit']

    output_path_json = f"../data/retained_indices_{int(config['train prop']*100)}_{int(config['val prop']*100)}_{int(config['test prop']*100)}_{len_images}_{int(config['prop zeros']*100)}.json"

    if os.path.exists(output_path_json):
        print("Indices already exist")
        with open(output_path_json, 'r') as file:
            retained_indices = json.load(file)

        indices_train = retained_indices['indices_train']
        indices_val = retained_indices['indices_validation']
        indices_test = retained_indices['indices_test']

        X_train = X[indices_train]
        X_val = X[indices_val]
        X_test = X[indices_test]

        y_train = y[indices_train]
        y_val = y[indices_val]
        y_test = y[indices_test]

    else:
        X_balanced, y_balanced, retained_indices = crop_and_balance_data(
            X=X, y=y, sample_size=len_images, prop_of_zeros=config['prop zeros']
            )

        (X_train, y_train, indices_train), (X_val, y_val, indices_val), (X_test, y_test, indices_test) = split_data(
            X_balanced, y_balanced, retained_indices, config['train prop'], config['val prop'], config['test prop']
            )

        retained_indices_json = {
            "indices_train": indices_train.tolist(),
            "indices_validation": indices_val.tolist(),
            "indices_test": indices_test.tolist()
            }
        with open(output_path_json, "w") as f:
            json.dump(retained_indices_json, f)

        upload_json_to_s3(retained_indices_json, output_path_json.split("/")[-1])

    return (X_train, y_train, indices_train), (X_val, y_val, indices_val), (X_test, y_test, indices_test)
