import gc
import os
import sys
import torch.nn as nn

import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from yaml.loader import SafeLoader

from optim.optimizer import generate_optimization_elements

from src.data.handle_dataset import generate_transform
from src.data.prepare_data import balance_data, split_data
from src.data.components import ResNet18_Dataset
from src.models.resnet18_module import ResNet18Module

from src.optim.evaluation_model import metrics_quality

from src.data.download_data import download_s3_folder, load_data


def instantiate_dataset(X, y):
    """
    Instantiates the appropriate dataset object.

    Returns:
        A dataset object of the specified type.
    """
    full_dataset = ResNet18_Dataset(X, y)
    return full_dataset


def instantiate_dataloader(X, y, config):
    """
    Instantiates and returns the data loaders for
    training, validation, and testing datasets.

    Args:
    - X: images
    - y: labels
    - config (dict)

    Returns:
    - train_dataloader (torch.utils.data.DataLoader):
    The data loader for the training dataset.
    - valid_dataloader (torch.utils.data.DataLoader):
    The data loader for the validation dataset.
    - test_dataloader (torch.utils.data.DataLoader):
    The data loader for the testing dataset.
    """

    print("Entre dans la fonction instantiate_dataloader")

    print("Balance and split data")

    X_balanced, y_balanced = balance_data(X, y)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X_balanced, y_balanced, config['prop_zeros'])

    print("Instantiate dataset")

    # Retrieving the desired Dataset class
    train_dataset = instantiate_dataset(
        X_train, y_train
    )

    valid_dataset = instantiate_dataset(
        X_val, y_val
    )

    test_dataset = instantiate_dataset(
        X_test, y_test
    )

    t_aug, t_preproc = generate_transform(
        config['augmentation']
    )

    train_dataset.transforms = t_aug
    valid_dataset.transforms = t_preproc
    test_dataset.transforms = t_preproc

    # Creation of the dataloaders
    shuffle_bool = [True, False, False]
    batch_size = config["batch size"]
    batch_size_test = config["batch size test"]

    train_dataloader, valid_dataloader, test_dataloader = [
        DataLoader(
            ds, batch_size=size, shuffle=boolean, num_workers=0, drop_last=True
        )
        for ds, boolean, size in zip([train_dataset, valid_dataset, test_dataset], shuffle_bool, [batch_size, batch_size, batch_size_test])
    ]
    return train_dataloader, valid_dataloader, test_dataloader


def instantiate_model(config):
    """
    Instantiate a module based on the provided module type.

    Args:
        module_type (str): Type of module to instantiate.

    Returns:
        object: Instance of the specified module.
    """
    print("Entre dans la fonction instantiate_model")
    module_type = config["module"]
    nbands = config["n bands"]

    if module_type in ["resnet18"]:
        return ResNet18Module(nbands)


def instantiate_loss(config):
    """
    intantiates an optimizer object with the parameters
    specified in the configuration file.

    Args:
        model: A PyTorch model object.
        config: A dictionary object containing the configuration parameters.

    Returns:
        An optimizer object from the `torch.optim` module.
    """

    print("Entre dans la fonction instantiate_loss")
    loss_type = config["loss"]

    if loss_type == "crossentropy":
        return nn.CrossEntropyLoss()


def instantiate_lightning_module(config):
    """
    Create a PyTorch Lightning module for segmentation
    with the given model and optimization configuration.

    Args:
        config (dict): Dictionary containing the configuration
        parameters for optimization.
        model: The PyTorch model to use for segmentation.

    Returns:
        A PyTorch Lightning module for segmentation.
    """
    print("Entre dans la fonction instantiate_lighting_module")
    list_params = generate_optimization_elements(config)

    if config['module'] == "resnet18":
        LightningModule = ResNet18Module

    lightning_module = LightningModule(
        model=instantiate_model(config),
        loss=instantiate_loss(config),
        optimizer=list_params[0],
        optimizer_params=list_params[1],
        scheduler=list_params[2],
        scheduler_params=list_params[3],
        scheduler_interval=list_params[4],
    )

    return lightning_module

def instantiate_trainer(config, lightning_module):
    """
    Create a PyTorch Lightning module for segmentation with
    the given model and optimization configuration.

    Args:
        config (dict): Dictionary containing the configuration
        parameters for optimization.
        model: The PyTorch model to use for segmentation.

    Returns:
        trainer: return a trainer object
    """
    # def callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="validation_loss", save_top_k=1, save_last=True, mode="min"
    )

    early_stop_callback = EarlyStopping(
        monitor="validation_loss", mode="min", patience=5
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    list_callbacks = [lr_monitor, checkpoint_callback, early_stop_callback]

    strategy = "auto"

    trainer = pl.Trainer(
        callbacks=list_callbacks,
        max_epochs=config["max epochs"],
        num_sanity_val_steps=2,
        strategy=strategy,
        log_every_n_steps=2,
        accumulate_grad_batches=config["accumulate batch"],
    )

    return trainer


def run_pipeline(run_name):
    """
    Runs the pipeline
    """
    # Open the file and load the file
    with open("../config.yml") as f:
        config = yaml.load(f, Loader=SafeLoader)

    download_s3_folder()

    X, y = load_data()

    train_dl, valid_dl, test_dl = instantiate_dataloader(X, y, config)

    torch.cuda.empty_cache()
    gc.collect()

    # train_dl.dataset[0][0].shape
    light_module = instantiate_lightning_module(config)
    trainer = instantiate_trainer(config, light_module)

    torch.cuda.empty_cache()
    gc.collect()

    if config["mlflow"]:
        remote_server_uri = "https://projet-slums-detection-mlflow.user.lab.sspcloud.fr"
        experiment_name = "challenge_mexicain"
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://minio.lab.sspcloud.fr"
        mlflow.end_run()
        mlflow.set_tracking_uri(remote_server_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name):
            mlflow.autolog()
            mlflow.log_artifact(
                "config.yml",
                artifact_path="config.yml"
            )
            for key, value in config.items():
                mlflow.log_param(key, value)

            trainer.fit(light_module, train_dl, valid_dl)

            light_module = light_module.load_from_checkpoint(
                loss=instantiate_loss(config),
                checkpoint_path=trainer.checkpoint_callback.best_model_path, #je créé un module qui charge
                model=light_module.model,
                optimizer=light_module.optimizer,
                optimizer_params=light_module.optimizer_params,
                scheduler=light_module.scheduler,
                scheduler_params=light_module.scheduler_params,
                scheduler_interval=light_module.scheduler_interval
            )
            torch.cuda.empty_cache()
            gc.collect()

            model = light_module.model

            accuracy, precision, recall, f1 = metrics_quality(test_dl, model)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

    else:
        trainer.fit(light_module, train_dl, valid_dl)

        light_module = light_module.load_from_checkpoint(
            loss=instantiate_loss(config),
            checkpoint_path=trainer.checkpoint_callback.best_model_path,
            model=light_module.model,
            optimizer=light_module.optimizer,
            optimizer_params=light_module.optimizer_params,
            scheduler=light_module.scheduler,
            scheduler_params=light_module.scheduler_params,
            scheduler_interval=light_module.scheduler_interval
        )

        torch.cuda.empty_cache()
        gc.collect()

        model = light_module.model

        accuracy, precision, recall, f1 = metrics_quality(test_dl, model)
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")


if __name__ == "__main__":
    # MLFlow param
    run_name = sys.argv[1]
    run_pipeline(run_name)
