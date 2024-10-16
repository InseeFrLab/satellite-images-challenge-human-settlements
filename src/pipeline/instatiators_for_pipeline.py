import torch.nn as nn
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader
from optim.optimizer import generate_optimization_elements

from data.handle_dataset import generate_transform
from data.prepare_data import generate_data_for_train
from data.components.classification_dataset import ClassificationDataset
from models.classification_module import ClassificationLightningModule
from models.components.resnet18_model import ResNet18Module
from models.components.mobilenet_v2_model import MobileNetV2Module
from models.components.mobilenet_v3_small_model import MobileNetV3SmallModule
from models.components.resnet34_model import ResNet34Module
from models.components.vgg11_model import VGG11Module
from models.components.vgg11_bn_model import VGG11BNModule


models_dict = {
    "mobilenet_v2": MobileNetV2Module,
    "resnet18": ResNet18Module,
    "mobilenet_v3_small": MobileNetV3SmallModule,
    "resnet34": ResNet34Module,
    "vgg11": VGG11Module,
    "vgg11_bn": VGG11BNModule
}

losses_dict = {
    "crossentropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bcelogits": nn.BCEWithLogitsLoss,
}


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

    print("*****Entre dans la fonction instantiate_dataloader*****")

    (X_train, y_train, indices_train), (X_val, y_val, indices_val), (X_test, y_test, indices_test) = generate_data_for_train(X, y, config)

    print("*****Instantiate dataset*****")

    # Retrieving the desired Dataset class
    train_dataset = ClassificationDataset(
        X_train, y_train
    )

    valid_dataset = ClassificationDataset(
        X_val, y_val
    )

    test_dataset = ClassificationDataset(
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
            ds, batch_size=size, shuffle=boolean, num_workers=72, drop_last=True
        )
        for ds, boolean, size in zip([train_dataset, valid_dataset, test_dataset], shuffle_bool, [batch_size, batch_size, batch_size_test])
    ]
    return (train_dataloader, valid_dataloader, test_dataloader), (indices_train, indices_val, indices_test)


def instantiate_dataloader_eval(X_eval, y_eval, config, ids_dict):
    """
    Instantiates and returns the data loaders for
    training, validation, and testing datasets.

    Args:
    - X_eval: images
    - y_eval: labels
    - config (dict)

    Returns:
    - eval_dataloader (torch.utils.data.DataLoader):
    The data loader for the evluation dataset.
    """

    print("*****Entre dans la fonction instantiate_dataloader_eval*****")

    # Retrieving the desired Dataset class
    eval_dataset = ClassificationDataset(
        X_eval, y_eval, ids_dict
    )

    __, t_preproc = generate_transform(
        config['augmentation']
    )

    eval_dataset.transforms = t_preproc

    # Creation of the dataloaders
    eval_dataloader = DataLoader(
            eval_dataset, batch_size=64, shuffle=False, num_workers=72, drop_last=True
        )

    return eval_dataloader


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
    nbands = len(config["bands"])

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if module_type in models_dict:
        model = models_dict[module_type]
        return model(nbands).to(device)
    else:
        print(f"Le modèle {module_type} n'a pas été implémenté")


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

    if loss_type in losses_dict:
        return losses_dict[loss_type]()
    else:
        print(f"La loss {loss_type} n'a pas été implémenté")


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

    LightningModule = ClassificationLightningModule

    lightning_module = LightningModule(
        model=instantiate_model(config),
        loss=instantiate_loss(config),
        optimizer=list_params[0],
        optimizer_params=list_params[1],
        scheduler=list_params[2],
        scheduler_params=list_params[3],
        scheduler_interval=list_params[4],
    )

    return lightning_module, LightningModule


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
