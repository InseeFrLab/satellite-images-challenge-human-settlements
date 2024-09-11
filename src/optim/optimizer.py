import torch


def generate_optimization_elements(config):
    """
    Returns the optimization elements required for PyTorch training.

    Args:
        config (dict): The configuration dictionary
        containing the optimization parameters.

    Returns:
        tuple: A tuple containing the optimizer, optimizer parameters,
        scheduler, scheduler parameters, and scheduler interval.

    """
    if config['optim'] == "adam":
        optimizer = torch.optim.Adam
        optimizer_params = {
            "lr": config["lr"],
        }
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
        scheduler_params = {
            "mode": "min",  # Mode de réduction, soit "min" ou "max"
            "factor": 0.1,  # Facteur par lequel le taux d'apprentissage sera multiplié
            "patience": 10,  # Nombre d'époques sans amélioration avant de réduire le taux d'apprentissage
        }
        scheduler_interval = "epoch"

    elif config['optim'] == "sgd":
        optimizer = torch.optim.SGD
        optimizer_params = {
            "lr": config["lr"],
            "momentum": config["momentum"],
        }
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
        scheduler_params = {}
        scheduler_interval = "epoch"

    return (
        optimizer,
        optimizer_params,
        scheduler,
        scheduler_params,
        scheduler_interval,
    )
