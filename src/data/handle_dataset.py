import albumentations as album
from albumentations.pytorch.transforms import ToTensorV2


def generate_transform(augmentation, tile_size=16):
    """
    Generates PyTorch transforms for data augmentation and preprocessing\
        for images.

    Args:
        tile_size (int): The size of the image tiles. Default to 16.
        augmentation (bool): Whether or not to include data augmentation.

    Returns:
        (albumentations.core.composition.Compose,
        albumentations.core.composition.Compose):
        A tuple containing the augmentation and preprocessing transforms.

    """
    transforms_preprocessing = album.Compose(
        [
            # album.Normalize(),
            ToTensorV2(),
        ]
    )

    if augmentation:
        transforms_augmentation = album.Compose(
            [
                album.HorizontalFlip(),
                album.VerticalFlip(),
                # album.Normalize(),
                ToTensorV2(),
            ]
        )
    else:
        transforms_augmentation = transforms_preprocessing

    return transforms_augmentation, transforms_preprocessing
