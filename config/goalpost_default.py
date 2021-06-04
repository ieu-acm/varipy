""" Project configuration parameters """
import imgaug

class config:
    path = "data" # Relative to home directory of repository, 
                  # includes "masked" and "original" sub-directories

    input_shape = (256,256,3)
    num_workers = 4
    val_ratio = 0.2

    valid_transforms = None

    weights_path = "weights"
    epochs = 50
    batch_size = 16

    train_transforms = iaa.Sequential([
        iaa.Crop(px=(1,16),keep_size=False),
        iaa.Fliplr(0.5),
        iaa.MotionBlur(),
        iaa.FastSnowyLandscape(
        lightness_threshold=[128, 200],
        lightness_multiplier=(1.5, 3.5)),
        iaa.Rain(speed=(0.1, 0.3)),
        iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05)),
        iaa.Fog(),
        iaa.imgcorruptlike.Brightness(severity=2),
        iaa.ShearX((-20, 20))
    ])
