""" Project configuration parameters """

class config:
    path = "data" # Relative to home directory of repository, 
                  # includes "masked" and "original" sub-directories

    input_shape = (256,256,3)
    num_workers = 4
    val_ratio = 0.2

    train_transforms = None
    valid_transforms = None

    weights_path = "weights"
    epochs = 50
    batch_size = 16