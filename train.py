import os
import time
import glob
import argparse
import importlib

import torch
from torch.optim import Adam
from src.loss import dice_coef_loss
from src.data import GoalpostDataLoader
import segmentation_models_pytorch as smp
from src.train_utils import TrainingManager

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', default="goalpost_default", \
    help="Specify the config name. (It should be located in ./config folder!")
args = parser.parse_args()

try:
    config = importlib.import_module(f"config.{args.config}").config
except ImportError:
    config = importlib.import_module("config.goalpost_default")
    print("Default configuration file loaded")

weights_path = config.weights_path
if not os.path.exists(weights_path):
    os.makedirs(weights_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

all_images = glob.glob(os.path.join(config.path, "original", "*"))
image_ids = []
for image_name in all_images:
    image_id = os.path.basename(image_name).split(".")[0]
    image_ids.append(image_id)

dataloader = GoalpostDataLoader(image_ids = image_ids,
                                base_path = config.path,
                                input_shape = config.input_shape,
                                transforms = config.train_transforms,
                                batch_size = config.batch_size,
                                val_ratio = config.val_ratio,
                                num_workers = config.num_workers
                                )

train_loader = dataloader.get("train")
val_loader = dataloader.get("val")

model = smp.Unet(encoder_name = "resnet18",
                 in_channels = 3,
                 classes = 1)

optimizer = Adam(model.parameters(), lr=1e-4)
criterion = dice_coef_loss

tm = TrainingManager(model=model,
                    optimizer=optimizer,
                    loss_fn=criterion,
                    train_dloader=train_loader,
                    val_dloader=val_loader,
                    device=device)

for epoch in range(config.epochs):
    tm.train_epoch(epoch)
    tm.validate(epoch)
    modelpt = tm.get_model()
    torch.save(modelpt.state_dict(), os.path.join(weights_path, \
        f'goalpost_{epoch}_{time.time()}.pth'))
