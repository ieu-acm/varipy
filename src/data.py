"""Data Processing and Batching Methods
"""

# Dependecies
import os
import glob
from typing import Tuple, Optional

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset

class GoalpostDataset(Dataset):
    """ Goalpost Dataset Collecter and Augmenter

    Args:
        image_ids (list): Filenames that will be used to generate inputs
        base_path (str): Folder path that contains original images in
            "original" subfolder and masks in "masked" subfolder
        input_shape (Tuple[int, int, int]): Output image and mask size
        transforms: List of Albumentations augmentations
    """

    def __init__(self, image_ids:list,
                       base_path:str,
                       input_shape:Tuple[int, int, int],
                       transforms:Optional):

        self.__image_ids = image_ids
        self.__base_path = base_path
        self.__input_shape = input_shape
        self.__transforms = transforms

    def __id2path(self, image_id:str) -> dict:
        """ Find original and mask image paths from their ids.

        Args:
            image_id (str): Represents file name of original images and masks

        Returns:
            (dict): Original image global path in "image_path"
                key and mask global path in "mask_path" key
        """

        original_image_search_path = os.path.join(self.__base_path, "original", f"{image_id}.*")
        original_image_path = glob.glob(original_image_search_path)[0]

        masked_image_search_path = os.path.join(self.__base_path, "masked", f"{image_id}.*")
        masked_image_path = glob.glob(masked_image_search_path)[0]

        paths = dict()
        paths["image_path"] = original_image_path
        paths["mask_path"] = masked_image_path

        return paths

    def __len__(self) -> int:
        """ Total number of image-mask

        Args:
            -

        Returns:
            (int): Length of image id list
        """

        return len(self.__image_ids)

    def __load_n_preprocess(self, path:str) -> np.array:
        """ Load, resize and normalize RGB image.

        Args:
            path (str): Image path

        Returns:
            (np.array): Preprocessed image
        """

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.__input_shape[:2])
        image = (image-image.min()) / (image.max()-image.min())

        return image

    def __getitem__(self, i:int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Input-Output Generator by index

        Args:
            i (int): Order of data

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): Image and mask PyTorch tensor
        """

        _id = self.__image_ids[i]
        paths = self.__id2path(_id)

        image = self.__load_n_preprocess(paths["image_path"])
        mask = self.__load_n_preprocess(paths["mask_path"])

        if self.__transforms is not None:
            transformed = self.__transforms(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        image = image.permute(2,0,1)
        mask = mask.permute(2,0,1)

        return image, mask
