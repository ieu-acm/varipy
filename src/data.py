"""Data Processing and Batching Methods
"""

# Dependecies
import os
import glob
from typing import Tuple, Optional

import cv2
import numpy as np
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

import torch
from torch.utils.data import Dataset, DataLoader, random_split


class GoalpostDataset(Dataset):
    """ Goalpost Dataset Collecter and Augmenter

    Args:
        image_ids (list): Filenames that will be used to generate inputs
        base_path (str): Folder path that contains original images in
            "original" subfolder and masks in "masked" subfolder
        input_shape (Tuple[int, int, int]): Output image and mask size
        transforms: List of Albumentations augmentations
    """

    def __init__(self, image_ids: list,
                       base_path: str,
                       input_shape: Tuple[int, int, int],
                       transforms: Optional):

        self.__image_ids = image_ids
        self.__base_path = base_path
        self.__input_shape = input_shape
        self.__transforms = transforms

    def __id2path(self, image_id: str) -> dict:
        """ Find original and mask image paths from their ids.

        Args:
            image_id (str): Represents file name of original images and masks

        Returns:
            (dict): Original image global path in "image_path"
                key and mask global path in "mask_path" key
        """

        orig_img_search_path = os.path.join(self.__base_path,
                                            "original",
                                            f"{image_id}.*")

        original_image_path = glob.glob(orig_img_search_path)[0]

        masked_img_search_path = os.path.join(self.__base_path,
                                              "masked",
                                              f"{image_id}.*")
        masked_image_path = glob.glob(masked_img_search_path)[0]

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

    def __load_n_preprocess(self, path: str) -> np.array:
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

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
            segmap = SegmentationMapsOnImage(mask, shape=mask.shape)
            image, mask = self.__transforms(image=image.astype(np.uint8), segmentation_maps=segmap)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        image = image.permute(2, 0, 1)
        mask = mask.permute(2, 0, 1)

        return image, mask


class GoalpostDataLoader:
    """ Goalpost Data Loader

    Args:
        image_ids (list): Filenames that will be used to generate inputs
        base_path (str): Folder path that contains original images in
            "original" subfolder and masks in "masked" subfolder
        input_shape (Tuple[int, int, int]): Output image and mask size
        transforms: List of Albumentations augmentations
        batch_size (int): To be created batch's size
        val_ratio (float): Validation split rate, 0 < val_ratio < 1
        num_workers (int): Number of parallel workers
    """

    def __init__(self, image_ids: list,
                       base_path: str,
                       input_shape: Tuple[int, int, int],
                       transforms: Optional,
                       batch_size: int,
                       val_ratio: float,
                       num_workers: int):

        self.__batch_size = batch_size
        self.__num_workers = num_workers

        dataset = GoalpostDataset(image_ids=image_ids,
                                  base_path=base_path,
                                  input_shape=input_shape,
                                  transforms=transforms)

        val_sample_size = int(len(dataset)*val_ratio)
        train_sample_size = len(dataset) - val_sample_size

        self.__train_dataset, self.__val_dataset = \
            random_split(dataset, [train_sample_size, val_sample_size])

    def get(self, data_type: str) -> DataLoader:
        """ Return data loader with specified data_type

        Args:
            data_type (str): Data type to getting dataset \
                and feeding the data loader

        Returns:
            (DataLoader): DataLoader with specified dataset
        """

        if data_type == "train":
            dataset = self.__train_dataset
        elif data_type == "val":
            dataset = self.__val_dataset
        else:
            raise ValueError("Invalid choice of data type. \
                Available values: 'train', 'val'")

        dataloader = DataLoader(dataset,
                                batch_size=self.__batch_size,
                                shuffle=True,
                                num_workers=self.__num_workers)

        return dataloader
