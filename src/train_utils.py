""" Data training and validation tools """
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


class TrainingManager:
    """ PyTorch Training Manager with Mixed Precision Support

    Args:
        model: PyTorch model to be trained
        loss_fn: Cost function
        train_dloader: Training data loader
        val_dloader: Validation data loader
        device: PyTorch device that specify model training hardware: \
            torch.device("cpu") or torch.device("cuda") available
    """
    def __init__(self, model, loss_fn, optimizer,
                       train_dloader, val_dloader, device, amp=False):
        self.__model = model
        self.__loss_fn = loss_fn
        self.__optimizer = optimizer
        self.__train_dloader = train_dloader
        self.__val_dloader = val_dloader
        self.__device = device
        self.__amp = amp

        if self.__amp:
            self.__scaler = GradScaler()

    def train_epoch(self, epoch: int):
        """ Train a epoch

        Args:
            epoch (int): Current epoch number
        """
        self.__model.train()

        total_loss = 0
        total_samples = 0

        pbar = tqdm(self.__train_dloader, total=len(self.__train_dloader))
        for (_input, _output) in pbar:
            self.__optimizer.zero_grad()

            _input = _input.to(self.__device).float()
            _output = _output.to(self.__device).float()

            if self.__amp:
                with autocast():
                    _prediction = self.__model(_input)
                    loss = self.__loss_fn(_prediction, _output)

                self.__scaler.scale(loss).backward()
                self.__scaler.step(self.__optimizer)
                self.__scaler.update()

            else:
                _prediction = self.__model(_input)
                loss = self.__loss_fn(_prediction, _output)

                loss.backward()
                self.__optimizer.step()

            total_loss += loss.item()
            total_samples += _input.size()[0]
            pbar.set_description(f'epoch {epoch} loss: {loss.item():.4f}')

        pbar.set_description(f'epoch {epoch} \
            average_loss: {total_loss/total_samples:.4f}')

    def validate(self, epoch):
        """ Validate a epoch

        Args:
            epoch (int): Current epoch number
        """

        self.__model.eval()

        total_loss = 0
        total_samples = 0

        pbar = tqdm(self.__val_dloader, total=len(self.__val_dloader))
        for (_input, _output) in pbar:
            _input = _input.to(self.__device).float()
            _output = _output.to(self.__device).float()

            with torch.no_grad():
                _prediction = self.__model(_input)
                loss = self.__loss_fn(_prediction, _output)

            total_loss += loss
            total_samples += _input.size()[0]

            pbar.set_description(f'epoch {epoch} val_loss: {loss.item():.4f}')

        pbar.set_description(f'epoch {epoch} \
            average_val_loss: {total_loss/total_samples:.4f}')
    
    def get_model(self):
        """ Return model architecture with weights & biases
        """    
        return self.__model
