import sys
import time
from abc import abstractmethod, ABCMeta
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


def accuracy(output: torch.tensor, label: torch.tensor) -> float:
    """calculate accuracy
    Args:
        output (torch.tensor): model prediction
        label (torch.tensor): label
    Returns:
        float: accuracy
    """
    total = len(output)
    label_array = np.array(label)
    output_array = np.array(output)

    assert len(label_array) == len(output_array)
    match = np.sum(label_array == output_array)
    return match / total


def train_progressbar(total: int, i: int, bar_length: int = 50, prefix: str = '', suffix: str = '') -> None:
    """progressbar
    """
    dot_num = int((i + 1) / total * bar_length)
    dot = '■' * dot_num
    empty = ' ' * (bar_length - dot_num)
    sys.stdout.write(f'\r {prefix} [{dot}{empty}] {i / total * 100:3.2f}% Done {suffix}')


class TorchModelInterface(nn.Module, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def _compute_loss(self, data: Iterable, loss_func, optimizer=None, scheduler=None, train=True):
        """ method for get loss from model
        Args:
            data (Iterable): batch data, some iterable object like list or tuple
            loss_func (func): loss function ex) nn.CrossEntropyLoss()
            optimizer (optimizer): model optimizer ex) AdamW
            scheduler (func): learning late scheduler
            train (bool): if True conducting back-propagation else it will return loss without back-propagation

        Returns: loss object(torch.loss), y(list), y_hat(list)
        """
        pass

    def save(self, file):
        """save model"""
        torch.save(self.state_dict(), file)

    def load(self, file, **kwargs):
        """load model"""
        self.load_state_dict(torch.load(file, **kwargs))

    def fit(self, epoch: 10, train_dataloader, test_dataloader, loss_func=None, optimizer=None, scheduler=None,
            callback=None, sample=1., last_epoch=0):
        if callback is None:
            callback = []
        self.zero_grad()
        total_step = int(len(train_dataloader) * sample)
        history = {}

        for e in range(epoch):
            # ------ epoch start ------
            e += last_epoch
            self.train()

            start_epoch_time = time.time()
            train_loss = 0
            output, label = [], []
            # output = torch.empty([len(train_dataloader) * batch_size])
            # label = torch.empty([len(train_dataloader) * batch_size])
            # output[step * batch_size: (step + 1) * batch_size] = y_hat
            # label[step * batch_size: (step + 1) * batch_size] = y

            for step, data in enumerate(train_dataloader):
                # ------ step start ------
                if ((step + 1) % 50 == 0) | (step + 1 >= total_step):
                    train_progressbar(
                        total_step, step + 1, bar_length=30,
                        prefix=f'train {e + 1:03d}/{epoch} epoch', suffix=f'{time.time() - start_epoch_time:0.2f} sec '
                    )

                loss, y, y_hat = self._compute_loss(data, loss_func, optimizer, scheduler, train=True)
                train_loss += loss.item()

                output.extend(y_hat)
                label.extend(y)

                if step >= total_step:
                    break
                # ------ step end ------

            history['epoch'] = e + 1
            history['time'] = np.round(time.time() - start_epoch_time, 2)

            history['train_loss'] = train_loss / total_step
            history['train_acc'] = accuracy(output, label)

            train_result = f"loss : {history['train_loss']:3.3f} acc : {history['train_acc']:3.3f}"
            sys.stdout.write(train_result)

            epoch_val_loss, output, label = self.validation(test_dataloader, loss_func)
            history['val_loss'] = epoch_val_loss
            history['val_acc'] = accuracy(output, label)
            print(f"  val_loss : {history['val_loss']:3.3f}  val_acc : {history['val_acc']:3.3f}")

            for func in callback:
                func(self, history)
            # ------ epoch end ------

    def validation(self, test_dataloader, loss_func):
        self.eval()
        total_step = len(test_dataloader)
        val_loss = 0
        output, label = [], []

        with torch.no_grad():
            for step, data in enumerate(test_dataloader):

                loss, y, y_hat = self._compute_loss(data, loss_func, train=False)
                val_loss += loss.item()
                output.extend(y_hat)
                label.extend(y)

                if step >= total_step:
                    break

        val_loss = val_loss / total_step
        return val_loss, output, label
