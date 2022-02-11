from typing import Tuple, Iterable, Callable

import torch
from torch import nn, Tensor
from torch.nn.functional import one_hot

from model.model_util import TorchInterfaceRecomm


class HRNN(TorchInterfaceRecomm):

    def __init__(self, hidden_size: int, item_size: int, dropout: float = 0.5, device: torch.device = None, k: int = 5):
        """HRNN(Hierarchical Recurrent Neural Networks)
        Personalizing Session-based Recommendations with Hierarchical Recurrent Neural Networks
        paper : https://arxiv.org/pdf/1706.04148.pdf

        Args:
            hidden_size: hidden dim for user/session GRU layers
            item_size: number of item, for output dim
            dropout: dropout rate 0. ~ 1.
            device: cpu for cuda
            k: parameter for metrics like recall@k, nDCG ....
        """
        super().__init__()
        self.k = k
        self.hidden_size = hidden_size
        self.item_size = item_size
        self.dropout_rate = dropout
        self.device = device

        self.user_gru = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.user_dropout = nn.Dropout(self.dropout_rate)

        self.user_2_session = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh()
        )
        self.session_gru = nn.GRUCell(self.item_size, self.hidden_size)
        self.session_dropout = nn.Dropout(self.dropout_rate)

        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.item_size),
            nn.Tanh()
        )

        self.user_repr = None
        self.session_repr = None

        self.to(device=self.device)

    def forward(self, input_item: Tensor, user_mask: Tensor, session_mask: Tensor, user_repr: Tensor,
                session_repr: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """paper: https://arxiv.org/pdf/1706.04148.pdf

        Args:
            input_item: input item indices, dim : [batch size], ex) tensor([1003,  124, 1194,    5])
            user_mask: user change indicator, dim : [batch size, 1], ex) tensor([[0],  [1], [0],    [0]])
            session_mask: session change indicator, dim : [batch size, 1], ex) tensor([[0],  [1], [0],    [1]])
            user_repr: hidden state for user GRU if first item is 0 , dim : [batch size, hidden dim]
            session_repr: hidden state for session GRU if first item is 0 , dim : [batch size, hidden dim]

        Returns:
            x: prediction result, dim : [batch size, item size]
            user_repr: hidden state for next user GRU, dim : [batch size, hidden dim]
            session_repr: hidden state for next session GRU, dim : [batch size, hidden dim]

        """
        item_vec = one_hot(input_item, self.item_size).float()

        # {the user-level GRU takes as input the session-level representation}
        user_repr_update = self.user_gru(session_repr, user_repr)
        user_repr_update = self.user_dropout(user_repr_update)
        # update user-representation only when session end
        user_repr = (user_repr_update * session_mask) + (user_repr * (1 - session_mask))
        user_repr = user_repr * (1 - user_mask)  # reset user_repr for new user

        session_repr_update = self.user_2_session(user_repr)

        # when new session start user-representation effect session init state
        session_repr = (session_repr_update * session_mask) + (session_repr * (1 - session_mask))

        # reset session representation for new user
        session_repr = session_repr * (1 - user_mask)

        session_repr = self.session_gru(item_vec, session_repr)
        session_repr = self.session_dropout(session_repr)
        x = self.output_layer(session_repr)

        return x, user_repr, session_repr

    def _compute_loss(self, data: list, loss_func, optimizer=None, scheduler=None, train=True) -> Tuple[
        Tensor, list, list]:
        """

        Args:
            data: list of tensor, model inputs, [input_item, output_item, user_mask, session_mask]
            loss_func: loss function
            optimizer: optimizer
            scheduler: learning rate scheduler
            train: default True, if set False no backpropagation is performed.

        Returns:
            loss: loss value
            y: target value dim : [batch size]
            y_hat: predict value, dim : [batch size, k]
        """
        input_item, output_item, user_mask, session_mask = data

        if self.user_repr is None:
            self.user_repr = torch.zeros(input_item.shape[0], self.hidden_size, requires_grad=False,
                                         device=self.device)  # batch size, hidden size

        if self.session_repr is None:
            self.session_repr = torch.zeros(input_item.shape[0], self.hidden_size, requires_grad=False,
                                            device=self.device)  # batch size, hidden size

        output, user_repr, session_repr = self.forward(
            input_item, user_mask, session_mask, self.user_repr, self.session_repr
        )
        self.user_repr = user_repr.detach()
        self.session_repr = session_repr.detach()

        loss = loss_func(output, output_item)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule
            self.zero_grad()

        _, indices = torch.topk(output, self.k, dim=-1)
        y_hat = indices.cpu().tolist()
        y = output_item.cpu().tolist()
        y = y[:len(y_hat)]  # drop negative sample

        # # context session 제외
        # context = [not c for c in context]
        # y_hat = list(compress(y_hat, context))
        # y = list(compress(y, context))

        return loss, y, y_hat

    def _validation(self, test_iterator: Iterable, loss_func: Callable) -> tuple[float, list[list], list[list]]:

        self.eval()
        total_step = len(test_iterator)
        val_loss = 0
        output, label = [], []

        with torch.no_grad():
            for step, (item, samples, last_item, iterator) in enumerate(test_iterator):

                self.user_repr = None
                self.session_repr = None

                for data in iterator:
                    # user_repr과 session_repr 업데이트
                    loss, _, _ = self._compute_loss(data, loss_func, train=False)

                val_loss += loss.item()
                x, _, _ = self.forward(
                    last_item, torch.tensor([[0.]]), torch.tensor([[1.]]), self.user_repr, self.session_repr
                )
                x = x[0, samples]

                _, indices = torch.topk(x, k=self.k)
                output.append(samples[indices].cpu().numpy())
                label.append([item.cpu().item()])  # positive item

        val_loss = val_loss / total_step
        return val_loss, output, label

    def get_recommendation(self, dataloader, k):

        self.k = k
        # initialize representation
        self.user_repr = None
        self.session_repr = None

        for step, data in enumerate(dataloader):
            _, _, recommend_items = self._compute_loss(data, loss_func=lambda a, b: torch.tensor([0.0]), train=False)

        return recommend_items
