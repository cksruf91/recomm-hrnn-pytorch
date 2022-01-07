import torch
from torch import nn, Tensor
from torch.nn.functional import one_hot

from model.model_util import TorchModelInterface


class HRNN(TorchModelInterface):

    def __init__(self, hidden_size: int, item_size: int, dropout: float = 0.5, device: torch.device = None):
        """HRNN(Hierarchical Recurrent Neural Networks)
        Personalizing Session-based Recommendations with Hierarchical Recurrent Neural Networks
        paper : https://arxiv.org/pdf/1706.04148.pdf

        Args:
            hidden_size:
            item_size:
            dropout:
            device:
        """
        super().__init__()
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

    def forward(self, input_item: Tensor, user_mask, session_mask, user_repr, session_repr):
        """
        paper: https://arxiv.org/pdf/1706.04148.pdf
        """
        item_vec = one_hot(input_item, self.item_size).float()

        user_repr_update = self.user_gru(user_repr, session_repr)
        user_repr_update = self.user_dropout(user_repr_update)
        user_repr = (user_repr_update * session_mask) + (user_repr * (1 - session_mask))
        user_repr = user_repr * (1 - user_mask)  # reset user_repr for new user

        session_repr_update = self.user_2_session(user_repr)

        # update session representation only when session changed
        session_repr = (session_repr_update * session_mask) + (session_repr * (1 - session_mask))
        # reset session representation for new user
        session_repr = session_repr * (1 - user_mask)

        x = self.session_gru(item_vec, session_repr)
        x = self.session_dropout(x)
        x = self.output_layer(x)

        return x, user_repr, session_repr

    def _compute_loss(self, data: list, loss_func, optimizer=None, scheduler=None, train=True):
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

        y_hat = torch.argmax(output, dim=-1).cpu().tolist()
        y = output_item.cpu().tolist()

        return loss, y, y_hat
