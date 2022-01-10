import torch


class BPRLoss:
    def __init__(self):
        """Bayesian Personalized Ranking loss
        SESSION-BASED RECOMMENDATIONS WITH RECURRENT NEURAL NETWORKS 논문에 BPR loss pytorch 버전
        defined : https://arxiv.org/pdf/1511.06939.pdf -> 3.1.3 Ranking loss 파트 BPR 부분
        (https://soobarkbar.tistory.com/147)

        example)
            output = torch.softmax(torch.rand([3, 100]), axis=1)
            label = torch.tensor([10, 32, 29, 30])
            bpr_loss = BPRLoss()
            loss = bpr_loss(output, label)
        """
        super().__init__()

    def __call__(self, output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """get BPRLoss
        Args:
            output: tensor of output score, dim: [batch size, item size]
            label: index of item, dim: [batch size + alpha]
                alpha = extra (negative) sample item

        Returns: BPR loss
        """
        score = output[:, label]
        diff = score - score.diag().view(-1, 1)
        return -torch.mean(torch.log(torch.sigmoid(diff)))


class TOP1Loss:

    def __init__(self):
        """TOP1 Loss
        SESSION-BASED RECOMMENDATIONS WITH RECURRENT NEURAL NETWORKS 논문에 TOP1 loss pytorch 버전
        defined : https://arxiv.org/pdf/1511.06939.pdf -> 3.1.3 Ranking loss TOP1 부분

        example)
            output = torch.softmax(torch.rand([3, 100]), axis=1)
            label = torch.tensor([10, 32, 29, 30])
            top1_loss = TOP1Loss()
            loss = top1_loss(output, label)
        """
        super().__init__()

    def __call__(self, output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """get TOP1 loss

        Args:
            output: tensor of output score, dim: [batch size, item size]
            label: index of item, dim: [batch size + alpha]
                alpha = extra (negative) sample item

        Returns: top1 loss
        """
        score = output[:, label]
        batch_size = score.shape[0]
        diff = score.diag().view(-1, 1) - score
        loss = torch.mean(torch.sigmoid(diff) + torch.sigmoid(score ** 2), axis=1)
        diag_l2 = torch.sigmoid(score.diag() ** 2) / batch_size
        return torch.mean(loss - diag_l2)
