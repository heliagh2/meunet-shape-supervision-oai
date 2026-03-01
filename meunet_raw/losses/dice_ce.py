import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceCELoss(nn.Module):
    """
    Cross-Entropy (ignore_index) + soft Dice.
    If logits spatial size != target size, logits are interpolated to target size.
    Assumes target in {-1,0,1,...,n_classes-1}, where -1 is ignore.
    """
    def __init__(self, n_classes: int, ignore_index: int = -1, dice_eps: float = 1e-5):
        super().__init__()
        self.n_classes = int(n_classes)
        self.ignore_index = int(ignore_index)
        self.dice_eps = float(dice_eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B,C,D,H,W); target: (B,D,H,W)
        if logits.shape[2:] != target.shape[-3:]:
            logits = F.interpolate(logits, size=target.shape[-3:], mode='trilinear', align_corners=False)

        ce = F.cross_entropy(logits, target.long(), ignore_index=self.ignore_index)

        probs = torch.softmax(logits, dim=1)  # (B,C,D,H,W)
        tgt = torch.clamp(target, min=0)
        onehot = F.one_hot(tgt.long(), num_classes=self.n_classes).permute(0,4,1,2,3).float()

        valid = (target != self.ignore_index).unsqueeze(1).float()
        probs = probs * valid
        onehot = onehot * valid

        inter = (probs * onehot).sum(dim=(0,2,3,4))
        den = probs.sum(dim=(0,2,3,4)) + onehot.sum(dim=(0,2,3,4)) + self.dice_eps
        dice = 2.0 * inter / den
        dice_loss = 1.0 - dice.mean()
        return ce + dice_loss
