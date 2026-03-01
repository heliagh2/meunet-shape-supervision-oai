import torch, torch.nn as nn

def conv_block(cin, cout, norm='batch'):
    norml = nn.BatchNorm3d(cout) if norm=='batch' else nn.InstanceNorm3d(cout, affine=True)
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1, bias=False), norml, nn.ReLU(inplace=True),
        nn.Conv3d(cout, cout, 3, padding=1, bias=False),
        (nn.BatchNorm3d(cout) if norm=='batch' else nn.InstanceNorm3d(cout, affine=True)),
        nn.ReLU(inplace=True)
    )

class Down(nn.Module):
    def __init__(self, cin, cout, norm='batch'):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.block = conv_block(cin, cout, norm)
    def forward(self, x): return self.block(self.pool(x))

class Up(nn.Module):
    def __init__(self, cin, cout, norm='batch'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = conv_block(cin, cout, norm)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class MEUNet3D(nn.Module):
    """
    Two heads: head1 at decoder level-1, head2 at decoder level-2.
    For expanded patches, detach after enc1 to skip grads at level-1.
    """
    def __init__(self, in_ch, n_classes, enc_ch, dec_ch, norm='batch'):
        super().__init__()
        c1,c2,c3,c4 = enc_ch
        d1,d2,d3 = dec_ch
        self.enc1 = conv_block(in_ch, c1, norm)
        self.enc2 = Down(c1, c2, norm)
        self.enc3 = Down(c2, c3, norm)
        self.enc4 = Down(c3, c4, norm)
        self.up3 = Up(c4 + c3, d3, norm)
        self.up2 = Up(d3 + c2, d2, norm)  # head2 here
        self.up1 = Up(d2 + c1, d1, norm)  # head1 here
        self.head1 = nn.Conv3d(d1, n_classes, 1)
        self.head2 = nn.Conv3d(d2, n_classes, 1)

    def forward(self, x, expanded: bool):
        e1 = self.enc1(x)
        if expanded:
            e1 = e1.detach()
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d3 = self.up3(e4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return {"logit1": self.head1(d1), "logit2": self.head2(d2)}
