import torch
from se3_transformer_pytorch.se3_transformer_pytorch import SE3Transformer
from se3_transformer_pytorch.irr_repr import rot

def test_transformer():
    model = SE3Transformer(
        dim = 64,
        depth = 1,
        num_degrees = 2
    )

    feats = torch.randn(1, 32, 64)
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    out = model(feats, coors, mask, return_type = 0)
    assert out.shape == (1, 32, 64, 1), 'output must be of the right shape'

def test_equivariance():
    model = SE3Transformer(
        dim = 64,
        depth = 2,
        attend_self = True,
        num_degrees = 2,
        output_degrees = 2
    )

    feats = torch.randn(1, 32, 64)
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    R   = rot(15, 0, 45)
    out1 = model(feats, coors @ R, mask, return_type = 1)
    out2 = model(feats, coors, mask, return_type = 1) @ R

    diff = (out1 - out2).max()
    assert diff < 1e-4, 'is not equivariant'
