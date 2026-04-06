import torch
from e3nn import o3
from e3nn.o3 import TensorProduct, Irreps


const_wigner2gaunt = torch.load("./const_wigner2gaunt.pt")


"""
    1.  Reference: https://github.com/c-tl/Efficient-TP-Exp-Preview-before-Releasing/blob/2c242126b42719e077a98f9051347dcaae2d0562/test_equi_many-body.ipynb
"""
class GauntTensorProductDepthwise(TensorProduct):

    def __init__(
        self, irreps_in1, irreps_in2, irreps_out, irrep_normalization: str = None, path_normalization: str = None, **kwargs
    ):
        irreps_in1 = o3.Irreps(irreps_in1)
        irreps_in2 = o3.Irreps(irreps_in2)
        irreps_out = o3.Irreps(irreps_out)

        instr = [
            (i_1, i_2, i_out, "uuu", True, const_wigner2gaunt[ir_out.l, ir_1.l, ir_2.l] ** 2)
            for i_1, (_, ir_1) in enumerate(irreps_in1)
            for i_2, (_, ir_2) in enumerate(irreps_in2)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_out in ir_1 * ir_2
        ]
        super().__init__(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instr,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
            **kwargs,
        )


def irreps2array(inputs_irreps, lmax, num_channels):
    
    out = []
    start_idx = 0
    
    for l in range(lmax + 1):
        length = (2 * l + 1) * num_channels
        feature = inputs_irreps.narrow(1, start_idx, length)
        feature = feature.view(-1, num_channels, (2 * l + 1))
        feature = feature.transpose(1, 2).contiguous()
        out.append(feature)
        start_idx = start_idx + length
    out = torch.cat(out, dim=1)
    
    return out


def array2irreps(inputs_array, lmax, num_channels):
    out = []
    
    for l in range(lmax + 1):
        start_idx = l ** 2
        length = 2 * l + 1
        feature = inputs_array.narrow(1, start_idx, length)
        feature = feature.transpose(1, 2).contiguous()
        feature = feature.view(feature.shape[0], length * num_channels)
        out.append(feature)
    out = torch.cat(out, dim=1)
    return out