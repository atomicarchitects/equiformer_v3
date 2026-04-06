import torch
from e3nn import o3
from e3nn.o3 import TensorProduct, Irreps
from models.equiformer_v3.so3 import SO3Grid


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


class SelfTensorProductS2Grid(torch.nn.Module):
    def __init__(self, lmax, resolution_list):
        super().__init__()
        self.lmax = lmax
        self.resolution_list = resolution_list
        self.so3_grid = SO3Grid(
            lmax=self.lmax,
            mmax=self.lmax,
            normalization='integral',   # setting to "component" does not work since different degrees are scaled differently
            resolution_list=self.resolution_list,
            use_m_primary=False
        )

    
    def forward(self, inputs):
        """
            1.  `inputs` shape: (N, (lmax + 1)**2, C)
        """
        assert len(inputs.shape) == 3
        grid = self.so3_grid.to_grid(inputs)
        grid = grid * grid
        outputs = self.so3_grid.from_grid(grid)
        return outputs


def irreps2array(inputs_irreps, lmax, num_channels):
    assert len(inputs_irreps.shape) == 2

    outputs = []
    start_idx = 0
    
    for l in range(lmax + 1):
        length = (2 * l + 1) * num_channels
        feature = inputs_irreps.narrow(1, start_idx, length)
        feature = feature.view(-1, num_channels, (2 * l + 1))
        feature = feature.transpose(1, 2).contiguous()
        outputs.append(feature)
        start_idx = start_idx + length
    outputs = torch.cat(outputs, dim=1)    
    return outputs


if __name__ == '__main__':
    torch.manual_seed(0)

    _num_nodes = 123
    _lmax = 6
    _num_channels = 19
    _resolution_list = [20, 20]

    irreps = Irreps([(_num_channels, (l, (-1)**l)) for l in range(_lmax + 1)])
    gtp_depthwise = GauntTensorProductDepthwise(
        irreps_in1=irreps, 
        irreps_in2=irreps, 
        irreps_out=irreps,
        irrep_normalization='none', 
        path_normalization='none', 
        internal_weights=True, 
        shared_weights=True
    )
    gtp_depthwise.weight.data = torch.ones_like(gtp_depthwise.weight.data)
    print(gtp_depthwise)

    stp_s2grid = SelfTensorProductS2Grid(
        lmax=_lmax,
        resolution_list=_resolution_list,
    )
    print(stp_s2grid)

    inputs_irreps = torch.randn(
        _num_nodes,
        _num_channels * ((_lmax + 1) ** 2)
    )
    outputs_gtp_irreps = gtp_depthwise(inputs_irreps, inputs_irreps)

    inputs_array = irreps2array(inputs_irreps, _lmax, _num_channels)
    outputs_s2grid_array = stp_s2grid(inputs_array)
    
    outputs_gtp_array = irreps2array(outputs_gtp_irreps, _lmax, _num_channels)

    print('lmax = {}, resolution_list = {}'.format(_lmax, _resolution_list))
    print('Max difference between depthwise self-tensor products implemented by Gaunt tensor product and S2 grid: {}'.format(
        torch.max(torch.abs(outputs_gtp_array - outputs_s2grid_array))
        )
    )