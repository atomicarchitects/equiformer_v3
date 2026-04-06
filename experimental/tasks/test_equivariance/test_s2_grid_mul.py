import torch
from e3nn import o3
from models.equiformer_v3.so3 import SO3Grid


def test_fashp_elementwise_mul_equivariance_error(lmax, resolution_list):
    _lmax = lmax
    _resolution_list = resolution_list

    inputs_1 = torch.randn(1023, (1 + _lmax) ** 2, 2049)
    inputs_2 = torch.randn(1023, (1 + _lmax) ** 2, 2049)
    
    rot = o3.rand_matrix()
    irreps_list = ['1x{}e'.format(i) for i in range(_lmax + 1)]
    irreps = o3.Irreps('+'.join(irreps_list))
    wigner = irreps.D_from_matrix(rot)

    inputs_1_rot = torch.einsum('ji, nic -> njc', wigner, inputs_1)
    inputs_2_rot = torch.einsum('ji, nic -> njc', wigner, inputs_2)

    so3_grid = SO3Grid(lmax=_lmax, mmax=_lmax, resolution_list=_resolution_list)
    
    inputs_1_grid = so3_grid.to_grid(inputs_1)
    inputs_2_grid = so3_grid.to_grid(inputs_2)
    outputs_grid = inputs_1_grid * inputs_2_grid
    outputs = so3_grid.from_grid(outputs_grid)

    inputs_1_rot_grid = so3_grid.to_grid(inputs_1_rot)
    inputs_2_rot_grid = so3_grid.to_grid(inputs_2_rot)
    outputs_rot_grid = inputs_1_rot_grid * inputs_2_rot_grid
    outputs_rot = so3_grid.from_grid(outputs_rot_grid)
    
    diff = torch.einsum('ji, nic -> njc', wigner, outputs) - outputs_rot
    #print(diff)
    print(torch.max(torch.abs(diff)))

    return diff


if __name__ == '__main__':
    """
        Empirical results:
            lmax = 1 -> resolution_list = [ 4,  4]
            lmax = 2 -> resolution_list = [ 8,  8]
            lmax = 3 -> resolution_list = [10, 10]
            lmax = 4 -> resolution_list = [14, 14]
            lmax = 6 -> resolution_list = [20, 20]
    """
    _lmax = 1
    _resolution_list = [4, 4]
    _times = 5

    max_errors = []
    mean_errors = []
    for i in range(_times):
        torch.manual_seed(i)
        diff = test_fashp_elementwise_mul_equivariance_error(_lmax, _resolution_list)
        max_errors.append(torch.max(torch.abs(diff)))
        mean_errors.append(torch.mean(torch.abs(diff)))
    
    print('lmax={}, resolution_list={}: average max error={}, average mean error={}'.format(
        _lmax,
        _resolution_list,
        sum(max_errors) / len(max_errors),
        sum(mean_errors) / len(mean_errors)
        )
    )

