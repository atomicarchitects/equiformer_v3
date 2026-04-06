import torch
from e3nn import o3
from torch_cluster import radius_graph

from models.equiformer_v3.radial_function import GaussianSmearing
from models.equiformer_v3.so3 import SO3Grid, SO3Rotation
from models.equiformer_v3.transformer_block import TransBlockV3
from models.equiformer_v3.edge_rot_mat import init_edge_rot_mat


_num_nodes  = 127
_radius     = 5.0
_lmax       = 4
_mmax       = 2
_attn_grid_resolution_list = [14,  7]
_ffn_grid_resolution_list  = [14, 14]
_num_channels = 128
_num_atom_types = 128
_num_basis = 8

_times = 10


def prepare_data():
    
    inputs = torch.randn((_num_nodes, (_lmax + 1) ** 2, _num_channels))

    atomic_numbers = torch.randint(low=0, high=(_num_atom_types - 1), size=(_num_nodes, ))

    pos = torch.randn((_num_nodes, 3))    
    edge_src, edge_dst = radius_graph(pos, _radius, max_num_neighbors=_num_nodes-1)
    edge_index = (edge_src, edge_dst)

    edge_distance_vec = pos[edge_src] - pos[edge_dst]
    rbf = GaussianSmearing(
        start=0.0,
        stop=_radius,
        num_gaussians=_num_basis,
        basis_width_scalar=2.0
    )
    edge_distance = rbf(edge_distance_vec.norm(dim=-1))

    batch = torch.zeros((_num_nodes, ), dtype=torch.int64)

    return inputs, atomic_numbers, pos, edge_index, edge_distance_vec, edge_distance, batch


def test_transformer_blocks():

    inputs, atomic_numbers, pos, edge_index, edge_distance_vec, edge_distance, batch = prepare_data()
    source_atomic_numbers = atomic_numbers[edge_index[0]]
    target_atomic_numbers = atomic_numbers[edge_index[1]]

    so3_rotation = SO3Rotation(
        lmax=_lmax,
        mmax=_mmax,
        use_rotation_mask=True
    )

    block = TransBlockV3(
        num_in_channels=_num_channels,
        attn_hidden_channels=_num_channels,
        num_heads=8,
        attn_alpha_channels=_num_channels,
        attn_value_channels=_num_channels,
        ffn_hidden_channels=_num_channels,
        num_out_channels=_num_channels,
        lmax=_lmax,
        mmax=_mmax,
        so3_rotation=so3_rotation,
        attn_grid_resolution_list=_attn_grid_resolution_list,
        ffn_grid_resolution_list=_ffn_grid_resolution_list,
        max_num_elements=_num_atom_types,
        edge_channels_list=[_num_basis, 128, 128],
        use_atom_edge_embedding=True,
        attn_activation='sep-merge_gates2_swiglu',
        use_attn_renorm=True,
        use_add_merge=False,
        use_rad_l_parametrization=True,
        softcap=None,
        ffn_activation='sep-merge_gates2_swiglu',
        use_grid_mlp=True,
        norm_type='merge_rms_norm',
        alpha_drop=0.0,
        attn_mask_rate=0.0,
        attn_weights_drop=0.0,
        value_drop=0.0,
        drop_path_rate=0.0,
        proj_drop=0.0,
        ffn_drop=0.0
    )
    #print(block)
    
    rot = o3.rand_matrix()
    irreps_list = ['1x{}e'.format(i) for i in range(_lmax + 1)]
    irreps = o3.Irreps('+'.join(irreps_list))
    wigner = irreps.D_from_matrix(rot)

    inputs_rot = torch.einsum('ji, nic -> njc', wigner, inputs)

    edge_rot_mat = init_edge_rot_mat(edge_distance_vec)
    so3_rotation.set_wigner(edge_rot_mat)
    outputs = block.forward(
        x=inputs,
        source_atomic_numbers=source_atomic_numbers,
        target_atomic_numbers=target_atomic_numbers,
        edge_distance=edge_distance,
        edge_index=edge_index,
        edge_envelope_weight=None,  # for smooth cutoff
        batch=batch                 # for GraphDropPath
    )

    edge_distance_vec_rot = torch.einsum('ji, ni -> nj', rot, edge_distance_vec)
    edge_rot_mat_rot = init_edge_rot_mat(edge_distance_vec_rot)
    so3_rotation.set_wigner(edge_rot_mat_rot)
    outputs_rot = block.forward(
        x=inputs_rot,
        source_atomic_numbers=source_atomic_numbers,
        target_atomic_numbers=target_atomic_numbers,
        edge_distance=edge_distance,
        edge_index=edge_index,
        edge_envelope_weight=None,  # for smooth cutoff
        batch=batch                 # for GraphDropPath
    )

    diff = torch.einsum('ji, nic -> njc', wigner, outputs) - outputs_rot
    print('Max absolute difference: {}'.format(torch.max(torch.abs(diff))))

    return diff


if __name__ == '__main__':
    torch.manual_seed(0)

    print('lmax = {}'.format(_lmax))
    print('mmax = {}'.format(_mmax))
    print('attn_grid_resolution_list = {}'.format(_attn_grid_resolution_list))
    print('ffn_grid_resolution_list  = {}'.format(_ffn_grid_resolution_list))
    for i in range(_times):
        test_transformer_blocks()