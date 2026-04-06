import torch
from torch_scatter import scatter
from .transformer_block import (
    EquivariantGraphAttention,
    FeedForwardNetwork,
)


_AVG_NUM_NODES = 77.81317


def get_stress_cg_change_mat():
    change_mat = torch.tensor(
        [
            [3 ** (-0.5), 0, 0, 0, 3 ** (-0.5), 0, 0, 0, 3 ** (-0.5)],
            [0, 0, 0, 0, 0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0],
            [0, 0, -(2 ** (-0.5)), 0, 0, 0, 2 ** (-0.5), 0, 0],
            [0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0, 0, 0, 0, 0],
            [0, 0, 0.5**0.5, 0, 0, 0, 0.5**0.5, 0, 0],
            [0, 2 ** (-0.5), 0, 2 ** (-0.5), 0, 0, 0, 0, 0],
            [
                -(6 ** (-0.5)),
                0,
                0,
                0,
                2 * 6 ** (-0.5),
                0,
                0,
                0,
                -(6 ** (-0.5)),
            ],
            [0, 0, 0, 0, 0, 2 ** (-0.5), 0, 2 ** (-0.5), 0],
            [-(2 ** (-0.5)), 0, 0, 0, 0, 0, 0, 0, 2 ** (-0.5)],
        ],
    )
    return change_mat


class ScalarFeedForwardNetwork(torch.nn.Module):
    """
        Args:
            num_in_channels (int):      Number of input channels
            num_hidden_channels (int):  Number of hidden channels
            num_out_channels (int):     Number of output channels
            dropout (float):            Dropout rate for the hidden features.
    """
    def __init__(
        self,
        num_in_channels,
        num_hidden_channels,
        num_out_channels,
        dropout=0.0
    ):
        super().__init__()
        self.num_in_channels = num_in_channels
        self.num_hidden_channels = num_hidden_channels
        self.num_out_channels = num_out_channels
        self.linear_1 = torch.nn.Linear(self.num_in_channels, self.num_hidden_channels, bias=True)
        self.act = torch.nn.SiLU()
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity()
        self.linear_2 = torch.nn.Linear(self.num_hidden_channels, self.num_out_channels, bias=True)


    def forward(self, inputs):
        outputs = self.linear_1(inputs)
        outputs = self.act(outputs)
        outputs = self.dropout(outputs)
        outputs = self.linear_2(outputs)
        return outputs


class EquivariantGraphAttentionStressHead(EquivariantGraphAttention):
    """
        Args:
            num_in_channels (int):      Number of input channels
            num_hidden_channels (int):  Number of hidden channels
            num_heads (int):            Number of attention heads
            attn_alpha_head (int):      Number of channels for alpha vector in each attention head
            attn_value_head (int):      Number of channels for value vector in each attention head
            num_out_channels (int):     Number of output channels
            lmax (int):                 Maximum degrees (l)
            mmax (int):                 Maximum order (m)

            so3_rotation (SO3Rotation): Class to calculate Wigner-D matrices and rotate embeddings
            grid_resolution_list (list:int):      
                                        Grid resolution list in class `SO3Grid`

            max_num_elements (int):     Maximum number of atomic numbers
            edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                            The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
            use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative distance for edge scalar features
            
            activation (str):           Type of activation function.
                                        Please see `check_activation_name()` in `./activation.py`.
            
            use_attn_renorm (bool):     Whether to re-normalize attention weights
            use_add_merge (bool):       Default: False
                                        If True, use addition to merge the source/target node features instead of concat, 
                                        which can save 2x compute when rotating with Wigner-D matrices.
            use_rad_l_parametrization (bool):
                                        Default: True
                                        If True, all the m components within the same type-L vector will share the same
                                        weight from the radial function.
            softcap (float):            Default: None
                                        If not None, use soft capping to limit the range of attention logits to
                                        [- `softcap`, + `softcap`].
            
            alpha_drop (float):         Dropout rate for the hidden features in non-linear MLP attention
            attn_mask_rate (float):     Mask rate for neighbors considered in attention
            attn_weights_drop (float):  Dropout rate for attention weights
            value_drop (float):         Dropout rate for the hidden features in non-linear value vectors
    """
    def __init__(
        self,
        num_in_channels,
        num_hidden_channels,
        num_heads,
        attn_alpha_channels,
        attn_value_channels,
        num_out_channels,
        lmax,
        mmax,
        so3_rotation,
        grid_resolution_list,
        max_num_elements,
        edge_channels_list,
        use_atom_edge_embedding=True,
        activation='sep-merge_s2_swiglu',
        use_attn_renorm=True,
        use_add_merge=False,
        use_rad_l_parametrization=True,
        softcap=None,
        alpha_drop=0.0,
        attn_mask_rate=0.0,
        attn_weights_drop=0.0,
        value_drop=0.0
    ):
        assert num_out_channels == 1
        super().__init__(
            num_in_channels,
            num_hidden_channels,
            num_heads,
            attn_alpha_channels,
            attn_value_channels,
            num_out_channels,
            lmax,
            mmax,
            so3_rotation,
            grid_resolution_list,
            max_num_elements,
            edge_channels_list,
            use_atom_edge_embedding,
            activation,
            use_attn_renorm,
            use_add_merge,
            use_rad_l_parametrization,
            softcap,
            alpha_drop,
            attn_mask_rate,
            attn_weights_drop,
            value_drop
        )
        zero_padded_change_matrix = get_stress_cg_change_mat()
        zero_padded_change_matrix[1:4, :] = 0.0
        self.register_buffer('zero_padded_change_matrix', zero_padded_change_matrix)


    def forward(
        self,
        x,
        source_atomic_numbers,
        target_atomic_numbers,
        edge_distance,
        edge_index,
        edge_envelope_weight=None,
        batch_size=None,
        batch=None,
    ):
        outputs = super().forward(
            x,
            source_atomic_numbers,
            target_atomic_numbers,
            edge_distance,
            edge_index,
            edge_envelope_weight
        )   # (num_nodes, (self.lmax + 1) ** 2, 1)
        outputs = outputs.view(outputs.shape[0], ((self.lmax + 1) ** 2))

        #stress = torch.zeros(
        #    (batch_size, 9), 
        #    device=outputs.device, 
        #    dtype=outputs.dtype
        #)
        #stress.index_add_(
        #    0, 
        #    batch, 
        #    outputs.narrow(1, 0, 9)
        #)
        #stress = stress / self.avg_num_nodes
        
        stress = scatter(
            src=outputs.narrow(1, 0, 9),
            index=batch,
            dim=0,
            dim_size=batch_size,
            reduce='mean'
        )
        
        stress = torch.einsum(
            'ni, ij -> nj',
            stress,
            self.zero_padded_change_matrix
        )   # (batch_size, 9)
        return stress


class FeedForwardNetworkStressHead(FeedForwardNetwork):
    """
        Args:
            num_in_channels (int):      Number of input channels
            num_hidden_channels (int):  Number of hidden channels
            num_out_channels (int):     Number of output channels

            lmax (int):                 Maximum degrees (l)
            mmax (int):                 Maximum order (m)

            grid_resolution_list (list:int):      
                                        Grid resolution list in class `SO3Grid`

            activation (str):           Type of activation function
            use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs.
            
            dropout (float):            Dropout rate for the hidden features.
    """
    def __init__(
        self,
        num_in_channels,
        num_hidden_channels,
        num_out_channels,
        lmax,
        mmax,
        grid_resolution_list,
        activation='sep-merge_s2_swiglu',
        use_grid_mlp=True,
        dropout=0.0,
    ):
        assert num_out_channels == 1
        super().__init__(
            num_in_channels,
            num_hidden_channels,
            num_out_channels,
            lmax,
            mmax,
            grid_resolution_list,
            activation,
            use_grid_mlp,
            dropout
        )
        zero_padded_change_matrix = get_stress_cg_change_mat()
        zero_padded_change_matrix[1:4, :] = 0.0
        self.register_buffer('zero_padded_change_matrix', zero_padded_change_matrix)
    

    def forward(
        self,
        x,
        batch_size,
        batch,
    ):
        outputs = super().forward(x)    # (num_nodes, (self.lmax + 1) ** 2, 1)
        outputs = outputs.view(outputs.shape[0], ((self.lmax + 1) ** 2))

        #stress = torch.zeros(
        #    (batch_size, 9), 
        #    device=outputs.device, 
        #    dtype=outputs.dtype
        #)
        #stress.index_add_(
        #    0, 
        #    batch, 
        #    outputs.narrow(1, 0, 9)
        #)
        #stress = stress / self.avg_num_nodes

        stress = scatter(
            src=outputs.narrow(1, 0, 9),
            index=batch,
            dim=0,
            dim_size=batch_size,
            reduce='mean'
        )
        
        stress = torch.einsum(
            'ni, ij -> nj',
            stress,
            self.zero_padded_change_matrix
        )   # (batch_size, 9)
        return stress