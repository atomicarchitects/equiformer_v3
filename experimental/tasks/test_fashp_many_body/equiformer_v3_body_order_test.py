"""
    1.  This file creates a specific version of EquiformerV3 for testing body order neighborhood.
        (1) We remove unused input arguments.
        (2) We add `num_ffns` for additional feedforward networks within a single Transformer block.
        (3) We add `num_output_channels` as the outputs in body order tests are logits for classification.
        (4) We add `use_attn` so that we can remove attention in Transformer blocks. This is to make sure
            we can handle the case of having only one sum aggregation (because of edge-degree embedding).
"""

import math
import torch
from functools import partial

from e3nn import o3

from fairchem.core.models.base import GraphModelMixin

from models.equiformer_v3.edge_rot_mat import init_edge_rot_mat
from models.equiformer_v3.envelope import PolynomialEnvelope
from models.equiformer_v3.so3 import (
    SO3Rotation,
    SO3Linear
)
from models.equiformer_v3.radial_function import (
    GaussianSmearing,
    RadialFunction
)
from models.equiformer_v3.layer_norm import (
    EquivariantLayerNorm,
    EquivariantSeparableLayerNorm,
    EquivariantMergeLayerNorm,
    RMSNorm
)
from models.equiformer_v3.transformer_block import (
    EquivariantGraphAttention,
    FeedForwardNetwork,
)
from models.equiformer_v3.drop import (
    GraphDropPath,
    EquivariantDropout
)

from models.equiformer_v3.input_block import EdgeDegreeEmbedding
from models.equiformer_v3.output_block import (
    ScalarFeedForwardNetwork,
)

from gaunt_self_tensor_product import (
    GauntTensorProductDepthwise,
    irreps2array,
    array2irreps,
)


# Statistics of IS2RE 100K
_AVG_NUM_NODES = 77.81317
_AVG_DEGREE = 23.395238876342773    # IS2RE: 100k, max_radius = 5, max_neighbors = 100

_NORM_SCALE_NODES = math.sqrt(_AVG_NUM_NODES)   # 8.82117735906041
_NORM_SCALE_DEGREE = math.sqrt(_AVG_DEGREE)     # 4.836862503353054


_NORM_TYPE_LIST = [
    'equivariant_layer_norm',
    'sep_layer_norm',
    'merge_layer_norm',
    'merge_layer_norm_attn_rms_norm',   # Use `EquivariantMergeLayerNorm` for the pre-norm layer 
                                        # and `RMSNorm` for attention re-normalization
    'merge_rms_norm',
    'none'
]


def get_normalization_layer(norm_type, lmax, num_channels, eps=1e-5, affine=True, normalization='component'):
    assert norm_type in _NORM_TYPE_LIST
    if norm_type == 'equivariant_layer_norm':
        norm_class = EquivariantLayerNorm
    elif norm_type == 'sep_layer_norm':
        norm_class = EquivariantSeparableLayerNorm
    elif norm_type in ['merge_layer_norm', 'merge_layer_norm_attn_rms_norm']:
        norm_class = EquivariantMergeLayerNorm
    elif norm_type == 'merge_rms_norm':
        norm_class = partial(EquivariantMergeLayerNorm, centering=False)
    elif norm_type == 'none':
        return torch.nn.Identity()
    else:
        raise ValueError
    return norm_class(lmax, num_channels, eps, affine, normalization)


class NormFeedForwardNetworkAdd(torch.nn.Module):
    """
        Args:
            num_in_channels (int):      Number of input channels
            num_hidden_channels (int):  Number of hidden channels used during feedforward network
            num_out_channels (int):     Number of output channels

            lmax (int):                 Maximum degrees (l)
            mmax (int):                 Maximum order (m)

            grid_resolution_list (list:int):      
                                        Grid resolution list in class `SO3Grid` in feedforward network
            activation (str):           Type of activation function for feedforward network
            use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFN.

            norm_type (str):            Type of normalization layer
            
            drop_path_rate (float):     Drop path rate
            proj_drop (float):          Dropout rate for outputs of attention and FFN
            dropout (float):            Dropout rate for the hidden features in FFN
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
        norm_type='merge_layer_norm',
        drop_path_rate=0.0,
        proj_drop=0.0,
        dropout=0.0, 
    ):
        super().__init__()
        self.norm = get_normalization_layer(norm_type, lmax=lmax, num_channels=num_in_channels)
        self.ffn = FeedForwardNetwork(
            num_in_channels=num_in_channels,
            num_hidden_channels=num_hidden_channels,
            num_out_channels=num_out_channels,
            lmax=lmax,
            mmax=mmax,
            grid_resolution_list=grid_resolution_list,
            activation=activation,
            use_grid_mlp=use_grid_mlp,
            dropout=dropout,
        )
        if hasattr(self.ffn, 'gating_linear'):
            self.ffn.gating_linear.weight.data.mul_(0.0)
        self.drop_path = GraphDropPath(drop_path_rate) if drop_path_rate > 0.0 else None
        self.proj_drop = EquivariantDropout(lmax=lmax, mmax=lmax, drop_prob=proj_drop) if proj_drop > 0.0 else torch.nn.Identity()
        self.ffn_shortcut = SO3Linear(num_in_channels, num_out_channels, lmax=lmax) if num_in_channels != num_out_channels else torch.nn.Identity()


    def forward(self, inputs, batch):
        x_res = inputs

        outputs = self.norm(inputs)
        outputs = self.ffn(outputs)
        if self.drop_path is not None:
            outputs = self.drop_path(outputs, batch)
        outputs = self.proj_drop(outputs)

        x_res = self.ffn_shortcut(x_res)

        outputs = outputs + x_res

        return outputs


class NormFeedForwardNetworkGauntSelfTensorProductAdd(torch.nn.Module):
    """
        Args:
            num_in_channels (int):      Number of input channels
            num_hidden_channels (int):  Number of hidden channels used during feedforward network
            num_out_channels (int):     Number of output channels

            lmax (int):                 Maximum degrees (l)
            mmax (int):                 Maximum order (m)

            grid_resolution_list (list:int):      
                                        Grid resolution list in class `SO3Grid` in feedforward network
            activation (str):           Type of activation function for feedforward network
            use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFN.

            norm_type (str):            Type of normalization layer
            
            drop_path_rate (float):     Drop path rate
            proj_drop (float):          Dropout rate for outputs of attention and FFN
            dropout (float):            Dropout rate for the hidden features in FFN
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
        norm_type='merge_layer_norm',
        drop_path_rate=0.0,
        proj_drop=0.0,
        dropout=0.0, 
    ):
        super().__init__()

        self.lmax = lmax
        self.num_hidden_channels = num_hidden_channels

        self.norm = get_normalization_layer(norm_type, lmax=lmax, num_channels=num_in_channels)

        self.linear_1 = SO3Linear(
            in_features=num_in_channels, 
            out_features=num_hidden_channels, 
            lmax=lmax, 
            bias=True
        )

        irreps = o3.Irreps([(num_hidden_channels, (l, (-1)**l)) for l in range(lmax + 1)])
        self.gaunt_self_tp = GauntTensorProductDepthwise(
            irreps,
            irreps,
            irreps,
            internal_weights=True, 
            shared_weights=True,
            irrep_normalization='component', 
            path_normalization='path', 
        )

        self.linear_2 = SO3Linear(
            in_features=num_hidden_channels, 
            out_features=num_out_channels, 
            lmax=lmax, 
            bias=True
        )

        self.drop_path = GraphDropPath(drop_path_rate) if drop_path_rate > 0.0 else None
        self.proj_drop = EquivariantDropout(lmax=lmax, mmax=lmax, drop_prob=proj_drop) if proj_drop > 0.0 else torch.nn.Identity()
        self.ffn_shortcut = SO3Linear(num_in_channels, num_out_channels, lmax=lmax) if num_in_channels != num_out_channels else torch.nn.Identity()


    def forward(self, inputs, batch):
        x_res = inputs

        outputs = self.norm(inputs)
        outputs = self.linear_1(outputs)
        outputs = array2irreps(outputs, lmax=self.lmax, num_channels=self.num_hidden_channels)
        outputs = self.gaunt_self_tp(outputs, outputs)
        outputs = irreps2array(outputs, lmax=self.lmax, num_channels=self.num_hidden_channels)
        outputs = self.linear_2(outputs)
        if self.drop_path is not None:
            outputs = self.drop_path(outputs, batch)
        outputs = self.proj_drop(outputs)

        x_res = self.ffn_shortcut(x_res)

        outputs = outputs + x_res

        return outputs
    

class TransBlockV3BodyOrderTest(torch.nn.Module):
    """
        Args:
            num_in_channels (int):      Number of input channels
            attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
            num_heads (int):            Number of attention heads
            attn_alpha_head (int):      Number of channels for alpha vector in each attention head
            attn_value_head (int):      Number of channels for value vector in each attention head
            ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
            num_out_channels (int):     Number of output channels

            lmax (int):                 Maximum degrees (l)
            mmax (int):                 Maximum order (m)

            so3_rotation (SO3Rotation): Class to calculate Wigner-D matrices and rotate embeddings
            attn_grid_resolution_list (list:int):      
                                        Grid resolution list in class `SO3Grid` in attention
            ffn_grid_resolution_list (list:int):      
                                        Grid resolution list in class `SO3Grid` in feedforward network
            
            max_num_elements (int):     Maximum number of atomic numbers
            edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                            The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
            use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative distance for edge scalar features
            
            attn_activation (str):      Type of activation function for equivariant graph attention
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

            ffn_activation (str):       Type of activation function for feedforward network
            use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFN.
            
            norm_type (str):            Type of normalization layer

            alpha_drop (float):         Dropout rate for the hidden features in non-linear MLP attention
            attn_mask_rate (float):     Mask rate for neighbors considered in attention
            attn_weights_drop (float):  Dropout rate for attention weights
            value_drop (float):         Dropout rate for the hidden features in non-linear value vectors.
            drop_path_rate (float):     Drop path rate
            proj_drop (float):          Dropout rate for outputs of attention and FFN
            ffn_drop (float):           Dropout rate for the hidden features in FFN

            use_attn (bool):            Default: True
                                        Whether to have the attention sub-block
            num_ffns (int):             Number of feedforward networks
            use_gaunt_self_tensor_product (bool):
                                        Default: False
                                        Whether to use Gaunt tensor products with path-level learnable weights
                                        in feedforward networks
    """
    def __init__(
        self,
        num_in_channels,
        attn_hidden_channels,
        num_heads,
        attn_alpha_channels,
        attn_value_channels,
        ffn_hidden_channels,
        num_out_channels,
        lmax,
        mmax,
        so3_rotation,
        attn_grid_resolution_list,
        ffn_grid_resolution_list,
        max_num_elements,
        edge_channels_list,
        use_atom_edge_embedding=True,
        attn_activation='sep-merge_s2_swiglu',
        use_attn_renorm=True,
        use_add_merge=False,
        use_rad_l_parametrization=True,
        softcap=None,
        ffn_activation='sep-merge_s2_swiglu',
        use_grid_mlp=True,
        norm_type='sep_layer_norm',
        alpha_drop=0.0,
        attn_mask_rate=0.0,
        attn_weights_drop=0.0,
        value_drop=0.0,
        drop_path_rate=0.0,
        proj_drop=0.0,
        ffn_drop=0.0,
        use_attn=True,
        num_ffns=1,
        use_gaunt_self_tensor_product=False,
    ):
        super().__init__()

        self.norm_1 = get_normalization_layer(norm_type, lmax=lmax, num_channels=num_in_channels)

        self.ga = EquivariantGraphAttention(
            num_in_channels=num_in_channels,
            num_hidden_channels=attn_hidden_channels,
            num_heads=num_heads,
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels,
            num_out_channels=num_in_channels,
            lmax=lmax,
            mmax=mmax,
            so3_rotation=so3_rotation,
            grid_resolution_list=attn_grid_resolution_list,
            max_num_elements=max_num_elements,
            edge_channels_list=edge_channels_list,
            use_atom_edge_embedding=use_atom_edge_embedding,
            activation=attn_activation,
            use_attn_renorm=use_attn_renorm,
            use_add_merge=use_add_merge,
            use_rad_l_parametrization=use_rad_l_parametrization,
            softcap=softcap,
            alpha_drop=alpha_drop,
            attn_mask_rate=attn_mask_rate,
            attn_weights_drop=attn_weights_drop,
            value_drop=value_drop
        )

        if 'rms_norm' in norm_type:
            if self.ga.alpha_norm is not None:
                del self.ga.alpha_norm
                self.ga.alpha_norm = RMSNorm(attn_alpha_channels)

        self.use_attn = use_attn
        if not self.use_attn:
            del self.norm_1
            del self.ga

        self.drop_path = GraphDropPath(drop_path_rate) if drop_path_rate > 0.0 else None
        self.proj_drop = EquivariantDropout(lmax=lmax, mmax=lmax, drop_prob=proj_drop) if proj_drop > 0.0 else None

        assert num_in_channels == num_out_channels

        self.ffn_module = torch.nn.ModuleList()

        ffn_class = NormFeedForwardNetworkAdd if not use_gaunt_self_tensor_product else NormFeedForwardNetworkGauntSelfTensorProductAdd
        for i in range(num_ffns):
            self.ffn_module.append(
                ffn_class(
                    num_in_channels,
                    ffn_hidden_channels,
                    num_out_channels,
                    lmax,
                    mmax,
                    ffn_grid_resolution_list,
                    activation=ffn_activation,
                    use_grid_mlp=use_grid_mlp,
                    norm_type=norm_type,
                    drop_path_rate=drop_path_rate,
                    proj_drop=proj_drop,
                    dropout=ffn_drop,
                )
            )


    def forward(
        self,
        x,                          # torch.Tensor
        source_atomic_numbers,
        target_atomic_numbers,
        edge_distance,
        edge_index,
        edge_envelope_weight=None,  # for smooth cutoff
        batch=None                  # for GraphDropPath
    ):
        outputs = x
        x_res = x

        if self.use_attn:
            outputs = self.norm_1(outputs)
            outputs = self.ga(
                outputs,
                source_atomic_numbers,
                target_atomic_numbers,
                edge_distance,
                edge_index,
                edge_envelope_weight
            )

            if self.drop_path is not None:
                outputs = self.drop_path(outputs, batch)
            if self.proj_drop is not None:
                outputs = self.proj_drop(outputs)

            outputs = outputs + x_res

        #x_res = outputs
        #outputs = self.norm_2(outputs)
        for i in range(len(self.ffn_module)):
            outputs = self.ffn_module[i](outputs, batch)
        #outputs = outputs + x_res
        return outputs
    

class EquiformerV3BodyOrderTest(torch.nn.Module, GraphModelMixin):
    """
    Args:
        max_neighbors (int):    Maximum number of neighbors per atom
        max_radius (float):     Maximum distance between nieghboring atoms in Angstroms
        num_radial_basis (int): Number of radial basis functions
        max_num_elements (int): Maximum atomic number

        num_layers (int):           Number of layers in the GNN
        num_channels (int):         Number of channels in node embeddings
        attn_hidden_channels (int): Number of hidden channels in equivariant graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_channels (int):  Number of channels for alpha vector in each attention head
        attn_value_channels (int):  Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels in feedforward network
        norm_type (str):            Type of normalization layer 
                                    (['sep_layer_norm', 'merge_layer_norm', 
                                    'merge_layer_norm_attn_rms_norm', 'merge_rms_norm'])

        lmax (int):                 Maximum degrees (l)
        mmax (int):                 Maximum order (m)
        attn_grid_resolution_list (list:int):      
                                    Grid resolution list in class `SO3Grid` in attention
        ffn_grid_resolution_list (list:int):      
                                    Grid resolution list in class `SO3Grid` in feedforward network

        edge_channels (int):                Number of channels for edge-wise invariant features
        use_atom_edge_embedding (bool):     Whether to use atomic embedding along with relative distance for edge scalar features
        use_envelope (bool):        Whether to apply an envelope function to attention
        
        attn_activation (str):      Type of activation function in equivariant graph attention
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
        ffn_activation (str):       Type of activation function for feedforward network
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFNs.

        alpha_drop (float):         Dropout rate for the hidden features in non-linear MLP attention
        attn_mask_rate (float):     Mask rate for neighbors considered in attention
        attn_weights_drop (float):  Dropout rate for attention weights
        value_drop (float):         Dropout rate for the hidden features in non-linear value vectors
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN in Transformer blocks
        ffn_drop (float):           Dropout rate for the hidden features in FFN
        
        avg_num_nodes (float):   Normalization factor for sum aggregation over nodes
        avg_degree (float):      Normalization factor for sum aggregation over edges

        use_attn (bool):            Default: True
                                    Whether to have the attention sub-block
        num_ffns (int):             Number of feedforward networks within a single Transformer block
        use_gaunt_self_tensor_product (bool):
                                    Default: False
                                    Whether to use Gaunt tensor products with path-level learnable weights
                                    in feedforward networks
        num_output_channels (int):  Number of output channels
    """
    def __init__(
        self,

        max_neighbors=20,
        max_radius=12.0,
        num_radial_basis=600,
        max_num_elements=128,

        num_layers=12,
        num_channels=128,
        attn_hidden_channels=64,
        num_heads=8,
        attn_alpha_channels=32,
        attn_value_channels=16,
        ffn_hidden_channels=128,
        norm_type='merge_rms_norm',

        lmax=6,
        mmax=2,
        attn_grid_resolution_list=[14, 5],
        ffn_grid_resolution_list=[14, 15],

        edge_channels=128,
        use_atom_edge_embedding=True,
        use_envelope=False,

        attn_activation='sep-merge_s2_swiglu',
        use_attn_renorm=True,
        use_add_merge=False,
        use_rad_l_parametrization=True,
        softcap=None,
        ffn_activation='sep-merge_s2_swiglu',
        use_grid_mlp=True,

        alpha_drop=0.0,
        attn_mask_rate=0.0,
        attn_weights_drop=0.1,
        value_drop=0.0,
        drop_path_rate=0.05,
        proj_drop=0.0,
        ffn_drop=0.0,

        avg_num_nodes=_AVG_NUM_NODES,
        avg_degree=_AVG_DEGREE,

        use_attn=True,
        num_ffns=1,
        use_gaunt_self_tensor_product=False,
        num_output_channels=1,
    ):
        super().__init__()
        
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.cutoff = max_radius
        self.num_radial_basis = num_radial_basis
        self.max_num_elements = max_num_elements

        self.num_layers = num_layers
        self.num_channels = num_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type

        self.lmax = lmax
        self.mmax = mmax
        self.attn_grid_resolution_list = attn_grid_resolution_list
        self.ffn_grid_resolution_list = ffn_grid_resolution_list

        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.use_envelope = use_envelope

        self.attn_activation = attn_activation
        self.use_attn_renorm = use_attn_renorm
        self.use_add_merge = use_add_merge
        self.use_rad_l_parametrization = use_rad_l_parametrization
        self.softcap = softcap
        self.ffn_activation = ffn_activation
        self.use_grid_mlp = use_grid_mlp

        self.alpha_drop = alpha_drop
        self.attn_mask_rate = attn_mask_rate
        self.attn_weights_drop = attn_weights_drop
        self.value_drop = value_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop
        self.ffn_drop = ffn_drop

        self.avg_num_nodes = avg_num_nodes
        self.avg_degree = avg_degree

        self.use_attn = use_attn
        self.num_ffns = num_ffns
        self.use_gaunt_self_tensor_product = use_gaunt_self_tensor_product
        self.num_output_channels = num_output_channels

        # Atom-type embedding
        self.sphere_embedding = torch.nn.Embedding(self.max_num_elements, self.num_channels)

        # Radial basis function
        self.distance_expansion = GaussianSmearing(
            0.0,
            self.cutoff,
            self.num_radial_basis,
            2.0,
        )
        edge_input_channels = int(self.distance_expansion.num_output)
        
        # The sizes of radial functions (input channels and 2 hidden channels)
        self.edge_channels_list = [edge_input_channels] + [self.edge_channels] * 2

        # Envelope function
        self.envelope_func = PolynomialEnvelope(
            cutoff=self.cutoff,
            exponent=5
        ) if self.use_envelope else None

        # Computing Wigner-D matrices
        self.so3_rotation = SO3Rotation(self.lmax, self.mmax, use_rotation_mask=True)

        # Edge-degree embedding
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            num_channels=self.num_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            so3_rotation=self.so3_rotation,
            max_num_elements=self.max_num_elements,
            edge_channels_list=self.edge_channels_list,
            use_atom_edge_embedding=self.use_atom_edge_embedding,
            rescale_factor=self.avg_degree
        )

        # Transformer block
        self.blocks = torch.nn.ModuleList()
        for i in range(self.num_layers):
            attn_activation = self.attn_activation
            ffn_activation  = self.ffn_activation
            block_config_dict = dict(
                num_in_channels=self.num_channels,
                attn_hidden_channels=self.attn_hidden_channels,
                num_heads=self.num_heads,
                attn_alpha_channels=self.attn_alpha_channels,
                attn_value_channels=self.attn_value_channels,
                ffn_hidden_channels=self.ffn_hidden_channels,
                num_out_channels=self.num_channels,
                lmax=self.lmax,
                mmax=self.mmax,
                so3_rotation=self.so3_rotation,
                attn_grid_resolution_list=self.attn_grid_resolution_list,
                ffn_grid_resolution_list=self.ffn_grid_resolution_list,
                max_num_elements=self.max_num_elements,
                edge_channels_list=self.edge_channels_list,
                use_atom_edge_embedding=self.use_atom_edge_embedding,
                attn_activation=attn_activation,
                use_attn_renorm=self.use_attn_renorm,
                use_add_merge=self.use_add_merge,
                use_rad_l_parametrization=self.use_rad_l_parametrization,
                softcap=self.softcap,
                ffn_activation=ffn_activation,
                use_grid_mlp=self.use_grid_mlp,
                norm_type=self.norm_type,
                alpha_drop=self.alpha_drop,
                attn_mask_rate=self.attn_mask_rate,
                attn_weights_drop=attn_weights_drop,
                value_drop=self.value_drop,
                drop_path_rate=self.drop_path_rate,
                proj_drop=self.proj_drop,
                ffn_drop=self.ffn_drop,
                num_ffns=self.num_ffns,
                use_gaunt_self_tensor_product=self.use_gaunt_self_tensor_product,
                use_attn=self.use_attn,
            )
            block_class = TransBlockV3BodyOrderTest
            self.blocks.append(block_class(**block_config_dict))

        # Output blocks for energy and forces (and optionally stress)
        self.norm = get_normalization_layer(
            self.norm_type, 
            lmax=self.lmax, 
            num_channels=self.num_channels
        )
        self.energy_block = ScalarFeedForwardNetwork(
            num_in_channels=self.num_channels,
            num_hidden_channels=self.ffn_hidden_channels,
            num_out_channels=self.num_output_channels,
        )
        self.apply(self._init_weights)


    def _forward_edge(
        self, 
        edge_distance, 
        edge_distance_vec
    ):
        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(edge_distance_vec)

        # Compute Wigner-D matrices
        self.so3_rotation.set_wigner(edge_rot_mat)

        # Envelope function
        edge_envelope_weight = self.envelope_func(edge_distance) if self.envelope_func is not None else None

        # Radial basis function
        edge_distance = self.distance_expansion(edge_distance)

        return edge_distance, edge_envelope_weight
    

    def _forward_embedding(
        self, 
        atomic_numbers, 
        edge_distance, 
        edge_index, 
        edge_envelope_weight
    ):
        num_atoms = len(atomic_numbers)
                
        # Initialize node embedding
        x = torch.zeros(
            (
                num_atoms,
                ((self.lmax + 1) ** 2),
                self.num_channels
            ),
            device=self.device,
            dtype=self.dtype
        )

        # Atom-type embedding
        atom_embedding = self.sphere_embedding(atomic_numbers)
        x[:, 0, :] = atom_embedding

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            atomic_numbers,
            edge_distance,
            edge_index,
            edge_envelope_weight
        )
        x = x + edge_degree

        return x
    

    def _forward_blocks(
        self,
        x,
        source_atomic_numbers, 
        target_atomic_numbers, 
        edge_distance, 
        edge_index,
        edge_envelope_weight,
        batch
    ):
        for i in range(self.num_layers):
            x = self.blocks[i](
                x, 
                source_atomic_numbers, 
                target_atomic_numbers, 
                edge_distance, 
                edge_index,
                edge_envelope_weight,
                batch,     # for GraphDropPath
            )
        x = self.norm(x)
        x_scalar = x.narrow(1, 0, 1)
        x_scalar = x_scalar.view(x_scalar.shape[0], self.num_channels)
        return x_scalar, x
    

    def _forward_direct(self, data):
        self.batch_size = max(data.batch) + 1
        self.dtype = data.pos.dtype
        self.device = data.pos.device

        (
            data,
            edge_index,
            edge_distance_vec,
            edge_distance,
        ) = self.generate_graph(data)

        atomic_numbers = data.atomic_numbers.long()
        source_atomic_numbers = atomic_numbers[edge_index[0]]
        target_atomic_numbers = atomic_numbers[edge_index[1]]

        edge_distance, edge_envelope_weight = self._forward_edge(edge_distance, edge_distance_vec)
        x = self._forward_embedding(atomic_numbers, edge_distance, edge_index, edge_envelope_weight)
        x_scalar, x = self._forward_blocks(
            x,
            source_atomic_numbers, 
            target_atomic_numbers, 
            edge_distance, 
            edge_index,
            edge_envelope_weight,
            data.batch
        )
        node_energy = self.energy_block(x_scalar)
        energy = torch.zeros((self.batch_size, self.num_output_channels), device=node_energy.device, dtype=node_energy.dtype)
        energy.index_add_(0, data.batch, node_energy.view(-1, self.num_output_channels))
        energy = energy / self.avg_num_nodes
        return energy


    def forward(self, data):
        outputs = self._forward_direct(data)
        return outputs


    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec, use_rotation_mask=True)


    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


    def _init_weights(self, m):
        if (isinstance(m, torch.nn.Linear)
            or isinstance(m, SO3Linear)
        ):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif (isinstance(m, RadialFunction)):
            m.apply(self._uniform_init_linear_weights)


    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)


    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.Embedding)
                or isinstance(module, torch.nn.Linear)
                or isinstance(module, SO3Linear)
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, RMSNorm)
                or isinstance(module, EquivariantLayerNorm)
                or isinstance(module, EquivariantSeparableLayerNorm)
                or isinstance(module, EquivariantMergeLayerNorm)
            ):
                for parameter_name, _ in module.named_parameters():
                    if (isinstance(module, torch.nn.Linear)
                        or isinstance(module, SO3Linear)
                    ):
                        if 'weight' in parameter_name:
                            continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)


    @torch._dynamo.disable
    def generate_graph(
        self, 
        data,
    ):
        data.atomic_numbers = data.atoms
        edge_index = data.edge_index
        edge_distance_vec = data.pos[data.edge_index[0]] - data.pos[data.edge_index[1]]
        edge_distance = edge_distance_vec.norm(dim=-1)
        return (
            data,
            edge_index,
            edge_distance_vec,
            edge_distance
        )