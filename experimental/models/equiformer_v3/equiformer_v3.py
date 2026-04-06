import math
import torch

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.base import GraphModelMixin

from .edge_rot_mat import init_edge_rot_mat
from .envelope import PolynomialEnvelope
from .so3 import (
    SO3Rotation,
    SO3Linear
)
from .radial_function import (
    GaussianSmearing,
    RadialFunction
)
from .layer_norm import (
    EquivariantLayerNorm,
    EquivariantSeparableLayerNorm,
    EquivariantMergeLayerNorm,
    RMSNorm,
    get_normalization_layer
)
from .transformer_block import (
    EquivariantGraphAttention,
    FeedForwardNetwork,
    TransBlockV3,
)
from .input_block import EdgeDegreeEmbedding
from .output_block import (
    ScalarFeedForwardNetwork,
    FeedForwardNetworkStressHead,
    EquivariantGraphAttentionStressHead
)


# Statistics of IS2RE 100K
_AVG_NUM_NODES = 77.81317
_AVG_DEGREE = 23.395238876342773    # IS2RE: 100k, max_radius = 5, max_neighbors = 100

_NORM_SCALE_NODES = math.sqrt(_AVG_NUM_NODES)   # 8.82117735906041
_NORM_SCALE_DEGREE = math.sqrt(_AVG_DEGREE)     # 4.836862503353054


@registry.register_model("equiformer_v3")
class EquiformerV3_OC(torch.nn.Module, GraphModelMixin):
    """
    Args:
        use_pbc (bool):         Use periodic boundary conditions
        use_pbc_single (bool, optional):    Process batch PBC graphs one at a time.
                                            This argument is set to True when training on MPTraj.
        otf_graph (bool):       Compute graph On The Fly (OTF)

        regress_forces (bool):  Compute forces
        regress_stress (bool):  Compute stress
        direct_prediction (bool):   Whether to use direct methods to predict forces and stress

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
        attn_eps (float):           Default: 1e-16
                                    Epsilon value used in the softmax operation of attention
        ffn_activation (str):       Type of activation function for feedforward network
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFNs.

        use_gate_force_head (bool): If `True`, use `GateActivation` in the equivariant attention of the force prediction head.

        alpha_drop (float):         Dropout rate for the hidden features in non-linear MLP attention
        attn_mask_rate (float):     Mask rate for neighbors considered in attention
        attn_weights_drop (float):  Dropout rate for attention weights
        value_drop (float):         Dropout rate for the hidden features in non-linear value vectors
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN in Transformer blocks
        ffn_drop (float):           Dropout rate for the hidden features in FFN
        use_head_reg (bool):        Whether to apply regularization to output head (dummy argument for backend compatibility)
        
        gradient_checkpointing_block_list (list):
                                    A list indicating which block we apply gradient/activation checkpointing to save memory.

        avg_num_nodes (float):   Normalization factor for sum aggregation over nodes
        avg_degree (float):      Normalization factor for sum aggregation over edges

        enforce_max_neighbors_strictly (bool):      When edges are subselected based on the `max_neighbors` arg, arbitrarily select amongst equidistant / degenerate edges to have exactly the correct number.
    """
    def __init__(
        self,

        use_pbc=True,
        use_pbc_single=False, 
        otf_graph=True,

        regress_forces=True,
        regress_stress=False,
        direct_prediction=True,
        
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
        norm_type='merge_layer_norm',

        lmax=6,
        mmax=2,
        attn_grid_resolution_list=[20, 8],
        ffn_grid_resolution_list=[20, 20],

        edge_channels=128,
        use_atom_edge_embedding=True,
        use_envelope=True,

        attn_activation='sep-merge_gates2_swiglu',
        use_attn_renorm=True,
        use_add_merge=False,
        use_rad_l_parametrization=True,
        softcap=None,
        attn_eps=1e-16,
        ffn_activation='sep-merge_gates2_swiglu',
        use_grid_mlp=True,

        use_gate_force_head=True,

        alpha_drop=0.0,
        attn_mask_rate=0.0,
        attn_weights_drop=0.1,
        value_drop=0.0,
        drop_path_rate=0.05,
        proj_drop=0.0,
        ffn_drop=0.0,
        use_head_reg=False,
        
        gradient_checkpointing_block_list=None,

        avg_num_nodes=_AVG_NUM_NODES,
        avg_degree=_AVG_DEGREE,

        enforce_max_neighbors_strictly=True,
    ):
        super().__init__()

        self.use_pbc = use_pbc
        self.use_pbc_single = use_pbc_single
        self.otf_graph = otf_graph
        
        self.regress_forces = regress_forces
        self.regress_stress = regress_stress
        self.direct_prediction = direct_prediction
        
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
        self.attn_eps = attn_eps
        self.ffn_activation = ffn_activation
        self.use_grid_mlp = use_grid_mlp

        self.use_gate_force_head = use_gate_force_head

        self.alpha_drop = alpha_drop
        self.attn_mask_rate = attn_mask_rate
        self.attn_weights_drop = attn_weights_drop
        self.value_drop = value_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop
        self.ffn_drop = ffn_drop
        self.use_head_reg = use_head_reg

        self.gradient_checkpointing_block_list = gradient_checkpointing_block_list
        if self.gradient_checkpointing_block_list is not None:
            assert len(self.gradient_checkpointing_block_list) == self.num_layers
        else:
            self.gradient_checkpointing_block_list = [0] * self.num_layers

        self.avg_num_nodes = avg_num_nodes
        self.avg_degree = avg_degree

        self.enforce_max_neighbors_strictly = enforce_max_neighbors_strictly

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
        self.so3_rotation = SO3Rotation(self.lmax, self.mmax, use_rotation_mask=(not self.direct_prediction))

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
            if self.gradient_checkpointing_block_list[i] == 1:
                attn_activation = self.attn_activation.replace('_mem', '')
                ffn_activation  = self.ffn_activation.replace('_mem', '')
            else:
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
                attn_eps=self.attn_eps,
                ffn_activation=ffn_activation,
                use_grid_mlp=self.use_grid_mlp,
                norm_type=self.norm_type,
                alpha_drop=self.alpha_drop,
                attn_mask_rate=self.attn_mask_rate,
                attn_weights_drop=attn_weights_drop,
                value_drop=self.value_drop,
                drop_path_rate=self.drop_path_rate,
                proj_drop=self.proj_drop,
                ffn_drop=self.ffn_drop
            )
            block_class = TransBlockV3
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
            num_out_channels=1,
            dropout=0.0
        )
        if self.direct_prediction:
            if self.regress_forces:
                self.force_block = EquivariantGraphAttention(
                    num_in_channels=self.num_channels,
                    num_hidden_channels=self.attn_hidden_channels,
                    num_heads=self.num_heads,
                    attn_alpha_channels=self.attn_alpha_channels,
                    attn_value_channels=self.attn_value_channels,
                    num_out_channels=1,
                    lmax=self.lmax,
                    mmax=self.mmax,
                    so3_rotation=self.so3_rotation,
                    grid_resolution_list=self.attn_grid_resolution_list,
                    max_num_elements=self.max_num_elements,
                    edge_channels_list=self.edge_channels_list,
                    use_atom_edge_embedding=self.use_atom_edge_embedding,
                    activation=('sep_s2' if not self.use_gate_force_head else 'gate'),
                    use_attn_renorm=self.use_attn_renorm,
                    use_add_merge=self.use_add_merge,
                    use_rad_l_parametrization=self.use_rad_l_parametrization,
                    softcap=self.softcap,
                    eps=self.attn_eps,
                    alpha_drop=0.0, 
                    attn_mask_rate=0.0,
                    attn_weights_drop=0.0,
                    value_drop=0.0
                )
                if 'rms_norm' in norm_type:
                    if self.force_block.alpha_norm is not None:
                        del self.force_block.alpha_norm
                        self.force_block.alpha_norm = RMSNorm(self.attn_alpha_channels)
            if self.regress_stress:
                self.stress_block = FeedForwardNetworkStressHead(
                    num_in_channels=self.num_channels,
                    num_hidden_channels=self.ffn_hidden_channels,
                    num_out_channels=1,
                    lmax=self.lmax,
                    mmax=self.mmax,
                    grid_resolution_list=self.ffn_grid_resolution_list,
                    activation='gate',
                    use_grid_mlp=False,
                    dropout=0.0,
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
        # Transformer blocks
        for i in range(self.num_layers):
            if self.gradient_checkpointing_block_list[i] == 0:
                x = self.blocks[i](
                    x, 
                    source_atomic_numbers, 
                    target_atomic_numbers, 
                    edge_distance, 
                    edge_index,
                    edge_envelope_weight,
                    batch,     # for GraphDropPath
                )
            elif self.gradient_checkpointing_block_list[i] == 1:
                x = torch.utils.checkpoint.checkpoint(
                    self.blocks[i],
                    x,                  
                    source_atomic_numbers,
                    target_atomic_numbers,
                    edge_distance,
                    edge_index,
                    edge_envelope_weight,
                    batch,     # for GraphDropPath
                    use_reentrant=False
                )
            else:
                raise ValueError
            
        # Final layer norm
        x = self.norm(x)
        x_scalar = x.narrow(1, 0, 1)
        x_scalar = x_scalar.view(x_scalar.shape[0], self.num_channels)
        return x_scalar, x
        

    def _forward_direct(self, data):
        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype
        self.device = data.pos.device

        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(
            data,
            enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
            use_pbc_single=self.use_pbc_single
        )

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

        outputs = {}

        # Energy prediction
        node_energy = self.energy_block(x_scalar)
        energy = torch.zeros(self.batch_size, device=node_energy.device, dtype=node_energy.dtype)
        energy.index_add_(0, data.batch, node_energy.view(-1))
        energy = energy / self.avg_num_nodes
        outputs['energy'] = energy

        # Force prediction
        if self.regress_forces:
            forces = self.force_block(
                x,
                source_atomic_numbers,
                target_atomic_numbers,
                edge_distance,
                edge_index,
                edge_envelope_weight
            )
            forces = forces.narrow(1, 1, 3)
            forces = forces.view(-1, 3)
            outputs['forces'] = forces
        
        # Stress Prediction
        if self.regress_stress:
            stress = self.stress_block(
                x,
                batch_size=self.batch_size,
                batch=data.batch
            )
            outputs['stress'] = stress
                
        return outputs
    

    @conditional_grad(torch.enable_grad())
    def _forward_gradient(self, data):
        """
            1.  We have additional `@conditional_grad` as the decorator since the decorator might not be compatible with 
                `torch.compile()` in direct methods.
        """
        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype
        self.device = data.pos.device

        """
            For gradient methods
        """
        displacement = None
        orig_cell = None
        if self.regress_stress and self.regress_forces:
            displacement = torch.zeros(
                (3, 3),
                dtype=self.dtype,
                device=self.device,
            )
            displacement = displacement.view(-1, 3, 3).expand(self.batch_size, 3, 3)
            displacement.requires_grad = True
            symmetric_displacement = 0.5 * (
                displacement + displacement.transpose(-1, -2)
            )

            data.pos.requires_grad = True
            data.pos = data.pos + torch.bmm(
                data.pos.unsqueeze(-2),
                torch.index_select(symmetric_displacement, 0, data.batch)
            ).squeeze(-2)

            orig_cell = data.cell
            data.cell = data.cell + torch.bmm(
                data.cell, symmetric_displacement
            )
        elif not self.regress_stress and self.regress_forces:
            data.pos.requires_grad = True

        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(
            data,
            enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
            use_pbc_single=self.use_pbc_single
        )

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

        outputs = {}

        # Energy prediction
        node_energy = self.energy_block(x_scalar)
        energy = torch.zeros(self.batch_size, device=node_energy.device, dtype=node_energy.dtype)
        energy.index_add_(0, data.batch, node_energy.view(-1))
        energy = energy / self.avg_num_nodes
        outputs['energy'] = energy

        if self.regress_stress and self.regress_forces:
            # Stress and forces prediction
            grads = torch.autograd.grad(
                [energy.sum()],
                [data.pos, displacement],
                create_graph=self.training,
            )
            forces = torch.neg(grads[0])
            virial = grads[1].view(-1, 3, 3)
            volume = torch.det(data.cell).abs().unsqueeze(-1)
            stress = virial / volume.view(-1, 1, 1)
            virial = torch.neg(virial)
            outputs['forces'] = forces
            outputs['stress'] = stress.view(-1, 9)
            data.cell = orig_cell
    
        elif not self.regress_stress and self.regress_forces:
            forces = (
                -1
                * torch.autograd.grad(
                    energy.sum(), data.pos, create_graph=self.training
                )[0]
            )
            outputs['forces'] = forces

        return outputs
    

    def forward(self, data):
        if self.direct_prediction:
            outputs = self._forward_direct(data)
        else:
            outputs = self._forward_gradient(data)
        return outputs


    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec, use_rotation_mask=(not self.direct_prediction))


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
        cutoff=None,
        max_neighbors=None,
        use_pbc=None,
        otf_graph=None,
        enforce_max_neighbors_strictly=None,
        use_pbc_single=False,
    ):
        graph_data = super().generate_graph(
            data,
            cutoff,
            max_neighbors,
            use_pbc,
            otf_graph,
            enforce_max_neighbors_strictly,
            use_pbc_single
        )

        edge_index   = graph_data.edge_index
        edge_dist    = graph_data.edge_distance
        distance_vec = graph_data.edge_distance_vec
        cell_offsets = graph_data.cell_offsets
        cell_offset_distances = graph_data.offset_distances
        neighbors    = graph_data.neighbors

        return (
            edge_index,
            edge_dist,
            distance_vec,
            cell_offsets,
            cell_offset_distances,
            neighbors,
        )