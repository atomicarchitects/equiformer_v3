import torch

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad

import e3nn
import math

from .so3 import SO3Linear
from .layer_norm import (
    RMSNorm,
    get_normalization_layer
)
from .transformer_block import EquivariantGraphAttention, FeedForwardNetwork
from .equiformer_v3 import EquiformerV3_OC


# Statistics of IS2RE 100K
_AVG_NUM_NODES = 77.81317
_AVG_DEGREE = 23.395238876342773    # IS2RE: 100k, max_radius = 5, max_neighbors = 100


@registry.register_model("equiformer_v3_dens")
class EquiformerV3DeNS_OC(EquiformerV3_OC):
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
                                    (['sep_layer_norm', 'sep_layer_norm_liger', 'merge_layer_norm', 
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
        super().__init__(
            use_pbc,
            use_pbc_single, 
            otf_graph,

            regress_forces,
            regress_stress,
            direct_prediction,

            max_neighbors,
            max_radius,
            num_radial_basis,
            max_num_elements,

            num_layers,
            num_channels,
            attn_hidden_channels,
            num_heads,
            attn_alpha_channels,
            attn_value_channels,
            ffn_hidden_channels,
            norm_type,
            
            lmax,
            mmax,
            attn_grid_resolution_list,
            ffn_grid_resolution_list,
            
            edge_channels,
            use_atom_edge_embedding,
            use_envelope,

            attn_activation,
            use_attn_renorm,
            use_add_merge,
            use_rad_l_parametrization,
            softcap,
            attn_eps,
            ffn_activation,
            use_grid_mlp,

            use_gate_force_head,
            
            alpha_drop,
            attn_mask_rate,
            attn_weights_drop,
            value_drop,
            drop_path_rate,
            proj_drop,
            ffn_drop,
            use_head_reg,
            
            gradient_checkpointing_block_list,

            avg_num_nodes,
            avg_degree,

            enforce_max_neighbors_strictly,
        )

        # Force encoding
        self.irreps_sh = e3nn.o3.Irreps.spherical_harmonics(lmax=self.lmax, p=1)
        self.force_embedding = SO3Linear(
            in_features=1,
            out_features=self.num_channels,
            lmax=self.lmax,
            bias=True
        )
        
        if self.regress_forces:
            self.dens_block = EquivariantGraphAttention(
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
                if self.dens_block.alpha_norm is not None:
                    del self.dens_block.alpha_norm
                    self.dens_block.alpha_norm = RMSNorm(self.attn_alpha_channels)

        self.apply(self._init_weights)


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
        force_embedding, noise_mask_tensor, dens_batch_mask_tensor, dens_mask_tensor = self._forward_dens_force_encoding(data)
        x = x + force_embedding
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
            
            # for DeNS
            denoising_pos_vec = self.dens_block(
                x,
                source_atomic_numbers,
                target_atomic_numbers,
                edge_distance,
                edge_index,
                edge_envelope_weight
            )
            denoising_pos_vec = denoising_pos_vec.narrow(1, 1, 3)
            denoising_pos_vec = denoising_pos_vec.view(-1, 3)
        
            outputs['forces'] = forces * (~noise_mask_tensor) + denoising_pos_vec * noise_mask_tensor
        
        # Stress Prediction
        if self.regress_stress:
            stress = self.stress_block(
                x,
                batch_size=self.batch_size,
                batch=data.batch
            )
            outputs['stress'] = stress * (~dens_batch_mask_tensor) # not predict stress during DeNS

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
        force_embedding, noise_mask_tensor, dens_batch_mask_tensor, dens_mask_tensor = self._forward_dens_force_encoding(data)
        x = x + force_embedding
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
            outputs['stress'] = stress.view(-1, 9) * (~dens_batch_mask_tensor) # not predict stress during DeNS
            data.cell = orig_cell
        elif not self.regress_stress and self.regress_forces:
            forces = (
                -1
                * torch.autograd.grad(
                    energy.sum(), data.pos, create_graph=self.training
                )[0]
            )
            outputs['forces'] = forces
        
        # for DeNS
        if self.regress_forces:
            denoising_pos_vec = self.dens_block(
                x,
                source_atomic_numbers,
                target_atomic_numbers,
                edge_distance,
                edge_index,
                edge_envelope_weight
            )
            denoising_pos_vec = denoising_pos_vec.narrow(1, 1, 3)
            denoising_pos_vec = denoising_pos_vec.view(-1, 3)
            outputs['forces'] = forces * (~noise_mask_tensor) + denoising_pos_vec * noise_mask_tensor # not predict forces during DeNS

        return outputs


    @torch._dynamo.disable
    def _generate_dens_data(self, data):
        num_atoms = len(data.atomic_numbers)
        if hasattr(data, "denoising_pos_forward") and data.denoising_pos_forward:
            force_data = data.forces
            if hasattr(data, "noise_mask"):
                noise_mask_tensor = data.noise_mask.view(-1, 1)
            else:
                noise_mask_tensor = torch.ones((num_atoms, 1), dtype=torch.bool, device=self.device)
            if hasattr(data, "dens_batch_mask"):
                dens_batch_mask_tensor = data.dens_batch_mask.view(-1, 1)
            else:
                dens_batch_mask_tensor = torch.ones((len(data.natoms), 1), dtype=torch.bool, device=self.device)
            dens_mask_tensor = torch.ones((1, 1), dtype=torch.bool, device=self.device)
        else:
            force_data = torch.zeros((num_atoms, 3), dtype=self.dtype, device=self.device)
            noise_mask_tensor = torch.zeros((num_atoms, 1), dtype=torch.bool, device=self.device)
            dens_batch_mask_tensor = torch.zeros((len(data.natoms), 1), dtype=torch.bool, device=self.device)
            dens_mask_tensor = torch.zeros((1, 1), dtype=torch.bool, device=self.device)
        force_sh = e3nn.o3.spherical_harmonics(
            l=self.irreps_sh,
            x=force_data,
            normalize=True,
            normalization='component'
        )
        
        return force_data, force_sh, noise_mask_tensor, dens_batch_mask_tensor, dens_mask_tensor


    def _forward_dens_force_encoding(self, data):
        force_data, force_sh, noise_mask_tensor, dens_batch_mask_tensor, dens_mask_tensor = self._generate_dens_data(data)
        force_norm = force_data.norm(dim=-1, keepdim=True)
        force_norm = force_norm / math.sqrt(3.0)
        force_embedding = force_sh * force_norm
        force_embedding = force_embedding.view(force_embedding.shape[0], -1, 1)
        force_embedding = self.force_embedding(force_embedding)
        noise_mask_tensor = noise_mask_tensor.view(-1, 1, 1)
        force_embedding = force_embedding * noise_mask_tensor
        noise_mask_tensor = noise_mask_tensor.view(-1, 1)

        return force_embedding, noise_mask_tensor, dens_batch_mask_tensor, dens_mask_tensor