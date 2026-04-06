"""
    1.  Reference: https://github.com/chaitjo/geometric-gnn-dojo/blob/f86a212a40d2ff418fd13320e9890c4f663dad8e/experiments/incompleteness.ipynb
    2.  We use `equivariant_layer_norm` proposed in Equiformer(V1) instead of `merge_layer_norm` proposed in EquiformerV3
        as `merge_layer_norm` can make some body order test simpler.
"""

import torch
import e3nn
import time

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected

from equiformer_v3_body_order_test import EquiformerV3BodyOrderTest


def create_many_body_dataset(body_order):
    if body_order == '2':
        return create_two_body_envs()
    elif body_order == '3':
        return create_three_body_envs()
    elif body_order == '4_chiral':
        return create_four_body_chiral_envs()
    elif body_order == '4_nonchiral':
        return create_four_body_nonchiral_envs()
    else:
        raise ValueError
    

def create_two_body_envs():
    dataset = []

    # Environment 0
    atoms = torch.LongTensor([ 0, 0, 0 ])
    edge_index = torch.LongTensor([ [0, 0], [1, 2] ])
    pos = torch.FloatTensor([ 
        [0, 0, 0],
        [5, 0, 0],
        [3, 0, 4]
    ])
    y = torch.LongTensor([0])  # Label 0
    data1 = Data(atoms=atoms, edge_index=edge_index, pos=pos, y=y)
    data1.edge_index = to_undirected(data1.edge_index)
    dataset.append(data1)
    
    # Environment 1
    atoms = torch.LongTensor([ 0, 0, 0 ])
    edge_index = torch.LongTensor([ [0, 0], [1, 2] ])
    pos = torch.FloatTensor([ 
        [0, 0, 0],
        [5, 0, 0],
        [-5, 0, 0]
    ])
    y = torch.LongTensor([1])  # Label 1
    data2 = Data(atoms=atoms, edge_index=edge_index, pos=pos, y=y)
    data2.edge_index = to_undirected(data2.edge_index)
    dataset.append(data2)
    
    return dataset


def create_three_body_envs():
    dataset = []

    a_x, a_y, a_z = 5, 0, 5
    b_x, b_y, b_z = 5, 5, 5
    c_x, c_y, c_z = 0, 5, 5
    
    # Environment 0
    atoms = torch.LongTensor([ 0, 0, 0, 0, 0 ])
    edge_index = torch.LongTensor([ [0, 0, 0, 0], [1, 2, 3, 4] ])
    pos = torch.FloatTensor([ 
        [0, 0, 0],
        [a_x, a_y, a_z],
        [+b_x, +b_y, b_z],
        [-b_x, -b_y, b_z],
        [c_x, +c_y, c_z],
    ])
    y = torch.LongTensor([0])  # Label 0
    data1 = Data(atoms=atoms, edge_index=edge_index, pos=pos, y=y)
    data1.edge_index = to_undirected(data1.edge_index)
    dataset.append(data1)
    
    # Environment 1
    atoms = torch.LongTensor([ 0, 0, 0, 0, 0 ])
    edge_index = torch.LongTensor([ [0, 0, 0, 0], [1, 2, 3, 4] ])
    pos = torch.FloatTensor([ 
        [0, 0, 0],
        [a_x, a_y, a_z],
        [+b_x, +b_y, b_z],
        [-b_x, -b_y, b_z],
        [c_x, -c_y, c_z],
    ])
    y = torch.LongTensor([1])  # Label 1
    data2 = Data(atoms=atoms, edge_index=edge_index, pos=pos, y=y)
    data2.edge_index = to_undirected(data2.edge_index)
    dataset.append(data2)
    
    return dataset


def create_four_body_nonchiral_envs():
    dataset = []

    a1_x, a1_y, a1_z = 3, 2, -4
    a2_x, a2_y, a2_z = 0, 2, 5
    a3_x, a3_y, a3_z = 0, 2, -5
    b1_x, b1_y, b1_z = 3, -2, -4
    b2_x, b2_y, b2_z = 0, -2, 5
    b3_x, b3_y, b3_z = 0, -2, -5
    c_x, c_y, c_z = 0, 5, 0

    angle = 1 * torch.pi / 10 # random angle
    Q = e3nn.o3.matrix_y(torch.tensor(angle)).numpy()

    # Environment 0
    atoms = torch.LongTensor([ 0, 0, 0, 0, 0, 0, 0, 0 ])
    edge_index = torch.LongTensor([ [0, 0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6, 7] ])
    pos = torch.FloatTensor([ 
        [0, 0, 0],
        [a1_x, a1_y, a1_z],
        [a2_x, a2_y, a2_z],
        [a3_x, a3_y, a3_z],
        [b1_x, b1_y, b1_z] @ Q,
        [b2_x, b2_y, b2_z] @ Q,
        [b3_x, b3_y, b3_z] @ Q,
        [c_x, +c_y, c_z],
    ])  #.to(dtype=torch.float64)        
    y = torch.LongTensor([0])  # Label 0
    data1 = Data(atoms=atoms, edge_index=edge_index, pos=pos, y=y)
    data1.edge_index = to_undirected(data1.edge_index)
    dataset.append(data1)
    
    # Environment 1
    atoms = torch.LongTensor([ 0, 0, 0, 0, 0, 0, 0, 0 ])
    edge_index = torch.LongTensor([ [0, 0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6, 7] ])
    pos = torch.FloatTensor([ 
        [0, 0, 0],
        [a1_x, a1_y, a1_z],
        [a2_x, a2_y, a2_z],
        [a3_x, a3_y, a3_z],
        [b1_x, b1_y, b1_z] @ Q,
        [b2_x, b2_y, b2_z] @ Q,
        [b3_x, b3_y, b3_z] @ Q,
        [c_x, -c_y, c_z],
    ])  #.to(dtype=torch.float64)
    y = torch.LongTensor([1])  # Label 1
    data2 = Data(atoms=atoms, edge_index=edge_index, pos=pos, y=y)
    data2.edge_index = to_undirected(data2.edge_index)
    dataset.append(data2)
    
    return dataset


def create_four_body_chiral_envs():
    dataset = []

    a1_x, a1_y, a1_z = 3, 0, -4
    a2_x, a2_y, a2_z = 0, 0, 5
    a3_x, a3_y, a3_z = 0, 0, -5
    c_x, c_y, c_z = 0, 5, 0

    # Environment 0
    atoms = torch.LongTensor([ 0, 0, 0, 0, 0 ])
    edge_index = torch.LongTensor([ [0, 0, 0, 0], [1, 2, 3, 4] ])
    pos = torch.FloatTensor([ 
        [0, 0, 0],
        [a1_x, a1_y, a1_z],
        [a2_x, a2_y, a2_z],
        [a3_x, a3_y, a3_z],
        [c_x, +c_y, c_z],
    ])
    y = torch.LongTensor([0])  # Label 0
    data1 = Data(atoms=atoms, edge_index=edge_index, pos=pos, y=y)
    data1.edge_index = to_undirected(data1.edge_index)
    dataset.append(data1)
    
    # Environment 1
    atoms = torch.LongTensor([ 0, 0, 0, 0, 0 ])
    edge_index = torch.LongTensor([ [0, 0, 0, 0], [1, 2, 3, 4] ])
    pos = torch.FloatTensor([ 
        [0, 0, 0],
        [a1_x, a1_y, a1_z],
        [a2_x, a2_y, a2_z],
        [a3_x, a3_y, a3_z],
        [c_x, -c_y, c_z],
    ])
    y = torch.LongTensor([1])  # Label 1
    data2 = Data(atoms=atoms, edge_index=edge_index, pos=pos, y=y)
    data2.edge_index = to_undirected(data2.edge_index)
    dataset.append(data2)
    
    return dataset


def _train_one_example(
    model, 
    train_loader, 
    val_loader, 
    num_epochs=100,
    print_freq=10,
    device='cpu'
):
    """
        1.  Run the whole training process on one example (e.g., two-body counterexample).
    """
    model = model.to(device)

    # Default optimizer used in "On the Expressive Power of Geometric Graph Neural Networks"
    params_learnable = []
    params_non_learnable = []
    name_non_learnable = []
    for n, p in model.named_parameters():
        if 'gating_linear' in n:
            params_non_learnable.append(p)
            name_non_learnable.append(n)
        else:
            params_learnable.append(p)
    print('Non-learnable parameters: {}'.format(name_non_learnable))
    optimizer = torch.optim.Adam(
        [
            {
                'params': params_learnable,
                'lr': 1e-4,
            },
            {   
                'params': params_non_learnable, 
                'lr': 0.0,
            }
        ]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.9, 
        patience=25, 
        min_lr=0.00001
    )
    
    best_val_acc = None
    perf_per_epoch = [] # Track validation performace vs. epoch (for plotting)
    train_start_time = time.perf_counter()
    for epoch in range(1, num_epochs + 1):
        start_time = time.perf_counter()

        # Train model for one epoch, return avg. training loss
        loss = train_one_epoch(model, train_loader, optimizer, device)
        
        # Evaluate model on validation set
        val_acc = eval(model, val_loader, device)

        if best_val_acc is None or val_acc >= best_val_acc:
            best_val_acc = val_acc
        
        if (epoch == 1) or (epoch % print_freq == 0):
            info_str = 'Epoch: [{epoch}]\t loss: {loss:.5f}, val acc: {val_acc:.3f}, lr={lr:.2e}, time={time_per_epoch:.0f}ms'.format( 
                epoch=epoch, 
                loss=loss,
                val_acc=val_acc,
                lr=optimizer.param_groups[0]['lr'],
                time_per_epoch=((time.perf_counter() - start_time) * 1000)
            )
            print(info_str)
        perf_per_epoch.append((val_acc, epoch))

        scheduler.step(val_acc)
    
    train_time = time.perf_counter() - train_start_time
    print("Training time: {train_time:.2f}s".format(train_time=train_time))
    print("Best validation accuracy: {best_val_acc:.3f}".format(best_val_acc=best_val_acc))
    
    return best_val_acc, train_time, perf_per_epoch


def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    loss_all = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        y_pred = model(batch)
        loss = torch.nn.functional.cross_entropy(y_pred, batch.y)
        loss.backward()
        loss_all += loss.item() * batch.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def accuracy_score(y_true, y_pred):
    score = y_true == y_pred
    score = score.to(torch.float)
    score = torch.mean(score)
    return score.item()


def eval(model, loader, device):
    model.eval()
    y_pred = []
    y_true = []
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            y_pred.append(model(batch).detach().cpu())
            y_true.append(batch.y.detach().cpu())
    return accuracy_score(
        torch.concat(y_true, dim=0), 
        torch.argmax(torch.concat(y_pred, dim=0), dim=1)
    )


if __name__ == '__main__':
    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """
        Adjust the following hyper-parameters for different counterexamples and model configs:
            _body_order
            _model_config
            _num_ffns
            _num_layers
            _use_attn
            _use_gaunt_self_tensor_product
    """
    # ['2', '3', '4_chiral', '4_nonchiral']
    _body_order = '4_nonchiral'
    
    # ['gate', 'sep_s2', 'sep_s2_swiglu', 'sep-merge_s2_swiglu', 'sep-merge_gates2_swiglu']
    _model_config = 'sep-merge_gates2_swiglu'

    _num_ffns = 2
    _num_layers = 1
    _use_attn = False

    """
        1.  When setting `use_gaunt_self_tensor_product` == `True`, we will ignore the `_model_config`
            for feedforward networks.
        2.  `use_gaunt_self_tensor_product` == `True` will have path-level learnable weights, which is
            found to be helpful for '4_non_chiral'.
    """
    _use_gaunt_self_tensor_product = False

    model_config_zoo = {
        'gate': ('gate', 'gate', False),
        'sep_s2': ('sep_s2', 'sep_s2', False),
        'sep_s2_swiglu': ('sep_s2_swiglu', 'sep_s2_swiglu', False),
        'sep-merge_s2_swiglu': ('sep-merge_s2_swiglu', 'sep-merge_s2_swiglu', False),
        'sep-merge_gates2_swiglu': ('sep-merge_gates2_swiglu', 'sep-merge_gates2_swiglu', False)
    }
    attn_activation, ffn_activation, use_grid_mlp = model_config_zoo[_model_config]

    model = EquiformerV3BodyOrderTest(
        max_neighbors=50,
        max_radius=10.0,
        num_radial_basis=128,
        max_num_elements=128,
        num_layers=_num_layers,
        num_channels=128,
        attn_hidden_channels=64,
        num_heads=8,
        attn_alpha_channels=64,
        attn_value_channels=16,
        ffn_hidden_channels=512,
        norm_type='equivariant_layer_norm',

        lmax=6,
        mmax=2,
        attn_grid_resolution_list=[20, 8],
        ffn_grid_resolution_list=[20, 20],

        edge_channels=128,
        use_atom_edge_embedding=True,
        use_envelope=True,

        attn_activation=attn_activation,
        use_attn_renorm=True,
        use_add_merge=False,
        use_rad_l_parametrization=True,
        softcap=None,
        ffn_activation=ffn_activation,
        use_grid_mlp=use_grid_mlp,

        alpha_drop=0.0,
        attn_mask_rate=0.0,
        attn_weights_drop=0.0,
        value_drop=0.0,
        drop_path_rate=0.0,
        proj_drop=0.0,
        ffn_drop=0.0,

        avg_num_nodes=1,
        avg_degree=1,

        use_attn=_use_attn,
        num_ffns=_num_ffns,
        use_gaunt_self_tensor_product=_use_gaunt_self_tensor_product,
        num_output_channels=2,
    )
    print(model)

    # Create two-body, three-body, and four-body counterexamples
    dataset = create_many_body_dataset(body_order=_body_order)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    _train_one_example(
        model, 
        train_loader=dataloader, 
        val_loader=val_loader, 
        num_epochs=1000,
        print_freq=10,
        device=device
    )