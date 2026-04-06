import torch
import typer
from typing import Annotated, List


def remove_key_from_checkpoint_and_save(
    input_path: Annotated[
        str, typer.Option(help="Input path to checkpoints")
    ],
    remove_name: Annotated[
        List[str], typer.Option(help="List of names of keys to be removed")
    ]
) -> None:
    """
        1.  `input_path` should be something like `.../checkpoint.pt`.
        2.  This will generate a new .pt file named `.../checkpoint_no-{}.pt`, which 
            removes key names specified in `remove_name`. 
            The {} would be the concatentation of the names of keys being removed.
    """
    checkpoint = torch.load(input_path, map_location='cpu')
    state_dict = {}
    remove_key_name_list = remove_name
    for k in checkpoint['state_dict']:
        if any(remove_key_name in k for remove_key_name in remove_key_name_list):
            print('Remove {} from the updated checkpoint'.format(k))
            continue
        state_dict[k] = checkpoint['state_dict'][k]
    checkpoint['state_dict'] = state_dict

    update_state_dict_name = '-'.join(remove_name)
    torch.save(
        checkpoint, 
        input_path.replace(
            '.pt', 
            '_no-{}.pt'.format(update_state_dict_name)
        )
    )


if __name__ == "__main__":
    typer.run(remove_key_from_checkpoint_and_save)