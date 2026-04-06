import torch
import typer
from typing import Annotated


def remove_torch_compile_from_checkpoint_and_save(
    input_path: Annotated[
        str, typer.Option(help="Input path to checkpoints")
    ]
) -> None:
    """
        1.  `input_path` should be something like `.../checkpoint.pt`.
        2.  This will generate a new .pt file named `.../checkpoint_no-torch-compile.pt`, which 
            removes `torch.compile()` related keys.
    """
    checkpoint = torch.load(input_path, map_location='cpu')
    state_dict = {}
    for k in checkpoint['state_dict']:
        new_key = k.replace('_orig_mod.module.', '')
        state_dict[new_key] = checkpoint['state_dict'][k]
    checkpoint['state_dict'] = state_dict
    torch.save(checkpoint, input_path.replace('.pt', '_no-torch-compile.pt'))


if __name__ == "__main__":
    typer.run(remove_torch_compile_from_checkpoint_and_save)