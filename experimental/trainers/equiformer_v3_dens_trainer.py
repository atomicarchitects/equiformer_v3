import logging
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch_geometric
from tqdm import tqdm
import copy
from torch.nn.parallel.distributed import DistributedDataParallel

from torch_geometric.utils import scatter

from fairchem.core.common import distutils
from fairchem.core.common.registry import registry
from fairchem.core.models.equiformer_v2.trainers.forces_trainer import (
    EquiformerV2ForcesTrainer,
)
from fairchem.core.models.equiformer_v2.trainers.lr_scheduler import LRScheduler
from fairchem.core.modules.evaluator import mae
from fairchem.core.modules.normalization.normalizer import Normalizer
from fairchem.core.modules.scaling.util import ensure_fitted

from .exponential_moving_average_v2 import ExponentialMovingAverageV2   # for moving EMA to CPU


@dataclass
class DenoisingPosParams:
    prob: float = 0.0
    fixed_noise_std: bool = True
    std: float = None
    corrupt_ratio: float = None
    all_atoms: bool = False
    denoising_pos_coefficient: float = None
    coefficient_linear_decay_min_factor: float = None
    num_steps: int = None
    std_low: float = None
    std_high: float = None
    min_num_atoms: int = None
    strict_max_ratio: float = None
    max_force_norm: float = None
    max_stress_norm: float = None
    early_epochs: float = None
    max_mean_force_norm: float = None


def add_gaussian_noise_to_position(
    batch, 
    std, 
    corrupt_ratio=None, 
    all_atoms=False, 
    min_num_atoms=None,
    strict_max_ratio=None,
    max_force_norm=None,
    max_stress_norm=None,
    max_mean_force_norm=None,
):
    """
        1.  Update `pos` in `batch`.
        2.  Add `noise_vec` to `batch`, which will serve as the target for denoising positions.
        3.  Add `denoising_pos_forward` to switch to denoising mode during training.
        4.  Add `noise_mask` for partially corrupted structures when `corrupt_ratio` is not None.
        5.  If `all_atoms` == True, we add noise to all atoms including fixed ones.
        6.  Check whether `batch` has `skip_dens`. We do not add noise to structures when `skip_dens` == True.
        7.  If `min_num_atoms` != None, we do not add noise to structures with numbers of atoms
            less than `min_num_atoms`.
        8.  Add `dens_batch_mask` to specify which graphs we apply DeNS. 
            This is used when `min_num_atoms` is not None or `strict_ratio` == True.
            `dens_batch_mask` is to be used to mask out stress prediction during DeNS.
        9.  If `strict_max_ratio` is not `None`, we skip denoising for a certain structure if the number of
            corrupted atoms is great than `strict_max_ratio` * number of atoms in that structure.
        10. If `max_force_norm` is not `None`, we skip denoising if the maximum of L2 norm of forces is
            greater than `max_force_norm`.
        11. If `max_stress_norm` is not `None`, we do not add noise to atoms with 
            stress norm > `max_stress_norm`.
        12. If `max_mean_force_norm` is not `None`, we skip denoising if the L2 norm of the sum of atomwise forces
            is greater than `max_mean_force_norm`.
    """
    noise_vec = torch.zeros_like(batch.pos)
    noise_vec = noise_vec.normal_(mean=0.0, std=std)

    dens_batch_mask = torch.ones(
        (len(batch.natoms), ),
        device=batch.pos.device,
        dtype=torch.bool,
    )

    if corrupt_ratio is not None:
        noise_mask = torch.rand(
            (batch.pos.shape[0]),
            dtype=batch.pos.dtype,
            device=batch.pos.device,
        )
        noise_mask = noise_mask < corrupt_ratio
        noise_vec[(~noise_mask)] *= 0
        batch.noise_mask = noise_mask
    
    # Not add noise to structures with `skip_dens` == True
    if hasattr(batch, 'skip_dens'):
        batch_index = batch.batch
        skip_dens_index = batch.skip_dens.bool()
        dens_batch_mask = dens_batch_mask * (~skip_dens_index)
        skip_dens_index = skip_dens_index[batch_index]
        noise_mask = (~skip_dens_index)
        noise_vec[(~noise_mask)] *= 0
        if hasattr(batch, 'noise_mask'):
            batch.noise_mask = batch.noise_mask * noise_mask
        else:
            batch.noise_mask = noise_mask

    if min_num_atoms is not None:
        batch_index = batch.batch
        num_atoms = batch.natoms
        noise_mask = num_atoms >= min_num_atoms
        dens_batch_mask = dens_batch_mask * noise_mask
        noise_mask = noise_mask[batch_index]
        noise_vec[(~noise_mask)] *= 0
        if hasattr(batch, 'noise_mask'):
            batch.noise_mask = batch.noise_mask * noise_mask
        else:
            batch.noise_mask = noise_mask

    if strict_max_ratio is not None:
        assert corrupt_ratio is not None
        batch_index = batch.batch
        noise_mask_tensor = batch.noise_mask.float()
        num_corrupted_atoms = torch.zeros(
            (len(batch.natoms), ),
            device=batch.pos.device,
            dtype=batch.pos.dtype,
        )
        num_corrupted_atoms.index_add_(0, batch_index, noise_mask_tensor)
        noise_mask = (num_corrupted_atoms <= (batch.natoms * strict_max_ratio))
        dens_batch_mask = dens_batch_mask * noise_mask
        noise_mask = noise_mask[batch_index]
        noise_vec[(~noise_mask)] *= 0
        batch.noise_mask = batch.noise_mask * noise_mask

    if max_force_norm is not None:
        batch_index = batch.batch
        force_norm = torch.norm(batch.forces, dim=-1)
        force_norm_max_reduce = scatter(force_norm, batch_index, 0, reduce='max')
        noise_mask = force_norm_max_reduce <= max_force_norm
        dens_batch_mask = dens_batch_mask * noise_mask
        noise_mask = noise_mask[batch_index]
        noise_vec[(~noise_mask)] *= 0
        if hasattr(batch, 'noise_mask'):
            batch.noise_mask = batch.noise_mask * noise_mask
        else:
            batch.noise_mask = noise_mask
    
    if max_stress_norm is not None:
        batch_index = batch.batch
        stress_norm = batch.stress.reshape(-1, 9)
        stress_norm = stress_norm ** 2
        stress_norm = torch.sqrt(torch.sum(stress_norm, dim=1))
        noise_mask = stress_norm <= max_stress_norm
        dens_batch_mask = dens_batch_mask * noise_mask
        noise_mask = noise_mask[batch_index]
        noise_vec[(~noise_mask)] *= 0
        if hasattr(batch, 'noise_mask'):
            batch.noise_mask = batch.noise_mask * noise_mask
        else:
            batch.noise_mask = noise_mask

    if max_mean_force_norm is not None:
        batch_index = batch.batch
        forces_reduce = scatter(
            src=batch.forces, 
            index=batch_index, 
            dim=0, 
            reduce='sum'
        )
        forces_reduce_norm = torch.norm(forces_reduce, dim=-1)
        noise_mask = forces_reduce_norm <= max_mean_force_norm
        dens_batch_mask = dens_batch_mask * noise_mask
        noise_mask = noise_mask[batch_index]
        noise_vec[(~noise_mask)] *= 0
        if hasattr(batch, 'noise_mask'):
            batch.noise_mask = batch.noise_mask * noise_mask
        else:
            batch.noise_mask = noise_mask

    pos = batch.pos
    new_pos = pos + noise_vec
    if all_atoms:
        batch.pos = new_pos
    else:
        free_mask = batch.fixed == 0.0
        batch.pos[free_mask] = new_pos[free_mask]

    batch.noise_vec = noise_vec
    batch.denoising_pos_forward = True
    batch.dens_batch_mask = dens_batch_mask

    return batch


def denoising_pos_eval(
    evaluator, prediction, target, prev_metrics={}, denoising_pos_forward=False
):
    """
        1.  Overwrite the original Evaluator.eval() here: https://github.com/facebookresearch/fairchem/blob/977a80328f2be44649b414a9907a1d6ef2f81e95/src/fairchem/core/modules/evaluator.py#L88
        2.  This is to make sure we separate forces MAE and denoising positions MAE.
        3.  When `denoising_pos_forward` == True, we only evaluate MAE for simplicity.
    """
    if not denoising_pos_forward:
        return evaluator.eval(prediction, target, prev_metrics)

    metrics = prev_metrics
    for target_property in evaluator.target_metrics:
        metric_name = 'denoising_{}_mae'.format(target_property)    # only consider MAE here
        if target_property != 'forces':
            res = eval('mae')(prediction, target, target_property)
            metrics = evaluator.update(metric_name, res, metrics)
        else:
            if target.get("noise_mask", None) is None:
                res = eval("mae")(prediction, target, "forces")     # `forces` here corresponds to noise prediction
                metrics = evaluator.update("denoising_pos_mae", res, metrics)
            else:
                """
                    Handle the case of partially corrupted structures:
                    1.  `target["forces"]` contains both force and noise predictions.
                    2.  We separate force and noise predictions based on `noise_mask`.
                """
                target_tensor = target["forces"]
                prediction_tensor = prediction["forces"]
                noise_mask = target["noise_mask"]
                forces_index = torch.where(noise_mask == 0)
                forces_prediction = {"forces": prediction_tensor[forces_index]}
                forces_target = {"forces": target_tensor[forces_index]}
                res = eval("mae")(forces_prediction, forces_target, "forces")
                if res["numel"] != 0:
                    metrics = evaluator.update("denoising_force_mae", res, metrics)
                denoising_pos_index = torch.where(noise_mask == 1)
                denoising_pos_prediction = {"forces": prediction_tensor[denoising_pos_index]}
                denoising_pos_target = {"forces": target_tensor[denoising_pos_index]}
                res = eval("mae")(
                    denoising_pos_prediction, denoising_pos_target, "forces"
                )
                if res["numel"] != 0:
                    metrics = evaluator.update("denoising_pos_mae", res, metrics)
    return metrics


def compute_atomwise_denoising_pos_and_force_hybrid_loss(
    pred, target, noise_mask, force_mult, denoising_pos_mult, mask=None
):
    loss = torch.norm(pred - target, p=2, dim=-1, keepdim=True)
    force_index = torch.where(noise_mask == 0)
    denoising_pos_index = torch.where(noise_mask == 1)
    mult_tensor = torch.ones_like(loss)
    mult_tensor[force_index] *= force_mult
    mult_tensor[denoising_pos_index] *= denoising_pos_mult
    loss = loss * mult_tensor
    if mask is not None:
        loss = loss[mask]
    loss = torch.mean(loss)
    return loss


@registry.register_trainer("equiformer_v3_dens_trainer")
class EquiformerV3DeNSTrainer(EquiformerV2ForcesTrainer):
    """
    Args:
        task (dict): Task configuration.
        model (dict): Model configuration.
        outputs (dict): Dictionary of model output configuration.
        dataset (dict): Dataset configuration.
        optimizer (dict): Optimizer configuration.
        loss_functions (dict): Loss function configuration.
        evaluation_metrics (dict): Evaluation metrics configuration.
        identifier (str): Experiment identifier that is appended to log directory.
        run_dir (str, optional): Path to the run directory where logs are to be saved.
            (default: :obj:`None`)
        timestamp_id (str, optional): timestamp identifier.
        is_debug (bool, optional): Run in debug mode.
            (default: :obj:`False`)
        print_every (int, optional): Frequency of printing logs.
            (default: :obj:`100`)
        seed (int, optional): Random number seed.
            (default: :obj:`None`)
        logger (str, optional): Type of logger to be used.
            (default: :obj:`wandb`)
        local_rank (int, optional): Local rank of the process, only applicable for distributed training.
            (default: :obj:`0`)
        amp (bool, optional): Run using automatic mixed precision.
            (default: :obj:`False`)
        cpu (bool): If True will run on CPU. Default is False, will attempt to use cuda.
        name (str): Trainer name.
        slurm (dict): Slurm configuration. Currently just for keeping track.
            (default: :obj:`{}`)
        gp_gpus (int, optional): Number of graph parallel GPUs.
        inference_only (bool): If true trainer will be loaded for inference only.
            (ie datasets, optimizer, schedular, etc, will not be instantiated)
    """

    def __init__(
        self,
        task,
        model,
        outputs,
        dataset,
        optimizer,
        loss_functions,
        evaluation_metrics,
        identifier,
        timestamp_id=None,
        run_dir=None,
        is_debug=False,
        print_every=100,
        seed=None,
        logger="wandb",
        local_rank=0,
        amp=False,
        cpu=False,
        name="ocp",
        slurm=None,
        gp_gpus=None,
        inference_only=False,
    ):
        if slurm is None:
            slurm = {}
        super().__init__(
            task=task,
            model=model,
            outputs=outputs,
            dataset=dataset,
            optimizer=optimizer,
            loss_functions=loss_functions,
            evaluation_metrics=evaluation_metrics,
            identifier=identifier,
            timestamp_id=timestamp_id,
            run_dir=run_dir,
            is_debug=is_debug,
            print_every=print_every,
            seed=seed,
            logger=logger,
            local_rank=local_rank,
            amp=amp,
            cpu=cpu,
            slurm=slurm,
            name=name,
            gp_gpus=gp_gpus,
            inference_only=inference_only,
        )

        # for denoising positions
        self.use_denoising_pos = self.config["optim"]["use_denoising_pos"]
        self.denoising_pos_params = DenoisingPosParams(
            **self.config["optim"]["denoising_pos_params"]
        )
        assert self.denoising_pos_params.fixed_noise_std, 'This trainer only supports `fixed_noise_std` == True'
        if self.denoising_pos_params.coefficient_linear_decay_min_factor is not None:
            assert (
                (
                    self.denoising_pos_params.coefficient_linear_decay_min_factor >= 0.0
                ) and
                (
                    self.denoising_pos_params.coefficient_linear_decay_min_factor <= 1.0
                )
            )
            self.total_steps = len(self.train_loader) * self.config["optim"]["max_epochs"]
            
        self.denoising_pos_params.denoising_pos_coefficient = self.config["optim"][
            "denoising_pos_coefficient"
        ]
        self.normalizers["denoising_pos_target"] = Normalizer(
            mean=0.0,
            rmsd=(
                self.denoising_pos_params.std 
                if self.denoising_pos_params.fixed_noise_std
                else self.denoising_pos_params.std_high
            ),
        )
        self.normalizers["denoising_pos_target"].to(self.device)

        if self.config['optim'].get('use_compile', False):
            self.model = torch.compile(self.model, dynamic=True)
            torch._dynamo.config.optimize_ddp = False


    def train(self, disable_eval_tqdm=False):
        ensure_fitted(self._unwrapped_model, warn=True)

        eval_every = self.config["optim"].get(
            "eval_every", len(self.train_loader)
        )
        checkpoint_every = self.config["optim"].get(
            "checkpoint_every", eval_every
        )
        primary_metric = self.evaluation_metrics.get(
            "primary_metric", self.evaluator.task_primary_metric[self.name]
        )
        if (
            not hasattr(self, "primary_metric")
            or self.primary_metric != primary_metric
        ):
            self.best_val_metric = 1e9 if "mae" in primary_metric else -1.0
        else:
            primary_metric = self.primary_metric
        self.metrics = {}

        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // len(self.train_loader)

        for epoch_int in range(
            start_epoch, self.config["optim"]["max_epochs"]
        ):
            skip_steps = self.step % len(self.train_loader)
            self.train_sampler.set_epoch_and_start_iteration(
                epoch_int, skip_steps
            )
            train_loader_iter = iter(self.train_loader)

            self.metrics = {}

            for i in range(skip_steps, len(self.train_loader)):
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1
                self.model.train()
                
                # Get a batch.
                batch = next(train_loader_iter)

                # for denoising positions
                if self.use_denoising_pos:
                    if (self.denoising_pos_params.early_epochs is None) or (self.epoch <= self.denoising_pos_params.early_epochs):
                        if np.random.rand() < self.denoising_pos_params.prob:
                            if self.denoising_pos_params.fixed_noise_std:
                                batch = add_gaussian_noise_to_position(
                                    batch,
                                    std=self.denoising_pos_params.std,
                                    corrupt_ratio=self.denoising_pos_params.corrupt_ratio,
                                    all_atoms=self.denoising_pos_params.all_atoms,
                                    min_num_atoms=self.denoising_pos_params.min_num_atoms,
                                    strict_max_ratio=self.denoising_pos_params.strict_max_ratio,
                                    max_force_norm=self.denoising_pos_params.max_force_norm,
                                    max_stress_norm=self.denoising_pos_params.max_stress_norm,
                                    max_mean_force_norm=self.denoising_pos_params.max_mean_force_norm,
                                )

                # Forward, loss, backward.
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    out = self._forward(batch)
                    loss = self._compute_loss(out, batch)
                loss = self.scaler.scale(loss) if self.scaler else loss
                if self.grad_accumulation_steps != 1:
                    loss = loss / self.grad_accumulation_steps
                self._backward(loss)
                scale = self.scaler.get_scale() if self.scaler else 1.0
    
                # Compute metrics.
                self.metrics = self._compute_metrics(
                    out,
                    batch,
                    self.evaluator,
                    self.metrics,
                )
                self.metrics = self.evaluator.update(
                    "loss", loss.item() / scale * self.grad_accumulation_steps, self.metrics
                )

                # Log metrics.
                log_dict = {k: self.metrics[k]["metric"] for k in self.metrics}
                log_dict.update(
                    {
                        "lr": self.scheduler.get_lr(),
                        "epoch": self.epoch,
                        "step": self.step,
                    }
                )
                if (
                    self.step % self.config["cmd"]["print_every"] == 0
                    or i == 0
                    or i == (len(self.train_loader) - 1)
                ) and distutils.is_master():
                    log_str = [
                        "{}: {:.2e}".format(k, v) for k, v in log_dict.items()
                    ]
                    logging.info(", ".join(log_str))
                    self.metrics = {}

                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split="train",
                    )

                if (
                    checkpoint_every != -1
                    and self.step % checkpoint_every == 0
                ):
                    self.save(
                        checkpoint_file="checkpoint.pt", training_state=True
                    )

                # Evaluate on val set every `eval_every` iterations.
                if self.step % eval_every == 0 or i == (
                    len(self.train_loader) - 1
                ):
                    if self.val_loader is not None:
                        if i == (len(self.train_loader) - 1):
                            self.save(
                                checkpoint_file="checkpoint.pt",
                                training_state=True,
                            )

                        val_metrics = self.validate(
                            split="val", disable_tqdm=disable_eval_tqdm
                        )
                        self.update_best(
                            primary_metric,
                            val_metrics,
                            disable_eval_tqdm=disable_eval_tqdm,
                        )

                    if self.config["task"].get("eval_relaxations", False):
                        if "relax_dataset" not in self.config["task"]:
                            logging.warning(
                                "Cannot evaluate relaxations, relax_dataset not specified"
                            )
                        else:
                            self.run_relaxations()

                if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                    if self.step % eval_every == 0:
                        self.scheduler.step(
                            metrics=val_metrics[primary_metric]["metric"],
                        )
                else:
                    if self.step % self.grad_accumulation_steps == 0:
                        self.scheduler.step()

            # torch.cuda.empty_cache()

            if checkpoint_every == -1:
                self.save(checkpoint_file="checkpoint.pt", training_state=True)

        if hasattr(self.train_dataset, 'close_db'):
            self.train_dataset.close_db()
        if self.config.get("val_dataset", False):
            if hasattr(self.val_dataset, 'close_db'):
                self.val_dataset.close_db()
        if self.config.get("test_dataset", False):
            if hasattr(self.test_dataset, 'close_db'):
                self.test_dataset.close_db()


    def _compute_current_denoising_pos_coefficient(self):
        if self.denoising_pos_params.coefficient_linear_decay_min_factor is None:
            return self.denoising_pos_params.denoising_pos_coefficient

        progress = min(1.0, ((self.step + 0.0) / self.total_steps))
        range = (1.0 - self.denoising_pos_params.coefficient_linear_decay_min_factor)
        weight = 1.0 - range * progress
        weight = weight * self.denoising_pos_params.denoising_pos_coefficient
        return weight
    

    def _compute_loss(self, out, batch):
        batch_size = batch.natoms.numel()
        fixed = batch.fixed
        mask = fixed == 0

        loss = []
        for loss_fn in self.loss_functions:
            target_name, loss_info = loss_fn

            if target_name == "forces" and batch.get(
                "denoising_pos_forward", False
            ):
                denoising_pos_target = batch.noise_vec
                if self.normalizers.get("denoising_pos_target", False):
                    denoising_pos_target = self.normalizers[
                        "denoising_pos_target"
                    ].norm(denoising_pos_target)

                if hasattr(batch, "noise_mask"):
                    # for partially corrupted structures
                    target = batch.forces
                    if self.normalizers.get("forces", False):
                        target = self.normalizers["forces"].norm(target)
                    noise_mask = batch.noise_mask.view(-1, 1)
                    target = denoising_pos_target * noise_mask + target * (~noise_mask)
                else:
                    target = denoising_pos_target

                pred = out["forces"]
                natoms = batch.natoms
                natoms = torch.repeat_interleave(natoms, natoms)

                force_mult = loss_info["coefficient"]

                #denoising_pos_mult = self.denoising_pos_params.denoising_pos_coefficient
                denoising_pos_mult = self._compute_current_denoising_pos_coefficient()
                
                if (
                    self.output_targets[target_name]["level"] == "atom"
                    and self.output_targets[target_name]["train_on_free_atoms"]
                ):
                    # If `all_atoms` == True when training on only free atoms,
                    # we also add noise to and denoise fixed atoms.
                    if self.denoising_pos_params.all_atoms:
                        if hasattr(batch, "noise_mask"):
                            mask = mask.view(-1, 1) | noise_mask
                        else:
                            mask = torch.ones_like(
                                mask, dtype=torch.bool, device=mask.device
                            ).view(-1, 1)

                    if hasattr(batch, "noise_mask"):
                        # for partially corrupted structures
                        loss.append(
                            compute_atomwise_denoising_pos_and_force_hybrid_loss(
                                pred=pred,
                                target=target,
                                noise_mask=noise_mask,
                                force_mult=force_mult,
                                denoising_pos_mult=denoising_pos_mult,
                                mask=mask,
                            )
                        )
                    else:
                        target = target[mask]
                        pred = pred[mask]
                        natoms = natoms[mask]

                        loss.append(
                            denoising_pos_mult
                            * loss_info["fn"](
                                pred,
                                target,
                                natoms=natoms,
                                #batch_size=batch_size,
                            )
                        )
                else:
                    if hasattr(batch, "noise_mask"):
                        # for partially corrupted structures
                        loss.append(
                            compute_atomwise_denoising_pos_and_force_hybrid_loss(
                                pred=pred,
                                target=target,
                                noise_mask=noise_mask,
                                force_mult=force_mult,
                                denoising_pos_mult=denoising_pos_mult,
                                mask=None,
                            )
                        )
                    else:
                        loss.append(
                            denoising_pos_mult
                            * loss_info["fn"](
                                pred,
                                target,
                                natoms=natoms,
                                #batch_size=batch_size,
                            )
                        )
            else:
                target = batch[target_name]
                pred = out[target_name]
                natoms = batch.natoms
                natoms = torch.repeat_interleave(natoms, natoms)

                if (
                    self.output_targets[target_name]["level"] == "atom"
                    and self.output_targets[target_name]["train_on_free_atoms"]
                ):
                    target = target[mask]
                    pred = pred[mask]
                    natoms = natoms[mask]

                num_atoms_in_batch = natoms.numel()
                ### reshape accordingly: num_atoms_in_batch, -1 or num_systems_in_batch, -1
                if self.output_targets[target_name]["level"] == "atom":
                    target = target.view(num_atoms_in_batch, -1)
                else:
                    target = target.view(batch_size, -1)

                # to keep the loss coefficient weights balanced we remove linear references
                # subtract element references from target data
                if target_name in self.elementrefs:
                    target = self.elementrefs[target_name].dereference(target, batch)

                if self.normalizers.get(target_name, False):
                    target = self.normalizers[target_name].norm(target)

                mult = loss_info["coefficient"]
                loss.append(
                    mult
                    * loss_info["fn"](
                        pred,
                        target,
                        natoms=batch.natoms,
                        #batch_size=batch_size,
                    )
                )

        # Sanity check to make sure the compute graph is correct.
        for lc in loss:
            assert hasattr(lc, "grad_fn")

        loss = sum(loss)
        return loss

    def _compute_metrics(self, out, batch, evaluator, metrics={}):
        # this function changes the values in the out dictionary,
        # make a copy instead of changing them in the callers version
        out = {k: v.clone() for k, v in out.items()}

        # This assumes batch.fixed is specified correctly for each dataset.

        natoms = batch.natoms
        batch_size = natoms.numel()

        ### Retrieve free atoms
        fixed = batch.fixed
        mask = fixed == 0

        s_idx = 0
        natoms_free = []
        for _natoms in natoms:
            natoms_free.append(torch.sum(mask[s_idx : s_idx + _natoms]).item())
            s_idx += _natoms
        natoms = torch.LongTensor(natoms_free).to(self.device)

        denoising_pos_forward = False
        if batch.get("denoising_pos_forward", False):
            denoising_pos_forward = True

        targets = {}
        for target_name in self.output_targets:
            num_atoms_in_batch = batch.natoms.sum()

            if denoising_pos_forward and target_name == "forces":
                if hasattr(batch, "noise_mask"):
                    force_target = batch.forces
                    denoising_pos_target = batch.noise_vec
                    noise_mask = batch.noise_mask
                    forces_index = torch.where(noise_mask == 0)
                    denoising_pos_index = torch.where(noise_mask == 1)
                    noise_mask_tensor = noise_mask.view(-1, 1)
                    targets["forces"] = (
                        denoising_pos_target * noise_mask_tensor
                        + force_target * (~noise_mask_tensor)
                    )
                    targets["noise_mask"] = noise_mask
                else:
                    targets["forces"] = batch.noise_vec

                if self.normalizers.get("denoising_pos_target", False):
                    if hasattr(batch, "noise_mask"):
                        out["forces"][denoising_pos_index] = self.normalizers[
                            "denoising_pos_target"
                        ].denorm(out["forces"][denoising_pos_index])
                    else:
                        out["forces"] = self.normalizers[
                            "denoising_pos_target"
                        ].denorm(out["forces"])

                if hasattr(batch, "noise_mask"):
                    out["forces"][forces_index] = self.normalizers[
                        "forces"
                    ].denorm(out["forces"][forces_index])

                if (
                    self.output_targets[target_name]["level"] == "atom"
                    and self.output_targets[target_name]["eval_on_free_atoms"]
                ):
                    if self.denoising_pos_params.all_atoms:
                        if hasattr(batch, "noise_mask"):
                            mask = mask | noise_mask
                        else:
                            mask = torch.ones_like(
                                mask, dtype=torch.bool, device=mask.device
                            )

                    targets["forces"] = targets["forces"][mask]
                    out["forces"] = out["forces"][mask]
                    num_atoms_in_batch = natoms.sum()
                    if "noise_mask" in targets:
                        targets["noise_mask"] = targets["noise_mask"][mask]
            else:
                target = batch[target_name]

                if (
                    self.output_targets[target_name]["level"] == "atom"
                    and self.output_targets[target_name]["eval_on_free_atoms"]
                ):
                    target = target[mask]
                    out[target_name] = out[target_name][mask]
                    num_atoms_in_batch = natoms.sum()

                ### reshape accordingly: num_atoms_in_batch, -1 or num_systems_in_batch, -1
                if self.output_targets[target_name]["level"] == "atom":
                    target = target.view(num_atoms_in_batch, -1)
                else:
                    target = target.view(batch_size, -1)

                out[target_name] = self._denorm_preds(
                    target_name, out[target_name], batch
                )
                targets[target_name] = target
                
        targets["natoms"] = natoms
        out["natoms"] = natoms

        metrics = denoising_pos_eval(
            evaluator,
            out,
            targets,
            prev_metrics=metrics,
            denoising_pos_forward=denoising_pos_forward,
        )

        return metrics


    @torch.no_grad()
    def predict(
        self,
        data_loader,
        per_image: bool = True,
        results_file: Optional[str] = None,
        disable_tqdm: bool = False,
    ):
        if self.is_debug and per_image:
            raise FileNotFoundError(
                "Predictions require debug mode to be turned off."
            )

        ensure_fitted(self._unwrapped_model, warn=True)

        if distutils.is_master() and not disable_tqdm:
            logging.info("Predicting on test.")
        assert isinstance(
            data_loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
            ),
        )
        rank = distutils.get_rank()

        if isinstance(data_loader, torch_geometric.data.Batch):
            data_loader = [data_loader]

        self.model.eval()
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        predictions = defaultdict(list)

        for key in self.normalizers.keys():
            self.normalizers[key].to(self.device)

        for i, batch in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                batch = batch.to(self.device)
                out = self._forward(batch)

            for key in out.keys():
                out[key] = out[key].float()

            for target_key in self.config["outputs"]:
                pred = self._denorm_preds(target_key, out[target_key], batch)

                if per_image:
                    ### Save outputs in desired precision, default float16
                    if (
                        self.config["outputs"][target_key].get(
                            "prediction_dtype", "float16"
                        )
                        == "float32"
                        or self.config["task"].get(
                            "prediction_dtype", "float16"
                        )
                        == "float32"
                        or self.config["task"].get("dataset", "lmdb")
                        == "oc22_lmdb"
                    ):
                        dtype = torch.float32
                    else:
                        dtype = torch.float16

                    #pred = pred.cpu().detach().to(dtype)
                    pred = pred.detach().cpu().to(dtype)
                    
                    ### Split predictions into per-image predictions
                    if self.config["outputs"][target_key]["level"] == "atom":
                        batch_natoms = batch.natoms
                        batch_fixed = batch.fixed
                        per_image_pred = torch.split(
                            pred, batch_natoms.tolist()
                        )

                        ### Save out only free atom, EvalAI does not need fixed atoms
                        _per_image_fixed = torch.split(
                            batch_fixed, batch_natoms.tolist()
                        )
                        _per_image_free_preds = [
                            _pred[(fixed == 0).tolist()].numpy()
                            for _pred, fixed in zip(
                                per_image_pred, _per_image_fixed
                            )
                        ]
                        _chunk_idx = np.array(
                            [
                                free_pred.shape[0]
                                for free_pred in _per_image_free_preds
                            ]
                        )
                        per_image_pred = _per_image_free_preds
                    ### Assumes system level properties are of the same dimension
                    else:
                        per_image_pred = pred.numpy()
                        _chunk_idx = None

                    predictions[f"{target_key}"].extend(per_image_pred)
                    ### Backwards compatibility, retain 'chunk_idx' for forces.
                    if _chunk_idx is not None:
                        if target_key == "forces":
                            predictions["chunk_idx"].extend(_chunk_idx)
                        else:
                            predictions[f"{target_key}_chunk_idx"].extend(
                                _chunk_idx
                            )
                else:
                    predictions[f"{target_key}"] = pred.detach()

            if not per_image:
                return predictions

            ### Get unique system identifiers
            sids = (
                batch.sid.tolist()
                if isinstance(batch.sid, torch.Tensor)
                else batch.sid
            )
            ## Support naming structure for OC20 S2EF
            if "fid" in batch:
                fids = (
                    batch.fid.tolist()
                    if isinstance(batch.fid, torch.Tensor)
                    else batch.fid
                )
                systemids = [f"{sid}_{fid}" for sid, fid in zip(sids, fids)]
            else:
                systemids = [f"{sid}" for sid in sids]

            predictions["ids"].extend(systemids)

        for key in predictions:
            if isinstance(predictions[key][0], np.ndarray):
                predictions[key] = np.concatenate(predictions[key], axis=0)
            else:
                predictions[key] = np.array(predictions[key])
        
        self.save_results(predictions, results_file)

        if self.ema:
            self.ema.restore()

        return predictions


    def update_best(
        self,
        primary_metric,
        val_metrics,
        disable_eval_tqdm: bool = True,
    ) -> None:
        '''
            1.  We also set `training_state` == True when saving the best checkpoint.
        '''
        if (
            "mae" in primary_metric
            and val_metrics[primary_metric]["metric"] < self.best_val_metric
        ) or (
            "mae" not in primary_metric
            and val_metrics[primary_metric]["metric"] > self.best_val_metric
        ):
            self.best_val_metric = val_metrics[primary_metric]["metric"]
            self.save(
                metrics=val_metrics,
                checkpoint_file="best_checkpoint.pt",
                training_state=True,
            )
            if self.test_loader is not None:
                self.predict(
                    self.test_loader,
                    results_file="predictions",
                    disable_tqdm=disable_eval_tqdm,
                )
    

    @torch.no_grad()
    def validate(self, split: str = "val", disable_tqdm: bool = False):
        metrics = super().validate(split, disable_tqdm)
        if self.ema:
            del self.ema.collected_params
            self.ema.collected_params = []
        return metrics
    

    def load_extras(self) -> None:
        """
            1.  Suppport gradient accumulation (`self.grad_accumulation_steps`).
            2.  Use ExponentialMovingAverageV2. 
        """
        def multiply(obj, num):
            if isinstance(obj, list):
                for i in range(len(obj)):
                    obj[i] = obj[i] * num
            else:
                obj = obj * num
            return obj

        self.config["optim"]["scheduler_params"]["epochs"] = self.config[
            "optim"
        ]["max_epochs"]
        self.config["optim"]["scheduler_params"]["lr"] = self.config["optim"][
            "lr_initial"
        ]

        self.grad_accumulation_steps = self.config['optim'].get('grad_accumulation_steps', 1)

        # convert epochs into number of steps
        if self.train_loader is None:
            logging.warning("Skipping scheduler setup. No training set found.")
            self.scheduler = None
        else:
            n_iter_per_epoch = len(self.train_loader)
            if self.grad_accumulation_steps != 1:
                n_iter_per_epoch = n_iter_per_epoch // self.grad_accumulation_steps
            scheduler_params = self.config["optim"]["scheduler_params"]
            for k in scheduler_params.keys():
                if "epochs" in k:
                    if isinstance(scheduler_params[k], (int, float)):
                        scheduler_params[k] = int(
                            multiply(scheduler_params[k], n_iter_per_epoch)
                        )
                    elif isinstance(scheduler_params[k], list):
                        scheduler_params[k] = [
                            int(x)
                            for x in multiply(
                                scheduler_params[k], n_iter_per_epoch
                            )
                        ]
            self.scheduler = LRScheduler(self.optimizer, self.config["optim"])

        self.clip_grad_norm = self.config["optim"].get("clip_grad_norm")
        self.ema_decay = self.config["optim"].get("ema_decay")
        if self.ema_decay:
            # for moving EMA to CPU
            self.ema = ExponentialMovingAverageV2(
                self.model.parameters(),
                self.ema_decay,
            )
        else:
            self.ema = None


    def load_model(self) -> None:
        """
            1.  Overwrite the original `self.load_model()` so that we can load 
                only model weihgts from a checkpoint without loading others like
                optimizer states.
                This is to support direct pre-training and gradient fine-tuning.
            2.  We will automatically remove `torch.compile()`-related keywords 
                (i.e., "_orig_mod.module.").
        """
        if distutils.is_master():
            logging.info(f"Loading model: {self.config['model']['name']}")

        model_config_copy = copy.deepcopy(self.config["model"])
        model_name = model_config_copy.pop("name")
        self.model = registry.get_model_class(model_name)(
            **model_config_copy,
        ).to(self.device)

        num_params = sum(p.numel() for p in self.model.parameters())

        if distutils.is_master():
            logging.info(self.model)
            logging.info(
                f"Loaded {self.model.__class__.__name__} with "
                f"{num_params} parameters."
            )

        if self.config['optim'].get('load_pretrained_weights', None):
            if distutils.is_master():
                logging.info('Loading pre-trained model weights from {}'.format(
                    self.config['optim']['load_pretrained_weights'])
                )
            checkpoint = torch.load(
                self.config['optim']['load_pretrained_weights'],
                map_location='cpu'
            )
            model_state_dict = self.model.state_dict()
            for k in checkpoint['state_dict'].keys():
                # Remove `torch.compile()`-related or distributed training related keywords
                if k.startswith('_orig_mod.module.'):
                    new_key = k.replace('_orig_mod.module.', '')
                elif k.startswith('module.'):
                    new_key = k.replace('module.', '')
                else:
                    new_key = k
                if new_key in model_state_dict.keys():
                    assert model_state_dict[new_key].shape == checkpoint['state_dict'][k].shape
                    model_state_dict[new_key] = checkpoint['state_dict'][k]
                else:
                    if distutils.is_master():
                        logging.info('{} in the checkpoint but not in the state_dict of model'.format(new_key))
            self.model.load_state_dict(model_state_dict)            

        if self.logger is not None:
            # only "watch" model if user specify watch: True because logging gradients
            # spews too much data into W&B and makes the UI slow to respond
            if "watch" in self.config["logger"]:
                self.logger.watch(
                    self.model, log_freq=int(self.config["logger"]["watch"])
                )
            self.logger.log_summary({"num_params": num_params})

        if distutils.initialized():
            self.model = DistributedDataParallel(
                self.model,
            )

    def _backward(self, loss) -> None:
        """
            1.  Add gradient accumulation.
        """
        if self.grad_accumulation_steps == 1:
            self.optimizer.zero_grad()
        
        loss.backward()
        
        # Scale down the gradients of shared parameters
        if hasattr(self.model, "shared_parameters"):
            for p, factor in self.model.shared_parameters:
                if hasattr(p, "grad") and p.grad is not None:
                    p.grad.detach().div_(factor)
                else:
                    if not hasattr(self, "warned_shared_param_no_grad"):
                        self.warned_shared_param_no_grad = True
                        logging.warning(
                            "Some shared parameters do not have a gradient. "
                            "Please check if all shared parameters are used "
                            "and point to PyTorch parameters."
                        )
        
        if (self.grad_accumulation_steps != 1):
            if (self.step % self.grad_accumulation_steps != 0):
                return 

        if self.clip_grad_norm:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.clip_grad_norm,
            )
            if self.logger is not None:
                self.logger.log(
                    {"grad_norm": grad_norm}, step=self.step, split="train"
                )
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.ema:
            self.ema.update()
        
        if (self.grad_accumulation_steps != 1):
            if (self.step % self.grad_accumulation_steps == 0):
                self.optimizer.zero_grad() 
