import pandas as pd

import numpy as np

import torch

from experiments.dense_decoder import (
    RADII,
    disk_average_probabilities,
    select_decoder_from_scores,
    states_from_masks,
)


class Trainer:

    def __init__(

        self,

        data_loaders,

        criterion,

        device,

        scheduler=None,

        on_after_epoch=None,

        use_amp=False,

        early_stopping_patience=None,

        recon_schedule=None,

        early_stopping_metric="val_loss",

        dense_site_xy=None,

    ):

        self.data_loaders = data_loaders

        self.criterion = criterion

        self.device = device

        self.history = []

        self.on_after_epoch = on_after_epoch

        self.scheduler = scheduler

        self.use_amp = use_amp and device.type == "cuda"

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.early_stopping_patience = early_stopping_patience

        self.recon_schedule = recon_schedule

        self.early_stopping_metric = early_stopping_metric

        self.dense_site_xy = None if dense_site_xy is None else np.asarray(dense_site_xy)


    def train(self, model, optimizer, num_epochs):

        self.history = []

        maximize = self.early_stopping_metric == "val_ion_acc"

        best_metric = float("-inf") if maximize else float("inf")

        no_improve = 0


        for epoch in range(num_epochs):

            set_temp = getattr(model, "set_temperature_progress", None)

            if set_temp is not None:

                progress = epoch / max(num_epochs - 1, 1)

                set_temp(progress)


            train_stats = self._train_on_epoch(model, optimizer, epoch)

            val_stats = self._val_on_epoch(model, epoch)


            if self.scheduler is not None:

                self.scheduler.step(val_stats["loss"])


            hist = {

                "epoch": epoch,

                "train_loss": train_stats.get("loss", 0.0),

                "val_loss": val_stats.get("loss", 0.0),

                "train_loss_bce": train_stats.get("loss_bce", 0.0),

                "val_loss_bce": val_stats.get("loss_bce", 0.0),

                "train_loss_dice": train_stats.get("loss_dice", 0.0),

                "val_loss_dice": val_stats.get("loss_dice", 0.0),

                "train_loss_centroid": train_stats.get("loss_centroid", 0.0),

                "val_loss_centroid": val_stats.get("loss_centroid", 0.0),

                "train_loss_state": train_stats.get("loss_state", 0.0),

                "val_loss_state": val_stats.get("loss_state", 0.0),

                "train_loss_coord": train_stats.get("loss_coord", 0.0),

                "val_loss_coord": val_stats.get("loss_coord", 0.0),

                "train_loss_exist": train_stats.get("loss_exist", 0.0),

                "val_loss_exist": val_stats.get("loss_exist", 0.0),

                "train_loss_recon": train_stats.get("loss_recon", 0.0),

                "val_loss_recon": val_stats.get("loss_recon", 0.0),

                "train_ion_acc": train_stats.get("ion_acc", 0.0),

                "val_ion_acc": val_stats.get("ion_acc", 0.0),

                "decoder_radius": val_stats.get("decoder_radius", float("nan")),

                "decoder_threshold": val_stats.get("decoder_threshold", float("nan")),

                "decoder_youden_j": val_stats.get("decoder_youden_j", float("nan")),

                "current_lr": round(optimizer.param_groups[0]["lr"], 8),

            }

            self.history.append(hist)


            if self.on_after_epoch is not None:

                self.on_after_epoch(model, pd.DataFrame(self.history))


            if self.early_stopping_patience is not None:

                current = (
                    val_stats.get("ion_acc", float("-inf"))
                    if maximize
                    else val_stats["loss"]
                )

                improved = (
                    current > best_metric + 1e-6
                    if maximize
                    else current < best_metric - 1e-4
                )

                if improved:

                    best_metric = current

                    no_improve = 0

                else:

                    no_improve += 1

                if no_improve >= self.early_stopping_patience:

                    print(
                        f"Early stopping at epoch {epoch} on {self.early_stopping_metric} "
                        f"(no improvement for {no_improve} epochs)"
                    )

                    break


        return pd.DataFrame(self.history)


    def _forward_and_loss(self, model, batch, epoch=0):

        inputs = batch["image"].to(self.device)

        if "site_coords" in batch:

            site_coords = batch["site_coords"].to(self.device)

            outputs = model(inputs, site_coords=site_coords)

            if self.recon_schedule is not None:

                recon_weight = self.recon_schedule.weight_at(epoch)

                detach_state_grad = self.recon_schedule.detach_state_grad_at(epoch)

                loss_dict = self.criterion(

                    outputs, batch, recon_weight=recon_weight, detach_state_grad=detach_state_grad

                )

            else:

                loss_dict = self.criterion(outputs, batch)

        else:

            labels = batch["mask"].to(self.device)

            centers_gt = batch["centers_gt"].to(self.device)

            centers_valid = batch["centers_valid"].to(self.device)

            outputs = model(inputs)

            loss_dict = self.criterion(outputs, labels, centers_gt, centers_valid)

        return inputs, loss_dict, outputs


    def _accumulate(self, running, loss_dict, batch_size):

        for k, v in loss_dict.items():

            if not torch.is_tensor(v) or v.numel() != 1:

                continue

            running.setdefault(k, 0.0)

            running[k] += float(v.item()) * batch_size


    def _finalize(self, running, dataset_size):

        for k in list(running.keys()):

            running[k] /= dataset_size

        return running


    def _train_on_epoch(self, model, optimizer, epoch=0):

        model.train()

        data_loader = self.data_loaders[0]

        running = {}


        for batch in data_loader:

            optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(True):

                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):

                    inputs, loss_dict, _ = self._forward_and_loss(model, batch, epoch)

                    loss = loss_dict["loss"]

                self.scaler.scale(loss).backward()

                self.scaler.step(optimizer)

                self.scaler.update()


            self._accumulate(running, loss_dict, inputs.size(0))


        return self._finalize(running, len(data_loader.dataset))


    def _val_on_epoch(self, model, epoch=0):

        model.eval()

        data_loader = self.data_loaders[1]

        running = {}

        scores_by_radius = {radius: [] for radius in RADII}

        state_labels = []


        for batch in data_loader:

            with torch.set_grad_enabled(False):

                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):

                    inputs, loss_dict, outputs = self._forward_and_loss(model, batch, epoch)

                    if self.dense_site_xy is not None:

                        if not torch.is_tensor(outputs):

                            raise TypeError("dense decoder validation requires tensor mask logits")

                        maps = torch.sigmoid(outputs).detach().cpu().numpy()

                        masks = batch["mask"].detach().cpu().numpy()

                        for radius in RADII:

                            scores_by_radius[radius].append(

                                disk_average_probabilities(maps, self.dense_site_xy, radius)

                            )

                        state_labels.append(states_from_masks(masks, self.dense_site_xy))

            self._accumulate(running, loss_dict, inputs.size(0))


        stats = self._finalize(running, len(data_loader.dataset))

        if state_labels:

            labels = np.concatenate(state_labels, axis=0)

            scores = {

                radius: np.concatenate(chunks, axis=0)

                for radius, chunks in scores_by_radius.items()

            }

            selected = select_decoder_from_scores(scores, labels)

            stats.update(

                ion_acc=selected.accuracy,

                decoder_radius=selected.radius,

                decoder_threshold=selected.threshold,

                decoder_youden_j=selected.youden_j,

            )

        return stats
