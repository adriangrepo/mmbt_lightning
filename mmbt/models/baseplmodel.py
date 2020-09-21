import numpy as np
import time
from collections import OrderedDict
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader, RandomSampler
import torch.optim as optim
import torch
from torch import nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.metrics.functional import accuracy
from pytorch_transformers.optimization import (
    AdamW,
    WarmupConstantSchedule,
    WarmupLinearSchedule,
)
from pytorch_pretrained_bert import BertAdam

from mmbt.data.helpers import get_data_loaders
from mmbt.utils.utils import store_preds_to_disk

class EpochTrackingCallback(Callback):
    def __init__(self):
        super(EpochTrackingCallback, self).__init__()
        self.val_batch = 0
        self.train_batch = 0

    def on_validation_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self.val_batch += 1

    def on_batch_end(self, trainer, pl_module):
        self.train_batch += 1


class BasePLModel(pl.LightningModule):
    def __init__(self, args):
        super(BasePLModel, self).__init__()
        self.args=args
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(args)
        self.__build_loss()
        self.train_batch_times=[]
        self.val_batch_times=[]

    def __build_loss(self):
        if self.args.task_type == "multilabel":
            if self.args.weight_classes:
                freqs = [self.args.label_freqs[l] for l in self.args.labels]
                label_weights = (torch.FloatTensor(freqs) / self.args.train_data_len) ** -1
                criterion = nn.BCEWithLogitsLoss(pos_weight=label_weights)
            else:
                criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        self.loss= criterion

    def run_fwd(self, batch):
        txt, segment, mask, img, tgt = batch

        freeze_img = self.current_epoch < self.args.freeze_img
        freeze_txt = self.current_epoch < self.args.freeze_txt

        if self.args.model == "bow":
            out = self(txt)
        elif self.args.model == "img":
            out = self(img)
        elif self.args.model in ["concatbow"]:
            out = self(txt, img)
        elif self.args.model == "bert":
            out = self(txt, mask, segment)
        elif self.args.model == "concatbert":
            out = self(txt, mask, segment, img)
        else:
            assert self.args.model == "mmbt"
            for param in self.enc.img_encoder.parameters():
                param.requires_grad = not freeze_img
            for param in self.enc.encoder.parameters():
                param.requires_grad = not freeze_txt

            out = self(txt, mask, segment, img)

        loss = self.loss(out, tgt)
        return loss, out, tgt

    def training_step(self, batch, batch_idx):
        batch_start = time.time()
        loss, out, tgt = self.run_fwd(batch)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        tqdm_dict = {"train_loss": loss}
        output = OrderedDict(
            {"loss": loss, "out": out, "tgt":tgt, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )
        self.train_batch_times.append(time.time() - batch_start)
        return output

    def training_epoch_end(self, output):
        print(f'bs: {self.args.batch_sz}, batches: {len(self.train_batch_times)}, average batch train time: {sum(self.train_batch_times) / len(self.train_batch_times)}')
        self.train_batch_times = []

    def validation_step(self, batch, batch_idx, store_preds=False):
        batch_start = time.time()
        loss_val, out, tgt = self.run_fwd(batch)
        if self.args.task_type == "multilabel":
            pred = torch.sigmoid(out) > 0.5
        else:
            pred_soft = torch.nn.functional.softmax(out, dim=1)
            pred = pred_soft.argmax(dim=1)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        tqdm_dict = {"val_loss": loss_val}
        output = OrderedDict({"loss": loss_val, 'pred':pred, 'target': tgt, 'log': tqdm_dict})
        self.val_batch_times.append(time.time() - batch_start)
        return output

    def validation_epoch_end(self, outputs, output_gates=False):
        val_losses = torch.stack([x["loss"] for x in outputs])
        val_loss_mean = val_losses.mean()
        metrics = {"val_loss_mean": val_loss_mean}
        preds = [x["pred"].cpu().detach().numpy() for x in outputs]
        tgts = [x["target"].cpu().detach().numpy() for x in outputs]

        if self.args.task_type == "multilabel":
            tgts = np.vstack(tgts)
            preds = np.vstack(preds)
            metrics["macro_f1"] = f1_score(tgts, preds, average="macro")
            metrics["micro_f1"] = f1_score(tgts, preds, average="micro")
            ret = OrderedDict({"val_loss": val_loss_mean, "macro_f1": metrics["macro_f1"], "micro_f1": metrics["micro_f1"], 'log': metrics})
        else:
            tgts = [l for sl in tgts for l in sl]
            preds = [l for sl in preds for l in sl]
            metrics["val_acc"] = accuracy_score(tgts, preds)
            ret = OrderedDict({"val_loss": val_loss_mean, "val_acc": metrics["val_acc"], 'log': metrics})
        print(f'Average val batch eval time: {sum(self.val_batch_times) / len(self.val_batch_times)}')
        self.val_batch_times = []
        return ret

    def test_step(self, batch, batch_idx, args):
        loss_val, out, tgt = self.run_fwd(batch)
        if self.args.task_type == "multilabel":
            pred = torch.sigmoid(out) > 0.5
        else:
            pred_soft = torch.nn.functional.softmax(out, dim=1)
            pred = pred_soft.argmax(dim=1)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        tqdm_dict = {"val_loss": loss_val}
        output = OrderedDict({"loss": loss_val, 'pred': pred, 'target': tgt, 'log': tqdm_dict})
        return output

    def test_epoch_end(self, outputs, store_preds=True):
        test_losses = torch.stack([x["loss"] for x in outputs])
        test_loss_mean = test_losses.mean()
        metrics = {"test_loss_mean": test_loss_mean}
        preds = [x["pred"].cpu().detach().numpy() for x in outputs]
        tgts = [x["target"].cpu().detach().numpy() for x in outputs]

        if self.args.task_type == "multilabel":
            tgts = np.vstack(tgts)
            preds = np.vstack(preds)
            metrics["macro_f1"] = f1_score(tgts, preds, average="macro")
            metrics["micro_f1"] = f1_score(tgts, preds, average="micro")
        else:
            tgts = [l for sl in tgts for l in sl]
            preds = [l for sl in preds for l in sl]
            metrics["test_acc"] = accuracy_score(tgts, preds)

        ret = OrderedDict({"val_loss": test_loss_mean, "test_acc": metrics["test_acc"], 'log': metrics})

        if store_preds:
            store_preds_to_disk(tgts, preds, self.args)
        return ret

    def _get_scheduler(optimizer, args):
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
        )

    def configure_optimizers(self):
        if self.args.model in ["bert", "concatbert", "mmbt"]:
            total_steps = (
                    self.args.train_data_len
                    / self.args.batch_sz
                    / self.args.gradient_accumulation_steps
                    * self.args.max_epochs
            )
            param_optimizer = list(self.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 "weight_decay": 0.01},
                {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0, },
            ]
            optimizer = BertAdam(
                optimizer_grouped_parameters,
                lr=self.args.lr,
                warmup=self.args.warmup,
                t_total=total_steps,
            )
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.args.lr)

        return optimizer

    def forward(self, batch):
        #abstract method
        pass

    def train_dataloader(self) -> DataLoader:
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        return self.val_loader

    def test_dataloader(self) -> DataLoader:
        return self.test_loader