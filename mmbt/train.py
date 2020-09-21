#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from mmbt.models import get_model
from mmbt.utils.utils import set_seed, setup_testube_logger

BASE_PATH = '../data'
ML_PATH = '../ml_data'
OUTPUT_PATH=BASE_PATH+'/mmbt/'
RUN_ID="02_20200920"

def get_args():
    parser = ArgumentParser()
    #lightning
    parser.add_argument('--gpus', default='2', type=str)
    parser.add_argument('--distributed_backend', default=None, type=str)
    parser.add_argument('--precision', default=16, type=int)
    parser.add_argument('--amp_level', default='02', type=str)
    parser.add_argument('--strategy', default='random_search', type=str)
    parser.add_argument('--monitor', default='macro_f1', type=str)
    parser.add_argument('--min_epochs', type=int, default=3)
    parser.add_argument('--metric_mode', default='max', type=str, choices=['auto', 'min', 'max'])
    parser.add_argument('--accumulate_grad_batches', default=2, type=int)
    parser.add_argument('--val_percent_check', default=1.0, type=float)
    parser.add_argument('--save_top_k', default=1, type=int, help='The best k models to be saved.')
    #mmbt
    parser.add_argument("--batch_sz", type=int, default=4)
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased", choices=["bert-base-uncased", "bert-large-uncased"])
    parser.add_argument("--data_path", type=str, default=BASE_PATH)
    parser.add_argument("--dataset_image_name", type=str, default="image", \
                        help="some versions of mmimdb seem to have 'img' as name in jsonl others have 'image'")
    parser.add_argument("--data_has_test_gt", type=bool, default=False)
    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed_sz", type=int, default=300)
    parser.add_argument("--freeze_img", type=int, default=0)
    parser.add_argument("--freeze_txt", type=int, default=0)
    parser.add_argument("--glove_path", type=str, default=f"{ML_PATH}/glove_embeds/glove.840B.300d.txt")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=24)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--model", type=str, default="mmbt", choices=["bow", "img", "bert", "concatbow", "concatbert", "mmbt"])
    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--name", type=str, default="mmimdb_mmbt")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--savedir", type=str, default=OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--task", type=str, default="mmimdb", choices=["mmimdb", "vsnli", "food101"])
    parser.add_argument("--task_type", type=str, default="multilabel", choices=["multilabel", "classification"])
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_classes", type=int, default=1)
    #misc
    parser.add_argument("--run_id", type=str, default=RUN_ID)
    return parser

def main(parser, fast_dev_run) -> None:
    args = parser.parse_args()
    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    model = get_model(args)

    early_stop_callback = EarlyStopping(
        monitor=args.monitor,
        min_delta=0.0,
        patience=args.patience,
        verbose=True,
        mode=args.metric_mode,
    )

    trainer = Trainer(
        logger=setup_testube_logger(args),
        checkpoint_callback=True,
        early_stop_callback=early_stop_callback,
        default_root_dir=args.savedir,
        gpus=args.gpus,
        distributed_backend=args.distributed_backend,
        precision=args.precision,
        amp_level=args.amp_level,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_percent_check=args.val_percent_check,
        fast_dev_run=fast_dev_run,
        num_sanity_val_steps=0
    )

    ckpt_path = os.path.join(
        trainer.default_root_dir,
        trainer.logger.name,
        f"version_{trainer.logger.version}",
        "checkpoints",
    )
    # initialize Model Checkpoint Saver
    checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_path,
        save_top_k=args.save_top_k,
        verbose=True,
        monitor=args.monitor,
        period=1,
        mode=args.metric_mode,
    )
    trainer.checkpoint_callback = checkpoint_callback

    trainer.fit(model)

def run():
    parser=get_args()
    main(parser, fast_dev_run=False)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    run()
