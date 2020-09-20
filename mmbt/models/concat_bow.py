#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

from mmbt.models.bow import GloveBowEncoder
from mmbt.models.image import ImageEncoder
from mmbt.models.baseplmodel import BasePLModel

class MultimodalConcatBowClf(BasePLModel):
    def __init__(self, args):
        super(MultimodalConcatBowClf, self).__init__(args)
        self.args = args
        self.clf = nn.Linear(
            args.embed_sz + (args.img_hidden_sz * args.num_image_embeds), args.n_classes
        )
        self.txtenc = GloveBowEncoder(args)
        self.imgenc = ImageEncoder(args)

    def forward(self, txt, img):
        txt = self.txtenc(txt)
        img = self.imgenc(img)
        img = torch.flatten(img, start_dim=1)
        cat = torch.cat([txt, img], -1)
        return self.clf(cat)
