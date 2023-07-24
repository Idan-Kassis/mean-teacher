# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import sys
import math
import itertools

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable, Function

from .utils import export, parameter_count

from transformers import AutoFeatureExtractor, SwinForImageClassification

@export
def swin(pretrained=False, **kwargs):
    model = SwinTransformer()
    return model



class SwinTransformer(nn.Module):
    def __init__(self):
        super(SwinTransformer, self).__init__()
        # Load the feature extractor
        model_name_or_path = "microsoft/swin-base-patch4-window12-384-in22k"
        labels = [0, 1]

        # Load the SwinForImageClassification model
        self.backbone = SwinForImageClassification.from_pretrained(
            model_name_or_path,
            ignore_mismatched_sizes=True,
            num_labels=len(labels),
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)}
        )

    def forward(self, inputs):
        out = self.backbone(inputs)
        logits = out.logits
        return logits

