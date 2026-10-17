"""Input-coordinate numerical channels and mixed-type feature bookkeeping."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class MixedCodec(nn.Module):
    def __init__(self, schema, knots, numerical_transform='input'):
        super().__init__()
        if numerical_transform != 'input':
            raise ValueError('The final model uses input numerical coordinates.')
        self.numerical_transform = numerical_transform
        self.schema = schema
        self.num_features = [i for i, f in enumerate(schema.features) if f['kind'] == 'continuous']
        self.cat_features = [i for i, f in enumerate(schema.features)
                             if f['kind'] == 'categorical' and f['name'] != 'scenario']
        self.categories = [len(schema.features[i]['values']) for i in self.cat_features]
        self.register_buffer('num_columns', torch.tensor([schema.blocks[i][0] for i in self.num_features]))
        self.cat_blocks = [schema.blocks[i] for i in self.cat_features]
        self.register_buffer('knots', torch.tensor(knots, dtype=torch.float32))
        if self.knots.shape[1] != 0:
            raise ValueError('Input-coordinate checkpoints use empty numerical knots.')
        self.register_buffer('scores', torch.empty(0, dtype=torch.float32))
        self.register_buffer('num_is_state', torch.tensor([i < schema.obs_dim for i in self.num_features]))
        self.register_buffer('cat_is_state', torch.tensor([i < schema.obs_dim for i in self.cat_features]))

    @classmethod
    def fit(cls, schema, encoded_training, numerical_transform='input'):
        cols = [a for f, (a, _) in zip(schema.features, schema.blocks) if f['kind'] == 'continuous']
        if not cols:
            raise ValueError('The driving diffusion requires at least one numerical feature.')
        return cls(schema, np.empty((len(cols), 0), dtype=np.float32).tolist(), numerical_transform)

    def split(self, encoded):
        num = encoded[..., self.num_columns]
        cat = torch.stack([encoded[..., a:b].argmax(-1) for a, b in self.cat_blocks], -1)
        return num, cat

    def merge(self, num, cat, conditions):
        result = num.new_zeros(*num.shape[:-1], self.schema.encoded_dim)
        result[..., self.num_columns] = num
        for j, (a, b) in enumerate(self.cat_blocks):
            result[..., a:b] = torch.nn.functional.one_hot(cat[..., j], b - a).to(num.dtype)
        a, b = self.schema.blocks[self.schema.obs_dim + self.schema.act_dim]
        phases = torch.tensor([0, 1, 0, 2, 0, 3, 0, 4, 0], device=num.device)
        levels = self.schema.features[self.schema.obs_dim + self.schema.act_dim]['values']
        if levels != [0., 1., 2., 3., 4.]:
            raise ValueError('The complete driving plan requires scenario levels 0..4.')
        result[..., a:b] = torch.nn.functional.one_hot(phases[conditions[..., :9].argmax(-1)], 5).to(num.dtype)
        return result
