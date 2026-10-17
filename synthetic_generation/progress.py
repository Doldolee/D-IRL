"""Causal segment-anchor conditioning and supervised cumulative displacement.

The per-phase goals are TRAIN aggregates. A segment sees only its preceding
accumulated position, never individual future positions. Source future
increments enter the training objective as labels, not as conditions.
"""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .trajectory import CONDITION_DIM, PHASE_SCENARIOS, phase_runs


def fit_progress_spec(data, train_spans, road, increments):
    endpoints = []
    for a, b in train_spans:
        pattern, durations = phase_runs(data.arrays['scenarios'][a:b])
        if pattern != PHASE_SCENARIOS:
            raise ValueError('Progress goals require complete ordered TRAIN drives.')
        ends = np.cumsum(durations)
        rows = np.vstack([data.arrays['states'][a + ends[:-1]], data.arrays['next_states'][b - 1]])
        endpoints.append(road.project(road.physical_xy(rows.astype(float)))[:, 0])
    goals = np.median(np.stack(endpoints), axis=0)
    if not np.isfinite(goals).all() or not (np.diff(goals) > 0).all():
        raise ValueError('Aggregate phase endpoints must be finite and ordered.')
    return dict(version=1, phase_endpoints_m=goals.tolist(), train_drives=len(train_spans),
                initial_progress_m=increments.state['initial_progress_m'],
                increment_mean_m=increments.mean, increment_scale_m=increments.scale,
                progress_index=increments.index, condition_center_m=float(road.center[0]),
                condition_scale_m=float(road.spread[0]),
                source='TRAIN median projected position at each phase end; not individual future endpoints',
                anchor='position immediately before the target segment; generated-only during rollout',
                conditions=['standardized segment-start progress', 'phase-goal minus segment-start progress'],
                progress_noise_m=0., endpoint_correction=False)


def add_progress_metadata(windows, spans, schema, spec, context_length, segment_length):
    """Append accumulated position as metadata, before the final validity flag.

    Only row k-1 of this metadata can be used to condition a target segment.
    GPU and CPU window gather paths retain exactly the same row order.
    """
    expected = schema.encoded_dim + CONDITION_DIM + 1
    if windows.values.shape[1] != expected:
        raise ValueError('Unexpected base window layout for progress metadata.')
    column = schema.blocks[spec['progress_index']][0]
    metadata = np.zeros((len(windows.values), 1), np.float32)
    offset = 0
    for a, b in spans:
        count = b - a + 1
        start = offset + context_length
        values = windows.values[start:start + count, column].astype(np.float64)
        ds = values * spec['increment_scale_m'] + spec['increment_mean_m']
        ds[0] = 0.
        metadata[start:start + count, 0] = spec['initial_progress_m'] + np.cumsum(ds)
        offset += context_length + count + 2 * segment_length - 1
    if offset != len(windows.values):
        raise ValueError('Progress metadata does not match padded trajectory spans.')
    windows.values = np.column_stack([windows.values[:, :-1], metadata, windows.values[:, -1]])


class ProgressConditioner(nn.Module):
    def __init__(self, spec, codec, loss_scale_m=10.):
        super().__init__()
        self.spec = deepcopy(spec)
        if spec.get('version') != 1:
            raise ValueError('Unsupported progress-condition version.')
        self.column = codec.num_features.index(spec['progress_index'])
        self.start = float(spec['initial_progress_m'])
        self.center = float(spec['condition_center_m'])
        self.scale = float(spec['condition_scale_m'])
        self.ds_mean = float(spec['increment_mean_m'])
        self.ds_scale = float(spec['increment_scale_m'])
        self.loss_scale = float(loss_scale_m)
        values = [self.start, self.center, self.scale, self.ds_mean, self.ds_scale, self.loss_scale]
        goals = np.asarray(spec['phase_endpoints_m'])
        if (not np.isfinite(values).all() or min(self.scale, self.ds_scale, self.loss_scale) <= 0
                or goals.shape != (9,) or not np.isfinite(goals).all() or not (np.diff(goals) > 0).all()):
            raise ValueError('Invalid progress-condition calibration.')
        self.register_buffer('goals', torch.tensor(goals, dtype=torch.float32))

    def training_anchor(self, packed, encoded_dim, context_length):
        if packed.shape[-1] != encoded_dim + CONDITION_DIM + 2:
            raise ValueError('Progress training requires aligned accumulated-position metadata.')
        # Never read target-row or future-row positions for conditioning.
        previous = packed[:, context_length - 1]
        return torch.where(previous[:, -1].bool(), previous[:, encoded_dim + CONDITION_DIM],
                           previous.new_full((len(previous),), self.start))

    def augment(self, conditions, anchor):
        if conditions.shape[-1] != CONDITION_DIM or anchor.shape != (len(conditions),):
            raise ValueError('Expected base plan and one preceding position per segment.')
        goal = self.goals[conditions[..., :9].argmax(-1)]
        current = anchor[:, None].expand_as(goal)
        extra = torch.stack([(current - self.center) / self.scale, (goal - current) / self.scale], -1)
        return torch.cat([conditions, extra], -1)

    def cumulative_loss(self, predicted, clean, target, base_conditions, sigma, context_length):
        k = context_length
        mask = target[:, k:] & (base_conditions[:, k:, 10] > -1.)
        # The first incoming ds is definitionally zero, as in exported rollouts.
        error = (predicted[:, k:, self.column].float() - clean[:, k:, self.column].float()) * self.ds_scale
        accumulated = torch.cumsum(error * mask, dim=1)
        length = accumulated.shape[1]
        selected = torch.zeros_like(mask)
        for horizon in (16, 32, 64, 128):
            if horizon <= length:
                selected[:, horizon - 1] = True
        # Always supervise the last valid row of a short terminal window.
        last = target[:, k:].long().sum(1).sub(1).clamp_min(0)
        selected.scatter_(1, last[:, None], True)
        selected &= target[:, k:]
        confidence = 1. / (1. + sigma[:, :, self.column].float().square())
        values = F.smooth_l1_loss(accumulated / self.loss_scale, torch.zeros_like(accumulated), reduction='none')
        return (values * selected * confidence).sum() / selected.sum().clamp_min(1)
