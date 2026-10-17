"""Train and sample the final progress-conditioned temporal TabDiff model."""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json
import time

import numpy as np
import torch

from .data import Tuples, new_directory, split_spans, write_json
from .diffusion import TemporalTabDiff
from .diffusion_codec import MixedCodec
from .execution import WindowBatches
from .schema import Schema
from .train import fingerprint, seed_all, source_group_name
from .trajectory import ConnectedWindows, DurationPlan, complete_spans, CONDITION_DIM


@dataclass
class DiffusionConfig:
    sequence_length: int = 128
    context_length: int = 32
    stride: int = 64
    steps: int = 20000
    batch_size: int = 8
    hidden_dim: int = 32
    heads: int = 4
    layers: int = 2
    learning_rate: float = 2e-4
    lambda_delta: float = .1
    lambda_boundary: float = 1.
    position_representation: str = 'frenet_delta'
    road_spec: dict = field(default_factory=dict)
    progress_conditioning: int = 1
    progress_spec: dict = field(default_factory=dict)
    lambda_progress: float = .1
    progress_loss_scale_m: float = 10.
    context_dropout: float = .1
    context_noise: float = .02
    ema_decay: float = .995
    numerical_transform: str = 'input'
    sampling_steps: int = 32
    second_order: int = 1
    validation_fraction: float = .2
    seed: int = 42
    split_seed: int = 42  # Keep the original data split fixed across model seeds.
    log_every: int = 25
    save_every: int = 500
    device: str = 'cpu'
    threads: int = 2
    data_cache: str = 'auto'
    precision: str = 'float32'
    matmul_precision: str = 'highest'

    def validate(self):
        for name in ('sequence_length', 'context_length', 'stride', 'steps', 'batch_size',
                     'hidden_dim', 'heads', 'layers', 'log_every', 'save_every', 'threads'):
            if getattr(self, name) < 1: raise ValueError(f'{name} must be positive.')
        if self.context_length > self.sequence_length or self.sequence_length < 2:
            raise ValueError('Require 1 <= context_length <= sequence_length, and length >= 2.')
        if self.hidden_dim % self.heads or self.hidden_dim % 2:
            raise ValueError('hidden_dim must be even and divisible by heads.')
        if self.sampling_steps < 2 or self.learning_rate <= 0:
            raise ValueError('Invalid sampling steps or learning rate.')
        if self.numerical_transform != 'input':
            raise ValueError('The final model uses input numerical coordinates.')
        if not 0 <= self.context_dropout < 1 or not 0 <= self.ema_decay < 1:
            raise ValueError('Invalid context dropout or EMA decay.')
        if self.position_representation != 'frenet_delta' or not self.road_spec:
            raise ValueError('The final model requires frenet_delta and road_spec.')
        if self.progress_conditioning != 1:
            raise ValueError('The final model requires progress conditioning.')
        if (not np.isfinite([self.lambda_progress, self.progress_loss_scale_m]).all()
                or self.lambda_progress < 0 or self.progress_loss_scale_m <= 0):
            raise ValueError('Invalid cumulative progress loss settings.')
        if self.lambda_progress and not self.progress_conditioning:
            raise ValueError('Cumulative progress loss requires progress conditioning.')
        if min(self.context_noise, self.lambda_delta, self.lambda_boundary) < 0:
            raise ValueError('Noise and loss weights cannot be negative.')
        if self.second_order not in (0, 1): raise ValueError('second_order must be 0 or 1.')
        if self.device not in ('cpu', 'cuda') or (self.device == 'cuda' and not torch.cuda.is_available()):
            raise ValueError('Select an available CPU or CUDA device.')
        if self.precision not in ('float32', 'bfloat16') or (self.device == 'cpu' and self.precision != 'float32'):
            raise ValueError('Use float32, or bfloat16 on CUDA.')
        if self.data_cache not in ('auto', 'cpu', 'cuda') or (self.device == 'cpu' and self.data_cache == 'cuda'):
            raise ValueError('Invalid data cache for this device.')
        if self.matmul_precision not in ('highest', 'high', 'medium'):
            raise ValueError('Invalid matmul precision.')


def autocast_for(model):
    return (torch.autocast('cuda', dtype=torch.bfloat16)
            if next(model.parameters()).is_cuda and model.config['precision'] == 'bfloat16' else nullcontext())


def fit_diffusion(data_dir, output, config, spec=None, split_groups=None,
                  source_group=None, input_manifest=None, coordinate_state=None):
    started = time.perf_counter()
    config.validate()
    source_group_name(source_group)
    manifest = None
    manifest_sha = None
    if input_manifest is not None:
        import hashlib
        payload = Path(input_manifest).read_bytes()
        manifest_sha = hashlib.sha256(payload).hexdigest()
        manifest = json.loads(payload)
        declared = manifest.get('source_group', manifest.get('source_cohort'))
        if declared is not None and declared != source_group:
            raise ValueError('Input manifest and requested source_group disagree.')
    output = new_directory(output)
    seed_all(config.seed, config.threads)
    data = Tuples.load(data_dir)
    original_train, original_val = split_spans(data, config.validation_fraction, config.split_seed, split_groups)
    train_spans, durations, excluded_train = complete_spans(data, original_train)
    val_spans, _, excluded_val = complete_spans(data, original_val)
    if not val_spans: raise ValueError('No complete held-out trajectory.')
    planner = DurationPlan.fit(durations)
    source_fingerprint = fingerprint(data)
    road = None
    road_extra = {}
    from .road_coordinates import RoadCoordinates
    road = (RoadCoordinates.fit(data, train_spans, config.road_spec)
            if coordinate_state is None else RoadCoordinates(coordinate_state['road_coordinates']))
    original_raw = data.matrix()
    original_terminal = np.stack([original_raw[b - 1].copy() for _, b in train_spans])
    original_terminal[:, :data.obs_dim] = np.stack([data.arrays['next_states'][b - 1] for _, b in train_spans])
    export_schema = Schema.fit(np.concatenate([*[original_raw[a:b] for a, b in train_spans], original_terminal]),
                               data.obs_dim, data.act_dim, spec)
    del original_raw, original_terminal
    transform = road
    from .road_increments import RoadProgressIncrements
    transform = (RoadProgressIncrements.fit(data, train_spans, road)
                 if coordinate_state is None else RoadProgressIncrements(road, coordinate_state['road_increments']))
    from .progress import fit_progress_spec
    config.progress_spec = (fit_progress_spec(data, train_spans, road, transform)
                            if coordinate_state is None else deepcopy(coordinate_state['config']['progress_spec']))
    write_json(output / 'progress_spec.json', config.progress_spec)
    data, diagnostics = transform.transform_data(data, train_spans, val_spans)
    road_extra = {'road_coordinates': road.state, 'export_schema': export_schema.to_dict(),
                  'road_roundtrip': diagnostics}
    write_json(output / 'road_coordinates.json', road.state)
    write_json(output / 'road_roundtrip.json', diagnostics)
    road_extra['road_increments'] = transform.state
    write_json(output / 'road_increments.json', transform.state)
    spec = transform.internal_spec(spec, data.obs_dim)
    print(json.dumps({'road_roundtrip': diagnostics}), flush=True)
    raw = data.matrix()
    terminal = np.stack([raw[b - 1].copy() for _, b in train_spans])
    terminal[:, :data.obs_dim] = np.stack([data.arrays['next_states'][b - 1] for _, b in train_spans])
    training_rows = np.concatenate([*[raw[a:b] for a, b in train_spans], terminal])
    schema = Schema.fit(training_rows, data.obs_dim, data.act_dim, spec)
    packing = 'input'
    codec = MixedCodec.fit(schema, schema.encode(training_rows, continuous_transform=packing),
                           numerical_transform=config.numerical_transform)
    del training_rows
    windows = ConnectedWindows(data, raw, schema, train_spans, planner, config.sequence_length,
                               config.context_length, config.stride, continuous_transform=packing)
    # Predict each target segment jointly on the trajectory start grid.
    windows.length = config.context_length + config.sequence_length
    validation = ConnectedWindows(data, raw, schema, val_spans, planner, config.sequence_length,
                                  config.context_length, config.stride, continuous_transform=packing)
    del raw
    from .progress import add_progress_metadata
    for dataset, spans in [(windows, train_spans), (validation, val_spans)]:
        add_progress_metadata(dataset, spans, schema, config.progress_spec,
                              config.context_length, config.sequence_length)
    batches = WindowBatches(windows, config)
    model = TemporalTabDiff(schema, codec.knots.tolist(), asdict(config)).to(config.device)
    model.train()
    ema = deepcopy(model).eval().requires_grad_(False)
    options = {'lr': config.learning_rate, 'weight_decay': 0.}
    if config.device == 'cuda': options['fused'] = True
    optimizer = torch.optim.AdamW(model.parameters(), **options)
    provenance = {
        **road_extra,
        'source_group': source_group, 'source_cohort': source_group,
        'input_manifest': manifest, 'input_manifest_sha256': manifest_sha,
        'active_package': 'synthetic_generation',
        'coordinate_statistics': ('fitted on supplied training trajectories' if coordinate_state is None
                                  else 'frozen aggregate road/increment/progress metadata from bundled cohort generator'),
        'format_version': 3, 'model_type': 'temporal_tabdiff', 'unit': 'complete_trajectory',
        'config': asdict(config), 'schema': schema.to_dict(), 'quantile_knots': codec.knots.tolist(),
        'duration_plan': asdict(planner), 'input_fingerprint': source_fingerprint, 'input_summary': data.summary(),
        'train_spans': train_spans, 'validation_spans': val_spans,
        'original_train_spans': original_train, 'original_validation_spans': original_val,
        'excluded_train_patterns': excluded_train, 'excluded_validation_patterns': excluded_val,
        'train_windows': len(windows), 'validation_windows': len(validation),
        'split_unit': 'participant_group' if split_groups is not None else 'trajectory',
        'generator_parameters': sum(p.numel() for p in model.parameters()),
        'reference': 'https://openreview.net/forum?id=swvURjrt8z',
        'upstream_commit': '5ecdb3356261aea72716cc9a779f31d7ad083bf4',
        'implementation': 'Temporal TabDiff: axial attention, clean prefix, road increments, cumulative progress loss and EMA',
        'numerical_coordinates': 'supplied tuple units; schema affine packing is bypassed',
        'numerical_loss_coordinates': config.numerical_transform,
        'export_range': 'continuous values clipped to fitted training bounds by the shared schema',
        'continuation': 'previous exported rows condition each generated segment; synthetic-only start',
        'end_rule': 'one terminal after the sampled final road phase; not a learned done head',
        'checkpoint_use': 'EMA inference weights; exact training resume is not implemented',
        'cuda_data_cache': batches.cached, 'torch_version': str(torch.__version__),
        'artificial_fixture': (Path(data_dir) / 'toy_manifest.json').is_file(),
    }
    if road is not None:
        provenance.update(numerical_coordinates='train-standardized road s,d; other supplied tuple units unchanged',
                          numerical_loss_coordinates='mixed: s,d standardized with training mean/std; all other original coordinates',
                          continuation='previous clipped internal s,d rows; synthetic-only start; final export restores XY',
                          export_range='internal training bounds including s,d; no additional XY clipping or road snapping')
    provenance.update(numerical_coordinates='train-standardized incoming ds and absolute d; other original tuple units',
                      numerical_loss_coordinates='mixed: ds,d standardized; derivative loss now penalizes differences of ds',
                      continuation='previous generated ds,d rows; full-drive integration carries s across segments',
                      initial_progress='fixed TRAIN median start; first incoming ds forced to zero before feedback',
                      absolute_progress_conditioning='existing phase/time progress only; accumulated s is not an extra model input',
                      export_range='training ds,d bounds; integrated s is neither clipped nor endpoint-corrected')
    provenance.update(progress_conditioning=config.progress_spec,
                      absolute_progress_conditioning='one preceding accumulated position per segment; no real future positions',
                      cumulative_progress_loss='SmoothL1 of cumulative displacement error at 16/32/64/128 steps and last valid row; low-noise confidence; first ds=0',
                      condition_dimensions=CONDITION_DIM + 2)
    write_json(output / 'run.json', provenance)
    write_json(output / 'schema.json', schema.to_dict())
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(config.matmul_precision)
    elapsed = []
    try:
        if config.device == 'cuda': torch.cuda.synchronize()
        training_started = time.perf_counter()
        with (output / 'training.jsonl').open('w') as log:
            for step in range(1, config.steps + 1):
                step_started = time.perf_counter()
                packed = batches.next()
                optimizer.zero_grad(set_to_none=True)
                with autocast_for(model): losses = model.loss(packed)
                if not torch.isfinite(losses['loss']): raise FloatingPointError('Nonfinite diffusion loss.')
                losses['loss'].backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                with torch.no_grad():
                    # Warm-started EMA avoids averaging almost entirely untrained weights in short trials.
                    decay = min(config.ema_decay, (1 + step) / (10 + step))
                    for target, source in zip(ema.parameters(), model.parameters()):
                        target.lerp_(source, 1 - decay)
                metrics = torch.stack([*[v.detach().float() for v in losses.values()], norm.float()]).cpu().tolist()
                elapsed.append(time.perf_counter() - step_started)
                record = {'step': step, **dict(zip([*losses, 'gradient_norm_before_clip'], metrics)),
                          'step_seconds': elapsed[-1], 'training_elapsed_seconds': time.perf_counter() - training_started}
                log.write(json.dumps(record, allow_nan=False) + '\n'); log.flush()
                if step == 1 or step % config.log_every == 0 or step == config.steps:
                    print(json.dumps(record), flush=True)
                if step % config.save_every == 0 or step == config.steps:
                    if not all(torch.isfinite(v).all() for v in ema.state_dict().values()):
                        raise FloatingPointError('Nonfinite EMA checkpoint.')
                    state = {**provenance, 'step': step,
                             'generator': {k: v.detach().cpu() for k, v in ema.state_dict().items()}}
                    temporary = output / 'checkpoint.tmp'
                    torch.save(state, temporary); temporary.replace(output / 'checkpoint.pt')
        finished = time.perf_counter()
    finally:
        torch.set_float32_matmul_precision(previous_precision)
    write_json(output / 'timing.json', {
        'preparation_seconds': training_started - started,
        'training_seconds_including_logging_and_checkpoints': finished - training_started,
        'total_seconds': finished - started, 'completed_steps': config.steps,
        'mean_step_seconds': float(np.mean(elapsed)),
        'mean_step_seconds_excluding_first': float(np.mean(elapsed[1:])) if len(elapsed) > 1 else None,
        'real_batches_consumed': config.steps, 'batches_per_data_epoch': len(batches),
        'data_epochs': config.steps / len(batches), 'device': config.device,
        'precision': config.precision, 'cuda_data_cache': batches.cached,
        'max_cuda_memory_allocated': torch.cuda.max_memory_allocated() if config.device == 'cuda' else None,
    })
    return output / 'checkpoint.pt'


@torch.no_grad()
def draw_diffusion_trajectories(model, state, count, seed, batch_size=8, categorical_mode='sample', sampling_steps=None):
    if not 1 <= count <= 32767 or batch_size < 1: raise ValueError('Invalid count or batch size.')
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    planner = DurationPlan(**state['duration_plan'])
    plans = planner.sample(count, np.random.default_rng(seed))
    c, d = state['config'], model.schema.encoded_dim
    packing = 'input'
    length, k = c['sequence_length'], c['context_length']
    result = []
    for start in range(0, count, batch_size):
        selected = plans[start:start + batch_size]
        sizes = selected.sum(1) + 1
        total = int(np.ceil(sizes.max() / length)) * length
        conditions = np.zeros((len(selected), total, CONDITION_DIM), np.float32)
        valid = np.zeros((len(selected), total, 1), np.float32)
        for i, (plan, size) in enumerate(zip(selected, sizes)):
            conditions[i, :size] = planner.conditions(plan); valid[i, :size] = 1
        conditions = torch.from_numpy(conditions).to(device); valid = torch.from_numpy(valid).to(device)
        context = torch.zeros(len(selected), k, d + CONDITION_DIM + 1, device=device)
        chunks = []
        progress = np.full(len(selected), c['progress_spec']['initial_progress_m'], np.float64) if c.get('progress_conditioning') else None
        for pos in range(0, total, length):
            phase = conditions[:, pos:pos + length]; mask = valid[:, pos:pos + length]
            with autocast_for(model):
                extra = {} if progress is None else {'progress_anchor': torch.as_tensor(progress, device=device, dtype=torch.float32)}
                encoded = model.sample_segment(context, phase, mask, rng, sampling_steps, categorical_mode, **extra)
            decoded = model.schema.decode(encoded.float().cpu().numpy(), np.random.default_rng(0), 'argmax',
                                          continuous_transform=packing).astype(np.float32)
            if state.get('road_increments') is not None and pos == 0:
                inc = state['road_increments']
                decoded[:, 0, state['road_coordinates']['indices'][0]] = -inc['increment_mean_m'] / inc['increment_scale_m']
            # Categories were sampled by the reverse chain, not by Schema.decode.
            if not np.isfinite(decoded).all(): raise FloatingPointError('Nonfinite decoded trajectory.')
            if progress is not None:
                inc = state['road_increments']
                ds = decoded[..., state['road_coordinates']['indices'][0]].astype(np.float64) * inc['increment_scale_m'] + inc['increment_mean_m']
                if pos == 0: ds[:, 0] = 0.
                progress += (ds * mask[..., 0].cpu().numpy()).sum(1)
            chunks.append(decoded)
            feedback = torch.from_numpy(model.schema.encode(decoded, strict=True, continuous_transform=packing)).to(device) * mask
            context = torch.cat([context, torch.cat([feedback, phase, mask], -1)], 1)[:, -k:]
        rows = np.concatenate(chunks, axis=1)
        result.extend(rows[i, :int(size)].copy() for i, size in enumerate(sizes))
    from .road_coordinates import RoadCoordinates
    road = RoadCoordinates(state['road_coordinates'])
    transform = road
    from .road_increments import RoadProgressIncrements
    transform = RoadProgressIncrements(road, state['road_increments'])
    result = [transform.decode_rows(rows).astype(np.float32) for rows in result]
    return result, plans
