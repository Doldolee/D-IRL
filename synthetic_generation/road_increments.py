"""Road progress as incoming displacement, with absolute lateral offsets.

Each complete drive has ds[0] = 0 and ds[t] = s[t] - s[t-1].  Sampling
integrates once across the entire drive, including segment boundaries.  The
initial s is a TRAIN aggregate; no real prefix or future endpoint is needed.
"""
from copy import deepcopy
import numpy as np

from .data import Tuples


class RoadProgressIncrements:
    def __init__(self, road, state):
        self.road, self.state = road, deepcopy(state)
        self.index = road.indices[0]
        if state['version'] != 1:
            raise ValueError('Unsupported road-increment version.')
        self.mean = float(state['increment_mean_m'])
        self.scale = float(state['increment_scale_m'])
        self.start = float(state['initial_progress_m'])
        if not np.isfinite([self.mean, self.scale, self.start]).all() or self.scale <= 0:
            raise ValueError('Invalid road-increment calibration.')

    @classmethod
    def fit(cls, data, train_spans, road):
        steps, starts = [], []
        for a, b in train_spans:
            rows = np.vstack([data.arrays['states'][a:b], data.arrays['next_states'][b-1]])
            s = road.project(road.physical_xy(rows))[:, 0]
            starts.append(s[0]); steps.append(np.diff(s))
        ds = np.concatenate(steps)
        return cls(road, dict(version=1, increment_mean_m=float(ds.mean()),
            increment_scale_m=float(max(ds.std(), 1e-6)),
            initial_progress_m=float(np.median(starts)),
            train_drive_count=len(train_spans), training_step_count=len(ds),
            initial_progress_range_m=[float(min(starts)), float(max(starts))],
            training_increment_range_m=[float(ds.min()), float(ds.max())],
            convention='incoming displacement: ds[0]=0; ds[t]=s[t]-s[t-1]',
            normalization='TRAIN-only mean/std over real transitions; initial zero rows excluded',
            initial_position='fixed median TRAIN initial s; lateral start remains generated',
            integration='float64 cumulative sum over the complete drive, never reset at segments',
            accumulated_progress_clipping=False, endpoint_correction=False,
            absolute_progress_conditioning=False))

    def internal_spec(self, spec, obs_dim):
        result = self.road.internal_spec(spec, obs_dim)
        result['state_names'][self.index] = 'road_progress_increment'
        result['types'].pop('road_progress_s', None)
        result['types']['road_progress_increment'] = 'continuous'
        return result

    def encode_rows(self, rows):
        """Encode one full drive, including its terminal next_state."""
        encoded = self.road.encode_rows(rows)
        if encoded.ndim != 2 or not len(encoded):
            raise ValueError('Expected a nonempty complete trajectory matrix.')
        s = encoded[:, self.index] * self.road.spread[0] + self.road.center[0]
        encoded[:, self.index] = (np.r_[0., np.diff(s)] - self.mean) / self.scale
        return encoded

    def decode_rows(self, rows, start_s=None, initial=True):
        """Restore XY; initial=False continues from the preceding segment's s.

        start_s overrides the aggregate start only for reversibility diagnostics
        or carrying an already generated preceding s, never real rollout history.
        """
        values = np.asarray(rows).astype(np.float64, copy=True)
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError('Expected a finite nonempty trajectory matrix.')
        ds = values[:, self.index] * self.scale + self.mean
        if initial:
            ds[0] = 0.
        s = (self.start if start_s is None else float(start_s)) + np.cumsum(ds, dtype=np.float64)
        values[:, self.index] = (s - self.road.center[0]) / self.road.spread[0]
        return self.road.decode_rows(values)

    def transform_data(self, data, train_spans, validation_spans, tolerance_m=.001):
        arrays = dict(data.arrays)
        arrays['states'] = arrays['states'].copy()
        arrays['next_states'] = arrays['next_states'].copy()
        report = {}
        for split, spans in [('train', train_spans), ('validation', validation_spans)]:
            errors, errors32, starts, steps = [], [], [], []
            for a, b in spans:
                original = np.vstack([data.arrays['states'][a:b], data.arrays['next_states'][b-1]])
                start = self.road.project(self.road.physical_xy(original[:1]))[0, 0]
                encoded = self.encode_rows(original)
                restored = self.decode_rows(encoded, start_s=start)
                restored32 = self.decode_rows(encoded.astype(np.float32), start_s=start).astype(np.float32)
                error = np.linalg.norm(self.road.physical_xy(restored) - self.road.physical_xy(original), axis=-1)
                error32 = np.linalg.norm(self.road.physical_xy(restored32) - self.road.physical_xy(original), axis=-1)
                if not np.isfinite(error).all() or not np.isfinite(error32).all() or max(error.max(), error32.max()) > tolerance_m:
                    raise ValueError(f'Increment round-trip failed ({split}).')
                arrays['states'][a:b] = encoded[:-1]
                arrays['next_states'][a:b] = encoded[1:]
                errors.append(error); errors32.append(error32); starts.append(start)
                steps.append(encoded[1:, self.index] * self.scale + self.mean)
            ds = np.concatenate(steps)
            report[split] = dict(drives=len(spans), points=sum(map(len, errors)),
                roundtrip_max_m=float(np.concatenate(errors).max()),
                float32_roundtrip_max_m=float(np.concatenate(errors32).max()),
                true_start_used_only_for_roundtrip_check=True,
                aggregate_start_max_error_m=float(np.max(np.abs(np.asarray(starts) - self.start))),
                increment_min_m=float(ds.min()), increment_max_m=float(ds.max()),
                increment_mean_m=float(ds.mean()), increment_std_m=float(ds.std()))
        return Tuples(arrays, data.path), report
