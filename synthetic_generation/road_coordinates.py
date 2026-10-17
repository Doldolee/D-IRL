"""Training-only shared reference curve and reversible road-relative positions.

The reference is an aggregate of driving paths, not a surveyed road centerline.
Only XY is reparameterized; lateral offsets and backward motion are retained.
No individual history, source trajectory, or projection is used at sampling.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from .trajectory import PHASE_SCENARIOS, phase_runs


class RoadCoordinates:
    def __init__(self, state):
        self.state = deepcopy(state)
        if state['version'] != 1:
            raise ValueError('Unsupported road-coordinate version.')
        self.indices = list(state['indices'])
        self.scale = np.asarray(state['xy_scale'], dtype=float)
        self.mean = np.asarray(state['xy_mean'], dtype=float)
        self.units = float(state['position_units_per_meter'])
        self.knots = np.asarray(state['curve_s_m'], dtype=float)
        self.curve = CubicSpline(self.knots, np.asarray(state['curve_xy_m']), axis=0)
        self.length = float(self.knots[-1])
        self.center = np.asarray(state.get('sd_mean', [0., 0.]))
        self.spread = np.asarray(state.get('sd_scale', [1., 1.]))
        if (len(self.indices) != 2 or len(set(self.indices)) != 2 or
                self.units <= 0 or (self.scale <= 0).any() or (self.spread <= 0).any()):
            raise ValueError('Invalid position calibration.')
        self._grid = np.linspace(0, self.length, max(33, int(np.ceil(self.length / .5)) + 1))
        self._tree = cKDTree(self.curve(self._grid))

    def physical_xy(self, rows):
        return (np.asarray(rows)[..., self.indices] * self.scale + self.mean) / self.units

    def geometry(self, s, derivative=0):
        s = np.asarray(s, dtype=float)
        clipped = np.clip(s, 0, self.length)
        value = self.curve(clipped, derivative)
        outside = s != clipped
        tangent = self.curve(clipped, 1)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-12)
        if derivative == 0:
            value += (s - clipped)[..., None] * tangent
        elif derivative == 1:
            value = np.where(outside[..., None], tangent, value)
        elif derivative == 2:
            value = np.where(outside[..., None], 0., value)
        return value

    def to_xy(self, sd):
        sd = np.asarray(sd, dtype=float)
        s, d = sd[..., 0], sd[..., 1]
        tangent = self.geometry(s, 1)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-12)
        normal = np.stack([-tangent[..., 1], tangent[..., 0]], axis=-1)
        return self.geometry(s) + d[..., None] * normal

    def project(self, xy):
        """Orthogonal projection onto a smooth curve with straight end extensions.

        Projection is preprocessing/evaluation only. A strict round-trip gate in
        training rejects unsuitable reference curves rather than altering data.
        """
        xy = np.asarray(xy, dtype=float)
        shape = xy.shape
        if shape[-1] != 2 or not np.isfinite(xy).all():
            raise ValueError('Expected finite XY points.')
        pieces = []
        points = xy.reshape(-1, 2)
        # Bounded chunks keep preprocessing memory independent of file size.
        for start in range(0, len(points), 65536):
            p = points[start:start + 65536]
            s = self._grid[self._tree.query(p, workers=1)[1]]
            for _ in range(16):
                residual = self.geometry(s) - p
                tangent = self.geometry(s, 1)
                denominator = (tangent * tangent).sum(-1) + (residual * self.geometry(s, 2)).sum(-1)
                safe = np.where(np.abs(denominator) < .05, .05, denominator)
                step = np.clip((residual * tangent).sum(-1) / safe, -20., 20.)
                s -= step
                if not len(step) or np.max(np.abs(step)) < 1e-10:
                    break
            tangent = self.geometry(s, 1)
            tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-12)
            normal = np.stack([-tangent[:, 1], tangent[:, 0]], -1)
            d = ((p - self.geometry(s)) * normal).sum(-1)
            pieces.append(np.column_stack([s, d]))
        return np.concatenate(pieces).reshape(shape)

    def encode_rows(self, rows):
        result = np.asarray(rows).astype(float, copy=True)
        sd = self.project(self.physical_xy(rows))
        result[..., self.indices] = (sd - self.center) / self.spread
        return result

    def decode_rows(self, rows):
        result = np.asarray(rows).astype(float, copy=True)
        sd = result[..., self.indices] * self.spread + self.center
        result[..., self.indices] = (self.to_xy(sd) * self.units - self.mean) / self.scale
        return result

    def internal_spec(self, spec, obs_dim):
        result = deepcopy(spec or {})
        names = result.get('state_names', [f'state_{i:02d}' for i in range(obs_dim)])
        types = result.setdefault('types', {})
        for i, name in zip(self.indices, ('road_progress_s', 'road_lateral_d')):
            types.pop(names[i], None)
            names[i] = name
            types[name] = 'continuous'
        result['state_names'] = names
        return result

    @classmethod
    def fit(cls, data, train_spans, spec):
        required = ('indices', 'xy_scale', 'xy_mean', 'position_units_per_meter')
        if any(k not in spec for k in required):
            raise ValueError('road_spec requires indices, xy_scale, xy_mean, position_units_per_meter.')
        indices = list(spec['indices'])
        if len(indices) != 2 or len(set(indices)) != 2 or min(indices) < 0 or max(indices) >= data.obs_dim:
            raise ValueError('road_spec indices must identify two different state columns.')
        if len(train_spans) < 2:
            raise ValueError('At least two complete training drives are required.')
        scale, mean, units = np.asarray(spec['xy_scale']), np.asarray(spec['xy_mean']), float(spec['position_units_per_meter'])
        samples = int(spec.get('samples_per_phase', 49))
        if (samples < 5 or not np.isfinite(units) or units <= 0 or scale.shape != (2,) or mean.shape != (2,)
                or not np.isfinite(scale).all() or not np.isfinite(mean).all() or (scale <= 0).any()):
            raise ValueError('Invalid road fitting specification.')
        paths = []
        for a, b in train_spans:
            pattern, durations = phase_runs(data.arrays['scenarios'][a:b])
            if pattern != PHASE_SCENARIOS:
                raise ValueError('Road fitting requires complete phase-ordered drives.')
            xy = (np.vstack([data.arrays['states'][a:b, indices],
                             data.arrays['next_states'][b - 1, indices]]) * scale + mean) / units
            pieces = []
            for j, (start, end) in enumerate(zip(np.r_[0, np.cumsum(durations)[:-1]], np.cumsum(durations))):
                part = xy[start:end + 1]
                # Reduce stationary coordinate noise only while fitting geometry.
                # All source rows, including stops and deviations, remain targets.
                keep = [0]
                for k in range(1, len(part) - 1):
                    if np.linalg.norm(part[k] - part[keep[-1]]) >= .5:
                        keep.append(k)
                keep.append(len(part) - 1)
                part = part[keep]
                arc = np.r_[0., np.linalg.norm(np.diff(part, axis=0), axis=1).cumsum()]
                unique = np.r_[True, np.diff(arc) > 1e-8]
                arc, part = arc[unique], part[unique]
                if arc[-1] < 1e-5:
                    raise ValueError('A phase has no measurable geometric extent.')
                grid = np.linspace(0., arc[-1], samples)
                sampled = np.column_stack([np.interp(grid, arc, part[:, k]) for k in (0, 1)])
                pieces.append(sampled if j == 8 else sampled[:-1])
            paths.append(np.concatenate(pieces))
        centerline = np.median(np.stack(paths), axis=0)
        chord = np.r_[0., np.linalg.norm(np.diff(centerline, axis=0), axis=1).cumsum()]
        keep = np.r_[True, np.diff(chord) > 1e-5]
        curve = CubicSpline(chord[keep], centerline[keep], axis=0)
        fine_u = np.linspace(0., chord[-1], max(257, int(np.ceil(chord[-1] / .1)) + 1))
        fine_xy = curve(fine_u)
        arc = np.r_[0., np.linalg.norm(np.diff(fine_xy, axis=0), axis=1).cumsum()]
        grid = np.linspace(0., arc[-1], max(65, int(np.ceil(arc[-1])) + 1))
        xy = curve(np.interp(grid, arc, fine_u))
        state = dict(version=1, indices=indices, xy_scale=scale.tolist(), xy_mean=mean.tolist(),
                     position_units_per_meter=units, curve_s_m=grid.tolist(), curve_xy_m=xy.tolist(),
                     fitting='coordinatewise median of training paths aligned by phase and within-phase arc fraction',
                     train_drive_count=len(train_spans), samples_per_phase=samples,
                     source='training-only aggregate reference path; not a surveyed lane centerline',
                     lateral_sign='positive to the left of increasing reference arc length')
        road = cls(state)
        sd = np.concatenate([road.project(road.physical_xy(np.vstack([
            data.arrays['states'][a:b], data.arrays['next_states'][b - 1]]))) for a, b in train_spans])
        state.update(sd_mean=sd.mean(0).tolist(), sd_scale=np.maximum(sd.std(0), 1e-6).tolist(),
                     normalization='train-only affine standardization of newly defined s,d; other features unchanged')
        return cls(state)

    def transform_data(self, data, train_spans, validation_spans, tolerance_m=.001):
        """Transform selected drives without mutating source arrays; verify every row."""
        from .data import Tuples
        arrays = dict(data.arrays)
        arrays['states'] = arrays['states'].copy()
        arrays['next_states'] = arrays['next_states'].copy()
        report = {}
        for split, spans in [('train', train_spans), ('validation', validation_spans)]:
            errors, float32_errors, offsets, steps = [], [], [], []
            for a, b in spans:
                original = np.vstack([data.arrays['states'][a:b], data.arrays['next_states'][b - 1]])
                encoded = self.encode_rows(original)
                restored = self.decode_rows(encoded)
                error = np.linalg.norm(self.physical_xy(restored) - self.physical_xy(original), axis=-1)
                if not np.isfinite(error).all() or error.max() > tolerance_m:
                    raise ValueError(f'Road-coordinate round-trip failed ({split}, max {error.max():.6g} m).')
                arrays['states'][a:b] = encoded[:-1]
                arrays['next_states'][a:b] = encoded[1:]
                errors.append(error)
                restored32 = self.decode_rows(encoded.astype(np.float32)).astype(np.float32)
                error32 = np.linalg.norm(self.physical_xy(restored32) - self.physical_xy(original), axis=-1)
                if not np.isfinite(error32).all() or error32.max() > tolerance_m:
                    raise ValueError(f'Float32 road-coordinate round-trip failed ({split}).')
                float32_errors.append(error32)
                sd = encoded[:, self.indices] * self.spread + self.center
                offsets.append(sd[:, 1]); steps.append(np.diff(sd[:, 0]))
            error, d, ds = map(np.concatenate, (errors, offsets, steps))
            report[split] = dict(drives=len(spans), points=len(error), roundtrip_max_m=float(error.max()),
                                 roundtrip_mean_m=float(error.mean()),
                                 float32_roundtrip_max_m=float(np.concatenate(float32_errors).max()),
                                 lateral_std_m=float(d.std()),
                                 lateral_range_m=[float(d.min()), float(d.max())],
                                 progress_step_min_m=float(ds.min()), progress_step_max_m=float(ds.max()),
                                 backward_over_1m_fraction=float((ds < -1).mean()))
        return Tuples(arrays, data.path), report
