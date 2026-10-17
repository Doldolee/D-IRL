"""Temporal extension of TabDiff's mixed continuous-time diffusion.

Feature attention and temporal attention share numerical/categorical tokens.
Previous exported rows are clean conditioning, never resampled during rollout.
This is an adaptation, not a reproduction of the original tabular architecture.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .diffusion_codec import MixedCodec
from .third_party.tabdiff.noise_schedule import PowerMeanNoise_PerColumn, LogLinearNoise_PerColumn
from .trajectory import CONDITION_DIM


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width),
                                nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, x, valid=None):
        b, length, width = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, length, 3, self.heads, width // self.heads).unbind(2)
        mask = None
        if valid is not None:
            # Some attention backends produce NaNs when every key is padding.
            # Attend to one dummy key in that case, then zero all padded outputs.
            keys = valid.clone()
            keys[:, 0] |= ~valid.any(-1)
            mask = keys[:, None, None, :]
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                          v.transpose(1, 2), attn_mask=mask)
        x = x + self.out(y.transpose(1, 2).reshape(b, length, width))
        x = x + self.ff(x)
        return x if valid is None else x * valid[..., None]


class TemporalDenoiser(nn.Module):
    def __init__(self, numerical, categories, width=32, heads=4, layers=2, condition_dim=CONDITION_DIM):
        super().__init__()
        self.numerical, self.categories = numerical, categories
        self.width = width
        self.features = numerical + len(categories)
        self.num_weight = nn.Parameter(torch.randn(numerical, width) / math.sqrt(width))
        self.feature_bias = nn.Parameter(torch.randn(self.features, width) * .02)
        offsets = [0]
        for size in categories: offsets.append(offsets[-1] + size + 1)
        self.register_buffer('offsets', torch.tensor(offsets[:-1]))
        self.register_buffer('mask_ids', torch.tensor(categories))
        self.embedding = nn.Embedding(offsets[-1], width)
        self.condition = nn.Linear(condition_dim + 2, width)  # known-row flag and diffusion time
        self.num_noise = nn.Linear(1, width)
        self.cat_noise = nn.Linear(1, width)
        self.feature_attention = nn.ModuleList([Attention(width, heads) for _ in range(layers)])
        self.temporal_attention = nn.ModuleList([Attention(width, heads) for _ in range(layers)])
        self.num_out = nn.Parameter(torch.randn(numerical, width) * .02)
        self.num_bias = nn.Parameter(torch.zeros(numerical))
        self.cat_out = nn.Parameter(torch.randn(len(categories), width, max(categories)) * .02)
        self.cat_bias = nn.Parameter(torch.zeros(len(categories), max(categories)))
        allowed = torch.arange(max(categories))[None] < torch.tensor(categories)[:, None]
        self.register_buffer('allowed', allowed)

    def forward(self, num, cat_embeddings, sigma, cat_noise, conditions, valid, known, t):
        b, length, _ = num.shape
        # Fixed EDM reference scale. Input mode retains supplied feature scales;
        # this preconditioning does not normalize those data to unit variance.
        sigma_data = 1.0
        c_in = (sigma.square() + sigma_data ** 2).rsqrt()
        num_tokens = (num * c_in)[..., None] * self.num_weight
        num_tokens = num_tokens + self.num_noise(sigma.clamp_min(.002).log()[..., None] / 4)
        cat_tokens = cat_embeddings + self.cat_noise(cat_noise[..., None])
        x = torch.cat([num_tokens, cat_tokens], dim=2) + self.feature_bias
        side = torch.cat([conditions, known[..., None].float(), t.expand(b, length, 1)], -1)
        x = x + self.condition(side)[:, :, None]
        # Relative offsets distinguish positions even within a short constant phase.
        pos = torch.arange(length, device=x.device, dtype=x.dtype)[:, None]
        freq = torch.exp(-math.log(10000) * torch.arange(0, self.width, 2, device=x.device) / self.width)
        position = torch.stack([(pos * freq).sin(), (pos * freq).cos()], -1).flatten(-2)
        x = (x + position[None, :, None]) * valid[:, :, None, None]
        temporal_mask = valid[:, None].expand(-1, self.features, -1).reshape(b * self.features, length)
        for feature_layer, time_layer in zip(self.feature_attention, self.temporal_attention):
            x = feature_layer(x.reshape(b * length, self.features, self.width)).reshape(b, length, self.features, self.width)
            x = x.permute(0, 2, 1, 3).reshape(b * self.features, length, self.width)
            x = time_layer(x, temporal_mask).reshape(b, self.features, length, self.width).permute(0, 2, 1, 3)
        raw_num = (x[:, :, :self.numerical] * self.num_out).sum(-1) + self.num_bias
        logits = torch.einsum('btch,chk->btck', x[:, :, self.numerical:], self.cat_out) + self.cat_bias
        logits = logits.masked_fill(~self.allowed, -1e4)
        c_skip = sigma_data ** 2 / (sigma.square() + sigma_data ** 2)
        c_out = sigma * sigma_data * c_in
        return c_skip * num + c_out * raw_num, logits


class TemporalTabDiff(nn.Module):
    def __init__(self, schema, knots, config):
        super().__init__()
        self.schema, self.config = schema, config
        if (config["numerical_transform"] != "input"
                or config["position_representation"] != "frenet_delta"
                or not config["progress_conditioning"]
                or config.get("lambda_physics", 0.) != 0.):
            raise ValueError("Use the final input-coordinate, progress-conditioned road-increment model.")
        self.codec = MixedCodec(schema, knots, config["numerical_transform"])
        n, cats = len(self.codec.num_features), self.codec.categories
        if not n or not cats:
            raise ValueError('This driving adapter requires numerical and categorical features.')
        self.num_schedule = PowerMeanNoise_PerColumn(n)
        self.cat_schedule = LogLinearNoise_PerColumn(len(cats))
        self.denoiser = TemporalDenoiser(n, cats, config['hidden_dim'], config['heads'], config['layers'],
                                        CONDITION_DIM + (2 if config.get('progress_conditioning') else 0))
        self.progress = None
        if config.get('progress_conditioning'):
            from .progress import ProgressConditioner
            self.progress = ProgressConditioner(config['progress_spec'], self.codec, config.get('progress_loss_scale_m', 10.))

    def _embedding(self, cat):
        return self.denoiser.embedding(cat + self.denoiser.offsets)

    def predict(self, num, cat_embedding, sigma, move, conditions, valid, known, t):
        num = torch.where(valid[..., None], num, 0)
        cat_embedding = torch.where(valid[..., None, None], cat_embedding, 0)
        sigma = sigma.expand_as(num).clone()
        # Prefix rows are explicitly clean, separately marked as known.
        sigma = torch.where(known[..., None], torch.zeros_like(sigma), sigma)
        move = move.expand(*num.shape[:2], len(self.codec.categories))
        move = torch.where(known[..., None], torch.zeros_like(move), move)
        conditions = torch.where(valid[..., None], conditions, 0)
        return self.denoiser(num, cat_embedding, sigma, move, conditions, valid, known, t)

    def loss(self, packed):
        c, d = self.config, self.schema.encoded_dim
        k, length = c['context_length'], c['sequence_length']
        packed = packed[:, :k + length]
        valid = packed[..., -1].bool()
        base_conditions = packed[..., d:d + CONDITION_DIM]
        conditions = base_conditions
        if self.progress is not None:
            anchor = self.progress.training_anchor(packed, d, k)
            conditions = self.progress.augment(base_conditions, anchor)
        clean, cat = self.codec.split(packed[..., :d])
        known = torch.zeros_like(valid); known[:, :k] = valid[:, :k]
        # Train unconditional starts and robustness to an imperfect generated prefix.
        if self.training:
            drop = torch.rand(len(packed), 1, device=packed.device) < c['context_dropout']
            valid = valid & ~(known & drop)
            known = known & ~drop
        clean = torch.where(valid[..., None], clean, 0)
        cat = torch.where(valid[..., None], cat, 0)
        t = torch.rand(len(packed), 1, 1, device=packed.device).clamp(.00001, .99999)
        sigma = self.num_schedule.total_noise(t[:, 0])[:, None]
        total = self.cat_schedule.total_noise(t[:, 0])[:, None]
        rate = self.cat_schedule.rate_noise(t[:, 0])[:, None]
        move = -torch.expm1(-total)
        # TabDiff uses a straight-through Gumbel mask to train column schedules.
        probabilities = torch.stack([1 - move, move], -1).expand(-1, clean.shape[1], -1, -1)
        choice = F.gumbel_softmax(probabilities.clamp_min(1e-12).log(), tau=.01, hard=True, dim=-1)
        masked = choice[..., 1]
        masked = torch.where(known[..., None], torch.zeros_like(masked), masked)
        clean_embed = self._embedding(cat)
        mask_embed = self._embedding(self.denoiser.mask_ids[None, None].expand_as(cat))
        cat_embed = clean_embed * (1 - masked[..., None]) + mask_embed * masked[..., None]
        noisy = clean + sigma * torch.randn_like(clean)
        prefix = clean + c['context_noise'] * torch.randn_like(clean) if self.training else clean
        noisy = torch.where(known[..., None], prefix, noisy)
        predicted, logits = self.predict(noisy, cat_embed, sigma, move, conditions, valid, known, t)
        target = valid & ~known
        target[:, :k] = False  # Dropped prefixes are not prediction targets.
        terminal = base_conditions[..., -1].bool()
        num_mask = target[..., None] & (~terminal[..., None] | self.codec.num_is_state)
        cat_mask = target[..., None] & (~terminal[..., None] | self.codec.cat_is_state)
        weight = (sigma.square() + 1) / sigma.square()
        numerical = ((predicted - clean).square() * weight * num_mask).sum() / num_mask.sum().clamp_min(1)
        ce = -logits.log_softmax(-1).gather(-1, cat[..., None]).squeeze(-1)
        # -alpha'(t)/(1-alpha(t)); only absorbed (masked) entries incur CE.
        cat_weight = rate / torch.expm1(total)
        categorical = (ce * masked * cat_weight * cat_mask).sum() / cat_mask.sum().clamp_min(1)
        recovered = torch.where(known[..., None], clean, predicted)
        pair_mask = num_mask[:, 1:] & valid[:, :-1, None]
        errors = ((recovered[:, 1:] - recovered[:, :-1]) - (clean[:, 1:] - clean[:, :-1])).square()
        # Low-noise predictions provide a meaningful supervised derivative target.
        confidence = 1 / (1 + sigma.square())
        delta = (errors * pair_mask * confidence).sum() / pair_mask.sum().clamp_min(1)
        seam_mask = num_mask[:, k] & known[:, k - 1, None]
        boundary = (errors[:, k - 1] * seam_mask * confidence[:, 0]).sum() / seam_mask.sum().clamp_min(1)
        total_loss = numerical + categorical + c['lambda_delta'] * delta + c['lambda_boundary'] * boundary
        result = {'loss': total_loss, 'numerical': numerical, 'categorical': categorical,
                  'delta': delta, 'boundary': boundary}
        if self.progress is not None and c.get('lambda_progress', 0.):
            cumulative = self.progress.cumulative_loss(predicted, clean, target, base_conditions, sigma, k)
            result['progress'] = cumulative
            result['loss'] = result['loss'] + c['lambda_progress'] * cumulative
        return result

    @torch.no_grad()
    def sample_segment(self, context, conditions, valid, rng, steps=None, categorical_mode='sample', progress_anchor=None):
        if categorical_mode not in ('sample', 'argmax'):
            raise ValueError('categorical_mode must be sample or argmax.')
        steps = self.config['sampling_steps'] if steps is None else steps
        if steps < 2: raise ValueError('Use at least two diffusion sampling steps.')
        k, d = self.config['context_length'], self.schema.encoded_dim
        b, length, _ = conditions.shape
        context_num, context_cat = self.codec.split(context[..., :d])
        context_valid = context[..., -1].bool()
        full_valid = torch.cat([context_valid, valid.squeeze(-1).bool()], 1)
        known = torch.cat([context_valid, torch.zeros(b, length, dtype=torch.bool, device=context.device)], 1)
        plan = torch.cat([context[..., d:-1], conditions], 1)
        if self.progress is not None:
            if progress_anchor is None:
                raise ValueError('Progress-conditioned sampling requires a generated preceding position.')
            plan = self.progress.augment(plan, progress_anchor)
        elif progress_anchor is not None:
            raise ValueError('This model does not use progress conditioning.')
        times = torch.linspace(1, 0, steps, device=context.device)
        sigmas = self.num_schedule.total_noise(times[:, None])
        moves = -torch.expm1(-self.cat_schedule.total_noise(times[:, None]))
        num = torch.randn(b, length, len(self.codec.num_features), device=context.device, generator=rng) * sigmas[0]
        cat = self.denoiser.mask_ids[None, None].expand(b, length, -1).clone()
        def estimate(x, y, sigma, move, time):
            values = torch.cat([context_num, x], 1)
            embeddings = self._embedding(torch.cat([context_cat, y], 1))
            pred, logits = self.predict(values, embeddings, sigma[None, None], move[None, None], plan,
                                        full_valid, known, time.expand(b, 1, 1))
            return pred[:, k:], logits[:, k:]
        for i, time in enumerate(times):
            sigma, move = sigmas[i], moves[i]
            next_sigma = sigmas[i + 1] if i + 1 < steps else torch.zeros_like(sigma)
            next_move = moves[i + 1] if i + 1 < steps else torch.zeros_like(move)
            denoised, logits = estimate(num, cat, sigma, move, time)
            derivative = (num - denoised) / sigma
            proposed = num + (next_sigma - sigma) * derivative
            # Heun correction with the current categories, as in TabDiff's solver.
            if self.config['second_order'] and i + 1 < steps:
                corrected, _ = estimate(proposed, cat, next_sigma, next_move, times[i + 1])
                proposed = num + .5 * (next_sigma - sigma) * (derivative + (proposed - corrected) / next_sigma)
            num = proposed
            unmask_probability = ((move - next_move) / move).clamp(0, 1)
            reveal = torch.rand(cat.shape, device=cat.device, generator=rng) < unmask_probability
            if categorical_mode == 'sample':
                uniform = torch.rand(logits.shape, device=cat.device, generator=rng).clamp(1e-8, 1 - 1e-8)
                sampled = (logits - (-uniform.log()).log()).argmax(-1)
            else:
                sampled = logits.argmax(-1)
            cat = torch.where((cat == self.denoiser.mask_ids) & reveal, sampled, cat)
        if not torch.isfinite(num).all() or (cat >= self.denoiser.mask_ids).any():
            raise FloatingPointError('Invalid diffusion sample; no data was exported.')
        return self.codec.merge(num, cat, conditions) * valid
