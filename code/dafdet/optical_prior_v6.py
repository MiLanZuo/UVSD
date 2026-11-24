"""Optical-prior extension of the frozen six-row AB06 decoder.

Only the modality prior in MADFD changes.  The original parameter names,
tensor shapes, VSAQS, reference mapping, and auxiliary losses are retained.
The scalar is configuration, not a trained parameter: callers must include
``optical_prior_alpha`` in checkpoint/resume fingerprints.  Bare state_dicts
do not encode this configuration and must not be treated as complete models.
"""
from __future__ import annotations

import math

import paddle
from paddle.nn import functional as F

from .components import RegionAttention
from .six_row import SixRowDAFDet


PLANNED_OPTICAL_PRIORS = (0.60, 0.75, 0.90)


def validate_optical_prior(alpha):
    """Return a finite probability; 0.5 is the exact neutral control."""
    try:
        value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError('optical_prior_alpha must be finite and strictly between 0 and 1') from exc
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError('optical_prior_alpha must be finite and strictly between 0 and 1')
    return value


class OpticalPriorRegionAttention(RegionAttention):
    """Existing normalized deformable attention with fixed optical log odds.

For fixed raw logits, this increases optical attention mass when alpha > .5.
The learned logits can compensate for the prior; alpha is not a guaranteed
post-training optical mass or a guaranteed improvement in detection quality.
Instances are promoted in place by ``attach_optical_prior`` without creating
parameters or consuming random numbers.
"""

    def forward(self, q, features, regions, valid, diagnostics=False):
        alpha = self.optical_prior_alpha
        if alpha == 0.5:
            # Exact existing operations, including their order, for the control.
            out, detail = super().forward(q, features, regions, valid, diagnostics)
        else:
            b, n, c = q.shape
            off = self.offsets(q).reshape([b, n, self.h, self.m, self.l, self.p, 2])
            logits = self.weights(q).reshape([b, n, self.h, self.m, self.l, self.p])
            # Adding (log(alpha/(1-alpha)), 0) is softmax-equivalent to
            # adding (log(alpha), log(1-alpha)).  This does not change values,
            # sampling offsets, box coordinates, or encoder classification.
            prior = paddle.to_tensor(
                [self.optical_prior_log_odds, 0.0], dtype=logits.dtype, place=logits.place
            ).reshape([1, 1, 1, 2, 1, 1])
            logits = logits + prior
            mask = valid[:, :, None, :, None, None]
            logits = paddle.where(mask, logits, paddle.full_like(logits, -1e9))
            weights = F.softmax(logits.reshape([b, n, self.h, -1]), axis=-1).reshape(logits.shape)
            weights = weights * mask.astype(weights.dtype)
            weights = weights / paddle.clip(weights.sum([3, 4, 5], keepdim=True), min=1e-12)
            xy = regions[:, :, None, :, None, None, :2] + regions[:, :, None, :, None, None, 2:] * off
            total = paddle.zeros([b, n, self.h, self.d], dtype=q.dtype)
            for m in range(self.m):
                for level in range(self.l):
                    feat = features[m][level]
                    hh, ww = feat.shape[-2:]
                    value = self.value(feat.flatten(2).transpose([0, 2, 1]))
                    value = value.reshape([b, hh, ww, self.h, self.d]).transpose([0, 3, 4, 1, 2])
                    value = value.reshape([b * self.h, self.d, hh, ww])
                    grid = (2 * xy.reshape([b, n, self.h, self.m * self.l, self.p, 2])[
                        :, :, :, m * self.l + level
                    ] - 1).transpose([0, 2, 1, 3, 4]).reshape([b * self.h, n, self.p, 2])
                    sample = F.grid_sample(value, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
                    sample = sample.reshape([b, self.h, self.d, n, self.p]).transpose([0, 3, 1, 4, 2])
                    total += (sample * weights[:, :, :, m, level, :, None]).sum(3)
            out = self.output(total.reshape([b, n, c]))
            detail = dict(sampling_locations=xy, attention_weights=weights, effective_offsets=off,
                          reference_regions=regions, region_valid=valid) if diagnostics else None
        if diagnostics:
            detail['optical_prior_alpha'] = alpha
            detail['optical_attention_mass'] = detail['attention_weights'][:, :, :, 0].sum([3, 4])
            detail['sonar_attention_mass'] = detail['attention_weights'][:, :, :, 1].sum([3, 4])
        return out, detail


def attach_optical_prior(layer, alpha):
    """Promote a two-modality layer in place without changing its state or RNG."""
    value = validate_optical_prior(alpha)
    if type(layer) not in (RegionAttention, OpticalPriorRegionAttention) or layer.m != 2:
        raise ValueError('Optical prior requires the existing two-modality RegionAttention')
    layer.__class__ = OpticalPriorRegionAttention
    layer.optical_prior_alpha = value
    layer.optical_prior_log_odds = 0.0 if value == 0.5 else math.log(value) - math.log1p(-value)
    return layer


class OpticalPriorDAFDet(SixRowDAFDet):
    """Complete DAFDet with one explicit scalar prior shared by its decoder layers."""

    def __init__(self, config, geometry=None):
        cfg = dict(config)
        if cfg.get('row', 6) != 6:
            raise ValueError('OpticalPriorDAFDet extends only the complete row 6 configuration')
        if 'optical_prior_alpha' not in cfg:
            raise ValueError('An explicit optical_prior_alpha is required for provenance')
        alpha = validate_optical_prior(cfg['optical_prior_alpha'])
        cfg.update(row=6, optical_prior_alpha=alpha)
        super().__init__(cfg, geometry)
        self.optical_prior_alpha = alpha
        for layer in self.decoder:
            attach_optical_prior(layer.cross, alpha)
        self.protocol.update(
            optical_prior_alpha=alpha,
            optical_prior_location='MADFD_attention_logits_only',
            optical_prior_rule='optical_log_odds_added_then_joint_modality_level_point_softmax',
            optical_prior_trainable=False,
            optical_prior_neutral_alpha=0.5,
            optical_prior_query_selection_unchanged=True,
            optical_prior_auxiliary_losses_unchanged=True,
            optical_prior_checkpoint_requires_config=True,
        )


__all__ = ['OpticalPriorDAFDet', 'OpticalPriorRegionAttention', 'attach_optical_prior',
           'validate_optical_prior', 'PLANNED_OPTICAL_PRIORS']
