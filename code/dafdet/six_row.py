"""Six controlled configurations for the revised Section 4.4.

This implementation owns the revised experiment policy independently.
All final detector outputs and the main detection loss use optical-crop boxes.
Rows 4/6 share sonar-query initialization regardless of the decoder choice.
Rows 5/6 alone use separate-modality geometric decoder reference regions.

The sonar-only row directly learns optical-coordinate boxes. Its normalized
references are also learned sampling parameters on the sonar feature grid;
this is NOT a claim that the two physical image coordinates are aligned.
It never requires optical pixels, although optical GT/size metadata are used.
"""
import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F

from .components import (
    SharedDetectionScaffold, CommonDetectionCriterion, DecoderLayer, gather, inverse_sigmoid,
    select_queries, get_contrastive_denoising_training_group, postprocess,
)


class SixRowDAFDet(SharedDetectionScaffold):
    """Common ResNet50-vd/HybridEncoder scaffold with explicit six-row toggles.

    config: row=1..6, num_queries=300, decoder_layers=6, hidden_dim=256.
    Geometry objects implement normalized(boxes, direction, flipped=False)
    and expose config['status']. Coarse geometry remains explicitly unverified.
    Reduced query counts/dimensions are allowed only for structural tests;
    experiment launchers must enforce their frozen, shared configuration.
    """

    def __init__(self, config, geometry=None):
        cfg = dict(config)
        self.row = int(cfg['row'])
        if self.row not in range(1, 7):
            raise ValueError('Six-row configuration must be 1 through 6')
        self.input_modalities = ('optical',) if self.row == 1 else (
            ('sonar',) if self.row == 2 else ('optical', 'sonar'))
        self.vsaqs = self.row in (4, 6)
        self.madfd = self.row in (5, 6)
        requires_geometry = self.vsaqs or self.madfd
        if requires_geometry:
            if geometry is None or not callable(getattr(geometry, 'normalized', None)):
                raise ValueError('Rows 4--6 require an explicit coordinate-mapping object')
            status = getattr(geometry, 'config', {}).get('status')
            allowed = {'verified', 'approximate_unverified', 'coarse_unverified'}
            if cfg.get('allow_test_geometry'):
                allowed.add('synthetic_test_only')
            if status not in allowed:
                raise ValueError('Geometry must explicitly declare its provenance/status')
        # The scaffold contains no legacy experiment-variant policy.
        super().__init__(cfg, len(self.input_modalities), decoder_modalities=2 if self.madfd else 1)
        self.geometry = geometry
        self.variant = 'AB%02d' % self.row
        self.use_geometry = self.madfd
        # The actual query content is always selected encoder features.
        self.protocol = dict(
            row=self.row, input_modalities=list(self.input_modalities),
            num_queries=self.k, query_content_initialization='selected_encoder_features',
            query_selection='shared_modalities_topk' if self.vsaqs else 'single_pool_topk',
            query_pool=('separate_modalities' if self.vsaqs else
                        'elementwise_sum' if len(self.input_modalities) == 2 else self.input_modalities[0]),
            decoder=('separate_modality_geometry' if self.madfd else
                     'standard_on_elementwise_sum' if len(self.input_modalities) == 2 else 'standard_single_modality'),
            output_coordinate_system='optical_crop_normalized_cxcywh',
            main_supervision_coordinate_system='optical_crop_normalized_cxcywh',
            sonar_only_sampling_semantics=('learned_optical_normalized_reference_on_sonar_grid'
                                          if self.row == 2 else None),
            geometry_status=(getattr(geometry, 'config', {}).get('status') if requires_geometry else 'not_used'),
            encoder_auxiliary_supervision=(
                ['visual_native_to_optical_GT', 'sonar_native_to_sonar_GT', 'added_to_optical_GT']
                if len(self.input_modalities) == 2 else ['sole_branch_to_optical_GT']),
            encoder_auxiliary_weights=[1., 1., 1.] if len(self.input_modalities) == 2 else [1.],
            added_auxiliary_present_in_all_multimodal_rows=True,
        )

    def _images(self, optical, sonar):
        if self.row == 1:
            if optical is None or sonar is not None:
                raise ValueError('Row 1 accepts optical pixels only')
            return [optical]
        if self.row == 2:
            if sonar is None or optical is not None:
                raise ValueError('Row 2 accepts sonar pixels only; optical must be None')
            return [sonar]
        if optical is None or sonar is None:
            raise ValueError('Rows 3--6 require both modalities')
        return [optical, sonar]

    @staticmethod
    def _add_features(features):
        if len(features) != 2 or len(features[0]) != len(features[1]):
            raise ValueError('Element-wise addition requires matching feature levels')
        for visual, sonar in zip(*features):
            if list(visual.shape) != list(sonar.shape):
                raise ValueError('Element-wise addition requires matching tensor shapes')
        return [visual + sonar for visual, sonar in zip(*features)]

    def _query_and_decoder_features(self, features):
        if len(self.input_modalities) == 1:
            return features, features
        added = None if self.vsaqs and self.madfd else self._add_features(features)
        query_features = features if self.vsaqs else [added]
        decoder_features = features if self.madfd else [added]
        return query_features, decoder_features

    def _initialize_optical_references(self, source_boxes, source_modality, flipped):
        if not self.vsaqs:
            return source_boxes, paddle.ones(source_modality.shape, dtype='bool')
        projected, valid = self.geometry.normalized(
            source_boxes.detach().numpy(), 'sonar_to_optical', flipped)
        projected = np.asarray(projected, dtype='float32')
        valid = np.asarray(valid, dtype=bool)
        if projected.shape != tuple(source_boxes.shape) or valid.shape != tuple(source_modality.shape):
            raise ValueError('Invalid query mapping output shape')
        if not np.isfinite(projected[valid]).all():
            raise ValueError('Nonfinite valid projected query reference')
        # Broad search fallback is explicit and shared by rows 4 and 6.
        projected = np.where(valid[..., None], projected,
                             np.array([.5, .5, 1., 1.], dtype='float32'))
        is_sonar = source_modality == 1
        return (paddle.where(is_sonar.unsqueeze(-1), paddle.to_tensor(projected), source_boxes),
                ~is_sonar | paddle.to_tensor(valid))

    def _decoder_regions(self, optical_reference, flipped):
        batch, queries, _ = optical_reference.shape
        if not self.madfd:
            return optical_reference.unsqueeze(2), paddle.ones([batch, queries, 1], dtype='bool')
        sonar_reference, sonar_valid = self.geometry.normalized(
            optical_reference.detach().numpy(), 'optical_to_sonar', flipped)
        sonar_reference = np.asarray(sonar_reference, dtype='float32')
        sonar_valid = np.asarray(sonar_valid, dtype=bool)
        if sonar_reference.shape != tuple(optical_reference.shape) or sonar_valid.shape != (batch, queries):
            raise ValueError('Invalid decoder mapping output shape')
        if not np.isfinite(sonar_reference[sonar_valid]).all():
            raise ValueError('Nonfinite valid sonar sampling reference')
        sonar_reference = np.where(sonar_valid[..., None], sonar_reference, 0.)
        return (paddle.stack([optical_reference, paddle.to_tensor(sonar_reference)], 2),
                paddle.stack([paddle.ones([batch, queries], dtype='bool'), paddle.to_tensor(sonar_valid)], 2))

    def _encoder_predictions(self, levels):
        memory = paddle.concat([f.flatten(2).transpose([0, 2, 1]) for f in levels], 1)
        anchors = self._anchors(levels)
        valid = ((anchors > .01) & (anchors < .99)).all(-1).expand([memory.shape[0], -1])
        encoded = self.memory_proj(memory * valid.astype(memory.dtype).unsqueeze(-1))
        logits = self.enc_score_head(encoded)
        boxes = F.sigmoid(self.enc_box_head(encoded) + inverse_sigmoid(anchors))
        return dict(memory=memory, encoded=encoded, pred_boxes=boxes, pred_logits=logits, valid=valid)

    def forward(self, optical=None, sonar=None, targets=None, diagnostics=False, flipped=False):
        images = self._images(optical, sonar)
        features = [enc(backbone({'image':image}))
                    for backbone, enc, image in zip(self.backbones, self.encoders, images)]
        query_features, decoder_features = self._query_and_decoder_features(features)
        # Exactly the same three auxiliary heads/loss inputs in every multimodal
        # row, including the added-feature head when VSAQS selects native pools.
        # Weights are shared; this introduces no extra learnable parameter set.
        encoder_predictions = [self._encoder_predictions(levels) for levels in features]
        if len(self.input_modalities) == 2:
            added = self._add_features(features) if self.vsaqs and self.madfd else (
                decoder_features[0] if self.vsaqs else query_features[0])
            encoder_predictions.append(self._encoder_predictions(added))
            query_predictions = encoder_predictions[:2] if self.vsaqs else encoder_predictions[2:]
        else:
            query_predictions = encoder_predictions
        memory_parts = [p['memory'] for p in query_predictions]
        encoded = paddle.concat([p['encoded'] for p in query_predictions], 1)
        logits = paddle.concat([p['pred_logits'] for p in query_predictions], 1)
        boxes = paddle.concat([p['pred_boxes'] for p in query_predictions], 1)
        valid = paddle.concat([p['valid'] for p in query_predictions], 1)
        if self.k > encoded.shape[1]:
            raise ValueError('Fewer encoder candidates than requested queries')
        indices, origin = select_queries(logits, memory_parts[0].shape[1], self.k, 'competitive', valid)
        if not self.vsaqs:
            # -1 means a candidate from added features, not a native modality.
            source = 1 if self.row == 2 else 0 if self.row == 1 else -1
            origin = paddle.full(indices.shape, source, dtype='int64')
        content = gather(encoded, indices)
        source_boxes = gather(boxes, indices)
        reference, initialization_valid = self._initialize_optical_references(source_boxes, origin, flipped)
        initial = reference
        dn_meta = mask = None
        dn_count = 0
        if self.training and targets is not None:
            dn_query, dn_box, mask, dn_meta = get_contrastive_denoising_training_group(
                targets, 4, self.k, self.dn_embed.weight,
                num_denoising=self.config.get('num_denoising', 100),
                label_noise_ratio=.5, box_noise_scale=1.)
            if dn_meta is not None:
                dn_count = dn_meta['dn_num_split'][0]
                content = paddle.concat([dn_query, content], 1)
                reference = paddle.concat([F.sigmoid(dn_box), reference], 1)
        query = (content.detach() if not dn_count else
                 paddle.concat([content[:, :dn_count], content[:, dn_count:].detach()], 1))
        reference = reference.detach()
        all_boxes, all_logits, layer_diagnostics = [], [], []
        for layer, classifier, box_head in zip(self.decoder, self.cls_heads, self.box_heads):
            regions, region_valid = self._decoder_regions(reference, flipped)
            query, detail = layer(query, self.pos(reference), decoder_features, regions,
                                  region_valid, mask, diagnostics)
            updated = F.sigmoid(box_head(query) + inverse_sigmoid(reference))
            all_boxes.append(updated)
            all_logits.append(classifier(query))
            reference = updated.detach()
            if diagnostics:
                layer_diagnostics.append(detail)
        stacked_boxes, stacked_logits = paddle.stack(all_boxes), paddle.stack(all_logits)
        result = dict(
            pred_boxes=stacked_boxes[-1, :, dn_count:], pred_logits=stacked_logits[-1, :, dn_count:],
            all_boxes=stacked_boxes[:, :, dn_count:], all_logits=stacked_logits[:, :, dn_count:],
            aux_outputs=[dict(pred_boxes=b[:, dn_count:], pred_logits=l[:, dn_count:])
                         for b, l in zip(all_boxes[:-1], all_logits[:-1])],
            dn_meta=dn_meta, dn_boxes=stacked_boxes[:, :, :dn_count] if dn_count else None,
            dn_logits=stacked_logits[:, :, :dn_count] if dn_count else None,
            encoder_outputs=[{key: p[key] for key in ('pred_boxes', 'pred_logits', 'valid')}
                             for p in encoder_predictions],
            encoder_coordinate_systems=(['optical', 'sonar', 'optical']
                                        if len(self.input_modalities) == 2 else ['optical']),
        )
        if diagnostics:
            result['diagnostics'] = dict(
                source_modality=origin, source_box=source_boxes,
                initial_optical_reference_box=initial, initial_mapping_valid=initialization_valid,
                candidate_index=indices, selected_encoder_content=gather(encoded, indices),
                query_pool_count=len(query_features), decoder_feature_pool_count=len(decoder_features),
                feature_shapes=[[list(f.shape) for f in levels] for levels in features],
                layers=layer_diagnostics, dn_query_count_excluded=dn_count,
            )
        return result


class SixRowCriterion(CommonDetectionCriterion):
    """Optical main loss and fixed, equal-weight encoder auxiliary losses.

    Every multimodal row has visual->optical, sonar->sonar, added->optical
    auxiliary supervision. Thus the auxiliary losses do not change with the
    VSAQS/MADFD toggles. Single-modality rows each have one optical-coordinate
    auxiliary loss. Complete sonar GT is required for the multimodal recipe.
    """

    def forward(self, out, targets):
        systems = out['encoder_coordinate_systems']
        if systems not in (['optical'], ['optical', 'sonar', 'optical']):
            raise ValueError('Unexpected encoder supervision coordinate systems')
        if len(systems) == 3 and ('sonar_gt_bbox' not in targets or 'sonar_gt_class' not in targets):
            raise ValueError('Fixed multimodal auxiliary supervision requires complete sonar GT')
        losses = super().forward(out, targets)
        if len(systems) == 3:
            added = out['encoder_outputs'][2]
            valid_ids = paddle.nonzero(added['valid'][0]).flatten()
            boxes = paddle.index_select(added['pred_boxes'], valid_ids, axis=1).unsqueeze(0)
            logits = paddle.index_select(added['pred_logits'], valid_ids, axis=1).unsqueeze(0)
            losses.update(self.encoder(boxes, logits, targets['gt_bbox'], targets['gt_class'], postfix='_enc_2'))
        return losses


__all__ = ['SixRowDAFDet', 'SixRowCriterion', 'postprocess']
