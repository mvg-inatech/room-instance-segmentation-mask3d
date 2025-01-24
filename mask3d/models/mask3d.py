from typing import Dict, List, Union
import torch
import torch.nn as nn
import MinkowskiEngine.MinkowskiOps as me
from MinkowskiEngine.MinkowskiPooling import MinkowskiAvgPooling
import numpy as np
from torch.nn import functional as F
from models.modules.common import conv
from models.position_embedding import PositionEmbeddingCoordsSine
from third_party.pointnet2.pointnet2_utils import furthest_point_sample
from models.modules.helpers_3detr import GenericMLP
from torch_scatter import scatter_mean, scatter_max, scatter_min
from torch.cuda.amp import autocast
import utils.measure_runtime as measure_runtime


class SingleModelPredictions:
    def __init__(self, pred_class: torch.Tensor, pred_mask: List[torch.Tensor]) -> None:
        self.pred_class = pred_class
        self.pred_mask = pred_mask


class SinglePointRuntimeError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("only a single point gives nans in cross-attention")


class ModelOutput:
    def __init__(
        self,
        # Tensor of dim [batch_size, num_queries, num_classes+1] with the classification logits (no softmax applied). Predictions of the final decoder.
        pred_class: torch.Tensor,
        # List of Tensor of dim [batch_size, num_queries, num_classes+1] with the classification logits (no softmax applied). Each list item is the prediction of a decoder.
        pred_class_all_decoders: List[torch.Tensor],
        # List of Tensor of dim [num_points, num_queries] with the predicted masks. Each list item is one batch item. Predictions of the final decoder.
        pred_mask: List[torch.Tensor],
        # List of Tensor of dim [num_points, num_queries] with the predicted masks. Each inner list item is one batch item, each outer list item is the prediction of a decoder.
        pred_mask_all_decoders: List[List[torch.Tensor]],
        sampled_coords: np.ndarray | None,
        backbone_features: me.SparseTensor,
    ) -> None:
        self.pred_class = pred_class
        self.pred_class_all_decoders = pred_class_all_decoders
        self.pred_mask = pred_mask
        self.pred_mask_all_decoders = pred_mask_all_decoders
        self.sampled_coords = sampled_coords
        self.backbone_features = backbone_features

    def get_device(self) -> torch.device:
        # TODO assert all of the class attributes are on the same device
        return self.pred_class.device

    def get_single_predictions(self) -> SingleModelPredictions:
        return SingleModelPredictions(self.pred_class, self.pred_mask)

    def to(self, device: torch.device):
        self.pred_class = self.pred_class.to(device)
        self.pred_class_all_decoders = [t.to(device) for t in self.pred_class_all_decoders]
        self.pred_mask = [t.to(device) for t in self.pred_mask]
        self.pred_mask_all_decoders = [[t.to(device) for t in inner] for inner in self.pred_mask_all_decoders]
        # TODO handle self.backbone_features
        return self

    def detach(self):
        self.pred_class = self.pred_class.detach()
        self.pred_class_all_decoders = [t.detach() for t in self.pred_class_all_decoders]
        self.pred_mask = [t.detach() for t in self.pred_mask]
        self.pred_mask_all_decoders = [[t.detach() for t in inner] for inner in self.pred_mask_all_decoders]
        # TODO handle self.backbone_features
        return self


class Mask3D(nn.Module):
    def __init__(
        self,
        config,
        hidden_dim,
        num_queries,
        num_heads,
        dim_feedforward,
        sample_sizes,
        shared_decoder,
        num_classes,
        num_decoders,
        dropout,
        pre_norm,
        positional_encoding_type,
        non_parametric_queries,
        normalize_pos_enc,
        use_level_embed,
        scatter_type,
        hlevels,
        use_np_features,
        max_sample_size,
        random_queries,
        gauss_scale,
        random_query_both,
        random_normal,
    ):
        super().__init__()

        self.random_normal = random_normal
        self.random_query_both = random_query_both
        self.random_queries = random_queries
        self.max_sample_size = max_sample_size
        self.gauss_scale = gauss_scale
        self.scatter_type = scatter_type
        self.hlevels = hlevels
        self.use_level_embed = use_level_embed
        self.normalize_pos_enc = normalize_pos_enc
        self.num_decoders = num_decoders
        self.num_classes = num_classes  # Number of semantic classes C, excluding the "no instance"/"ignore" class (see paper)
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.shared_decoder = shared_decoder
        self.sample_sizes = sample_sizes
        self.non_parametric_queries = non_parametric_queries
        self.use_np_features = use_np_features
        self.mask_dim = hidden_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.pos_enc_type = positional_encoding_type

        self.backbone = config.backbone

        self.num_levels = len(self.hlevels)
        sizes = self.backbone.PLANES[-5:]

        self.mask_features_head = conv(
            self.backbone.PLANES[7],
            self.mask_dim,
            kernel_size=1,
            stride=1,
            bias=True,
            D=3,
        )

        if self.scatter_type == "mean":
            self.scatter_fn = scatter_mean
        elif self.scatter_type == "max":
            self.scatter_fn = lambda mask, p2s, dim: scatter_max(mask, p2s, dim=dim)[0]
        else:
            assert False, "Scatter function not known"

        assert (not use_np_features) or non_parametric_queries, "np features only with np queries"

        if self.non_parametric_queries:
            self.query_projection = GenericMLP(
                input_dim=self.mask_dim,
                hidden_dims=[self.mask_dim],
                output_dim=self.mask_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True,
            )

            if self.use_np_features:
                self.np_feature_projection = nn.Sequential(
                    nn.Linear(sizes[-1], hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
        elif self.random_query_both:
            self.query_projection = GenericMLP(
                input_dim=2 * self.mask_dim,
                hidden_dims=[2 * self.mask_dim],
                output_dim=2 * self.mask_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True,
            )
        else:
            # PARAMETRIC QUERIES
            # learnable query features
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            # learnable query p.e.
            self.query_pos = nn.Embedding(num_queries, hidden_dim)

        if self.use_level_embed:
            # learnable scale-level embedding
            self.level_embed = nn.Embedding(self.num_levels, hidden_dim)

        self.mask_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.class_embed_head = nn.Linear(hidden_dim, self.num_classes + 1)  # +1 for the "no instance"/"ignore" class (see paper)

        if self.pos_enc_type == "legacy":
            self.pos_enc = PositionalEncoding3D(channels=self.mask_dim)
        elif self.pos_enc_type == "fourier":
            self.pos_enc = PositionEmbeddingCoordsSine(
                pos_type="fourier",
                d_pos=self.mask_dim,
                gauss_scale=self.gauss_scale,
                normalize=self.normalize_pos_enc,
            )
        elif self.pos_enc_type == "sine":
            self.pos_enc = PositionEmbeddingCoordsSine(
                pos_type="sine",
                d_pos=self.mask_dim,
                normalize=self.normalize_pos_enc,
            )
        else:
            assert False, "pos enc type not known"

        self.pooling = MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)

        self.masked_transformer_decoder = nn.ModuleList()
        self.cross_attention = nn.ModuleList()
        self.self_attention = nn.ModuleList()
        self.ffn_attention = nn.ModuleList()
        self.lin_squeeze = nn.ModuleList()

        num_shared = self.num_decoders if not self.shared_decoder else 1

        for _ in range(num_shared):
            tmp_cross_attention = nn.ModuleList()
            tmp_self_attention = nn.ModuleList()
            tmp_ffn_attention = nn.ModuleList()
            tmp_squeeze_attention = nn.ModuleList()
            for i, hlevel in enumerate(self.hlevels):
                tmp_cross_attention.append(
                    CrossAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm,
                    )
                )

                tmp_squeeze_attention.append(nn.Linear(sizes[hlevel], self.mask_dim))

                tmp_self_attention.append(
                    SelfAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm,
                    )
                )

                tmp_ffn_attention.append(
                    FFNLayer(
                        d_model=self.mask_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm,
                    )
                )

            self.cross_attention.append(tmp_cross_attention)
            self.self_attention.append(tmp_self_attention)
            self.ffn_attention.append(tmp_ffn_attention)
            self.lin_squeeze.append(tmp_squeeze_attention)

        self.decoder_norm = nn.LayerNorm(hidden_dim)

    def get_pos_encs(self, coords):
        pos_encodings_pcd = []

        for feature_map_idx in range(len(coords)):
            pos_encodings_pcd.append([[]])
            for item_coords in coords[feature_map_idx].decomposed_features:
                scene_min = item_coords.min(dim=0)[0][None, ...]
                scene_max = item_coords.max(dim=0)[0][None, ...]

                with autocast(enabled=False):
                    tmp = self.pos_enc(
                        item_coords[None, ...].float(),
                        input_range=[scene_min, scene_max],
                    )

                pos_encodings_pcd[-1][0].append(tmp.squeeze(0).permute((1, 0)))

        return pos_encodings_pcd

    def forward(self, input_batch, is_eval=False) -> ModelOutput:
        backbone_out, backbone_feature_maps = self.backbone(input_batch)
        measure_runtime.add_timing("model_forward_backbone_part1")

        batch_size = len(input_batch.decomposed_coordinates)

        with torch.no_grad():
            coordinates = me.SparseTensor(
                features=input_batch.coordinates[
                    :, 1:4
                ].float(),  # Index 0 is MinkowskiEngine batch index # TODO multiply with rasterization_factor to make it scaling-independent?
                coordinate_manager=backbone_feature_maps[-1].coordinate_manager,
                coordinate_map_key=backbone_feature_maps[-1].coordinate_map_key,
                device=backbone_feature_maps[-1].device,
            )
            coordinates_list_modified = [coordinates]
            for _ in reversed(range(len(backbone_feature_maps) - 1)):
                coordinates_list_modified.append(self.pooling(coordinates_list_modified[-1]))

            coordinates_list_modified.reverse()

        pos_encodings_pcd = self.get_pos_encs(coordinates_list_modified)
        measure_runtime.add_timing("model_forward_pos_enc")
        mask_features = self.mask_features_head(backbone_out)
        measure_runtime.add_timing("model_forward_backbone_part2")

        sampled_coords = None

        if self.non_parametric_queries:
            fps_idx = [
                furthest_point_sample(
                    input_batch.decomposed_coordinates[item_idx_in_batch][None, ...].float(),
                    self.num_queries,
                )
                .squeeze(0)
                .long()
                for item_idx_in_batch in range(len(input_batch.decomposed_coordinates))
            ]

            sampled_coords = torch.stack(
                [coordinates.decomposed_features[item_idx_in_batch][fps_idx[item_idx_in_batch].long(), :] for item_idx_in_batch in range(len(fps_idx))]
            )  # Note that the features of the sparse tensor are the coordinate values

            mins = torch.stack(
                [coordinates.decomposed_features[item_idx_in_batch].min(dim=0)[0] for item_idx_in_batch in range(len(coordinates.decomposed_features))]
            )
            maxs = torch.stack(
                [coordinates.decomposed_features[item_idx_in_batch].max(dim=0)[0] for item_idx_in_batch in range(len(coordinates.decomposed_features))]
            )

            query_pos = self.pos_enc(sampled_coords.float(), input_range=[mins, maxs])  # Batch, Dim, queries
            query_pos = self.query_projection(query_pos)

            if not self.use_np_features:
                queries = torch.zeros_like(query_pos).permute((0, 2, 1))
            else:
                queries = torch.stack([backbone_out.decomposed_features[i][fps_idx[i].long(), :] for i in range(len(fps_idx))])
                queries = self.np_feature_projection(queries)
            query_pos = query_pos.permute((2, 0, 1))
        elif self.random_queries:
            query_pos = (
                torch.rand(
                    batch_size,
                    self.mask_dim,
                    self.num_queries,
                    device=input_batch.device,
                )
                - 0.5
            )

            queries = torch.zeros_like(query_pos).permute((0, 2, 1))
            query_pos = query_pos.permute((2, 0, 1))
        elif self.random_query_both:
            if not self.random_normal:
                query_pos_feat = (
                    torch.rand(
                        batch_size,
                        2 * self.mask_dim,
                        self.num_queries,
                        device=input_batch.device,
                    )
                    - 0.5
                )
            else:
                query_pos_feat = torch.randn(
                    batch_size,
                    2 * self.mask_dim,
                    self.num_queries,
                    device=input_batch.device,
                )

            queries = query_pos_feat[:, : self.mask_dim, :].permute((0, 2, 1))
            query_pos = query_pos_feat[:, self.mask_dim :, :].permute((2, 0, 1))
        else:
            # PARAMETRIC QUERIES
            queries = self.query_feat.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            query_pos = self.query_pos.weight.unsqueeze(1).repeat(1, batch_size, 1)

        measure_runtime.add_timing("model_forward_queries")

        predictions_class = []
        predictions_mask = []

        decoder_num = 0
        for decoder_counter in range(self.num_decoders):
            if self.shared_decoder:
                decoder_counter = 0

            for item_idx, hlevel in enumerate(self.hlevels):
                output_class, outputs_mask, attn_mask = self.mask_module(
                    queries,
                    mask_features,
                    len(backbone_feature_maps) - hlevel - 1,
                    ret_attn_mask=True,
                    coords=coordinates,
                )

                feature_map_level_features = backbone_feature_maps[hlevel].decomposed_features
                decomposed_attn = attn_mask.decomposed_features

                curr_sample_size = max([pcd.shape[0] for pcd in feature_map_level_features])

                if min([pcd.shape[0] for pcd in feature_map_level_features]) == 1:
                    raise SinglePointRuntimeError()

                if not (self.max_sample_size or is_eval):
                    curr_sample_size = min(curr_sample_size, self.sample_sizes[hlevel])

                item_used_point_idxs = []
                mask_idx = []
                for k in range(len(feature_map_level_features)):
                    pcd_size = feature_map_level_features[k].shape[0]
                    if pcd_size <= curr_sample_size:
                        # we do not need to sample
                        # take all points and pad the rest with zeroes and mask it
                        idx = torch.zeros(
                            curr_sample_size,
                            dtype=torch.long,
                            device=queries.device,
                        )

                        midx = torch.ones(
                            curr_sample_size,
                            dtype=torch.bool,
                            device=queries.device,
                        )

                        idx[:pcd_size] = torch.arange(pcd_size, device=queries.device)

                        midx[:pcd_size] = False  # attend to first points
                    else:
                        # we have more points in pcd as we like to sample
                        # take a subset (no padding or masking needed)
                        idx = torch.randperm(feature_map_level_features[k].shape[0], device=queries.device)[:curr_sample_size]
                        midx = torch.zeros(
                            curr_sample_size,
                            dtype=torch.bool,
                            device=queries.device,
                        )  # attend to all

                    item_used_point_idxs.append(idx)
                    mask_idx.append(midx)

                batched_aux = torch.stack(
                    [
                        feature_map_level_features[item_idx_in_batch][item_used_point_idxs[item_idx_in_batch], :]
                        for item_idx_in_batch in range(len(item_used_point_idxs))
                    ]
                )

                batched_attn_mask = torch.stack(
                    [
                        decomposed_attn[item_idx_in_batch][item_used_point_idxs[item_idx_in_batch], :]
                        for item_idx_in_batch in range(len(item_used_point_idxs))
                    ]
                )
                batched_attn_mask.permute((0, 2, 1))[batched_attn_mask.sum(1) == item_used_point_idxs[0].shape[0]] = False
                m = torch.stack(mask_idx)
                batched_attn_mask = torch.logical_or(batched_attn_mask, m[..., None])

                batched_pos_enc = torch.stack(
                    [
                        pos_encodings_pcd[hlevel][0][item_idx_in_batch][item_used_point_idxs[item_idx_in_batch], :]
                        for item_idx_in_batch in range(len(item_used_point_idxs))
                    ]
                )

                src_pcd = self.lin_squeeze[decoder_counter][item_idx](batched_aux.permute((1, 0, 2)))
                if self.use_level_embed:
                    src_pcd += self.level_embed.weight[item_idx]

                output = self.cross_attention[decoder_counter][item_idx](
                    queries.permute((1, 0, 2)),
                    src_pcd,
                    memory_mask=batched_attn_mask.repeat_interleave(self.num_heads, dim=0).permute((0, 2, 1)),
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=batched_pos_enc.permute((1, 0, 2)),
                    query_pos=query_pos,
                )

                output = self.self_attention[decoder_counter][item_idx](
                    output,
                    tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_pos,
                )

                # FFN
                queries = self.ffn_attention[decoder_counter][item_idx](output).permute((1, 0, 2))

                # Append output of this decoder item
                predictions_class.append(output_class)
                predictions_mask.append(outputs_mask)
            
            measure_runtime.add_timing(f"model_forward_decoder_{decoder_num}")
            decoder_num += 1

        output_class, outputs_mask, _ = self.mask_module(
            queries,
            mask_features,
            0,
            ret_attn_mask=False,
            coords=coordinates,
        )
        measure_runtime.add_timing("model_forward_final_mask_module")
        # Append output of the final decoder item
        predictions_class.append(output_class)
        predictions_mask.append(outputs_mask)

        sampled_coords_npy: Union[np.ndarray, None] = sampled_coords.detach().cpu().numpy() if sampled_coords is not None else None

        return ModelOutput(
            pred_class=predictions_class[-1],
            pred_class_all_decoders=predictions_class,
            pred_mask=predictions_mask[-1],
            pred_mask_all_decoders=predictions_mask,
            sampled_coords=sampled_coords_npy,
            backbone_features=backbone_out,
        )

    def mask_module(
        self,
        query_feat,
        mask_features,
        num_pooling_steps,
        ret_attn_mask=True,
        coords=None,
    ):
        query_feat = self.decoder_norm(query_feat)
        mask_embed = self.mask_embed_head(query_feat)
        outputs_class = self.class_embed_head(query_feat)

        output_masks = []

        for i in range(mask_features.C[-1, 0] + 1):
            output_masks.append(mask_features.decomposed_features[i] @ mask_embed[i].T)

        output_masks = torch.cat(output_masks)
        outputs_mask = me.SparseTensor(
            features=output_masks,
            coordinate_manager=mask_features.coordinate_manager,
            coordinate_map_key=mask_features.coordinate_map_key,
        )

        if ret_attn_mask:
            attn_mask = outputs_mask
            for _ in range(num_pooling_steps):
                attn_mask = self.pooling(attn_mask.float())

            attn_mask = me.SparseTensor(
                features=(attn_mask.F.detach().sigmoid() < 0.5),
                coordinate_manager=attn_mask.coordinate_manager,
                coordinate_map_key=attn_mask.coordinate_map_key,
            )

            return (
                outputs_class,
                outputs_mask.decomposed_features,
                attn_mask,
            )

        return outputs_class, outputs_mask.decomposed_features, None


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        self.orig_ch = channels
        super(PositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor, input_range=None):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        pos_x, pos_y, pos_z = tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2]
        sin_inp_x = torch.einsum("bi,j->bij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("bi,j->bij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("bi,j->bij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)

        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)

        emb = torch.cat((emb_x, emb_y, emb_z), dim=-1)
        return emb[:, :, : self.orig_ch].permute((0, 2, 1))


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        tgt2 = self.norm(tgt)

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                memory_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
