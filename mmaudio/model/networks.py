import logging
from dataclasses import dataclass
from typing import Optional, Union
import math
import random
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmaudio.ext.rotary_embeddings import compute_rope_rotations
from mmaudio.model.embeddings import TimestepEmbedder
from mmaudio.model.low_level import MLP, ChannelLastConv1d, ConvMLP
from mmaudio.model.transformer_layers import (FinalBlock, JointBlock, MMDitSingleBlock)

log = logging.getLogger()


@dataclass
class PreprocessedConditions:
    clip_f: torch.Tensor
    sync_f: torch.Tensor
    text_f: torch.Tensor
    clip_f_c: torch.Tensor
    text_f_c: torch.Tensor
    extra_cond: Optional[torch.Tensor]


# Partially from https://github.com/facebookresearch/DiT
class MMAudio(nn.Module):

    def __init__(self,
                 *,
                 latent_dim: int,
                 clip_dim: int,
                 sync_dim: int,
                 text_dim: int,
                 hidden_dim: int,
                 depth: int,
                 fused_depth: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 latent_seq_len: int,
                 clip_seq_len: int,
                 sync_seq_len: int,
                 text_seq_len: int = 77,
                 latent_mean: Optional[torch.Tensor] = None,
                 latent_std: Optional[torch.Tensor] = None,
                 empty_string_feat: Optional[torch.Tensor] = None,
                 v2: bool = False,
                 extra_condition_info: Optional[dict] = None) -> None:
        super().__init__()

        self.v2 = v2
        self.latent_dim = latent_dim
        self._latent_seq_len = latent_seq_len
        self._clip_seq_len = clip_seq_len
        self._sync_seq_len = sync_seq_len
        self._text_seq_len = text_seq_len
        latent_downsample_rate = 2 # workaround
        self._extra_cond_seq_len = latent_downsample_rate * latent_seq_len if latent_downsample_rate is not None else None
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        if v2:
            self.audio_input_proj = nn.Sequential(
                ChannelLastConv1d(latent_dim, hidden_dim, kernel_size=7, padding=3),
                nn.SiLU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=7, padding=3),
            )

            self.clip_input_proj = nn.Sequential(
                nn.Linear(clip_dim, hidden_dim),
                nn.SiLU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            )

            self.sync_input_proj = nn.Sequential(
                ChannelLastConv1d(sync_dim, hidden_dim, kernel_size=7, padding=3),
                nn.SiLU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            )

            self.text_input_proj = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.SiLU(),
                MLP(hidden_dim, hidden_dim * 4),
            )
        else:
            self.audio_input_proj = nn.Sequential(
                ChannelLastConv1d(latent_dim, hidden_dim, kernel_size=7, padding=3),
                nn.SELU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=7, padding=3),
            )

            self.clip_input_proj = nn.Sequential(
                nn.Linear(clip_dim, hidden_dim),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            )

            self.sync_input_proj = nn.Sequential(
                ChannelLastConv1d(sync_dim, hidden_dim, kernel_size=7, padding=3),
                nn.SELU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            )

            self.text_input_proj = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                MLP(hidden_dim, hidden_dim * 4),
            )

        self.clip_cond_proj = nn.Linear(hidden_dim, hidden_dim)
        self.text_cond_proj = nn.Linear(hidden_dim, hidden_dim)
        self.global_cond_mlp = MLP(hidden_dim, hidden_dim * 4)
        # each synchformer output segment has 8 feature frames
        self.sync_pos_emb = nn.Parameter(torch.zeros((1, 1, 8, sync_dim)))

        self.final_layer = FinalBlock(hidden_dim, latent_dim)

        if v2:
            self.t_embed = TimestepEmbedder(hidden_dim,
                                            frequency_embedding_size=hidden_dim,
                                            max_period=1)
        else:
            self.t_embed = TimestepEmbedder(hidden_dim,
                                            frequency_embedding_size=256,
                                            max_period=10000)
        self.joint_blocks = nn.ModuleList([
            JointBlock(hidden_dim,
                       num_heads,
                       mlp_ratio=mlp_ratio,
                       pre_only=(i == depth - fused_depth - 1)) for i in range(depth - fused_depth)
        ])

        self.fused_blocks = nn.ModuleList([
            MMDitSingleBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, kernel_size=3, padding=1)
            for i in range(fused_depth)
        ])

        if latent_mean is None:
            # these values are not meant to be used
            # if you don't provide mean/std here, we should load them later from a checkpoint
            assert latent_std is None
            latent_mean = torch.ones(latent_dim).view(1, 1, -1).fill_(float('nan'))
            latent_std = torch.ones(latent_dim).view(1, 1, -1).fill_(float('nan'))
        else:
            assert latent_std is not None
            assert latent_mean.numel() == latent_dim, f'{latent_mean.numel()=} != {latent_dim=}'
        if empty_string_feat is None:
            empty_string_feat = torch.zeros((text_seq_len, text_dim))
        self.latent_mean = nn.Parameter(latent_mean.view(1, 1, -1), requires_grad=False)
        self.latent_std = nn.Parameter(latent_std.view(1, 1, -1), requires_grad=False)

        self.empty_string_feat = nn.Parameter(empty_string_feat, requires_grad=False)
        self.empty_clip_feat = nn.Parameter(torch.zeros(1, clip_dim), requires_grad=True)
        self.empty_sync_feat = nn.Parameter(torch.zeros(1, sync_dim), requires_grad=True)

        ## Extra conditions
        # extra_condition_info should include at least cond_type and cond_shape.
        if extra_condition_info is not None:
            if extra_condition_info["cond_type"] == "add":
                assert type(extra_condition_info["cond_dim"]) is int
                self.extra_cond_proj = nn.Linear(extra_condition_info["cond_dim"], latent_dim)
            elif extra_condition_info["cond_type"] == "mask_and_add":
                assert type(extra_condition_info["cond_dim"]) is int
                self.extra_cond_proj = nn.Linear(extra_condition_info["cond_dim"] * 2, latent_dim) ## simple linear
                self.mask_level_max = int(math.log2(extra_condition_info["cond_dim"] + 1))
            elif extra_condition_info["cond_type"] == "mask_and_add_with_sync_mod":
                assert type(extra_condition_info["cond_dim"]) is int
                self.extra_cond_proj = nn.Linear(extra_condition_info["cond_dim"] * 2, latent_dim) ## simple linear
                self.extra_cond_proj_for_sync = nn.Sequential(
                    nn.Linear(extra_condition_info["cond_dim"] * 2, hidden_dim),
                    nn.SiLU(),
                    MLP(hidden_dim, hidden_dim * 4)
                )
                self.mask_level_max = int(math.log2(extra_condition_info["cond_dim"] + 1))
            else:
                # unknown extra condition type
                raise ValueError(f"Unknown extra condition type: {extra_condition_info['cond_type']}")
            
        self.extra_condition_info = extra_condition_info

        self.initialize_weights()
        self.initialize_rotations()

    def initialize_rotations(self):
        base_freq = 1.0
        latent_rot = compute_rope_rotations(self._latent_seq_len,
                                            self.hidden_dim // self.num_heads,
                                            10000,
                                            freq_scaling=base_freq,
                                            device=self.device)
        clip_rot = compute_rope_rotations(self._clip_seq_len,
                                          self.hidden_dim // self.num_heads,
                                          10000,
                                          freq_scaling=base_freq * self._latent_seq_len /
                                          self._clip_seq_len,
                                          device=self.device)

        self.latent_rot = nn.Buffer(latent_rot, persistent=False)
        self.clip_rot = nn.Buffer(clip_rot, persistent=False)

    def update_seq_lengths(self, latent_seq_len: int, clip_seq_len: int, sync_seq_len: int, latent_downsample_rate: Optional[int] = None) -> None:
        self._latent_seq_len = latent_seq_len
        self._clip_seq_len = clip_seq_len
        self._sync_seq_len = sync_seq_len
        self._extra_cond_seq_len = latent_downsample_rate * latent_seq_len if latent_downsample_rate is not None else None
        self.initialize_rotations()

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.joint_blocks:
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.clip_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.clip_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].bias, 0)
        for block in self.fused_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        nn.init.constant_(self.final_layer.conv.bias, 0)

        # empty string feat shall be initialized by a CLIP encoder
        nn.init.constant_(self.sync_pos_emb, 0)
        nn.init.constant_(self.empty_clip_feat, 0)
        nn.init.constant_(self.empty_sync_feat, 0)

        # zero-out extra cond proj
        if self.extra_condition_info is not None:
            if self.extra_condition_info["cond_type"] == "add" or self.extra_condition_info["cond_type"] == "mask_and_add" or self.extra_condition_info["cond_type"] == "mask_and_add_with_sync_mod":
                if isinstance(self.extra_cond_proj, nn.Linear):
                    nn.init.constant_(self.extra_cond_proj.weight, 0)
                    nn.init.constant_(self.extra_cond_proj.bias, 0)
                else:
                    nn.init.constant_(self.extra_cond_proj[1].w2.weight, 0)
                
            # zero-out extra cond proj for sync modulation
            if self.extra_condition_info["cond_type"] == "mask_and_add_with_sync_mod":
                nn.init.constant_(self.extra_cond_proj_for_sync[-1].w2.weight, 0)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # return (x - self.latent_mean) / self.latent_std
        return x.sub_(self.latent_mean).div_(self.latent_std)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        # return x * self.latent_std + self.latent_mean
        return x.mul_(self.latent_std).add_(self.latent_mean)

    def preprocess_conditions(self, clip_f: torch.Tensor, sync_f: torch.Tensor,
                              text_f: torch.Tensor, extra_cond: Optional[Union[torch.Tensor, list]] = None, aug_arg: Optional[dict] = None) -> PreprocessedConditions:
        """
        cache computations that do not depend on the latent/time step
        i.e., the features are reused over steps during inference
        """
        assert clip_f.shape[1] == self._clip_seq_len, f'{clip_f.shape=} {self._clip_seq_len=}'
        assert sync_f.shape[1] == self._sync_seq_len, f'{sync_f.shape=} {self._sync_seq_len=}'
        assert text_f.shape[1] == self._text_seq_len, f'{text_f.shape=} {self._text_seq_len=}'

        bs = clip_f.shape[0]

        # B * num_segments (24) * 8 * 768
        num_sync_segments = self._sync_seq_len // 8
        sync_f = sync_f.view(bs, num_sync_segments, 8, -1) + self.sync_pos_emb
        sync_f = sync_f.flatten(1, 2)  # (B, VN, D)

        # extend vf to match x
        clip_f = self.clip_input_proj(clip_f)  # (B, VN, D)
        sync_f = self.sync_input_proj(sync_f)  # (B, VN, D)
        text_f = self.text_input_proj(text_f)  # (B, VN, D)

        # upsample the sync features to match the audio
        sync_f = sync_f.transpose(1, 2)  # (B, D, VN)
        sync_f = F.interpolate(sync_f, size=self._latent_seq_len, mode='nearest-exact')
        sync_f = sync_f.transpose(1, 2)  # (B, N, D)

        # get conditional features from the clip side
        clip_f_c = self.clip_cond_proj(clip_f.mean(dim=1))  # (B, D)
        text_f_c = self.text_cond_proj(text_f.mean(dim=1))  # (B, D)

        # To handle mask level 0 properly
        if extra_cond is not None and aug_arg is not None and "mask_level" in aug_arg and aug_arg["mask_level"] == "random":
            prob_null = 0.1
            dice = random.random()
            if dice < prob_null:
                aug_arg["mask_level"] = 0

        # process extra condition
        if extra_cond is not None and not (aug_arg is not None and "mask_level" in aug_arg and aug_arg["mask_level"] == 0):
            assert self.extra_condition_info is not None, 'extra condition info should be given if used'

            # if extra_cond is a list of dict, each element should contain "extra_condition" and "start_pos".
            # aug_arg may contain "med_filter_size", "temporal_mask", and "mask_level".

            if aug_arg is not None and "temporal_mask" in aug_arg: # train (random) or inference mode
                if aug_arg["temporal_mask"] == "random":
                    assert type(extra_cond) is torch.Tensor
                    options = ["no_mask", "continuation", "crop"]
                    weights = [0.5, 0.25, 0.25]
                    mask_type = random.choices(options, weights=weights)[0]
                    # mask_type = "no_mask" # for now
                    extra_cond = self.temporal_augmentation(extra_cond, mask_type)
                elif aug_arg["temporal_mask"] == "condition":
                    assert type(extra_cond) is list
                    # We assume that extra_cond has been created through create_extra_condition
                    pass
                else:
                    raise
            else: # default mode (no temporal mask)
                if type(extra_cond) is torch.Tensor:
                    extra_cond = [{"extra_condition": extra_cond, "start_pos": 0}]

            if self.extra_condition_info["cond_type"] == "mask_and_add" or self.extra_condition_info["cond_type"] == "mask_and_add_with_sync_mod":
                mask = torch.zeros(bs, self.extra_condition_info["cond_dim"], self._extra_cond_seq_len, device=self.device, dtype=extra_cond[0]["extra_condition"].dtype)

            # Data augmentation
            if aug_arg is not None and "med_filter_size" in aug_arg:
                if aug_arg["med_filter_size"] == "random": # train
                    kernel_size = random.randrange(1, self.extra_condition_info["med_filter_range"] // 2 * 2 + 1, 2)
                elif type(aug_arg["med_filter_size"]) is int:
                    kernel_size = aug_arg["med_filter_size"] // 2 * 2 + 1
                    kernel_size = min(kernel_size, self.extra_condition_info["med_filter_range"] // 2 * 2 + 1)
                else:
                    raise
            else: # default setting (test)
                kernel_size = self.extra_condition_info["med_filter_range"] // 2 // 2 * 2 + 1
            if kernel_size > 1:
                for i in range(len(extra_cond)):
                    kernel_size_cur = min(kernel_size, extra_cond[i]["extra_condition"].shape[-1] - 1)
                    extra_cond[i]["extra_condition"] = extra_cond[i]["extra_condition"].unfold(-1, kernel_size_cur, 1).median(-1)[0]
                    extra_cond[i]["extra_condition"] = torch.nn.functional.pad(extra_cond[i]["extra_condition"], (kernel_size_cur // 2, ) * 2, 'replicate')

            # Merge extra conditions
            extra_cond_all = torch.zeros(bs, self.extra_condition_info["cond_dim"], self._extra_cond_seq_len, device=self.device, dtype=extra_cond[0]["extra_condition"].dtype)
            for i in range(len(extra_cond)):
                start_time = extra_cond[i]["start_pos"]
                end_time = start_time + extra_cond[i]["extra_condition"].shape[-1]
                if end_time > self._extra_cond_seq_len:
                    end_time = self._extra_cond_seq_len
                extra_cond_all[:, :, start_time:end_time] = extra_cond[i]["extra_condition"][:, :, :end_time - start_time]
                if self.extra_condition_info["cond_type"] == "mask_and_add" or self.extra_condition_info["cond_type"] == "mask_and_add_with_sync_mod":
                    mask[:, :, start_time:end_time] = 1

            # Temporal interpolation
            if self.extra_condition_info["cond_type"] == "add" or self.extra_condition_info["cond_type"] == "mask_and_add" or self.extra_condition_info["cond_type"] == "mask_and_add_with_sync_mod":
                extra_cond_all = torch.clamp(extra_cond_all, min=-12)
                extra_cond_all = F.interpolate(extra_cond_all, self._latent_seq_len)
                extra_cond_all = extra_cond_all.transpose(1, 2) # [B, C, T] -> [B, T, C]
                if self.extra_condition_info["cond_type"] == "mask_and_add" or self.extra_condition_info["cond_type"] == "mask_and_add_with_sync_mod":
                    mask = F.interpolate(mask, self._latent_seq_len)
                    mask = mask.transpose(1, 2) # [B, C, T] -> [B, T, C]
            
            # Concat mask
            if self.extra_condition_info["cond_type"] == "mask_and_add" or self.extra_condition_info["cond_type"] == "mask_and_add_with_sync_mod":
                # Set mask level
                if aug_arg is not None and "mask_level" in aug_arg:
                    if aug_arg["mask_level"] == "random":
                        exp = np.arange(0, -self.mask_level_max+1, -1) - 1
                        exp = np.append(exp, -self.mask_level_max+1)
                        prob_list = (2.0 ** exp).tolist()
                        mask_level = random.choices(list(range(1, self.mask_level_max + 1)), weights=prob_list)[0]
                    elif type(aug_arg["mask_level"]) is int:
                        assert 0 <= aug_arg["mask_level"] <= self.mask_level_max
                        mask_level = aug_arg["mask_level"]
                    else:
                        raise
                else: # default setting
                    mask_level = self.mask_level_max - 1
                
                # Create mask
                ind_mask = 2 ** mask_level - 1
                if mask_level == 0:
                    mask[:, :, :] = 0
                else:
                    mask[:, :, :-ind_mask] = 0

                extra_cond_all = extra_cond_all * mask
                extra_cond_all = torch.cat([extra_cond_all, mask], dim=2)

        else:
            extra_cond_all = None

        return PreprocessedConditions(clip_f=clip_f,
                                      sync_f=sync_f,
                                      text_f=text_f,
                                      clip_f_c=clip_f_c,
                                      text_f_c=text_f_c,
                                      extra_cond=extra_cond_all)

    def predict_flow(self, latent: torch.Tensor, t: torch.Tensor,
                     conditions: PreprocessedConditions) -> torch.Tensor:
        """
        for non-cacheable computations
        """
        assert latent.shape[1] == self._latent_seq_len, f'{latent.shape=} {self._latent_seq_len=}'

        clip_f = conditions.clip_f
        sync_f = conditions.sync_f
        text_f = conditions.text_f
        clip_f_c = conditions.clip_f_c
        text_f_c = conditions.text_f_c

        if conditions.extra_cond is not None:
            if self.extra_condition_info["cond_type"] == "add" or self.extra_condition_info["cond_type"] == "mask_and_add" or self.extra_condition_info["cond_type"] == "mask_and_add_with_sync_mod":
                latent = latent + self.extra_cond_proj(conditions.extra_cond)

            if self.extra_condition_info["cond_type"] == "mask_and_add_with_sync_mod":
                # modulate the sync features using additive way
                sync_f = sync_f + self.extra_cond_proj_for_sync(conditions.extra_cond)

        latent = self.audio_input_proj(latent)  # (B, N, D)
        global_c = self.global_cond_mlp(clip_f_c + text_f_c)  # (B, D)

        global_c = self.t_embed(t).unsqueeze(1) + global_c.unsqueeze(1)  # (B, D)
        extended_c = global_c + sync_f

        for block in self.joint_blocks:
            latent, clip_f, text_f = block(latent, clip_f, text_f, global_c, extended_c,
                                           self.latent_rot, self.clip_rot)  # (B, N, D)

        for block in self.fused_blocks:
            latent = block(latent, extended_c, self.latent_rot)

        flow = self.final_layer(latent, global_c)  # (B, N, out_dim), remove t
        return flow

    def forward(self, latent: torch.Tensor, clip_f: torch.Tensor, sync_f: torch.Tensor,
                text_f: torch.Tensor, t: torch.Tensor, extra_cond: Optional[Union[torch.Tensor, list]] = None, aug_arg: Optional[dict] = None) -> torch.Tensor:
        """
        latent: (B, N, C) 
        vf: (B, T, C_V)
        t: (B,)
        """
        conditions = self.preprocess_conditions(clip_f, sync_f, text_f, extra_cond, aug_arg)
        flow = self.predict_flow(latent, t, conditions)
        return flow

    def get_empty_string_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_string_feat.unsqueeze(0).expand(bs, -1, -1)

    def get_empty_clip_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_clip_feat.unsqueeze(0).expand(bs, self._clip_seq_len, -1)

    def get_empty_sync_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_sync_feat.unsqueeze(0).expand(bs, self._sync_seq_len, -1)

    def get_empty_conditions(
            self,
            bs: int,
            *,
            negative_text_features: Optional[torch.Tensor] = None) -> PreprocessedConditions:
        if negative_text_features is not None:
            empty_text = negative_text_features
        else:
            empty_text = self.get_empty_string_sequence(1)

        empty_clip = self.get_empty_clip_sequence(1)
        empty_sync = self.get_empty_sync_sequence(1)
        conditions = self.preprocess_conditions(empty_clip, empty_sync, empty_text)
        conditions.clip_f = conditions.clip_f.expand(bs, -1, -1)
        conditions.sync_f = conditions.sync_f.expand(bs, -1, -1)
        conditions.clip_f_c = conditions.clip_f_c.expand(bs, -1)
        if negative_text_features is None:
            conditions.text_f = conditions.text_f.expand(bs, -1, -1)
            conditions.text_f_c = conditions.text_f_c.expand(bs, -1)

        return conditions

    def ode_wrapper(self, t: torch.Tensor, latent: torch.Tensor, conditions: PreprocessedConditions,
                    empty_conditions: PreprocessedConditions, cfg_strength: float) -> torch.Tensor:
        t = t * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)

        if cfg_strength < 1.0:
            return self.predict_flow(latent, t, conditions)
        else:
            return (cfg_strength * self.predict_flow(latent, t, conditions) +
                    (1 - cfg_strength) * self.predict_flow(latent, t, empty_conditions))

    def ode_wrapper_dual(self, t: torch.Tensor, latent: torch.Tensor, conditions: PreprocessedConditions,
                    proxy_conditions: PreprocessedConditions, empty_conditions: PreprocessedConditions, cfg_strength_1: float, cfg_strength_2: float) -> torch.Tensor:
        t = t * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)

        guide_1 = self.predict_flow(latent, t, proxy_conditions) - self.predict_flow(latent, t, empty_conditions)
        guide_2 = self.predict_flow(latent, t, conditions) - self.predict_flow(latent, t, proxy_conditions)

        return ((cfg_strength_1 - 1.0) * guide_1 + (cfg_strength_2 - 1.0) * guide_2) + self.predict_flow(latent, t, empty_conditions)

    def load_weights(self, src_dict, strict=True) -> None:
        if 't_embed.freqs' in src_dict:
            del src_dict['t_embed.freqs']
        if 'latent_rot' in src_dict:
            del src_dict['latent_rot']
        if 'clip_rot' in src_dict:
            del src_dict['clip_rot']

        self.load_state_dict(src_dict, strict=strict)

    def temporal_augmentation(self, extra_cond: torch.Tensor, mask_type: str) -> torch.Tensor:
        # mask_type: no_mask, continuation, crop
        # no_mask: use all
        # continuation: use the first or the second part
        # crop: crop one or two parts
        extra_cond_list = []
        min_len = 0.1
        if mask_type == "no_mask": # use all
            extra_cond_list.append({"extra_condition": extra_cond, "start_pos": 0})
        elif mask_type == "continuation":
            norm_pos = random.uniform(min_len, 1.0 - min_len)
            which_side = random.choice([0, 1])
            if which_side == 0: # use the first part
                extra_cond = extra_cond[:, :, :int(norm_pos * extra_cond.shape[-1])]
                extra_cond_list.append({"extra_condition": extra_cond, "start_pos": 0})
            else: # use the second part
                extra_cond = extra_cond[:, :, int(norm_pos * extra_cond.shape[-1]):]
                extra_cond_list.append({"extra_condition": extra_cond, "start_pos": int(norm_pos * extra_cond.shape[-1])})
        elif mask_type == "crop":
            num_of_crop = random.randint(1, 2)
            while True:
                norm_pos = sorted([random.random() for _ in range(num_of_crop * 2)])
                # make sure the normalized length of each crop is larger than min_len
                flag = 0
                for i in range(num_of_crop):
                    if norm_pos[i*2+1] - norm_pos[i*2] < min_len:
                        flag += 1
                if flag > 0:
                    continue
                # convert the normalized position to the position in the extra condition
                denorm_pos = [int(norm_pos[i] * extra_cond.shape[-1]) for i in range(num_of_crop * 2)]
                for i in range(num_of_crop):
                    extra_cond_list.append({"extra_condition": extra_cond[:, :, denorm_pos[i*2]:denorm_pos[i*2+1]], "start_pos": denorm_pos[i*2]})
                break
        else:
            raise

        return extra_cond_list

    def create_extra_condition(self, extra_features: Union[torch.Tensor, list], start_norm_pos: Union[float, list]) -> torch.Tensor:
        if type(extra_features) is torch.Tensor:
            assert type(start_norm_pos) is float
            extra_features = [extra_features]
            start_norm_pos = [start_norm_pos]
        
        extra_cond_list = []
        for i in range(len(extra_features)):
            start_pos = int(start_norm_pos[i] * self._extra_cond_seq_len)
            extra_cond_list.append({"extra_condition": extra_features[i], "start_pos": start_pos})

        return extra_cond_list

    @property
    def device(self) -> torch.device:
        return self.latent_mean.device

    @property
    def latent_seq_len(self) -> int:
        return self._latent_seq_len

    @property
    def clip_seq_len(self) -> int:
        return self._clip_seq_len

    @property
    def sync_seq_len(self) -> int:
        return self._sync_seq_len


def small_16k(**kwargs) -> MMAudio:
    num_heads = 7
    return MMAudio(latent_dim=20,
                   clip_dim=1024,
                   sync_dim=768,
                   text_dim=1024,
                   hidden_dim=64 * num_heads,
                   depth=12,
                   fused_depth=8,
                   num_heads=num_heads,
                   latent_seq_len=250,
                   clip_seq_len=64,
                   sync_seq_len=192,
                   **kwargs)


def small_44k(**kwargs) -> MMAudio:
    num_heads = 7
    return MMAudio(latent_dim=40,
                   clip_dim=1024,
                   sync_dim=768,
                   text_dim=1024,
                   hidden_dim=64 * num_heads,
                   depth=12,
                   fused_depth=8,
                   num_heads=num_heads,
                   latent_seq_len=345,
                   clip_seq_len=64,
                   sync_seq_len=192,
                   **kwargs)


def medium_16k(**kwargs) -> MMAudio:
    num_heads = 14
    return MMAudio(latent_dim=20,
                   clip_dim=1024,
                   sync_dim=768,
                   text_dim=1024,
                   hidden_dim=64 * num_heads,
                   depth=12,
                   fused_depth=8,
                   num_heads=num_heads,
                   latent_seq_len=250,
                   clip_seq_len=64,
                   sync_seq_len=192,
                   **kwargs)


def medium_44k(**kwargs) -> MMAudio:
    num_heads = 14
    return MMAudio(latent_dim=40,
                   clip_dim=1024,
                   sync_dim=768,
                   text_dim=1024,
                   hidden_dim=64 * num_heads,
                   depth=12,
                   fused_depth=8,
                   num_heads=num_heads,
                   latent_seq_len=345,
                   clip_seq_len=64,
                   sync_seq_len=192,
                   **kwargs)


def large_44k(**kwargs) -> MMAudio:
    num_heads = 14
    return MMAudio(latent_dim=40,
                   clip_dim=1024,
                   sync_dim=768,
                   text_dim=1024,
                   hidden_dim=64 * num_heads,
                   depth=21,
                   fused_depth=14,
                   num_heads=num_heads,
                   latent_seq_len=345,
                   clip_seq_len=64,
                   sync_seq_len=192,
                   **kwargs)


def large_44k_v2(**kwargs) -> MMAudio:
    num_heads = 14
    return MMAudio(latent_dim=40,
                   clip_dim=1024,
                   sync_dim=768,
                   text_dim=1024,
                   hidden_dim=64 * num_heads,
                   depth=21,
                   fused_depth=14,
                   num_heads=num_heads,
                   latent_seq_len=345,
                   clip_seq_len=64,
                   sync_seq_len=192,
                   v2=True,
                   **kwargs)


def medium_w_loudness_16k(**kwargs) -> MMAudio:
    num_heads = 14
    # The following setting is just a workaround for the extra condition info. 
    # It should be determined depending on the setting of feature utils and the duration of video clip.
    extra_condition_info = {"cond_type": "mask_and_add_with_sync_mod", "cond_dim": 31, "med_filter_range": 39}
    if "extra_condition_info" in kwargs:
        extra_condition_info = kwargs["extra_condition_info"]
        del kwargs["extra_condition_info"]
    return MMAudio(latent_dim=20,
                   clip_dim=1024,
                   sync_dim=768,
                   text_dim=1024,
                   hidden_dim=64 * num_heads,
                   depth=12,
                   fused_depth=8,
                   num_heads=num_heads,
                   latent_seq_len=250,
                   clip_seq_len=64,
                   sync_seq_len=192,
                   extra_condition_info=extra_condition_info,
                   **kwargs)


def get_my_mmaudio(name: str, **kwargs) -> MMAudio:
    if name == 'small_16k':
        return small_16k(**kwargs)
    if name == 'medium_16k':
        return medium_16k(**kwargs)
    if name == 'small_44k':
        return small_44k(**kwargs)
    if name == 'medium_44k':
        return medium_44k(**kwargs)
    if name == 'large_44k':
        return large_44k(**kwargs)
    if name == 'large_44k_v2':
        return large_44k_v2(**kwargs)
    if name == 'medium_w_loudness_16k':
        return medium_w_loudness_16k(**kwargs)

    raise ValueError(f'Unknown model name: {name}')


if __name__ == '__main__':
    network = get_my_mmaudio('small_16k')

    # print the number of parameters in terms of millions
    num_params = sum(p.numel() for p in network.parameters()) / 1e6
    print(f'Number of parameters: {num_params:.2f}M')
