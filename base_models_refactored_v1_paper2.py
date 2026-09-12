#!/usr/bin/env python3
"""
Refactored PyTorch models to exactly match TensorFlow/Keras architecture
Base models for MIMIC-CXR Multimodal Retrieval - PyTorch Version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        
        # 4 distinct conv blocks matching TensorFlow exactly
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Dense layers matching TensorFlow naming
        self.image_encoder_patch_dense_1 = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.ReLU()
        )
        self.image_encoder_patch_dense_2 = nn.Linear(embed_dim, embed_dim)
        self.image_encoder_patch_norm_1 = nn.LayerNorm(embed_dim)
    
    def forward(self, x, training=True, verbose=False):
        if verbose:
            print(f"Image Encoder Input shape: {x.shape}")
        
        # Apply conv blocks sequentially
        x = self.conv_block_1(x)
        if verbose:
            print(f"After Conv Block 1: {x.shape}")
            
        x = self.conv_block_2(x)
        if verbose:
            print(f"After Conv Block 2: {x.shape}")
            
        x = self.conv_block_3(x)
        if verbose:
            print(f"After Conv Block 3: {x.shape}")
            
        x = self.conv_block_4(x)
        if verbose:
            print(f"After Conv Block 4: {x.shape}")
        
        # Flatten to (B, H*W, C) just like TensorFlow
        batch_size = x.size(0)
        h, w = x.size(2), x.size(3)
        patches = x.permute(0, 2, 3, 1).reshape(batch_size, h*w, -1)
        
        # Apply dense layers
        patches = self.image_encoder_patch_dense_1(patches)
        patches = self.image_encoder_patch_dense_2(patches)
        patches = self.image_encoder_patch_norm_1(patches)
        
        if verbose:
            print(f"Final patch embeddings: {patches.shape}")
        
        return patches


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, max_length=None, embed_dim=256):
        super().__init__()
        
        # Use max_length from config if not specified
        if max_length is None:
            max_length = config.get_max_token_length()
        
        # Embedding with padding_idx=0 for masking
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )
        
        # First LSTM: Bidirectional, hidden_size=256, dropout=0.5
        self.lstm1 = nn.LSTM(
            input_size=embed_dim,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )
        
        # Second LSTM: Bidirectional, hidden_size=embed_dim//2, dropout=0.5
        self.lstm2 = nn.LSTM(
            input_size=512,  # 256*2 from bidirectional LSTM1
            hidden_size=embed_dim//2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )
        
        # Dense layers matching TensorFlow naming
        self.text_encoder_token_dense_1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
        self.text_encoder_token_dense_2 = nn.Linear(embed_dim, embed_dim)
        self.text_encoder_token_norm_1 = nn.LayerNorm(embed_dim)
    
    def forward(self, x, training=True, verbose=False):
        if verbose:
            print(f"Text Encoder Input shape: {x.shape}")
        
        # Apply embedding
        x = self.embedding(x)
        
        # Apply first LSTM (bidirectional)
        x, _ = self.lstm1(x)
        
        # Apply second LSTM (bidirectional)
        x, _ = self.lstm2(x)
        
        # Apply dense layers
        x = self.text_encoder_token_dense_1(x)
        x = self.text_encoder_token_dense_2(x)
        x = self.text_encoder_token_norm_1(x)
        
        if verbose:
            print(f"Final token embeddings: {x.shape}")
        
        return x


class HierarchicalCoAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, instance_name="coattn"):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.instance_name = instance_name
        
        # Learnable gate weights (matching TensorFlow exactly)
        self.local_image_gate_weights = nn.Parameter(torch.randn(embed_dim))
        self.local_text_gate_weights = nn.Parameter(torch.randn(embed_dim))
        self.global_image_gate_weights = nn.Parameter(torch.randn(embed_dim))
        self.global_text_gate_weights = nn.Parameter(torch.randn(embed_dim))

        # NEW (paper2, section-aware feedback -- not wired into forward() yet)
        self.section_find_gate_weights = nn.Parameter(torch.randn(embed_dim))
        self.section_imp_gate_weights = nn.Parameter(torch.randn(embed_dim))

        # Multi-head attention layers
        self.cross_attention1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attention2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.global_cross_attention1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.global_cross_attention2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Layer normalization matching TensorFlow naming
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)
        
        self.global_norm1 = nn.LayerNorm(embed_dim)
        self.global_norm2 = nn.LayerNorm(embed_dim)
        self.global_norm3 = nn.LayerNorm(embed_dim)
        self.global_norm4 = nn.LayerNorm(embed_dim)
        
        # Feed-forward networks matching TensorFlow structure
        self.ffn1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.global_ffn1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.global_ffn2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    
    def forward(self, image_tokens, text_tokens, findings_token_count=None, has_find=None, has_imp=None,
                token_ids=None, debug=False):
        # NEW (paper2) optional args, all default None -> when findings_token_count
        # is None (Paper 1's original call signature: forward(image_tokens, text_tokens)),
        # the code below runs the ORIGINAL path, completely unchanged. token_ids is
        # also new/required alongside findings_token_count/has_find/has_imp: the three
        # already-verified helper methods this branches into
        # (compute_section_pools/compute_section_global_attention/section_aware_feedback)
        # need the raw token ids to detect padding (text_tokens itself is continuous
        # post-LSTM/LayerNorm and its padded positions aren't reliably zero), and
        # compute_section_pools's signature -- verified in the previous step --
        # takes token_ids as a required positional argument.
        if debug:
            # TEMPORARY debug print for this verification step only -- confirms section
            # info actually reaches this specific layer instance, rather than inferring
            # it indirectly from output differences.
            print(f"[DEBUG] HierarchicalCoAttention({self.instance_name}): "
                  f"findings_token_count is not None = {findings_token_count is not None}")

        if findings_token_count is not None:
            assert has_find is not None and has_imp is not None and token_ids is not None, (
                "has_find, has_imp, and token_ids must all be provided together with findings_token_count"
            )

        # Local cross-attention: Image -> Text
        attended_image, _ = self.cross_attention1(
            query=image_tokens,
            key=text_tokens,
            value=text_tokens
        )
        
        # Apply local image gate (matching TensorFlow sigmoid gating)
        local_image_gate = torch.sigmoid(self.local_image_gate_weights).view(1, 1, self.embed_dim)
        gated_image = local_image_gate * attended_image + (1 - local_image_gate) * image_tokens
        
        image_tokens = self.norm1(image_tokens + gated_image)
        image_tokens = self.norm2(image_tokens + self.ffn1(image_tokens))
        
        # Local cross-attention: Text -> Image
        attended_text, _ = self.cross_attention2(
            query=text_tokens,
            key=image_tokens,
            value=image_tokens
        )
        
        # Apply local text gate
        local_text_gate = torch.sigmoid(self.local_text_gate_weights).view(1, 1, self.embed_dim)
        gated_text = local_text_gate * attended_text + (1 - local_text_gate) * text_tokens
        
        text_tokens = self.norm3(text_tokens + gated_text)
        text_tokens = self.norm4(text_tokens + self.ffn2(text_tokens))
        
        # Global tokens (mean pooling)
        global_image_token = torch.mean(image_tokens, dim=1, keepdim=True)
        global_text_token = torch.mean(text_tokens, dim=1, keepdim=True)
        
        # Global cross-attention: Global Image -> Text
        attended_global_image, _ = self.global_cross_attention1(
            query=global_image_token,
            key=text_tokens,
            value=text_tokens
        )
        
        # Apply global image gate
        global_image_gate = torch.sigmoid(self.global_image_gate_weights).view(1, 1, self.embed_dim)
        gated_global_image = global_image_gate * attended_global_image + (1 - global_image_gate) * global_image_token
        
        global_image_token = self.global_norm1(global_image_token + gated_global_image)
        global_image_token = self.global_norm2(global_image_token + self.global_ffn1(global_image_token))
        
        # Global cross-attention: Global Text -> Image
        attended_global_text, _ = self.global_cross_attention2(
            query=global_text_token,
            key=image_tokens,
            value=image_tokens
        )
        
        # Apply global text gate
        global_text_gate = torch.sigmoid(self.global_text_gate_weights).view(1, 1, self.embed_dim)
        gated_global_text = global_text_gate * attended_global_text + (1 - global_text_gate) * global_text_token
        
        global_text_token = self.global_norm3(global_text_token + gated_global_text)
        global_text_token = self.global_norm4(global_text_token + self.global_ffn2(global_text_token))
        
        # Combine local and global features (broadcast global to all positions)
        # Image-side feedback is COMPLETELY unchanged in both cases.
        image_tokens = image_tokens + global_image_token.expand(-1, image_tokens.size(1), -1)

        if findings_token_count is None:
            # ORIGINAL path -- byte-for-byte unchanged (critical safety fallback).
            text_tokens = text_tokens + global_text_token.expand(-1, text_tokens.size(1), -1)
        else:
            # NEW (paper2) section-aware path.
            section_pools = self.compute_section_pools(text_tokens, token_ids, findings_token_count)
            G_find, G_imp, G_report = self.compute_section_global_attention(section_pools, image_tokens)
            real_length = (token_ids != 0).sum(dim=1)
            text_tokens, _, _ = self.section_aware_feedback(
                text_tokens, G_find, G_imp, G_report, findings_token_count, real_length, has_find, has_imp
            )

        return image_tokens, text_tokens

    def compute_section_pools(self, text_tokens, token_ids, findings_token_count, pad_token_id=0):
        """
        NOT wired into forward() yet. Computes three separate mean-pooled
        global text vectors from the same per-position text_tokens that
        forward() operates on, splitting each sample's real (non-padded)
        tokens into a Findings span and an Impression span:

          - global_text_find:   mean over text_tokens[:findings_token_count]
          - global_text_imp:    mean over text_tokens[findings_token_count:real_length]
          - global_text_report: torch.mean(text_tokens, dim=1, keepdim=True) --
            EXACTLY the old, unmasked behavior (averages over all seq_len
            positions, padding included). This is the "whole report"
            fallback and is kept bit-identical to Paper 1's original
            computation on purpose, unlike global_text_find/global_text_imp
            which are masked to real tokens only.

        A section with zero real tokens (e.g. findings_token_count == 0, or
        real_length <= findings_token_count meaning no room left for an
        Impression span after truncation) gets a zero vector for that pool,
        and its has_find / has_imp flag is False.

        Args:
            text_tokens: (B, L, D) float tensor -- same tensor forward() uses
                for global_text_token (i.e. after the local text->image
                cross-attention step).
            token_ids: (B, L) long tensor of the ORIGINAL raw token ids for
                this batch (before embedding). Needed because text_tokens
                itself is continuous (post-LSTM/LayerNorm) and its padded
                positions are not reliably zero, so padding must be detected
                from the raw ids, using pad_token_id (0, matching
                pad_sequences' default pad value) as the pad marker.
            findings_token_count: (B,) int tensor, one value per sample --
                from section_boundaries_test_paper2.csv, already capped to
                the sequence length used when the shards were built.
            pad_token_id: token id used for padding (default 0).

        Returns:
            dict with keys:
              'global_text_find':   (B, 1, D) float tensor
              'global_text_imp':    (B, 1, D) float tensor
              'global_text_report': (B, 1, D) float tensor
              'has_find': (B,) bool tensor
              'has_imp':  (B,) bool tensor
        """
        batch_size, seq_len, _ = text_tokens.shape
        device = text_tokens.device

        findings_token_count = findings_token_count.to(device=device, dtype=torch.long).clamp(min=0, max=seq_len)
        real_length = (token_ids.to(device) != pad_token_id).sum(dim=1).to(torch.long).clamp(max=seq_len)

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)

        find_mask = positions < findings_token_count.unsqueeze(1)
        imp_mask = (positions >= findings_token_count.unsqueeze(1)) & (positions < real_length.unsqueeze(1))

        has_find = findings_token_count > 0
        has_imp = (real_length - findings_token_count) > 0

        def masked_mean(mask):
            mask_f = mask.unsqueeze(-1).to(text_tokens.dtype)  # (B, L, 1)
            summed = (text_tokens * mask_f).sum(dim=1)  # (B, D)
            counts = mask.sum(dim=1).clamp(min=1).unsqueeze(-1).to(text_tokens.dtype)  # avoid div-by-zero
            return (summed / counts).unsqueeze(1)  # (B, 1, D)

        return {
            'global_text_find': masked_mean(find_mask),
            'global_text_imp': masked_mean(imp_mask),
            'global_text_report': torch.mean(text_tokens, dim=1, keepdim=True),
            'has_find': has_find,
            'has_imp': has_imp,
        }

    def _global_text_attend_to_image(self, query, image_tokens):
        """
        Exactly the "Global Text -> Image" block from forward() (the
        global_cross_attention2 / global_text_gate_weights / global_norm3 /
        global_norm4 / global_ffn2 sequence), factored out unchanged so it
        can be reused for global_text_find / global_text_imp /
        global_text_report without duplicating logic or creating new layers.

        Args:
            query: (N, 1, D) -- N may be the full batch or a subset of it.
            image_tokens: (N, L_img, D) -- same N as query.
        Returns:
            (N, 1, D)
        """
        attended_global_text, _ = self.global_cross_attention2(
            query=query,
            key=image_tokens,
            value=image_tokens
        )

        global_text_gate = torch.sigmoid(self.global_text_gate_weights).view(1, 1, self.embed_dim)
        gated_global_text = global_text_gate * attended_global_text + (1 - global_text_gate) * query

        out = self.global_norm3(query + gated_global_text)
        out = self.global_norm4(out + self.global_ffn2(out))
        return out

    def compute_section_global_attention(self, section_pools, image_tokens):
        """
        NOT wired into forward() yet. Runs the same "Global Text -> Image"
        attention block (global_cross_attention2 + global_text_gate_weights +
        global_norm3/4 + global_ffn2 -- all reused, no new layers) separately
        for each of the three pooled vectors from compute_section_pools().

        Samples with has_find=False (or has_imp=False) skip attention
        entirely for that section -- their query is a meaningless zero
        vector, so running it through attention/gates would produce a
        non-zero but meaningless value. Those rows are left as exact zeros
        instead, computed only for the valid subset of the batch.

        global_text_report has no such validity flag (the whole-report
        fallback is always defined), so it is always computed on the full
        batch -- and since global_text_report is bit-identical to the old
        unmasked torch.mean(text_tokens, dim=1), and _global_text_attend_to_image
        reuses the exact same layers/equations as forward()'s original
        "Global Text -> Image" block, G_report is bit-identical to what
        forward() would have produced for global_text_token at that step.

        Args:
            section_pools: dict returned by compute_section_pools() --
                needs 'global_text_find', 'global_text_imp',
                'global_text_report', 'has_find', 'has_imp'.
            image_tokens: (B, L_img, D) -- same image_tokens forward() uses
                for global_cross_attention2 (i.e. after the local
                image<->text cross-attention step).

        Returns:
            (G_find, G_imp, G_report), each (B, 1, D).
        """
        global_text_find = section_pools['global_text_find']
        global_text_imp = section_pools['global_text_imp']
        global_text_report = section_pools['global_text_report']
        has_find = section_pools['has_find']
        has_imp = section_pools['has_imp']

        G_find = torch.zeros_like(global_text_find)
        find_idx = has_find.nonzero(as_tuple=True)[0]
        if find_idx.numel() > 0:
            G_find[find_idx] = self._global_text_attend_to_image(
                global_text_find[find_idx], image_tokens[find_idx]
            )

        G_imp = torch.zeros_like(global_text_imp)
        imp_idx = has_imp.nonzero(as_tuple=True)[0]
        if imp_idx.numel() > 0:
            G_imp[imp_idx] = self._global_text_attend_to_image(
                global_text_imp[imp_idx], image_tokens[imp_idx]
            )

        # No validity flag for the whole-report fallback -- always computed on the full batch.
        G_report = self._global_text_attend_to_image(global_text_report, image_tokens)

        return G_find, G_imp, G_report

    def section_aware_feedback(self, text_tokens, G_find, G_imp, G_report,
                                findings_token_count, real_length, has_find, has_imp):
        """
        NOT wired into forward() yet. Per-sample, per-position j:

            mask_find(j) = 1 if (j < findings_token_count AND has_find) else 0
            mask_imp(j)  = 1 if (findings_token_count <= j < real_length AND has_imp) else 0

            output[j] = text_tokens[j] + G_report
                        + mask_find(j) * sigmoid(section_find_gate_weights)  * G_find
                        + mask_imp(j)  * sigmoid(section_imp_gate_weights)   * G_imp

        Positions with j >= real_length (padding) get only "+ G_report" --
        mask_find/mask_imp are zero there by construction (real_length is
        the upper bound for both masks), so this matches Paper 1's original
        feedback broadcast exactly at padded positions too.

        Backward-compatibility guarantee: if has_find and has_imp are both
        all-False (i.e. "no section info available"), mask_find and
        mask_imp are zero for every position regardless of
        findings_token_count/real_length, and this reduces EXACTLY to
        Paper 1's original text_tokens + global_text_token.expand(...)
        (since G_report is bit-identical to the old global_text_token at
        this point in the pipeline) -- see test_section_aware_feedback_paper2.py.

        Args:
            text_tokens: (B, L, D)
            G_find, G_imp, G_report: (B, 1, D) each
            findings_token_count: (B,) long
            real_length: (B,) long
            has_find, has_imp: (B,) bool

        Returns:
            (output, mask_find, mask_imp) where output is (B, L, D) and
            mask_find/mask_imp are (B, L) bool (returned for inspection/testing).
        """
        batch_size, seq_len, embed_dim = text_tokens.shape
        device = text_tokens.device

        findings_token_count = findings_token_count.to(device=device, dtype=torch.long).clamp(min=0, max=seq_len)
        real_length = real_length.to(device=device, dtype=torch.long).clamp(min=0, max=seq_len)
        has_find = has_find.to(device=device)
        has_imp = has_imp.to(device=device)

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)  # (B, L)

        mask_find = (positions < findings_token_count.unsqueeze(1)) & has_find.view(-1, 1)
        mask_imp = (
            (positions >= findings_token_count.unsqueeze(1))
            & (positions < real_length.unsqueeze(1))
            & has_imp.view(-1, 1)
        )

        mask_find_f = mask_find.unsqueeze(-1).to(text_tokens.dtype)  # (B, L, 1)
        mask_imp_f = mask_imp.unsqueeze(-1).to(text_tokens.dtype)

        find_gate = torch.sigmoid(self.section_find_gate_weights).view(1, 1, embed_dim)
        imp_gate = torch.sigmoid(self.section_imp_gate_weights).view(1, 1, embed_dim)

        output = (
            text_tokens
            + G_report.expand(-1, seq_len, -1)
            + mask_find_f * find_gate * G_find.expand(-1, seq_len, -1)
            + mask_imp_f * imp_gate * G_imp.expand(-1, seq_len, -1)
        )

        return output, mask_find, mask_imp


class BranchEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, name="branch"):
        super().__init__()
        self.name = name
        
        # Co-attention layers
        self.co_attn_layers = nn.ModuleList([
            HierarchicalCoAttention(embed_dim, num_heads, instance_name=f"{name}_coattn_{i+1}") 
            for i in range(num_layers)
        ])
        
        # Final projection layers with TensorFlow-style naming
        self.image_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
    
    def forward(self, image_tokens, text_tokens, findings_token_count=None, has_find=None, has_imp=None,
                token_ids=None, debug=False):
        # Process through co-attention layers -- new (paper2) optional args passed through unchanged
        # to every layer; all default None, so an unmodified call behaves exactly as before.
        for layer in self.co_attn_layers:
            image_tokens, text_tokens = layer(
                image_tokens, text_tokens,
                findings_token_count=findings_token_count, has_find=has_find, has_imp=has_imp,
                token_ids=token_ids, debug=debug,
            )

        # Global pooling (mean reduction)
        image_emb = torch.mean(image_tokens, dim=1)
        text_emb = torch.mean(text_tokens, dim=1)
        
        # Final projections
        image_emb = self.image_proj(image_emb)
        text_emb = self.text_proj(text_emb)

        if findings_token_count is None:
            # ORIGINAL path -- byte-for-byte unchanged (backward compatible: same 2-tuple return).
            return image_emb, text_emb

        # NEW (paper2): pool Findings/Impression from the FINAL text_tokens (post all
        # co_attn_layers blocks), reusing the already-verified compute_section_pools logic.
        # compute_section_pools doesn't reference any layer-specific weights, so it's
        # safe/equivalent to call it via any one of this branch's co-attention layer
        # instances -- no new pooling logic, no new weights.
        section_pools = self.co_attn_layers[0].compute_section_pools(text_tokens, token_ids, findings_token_count)
        h_find_pooled = section_pools['global_text_find'].squeeze(1)  # (B, D)
        h_imp_pooled = section_pools['global_text_imp'].squeeze(1)    # (B, D)

        # SAME text_proj used for the normal branch text embedding above -- no new projection weights.
        h_find_pooled_proj = self.text_proj(h_find_pooled)
        h_imp_pooled_proj = self.text_proj(h_imp_pooled)

        return image_emb, text_emb, h_find_pooled_proj, h_imp_pooled_proj


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, _, embeddings):
        """
        Calculate the contrastive loss between image and text embeddings
        Args:
            _: Unused parameter for compatibility
            embeddings: Tuple of (image_embeddings, text_embeddings)
                       Each should be of shape (batch_size, embed_dim)
        """
        image_embeddings, text_embeddings = embeddings
        
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        
        # Calculate similarity matrix
        similarity_matrix = torch.matmul(image_embeddings, text_embeddings.t())
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create labels (diagonal is positive pairs)
        batch_size = image_embeddings.size(0)
        labels = torch.arange(batch_size, device=image_embeddings.device)
        
        # Calculate loss in both directions (image->text and text->image)
        loss_i2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2i = F.cross_entropy(similarity_matrix.t(), labels)
        
        # Average both directions
        total_loss = (loss_i2t + loss_t2i) / 2
        
        return total_loss


class SynergyLoss(nn.Module):
    """Loss for synergy branch: MAXIMIZE similarity between matching pairs"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, _, embeddings):
        """
        Calculate synergy loss - maximize similarity for positive pairs
        Args:
            _: Unused parameter for compatibility
            embeddings: Tuple of (image_embeddings, text_embeddings)
        """
        image_embeddings, text_embeddings = embeddings
        
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        
        # Calculate similarity matrix
        similarity_matrix = torch.matmul(image_embeddings, text_embeddings.t())
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create labels (diagonal is positive pairs)
        batch_size = image_embeddings.size(0)
        labels = torch.arange(batch_size, device=image_embeddings.device)
        
        # Standard contrastive loss (minimize this = maximize similarity)
        loss_i2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2i = F.cross_entropy(similarity_matrix.t(), labels)
        
        # Average both directions
        total_loss = (loss_i2t + loss_t2i) / 2
        
        return total_loss


class DifferenceLoss(nn.Module):
    """Loss for difference branch: MINIMIZE similarity between matching pairs"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, _, embeddings):
        """
        Calculate difference loss - minimize similarity for positive pairs
        Args:
            _: Unused parameter for compatibility
            embeddings: Tuple of (image_embeddings, text_embeddings)
        """
        image_embeddings, text_embeddings = embeddings
        
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        
        # Calculate similarity matrix
        similarity_matrix = torch.matmul(image_embeddings, text_embeddings.t())
        similarity_matrix = similarity_matrix / self.temperature
        
        
        # For difference branch: we want LOW similarity on diagonal
        # Create "anti-labels" - maximize similarity with WRONG pairs
        #batch_size = image_embeddings.size(0)
        
        # Create anti-labels (shift by 1: Image1 should match Text2, Image2 should match Text3, etc.)
        #anti_labels = torch.arange(1, batch_size + 1, device=image_embeddings.device) % batch_size
        
        # Anti-contrastive loss: maximize similarity with wrong pairs
        #loss_i2t = F.cross_entropy(similarity_matrix, anti_labels)
        #loss_t2i = F.cross_entropy(similarity_matrix.t(), anti_labels)
      
        batch_size = image_embeddings.size(0)
        labels = torch.arange(batch_size, device=image_embeddings.device)
        
        # Standard contrastive loss (minimize this = maximize similarity)
        loss_i2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2i = F.cross_entropy(similarity_matrix.t(), labels)
        
        
        # Average both directions
        total_loss = (loss_i2t + loss_t2i) / 2

        return total_loss


def _infonce_on_subset(z_img, z_text, mask, temperature):
    """
    Same InfoNCE logic as ContrastiveLoss/SynergyLoss/DifferenceLoss above
    (L2-normalize, similarity_matrix / temperature, symmetric cross-entropy
    with in-batch diagonal positives), restricted to the sub-batch selected
    by `mask`. Returns (loss_or_None, n_selected) -- None if fewer than 2
    samples are selected (in-batch negatives need >=2 samples to be meaningful).
    """
    idx = mask.nonzero(as_tuple=True)[0]
    n = idx.numel()
    if n < 2:
        return None, n

    img_sub = F.normalize(z_img[idx], p=2, dim=1)
    text_sub = F.normalize(z_text[idx], p=2, dim=1)

    similarity_matrix = torch.matmul(img_sub, text_sub.t()) / temperature
    labels = torch.arange(n, device=z_img.device)

    loss_i2t = F.cross_entropy(similarity_matrix, labels)
    loss_t2i = F.cross_entropy(similarity_matrix.t(), labels)

    return (loss_i2t + loss_t2i) / 2, n


def compute_granularity_loss(z_img, h_find_pooled_proj, h_imp_pooled_proj, has_find, has_imp, temperature=0.07):
    """
    NEW (paper2) standalone loss, NOT wired into the main training loop yet.

    Same InfoNCE logic used by ContrastiveLoss/SynergyLoss/DifferenceLoss
    (temperature=0.07 default, in-batch negatives), computed as two separate
    terms:
      - find_loss: InfoNCE(z_img, h_find_pooled_proj), restricted to the
        sub-batch where has_find=True.
      - imp_loss:  InfoNCE(z_img, h_imp_pooled_proj), restricted to the
        sub-batch where has_imp=True.

    If a sub-batch has fewer than 2 samples, that term is skipped (0.0
    contribution) rather than computed on a meaningless 0- or 1-sample
    in-batch-negatives matrix.

    Args:
        z_img: (B, D) image embedding (any consistent image embedding the
            caller wants to align these pooled text vectors against --
            e.g. final_image_emb).
        h_find_pooled_proj: (B, D) projected Findings-pooled text embedding
            (from BranchEncoder.forward()'s new return value).
        h_imp_pooled_proj: (B, D) projected Impression-pooled text embedding.
        has_find, has_imp: (B,) bool tensors.
        temperature: same default as the other losses in this file.

    Returns:
        L_gran (scalar tensor), find_loss (scalar tensor or 0.0),
        imp_loss (scalar tensor or 0.0), n_find (int), n_imp (int)
    """
    find_loss, n_find = _infonce_on_subset(z_img, h_find_pooled_proj, has_find, temperature)
    imp_loss, n_imp = _infonce_on_subset(z_img, h_imp_pooled_proj, has_imp, temperature)

    L_gran = z_img.new_tensor(0.0)
    if find_loss is not None:
        L_gran = L_gran + find_loss
    if imp_loss is not None:
        L_gran = L_gran + imp_loss

    return (
        L_gran,
        find_loss if find_loss is not None else 0.0,
        imp_loss if imp_loss is not None else 0.0,
        n_find,
        n_imp,
    )


class MultimodalFusion(nn.Module):
    def __init__(self, vocab_size, embed_dim=None, num_heads=None, num_layers=None):
        super().__init__()
        
        # Use config defaults if not specified
        if embed_dim is None:
            embed_dim = config.get_embed_dim()
        if num_heads is None:
            num_heads = config.get_current_config()["num_heads"]
        if num_layers is None:
            num_layers = config.get_current_config()["num_layers"]
        
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(vocab_size, embed_dim=embed_dim)
        
        self.synergy_branch = BranchEncoder(embed_dim, num_heads, num_layers, name="synergy")
        self.difference_branch = BranchEncoder(embed_dim, num_heads, num_layers, name="difference")
    
    def forward(self, inputs, training=False, verbose=False, return_branch_embeddings=False,
                findings_token_count=None, has_find=None, has_imp=None, token_ids=None, debug=False):
        images, texts = inputs

        # Get token embeddings from encoders
        image_tokens = self.image_encoder(images, training=training, verbose=verbose)
        text_tokens = self.text_encoder(texts, training=training, verbose=verbose)

        # NEW (paper2): section args go to the SYNERGY branch only. The Difference
        # branch's call is left completely unaffected by these new args (always the
        # original 2-arg call), per instructions.
        if findings_token_count is not None:
            synergy_img_emb, synergy_txt_emb, h_find_pooled_proj, h_imp_pooled_proj = self.synergy_branch(
                image_tokens, text_tokens,
                findings_token_count=findings_token_count, has_find=has_find, has_imp=has_imp,
                token_ids=token_ids, debug=debug,
            )
        else:
            synergy_img_emb, synergy_txt_emb = self.synergy_branch(image_tokens, text_tokens)
            h_find_pooled_proj, h_imp_pooled_proj = None, None

        diff_img_emb, diff_txt_emb = self.difference_branch(image_tokens, text_tokens)

        # Average and L2 normalize final embeddings (matching TensorFlow exactly)
        final_image_emb = F.normalize((synergy_img_emb + diff_img_emb) / 2, p=2, dim=-1)
        final_text_emb = F.normalize((synergy_txt_emb + diff_txt_emb) / 2, p=2, dim=-1)

        if findings_token_count is None:
            # ORIGINAL path -- byte-for-byte unchanged (backward compatible: same return arity).
            if return_branch_embeddings:
                return final_image_emb, final_text_emb, synergy_img_emb, synergy_txt_emb, diff_img_emb, diff_txt_emb
            else:
                return final_image_emb, final_text_emb

        # NEW (paper2): section info was provided -- L2-normalize the pooled
        # Findings/Impression projections the same way final_image_emb/final_text_emb are.
        z_find = F.normalize(h_find_pooled_proj, p=2, dim=-1)
        z_imp = F.normalize(h_imp_pooled_proj, p=2, dim=-1)

        if return_branch_embeddings:
            return (final_image_emb, final_text_emb, synergy_img_emb, synergy_txt_emb,
                    diff_img_emb, diff_txt_emb, z_find, z_imp)
        else:
            return final_image_emb, final_text_emb, z_find, z_imp