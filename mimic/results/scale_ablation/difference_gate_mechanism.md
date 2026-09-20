# Why the four Difference-branch section gates stay bit-identical to initialisation

**Observed.** In all six MG-G2L checkpoints of the scale ablation (attempt 2,
jobs 115261–115272), comparing `checkpoint_best.pth` against a fresh model
constructed under the same seed:

| branch | gates | result |
|---|---|---|
| `synergy_branch.co_attn_layers.{0,1}.section_{find,imp}_gate_weights` | 4 | **moved** from init, max abs diff 2.66–2.90 |
| `difference_branch.co_attn_layers.{0,1}.section_{find,imp}_gate_weights` | 4 | **bit-identical** to init, max abs diff exactly 0.0 |

4/4 and 4/4 in every one of the six runs. The same pattern was seen earlier in
the ReXGradient MG-G2L checkpoints, where the section mechanism was inert for a
different reason, so it is not specific to one dataset or to one training regime.

**Mechanism.** It follows from a single asymmetry in `MultimodalFusion.forward`
(`base_models_refactored_v1_paper2.py`). The synergy branch is called with the
section arguments; the difference branch is called without them:

```python
if findings_token_count is not None:
    synergy_img_emb, synergy_txt_emb, h_find_pooled_proj, h_imp_pooled_proj = self.synergy_branch(
        image_tokens, text_tokens,
        findings_token_count=findings_token_count, has_find=has_find, has_imp=has_imp,
        token_ids=token_ids, debug=debug,
    )
else:
    synergy_img_emb, synergy_txt_emb = self.synergy_branch(image_tokens, text_tokens)
    h_find_pooled_proj, h_imp_pooled_proj = None, None

diff_img_emb, diff_txt_emb = self.difference_branch(image_tokens, text_tokens)   # no section args
```

Inside `HierarchicalCoAttention.forward`, the section path is gated on
`findings_token_count is not None`:

```python
if findings_token_count is None:
    # ORIGINAL path -- byte-for-byte unchanged
    text_tokens = text_tokens + global_text_token.expand(-1, text_tokens.size(1), -1)
else:
    section_pools = self.compute_section_pools(text_tokens, token_ids, findings_token_count)
    G_find, G_imp, G_report = self.compute_section_global_attention(section_pools, image_tokens)
    real_length = (token_ids != 0).sum(dim=1)
    text_tokens, _, _ = self.section_aware_feedback(
        text_tokens, G_find, G_imp, G_report, findings_token_count, real_length, has_find, has_imp
    )
```

`section_find_gate_weights` and `section_imp_gate_weights` are read **only**
inside `section_aware_feedback`. For the difference branch that function is never
invoked, so the chain is:

1. no section args → `findings_token_count is None` → the original path is taken;
2. `section_aware_feedback` is never called, so the two gate Parameters are never
   read in the forward pass;
3. they are therefore not nodes in the autograd graph, and `backward()` leaves
   their `.grad` as `None`;
4. `torch.optim.Adam` skips any parameter whose `.grad` is `None`
   (`if p.grad is None: continue`), so no update is applied — **including weight
   decay**, which Adam implements by adding `wd * p` to an existing gradient
   rather than as a separate step.

Hence the difference-branch gates retain their `torch.randn(embed_dim)`
initialisation exactly, for the whole of training, at any number of epochs.

**Consequences to state in Methods.**

- MG-G2L's section-aware feedback is active in the **synergy branch only**. The
  difference branch is architecturally identical to Paper 1's.
- The 8 extra tensors in an MG-G2L checkpoint (289 vs Paper 1's 281) are
  therefore **4 trained parameters and 4 untouched random initialisations**. The
  effective additional parameter count is half of what the tensor count suggests.
- This is a property of the wiring, not a bug that changed any reported result:
  the difference branch behaves exactly as Paper 1's throughout, which is what
  the backward-compatibility tests assert.
- A useful ablation, if ever wanted, is to pass the section args to the
  difference branch as well and see whether the gain grows — untested here.

**Diagnostic that produced the table:**
`mimic/scripts/preflight_scale_ablation_attempt2.py` (step 2.3), job 115347.
