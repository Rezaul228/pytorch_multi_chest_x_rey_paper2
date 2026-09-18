#!/usr/bin/env python3
"""
Guards that make an inert MG-G2L section mechanism a hard failure.

Added 2026-09-18 after a ReXGradient MG-G2L arm was found to have trained for
100 epochs with has_find/has_imp all-False: the granularity loss was exactly
0.000000 on every logged epoch line, section_aware_feedback() applied no
gating, and the run silently degenerated towards the Paper 1 architecture while
still being reported as "MG-G2L".

Imported by train_retrieval_v2_paper2.py only -- the Paper 1 trainer
(train_retrieval_v2.py / data_loader_v1.py) has no section code path and is
completely unaffected. Kept dependency-free (no torch/numpy) so it can be
unit-tested in milliseconds.
"""

# ---------------------------------------------------------------------------
# NEW (2026-09-18): make an inert section mechanism a hard failure, not a warning.
# Both helpers are used ONLY on the section-aware (paper2) path, so Paper 1 runs
# via train_retrieval_v2.py / data_loader_v1.py are completely unaffected.
# ---------------------------------------------------------------------------
SECTION_INERT_CAUSE = (
    "Likely cause: the section-boundary CSV for this dataset_mode was not found or "
    "matched no study_ids, so every sample fell back to has_find=False/has_imp=False. "
    "compute_granularity_loss() then has no positive pairs and returns exactly 0.0, "
    "and section_aware_feedback() applies no section gating -- i.e. MG-G2L silently "
    "degenerates towards the Paper 1 architecture. Check the 'Loading section "
    "boundaries ONCE from:' line and paths.get_section_boundaries_path()."
)


def assert_section_mechanism_active(epoch, epoch_gran_losses):
    """Raise if the granularity loss was 0.0 for EVERY step of this epoch."""
    n_nonzero = sum(1 for g in epoch_gran_losses if float(g) != 0.0)
    if n_nonzero == 0:
        raise RuntimeError(
            f"SECTION MECHANISM INERT: granularity loss was exactly 0.0 for all "
            f"{len(epoch_gran_losses)} steps of epoch {epoch + 1}. " + SECTION_INERT_CAUSE
        )
    return n_nonzero


def assert_section_boundaries_loaded(dataset, split_name):
    """Raise if a section-aware dataset resolved no boundary entries, or if every
    sample missed the lookup. Called once per split, right after construction."""
    boundaries = getattr(dataset, "section_boundaries", None)
    if boundaries is None:
        raise RuntimeError(
            f"SECTION BOUNDARIES MISSING for split='{split_name}': dataset has no "
            f"section_boundaries attribute. " + SECTION_INERT_CAUSE
        )
    if len(boundaries) == 0:
        raise FileNotFoundError(
            f"SECTION BOUNDARIES EMPTY for split='{split_name}': 0 entries loaded "
            f"(dataset has {len(dataset)} samples). " + SECTION_INERT_CAUSE
        )
    samples = getattr(dataset, "samples", [])
    n_checked = min(len(samples), 2000)
    n_hit = sum(1 for sm in samples[:n_checked] if str(sm["study_ids"]) in boundaries)
    if n_checked and n_hit == 0:
        raise RuntimeError(
            f"SECTION BOUNDARIES DO NOT MATCH for split='{split_name}': 0/{n_checked} "
            f"sampled study_ids found among {len(boundaries)} boundary entries. "
            + SECTION_INERT_CAUSE
        )
    print(f"   [section guard] split='{split_name}': {len(boundaries)} boundary entries, "
          f"{n_hit}/{n_checked} sampled study_ids matched")
    return len(boundaries)
