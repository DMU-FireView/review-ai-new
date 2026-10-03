# Contrastive Hard Example Pilot v1

`annotation_pilot_v1.jsonl` is the source of truth.  The CSV contains the same
rows and columns as a reviewer-friendly view.

- Rows with `source_type=PENDING_HUMAN` are empty authoring slots.  They must
  remain `review_status=DRAFT` with a blank `split` until a person writes the
  content and provenance is updated.
- Rows with `source_type=synthetic` are Codex-generated candidates.  Human
  editing or review does not change that provenance.
- Only populated rows with `review_status=AGREED` or `ADJUDICATED` may later be
  considered training-ready.  This pilot currently contains no training-ready
  rows.
- `praise_intensity` and `evidence_level` are blank legacy annotations until a
  reviewer explicitly assigns an allowed enum value.  Missing or blank values
  do not invent evidence.
- `human_reviewed` and `human_approved` default to `false`.  Human approval does
  not change `source_type` or `is_llm_generated`.
- Validate without modifying the data:

  `python -m app.training.p_text.contrastive_pilot data/ptext_improvement/pilot_v1/annotation_pilot_v1.jsonl`

The pilot contains no Blind v3, challenge, or training-dataset sentence.  Those
corpora are inputs only to read-only duplicate checks.
