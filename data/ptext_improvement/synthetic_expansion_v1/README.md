# Synthetic Expansion v1

This directory is reserved for future P_text improvement data.  It intentionally
contains no annotation rows yet and is separate from `pilot_v1`.

New rows must follow `schema.json`.  AI-generated rows always retain
`source_type=synthetic` and `is_llm_generated=true`, even after human review or
approval.  `human_reviewed` and `human_approved` describe review state; they do
not rewrite generation provenance.

The label must be supported by a relationship visible in `content`.  Praise or
health-related keywords alone never determine the label.  Before any row is
used for training, it must pass the improvement validator, duplicate/leakage
checks, and the project annotation process.
