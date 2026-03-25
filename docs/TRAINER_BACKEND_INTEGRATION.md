# Trainer Backend Integration

This phase adds the first real bridge between trainer modules and runtime adapters.

## What is integrated now
- standardized checkpoint package helpers
- `MGN` model implementation for runtime-side checkpoint loading
- `MGNReIDExtractor.from_checkpoint(...)`
- `PersonAttributeExtractor.from_checkpoint(...)`
- trainer export modes that repack source checkpoints into the platform package format

## Why this matters
Before this phase, runtime adapters and trainer modules were still logically separate. After this phase, they share:
- a package format
- a loading contract
- model configuration metadata

## Current limits
- export support is implemented before full training support
- strict compatibility with every upstream checkpoint layout is not guaranteed yet
- the attribute runtime model is a unified baseline for platform integration, not yet a full vendored copy of all upstream training-time modules
