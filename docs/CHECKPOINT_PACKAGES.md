# Standardized Checkpoint Packages

This document defines the checkpoint package convention used by trainer modules and runtime adapters.

## Package structure
A checkpoint package is a Torch-saved dictionary with these main fields:
- `state_dict`
- `metadata`
- `model_config`
- optional `class_names`

The loader also accepts common legacy keys such as:
- `model_state_dict`
- `model`
- `net`

## Why this exists
The platform needs trainer/runtime decoupling. A trainer should be able to export a package once, and runtime adapters should be able to load it without depending on the original training script layout.

## Current integrated backends
### ReID MGN
Export mode records:
- backend = `mgn`
- package_type = `reid_extractor`
- model_name = `MGN`
- num_classes
- image_size

### Attributes
Export mode records:
- backend = `person_attribute_recognition`
- package_type = `attribute_extractor`
- model_name = `AttributeBaseline`
- num_classes
- backbone
- image_size
- low/high thresholds
- optional class names

## Current limitation
This phase standardizes packaging and runtime loading. It does not yet provide the full upstream training pipelines inside this consolidated repo.
