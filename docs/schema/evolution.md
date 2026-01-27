# Evolution Database Schema

This document describes the evolution storage tables added to BubbleLab's Drizzle schema.

## Overview

The evolution schema stores requests, generated designs, judge scores, results, and screenshots.
It complements the existing evolution graph tables (`evolution_runs`, `evolution_assets`,
`evolution_nodes`).

## Tables

### evolution_requests
- `id`: Primary key.
- `user_id`: Owner (references `users.clerk_id`).
- `content`: Original prompt or request text.
- `mode`: Evolution mode (default `standard`).
- `parameters`: JSON parameters for the run.
- `constraints`: JSON constraints for the run.
- `created_at`, `updated_at`: Timestamps.

### evolution_designs
- `id`: Primary key.
- `request_id`: Parent request (references `evolution_requests.id`).
- `generation`: Generation index.
- `html`, `css`: Rendered output.
- `metadata`: JSON metadata (mutation details, etc.).
- `created_at`: Timestamp.

### evolution_judge_scores
- `id`: Primary key.
- `design_id`: Parent design (references `evolution_designs.id`).
- `agent`: Judge agent identifier.
- `score`: Score value (0-1).
- `reasoning`: Text summary.
- `highlights`, `issues`, `recommendations`: JSON arrays.
- `created_at`: Timestamp.

### evolution_results
- `id`: Primary key.
- `request_id`: Parent request (references `evolution_requests.id`).
- `best_design_id`: Winning design (references `evolution_designs.id`).
- `summary`: JSON summary and metrics.
- `created_at`: Timestamp.

### evolution_screenshots
- `id`: Primary key.
- `design_id`: Parent design (references `evolution_designs.id`).
- `asset_id`: Optional asset reference (references `evolution_assets.id`).
- `kind`: Screenshot type (default `thumbnail`).
- `width`, `height`: Dimensions.
- `created_at`: Timestamp.

## Relationships

- `evolution_requests` -> `users` (owner).
- `evolution_designs` -> `evolution_requests`.
- `evolution_judge_scores` -> `evolution_designs`.
- `evolution_results` -> `evolution_requests` and `evolution_designs` (best).
- `evolution_screenshots` -> `evolution_designs` and `evolution_assets`.

## Indexes

- `evolution_requests_user_id_idx` on `evolution_requests.user_id`.
- `evolution_designs_request_generation_idx` on `evolution_designs.request_id, generation`.
- `evolution_judge_scores_design_id_idx` on `evolution_judge_scores.design_id`.
- `evolution_results_request_id_idx` on `evolution_results.request_id`.
- `evolution_screenshots_design_id_idx` on `evolution_screenshots.design_id`.
