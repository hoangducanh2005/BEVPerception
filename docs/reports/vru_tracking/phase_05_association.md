# Phase 5 — Inference-only Association Improvement

## A. Objective

Test whether BEV query similarity improves VRU association without training,
while preserving the classical Kalman/Hungarian baseline.

## B. Work completed

No association ablation was run. This phase correctly remains dependent on
Phase 2's verified tracker protocol and Phase 3's identity-discrimination
evidence.

## C. Experimental results

No E1–E5 comparison, handoff result, tracking-only latency, or reproducible
improvement is available. The existing Phase 2 query state is diagnostic only
and is not an association cost.

## D. Technical findings

It would be methodologically invalid to introduce a query-assisted cost before
proving that the embeddings separate same-ID from different-ID pairs. The
current lower nuScenes query cosine alone is insufficient evidence.

## E. Changes relative to previous phase

No association code or detector behavior was changed.

## F. Limitations

Data is unavailable, and prerequisite phases are blocked. No claim of tracking
improvement is made.

## G. Recommendation

After phases 1–4 are valid, compare classical baseline, corrected protocol,
optional low-confidence association, and query-assisted association using the
same cached detections and scenes. Keep query association optional and gated.

## H. Completion status

**BLOCKED**
