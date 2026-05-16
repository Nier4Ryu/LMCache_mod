# Modifications to LMCache_mod

## Upstream baseline

- **Repo**: https://github.com/LMCache/LMCache
- **Pinned commit**: 6fbec463e3c047fffb4e22c97508f03b057de3bc - current fork baseline before Putpocket special blending hooks
- **Our fork remote**: https://github.com/Nier4Ryu/LMCache_mod.git

Regenerate the diff against the baseline:

```bash
cd deps/LMCache_mod
git diff 6fbec463e3c047fffb4e22c97508f03b057de3bc -- .
```

## Modifications (active)

### M1 - Putpocket per-request special blending hook

**Status**: active - last updated 2026-05-16

**Files (current scope)**:
- `lmcache/integration/vllm/vllm_v1_adapter.py` - optional special blending routing, Putpocket plan metadata, and request-config pass-through

**Current state**: The vLLM v1 adapter recognizes an optional `putpocket_blending_mode` value in LMCache `extra_config`; `special` now means the Putpocket hook is available, not that every request must use it. A request enters the Putpocket path only when `SamplingParams.extra_args["kv_transfer_params"]` carries `putpocket.enable=True` with valid Putpocket metadata. Requests without valid metadata use normal LMCache lookup and CacheBlend behavior.

**Reason**: Putpocket needs an edit-aware KV reuse hook that can reuse exact spans, shifted spans, and dirty-span repair plans without embedding the algorithm inside LMCache. The algorithm is still experimental, so opt-in must be request-specific until a global mode can make decisions from token history alone.

**Iterations**:
- 2026-05-15 - initial: added dormant special-mode planning/execution hook and request-config pass-through for Putpocket plan metadata.
- 2026-05-16 - refined: changed special mode from global routing to per-request opt-in keyed by `kv_transfer_params["putpocket.enable"]`, with normal LMCache fallback when planning is absent or invalid.

**Upstream PR-worthy?**: no - this is project-specific integration glue for Putpocket experiments.

## Archive (superseded / reverted)

No archived modifications yet.
