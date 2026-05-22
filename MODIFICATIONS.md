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

**Status**: active - last updated 2026-05-21

**Files (current scope)**:
- `lmcache/integration/vllm/vllm_v1_adapter.py` - optional special blending routing, Putpocket plan metadata, and request-config pass-through
- `lmcache/v1/cache_engine.py` - layerwise retrieve cleanup ownership for request-level lookup pins
- `lmcache/v1/gpu_connector/gpu_connectors.py` - optional Putpocket exact-load position-remap bypass for layerwise GPU transfer

**Current state**: The vLLM v1 adapter recognizes an optional `putpocket_blending_mode` value in LMCache `extra_config`; `special` now means the Putpocket hook is available, not that every request must use it. A request enters the Putpocket path only when `SamplingParams.extra_args["kv_transfer_params"]` carries `putpocket.enable=True` with valid Putpocket metadata. Requests without valid metadata use normal LMCache lookup/retrieval behavior; special mode avoids eager CacheBlend construction because this vLLM worker path has not registered the model when the connector is initialized. Scheduler lookup normalizes vLLM token containers to plain Python lists before RPC serialization. The adapter passes vLLM tensor/pipeline parallel sizes and the vLLM runtime config into the Putpocket executor so unsupported TP layouts can fail explicitly and temp materializers can resolve the exact runtime RoPE/YaRN params before any KV tensor update is attempted; it can now instantiate a named Putpocket executor from `extra_config["putpocket_executor"]`. Putpocket plans may provide `lmcache_hit_token_len`, allowing query-boundary planners to reserve less than the full prompt as externally supplied KV; requests with `putpocket.profile_only=True` build/log the plan but fall back to normal LMCache lookup so experiments can collect planner data. Layerwise store/retrieve calls now forward `request_configs` into `LMCacheEngine.store_layer` and `retrieve_layer` so `lmcache.tag.*` namespaces affect LMCache keys consistently with the non-layerwise path; worker-side layerwise retrieve also forwards `req_id` so retrieve cleanup can distinguish request-level lookup pins from retrieve-owned staging pins. Layerwise store generator advancement is StopIteration-safe so partial-chunk storage does not crash vLLM workers; if storage completed early, any remaining issue surfaces as a later LMCache miss. Layerwise retrieve cleanup now preserves LocalCPU exact-hit objects whose pins are owned by `lookup_unpin(req_id)`, while still unpinning retrieve-owned staging objects such as disk-loaded CPU buffers. For Putpocket materialization requests, the scheduler first checks the target-tag exact LMCache namespace and uses normal retrieve on hit; on miss it attaches a Putpocket source-prefix plan and passes the active `LMCacheEngine` into the worker hook so Putpocket can load source-tagged KV into allocated vLLM slots before normal LMCache store persists the target-tagged result. Putpocket exact-load requests can mark layerwise GPU transfer with `putpocket_disable_position_remap=True`; this bypasses the CacheBlend fused-RoPE dependency because Putpocket stores fixed/full KV under the target token positions already.

**Reason**: Putpocket needs an edit-aware KV reuse hook that can reuse exact spans, shifted spans, and dirty-span repair plans without embedding the algorithm inside LMCache. The algorithm is still experimental, so opt-in must be request-specific until a global mode can make decisions from token history alone.

**Iterations**:
- 2026-05-15 - initial: added dormant special-mode planning/execution hook and request-config pass-through for Putpocket plan metadata.
- 2026-05-16 - refined: changed special mode from global routing to per-request opt-in keyed by `kv_transfer_params["putpocket.enable"]`, with normal LMCache fallback when planning is absent or invalid.
- 2026-05-16 - refined: delayed/disabled CacheBlend object construction in special mode to avoid `VLLMModelTracker` initialization-order failures during Qwen30B launcher startup.
- 2026-05-16 - refined: converted scheduler `token_ids` to a plain list before LMCache lookup so vLLM `ConstantList` objects do not reach the lookup RPC encoder.
- 2026-05-16 - refined: passed vLLM TP/PP sizes into the Putpocket executor so the initial TP=1-only path can reject TP>1 with an explicit TODO.
- 2026-05-17 - refined: added named Putpocket executor selection and scheduler support for plan-provided `lmcache_hit_token_len`, used by QueryBasedPPLE to reserve only the pre-query prompt region.
- 2026-05-17 - refined: added `putpocket.profile_only` fallback behavior so planner telemetry can be collected without invoking the unfinished worker-side KV patch hook.
- 2026-05-21 - refined: forwarded request configs through layerwise store/retrieve so tagged LMCache namespaces can distinguish Putpocket source/full/fix variants.
- 2026-05-21 - refined: added target-exact-first scheduling for Putpocket materialization, preserved attached plans across cached scheduler lookups, and passed `LMCacheEngine`/sync state to the worker hook for source-prefix runtime loading.
- 2026-05-21 - refined: passed `vllm_config` into Putpocket executor construction so the temp canonical materializer can snapshot runtime RoPE/YaRN params from the actual model config.
- 2026-05-21 - refined: made layerwise store generator advancement StopIteration-safe so enabling partial-chunk storage for Putpocket does not crash vLLM workers.
- 2026-05-21 - refined: added a Putpocket exact-load position-remap bypass so special mode can retrieve target-position KV without constructing CacheBlend's fused-RoPE helper.
- 2026-05-21 - refined: separated layerwise retrieve staging-pin cleanup from request-level lookup pins and forwarded worker request ids through normal layerwise retrieve to avoid double-unpin warnings on LocalCPU exact hits.

**Upstream PR-worthy?**: no - this is project-specific integration glue for Putpocket experiments.

---

### M2 - Putpocket replay attention bridge

**Status**: active - last updated 2026-05-17

**Files (current scope)**:
- `lmcache/integration/vllm/vllm_v1_adapter.py` - optional worker-side callback that forwards vLLM layer tensors and request metadata to Putpocket attention replay capture

**Current state**: The LMCache vLLM adapter exposes `capture_putpocket_attention_layer` for the optional vLLM attention hook. When the Putpocket executor exists and a request carries `putpocket.attention_replay.*` metadata, the adapter slices the request query/key tensors using attention metadata, derives the absolute local token span from scheduler progress metadata, passes request slot mapping/KV cache handles to Putpocket, and leaves non-opted-in requests untouched.

**Reason**: Putpocket needs replay attention scores after KV loading/modification, but LMCache should remain a bridge for request metadata and KV lifecycle rather than owning the localization algorithm.

**Iterations**:
- 2026-05-17 - initial: added the optional bridge from vLLM layer callbacks to Putpocket's attention localization module.
- 2026-05-17 - refined: carried per-step input token length through `ReqMeta` so chunked prefill attention capture can locate Query-2 even after `LoadSpec` is popped.
- 2026-05-17 - refined: sliced the layer key tensor per request as well as the query tensor so Putpocket can compute true causal query-to-context attention with Query-2 keys in the softmax denominator.
- 2026-05-17 - refined: forwarded the request-local token span to Putpocket so chunked prefill capture can merge cached prefix K with the current local K chunk.

**Upstream PR-worthy?**: no - this is project-specific instrumentation for Putpocket experiments.

---

### M3 - Putpocket sampled KV vector dump hook

**Status**: active - last updated 2026-05-18

**Files (current scope)**:
- `lmcache/integration/vllm/vllm_v1_adapter.py` - optional env/request-gated callout from the layerwise KV save path to Putpocket's sampled KV-vector dumper

**Current state**: When a request carries `putpocket.kv_vector_dump.enable=True`, the layerwise save hook forwards the current layer's paged KV tensor, token ids, slot mapping, and request metadata to `putpocket.src.kv_update.vllm_kv_vector_dump`, then skips normal LMCache storage for that request. Non-opted-in requests are untouched, and dump failures are logged without changing LMCache's normal save path.

**Reason**: Putpocket needs compact real-vLLM KV artifacts for layer/token L2 analysis on large Qwen runs, without writing the full paged cache for every token or depending on LMCache offload storage being healthy.

**Iterations**:
- 2026-05-18 - initial: added request-gated sampled KV-vector dump callout in the vLLM v1 layerwise save path.
- 2026-05-18 - refined: made dump-enabled requests bypass normal LMCache storage so the analysis hook can run even when CPU offload storage is unavailable.

**Upstream PR-worthy?**: no - this is project-specific analysis instrumentation.

## Archive (superseded / reverted)

No archived modifications yet.
