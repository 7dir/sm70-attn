# Porting sm70-attn into llama.cpp_v1.1.0 — Step-by-Step Instruction

**Goal:** migrate all sm70-attn components (fork base ≈ upstream `b11841`, Oct 2026)
into a clean **llama.cpp v1.1.0** tree (`7dir/llama.cpp_v1.1.0`), keeping every
behavior reversible to the stock path via environment variables.

**Strategy:** do NOT merge branches (histories diverge by ~933 upstream commits).
Instead, copy new standalone files verbatim and re-apply each hook manually on top
of v1.1.0 sources. Work layer by layer; build + test after every layer.

---

## Prerequisites

```bash
# two checkouts side by side
git clone https://github.com/7dir/sm70-attn          # SOURCE (call it $SRC)
git clone https://github.com/7dir/llama.cpp_v1.1.0   # TARGET (call it $DST)
cd $DST && git checkout -b sm70-port
export SRC=/path/to/sm70-attn DST=/path/to/llama.cpp_v1.1.0
```

Reference material that lives in `$SRC` and helps during the port:
* `sm70-hook.patch` — the original fattn.cu routing patch (Layer 2 baseline).
* `README-sm70.md` — env-var table, "what does not work" matrix, fix archives.
* `docs/sm70-*.md` — design deep dives (kvu-mask-driven, decode-tc-port-plan,
  determinism-findings, gdn-prologue-fusion, tcc-tp2-findings).

Build config used throughout (must be kept for every layer):
```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --target llama-server -j
```

Gate for every layer: **the build must compile, and with all env vars unset /
`LLAMA_SM70_D256=0` the binary must produce bit-identical output to stock v1.1.0.**

---

## Layer 0 — New standalone CUDA files (no conflicts, copy verbatim)

These files do not exist in stock v1.1.0, so there is nothing to patch — just copy.

```bash
# Attention kernels (prefill Split-D D256, decode TC, mask-driven unified pool)
cp $SRC/ggml/src/ggml-cuda/fattn-sm70-d256.{cu,cuh} \
   $SRC/ggml/src/ggml-cuda/fattn-sm70-d256-{decode,kernel,maskmd}.cuh \
   $DST/ggml/src/ggml-cuda/

# sm_70 dequant parity kernels (iq3_s / iq3_xxs / iq4_xs)
cp $SRC/ggml/src/ggml-cuda/dequant-sm70.cuh $DST/ggml/src/ggml-cuda/

# Skinny GEMM / repack stack
cp $SRC/ggml/src/ggml-cuda/{iq,q4k,q8}-skinny.{cu,cuh} \
   $SRC/ggml/src/ggml-cuda/q-skinny-common.cuh \
   $SRC/ggml/src/ggml-cuda/LICENSE.v100-skinny \
   $DST/ggml/src/ggml-cuda/

# P2P NVLink allreduce (TP2)
cp $SRC/ggml/src/ggml-cuda/allreduce-p2p.{cu,cuh} $DST/ggml/src/ggml-cuda/

# GDN chunked prefill kernels + vendored cute/cutlass/flash headers
cp -r $SRC/ggml/src/ggml-cuda/gdn-chunk-sm70 \
      $SRC/ggml/src/ggml-cuda/gdn-chunk-sm70.{cu,cuh} \
      $SRC/ggml/src/ggml-cuda/sm70-vendor \
      $DST/ggml/src/ggml-cuda/
```

**Do not build yet** — the CMake registration comes next.

### Layer 0b — CMakeLists changes

File: `ggml/src/ggml-cuda/CMakeLists.txt`. Diff `$SRC` vs `$DST` versions of this
file and re-apply these pieces onto v1.1.0's version:

1. Register the new `.cu` sources in `GGML_SOURCES_CUDA`
   (`fattn-sm70-d256.cu`, `*-skinny.cu`, `allreduce-p2p.cu`).
2. **Exclude `gdn-chunk-sm70.cu` from the main list** — it is compiled by a
   dedicated `sm_70` OBJECT target (`ggml-cuda-gdnchunk`) because the vendored
   CUTLASS needs truthful `__cplusplus` and sm_70-only flags. In `$SRC`'s
   CMakeLists see lines ~118–164 (`list(REMOVE_ITEM ...)`, `add_library(... OBJECT ...)`).
3. Add include dirs: `sm70-vendor` and `gdn-chunk-sm70` as PRIVATE includes of
   `ggml-cuda`, plus the MSVC `/Zc:__cplusplus` handling (~line 298).

✅ Gate: `cmake --build build` succeeds; server runs identically to stock
(kernels are present but never routed to yet).

---

## Layer 1 — fattn.cu routing hook (the core integration)

File to patch: `ggml/src/ggml-cuda/fattn.cu`. This is where the fork touches
stock code; use `sm70-hook.patch` from `$SRC` as the template, but apply it by
hand because v1.1.0's fattn.cu has drifted. Four edit sites (verified in `$SRC`):

1. **Extern declarations** (top of file, ~line 14–20):
   ```cpp
   extern bool   ggml_cuda_sm70_d256_supported(int cc, const ggml_tensor * dst);
   extern size_t ggml_cuda_sm70_d256_alloc_size(const ggml_tensor * dst);
   extern void   ggml_cuda_flash_attn_ext_sm70_d256(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
   ```
2. **New enum value** in the kernel-selection enum (~line 535):
   `BEST_FATTN_KERNEL_SM70_D256 = 500,`
3. **Routing predicate** `sm70_d256_supported(cc, dst)` (~line 542) — checks
   SM70 + head_dim 256 + causal mask + prefill width (`ne[1] >= 256`) +
   `LLAMA_SM70_D256 != 0`; inside `get_best_fattn_kernel()` return the new enum
   when it passes (~line 706).
4. **Scratch-size branch** (~line 770): when kernel == SM70_D256, delegate to
   `ggml_cuda_sm70_d256_alloc_size(dst)`.
5. **Dispatch + dump hook** (~line 800+): call
   `ggml_cuda_flash_attn_ext_sm70_d256()` for the selected kernel; also port the
   `SM70_DUMP` / `SM70_DUMP_KV` per-call output dumper (`sm70_dump_attn_output`)
   — zero cost when the env var is unset, needed later for the blame tooling.

⚠️ **v1.1.0 drift risk:** if upstream renamed/restructured the selection function
or scratch computation between b11841 and v1.1.0, map each of the 5 sites onto
the equivalent place in the new file — do not blind-patch. After editing, run:
```bash
diff <(grep -n "sm70\|SM70" $SRC/ggml/src/ggml-cuda/fattn.cu) \
     <(grep -n "sm70\|SM70" $DST/ggml/src/ggml-cuda/fattn.cu)
```
and confirm every `$SRC` site has a counterpart.

✅ Gate: build passes; `LLAMA_SM70_D256=0` → bit-identical greedy tokens vs stock;
`=1` on V100 + Qwen D256 model → kernel actually engages (use
`LLAMA_SM70_D256_DEBUG=1` to see ACCEPT/REJECT probes).

---

## Layer 2 — Decode kernel + verify-width routing

Already inside `fattn-sm70-d256-decode.cuh` (copied in Layer 0). What must be
ported here is the **selection logic in the launcher** (`fattn-sm70-d256.cu`):

* `LLAMA_SM70_D256_DECODE` (default 0) enables the TC decode path for
  spec-verify widths `w >= 2`.
* `LLAMA_SM70_D256_DEC_Q4` — q4_0 in-kernel loads in decode.
* `LLAMA_SM70_D256_SPLITS` — adaptive KV splits (`kv_len/920`, cap 48).
* `LLAMA_SM70_D256_DEC_WMAX` (default 16) — widest token count accepted.

Check whether upstream changed how FLASH_ATTN_EXT decode batches are shaped
between b11841 and v1.1.0 (ubatch width semantics); adjust the `w>=2` gate if so.

✅ Gate: same-prompt A/B with `--spec-type draft-mtp --spec-draft-n-max 3`:
greedy ids identical ON vs OFF at short context; deep-context decode speedup
visible at ≥46k.

---

## Layer 3 — Unified KV pool / mask-driven attention (`-kvu`)

Files touched in `$SRC` (port each hunk by hand — these files moved a lot in
933 upstream commits):

1. `common/arg.cpp` (~line 1728): `-kvu/--kv-unified` and `-no-kvu` options.
   **First check v1.1.0**: upstream may already ship an equivalent — grep for
   `kv-unified` before adding.
2. `src/llama-kv-cache.cpp`: unified-pool allocation + the `LLAMA_SM70_KV_OCC`
   fragmentation probe (~line 1297).
3. `tools/server/server.cpp` (~line 183): server auto-sets
   `LLAMA_SM70_D256_MULTI_SEQ=1` when `-kvu` + `-np > 1`, but **never overrides a
   user-set value**. Port exactly that precedence rule.
4. The mask-driven kernels themselves (`fattn-sm70-d256-maskmd.cuh`) were copied
   in Layer 0; they read the KQ mask instead of deriving causality from positions.

Design doc to follow while porting: `$SRC/docs/sm70-kvu-mask-driven.md`.

✅ Gate: run `$SRC/verify/kvu_gate.py` adapted to the new build — `-kvu -np 4`,
identical greedy ids ON vs OFF, single card and TP2.

---

## Layer 4 — Skinny GEMM + IQ repack + fast dequant

Mostly self-contained now (files copied), but each has a hook into stock dispatch:

1. **Dequant parity** (`dequant-sm70.cuh`): wire into the iq3_s/iq3_xxs/iq4_xs
   dequant dispatch behind `LLAMA_SM70_DEQUANT_FAST` (default **1**, set 0 = stock).
   Both historical bugs live here — make sure the ported code contains:
   * the `& 0xF` nibble mask fix (IQ4_XS LUT overrun, commit `ac3c27910`),
   * the poisoned-output, error-checked launch pattern.
2. **Skinny repack** (`q4k-skinny.*`, `q8-skinny.*`, `iq-skinny.*`): hooks into
   the CUDA buffer load path (weight repack at model load) and into mul_mat
   dispatch for small-M GEMMs. Env knobs: `LLAMA_SM70_SKINNY` (0/1/2),
   `LLAMA_SM70_IQ_SKINNY` (only active when SKINNY != 0),
   `LLAMA_SM70_IQ_PARITY`, `LLAMA_SM70_SKINNY_DEBUG_STOCK`.
3. **Critical regression guard:** the all-zero-IQ3_S bug was caused by a device
   symbol defined in `iq-skinny.cu` but referenced `extern __device__` from
   `q4k-skinny.cu` without `-rdc`. Keep the fix: **internal linkage (one table
   copy per TU) + lazy init in the launching TU** (commit `6cb842973`). Do not
   "clean up" the duplicated tables during the port.

✅ Gate: `verify/sm70_dequant_parity.cu` (real launches, poisoned outputs) PASS;
`LLAMA_SM70_IQ_PARITY=1` reports `nbad = 0`; long-prompt sentinel retrieval works.

---

## Layer 5 — GDN (gated delta net) chunked prefill + prologue fusion

Files:
1. `ggml/src/ggml-cuda/gated_delta_net.cu` — **highest-conflict file**: upstream
   actively develops GDN. Diff `$SRC` vs its own base, then reconcile against
   v1.1.0's version hunk by hunk. Hooks to reproduce:
   * `#include "gdn-chunk-sm70.cuh"`;
   * `ggml_cuda_gdn_chunk_sm70_n_tokens(...)` sizing call + the
     `sm70_gdn_chunk_disabled()` gate (`LLAMA_SM70_GDN_CHUNK`, default 1);
   * l2norm prologue fusion behind `LLAMA_SM70_GDN_PROLOGUE` (default **0** —
     archived negative perf result, keep it off by default).
2. If v1.1.0 refactored GDN node building (out of ggml-cuda into graph ops in
   `src/models/`), the prologue-fusion part may need re-targeting; the chunk
   kernel itself only needs the launch-site hook.

✅ Gate: `verify/sm70_gdn_prologue_bitexact.cu` → 0/16384 differ; Qwen GDN model
runs with `LLAMA_SM70_GDN_CHUNK=1` and produces identical tokens to `=0`.

---

## Layer 6 — Speculative decoding fixes (mtmd / M-RoPE crash)

File: `common/speculative.cpp`. Three changes from `$SRC` (~lines 321, 646–650,
713, 762–766):
1. Skip embedding (multimodal) batches in `process()`; zero-fill the hole on the
   next token batch.
2. Place the draft's noise block at the draft cache's own `pos_max + 1`.
3. Demote the `pos_max < N-2` warning to a debug heuristic.

⚠️ **Before porting:** check whether upstream fixed
[ggml-org#27408] in v1.1.0 (`git log v1.1.0 -- common/speculative.cpp` and search
for mtmd/embedding handling). If already fixed upstream, skip this layer entirely.

Known remaining limitation to document, not fix: `draft-dflash` × `-sm tensor`
asserts in the meta backend — production spec decoder for TP2 is `draft-mtp`.

✅ Gate: image request under `--spec-type draft-mtp` returns 200 + correct
description (`verify/mtmd_vision_check.py`).

---

## Layer 7 — Meta backend / TP2 / P2P allreduce / step profiler

Files:
1. `ggml/src/ggml-backend-meta.cpp` + `ggml/src/ggml-backend-impl.h` +
   `ggml/src/ggml-backend-reg.cpp` + `ggml/src/ggml-step-prof.h` — the TP meta
   backend and the `LLAMA_SM70_STEP_PROF` round/subgraph wall ledger. **Diff
   carefully**: `ggml-backend-meta.cpp` did not exist in older trees and changed
   shape recently; port the profiler hooks (env-gated, zero cost when off) and
   the per-subgraph dispatch fixes, not whole-file replacement.
2. `allreduce-p2p.cu` tuning constants (already copied; verify defaults):
   window `GGML_CUDA_P2P_AR_MAX_BYTES=176 MiB`, `..._NBLOCKS=80`, local-source
   staging, `__threadfence()` publish. These four commits
   (`627961dc0 597028032 4463259d1 17cc119fc`) are what turned TP2 from slower-
   than-single-card into +50% — do not lose them.
3. `src/llama-context.cpp` (~line 1847): step-prof wiring (`LLAMA_SM70_STEP_PROF`).

✅ Gate: 2×V100 TP2 prefill at 22k within noise of AR-skipped timing;
`-sm tensor` runs end-to-end with `draft-mtp`.

---

## Layer 8 — Tooling, docs, GUI (low risk, last)

```bash
cp -r $SRC/verify $SRC/bench $SRC/tools/sm70gui $DST/...   # adjust paths
cp $SRC/docs/sm70-*.md $SRC/docs/tcc-tp2-findings.md $DST/docs/
cp $SRC/README-sm70.md $DST/            # update fork-base version numbers inside
```
Then run the full harness on the v1.1.0-based build:
* `bash verify/run.sh` → 23/23 kernel cases (note: md5 pin on kernel source —
  re-pin if you had to modify any copied header);
* `verify/sm70_verify_maskmd.cu` → 8/8 mask-driven cases;
* determinism gate: identical requests → bit-identical last-token logprobs at
  554 / 22k / 174k, single card and TP2;
* rollback gate: `LLAMA_SM70_D256=0` reproduces stock v1.1.0 bit-for-bit.

---

## Summary checklist

| # | Component | Files | Conflict risk | Gate |
|---|-----------|-------|---------------|------|
| 0 | New kernels (copy) | `fattn-sm70-*`, `dequant-sm70.cuh`, `*-skinny.*`, `allreduce-p2p.*`, `gdn-chunk-sm70*`, `sm70-vendor/` | none | compiles |
| 0b | Build system | `ggml-cuda/CMakeLists.txt` (+ gdn OBJECT target, vendor includes) | low | compiles |
| 1 | Attention routing | `fattn.cu` (5 sites, see `sm70-hook.patch`) | **high** | ON/OFF bit-identical rollback |
| 2 | TC decode | launcher logic in `fattn-sm70-d256.cu` | medium | spec-verify A/B |
| 3 | Unified pool `-kvu` | `common/arg.cpp`, `src/llama-kv-cache.cpp`, `tools/server/server.cpp` | **high** | kvu_gate.py |
| 4 | Skinny + dequant | dispatch in dequant/mul_mat/load paths | medium | parity nbad=0 |
| 5 | GDN chunk/prologue | `gated_delta_net.cu` | **highest** | bit-exact 0/16384 |
| 6 | Spec-decode fix | `common/speculative.cpp` | medium (may be upstreamed already) | mtmd 200 + acceptance |
| 7 | TP2/P2P/profiler | `ggml-backend-meta.cpp`, `ggml-step-prof.h`, `llama-context.cpp` | **high** | AR-skip A/B within noise |
| 8 | Tools/docs/GUI | `verify/`, `bench/`, `docs/sm70-*`, `sm70gui` | none | 23/23 + full gates |

**Golden rules during the whole port:**
1. Every feature stays env-gated; default-off features must not change stock
   numerics at all.
2. After each layer: build + rollback gate before starting the next.
3. Never re-derive a fix that exists in `$SRC` history — cite the commit hash
   (e.g. `ac3c27910`, `315dbb08e`, `6cb842973`) in the ported comment so future
   rebases know why the odd-looking code is there.
4. Check first whether v1.1.0 already contains an upstream equivalent
   (especially Layers 3 and 6) before porting fork-local code.
