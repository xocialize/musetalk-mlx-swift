# Efficiency Adoption Brief — `musetalk-mlx-swift` (MuseTalk, `talkingHead`)

> **For a session-specific agent.** Adopt engine 1.14 efficiency (engine 0.17.0+). Load the
> `mlx-swift-integration` skill; read references/package-efficiency.md (four levers + **"Measurement
> findings"**, esp. *in-app phys vs smoke MLX-peak* AND the *post-load-floor / flat-vs-climbing retention*
> note) + references/memory-harness.md. This is a **split + unload-clearCache** adoption, NOT a big
> encoder-evict. Audited 2026-06-30.

## Package at a glance
- Wrapper `MLXTalkingHead` (`TalkingHeadPackage: ModelPackage`) over core `MuseTalk`. Capability
  **`talkingHead`** (audio-driven lip-sync). Engine pinned `from: "0.5.0"`.
- **Components** (`load()`): `pipeline` = MuseTalk **VAE + UNet** (UNet quantized for int8/int4; VAE fp16);
  `whisper` = shared **whisper-tiny** audio encoder (+ `whisperPosEmb`); `bisenet` = face-parser (blend mask).
- **Footprints today (FLAT residentBytes only, NO transient):** fp16 **8 GB** · int8 **7 GB** · int4 **6.5 GB**.
  The manifest comment says measured peak ~7 GB fp16 at bs=8 — i.e. the flat number **bakes the bs=8
  activation into residency** (the over-reserve the 1.14 split fixes). `chipFloor: .pro`, tier 3 (heavy).
- `unload()` nils all four components but **does NOT `MLX.Memory.clearCache()`**.

## Audit vs. the four levers
| Lever | State | Finding | Priority |
|---|---|---|---|
| Engine dep | 🟡 | from 0.5.0 → 0.17.0 | **P0** |
| 1. Split footprint | ❌ | flat 8/7/6.5 GB, transient baked in (bs=8 peak ~7 GB) | **P1 (headline)** |
| 2. Per-stage evict | 🟡 minor | whisper-tiny encodes audio ONCE upfront then idles through the per-frame denoise — evictable, but it's ~75 MB so LOW value; bisenet runs per-frame (stays) | **P2 (optional, note)** |
| 3. mmap/lazy | 🟡 verify | confirm `MuseTalkWeights.load` is lazy (floor ≈ on-disk for the selected quant) | note |
| 4. BudgetAware | ➖ | quant config-chosen | defer |

## Plan
- **P0:** `swift package update` → 0.17.0; build + fix any drift (talkingHead surface is stable; verify).
- **P1 (HEADLINE):** split the flat footprint per quant. `residentBytes` = the weights floor that stays
  resident through the per-frame loop (VAE + UNet[quant] + bisenet + whisper-tiny); `peakActivationBytes` =
  the **bs=8 per-frame UNet+VAE transient** (the manifest's ~7 GB peak is the starting estimate — measure
  it). Adopt `QuantConfigured` so the engine selects the right per-quant footprint (3 quants already declared
  — confirm the `Configuration` reports its `quant`).
- **P2 (optional, low value):** whisper-tiny is encoded once in `runTalkingHead` step 1 (audio → per-frame
  cross-attn chunks) and unused through the denoise loop — you *could* evict it (`whisper = nil` +
  `Memory.clearCache()`) after the chunks are built, but at ~75 MB it's marginal. Note it; only implement if
  trivial and it doesn't complicate the run path. The real win is P1. (bisenet is per-frame → must stay.)
- **`unload()` must add `MLX.Memory.clearCache()`** after niling the components (eviction-frees-RSS rule).
  The wrapper already imports MLX (`whisperPosEmb: MLXArray?`), so the product is linked.

## Measurement — IMPORTANT (video watchdog + in-app phys lesson)
Video models trip the GPU watchdog on a full-pipeline CLI bench on this beta OS (the LTX lesson). So:
- Declare `residentBytes` from the **measured weight floor** (solid — load weights, `clearCache()`, read).
- Declare `peakActivationBytes` as a **best-effort estimate** from the package's own smoke/CLI at a SMALL
  envelope (few frames, small bs) OR the manifest's ~7 GB bs=8 figure, **explicitly FLAGGED** in the manifest
  + registry as "smoke/derived est, in-app phys re-baseline pending" — the in-app process `phys_footprint`
  (R-MEM-1/admission basis) reads ~2.5–2.9× higher than a smoke MLX-peak (BiRefNet 18→48). The video testing
  app (`MLXEngineVideo`/`LTXVideoTesting`) is the eventual re-baseline surface, like the image set.
- Don't fight the watchdog; prewarm weights if a cold load risks it (mirror LTX).

## Definition of done
- [ ] engine 0.17.0; `QuantConfigured`; P1 split declared per quant; `unload()` clearCache.
- [ ] residentBytes = measured weight floor; peakActivationBytes = bs=8 transient (FLAGGED smoke/derived).
- [ ] P2 whisper-evict implemented-if-trivial or noted N/A-low-value with reason.
- [ ] Smoke/CLI green at a small envelope (coherent lip-sync frames); split recorded; activation flagged.
- [ ] Registry: musetalk row Eff ⬜→✅ (note "activation = est, phys re-baseline pending"), Eng→0.17.0.

## Report back
flat→split per quant, which component dominates the resident floor, the bs=8 transient estimate (flagged),
whisper-evict decision, drift since 0.5.0, effort, commit SHAs. STAY IN SCOPE — four-lever adoption + this
brief + registry row only; no testing-app/shell/xcodeproj changes; stop-and-report if bigger.
