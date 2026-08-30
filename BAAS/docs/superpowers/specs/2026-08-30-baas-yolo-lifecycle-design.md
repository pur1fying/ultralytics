# BAAS YOLO Lifecycle and CUDA Inference Design

- **Date:** 2026-08-30
- **Status:** Approved master architecture, except detailed model quantization, which is explicitly deferred
- **Owner workspace:** `D:\github\ultralytics\BAAS`

## 1. Purpose

This document defines the cross-repository architecture for BAAS automatic-battle object detection: data import and
annotation, dataset versioning, model training and local experiment archival, CUDA inference comparison, model candidate
gates, and release distribution.

The design deliberately separates four concerns:

1. The browser provides a small annotation interface.
2. The Hugging Face dataset repository owns versioned training data.
3. The Ultralytics fork owns BAAS YOLO research, training, evaluation, export, benchmarking, and release preparation.
4. `BAAS_resource` publishes approved model and configuration assets without owning training or consumers.

Detailed INT8 calibration and quantization behavior is not approved by this document. It is a separate future design and
does not block the approved FP32/FP16 lifecycle.

## 2. Design Principles

- Keep the normal user workflow small. Engineering metadata exists for reproducibility but is not exposed as extra UI
  steps.
- Use immutable identities at repository boundaries: Git commit SHA, content SHA-256, model/variant ID, run ID, and
  hardware profile ID.
- Make backend selection explicit. Never silently change model, precision, execution provider, input size, calibration
  data, or preprocessing behavior.
- Store intermediate training artifacts locally. External publication always requires the user's explicit approval.
- Keep browser UI, local inference, training, dataset storage, and release distribution in their owning repositories.
- Treat the MediaTek implementation as an optimization reference, not as a CUDA API or file-format authority.

## 3. Repository Responsibilities

| Repository                                              | Responsibility                                                     | Produces                                                                                          | Must not own                                                                           |
| ------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `D:\github\ultralytics`                                 | BAAS YOLO research and lifecycle workspace under root `BAAS/`      | Run specifications, checkpoints, metrics, ONNX exports, CUDA benchmarks, local release candidates | Browser annotation UI, dataset history, GO_BAAS APIs, published TensorRT engines       |
| `D:\github\BlueArchive-YOLO-CharacterDetection-dataset` | Authoritative versioned dataset on Hugging Face                    | Images, YOLO labels, class schema, split/provenance manifest, immutable dataset commit SHA        | Training code, inference runtime, model releases                                       |
| `D:\github\baas-auto-fight-webui`                       | Frontend product and simple annotation UI                          | `/user/annotate/*` pages and calls to the local annotator API                                     | In-browser YOLO execution, independent dataset storage, independent workflow semantics |
| `D:\github\BAAS_Cpp`                                    | Automatic-battle runtime and local GPU inference consumer/provider | Runtime detections; optional native auto-label inference behind the annotator API                 | Training, dataset authority, model release ownership                                   |
| `D:\github\BAAS_resource`                               | GitHub Release distribution point                                  | Immutable Release assets and canonical release URLs                                               | Training, model selection policy, TensorRT engine building, consumer inventory         |
| `D:\github\GO_BAAS`                                     | Future backend for axis-file products                              | Axis-file product APIs and persistence                                                            | YOLO data, annotation, training, model lifecycle, model inference                      |
| `D:\github\blue_archive_auto_script`                    | BAAS user and integration documentation                            | Cross-links and user-facing integration guidance                                                  | YOLO training implementation or authoritative model artifacts                          |

Reference-only repositories:

- `D:\github\data_cleanup_ui` supplies interaction references for background pre-label jobs, explicit candidate
  acceptance, atomic saves, task progress, and diff review.
- `D:\github\cvat` supplies product references for annotation concepts and API boundaries. CVAT is not a runtime
  dependency and its annotation UI is not reused as the BAAS UI.
- `D:\github\MediaTek_neuron_pilot_deploy` supplies optimization references for fused preprocessing, buffer reuse,
  zero-copy-oriented data paths, accelerator binding, and measured end-to-end performance.

Reference code is copied only after confirming license compatibility. Otherwise the approved BAAS-specific contract is
implemented independently.

## 4. End-to-End Ownership Flow

```text
Images / folders / videos / BAAS_Cpp capture
                    |
                    v
baas-auto-fight-webui /user/annotate
                    |
          local annotator API
                    |
                    v
Hugging Face dataset commit SHA
                    |
                    v
ultralytics/BAAS local Run
  train -> validate -> export -> benchmark
                    |
                    v
Local Candidate -- explicit user approval --> BAAS_resource GitHub Release
                                                  |
                                                  v
                         BAAS_Cpp and any other consumer resolve their own lock
```

`GO_BAAS` is intentionally absent from this flow. It remains focused on the future axis-file backend.

## 5. Ultralytics BAAS Workspace Boundary

All BAAS-specific work in the Ultralytics fork lives below the root `BAAS/` directory. The root `AGENTS.md` records this
rule. No BAAS behavior is added to `ultralytics/`, upstream tests, root scripts, or other upstream-owned paths.

The target layout is:

```text
BAAS/
  README.md
  configs/
    train/
    export/
    benchmark/
    release/
  src/
  scripts/
  tests/
  docs/
    superpowers/specs/
  workspace/
    runs/
    cache/
    release-candidates/
```

Tracked source, configuration, tests, and documentation live in the first six directories. `BAAS/workspace/` is local
state and is ignored through this checkout's `.git/info/exclude`, not the tracked `.gitignore`.

Existing BAAS-related root files are legacy inputs. This design does not move, delete, or continue editing them. Migration
is outside the current scope and requires a separate explicit request identifying the files to move.

## 6. Annotation Product

### 6.1 Route and process boundary

The annotation product is separate from the automatic-battle GUI while remaining in the same WebUI application:

- `/user/annotate`
- `/user/annotate/workspaces/[workspaceId]`
- `/user/annotate/tasks/[taskId]`

The browser never runs YOLO. It talks to a loopback-only local sidecar under `/api/annotator/v1/*`. The sidecar exposes
two explicitly selectable adapters:

- `PythonYoloProvider`, which loads YOLO through Python; or
- `BaasCppProvider`, which calls a BAAS_Cpp-backed native inference path.

Provider selection is explicit configuration. If the selected provider fails or is unavailable, the request fails with a
structured error. It does not silently fall back to the other provider.

### 6.2 User workflow

The visible annotation workflow has only four steps:

1. **Import:** drag in images, a folder, or a video. Video frame extraction is automatic.
2. **Automatic annotation:** one action creates candidate boxes using the configured provider.
3. **Manual correction:** adjust boxes and classes, delete false detections, add missed objects, or mark a negative image.
4. **Submit dataset:** review a compact change summary, confirm, validate, and push the resulting dataset commit.

Progress is saved automatically. Frame navigation does not trigger inference. Auto-labeling is an explicit background job,
and its results are immutable candidates until accepted or edited by the user.

### 6.3 Hidden reliability behavior

The following behavior remains behind the four-step UI:

- Exact duplicate imports are detected by content SHA-256 and do not create a second sample; the existing sample remains
  available for label correction.
- Corrupt or unsupported files appear in one `Needs attention` list.
- Near-duplicate or ambiguous cases are sent to the single `Needs attention` list and are never silently discarded.
- Related frames from one source/time sequence remain in one split group to reduce train/validation leakage.
- Label saves use a temporary file plus atomic replacement.
- Empty YOLO label files are allowed only for explicitly confirmed negative images.
- Class IDs follow the versioned dataset schema and cannot be reordered during an ordinary annotation task.

The UI does not expose `CollectionSession`, `SelectionBatch`, `ImportPlan`, or individual gate screens. They are not public
resources in the first release and are never user-facing workflow steps.

## 7. Dataset Contract

The Hugging Face repository `Pur1fy/BlueArchive-YOLO-CharacterDetection-dataset` is the only dataset authority. A training
run references an exact Hugging Face commit SHA, never `main`, `latest`, a local modification time, or a mutable folder.

Tracked dataset content consists of:

- selected images;
- one YOLO text label file per image, including explicit empty files for negative examples;
- an ordered class ID/name schema;
- a compact dataset manifest containing stable image identity, content SHA-256, split/group identity, schema hash, and
  sanitized provenance.

Raw videos, unselected frames, absolute local paths, account names, device serials, sidecar jobs, undo history, and model
candidates stay local.

Before submission, the sidecar verifies:

- every image decodes and has exactly one label file;
- each label has a valid class ID, finite normalized coordinates, and positive width/height;
- content hashes and manifest entries agree;
- exact duplicates and related-frame groups do not cross splits;
- no unfinished candidate or unresolved blocking issue remains;
- image files are covered by Git LFS and no local-only artifacts are staged.

The user-facing `Submit dataset` action is the authorization boundary for the resulting Hugging Face push. A failed push
does not discard the local commit or annotation work.

## 8. Training Run and Local Archive

### 8.1 Minimal inputs

A user starts a run by selecting only:

1. a Hugging Face dataset commit SHA;
2. a versioned training preset; and
3. a base checkpoint, or an explicit from-scratch option.

The system resolves these selections into an immutable `RunSpec`. The resolved record includes:

- dataset repository, commit SHA, split manifest hash, and class schema hash;
- Ultralytics fork commit and a source digest for dirty builds;
- full resolved training arguments, preset identity, random seeds, and base checkpoint SHA-256;
- Python, PyTorch, CUDA, driver, dependency, GPU, and operating-system identities;
- evaluation metric definitions and gate thresholds.

Every training preset supplies explicit accuracy thresholds, and every benchmark profile supplies explicit latency and
resource thresholds. A run with a missing threshold may produce diagnostic artifacts but cannot become a Local Candidate.

Changing any identity creates a new run. A changed dataset, preset, seed, input size, or checkpoint cannot resume into an
existing run directory.

### 8.2 Run directory

Each run is self-describing and stored at `BAAS/workspace/runs/<run-id>/`:

```text
run.yaml
status.json
checkpoints/
  best.pt
  last.pt
metrics/
exports/
benchmarks/
candidate.json
```

`run.yaml` is immutable after start. `status.json` records the current stage, failure reason, and resumable checkpoint.
Metrics are stored in portable JSONL/JSON plus generated visualization files. No database or external experiment tracker
is required. External tracker integration is outside the first release; if added, it consumes the local event stream and
does not become the source of truth.

The first release does not implement automatic cleanup and never deletes completed runs. A separately approved cleanup
feature must list exact targets before deletion and retain at least the RunSpec, best/last checkpoints, metrics, exports,
benchmark reports, and candidate record.

### 8.3 Automatic stages

The approved non-quantized pipeline is:

```text
create Run -> train -> validate -> export FP32/FP16 ONNX -> ORT/TensorRT benchmark -> Local Candidate
```

Out-of-memory, unsupported-operator, export, or benchmark failures stop at their owning stage. The result contains a
structured failure and may include a proposed new RunSpec, but it never silently reduces batch size, changes image size,
changes precision, or switches backend.

## 9. CUDA Inference Architecture

### 9.1 Backends

The first PC implementation compares two explicit backends:

- `OrtCudaBackend`: ONNX Runtime with CUDA Execution Provider.
- `TensorRtBackend`: native TensorRT runtime using a locally built engine.

Backend selection is explicit and included in every result. A failed backend does not fall back to CPU, another provider,
another precision, or another model.

Both backends share the same surrounding pipeline:

- host input contract for RGB, BGR, or BGRA frames;
- a reusable pinned-host buffer pool, unless the caller already supplies registered/pinned memory;
- asynchronous H2D transfer;
- fused GPU color conversion, resize/letterbox, normalization, and layout conversion;
- preallocated device buffers and CUDA stream ownership;
- shared GPU decode and NMS;
- compact D2H transfer of final detections;
- identical detection result structures and error model.

The shared implementation prevents a benchmark from accidentally comparing a Python/CPU preprocessing path against a
GPU-fused path. A backend that cannot provide the device-resident outputs needed by the shared postprocessor fails G5;
postprocessing does not silently move to CPU.

### 9.2 TensorRT engine policy

TensorRT engines are local caches, not portable model assets. The cache key includes at least:

- ONNX SHA-256 and model I/O contract identity;
- TensorRT, CUDA, driver, and plugin versions;
- GPU identity and compute capability;
- static shape or optimization profile;
- build precision and flags.

A missing or mismatched cache entry is rebuilt locally from the verified ONNX. Engine files never enter
`BAAS_resource`.

### 9.3 Benchmark boundary

The latency boundary starts when a 1280x720 RGB/BGR/BGRA frame is available in host memory and ends when final detections
are usable by automatic battle. It includes:

- host staging and H2D;
- GPU preprocessing;
- inference;
- output decode and NMS;
- compact D2H; and
- per-frame orchestration.

It excludes emulator screenshot capture because that is a separate upstream producer. Reports include warmup policy,
sample order, run count, mean, P50, P95, P99, throughput, GPU memory, and the full hardware/software profile.

The MediaTek result of approximately 5.172 ms is a reference for the same broad end-to-end boundary, not a universal pass
threshold. CUDA targets are set per `HardwareProfile`. The first local target profile is the installed RTX 5090 system;
results are invalid without the associated driver, CUDA, ORT, TensorRT, clocks/power state, model, shape, and precision.

## 10. Candidate Gates

The lifecycle uses fail-closed gates:

- **G0 Lineage:** dataset, source, base checkpoint, resolved configuration, environment, and artifact hashes are complete.
- **G1 Dataset:** dataset structure, labels, schema, split grouping, and content hashes pass validation.
- **G2 Accuracy:** the trained checkpoint meets the exact overall and per-class thresholds frozen in the RunSpec.
- **G3 Export parity:** PyTorch and exported ONNX meet the configured numerical/output and evaluation parity policy.
- **G5 Runtime:** ORT CUDA and TensorRT execute the declared model and input contract without silent fallback, record the
  complete end-to-end benchmark, and meet the exact thresholds frozen in the selected HardwareProfile.

Only a run passing all applicable approved gates receives `candidate.json` and becomes a `Local Candidate`.

**G4 Quantization is disabled in this phase.** No INT8 model may be marked candidate or published until a separate
quantization design defines calibration data, graph format, accuracy policy, backend comparison, and failure behavior and
the user approves it.

## 11. Multi-Asset Model Release

One `BAAS_resource` GitHub Release contains multiple independent files, not one bundle containing every model.

An initial non-quantized Release may contain:

- `yolo-models-manifest-<version>.json`;
- `SHA256SUMS-<version>.txt`;
- ordered labels/configuration files;
- one or more FP32 ONNX model files;
- one or more FP16 ONNX model files; and
- accuracy, parity, and hardware-profile benchmark reports.

The verified manifest is a model catalog. It records:

- immutable release identity and schema version;
- model IDs and model versions;
- variant IDs, precision, supported backends, model I/O contract, and compatibility requirements;
- every asset's canonical Release URL, filename, byte size, SHA-256, media type, and role;
- explicit dependency edges from a model variant to labels, preprocessing/postprocessing configuration, and reports;
- training run and dataset lineage.

A consumer owns a lock containing:

```text
manifest_url
manifest_sha256
model_id
variant_id
```

The consumer downloads and verifies the manifest, resolves only the chosen model's dependency closure, downloads those
files into staging, verifies each file, validates cross-file contracts, and atomically activates the selection. It never
uses a mutable `latest` URL and never substitutes a missing variant.

`BAAS_resource` does not maintain a list of consumers and does not push weights into `BAAS_Cpp`. `BAAS_Cpp` and every
other consumer independently maintain their own lock and download from the published Release URL.

### 11.1 Publication authorization

Passing gates creates only a local release candidate. External publication follows this exact boundary:

1. Generate all candidate assets, manifest, hashes, and a human-readable summary locally.
2. Present tag, filenames, sizes, hashes, lineage, accuracy, and benchmark evidence to the user.
3. Continue only after explicit user approval for that release.
4. Create a GitHub Draft Release and upload every independent asset.
5. Verify remote filenames, sizes, and hashes while the release is still a draft.
6. Publish once after all verification passes.

Published tags and assets are immutable by policy. A correction requires a new version. Datasets, `.pt` checkpoints,
optimizer state, training cache, raw logs, and TensorRT engines are never Release assets.

## 12. Failure, Integrity, and Security Rules

- Local sidecars bind to loopback by default and restrict filesystem access to configured roots.
- User-provided paths are canonicalized and rejected when they escape the configured workspace or dataset checkout.
- Atomic file replacement protects labels, status, and manifests from interrupted writes.
- Downloads use temporary staging, byte-size verification, SHA-256 verification, and atomic activation.
- A stale Hugging Face base commit blocks dataset submission instead of overwriting new remote work.
- A stale model manifest, missing asset, incompatible model contract, or backend mismatch fails closed.
- Secrets, tokens, absolute paths, device serials, and account identifiers never enter dataset provenance, run reports, or
  release assets.
- Failed jobs preserve structured diagnostics and leave the last valid dataset, run, model selection, and release state
  unchanged.

## 13. Verification Strategy

### 13.1 Annotation and dataset

- Contract tests for the local annotator API and both inference provider adapters.
- Candidate acceptance/rejection, autosave, negative-image, restart, and provider-failure tests.
- Dataset validation fixtures covering corrupt images, exact duplicates, invalid classes/coordinates, empty labels, split
  leakage, stale base commits, and LFS mistakes.
- Browser tests for the four-step workflow and the single `Needs attention` path.

### 13.2 Training and archive

- RunSpec canonicalization and identity tests.
- Resume tests proving that only the identical RunSpec may resume a run.
- Failure injection at training, export, and benchmark stages.
- Artifact hashing, candidate gate, retention, and explicit-cleanup tests.

### 13.3 CUDA inference

- Shared preprocessing parity tests for RGB, BGR, BGRA, letterbox coordinates, normalization, and odd image sizes.
- Backend result parity on fixed fixtures, including empty and dense detections.
- Evidence that ORT uses the requested CUDA provider without CPU fallback.
- TensorRT cache invalidation tests across model, runtime, GPU, shape, plugin, and build-flag changes.
- Real RTX 5090 performance runs reporting the complete approved latency boundary and distribution.

### 13.4 Release

- Manifest schema and cross-file dependency tests.
- Corrupt, truncated, wrong-hash, missing-asset, unknown-model, and unsupported-backend tests.
- Atomic activation and rollback tests.
- Draft Release verification tests that cannot publish unless the uploaded asset set is complete.

## 14. Delivery Decomposition

This master architecture is intentionally too broad for one implementation plan. Delivery is split into independently
reviewed subprojects:

1. **Workspace and lifecycle foundation:** tracked `BAAS/` structure, local workspace exclusion, canonical identities,
   RunSpec/candidate schemas, and command boundaries.
2. **Simple annotation product:** WebUI routes, loopback sidecar, provider interface, four-step workflow, and Hugging Face
   submission.
3. **Training and local archive:** preset resolution, immutable run directories, metrics, resume, validation, and export.
4. **CUDA FP32/FP16 inference:** shared GPU pipeline, ORT CUDA, native TensorRT, cache identity, parity, and benchmark.
5. **Multi-asset release:** manifest catalog, consumer lock, staged download/activation, and user-approved Draft Release.
6. **INT8 quantization:** deferred; it receives its own design and implementation plan only after explicit approval.

Each subproject receives its own implementation plan, tests, and acceptance gate. The first implementation plan after this
master document is approved will cover only subproject 1.

## 15. Approved Decisions and Deferred Scope

Approved by the user:

- repository ownership and the `ultralytics/BAAS/` boundary;
- the non-quantized YOLO lifecycle and user-controlled publication gate;
- independent annotation routes under `/user/annotate/*`;
- the simplified four-step annotation experience;
- Hugging Face commit SHA as dataset version;
- one local directory per training run;
- ORT CUDA and native TensorRT as the first two PC backends;
- the MediaTek end-to-end measurement as a reference boundary;
- one GitHub Release with multiple independent model/config/report assets;
- consumer-owned locks and locally built TensorRT engines.

Explicitly deferred and not approved for implementation:

- calibration-set construction;
- PTQ or QAT algorithms;
- Q/DQ graph policy;
- INT8 accuracy thresholds and backend comparison;
- INT8 release variants.

The deferred items do not alter or block the approved FP32/FP16 architecture.
