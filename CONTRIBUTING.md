# Contributing

This repository is meant to be extended and welcomes contributions.

The most common contributions are:

- adding a new compression algorithm
- adding a new input or output format
- improving detector-aware validation and visualization (e.g. support of other detecotrs, factories, formats)
- extending/implementing the future C++ backend
- improving docs, examples, and CI

The sections below summarize what should usually be changed for each type of contribution.

## General comments

Typical local checks:

```bash
source venv/bin/activate
ruff check .
pytest -q
```

If you work on the ROOT/EDM4hep path, use the Key4hep environment described in the README.

Before adding new helpers, check whether the repo already has a reusable implementation you can build on.
For example, DD4hep cell-id decoding now lives in:

- `src/step2point/geometry/dd4hep/bitfield.py`

Recommendation:

- prefer reusing existing helpers where possible
- if logic is currently buried inside one feature, extract it into a small reusable module instead of reimplementing it elsewhere

## Adding a new compression algorithm

Put the implementation in:

- `src/step2point/algorithms/<name>.py`

In the current design, those are the algorithm requirements:

1. Accept one `Shower`.
2. Return a new `Shower` wrapped in `CompressionResult`.
3. Do not mutate the input shower.
4. Keep format assumptions out of the algorithm itself.

Checklist of additions/changes:

1. Add the algorithm class in `src/step2point/algorithms/`.
2. Export it from `src/step2point/__init__.py` if it is part of the public API.
3. Add it to user-facing examples where appropriate:
   - `examples/run_step2point_pipeline.py`
   - `examples/generate_validation_plots.py`
4. Add unit tests for the core behavior in `tests/unit/`.
   Typical unit-test scope:
   - input shower is not mutated
   - output shape/content is correct for a small hand-built shower
   - edge cases such as empty inputs, single-point showers, repeated `cell_id`, or missing optional fields
5. Add algorithm regression coverage:
   - add a dedicated test file in `tests/integration/`
   - add a committed reference file in `tests/data/`
   - add the new job entry to `.github/workflows/algorithms.yml`
6. Update `tests/data/README.md` if you add a new reference file.
7. Update `README.md` and/or `docs`.

Current conventions:

- `identity` is checked against the original input
- every non-identity algorithm should have a committed reference output and a dedicated regression test
- deterministic algorithms should use a `strict_regression` test with exact output/reference checks
- nondeterministic (on different hardware/OS) algorithms should add:
  - a `loose_regression` test that checks stable invariants such as point-count range, compression level, and energy conservation
  - a `strict_regression` test for the pinned reference CI job only
- the split of the tests should follow:
  - `tests/unit/` for focused algorithm behavior
  - `tests/integration/` for full pipeline output and reference regression

## Adding a new input/output format

Put readers and writers in:

- `src/step2point/io/`

Reader requirements:

1. Convert the source format into the canonical `Shower` object.
2. Preserve optional fields where available (`t`, `cell_id`, `pdg`, `track_id`, primary info), and do not extend the non-optional fields.
3. Keep detector- or framework-specific dependencies isolated in the reader.

Checklist of additions/changes:

1. Add the reader/writer under `src/step2point/io/`.
2. Export it from `src/step2point/io/__init__.py`.
3. Update `example/` scripts that auto-detect by file extension if needed.
4. Add integration tests for realistic sample files in `tests/integration/`.
   Typical pattern:
   - one reader-focused test file such as `tests/integration/test_<format>_reader_optional.py`
   - if the format is used end-to-end in examples or pipeline output, add a pipeline-level integration test there too
5. If the format requires a special environment, add or update the corresponding CI job.
6. Document required setup in `README.md` and/or `docs/getting_started_edm4hep.md`.

## Adding detector-aware geometry or visualization

Geometry extraction currently lives under:

- `src/step2point/geometry/dd4hep/`

As it supports a single DD4hep specific factory for polyhedra sampling calorimeters.

Plotting lives under:

- `src/step2point/vis/`
- `examples/plot_detector_cells.py`

Checklist of additions/changes:

1. Keep geometry extraction separate from plotting, extend current schema if needed (e.g. if adding other DD4hep factories).
2. Add tests for geometry when possible.
3. If the feature depends on external detector XML, make tests skip cleanly when the external checkout is absent.
4. Update `docs/validation.md` with example commands and screenshots if the workflow is user-facing.
5. If new plots should be produced in CI, add them to the appropriate artifact workflow:
   - `.github/workflows/regression.yml`

## Implementing/extending the C++ backend

The planned for future C++ backend lives in:

- `cpp/include/step2point/`
- `cpp/src/`
- `cpp/tests/`
- `cpp/bindings/`

Curently those are the guidelines (but can be rediscussed as this is only the prototype):

1. Keep the public Python contract stable unless there is a strong reason to change it.
2. Match the Python algorithm semantics closely.
3. Add C++ tests in `cpp/tests/`.
4. If Python bindings change, add Python-side regression coverage too.
5. Check the `cpp-build` workflow in `.github/workflows/ci.yml`.

## Updating examples

Examples are the public entry points for common workflows.

If you add a new example:

1. Keep it thin. Most logic should live in `src/step2point/...`.
2. Prefer auto-detecting the input/output formats from filename when possible.
3. Document any new example in `README.md` and `docs/validation.md`.
4. If the example is part of a regression workflow, add it to CI explicitly.

## Updating CI

Current workflows:

- `.github/workflows/ci.yml`
  main correctness checks: lint, unit tests, integration tests, physics tests, ROOT checks, docs, C++
- `.github/workflows/algorithms.yml`
  per-algorithm regressions:
  - `strict_regression` jobs for exact reference checks
  - `loose_regression` jobs for nondeterministic algorithms
- `.github/workflows/regression.yml`
  plot and artifact generation, broader regression-style runs

When changing CI:

1. Keep correctness checks and artifact-generation jobs separate.
2. Put per-algorithm regression in `algorithms.yml`.
3. Use `strict_regression` for deterministic algorithms and pinned exact-reference checks.
4. Use `loose_regression` for nondeterministic algorithms that need invariant/range-based checks in local and generic CI (and please justify this choice).
5. Put plot-generation and workflow artifacts in `regression.yml`.
6. Use the Key4hep container path for ROOT/EDM4hep jobs.
7. If an external detector checkout is required, make that explicit in the workflow.

## Updating the documentation

Main docs locations:

- `README.md` for quick-start and common user workflows
- `docs/` is the main directory for the website-deployed documentation:
   - `docs/getting_started*.md` for installation and input format setup
   - `docs/validation.md` for validation, inspection, and detector-cell workflows
   - ...

When behavior changes:

1. Update the relevant example command in docs.
2. Update screenshots if the visible output changed.
3. Keep docs concise and task-oriented.
