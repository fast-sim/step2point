# Future C++ backend

`step2point` is currently a Python-first project, but it is intentionally designed so that the heavy kernels can later be shared with a C++ implementation.

## Why C++?

There are two practical reasons.

On-the-fly clustering supports simulation production pipelines that allow for small stored output files instead of huge step-rich files that are cluster in the post-processing stage.

Moreover, clustering on-the-fly may enable access to additional, simulation-level information (like incident particle direction).

### Shared algorithms

If the core clustering logic exists in C++, then:

- C++ frameworks can call it directly
- Python can call the same implementation through bindings

That avoids maintaining two independent algorithm implementations.

## What is likely to move into C++

The most natural candidates are the numerically heavy, reusable parts:

- `merge_within_cell`
- future clustering kernels
- selected metric computations such as centroids or profile coordinates

## What should remain in Python

Python is still the right place for:

- plotting
- exploratory studies
- high-level validation reports
- user-facing orchestration

## Build system

When the C++ backend becomes active, the intended build system is **CMake**.

That keeps the C++ side clean and conventional:

```text
cpp/
├── CMakeLists.txt
├── include/step2point/
├── src/
├── tests/
└── bindings/
```

A future CMake configuration can then build:

- the C++ library
- C++ tests
- optional Python bindings

## Long-term picture

The long-term design is to maintain a single codebase:

- Python for usability, documentation, plotting, and studies
- C++ for the stable heavy kernels that may need to run inside event-processing frameworks