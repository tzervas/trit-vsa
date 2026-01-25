# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-01-24

### Added

- Core `Trit` type with balanced ternary values {-1, 0, +1}
- `Tryte3` type: 3 trits packed in a single byte (range -13 to +13)
- `Word6` type: 6 trits packed in u16 (range -364 to +364)
- Full arithmetic operations: add, multiply, negate for all types
- `PackedTritVec`: Bitsliced vector storage using plus/minus planes
  - Efficient storage: 2 bits per trit
  - SIMD-friendly memory layout
  - Support for dot product, addition, negation
- `SparseVec`: COO (Coordinate) format for high-sparsity vectors
  - Separate positive and negative index lists
  - Efficient for vectors with >90% zeros
- Vector Symbolic Architecture (VSA) operations:
  - `bind`: Association via subtraction mod 3
  - `unbind`: Recovery via addition mod 3
  - `bundle`: Superposition via majority voting
  - `cosine_similarity`: Similarity metric
  - `hamming_distance`: Distance metric
- SIMD acceleration stubs for AVX2 and NEON
- Comprehensive test suite (84 unit tests, 32 doc tests)
- Criterion benchmarks for performance-critical operations
- Serialization support via serde

### Technical Details

- Minimum Rust version: 1.70
- Zero unsafe code in core types
- Thread-safe (all types are Send + Sync)
- No-std compatible (with alloc)

[Unreleased]: https://github.com/tzervas/trit-vsa/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tzervas/trit-vsa/releases/tag/v0.1.0
