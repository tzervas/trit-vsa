// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! GPU-accelerated operations for trit-vsa using CubeCL.
//!
//! This module provides CUDA kernel implementations for all core VSA operations:
//!
//! - **Bind/Unbind**: Ternary composition via modular arithmetic
//! - **Bundle**: Majority voting across multiple vectors
//! - **Similarity**: Dot product and Hamming distance with parallel reductions
//! - **Random**: GPU-parallel xorshift PRNG for random vector generation
//!
//! ## Architecture
//!
//! The GPU implementation uses a two-layer design:
//!
//! 1. **Kernels** ([`kernels`]) - Low-level CubeCL kernel definitions
//! 2. **Ops** ([`ops`]) - High-level wrappers implementing `GpuDispatchable`
//!
//! ## Feature Gate
//!
//! This module requires the `cuda` feature:
//!
//! ```toml
//! [dependencies]
//! trit-vsa = { version = "0.1", features = ["cuda"] }
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use trit_vsa::{PackedTritVec, gpu::ops::GpuBind};
//! use rust_ai_core::{get_device, DeviceConfig, GpuDispatchable};
//!
//! let device = get_device(&DeviceConfig::default())?;
//! let a = PackedTritVec::random(10000);
//! let b = PackedTritVec::random(10000);
//!
//! let gpu_bind = GpuBind;
//! let result = gpu_bind.dispatch(&(a, b), &device)?;
//! ```

pub mod kernels;
pub mod ops;

pub use ops::{
    GpuBind, GpuBundle, GpuCosineSimilarity, GpuDotSimilarity, GpuHammingDistance, GpuRandom,
    GpuUnbind, RandomInput,
};
