#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "arbitrary")]
mod arbitrary;
pub(crate) mod bounds;
pub mod discrete;
pub(crate) mod helper;
mod interval;
mod ops;
mod set;
pub(crate) mod singleton;
mod str;
mod traits;

#[cfg(feature = "arbitrary")]
pub use self::arbitrary::BoundedInterval;

pub use self::{
    bounds::{Bounded, EmptyIntervalError, Endpoint, IntoBounds},
    helper::{OneOrPair, Size, ValOrInf},
    interval::Interval,
    set::SetOps,
    singleton::{Singleton, SingletonBounds},
    traits::{Linear, LinearIntRatio, Metric, MonotonicMeasure, Zero},
};
