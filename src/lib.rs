#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]
#![cfg_attr(not(feature = "std"), no_std)]

pub(crate) mod bounds;
pub mod discrete;
pub(crate) mod helper;
mod interval;
mod macros;
pub mod rounding;
mod set;
pub mod traits;

#[cfg(feature = "arbitrary")]
pub use self::interval::arbitrary::BoundedInterval;

pub use self::{
    bounds::{Bounded, EmptyIntervalError, Endpoint, IntoBounds},
    helper::{OneOrPair, Size, ValOrInf},
    interval::{singleton, Interval},
    set::SetOps,
};
