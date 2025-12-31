#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]
#![no_std]

#[cfg(feature = "arbitrary")]
mod arbitrary;
pub(crate) mod bounds;
pub(crate) mod helper;
mod interval;
mod ops;
pub(crate) mod singleton;
mod str;

#[cfg(feature = "arbitrary")]
pub use self::arbitrary::BoundedInterval;

pub use self::{
    bounds::{Bounded, EmptyIntervalError, Endpoint},
    helper::{OneOrPair, Size, Zero},
    interval::Interval,
    singleton::{Singleton, SingletonBounds},
};
