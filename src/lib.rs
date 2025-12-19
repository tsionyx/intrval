//! Generic single-dimensioned intervals with operations on them.
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

use self::helper::Pair;

pub use self::{
    bounds::Bounded,
    helper::{OneOrPair, Size, Zero},
    interval::Interval,
    singleton::{Singleton, SingletonBounds},
};
