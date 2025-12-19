//! Module to suppport singleton sets.
//! It depends on the `singleton` feature flag, therefore
//! the implementation is separated.

#[cfg(feature = "singleton")]
use core::ops::Bound;

use crate::Interval;

/// Create a representation of a singleton sets from the given value.
///
/// Separated into a trait to allow conditional compilation
/// depending on the presence of the `singleton` feature.
///
/// <https://en.wikipedia.org/wiki/Singleton_set>
pub trait Singleton<T> {
    /// Create a singleton set containing only the given value.
    fn singleton(x: T) -> Self;
}

#[allow(unnameable_types)]
/// Special trait to convert a value into [`Bound`][core::ops::Bound]-s representing a singleton interval.
///
/// This trait is only has a single method when the `singleton` feature is enabled.
/// Otherwise it is an empty trait and implemented for `Interval<T>` by default.
pub trait SingletonBounds<T> {
    #[cfg(feature = "singleton")]
    /// Convert the given value into [`Bound`]-s representing a singleton interval.
    fn value_into_bounds(x: T) -> (Bound<T>, Bound<T>);
}

#[cfg(not(feature = "singleton"))]
impl<T> Singleton<T> for Interval<T>
where
    T: Clone,
{
    fn singleton(x: T) -> Self {
        Self::Closed((x.clone(), x))
    }
}

#[cfg(not(feature = "singleton"))]
impl<T> SingletonBounds<T> for Interval<T> {}

#[cfg(feature = "singleton")]
impl<T> Singleton<T> for Interval<T> {
    fn singleton(x: T) -> Self {
        Self::Singleton(x)
    }
}

#[cfg(feature = "singleton")]
impl<T> SingletonBounds<T> for Interval<T>
where
    T: Clone,
{
    fn value_into_bounds(x: T) -> (Bound<T>, Bound<T>) {
        (Bound::Included(x.clone()), Bound::Included(x))
    }
}
