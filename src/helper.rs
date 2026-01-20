//! Auxiliary types and functions.
use core::cmp::Ordering;

/// The trait to define scalar (single-dimension) types
/// with a dedicated origin (zero) point.
///
/// Currently, it is blanket-implemented for all types that implement `TryFrom<u8>`,
/// which covers at least all core primitive numeric types
/// (like `iN`, `uN` and `fN` where N is the size in bits).
pub trait Zero {
    /// Produce the zero (neutral in terms of sum) element of a type.
    fn zero() -> Self;

    /// Determines how the value is comparable to zero.
    fn cmp_zero(&self) -> Option<Ordering>;
}

impl<T> Zero for T
where
    T: TryFrom<u8> + PartialOrd,
{
    fn zero() -> Self {
        Self::try_from(0).unwrap_or_else(|_| panic!("conversion from 0 failed"))
    }

    fn cmp_zero(&self) -> Option<Ordering> {
        let zero = Self::try_from(0).ok()?;
        self.partial_cmp(&zero)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Represent a length of the arbitrary interval.
pub enum Size<T> {
    /// Interval containing no points at all.
    Empty,
    /// A size of interval containing exactly
    /// one point (or the finite number of discrete points).
    SinglePoint,
    /// A finite size of interval bounded in both directions.
    Finite(T),
    /// A size of interval extending infinitely in at least one direction.
    Infinite,
}

impl<T> Size<T> {
    /// Check whether the size represents an empty interval.
    pub const fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// Convert the size into a finite difference value, if possible.
    pub fn into_diff(self) -> Option<T>
    where
        T: Zero,
    {
        match self {
            Self::Empty | Self::SinglePoint => Some(T::zero()),
            Self::Finite(v) => Some(v),
            Self::Infinite => None,
        }
    }
}

/// Alias for the 2-tuple of the same type `T`.
pub type Pair<T> = (T, T);

/// Convert a [`Pair`] of type `T` into a [`Pair`] of type `U`.
pub fn map_pair<T, U, F>((a, b): Pair<T>, mut f: F) -> Pair<U>
where
    F: FnMut(T) -> U,
{
    (f(a), f(b))
}

/// Reorder a pair of values into ascending order.
///
/// TODO: use `core::cmp::minmax` when stabilized.
pub fn minmax<T: Ord>(v1: T, v2: T) -> [T; 2] {
    if v2 < v1 {
        [v2, v1]
    } else {
        [v1, v2]
    }
}

/// Either a single value or a pair of values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OneOrPair<T> {
    /// A single value variant.
    One(T),
    /// A pair of values variant.
    Pair(Pair<T>),
}

impl<T> OneOrPair<T> {
    /// Convert into a single value, or return the pair wrapped in `Err`.
    ///
    /// # Errors
    /// An inner pair of values if the value is a `Self::Pair`.
    pub fn into_single(self) -> Result<T, Pair<T>> {
        match self {
            Self::One(v) => Ok(v),
            Self::Pair(v) => Err(v),
        }
    }

    /// Convert into a pair of values, or return the single value wrapped in `Err`.
    ///
    /// # Errors
    /// An inner single value if the value is a `Self::One`.
    pub fn into_pair(self) -> Result<Pair<T>, T> {
        match self {
            Self::One(v) => Err(v),
            Self::Pair(v) => Ok(v),
        }
    }
}

impl<T> TryInto<Pair<T>> for OneOrPair<T> {
    type Error = T;

    fn try_into(self) -> Result<Pair<T>, T> {
        self.into_pair()
    }
}

impl<T> From<T> for OneOrPair<T> {
    fn from(value: T) -> Self {
        Self::One(value)
    }
}

impl<T> From<Pair<T>> for OneOrPair<T> {
    fn from(value: Pair<T>) -> Self {
        Self::Pair(value)
    }
}
