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
        Self::try_from(0)
            .ok()
            .and_then(|zero| self.partial_cmp(&zero))
    }
}

#[derive(Debug, Clone, Copy)]
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

pub type Pair<T> = (T, T);

pub fn map_pair<T, U, F>((a, b): Pair<T>, mut f: F) -> Pair<U>
where
    F: FnMut(T) -> U,
{
    (f(a), f(b))
}

/// Either a single value or a pair of values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OneOrPair<T> {
    /// A single value variant.
    One(T),
    /// A pair of values variant.
    Pair(Pair<T>),
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
