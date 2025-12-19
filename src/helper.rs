use core::cmp::Ordering;

/// Marker trait to define scalar (single-dimension) types.
///
/// Currently, it is blanket-implemented for all types that implement `TryFrom<u8>`,
/// which covers at least all core primitive numeric types
/// (like `iN`, `uN` and `fN` where N is the size in bits).
pub trait Scalar: TryFrom<u8> {}

impl<T> Scalar for T where T: TryFrom<u8> {}

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

pub type Pair<T> = (T, T);

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
