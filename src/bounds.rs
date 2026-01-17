use core::{
    cmp::Ordering,
    fmt,
    ops::{Add, Bound, Neg, Not, Sub},
};

pub use self::impls::EmptyIntervalError;

// The lack of generic const expressions
// forces to use `bool` instead of `enum Side {Left, Right}`
pub const LEFT: bool = false;
pub const RIGHT: bool = true;

#[derive(Debug, Clone, Copy, Eq, Hash)]
/// The bound of an interval.
pub enum Endpoint<const SIDE: bool, T> {
    /// The point is included in the interval.
    Included(T),
    /// The point is excluded from the interval.
    Excluded(T),
    /// The interval is unbounded in this direction.
    Infinite,
}

pub type LBound<T> = Endpoint<LEFT, T>;
pub type RBound<T> = Endpoint<RIGHT, T>;
pub type BothBounds<T> = (LBound<T>, RBound<T>);

impl<const SIDE: bool, T> Endpoint<SIDE, T> {
    /// Convert [`Endpoint`] into a [`Bound`].
    pub fn into_bound(self) -> Bound<T> {
        match self {
            Self::Included(v) => Bound::Included(v),
            Self::Excluded(v) => Bound::Excluded(v),
            Self::Infinite => Bound::Unbounded,
        }
    }

    /// Convert [`Endpoint`] into an open [`Bound`].
    pub fn into_exclusive_bound(self) -> Bound<T> {
        match self.into_bound() {
            Bound::Included(v) => Bound::Excluded(v),
            other => other,
        }
    }

    fn swap_inclusion(self) -> Self {
        match self {
            Self::Included(v) => Self::Excluded(v),
            Self::Excluded(v) => Self::Included(v),
            Self::Infinite => Self::Infinite,
        }
    }

    fn diff_bound<U, Z>(self, rhs: Bound<U>) -> Bound<Z>
    where
        T: Sub<U, Output = Z>,
    {
        use Bound::{Excluded, Included, Unbounded};

        match (self.into_bound(), rhs) {
            (Included(a), Included(b)) => Included(a - b),
            (Included(a), Excluded(b)) | (Excluded(a), Included(b) | Excluded(b)) => {
                Excluded(a - b)
            }
            (Unbounded, _) | (_, Unbounded) => Unbounded,
        }
    }

    pub(crate) fn augment_with_inf(self) -> BothBounds<T> {
        #[allow(clippy::match_bool)]
        match SIDE {
            LEFT => (LBound::from(self.into_bound()), RBound::Infinite),
            RIGHT => (LBound::Infinite, RBound::from(self.into_bound())),
        }
    }
}

impl<const SIDE: bool, T> From<Endpoint<SIDE, T>> for Bound<T> {
    fn from(value: Endpoint<SIDE, T>) -> Self {
        value.into_bound()
    }
}

impl<const SIDE: bool, T> From<Bound<T>> for Endpoint<SIDE, T> {
    fn from(value: Bound<T>) -> Self {
        match value {
            Bound::Included(v) => Self::Included(v),
            Bound::Excluded(v) => Self::Excluded(v),
            Bound::Unbounded => Self::Infinite,
        }
    }
}

impl<const SIDE: bool, T> fmt::Display for Endpoint<SIDE, T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        #[allow(clippy::match_bool)]
        match SIDE {
            LEFT => match self {
                Self::Included(v) => {
                    write!(f, "[")?;
                    v.fmt(f)
                }
                Self::Excluded(v) => {
                    write!(f, "(")?;
                    v.fmt(f)
                }
                Self::Infinite => write!(f, "(-inf"),
            },
            RIGHT => match self {
                Self::Included(v) => {
                    v.fmt(f)?;
                    write!(f, "]")
                }
                Self::Excluded(v) => {
                    v.fmt(f)?;
                    write!(f, ")")
                }
                Self::Infinite => write!(f, "+inf)"),
            },
        }
    }
}

/// Used to convert an interval-like value into its endpoints.
///
/// TODO: consider matching with `core::ops::IntoBounds` when
/// the `feature = "range_into_bounds"` gets stabilized.
pub trait IntoBounds<T>: Sized {
    /// The error signalling conversion to [`Endpoint`]-s fails.
    type Error;

    /// Convert this value into the left and right bounds, consuming the value.
    ///
    /// # Errors
    /// Return [`Self::Error`] if the conversion fails.
    fn into_bounds(self) -> Result<BothBounds<T>, Self::Error>;
}

/// Extend the [`IntoBounds`] trait to allow creating
/// a value from a pair of [`Endpoint`]-s.
pub trait Bounded<T>: IntoBounds<T> {
    /// Create from the given pair of [`Endpoint`]-s.
    fn from_bounds(bounds: BothBounds<T>) -> Self;
}

pub const fn inf_ordering(side: bool) -> Ordering {
    #[allow(clippy::match_bool)]
    match side {
        LEFT => Ordering::Less,
        RIGHT => Ordering::Greater,
    }
}

impl<const SIDE: bool, T> Endpoint<SIDE, T> {
    /// Represent the direction of approaching to the endpoint as an [`Ordering`]:
    /// - `Ordering::Greater` for left endpoints (i.e. approaching from the right):
    ///   `(a, ...)` can also be represented as `[a + epsilon, ...)`
    /// - `Ordering::Less` for right endpoints (i.e. approaching from the left):
    ///   `(..., b)` can also be represented as `(..., b - epsilon]`.
    const fn value_approaching(&self) -> Ordering {
        match self {
            Self::Included(_) => Ordering::Equal,
            Self::Excluded(_) | Self::Infinite => inf_ordering(SIDE).reverse(),
        }
    }

    /// Get an `Endpoint` with referenced point value.
    pub const fn as_ref(&self) -> Endpoint<SIDE, &T> {
        match self {
            Self::Included(v) => Endpoint::Included(v),
            Self::Excluded(v) => Endpoint::Excluded(v),
            Self::Infinite => Endpoint::Infinite,
        }
    }

    /// Convert the underlying value of the endpoint
    /// preserving the inclusion/exclusion state.
    pub fn map<F, U>(self, f: F) -> Endpoint<SIDE, U>
    where
        F: FnOnce(T) -> U,
    {
        match self {
            Self::Included(v) => Endpoint::Included(f(v)),
            Self::Excluded(v) => Endpoint::Excluded(f(v)),
            Self::Infinite => Endpoint::Infinite,
        }
    }

    pub(crate) const fn bound_val(&self) -> Option<&T> {
        match self.as_ext_point() {
            ExtPoint::Finite((val, _ordering)) => Some(val),
            ExtPoint::Infinite(_) => None,
        }
    }

    /// Get the underlying value and the direction of the endpoint.
    /// The direction is represented as an [`Ordering`]:
    /// - `Ordering::Equal` for included endpoints (either left or right);
    /// - `Ordering::Less` for excluded right endpoints (i.e. approaching from the left);
    /// - `Ordering::Greater` for excluded left endpoints (i.e. approaching from the right).
    pub(crate) const fn as_ext_point(&self) -> ExtPoint<&T> {
        let ordering = self.value_approaching();
        match self {
            Self::Included(v) | Self::Excluded(v) => ExtPoint::Finite((v, ordering)),
            Self::Infinite => ExtPoint::Infinite(SIDE),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// The value of an endpoint related to a specific point
/// (along with the direction of approaching to this point) or infinity.
pub enum ExtPoint<T> {
    Finite((T, Ordering)),
    Infinite(bool),
}

impl<T: PartialOrd> PartialOrd for ExtPoint<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Self::Infinite(side_a), Self::Infinite(side_b)) => Some(side_a.cmp(side_b)),
            (Self::Infinite(side), Self::Finite(_)) => Some(inf_ordering(*side)),
            (Self::Finite(_), Self::Infinite(side)) => Some(inf_ordering(!*side)),
            (Self::Finite(a), Self::Finite(b)) => a.partial_cmp(b),
        }
    }
}

impl<const SIDE_L: bool, const SIDE_R: bool, T: PartialEq> PartialEq<Endpoint<SIDE_R, T>>
    for Endpoint<SIDE_L, T>
{
    fn eq(&self, other: &Endpoint<SIDE_R, T>) -> bool {
        self.as_ext_point() == other.as_ext_point()
    }
}

impl<const SIDE_L: bool, const SIDE_R: bool, T: PartialOrd> PartialOrd<Endpoint<SIDE_R, T>>
    for Endpoint<SIDE_L, T>
{
    fn partial_cmp(&self, other: &Endpoint<SIDE_R, T>) -> Option<Ordering> {
        self.as_ext_point().partial_cmp(&other.as_ext_point())
    }
}

impl<const SIDE: bool, T> PartialEq<T> for Endpoint<SIDE, T>
where
    T: PartialEq + Clone,
{
    fn eq(&self, point: &T) -> bool {
        self.eq(&Self::Included(point.clone()))
    }
}

impl<const SIDE: bool, T> PartialOrd<T> for Endpoint<SIDE, T>
where
    T: PartialOrd + Clone,
{
    fn partial_cmp(&self, point: &T) -> Option<Ordering> {
        self.partial_cmp(&Self::Included(point.clone()))
    }
}

impl<const SIDE: bool, T: Ord> Ord for Endpoint<SIDE, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other)
            .expect("comparison between Ord values failed")
    }
}

impl<const SIDE: bool, T> Neg for Endpoint<SIDE, T>
where
    T: Neg<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.map(Neg::neg)
    }
}

impl<const SIDE: bool, T, U, Z> Add<Endpoint<SIDE, U>> for Endpoint<SIDE, T>
where
    T: Add<U, Output = Z>,
{
    type Output = Endpoint<SIDE, Z>;

    fn add(self, rhs: Endpoint<SIDE, U>) -> Self::Output {
        use Endpoint::{Excluded, Included, Infinite};
        match (self, rhs) {
            (Included(a), Included(b)) => Included(a + b),
            (Included(a) | Excluded(a), Excluded(b)) | (Excluded(a), Included(b)) => {
                Excluded(a + b)
            }
            (Infinite, _) | (_, Infinite) => Infinite,
        }
    }
}

impl<T> Not for LBound<T> {
    type Output = RBound<T>;

    fn not(self) -> Self::Output {
        Endpoint::from(Bound::from(self.swap_inclusion()))
    }
}

impl<T, U, Z> Sub<RBound<U>> for LBound<T>
where
    T: Sub<U, Output = Z>,
{
    type Output = LBound<Z>;

    fn sub(self, rhs: RBound<U>) -> Self::Output {
        self.diff_bound(rhs.into_bound()).into()
    }
}

impl<T> Not for RBound<T> {
    type Output = LBound<T>;

    fn not(self) -> Self::Output {
        Endpoint::from(Bound::from(self.swap_inclusion()))
    }
}

impl<T, U, Z> Sub<LBound<U>> for RBound<T>
where
    T: Sub<U, Output = Z>,
{
    type Output = RBound<Z>;

    fn sub(self, rhs: LBound<U>) -> Self::Output {
        self.diff_bound(rhs.into_bound()).into()
    }
}

mod impls {
    use core::fmt;

    use crate::{bounds::Endpoint, singleton::SingletonBounds, Interval};

    use super::{BothBounds, Bounded, IntoBounds};

    #[derive(Debug, Clone, Copy)]
    /// The operation error indicating that the interval is empty.
    pub struct EmptyIntervalError<T>(Interval<T>);

    impl<T: fmt::Display> fmt::Display for EmptyIntervalError<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "the interval is empty: ")?;
            self.0.fmt(f)
        }
    }

    impl<T> From<EmptyIntervalError<T>> for Interval<T> {
        fn from(err: EmptyIntervalError<T>) -> Self {
            err.0
        }
    }

    // https://blog.rust-lang.org/2024/09/05/Rust-1.81.0/#core-error-error
    #[rustversion::since(1.81)]
    impl<T: fmt::Debug + fmt::Display> core::error::Error for EmptyIntervalError<T> {}

    impl<T> IntoBounds<T> for Interval<T>
    where
        T: PartialOrd,
        Self: SingletonBounds<T>,
    {
        type Error = EmptyIntervalError<T>;

        fn into_bounds(self) -> Result<BothBounds<T>, Self::Error> {
            use Endpoint::{Excluded, Included, Infinite};

            let bounds = match self {
                Self::Empty => return Err(EmptyIntervalError(self)),
                Self::LessThan(b) => (Infinite, Excluded(b)),
                Self::LessThanOrEqual(b) => (Infinite, Included(b)),
                #[cfg(feature = "singleton")]
                Self::Singleton(x) => <Self as SingletonBounds<T>>::value_into_bounds(x),
                Self::GreaterThanOrEqual(a) => (Included(a), Infinite),
                Self::GreaterThan(a) => (Excluded(a), Infinite),
                Self::Open((ref a, ref b)) if a >= b => {
                    return Err(EmptyIntervalError(self));
                }
                Self::Open((a, b)) => (Excluded(a), Excluded(b)),
                Self::LeftOpen((ref a, ref b)) if a >= b => {
                    return Err(EmptyIntervalError(self));
                }
                Self::LeftOpen((a, b)) => (Excluded(a), Included(b)),
                Self::RightOpen((ref a, ref b)) if a >= b => {
                    return Err(EmptyIntervalError(self));
                }
                Self::RightOpen((a, b)) => (Included(a), Excluded(b)),
                Self::Closed((ref a, ref b)) if a > b => {
                    return Err(EmptyIntervalError(self));
                }
                Self::Closed((a, b)) => (Included(a), Included(b)),
                Self::Full => (Infinite, Infinite),
            };
            Ok(bounds)
        }
    }

    impl<T> Bounded<T> for Interval<T>
    where
        T: PartialOrd,
        Self: SingletonBounds<T>,
    {
        fn from_bounds(bounds: BothBounds<T>) -> Self {
            use Endpoint::{Excluded, Included, Infinite};

            match bounds {
                (Infinite, Infinite) => Self::Full,
                (Infinite, Included(b)) => Self::LessThanOrEqual(b),
                (Infinite, Excluded(b)) => Self::LessThan(b),
                (Included(a), Infinite) => Self::GreaterThanOrEqual(a),
                (Excluded(a), Infinite) => Self::GreaterThan(a),
                (Included(a), Included(b)) => Self::Closed((a, b)),
                (Included(a), Excluded(b)) => Self::RightOpen((a, b)),
                (Excluded(a), Included(b)) => Self::LeftOpen((a, b)),
                (Excluded(a), Excluded(b)) => Self::Open((a, b)),
            }
        }
    }

    impl<'a, T> IntoBounds<&'a T> for &'a Interval<T>
    where
        T: PartialOrd,
    {
        type Error = EmptyIntervalError<&'a T>;

        fn into_bounds(self) -> Result<BothBounds<&'a T>, Self::Error> {
            self.as_ref_bounds()
        }
    }

    impl<T> Interval<T> {
        /// Get referenced interval's bounds.
        ///
        /// # Errors
        /// [`EmptyIntervalError`] wrapping a reference to itself when the interval is empty.
        pub fn as_ref_bounds(
            &self,
        ) -> Result<BothBounds<&T>, <Interval<&T> as IntoBounds<&T>>::Error>
        where
            for<'a> Interval<&'a T>: IntoBounds<&'a T>,
        {
            self.as_ref().into_bounds()
        }
    }

    impl<T> From<BothBounds<T>> for Interval<T>
    where
        Self: Bounded<T>,
    {
        fn from(bounds: BothBounds<T>) -> Self {
            Self::from_bounds(bounds)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::interval;
    use Endpoint::{Excluded, Included, Infinite};

    use super::*;

    #[test]
    fn into_bounds() {
        assert!(interval!(0: i32).into_bounds().is_err());
        assert_eq!(
            interval!(<5).into_bounds().unwrap(),
            (Infinite, Excluded(5))
        );
        assert_eq!(
            interval!(<=5).into_bounds().unwrap(),
            (Infinite, Included(5))
        );
        assert_eq!(
            interval!(>5).into_bounds().unwrap(),
            (Excluded(5), Infinite)
        );
        assert_eq!(
            interval!(>=5).into_bounds().unwrap(),
            (Included(5), Infinite)
        );
        assert_eq!(
            interval!((3, 7)).into_bounds().unwrap(),
            (Excluded(3), Excluded(7))
        );
        assert_eq!(
            interval!((3, =7)).into_bounds().unwrap(),
            (Excluded(3), Included(7))
        );
        assert_eq!(
            interval!((=3, 7)).into_bounds().unwrap(),
            (Included(3), Excluded(7))
        );
        assert_eq!(
            interval!([3, 7]).into_bounds().unwrap(),
            (Included(3), Included(7))
        );
        assert_eq!(
            interval!(U: i32).into_bounds().unwrap(),
            (Infinite, Infinite)
        );
    }

    fn left(a: Bound<i32>) -> LBound<i32> {
        a.into()
    }

    fn right(b: Bound<i32>) -> RBound<i32> {
        b.into()
    }

    #[test]
    fn unbounded_infimum() {
        use Bound::{Excluded, Included, Unbounded};
        assert!(left(Unbounded) == left(Unbounded));
        assert!(left(Unbounded) < left(Included(i32::MIN)));
        assert!(left(Unbounded) < left(Excluded(i32::MIN)));
        assert!(left(Unbounded) < left(Included(0)));
        assert!(left(Unbounded) < left(Excluded(0)));
        assert!(left(Unbounded) < left(Included(0)));
        assert!(left(Unbounded) < left(Excluded(0)));
    }

    #[test]
    fn forward_inner_inequality_for_lower() {
        use Bound::{Excluded, Included};
        assert!(left(Included(i32::MIN)) < left(Included(-1_000)));
        assert!(left(Included(i32::MIN)) < left(Excluded(-1_000)));

        assert!(left(Excluded(0)) < left(Excluded(1)));
        assert!(left(Excluded(-1)) < left(Included(1)));
        assert!(left(Included(0)) < left(Excluded(1)));
        assert!(left(Included(0)) < left(Included(1)));

        assert!(left(Excluded(5)) > left(Excluded(1)));
        assert!(left(Excluded(8)) > left(Included(7)));
        assert!(left(Included(0)) > left(Excluded(-1)));
        assert!(left(Included(2)) > left(Included(1)));

        assert!(left(Included(i32::MAX)) > left(Included(1_000)));
        assert!(left(Included(i32::MAX)) > left(Excluded(1_000)));
    }

    #[test]
    fn resolve_equal_included_excluded_lower() {
        use Bound::{Excluded, Included};
        // '>=5' < '>5'
        assert!(left(Included(5)) < left(Excluded(5)));
        // '>100' > '>=100'
        assert!(left(Excluded(100)) > left(Included(100)));
    }

    #[test]
    fn unbounded_supremum() {
        use Bound::{Excluded, Included, Unbounded};
        assert!(right(Unbounded) == right(Unbounded));
        assert!(right(Unbounded) > right(Included(i32::MIN)));
        assert!(right(Unbounded) > right(Excluded(i32::MIN)));
        assert!(right(Unbounded) > right(Included(0)));
        assert!(right(Unbounded) > right(Excluded(0)));
        assert!(right(Unbounded) > right(Included(0)));
        assert!(right(Unbounded) > right(Excluded(0)));
    }

    #[test]
    fn forward_inner_inequality_for_upper() {
        use Bound::{Excluded, Included};
        assert!(right(Included(i32::MIN)) < right(Included(-1_000)));
        assert!(right(Included(i32::MIN)) < right(Excluded(-1_000)));

        assert!(right(Excluded(0)) < right(Excluded(1)));
        assert!(right(Excluded(-1)) < right(Included(1)));
        assert!(right(Included(0)) < right(Excluded(1)));
        assert!(right(Included(0)) < right(Included(1)));

        assert!(right(Excluded(5)) > right(Excluded(1)));
        assert!(right(Excluded(8)) > right(Included(7)));
        assert!(right(Included(0)) > right(Excluded(-1)));
        assert!(right(Included(2)) > right(Included(1)));

        assert!(right(Included(i32::MAX)) > right(Included(1_000)));
        assert!(right(Included(i32::MAX)) > right(Excluded(1_000)));
    }

    #[test]
    fn resolve_equal_included_excluded_upper() {
        use Bound::{Excluded, Included};
        // '<=5' > '<5'
        assert!(right(Included(5)) > right(Excluded(5)));
        // '<100' < '<=100'
        assert!(right(Excluded(100)) < right(Included(100)));
    }

    #[test]
    fn cmp_left_and_right_inf() {
        use Bound::{Excluded, Included, Unbounded};
        assert!(left(Unbounded) < right(Unbounded));
        assert!(left(Unbounded) < right(Included(i32::MIN)));
        assert!(left(Unbounded) < right(Excluded(i32::MIN)));

        assert!(right(Unbounded) > left(Unbounded));
        assert!(right(Unbounded) > left(Included(i32::MAX)));
        assert!(right(Unbounded) > left(Excluded(i32::MAX)));
    }

    #[test]
    fn cmp_left_and_right_finite() {
        use Bound::{Excluded, Included};
        assert!(left(Included(-100)) < right(Included(100)));
        assert!(left(Included(-100)) < right(Included(0)));

        assert!(left(Included(100)) == right(Included(100)));
        assert!(left(Included(100)) > right(Excluded(100)));
        assert!(left(Excluded(100)) > right(Included(100)));
        assert!(left(Excluded(100)) > right(Excluded(100)));
    }
}
