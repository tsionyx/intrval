use core::{
    cmp::Ordering,
    fmt,
    ops::{Bound, Neg},
};

use crate::{helper::Pair, singleton::SingletonBounds, Interval, OneOrPair};

pub const LEFT: bool = false;
pub const RIGHT: bool = true;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// The bound of an interval.
pub enum Endpoint<const SIDE: bool, T> {
    /// The point is included in the interval.
    Included(T),
    /// The point is excluded from the interval.
    Excluded(T),
    /// The interval is unbounded in this direction.
    Infinite,
}

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

    pub(crate) fn augment_with_inf(self) -> BothBounds<T> {
        #[allow(clippy::match_bool)]
        match SIDE {
            LEFT => (Endpoint::from(self.into_bound()), Endpoint::Infinite),
            RIGHT => (Endpoint::Infinite, Endpoint::from(self.into_bound())),
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

pub type BothBounds<T> = (Endpoint<LEFT, T>, Endpoint<RIGHT, T>);

/// Convert the two [`Bounded`] values into a pair of `BothBounds`,
/// returning the original pair of values
/// if at least one [conversion][Bounded::into_bounds] fails.
fn pair_into_bounds<B1, B2, T>(a: B1, b: B2) -> Result<Pair<BothBounds<T>>, (B1, B2)>
where
    B1: Bounded<T>,
    B2: Bounded<T>,
{
    let a_bounds = match a.into_bounds() {
        Ok(bounds) => bounds,
        Err(err) => return Err((err.into(), b)),
    };
    let b_bounds = match b.into_bounds() {
        Ok(bounds) => bounds,
        Err(err) => return Err((B1::from_bounds(a_bounds), err.into())),
    };

    Ok((a_bounds, b_bounds))
}

/// Used to convert a value into start and end endpoints, consuming the value.
///
/// TODO: consider matching with `cope::ops::IntoBounds` when
/// the `feature = "range_into_bounds"` gets stabilized.
pub trait Bounded<T>: Sized {
    /// The error signalling conversion to [`Endpoint`]-s fails.
    type Error: Into<Self>;

    /// Create from the given pair of [`Bound`]-s.
    fn from_bounds(bounds: BothBounds<T>) -> Self;

    /// Convert this range into the start and end bounds.
    ///
    /// # Errors
    /// Return [`Self::Error`] if the conversion fails.
    fn into_bounds(self) -> Result<BothBounds<T>, Self::Error>;

    /// Compute the intersection of `self` and `other`.
    ///
    /// # Errors
    /// Return a pair of original values if at least one of [`Self::into_bounds`] fails.
    fn intersect<R>(self, other: R) -> Result<BothBounds<T>, (Self, R)>
    where
        T: Ord,
        R: Bounded<T>,
    {
        let ((self_start, self_end), (other_start, other_end)) = pair_into_bounds(self, other)?;

        let start = self_start.max(other_start);
        let end = self_end.min(other_end);

        Ok((start, end))
    }

    /// The smallest span containing both `self` and `other`
    /// if the values [intersects][Self::intersect] (wrapped in [`OneOrPair::One`]).
    ///
    /// Otherwise, return a [pair][OneOrPair::Pair] of pairs of ordered [`Endpoint`]-s.
    ///
    /// # Errors
    /// Return a pair of original values if at least one of [`Self::into_bounds`] fails.
    fn union<R>(self, other: R) -> Result<OneOrPair<BothBounds<T>>, (Self, R)>
    where
        T: Ord,
        R: Bounded<T>,
    {
        // TODO: use `core::cmp::minmax` when stabilized.
        fn minmax<T: Ord>(v1: T, v2: T) -> [T; 2] {
            if v2 < v1 {
                [v2, v1]
            } else {
                [v1, v2]
            }
        }

        let ((self_start, self_end), (other_start, other_end)) = pair_into_bounds(self, other)?;

        let [min_start, max_start] = minmax(self_start, other_start);
        let [min_end, max_end] = minmax(self_end, other_end);

        let intersection = Interval::from_bounds((max_start.as_ref(), min_end.as_ref()));
        let one_or_pair = if intersection.is_empty() {
            OneOrPair::Pair(((min_start, min_end), (max_start, max_end)))
        } else {
            OneOrPair::One((min_start, max_end))
        };
        Ok(one_or_pair)
    }

    /// The smallest span containing both `self` and `other`.
    ///
    /// # Errors
    /// Return a pair of original values if at least one of [`Self::into_bounds`] fails.
    fn enclosure<R>(self, other: R) -> Result<BothBounds<T>, (Self, R)>
    where
        T: Ord,
        R: Bounded<T>,
    {
        let ((self_start, self_end), (other_start, other_end)) = pair_into_bounds(self, other)?;

        let start = self_start.min(other_start);
        let end = self_end.max(other_end);
        Ok((start, end))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

impl<T: fmt::Debug + fmt::Display> core::error::Error for EmptyIntervalError<T> {}

impl<T> Bounded<T> for Interval<T>
where
    Self: SingletonBounds<T>,
{
    type Error = EmptyIntervalError<T>;

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
            Self::Open((a, b)) => (Excluded(a), Excluded(b)),
            Self::LeftOpen((a, b)) => (Excluded(a), Included(b)),
            Self::RightOpen((a, b)) => (Included(a), Excluded(b)),
            Self::Closed((a, b)) => (Included(a), Included(b)),
            Self::Full => (Infinite, Infinite),
        };
        Ok(bounds)
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

impl<const SIDE: bool, T> Endpoint<SIDE, T> {
    /// Represent the result of operation `Infinite.cmp(Bounded)`,
    /// i.e. the comparison of infinity with the finite number.
    ///
    /// E.g.:
    /// - for the `LEFT` side: `Infinite == -inf < x == Bounded`;
    /// - for the `RIGHT` side: `Infinite == +inf > x == Bounded`;
    ///
    /// This is also the result of operation of comparing `Included` with `Excluded` bounds with the same underlying value:
    /// - for the `LEFT` side: `Included(x) < Excluded(x) ~= Included(x + epsilon)`;
    /// - for the `RIGHT` side: `Included(x) > Excluded(x) ~= Included(x - epsilon)`;
    pub(crate) const fn to_inf_ordering() -> Ordering {
        #[allow(clippy::match_bool)]
        match SIDE {
            // `(a, ...)` can also be represented as `[a + epsilon, ...)`
            // which leads to `[a, ...) < [a + epsilon, ...)`,
            // so `Included(a) < Included(a + epsilon) ~= Excluded(a)`
            LEFT => Ordering::Less,
            // `(..., b)` can also be represented as `(..., b - epsilon]`
            // which leads to `(..., b] > (..., b - epsilon]`
            // so `Included(b) > Included(b - epsilon) ~= Excluded(b)`
            RIGHT => Ordering::Greater,
        }
    }

    pub(crate) const fn as_ref(&self) -> Endpoint<SIDE, &T> {
        match self {
            Self::Included(v) => Endpoint::Included(v),
            Self::Excluded(v) => Endpoint::Excluded(v),
            Self::Infinite => Endpoint::Infinite,
        }
    }

    pub(crate) const fn bound_val(&self) -> Option<&T> {
        match self {
            Self::Included(v) | Self::Excluded(v) => Some(v),
            Self::Infinite => None,
        }
    }
}

impl<const SIDE: bool, T: PartialOrd> PartialOrd for Endpoint<SIDE, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        use Endpoint::{Excluded, Included, Infinite};

        let to_inf_ordering = Self::to_inf_ordering();
        let to_zero_ordering = to_inf_ordering.reverse();

        match (self, other) {
            (Infinite, Infinite) => Some(Ordering::Equal),
            (Infinite, _) => Some(to_inf_ordering),
            (_, Infinite) => Some(to_zero_ordering),

            (Included(a), Included(b)) | (Excluded(a), Excluded(b)) => a.partial_cmp(b),
            (Included(i), Excluded(e)) => i.partial_cmp(e).map(|x| x.then(to_inf_ordering)),
            (Excluded(e), Included(i)) => e.partial_cmp(i).map(|x| x.then(to_zero_ordering)),
        }
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

impl<T> Neg for Endpoint<LEFT, T> {
    type Output = Endpoint<RIGHT, T>;

    fn neg(self) -> Self::Output {
        Endpoint::from(Bound::from(self.swap_inclusion()))
    }
}

impl<T> Neg for Endpoint<RIGHT, T> {
    type Output = Endpoint<LEFT, T>;

    fn neg(self) -> Self::Output {
        Endpoint::from(Bound::from(self.swap_inclusion()))
    }
}

#[cfg(test)]
mod tests {
    use crate::interval;
    use Endpoint::{Excluded, Included, Infinite};

    use super::*;

    #[test]
    fn into_bounds() {
        assert!(interval!(_: i32).into_bounds().is_err());
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
            interval!(..: i32).into_bounds().unwrap(),
            (Infinite, Infinite)
        );
    }

    #[test]
    fn intersect() {
        let a = interval!([3, 7]);
        let b = interval!((5, 10));
        assert_eq!(a.intersect(b).unwrap(), (Excluded(5), Included(7)));

        let a = interval!(<5);
        let b = interval!(>3);
        assert_eq!(a.intersect(b).unwrap(), (Excluded(3), Excluded(5)));

        let a = interval!(<=5);
        let b = interval!(>=-3);
        assert_eq!(a.intersect(b).unwrap(), (Included(-3), Included(5)));

        let a = interval!(..: i32);
        let b = interval!(_: i32);
        assert!(matches!(
            a.intersect(b).unwrap_err(),
            (interval!(..), interval!(_))
        ));
    }

    #[test]
    fn intersect_empty() {
        let a = interval!([1, 2]);
        let b = interval!([3, 4]);
        assert_eq!(a.intersect(b).unwrap(), (Included(3), Included(2)));
        assert!(Interval::from_bounds(a.intersect(b).unwrap())
            .reduce()
            .is_empty());

        let a = interval!(>6);
        let b = interval!(<3);
        assert_eq!(a.intersect(b).unwrap(), (Excluded(6), Excluded(3)));
        assert!(Interval::from_bounds(a.intersect(b).unwrap())
            .reduce()
            .is_empty());

        let a = interval!(>=6);
        let b = interval!(<6);
        assert_eq!(a.intersect(b).unwrap(), (Included(6), Excluded(6)));
        assert!(Interval::from_bounds(a.intersect(b).unwrap())
            .reduce()
            .is_empty());

        let a = interval!((2, =4));
        let b = interval!((=3, 1));
        assert_eq!(a.intersect(b).unwrap(), (Included(3), Excluded(1)));
        assert!(Interval::from_bounds(a.intersect(b).unwrap())
            .reduce()
            .is_empty());
    }

    #[test]
    fn intersect_single() {
        let a = interval!(>=6);
        let b = interval!(<=6);
        assert_eq!(a.intersect(b).unwrap(), (Included(6), Included(6)));
        assert_eq!(
            Interval::from_bounds(a.intersect(b).unwrap()).reduce(),
            interval!(=6)
        );

        let a = interval!((2, =3));
        let b = interval!((=3, 8));
        assert_eq!(a.intersect(b).unwrap(), (Included(3), Included(3)));
        assert_eq!(
            Interval::from_bounds(a.intersect(b).unwrap()).reduce(),
            interval!(==3)
        );
    }

    #[test]
    fn enclosure() {
        let a = interval!([3, 7]);
        let b = interval!((5, 10));
        assert_eq!(a.enclosure(b).unwrap(), (Included(3), Excluded(10)));

        let a = interval!(<5);
        let b = interval!(>3);
        assert_eq!(a.enclosure(b).unwrap(), (Infinite, Infinite));

        let a = interval!(<=-100);
        let b = interval!(>=100);
        assert_eq!(a.enclosure(b).unwrap(), (Infinite, Infinite));

        let a = interval!([1, 2]);
        let b = interval!([3, 4]);
        assert_eq!(a.enclosure(b).unwrap(), (Included(1), Included(4)));
    }

    fn left(a: Bound<i32>) -> Endpoint<LEFT, i32> {
        a.into()
    }

    fn right(b: Bound<i32>) -> Endpoint<RIGHT, i32> {
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
}
