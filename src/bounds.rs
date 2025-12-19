use core::{
    cmp::Ordering,
    ops::{Bound, Not, RangeBounds},
};

use crate::{singleton::SingletonBounds, Interval, OneOrPair, Pair};

/// Used to convert a range into start and end bounds, consuming the
/// range by value.
///
/// TODO: consider matching with into `cope::ops::IntoBounds` when
/// the `feature = "range_into_bounds"` gets stabilized.
pub trait Bounded<T>: RangeBounds<T> {
    /// Create from the given pair of [`Bound`]-s.
    fn from_bounds(bounds: Pair<Bound<T>>) -> Self;

    /// Convert this range into the start and end [`Bound`]-s.
    fn into_bounds(self) -> Pair<Bound<T>>;

    /// Compute the intersection of `self` and `other`.
    fn intersect<R>(self, other: R) -> Pair<Bound<T>>
    where
        Self: Sized,
        T: Ord,
        R: Sized + Bounded<T>,
    {
        let (self_start, self_end) = sided_bounds(self);
        let (other_start, other_end) = sided_bounds(other);

        let start = Ord::max(self_start, other_start).into();
        let end = Ord::min(self_end, other_end).into();

        (start, end)
    }

    /// The smallest span containing both `self` and `other`
    /// if the values [intersects][Self::intersect] (wrapped in [`OneOrPair::One`]).
    ///
    /// Otherwise, return a [pair][OneOrPair::Pair] of pairs of ordered [`Bound`]-s.
    fn union<R>(self, other: R) -> OneOrPair<Pair<Bound<T>>>
    where
        Self: Sized,
        T: Ord,
        R: Sized + Bounded<T>,
    {
        // TODO: use `core::cmp::minmax` when stabilized.
        fn minmax<T: Ord>(v1: T, v2: T) -> [T; 2] {
            if v2 < v1 {
                [v2, v1]
            } else {
                [v1, v2]
            }
        }

        let (self_start, self_end) = sided_bounds(self);
        let (other_start, other_end) = sided_bounds(other);

        let [min_start, max_start] = minmax(self_start, other_start);
        let [min_end, max_end] = minmax(self_end, other_end);

        let intersection = Interval::from((max_start.as_ref(), min_end.as_ref()));
        if intersection.is_empty() {
            OneOrPair::Pair((
                (min_start.into(), min_end.into()),
                (max_start.into(), max_end.into()),
            ))
        } else {
            OneOrPair::One((min_start.into(), max_end.into()))
        }
    }

    /// The smallest span containing both `self` and `other`.
    fn enclosure<R>(self, other: R) -> Pair<Bound<T>>
    where
        Self: Sized,
        T: Ord,
        R: Sized + Bounded<T>,
    {
        let (self_start, self_end) = sided_bounds(self);
        let (other_start, other_end) = sided_bounds(other);

        let start = Ord::min(self_start, other_start).into();
        let end = Ord::max(self_end, other_end).into();
        (start, end)
    }
}

impl<T> Bounded<T> for Interval<T>
where
    Self: SingletonBounds<T>,
{
    fn from_bounds(bounds: Pair<Bound<T>>) -> Self {
        use Bound::{Excluded, Included, Unbounded};

        match bounds {
            (Unbounded, Unbounded) => Self::Full,
            (Unbounded, Included(b)) => Self::LessThanOrEqual(b),
            (Unbounded, Excluded(b)) => Self::LessThan(b),
            (Included(a), Unbounded) => Self::GreaterThanOrEqual(a),
            (Excluded(a), Unbounded) => Self::GreaterThan(a),
            (Included(a), Included(b)) => Self::Closed((a, b)),
            (Included(a), Excluded(b)) => Self::RightOpen((a, b)),
            (Excluded(a), Included(b)) => Self::LeftOpen((a, b)),
            (Excluded(a), Excluded(b)) => Self::Open((a, b)),
        }
    }

    fn into_bounds(self) -> Pair<Bound<T>> {
        use Bound::{Excluded, Included, Unbounded};

        // TODO: separate the `Interval<T>` into enum variants for empty and non-empty intervals.

        #[allow(clippy::match_same_arms)]
        match self {
            // TODO: provide a better solution, e.g. `(Excluded(T::default()), Excluded(T::default()))`
            // in order for `Self::contains` to always return `false`.
            Self::Empty => (Unbounded, Unbounded),
            Self::LessThan(b) => (Unbounded, Excluded(b)),
            Self::LessThanOrEqual(b) => (Unbounded, Included(b)),
            #[cfg(feature = "singleton")]
            Self::Singleton(x) => <Self as SingletonBounds<T>>::value_into_bounds(x),
            Self::GreaterThanOrEqual(a) => (Included(a), Unbounded),
            Self::GreaterThan(a) => (Excluded(a), Unbounded),
            Self::Open((a, b)) => (Excluded(a), Excluded(b)),
            Self::LeftOpen((a, b)) => (Excluded(a), Included(b)),
            Self::RightOpen((a, b)) => (Included(a), Excluded(b)),
            Self::Closed((a, b)) => (Included(a), Included(b)),
            Self::Full => (Unbounded, Unbounded),
        }
    }
}

impl<T> From<(SidedBound<LEFT, T>, SidedBound<RIGHT, T>)> for Interval<T>
where
    Self: Bounded<T>,
{
    fn from((a, b): (SidedBound<LEFT, T>, SidedBound<RIGHT, T>)) -> Self {
        let bounds = (Bound::from(a), Bound::from(b));
        Self::from_bounds(bounds)
    }
}

pub const LEFT: bool = false;
pub const RIGHT: bool = true;

#[allow(unnameable_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SidedBound<const SIDE: bool, T>(Bound<T>);

impl<const SIDE: bool, T> SidedBound<SIDE, T> {
    pub(crate) const fn new(value: Bound<T>) -> Self {
        Self(value)
    }

    /// Represent the result of operation `Unbounded.cmp(Bounded)`,
    /// i.e. the comparison of infinity with the finite number.
    ///
    /// E.g.:
    /// - for the `LEFT` side: `Unbounded == -inf < x == Bounded`;
    /// - for the `RIGHT` side: `Unbounded == +inf > x == Bounded`;
    ///
    /// This is also the result of operation of comparing `Included` with `Excluded` bounds with the same underlying value:
    /// - for the `LEFT` side: `Included(x) < Excluded(x) ~= Included(x + epsilon)`;
    /// - for the `RIGHT` side: `Included(x) > Excluded(x) ~= Included(x - epsilon)`;
    pub(crate) const fn inf_ordering() -> Ordering {
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

    pub(crate) const fn as_ref(&self) -> SidedBound<SIDE, &T> {
        let bound = match &self.0 {
            Bound::Included(v) => Bound::Included(v),
            Bound::Excluded(v) => Bound::Excluded(v),
            Bound::Unbounded => Bound::Unbounded,
        };
        SidedBound::new(bound)
    }

    pub(crate) const fn bound_val(&self) -> Option<&T> {
        match &self.0 {
            Bound::Included(v) | Bound::Excluded(v) => Some(v),
            Bound::Unbounded => None,
        }
    }
}

pub fn sided_bounds<I: Bounded<T>, T>(interval: I) -> (SidedBound<LEFT, T>, SidedBound<RIGHT, T>) {
    let (start, end) = interval.into_bounds();
    (SidedBound::new(start), SidedBound::new(end))
}

impl<const SIDE: bool, T> From<SidedBound<SIDE, T>> for Bound<T> {
    fn from(value: SidedBound<SIDE, T>) -> Self {
        value.0
    }
}

impl<const SIDE: bool, T: PartialOrd> PartialOrd for SidedBound<SIDE, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        use Bound::{Excluded, Included, Unbounded};

        let inf_ordering = Self::inf_ordering();
        let with_inf_ordering = inf_ordering.reverse();

        match (&self.0, &other.0) {
            (Unbounded, Unbounded) => Some(Ordering::Equal),
            (Unbounded, _) => Some(inf_ordering),
            (_, Unbounded) => Some(with_inf_ordering),

            (Included(a), Included(b)) | (Excluded(a), Excluded(b)) => a.partial_cmp(b),
            (Included(i), Excluded(e)) => i.partial_cmp(e).map(|x| x.then(inf_ordering)),
            (Excluded(e), Included(i)) => e.partial_cmp(i).map(|x| x.then(with_inf_ordering)),
        }
    }
}

impl<const SIDE: bool, T> PartialEq<T> for SidedBound<SIDE, T>
where
    T: PartialEq + Clone,
{
    fn eq(&self, point: &T) -> bool {
        self.eq(&Self::new(Bound::Included(point.clone())))
    }
}

impl<const SIDE: bool, T: PartialOrd + Clone> PartialOrd<T> for SidedBound<SIDE, T> {
    fn partial_cmp(&self, point: &T) -> Option<Ordering> {
        self.partial_cmp(&Self::new(Bound::Included(point.clone())))
    }
}

impl<const SIDE: bool, T: Ord> Ord for SidedBound<SIDE, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other)
            .expect("comparison between Ord values failed")
    }
}

fn bound_negate_inclusion<T>(bound: Bound<T>) -> Bound<T> {
    match bound {
        Bound::Included(v) => Bound::Excluded(v),
        Bound::Excluded(v) => Bound::Included(v),
        Bound::Unbounded => Bound::Unbounded,
    }
}

impl<T> Not for SidedBound<LEFT, T> {
    type Output = SidedBound<RIGHT, T>;

    fn not(self) -> Self::Output {
        SidedBound::new(bound_negate_inclusion(self.0))
    }
}

impl<T> Not for SidedBound<RIGHT, T> {
    type Output = SidedBound<LEFT, T>;

    fn not(self) -> Self::Output {
        SidedBound::new(bound_negate_inclusion(self.0))
    }
}

impl<const SIDE: bool, T> From<SidedBound<SIDE, T>> for Pair<Bound<T>> {
    fn from(value: SidedBound<SIDE, T>) -> Self {
        #[allow(clippy::match_bool)]
        match SIDE {
            LEFT => (value.0, Bound::Unbounded),
            RIGHT => (Bound::Unbounded, value.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use Bound::{Excluded, Included, Unbounded};

    use crate::interval;

    use super::*;

    #[test]
    fn into_bounds() {
        assert_eq!(interval!(_: i32).into_bounds(), (Unbounded, Unbounded));
        assert_eq!(interval!(<5).into_bounds(), (Unbounded, Excluded(5)));
        assert_eq!(interval!(<=5).into_bounds(), (Unbounded, Included(5)));
        assert_eq!(interval!(>5).into_bounds(), (Excluded(5), Unbounded));
        assert_eq!(interval!(>=5).into_bounds(), (Included(5), Unbounded));
        assert_eq!(interval!((3, 7)).into_bounds(), (Excluded(3), Excluded(7)));
        assert_eq!(interval!((3, =7)).into_bounds(), (Excluded(3), Included(7)));
        assert_eq!(interval!((=3, 7)).into_bounds(), (Included(3), Excluded(7)));
        assert_eq!(interval!([3, 7]).into_bounds(), (Included(3), Included(7)));
        assert_eq!(interval!(..: i32).into_bounds(), (Unbounded, Unbounded));
    }

    #[test]
    fn intersect() {
        let a = interval!([3, 7]);
        let b = interval!((5, 10));
        assert_eq!(a.intersect(b), (Excluded(5), Included(7)));

        let a = interval!(<5);
        let b = interval!(>3);
        assert_eq!(a.intersect(b), (Excluded(3), Excluded(5)));

        let a = interval!(<=5);
        let b = interval!(>=-3);
        assert_eq!(a.intersect(b), (Included(-3), Included(5)));

        let a = interval!(..: i32);
        let b = interval!(_: i32);
        assert_eq!(a.intersect(b), (Unbounded, Unbounded));
    }

    #[test]
    fn intersect_empty() {
        let a = interval!([1, 2]);
        let b = interval!([3, 4]);
        assert_eq!(a.intersect(b), (Included(3), Included(2)));
        assert!(Interval::from_bounds(a.intersect(b)).reduce().is_empty());

        let a = interval!(>6);
        let b = interval!(<3);
        assert_eq!(a.intersect(b), (Excluded(6), Excluded(3)));
        assert!(Interval::from_bounds(a.intersect(b)).reduce().is_empty());

        let a = interval!(>=6);
        let b = interval!(<6);
        assert_eq!(a.intersect(b), (Included(6), Excluded(6)));
        assert!(Interval::from_bounds(a.intersect(b)).reduce().is_empty());

        let a = interval!((2, =4));
        let b = interval!((=3, 1));
        assert_eq!(a.intersect(b), (Included(3), Excluded(1)));
        assert!(Interval::from_bounds(a.intersect(b)).reduce().is_empty());
    }

    #[test]
    fn intersect_single() {
        let a = interval!(>=6);
        let b = interval!(<=6);
        assert_eq!(a.intersect(b), (Included(6), Included(6)));
        assert_eq!(
            Interval::from_bounds(a.intersect(b)).reduce(),
            interval!(=6)
        );

        let a = interval!((2, =3));
        let b = interval!((=3, 8));
        assert_eq!(a.intersect(b), (Included(3), Included(3)));
        assert_eq!(
            Interval::from_bounds(a.intersect(b)).reduce(),
            interval!(==3)
        );
    }

    #[test]
    fn enclosure() {
        let a = interval!([3, 7]);
        let b = interval!((5, 10));
        assert_eq!(a.enclosure(b), (Included(3), Excluded(10)));

        let a = interval!(<5);
        let b = interval!(>3);
        assert_eq!(a.enclosure(b), (Unbounded, Unbounded));

        let a = interval!(<=-100);
        let b = interval!(>=100);
        assert_eq!(a.enclosure(b), (Unbounded, Unbounded));

        let a = interval!([1, 2]);
        let b = interval!([3, 4]);
        assert_eq!(a.enclosure(b), (Included(1), Included(4)));
    }

    fn left(b: Bound<i32>) -> SidedBound<LEFT, i32> {
        SidedBound::new(b)
    }

    fn right(b: Bound<i32>) -> SidedBound<RIGHT, i32> {
        SidedBound::new(b)
    }

    #[test]
    fn unbounded_infimum() {
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
        // '>=5' < '>5'
        assert!(left(Included(5)) < left(Excluded(5)));
        // '>100' > '>=100'
        assert!(left(Excluded(100)) > left(Included(100)));
    }

    #[test]
    fn unbounded_supremum() {
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
        // '<=5' > '<5'
        assert!(right(Included(5)) > right(Excluded(5)));
        // '<100' < '<=100'
        assert!(right(Excluded(100)) < right(Included(100)));
    }
}
