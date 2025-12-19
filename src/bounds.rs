use core::ops::{Bound, RangeBounds};

use crate::{sided::sided_bounds, singleton::SingletonBounds, Interval, Pair};

/// Used to convert a range into start and end bounds, consuming the
/// range by value.
///
/// TODO: consider moving into `IntoBounds::into_bounds` when
/// the `feature = "range_into_bounds"` gets stabilized.
pub trait IntoBounds<T>: RangeBounds<T> {
    /// Convert this range into the start and end bounds.
    fn into_bounds(self) -> Pair<Bound<T>>;

    /// Compute the intersection of `self` and `other`.
    fn intersect<R>(self, other: R) -> Pair<Bound<T>>
    where
        Self: Sized,
        T: Ord,
        R: Sized + IntoBounds<T>,
    {
        let (self_start, self_end) = sided_bounds(self);
        let (other_start, other_end) = sided_bounds(other);

        let start = Ord::max(self_start, other_start).into();
        let end = Ord::min(self_end, other_end).into();

        (start, end)
    }

    // TODO: separate `union` function returning `OneOrPair<Self>`

    /// The smallest span containing both `self` and `other`.
    fn enclosure<R>(self, other: R) -> Pair<Bound<T>>
    where
        Self: Sized,
        T: Ord,
        R: Sized + IntoBounds<T>,
    {
        let (self_start, self_end) = sided_bounds(self);
        let (other_start, other_end) = sided_bounds(other);

        let start = Ord::min(self_start, other_start).into();
        let end = Ord::max(self_end, other_end).into();
        (start, end)
    }
}

impl<T> IntoBounds<T> for Interval<T>
where
    Self: SingletonBounds<T>,
{
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
        assert!(Interval::from(a.intersect(b)).reduce().is_empty());

        let a = interval!(>6);
        let b = interval!(<3);
        assert_eq!(a.intersect(b), (Excluded(6), Excluded(3)));
        assert!(Interval::from(a.intersect(b)).reduce().is_empty());

        let a = interval!(>=6);
        let b = interval!(<6);
        assert_eq!(a.intersect(b), (Included(6), Excluded(6)));
        assert!(Interval::from(a.intersect(b)).reduce().is_empty());

        let a = interval!((2, =4));
        let b = interval!((=3, 1));
        assert_eq!(a.intersect(b), (Included(3), Excluded(1)));
        assert!(Interval::from(a.intersect(b)).reduce().is_empty());
    }

    #[test]
    fn intersect_single() {
        let a = interval!(>=6);
        let b = interval!(<=6);
        assert_eq!(a.intersect(b), (Included(6), Included(6)));
        assert_eq!(Interval::from(a.intersect(b)).reduce(), interval!(=6));

        let a = interval!((2, =3));
        let b = interval!((=3, 8));
        assert_eq!(a.intersect(b), (Included(3), Included(3)));
        assert_eq!(Interval::from(a.intersect(b)).reduce(), interval!(==3));
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
}
