use core::ops::{
    Add, Bound, Neg, Not, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo,
    RangeToInclusive, Sub,
};

use crate::{sided::sided_bounds, Interval, IntoBounds, OneOrPair, Pair};

impl<T> From<RangeFull> for Interval<T> {
    fn from(_: RangeFull) -> Self {
        Self::Full
    }
}

impl<T> From<RangeFrom<T>> for Interval<T> {
    fn from(RangeFrom { start }: RangeFrom<T>) -> Self {
        Self::GreaterThanOrEqual(start)
    }
}

impl<T> From<RangeTo<T>> for Interval<T> {
    fn from(RangeTo { end }: RangeTo<T>) -> Self {
        Self::LessThan(end)
    }
}

impl<T> From<Range<T>> for Interval<T> {
    fn from(Range { start, end }: Range<T>) -> Self {
        Self::RightOpen((start, end))
    }
}

impl<T> From<RangeInclusive<T>> for Interval<T> {
    fn from(value: RangeInclusive<T>) -> Self {
        Self::Closed(value.into_inner())
    }
}

impl<T> From<RangeToInclusive<T>> for Interval<T> {
    fn from(RangeToInclusive { end }: RangeToInclusive<T>) -> Self {
        Self::LessThanOrEqual(end)
    }
}

#[allow(clippy::match_same_arms)]
impl<T> RangeBounds<T> for Interval<T> {
    fn start_bound(&self) -> Bound<&T> {
        self.as_ref().into_bounds().0
    }

    fn end_bound(&self) -> Bound<&T> {
        self.as_ref().into_bounds().1
    }

    fn contains<U>(&self, item: &U) -> bool
    where
        T: PartialOrd<U>,
        U: ?Sized + PartialOrd<T>,
    {
        if matches!(self, Self::Empty) {
            return false;
        }

        (match self.start_bound() {
            Bound::Included(start) => start <= item,
            Bound::Excluded(start) => start < item,
            Bound::Unbounded => true,
        }) && (match self.end_bound() {
            Bound::Included(end) => item <= end,
            Bound::Excluded(end) => item < end,
            Bound::Unbounded => true,
        })
    }
}

impl<T> From<Pair<Bound<T>>> for Interval<T> {
    fn from(bounds: Pair<Bound<T>>) -> Self {
        match bounds {
            (Bound::Unbounded, Bound::Unbounded) => Self::Full,
            (Bound::Unbounded, Bound::Included(b)) => Self::LessThanOrEqual(b),
            (Bound::Unbounded, Bound::Excluded(b)) => Self::LessThan(b),
            (Bound::Included(a), Bound::Unbounded) => Self::GreaterThanOrEqual(a),
            (Bound::Excluded(a), Bound::Unbounded) => Self::GreaterThan(a),
            (Bound::Included(a), Bound::Included(b)) => Self::Closed((a, b)),
            (Bound::Included(a), Bound::Excluded(b)) => Self::RightOpen((a, b)),
            (Bound::Excluded(a), Bound::Included(b)) => Self::LeftOpen((a, b)),
            (Bound::Excluded(a), Bound::Excluded(b)) => Self::Open((a, b)),
        }
    }
}

impl<T: Add<Output = T> + Clone> Add<T> for Interval<T> {
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        self.map(|x| x + rhs.clone())
    }
}

impl<T: Sub<Output = T> + Clone> Sub<T> for Interval<T> {
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        self.map(|x| x - rhs.clone())
    }
}

// symmetric relative to the _origin_ point (`x::T: x == -x`)
impl<T: Neg<Output = T>> Neg for Interval<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Self::Empty => Self::Empty,
            Self::LessThan(x) => Self::GreaterThan(-x),
            Self::LessThanOrEqual(x) => Self::GreaterThanOrEqual(-x),
            #[cfg(feature = "singleton")]
            Self::Singleton(x) => Self::Singleton(-x),
            Self::GreaterThanOrEqual(x) => Self::LessThanOrEqual(-x),
            Self::GreaterThan(x) => Self::LessThan(-x),
            Self::Open((a, b)) => Self::Open((-b, -a)),
            Self::LeftOpen((a, b)) => Self::RightOpen((-b, -a)),
            Self::RightOpen((a, b)) => Self::LeftOpen((-b, -a)),
            Self::Closed((a, b)) => Self::Closed((-b, -a)),
            Self::Full => Self::Full,
        }
    }
}

#[cfg(feature = "num-traits")]
mod mul_impl {
    use core::ops::Mul;

    use num_traits::Signed;

    use super::{Bound, Interval, IntoBounds, Neg, Pair};

    impl<T, N, Z> Mul<N> for Interval<T>
    where
        N: Signed,
        T: Mul<N, Output = Z>,
        Z: Neg<Output = Z>,
    {
        type Output = Interval<Z>;

        fn mul(self, rhs: N) -> Self::Output {
            let abs = self.map(|x| x * rhs.abs());
            if rhs.is_negative() {
                -abs
            } else {
                abs
            }
        }
    }

    impl<T> Interval<T> {
        /// Auxiliary function to help with multiplying two [`Intervals`].
        fn mul_bound<N, Z>(self, rhs: Bound<N>) -> Interval<Z>
        where
            Self: Mul<N, Output = Interval<Z>>,
        {
            match rhs {
                Bound::Included(n) => self * n,
                Bound::Excluded(n) => (self * n).into_interior(),
                Bound::Unbounded => Interval::Full,
            }
        }
    }

    // TODO: tests
    // (-2, 3] * [-4, 5) == [-12, 15)
    // (-2, 3] * -4 => [-12, 8)
    // (-2, 3] * 5) => (-10, 15)  => [-12, 15)
    // [-4, 5) * (-2 => (-10, 8)
    // [-4, 5) * 3 => [-12, 15]

    impl<T, N, Z> Mul<Interval<N>> for Interval<T>
    where
        // TODO: remove `PartialOrd` bounds by preventing `empty` case
        T: Clone + PartialOrd + Mul<N, Output = Z>,
        N: PartialOrd + Signed,
        Interval<N>: IntoBounds<N>,
        Z: Ord + Neg<Output = Z>,
        Interval<Z>: IntoBounds<Z>,
        // Self: Mul<N, Output = Interval<Z>>,
    {
        type Output = Interval<Z>;

        fn mul(self, rhs: Interval<N>) -> Self::Output {
            if self.is_empty() || rhs.is_empty() {
                return Interval::Empty;
            }

            if self.is_full() || rhs.is_full() {
                return Interval::Full;
            }

            let (rhs_start, rhs_end): Pair<Bound<N>> = rhs.into_bounds();

            let left = self.clone().mul_bound(rhs_start);
            let right = self.mul_bound(rhs_end);
            left.enclosure(right).into()
        }
    }
}

impl<T> Not for Interval<T>
where
    Self: IntoBounds<T>,
{
    type Output = OneOrPair<Self>;

    fn not(self) -> Self::Output {
        if self.is_full() {
            return Self::Empty.into();
        }

        let (start, end) = sided_bounds(self);
        let not_start = Self::from(Pair::from(!start));
        let not_end = Self::from(Pair::from(!end));

        match (not_start, not_end) {
            (Self::Full, i) | (i, Self::Full) => OneOrPair::One(i),
            pair => OneOrPair::Pair(pair),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::interval;

    use super::*;

    #[test]
    fn empty() {
        let i: Interval<f64> = interval!(_);
        assert!(!i.contains(&-100.0));
        assert!(!i.contains(&0.0));
        assert!(!i.contains(&100.0));
    }

    #[test]
    fn less() {
        let i: Interval<f64> = interval!(< -5.0);
        assert!(i.contains(&-100.0));
        assert!(i.contains(&-5.000_001));
        assert!(!i.contains(&-5.0));
        assert!(!i.contains(&-4.999_999));
        assert!(!i.contains(&0.0));
        assert!(!i.contains(&100.0));

        assert_eq!(i, (..-5.0).into());
    }

    #[test]
    fn less_eq() {
        let i: Interval<f64> = interval!(<= -5.0);
        assert!(i.contains(&-100.0));
        assert!(i.contains(&-5.000_001));
        assert!(i.contains(&-5.0));
        assert!(!i.contains(&-4.999_999));
        assert!(!i.contains(&0.0));
        assert!(!i.contains(&100.0));

        assert_eq!(i, (..=-5.0).into());
    }

    #[test]
    fn eq() {
        let i: Interval<f64> = interval!(= 5.0);
        assert!(!i.contains(&-100.0));
        assert!(!i.contains(&0.0));
        assert!(!i.contains(&4.999_999));
        assert!(i.contains(&5.0));
        assert!(!i.contains(&5.000_001));
        assert!(!i.contains(&100.0));
    }

    #[test]
    fn greater_eq() {
        let i: Interval<f64> = interval!(>= 5.0);
        assert!(!i.contains(&-100.0));
        assert!(!i.contains(&0.0));
        assert!(!i.contains(&4.999_999));
        assert!(i.contains(&5.0));
        assert!(i.contains(&5.000_001));
        assert!(i.contains(&100.0));

        assert_eq!(i, (5.0..).into());
    }

    #[test]
    fn greater() {
        let i: Interval<f64> = interval!(> 5.0);
        assert!(!i.contains(&-100.0));
        assert!(!i.contains(&0.0));
        assert!(!i.contains(&4.999_999));
        assert!(!i.contains(&5.0));
        assert!(i.contains(&5.000_001));
        assert!(i.contains(&100.0));
    }

    #[test]
    fn open() {
        let i: Interval<f64> = interval!((5.0, 7.0));
        assert!(!i.contains(&-100.0));
        assert!(!i.contains(&0.0));
        assert!(!i.contains(&4.999_999));
        assert!(!i.contains(&5.0));
        assert!(i.contains(&5.000_001));
        assert!(i.contains(&6.999_999));
        assert!(!i.contains(&7.0));
        assert!(!i.contains(&7.000_001));
        assert!(!i.contains(&100.0));
    }

    #[test]
    fn left_open() {
        let i: Interval<f64> = interval!((5.0, =7.0));
        assert!(!i.contains(&-100.0));
        assert!(!i.contains(&0.0));
        assert!(!i.contains(&4.999_999));
        assert!(!i.contains(&5.0));
        assert!(i.contains(&5.000_001));
        assert!(i.contains(&6.999_999));
        assert!(i.contains(&7.0));
        assert!(!i.contains(&7.000_001));
        assert!(!i.contains(&100.0));
    }

    #[test]
    fn right_open() {
        let i: Interval<f64> = interval!((=5.0, 7.0));
        assert!(!i.contains(&-100.0));
        assert!(!i.contains(&0.0));
        assert!(!i.contains(&4.999_999));
        assert!(i.contains(&5.0));
        assert!(i.contains(&5.000_001));
        assert!(i.contains(&6.999_999));
        assert!(!i.contains(&7.0));
        assert!(!i.contains(&7.000_001));
        assert!(!i.contains(&100.0));

        assert_eq!(i, (5.0..7.0).into());
    }

    #[test]
    fn closed() {
        let i: Interval<f64> = interval!([5.0, 7.0]);
        assert!(!i.contains(&-100.0));
        assert!(!i.contains(&0.0));
        assert!(!i.contains(&4.999_999));
        assert!(i.contains(&5.0));
        assert!(i.contains(&5.000_001));
        assert!(i.contains(&6.999_999));
        assert!(i.contains(&7.0));
        assert!(!i.contains(&7.000_001));
        assert!(!i.contains(&100.0));

        assert_eq!(i, (5.0..=7.0).into());
    }

    #[test]
    fn full() {
        let i: Interval<f64> = interval!(..);
        assert!(i.contains(&-100.0));
        assert!(i.contains(&0.0));
        assert!(i.contains(&4.999_999));
        assert!(i.contains(&5.0));
        assert!(i.contains(&5.000_001));
        assert!(i.contains(&6.999_999));
        assert!(i.contains(&7.0));
        assert!(i.contains(&7.000_001));
        assert!(i.contains(&100.0));

        assert_eq!(i, (..).into());
    }
}

#[cfg(test)]
mod ops_tests {
    use crate::interval;

    use super::*;

    #[test]
    fn neg() {
        let i = -interval!(_: i8);
        assert_eq!(i, interval!(_));

        let i = -interval!(< 5.0);
        assert_eq!(i, interval!(> -5.0));

        let i = -interval!(<= 5.0);
        assert_eq!(i, interval!(>= -5.0));

        let i = -interval!(= 5.0);
        assert_eq!(i, interval!(== -5.0));

        let i = -interval!(>= 5.0);
        assert_eq!(i, interval!(<= -5.0));

        let i = -interval!(> 5.0);
        assert_eq!(i, interval!(< -5.0));

        let i = -interval!((5.0, 7.0));
        assert_eq!(i, interval!((-7.0, -5.0)));

        let i = -interval!((5.0, =7.0));
        assert_eq!(i, interval!((=-7.0, -5.0)));

        let i = -interval!((=5.0, 7.0));
        assert_eq!(i, interval!((-7.0, =-5.0)));

        let i = -interval!([5.0, 7.0]);
        assert_eq!(i, interval!([-7.0, -5.0]));

        let i = -interval!(..: f64);
        assert_eq!(i, interval!(..));
    }

    #[test]
    fn not() {
        let i = !interval!(_: i16);
        assert_eq!(i, OneOrPair::One(interval!(..)));

        let i = !interval!(< 5.0);
        assert_eq!(i, OneOrPair::One(interval!(>= 5.0)));

        let i = !interval!(<= 5.0);
        assert_eq!(i, OneOrPair::One(interval!(> 5.0)));

        let i = interval!(= 5.0);
        assert_eq!(!i, OneOrPair::Pair((interval!(< 5.0), interval!(> 5.0))));

        let i = !interval!(>= 5.0);
        assert_eq!(i, OneOrPair::One(interval!(< 5.0)));

        let i = !interval!(> 5.0);
        assert_eq!(i, OneOrPair::One(interval!(<= 5.0)));

        let i = !interval!((5.0, 7.0));
        assert_eq!(i, OneOrPair::Pair((interval!(<= 5.0), interval!(>= 7.0))));

        let i = !interval!((5.0, =7.0));
        assert_eq!(i, OneOrPair::Pair((interval!(<= 5.0), interval!(> 7.0))));

        let i = !interval!((=5.0, 7.0));
        assert_eq!(i, OneOrPair::Pair((interval!(< 5.0), interval!(>= 7.0))));

        let i = !interval!([5.0, 7.0]);
        assert_eq!(i, OneOrPair::Pair((interval!(< 5.0), interval!(> 7.0))));

        let i = !interval!(..: f64);
        assert_eq!(i, OneOrPair::One(interval!(_)));
    }
}
