use core::{
    cmp::Ordering,
    ops::{
        Add, Bound, Mul, Neg, Not, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive,
        RangeTo, RangeToInclusive, Sub,
    },
};

use crate::{sided::sided_bounds, Interval, IntoBounds, OneOrPair, Pair, Scalar, Zero};

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
        self.reverse().map(Neg::neg)
    }
}

impl<T, N, Z> Mul<N> for Interval<T>
where
    N: Clone + Scalar + Zero,
    T: PartialOrd + Mul<N, Output = Z>,
    Z: PartialOrd + Zero,
{
    type Output = Interval<Z>;

    fn mul(self, rhs: N) -> Self::Output {
        let was_empty = self.is_empty();
        // Do not simply return `Interval::Empty` on `self.is_empty()`
        // to preserve a (possibly erroneous) structure of the `Interval`,
        // e.g. `[a+1, a) * 2 = [2*a+2, 2*a)`, not a simple `∅`.
        //
        let product = self.map(|x| x * rhs.clone());

        match rhs.cmp_zero() {
            Some(Ordering::Equal) => {
                if was_empty {
                    if product.is_empty() {
                        product
                    } else {
                        // sometimes the invalid interval can become a valid one,
                        // e.g. `[4, 6] * 0 -> [0, 0]`, so we have to force it to be empty again
                        Interval::Empty
                    }
                } else {
                    // non-empty interval should map to the singleton `[0]` interval
                    Interval::Closed((Z::zero(), Z::zero()))
                }
            }
            // if the interval changed its "direction",
            // e.g. `[3, 5] * -1 = [-3, -5]`, we have to reverse it (without changing the signs)
            Some(Ordering::Less) => product.reverse(),
            _ => product,
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

impl<T, N, Z> Mul<Interval<N>> for Interval<T>
where
    N: PartialOrd,
    Interval<N>: IntoBounds<N>,
    T: Clone + PartialOrd + Mul<N, Output = Z>,
    Self: Mul<N, Output = Interval<Z>>,
    Z: Ord,
    Interval<Z>: IntoBounds<Z>,
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

#[cfg(test)]
mod mul_tests {
    use crate::interval;

    use super::*;

    #[test]
    fn mul_by_positive_scalar() {
        assert_eq!(interval!(_: u8) * 100, interval!(_));
        assert_eq!(interval!(< -2) * 5, interval!(< -10));
        assert_eq!(interval!((2, 3)) * 2, interval!((4, 6)));
        assert_eq!(interval!([5.0, 11.0]) * 3.5, interval!([17.5, 38.5]));
        assert_eq!(interval!(..: u8) * 100, interval!(..));
    }

    #[test]
    fn mul_invalid_preserves_structure() {
        assert_eq!(interval!((=4, 3)) * 2, interval!((=8, 6)));
        assert_eq!(interval!([10, -5]) * 6, interval!([60, -30]));
        assert_eq!(interval!([10, -5]) * -6, interval!([30, -60]));
    }

    #[test]
    fn mul_by_negative_scalar() {
        assert_eq!(interval!(_: i8) * -100, interval!(_));
        assert_eq!(interval!(>= 13) * -2, interval!(<= -26));
        assert_eq!(interval!((-2, 3)) * -2, interval!((-6, 4)));
        assert_eq!(interval!([5.0, 11.0]) * -3.5, interval!([-38.5, -17.5]));
        assert_eq!(interval!(..: i8) * -100, interval!(..));
    }

    #[test]
    #[allow(clippy::erasing_op)]
    fn mul_by_zero() {
        assert_eq!(interval!(_: u8) * 0, interval!(_));
        assert_eq!(interval!(< -2) * 0, interval!([0, 0]));
        assert_eq!(interval!((2, 3)) * 0, interval!([0, 0]));
        assert_eq!(interval!([5.0, 11.0]) * 0.0, interval!([0.0, 0.0]));
        assert_eq!(interval!(..: u8) * 0, interval!([0, 0]));
    }

    #[test]
    #[ignore = "unstable mul (inf * 0) operation"]
    #[allow(clippy::erasing_op)]
    fn two_zero_based() {
        let r1 = interval!(< 0);
        let r2 = interval!((0, =1));

        assert_eq!(r1 * 0, interval!(< 0));
        assert_eq!(r1 * 1, interval!(< 0));
        assert_eq!(r1 * r2, interval!(< 0));

        assert_eq!(r2 * 0, interval!([0, 0]));
        assert_eq!(r2 * r1, interval!(< 0));
    }

    #[test]
    fn two_intervals() {
        let ((a, b), (c, d)) = ((-2, 3), (-4, 5));

        assert_eq!(interval!((a, =b)) * c, interval!((= -12, 8)));
        assert_eq!(interval!((a, =b)) * d, interval!((-10, =15)));
        assert_eq!(interval!((=c, d)) * a, interval!((-10, =8)));
        assert_eq!(interval!((=c, d)) * b, interval!((= -12, 15)));

        assert_eq!(
            interval!((a, =b)).mul_bound(Bound::Included(c)),
            interval!((= -12, 8))
        );
        assert_eq!(
            interval!((a, =b)).mul_bound(Bound::Excluded(d)),
            interval!((-10, 15))
        );

        assert_eq!(
            interval!((=c, d)).mul_bound(Bound::Excluded(a)),
            interval!((-10, 8))
        );
        assert_eq!(
            interval!((=c, d)).mul_bound(Bound::Included(b)),
            interval!((= -12, 15))
        );

        assert_eq!(
            interval!((a, =b)) * interval!((=c, d)),
            interval!((= -12, 15))
        );
    }
}
