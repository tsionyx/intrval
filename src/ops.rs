use core::{
    cmp::Ordering,
    ops::{
        Add, Bound, Mul, Neg, Not, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive,
        RangeTo, RangeToInclusive, Sub,
    },
};

use crate::{
    bounds::{sided_bounds, SidedBound, LEFT, RIGHT},
    helper::map_pair,
    singleton::SingletonBounds,
    Bounded, Interval, OneOrPair, Pair, Zero,
};

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
        self.contains(item)
    }
}

impl<T: PartialOrd> Interval<T> {
    /// Whether the given interval is completely contained
    /// within this interval.
    pub fn contains_other<R>(&self, other: R) -> bool
    where
        R: for<'a> Bounded<&'a T>,
    {
        let (self_start, self_end) = sided_bounds(self.as_ref());
        let (other_start, other_end) = sided_bounds(other);
        self_start <= other_start && self_end >= other_end
    }

    /// Whether the given point lies completely
    /// to the left/right of the interval,
    /// or maybe completely matches it (in a singleton case).
    fn point_cmp(&self, point: &T) -> Option<Ordering> {
        use Ordering::{Equal, Greater, Less};

        if self.is_empty() {
            return None;
        }

        let (start, end) = sided_bounds(self.as_ref());

        match (start.partial_cmp(&point)?, end.partial_cmp(&point)?) {
            (Less, Less | Equal) | (Equal, Less) => Some(Less),
            (Equal, Equal) => Some(Equal),
            (Equal, Greater) | (Greater, Equal | Greater) => Some(Greater),
            (Less, Greater) | (Greater, Less) => None,
        }
    }
}

impl<T> From<Pair<Bound<T>>> for Interval<T>
where
    Self: Bounded<T>,
{
    fn from(bounds: Pair<Bound<T>>) -> Self {
        Self::from_bounds(bounds)
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
    N: Clone + Zero,
    T: PartialOrd + Mul<N, Output = Z>,
    Z: PartialOrd + Zero,
{
    type Output = Interval<Z>;

    fn mul(self, rhs: N) -> Self::Output {
        let was_empty = self.is_empty();
        // Do not simply return `Interval::Empty` on `self.is_empty()`
        // to preserve a (possibly erroneous) structure of the `Interval`,
        // e.g. `[a+1, a) * 2 = [2*a+2, 2*a)`, not a simple `{}`.
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

#[derive(Debug, PartialEq, Eq)]
/// Attach a tag to data to signify its priority.
enum Prioritized<T> {
    Low(T),
    Normal(T),
}

impl<const SIDE: bool, T: PartialOrd> PartialOrd for Prioritized<SidedBound<SIDE, T>> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let normal_with_low = SidedBound::<SIDE, T>::inf_ordering();
        let low_with_normal = normal_with_low.reverse();

        let (a, b) = map_pair((self, other), |p| p.as_ref().into_data().bound_val());
        match (self, other) {
            (Self::Low(_), Self::Normal(_)) if a.is_some() && b.is_some() && a != b => {
                Some(low_with_normal)
            }
            (Self::Normal(_), Self::Low(_)) if a.is_some() && b.is_some() && a != b => {
                Some(normal_with_low)
            }
            other => {
                let (a, b) = map_pair(other, |x| x.as_ref().into_data());
                a.partial_cmp(b)
            }
        }
    }
}

impl<const SIDE: bool, T: Ord> Ord for Prioritized<SidedBound<SIDE, T>> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other)
            .expect("comparison between Ord values failed")
    }
}

impl<T> Prioritized<T> {
    fn into_data(self) -> T {
        match self {
            Self::Low(v) | Self::Normal(v) => v,
        }
    }

    fn map<U, F>(self, mut f: F) -> Prioritized<U>
    where
        F: FnMut(T) -> U,
    {
        match self {
            Self::Low(v) => Prioritized::Low(f(v)),
            Self::Normal(v) => Prioritized::Normal(f(v)),
        }
    }

    const fn as_ref(&self) -> Prioritized<&T> {
        match self {
            Self::Low(v) => Prioritized::Low(v),
            Self::Normal(v) => Prioritized::Normal(v),
        }
    }
}

impl<T> From<T> for Prioritized<T> {
    fn from(value: T) -> Self {
        Self::Normal(value)
    }
}

impl<T> Interval<T> {
    /// Auxiliary function to help with multiplying two [`Intervals`].
    fn mul_bound<const SIDE: bool, N, Z>(
        self,
        rhs: SidedBound<SIDE, N>,
    ) -> Pair<Prioritized<Bound<Z>>>
    where
        Self: Bounded<T> + Mul<N, Output = Interval<Z>>,
        T: Zero + PartialOrd,
        Z: Zero,
        Interval<Z>: SingletonBounds<Z>,
    {
        let zero_point = T::zero();
        let zero_result = || Z::zero();

        let bounds = match Bound::from(rhs) {
            Bound::Included(n) => (self * n).into_bounds(),
            Bound::Excluded(n) => {
                let ref_bounds = <[_; _]>::from(self.as_ref().into_bounds());
                let preserve_zero = ref_bounds.contains(&Bound::Included(&zero_point));
                let product = (self * n).into_interior();

                map_pair(product.into_bounds(), |bound| {
                    if matches!(bound, Bound::Excluded(ref t) if t.cmp_zero().map_or(false, Ordering::is_eq))
                        && preserve_zero
                    {
                        Bound::Included(zero_result())
                    } else {
                        bound
                    }
                })
            }
            Bound::Unbounded => {
                let zero_bound = || {
                    if self.contains(&zero_point) {
                        Bound::Included(zero_result())
                    } else {
                        Bound::Excluded(zero_result())
                    }
                };
                let inf = || Bound::Unbounded;

                match (self.point_cmp(&zero_point), SIDE) {
                    // the interval is entirely non-negative multiplied by +inf (`[+a, +b] * +inf = [0, +inf)`)
                    // the interval is entirely non-positive multiplied by -inf (`[-a, -b] * -inf = [0, +inf)`)
                    (Some(Ordering::Greater), RIGHT) | (Some(Ordering::Less), LEFT) => {
                        return (Prioritized::Low(zero_bound()), inf().into());
                    }
                    // the interval is entirely non-positive multiplied by +inf (`[-a, -b] * +inf = (-inf, 0]`)
                    // the interval is entirely non-negative multiplied by -inf (`[+a, +b] * -inf = (-inf, 0]`)
                    (Some(Ordering::Less), RIGHT) | (Some(Ordering::Greater), LEFT) => {
                        return (inf().into(), Prioritized::Low(zero_bound()));
                    }
                    // [0] * +/- inf = [-inf, +inf] * 0 = [0, 0]
                    (Some(Ordering::Equal), _) => (zero_bound(), zero_bound()),
                    // the interval spans both negative and positive values
                    (None, _) => (inf(), inf()),
                }
            }
        };
        map_pair(bounds, Prioritized::Normal)
    }
}

impl<T, N, Z> Mul<Interval<N>> for Interval<T>
where
    N: Clone + PartialOrd + Zero,
    T: Clone + PartialOrd + Mul<N, Output = Z> + Zero,
    Z: Ord + Zero,
    Self: Bounded<T>,
    Interval<Z>: SingletonBounds<Z>,
{
    type Output = Interval<Z>;

    fn mul(self, rhs: Interval<N>) -> Self::Output {
        if self.is_empty() || rhs.is_empty() {
            return Interval::Empty;
        }

        let (rhs_start, rhs_end) = sided_bounds(rhs);

        let (left_start, left_end) = self.clone().mul_bound(rhs_start);
        let (right_start, right_end) = self.mul_bound(rhs_end);

        let (start1, start2) = map_pair((left_start, right_start), |x| {
            x.map(SidedBound::<LEFT, _>::new)
        });
        let (end1, end2) = map_pair((left_end, right_end), |x| {
            x.map(SidedBound::<RIGHT, _>::new)
        });

        let start = start1.min(start2).into_data();
        let end = end1.max(end2).into_data();

        let interval = Interval::from((start, end));
        if interval.is_empty() {
            interval.into_closure()
        } else {
            interval
        }
    }
}

impl<T> Interval<T>
where
    Self: Bounded<T>,
{
    /// Compute the complement of this interval.
    ///
    /// If the [`Interval`] has two endpoints, the complement will be represented
    /// as a pair of intervals (one to the left of the start, one to the right of the end).
    pub fn complement(self) -> OneOrPair<Self> {
        if self.is_full() {
            return Self::Empty.into();
        }

        let (start, end) = sided_bounds(self);
        let not_start = Self::from_bounds((!start).into());
        let not_end = Self::from_bounds((!end).into());

        match (not_start, not_end) {
            (Self::Full, i) | (i, Self::Full) => OneOrPair::One(i),
            pair => OneOrPair::Pair(pair),
        }
    }
}

impl<T> Not for Interval<T>
where
    Self: Bounded<T>,
{
    type Output = OneOrPair<Self>;

    fn not(self) -> Self::Output {
        self.complement()
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

    fn cmp_mul_bound<T: PartialEq + core::fmt::Debug>(
        product: Pair<Prioritized<Bound<T>>>,
        result_interval: Interval<T>,
    ) where
        Interval<T>: Bounded<T>,
    {
        let pair = map_pair(product, Prioritized::into_data);
        assert_eq!(pair, result_interval.into_bounds());
    }

    #[test]
    fn mul_to_infinite_bound() {
        let (neg_inf, pos_inf) = sided_bounds(interval!(..: i32));
        cmp_mul_bound(interval!(>= 0).mul_bound(pos_inf), interval!(>= 0));
        cmp_mul_bound(interval!(> 0).mul_bound(pos_inf), interval!(> 0));
        cmp_mul_bound(interval!(>= 1).mul_bound(pos_inf), interval!(> 0));
        cmp_mul_bound(interval!(<= 0).mul_bound(pos_inf), interval!(<= 0));
        cmp_mul_bound(interval!(< 0).mul_bound(pos_inf), interval!(< 0));
        cmp_mul_bound(interval!(<= -1).mul_bound(pos_inf), interval!(< 0));
        cmp_mul_bound(interval!(<= 1).mul_bound(pos_inf), interval!(..));

        cmp_mul_bound(interval!(>= 0).mul_bound(neg_inf), interval!(<= 0));
        cmp_mul_bound(interval!(> 0).mul_bound(neg_inf), interval!(< 0));
        cmp_mul_bound(interval!(>= 1).mul_bound(neg_inf), interval!(< 0));
        cmp_mul_bound(interval!(<= 0).mul_bound(neg_inf), interval!(>= 0));
        cmp_mul_bound(interval!(< 0).mul_bound(neg_inf), interval!(> 0));
        cmp_mul_bound(interval!(<= -1).mul_bound(neg_inf), interval!(> 0));
        cmp_mul_bound(interval!(<= 1).mul_bound(neg_inf), interval!(..));
    }

    #[test]
    fn zero_unbounded_to_finite() {
        let upper_bound = SidedBound::<RIGHT, _>::new(Bound::Excluded(5));
        cmp_mul_bound(interval!(>= 0).mul_bound(upper_bound), interval!(>= 0));
    }

    #[test]
    #[allow(clippy::erasing_op)]
    fn two_zero_based() {
        let r1 = interval!(< 0);
        let r2 = interval!((0, =1));

        assert_eq!(r1 * 0, interval!([0, 0]));
        assert_eq!(r1 * 1, interval!(< 0));
        assert_eq!(r1 * r2, interval!(< 0));

        assert_eq!(r2 * 0, interval!([0, 0]));

        let (rhs_start, rhs_end) = sided_bounds(r1);
        cmp_mul_bound(r2.mul_bound(rhs_start), interval!(< 0));
        cmp_mul_bound(r2.mul_bound(rhs_end), interval!((0, 0)));
        assert_eq!(r2 * r1, interval!(< 0));
    }

    #[test]
    fn zero_singleton_with_strict_positive() {
        let r1 = interval!(== 0);
        let r2 = interval!(> 0);

        assert_eq!(r1 * r2, interval!([0, 0]));
        assert_eq!(r2 * r1, interval!([0, 0]));
    }

    #[test]
    fn zero_singleton_with_pos_and_neg() {
        let r1 = interval!(== 0);
        let r2 = interval!(> -1);

        assert_eq!(r1 * r2, interval!([0, 0]));
        assert_eq!(r2 * r1, interval!([0, 0]));
    }

    #[test]
    fn zero_singleton_with_open() {
        let r1 = interval!(== 0);
        let r2 = interval!((0, 1));

        assert_eq!(r1 * r2, interval!([0, 0]));
        assert_eq!(r2 * r1, interval!([0, 0]));
    }

    #[test]
    fn singleton_with_negative() {
        let r1 = interval!(== 1);
        let r2 = interval!(<= -1);

        assert_eq!(r1 * r2, interval!(<= -1));
        assert_eq!(r2 * r1, interval!(<= -1));
    }

    #[test]
    fn singleton_with_strict_negative() {
        let r1 = interval!(== 1);
        let r2 = interval!(< -1);

        assert_eq!(r1 * r2, interval!(< -1));
        assert_eq!(r2 * r1, interval!(< -1));
    }

    #[test]
    fn open_to_closed_infinite_zero_based() {
        let r1 = interval!(>= 0);
        let r2 = interval!(> 0);

        assert_eq!(r1 * r2, interval!(>= 0));
        assert_eq!(r2 * r1, interval!(>= 0));
    }

    #[test]
    fn open_to_closed_infinite() {
        let r1 = interval!(> 1);
        let r2 = interval!(>= 1);

        assert_eq!(r1 * r2, interval!(> 1));
        assert_eq!(r2 * r1, interval!(> 1));
    }

    #[test]
    fn open_to_closed_infinite_with_zero_inside() {
        let r1 = interval!(> -1);
        let r2 = interval!(>= -1);

        assert_eq!(r1 * r2, interval!(..));
        assert_eq!(r2 * r1, interval!(..));
    }

    #[test]
    fn bounded_to_unbounded_zero() {
        let r1 = interval!(>= 0);
        let r2 = interval!((0, 5));

        assert_eq!(r1 * r2, interval!(>= 0));
        assert_eq!(r2 * r1, interval!(>= 0));
    }

    #[test]
    fn bounded_to_unbounded() {
        let r1 = interval!([0, 5]);
        let r2 = interval!(> -1);

        assert_eq!(r1 * r2, interval!(> -5));
        assert_eq!(r2 * r1, interval!(> -5));
    }

    #[test]
    fn two_intervals() {
        let ((a, b), (c, d)) = ((-2, 3), (-4, 5));

        assert_eq!(interval!((a, =b)) * c, interval!((= -12, 8)));
        assert_eq!(interval!((a, =b)) * d, interval!((-10, =15)));
        assert_eq!(interval!((=c, d)) * a, interval!((-10, =8)));
        assert_eq!(interval!((=c, d)) * b, interval!((= -12, 15)));

        let bounds1 = sided_bounds(interval!((a, =b)));
        let bounds2 = sided_bounds(interval!((=c, d)));
        cmp_mul_bound(
            interval!((a, =b)).mul_bound(bounds2.0),
            interval!((= -12, 8)),
        );
        cmp_mul_bound(
            interval!((a, =b)).mul_bound(bounds2.1),
            interval!((-10, 15)),
        );

        cmp_mul_bound(interval!((=c, d)).mul_bound(bounds1.0), interval!((-10, 8)));
        cmp_mul_bound(
            interval!((=c, d)).mul_bound(bounds1.1),
            interval!((= -12, 15)),
        );

        assert_eq!(
            interval!((a, =b)) * interval!((=c, d)),
            interval!((= -12, 15))
        );
    }
}
