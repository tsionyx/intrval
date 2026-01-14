use core::{
    cmp::Ordering,
    ops::{
        Add, BitAnd, BitOr, Bound, Div, Mul, Neg, Not, Range, RangeBounds, RangeFrom, RangeFull,
        RangeInclusive, RangeTo, RangeToInclusive, Sub,
    },
};

use crate::{
    bounds::{inf_ordering, Bounded, Endpoint, LBound, RBound, LEFT, RIGHT},
    helper::map_pair,
    Interval, OneOrPair, Zero,
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
impl<T> RangeBounds<T> for Interval<T>
where
    T: PartialOrd,
{
    fn start_bound(&self) -> Bound<&T> {
        self.as_ref()
            .into_bounds()
            .map_or(Bound::Unbounded, |(a, _b)| a.into_bound())
    }

    fn end_bound(&self) -> Bound<&T> {
        self.as_ref()
            .into_bounds()
            .map_or(Bound::Unbounded, |(_a, b)| b.into_bound())
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
    /// Returns `true` if there is a non-empty gap between `self` and `other`.
    /// This implies the `self.union(other)` guaranteed to be a whole `Interval`
    /// without [jumps](https://en.wikipedia.org/wiki/Classification_of_discontinuities)
    ///
    /// This is **not equivalent** to checking for an empty intersection.
    /// because two intervals can 'touch' at a single point, where
    /// one of them includes the point and another exclusively approaches it (open interval).
    /// This case is considered as _not disjoint_ (i.e. _joint_),
    /// because there is no gap between the intervals, even though their intersection is empty.
    pub fn is_disjoint<'other, R>(&self, other: R) -> bool
    where
        T: Ord + 'other,
        R: Bounded<&'other T>,
    {
        // if at least one of the intervals is empty, they cannot be disjoint

        let Ok((self_start, self_end)) = self.as_ref().into_bounds() else {
            return false;
        };
        let Ok((other_start, other_end)) = other.into_bounds() else {
            return false;
        };

        let max_start = self_start.max(other_start);
        let min_end = self_end.min(other_end);

        let empty_intersection = max_start > min_end;
        // `[x, x)` or `(x, x]` is an empty gap for empty intersection => intervals are joint
        // `(x, x)` is an empty gap for empty intersection `(x, x)` => intervals are disjoint
        let empty_gap = !min_end > !max_start;
        empty_intersection && !empty_gap
    }

    /// Returns `true` if the interval lies completely within another,
    /// i.e., `other` contains at least all the values in `self`.
    pub fn is_sub<'other, R>(&self, other: R) -> bool
    where
        T: 'other,
        R: Bounded<&'other T>,
    {
        let Ok((self_start, self_end)) = self.as_ref().into_bounds() else {
            // the empty interval is contained in any interval
            return true;
        };

        let Ok((other_start, other_end)) = other.into_bounds() else {
            // no interval can be inside an empty interval
            // (except the empty one, which is handled above)
            return false;
        };

        other_start <= self_start && other_end >= self_end
    }

    /// Returns `true` if the interval completely contains another,
    /// i.e., `self` contains at least all the values in `other`.
    pub fn is_super<'other, R>(&self, other: R) -> bool
    where
        T: 'other,
        R: Bounded<&'other T>,
    {
        let Ok((other_start, other_end)) = other.into_bounds() else {
            // the empty interval is contained in any interval
            return true;
        };

        let Ok((self_start, self_end)) = self.as_ref().into_bounds() else {
            // no interval can be inside an empty interval
            // (except the empty one, which is handled above)
            return false;
        };

        self_start <= other_start && self_end >= other_end
    }

    /// Whether the given point lies completely
    /// to the left/right of the interval,
    /// or maybe completely matches it (in a singleton case).
    ///
    /// This should be an impl of `PartialOrd<T> for Interval<T>`,
    /// but this would require an impl of `PartialEq<T> for Interval<T>`,
    /// which does not make much sense.
    ///
    /// Even worse, multiple existing implementations of
    /// `PartialEq` for `Interval<T>` breaks
    /// the type inference system in expressions like
    /// `interval < interval_like.into()`.
    /// This lowers much of ergonomics of using compare operators.
    pub fn point_cmp(&self, point: &T) -> Option<Ordering> {
        use Ordering::{Equal, Greater, Less};

        let Ok((start, end)) = self.as_ref().into_bounds() else {
            return None;
        };

        match (start.partial_cmp(&point)?, end.partial_cmp(&point)?) {
            (Less, Less | Equal) | (Equal, Less) => Some(Less),
            (Equal, Equal) => Some(Equal),
            (Equal, Greater) | (Greater, Equal | Greater) => Some(Greater),
            (Less, Greater) | (Greater, Less) => None,
        }
    }
}

impl<T: Add<Output = T> + Clone> Add<T> for Interval<T> {
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        self.map(|x| x + rhs.clone())
    }
}

impl<T, U, Z> Add<Interval<U>> for Interval<T>
where
    T: Zero + Add<U, Output = Z>,
    U: Zero,
    Self: Bounded<T>,
    Interval<U>: Bounded<U>,
    Interval<Z>: Bounded<Z>,
{
    type Output = Interval<Z>;

    fn add(self, rhs: Interval<U>) -> Self::Output {
        let (left, right) = (self.into_bounds().ok(), rhs.into_bounds().ok());
        if left.is_none() && right.is_none() {
            return Interval::Empty;
        }

        let (lhs_start, lhs_end) =
            left.unwrap_or_else(|| (Endpoint::Included(T::zero()), Endpoint::Included(T::zero())));

        let (rhs_start, rhs_end) =
            right.unwrap_or_else(|| (Endpoint::Included(U::zero()), Endpoint::Included(U::zero())));

        Interval::from_bounds((lhs_start + rhs_start, lhs_end + rhs_end))
    }
}

impl<T: Sub<Output = T> + Clone> Sub<T> for Interval<T> {
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        self.map(|x| x - rhs.clone())
    }
}

impl<T, U, Z> Sub<Interval<U>> for Interval<T>
where
    T: Zero + Sub<U, Output = Z>,
    U: Zero,
    Self: Bounded<T>,
    Interval<U>: Bounded<U>,
    Interval<Z>: Bounded<Z>,
{
    type Output = Interval<Z>;

    fn sub(self, rhs: Interval<U>) -> Self::Output {
        let (left, right) = (self.into_bounds().ok(), rhs.into_bounds().ok());
        if left.is_none() && right.is_none() {
            return Interval::Empty;
        }

        let (lhs_start, lhs_end) =
            left.unwrap_or_else(|| (Endpoint::Included(T::zero()), Endpoint::Included(T::zero())));

        let (rhs_start, rhs_end) =
            right.unwrap_or_else(|| (Endpoint::Included(U::zero()), Endpoint::Included(U::zero())));

        Interval::from_bounds((lhs_start - rhs_end, lhs_end - rhs_start))
    }
}

// symmetric relative to the _origin_ point (`x::T: x == -x`)
impl<T: Neg<Output = T>> Neg for Interval<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.reverse().map(Neg::neg)
    }
}

impl<T> Interval<T> {
    fn adjust_multiplied<N>(self, factor: &N, was_empty: bool) -> Self
    where
        N: Zero,
        T: PartialOrd + Zero,
    {
        match factor.cmp_zero() {
            Some(Ordering::Equal) => {
                if was_empty {
                    if self.is_empty() {
                        self
                    } else {
                        // sometimes a degenerate interval can become a valid one,
                        // e.g. `[2, 1] * 0 -> [0, 0]`, so we have to force it to be empty again
                        Self::Empty
                    }
                } else {
                    // non-empty interval times `0` should map to the singleton `[0]` interval
                    Self::Closed((T::zero(), T::zero()))
                }
            }
            // if the interval changed its "direction",
            // e.g. `[3, 5] * -1 = [-3, -5]`, we have to reverse it (without changing the signs)
            Some(Ordering::Less) => self.reverse(),
            None | Some(Ordering::Greater) => self,
        }
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
        self.map(|x| x * rhs.clone())
            .adjust_multiplied(&rhs, was_empty)
    }
}

impl<T, N, Z> Div<N> for Interval<T>
where
    N: Clone + Zero,
    T: PartialOrd + Div<N, Output = Z>,
    Z: PartialOrd + Zero,
{
    type Output = Interval<Z>;

    fn div(self, rhs: N) -> Self::Output {
        let was_empty = self.is_empty();
        self.map(|x| x / rhs.clone())
            .adjust_multiplied(&rhs, was_empty)
    }
}

#[derive(Debug, PartialEq, Eq)]
/// Attach a tag to data to signify its priority.
enum Prioritized<T> {
    Low(T),
    Normal(T),
}

impl<const SIDE: bool, T: PartialOrd> PartialOrd for Prioritized<Endpoint<SIDE, T>> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let to_inf_ordering = inf_ordering(SIDE);
        let to_zero_ordering = to_inf_ordering.reverse();

        // `Low` variant always here to 'shrink' the interval, i.e.
        // - it is `Greater` than any `Normal` (except equivalent one) for `Endpoint<LEFT, T>`
        //   (preferrable in `max`, _low_ priority in `min`);
        // - it is `Less` than any `Normal` (except equivalent one) for `Endpoint<RIGHT, T>`
        //   (preferrable in `min`, _low_ priority in `max`).
        let (a, b) = map_pair((self, other), |p| p.as_ref().into_data().bound_val());
        match (self, other) {
            (Self::Low(_), Self::Normal(_)) if a.is_some() && b.is_some() && a != b => {
                Some(to_zero_ordering)
            }
            (Self::Normal(_), Self::Low(_)) if a.is_some() && b.is_some() && a != b => {
                Some(to_inf_ordering)
            }
            other => {
                let (a, b) = map_pair(other, |x| x.as_ref().into_data());
                a.partial_cmp(b)
            }
        }
    }
}

impl<const SIDE: bool, T: Ord> Ord for Prioritized<Endpoint<SIDE, T>> {
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

type PrioritizedBounds<T> = (Prioritized<LBound<T>>, Prioritized<RBound<T>>);

impl<T> Interval<T> {
    /// Auxiliary function to help with multiplying two [`Intervals`].
    fn mul_bound<const SIDE: bool, N, Z, E>(
        self,
        rhs: Endpoint<SIDE, N>,
    ) -> Result<PrioritizedBounds<Z>, E>
    where
        Self: Mul<N, Output = Interval<Z>>,
        T: Zero + PartialOrd,
        Z: Zero,
        Interval<Z>: Bounded<Z, Error = E>,
    {
        let zero_point = T::zero();
        let zero_result = || Z::zero();

        let (a, b) = match rhs {
            Endpoint::Included(n) => (self * n).into_bounds()?,
            Endpoint::Excluded(n) => {
                let preserve_zero = match self.as_ref().into_bounds() {
                    Ok((ref_a, ref_b)) => [ref_a.into_bound(), ref_b.into_bound()]
                        .contains(&Bound::Included(&zero_point)),
                    Err(_) => false,
                };

                let product = (self * n)
                    .into_bounds()
                    .map(|(a, b)| (a.into_exclusive_bound(), b.into_exclusive_bound()))?;

                let (a, b) = map_pair(product, |bound| {
                    if matches!(bound, Bound::Excluded(ref t) if t.cmp_zero().map_or(false, Ordering::is_eq))
                        && preserve_zero
                    {
                        Bound::Included(zero_result())
                    } else {
                        bound
                    }
                });
                (a.into(), b.into())
            }
            Endpoint::Infinite => {
                let zero_bound = || {
                    if self.contains(&zero_point) {
                        Bound::Included(zero_result())
                    } else {
                        Bound::Excluded(zero_result())
                    }
                };

                match (self.point_cmp(&zero_point), SIDE) {
                    // the interval is entirely non-negative multiplied by +inf (`[+a, +b] * +inf = [0, +inf)`)
                    // the interval is entirely non-positive multiplied by -inf (`[-a, -b] * -inf = [0, +inf)`)
                    (Some(Ordering::Greater), RIGHT) | (Some(Ordering::Less), LEFT) => {
                        return Ok((
                            Prioritized::Low(zero_bound().into()),
                            Prioritized::Normal(Endpoint::Infinite),
                        ));
                    }
                    // the interval is entirely non-positive multiplied by +inf (`[-a, -b] * +inf = (-inf, 0]`)
                    // the interval is entirely non-negative multiplied by -inf (`[+a, +b] * -inf = (-inf, 0]`)
                    (Some(Ordering::Less), RIGHT) | (Some(Ordering::Greater), LEFT) => {
                        return Ok((
                            Prioritized::Normal(Endpoint::Infinite),
                            Prioritized::Low(zero_bound().into()),
                        ));
                    }
                    // [0] * +/- inf = [-inf, +inf] * 0 = [0, 0]
                    (Some(Ordering::Equal), _) => (zero_bound().into(), zero_bound().into()),
                    // the interval spans both negative and positive values
                    (None, _) => (Endpoint::Infinite, Endpoint::Infinite),
                }
            }
        };
        Ok((Prioritized::Normal(a), Prioritized::Normal(b)))
    }
}

impl<T, N, Z> Mul<Interval<N>> for Interval<T>
where
    N: Clone + PartialOrd + Zero,
    T: Clone + PartialOrd + Mul<N, Output = Z> + Zero,
    Z: Ord + Zero,
    Interval<Z>: Bounded<Z>,
{
    type Output = Interval<Z>;

    fn mul(self, rhs: Interval<N>) -> Self::Output {
        if self.is_empty() {
            return Interval::Empty;
        }

        let Ok((rhs_start, rhs_end)) = rhs.into_bounds() else {
            return Interval::Empty;
        };

        let Ok((left_start, left_end)) = self.clone().mul_bound(rhs_start) else {
            panic!(
                "multiplication of non-empty interval to an Endpoint produced an empty interval"
            );
        };
        let Ok((right_start, right_end)) = self.mul_bound(rhs_end) else {
            panic!(
                "multiplication of non-empty interval to an Endpoint produced an empty interval"
            );
        };

        let start = left_start.min(right_start).into_data();
        let end = left_end.max(right_end).into_data();

        let interval = Interval::from_bounds((start, end));
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
        let Ok((start, end)) = self.into_bounds() else {
            return Self::Full.into();
        };

        let not_start = Self::from_bounds((!start).augment_with_inf());
        let not_end = Self::from_bounds((!end).augment_with_inf());

        match (not_start, not_end) {
            (Self::Full, Self::Full) => OneOrPair::One(Self::Empty),
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

impl<T, U> BitAnd<U> for Interval<T>
where
    Self: Bounded<T>,
    T: Ord,
    U: Bounded<T>,
{
    type Output = Self;

    fn bitand(self, rhs: U) -> Self::Output {
        self.intersect(rhs).unwrap_or_else(|_| Self::Empty)
    }
}

impl<T, U> BitOr<U> for Interval<T>
where
    Self: Bounded<T>,
    T: Ord,
    U: Bounded<T>,
{
    type Output = OneOrPair<Self>;

    fn bitor(self, rhs: U) -> Self::Output {
        self.union(rhs).unwrap_or_else(|(left, right)| {
            OneOrPair::One(right.into_bounds().map_or(left, Self::from_bounds))
        })
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
mod add_tests {
    use crate::interval;

    #[test]
    fn adding_empty_is_preserving() {
        let i = interval!(> 2);
        assert_eq!(i + interval!((6, 5)), i);
        assert_eq!(interval!((5, 5)) + i, i);
    }

    #[test]
    fn bounded_with_unbounded_sum() {
        let a = interval!(>= 2);
        let b = interval!([3, 10]);
        assert_eq!(a + b, interval!(>= 5));

        assert_eq!(a.into_interior() + b, interval!(> 5));
        assert_eq!(b.into_interior() + a, interval!(> 5));
    }

    #[test]
    fn two_unbounded_sum() {
        let a = interval!(>= 2);
        let b = interval!(> 8);
        assert_eq!(a + b, interval!(> 10));

        let c = interval!(< 6);
        assert_eq!(a + c, interval!(..));
    }

    #[test]
    fn two_bounded_sum() {
        let a = interval!([-5, 12]);
        let b = interval!((120, =287));
        assert_eq!(a + b, interval!((115, =299)));
    }

    #[test]
    fn sub_empty_is_preserving() {
        let i = interval!(> 2);
        assert_eq!(i - interval!((6, 5)), i);
        assert_eq!(interval!((5, 5)) - i, -i);
    }

    #[test]
    fn bounded_with_unbounded_diff() {
        let a = interval!(>= 2);
        let b = interval!([3, 10]);
        assert_eq!(a - b, interval!(>= -8));

        assert_eq!(a.into_interior() - b, interval!(> -8));
        assert_eq!(b.into_interior() - a, interval!(< 8));
    }

    #[test]
    fn two_unbounded_diff() {
        let a = interval!(>= 2);
        let b = interval!(> 8);
        assert_eq!(a - b, interval!(..));

        let c = interval!(< 6);
        assert_eq!(a - c, interval!(> -4));
    }

    #[test]
    fn two_bounded_diff() {
        let a = interval!([-5, 12]);
        let b = interval!((120, =287));
        assert_eq!(a - b, interval!((=-292, -108)));
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
        bounds: Result<PrioritizedBounds<T>, crate::bounds::EmptyIntervalError<T>>,
        result_interval: Interval<T>,
    ) where
        Interval<T>: Bounded<T>,
        <Interval<T> as Bounded<T>>::Error: core::fmt::Debug,
    {
        let (a, b) = bounds.unwrap();
        let pair = (a.into_data(), b.into_data());
        assert_eq!(pair, result_interval.into_bounds().unwrap());
    }

    #[test]
    fn mul_to_infinite_bound() {
        let (neg_inf, pos_inf) = interval!(..: i32).into_bounds().unwrap();

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
        let upper_bound = RBound::<_>::from(Bound::Excluded(5));
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

        let (rhs_start, rhs_end) = r1.into_bounds().unwrap();
        cmp_mul_bound(r2.mul_bound(rhs_start), interval!(< 0));
        assert_eq!(
            r2.mul_bound(rhs_end)
                .map(|(a, b)| (a.into_data(), b.into_data()))
                .unwrap(),
            (Endpoint::Excluded(0), Endpoint::Excluded(0)),
        );
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

        let bounds1 = interval!((a, =b)).into_bounds().unwrap();
        let bounds2 = interval!((=c, d)).into_bounds().unwrap();
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
