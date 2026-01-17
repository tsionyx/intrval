use crate::{
    bounds::{BothBounds, Bounded, IntoBounds},
    helper::{minmax, OneOrPair, Pair},
};

/// Provides a bunch of set operations for [`Bounded`] types.
pub trait SetOps<T>: Bounded<T> {
    // TODO: difference, symmetric_difference (+core::ops::Xor)

    /// Compute the intersection of `self` and `other`.
    ///
    /// # Errors
    /// Return a pair of original values if at least one of [`IntoBounds::into_bounds`] fails.
    fn intersect<R>(self, other: R) -> Result<Self, (Self, R)>
    where
        T: Ord,
        Self: From<Self::Error>,
        R: IntoBounds<T> + From<R::Error>,
    {
        let ((self_start, self_end), (other_start, other_end)) = pair_into_bounds(self, other)?;

        let start = self_start.max(other_start);
        let end = self_end.min(other_end);

        Ok(Self::from_bounds((start, end)))
    }

    /// The smallest span containing both `self` and `other`
    /// if the values [intersects][Self::intersect] (wrapped in [`OneOrPair::One`]).
    ///
    /// Otherwise (when the intervals are disjoint),
    /// return a [pair][OneOrPair::Pair] of pairs of ordered ranges
    ///
    /// # Errors
    /// Return a pair of original values if at least one of [`IntoBounds::into_bounds`] fails.
    fn union<R>(self, other: R) -> Result<OneOrPair<Self>, (Self, R)>
    where
        T: Ord,
        Self: From<Self::Error>,
        R: IntoBounds<T> + From<R::Error>,
    {
        let ((self_start, self_end), (other_start, other_end)) = pair_into_bounds(self, other)?;

        let [min_start, max_start] = minmax(self_start, other_start);
        let [min_end, max_end] = minmax(self_end, other_end);

        let are_disjoint = {
            let (l, r) = (max_start.as_ref(), min_end.as_ref());
            let empty_intersection = l > r;

            // `[x, x)` or `(x, x]` is an empty gap for empty intersection => intervals are joint
            // `(x, x)` is an empty gap for empty intersection `(x, x)` => intervals are disjoint
            let empty_gap = !r > !l;
            empty_intersection && !empty_gap
        };

        let one_or_pair = if are_disjoint {
            OneOrPair::Pair((
                Self::from_bounds((min_start, min_end)),
                Self::from_bounds((max_start, max_end)),
            ))
        } else {
            OneOrPair::One(Self::from_bounds((min_start, max_end)))
        };
        Ok(one_or_pair)
    }

    /// The smallest span containing both `self` and `other`.
    ///
    /// # Errors
    /// Return a pair of original values if at least one of [`IntoBounds::into_bounds`] fails.
    fn enclosure<R>(self, other: R) -> Result<Self, (Self, R)>
    where
        T: Ord,
        Self: From<Self::Error>,
        R: IntoBounds<T> + From<R::Error>,
    {
        let ((self_start, self_end), (other_start, other_end)) = pair_into_bounds(self, other)?;

        let start = self_start.min(other_start);
        let end = self_end.max(other_end);
        Ok(Self::from_bounds((start, end)))
    }

    // TODO: contains

    /// Returns `true` if there is a non-empty gap between `self` and `other`.
    /// This implies the `self.union(other)` guaranteed to be a [whole span][OneOrPair::One]
    /// without [jumps](https://en.wikipedia.org/wiki/Classification_of_discontinuities)
    ///
    /// This is **not equivalent** to checking for an empty intersection.
    /// because two intervals can 'touch' at a single point, where
    /// one of them includes the point and another exclusively approaches it (open interval).
    /// This case is considered as _not disjoint_ (i.e. _joint_),
    /// because there is no gap between the intervals, even though their intersection is empty.
    fn is_disjoint<'a, R>(&'a self, other: R) -> bool
    where
        T: Ord + 'a,
        &'a Self: IntoBounds<&'a T>,
        R: IntoBounds<&'a T>,
    {
        // if at least one of the intervals is empty, they cannot be disjoint

        let Ok((self_start, self_end)) = self.into_bounds() else {
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
    fn is_sub<'a, R>(&'a self, other: R) -> bool
    where
        T: PartialOrd + 'a,
        &'a Self: IntoBounds<&'a T>,
        R: IntoBounds<&'a T>,
    {
        let Ok((self_start, self_end)) = self.into_bounds() else {
            // the degenerate interval (with no valid bounds) is contained in any interval
            return true;
        };

        let Ok((other_start, other_end)) = other.into_bounds() else {
            // no interval can be inside a degenerate interval (with no valid bounds)
            // (except the empty one, which is handled above)
            return false;
        };

        other_start <= self_start && other_end >= self_end
    }

    /// Returns `true` if the interval completely contains another,
    /// i.e., `self` contains at least all the values in `other`.
    fn is_super<'a, R>(&'a self, other: R) -> bool
    where
        T: PartialOrd + 'a,
        &'a Self: IntoBounds<&'a T>,
        R: IntoBounds<&'a T>,
    {
        let Ok((other_start, other_end)) = other.into_bounds() else {
            // the degenerate interval (with no valid bounds) is contained in any interval
            return true;
        };

        let Ok((self_start, self_end)) = self.into_bounds() else {
            // no interval can be inside a degenerate interval (with no valid bounds)
            // (except the empty one, which is handled above)
            return false;
        };

        self_start <= other_start && self_end >= other_end
    }
}

impl<T, X> SetOps<T> for X where X: Bounded<T> + From<Self::Error> {}

/// Convert the two [`Bounded`] values into a pair of `BothBounds`,
/// returning the original pair of values
/// if at least one [conversion][IntoBounds::into_bounds] fails.
fn pair_into_bounds<B1, B2, T>(a: B1, b: B2) -> Result<Pair<BothBounds<T>>, (B1, B2)>
where
    B1: Bounded<T> + From<B1::Error>,
    B2: IntoBounds<T> + From<B2::Error>,
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

#[cfg(test)]
mod tests {
    use crate::{interval, Interval};

    use super::*;

    #[test]
    fn intersect() {
        let a = interval!([3, 7]);
        let b = interval!((5, 10));
        assert_eq!(a.intersect(b).unwrap(), interval!((5, =7)));

        let a = interval!(<5);
        let b = interval!(>3);
        assert_eq!(a & b, interval!((3, 5)));

        let a = interval!(<=5);
        let b = interval!(>=-3);
        assert_eq!(a & b, interval!([-3, 5]));

        let a = interval!(U: i32);
        let b = interval!(0: i32);
        assert!(matches!(
            a.intersect(b).unwrap_err(),
            (interval!(U), interval!(0))
        ));
    }

    #[test]
    fn intersect_empty() {
        let a = interval!([1, 2]);
        let b = interval!([3, 4]);
        assert_eq!(a.intersect(b).unwrap(), interval!([3, 2]));
        assert!((a & b).is_empty());

        let a = interval!(>6);
        let b = interval!(<3);
        assert_eq!(a.intersect(b).unwrap(), interval!((6, 3)));
        assert!((a & b).is_empty());

        let a = interval!(>=6);
        let b = interval!(<6);
        assert_eq!(a.intersect(b).unwrap(), interval!((=6, 6)));
        assert!((a & b).is_empty());

        let a = interval!((2, =4));
        let b = interval!((=3, 1));
        assert_eq!(a.intersect(b).unwrap_err(), (a, b));
    }

    #[test]
    fn intersect_single() {
        let a = interval!(>=6);
        let b = interval!(<=6);
        assert_eq!(a.intersect(b).unwrap(), interval!([6, 6]));
        assert_eq!((a & b), interval!(=6));

        let a = interval!((2, =3));
        let b = interval!((=3, 8));
        assert_eq!(a.intersect(b).unwrap(), interval!([3, 3]));
        assert_eq!((a & b), interval!(==3));
    }

    #[test]
    fn enclosure() {
        let a = interval!([3, 7]);
        let b = interval!((5, 10));
        assert_eq!(a.enclosure(b).unwrap(), interval!((=3, 10)));

        let a = interval!(<5);
        let b = interval!(>3);
        assert_eq!(a.enclosure(b).unwrap(), Interval::Full);

        let a = interval!(<=-100);
        let b = interval!(>=100);
        assert_eq!(a.enclosure(b).unwrap(), Interval::Full);

        let a = interval!([1, 2]);
        let b = interval!([3, 4]);
        assert_eq!(a.enclosure(b).unwrap(), interval!([1, 4]));
    }

    #[test]
    fn union_touching() {
        let a = interval!([1, 2]);
        let b = interval!([2, 4]);
        assert_eq!(
            a.union(b).unwrap().into_single().unwrap(),
            interval!([1, 4])
        );

        let a = interval!((1, 2));
        let b = interval!([2, 4]);
        assert_eq!((a | b).into_single().unwrap(), interval!((1, =4)));

        let a = interval!([1, 2]);
        let b = interval!((2, 4));
        assert_eq!(
            a.union(b).unwrap().into_single().unwrap(),
            interval!((=1, 4))
        );

        let a = interval!((1, 2));
        let b = interval!((2, 4));
        assert_eq!(
            (a | b).into_pair().unwrap(),
            (interval!((1, 2)), interval!((2, 4)))
        );
    }
}
