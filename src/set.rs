use crate::{
    bounds::{BothBounds, Bounded, IntoBounds},
    helper::{minmax, OneOrPair, Pair},
};

/// Provides a bunch of set operations for [`Bounded`] types.
pub trait SetOps<T>: Bounded<T> {
    // TODO: difference, symmetric_difference (+core::ops::Xor)

    /// Compute the intersection of `self` and `other`.
    ///
    /// # Notes
    /// If the `self` is degenerate, just return it as is.
    ///
    /// # Errors
    /// Return `other` if it is degenerate (for the normal `self`).
    fn intersect<R>(self, other: R) -> Result<Self, R>
    where
        T: Ord,
        Self: From<Self::Error>,
        R: IntoBounds<T> + From<R::Error>,
    {
        let ((a, b), (c, d)) = match pair_into_bounds(self, other) {
            Ok(bounds) => bounds,
            Err((left, right)) => {
                let left_degenerate = left.into_bounds().err();
                // if _left_ is degenerate, return it as is
                return left_degenerate.map(Self::from).ok_or({
                    // otherwise, _left_ is valid and _right_ is degenerate,
                    // but we cannot construct a value of `Self`
                    // from a degenerate _right_, so just return it as `Err`
                    right
                });
            }
        };

        let start = a.max(c);
        let end = b.min(d);

        Ok(Self::from_bounds((start, end)))
    }

    /// The smallest span containing both `self` and `other`
    /// if the values [intersects][Self::intersect] (wrapped in [`OneOrPair::One`]).
    ///
    /// Otherwise (when the intervals are disjoint),
    /// return a [pair][OneOrPair::Pair] of pairs of ordered ranges
    ///
    /// # Notes
    /// If at least one of [`IntoBounds::into_bounds`] fails (the value is degenerate),
    /// return the other value wrapped in [`OneOrPair::One`].
    fn union<R>(self, other: R) -> OneOrPair<Self>
    where
        T: Ord,
        Self: From<Self::Error>,
        R: IntoBounds<T> + From<R::Error>,
    {
        let ((a, b), (c, d)) = match pair_into_bounds(self, other) {
            Ok(bounds) => bounds,
            Err((left, right)) => {
                // - if _right_ is degenerate, return _left_ as is (either it is degenerate or not)
                // - if _right_ is not degenerate (then _left_ should be degenerate, since the pair failed),
                //   then transform the _right_ into `Self` and return it.
                return OneOrPair::One(right.into_bounds().map_or(left, Self::from_bounds));
            }
        };

        let [min_start, max_start] = minmax(a, c);
        let [min_end, max_end] = minmax(b, d);

        let are_disjoint = {
            let (l, r) = (max_start.as_ref(), min_end.as_ref());
            let empty_intersection = l > r;

            // `[x, x)` or `(x, x]` is an empty gap for empty intersection => intervals are joint
            // `(x, x)` is an empty gap for empty intersection `(x, x)` => intervals are disjoint
            let empty_gap = !r > !l;
            empty_intersection && !empty_gap
        };

        if are_disjoint {
            OneOrPair::Pair((
                Self::from_bounds((min_start, min_end)),
                Self::from_bounds((max_start, max_end)),
            ))
        } else {
            OneOrPair::One(Self::from_bounds((min_start, max_end)))
        }
    }

    #[must_use = "Computing the enclosure returns a new value from the inputs"]
    /// The smallest span containing both `self` and `other`,
    /// also covering possible 'gap' between them.
    ///
    /// # Notes
    /// If at least one of [`IntoBounds::into_bounds`] fails (the value is degenerate),
    /// return the other value wrapped in [`OneOrPair::One`].
    fn enclosure<R>(self, other: R) -> Self
    where
        T: Ord,
        Self: From<Self::Error>,
        R: IntoBounds<T> + From<R::Error>,
    {
        let ((a, b), (c, d)) = match pair_into_bounds(self, other) {
            Ok(bounds) => bounds,
            Err((left, right)) => {
                // - if _right_ is degenerate, return _left_ as is (either it is degenerate or not)
                // - if _right_ is not degenerate (then _left_ should be degenerate, since the pair failed),
                //   then transform the _right_ into `Self` and return it.
                return right.into_bounds().map_or(left, Self::from_bounds);
            }
        };

        let start = a.min(c);
        let end = b.max(d);
        Self::from_bounds((start, end))
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
        // if at least one of the intervals is degenerate, they cannot be disjoint

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
        assert!(matches!(a.intersect(b).unwrap_err(), interval!(0)));
        assert!(matches!(b.intersect(a).unwrap(), interval!(0)));
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
        assert_eq!(a.intersect(b).unwrap_err(), b);
        assert_eq!(b.intersect(a).unwrap(), b);
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
        assert_eq!(a.enclosure(b), interval!((=3, 10)));

        let a = interval!(<5);
        let b = interval!(>3);
        assert_eq!(a.enclosure(b), Interval::Full);

        let a = interval!(<=-100);
        let b = interval!(>=100);
        assert_eq!(a.enclosure(b), Interval::Full);

        let a = interval!([1, 2]);
        let b = interval!([3, 4]);
        assert_eq!(a.enclosure(b), interval!([1, 4]));
    }

    #[test]
    fn union_touching() {
        let a = interval!([1, 2]);
        let b = interval!([2, 4]);
        assert_eq!(a.union(b).into_single().unwrap(), interval!([1, 4]));

        let a = interval!((1, 2));
        let b = interval!([2, 4]);
        assert_eq!((a | b).into_single().unwrap(), interval!((1, =4)));

        let a = interval!([1, 2]);
        let b = interval!((2, 4));
        assert_eq!(a.union(b).into_single().unwrap(), interval!((=1, 4)));

        let a = interval!((1, 2));
        let b = interval!((2, 4));
        assert_eq!(
            (a | b).into_pair().unwrap(),
            (interval!((1, 2)), interval!((2, 4)))
        );
    }
}
