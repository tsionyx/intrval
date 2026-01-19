//! Set operations for bounded types
//! independent of the specific interval implementation.
use crate::{
    bounds::{BothBounds, Bounded, IntoBounds},
    helper::{minmax, OneOrPair, Pair},
};

/// The trait for types that can contain the items of type `T`.
pub trait Container<T> {
    /// Whether the `self` contains a given point comparable to `T`.
    fn contains<U>(self, point: U) -> bool
    where
        T: PartialOrd + PartialOrd<U>,
        U: PartialOrd<T>;
}

impl<Z, T> Container<T> for Z
where
    Self: IntoBounds<T>,
{
    fn contains<U>(self, point: U) -> bool
    where
        T: PartialOrd + PartialOrd<U>,
        U: PartialOrd<T>,
    {
        use crate::bounds::Endpoint::{Excluded, Included, Infinite};

        let Ok((a, b)) = self.into_bounds() else {
            // an empty interval does not contain any point
            return false;
        };

        (match a {
            Included(start) => start <= point,
            Excluded(start) => start < point,
            Infinite => true,
        }) && (match b {
            Included(end) => point <= end,
            Excluded(end) => point < end,
            Infinite => true,
        })
    }
}

/// Provides a bunch of set operations for [`Bounded`] types.
pub trait SetOps<T>: Bounded<T> {
    /// Get the set difference between `self` and `other`
    /// i.e. the span(s) of values that are in `self` but **not** in `other`.
    ///
    /// # Errors
    /// Return `Err((self, other))` if `self` is a (not degenerate) subset of `other`.
    /// In this case it is safe to assume the difference is _degenerate_.
    fn difference<R>(self, other: R) -> Result<OneOrPair<Self>, (Self, R)>
    where
        T: Ord,
        Self: From<Self::Error>,
        R: IntoBounds<T> + From<<R as IntoBounds<T>>::Error>,

        for<'a> &'a Self: IntoBounds<&'a T>,
        for<'a> &'a R: IntoBounds<&'a T>,
    {
        if self.is_sub(&other) {
            return if (&self).into_bounds().is_err() {
                // if `self` is degenerate, just return it as is
                Ok(self.into())
            } else {
                Err((self, other))
            };
        }

        if self.is_disjoint(&other) {
            return Ok(self.into());
        }

        let ((a, b), (c, d)) = match pair_into_bounds(self, other) {
            Ok(bounds) => bounds,
            Err((left, _right)) => {
                // if either _left_ or _right_ is degenerate, return _left_ as is:
                // - if _left_ is degenerate, then we have nothing to subtract from;
                // - if _right_ is degenerate, then there is nothing to subtract, just return _left_ as is.
                return Ok(left.into());
            }
        };

        let c_finite = c.is_finite();
        let d_finite = d.is_finite();

        let (left, right) = (Self::from_bounds((a, !c)), Self::from_bounds((!d, b)));

        match (c_finite, d_finite) {
            (true, true) => Ok(left.union(right)),
            (true, false) => {
                // the `right` is `(-inf, b)` is a super of the `self` and useless for diff
                Ok(left.into())
            }
            (false, true) => {
                // the `left` is `(a, +inf)` is a super of the `self` and useless for diff
                Ok(right.into())
            }
            (false, false) => {
                unreachable!(
                    "This variant is only possible when the `right` is universal, which is handled above"
                )
            }
        }
    }

    /// Get the symmetric difference between `self` and `other`
    /// i.e. the span(s) of values that are
    /// in one of the intervals but **not** in other.
    ///
    /// # Notes
    /// When the inputs do not overlap, the result is equivalent to their [union][Self::union].
    ///
    /// # Errors
    /// Return `other` if the inputs are equal and **not degenerate**.
    /// In this case it is safe to assume the symmetric difference is _degenerate_.
    ///
    /// Is is tempting here to make the return type `Result<_, Self>`,
    /// because it is very easy for the `Err` branch
    /// to convert the `R` into `Self`, but this can lead
    /// to accident use of `.unwrap_or_else(OneOrPair::One)` which is not correct:
    /// the `Err` data is here only for diagnostic use and should be fallen back
    /// with manually constructed _degenerate_ interval.
    fn symmetric_difference<R>(self, other: R) -> Result<OneOrPair<Self>, R>
    where
        T: Ord,
        Self: From<Self::Error>,
        R: IntoBounds<T> + From<<R as IntoBounds<T>>::Error>,

        for<'a> &'a Self: IntoBounds<&'a T>,
        for<'a> &'a R: IntoBounds<&'a T>,
    {
        if self.is_disjoint(&other) {
            return Ok(self.union(other));
        }

        let has_equal_bounds = match ((&self).into_bounds(), (&other).into_bounds()) {
            (Ok(left), Ok(right)) => Some(left == right),
            (Ok(_), Err(_)) | (Err(_), Ok(_)) => Some(false),
            (Err(_), Err(_)) => None,
        };

        match has_equal_bounds {
            Some(true) => {
                // the intervals are the same => symmetric difference is empty (degenerate)
                return Err(other);
            }
            Some(false) => {}
            None => {
                // both are degenerate, just return `self` as is
                return Ok(self.into());
            }
        }

        let ((a, b), (c, d)) = match pair_into_bounds(self, other) {
            Ok(bounds) => bounds,
            Err((left, right)) => {
                // if _right_ is degenerate, just return `left` as is,
                // otherwise the `left` should be degenerate, so transform `right` into `Self` and return it.
                let res = right.into_bounds().map_or(left, Self::from_bounds);
                return Ok(res.into());
            }
        };

        let [min_start, max_start] = minmax(a, c);
        let [min_end, max_end] = minmax(b, d);

        let c_finite = max_start.is_finite();
        let d_finite = min_end.is_finite();

        let (left, right) = (
            Self::from_bounds((min_start, !max_start)),
            Self::from_bounds((!min_end, max_end)),
        );

        match (c_finite, d_finite) {
            (true, true) => Ok(left.union(right)),
            (true, false) => {
                // the `right` is `(-inf, +inf)` and useless for `union`
                Ok(left.into())
            }
            (false, true) => {
                // the `left` is `(-inf, +inf)` and useless for `union`
                Ok(right.into())
            }
            (false, false) => {
                unreachable!(
                    "This variant is only possible when both intervals are universal, which is handled above"
                )
            }
        }
    }

    /// Compute the intersection of `self` and `other`.
    ///
    /// # Notes
    /// If the `self` is degenerate, just return it as is.
    ///
    /// # Errors
    /// Return `other` if it is degenerate (for the normal `self`).
    /// In this case it is safe to assume the intersection is _degenerate_.
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
    fn diff() {
        let a = interval!((0, 1));
        let b = interval!([-2, 0]);
        assert_eq!(a.difference(b).unwrap().into_single().unwrap(), a);
        assert_eq!(b.difference(a).unwrap().into_single().unwrap(), b);

        let a = interval!([3, 7]);
        let b = interval!((5, 10));
        assert_eq!(
            a.difference(b).unwrap().into_single().unwrap(),
            interval!([3, 5])
        );
        assert_eq!(
            b.difference(a).unwrap().into_single().unwrap(),
            interval!((7, 10))
        );

        let a = interval!(<5);
        let b = interval!(>3);
        assert_eq!(
            a.difference(b).unwrap().into_single().unwrap(),
            interval!(<=3)
        );
        assert_eq!(
            b.difference(a).unwrap().into_single().unwrap(),
            interval!(>=5)
        );

        let a = interval!(<=5);
        let b = interval!(>=-3);
        assert_eq!(
            a.difference(b).unwrap().into_single().unwrap(),
            interval!(< -3)
        );
        assert_eq!(
            b.difference(a).unwrap().into_single().unwrap(),
            interval!(>5)
        );

        let a = interval!(U: i32);
        let b = interval!(0: i32);
        assert_eq!(
            a.difference(b).unwrap().into_single().unwrap(),
            interval!(U),
        );
        assert_eq!(
            b.difference(a).unwrap().into_single().unwrap(),
            interval!(0)
        );
    }

    #[test]
    fn symm_diff() {
        let a = interval!((0, 1));
        let b = interval!((=-2, 0));
        assert_eq!(
            a.symmetric_difference(b).unwrap().into_pair().unwrap(),
            (b, a)
        );
        assert_eq!(
            b.symmetric_difference(a).unwrap().into_pair().unwrap(),
            (b, a)
        );

        let a = interval!((0, 1));
        let b = interval!([-2, 0]);
        // joined in `0` into a single one
        assert_eq!(
            a.symmetric_difference(b).unwrap().into_single().unwrap(),
            interval!((=-2, 1))
        );
        assert_eq!(
            a.symmetric_difference(b).unwrap().into_single().unwrap(),
            interval!((=-2, 1))
        );

        let a = interval!([3, 7]);
        let b = interval!((5, 10));
        assert_eq!(
            a.symmetric_difference(b).unwrap().into_pair().unwrap(),
            (interval!([3, 5]), interval!((7, 10)))
        );
        assert_eq!(
            b.symmetric_difference(a).unwrap().into_pair().unwrap(),
            (interval!([3, 5]), interval!((7, 10)))
        );

        let a = interval!(<5);
        let b = interval!(>3);
        assert_eq!(
            a.symmetric_difference(b).unwrap().into_pair().unwrap(),
            (interval!(<=3), interval!(>=5))
        );
        assert_eq!(
            b.symmetric_difference(a).unwrap().into_pair().unwrap(),
            (interval!(<=3), interval!(>=5))
        );

        let a = interval!(<=5);
        let b = interval!(>=-3);
        assert_eq!(
            a.symmetric_difference(b).unwrap().into_pair().unwrap(),
            (interval!(< -3), interval!(>5))
        );
        assert_eq!(
            b.symmetric_difference(a).unwrap().into_pair().unwrap(),
            (interval!(< -3), interval!(>5))
        );

        let a = interval!([1, 9]);
        let b = interval!(0);
        assert_eq!(a.symmetric_difference(b).unwrap().into_single().unwrap(), a);
        assert_eq!(b.symmetric_difference(a).unwrap().into_single().unwrap(), a);

        let a = interval!(U: i32);
        let b = interval!(0: i32);
        assert_eq!(
            a.symmetric_difference(b).unwrap().into_single().unwrap(),
            interval!(U)
        );
        assert_eq!(
            b.symmetric_difference(a).unwrap().into_single().unwrap(),
            interval!(U)
        );
    }

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
