use core::{
    cmp::Ordering,
    ops::{Div, Mul, Shl, Shr},
};

use crate::{helper::Zero, interval::Interval};

use super::LinearSpace;

// Implement shift operations for LinearSpace.
//
// Note that the shift operations only shift the bounds of the linear space, but do not change the step size.
//
// The implementation of `{Add, Sub}` are intentionally skipped because they have
// richer semantics and could be assumed to also change the step size.

impl<T> Shl<T> for LinearSpace<T>
where
    Interval<T>: Shl<T, Output = Interval<T>>,
{
    type Output = Self;

    /// Shift the linear space's bounds to the _left_ **without changing the step size**.
    fn shl(self, rhs: T) -> Self::Output {
        let Self { bounds, step } = self;

        Self {
            bounds: bounds << rhs,
            step,
        }
    }
}

impl<T> Shr<T> for LinearSpace<T>
where
    Interval<T>: Shr<T, Output = Interval<T>>,
{
    type Output = Self;

    /// Shift the linear space's bounds to the _right_ **without changing the step size**.
    fn shr(self, rhs: T) -> Self::Output {
        let Self { bounds, step } = self;

        Self {
            bounds: bounds >> rhs,
            step,
        }
    }
}

impl<T, U, Z> Mul<U> for LinearSpace<T>
where
    T: Mul<U, Output = Z>,
    U: Clone,
    Interval<T>: Mul<U, Output = Interval<Z>>,
    Z: Zero,
{
    type Output = Option<LinearSpace<Z>>;

    /// Scale the bounds and step size using some scalar value.
    ///
    /// # Returns
    /// `None`, if the multiplier is a negative number,
    /// thus producing an invalid `step` as a product.
    fn mul(self, rhs: U) -> Self::Output {
        let Self { bounds, step } = self;

        let step = step * rhs.clone();
        let bounds = bounds * rhs;
        LinearSpace::try_bounded(bounds, step)
    }
}

impl<T, U, Z> Div<U> for LinearSpace<T>
where
    T: Div<U, Output = Z>,
    U: Clone + Zero,
    Interval<T>: Div<U, Output = Interval<Z>>,
    Z: Zero,
{
    type Output = Option<LinearSpace<Z>>;

    /// Scale the bounds and step size using some scalar value.
    ///
    /// # Returns
    /// `None`:
    /// - if the multiplier is a negative number,
    ///   thus producing an invalid `step` as a product;
    /// - if the `scalar` is zero, leading to division by zero.
    fn div(self, rhs: U) -> Self::Output {
        if rhs.cmp_zero() == Some(Ordering::Equal) {
            return None;
        }

        let Self { bounds, step } = self;

        let step = step / rhs.clone();
        let bounds = bounds / rhs;
        LinearSpace::try_bounded(bounds, step)
    }
}

impl<T, U, Z> Mul<LinearSpace<U>> for LinearSpace<T>
where
    T: Mul<U, Output = Z>,
    Interval<T>: Mul<Interval<U>, Output = Interval<Z>>,
{
    type Output = LinearSpace<Z>;

    /// Pairwise multiplies the bounds and step size of both spaces.
    fn mul(self, rhs: LinearSpace<U>) -> Self::Output {
        let Self { bounds, step } = self;
        let LinearSpace {
            bounds: rhs_bounds,
            step: rhs_step,
        } = rhs;

        let step = step * rhs_step;
        let bounds = bounds * rhs_bounds;
        // the existence of the two `LinearSpace` guarantees the validity (positiveness)
        // of their `step`-s, so we can directly construct a new `LinearSpace`
        // without checking the validity of the new (product) `step`.
        LinearSpace { bounds, step }
    }
}

#[cfg(test)]
mod tests {
    use crate::interval;

    use super::*;

    #[test]
    fn shift_left() {
        let space = LinearSpace::try_bounded(interval!(>10), 2_u8).unwrap();
        let shifted = space << 5;
        assert_eq!(shifted.bounds(), &interval!(> 5));
        assert_eq!(shifted.step(), &2);

        let space = LinearSpace::try_bounded(interval!((90, =2048)), 100).unwrap();
        let shifted = space << 42;
        assert_eq!(shifted.bounds(), &Interval::LeftOpen((48, 2006)));
        assert_eq!(shifted.step(), &100);
    }

    #[test]
    fn shift_right() {
        let space = LinearSpace::try_bounded(interval!(>= 18), 3).unwrap();
        let shifted = space >> 7;
        assert_eq!(shifted.bounds(), &interval!(>= 25));
        assert_eq!(shifted.step(), &3);

        let space = LinearSpace::try_bounded(interval!((90, =2048)), 100).unwrap();
        let shifted = space >> 112;
        assert_eq!(shifted.bounds(), &Interval::LeftOpen((202, 2160)));
        assert_eq!(shifted.step(), &100);
    }

    #[test]
    fn mult_scalar() {
        let space = LinearSpace::try_bounded(interval!((90, =2048)), 100).unwrap();
        let twice = (space * 2).unwrap();
        assert_eq!(twice.bounds(), &Interval::LeftOpen((180, 4096)));
        assert_eq!(twice.step(), &200);

        let neg_invalid = space * -1;
        assert!(neg_invalid.is_none());
    }

    #[test]
    fn mult_two_spaces() {
        let space1 = LinearSpace::try_bounded(interval!((8, =20)), 2).unwrap();
        let space2 = LinearSpace::try_bounded(interval!([5, 10]), 3).unwrap();
        let prod = space1 * space2;

        assert_eq!(prod.bounds(), &Interval::LeftOpen((40, 200)));
        assert_eq!(prod.step(), &6);
    }
}
