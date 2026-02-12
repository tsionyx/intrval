//! Routines to perform rounding operations on arbitrary numeric types.
//!
//! <https://en.wikipedia.org/wiki/Rounding>

use core::ops::Sub;

use crate::helper::{minmax, OneOrPair, Zero};

use super::DiscreteOrdSet;

mod modes;
mod rand;
#[cfg(test)]
mod tests;

use self::rand::RandRng;

pub use self::{
    modes::{DirectedMode, Mode, TieBreakingMode},
    rand::Distance,
};

#[cfg(feature = "random")]
pub use self::rand::Probability;

/// Extend the [`DiscreteOrdSet`] to support rounding.
pub trait Roundable: DiscreteOrdSet
where
    Self::Point: Zero + Ord,
{
    /// Round the given point according to the specified [`Mode`].
    ///
    /// # Returns
    ///
    /// `None` when there are no nearest points to round to.
    fn round(&self, point: &Self::Point, mode: impl Into<Mode>) -> Option<Self::Point>
    where
        for<'any> &'any Self::Point: Sub,
        for<'any> <&'any Self::Point as Sub>::Output: Distance,
    {
        round(self, point, mode.into(), None)
    }

    #[cfg(feature = "random")]
    #[allow(single_use_lifetimes)] // error[E0658]: anonymous lifetimes in `impl Trait` are unstable
    /// Round the given point according to the specified [`Mode`].
    ///
    /// Optional random number generator can be provided
    /// (only valid for [stochastic rounding][Mode::Stochastic]
    /// or [random tie-breaking][TieBreakingMode::Random]).
    ///
    /// # Returns
    ///
    /// `None` when there are no nearest points to round to.
    fn round_with_rng<'r>(
        &self,
        point: &Self::Point,
        mode: impl Into<Mode>,
        rng: impl Into<Option<&'r mut dyn RandRng>>,
    ) -> Option<Self::Point>
    where
        for<'any> &'any Self::Point: Sub,
        for<'any> <&'any Self::Point as Sub>::Output: Distance,
    {
        round(self, point, mode.into(), rng.into())
    }

    /// Sort and normalize the [`nearest points`][DiscreteOrdSet::get_nearest].
    ///
    /// This orders the two bounds in ascending order and, if both bounds
    /// are equal (`OneOrPair::Pair((x, x))`), collapses them into `OneOrPair::One(x)`.
    fn get_nearest_ordered(&self, point: &Self::Point) -> Option<OneOrPair<Self::Point>> {
        self.get_nearest(point).map(|nearest| match nearest {
            x @ OneOrPair::One(_) => x,
            OneOrPair::Pair((a, b)) => {
                if a == b {
                    OneOrPair::One(a)
                } else {
                    OneOrPair::Pair(minmax(a, b).into())
                }
            }
        })
    }
}

fn round<S, T>(space: &S, point: &T, mode: Mode, rng: Option<&mut dyn RandRng>) -> Option<T>
where
    S: Roundable<Point = T> + ?Sized,
    T: Zero + Ord,
    for<'any> &'any T: Sub,
    for<'any> <&'any T as Sub>::Output: Distance,
{
    let (a, b) = match space.get_nearest_ordered(point)? {
        OneOrPair::Pair(x) => x,
        OneOrPair::One(v) => {
            return Some(v);
        }
    };

    // check that the nearest points returned by [`DiscreteOrdSet::get_nearest`]
    // (in case of a `OneOrPair::Pair` result)  are correct,
    // i.e. that the given point is between them.
    debug_assert!(
        point >= &a && point <= &b,
        "The pair of nearest points do not match the point to round",
    );

    // the inequality should not be allowed in a correct
    // implementation of [`DiscreteOrdSet::get_nearest`],
    // but we handle it gracefully by clamping to the nearest bound.
    if point <= &a {
        return Some(a);
    }
    if point >= &b {
        return Some(b);
    }

    Some(mode.round(point, (a, b), rng))
}

impl<S> Roundable for S
where
    S: DiscreteOrdSet,
    S::Point: Zero + Ord,
{
}
