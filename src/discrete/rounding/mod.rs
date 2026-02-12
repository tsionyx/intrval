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

pub use self::{
    modes::{DirectedMode, Mode, TieBreakingMode},
    rand::Distance,
};

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
        let (a, b) = match self.get_nearest_ordered(point)? {
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

        Some(mode.into().round(point, (a, b)))
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

impl<S> Roundable for S
where
    S: DiscreteOrdSet,
    S::Point: Zero + Ord,
{
}
