//! Routines to perform rounding operations on arbitrary numeric types.
//!
//! <https://en.wikipedia.org/wiki/Rounding>

use core::{fmt, ops::Sub};

use crate::{
    helper::{minmax, OneOrPair},
    traits::Zero,
};

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

/// Extend the [`DiscreteOrdSet`] to support rounding.
pub trait Roundable: DiscreteOrdSet
where
    Self::Point: Zero + Ord,
{
    /// Round the given point according to the specified [`Mode`].
    ///
    /// # Errors
    ///
    /// `Err(RoundError)` when the candidate points to round to returned by
    ///  [`DiscreteOrdSet::get_nearest`] either absent or invalid.
    fn round(
        &self,
        point: &Self::Point,
        mode: impl Into<Mode>,
    ) -> Result<Self::Point, RoundError<Self::Point>>
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
    /// # Errors
    ///
    /// `Err(RoundError)` when the candidate points to round to returned by
    ///  [`DiscreteOrdSet::get_nearest`] either absent or invalid.
    fn round_with_rng<'r>(
        &self,
        point: &Self::Point,
        mode: impl Into<Mode>,
        rng: impl Into<Option<&'r mut dyn RandRng>>,
    ) -> Result<Self::Point, RoundError<Self::Point>>
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Error while rounding a point to the nearest representable value in a space.
pub enum RoundError<T> {
    /// The rounded value, although being the nearest representable value
    /// to the input `point` is not in the requested direction
    /// (e.g., the rounded value is greater than the input point
    /// when the [floor mode][DirectedMode::TowardNegativeInfinity] is requested).
    InvalidDirection {
        /// The nearest representable value.
        rounded: T,
        /// The mode requested for rounding, which the `rounded` value does not satisfy.
        direction: DirectedMode,
    },
    /// No rounding candidates provided.
    NoCandidates,
}

impl<T> fmt::Display for RoundError<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDirection { rounded, direction } => {
                write!(f, "The rounded value ")?;
                rounded.fmt(f)?;
                write!(f, " does not match the requested direction {direction:?}")
            }
            Self::NoCandidates => write!(f, "No candidates to round to"),
        }
    }
}

fn round<S, T>(
    space: &S,
    point: &T,
    mode: Mode,
    rng: Option<&mut dyn RandRng>,
) -> Result<T, RoundError<T>>
where
    S: Roundable<Point = T> + ?Sized,
    T: Zero + Ord,
    for<'any> &'any T: Sub,
    for<'any> <&'any T as Sub>::Output: Distance,
{
    match space
        .get_nearest_ordered(point)
        .ok_or(RoundError::NoCandidates)?
    {
        OneOrPair::Pair((a, b)) => {
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
                return Ok(a);
            }
            if point >= &b {
                return Ok(b);
            }

            mode.round(point, OneOrPair::Pair((a, b)), rng)
        }
        single @ OneOrPair::One(_) => mode.round(point, single, rng),
    }
}

impl<S> Roundable for S
where
    S: DiscreteOrdSet,
    S::Point: Zero + Ord,
{
}
