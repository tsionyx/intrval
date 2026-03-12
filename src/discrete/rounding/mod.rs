//! Routines to perform rounding operations on arbitrary numeric types.
//! using discrete sets as the space of representable values.
//!
//! <https://en.wikipedia.org/wiki/Rounding>

use core::fmt;

use crate::{
    helper::{minmax, OneOrPair, StdError},
    traits::Zero,
};

use super::DiscreteOrdSet;

mod modes;
#[cfg(feature = "random")]
mod rand;
#[cfg(not(feature = "random"))]
mod rand {
    /// Dummy trait when the `random` feature is disabled.
    pub trait RandRng {}
}

#[cfg(test)]
mod tests;

pub use self::rand::RandRng;

pub use self::modes::{DirectedMode, NearestMode};

#[cfg(feature = "random")]
pub use self::rand::{Probability, RandomTie, StochasticMode};

/// Extend the [`DiscreteOrdSet`] to support rounding.
pub trait Rounding: DiscreteOrdSet
where
    Self::Point: Zero + Ord,
{
    /// Round the given point according to the specified [`RoundingMode`].
    ///
    ///
    /// # Note (for the case where _feature = "random"_ enabled)
    ///
    /// Performing a rounding with one of the [random-based modes][RoundingMode::is_stochastic]
    /// will use the fallback RNG for any random choices.
    /// This is `no_std` friendly but provides low-quality
    /// and cryptographically insecure predetermined results.
    /// It is recommended to set the environment variable `CONST_RANDOM_SEED=<RANDOM_STRING>`
    /// at compile time (during `cargo build`) to get a better quality of randomness.
    ///
    /// Also be aware that the fallback RNG is global and shared across all callers/threads in `no_std` environment.
    /// That means stochastic rounding results depend on cross-thread interleaving
    /// and on prior uses elsewhere in the process, which can make behavior hard
    /// to reproduce and tests order-dependent. If you experience any difficulties with this,
    /// consider providing your own RNG (and call it with `round_with_rng` method instead)
    /// or switch to one of the [deterministic modes][RoundingMode::is_deterministic].
    ///
    ///
    /// # Errors
    ///
    /// `Err(RoundError)` when the candidate points to round to returned by
    ///  [`DiscreteOrdSet::get_nearest`] either absent or invalid.
    fn round(
        &self,
        point: &Self::Point,
        mode: impl RoundingMode<Self::Point>,
    ) -> Result<Self::Point, RoundError<Self::Point>> {
        round(self, point, &mode, None)
    }

    #[cfg(feature = "random")]
    #[allow(single_use_lifetimes)] // error[E0658]: anonymous lifetimes in `impl Trait` are unstable
    /// Round the given point according to the specified [`RoundingMode`].
    ///
    /// Optional random number generator can be provided
    /// This method only makes sense for [stochastic rounding][StochasticMode]
    /// or [random tie-breaking][RandomTie] modes.
    /// If you are using fully [deterministic][RoundingMode::is_deterministic] mode,
    /// you should probably use the [`Self::round`] instead.
    ///
    ///
    /// # Note
    ///
    /// Performing a rounding with one of the [random-based modes][RoundingMode::is_stochastic]
    /// with `rng=None` will use the fallback [small rng][::rand::rngs::SmallRng] for any random choices.
    /// This is `no_std` friendly but provides low-quality
    /// and cryptographically insecure predetermined results.
    /// It is recommended to set the environment variable `CONST_RANDOM_SEED=<RANDOM_STRING>`
    /// at compile time (during `cargo build`) to get a better quality of randomness.
    ///
    /// Also be aware that the fallback RNG is global and shared across all callers/threads in `no_std` environment.
    /// That means stochastic rounding results depend on cross-thread interleaving and on prior uses elsewhere in the process,
    /// which can make behavior hard to reproduce and tests order-dependent.
    /// If you experience any difficulties with this, consider providing your own RNG
    /// or switch to one of the [deterministic modes][RoundingMode::is_deterministic].
    ///
    ///
    /// # Errors
    ///
    /// `Err(RoundError)` when the candidate points to round to returned by
    ///  [`DiscreteOrdSet::get_nearest`] either absent or invalid.
    fn round_with_rng<'r>(
        &self,
        point: &Self::Point,
        mode: impl RoundingMode<Self::Point>,
        rng: impl Into<Option<&'r mut dyn RandRng>>,
    ) -> Result<Self::Point, RoundError<Self::Point>> {
        round(self, point, &mode, rng.into())
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

impl<S> Rounding for S
where
    S: DiscreteOrdSet,
    S::Point: Zero + Ord,
{
}

/// Rounding modes supported by the rounding routines.
pub trait RoundingMode<T> {
    /// Round the given point with the mode, providing nearest point(s).
    ///
    /// # Errors
    ///
    /// Return the nearest point (if it is [single][OneOrPair::One])
    /// and the rounding cannot be made, e.g. the `nearest > point`
    /// for the [floor mode][DirectedMode::TowardNegativeInfinity].
    fn round(
        &self,
        point: &T,
        nearest: OneOrPair<T>,
        rng: Option<&mut dyn RandRng>,
    ) -> Result<T, RoundError<T>>;

    /// Check if the rounding mode is stochastic (i.e., involves random choices).
    fn is_stochastic(&self) -> bool {
        false
    }

    /// Check if the rounding mode is deterministic (i.e., does not involve random choices).
    fn is_deterministic(&self) -> bool {
        !self.is_stochastic()
    }
}

#[derive(Debug, Clone, Copy)]
enum TieSelection {
    Left,
    Right,
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

impl<T> StdError for RoundError<T> where T: fmt::Debug + fmt::Display {}

fn round<S, M, T>(
    space: &S,
    point: &T,
    mode: &M,
    rng: Option<&mut dyn RandRng>,
) -> Result<T, RoundError<T>>
where
    S: Rounding<Point = T> + ?Sized,
    T: Zero + Ord,
    M: RoundingMode<T>,
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
