use core::{cmp::Ordering, ops::Sub};

use crate::{
    helper::{OneOrPair, Pair},
    traits::Zero,
};

use super::{
    rand::{Distance, RandRng},
    RoundError,
};

#[cfg(feature = "random")]
use super::rand::{bernoulli_sample, UNIFORM_CHOICE_PROB};

/// Rounding modes supported by the rounding routines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Directed rounding without tie situations.
    Directed(DirectedMode),

    /// Round to the nearest representable value,
    /// with a specified tie-breaking strategy.
    Nearest(TieBreakingMode),

    #[cfg(feature = "random")]
    /// Stochastic rounding, picking between two nearest values
    /// with probability proportional to their distance from the original value.
    Stochastic,
}

impl Mode {
    /// Round the given point with the mode, providing nearest point(s).
    ///
    /// # Errors
    ///
    /// Return the nearest point (if it is [single][OneOrPair::One])
    /// and the rounding cannot be made, e.g. the `nearest > point`
    /// for the [floor mode][DirectedMode::TowardNegativeInfinity].
    pub(super) fn round<T>(
        self,
        point: &T,
        nearest: OneOrPair<T>,
        rng: Option<&mut dyn RandRng>,
    ) -> Result<T, RoundError<T>>
    where
        T: Zero + PartialOrd,
        for<'any> &'any T: Sub,
        for<'any> <&'any T as Sub>::Output: Distance,
    {
        match self {
            Self::Directed(dir_mode) => dir_mode.round(point, nearest),
            Self::Nearest(tie_mode) => {
                Ok(nearest.single_or_fold(|nearest_lower, nearest_upper| {
                    let selection = {
                        let to_upper = distance(&nearest_upper, point);
                        let to_lower = distance(&nearest_lower, point);
                        match to_upper.partial_cmp(&to_lower) {
                            Some(Ordering::Less) => TieSelection::Right,
                            Some(Ordering::Greater) => TieSelection::Left,
                            Some(Ordering::Equal) | None => {
                                // when the distances are equal, use the tie-breaking mode
                                tie_mode.select((&nearest_lower, &nearest_upper), rng)
                            }
                        }
                    };
                    match selection {
                        TieSelection::Left => nearest_lower,
                        TieSelection::Right => nearest_upper,
                    }
                }))
            }
            #[cfg(feature = "random")]
            Self::Stochastic => {
                Ok(nearest.single_or_fold(|nearest_lower, nearest_upper| {
                    let total: f64 = distance(&nearest_upper, &nearest_lower)
                        .try_into()
                        .unwrap_or(1.0);
                    let to_lower: f64 = distance(&nearest_lower, point).try_into().unwrap_or(0.0);

                    // the closer (_less_ distance) to `lower`, the _lower_ the probability to pick `upper`
                    //
                    // Note: division by zero is safe here because the +/- inf handled separately.
                    let prob_upper = to_lower / total;
                    let prob_upper = if prob_upper.is_finite() {
                        prob_upper.clamp(0.0, 1.0)
                    } else {
                        0.0
                    };

                    let select_upper = bernoulli_sample(prob_upper, rng);
                    if select_upper {
                        nearest_upper
                    } else {
                        nearest_lower
                    }
                }))
            }
        }
    }
}

/// Directed rounding aims towards/away from a predetermined limit point.
///
/// For [`Self::TowardPositiveInfinity`] and [`Self::TowardNegativeInfinity`]
/// there is always a unique representable value in the requested direction,
/// so no tie between candidate values can occur.
///
/// For [`Self::TowardZero`] and [`Self::AwayFromZero`] a tie can occur
/// when the two closest representable values are equidistant
/// on either side of the _zero point_ (or _incomparable to zero_).
/// In that case, the implementation falls back to [`Mode::Nearest`]
/// and uses its [`TieBreakingMode`] to select the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectedMode {
    #[doc(alias = "truncate")]
    /// Round towards zero.
    ///
    /// If `x` is positive, it is the same as [round-down][Self::TowardNegativeInfinity].
    /// If `x` is negative, it is the same as [round-up][Self::TowardPositiveInfinity].
    TowardZero,

    /// Round away from zero.
    ///
    /// If `x` is positive, it is the same as [round-up][Self::TowardPositiveInfinity].
    /// If `x` is negative, it is the same as [round-down][Self::TowardNegativeInfinity].
    AwayFromZero,

    #[doc(alias = "ceiling")]
    /// Round up (towards positive infinity) aka `ceiling`.
    ///
    /// If `x` is positive, it is the same as [round-away-from-zero][Self::AwayFromZero].
    /// If `x` is negative, it is the same as [round-toward-zero][Self::TowardZero].
    TowardPositiveInfinity,

    #[doc(alias = "floor")]
    /// Round down (towards negative infinity) aka `floor`.
    ///
    /// If `x` is positive, it is the same as [round-toward-zero][Self::TowardZero].
    /// If `x` is negative, it is the same as [round-away-from-zero][Self::AwayFromZero].
    TowardNegativeInfinity,
}

impl DirectedMode {
    fn round<T>(self, point: &T, nearest: OneOrPair<T>) -> Result<T, RoundError<T>>
    where
        T: Zero + PartialOrd,
        for<'any> &'any T: Sub,
        for<'any> <&'any T as Sub>::Output: Distance,
    {
        let rounded = match nearest {
            OneOrPair::One(nearest) => nearest,
            OneOrPair::Pair((nearest_lower, nearest_upper)) => match self {
                Self::TowardZero | Self::AwayFromZero => {
                    match self.select((&nearest_lower, &nearest_upper)) {
                        Some(TieSelection::Left) => nearest_lower,
                        Some(TieSelection::Right) => nearest_upper,
                        None => Mode::Nearest(TieBreakingMode::Directed(self)).round(
                            point,
                            OneOrPair::Pair((nearest_lower, nearest_upper)),
                            None,
                        )?,
                    }
                }
                Self::TowardPositiveInfinity => nearest_upper,
                Self::TowardNegativeInfinity => nearest_lower,
            },
        };
        if self.sanity_check(point, &rounded) {
            Ok(rounded)
        } else {
            Err(RoundError::InvalidDirection {
                rounded,
                direction: self,
            })
        }
    }

    fn sanity_check<T>(self, point: &T, rounded: &T) -> bool
    where
        T: Zero + PartialOrd,
    {
        match (self, point.cmp_zero()) {
            (Self::TowardZero, ord) => match ord {
                Some(Ordering::Greater) => {
                    rounded <= point && rounded.cmp_zero().is_some_and(Ordering::is_ge)
                }
                Some(Ordering::Equal) => rounded == point,
                Some(Ordering::Less) => {
                    rounded >= point && rounded.cmp_zero().is_some_and(Ordering::is_le)
                }
                None => false, // incomparable to zero
            },
            (Self::AwayFromZero, ord) => {
                match ord {
                    Some(Ordering::Greater) => rounded >= point,
                    Some(Ordering::Equal) => true,
                    Some(Ordering::Less) => rounded <= point,
                    None => false, // incomparable to zero
                }
            }
            (Self::TowardPositiveInfinity, _) => rounded >= point,
            (Self::TowardNegativeInfinity, _) => rounded <= point,
        }
    }

    /// Select which of the two values to pick based on its distance to zero.
    ///
    /// In case of a tie, return `None`.
    fn select<T>(self, (left, right): Pair<&T>) -> Option<TieSelection>
    where
        T: Zero + PartialOrd,
        for<'any> &'any T: Sub,
        for<'any> <&'any T as Sub>::Output: PartialOrd,
    {
        use TieSelection::{Left, Right};

        let zero = T::zero();
        match self {
            Self::TowardZero => {
                let right_abs = distance(right, &zero);
                let left_abs = distance(left, &zero);
                match right_abs.partial_cmp(&left_abs) {
                    Some(Ordering::Greater) => Some(Left),
                    Some(Ordering::Less) => Some(Right),
                    Some(Ordering::Equal) | None => None,
                }
            }
            Self::AwayFromZero => {
                let right_abs = distance(right, &zero);
                let left_abs = distance(left, &zero);
                match right_abs.partial_cmp(&left_abs) {
                    Some(Ordering::Greater) => Some(Right),
                    Some(Ordering::Less) => Some(Left),
                    Some(Ordering::Equal) | None => None,
                }
            }
            Self::TowardPositiveInfinity => Some(Right),
            Self::TowardNegativeInfinity => Some(Left),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum TieSelection {
    Left,
    Right,
}

/// The strategy to break ties (pick one of two values) when the rounded value
/// is exactly halfway between two representable values.
#[derive(Debug, Clone, Copy)]
pub enum TieBreakingMode {
    /// Pick the value in given [direction][DirectedMode].
    Directed(DirectedMode),

    // TODO: implement evenness tie-breaking
    // /// Pick the nearest even/odd representable value.
    // /// Only applicable for integer rounding.
    // ///
    // /// <https://en.wikipedia.org/wiki/Rounding#Rounding_half_to_even>
    // Evenness(bool),
    //
    #[cfg(feature = "random")]
    /// Pick between two values at random with the given probability:
    /// - right with probability `p = prob_upper`.
    /// - left with probability `q = 1 - prob_upper`;
    Random {
        /// Probability to pick the upper value.
        /// The value, if specified, will be clamped into `[0, 1]` interval.
        ///
        /// Defaults to `0.5` (if `None`) for uniform distribution.
        prob_upper: Option<f64>,
    },
}

// ---- manual PartialEq and Eq implementations to deal with `f64: !Eq` ---
impl PartialEq for TieBreakingMode {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Directed(a), Self::Directed(b)) => a == b,
            #[cfg(feature = "random")]
            (Self::Random { prob_upper: a }, Self::Random { prob_upper: b }) => a
                .unwrap_or_default()
                .total_cmp(&b.unwrap_or_default())
                .is_eq(),
            #[cfg(feature = "random")]
            (Self::Directed(_), Self::Random { .. }) | (Self::Random { .. }, Self::Directed(_)) => {
                false
            }
        }
    }
}

impl Eq for TieBreakingMode {}

impl TieBreakingMode {
    /// If the tie breaking fails to break a tie, default to `Right` (upper).
    const fn last_resort_for_equidistant_to_zero(self) -> TieSelection {
        #![allow(clippy::unused_self)]
        TieSelection::Right
    }

    #[allow(clippy::needless_pass_by_value)]
    fn select<T>(self, pair: Pair<&T>, rng: Option<&mut dyn RandRng>) -> TieSelection
    where
        T: Zero + PartialOrd,
        for<'any> &'any T: Sub,
        for<'any> <&'any T as Sub>::Output: PartialOrd,
    {
        #[cfg(not(feature = "random"))]
        let _ = rng;

        match self {
            Self::Directed(dir_mode) => dir_mode
                .select(pair)
                .unwrap_or_else(|| self.last_resort_for_equidistant_to_zero()),
            #[cfg(feature = "random")]
            Self::Random { prob_upper } => {
                let prob_upper = prob_upper
                    .filter(|p| p.is_finite())
                    .map_or(UNIFORM_CHOICE_PROB, |p| p.clamp(0.0, 1.0));

                let select_upper = bernoulli_sample(prob_upper, rng);
                if select_upper {
                    TieSelection::Right
                } else {
                    TieSelection::Left
                }
            }
        }
    }
}

fn distance<T, Diff>(x: T, y: T) -> Diff
where
    T: PartialOrd + Sub<Output = Diff>,
{
    if x >= y {
        x - y
    } else {
        y - x
    }
}

impl From<DirectedMode> for Mode {
    fn from(mode: DirectedMode) -> Self {
        Self::Directed(mode)
    }
}

impl From<DirectedMode> for TieBreakingMode {
    fn from(mode: DirectedMode) -> Self {
        Self::Directed(mode)
    }
}
