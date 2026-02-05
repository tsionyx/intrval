use core::{cmp::Ordering, ops::Sub};

use crate::{helper::Pair, traits::Zero};

use super::rand::{Distance, RandRng};

#[cfg(feature = "random")]
use super::rand::{bernoulli_sample, Probability};

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
    pub(super) fn round<T>(
        self,
        point: &T,
        (nearest_lower, nearest_upper): Pair<T>,
        rng: Option<&mut dyn RandRng>,
    ) -> T
    where
        T: Zero + PartialOrd,
        for<'any> &'any T: Sub,
        for<'any> <&'any T as Sub>::Output: Distance,
    {
        match self {
            Self::Directed(dir_mode) => dir_mode.round(point, (nearest_lower, nearest_upper)),
            Self::Nearest(tie_mode) => {
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
            }
            #[cfg(feature = "random")]
            Self::Stochastic => {
                let total: Option<f64> = distance(&nearest_upper, &nearest_lower).try_into().ok();
                let to_lower: Option<f64> = distance(&nearest_lower, point).try_into().ok();

                // the closer (_less_ distance) to `lower`, the _lower_ the probability to pick `upper`
                //
                // Note: division by zero is safe here because the +/- inf handled separately.
                let prob_upper = Probability::new(
                    total.and_then(|total| to_lower.map(|to_lower| to_lower / total)),
                );

                let select_upper = bernoulli_sample(prob_upper.get_f64(), rng);
                if select_upper {
                    nearest_upper
                } else {
                    nearest_lower
                }
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
    /// An alias for [`Self::TowardPositiveInfinity`].
    pub const UP: Self = Self::TowardPositiveInfinity;

    /// An alias for [`Self::TowardPositiveInfinity`].
    pub const CEILING: Self = Self::TowardPositiveInfinity;

    /// An alias for [`Self::TowardNegativeInfinity`].
    pub const DOWN: Self = Self::TowardNegativeInfinity;
    /// An alias for [`Self::TowardNegativeInfinity`].
    pub const FLOOR: Self = Self::TowardNegativeInfinity;

    /// An alias for [`Self::TowardZero`].
    pub const TRUNCATE: Self = Self::TowardZero;

    fn round<T>(self, point: &T, (nearest_lower, nearest_upper): Pair<T>) -> T
    where
        T: Zero + PartialOrd,
        for<'any> &'any T: Sub,
        for<'any> <&'any T as Sub>::Output: Distance,
    {
        match self {
            Self::TowardZero | Self::AwayFromZero => {
                match self.select((&nearest_lower, &nearest_upper)) {
                    Some(TieSelection::Left) => nearest_lower,
                    Some(TieSelection::Right) => nearest_upper,
                    None => Mode::Nearest(TieBreakingMode::Directed(self)).round(
                        point,
                        (nearest_lower, nearest_upper),
                        None,
                    ),
                }
            }
            Self::TowardPositiveInfinity => nearest_upper,
            Self::TowardNegativeInfinity => nearest_lower,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        prob_upper: Probability,
    },
}

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
                let prob_upper = prob_upper.get_f64();
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
