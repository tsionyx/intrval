use core::{cmp::Ordering, ops::Sub};

use crate::{
    helper::{OneOrPair, Pair},
    traits::Zero,
};

use super::{distance, rand::RandRng, RoundError, RoundingMode, TieSelection};

/// Directed rounding aims towards/away from a predetermined limit point.
///
/// For [`Self::TowardPositiveInfinity`] and [`Self::TowardNegativeInfinity`]
/// there is always a unique representable value in the requested direction,
/// so no tie between candidate values can occur.
///
/// For [`Self::TowardZero`] and [`Self::AwayFromZero`] a tie can occur
/// when the two closest representable values are equidistant
/// on either side of the _zero point_ (or _incomparable to zero_).
/// In that case, the implementation falls back to [`NearestMode`]
/// and uses `self` as `TieBreaking` to select the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "SCREAMING_SNAKE_CASE"))]
pub enum DirectedMode {
    /// Round towards zero.
    ///
    /// If `x` is positive, it is (almost) the same as [round-down][Self::TowardNegativeInfinity].
    /// If `x` is negative, it is (almost) the same as [round-up][Self::TowardPositiveInfinity].
    ///
    /// The only difference from the _floor/ceiling_ modes is that the rounding
    /// is **sign-preserving**, i.e., it is not allowed for a rounded value to cross zero.
    /// E.g., when rounding a positive number towards zero, the result cannot be negative,
    /// so if the nearest representable value(s) are negative, the rounding will
    /// yield an [error][RoundError::InvalidDirection].
    #[doc(alias = "truncate")]
    #[cfg_attr(feature = "serde", serde(alias = "TRUNCATE"))]
    TowardZero,

    /// Round away from zero.
    ///
    /// If `x` is positive, it is the same as [round-up][Self::TowardPositiveInfinity].
    /// If `x` is negative, it is the same as [round-down][Self::TowardNegativeInfinity].
    AwayFromZero,

    /// Round up (towards positive infinity) aka `ceiling`.
    ///
    /// If `x` is positive, it is the same as [round-away-from-zero][Self::AwayFromZero].
    /// If `x` is negative, it is the same as [round-toward-zero][Self::TowardZero].
    #[doc(alias = "ceiling")]
    #[cfg_attr(feature = "serde", serde(alias = "UP", alias = "CEILING"))]
    TowardPositiveInfinity,

    /// Round down (towards negative infinity) aka `floor`.
    ///
    /// If `x` is positive, it is the same as [round-toward-zero][Self::TowardZero].
    /// If `x` is negative, it is the same as [round-away-from-zero][Self::AwayFromZero].
    #[doc(alias = "floor")]
    #[cfg_attr(feature = "serde", serde(alias = "DOWN", alias = "FLOOR"))]
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
}

impl<T> RoundingMode<T> for DirectedMode
where
    T: Zero + PartialOrd,
    for<'any> &'any T: Sub,
    for<'any> <&'any T as Sub>::Output: PartialOrd,
{
    fn round(
        &self,
        point: &T,
        nearest: OneOrPair<T>,
        rng: Option<&mut dyn RandRng>,
    ) -> Result<T, RoundError<T>> {
        let rounded = match nearest {
            OneOrPair::One(nearest) => nearest,
            OneOrPair::Pair((nearest_lower, nearest_upper)) => match self {
                Self::TowardZero | Self::AwayFromZero => {
                    match self.select_opt((&nearest_lower, &nearest_upper), None) {
                        Some(TieSelection::Left) => nearest_lower,
                        Some(TieSelection::Right) => nearest_upper,
                        None => NearestMode(*self).round(
                            point,
                            OneOrPair::Pair((nearest_lower, nearest_upper)),
                            rng,
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
                direction: *self,
            })
        }
    }
}

/// The mode wrapping a rounding strategy to always select
/// nearest of the two values, with a specified tie-breaking strategy
/// when the two values are equidistant from the point to round.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(
        from = "serde_repr::Nearest<Tie>",
        into = "serde_repr::Nearest<Tie>",
        bound = "Tie: Clone + serde::Serialize + serde::de::DeserializeOwned"
    )
)]
pub struct NearestMode<Tie>(pub Tie);

#[cfg(feature = "serde")]
/// Ser/de [`NearestMode`] using the '{"NEAREST": &lt;Tie&gt;}' format,
/// where `Tie` is the serialization of the underlying tie-breaking strategy.
mod serde_repr {
    use super::NearestMode;

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    #[serde(rename_all = "SCREAMING_SNAKE_CASE")]
    pub(super) struct Nearest<Tie> {
        nearest: Tie,
    }

    impl<Tie> From<Nearest<Tie>> for NearestMode<Tie> {
        fn from(value: Nearest<Tie>) -> Self {
            Self(value.nearest)
        }
    }

    impl<Tie> From<NearestMode<Tie>> for Nearest<Tie> {
        fn from(value: NearestMode<Tie>) -> Self {
            Self { nearest: value.0 }
        }
    }
}

impl<Tie, T> RoundingMode<T> for NearestMode<Tie>
where
    Tie: TieBreaking<T>,
    T: Zero + PartialOrd,
    for<'any> &'any T: Sub,
    for<'any> <&'any T as Sub>::Output: PartialOrd,
{
    fn round(
        &self,
        point: &T,
        nearest: OneOrPair<T>,
        rng: Option<&mut dyn RandRng>,
    ) -> Result<T, RoundError<T>> {
        let res = nearest.single_or_fold(|nearest_lower, nearest_upper| {
            let selection = {
                let to_upper = distance(&nearest_upper, point);
                let to_lower = distance(&nearest_lower, point);
                match to_upper.partial_cmp(&to_lower) {
                    Some(Ordering::Less) => TieSelection::Right,
                    Some(Ordering::Greater) => TieSelection::Left,
                    Some(Ordering::Equal) | None => {
                        // when the distances are equal, use the tie-breaking mode
                        self.0.select((&nearest_lower, &nearest_upper), rng)
                    }
                }
            };
            match selection {
                TieSelection::Left => nearest_lower,
                TieSelection::Right => nearest_upper,
            }
        });
        Ok(res)
    }

    fn is_stochastic(&self) -> bool {
        self.0.is_stochastic()
    }
}

/// The strategy to break ties (pick one of two values) when the rounded value
/// is exactly halfway between two representable values.
pub(super) trait TieBreaking<T> {
    // The selector function **always** returning a [`TieSelection`] result.
    fn select(&self, pair: Pair<&T>, rng: Option<&mut dyn RandRng>) -> TieSelection {
        self.select_opt(pair, rng)
            .unwrap_or_else(|| self.last_resort())
    }

    /// The selector function to pick one of the two values.
    ///
    /// This function can fail to select a particular value.
    fn select_opt(&self, pair: Pair<&T>, rng: Option<&mut dyn RandRng>) -> Option<TieSelection>;

    /// The last resort selection when the `select_opt` fails to select a particular value.
    fn last_resort(&self) -> TieSelection;

    /// Check if the tie breaking mode is stochastic (i.e., involves random choices).
    fn is_stochastic(&self) -> bool {
        false
    }
}

impl<T> TieBreaking<T> for DirectedMode
where
    T: Zero + PartialOrd,
    for<'any> &'any T: Sub,
    for<'any> <&'any T as Sub>::Output: PartialOrd,
{
    /// Select which of the two values to pick based on its distance to zero.
    ///
    /// In case of a tie, return `Self::last_resort_for_equidistant_to_zero()`.
    fn select_opt(
        &self,
        (left, right): Pair<&T>,
        _rng: Option<&mut dyn RandRng>,
    ) -> Option<TieSelection> {
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

    /// If the tie breaking fails to break a tie
    /// (when the candidates are equidistant to zero),
    /// default to `Right` (upper).
    fn last_resort(&self) -> TieSelection {
        TieSelection::Right
    }
}

// TODO: implement evenness tie-breaking
// /// Pick the nearest even/odd representable value.
// /// Only applicable for integer rounding.
// ///
// /// <https://en.wikipedia.org/wiki/Rounding#Rounding_half_to_even>
// pub struct Evenness(bool),

#[cfg(all(feature = "serde", test))]
mod deser_tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn toward_zero() {
        let j = json!("TOWARD_ZERO");
        let mode: DirectedMode = serde_json::from_value(j).unwrap();
        assert_eq!(mode, DirectedMode::TowardZero);
    }

    #[test]
    fn truncate() {
        let j = json!("TRUNCATE");
        let mode: DirectedMode = serde_json::from_value(j).unwrap();
        assert_eq!(mode, DirectedMode::TowardZero);
    }

    #[test]
    fn away_from_zero() {
        let j = json!("AWAY_FROM_ZERO");
        let mode: DirectedMode = serde_json::from_value(j).unwrap();
        assert_eq!(mode, DirectedMode::AwayFromZero);
    }

    #[test]
    fn toward_positive_infinity() {
        let j = json!("TOWARD_POSITIVE_INFINITY");
        let mode: DirectedMode = serde_json::from_value(j).unwrap();
        assert_eq!(mode, DirectedMode::TowardPositiveInfinity);
    }

    #[test]
    fn up() {
        let j = json!("UP");
        let mode: DirectedMode = serde_json::from_value(j).unwrap();
        assert_eq!(mode, DirectedMode::TowardPositiveInfinity);
    }

    #[test]
    fn ceiling() {
        let j = json!("CEILING");
        let mode: DirectedMode = serde_json::from_value(j).unwrap();
        assert_eq!(mode, DirectedMode::TowardPositiveInfinity);
    }

    #[test]
    fn toward_negative_infinity() {
        let j = json!("TOWARD_NEGATIVE_INFINITY");
        let mode: DirectedMode = serde_json::from_value(j).unwrap();
        assert_eq!(mode, DirectedMode::TowardNegativeInfinity);
    }

    #[test]
    fn down() {
        let j = json!("DOWN");
        let mode: DirectedMode = serde_json::from_value(j).unwrap();
        assert_eq!(mode, DirectedMode::TowardNegativeInfinity);
    }

    #[test]
    fn floor() {
        let j = json!("FLOOR");
        let mode: DirectedMode = serde_json::from_value(j).unwrap();
        assert_eq!(mode, DirectedMode::TowardNegativeInfinity);
    }

    #[test]
    fn nearest_mode() {
        let j = json!({
            "NEAREST": "TOWARD_ZERO",
        });
        let mode: NearestMode<DirectedMode> = serde_json::from_value(j).unwrap();
        assert_eq!(mode.0, DirectedMode::TowardZero);
    }

    #[test]
    fn nearest_mode_with_alias() {
        let j = json!({
            "NEAREST": "FLOOR",
        });
        let mode: NearestMode<DirectedMode> = serde_json::from_value(j).unwrap();
        assert_eq!(mode.0, DirectedMode::TowardNegativeInfinity);
    }
}
