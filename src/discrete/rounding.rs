//! Routines to perform rounding operations on arbitrary numeric types.
//!
//! <https://en.wikipedia.org/wiki/Rounding>

use core::{cmp::Ordering, ops::Sub};

use crate::helper::{minmax, OneOrPair, Pair, Zero};

use super::DiscreteOrdSet;

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
    fn round<T>(self, point: &T, (nearest_lower, nearest_upper): Pair<T>) -> T
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
                            tie_mode.select((&nearest_lower, &nearest_upper))
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
                let total: f64 = distance(&nearest_upper, &nearest_lower)
                    .try_into()
                    .unwrap_or(1.0);
                let to_lower: f64 = distance(&nearest_lower, point).try_into().unwrap_or(0.0);

                // the closer (_less_ distance) to `lower`, the _lower_ the probability to pick `upper`
                //
                // Note: division by zero is safe here because clamping will handle it:
                // - `+inf.clamp(0.0, 1.0)` -> `1.0`;
                // - `-inf.clamp(0.0, 1.0)` -> `0.0`;
                let prob_upper = (to_lower / total).clamp(0.0, 1.0);

                if rand::random_bool(prob_upper) {
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
                    None => Mode::Nearest(TieBreakingMode::Directed(self))
                        .round(point, (nearest_lower, nearest_upper)),
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
    /// Pick between two values at random with equal probability `p=0.5`.
    Random,
}

impl TieBreakingMode {
    /// If the tie breaking fails to break a tie, default to `Right` (upper).
    const fn last_resort_for_equidistant_to_zero(self) -> TieSelection {
        #![allow(clippy::unused_self)]
        TieSelection::Right
    }

    fn select<T>(self, pair: Pair<&T>) -> TieSelection
    where
        T: Zero + PartialOrd,
        for<'any> &'any T: Sub,
        for<'any> <&'any T as Sub>::Output: PartialOrd,
    {
        match self {
            Self::Directed(dir_mode) => dir_mode
                .select(pair)
                .unwrap_or_else(|| self.last_resort_for_equidistant_to_zero()),
            #[cfg(feature = "random")]
            Self::Random => {
                if rand::random_bool(0.5) {
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

#[cfg(not(feature = "random"))]
/// When the `random` feature is disabled, we only require `PartialOrd`.
pub trait Distance: PartialOrd {}
#[cfg(not(feature = "random"))]
impl<T> Distance for T where T: PartialOrd {}

#[cfg(feature = "random")]
/// When the `random` feature is enabled, we require additional traits
/// to support stochastic rounding using the distances,
pub trait Distance: PartialOrd + TryInto<f64> {}
#[cfg(feature = "random")]
impl<T> Distance for T where T: PartialOrd + TryInto<f64> {}

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

#[cfg(test)]
mod tests {
    use core::fmt::Debug;

    use crate::interval;

    use super::{super::LinearSpace, *};

    fn half_open_multiples_of_3() -> LinearSpace<i32> {
        LinearSpace::try_bounded(interval!((-20, =40)), 3).unwrap()
    }

    fn half_unbounded_odd() -> LinearSpace<i32> {
        LinearSpace::try_bounded(interval!(> 17), 2).unwrap()
    }

    fn hundreds() -> LinearSpace<i32> {
        LinearSpace::try_new(100).unwrap()
    }

    type F64 = ordered_float::OrderedFloat<f64>;

    const fn f(x: f64) -> F64 {
        ordered_float::OrderedFloat(x)
    }

    fn integer_floats() -> LinearSpace<F64> {
        LinearSpace::try_new(f(1.0)).unwrap()
    }

    fn half_open_halves() -> LinearSpace<F64> {
        LinearSpace::try_bounded(interval!((=f(-42.0), f(13.55))), f(0.5)).unwrap()
    }

    fn assert_rounding_cases<T, R>(space: &R, mode: impl Into<Mode>, tests: &[(T, T)])
    where
        R: Roundable<Point = T>,
        T: Zero + Ord + Debug,
        for<'any> &'any T: Sub,
        for<'any> <&'any T as Sub>::Output: Distance,
    {
        let mode = mode.into();
        for (input, expected) in tests {
            assert_eq!(
                space.round(input, mode).unwrap(),
                *expected,
                "Rounding {input:?} with mode {mode:?} != {expected:?}",
            );
        }
    }

    #[test]
    fn directed_zero_rounding_with_excluded_zero_always_choose_upper_on_tie() {
        let space = LinearSpace::try_bounded(interval!([-10, 10]), 4).unwrap();

        assert_rounding_cases(
            &space,
            DirectedMode::TowardZero,
            &[
                (-10, -10),
                (-9, -6),
                (-5, -2),
                (-2, -2),
                (-1, -2), // tie: choose the one closer to original point
                (0, 2),   // tie: choose upper as a last resort tie breaking
                (1, 2),   // tie: choose the one closer to original point
                (2, 2),
                (5, 2),
                (7, 6),
                (10, 10),
            ],
        );

        assert_rounding_cases(
            &space,
            DirectedMode::AwayFromZero,
            &[
                (-10, -10),
                (-9, -10),
                (-5, -6),
                (-2, -2),
                (-1, -2), // tie: choose the one closer to original point
                (0, 2),   // tie: choose upper as a last resort tie breaking
                (1, 2),   // tie: choose the one closer to original point
                (2, 2),
                (5, 6),
                (7, 10),
                (10, 10),
            ],
        );
    }

    mod truncate {
        use super::*;

        const MODE: DirectedMode = DirectedMode::TowardZero;

        #[test]
        fn int_3() {
            let space = half_open_multiples_of_3();

            assert_rounding_cases(
                &space,
                MODE,
                &[
                    (-100, -17),
                    (-21, -17),
                    (-19, -17),
                    (-17, -17),
                    (-16, -14),
                    (-8, -8),
                    (-7, -5),
                    (-2, -2),
                    (-1, 1), // between -2 and 1 select the one closer to 0
                    (0, 1),  // between -2 and 1 select the one closer to 0
                    (1, 1),
                    (2, 1),
                    (3, 1),
                    (4, 4),
                    (10, 10),
                    (11, 10),
                    (40, 40),
                    (41, 40),
                    (100, 40),
                ],
            );
        }

        #[test]
        fn int_odd() {
            let space = half_unbounded_odd();

            assert_rounding_cases(
                &space,
                MODE,
                &[
                    (-100, 19),
                    (0, 19),
                    (17, 19),
                    (18, 19),
                    (19, 19),
                    (20, 19),
                    (21, 21),
                    (99, 99),
                    (100, 99),
                    (101, 101),
                    (1_000, 999),
                    (1_001, 1_001),
                ],
            );
        }

        #[test]
        fn int_hundreds() {
            let space = hundreds();

            assert_rounding_cases(
                &space,
                MODE,
                &[
                    (-1_000, -1_000),
                    (-999, -900),
                    (-100, -100),
                    (-99, 0),
                    (-1, 0),
                    (0, 0),
                    (1, 0),
                    (99, 0),
                    (100, 100),
                    (901, 900),
                    (999, 900),
                    (1_000, 1_000),
                    (1_234_567, 1_234_500),
                ],
            );
        }

        #[test]
        fn float_integers() {
            let space = integer_floats();

            assert_rounding_cases(
                &space,
                MODE,
                &[
                    (f(-10.7), f(-10.0)),
                    (f(-10.5), f(-10.0)),
                    (f(-10.0), f(-10.0)),
                    (f(-9.5), f(-9.0)),
                    (f(-9.3), f(-9.0)),
                    (f(-0.5), f(-0.0)),
                    (f(0.0), f(0.0)),
                    (f(0.5), f(0.0)),
                    (f(5.5), f(5.0)),
                    (f(5.9), f(5.0)),
                    (f(10.42), f(10.0)),
                    (f(10.5), f(10.0)),
                    (f(13.0), f(13.0)),
                    (f(13.5), f(13.0)),
                    (f(14.0), f(14.0)),
                    (f(14.5), f(14.0)),
                    (f(15.0), f(15.0)),
                ],
            );
        }

        #[test]
        fn float_halves() {
            let space = half_open_halves();

            assert_rounding_cases(
                &space,
                MODE,
                &[
                    (f(-50.1), f(-42.0)),
                    (f(-42.3), f(-42.0)),
                    (f(-41.999), f(-41.5)),
                    (f(-10.7), f(-10.5)),
                    (f(-10.5), f(-10.5)),
                    (f(-10.0), f(-10.0)),
                    (f(-9.5), f(-9.5)),
                    (f(-9.3), f(-9.0)),
                    (f(-0.5), f(-0.5)),
                    (f(0.0), f(0.0)),
                    (f(0.5), f(0.5)),
                    (f(5.5), f(5.5)),
                    (f(5.9), f(5.5)),
                    (f(10.42), f(10.0)),
                    (f(10.5), f(10.5)),
                    (f(13.0), f(13.0)),
                    (f(13.5), f(13.5)),
                    (f(14.0), f(13.5)),
                    (f(14.5), f(13.5)),
                    (f(15.0), f(13.5)),
                ],
            );
        }
    }

    mod away_from_zero {
        use super::*;

        const MODE: DirectedMode = DirectedMode::AwayFromZero;

        #[test]
        fn int_3() {
            let space = half_open_multiples_of_3();

            assert_rounding_cases(
                &space,
                MODE,
                &[
                    (-100, -17),
                    (-21, -17),
                    (-19, -17),
                    (-17, -17),
                    (-16, -17),
                    (-8, -8),
                    (-7, -8),
                    (-2, -2),
                    (-1, -2), // between -2 and 1 select the one further from 0
                    (0, -2),  // between -2 and 1 select the one further from 0
                    (1, 1),
                    (2, 4),
                    (3, 4),
                    (4, 4),
                    (10, 10),
                    (11, 13),
                    (40, 40),
                    (41, 40),
                    (100, 40),
                ],
            );
        }

        #[test]
        fn int_odd() {
            let space = half_unbounded_odd();

            assert_rounding_cases(
                &space,
                MODE,
                &[
                    (-100, 19),
                    (0, 19),
                    (17, 19),
                    (18, 19),
                    (19, 19),
                    (20, 21),
                    (21, 21),
                    (99, 99),
                    (100, 101),
                    (101, 101),
                    (1_000, 1_001),
                    (1_001, 1_001),
                ],
            );
        }

        #[test]
        fn int_hundreds() {
            let space = hundreds();

            assert_rounding_cases(
                &space,
                MODE,
                &[
                    (-1_000, -1_000),
                    (-999, -1_000),
                    (-100, -100),
                    (-99, -100),
                    (-1, -100),
                    (0, 0),
                    (1, 100),
                    (99, 100),
                    (100, 100),
                    (901, 1_000),
                    (999, 1_000),
                    (1_000, 1_000),
                    (1_234_567, 1_234_600),
                ],
            );
        }

        #[test]
        fn float_integers() {
            let space = integer_floats();

            assert_rounding_cases(
                &space,
                MODE,
                &[
                    (f(-10.7), f(-11.0)),
                    (f(-10.5), f(-11.0)),
                    (f(-10.0), f(-10.0)),
                    (f(-9.5), f(-10.0)),
                    (f(-9.3), f(-10.0)),
                    (f(-0.5), f(-1.0)),
                    (f(0.0), f(0.0)),
                    (f(0.5), f(1.0)),
                    (f(5.5), f(6.0)),
                    (f(5.9), f(6.0)),
                    (f(10.42), f(11.0)),
                    (f(10.5), f(11.0)),
                    (f(13.0), f(13.0)),
                    (f(13.5), f(14.0)),
                    (f(14.0), f(14.0)),
                    (f(14.5), f(15.0)),
                    (f(15.0), f(15.0)),
                ],
            );
        }

        #[test]
        fn float_halves() {
            let space = half_open_halves();

            assert_rounding_cases(
                &space,
                MODE,
                &[
                    (f(-50.1), f(-42.0)),
                    (f(-42.3), f(-42.0)),
                    (f(-41.999), f(-42.0)),
                    (f(-10.7), f(-11.0)),
                    (f(-10.5), f(-10.5)),
                    (f(-10.0), f(-10.0)),
                    (f(-9.5), f(-9.5)),
                    (f(-9.3), f(-9.5)),
                    (f(-0.5), f(-0.5)),
                    (f(0.0), f(0.0)),
                    (f(0.5), f(0.5)),
                    (f(5.5), f(5.5)),
                    (f(5.9), f(6.0)),
                    (f(10.42), f(10.5)),
                    (f(10.5), f(10.5)),
                    (f(13.0), f(13.0)),
                    (f(13.5), f(13.5)),
                    (f(14.0), f(13.5)),
                    (f(14.5), f(13.5)),
                    (f(15.0), f(13.5)),
                ],
            );
        }
    }

    mod ceiling {
        use super::*;

        const MODE: DirectedMode = DirectedMode::TowardPositiveInfinity;

        #[test]
        fn int_3() {
            let space = half_open_multiples_of_3();

            assert_rounding_cases(
                &space,
                MODE,
                &[
                    (-100, -17),
                    (-21, -17),
                    (-19, -17),
                    (-17, -17),
                    (-16, -14),
                    (-8, -8),
                    (-7, -5),
                    (-2, -2),
                    (-1, 1),
                    (0, 1),
                    (1, 1),
                    (2, 4),
                    (3, 4),
                    (4, 4),
                    (10, 10),
                    (11, 13),
                    (40, 40),
                    (41, 40),
                    (100, 40),
                ],
            );
        }

        #[test]
        fn int_odd() {
            let space = half_unbounded_odd();

            assert_rounding_cases(
                &space,
                MODE,
                &[
                    (-100, 19),
                    (0, 19),
                    (17, 19),
                    (18, 19),
                    (19, 19),
                    (20, 21),
                    (21, 21),
                    (99, 99),
                    (100, 101),
                    (101, 101),
                    (1_000, 1_001),
                    (1_001, 1_001),
                ],
            );
        }

        #[test]
        fn int_hundreds() {
            let space = hundreds();

            assert_rounding_cases(
                &space,
                MODE,
                &[
                    (-1_000, -1_000),
                    (-999, -900),
                    (-100, -100),
                    (-99, 0),
                    (-1, 0),
                    (0, 0),
                    (1, 100),
                    (99, 100),
                    (100, 100),
                    (901, 1_000),
                    (999, 1_000),
                    (1_000, 1_000),
                    (1_234_567, 1_234_600),
                ],
            );
        }

        #[test]
        fn float_integers() {
            let space = integer_floats();

            assert_rounding_cases(
                &space,
                MODE,
                &[
                    (f(-10.7), f(-10.0)),
                    (f(-10.5), f(-10.0)),
                    (f(-10.0), f(-10.0)),
                    (f(-9.5), f(-9.0)),
                    (f(-9.3), f(-9.0)),
                    (f(-0.5), f(0.0)),
                    (f(0.0), f(0.0)),
                    (f(0.5), f(1.0)),
                    (f(5.5), f(6.0)),
                    (f(5.9), f(6.0)),
                    (f(10.42), f(11.0)),
                    (f(10.5), f(11.0)),
                    (f(13.0), f(13.0)),
                    (f(13.5), f(14.0)),
                    (f(14.0), f(14.0)),
                    (f(14.5), f(15.0)),
                    (f(15.0), f(15.0)),
                ],
            );
        }

        #[test]
        fn float_halves() {
            let space = half_open_halves();

            assert_rounding_cases(
                &space,
                MODE,
                &[
                    (f(-50.1), f(-42.0)),
                    (f(-42.3), f(-42.0)),
                    (f(-41.999), f(-41.5)),
                    (f(-10.7), f(-10.5)),
                    (f(-10.5), f(-10.5)),
                    (f(-10.0), f(-10.0)),
                    (f(-9.5), f(-9.5)),
                    (f(-9.3), f(-9.0)),
                    (f(-0.5), f(-0.5)),
                    (f(0.0), f(0.0)),
                    (f(0.5), f(0.5)),
                    (f(5.5), f(5.5)),
                    (f(5.9), f(6.0)),
                    (f(10.42), f(10.5)),
                    (f(10.5), f(10.5)),
                    (f(13.0), f(13.0)),
                    (f(13.5), f(13.5)),
                    (f(14.0), f(13.5)),
                    (f(14.5), f(13.5)),
                    (f(15.0), f(13.5)),
                ],
            );
        }
    }

    mod floor {
        use super::*;

        const MODE: DirectedMode = DirectedMode::TowardNegativeInfinity;

        #[test]
        fn int_3() {
            let space = half_open_multiples_of_3();

            assert_rounding_cases(
                &space,
                MODE,
                &[
                    (-100, -17),
                    (-21, -17),
                    (-19, -17),
                    (-17, -17),
                    (-16, -17),
                    (-8, -8),
                    (-7, -8),
                    (-2, -2),
                    (-1, -2),
                    (0, -2),
                    (1, 1),
                    (2, 1),
                    (3, 1),
                    (4, 4),
                    (10, 10),
                    (11, 10),
                    (40, 40),
                    (41, 40),
                    (100, 40),
                ],
            );
        }

        #[test]
        fn int_odd() {
            let space = half_unbounded_odd();

            assert_rounding_cases(
                &space,
                MODE,
                &[
                    (-100, 19),
                    (0, 19),
                    (17, 19),
                    (18, 19),
                    (19, 19),
                    (20, 19),
                    (21, 21),
                    (99, 99),
                    (100, 99),
                    (101, 101),
                    (1_000, 999),
                    (1_001, 1_001),
                ],
            );
        }

        #[test]
        fn int_hundreds() {
            let space = hundreds();

            assert_rounding_cases(
                &space,
                MODE,
                &[
                    (-1_000, -1_000),
                    (-999, -1_000),
                    (-100, -100),
                    (-99, -100),
                    (-1, -100),
                    (0, 0),
                    (1, 0),
                    (99, 0),
                    (100, 100),
                    (901, 900),
                    (999, 900),
                    (1_000, 1_000),
                    (1_234_567, 1_234_500),
                ],
            );
        }

        #[test]
        fn float_integers() {
            let space = integer_floats();

            assert_rounding_cases(
                &space,
                MODE,
                &[
                    (f(-10.7), f(-11.0)),
                    (f(-10.5), f(-11.0)),
                    (f(-10.0), f(-10.0)),
                    (f(-9.5), f(-10.0)),
                    (f(-9.3), f(-10.0)),
                    (f(-0.5), f(-1.0)),
                    (f(0.0), f(0.0)),
                    (f(0.5), f(0.0)),
                    (f(5.5), f(5.0)),
                    (f(5.9), f(5.0)),
                    (f(10.42), f(10.0)),
                    (f(10.5), f(10.0)),
                    (f(13.0), f(13.0)),
                    (f(13.5), f(13.0)),
                    (f(14.0), f(14.0)),
                    (f(14.5), f(14.0)),
                    (f(15.0), f(15.0)),
                ],
            );
        }

        #[test]
        fn float_halves() {
            let space = half_open_halves();

            assert_rounding_cases(
                &space,
                MODE,
                &[
                    (f(-50.1), f(-42.0)),
                    (f(-42.3), f(-42.0)),
                    (f(-41.999), f(-42.0)),
                    (f(-10.7), f(-11.0)),
                    (f(-10.5), f(-10.5)),
                    (f(-10.0), f(-10.0)),
                    (f(-9.5), f(-9.5)),
                    (f(-9.3), f(-9.5)),
                    (f(-0.5), f(-0.5)),
                    (f(0.0), f(0.0)),
                    (f(0.5), f(0.5)),
                    (f(5.5), f(5.5)),
                    (f(5.9), f(5.5)),
                    (f(10.42), f(10.0)),
                    (f(10.5), f(10.5)),
                    (f(13.0), f(13.0)),
                    (f(13.5), f(13.5)),
                    (f(14.0), f(13.5)),
                    (f(14.5), f(13.5)),
                    (f(15.0), f(13.5)),
                ],
            );
        }
    }

    mod nearest {
        use super::*;

        #[test]
        fn no_ties_for_int_3() {
            let space = half_open_multiples_of_3();

            for mode in [
                DirectedMode::TowardZero,
                DirectedMode::AwayFromZero,
                DirectedMode::TowardPositiveInfinity,
                DirectedMode::TowardNegativeInfinity,
            ] {
                let mode = Mode::Nearest(mode.into());
                assert_rounding_cases(
                    &space,
                    mode,
                    &[
                        (-100, -17),
                        (-21, -17),
                        (-19, -17),
                        (-17, -17),
                        (-16, -17),
                        (-8, -8),
                        (-7, -8),
                        (-2, -2),
                        (-1, -2),
                        (0, 1),
                        (1, 1),
                        (2, 1),
                        (3, 4),
                        (4, 4),
                        (10, 10),
                        (11, 10),
                        (40, 40),
                        (41, 40),
                        (100, 40),
                    ],
                );
            }
        }

        #[test]
        fn int_odd() {
            let space = half_unbounded_odd();

            for mode in [
                DirectedMode::TowardZero,
                DirectedMode::TowardNegativeInfinity,
            ] {
                let mode = Mode::Nearest(mode.into());
                assert_rounding_cases(
                    &space,
                    mode,
                    &[
                        (-100, 19),
                        (0, 19),
                        (17, 19),
                        (18, 19),
                        (19, 19),
                        (20, 19),
                        (21, 21),
                        (99, 99),
                        (100, 99),
                        (101, 101),
                        (1_000, 999),
                        (1_001, 1_001),
                    ],
                );
            }

            for mode in [
                DirectedMode::AwayFromZero,
                DirectedMode::TowardPositiveInfinity,
            ] {
                let mode = Mode::Nearest(mode.into());
                assert_rounding_cases(
                    &space,
                    mode,
                    &[
                        (-100, 19),
                        (0, 19),
                        (17, 19),
                        (18, 19),
                        (19, 19),
                        (20, 21),
                        (21, 21),
                        (99, 99),
                        (100, 101),
                        (101, 101),
                        (1_000, 1_001),
                        (1_001, 1_001),
                    ],
                );
            }
        }

        #[test]
        fn int_hundreds_no_ties() {
            let space = hundreds();

            for mode in [
                DirectedMode::TowardZero,
                DirectedMode::AwayFromZero,
                DirectedMode::TowardPositiveInfinity,
                DirectedMode::TowardNegativeInfinity,
            ] {
                let mode = Mode::Nearest(mode.into());
                assert_rounding_cases(
                    &space,
                    mode,
                    &[
                        (-1_000, -1_000),
                        (-999, -1_000),
                        (-100, -100),
                        (-99, -100),
                        (-1, 0),
                        (0, 0),
                        (1, 0),
                        (99, 100),
                        (100, 100),
                        (901, 900),
                        (999, 1_000),
                        (1_000, 1_000),
                        (1_234_567, 1_234_600),
                    ],
                );
            }
        }

        #[test]
        fn float_integers() {
            let space = integer_floats();

            let mode = Mode::Nearest(DirectedMode::TowardZero.into());
            assert_rounding_cases(
                &space,
                mode,
                &[
                    (f(-10.7), f(-11.0)),
                    (f(-10.5), f(-10.0)),
                    (f(-10.0), f(-10.0)),
                    (f(-9.5), f(-9.0)),
                    (f(-9.3), f(-9.0)),
                    (f(-0.5), f(-0.0)),
                    (f(0.0), f(0.0)),
                    (f(0.5), f(0.0)),
                    (f(5.5), f(5.0)),
                    (f(5.9), f(6.0)),
                    (f(10.42), f(10.0)),
                    (f(10.5), f(10.0)),
                    (f(13.0), f(13.0)),
                    (f(13.5), f(13.0)),
                    (f(14.0), f(14.0)),
                    (f(14.5), f(14.0)),
                    (f(15.0), f(15.0)),
                ],
            );

            let mode = Mode::Nearest(DirectedMode::AwayFromZero.into());
            assert_rounding_cases(
                &space,
                mode,
                &[
                    (f(-10.7), f(-11.0)),
                    (f(-10.5), f(-11.0)),
                    (f(-10.0), f(-10.0)),
                    (f(-9.5), f(-10.0)),
                    (f(-9.3), f(-9.0)),
                    (f(-0.5), f(-1.0)),
                    (f(0.0), f(0.0)),
                    (f(0.5), f(1.0)),
                    (f(5.5), f(6.0)),
                    (f(5.9), f(6.0)),
                    (f(10.42), f(10.0)),
                    (f(10.5), f(11.0)),
                    (f(13.0), f(13.0)),
                    (f(13.5), f(14.0)),
                    (f(14.0), f(14.0)),
                    (f(14.5), f(15.0)),
                    (f(15.0), f(15.0)),
                ],
            );

            let mode = Mode::Nearest(DirectedMode::TowardPositiveInfinity.into());
            assert_rounding_cases(
                &space,
                mode,
                &[
                    (f(-10.7), f(-11.0)),
                    (f(-10.5), f(-10.0)),
                    (f(-10.0), f(-10.0)),
                    (f(-9.5), f(-9.0)),
                    (f(-9.3), f(-9.0)),
                    (f(-0.5), f(-0.0)),
                    (f(0.0), f(0.0)),
                    (f(0.5), f(1.0)),
                    (f(5.5), f(6.0)),
                    (f(5.9), f(6.0)),
                    (f(10.42), f(10.0)),
                    (f(10.5), f(11.0)),
                    (f(13.0), f(13.0)),
                    (f(13.5), f(14.0)),
                    (f(14.0), f(14.0)),
                    (f(14.5), f(15.0)),
                    (f(15.0), f(15.0)),
                ],
            );

            let mode = Mode::Nearest(DirectedMode::TowardNegativeInfinity.into());
            assert_rounding_cases(
                &space,
                mode,
                &[
                    (f(-10.7), f(-11.0)),
                    (f(-10.5), f(-11.0)),
                    (f(-10.0), f(-10.0)),
                    (f(-9.5), f(-10.0)),
                    (f(-9.3), f(-9.0)),
                    (f(-0.5), f(-1.0)),
                    (f(0.0), f(0.0)),
                    (f(0.5), f(0.0)),
                    (f(5.5), f(5.0)),
                    (f(5.9), f(6.0)),
                    (f(10.42), f(10.0)),
                    (f(10.5), f(10.0)),
                    (f(13.0), f(13.0)),
                    (f(13.5), f(13.0)),
                    (f(14.0), f(14.0)),
                    (f(14.5), f(14.0)),
                    (f(15.0), f(15.0)),
                ],
            );
        }

        #[test]
        fn float_halves_no_ties() {
            let space = half_open_halves();

            for mode in [
                DirectedMode::TowardZero,
                DirectedMode::AwayFromZero,
                DirectedMode::TowardPositiveInfinity,
                DirectedMode::TowardNegativeInfinity,
            ] {
                let mode = Mode::Nearest(mode.into());
                assert_rounding_cases(
                    &space,
                    mode,
                    &[
                        (f(-50.1), f(-42.0)),
                        (f(-42.3), f(-42.0)),
                        (f(-41.999), f(-42.0)),
                        (f(-10.7), f(-10.5)),
                        (f(-10.5), f(-10.5)),
                        (f(-10.0), f(-10.0)),
                        (f(-9.5), f(-9.5)),
                        (f(-9.3), f(-9.5)),
                        (f(-0.5), f(-0.5)),
                        (f(0.0), f(0.0)),
                        (f(0.5), f(0.5)),
                        (f(5.5), f(5.5)),
                        (f(5.9), f(6.0)),
                        (f(10.42), f(10.5)),
                        (f(10.5), f(10.5)),
                        (f(13.0), f(13.0)),
                        (f(13.5), f(13.5)),
                        (f(14.0), f(13.5)),
                        (f(14.5), f(13.5)),
                        (f(15.0), f(13.5)),
                    ],
                );
            }
        }
    }

    #[cfg(feature = "random")]
    mod random_rounds {
        extern crate std;
        use std::collections::HashMap;

        use super::*;

        #[test]
        fn equal_probability_for_tie() {
            let space = half_unbounded_odd();
            let mode = Mode::Nearest(TieBreakingMode::Random);

            let x = 100;
            let n = 1_000;
            let results = (0..n).map(|_| space.round(&x, mode).unwrap());

            let mut distrib = HashMap::new();
            for res in results {
                *distrib.entry(res).or_insert(0_u16) += 1;
            }

            let f_lower = distrib.remove(&(x - 1)).unwrap();
            let f_upper = distrib.remove(&(x + 1)).unwrap();
            assert!(distrib.is_empty());

            let max_allowed_deviation = n / 20; // ~sqrt(n)
            let expected_interval =
                (n / 2 - max_allowed_deviation)..(n / 2 + max_allowed_deviation);
            assert!(
                expected_interval.contains(&f_lower),
                "lower frequency {f_lower} not in {expected_interval:?}"
            );
            assert!(
                expected_interval.contains(&f_upper),
                "upper frequency {f_upper} not in {expected_interval:?}"
            );
        }

        #[test]
        fn stochastic() {
            let space = hundreds();
            let mode = Mode::Stochastic;

            let prec = 100;
            let x = 578;
            let n = 8_000;
            let results = (0..n).map(|_| space.round(&x, mode).unwrap());

            let mut distrib = HashMap::new();
            for res in results {
                *distrib.entry(res).or_insert(0_i32) += 1;
            }

            let lower_val = (x / prec) * prec;
            let f_lower = distrib.remove(&lower_val).unwrap();
            let f_upper = distrib.remove(&(lower_val + prec)).unwrap();
            assert!(distrib.is_empty());

            let max_allowed_deviation = n / 60; // ~sqrt(n)
            let expected_lower = n / prec * (prec - (x - lower_val));
            let expected_upper = n / prec * (x - lower_val);

            let expected_interval_lower =
                (expected_lower - max_allowed_deviation)..(expected_lower + max_allowed_deviation);
            assert!(
                expected_interval_lower.contains(&f_lower),
                "lower frequency {f_lower} not in {expected_interval_lower:?}"
            );

            let expected_interval_upper =
                (expected_upper - max_allowed_deviation)..(expected_upper + max_allowed_deviation);
            assert!(
                expected_interval_upper.contains(&f_upper),
                "upper frequency {f_upper} not in {expected_interval_upper:?}"
            );
        }
    }
}
