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

#[derive(Debug, Copy, Clone)]
enum Expected<T> {
    Success(T),
    Failure(T),
}

fn good<T>(x: T) -> Expected<T> {
    Expected::Success(x)
}

fn bad<T>(x: T, _reason: &str) -> Expected<T> {
    Expected::Failure(x)
}

fn assert_rounding_cases<T, R>(space: &R, mode: impl Into<Mode>, tests: &[(T, Expected<T>)])
where
    R: Roundable<Point = T>,
    T: Zero + Ord + Debug,
    for<'any> &'any T: Sub,
    for<'any> <&'any T as Sub>::Output: Distance,
{
    let mode = mode.into();
    for (input, expected) in tests {
        match expected {
            Expected::Success(expected) => {
                assert_eq!(
                    space.round(input, mode).as_ref(),
                    Ok(expected),
                    "Rounding {input:?} with mode {mode:?} != {expected:?}",
                );
            }
            Expected::Failure(failed) => {
                let err = space.round(input, mode).unwrap_err();
                assert!(
                    matches!(err, RoundError::InvalidDirection {
                    ref rounded,
                    direction,
                } if rounded == failed && Mode::Directed(direction) == mode),
                    "Rounding {input:?} with mode {mode:?} should fail with {failed:?}, got {err:?}"
                );
            }
        }
    }
}

#[test]
fn directed_zero_rounding_with_excluded_zero_always_choose_upper_on_tie() {
    let space = LinearSpace::try_bounded(interval!([-10, 10]), 4).unwrap();

    assert_rounding_cases(
        &space,
        DirectedMode::TowardZero,
        &[
            (-10, good(-10)),
            (-9, good(-6)),
            (-5, good(-2)),
            (-2, good(-2)),
            (-1, bad(-2, "goes away from 0")), // tie: choose the one closer to original point
            (0, bad(2, "goes away from 0")),   // tie: choose upper as a last resort tie breaking
            (1, bad(2, "goes away from 0")),   // tie: choose the one closer to original point
            (2, good(2)),
            (5, good(2)),
            (7, good(6)),
            (10, good(10)),
        ],
    );

    assert_rounding_cases(
        &space,
        DirectedMode::AwayFromZero,
        &[
            (-10, good(-10)),
            (-9, good(-10)),
            (-5, good(-6)),
            (-2, good(-2)),
            (-1, good(-2)), // tie: choose the one closer to original point
            (0, good(2)),   // tie: choose upper as a last resort tie breaking
            (1, good(2)),   // tie: choose the one closer to original point
            (2, good(2)),
            (5, good(6)),
            (7, good(10)),
            (10, good(10)),
        ],
    );
}

#[test]
fn directed_zero_rounding_with_odd_step() {
    let space = LinearSpace::try_bounded(interval!([-10, 10]), 3).unwrap();

    assert_rounding_cases(
        &space,
        DirectedMode::TowardZero,
        &[
            (-10, good(-10)),
            (-9, good(-7)),
            (-5, good(-4)),
            (-2, good(-1)),
            (-1, good(-1)),
            (0, bad(-1, "goes away from 0")), // choose the one closer to zero
            (1, bad(-1, "goes past and away from 0")), // choose the one closer to zero
            (2, good(2)),
            (5, good(5)),
            (7, good(5)),
            (10, good(8)),
        ],
    );

    assert_rounding_cases(
        &space,
        DirectedMode::AwayFromZero,
        &[
            (-10, good(-10)),
            (-9, good(-10)),
            (-5, good(-7)),
            (-2, good(-4)),
            (-1, good(-1)),
            (0, good(2)), // choose the one further from zero
            (1, good(2)),
            (2, good(2)),
            (5, good(5)),
            (7, good(8)),
            (10, bad(8, "goes toward 0")),
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
                (-100, good(-17)),
                (-21, good(-17)),
                (-19, good(-17)),
                (-17, good(-17)),
                (-16, good(-14)),
                (-8, good(-8)),
                (-7, good(-5)),
                (-2, good(-2)),
                (-1, bad(1, "goes past and away from 0")), // between -2 and 1 select the one closer to 0
                (0, bad(1, "goes away from 0")), // between -2 and 1 select the one closer to 0
                (1, good(1)),
                (2, good(1)),
                (3, good(1)),
                (4, good(4)),
                (10, good(10)),
                (11, good(10)),
                (40, good(40)),
                (41, good(40)),
                (100, good(40)),
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
                (-100, bad(19, "goes past and away from 0")),
                (0, bad(19, "goes away from 0")),
                (17, bad(19, "goes away from 0")),
                (18, bad(19, "goes away from 0")),
                (19, good(19)),
                (20, good(19)),
                (21, good(21)),
                (99, good(99)),
                (100, good(99)),
                (101, good(101)),
                (1_000, good(999)),
                (1_001, good(1_001)),
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
                (-1_000, good(-1_000)),
                (-999, good(-900)),
                (-100, good(-100)),
                (-99, good(0)),
                (-1, good(0)),
                (0, good(0)),
                (1, good(0)),
                (99, good(0)),
                (100, good(100)),
                (901, good(900)),
                (999, good(900)),
                (1_000, good(1_000)),
                (1_234_567, good(1_234_500)),
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
                (f(-10.7), good(f(-10.0))),
                (f(-10.5), good(f(-10.0))),
                (f(-10.0), good(f(-10.0))),
                (f(-9.5), good(f(-9.0))),
                (f(-9.3), good(f(-9.0))),
                (f(-0.5), good(f(-0.0))),
                (f(0.0), good(f(0.0))),
                (f(0.5), good(f(0.0))),
                (f(5.5), good(f(5.0))),
                (f(5.9), good(f(5.0))),
                (f(10.42), good(f(10.0))),
                (f(10.5), good(f(10.0))),
                (f(13.0), good(f(13.0))),
                (f(13.5), good(f(13.0))),
                (f(14.0), good(f(14.0))),
                (f(14.5), good(f(14.0))),
                (f(15.0), good(f(15.0))),
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
                (f(-50.1), good(f(-42.0))),
                (f(-42.3), good(f(-42.0))),
                (f(-41.999), good(f(-41.5))),
                (f(-10.7), good(f(-10.5))),
                (f(-10.5), good(f(-10.5))),
                (f(-10.0), good(f(-10.0))),
                (f(-9.5), good(f(-9.5))),
                (f(-9.3), good(f(-9.0))),
                (f(-0.5), good(f(-0.5))),
                (f(0.0), good(f(0.0))),
                (f(0.5), good(f(0.5))),
                (f(5.5), good(f(5.5))),
                (f(5.9), good(f(5.5))),
                (f(10.42), good(f(10.0))),
                (f(10.5), good(f(10.5))),
                (f(13.0), good(f(13.0))),
                (f(13.5), good(f(13.5))),
                (f(14.0), good(f(13.5))),
                (f(14.5), good(f(13.5))),
                (f(15.0), good(f(13.5))),
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
                (-100, bad(-17, "goes toward 0")),
                (-21, bad(-17, "goes toward 0")),
                (-19, bad(-17, "goes toward 0")),
                (-17, good(-17)),
                (-16, good(-17)),
                (-8, good(-8)),
                (-7, good(-8)),
                (-2, good(-2)),
                (-1, good(-2)), // between -2 and 1 select the one further from 0
                (0, good(-2)),  // between -2 and 1 select the one further from 0
                (1, good(1)),
                (2, good(4)),
                (3, good(4)),
                (4, good(4)),
                (10, good(10)),
                (11, good(13)),
                (40, good(40)),
                (41, bad(40, "goes toward 0")),
                (100, bad(40, "goes toward 0")),
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
                (-100, bad(19, "goes toward and past 0")),
                (0, good(19)),
                (17, good(19)),
                (18, good(19)),
                (19, good(19)),
                (20, good(21)),
                (21, good(21)),
                (99, good(99)),
                (100, good(101)),
                (101, good(101)),
                (1_000, good(1_001)),
                (1_001, good(1_001)),
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
                (-1_000, good(-1_000)),
                (-999, good(-1_000)),
                (-100, good(-100)),
                (-99, good(-100)),
                (-1, good(-100)),
                (0, good(0)),
                (1, good(100)),
                (99, good(100)),
                (100, good(100)),
                (901, good(1_000)),
                (999, good(1_000)),
                (1_000, good(1_000)),
                (1_234_567, good(1_234_600)),
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
                (f(-10.7), good(f(-11.0))),
                (f(-10.5), good(f(-11.0))),
                (f(-10.0), good(f(-10.0))),
                (f(-9.5), good(f(-10.0))),
                (f(-9.3), good(f(-10.0))),
                (f(-0.5), good(f(-1.0))),
                (f(0.0), good(f(0.0))),
                (f(0.5), good(f(1.0))),
                (f(5.5), good(f(6.0))),
                (f(5.9), good(f(6.0))),
                (f(10.42), good(f(11.0))),
                (f(10.5), good(f(11.0))),
                (f(13.0), good(f(13.0))),
                (f(13.5), good(f(14.0))),
                (f(14.0), good(f(14.0))),
                (f(14.5), good(f(15.0))),
                (f(15.0), good(f(15.0))),
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
                (f(-50.1), bad(f(-42.0), "goes toward 0")),
                (f(-42.3), bad(f(-42.0), "goes toward 0")),
                (f(-41.999), good(f(-42.0))),
                (f(-10.7), good(f(-11.0))),
                (f(-10.5), good(f(-10.5))),
                (f(-10.0), good(f(-10.0))),
                (f(-9.5), good(f(-9.5))),
                (f(-9.3), good(f(-9.5))),
                (f(-0.5), good(f(-0.5))),
                (f(0.0), good(f(0.0))),
                (f(0.5), good(f(0.5))),
                (f(5.5), good(f(5.5))),
                (f(5.9), good(f(6.0))),
                (f(10.42), good(f(10.5))),
                (f(10.5), good(f(10.5))),
                (f(13.0), good(f(13.0))),
                (f(13.5), good(f(13.5))),
                (f(14.0), bad(f(13.5), "goes toward 0")),
                (f(14.5), bad(f(13.5), "goes toward 0")),
                (f(15.0), bad(f(13.5), "goes toward 0")),
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
                (-100, good(-17)),
                (-21, good(-17)),
                (-19, good(-17)),
                (-17, good(-17)),
                (-16, good(-14)),
                (-8, good(-8)),
                (-7, good(-5)),
                (-2, good(-2)),
                (-1, good(1)),
                (0, good(1)),
                (1, good(1)),
                (2, good(4)),
                (3, good(4)),
                (4, good(4)),
                (10, good(10)),
                (11, good(13)),
                (40, good(40)),
                (41, bad(40, "going down")),
                (100, bad(40, "going down")),
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
                (-100, good(19)),
                (0, good(19)),
                (17, good(19)),
                (18, good(19)),
                (19, good(19)),
                (20, good(21)),
                (21, good(21)),
                (99, good(99)),
                (100, good(101)),
                (101, good(101)),
                (1_000, good(1_001)),
                (1_001, good(1_001)),
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
                (-1_000, good(-1_000)),
                (-999, good(-900)),
                (-100, good(-100)),
                (-99, good(0)),
                (-1, good(0)),
                (0, good(0)),
                (1, good(100)),
                (99, good(100)),
                (100, good(100)),
                (901, good(1_000)),
                (999, good(1_000)),
                (1_000, good(1_000)),
                (1_234_567, good(1_234_600)),
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
                (f(-10.7), good(f(-10.0))),
                (f(-10.5), good(f(-10.0))),
                (f(-10.0), good(f(-10.0))),
                (f(-9.5), good(f(-9.0))),
                (f(-9.3), good(f(-9.0))),
                (f(-0.5), good(f(0.0))),
                (f(0.0), good(f(0.0))),
                (f(0.5), good(f(1.0))),
                (f(5.5), good(f(6.0))),
                (f(5.9), good(f(6.0))),
                (f(10.42), good(f(11.0))),
                (f(10.5), good(f(11.0))),
                (f(13.0), good(f(13.0))),
                (f(13.5), good(f(14.0))),
                (f(14.0), good(f(14.0))),
                (f(14.5), good(f(15.0))),
                (f(15.0), good(f(15.0))),
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
                (f(-50.1), good(f(-42.0))),
                (f(-42.3), good(f(-42.0))),
                (f(-41.999), good(f(-41.5))),
                (f(-10.7), good(f(-10.5))),
                (f(-10.5), good(f(-10.5))),
                (f(-10.0), good(f(-10.0))),
                (f(-9.5), good(f(-9.5))),
                (f(-9.3), good(f(-9.0))),
                (f(-0.5), good(f(-0.5))),
                (f(0.0), good(f(0.0))),
                (f(0.5), good(f(0.5))),
                (f(5.5), good(f(5.5))),
                (f(5.9), good(f(6.0))),
                (f(10.42), good(f(10.5))),
                (f(10.5), good(f(10.5))),
                (f(13.0), good(f(13.0))),
                (f(13.5), good(f(13.5))),
                (f(14.0), bad(f(13.5), "going down")),
                (f(14.5), bad(f(13.5), "going down")),
                (f(15.0), bad(f(13.5), "going down")),
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
                (-100, bad(-17, "going up")),
                (-21, bad(-17, "going up")),
                (-19, bad(-17, "going up")),
                (-17, good(-17)),
                (-16, good(-17)),
                (-8, good(-8)),
                (-7, good(-8)),
                (-2, good(-2)),
                (-1, good(-2)),
                (0, good(-2)),
                (1, good(1)),
                (2, good(1)),
                (3, good(1)),
                (4, good(4)),
                (10, good(10)),
                (11, good(10)),
                (40, good(40)),
                (41, good(40)),
                (100, good(40)),
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
                (-100, bad(19, "going up")),
                (0, bad(19, "going up")),
                (17, bad(19, "going up")),
                (18, bad(19, "going up")),
                (19, good(19)),
                (20, good(19)),
                (21, good(21)),
                (99, good(99)),
                (100, good(99)),
                (101, good(101)),
                (1_000, good(999)),
                (1_001, good(1_001)),
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
                (-1_000, good(-1_000)),
                (-999, good(-1_000)),
                (-100, good(-100)),
                (-99, good(-100)),
                (-1, good(-100)),
                (0, good(0)),
                (1, good(0)),
                (99, good(0)),
                (100, good(100)),
                (901, good(900)),
                (999, good(900)),
                (1_000, good(1_000)),
                (1_234_567, good(1_234_500)),
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
                (f(-10.7), good(f(-11.0))),
                (f(-10.5), good(f(-11.0))),
                (f(-10.0), good(f(-10.0))),
                (f(-9.5), good(f(-10.0))),
                (f(-9.3), good(f(-10.0))),
                (f(-0.5), good(f(-1.0))),
                (f(0.0), good(f(0.0))),
                (f(0.5), good(f(0.0))),
                (f(5.5), good(f(5.0))),
                (f(5.9), good(f(5.0))),
                (f(10.42), good(f(10.0))),
                (f(10.5), good(f(10.0))),
                (f(13.0), good(f(13.0))),
                (f(13.5), good(f(13.0))),
                (f(14.0), good(f(14.0))),
                (f(14.5), good(f(14.0))),
                (f(15.0), good(f(15.0))),
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
                (f(-50.1), bad(f(-42.0), "going up")),
                (f(-42.3), bad(f(-42.0), "going up")),
                (f(-41.999), good(f(-42.0))),
                (f(-10.7), good(f(-11.0))),
                (f(-10.5), good(f(-10.5))),
                (f(-10.0), good(f(-10.0))),
                (f(-9.5), good(f(-9.5))),
                (f(-9.3), good(f(-9.5))),
                (f(-0.5), good(f(-0.5))),
                (f(0.0), good(f(0.0))),
                (f(0.5), good(f(0.5))),
                (f(5.5), good(f(5.5))),
                (f(5.9), good(f(5.5))),
                (f(10.42), good(f(10.0))),
                (f(10.5), good(f(10.5))),
                (f(13.0), good(f(13.0))),
                (f(13.5), good(f(13.5))),
                (f(14.0), good(f(13.5))),
                (f(14.5), good(f(13.5))),
                (f(15.0), good(f(13.5))),
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
                    (-100, good(-17)),
                    (-21, good(-17)),
                    (-19, good(-17)),
                    (-17, good(-17)),
                    (-16, good(-17)),
                    (-8, good(-8)),
                    (-7, good(-8)),
                    (-2, good(-2)),
                    (-1, good(-2)),
                    (0, good(1)),
                    (1, good(1)),
                    (2, good(1)),
                    (3, good(4)),
                    (4, good(4)),
                    (10, good(10)),
                    (11, good(10)),
                    (40, good(40)),
                    (41, good(40)),
                    (100, good(40)),
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
                    (-100, good(19)),
                    (0, good(19)),
                    (17, good(19)),
                    (18, good(19)),
                    (19, good(19)),
                    (20, good(19)),
                    (21, good(21)),
                    (99, good(99)),
                    (100, good(99)),
                    (101, good(101)),
                    (1_000, good(999)),
                    (1_001, good(1_001)),
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
                    (-100, good(19)),
                    (0, good(19)),
                    (17, good(19)),
                    (18, good(19)),
                    (19, good(19)),
                    (20, good(21)),
                    (21, good(21)),
                    (99, good(99)),
                    (100, good(101)),
                    (101, good(101)),
                    (1_000, good(1_001)),
                    (1_001, good(1_001)),
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
                    (-1_000, good(-1_000)),
                    (-999, good(-1_000)),
                    (-100, good(-100)),
                    (-99, good(-100)),
                    (-1, good(0)),
                    (0, good(0)),
                    (1, good(0)),
                    (99, good(100)),
                    (100, good(100)),
                    (901, good(900)),
                    (999, good(1_000)),
                    (1_000, good(1_000)),
                    (1_234_567, good(1_234_600)),
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
                (f(-10.7), good(f(-11.0))),
                (f(-10.5), good(f(-10.0))),
                (f(-10.0), good(f(-10.0))),
                (f(-9.5), good(f(-9.0))),
                (f(-9.3), good(f(-9.0))),
                (f(-0.5), good(f(-0.0))),
                (f(0.0), good(f(0.0))),
                (f(0.5), good(f(0.0))),
                (f(5.5), good(f(5.0))),
                (f(5.9), good(f(6.0))),
                (f(10.42), good(f(10.0))),
                (f(10.5), good(f(10.0))),
                (f(13.0), good(f(13.0))),
                (f(13.5), good(f(13.0))),
                (f(14.0), good(f(14.0))),
                (f(14.5), good(f(14.0))),
                (f(15.0), good(f(15.0))),
            ],
        );

        let mode = Mode::Nearest(DirectedMode::AwayFromZero.into());
        assert_rounding_cases(
            &space,
            mode,
            &[
                (f(-10.7), good(f(-11.0))),
                (f(-10.5), good(f(-11.0))),
                (f(-10.0), good(f(-10.0))),
                (f(-9.5), good(f(-10.0))),
                (f(-9.3), good(f(-9.0))),
                (f(-0.5), good(f(-1.0))),
                (f(0.0), good(f(0.0))),
                (f(0.5), good(f(1.0))),
                (f(5.5), good(f(6.0))),
                (f(5.9), good(f(6.0))),
                (f(10.42), good(f(10.0))),
                (f(10.5), good(f(11.0))),
                (f(13.0), good(f(13.0))),
                (f(13.5), good(f(14.0))),
                (f(14.0), good(f(14.0))),
                (f(14.5), good(f(15.0))),
                (f(15.0), good(f(15.0))),
            ],
        );

        let mode = Mode::Nearest(DirectedMode::TowardPositiveInfinity.into());
        assert_rounding_cases(
            &space,
            mode,
            &[
                (f(-10.7), good(f(-11.0))),
                (f(-10.5), good(f(-10.0))),
                (f(-10.0), good(f(-10.0))),
                (f(-9.5), good(f(-9.0))),
                (f(-9.3), good(f(-9.0))),
                (f(-0.5), good(f(-0.0))),
                (f(0.0), good(f(0.0))),
                (f(0.5), good(f(1.0))),
                (f(5.5), good(f(6.0))),
                (f(5.9), good(f(6.0))),
                (f(10.42), good(f(10.0))),
                (f(10.5), good(f(11.0))),
                (f(13.0), good(f(13.0))),
                (f(13.5), good(f(14.0))),
                (f(14.0), good(f(14.0))),
                (f(14.5), good(f(15.0))),
                (f(15.0), good(f(15.0))),
            ],
        );

        let mode = Mode::Nearest(DirectedMode::TowardNegativeInfinity.into());
        assert_rounding_cases(
            &space,
            mode,
            &[
                (f(-10.7), good(f(-11.0))),
                (f(-10.5), good(f(-11.0))),
                (f(-10.0), good(f(-10.0))),
                (f(-9.5), good(f(-10.0))),
                (f(-9.3), good(f(-9.0))),
                (f(-0.5), good(f(-1.0))),
                (f(0.0), good(f(0.0))),
                (f(0.5), good(f(0.0))),
                (f(5.5), good(f(5.0))),
                (f(5.9), good(f(6.0))),
                (f(10.42), good(f(10.0))),
                (f(10.5), good(f(10.0))),
                (f(13.0), good(f(13.0))),
                (f(13.5), good(f(13.0))),
                (f(14.0), good(f(14.0))),
                (f(14.5), good(f(14.0))),
                (f(15.0), good(f(15.0))),
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
                    (f(-50.1), good(f(-42.0))),
                    (f(-42.3), good(f(-42.0))),
                    (f(-41.999), good(f(-42.0))),
                    (f(-10.7), good(f(-10.5))),
                    (f(-10.5), good(f(-10.5))),
                    (f(-10.0), good(f(-10.0))),
                    (f(-9.5), good(f(-9.5))),
                    (f(-9.3), good(f(-9.5))),
                    (f(-0.5), good(f(-0.5))),
                    (f(0.0), good(f(0.0))),
                    (f(0.5), good(f(0.5))),
                    (f(5.5), good(f(5.5))),
                    (f(5.9), good(f(6.0))),
                    (f(10.42), good(f(10.5))),
                    (f(10.5), good(f(10.5))),
                    (f(13.0), good(f(13.0))),
                    (f(13.5), good(f(13.5))),
                    (f(14.0), good(f(13.5))),
                    (f(14.5), good(f(13.5))),
                    (f(15.0), good(f(13.5))),
                ],
            );
        }
    }
}

#[cfg(feature = "random")]
mod random_rounds {
    extern crate std;
    use std::collections::HashMap;

    use ::rand::{rngs::SmallRng, SeedableRng as _};

    use super::*;

    #[test]
    fn equal_probability_for_tie() {
        let space = half_unbounded_odd();
        let mode = Mode::Nearest(TieBreakingMode::Random {
            prob_upper: Probability::default(),
        });

        let x = 100;
        let n = 1_000;

        // random fixed seed to preserve test results between identical runs
        let mut rng = SmallRng::seed_from_u64(13_015_868_539_724_329_586);
        let results = (0..n).map(|_| {
            let rng: &mut dyn RandRng = &mut rng;
            space.round_with_rng(&x, mode, rng).unwrap()
        });

        let mut distrib = HashMap::new();
        for res in results {
            *distrib.entry(res).or_insert(0_u16) += 1;
        }

        let f_lower = distrib.remove(&(x - 1)).unwrap();
        let f_upper = distrib.remove(&(x + 1)).unwrap();
        assert!(distrib.is_empty());

        let max_allowed_deviation = n / 20; // ~sqrt(n)
        let expected_interval = (n / 2 - max_allowed_deviation)..(n / 2 + max_allowed_deviation);
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
    /// This test uses the global implicit RNG (`DEFAULT_RNG`) to perform
    /// genuinely stochastic rounding.
    ///
    /// Other tests in this module may call rounding with `rng = None` as well,
    /// but only in deterministic configurations (probabilities 0 or 1), so
    /// their results do not depend on the RNG state.
    ///
    /// For stochastic behavior like in this test, you have to ensure this is
    /// the only test that relies on the shared `DEFAULT_RNG`; otherwise the
    /// RNG (shared between threads) can advance its state non-deterministically
    /// (from the point of view of a single thread) and test results may become
    /// order-dependent, causing intermittent assertion failures.
    fn global_rng_works() {
        let space = half_unbounded_odd();
        let prob_upper = 0.4;
        let mode = Mode::Nearest(TieBreakingMode::Random {
            prob_upper: Probability::new(prob_upper),
        });

        let x = 100;
        let n = 1_000;

        let results = (0..n).map(|_| space.round_with_rng(&x, mode, None).unwrap());

        let mut distrib = HashMap::new();
        for res in results {
            *distrib.entry(res).or_insert(0_u16) += 1;
        }

        let f_lower = distrib.remove(&(x - 1)).unwrap();
        let f_upper = distrib.remove(&(x + 1)).unwrap();
        assert!(distrib.is_empty());

        let (expected_lower, expected_upper) = {
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss
            )]
            (
                (f64::from(n) * (1.0 - prob_upper)) as u16,
                (f64::from(n) * prob_upper) as u16,
            )
        };

        let max_allowed_deviation = n / 20; // ~sqrt(n)
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

    #[test]
    fn stochastic() {
        let space = hundreds();
        let mode = Mode::Stochastic;

        let prec = 100;
        let x = 578;
        let n = 8_000;

        // random fixed seed to preserve test results between identical runs
        let mut rng = SmallRng::seed_from_u64(17_353_928_030_973_914_206);
        let results = (0..n).map(|_| {
            let rng: &mut dyn RandRng = &mut rng;
            space.round_with_rng(&x, mode, rng).unwrap()
        });

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

    #[test]
    fn nearest_mode_with_determined_tie_breaking() {
        let n = 1_000;

        let mode_always_upper = Mode::Nearest(TieBreakingMode::Random {
            prob_upper: Probability::new(1.0),
        });
        let mode_always_lower = Mode::Nearest(TieBreakingMode::Random {
            prob_upper: Probability::new(0.0),
        });
        for _ in 0..n {
            assert_eq!(
                mode_always_upper
                    .round(&105, (100, 110).into(), None)
                    .unwrap(),
                110
            );
            assert_eq!(
                mode_always_lower
                    .round(&105, (100, 110).into(), None)
                    .unwrap(),
                100
            );
        }
    }

    #[test]
    fn stochastic_with_equal_to_either_bound() {
        // integer values are represented exactly in floating point, so no precision issues here
        #![allow(clippy::float_cmp)]
        let n = 1_000;

        let mode = Mode::Stochastic;
        for _ in 0..n {
            assert_eq!(mode.round(&10.0, (9.0, 10.0).into(), None).unwrap(), 10.0);
            assert_eq!(mode.round(&9.0, (9.0, 10.0).into(), None).unwrap(), 9.0);
            assert_eq!(mode.round(&10.0, (10.0, 10.0).into(), None).unwrap(), 10.0);
        }
    }
}
