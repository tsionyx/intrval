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
