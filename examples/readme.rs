//! The example below demonstrates the basic usage of the `intrval` crate.
#![allow(clippy::unwrap_used)]

fn main() {
    macro_syntax();
    common_functions();
    scalar_arithmetic();
    interval_arithmetic();
    set_operations();

    discrete_constructors_and_getters();
    discrete_arithmetic_operations();
    discrete_iterators();
    discrete_rounding();
}

/// Allows to simplify the definitions of an interval
/// in common inequality and ranges terms.
///
/// For the half-open intervals, the inclusive bound is marked with `=` symbol;
/// for the closed interval both `[a, b]` and `(=a, =b)` definitions are possible.
fn macro_syntax() {
    use intrval::{interval, Interval};

    let i0: Interval<i16> = interval!(0);
    assert_eq!(i0, Interval::Empty);

    let igt2: Interval<i16> = interval!(> 2);
    assert_eq!(igt2, Interval::GreaterThan(2));

    let igt10: Interval<i16> = interval!(> 10);
    assert_eq!(igt10, Interval::GreaterThan(10));

    let i_2to10_incl: Interval<i16> = interval!([-2, 10]);
    assert_eq!(i_2to10_incl, Interval::Closed((-2, 10)));

    let i5to20_excl: Interval<i16> = interval!((5, =20));
    assert_eq!(i5to20_excl, Interval::LeftOpen((5, 20)));

    let iuni: Interval<i16> = interval!(U);
    assert_eq!(iuni, Interval::Full);
}

/// Functions not falling into _arithmetic_ or _set_ categories,
/// but still common for all intervals.
fn common_functions() {
    use core::cmp::Ordering;
    use intrval::{interval, Size};

    assert!(interval!(0: i32).is_empty());
    assert!(interval!((1, 0)).is_empty());
    assert!(interval!((0, 0)).is_empty());
    assert!(!(interval!([0, 0]).is_empty()));
    assert!(interval!(U: i32).is_full());

    let igt10 = interval!(> 10);
    assert!(igt10.contains(&11));
    assert!(!(igt10.contains(&10)));
    assert!(!(interval!(0).contains(&0)));

    assert_eq!(interval!(0: i8).len(), Size::Empty);
    assert_eq!(interval!((1, 0)).len(), Size::Empty);
    assert_eq!(interval!([1, 1]).len(), Size::SinglePoint);
    assert_eq!(interval!((-10, =10)).len(), Size::Finite(20));
    assert_eq!(interval!(> 2).len(), Size::Infinite);

    let i_left_open = interval!((2, =5));
    assert_eq!(i_left_open.clamp(10).unwrap(), (Ordering::Equal, 5));
    assert_eq!(i_left_open.clamp(3).unwrap(), (Ordering::Equal, 3));
    assert_eq!(i_left_open.clamp(2).unwrap(), (Ordering::Greater, 2));
    assert_eq!(i_left_open.clamp(0).unwrap(), (Ordering::Greater, 2));
}

/// Add, subtract or multiply the interval bounds with a scalar value of type `U`
/// if the underlying type `T: {Add, Sub, Mul}<U>`.
fn scalar_arithmetic() {
    use intrval::{interval, Interval};

    // negation changes the sign and flips the bounds
    assert_eq!(-interval!(> 2), interval!(< -2));
    assert_eq!(-interval!([-2, 10]), interval!([-10, 2]));

    // full and empy does not change with scalars
    assert_eq!(interval!(0) + 5, interval!(0));
    assert_eq!(interval!(0: i32) * 2, interval!(0));
    assert_eq!(interval!(U) - 100, interval!(U));
    assert_eq!(interval!(U: i32) * -5, interval!(U));
    // however, multiplying by 0 is different
    #[allow(clippy::erasing_op)]
    {
        assert_eq!(interval!(U: i32) * 0, interval!([0, 0]));
    };

    assert_eq!(interval!(> 2) + 3, interval!(> 5));
    assert_eq!(interval!(> 10) - 5, interval!(> 5));
    assert_eq!(interval!(> 2) * 5, interval!(> 10));
    assert_eq!(interval!((5, =20)) / 5, Interval::LeftOpen((1, 4)));
    // multiplying/dividing by negative flips the bounds
    assert_eq!(interval!([-2, 10]) * -4, interval!([-40, 8]));
    assert_eq!(interval!([16, 79]) / -8, Interval::Closed((-9, -2)));
}

/// Add, subtract or multiply an `Interval<T>` with an `Interval<U>`
/// to produce another `Interval<Z>`
/// if the underlying type `T: {Add, Sub, Mul}<U, Output=Z>`.
fn interval_arithmetic() {
    use intrval::interval;

    let i0 = interval!(0: i32);
    let igt10 = interval!(> 10);
    let i_2to10_incl = interval!([-2, 10]);
    let i5to20_excl = interval!((5, =20));
    let iuni = interval!(U: i32);

    assert_eq!(igt10 + i_2to10_incl, interval!(> 8));
    // adding an empty interval one does not change the proper one
    assert_eq!(igt10 + interval!((1, 0)), igt10);
    assert_eq!(interval!((1, 0)) + igt10, igt10);

    assert_eq!(i_2to10_incl - i5to20_excl, interval!((=-22, 5)));
    // subtracting an empty interval does not change the proper one
    assert_eq!(igt10 - interval!((1, 0)), igt10);
    // subtracting _from_ an empty interval negates the proper one
    assert_eq!(interval!((2, 0)) - i_2to10_incl, -i_2to10_incl);

    // Interval::Empty is neutral over multiplication
    assert_eq!(i0 * i_2to10_incl, i0);
    // positive (+inf) times positive is positive
    assert_eq!(interval!(> 2) * igt10, interval!(> 20));
    // positive (+inf) times (negative and positive) is (-inf, +inf)
    assert_eq!(igt10 * i_2to10_incl, interval!(U));
    assert_eq!(igt10 * i5to20_excl, interval!(> 50));
    // Interval::Full is neutral over multiplication
    assert_eq!(i5to20_excl * iuni, iuni);
}

/// Representation of an `Interval`-s as a sets of points
/// allows to apply various set operations on them.
fn set_operations() {
    use intrval::{interval, SetOps as _};

    let igt2 = interval!(> 2);
    let igt_e2 = interval!(>= 2);
    let igt10 = interval!(> 10);

    assert!(igt_e2.is_super(&igt2));
    assert!(igt2.is_sub(&igt_e2));
    assert!(!igt2.is_super(&igt_e2));
    assert!(!igt_e2.is_sub(&igt2));
    assert!(igt2.is_super(&igt10));
    assert!(igt10.is_sub(&igt2));
    assert!(igt2.is_disjoint(&interval!(< 0)));
    assert!(igt2.is_disjoint(&interval!(< 2)));
    assert!(!igt2.is_disjoint(&interval!(<= 2)));

    assert_eq!(igt2.complement().into_single().unwrap(), interval!(<= 2));
    // `.complement` is aliased with `!`
    assert_eq!(
        (!interval!([-2, 10])).into_pair().unwrap(),
        (interval!(< -2), igt10)
    );

    assert_eq!(
        igt2.difference(igt10).unwrap().into_single().unwrap(),
        interval!((2, =10)),
    );
    assert_eq!(
        igt2.symmetric_difference(interval!(<= 5))
            .unwrap()
            .into_pair()
            .unwrap(),
        (interval!(<= 2), interval!(> 5)),
    );
    // `.symmetric_difference` is aliased with `^` (falling back to Interval::Empty)
    assert_eq!(
        (interval!(<= 5) ^ interval!((0, 5))).into_pair().unwrap(),
        (interval!(<= 0), interval!(== 5)),
    );

    assert_eq!(igt2.intersect(igt10).unwrap(), igt10);
    // `.intersect` is aliased with `&` (falling back to Interval::Empty)
    assert!((igt10 & interval!([-2, 10])).is_empty());
    assert_eq!(
        interval!([-2, 10]) & interval!((5, =20)),
        interval!((5, =10))
    );

    assert_eq!(
        igt10.union(interval!([-2, 10])).into_single().unwrap(),
        interval!(>= -2)
    );
    // `.union` is aliased with `|`
    assert_eq!(
        (interval!([-2, 10]) | igt2).into_single().unwrap(),
        interval!(>= -2)
    );
    assert_eq!(
        (interval!((5, 10)) | interval!([3, 4]))
            .into_pair()
            .unwrap(),
        // reorders the input intervals in left-to-right order if they do not intersect
        (interval!([3, 4]), interval!((5, 10)))
    );
    assert_eq!((interval!(0) | igt2).into_single().unwrap(), igt2);
    assert_eq!((igt2 | interval!(0)).into_single().unwrap(), igt2);

    assert_eq!(interval!([-2, 10]).enclosure(igt10 * 2), interval!(>= -2));
}

/// Create and deconstruct discrete intervals.
fn discrete_constructors_and_getters() {
    use core::convert::identity;
    use intrval::{discrete::LinearSpace, interval};

    let space = LinearSpace::try_bounded(interval!([-2, 10]), 3_i8).unwrap();
    assert_eq!(space.bounds(), &interval!([-2, 10]));
    assert_eq!(space.step(), &3);
    assert_eq!(space.into_parts(), (interval!([-2, 10]), 3));
    assert_eq!(
        space.map(|x| x * 2, identity).unwrap().into_parts(),
        (interval!([-4, 20]), 3)
    );

    let space = LinearSpace::<i8, i8>::try_new(10).unwrap();
    assert_eq!(space.bounds(), &interval!(U));
    assert_eq!(space.step(), &10);
    assert_eq!(space.into_parts(), (interval!(U), 10));
    assert_eq!(
        space.map(|x| x + 2, |x| x + 2).unwrap().into_parts(),
        (interval!(U), 12)
    );
    assert!(space.map(identity, |x| x - 10).is_none());
}

/// Modify the space by doing some simple operations
/// using underlying `Interval` implementations.
fn discrete_arithmetic_operations() {
    use intrval::{discrete::LinearSpace, interval};

    let space = LinearSpace::try_bounded(interval!((20, =142)), 10_u16).unwrap();

    // `Shl` and `Shr` (forwarded to underlying `Interval`, without changing the `step`)
    assert_eq!((space >> 5).into_parts(), (interval!((25, =147)), 10));
    assert_eq!((space << 20).into_parts(), (interval!((0, =122)), 10));

    // `Mul`/`Div` by scalar to extend/shrink the `Interval` along with the `step`
    assert_eq!(
        (space * 10).unwrap().into_parts(),
        (interval!((200, =1420)), 100)
    );
    assert_eq!((space / 3).unwrap().into_parts(), (interval!((6, =47)), 3));

    // `Mul` by another `LinearSpace`
    let space2 = LinearSpace::try_bounded(interval!([5, 10]), 3).unwrap();
    assert_eq!((space * space2).into_parts(), (interval!((100, =1420)), 30));
}

/// Convert a space into forward and backward iterators.
fn discrete_iterators() {
    use intrval::{discrete::LinearSpace, interval};

    let space = LinearSpace::try_bounded(interval!((20_u16, =80)), 10).unwrap();
    assert_eq!(
        space.try_into_forward_iter().unwrap().collect::<Vec<_>>(),
        [30, 40, 50, 60, 70, 80]
    );
    assert_eq!(
        space.into_forward_iter_from(65).collect::<Vec<_>>(),
        [70, 80]
    );

    let space_unbounded_lower = LinearSpace::try_bounded(interval!(<= 20_u8), 5).unwrap();
    let _err = space_unbounded_lower.try_into_forward_iter().unwrap_err();
    assert_eq!(
        space_unbounded_lower
            .into_forward_iter_from(8)
            .collect::<Vec<_>>(),
        [10, 15, 20]
    );

    assert_eq!(
        space.try_into_backward_iter().unwrap().collect::<Vec<_>>(),
        [80, 70, 60, 50, 40, 30]
    );
    assert_eq!(
        space.into_backward_iter_up_to(42).collect::<Vec<_>>(),
        [40, 30]
    );

    let space_unbounded_upper = LinearSpace::try_bounded(interval!(> 100_u8), 5).unwrap();
    let _err = space_unbounded_upper.try_into_backward_iter().unwrap_err();
    assert_eq!(
        space_unbounded_upper
            .into_backward_iter_up_to(127)
            .collect::<Vec<_>>(),
        [125, 120, 115, 110, 105]
    );
}

/// Round (using `Roundable` trait) a `point: T` to the values of the space.
fn discrete_rounding() {
    use intrval::{
        discrete::{
            rounding::{DirectedMode, NearestMode, RoundError, Roundable as _},
            LinearSpace,
        },
        interval,
    };

    let space = LinearSpace::try_bounded(interval!(> 100_u8), 4).unwrap();

    assert_eq!(space.round(&102, DirectedMode::UP).unwrap(), 104);
    assert_eq!(space.round(&117, DirectedMode::DOWN).unwrap(), 116);
    assert_eq!(
        space.round(&253, DirectedMode::AwayFromZero).unwrap_err(),
        RoundError::InvalidDirection {
            rounded: 252,
            direction: DirectedMode::AwayFromZero
        }
    );
    assert_eq!(
        space.round(&142, NearestMode(DirectedMode::DOWN)).unwrap(),
        140,
    );

    #[cfg(feature = "random")]
    {
        use intrval::discrete::rounding::StochasticMode;
        let rounded = space
            .round_with_rng(
                &141,
                StochasticMode,
                // you should provide an optional RNG for the stable results,
                // see the caveats of using fallback RNG in docs for `round_with_rng` method.
                None,
            )
            .unwrap();
        assert!([140, 144].contains(&rounded));
    }
}
