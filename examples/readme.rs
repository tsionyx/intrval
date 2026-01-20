//! The example below demonstrates the basic usage of the `intrval` crate.
#![allow(clippy::unwrap_used)]

use core::cmp::Ordering;

use intrval::{interval, Interval, SetOps as _, Size};

#[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
fn main() {
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

    // ===== common functions =====
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

    // ===== scalar arithmetic =====
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

    // ===== interval arithmetic =====
    let i0 = interval!(0: i32);
    let igt10 = interval!(> 10);
    let i_2to10_incl = interval!([-2, 10]);
    let i5to20_excl = interval!((5, =20));
    let iuni = interval!(U: i32);

    assert_eq!(igt10 + i_2to10_incl, interval!(> 8));
    // adding degenerate does not change the normal one
    assert_eq!(igt10 + interval!((1, 0)), igt10);
    assert_eq!(interval!((1, 0)) + igt10, igt10);

    assert_eq!(i_2to10_incl - i5to20_excl, interval!((=-22, 5)));
    // subtracting degenerate does not change the normal one
    assert_eq!(igt10 - interval!((1, 0)), igt10);
    // subtracting _from_ degenerate negates the normal one
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

    // ===== set operations =====
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
