intrvals
========

[![Build Status](https://github.com/tsionyx/intrval/actions/workflows/rust.yml/badge.svg)](https://github.com/tsionyx/intrval/actions)
[![Latest version](https://img.shields.io/crates/v/intrval.svg)](https://crates.io/crates/intrval)
[![MSRV](https://img.shields.io/crates/msrv/intrval)](https://crates.io/crates/intrval)
[![Documentation](https://docs.rs/intrval/badge.svg)](https://docs.rs/intrval)
[![codecov](https://codecov.io/github/tsionyx/intrval/graph/badge.svg?token=6ODY9VWDZE)](https://codecov.io/github/tsionyx/intrval)


`intrvals` is a generic intervals library with basic
[arithmetic](https://en.wikipedia.org/wiki/Interval_arithmetic) (+, -, *)
and sets (_union_, _intersect_, _complement_) operations support.

Its main structure is an `Interval` which represents
a single-dimensioned (possibly unbounded) span of values instantiated by any type `T`.

## Continuous intervals

If the underlying type `T` is discrete like one of the integer types,
the `Interval` is not really _continuous_, but we still consider
it contains **every value** between the given bounds.

### macro syntax

Allows to simplify the definitions of an interval
in common inequality and ranges terms.

For the half-open intervals, the inclusive bound is marked with `=` symbol; 
for the closed interval both `[a, b]` and `(=a, =b)` definitions are possible.

```rust
# use intrval::{interval, Interval};

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
```

### common functions

Functions not falling into _arithmetic_ or _set_ categories,
but still common for all intervals.

```rust
# use core::cmp::Ordering;
# use intrval::{interval, Size};

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
```

### scalar arithmetic

Add, subtract or multiply the interval bounds with a scalar value of type `U`
if the underlying type `T: {Add, Sub, Mul}<U>`.

```rust
# use intrval::{interval, Interval};

// negation changes the sign and flips the bounds
assert_eq!(-interval!(> 2), interval!(< -2));
assert_eq!(-interval!([-2, 10]), interval!([-10, 2]));

// full and empy does not change with scalars
assert_eq!(interval!(0) + 5, interval!(0));
assert_eq!(interval!(0: i32) * 2, interval!(0));
assert_eq!(interval!(U) - 100, interval!(U));
assert_eq!(interval!(U: i32) * -5, interval!(U));
// however, multiplying by 0 is different
assert_eq!(interval!(U: i32) * 0, interval!([0, 0]));

assert_eq!(interval!(> 2) + 3, interval!(> 5));
assert_eq!(interval!(> 10) - 5, interval!(> 5));
assert_eq!(interval!(> 2) * 5, interval!(> 10));
assert_eq!(interval!((5, =20)) / 5, Interval::LeftOpen((1, 4)));
// multiplying/dividing by negative flips the bounds
assert_eq!(interval!([-2, 10]) * -4, interval!([-40, 8]));
assert_eq!(interval!([16, 79]) / -8, Interval::Closed((-9, -2)));
```

### interval arithmetic

Add, subtract or multiply an `Interval<T>` with an `Interval<U>`
to produce another `Interval<Z>`
if the underlying type `T: {Add, Sub, Mul}<U, Output=Z>`.

```rust
# use intrval::interval;

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
```

### set operations

Representation of an `Interval`-s as a sets of points
allows to apply various set operations on them.

```rust
# use intrval::{interval, SetOps as _};

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
```


## Discrete intervals

`LinearSpace` is an `Interval` combined with the notion of `step` to
represent discrete space of points.

### constructors (injectors) and getters (projectors)

Create and deconstruct discrete intervals.

```rust
# use core::convert::identity;
# use intrval::{discrete::LinearSpace, interval};

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
```

### arithmetic operations

Modify the space by doing some simple operations
using underlying `Interval` implementations.

```rust
# use intrval::{discrete::LinearSpace, interval};

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
```


### iterators

Convert a space into forward and backward iterators.

```rust
# use intrval::{discrete::LinearSpace, interval};

let space = LinearSpace::try_bounded(interval!((20_u16, =80)), 10).unwrap();
assert_eq!(
    space.try_into_forward_iter().unwrap().collect::<Vec<_>>(),
    [30, 40, 50, 60, 70, 80]
);
assert_eq!(
    space.into_forward_iter_from(65).collect::<Vec<_>>(),
    [70, 80]
);

let space_unbounded_lower = LinearSpace::try_bounded(interval!(<= 20_u16), 5).unwrap();
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
```

### rounding

Round (using `Roundable` trait) a `point: T` to the values of the space.

```rust
# use intrval::{
#     discrete::{
#         rounding::{DirectedMode, NearestMode, RoundError, Roundable as _},
#         LinearSpace,
#     },
#     interval,
# };

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
    space
        .round(&142, NearestMode(DirectedMode::DOWN))
        .unwrap(),
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
```


## Features

By default, all the features below are disabled to ensure minimalistic
no-dependency library.

### serde

Enables the support of `serde::{Serialize, Deserialize}` for `Interval<T>`.

### arbitrary

Enables the `proptest::Arbitrary` for `Interval<T>` along with the property tests
(could be invoked with `cargo test prop_test --features=arbitrary`).


### random

Enables the ability to use _stochastic_ rounding (`discrete::rounding::Mode::Stochastic`)
and _random tie-breaking_ (in the `discrete::rounding::Mode::Nearest`).

This feature also enables the `Roundable::round_with_rng`
to be able to provide a custom RNG (random number generator implementing `rand::RngCore`).
If you do not provide an RNG (either calling `Roundable::round_with_rng` with `None`
or calling `Roundable::round`) using any of the modes mentioned above,
be aware that the default very simple RNG (`rand::SmallRng`) will be used.
Its seed can be provided as a `CONST_RANDOM_SEED` environment variable **at compile time**
(during `cargo build` or `cargo run`).


### singleton

This feature is useful if you want to have a dedicated `Interval::Singleton`
variant (otherwise represented as `Interval::Closed`).
This allows to create an `Interval` from a single value
without cloning it.
However, to get the bounds of an `Interval`, you have to meet
the requirement of `T: Clone` again.

The table below summarizes these limitations for the `Interval<T>`

| use case | `singleton` feature | no `singleton` feature |
|----------|:-------------------:|:----------------------:|
| create an `Interval` from a single point <br/>(`Singleton::singleton`) | - | `T: Clone` |
| convert an `Interval` into a pair of `Endpoint`-s<br/>(`Bounded::into_bounds`) | `T: Clone` | - |

There are two helper traits used to implement the desired feature-gated behaviour:

- `trait Singleton` to create a point-sized `Interval` from a single value
  (when the feature is disabled, create an `Interval::Closed((x.clone(), x))` instead);
- `trait SingletonBounds` to convert a single value into a pair of `Endpoint`-s
  (when the feature is disabled, this trait has no methods
  and implemented for an `Interval<T>` unconditionally).
