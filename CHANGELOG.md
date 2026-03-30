# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# [0.2.0] - 2026-03-30

## Added

### feature="std"

Main features:
- impls of numeric traits for `std::time::SystemTime` enables rounding for it;
- using `{f32,f64}::trunc()` instead of manual implementation;
- using `thread_local!(RefCell)` for default RNG (with `random` feature).

More:
- unify `StdError` for all configurations
  (either from `core` or from `std`, depending on Rust version);
- reduce the number of property test cases in CI to 1000
  (one more feature increases the power set of features twice);

### rounding

- `discrete::LinearRoundable` as an extension trait
  for any _roundable_ `MonotonicMeasure` type.
  It allows for inline creation of `LinearSpace` with no bounds
  and round the value with it;
- round `SystemTime` using `Duration` as a _distance_;
- impl necessary rounding traits for `rust_decimal::Decimal`
  to showcase in tests;


## Changed

### LinearSpace

- make it generic over `T` (for interval points) and `D` (for distance between points);
  - `LinearSpace::map` now accepts two separate mapping functions for `T` and `D`;
- additional constraint `T: Zero` for `impl DiscreteOrdSet` to find origin point during rounding
  (the `step` value was used before, as its type was the same `T`);
- non-fallible constructors with `NonZero` (statically guaranteed positive)
  step value as a wrapper around type `D`.


Internal changes:
- use `MonotonicMeasure::origin()` instead of `x - x` to find a rounded value of _zero_ for the type
  later used as an anchor;
- use the commutativeness of `checked_diff` to simplify `find_stepped` function;
- use getter functions instead of direct fields' referencing;


### Rounding supporting traits

- rename:
  - `Roundable` -> `Rounding`;
  - `Measure` -> `Metric`;
  - `IntDiv` -> `LinearIntRatio`;
  - `MonotonicLinear` -> `MonotonicMeasure`;

- simplify `impl RoundingMode` by using `Metric::distance`
  instead of `for<'any> &'any T: ops::Sub`;

- get rid of blanket impls (depending on some ops)
  for `Zero`, `Linear`, `LinearIntRatio`;
  - explicit methods `Linear::{mul_scalar(), get_ratio()}`;
  - use (exported) macros instead to impl those traits for core numeric types;

- new methods:
  - `LinearIntRatio::quantize` combining `LinearIntRatio::int_ratio`
    and `Linear::mul_scalar` to round a value to an integer number of steps;

  - `MonotonicMeasure`:
    - separate `monotonic_sub()` and `checked_diff()` (commutative one);
    - `origin()` to optionally shortcut rounding;

  - `RoundError::fit` to use a rounded value even if it contradicts
    the selected rounding direction;

## Fixed

- `impl Mul for LinearSpace` outputs an `Option<Self>`
  because the product of steps could become non-positive;


## Other

- restructured the project files;
  - move `discrete::rounding` module into top-level;
  - move `Interval` specific implementation into `interval` submodule;


# [0.1.4] - 2026-02-26

## Added

- `serde` support for the types covering _rounding modes_:
  - `DirectedMode`;
  - `NearestMode`;
  - `RandomTie`;
  - `Probability`;


# [0.1.3] - 2026-02-22

## Added

- `Interval`'s shifting operations as aliases for arithmetic operations:
  `core::ops::Shl(Shr)` based on `Sub(Add)` and can be thought as shifting
  both of interval's bounds to the left(right) along a rational line
  (i.e. subtracting(adding) some offset from(to) both bounds);

- fine-grained hierarchy of numeric traits:
  - `Zero` (blanket impl for `TryFrom<u8> + PartialOrd`);
  - `Measure` (`Add & Sub`), blanket impl for types _closed under those ops_;
    - `Linear` (+ `Mul & Div` by scalar),
      blanket impl for `Measure`-d types with `Mul` and `Div` defined;
      - `MonotonicLinear` (`Add & Sub` are guaranteed to be monotonic),
        impl for _core numeric types_
        (`checked_{add,sub}` for integers, manual check for floating);
  - `IntDiv` (`Div` with the ability to round a quotient to integer),
    impl for _core numeric types_
    (identity quotient for integers, truncated quotient for floating);


### Discrete sets

- `DiscreteOrdSet` trait to define a generic _discrete_ set of given `Point`-s
  - `LinearSpace` as the implementor of `DiscreteOrdSet` representing
    an `Interval` with a `step > 0` defined.
    Some additional behaviour also included:
    - constructors:
      - `try_bounded(bounds, step)` with the `Interval` provided;
      - `try_new(step)` with the implicit `Interval::Full` provided;
    - getters and projectors:
      - `bounds()`;
      - `step()`;
      - `into_parts()`;
      - `map(f)`;
    - arithmetic operations:
      - `Shl` and `Shr` (forwarded to underlying `Interval`,
        without changing the `step`);
      - `Mul/Div` by scalar to extend/shrink the `Interval` along with the `step`;
      - `Mul` by another `LinearSpace`;
    - conversions into _forward_ and _backward_ iterators:
      - `try_into_forward_iter()`;
      - `into_forward_iter_from(start)`;
      - `try_into_backward_iter()`;
      - `into_backward_iter_up_to(end)`;

- `Roundable` as an extension of `DiscreteOrdSet` to allow
  an arbitrary point `T` to be rounded using one of the rounding modes
  to one of the discrete points defined by `DiscreteOrdSet`;
  - `random` feature to enable some _stochastic_ modes;
    - `rand` dependency with `SmallRng` as a fallback (used while
      performing rounding using one of the _stochastic_ modes provided **no _RNG_** is available);
      - `CONST_RANDOM_SEED` environment variable can be used during `cargo build`
        to set a custom seed for the default RNG;
  - `ordered-float` dependency to test rounding on `LinearSpace` with ordered floating-point numbers;


### Helpers

- `OneOrPair::{fold,single_or_fold}` to transform it into a value;
- `ValOrInf` (isomorphic to `Option`) to extend an arbitrary type with the `Infinite` notion;
- `slice_to_array_*` conversions to get an array `[T; N]` from the slice `&[T]`;
- `OnceLock`, a very basic primitive for performing mutable operations on shared global data in `no_std` mode.

  The possible problems of using a global state behind this `OnceLock` include:
  1. No fairness guarantee: A thread spinning on the lock could starve other threads indefinitely.
  2. Priority inversion: On systems with thread priorities, a low-priority thread holding the lock can block high-priority threads.
  3. Performance on single-core systems: Spinning wastes CPU cycles when only one thread can make progress.


## Changed

- `proptest` library with `default-features=false`,
  using `features = ["std"]` only for tests (in `dev-dependencies`);

- loosened the level of `unsafe_code` lint to `deny` to implement `OnceLock` (see above);



# [0.1.2] - 2026-02-10

## Fixed

- misconception about _degenerate_ intervals fixed in comments and variables.
  A _degenerate_ interval is essentially a _singleton_ interval, i.e. having `len=0` (zero measure),
  but not empty, because **there is a point that belongs to it**.

## Changed

- bump MSRV to _**1.71**_:
  - transitive dependencies are updated to include MSRV=1.71:
    - unicode-ident@1.0.23
    - zmij@1.0.20

- the syntax of [CHANGELOG](CHANGELOG.md) (this file) changed slightly to
  support the [Keep a Changelog](https://keepachangelog.com) scheme.



# [0.1.1] - 2026-01-20

## Fixed

- bump MSRV to _**1.69**_:
  - [namespaced features](https://blog.rust-lang.org/2022/04/07/Rust-1.60.0/#new-syntax-for-cargo-features)
    in _Cargo.toml_ (_**1.60**_);
  - [_sparse_ protocol](https://blog.rust-lang.org/2023/03/09/Rust-1.68.0/#cargo-s-sparse-protocol)
    for the index of crate dependencies (_**1.68**_);
  - the _clippy_'s setting
    [missing-docs-in-crate-items](https://github.com/rust-lang/rust-clippy/blob/master/CHANGELOG.md#rust-169)
    for the [missing_docs_in_private_items](https://rust-lang.github.io/rust-clippy/master/index.html#missing_docs_in_private_items)
    lint (_**1.69**_);

- `impl<T: PartialOrd> PartialEq for Interval<T>` to allow to compare
  different representations without `reduce`-ing:
  - all _empty_ intervals are equal to each other;
  - the `Singleton(x)` equals to `Closed((x, x))`;

## Changed

- split the `Bounded` trait into hierarchy of traits:
  - `IntoBounds` to define only conversion to a pair of `Endpoint`-s;
    - this reduction allows to implement it for `&Interval<T>`,
      improving ergonomics (skipping some `.as_ref()` calls);
  - `Bounded<T>`:
    - adds the `.from_bounds()`
    - removes the requirement `type Error: Into<Self>`;
  - `SetOps<T>` to define operations:
    - `.difference(self, other: impl IntoBounds)` to ignore the intersection;
    - `.symmetric_difference(self, other: impl IntoBounds)`
      to only take values from one of the spans;
    - `.intersect(self, other: impl IntoBounds)`
      (return type changed: `Result<Self, (Self, R)>` -> `Result<Self, R>`);
    - `.union(self, other: impl IntoBounds)`
      (return type changed: `Result<OneOrPair<Self>, (Self, R)>` -> `OneOrPair<Self>`)
    - `.enclosure(self, other: impl IntoBounds)`
      (return type changed: `Result<Self, (Self, R)>` -> `Self`);
    - `.is_disjoint(&self, other: impl IntoBounds<&T>)`
      to determine whether the two `Interval`-s are separated
      and could not be merged into a single one;
    - `.is_sub(&self, other: impl IntoBounds<&T>)` (reverse to `is_super`);
    - `.is_super(&self, other: impl IntoBounds<&T>)`
      (previously named `Interval::contains_other`);

- `Interval::point_cmp` to determine whether a point `T`
  lies to the right or left of the `Interval<T>`.
  The signature of the method matches the `PartialOrd::<T>::partial_cmp`,
  but implementing the latter would result in poor ergonomics
  in other parts of the codebase.

- improve the `Partial{Eq,Ord}` for `Endpoint` by enabling arbitrary `SIDE`-s to be comparable.
  This became possible thanks to the use of `ExtPoint` representing
  a (possibly infinite) point with its
  [neighbourhood](https://en.wikipedia.org/wiki/Neighbourhood_(mathematics)#Neighbourhood_of_a_point).

- change the `interval!` macro syntax to create:
  - empty `Interval::Empty` with `interval!(0)`;
  - universe `Interval::Full` with `interval!(U)`;

- set up more lints by carefully exploring the latest rustc and clippy lint groups;

## Added

- `Interval::as_ref_bounds` as a synonym for `Interval::into_bounds(Interval::as_ref)`;
- the internal `Container` trait to abstract away the `.contains` method using only `IntoBounds`;
  - the `Interval::contains` just reuses the blanket implementation of it;

## Removed

- `impl RangeBounds<T> for Interval<T>`;


# [0.1.0] - 2026-01-07

_The first release_


## Added


### Members


#### `Interval`

Represent a subset of single-dimensioned ordered set bounded by at most 2 points (left and right).

- macro `interval!` using short syntax to create `Interval`;

- `impl From<core::ops::Range*> for Interval<T>`
  to easily convert from standard ranges;

- scalar arithmetic: `Add<T>`, `Sub<T>`, `Mul<T>`, `Div<T>` for `Interval<T>`;

- interval arithmetic: 
  - unary operations:
    
    `Neg<Interval<T>> -> Interval<T> where T: Neg<Output=T>` (`-` operator)
    with bounds negated and swapped

  - binary operations:

    `Interval<T>` _\<OPERATION>_ `Interval<U>` -> `Interval<Z>`
    
    where `T` _\<OPERATION>_ `U` -> `Z`

    _\<OPERATION>_ is `{Add, Sub, Mul}`.

- set operations:
  - `complement` (aliased with `Not` (`!`) operator);
  - `contains` to determine whether the `Interval` contains a given point;
  - `contains_other` to check for full containment of another `Interval`;
  - via `Bounded` trait: 
    - `intersect` (aliased with `BitAnd` (`&`) operator);
    - `union` (aliased with `BitOr` (`|`) operator);
    - `enclosure`;

- other methods:
  - `is_empty` to determine whether the `Interval` contains no points;
  - `is_full` to determine whether the `Interval` contains all possible points (_universe_);
  - `len` to get a measure of the `Interval` (as the `Size<Diff<T>>` type);
  - `clamp` to force the given point to fall into the `Interval`;
  
  - `as_ref` to represent the borrowed version `Interval<&T>`
    (useful in many methods to avoid cloning when doing `Interval::into_bounds`);
  - `map` to convert to another `Interval<U>` given transformation function
    `Fn(T) -> (U)`;
  - `reduce` to simplify the definition of an `Interval` to the equivalent one:
    - the _empty_ reduced to the `Interval::Empty`;
    - the _singleton_ (_degenerate_) interval `[x, x]` reduced to `Interval::Singleton(x)`
      (if the _singleton_ feature enabled);

  - `into_closure` to convert into an `Interval` with closed bounds;
  - `into_interior` to convert into an `Interval` with open bounds;


#### `Endpoint`

- `impl From<core::ops::Bound<T>> for Endpoint<T>`
  to easily convert to/from standard `Bound`;

- arithmetic: 
  - unary operations:
    
    `Neg<Endpoint<SIDE, T>> -> Endpoint<SIDE, T> where T: Neg<Output=T>` (`-` operator)
    with the internal value negated (keeping the same `SIDE`);

  - binary operations:
    - summing the same-`SIDE` `Endpoint`-s:
      `Endpoint<SIDE, T> + Endpoint<SIDE, U> -> Endpoint<SIDE, Z>`
      where `T: Add<U, Output=Z>`;
    - the difference between opposite-`SIDE` `Endpoint`-s:
      `Endpoint<SIDE, T> - Endpoint<NEG_SIDE, U> -> Endpoint<SIDE, Z>`
      where `T: Sub<U, Output=Z>`;

- the `Not` (`!`) operation to represent the complementary `Endpoint`
  with its side and inclusion swapped;

- comparison traits:
  - custom `impl PartialOrd` for the same-`SIDE` `Endpoint`-s
    taking into account the notion of infinity and (in/ex)-clusion
    to properly order its instances;
  - custom `impl Partial{Eq,Ord}<T> for Endpoint<T>`
    to compare with the point (converting it to `Endpoint::Inclusive`);


#### `Bounded`

The main trait implemented by `Interval<T: PartialOrd>`
to convert it to/from a pair `(Endpoint<LEFT, T>, Endpoint<RIGHT, T>)`.
See the `Interval`'s set operations above for provided methods.


#### `Zero`

The trait representing a scalar type with the zero point.
Now it is blanket-implemented for any type `T: TryFrom<u8>`,
i.e. any primitive numeric type from core library.


### Features

All features are disabled by default:

- _serde_ for (de)serialization support;
- _arbitrary_ for the property tests support;
- _singleton_ for `Interval::Singleton` variant support;
