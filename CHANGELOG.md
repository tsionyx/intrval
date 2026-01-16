This is a list of all notable changes.

# Unreleased

_2026-01-16_

## Fixed

- bump MSRV to _**1.68**_:
  - [namespaced features](https://blog.rust-lang.org/2022/04/07/Rust-1.60.0/#new-syntax-for-cargo-features)
    in _Cargo.toml_ (_**1.60**_);
  - [_sparse_ protocol](https://blog.rust-lang.org/2023/03/09/Rust-1.68.0/#cargo-s-sparse-protocol)
    for the index of crate dependencies (_**1.68**_);

- `impl<T: PartialOrd> PartialEq for Interval<T>` to allow to compare
  different representations without `reduce`-ing:
  - all _degenerate_ (empty) intervals are equal to each other;
  - the `Singleton(x)` equals to `Closed((x, x))`;

## Changed

- `Interval::point_cmp` to answer whether a point `T`
  lies to the right/left of the `Interval<T>`.
  The signature of the method matches the `PartialOrd::<T>::partial_cmp`,
  but implementing the latter would result in poor ergonomics
  in other parts of codebase.

- improve the `Partial{Eq,Ord}` for `Endpoint` by enabling arbitrary `SIDE`-s to be comparable.
  This became possible thanks to use of `ExtPoint` representing
  a (possibly infinite) point with its
  [neighbourhood](https://en.wikipedia.org/wiki/Neighbourhood_(mathematics)#Neighbourhood_of_a_point).

- improve set operations for `Interval<T>`:
  - rename the `contains_other` into `is_super`;
  - add the `is_sub` (reverse to `is_super`) method;
  - add the `is_disjoint` to check whether the two `Interval`-s could be merged into a single one;

- minor changes for the `interval!` macro to allow to specify type of the produced `Interval`.

## Added

- `Interval::as_ref_bounds` as a synonym for `Interval::into_bounds(Interval::as_ref)`;


# v0.1.0

_2026-01-07_ (The first release)


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
  - `contains` to check the `Interval` contains a given point;
  - `contains_other` to check for fully-containness of another `Interval`;
  - via `Bounded` trait: 
    - `intersect` (aliased with `BitAnd` (`&`) operator);
    - `union` (aliased with `BitOr` (`|`) operator);
    - `enclosure`;

- other methods:
  - `is_empty` to check whether an interval contain no points (_degenerate_);
  - `is_full` to check whether an interval contain all possible points (_universe_);
  - `len` to get a measure of an interval (as the `Size<Diff<T>>` type);
  - `clamp` to force the given point to fall into the `Interval`;
  
  - `as_ref` to represent the borrowed version `Interval<&T>`
    (useful in many methods to avoid cloning doing `Interval::into_bounds`);
  - `map` to convert to another `Interval<U>` given transformation function
    `Fn(T) -> (U)`;
  - `reduce` to simplify the defintion of an `Interval` to the equivalent one:
    - the _degenerate_ reduced to the `Interval::Empty`;
    - the _singleton_ interval `[x, x]` reduced to `Interval::Singleton(x)`
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
