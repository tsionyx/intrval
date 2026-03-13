//! The traits describing some numerical behaviour.
use core::cmp::Ordering;

/// The trait to define scalar (single-dimension) types
/// with a dedicated origin (zero) point.
///
/// Currently, it is implemented for all core primitive numeric types
/// (like `iN`, `uN` and `fN` where N is the size in bits)
/// as well as for `Duration` and, when the `std` feature is enabled,
/// for certain `std` types such as `SystemTime`.
pub trait Zero {
    /// Produce the zero (neutral in terms of sum) element of a type.
    fn zero() -> Self;

    /// Determines how the value is comparable to zero.
    fn cmp_zero(&self) -> Option<Ordering>;
}

/// The trait representing ability to have a distance
/// between two values of the type.
pub trait Metric: Sized {
    /// The type representing a distance (difference) between two items.
    type Distance;

    /// Calculate the distance between two points.
    ///
    /// This method is expected to be commutative.
    fn distance(&self, rhs: &Self) -> Self::Distance;
}

/// Helper trait combining the basic arithmetic _linear_ operations:
/// - addition / subtraction;
/// - multiplying to scalar value;
/// - dividing to get scalar (ratio) value.
///
/// <https://en.wikipedia.org/wiki/Linear_space>
pub trait Linear: Metric {
    /// The scalar type to be used in Mul/Div operations.
    type Scalar;

    #[must_use]
    /// Multiply a value to scalar getting another value.
    fn mul_scalar(self, scalar: Self::Scalar) -> Self;

    /// Get a ratio of two values as a [Scalar][Self::Scalar] value.
    fn get_ratio(self, rhs: Self) -> Option<Self::Scalar>;
}

/// Extend a [`Linear`] with integer ratio.
pub trait LinearIntRatio: Linear {
    /// Ensure the ratio to be integer by rounding it.
    ///
    /// The actual direction of the rounding is irrelevant, since the rounding algorithm
    /// will adjust the value anyway. By convention it is better to use
    /// the truncation (rounding towards zero).
    fn trunc_scalar(ratio: Self::Scalar) -> Option<Self::Scalar>;

    /// Extension of the [`Linear::get_ratio`] method to get an integer ratio.
    ///
    /// Performs integer division by rounding the ratio to integer
    /// using the [`Self::trunc_scalar`] method.
    fn int_ratio(self, other: Self) -> Option<Self::Scalar>;

    #[must_use]
    /// Quantize the value to one of the nearest multiple of `step`
    /// (usually truncating `self` towards zero,
    /// see the corresponding [`Self::int_ratio`] implementation).
    fn quantize(self, step: Self) -> Self
    where
        Self: Clone,
    {
        #![allow(clippy::option_if_let_else)]

        match self.clone().int_ratio(step.clone()) {
            Some(no_steps) => step.mul_scalar(no_steps),
            None => self,
        }
    }
}

/// Extension of [linear types][Linear] with monotonic addition and subtraction.
///
/// The operation `monotonic_{add,sub}` is similar to the standard addition/subtraction,
/// but contains additional checks to ensure that the result is greater/less than the `lhs`
/// depending on the `rhs` of the operation.
///
/// This generalizes the concept of `checked_*` arithmetic methods for integer types.
/// to other linear types.
/// E.g. for floating-point types, the `monotonic_{add,sub}` would check for `NaN` results
/// or the loss of precision when the operands' magnitudes differ significantly.
pub trait MonotonicMeasure: Metric + PartialOrd {
    /// Monotonically perform an addition of a distance.
    ///
    /// The operation should ensure the sum is:
    /// - greater than `self` when the `rhs` is greater than zero;
    /// - less than `self` when the `rhs` is less than zero;
    /// - equal to `self` when the `rhs` is equal to zero.
    ///
    /// or in pseudocode:
    /// ```no_compile
    /// let zero_ord = diff.cmp_zero()?;
    /// let result = self.clone() + diff;
    /// (result.partial_cmp(&self)? == zero_ord).then_some(result)
    /// ```
    fn monotonic_add(self, diff: Self::Distance) -> Option<Self>;

    /// Monotonically perform a subtraction of a distance.
    ///
    /// The operation should ensure the difference is:
    /// - less than `self` when the `rhs` is greater than zero;
    /// - greater than `self` when the `rhs` is less than zero;
    /// - equal to `self` when the `rhs` is equal to zero.
    ///
    /// or in pseudocode:
    /// ```no_compile
    /// let zero_ord = diff.cmp_zero()?;
    /// let result = self.clone() - diff;
    /// (self.partial_cmp(&result)? == zero_ord).then_some(result)
    /// ```
    fn monotonic_sub(self, diff: Self::Distance) -> Option<Self>;

    /// Get a (non-negative) distance between points.
    ///
    /// The operation should ensure the distance is:
    /// - greater than or equal to zero;
    /// - commutative, i.e., `self.checked_diff(rhs) == rhs.checked_diff(self)`;
    /// - `None` if overflow or loss of precision occurs, i.e.,
    ///   if the result is not a valid distance between the two points.
    fn checked_diff(self, rhs: Self) -> Option<Self::Distance>;

    /// Get the origin (zero) point of the type, if it exists.
    ///
    /// It is optional but allows to shortcut some operations,
    /// e.g., calculating the distance to the origin point
    /// is often simpler than calculating the distance between two arbitrary points.
    fn origin() -> Option<Self>;
}

mod impls {
    use core::{ops::Sub as _, time::Duration};

    use super::{Linear, LinearIntRatio, MonotonicMeasure, Zero};

    #[macro_export]
    /// Helper macro to implement [`Zero`][crate::Zero] for numeric types
    /// using provided zero value and the natural ordering of the type.
    macro_rules! impl_zero {
        (using $z:expr => $($n:ty),+ $(,)?) => {$(
            impl $crate::Zero for $n {
                fn zero() -> Self {
                    $z
                }

                fn cmp_zero(&self) -> Option<core::cmp::Ordering> {
                    self.partial_cmp(&$z)
                }
            }
        )+};
    }

    impl_zero!(using 0 => i8, u8, i16, u16, i32, u32, i64, u64, isize, usize, i128, u128);
    impl_zero!(using 0.0 => f32, f64);
    impl_zero!(using Duration::ZERO => Duration);

    #[macro_export]
    /// Helper macro to implement [`Metric`][crate::Metric] for numeric types
    /// which implement a `Copy + PartialOrd` and have a `$diff` method.
    macro_rules! impl_metric {
        (using $diff:ident for $($n:ty),+ $(,)?) => {$(
            impl $crate::Metric for $n {
                type Distance = Self;

                fn distance(&self, rhs: &Self) -> Self::Distance {
                    if self < rhs {
                        rhs.$diff(*self)
                    } else {
                        self.$diff(*rhs)
                    }
                }
            }
        )+};
    }
    impl_metric!(using saturating_sub for i8, u8, i16, u16, i32, u32, i64, u64, isize, usize, i128, u128);
    impl_metric!(using sub for f32, f64);
    impl_metric!(using saturating_sub for Duration);

    #[macro_export]
    /// Helper macro to implement [`Linear`][crate::Linear] for numeric types
    /// which implement `core::ops::{Mul, Div}`.
    macro_rules! impl_linear {
        ($($t:ty),+ $(,)?) => {$(
            impl $crate::Linear for $t {
                type Scalar = <Self as core::ops::Div>::Output;

                fn mul_scalar(self, scalar: Self::Scalar) -> Self {
                    self * scalar
                }

                fn get_ratio(self, rhs: Self) -> Option<Self::Scalar> {
                    <Self as $crate::Zero>::cmp_zero(&rhs)
                        .and_then(|ord| ord.is_ne().then_some(self / rhs))
                }
            }
        )+};
    }

    impl_linear!(i8, u8, i16, u16, i32, u32, i64, u64, isize, usize, i128, u128);
    impl_linear!(f32, f64);

    impl Linear for Duration {
        type Scalar = u128;

        fn mul_scalar(self, scalar: Self::Scalar) -> Self {
            let total_nanos: u128 = self.as_nanos().saturating_mul(scalar);
            u64::try_from(total_nanos)
                .ok()
                .map_or(Self::MAX, Self::from_nanos)
        }

        fn get_ratio(self, rhs: Self) -> Option<Self::Scalar> {
            if rhs == Self::zero() {
                None
            } else {
                Some(self.as_nanos() / rhs.as_nanos())
            }
        }
    }

    impl LinearIntRatio for Duration {
        fn trunc_scalar(ratio: Self::Scalar) -> Option<Self::Scalar> {
            Some(ratio)
        }

        fn int_ratio(self, other: Self) -> Option<Self::Scalar> {
            self.get_ratio(other)
        }
    }

    /// Implement `LinearIntRatio` for integer types by simply returning the ratio as is.
    macro_rules! impl_linear_int_for_int {
        ($($int:ty),+ $(,)?) => {$(
            impl LinearIntRatio for $int {
                fn trunc_scalar(ratio: Self::Scalar) -> Option<Self::Scalar> { Some(ratio) }
                fn int_ratio(self, other: Self) ->  Option<Self::Scalar> { (other != 0).then_some(self / other) }
            }
        )+};
    }
    impl_linear_int_for_int!(i8, u8, i16, u16, i32, u32, i64, u64, isize, usize, i128, u128);

    /// Implement `LinearIntRatio` for floating-point types by truncating the ratio.
    ///
    /// # Note
    ///
    /// - `NaN` values cause [`trunc_scalar`](LinearIntRatio::trunc_scalar) to return `None`;
    /// - infinite values are clamped to the representable finite range for this type;
    ///   (i.e., `+inf` is clamped to `MAX` and `-inf` is clamped to `MIN`).
    macro_rules! impl_linear_int_for_float {
        ($($f:ty => $int_ty:ty),+ $(,)?) => {$(
            impl LinearIntRatio for $f {
                #[cfg(feature = "std")]
                fn trunc_scalar(ratio: Self::Scalar) -> Option<Self::Scalar> {
                    #![allow(
                        trivial_numeric_casts,
                        clippy::as_conversions,
                    )]
                    (!ratio.is_nan()).then(|| {
                        if ratio.is_infinite() {
                            if ratio.is_sign_positive() {
                                Self::Scalar::MAX
                            } else {
                                Self::Scalar::MIN
                            }
                        } else {
                            ratio.trunc()
                        }
                    })
                }

                #[cfg(not(feature = "std"))]
                fn trunc_scalar(ratio: Self::Scalar) -> Option<Self::Scalar> {
                    #![allow(
                        trivial_numeric_casts,
                        clippy::as_conversions,
                        clippy::cast_possible_truncation,
                        clippy::cast_precision_loss,
                        clippy::cast_sign_loss,
                    )]
                    (!ratio.is_nan()).then(|| {
                        // `f{32,64}.trunc()` is still unstable in `core` as of _Rust 1.93_
                        // (https://github.com/rust-lang/rust/issues/137578),
                        // so emulate it with casting to integer and back.
                        // This way will clamp some values with extremely large
                        // absolute value (including +/- infinity) into integer range as well.
                        let sign = ratio.is_sign_positive();

                        let truncated_abs = {
                            let r_abs_finite = {
                                // Make it a positive and clamp to integer range before casting.
                                //
                                // Note: the `{float}::abs()` was only stabilized in core in _Rust 1.84_
                                // (https://github.com/rust-lang/rust/releases/tag/1.84.0)
                                let r_abs = if sign {
                                    ratio
                                } else {
                                    -ratio   // use unary negation instead of calling `abs()`
                                };

                                // return the clamped value when `|r|` is infinity
                                if r_abs.is_infinite() {
                                    Self::Scalar::MAX
                                } else {
                                    r_abs
                                }
                            };

                            let max = <$int_ty>::MAX as Self::Scalar;
                            if r_abs_finite >= max {
                                // return unchanged when `|r| >= 2^mantissa_bits`
                                // (it has no fractional part in IEEE-754)
                                r_abs_finite
                            } else {
                                // the cast to integer will truncate the fractional part,
                                // and the cast back to float will restore the original integer value
                                // (it should be within the range of the integer type in this branch)
                                (r_abs_finite as $int_ty) as Self::Scalar
                            }
                        };

                        if sign {
                            truncated_abs
                        } else {
                            -truncated_abs
                        }
                    })
                }

                fn int_ratio(self, other: Self) -> Option<Self::Scalar> {
                    if other == 0.0 {
                        None
                    } else {
                        Self::trunc_scalar(self / other)
                    }
                }
            }
        )+};
    }

    impl_linear_int_for_float!(f32 => i64, f64 => i128);

    #[macro_export]
    /// Helper macro to implement `LinearIntRatio` for numeric types
    /// which ratio could be (fallibly) converted into/from core numeric types that
    /// already implement `LinearIntRatio`.
    ///
    /// The following underlying primitive types are now supported
    /// (and can be used on the right hand of `as` in the macro):
    /// - integers: i8, u8, i16, u16, i32, u32, i64, u64, isize, usize, i128, u128
    /// - floats: f32, f64
    ///
    /// # TODO
    ///
    /// consider using feature flags to implement for more numeric types,
    /// like the ones listed in:
    ///   - <https://crates.io/keywords/int>
    ///   - <https://crates.io/keywords/decimal>
    ///
    /// The naive blanket impl approach `impl for T where T: TryInto<f64> + TryFrom<f64>`
    /// does not work due to orphan rule.
    macro_rules! impl_linear_int {
        ($($num_ty:ty as $core_ty:ty),+ $(,)?) => {$(
            impl $crate::LinearIntRatio for $num_ty {
                fn trunc_scalar(ratio: Self::Scalar) -> Option<Self::Scalar> {
                    let core_num_ratio = ratio.try_into().ok()?;
                    let ratio_rounded = <$core_ty>::trunc_scalar(core_num_ratio)?;
                    ratio_rounded.try_into().ok()
                }

                fn int_ratio(self, other: Self) -> Option<Self::Scalar> {
                    <Self as $crate::Zero>::cmp_zero(&other)
                        .and_then(|ord| if ord.is_eq() {
                            None
                        } else {
                            Self::trunc_scalar(self / other)
                        })
                }
            }
        )+};
    }

    /// Implement `MonotonicMeasure` for integer types using the `checked_*` methods.
    macro_rules! impl_monotonic_for_int {
        ($($int:ty),+ $(,)?) => {$(
            impl MonotonicMeasure for $int {
                fn monotonic_add(self, diff: Self::Distance) -> Option<Self> {
                    self.checked_add(diff)
                }

                fn monotonic_sub(self, diff: Self::Distance) -> Option<Self> {
                    self.checked_sub(diff)
                }

                fn checked_diff(self, rhs: Self) -> Option<Self::Distance> {
                    // simple `.abs_diff()` could overflow
                    if self > rhs {
                        self.checked_sub(rhs)
                    }
                    else {
                        rhs.checked_sub(self)
                    }
                }

                fn origin() -> Option<Self> { Some(0) }
            }
        )+};
    }
    impl_monotonic_for_int!(i8, u8, i16, u16, i32, u32, i64, u64, isize, usize, i128, u128);

    /// Implement `MonotonicMeasure` for floating types by explicitly checking the result.
    macro_rules! impl_monotonic_for_float {
        ($($f:ty),+ $(,)?) => {$(
            impl MonotonicMeasure for $f {
                fn monotonic_add(self, diff: Self::Distance) -> Option<Self> {
                    let zero_ord = diff.cmp_zero()?;
                    let result = self + diff;
                    if !result.is_finite() || result.is_nan() {
                        return None;
                    }
                    (result.partial_cmp(&self)? == zero_ord).then_some(result)
                }

                fn monotonic_sub(self, diff: Self::Distance) -> Option<Self> {
                    let zero_ord = diff.cmp_zero()?;
                    let result = self - diff;
                    if !result.is_finite() || result.is_nan() {
                        return None;
                    }
                    (self.partial_cmp(&result)? == zero_ord).then_some(result)
                }

                fn checked_diff(self, rhs: Self) -> Option<Self::Distance> {
                    let result = if self > rhs {
                        self - rhs
                    }
                    else {
                        rhs - self
                    };
                    if !result.is_finite() || result.is_nan() {
                        return None;
                    }

                    {
                        #![allow(clippy::float_cmp)]

                        let rhs_is_zero = rhs.cmp_zero()?.is_eq();
                        let self_abs = if self.is_sign_positive() {
                            self
                        } else {
                            -self
                        };

                        let self_is_zero = self.cmp_zero()?.is_eq();
                        let rhs_abs = if rhs.is_sign_positive() {
                            rhs
                        } else {
                            -rhs
                        };

                        // if the magnitudes differ significantly, the result may lose precision
                        // and struggle to change the larger operand, thus failing the monotonicity check

                        // one `argument!=0` implies result differs from the `abs()` of another operand
                        ((
                            rhs_is_zero ||
                            !self_abs.partial_cmp(&result)?.is_eq() ||
                            self * 2.0 == rhs // `self_abs` could be equal to `result` in a valid situation when `rhs - self = self`
                        ) && (
                            self_is_zero ||
                            !rhs_abs.partial_cmp(&result)?.is_eq() ||
                            rhs * 2.0 == self // `rhs_abs` could be equal to `result` in a valid situation when `self - rhs = rhs`
                        )).then_some(result)
                    }
                }

                fn origin() -> Option<Self> { Some(0.0) }
            }
        )+};
    }

    impl_monotonic_for_float!(f32, f64);

    #[macro_export]
    /// Helper macro to implement `MonotonicMeasure` for numeric types
    /// which could be converted into/from core numeric types that
    /// already implement `MonotonicMeasure`.
    ///
    /// The following underlying primitive types are now supported
    /// (and can be used on the right hand of `as` in the macro):
    /// - integers: i8, u8, i16, u16, i32, u32, i64, u64, isize, usize, i128, u128
    /// - floats: f32, f64
    macro_rules! impl_monotonic {
        ($($num_ty:ty as $core_ty:ty),+ $(,)?) => {$(
            impl $crate::MonotonicMeasure for $num_ty {
                fn monotonic_add(self, diff: Self::Distance) -> Option<Self> {
                    let core_self: $core_ty = self.into();
                    let core_diff: <$core_ty as $crate::Metric>::Distance = diff.into();
                    core_self.monotonic_add(core_diff).map(Into::into)
                }

                fn monotonic_sub(self, diff: Self::Distance) -> Option<Self> {
                    let core_self: $core_ty = self.into();
                    let core_diff: <$core_ty as $crate::Metric>::Distance = diff.into();
                    core_self.monotonic_sub(core_diff).map(Into::into)
                }

                fn checked_diff(self, rhs: Self) -> Option<Self::Distance> {
                    let core_self: $core_ty = self.into();
                    let core_rhs: $core_ty = rhs.into();
                    core_self.checked_diff(core_rhs).map(Into::into)
                }

                fn origin() -> Option<Self> {
                    <$core_ty>::origin().map(Into::into)
                }
            }
        )+};
    }

    #[cfg(feature = "std")]
    mod std {
        use core::time::Duration;
        use std::time::{Instant, SystemTime};

        use super::super::{Metric, MonotonicMeasure};

        impl_zero!(using SystemTime::UNIX_EPOCH => SystemTime);

        impl Metric for Instant {
            type Distance = Duration;

            fn distance(&self, rhs: &Self) -> Self::Distance {
                self.checked_duration_since(*rhs)
                    .unwrap_or_else(|| rhs.duration_since(*self))
            }
        }

        impl Metric for SystemTime {
            type Distance = Duration;

            fn distance(&self, rhs: &Self) -> Self::Distance {
                self.duration_since(*rhs)
                    .unwrap_or_else(|err| err.duration())
            }
        }

        impl MonotonicMeasure for Instant {
            fn monotonic_add(self, diff: Self::Distance) -> Option<Self> {
                self.checked_add(diff)
            }

            fn monotonic_sub(self, diff: Self::Distance) -> Option<Self> {
                self.checked_sub(diff)
            }

            fn checked_diff(self, rhs: Self) -> Option<Self::Distance> {
                Some(self.distance(&rhs))
            }

            fn origin() -> Option<Self> {
                // FIXME: `impl_zero!(using zero_instant() => Instant)`
                // or a proper `MonotonicMeasure::origin` for `Instant`.
                // Otherwise, the `LinearSpace<Instant, Duration>` cannot be used to round `Instant` values.
                None
            }
        }

        impl MonotonicMeasure for SystemTime {
            fn monotonic_add(self, diff: Self::Distance) -> Option<Self> {
                self.checked_add(diff)
            }

            fn monotonic_sub(self, diff: Self::Distance) -> Option<Self> {
                self.checked_sub(diff)
            }

            fn checked_diff(self, rhs: Self) -> Option<Self::Distance> {
                Some(self.distance(&rhs))
            }

            fn origin() -> Option<Self> {
                Some(Self::UNIX_EPOCH)
            }
        }
    }

    #[cfg(test)]
    mod tests {
        #![allow(
            clippy::as_conversions,
            clippy::cast_precision_loss,
            clippy::excessive_precision,
            // the rounded values have no fractional parts, so direct comparison is fine
            clippy::float_cmp,
            clippy::unreadable_literal,
        )]

        use super::*;

        fn f64_inputs() -> [f64; 10] {
            let inf_p = 1.0 / 0.0;
            let inf_n = -1.0 / 0.0;
            let nan_p = inf_p * 0.0;
            let nan_n = inf_n * 0.0;

            [
                -84785459459999193493494549584.55,
                -123.343435543559,
                -0.0,
                0.0,
                8989898.44489348939775,
                147326823945405434343.87878,
                inf_p,
                inf_n,
                nan_p,
                nan_n,
            ]
        }

        #[test]
        fn f64_trunc() {
            let inputs = f64_inputs();
            let inf_repr = f64::MAX;
            let expected = [
                Some(-84785459459999193493494549584.0),
                Some(-123.0),
                Some(-0.0),
                Some(0.0),
                Some(8989898.0),
                Some(147326823945405434343.0),
                Some(inf_repr),
                Some(-inf_repr),
                None,
                None,
            ];

            for (&input, exp) in inputs.iter().zip(expected) {
                let result = <f64 as LinearIntRatio>::trunc_scalar(input);
                assert_eq!(
                    result, exp,
                    "f64::trunc_scalar({input}) = {result:?}, expected {exp:?}",
                );
            }
        }

        #[test]
        fn f32_trunc() {
            let inf_p = 1.0 / 0.0;
            let inf_n = -1.0 / 0.0;
            let nan_p = inf_p * 0.0;
            let nan_n = inf_n * 0.0;

            let inf_repr = f32::MAX;

            let inputs = [
                -84785459459999193493494549584.55,
                -123.343435543559,
                -0.0,
                0.0,
                8989898.44489348939775,
                147326823945405434343.87878,
                inf_p,
                inf_n,
                nan_p,
                nan_n,
            ];

            let expected = [
                Some(-84785459459999193493494549584.0),
                Some(-123.0),
                Some(-0.0),
                Some(0.0),
                Some(8989898.0),
                Some(147326823945405434343.0),
                Some(inf_repr),
                Some(-inf_repr),
                None,
                None,
            ];

            for (&input, exp) in inputs.iter().zip(expected) {
                let result = <f32 as LinearIntRatio>::trunc_scalar(input);
                assert_eq!(
                    result, exp,
                    "f32::trunc_scalar({input}) = {result:?}, expected {exp:?}",
                );
            }
        }

        #[test]
        fn f64_checked_diff_commutative() {
            let inputs = f64_inputs();

            // commutative
            for &a in &inputs {
                for &b in &inputs {
                    assert_eq!(a.checked_diff(b), b.checked_diff(a));
                }
            }
        }

        #[test]
        fn f64_checked_diff_results() {
            let inputs = f64_inputs();

            let expected_shift1 = [
                None, // `Some(84785459459999180000000000000.0)` lost necessary precision
                Some(123.343435543559),
                Some(0.0),
                Some(8989898.444893489),
                Some(147326823945396440000.0),
                None,
                None,
                None,
                None,
                None,
            ];
            for ((&a, &b), exp) in inputs
                .iter()
                // rotate_left(1)
                .zip(inputs.iter().skip(1).chain(&inputs[..1]))
                .zip(expected_shift1)
            {
                let result = a.checked_diff(b);
                assert_eq!(
                    result, exp,
                    "checked_diff({a}, {b}) = {result:?}, expected {exp:?}",
                );
            }

            let expected_shift2 = [
                Some(84785459459999180000000000000.0),
                Some(123.343435543559),
                Some(8989898.444893489),
                Some(147326823945405430000.0),
                None,
                None,
                None,
                None,
                None,
            ];
            for ((&a, &b), exp) in inputs
                .iter()
                // rotate_left(2)
                .zip(inputs.iter().skip(2).chain(&inputs[..2]))
                .zip(expected_shift2)
            {
                let result = a.checked_diff(b);
                assert_eq!(
                    result, exp,
                    "checked_diff({a}, {b}) = {result:?}, expected {exp:?}",
                );
            }
        }

        #[test]
        fn f64_checked_diff_double_regression() {
            let cases = [
                (84785459459999193493494549584.55, 123.343, None),
                (10.0, 5.0, Some(5.0)),
                (5.0, 0.0, Some(5.0)),
                (5.0, 10.0, Some(5.0)),
                (-10.0, 5.0, Some(15.0)),
                (-5.0, 10.0, Some(15.0)),
                (-5.0, 0.0, Some(5.0)),
                (-10.0, -5.0, Some(5.0)),
                (-5.0, -10.0, Some(5.0)),
            ];
            for (a, b, exp) in cases {
                let result = a.checked_diff(b);
                assert_eq!(
                    result, exp,
                    "checked_diff({a}, {b}) = {result:?}, expected {exp:?}",
                );
            }
        }
    }
}
