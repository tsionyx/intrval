//! Macros to implement the numeric [traits][super::traits] for various types.

#[macro_export]
/// Helper macro to implement [`Zero`][crate::traits::Zero] for numeric types
/// using provided zero value and the natural ordering of the type.
macro_rules! impl_zero {
    (using $z:expr => $($n:ty),+ $(,)?) => {$(
        impl $crate::traits::Zero for $n {
            fn zero() -> Self {
                $z
            }

            fn cmp_zero(&self) -> Option<core::cmp::Ordering> {
                self.partial_cmp(&$z)
            }
        }
    )+};
}

#[macro_export]
/// Helper macro to implement [`Metric`][crate::traits::Metric] for numeric types
/// which implement a `Copy + PartialOrd` and have a `$diff` method.
macro_rules! impl_metric {
    (using $diff:ident for $($n:ty),+ $(,)?) => {$(
        impl $crate::traits::Metric for $n {
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

#[macro_export]
/// Helper macro to implement [`Linear`][crate::traits::Linear] for numeric types
/// which implement `core::ops::{Mul, Div}`.
macro_rules! impl_linear {
    ($($t:ty),+ $(,)?) => {$(
        impl $crate::traits::Linear for $t {
            type Scalar = <Self as core::ops::Div>::Output;

            fn mul_scalar(self, scalar: Self::Scalar) -> Self {
                self * scalar
            }

            fn get_ratio(self, rhs: Self) -> Option<Self::Scalar> {
                <Self as $crate::traits::Zero>::cmp_zero(&rhs)
                    .and_then(|ord| ord.is_ne().then_some(self / rhs))
            }
        }
    )+};
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

#[macro_export]
/// Helper macro to implement [`LinearIntRatio`][crate::traits::LinearIntRatio]
/// for numeric types which ratio could be (fallibly) converted into/from types that
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
        impl $crate::traits::LinearIntRatio for $num_ty {
            fn trunc_scalar(ratio: Self::Scalar) -> Option<Self::Scalar> {
                let core_num_ratio = ratio.try_into().ok()?;
                let ratio_rounded = <$core_ty>::trunc_scalar(core_num_ratio)?;
                ratio_rounded.try_into().ok()
            }

            fn int_ratio(self, other: Self) -> Option<Self::Scalar> {
                <Self as $crate::traits::Zero>::cmp_zero(&other)
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

#[macro_export]
/// Helper macro to implement [`MonotonicMeasure`][crate::traits::MonotonicMeasure]
/// for numeric types which could be converted into/from types that
/// already implement `MonotonicMeasure`.
///
/// The following underlying primitive types are now supported
/// (and can be used on the right hand of `as` in the macro):
/// - integers: i8, u8, i16, u16, i32, u32, i64, u64, isize, usize, i128, u128
/// - floats: f32, f64
macro_rules! impl_monotonic {
    ($($num_ty:ty as $core_ty:ty),+ $(,)?) => {$(
        impl $crate::traits::MonotonicMeasure for $num_ty {
            fn monotonic_add(self, diff: Self::Distance) -> Option<Self> {
                let core_self: $core_ty = self.into();
                let core_diff: <$core_ty as $crate::traits::Metric>::Distance = diff.into();
                core_self.monotonic_add(core_diff).map(Into::into)
            }

            fn monotonic_sub(self, diff: Self::Distance) -> Option<Self> {
                let core_self: $core_ty = self.into();
                let core_diff: <$core_ty as $crate::traits::Metric>::Distance = diff.into();
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

pub(crate) use {
    impl_linear_int_for_float, impl_linear_int_for_int, impl_monotonic_for_float,
    impl_monotonic_for_int,
};
