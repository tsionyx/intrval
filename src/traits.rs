//! The traits describing some numerical behaviour.
use core::{
    cmp::Ordering,
    ops::{Add, Div, Mul, Sub},
};

/// The trait to define scalar (single-dimension) types
/// with a dedicated origin (zero) point.
///
/// Currently, it is blanket-implemented for all types that implement `TryFrom<u8>`,
/// which covers at least all core primitive numeric types
/// (like `iN`, `uN` and `fN` where N is the size in bits).
pub trait Zero {
    /// Produce the zero (neutral in terms of sum) element of a type.
    fn zero() -> Self;

    /// Determines how the value is comparable to zero.
    fn cmp_zero(&self) -> Option<Ordering>;
}

impl<T> Zero for T
where
    T: TryFrom<u8> + PartialOrd,
{
    fn zero() -> Self {
        Self::try_from(0).unwrap_or_else(|_| panic!("conversion from 0 failed"))
    }

    fn cmp_zero(&self) -> Option<Ordering> {
        let zero = Self::try_from(0).ok()?;
        self.partial_cmp(&zero)
    }
}

/// The ability to have a distance between two values of the type.
pub trait Measure:
    Sized + Add<Self::Distance, Output = Self> + Sub<Self::Distance, Output = Self>
{
    /// The type representing a distance (difference) between two quantities.
    type Distance;
}

// blanket impl for homogeneous addition/subtraction
impl<T> Measure for T
where
    Self: Add<Output = Self> + Sub<Output = Self>,
{
    type Distance = Self;
}

/// Helper trait combining the four basic arithmetic _linear_ operations:
/// - addition / subtraction;
/// - multiplying to scalar value;
/// - dividing to get scalar (ratio) value.
///
/// <https://en.wikipedia.org/wiki/Linear_space>
pub trait Linear: Measure<Distance = Self> + Mul<Ratio<Self>, Output = Self> + Div<Self> {}

type Ratio<T> = <T as Div<T>>::Output;

impl<T> Linear for T where T: Measure<Distance = Self> + Mul<Ratio<Self>, Output = Self> + Div<Self> {}

/// Extend a [self-divisible type][Div] with integer division.
pub trait IntDiv: Div + Sized {
    /// Ensure the ratio to be integer by rounding it (towards zero).
    fn round_to_int(r: Ratio<Self>) -> Ratio<Self>;

    /// Perform integer division by rounding the ratio to integer (towards zero)
    /// using the [`Self::round_to_int`] method.
    fn int_div(self, other: Self) -> Ratio<Self> {
        Self::round_to_int(self / other)
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
pub trait MonotonicLinear: Linear + PartialOrd {
    /// Check and perform monotonic addition.
    ///
    /// The operation should ensure the sum is:
    /// - greater than `self` when the `rhs` is greater than zero;
    /// - less than `self` when the `rhs` is less than zero;
    /// - equal to `self` when the `rhs` is equal to zero.
    ///
    /// or in pseudocode:
    /// ```no_compile
    /// let zero_ord = rhs.cmp_zero()?;
    /// let result = self.clone() + rhs;
    /// (result.partial_cmp(&self)? == zero_ord).then_some(result)
    /// ```
    fn monotonic_add(self, rhs: Self) -> Option<Self>;

    /// Check and perform monotonic subtraction.
    ///
    /// The operation should ensure the difference is:
    /// - less than `self` when the `rhs` is greater than zero;
    /// - greater than `self` when the `rhs` is less than zero;
    /// - equal to `self` when the `rhs` is equal to zero.
    ///
    /// or in pseudocode:
    /// ```no_compile
    /// let zero_ord = rhs.cmp_zero()?;
    /// let result = self.clone() - rhs;
    /// (result.partial_cmp(&self)? == zero_ord.reverse()).then_some(result)
    /// ```
    fn monotonic_sub(self, rhs: Self) -> Option<Self>;
}

mod impls {
    use super::{IntDiv, MonotonicLinear, Ratio, Zero};

    // FIXME: impl for `std` types (not `core`):
    // impl Measure for std::time::{SystemTime, Instant} {
    //     type Distance = core::time::Duration;
    // }

    /// Implement `IntDiv` for integer types by simply returning the ratio as is.
    macro_rules! impl_int_div_for_int {
        ($($int:ty),+ $(,)?) => {$(
            impl IntDiv for $int {
                fn round_to_int(r: Ratio<Self>) -> Ratio<Self> {
                    r
                }
            }
        )+};
    }
    impl_int_div_for_int!(i8, u8, i16, u16, i32, u32, i64, u64, isize, usize, i128, u128);

    /// Implement `IntDiv` for floating-point types by truncating the ratio.
    ///
    /// # Note
    ///
    /// The `NaN` values will be treated as zero.
    macro_rules! impl_int_div_for_float {
        ($($f:ty => $int_ty:ty),+ $(,)?) => {$(
            impl IntDiv for $f {
                fn round_to_int(r: Ratio<Self>) -> Ratio<Self> {
                    #![allow(
                        trivial_numeric_casts,
                        clippy::as_conversions,
                        clippy::cast_possible_truncation,
                        clippy::cast_precision_loss,
                        clippy::cast_sign_loss,
                    )]
                    if r.is_nan() {
                        0.0 as Ratio<Self>
                    } else {
                        // `f{32,64}.trunc()` is still unstable in `core` as of _Rust 1.93_
                        // (https://github.com/rust-lang/rust/issues/137578),
                        // so emulate it with casting to integer and back.
                        // This way will clamp some values with extremely large
                        // absolute value (including +/- infinity) into integer range as well.
                        let sign = r.is_sign_positive();

                        let truncated_abs = {
                            let r_abs_finite = {
                                // Make it a positive and clamp to integer range before casting.
                                //
                                // Note: the `{float}::abs()` was only stabilized in core in _Rust 1.84_
                                // (https://github.com/rust-lang/rust/releases/tag/1.84.0)
                                let r_abs = if sign {
                                    r
                                } else {
                                    -r   // use unary negation instead of calling `abs()`
                                };

                                // return the clamped value when `|r|` is infinity
                                if r_abs.is_infinite() {
                                    Ratio::<Self>::MAX
                                } else {
                                    r_abs
                                }
                            };

                            let max = <$int_ty>::MAX as Ratio<Self>;
                            if r_abs_finite >= max {
                                // return unchanged when `|r| >= 2^mantissa_bits`
                                // (it has no fractional part in IEEE-754)
                                r_abs_finite
                            } else {
                                // the cast to integer will truncate the fractional part,
                                // and the cast back to float will restore the original integer value
                                // (it should be within the range of the integer type in this branch)
                                (r_abs_finite as $int_ty) as Ratio<Self>
                            }
                        };

                        if sign {
                            truncated_abs
                        } else {
                            -truncated_abs
                        }
                    }
                }
            }
        )+};
    }

    impl_int_div_for_float!(f32 => i64, f64 => i128);

    #[macro_export]
    /// Helper macro to implement `IntDiv` for numeric types
    /// which ratio could be (fallibly) converted to/from core numeric types that
    /// already implement `IntDiv`.
    ///
    /// The following underlying primitive types are now supported
    /// (and can be used on the right hand of `as` in the macro):
    /// - integers: i8, u8, i16, u16, i32, u32, i64, u64, isize, usize, i128, u128
    /// - floats: f32, f64
    ///
    /// # Note
    /// The macro uses `unwrap_or_default()`, which means type conversion failures
    /// will result in default values (typically zero) being used instead of propagating errors.
    /// If you cannot rely on this behaviour, it is better to provide the manual implementation
    /// of the `IntDiv` instead to cover the edge cases of the type conversions.
    ///
    /// # TODO
    ///
    /// consider using feature flags to implement for more numeric types,
    /// like the ones listed in:
    ///   - <https://crates.io/keywords/int>
    ///   - <https://crates.io/keywords/decimal>
    ///
    /// The naive blanket impl approach `impl for T where T: Into<f64> + From<f64>`
    /// does not work due to orphan rule.
    macro_rules! impl_int_div {
        ($num_ty:ty as $core_ty:ty) => {
            impl $crate::IntDiv for $num_ty {
                fn round_to_int(
                    r: <Self as core::ops::Div>::Output,
                ) -> <Self as core::ops::Div>::Output {
                    let core_num_ratio = r.try_into().unwrap_or_default();
                    let ratio_rounded = <$core_ty>::round_to_int(core_num_ratio);
                    ratio_rounded.try_into().unwrap_or_default()
                }
            }
        };
    }

    /// Implement `MonotonicLinear` for integer types using the `checked_*` methods.
    macro_rules! impl_monotonic_for_int {
        ($($int:ty),+ $(,)?) => {$(
            impl MonotonicLinear for $int {
                fn monotonic_add(self, rhs: Self) -> Option<Self> {
                    self.checked_add(rhs)
                }

                fn monotonic_sub(self, rhs: Self) -> Option<Self> {
                    self.checked_sub(rhs)
                }
            }
        )+};
    }
    impl_monotonic_for_int!(i8, u8, i16, u16, i32, u32, i64, u64, isize, usize, i128, u128);

    /// Implement `MonotonicLinear` for floating types by explicitly checking the result.
    macro_rules! impl_monotonic_for_float {
        ($($f:ty),+ $(,)?) => {$(
            impl MonotonicLinear for $f {
                fn monotonic_add(self, rhs: Self) -> Option<Self> {
                    let zero_ord = rhs.cmp_zero()?;
                    let result = self + rhs;
                    if !result.is_finite() {
                        return None;
                    }
                    (result.partial_cmp(&self)? == zero_ord).then_some(result)
                }

                fn monotonic_sub(self, rhs: Self) -> Option<Self> {
                    let zero_ord = rhs.cmp_zero()?;
                    let result = self - rhs;
                    if !result.is_finite() {
                        return None;
                    }
                    (result.partial_cmp(&self)? == zero_ord.reverse()).then_some(result)
                }
            }
        )+};
    }

    impl_monotonic_for_float!(f32, f64);

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

        #[test]
        fn f64_trunc() {
            let inf_p = 1.0 / 0.0;
            let inf_n = -1.0 / 0.0;
            let nan_p = inf_p * 0.0;
            let nan_n = inf_n * 0.0;

            let inf_repr = f64::MAX;

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
                -84785459459999193493494549584.0,
                -123.0,
                -0.0,
                0.0,
                8989898.0,
                147326823945405434343.0,
                inf_repr,
                -inf_repr,
                0.0,
                0.0,
            ];

            for (&input, exp) in inputs.iter().zip(expected) {
                let result = <f64 as IntDiv>::round_to_int(input);
                assert!(
                    (result == exp) || (result.is_nan() && exp.is_nan()),
                    "f64::round_to_int({input}) = {result}, expected {exp}",
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
                -84785459459999193493494549584.0,
                -123.0,
                -0.0,
                0.0,
                8989898.0,
                147326823945405434343.0,
                inf_repr,
                -inf_repr,
                0.0,
                0.0,
            ];

            for (&input, exp) in inputs.iter().zip(expected) {
                let result = <f32 as IntDiv>::round_to_int(input);
                assert!(
                    (result == exp) || (result.is_nan() && exp.is_nan()),
                    "f32::round_to_int({input}) = {result}, expected {exp}",
                );
            }
        }
    }
}
