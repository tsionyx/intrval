//! Collection of helper traits to implement the operations for number-like types.
//!
//! By default the traits are implemented for core primitive numeric types
//! (like `iN`, `uN` and `fN` where N is the size in bits) and `core::time::Duration`.
//! When the `std` feature is enabled, they are also implemented for certain `std` types such as `SystemTime`.

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

    use crate::{
        impl_linear, impl_metric, impl_zero,
        macros::{
            impl_linear_int_for_float, impl_linear_int_for_int, impl_monotonic_for_float,
            impl_monotonic_for_int,
        },
    };

    use super::{Linear, LinearIntRatio, MonotonicMeasure, Zero};

    impl_zero!(using 0 => i8, u8, i16, u16, i32, u32, i64, u64, isize, usize, i128, u128);
    impl_zero!(using 0.0 => f32, f64);
    impl_zero!(using Duration::ZERO => Duration);

    impl_metric!(using saturating_sub for i8, u8, i16, u16, i32, u32, i64, u64, isize, usize, i128, u128);
    impl_metric!(using sub for f32, f64);
    impl_metric!(using saturating_sub for Duration);

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

    impl_linear_int_for_int!(i8, u8, i16, u16, i32, u32, i64, u64, isize, usize, i128, u128);
    impl_linear_int_for_float!(f32 => i64, f64 => i128);

    impl_monotonic_for_int!(i8, u8, i16, u16, i32, u32, i64, u64, isize, usize, i128, u128);
    impl_monotonic_for_float!(f32, f64);

    #[cfg(feature = "std")]
    mod std {
        use core::time::Duration;
        use std::time::{Instant, SystemTime};

        use super::{
            super::{Metric, MonotonicMeasure},
            impl_zero,
        };

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
