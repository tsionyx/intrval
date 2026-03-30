use core::{
    fmt,
    num::{NonZeroU128, NonZeroU16, NonZeroU32, NonZeroU64, NonZeroU8, NonZeroUsize},
};

use crate::{
    helper::StdError,
    interval::Interval,
    rounding::{RoundError, Rounding, RoundingMode},
    traits::{LinearIntRatio, Metric, MonotonicMeasure, Zero},
};

mod discrete;
pub mod iter;
mod ops;

#[derive(Debug, Copy, Clone)]
/// A discrete linear space representing
/// a set of evenly spaced points within given bounds.
pub struct LinearSpace<T, D> {
    bounds: Interval<T>,
    step: D,
}

impl<T, D> LinearSpace<T, D>
where
    D: Zero,
{
    /// Create a new linear space with the given `bounds` and `step` size.
    ///
    /// The valid values for this space are such multiples with fixed `step` value
    /// that are within the specified `bounds`.
    ///
    /// Depending on the lower bound of the `bounds`, the multiples may be shifted:
    /// - for an inclusive lower bound `a`, the valid values are
    ///   `a, a+step, a+2*step, ...up to the upper bound`;
    /// - for an exclusive lower bound `a`, the valid values are
    ///   `a+step, a+2*step, ...up to the upper bound`;
    /// - for an unbounded lower bound, the valid values are
    ///   `..., upper-3*step, upper-2*step, upper - step, upper`.
    /// - for unbounded from both sides, the valid values are
    ///   `..., step-3*step, step-2*step, step-step, step, 2*step, ...`.
    ///
    /// # Results
    /// - if the `step` is strictly positive, `Some(Self)` is returned.
    /// - otherwise, i.e., for the non-positive (or non-comparable) `step`,
    ///   `None` is returned.
    pub fn try_bounded(bounds: Interval<T>, step: D) -> Option<Self> {
        step.cmp_zero()?
            .is_gt()
            .then_some(Self::new_raw(bounds, step))
    }

    /// Create a new unbounded linear space with `step` size.
    ///
    /// The valid values for this space are multiples of
    /// every value of `T` with a fixed `step` value.
    ///
    /// E.g.:
    /// - `step=2` for signed integers
    ///   denotes all numbers `..., -6, -4, -2, 0, 2, 4, 6, ...`;
    /// - `step=1.0` for floating-point numbers
    ///   denotes all integers `..., -2.0, -1.0, 0.0, 1.0, 2.0, ...`.
    ///
    /// # Results
    /// - if the `step` is strictly positive, `Some(Self)` is returned.
    /// - otherwise, i.e., for the non-positive (or non-comparable) `step`,
    ///   `None` is returned.
    pub fn try_new(step: D) -> Option<Self> {
        Self::try_bounded(Interval::Full, step)
    }
}

impl<T, D> LinearSpace<T, D> {
    const fn new_raw(bounds: Interval<T>, step: D) -> Self {
        Self { bounds, step }
    }
}

/// If the step size forced to be `>0`,
/// the constructed values are always valid.
macro_rules! impl_with_positive_step {
    ($($inner:ty => $non_zero:ty),+ $(,)?) => {$(
        impl<T> LinearSpace<T, $inner> {
            #[doc = "Create a new linear space with the given `bounds` and (wrapped in `NonZero`) `step` size."]
            #[doc = "\n"]
            #[doc = concat!("When the `step` is [`", stringify!($non_zero), "`]")]
            #[doc = "the result is guaranteed to be valid, as opposed to"]
            #[doc = "the generic constructor [`Self::try_bounded`]."]
            pub const fn bounded(bounds: Interval<T>, step: $non_zero) -> Self {
                Self::new_raw(bounds, step.get())
            }

            #[doc = "Create a new unbounded linear space with (wrapped in `NonZero`) `step` size."]
            #[doc = "\n"]
            #[doc = concat!("When the `step` is [`", stringify!($non_zero), "`]")]
            #[doc = "the result is guaranteed to be valid, as opposed to"]
            #[doc = "the generic constructor [`Self::try_new`]."]
            pub const fn new(step: $non_zero) -> Self {
                Self::bounded(Interval::Full, step)
            }
        }
    )+}
}

impl_with_positive_step! {
    u8 => NonZeroU8,
    u16 => NonZeroU16,
    u32 => NonZeroU32,
    u64 => NonZeroU64,
    u128 => NonZeroU128,
    usize => NonZeroUsize,
}

impl<T, D> LinearSpace<T, D> {
    /// Get the bounds as a reference to the inner [`Interval`].
    pub const fn bounds(&self) -> &Interval<T> {
        &self.bounds
    }

    /// Get the step size.
    pub const fn step(&self) -> &D {
        &self.step
    }

    /// Convert the space into its components,
    /// i.e., the bounds and step size.
    pub fn into_parts(self) -> (Interval<T>, D) {
        (self.bounds, self.step)
    }

    /// Convert to a [`LinearSpace`] of another type using
    /// the two mapping functions (one for the bounds and another for step size).
    ///
    /// # Returns
    ///
    /// If the `f_step` function produces a valid `step` value (i.e., a positive number)
    /// then `Some(LinearSpace<U, V>)` is returned, otherwise `None` is returned.
    pub fn map<U, V, F1, F2>(self, f_bounds: F1, f_step: F2) -> Option<LinearSpace<U, V>>
    where
        F1: Fn(T) -> U,
        F2: Fn(D) -> V,
        V: Zero,
    {
        let (bounds, step) = self.into_parts();
        let bounds = bounds.map(&f_bounds);
        let step = f_step(step);
        LinearSpace::try_bounded(bounds, step)
    }
}

impl<T, D> PartialEq for LinearSpace<T, D>
where
    T: PartialOrd,
    D: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.bounds() == other.bounds() && self.step() == other.step()
    }
}

impl<T, D> Eq for LinearSpace<T, D>
where
    T: PartialOrd,
    D: Eq,
{
}

/// Helper extension trait to ease rounding of any [`MonotonicMeasure`]
/// using an _unbounded_ [`LinearSpace`].
///
/// For more advanced use cases consider creating the [`LinearSpace`] manually
/// and use methods of [`Rounding`] trait directly.
pub trait LinearRoundable: MonotonicMeasure {
    /// Round a quantity linearly with the given `step`
    /// using a provided [mode][RoundingMode].
    ///
    /// # Errors
    /// - [`LinearRoundError::InvalidStep`] when the step
    ///   cannot be used to create a valid [`LinearSpace`];
    /// - [`LinearRoundError::Rounding`] if the underlying
    ///   [`Rounding::round`] operation failed.
    fn round(
        &self,
        step: Self::Distance,
        mode: impl RoundingMode<Self>,
    ) -> Result<Self, LinearRoundError<Self, Distance<Self>>>;
}

#[derive(Debug)]
/// An error while using a [`LinearRoundable::round`].
pub enum LinearRoundError<T, S> {
    /// Cannot create a [`LinearSpace`] because the given `step` is invalid.
    InvalidStep(S),
    /// Underlying [`RoundError`] while doing [`Rounding::round`].
    Rounding(RoundError<T>),
}

impl<T, S> fmt::Display for LinearRoundError<T, S>
where
    T: fmt::Display,
    S: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidStep(step) => {
                write!(f, "The given step ")?;
                step.fmt(f)?;
                write!(f, " cannot be used to create a `LinearSpace`")
            }
            Self::Rounding(err) => {
                write!(f, "Rounding::round failed: ")?;
                err.fmt(f)
            }
        }
    }
}

impl<T, S> StdError for LinearRoundError<T, S>
where
    T: fmt::Debug + fmt::Display,
    S: fmt::Debug + fmt::Display,
{
}

type Distance<T> = <T as Metric>::Distance;

impl<T> LinearRoundable for T
where
    T: Clone + Ord + Zero + MonotonicMeasure,
    Distance<T>: Clone + Zero + LinearIntRatio,
{
    fn round(
        &self,
        step: Distance<T>,
        mode: impl RoundingMode<Self>,
    ) -> Result<Self, LinearRoundError<Self, Distance<Self>>> {
        let space =
            LinearSpace::try_new(step.clone()).ok_or(LinearRoundError::InvalidStep(step))?;
        Rounding::round(&space, self, mode).map_err(LinearRoundError::Rounding)
    }
}

#[cfg(test)]
mod tests {
    use core::ops::Sub as _;

    use ordered_float::OrderedFloat as OF;
    use rust_decimal::Decimal;

    use crate::{
        impl_linear, impl_linear_int, impl_metric, impl_monotonic, impl_zero,
        traits::{Linear as _, LinearIntRatio, Metric as _, MonotonicMeasure},
    };

    type F32 = OF<f32>;
    type F64 = OF<f64>;

    impl_zero!(using OF(0.0) => F32, F64);
    impl_metric!(using sub for F32, F64);
    impl_linear!(F32, F64);
    impl_linear_int!(F32 as f32, F64 as f64);
    impl_monotonic!(F32 as f32, F64 as f64);

    impl_zero!(using Decimal::ZERO => Decimal);
    impl_metric!(using sub for Decimal);
    impl_linear!(Decimal);

    impl LinearIntRatio for Decimal {
        fn trunc_scalar(ratio: Self::Scalar) -> Option<Self::Scalar> {
            Some(ratio.trunc())
        }

        fn int_ratio(self, other: Self) -> Option<Self::Scalar> {
            self.get_ratio(other).and_then(Self::trunc_scalar)
        }
    }

    impl MonotonicMeasure for Decimal {
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
            Some(Self::ZERO)
        }
    }
}
