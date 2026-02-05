use crate::{interval::Interval, traits::Zero};

mod discrete;
mod ops;

#[derive(Debug, Copy, Clone)]
/// A discrete linear space representing
/// a set of evenly spaced points within given bounds.
pub struct LinearSpace<T> {
    bounds: Interval<T>,
    step: T,
}

impl<T> LinearSpace<T>
where
    T: Zero,
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
    pub fn try_bounded(bounds: Interval<T>, step: T) -> Option<Self> {
        step.cmp_zero()?.is_gt().then_some(Self { bounds, step })
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
    pub fn try_new(step: T) -> Option<Self> {
        Self::try_bounded(Interval::Full, step)
    }
}

impl<T> LinearSpace<T> {
    /// Get the bounds as a reference to the inner [`Interval`].
    pub const fn bounds(&self) -> &Interval<T> {
        &self.bounds
    }

    /// Get the step size.
    pub const fn step(&self) -> &T {
        &self.step
    }

    /// Convert the space into its components,
    /// i.e., the bounds and step size.
    pub fn into_parts(self) -> (Interval<T>, T) {
        (self.bounds, self.step)
    }

    /// Convert to a [`LinearSpace`] of another type using
    /// the given mapping function for the bounds and step size.
    ///
    /// # Returns
    ///
    /// If the mapping function produces a valid `step` value (i.e., a positive number)
    /// then `Some(LinearSpace<U>)` is returned, otherwise `None` is returned.
    pub fn map<U, F>(self, f: F) -> Option<LinearSpace<U>>
    where
        F: Fn(T) -> U,
        U: Zero,
    {
        let (bounds, step) = self.into_parts();
        let bounds = bounds.map(&f);
        let step = f(step);
        LinearSpace::try_bounded(bounds, step)
    }
}

impl<T: PartialOrd> PartialEq for LinearSpace<T> {
    fn eq(&self, other: &Self) -> bool {
        self.bounds == other.bounds && self.step == other.step
    }
}

impl<T: PartialOrd> Eq for LinearSpace<T> {}

#[cfg(test)]
mod tests {
    use crate::traits::{IntDiv, MonotonicLinear};

    type F32 = ordered_float::OrderedFloat<f32>;

    impl IntDiv for F32 {
        fn round_to_int(r: Self) -> Self {
            Self(f32::round_to_int(r.0))
        }
    }

    impl MonotonicLinear for F32 {
        fn monotonic_add(self, rhs: Self) -> Option<Self> {
            self.0.monotonic_add(rhs.0).map(Self)
        }

        fn monotonic_sub(self, rhs: Self) -> Option<Self> {
            self.0.monotonic_sub(rhs.0).map(Self)
        }
    }

    type F64 = ordered_float::OrderedFloat<f64>;

    impl IntDiv for F64 {
        fn round_to_int(r: Self) -> Self {
            Self(f64::round_to_int(r.0))
        }
    }

    impl MonotonicLinear for F64 {
        fn monotonic_add(self, rhs: Self) -> Option<Self> {
            self.0.monotonic_add(rhs.0).map(Self)
        }

        fn monotonic_sub(self, rhs: Self) -> Option<Self> {
            self.0.monotonic_sub(rhs.0).map(Self)
        }
    }
}
