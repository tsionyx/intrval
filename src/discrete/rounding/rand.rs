//! Implementation details dependent on the `random` feature,
//! which is used to support stochastic rounding.

#[cfg(feature = "random")]
pub use rand::RngCore as RandRng;

#[cfg(not(feature = "random"))]
/// Dummy trait when the `random` feature is disabled.
pub trait RandRng {}

#[cfg(not(feature = "random"))]
/// When the `random` feature is disabled, we only require `PartialOrd`.
pub trait Distance: PartialOrd {}
#[cfg(not(feature = "random"))]
impl<T> Distance for T where T: PartialOrd {}

#[cfg(feature = "random")]
/// When the `random` feature is enabled, we require additional traits
/// to support stochastic rounding using the distances,
pub trait Distance: PartialOrd + TryInto<f64> {}
#[cfg(feature = "random")]
impl<T> Distance for T where T: PartialOrd + TryInto<f64> {}

#[cfg(feature = "random")]
pub use self::prob::Probability;

#[cfg(feature = "random")]
mod prob {
    #[derive(Debug, Copy, Clone)]
    /// A newtype wrapping an (optional) f64 value denoting probability.
    pub struct Probability {
        value: Option<f64>,
    }

    #[cfg(feature = "random")]
    impl Probability {
        /// Create a new `Probability` with the given value.
        ///
        /// The value, if `Some`, will be clamped into `[0, 1]` interval.
        /// It defaults to `0.5` (if `None`) for uniform distribution.
        pub fn new(value: impl Into<Option<f64>>) -> Self {
            Self {
                value: value.into(),
            }
        }

        const UNIFORM_CHOICE_PROB: f64 = 0.5;

        #[must_use]
        /// Get the normalized probability value.
        ///
        /// If the original value is `None` or not finite, it defaults to `0.5` for uniform distribution.
        /// Otherwise, it is clamped into the range `[0, 1]`.
        pub fn get_f64(&self) -> f64 {
            self.value
                .filter(|p| p.is_finite())
                .map_or(Self::UNIFORM_CHOICE_PROB, |p| p.clamp(0.0, 1.0))
        }
    }

    impl Default for Probability {
        fn default() -> Self {
            Self::new(None)
        }
    }

    // ---- manual PartialEq and Eq implementations to deal with `f64: !Eq` ---
    impl PartialEq for Probability {
        fn eq(&self, other: &Self) -> bool {
            self.get_f64().total_cmp(&other.get_f64()).is_eq()
        }
    }

    impl Eq for Probability {}

    impl From<Option<f64>> for Probability {
        fn from(value: Option<f64>) -> Self {
            Self::new(value)
        }
    }
}

#[cfg(feature = "random")]
/// Obtain a Bernoulli outcome with the probability of `p`.
/// If the RNG is not provided, use the global RNG.
///
/// # Note
/// The functions `rand::random_bool` and `rand::Rng::random_bool`
/// were added in `rand@0.9.0`, so we can use both of them
/// to directly obtain a Bernoulli outcome.
/// <https://github.com/rust-random/rand/blob/master/CHANGELOG.md#090---2025-01-27>
pub fn bernoulli_sample(p: f64, rng: Option<&mut dyn RandRng>) -> bool {
    rng.map_or_else(
        || rand::random_bool(p),
        |rng| rand::Rng::random_bool(rng, p),
    )
}
