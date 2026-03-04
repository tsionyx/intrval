//! Implementation details dependent on the `random` feature,
//! which is used to support stochastic rounding.

use crate::{
    helper::{OneOrPair, Pair},
    traits::{Metric, Zero},
};

use super::{modes::TieBreaking, RoundError, RoundingMode, TieSelection};

pub use rand::RngCore as RandRng;

/// Stochastic rounding, picking between two nearest values
/// with probability proportional to their distance from the original value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StochasticMode;

impl<T> RoundingMode<T> for StochasticMode
where
    T: PartialOrd + Zero + Metric,
    <T as Metric>::Distance: TryInto<f64>,
{
    fn round(
        &self,
        point: &T,
        nearest: OneOrPair<T>,
        rng: Option<&mut dyn RandRng>,
    ) -> Result<T, RoundError<T>> {
        Ok(nearest.single_or_fold(|nearest_lower, nearest_upper| {
            let total: Option<f64> = nearest_upper.distance(&nearest_lower).try_into().ok();
            let to_lower: Option<f64> = point.distance(&nearest_lower).try_into().ok();

            // the closer (_less_ distance) to `lower`, the _lower_ the probability to pick `upper`
            //
            // Note: division by zero is safe here because the +/- inf handled separately.
            let prob_upper =
                Probability::new(total.and_then(|total| to_lower.map(|to_lower| to_lower / total)));

            let select_upper = bernoulli_sample(prob_upper.get_f64(), rng);
            if select_upper {
                nearest_upper
            } else {
                nearest_lower
            }
        }))
    }

    fn is_stochastic(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// Pick between two values at random with the given probability:
/// - right with probability `p = prob_upper`.
/// - left with probability `q = 1 - prob_upper`;
pub struct RandomTie {
    /// Probability to pick the upper value.
    pub prob_upper: Probability,
}

impl<T> TieBreaking<T> for RandomTie {
    fn select_opt(&self, _: Pair<&T>, rng: Option<&mut dyn RandRng>) -> Option<TieSelection> {
        let prob_upper = self.prob_upper.get_f64();
        let select_upper = bernoulli_sample(prob_upper, rng);
        Some(if select_upper {
            TieSelection::Right
        } else {
            TieSelection::Left
        })
    }

    fn last_resort(&self) -> TieSelection {
        if self.prob_upper.get_f64() >= 0.5 {
            TieSelection::Right
        } else {
            TieSelection::Left
        }
    }

    fn is_stochastic(&self) -> bool {
        true
    }
}

#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
/// A newtype wrapping an (optional) f64 value denoting probability.
pub struct Probability {
    value: Option<f64>,
}

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

/// Obtain a Bernoulli outcome with the probability of `p`.
/// If the RNG is not provided, use the global RNG.
///
/// # Note
/// The functions `rand::random_bool` and `rand::Rng::random_bool`
/// were added in `rand@0.9.0`, so we can use both of them
/// to directly obtain a Bernoulli outcome.
/// <https://github.com/rust-random/rand/blob/master/CHANGELOG.md#090---2025-01-27>
pub fn bernoulli_sample(p: f64, rng: Option<&mut dyn RandRng>) -> bool {
    use rand::Rng as _;

    #[allow(clippy::option_if_let_else)]
    if let Some(rng) = rng {
        rng.random_bool(p)
    } else {
        fallback_rng::with_default_rng(|rng| rng.random_bool(p))
    }
}

mod fallback_rng {
    use rand::{rngs::SmallRng, SeedableRng as _};

    pub use impl_global::with_default_rng;

    #[cfg(feature = "std")]
    mod impl_global {
        use super::*;
        use std::{cell::RefCell, thread_local};

        thread_local! {
            /// A global (per-thread) fallback RNG used for stochastic rounding
            /// when no RNG is provided.
            ///
            /// This is a more stable variant when the `std` feature is enabled,
            /// but it is not available in `no-std` environments.
            static DEFAULT_RNG: RefCell<SmallRng> = RefCell::new(SmallRng::seed_from_u64(get_seed()));
        }

        pub fn with_default_rng<F, R>(f: F) -> R
        where
            F: FnOnce(&mut SmallRng) -> R,
            R: 'static,
        {
            DEFAULT_RNG.with(|cell| f(&mut cell.borrow_mut()))
        }
    }

    #[cfg(not(feature = "std"))]
    mod impl_global {
        use super::*;
        use crate::helper::sync::OnceLock;

        /// A global (shared between threads) fallback RNG used for stochastic rounding
        /// when no RNG is provided.
        ///
        /// This is a version for `no-std` environments,
        /// but it probably should be replaced with a more stable variant
        /// (enable the `std` feature for this).
        static DEFAULT_RNG: OnceLock<SmallRng> = OnceLock::new();

        pub fn with_default_rng<F, R>(f: F) -> R
        where
            F: FnOnce(&mut SmallRng) -> R,
            R: 'static,
        {
            DEFAULT_RNG.with_mut_spin_lock(f, || {
                let seed = get_seed();
                SmallRng::seed_from_u64(seed)
            })
        }
    }

    fn get_seed() -> u64 {
        use crate::helper::slice_to_array_or_default;

        // inspired by https://github.com/tkaitchuck/constrandom/
        option_env!("CONST_RANDOM_SEED")
            .map(|value| u64::from_le_bytes(slice_to_array_or_default(value.as_bytes())))
            .unwrap_or(0x1234_5678_9abc_def0)
    }
}

#[cfg(all(feature = "serde", test))]
mod deser_tests {
    use serde_json::json;

    use super::{super::NearestMode, *};

    #[test]
    fn nearest_mode_with_random() {
        let j = json!({
            "NEAREST": {"prob_upper": 0.25}
        });

        let mode: NearestMode<RandomTie> = serde_json::from_value(j).unwrap();
        assert_eq!(
            mode.0,
            RandomTie {
                prob_upper: Probability::new(0.25)
            }
        );
    }

    #[test]
    fn nearest_mode_with_default_random_prob() {
        let j = json!({
            "NEAREST": {}
        });

        let mode: NearestMode<RandomTie> = serde_json::from_value(j).unwrap();
        assert_eq!(
            mode.0,
            RandomTie {
                prob_upper: Probability::default(),
            }
        );
    }
}
