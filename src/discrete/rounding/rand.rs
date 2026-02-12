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
pub const UNIFORM_CHOICE_PROB: f64 = 0.5;

#[cfg(feature = "random")]
/// Obtain a Bernoulli outcome with the probability of `p`.
/// If the RNG is not provided, use the global RNG.
///
/// # Note
/// The functions `rand::random_bool` and `rand::Rng::random_bool`
/// was added in `rand@0.9.0`, so we can use both of them
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

#[cfg(feature = "random")]
mod fallback_rng {
    use rand::{rngs::SmallRng, SeedableRng as _};

    use crate::helper::{slice_to_array_or_default, sync::OnceLock};

    static DEFAULT_RNG: OnceLock<SmallRng> = OnceLock::new();

    pub fn with_default_rng<F, R>(f: F) -> R
    where
        F: FnOnce(&mut SmallRng) -> R,
    {
        DEFAULT_RNG.with_mut_spin_lock(f, || {
            let seed = get_seed();
            SmallRng::seed_from_u64(seed)
        })
    }

    fn get_seed() -> u64 {
        // inspired by https://github.com/tkaitchuck/constrandom/
        option_env!("CONST_RANDOM_SEED")
            .map(|value| u64::from_le_bytes(slice_to_array_or_default(value.as_bytes())))
            .unwrap_or(0x1234_5678_9abc_def0)
    }
}
