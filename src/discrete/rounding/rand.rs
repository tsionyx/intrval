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
    rng.map_or_else(
        || rand::random_bool(p),
        |rng| rand::Rng::random_bool(rng, p),
    )
}
