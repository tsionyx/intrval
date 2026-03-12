//! Auxiliary types and functions.

use crate::traits::Zero;

// https://blog.rust-lang.org/2024/09/05/Rust-1.81.0/#core-error-error
#[rustversion::since(1.81)]
pub use core::error::Error as StdError;

#[rustversion::before(1.81)]
#[cfg(feature = "std")]
pub use std::error::Error as StdError;

#[rustversion::before(1.81)]
#[cfg(not(feature = "std"))]
/// Dummy trait representing a standard library's `error::Error`.
///
/// Enabled to support _no_std_ code before Rust 1.81 (where it was stabilized in `core`).
pub trait StdError: core::fmt::Debug + core::fmt::Display {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Represent a length of the arbitrary interval.
pub enum Size<T> {
    /// Interval containing no points at all.
    Empty,
    /// A size of interval containing exactly
    /// one point (or the finite number of discrete points).
    SinglePoint,
    /// A finite size of interval bounded in both directions.
    Finite(T),
    /// A size of interval extending infinitely in at least one direction.
    Infinite,
}

impl<T> Size<T> {
    /// Check whether the size represents an empty interval.
    pub const fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// Convert the size into a finite difference value, if possible.
    pub fn into_diff(self) -> Option<T>
    where
        T: Zero,
    {
        match self {
            Self::Empty | Self::SinglePoint => Some(T::zero()),
            Self::Finite(v) => Some(v),
            Self::Infinite => None,
        }
    }
}

/// Alias for the 2-tuple of the same type `T`.
pub type Pair<T> = (T, T);

/// Convert a [`Pair`] of type `T` into a [`Pair`] of type `U`.
pub fn map_pair<T, U, F>((a, b): Pair<T>, mut f: F) -> Pair<U>
where
    F: FnMut(T) -> U,
{
    (f(a), f(b))
}

/// Reorder a pair of values into ascending order.
///
/// TODO: use `core::cmp::minmax` when stabilized.
pub fn minmax<T: Ord>(v1: T, v2: T) -> [T; 2] {
    if v2 < v1 {
        [v2, v1]
    } else {
        [v1, v2]
    }
}

/// Either a single value or a pair of values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OneOrPair<T> {
    /// A single value variant.
    One(T),
    /// A pair of values variant.
    Pair(Pair<T>),
}

impl<T> OneOrPair<T> {
    /// Convert into a single value, or return the pair wrapped in `Err`.
    ///
    /// # Errors
    /// An inner pair of values if the value is a `Self::Pair`.
    pub fn into_single(self) -> Result<T, Pair<T>> {
        match self {
            Self::One(v) => Ok(v),
            Self::Pair(v) => Err(v),
        }
    }

    /// Convert into a pair of values, or return the single value wrapped in `Err`.
    ///
    /// # Errors
    /// An inner single value if the value is a `Self::One`.
    pub fn into_pair(self) -> Result<Pair<T>, T> {
        match self {
            Self::One(v) => Err(v),
            Self::Pair(v) => Ok(v),
        }
    }

    /// Apply one of the functions:
    /// - `f1` with a single value argument if `self` is the [`OneOrPair::One`],
    /// - `f2` to the two arguments if `self` is the [`OneOrPair::Pair`].
    pub fn fold<F1, F2, R>(self, f1: F1, f2: F2) -> R
    where
        F1: FnOnce(T) -> R,
        F2: FnOnce(T, T) -> R,
    {
        match self {
            Self::One(v) => f1(v),
            Self::Pair((v1, v2)) => f2(v1, v2),
        }
    }

    /// Unwrap the single value if `self` is the [`OneOrPair::One`],
    /// otherwise fold the pair of values into a single one using the provided function.
    pub fn single_or_fold<F>(self, f: F) -> T
    where
        F: FnOnce(T, T) -> T,
    {
        use core::convert::identity;
        self.fold(identity, f)
    }
}

impl<T> TryInto<Pair<T>> for OneOrPair<T> {
    type Error = T;

    fn try_into(self) -> Result<Pair<T>, T> {
        self.into_pair()
    }
}

impl<T> From<T> for OneOrPair<T> {
    fn from(value: T) -> Self {
        Self::One(value)
    }
}

impl<T> From<Pair<T>> for OneOrPair<T> {
    fn from(value: Pair<T>) -> Self {
        Self::Pair(value)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
/// Extend a type `T` by adding a notion of possible _infinity_ to it.
pub enum ValOrInf<T> {
    /// Just a value.
    Val(T),
    /// Infinite value.
    Inf,
}

impl<T> ValOrInf<T> {
    /// Get the reference to inner value,
    /// where `None` denotes the [infinity][ValOrInf::Inf].
    pub const fn get_val(&self) -> Option<&T> {
        match self {
            Self::Val(x) => Some(x),
            Self::Inf => None,
        }
    }

    /// Convert the [`ValOrInf`] into [`Option`]
    /// where `None` denotes the [infinity][ValOrInf::Inf].
    pub fn into_val(self) -> Option<T> {
        match self {
            Self::Val(x) => Some(x),
            Self::Inf => None,
        }
    }

    /// Is the value finite?
    pub const fn is_finite(&self) -> bool {
        matches!(self, Self::Val(_))
    }
}

impl<T> From<T> for ValOrInf<T> {
    fn from(value: T) -> Self {
        Self::Val(value)
    }
}

/// Synchronization primitives for internal use.
#[cfg(not(feature = "std"))]
pub mod sync {
    use core::{
        cell::UnsafeCell,
        hint,
        sync::atomic::{AtomicBool, Ordering},
    };

    /// RAII guard that acquires a given [switch][AtomicBool]
    /// on creation and releases it on drop.
    ///
    /// This allows to unlock the critical section even
    /// if the code panics while executing the former.
    struct SpinLockGuard<'a> {
        switch: &'a AtomicBool,
    }

    impl<'a> SpinLockGuard<'a> {
        fn acquire(switch: &'a AtomicBool) -> Self {
            // spin until we acquire the lock, thus entering the critical section
            while switch
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                hint::spin_loop();
            }
            Self { switch }
        }
    }

    impl Drop for SpinLockGuard<'_> {
        fn drop(&mut self) {
            // exit the critical section by releasing the lock
            self.switch.store(false, Ordering::Release);
        }
    }

    /// Very simple implementation of a spinlock-based `OnceLock` for internal use.
    pub struct OnceLock<T> {
        value: UnsafeCell<Option<T>>,
        lock: AtomicBool,
    }

    #[allow(unsafe_code)]
    // SAFETY:
    // - All access to `value: UnsafeCell<Option<T>>` happens inside `with_mut_spin_lock`,
    //   which first acquires a `SpinLockGuard`. The guard uses `lock: AtomicBool` with
    //   `compare_exchange` plus `Acquire`/`Release` ordering to provide mutual exclusion,
    //   so only one thread can read or modify the inner `T` at a time.
    // - `with_mut_spin_lock` takes `F: FnOnce(&mut T) -> R` and requires `R: 'static`.
    //   This prevents callers from returning references (or other non-'static borrows)
    //   tied to the protected `T`, so no reference to `T` can outlive the critical
    //   section while the lock is held.
    // - Because the spinlock guarantees that the inner value is only ever accessed
    //   with exclusive mutable access, it is never concurrently aliased through shared
    //   references across threads. Therefore `T` only needs to be `Send` (movable
    //   between threads); it does not need to be `Sync`.
    unsafe impl<T: Send> Sync for OnceLock<T> {}

    #[cfg_attr(not(feature = "random"), allow(dead_code))]
    impl<T> OnceLock<T> {
        /// Create a new [`OnceLock`] instance with uninitialized value and unlocked state.
        pub const fn new() -> Self {
            Self {
                value: UnsafeCell::new(None),
                lock: AtomicBool::new(false),
            }
        }

        /// Execute a closure `f` with mutable access to the inner value of the [`OnceLock`],
        /// initializing it with another closure `init` if it has not been initialized yet.
        ///
        /// The return type `R` must be `'static` to prevent returning references
        /// to the inner value that could outlive the lock's acquisition.
        pub fn with_mut_spin_lock<Init, F, R>(&self, f: F, init: Init) -> R
        where
            Init: FnOnce() -> T,
            F: FnOnce(&mut T) -> R,
            R: 'static,
        {
            let guard = SpinLockGuard::acquire(&self.lock);

            #[allow(unsafe_code)]
            let result = {
                // SAFETY: the access is safe because we have exclusive access through the `self.lock`
                let inner = unsafe { &mut *self.value.get() };
                let val = inner.get_or_insert_with(init);
                f(val)
            };

            drop(guard);
            result
        }
    }
}

#[cfg_attr(not(feature = "random"), allow(dead_code))]
/// Convert a slice into an array of fixed size `N`,
/// padding with the default value of `T` if the slice is shorter than `N`.
pub fn slice_to_array_or_default<T, const N: usize>(slice: &[T]) -> [T; N]
where
    T: Default + Clone,
{
    slice_to_array_or(slice, T::default())
}

/// Convert a slice into an array of fixed size `N`,
/// padding with a specified value if the slice is shorter than `N`.
pub fn slice_to_array_or<T, const N: usize>(slice: &[T], padding: T) -> [T; N]
where
    T: Clone,
{
    use core::array;

    let mut arr = array::from_fn(|_| padding.clone());

    // determine the number of elements that are safe to clone
    // (minimum of slice length and array length)
    let items_to_clone = slice.len().min(arr.len());

    // clone the available data to the start of the array
    arr[..items_to_clone].clone_from_slice(&slice[..items_to_clone]);

    arr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_arr() {
        let slice1 = &[1_u8, 2];
        let slice2 = &[10_u8, 20, 30, 40];
        let slice3 = &[1_000_u16, 2_000, 3_000, 4_000, 5_000, 6_000];

        assert_eq!(slice_to_array_or_default::<_, 4>(slice1), [1, 2, 0, 0]);
        assert_eq!(slice_to_array_or_default::<_, 4>(slice2), [10, 20, 30, 40]);
        assert_eq!(
            slice_to_array_or_default::<_, 4>(slice3),
            [1_000, 2_000, 3_000, 4_000]
        );
    }
}
