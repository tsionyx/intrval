//! Auxiliary types and functions.

use crate::traits::Zero;

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

    /// Unwrap the single value if it is wrapped into the [`OneOrPair::One`],
    /// otherwise fold the pair of values into a single one using the provided function.
    pub fn single_or_fold<F>(self, f: F) -> T
    where
        F: FnOnce(T, T) -> T,
    {
        match self {
            Self::One(v) => v,
            Self::Pair((v1, v2)) => f(v1, v2),
        }
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
    // SAFETY: We ensure exclusive access through the `self.lock` atomic boolean,
    // and `T: Send` guarantees that the inner value may be safely accessed across
    // threads. The `with_mut_spin_lock` method requires `R: 'static` to prevent
    // returning references to the inner value that could outlive the lock.
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
                let rng = inner.get_or_insert_with(init);
                f(rng)
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

    let mut array = array::from_fn(|_| padding.clone());

    // determine the number of elements that are safe to clone
    // (minimum of slice length and array length)
    let items_to_clone = slice.len().min(array.len());

    // clone the available data to the start of the array
    array[..items_to_clone].clone_from_slice(&slice[..items_to_clone]);

    array
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
