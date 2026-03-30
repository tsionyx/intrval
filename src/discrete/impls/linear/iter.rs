//! Implementation of the [`Iterator`] trait(s) for [`LinearSpace`][super::LinearSpace].

use core::iter::FusedIterator;

use crate::helper::ValOrInf;

use super::{super::super::DiscreteOrdSet, LinearSpace};

#[derive(Debug, Copy, Clone)]
/// An iterator for a [`LinearSpace`] that can be either
/// forward (increasing) or backward (decreasing).
pub struct It<const INCREASING: bool, T, D> {
    space: LinearSpace<T, D>,
    current: Option<T>,
}

impl<T, D> LinearSpace<T, D>
where
    Self: DiscreteOrdSet<Point = T>,
{
    /// Convert [`LinearSpace`] into an ordinary `Iterator`
    /// moving forward from the [minimum value][Self::get_min] of the space.
    ///
    /// # Errors
    ///
    /// Returns `Err(self)` if the minimum value of the space is infinite, i.e.,
    /// if the space is unbounded from below.
    pub fn try_into_forward_iter(self) -> Result<It<true, T, D>, Self> {
        match self.get_min() {
            Some(ValOrInf::Val(start)) => Ok(It {
                space: self,
                current: Some(start),
            }),
            Some(ValOrInf::Inf) => Err(self),
            None => Ok(It {
                space: self,
                current: None,
            }),
        }
    }

    /// Convert [`LinearSpace`] into an ordinary `Iterator`
    /// moving forward from the `max(start, min_value)`.
    pub fn into_forward_iter_from(self, start: T) -> It<true, T, D>
    where
        T: Ord,
    {
        let adjusted_start = if self.contains(&start) {
            Ok(start)
        } else {
            self.get_next(&start).ok_or(&start)
        };

        let min = self.get_min().and_then(ValOrInf::into_val);
        let start = if let Some(min_val) = min {
            match adjusted_start {
                Ok(start) => Some(min_val.max(start)),
                Err(init) => (init <= &min_val).then_some(min_val),
            }
        } else {
            adjusted_start.ok()
        };

        It {
            space: self,
            current: start,
        }
    }

    /// Convert [`LinearSpace`] into a 'backward' `Iterator`
    /// moving backward from the [maximum value][Self::get_max] of the space.
    ///
    /// # Errors
    ///
    /// Returns `Err(self)` if the maximum value of the space is infinite, i.e.,
    /// if the space is unbounded from above.
    pub fn try_into_backward_iter(self) -> Result<It<false, T, D>, Self> {
        match self.get_max() {
            Some(ValOrInf::Val(end)) => Ok(It {
                space: self,
                current: Some(end),
            }),
            Some(ValOrInf::Inf) => Err(self),
            None => Ok(It {
                space: self,
                current: None,
            }),
        }
    }

    /// Convert [`LinearSpace`] into a 'backward' `Iterator`
    /// moving backward from the `min(end, max_value)`.
    pub fn into_backward_iter_up_to(self, end: T) -> It<false, T, D>
    where
        T: Ord,
    {
        let adjusted_end = if self.contains(&end) {
            Ok(end)
        } else {
            self.get_prev(&end).ok_or(&end)
        };

        let max = self.get_max().and_then(ValOrInf::into_val);
        let end = if let Some(max_val) = max {
            match adjusted_end {
                Ok(end) => Some(max_val.min(end)),
                Err(init) => (init >= &max_val).then_some(max_val),
            }
        } else {
            adjusted_end.ok()
        };

        It {
            space: self,
            current: end,
        }
    }
}

impl<T, D> Iterator for It<true, T, D>
where
    LinearSpace<T, D>: DiscreteOrdSet<Point = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current.take()?;

        let next = self.space.get_next(&current);
        self.current = next;

        Some(current)
    }
}

impl<T, D> Iterator for It<false, T, D>
where
    LinearSpace<T, D>: DiscreteOrdSet<Point = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current.take()?;

        let prev = self.space.get_prev(&current);
        self.current = prev;

        Some(current)
    }
}

impl<const INCREASING: bool, T, D> FusedIterator for It<INCREASING, T, D> where Self: Iterator {}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    extern crate alloc;

    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    use crate::interval;

    use super::*;

    #[test]
    fn closed_interval_iter_forward_and_backward() {
        let space = LinearSpace::try_bounded(interval!([12_u8, 42]), 5).unwrap();

        let full = [12, 17, 22, 27, 32, 37, 42];
        let full_rev = [42, 37, 32, 27, 22, 17, 12];

        assert_eq!(
            space.try_into_forward_iter().unwrap().collect::<Vec<_>>(),
            full,
        );
        for (start, expected) in [
            (0, &full[..]),
            (10, &full[..]),
            (12, &full[..]),
            (25, &full[3..]),
            (27, &full[3..]),
            (100, &[]),
            (200, &[]),
            (254, &[]),
        ] {
            assert_eq!(
                space.into_forward_iter_from(start).collect::<Vec<_>>(),
                expected,
            );
        }

        assert_eq!(
            space.try_into_backward_iter().unwrap().collect::<Vec<_>>(),
            full_rev,
        );
        for (end, expected) in [
            (100, &full_rev[..]),
            (45, &full_rev[..]),
            (42, &full_rev[..]),
            (20, &full_rev[5..]),
            (17, &full_rev[5..]),
            (12, &full_rev[6..]),
            (11, &[]),
            (0, &[]),
        ] {
            assert_eq!(
                space.into_backward_iter_up_to(end).collect::<Vec<_>>(),
                expected,
            );
        }
    }

    #[test]
    fn half_bounded_below_can_only_iterate_forward() {
        let space = LinearSpace::try_bounded(interval!(> -38_i8), 16).unwrap();

        let full = [-22, -6, 10, 26, 42, 58, 74, 90, 106, 122];
        let full_rev = [122, 106, 90, 74, 58, 42, 26, 10, -6, -22];

        assert_eq!(
            space.try_into_forward_iter().unwrap().collect::<Vec<_>>(),
            full,
        );
        for (start, expected) in [
            (-100, &full[..]),
            (-38, &full[..]),
            (-30, &full[..]),
            (-22, &full[..]),
            (-21, &full[1..]),
            (0, &full[2..]),
            (58, &full[5..]),
            (100, &full[8..]),
            (121, &full[9..]),
            (122, &full[9..]),
            (123, &[]),
        ] {
            assert_eq!(
                space.into_forward_iter_from(start).collect::<Vec<_>>(),
                expected,
            );
        }

        let _err = space.try_into_backward_iter().unwrap_err();
        for (end, expected) in [
            (127, &full_rev[..]),
            (122, &full_rev[..]),
            (121, &full_rev[1..]),
            (100, &full_rev[2..]),
            (58, &full_rev[4..]),
            (0, &full_rev[8..]),
            (-10, &full_rev[9..]),
            (-22, &full_rev[9..]),
            (-30, &[]),
            (-37, &[]),
            (-38, &[]),
            (-40, &[]),
            (-100, &[]),
        ] {
            assert_eq!(
                space.into_backward_iter_up_to(end).collect::<Vec<_>>(),
                expected,
            );
        }
    }

    #[test]
    fn half_bounded_above_can_only_iterate_backward() {
        let space = LinearSpace::try_bounded(interval!(<= 66_u8), 17).unwrap();

        let full = [15, 32, 49, 66];
        let full_rev = [66, 49, 32, 15];

        let _err = space.try_into_forward_iter().unwrap_err();
        for (start, expected) in [
            (0, &full[..]),
            (14, &full[..]),
            (15, &full[..]),
            (16, &full[1..]),
            (20, &full[1..]),
            (40, &full[2..]),
            (49, &full[2..]),
            (65, &full[3..]),
            (66, &full[3..]),
            (67, &[]),
            (200, &[]),
            (255, &[]),
        ] {
            assert_eq!(
                space.into_forward_iter_from(start).collect::<Vec<_>>(),
                expected,
            );
        }

        assert_eq!(
            space.try_into_backward_iter().unwrap().collect::<Vec<_>>(),
            full_rev,
        );
        for (end, expected) in [
            (255, &full_rev[..]),
            (100, &full_rev[..]),
            (67, &full_rev[..]),
            (66, &full_rev[..]),
            (65, &full_rev[1..]),
            (50, &full_rev[1..]),
            (49, &full_rev[1..]),
            (48, &full_rev[2..]),
            (20, &full_rev[3..]),
            (15, &full_rev[3..]),
            (14, &[]),
            (0, &[]),
        ] {
            assert_eq!(
                space.into_backward_iter_up_to(end).collect::<Vec<_>>(),
                expected,
            );
        }
    }

    #[test]
    fn unbounded_does_not_iterate() {
        let space = LinearSpace::<u8, u8>::try_new(1).unwrap();

        let _err = space.try_into_forward_iter().unwrap_err();
        for start in [0, 14, 15, 16, 20, 40, 49, 65, 66, 67, 200, 255] {
            let total: Vec<_> = space.into_forward_iter_from(start).collect();
            assert!(!total.is_empty());
            let expected: Vec<_> = (start..=255).collect();
            assert_eq!(total, expected);
        }

        let _err = space.try_into_backward_iter().unwrap_err();
        for end in [255, 100, 67, 66, 65, 50, 49, 48, 20, 15, 14, 0] {
            let total: Vec<_> = space.into_backward_iter_up_to(end).collect();
            assert!(!total.is_empty());
            let mut expected: Vec<_> = (0..=end).collect();
            expected.reverse();
            assert_eq!(total, expected);
        }
    }
}
