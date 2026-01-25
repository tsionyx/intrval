use core::{
    cmp::Ordering,
    ops::{Add, Sub},
};

use crate::{
    bounds::Endpoint,
    helper::{OneOrPair, ValOrInf},
};

use super::{
    super::super::DiscreteOrdSet,
    ops_traits::{IntDiv, Linear},
    LinearSpace,
};

impl<T> DiscreteOrdSet for LinearSpace<T>
where
    T: PartialOrd + Clone + Linear + IntDiv,
{
    type Point = T;

    fn get_min(&self) -> Option<ValOrInf<Self::Point>> {
        let (lower, upper) = self.bounds.as_ref_bounds().ok()?;
        match lower {
            // if the lower bound is inclusive,
            // the space's first point is that `Included` value
            Endpoint::Included(lower) => Some(ValOrInf::Val(lower.clone())),
            Endpoint::Excluded(lower) => {
                let first_point = lower.clone() + self.step.clone();
                // the space is empty if the first valid point goes beyond `upper`
                (upper >= &first_point).then_some(ValOrInf::Val(first_point))
            }
            Endpoint::Infinite => Some(ValOrInf::Inf),
        }
    }

    fn get_max(&self) -> Option<ValOrInf<Self::Point>> {
        let min = self.get_min()?;
        let (_, upper) = self.bounds.as_ref_bounds().ok()?;

        #[allow(clippy::option_if_let_else)]
        let max = if let Some(upper_val) = upper.bound_val().copied() {
            // the point to start counting from
            let near_max = {
                let origin = min.into_val().unwrap_or_else(|| self.step.clone());
                find_stepped(upper_val.clone(), origin, &self.step)
            };

            let max_point =
                find_best_step(near_max, &self.step, Direction::Down, |max| upper >= max);

            ValOrInf::Val(max_point)
        } else {
            ValOrInf::Inf
        };
        Some(max)
    }

    fn get_nearest(&self, point: &Self::Point) -> Option<OneOrPair<Self::Point>> {
        let min = self.get_min()?;
        if let ValOrInf::Val(min_val) = &min {
            if point <= min_val {
                return Some(OneOrPair::One(min_val.clone()));
            }
        }

        let max = self.get_max()?;
        if let ValOrInf::Val(max_val) = &max {
            if point >= max_val {
                return Some(OneOrPair::One(max_val.clone()));
            }
        }

        // the point should be within the bounds now;
        // if it is not, it is preferable do not continue
        if !self.bounds.contains(point) {
            return None;
        }

        let point_stepped = {
            let origin = min.into_val().unwrap_or_else(|| self.step.clone());
            find_stepped(point.clone(), origin, &self.step)
        };

        let lower = find_best_step(
            point_stepped.clone(),
            &self.step,
            Direction::Down,
            |lower| lower <= point,
        );
        let upper = find_best_step(point_stepped, &self.step, Direction::Up, |upper| {
            upper >= point
        });

        Some(OneOrPair::Pair((lower, upper)))
    }

    fn get_next(&self, point: &Self::Point) -> Option<Self::Point> {
        let adjust_nearest = |nearest: T| match nearest.partial_cmp(point)? {
            Ordering::Greater => Some(nearest),
            Ordering::Equal => Some(nearest + self.step.clone()),
            Ordering::Less => None,
        };

        match self.get_nearest(point)? {
            OneOrPair::One(nearest) => adjust_nearest(nearest),
            OneOrPair::Pair((lower, upper)) => {
                adjust_nearest(upper).or_else(|| adjust_nearest(lower))
            }
        }
    }

    fn get_prev(&self, point: &Self::Point) -> Option<Self::Point> {
        let adjust_nearest = |nearest: T| match nearest.partial_cmp(point)? {
            Ordering::Greater => None,
            Ordering::Equal => Some(nearest - self.step.clone()),
            Ordering::Less => Some(nearest),
        };

        match self.get_nearest(point)? {
            OneOrPair::One(nearest) => adjust_nearest(nearest),
            OneOrPair::Pair((lower, upper)) => {
                adjust_nearest(lower).or_else(|| adjust_nearest(upper))
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Direction {
    /// Move down until the condition is satisfied.
    Down,
    /// Move up until the condition is satisfied.
    Up,
}

/// Find the point close to `x`, starting from `origin`
/// on a distance of integer number of `step`-s.
fn find_stepped<T>(x: T, origin: T, step: &T) -> T
where
    T: Clone + PartialOrd + Linear + IntDiv,
{
    let (direction, distance) = if x > origin {
        (true, x - origin.clone())
    } else {
        (false, origin.clone() - x)
    };
    let no_steps = distance.int_div(step.clone());
    if direction {
        origin + step.clone() * no_steps
    } else {
        origin - step.clone() * no_steps
    }
}

/// Find the point starting from `start` and moving in `step` increments/decrements
/// (depending on `dir`), using the `condition` predicate to locate the boundary:
/// first move in `dir` until `condition` is satisfied, then move one step in the
/// opposite direction until the next point would not satisfy `condition`, and
/// return the last point for which `condition` holds.
fn find_best_step<T, F>(start: T, step: &T, dir: Direction, mut condition: F) -> T
where
    T: Clone + Add<Output = T> + Sub<Output = T>,
    F: FnMut(&T) -> bool,
{
    let mut current = start;
    while !condition(&current) {
        // move down/up to find the condition is satisfied
        let next = match dir {
            Direction::Down => current.clone() - step.clone(),
            Direction::Up => current.clone() + step.clone(),
        };
        current = next;
    }

    loop {
        // move up/down until the condition with
        // the next point is not satisfied anymore
        let next = match dir {
            Direction::Down => current.clone() + step.clone(),
            Direction::Up => current.clone() - step.clone(),
        };

        if !condition(&next) {
            return current;
        }
        current = next;
    }
}

#[cfg(test)]
mod tests {
    use crate::{interval, Interval};

    use super::{
        OneOrPair::{One as I, Pair as P},
        *,
    };

    type Int = i16;

    fn even_numbers_space(bounds: Interval<Int>) -> LinearSpace<Int> {
        LinearSpace::try_bounded(bounds, 2).unwrap()
    }

    #[test]
    fn empty_interval() {
        let interval = even_numbers_space(interval!([10, 1]));
        assert!(interval.get_min().is_none());
        assert!(interval.get_max().is_none());

        assert!(interval.get_nearest(&-10).is_none());
        assert!(interval.get_nearest(&-5).is_none());
        assert!(interval.get_nearest(&-1).is_none());
        assert!(interval.get_nearest(&0).is_none());
        assert!(interval.get_nearest(&1).is_none());
        assert!(interval.get_nearest(&2).is_none());
        assert!(interval.get_nearest(&6).is_none());
        assert!(interval.get_nearest(&7).is_none());
        assert!(interval.get_nearest(&9).is_none());
        assert!(interval.get_nearest(&10).is_none());
        assert!(interval.get_nearest(&11).is_none());
        assert!(interval.get_nearest(&20).is_none());
    }

    #[test]
    fn closed_interval() {
        let interval = even_numbers_space(interval!([0, 10]));
        assert_eq!(interval.get_min().unwrap(), ValOrInf::Val(0));
        assert_eq!(interval.get_max().unwrap(), ValOrInf::Val(10));

        assert_eq!(interval.get_nearest(&-10).unwrap(), I(0));
        assert_eq!(interval.get_nearest(&-5).unwrap(), I(0));
        assert_eq!(interval.get_nearest(&-1).unwrap(), I(0));
        assert_eq!(interval.get_nearest(&0).unwrap(), I(0));
        assert_eq!(interval.get_nearest(&1).unwrap(), P((0, 2)));
        assert_eq!(interval.get_nearest(&2).unwrap(), P((2, 2)));
        assert_eq!(interval.get_nearest(&6).unwrap(), P((6, 6)));
        assert_eq!(interval.get_nearest(&7).unwrap(), P((6, 8)));
        assert_eq!(interval.get_nearest(&9).unwrap(), P((8, 10)));
        assert_eq!(interval.get_nearest(&10).unwrap(), I(10));
        assert_eq!(interval.get_nearest(&11).unwrap(), I(10));
        assert_eq!(interval.get_nearest(&20).unwrap(), I(10));
    }

    #[test]
    fn left_open_interval() {
        let interval = even_numbers_space(interval!((0, =10)));
        assert_eq!(interval.get_min().unwrap(), ValOrInf::Val(2));
        assert_eq!(interval.get_max().unwrap(), ValOrInf::Val(10));

        assert_eq!(interval.get_nearest(&-10).unwrap(), I(2));
        assert_eq!(interval.get_nearest(&-5).unwrap(), I(2));
        assert_eq!(interval.get_nearest(&-1).unwrap(), I(2));
        assert_eq!(interval.get_nearest(&0).unwrap(), I(2));
        assert_eq!(interval.get_nearest(&1).unwrap(), I(2));
        assert_eq!(interval.get_nearest(&2).unwrap(), I(2));
        assert_eq!(interval.get_nearest(&6).unwrap(), P((6, 6)));
        assert_eq!(interval.get_nearest(&7).unwrap(), P((6, 8)));
        assert_eq!(interval.get_nearest(&9).unwrap(), P((8, 10)));
        assert_eq!(interval.get_nearest(&10).unwrap(), I(10));
        assert_eq!(interval.get_nearest(&11).unwrap(), I(10));
        assert_eq!(interval.get_nearest(&20).unwrap(), I(10));
    }

    #[test]
    fn right_open_interval() {
        let interval = even_numbers_space(interval!((=0, 10)));
        assert_eq!(interval.get_min().unwrap(), ValOrInf::Val(0));
        assert_eq!(interval.get_max().unwrap(), ValOrInf::Val(8));

        assert_eq!(interval.get_nearest(&-10).unwrap(), I(0));
        assert_eq!(interval.get_nearest(&-5).unwrap(), I(0));
        assert_eq!(interval.get_nearest(&-1).unwrap(), I(0));
        assert_eq!(interval.get_nearest(&0).unwrap(), I(0));
        assert_eq!(interval.get_nearest(&1).unwrap(), P((0, 2)));
        assert_eq!(interval.get_nearest(&2).unwrap(), P((2, 2)));
        assert_eq!(interval.get_nearest(&6).unwrap(), P((6, 6)));
        assert_eq!(interval.get_nearest(&7).unwrap(), P((6, 8)));
        assert_eq!(interval.get_nearest(&9).unwrap(), I(8));
        assert_eq!(interval.get_nearest(&10).unwrap(), I(8));
        assert_eq!(interval.get_nearest(&11).unwrap(), I(8));
        assert_eq!(interval.get_nearest(&20).unwrap(), I(8));
    }

    #[test]
    fn open_interval() {
        let interval = even_numbers_space(interval!((0, 10)));
        assert_eq!(interval.get_min().unwrap(), ValOrInf::Val(2));
        assert_eq!(interval.get_max().unwrap(), ValOrInf::Val(8));

        assert_eq!(interval.get_nearest(&-10).unwrap(), I(2));
        assert_eq!(interval.get_nearest(&-5).unwrap(), I(2));
        assert_eq!(interval.get_nearest(&-1).unwrap(), I(2));
        assert_eq!(interval.get_nearest(&0).unwrap(), I(2));
        assert_eq!(interval.get_nearest(&1).unwrap(), I(2));
        assert_eq!(interval.get_nearest(&2).unwrap(), I(2));
        assert_eq!(interval.get_nearest(&6).unwrap(), P((6, 6)));
        assert_eq!(interval.get_nearest(&7).unwrap(), P((6, 8)));
        assert_eq!(interval.get_nearest(&9).unwrap(), I(8));
        assert_eq!(interval.get_nearest(&10).unwrap(), I(8));
        assert_eq!(interval.get_nearest(&11).unwrap(), I(8));
        assert_eq!(interval.get_nearest(&20).unwrap(), I(8));
    }

    #[test]
    fn left_inf_closed_interval() {
        let interval = even_numbers_space(interval!(<=10));
        assert_eq!(interval.get_min().unwrap(), ValOrInf::Inf);
        assert_eq!(interval.get_max().unwrap(), ValOrInf::Val(10));

        assert_eq!(interval.get_nearest(&-10).unwrap(), P((-10, -10)));
        assert_eq!(interval.get_nearest(&-5).unwrap(), P((-6, -4)));
        assert_eq!(interval.get_nearest(&-1).unwrap(), P((-2, 0)));
        assert_eq!(interval.get_nearest(&0).unwrap(), P((0, 0)));
        assert_eq!(interval.get_nearest(&1).unwrap(), P((0, 2)));
        assert_eq!(interval.get_nearest(&2).unwrap(), P((2, 2)));
        assert_eq!(interval.get_nearest(&6).unwrap(), P((6, 6)));
        assert_eq!(interval.get_nearest(&7).unwrap(), P((6, 8)));
        assert_eq!(interval.get_nearest(&9).unwrap(), P((8, 10)));
        assert_eq!(interval.get_nearest(&10).unwrap(), I(10));
        assert_eq!(interval.get_nearest(&11).unwrap(), I(10));
        assert_eq!(interval.get_nearest(&20).unwrap(), I(10));
    }

    #[test]
    fn left_inf_open_interval() {
        let interval = even_numbers_space(interval!(< 10));
        assert_eq!(interval.get_min().unwrap(), ValOrInf::Inf);
        assert_eq!(interval.get_max().unwrap(), ValOrInf::Val(8));

        assert_eq!(interval.get_nearest(&-10).unwrap(), P((-10, -10)));
        assert_eq!(interval.get_nearest(&-5).unwrap(), P((-6, -4)));
        assert_eq!(interval.get_nearest(&-1).unwrap(), P((-2, 0)));
        assert_eq!(interval.get_nearest(&0).unwrap(), P((0, 0)));
        assert_eq!(interval.get_nearest(&1).unwrap(), P((0, 2)));
        assert_eq!(interval.get_nearest(&2).unwrap(), P((2, 2)));
        assert_eq!(interval.get_nearest(&6).unwrap(), P((6, 6)));
        assert_eq!(interval.get_nearest(&7).unwrap(), P((6, 8)));
        assert_eq!(interval.get_nearest(&9).unwrap(), I(8));
        assert_eq!(interval.get_nearest(&10).unwrap(), I(8));
        assert_eq!(interval.get_nearest(&11).unwrap(), I(8));
        assert_eq!(interval.get_nearest(&20).unwrap(), I(8));
    }

    #[test]
    fn right_inf_closed_interval() {
        let interval = even_numbers_space(interval!(>= 0));
        assert_eq!(interval.get_min().unwrap(), ValOrInf::Val(0));
        assert_eq!(interval.get_max().unwrap(), ValOrInf::Inf);

        assert_eq!(interval.get_nearest(&-10).unwrap(), I(0));
        assert_eq!(interval.get_nearest(&-5).unwrap(), I(0));
        assert_eq!(interval.get_nearest(&-1).unwrap(), I(0));
        assert_eq!(interval.get_nearest(&0).unwrap(), I(0));
        assert_eq!(interval.get_nearest(&1).unwrap(), P((0, 2)));
        assert_eq!(interval.get_nearest(&2).unwrap(), P((2, 2)));
        assert_eq!(interval.get_nearest(&6).unwrap(), P((6, 6)));
        assert_eq!(interval.get_nearest(&7).unwrap(), P((6, 8)));
        assert_eq!(interval.get_nearest(&9).unwrap(), P((8, 10)));
        assert_eq!(interval.get_nearest(&10).unwrap(), P((10, 10)));
        assert_eq!(interval.get_nearest(&11).unwrap(), P((10, 12)));
        assert_eq!(interval.get_nearest(&20).unwrap(), P((20, 20)));
    }

    #[test]
    fn right_inf_open_interval() {
        let interval = even_numbers_space(interval!(> 0));
        assert_eq!(interval.get_min().unwrap(), ValOrInf::Val(2));
        assert_eq!(interval.get_max().unwrap(), ValOrInf::Inf);

        assert_eq!(interval.get_nearest(&-10).unwrap(), I(2));
        assert_eq!(interval.get_nearest(&-5).unwrap(), I(2));
        assert_eq!(interval.get_nearest(&-1).unwrap(), I(2));
        assert_eq!(interval.get_nearest(&0).unwrap(), I(2));
        assert_eq!(interval.get_nearest(&1).unwrap(), I(2));
        assert_eq!(interval.get_nearest(&2).unwrap(), I(2));
        assert_eq!(interval.get_nearest(&6).unwrap(), P((6, 6)));
        assert_eq!(interval.get_nearest(&7).unwrap(), P((6, 8)));
        assert_eq!(interval.get_nearest(&9).unwrap(), P((8, 10)));
        assert_eq!(interval.get_nearest(&10).unwrap(), P((10, 10)));
        assert_eq!(interval.get_nearest(&11).unwrap(), P((10, 12)));
        assert_eq!(interval.get_nearest(&20).unwrap(), P((20, 20)));
    }

    #[test]
    fn full_interval() {
        let interval = even_numbers_space(interval!(U));
        assert_eq!(interval.get_min().unwrap(), ValOrInf::Inf);
        assert_eq!(interval.get_max().unwrap(), ValOrInf::Inf);

        assert_eq!(interval.get_nearest(&-10).unwrap(), P((-10, -10)));
        assert_eq!(interval.get_nearest(&-5).unwrap(), P((-6, -4)));
        assert_eq!(interval.get_nearest(&-1).unwrap(), P((-2, 0)));
        assert_eq!(interval.get_nearest(&0).unwrap(), P((0, 0)));
        assert_eq!(interval.get_nearest(&1).unwrap(), P((0, 2)));
        assert_eq!(interval.get_nearest(&2).unwrap(), P((2, 2)));
        assert_eq!(interval.get_nearest(&6).unwrap(), P((6, 6)));
        assert_eq!(interval.get_nearest(&7).unwrap(), P((6, 8)));
        assert_eq!(interval.get_nearest(&9).unwrap(), P((8, 10)));
        assert_eq!(interval.get_nearest(&10).unwrap(), P((10, 10)));
        assert_eq!(interval.get_nearest(&11).unwrap(), P((10, 12)));
        assert_eq!(interval.get_nearest(&20).unwrap(), P((20, 20)));
    }
}
