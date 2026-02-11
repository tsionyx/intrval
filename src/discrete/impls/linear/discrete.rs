use core::cmp::Ordering;

use crate::{
    bounds::Endpoint,
    helper::{OneOrPair, ValOrInf},
    traits::{IntDiv, MonotonicLinear},
};

use super::{super::super::DiscreteOrdSet, LinearSpace};

impl<T> LinearSpace<T>
where
    T: PartialOrd + Clone + MonotonicLinear,
{
    fn min_value(&self) -> Option<ValOrInf<T>> {
        let (lower, upper) = self.bounds.as_ref_bounds().ok()?;
        match lower {
            // if the lower bound is inclusive,
            // the space's first point is that `Included` value
            Endpoint::Included(lower) => Some(ValOrInf::Val(lower.clone())),
            Endpoint::Excluded(lower) => {
                let first_point = lower.clone().monotonic_add(self.step.clone())?;
                // the space is empty if the first valid point goes beyond `upper`
                (upper >= &first_point).then_some(ValOrInf::Val(first_point))
            }
            Endpoint::Infinite => Some(ValOrInf::Inf),
        }
    }

    fn max_value(&self) -> Option<ValOrInf<T>>
    where
        T: IntDiv,
    {
        let min = self.min_value()?;
        let (lower, upper) = self.bounds.as_ref_bounds().ok()?;

        match min.into_val() {
            Some(min_v) => {
                #[allow(clippy::option_if_let_else)]
                let max = if let Some(upper_val) = upper.bound_val().copied() {
                    // the point to start counting from
                    let near_max = find_stepped(upper_val.clone(), min_v.clone(), &self.step);

                    let max_point =
                        find_best_step(near_max, &self.step, Direction::Down, |max| upper >= max);
                    // if `max_point` is underflowed, use the `min_v` as it has a finite value
                    ValOrInf::Val(max_point.unwrap_or(min_v))
                } else {
                    ValOrInf::Inf
                };
                Some(max)
            }
            None => {
                // if the min is infinite, the max depends solely on upper bound
                match upper {
                    // if the upper bound is inclusive,
                    // the space's max point is that `Included` value
                    Endpoint::Included(upper) => Some(ValOrInf::Val(upper.clone())),
                    Endpoint::Excluded(upper) => {
                        let last_point = upper.clone().monotonic_sub(self.step.clone())?;
                        // the space is empty if the last valid point goes beyond `lower`
                        (lower <= &last_point).then_some(ValOrInf::Val(last_point))
                    }
                    Endpoint::Infinite => Some(ValOrInf::Inf),
                }
            }
        }
    }
}

impl<T> DiscreteOrdSet for LinearSpace<T>
where
    T: PartialOrd + Clone + MonotonicLinear + IntDiv,
{
    type Point = T;

    fn contains(&self, point: &Self::Point) -> bool {
        // the point should be within the bounds
        if !self.bounds.contains(point) {
            return false;
        }

        let Some(min) = self.get_min() else {
            return false;
        };

        let Some(max) = self.get_max() else {
            return false;
        };

        // the point should be greater than or equal to the minimum
        if let ValOrInf::Val(ref min_val) = min {
            if point < min_val {
                return false;
            }
        }

        // the point should be less than or equal to the maximum
        if let ValOrInf::Val(ref max_val) = max {
            if point > max_val {
                return false;
            }
        }

        // the point should be a multiple of `step` from some origin
        let origin = min
            .into_val()
            .or_else(|| max.into_val())
            .unwrap_or_else(|| self.step.clone());
        let stepped = find_stepped(point.clone(), origin, &self.step);
        stepped == *point
    }

    fn get_min(&self) -> Option<ValOrInf<Self::Point>> {
        self.min_value().filter(|min| {
            if min.is_finite() {
                true
            } else {
                // extra check: if minimum is infinite,
                // but there is no maximum, the space is empty as well
                self.max_value().is_some()
            }
        })
    }

    fn get_max(&self) -> Option<ValOrInf<Self::Point>> {
        self.max_value().filter(|max| {
            if max.is_finite() {
                true
            } else {
                // extra check: if maximum is infinite,
                // but there is no minimum, the space is empty as well
                self.min_value().is_some()
            }
        })
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
            let origin = min
                .into_val()
                .or_else(|| max.into_val())
                .unwrap_or_else(|| self.step.clone());
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

        match (lower, upper) {
            (Some(l), Some(u)) => Some(OneOrPair::Pair((l, u))),
            (Some(x), None) | (None, Some(x)) => Some(OneOrPair::One(x)),
            (None, None) => None,
        }
    }

    fn get_next(&self, point: &Self::Point) -> Option<Self::Point> {
        let adjust_nearest = |nearest: T| match nearest.partial_cmp(point)? {
            Ordering::Greater => Some(nearest),
            Ordering::Equal => nearest.monotonic_add(self.step.clone()),
            Ordering::Less => None,
        };

        self.get_nearest(point)?
            .fold(adjust_nearest, |lower, upper| {
                adjust_nearest(upper).or_else(|| adjust_nearest(lower))
            })
    }

    fn get_prev(&self, point: &Self::Point) -> Option<Self::Point> {
        let adjust_nearest = |nearest: T| match nearest.partial_cmp(point)? {
            Ordering::Greater => None,
            Ordering::Equal => nearest.monotonic_sub(self.step.clone()),
            Ordering::Less => Some(nearest),
        };

        self.get_nearest(point)?
            .fold(adjust_nearest, |lower, upper| {
                adjust_nearest(lower).or_else(|| adjust_nearest(upper))
            })
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
    T: Clone + PartialOrd + MonotonicLinear + IntDiv,
{
    let direction = x > origin;

    let (mut x, mut additional_steps) = (x, 0_usize);
    let distance = loop {
        if direction {
            let dist = x.clone().monotonic_sub(origin.clone());
            if let Some(d) = dist {
                break d;
            }

            let decr_x = x.monotonic_sub(step.clone());
            if let Some(reduced_x) = decr_x {
                x = reduced_x;
                additional_steps += 1;
            } else {
                // if subtracting a `step` from `x` overflowed, we can just return the `origin`
                return origin;
            }
        } else {
            let dist = origin.clone().monotonic_sub(x.clone());
            if let Some(d) = dist {
                break d;
            }

            let incr_x = x.monotonic_add(step.clone());
            if let Some(increased_x) = incr_x {
                x = increased_x;
                additional_steps += 1;
            } else {
                // if summing a `step` to `x` overflowed, we can just return the `origin`
                return origin;
            }
        }
    };

    let no_steps = distance.int_div(step.clone());
    let delta = step.clone() * no_steps;
    if direction {
        origin.clone().monotonic_add(delta).map(|mut x| {
            for _ in 0..additional_steps {
                let next = x.clone().monotonic_add(step.clone());
                if let Some(next) = next {
                    x = next;
                } else {
                    break;
                }
            }
            x
        })
    } else {
        origin.clone().monotonic_sub(delta).map(|mut x| {
            for _ in 0..additional_steps {
                let prev = x.clone().monotonic_sub(step.clone());
                if let Some(prev) = prev {
                    x = prev;
                } else {
                    break;
                }
            }
            x
        })
    }
    .unwrap_or(origin)
}

/// Find the point starting from `start` and moving in `step` increments/decrements
/// (depending on `dir`), using the `condition` predicate to locate the boundary:
/// first move in `dir` until `condition` is satisfied, then move one step in the
/// opposite direction until the next point would not satisfy `condition`, and
/// return the last point for which `condition` holds.
fn find_best_step<T, F>(start: T, step: &T, dir: Direction, mut condition: F) -> Option<T>
where
    T: Clone + MonotonicLinear,
    F: FnMut(&T) -> bool,
{
    let mut current = start;
    while !condition(&current) {
        // move down/up to find the condition is satisfied
        let next = match dir {
            Direction::Down => current.clone().monotonic_sub(step.clone()),
            Direction::Up => current.clone().monotonic_add(step.clone()),
        };
        current = next?;
    }

    loop {
        // move up/down until the condition with
        // the next point is not satisfied anymore
        let next = match dir {
            Direction::Down => current.clone().monotonic_add(step.clone()),
            Direction::Up => current.clone().monotonic_sub(step.clone()),
        };

        if let Some(next) = next {
            if condition(&next) {
                current = next;
                continue;
            }
        }

        break Some(current);
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

    #[test]
    fn overflow_while_computing_min() {
        let space = LinearSpace::try_bounded(interval!(>250), 6_u8).unwrap();
        assert!(space.get_min().is_none());
    }

    #[test]
    fn overflow_while_computing_max() {
        let space = LinearSpace::try_bounded(interval!([250, 255]), 6_u8).unwrap();
        let min = space.get_min().unwrap();
        assert_eq!(min.get_val().unwrap(), &250);
        let max = space.get_max().unwrap();
        assert_eq!(max.get_val().unwrap(), &250);
    }

    #[test]
    fn overflow_while_computing_nearest() {
        let space = LinearSpace::try_bounded(interval!(>=250), 6_u8).unwrap();
        let min = space.get_min().unwrap();
        assert_eq!(min.get_val().unwrap(), &250);
        let max = space.get_max().unwrap();
        assert!(!max.is_finite());

        let nearest = space.get_nearest(&253).unwrap();
        assert_eq!(nearest.into_single().unwrap(), 250);
    }

    #[test]
    fn underflow_while_computing_nearest() {
        let space = LinearSpace::try_bounded(interval!(<=10), 10_u8).unwrap();
        let min = space.get_min().unwrap();
        assert!(!min.is_finite());
        let max = space.get_max().unwrap();
        assert_eq!(max.get_val().unwrap(), &10);

        let nearest = space.get_nearest(&0).unwrap();
        assert_eq!(nearest.into_pair().unwrap(), (0, 0));
    }

    #[test]
    fn regression_for_included() {
        let space_neg_inf = LinearSpace::try_bounded(interval!(<= -100), 29_i8).unwrap();
        let min = space_neg_inf.get_min().unwrap();
        assert!(!min.is_finite());
        let max = space_neg_inf.get_max().unwrap();
        assert_eq!(max.get_val().unwrap(), &-100);

        let space_pos_inf = LinearSpace::try_bounded(interval!(>=100), 28_i8).unwrap();
        let min = space_pos_inf.get_min().unwrap();
        assert_eq!(min.get_val().unwrap(), &100);
        let max = space_pos_inf.get_max().unwrap();
        assert!(!max.is_finite());

        for i in i8::MIN..=i8::MAX {
            let nearest = space_neg_inf.get_nearest(&i).unwrap();
            assert_eq!(nearest.into_single().unwrap(), -100, "{i}");

            let nearest = space_pos_inf.get_nearest(&i).unwrap();
            assert_eq!(nearest.into_single().unwrap(), 100, "{i}");
        }
    }
}

#[cfg(all(test, feature = "arbitrary"))]
mod prop_test {
    extern crate alloc;

    #[allow(unused_imports)] // will be used internally by `prop_assert` macros
    use alloc::format;

    use proptest::prelude::*;

    use crate::{traits::Zero, Interval};

    use super::*;

    impl<T> Arbitrary for LinearSpace<T>
    where
        T: Clone + Arbitrary + Zero + 'static,
    {
        type Parameters = <Interval<T> as Arbitrary>::Parameters;
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
            (
                Interval::<T>::arbitrary_with(args),
                T::arbitrary().prop_filter("only positive step", |x| {
                    x.cmp_zero() == Some(Ordering::Greater)
                }),
            )
                .prop_map(|(range, step)| {
                    Self::try_bounded(range, step)
                        .expect("The positiveness of step was checked before")
                })
                .boxed()
        }
    }

    // TODO: check for `ordered_float::OrderedFloat<f32>`
    type Int = i8;

    proptest! {
        // https://proptest-rs.github.io/proptest/proptest/tutorial/config.html
        #![proptest_config(ProptestConfig::with_cases(8000))]

        #[test]
        fn does_not_panic(space: LinearSpace<Int>, point: Int) {
            let _ = space.get_min();
            let _ = space.get_max();
            let _ = space.get_nearest(&point);
            let _ = space.get_next(&point);
            let _ = space.get_prev(&point);
        }

        #[test]
        fn having_min_or_max_equiv_non_empty(space: LinearSpace<Int>) {
            let min = space.get_min();
            prop_assert_eq!(min.is_none(), space.is_empty());

            let max = space.get_max();
            prop_assert_eq!(max.is_none(), space.is_empty());
        }

        #[test]
        fn having_min_equiv_having_max(space: LinearSpace<Int>) {
            let max = space.get_max();
            if let Some(min) = space.get_min() {
                let max = max.unwrap();
                if let (Some(min_v), Some(max_v)) = (min.into_val(), max.into_val()) {
                    prop_assert!(max_v >= min_v);
                }
            }
            else {
                prop_assert!(max.is_none());
            }
        }

        #[test]
        fn having_nearest_equiv_non_empty(space: LinearSpace<Int>, point: Int) {
            let nearest = space.get_nearest(&point);
            prop_assert_eq!(nearest.is_none(), space.is_empty());
        }

        #[test]
        fn nearest_always_contained(space: LinearSpace<Int>, point: Int) {
            let nearest = space.get_nearest(&point);
            if let Some(nearest) = nearest {
                match nearest {
                    OneOrPair::One(nearest) => prop_assert!(space.contains(&nearest)),
                    OneOrPair::Pair((lower, upper)) => {
                        prop_assert!(space.contains(&lower));
                        prop_assert!(space.contains(&upper));
                    }
                }
            }
        }

        #[test]
        fn next_and_prev_always_contained(space: LinearSpace<Int>, point: Int) {
            if let Some(next) = space.get_next(&point) {
                prop_assert!(space.contains(&next));
            }
            if let Some(prev) = space.get_prev(&point) {
                prop_assert!(space.contains(&prev));
            }
        }

        #[test]
        fn empty_is_either_empty_bounds_or_large_step_or_open(space in any::<LinearSpace<Int>>()
            .prop_filter("select empty spaces only", LinearSpace::is_empty)) {
                let bounds = space.bounds().as_ref();
                let size = bounds.len().into_diff();
                let is_open = bounds.into_interior() == bounds;

                let cond1 = bounds.is_empty();
                let cond2 = is_open && (Some(space.step()) == size.as_ref());
                let cond3 = || {
                    #[allow(clippy::option_if_let_else)]

                    if let Some(size) = size {
                        space.step() > &size
                    } else {
                        use crate::IntoBounds as _;

                        let (a, b) = bounds.into_bounds().unwrap();
                        let left_inf = (!a.is_finite()) &&
                            (b.bound_val().copied().unwrap().monotonic_sub(*space.step()).is_none());
                        let right_inf =(!b.is_finite()) &&
                            (a.bound_val().copied().unwrap().monotonic_add(*space.step()).is_none());

                        is_open && (left_inf || right_inf)
                    }
                };

                prop_assert!(cond1 || cond2 || cond3());
        }

        #[test]
        fn next_exists_when_non_empty_and_step_from_max(space in any::<LinearSpace<Int>>(), point: Int) {
            if let Some(max) = space.get_max() {
                let min = space.get_min().unwrap();
                if let Some(next) = space.get_next(&point) {
                    prop_assert!(next > point, "next should be strictly greater than point");

                    if let Some(min_v) = min.into_val() {
                        if point < min_v {
                            prop_assert_eq!(next, min_v,
                                "point is outside of space: next should be the min");
                        }
                        else {
                            prop_assert!(next - point <= *space.step(),
                                "point is inside of space bounds: next should be at most `step` away");
                        }
                    }
                }
                else if let Some(max_v) = max.into_val() {
                    prop_assert!(point >= max_v, "No next if point >= max");
                }
                else {
                    prop_assert!(point.monotonic_add(*space.step()).is_none(), "No next if `point+step` overflows");
                }
            }
            else {
                prop_assert!(space.get_next(&point).is_none(), "no next in empty space");
            }
        }

        #[test]
        fn prev_exists_when_non_empty_and_step_from_min(space in any::<LinearSpace<Int>>(), point: Int) {
            if let Some(min) = space.get_min() {
                let max = space.get_max().unwrap();
                if let Some(prev) = space.get_prev(&point) {
                    prop_assert!(prev < point, "prev should be strictly less than point");

                    if let Some(max_v) = max.into_val() {
                        if point > max_v {
                            prop_assert_eq!(prev, max_v,
                                "point is outside of space: prev should be the max");
                        }
                        else {
                            prop_assert!(point - prev <= *space.step(),
                                "point is inside of space bounds: prev should be at most `step` away");
                        }
                    }
                }
                else if let Some(min_v) = min.into_val() {
                    prop_assert!(point <= min_v, "No prev if point <= min");
                }
                else {
                    prop_assert!(point.monotonic_sub(*space.step()).is_none(), "No prev if `point-step` overflows");
                }
            }
            else {
                prop_assert!(space.get_prev(&point).is_none(), "no prev in empty space");
            }
        }
    }
}
