//! [`Arbitrary`] implementation for [`Interval`].
extern crate alloc;

#[allow(unused_imports)] // will be used internally by `prop_assert` macros
use alloc::format;
use core::{fmt::Debug, ops};

use proptest::prelude::*;

use super::Interval;

impl<T> Interval<T> {
    /// Creates a strategy that generates [`Arbitrary`] values from this `interval`.
    ///
    /// # Panics
    ///
    /// when the interval is [empty][Self::is_empty].
    pub fn get_strategy(self) -> BoxedStrategy<T>
    where
        T: PartialEq + Clone + Arbitrary + 'static,
        ops::Range<T>: Strategy<Value = T>,
        ops::RangeFrom<T>: Strategy<Value = T>,
        ops::RangeTo<T>: Strategy<Value = T>,
        ops::RangeInclusive<T>: Strategy<Value = T>,
        ops::RangeToInclusive<T>: Strategy<Value = T>,
    {
        let lower_bound_exclude_reason = "Excluding the lower bound";
        match self {
            Self::Empty => panic!("Cannot create a strategy for an empty interval"),
            Self::Full => any::<T>().boxed(),
            Self::LessThan(b) => (..b).boxed(),
            Self::LessThanOrEqual(b) => (..=b).boxed(),
            Self::GreaterThan(a) => (a.clone()..)
                .prop_filter(lower_bound_exclude_reason, move |x| x != &a)
                .boxed(),
            Self::GreaterThanOrEqual(a) => (a..).boxed(),
            #[cfg(feature = "singleton")]
            Self::Singleton(a) => Just(a).boxed(),
            Self::Open((a, b)) => (a.clone()..b)
                .prop_filter(lower_bound_exclude_reason, move |x| x != &a)
                .boxed(),
            Self::LeftOpen((a, b)) => (a.clone()..=b)
                .prop_filter(lower_bound_exclude_reason, move |x| x != &a)
                .boxed(),
            Self::RightOpen((a, b)) => (a..b).boxed(),
            Self::Closed((a, b)) => (a..=b).boxed(),
        }
    }

    fn arbitrary_with_bounds_strategy(input: BoxedStrategy<T>) -> impl Strategy<Value = Self>
    where
        T: Debug + Clone + 'static,
    {
        // simple `prop_oneof!` does not work here due to conditional compilation
        let s = prop::strategy::Union::new([
            Just(Self::Empty).boxed(),
            input.clone().prop_map(Self::LessThan).boxed(),
            input.clone().prop_map(Self::LessThanOrEqual).boxed(),
            input.clone().prop_map(Self::GreaterThanOrEqual).boxed(),
            input.clone().prop_map(Self::GreaterThan).boxed(),
            (input.clone(), input.clone()).prop_map(Self::Open).boxed(),
            (input.clone(), input.clone())
                .prop_map(Self::LeftOpen)
                .boxed(),
            (input.clone(), input.clone())
                .prop_map(Self::RightOpen)
                .boxed(),
            (input.clone(), input.clone())
                .prop_map(Self::Closed)
                .boxed(),
            Just(Self::Full).boxed(),
        ]);

        #[cfg(feature = "singleton")]
        let s = s.or(input.prop_map(Self::Singleton).boxed());
        s
    }
}

impl<T> Arbitrary for Interval<T>
where
    T: Clone + Arbitrary + 'static,
{
    type Parameters = T::Parameters;
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        let any = any_with::<T>(args).boxed();
        Self::arbitrary_with_bounds_strategy(any).boxed()
    }
}

#[derive(Debug, Copy, Clone)]
/// Wrapper for [`Interval<T>`] to generate arbitrary [`Interval`] values
/// using the [`Interval`] itself to generate the bounds.
pub struct BoundedInterval<T>(pub Interval<T>);

impl<T> Default for BoundedInterval<T> {
    fn default() -> Self {
        Self(Interval::Full)
    }
}

impl<T> From<Interval<T>> for BoundedInterval<T> {
    fn from(interval: Interval<T>) -> Self {
        Self(interval)
    }
}

impl<T> Arbitrary for BoundedInterval<T>
where
    T: PartialOrd + Clone + Arbitrary + 'static,
    <T as Arbitrary>::Strategy: 'static,
    ops::Range<T>: Strategy<Value = T>,
    ops::RangeFrom<T>: Strategy<Value = T>,
    ops::RangeTo<T>: Strategy<Value = T>,
    ops::RangeInclusive<T>: Strategy<Value = T>,
    ops::RangeToInclusive<T>: Strategy<Value = T>,
{
    type Parameters = Self; // the `Interval<T>` cannot be used because it is `!Default`
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        let interval_strategy = args.0.get_strategy();
        Interval::arbitrary_with_bounds_strategy(interval_strategy)
            .prop_map_into()
            .boxed()
    }
}

#[cfg(test)]
mod prop_test {
    use crate::{
        bounds::{Endpoint::Infinite, IntoBounds as _},
        interval,
        set::SetOps as _,
    };

    use super::*;

    type Int = i8;
    const PARAM_BOUND: Int = 100;

    fn params_range() -> impl Strategy<Value = Int> {
        #[allow(trivial_numeric_casts)]
        (-PARAM_BOUND..=PARAM_BOUND).boxed()
    }

    fn mul_range() -> ops::RangeInclusive<Int> {
        #[allow(trivial_numeric_casts, clippy::as_conversions)]
        (-11 as Int..=11)
    }

    fn mul_interval() -> Interval<Int> {
        mul_range().into()
    }

    // consider the `PARAM_BOUND` to prevent any case of addition overflow
    fn no_addition_overflow_interval() -> BoundedInterval<Int> {
        let min = Int::MIN + PARAM_BOUND;
        interval!((=min, -min)).into()
    }

    fn negated_interval() -> BoundedInterval<Int> {
        interval!((Int::MIN, =Int::MAX)).into()
    }

    fn non_empty() -> impl Strategy<Value = Interval<Int>> {
        any::<Interval<Int>>().prop_filter(
            "skip the empty interval as its bounds are not well-defined",
            |i| !i.is_empty(),
        )
    }

    fn one_or_zero_endpoints() -> impl Strategy<Value = Interval<Int>> {
        any::<Interval<Int>>().prop_filter("exclude both-bounded", |i| {
            if let Ok((a, b)) = i.as_ref_bounds() {
                matches!(a, Infinite) || matches!(b, Infinite)
            } else {
                true
            }
        })
    }

    fn single_endpoint() -> impl Strategy<Value = Interval<Int>> {
        any::<Interval<Int>>().prop_filter("only take one side-unbounded", |i| {
            if let Ok((a, b)) = i.as_ref_bounds() {
                matches!(a, Infinite) ^ matches!(b, Infinite)
            } else {
                false
            }
        })
    }

    proptest! {
        // https://proptest-rs.github.io/proptest/proptest/tutorial/config.html
        #![proptest_config(ProptestConfig::with_cases(8000))]

        #[test]
        // i + ZERO == i
        fn adding_zero_is_preserving(range: Interval<Int>) {
            let right = range + 0;
            prop_assert_eq!(range, right);
        }

        #[test]
        // i - x == i + (-x)
        fn sub_and_add_neg_is_equivalent(
            range in BoundedInterval::arbitrary_with(no_addition_overflow_interval()).prop_map(|i| i.0),
            delta in params_range(),
        ) {
            let sub = range - delta;
            let add_neg = range + (-delta);
            prop_assert_eq!(sub, add_neg);
        }

        #[test]
        // -(-i) == i
        fn double_neg_is_equivalent(
            range in BoundedInterval::arbitrary_with(negated_interval()).prop_map(|i| i.0),
        ) {
            let double_neg = -(-range);
            prop_assert_eq!(range, double_neg);
        }

        #[test]
        // i + x - x == i
        // i - x + x == i
        fn add_sub_roundtrip(
            range in BoundedInterval::arbitrary_with(no_addition_overflow_interval()).prop_map(|i| i.0),
            delta in params_range(),
        ) {
            let add_sub = (range + delta) - delta;
            prop_assert_eq!(range, add_sub);
            let sub_add = (range - delta) + delta;
            prop_assert_eq!(range, sub_add);
        }

        #[test]
        fn empty_and_full_are_preserving_under_add(increment in params_range()) {
            let empty = interval!(0: Int);
            prop_assert_eq!(empty + increment, empty);

            let full = interval!(U: Int);
            prop_assert_eq!(full + increment, full);
        }

        #[test]
        fn empty_is_reduced_and_not_clamped(range: Interval<Int>, x in params_range()) {
            let inv1 = range.is_empty();
            let inv2 = range == interval!(0);
            let inv3 = range.clamp(x).is_err();
            let invariants = [inv1, inv2, inv3];
            prop_assert!(invariants.iter().all(|&b| b) || invariants.iter().all(|&b| !b));
        }

        #[test]
        fn contains_reversed_with_complement(range: Interval<Int>, x in params_range()) {
            use crate::OneOrPair;

            let contains_in_original = range.contains(&x);
            let complement = !range;
            let contains_in_complement = match complement {
                OneOrPair::One(interval) => interval.contains(&x),
                OneOrPair::Pair((a, b)) => a.contains(&x) || b.contains(&x),
            };
            prop_assert_eq!(contains_in_original, !contains_in_complement);
        }

        #[test]
        fn contains_implies_clamp_preserving(range in non_empty(), x in params_range()) {
            use core::cmp::Ordering;
            use crate::bounds::{Endpoint, RIGHT, LEFT};

            fn bound_included<const SIDE: bool, T>(b: Endpoint<SIDE, T>) -> Option<T> {
                if let Endpoint::Included(v) = b {
                    Some(v)
                } else {
                    None
                }
            }

            if range.contains(&x) {
                let clamped = range.clamp(x);
                prop_assert_eq!(clamped, Ok((Ordering::Equal, x)));
            } else {
                let (a, b) = range.into_bounds().unwrap();
                let (ordering, clamped) = range.clamp(x).unwrap();
                match ordering {
                    Ordering::Less => {
                        prop_assert_eq!(b, Endpoint::<RIGHT, _>::Excluded(clamped));
                    }
                    Ordering::Equal => {
                        prop_assert_ne!(clamped, x);
                        prop_assert!(
                            [
                                bound_included(a),
                                bound_included(b)
                            ]
                            .contains(&Some(clamped))
                        );
                    }
                    Ordering::Greater => {
                        prop_assert_eq!(a, Endpoint::<LEFT, _>::Excluded(clamped));
                    }
                }
            }
        }
    }

    mod add_sub {
        use super::*;
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(8000))]

            #[test]
            // i + EMPTY == i
            fn adding_empty_interval_is_preserving(range in non_empty()) {
                let sum = range + interval!(0: Int);
                prop_assert_eq!(range, sum);

                let sum = range + interval!([0, -1]); // also empty
                prop_assert_eq!(range, sum);

                let sum = interval!((2, =2)) + range; // also empty
                prop_assert_eq!(range, sum);
            }

            #[test]
            // i + FULL == FULL
            fn adding_full_interval_is_full(range: Interval<Int>) {
                let sum = range + interval!(U: Int);
                prop_assert_eq!(sum, Interval::Full);
            }

            #[test]
            // i + Singleton(x) == i + x
            fn adding_singleton_is_like_adding_scalar(
                range in BoundedInterval::arbitrary_with(no_addition_overflow_interval())
                    .prop_filter(
                        "skip the empty interval as its bounds are not well-defined",
                        |i| !i.0.is_empty(),
                    ).prop_map(|i| i.0),
                delta in params_range(),
            ) {
                use crate::singleton::Singleton as _;

                let scalar_sum = range + delta;
                let sum = range + Interval::singleton(delta);
                prop_assert_eq!(sum, scalar_sum);
            }

            #[test]
            fn add_is_commutative(
                range1 in BoundedInterval::arbitrary_with(no_addition_overflow_interval()).prop_map(|i| i.0),
                range2 in BoundedInterval::arbitrary_with(no_addition_overflow_interval()).prop_map(|i| i.0)
            ) {
                let sum1 = range1 + range2;
                let sum2 = range2 + range1;
                prop_assert_eq!(sum1, sum2);
            }

            #[test]
            // i - EMPTY == i
            fn sub_empty_interval_is_preserving(range in non_empty()) {
                let diff = range - interval!(0: Int);
                prop_assert_eq!(range, diff);

                let diff = range - interval!([0, -1]); // also empty
                prop_assert_eq!(range, diff);
            }

            #[test]
            fn sub_from_empty_is_neg(
                range in BoundedInterval::arbitrary_with(negated_interval()).prop_map(|i| i.0),
            ) {
                let diff = interval!(0: Int) - range;
                prop_assert_eq!(range, -diff);

                let diff = interval!((2, =2)) - range; // also empty
                prop_assert_eq!(range, -diff);
            }

            #[test]
            // i - FULL == FULL
            // FULL - i == FULL
            fn sub_full_interval_is_full(range: Interval<Int>) {
                let diff = range - interval!(U: Int);
                prop_assert_eq!(diff, Interval::Full);

                let diff = interval!(U: Int) - range;
                prop_assert_eq!(diff, Interval::Full);
            }

            #[test]
            // i - Singleton(x) == i - x
            fn sub_singleton_is_like_sub_scalar(
                range in BoundedInterval::arbitrary_with(no_addition_overflow_interval())
                    .prop_filter(
                        "skip the empty interval as its bounds are not well-defined",
                        |i| !i.0.is_empty(),
                    ).prop_map(|i| i.0),
                delta in params_range(),
            ) {
                use crate::singleton::Singleton as _;

                let scalar_diff = range - delta;
                let diff = range - Interval::singleton(delta);
                prop_assert_eq!(diff, scalar_diff);
            }

            #[test]
            fn sub_is_equal_to_add_neg(
                range1 in BoundedInterval::arbitrary_with(no_addition_overflow_interval()).prop_map(|i| i.0),
                range2 in BoundedInterval::arbitrary_with(no_addition_overflow_interval()).prop_map(|i| i.0)
            ) {
                let diff1 = range1 - range2;
                let diff2 = range1 + (-range2);
                prop_assert_eq!(diff1, diff2);
            }

            #[test]
            fn sub_self_unbounded_is_full(
                range in single_endpoint(),
            ) {
                prop_assert_eq!(range - range, Interval::Full);
            }
        }
    }

    mod mult {
        use super::*;
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(8000))]

            #[test]
            fn mul_one_is_preserving(range: Interval<Int>) {
                prop_assert_eq!(range * 1, range);
            }

            #[test]
            fn mul_minus_one_is_neg(range in
                BoundedInterval::arbitrary_with(interval!(> Int::MIN).into()).prop_map(|i| i.0)
            ) {
                let mul_neg = range * -1;
                prop_assert_eq!(mul_neg, -range);
            }

            #[test]
            fn empty_and_full_are_preserving(factor in params_range()) {
                let empty = interval!(0: Int);
                prop_assert_eq!(empty * factor, empty);

                let full = interval!(U: Int);
                if factor == 0 {
                    prop_assert_eq!(full * factor, interval!([0, 0]));
                } else {
                    prop_assert_eq!(full * factor, full);
                }
            }

            #[test]
            fn with_zero_is_always_zero(right in
                BoundedInterval::arbitrary_with(mul_interval().into()).prop_map(|i| i.0)
            ) {
                let left = interval!(== 0);

                if right.is_empty() {
                    prop_assert!((left * right).is_empty());
                    prop_assert!((right * left).is_empty());
                }
                else {
                    prop_assert_eq!(left * right, interval!([0, 0]));
                    prop_assert_eq!(right * left, interval!([0, 0]));
                }
            }

            #[test]
            fn commutative_singleton(
                x in mul_range(),
                right in BoundedInterval::arbitrary_with(mul_interval().into()).prop_map(|i| i.0)
            ) {
                let left = interval!(== x);
                prop_assert_eq!(left * right, right * left);
            }

            #[test]
            fn commutative(
                left in BoundedInterval::arbitrary_with(mul_interval().into()).prop_map(|i| i.0),
                right in BoundedInterval::arbitrary_with(mul_interval().into()).prop_map(|i| i.0),
            ) {
                prop_assert_eq!(left * right, right * left);
            }
        }
    }

    mod bounds {
        use super::*;

        use crate::bounds::Bounded as _;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(8000))]

            #[test]
            fn roundtrip_into_bounds(range in non_empty()) {
                let bounds = range.into_bounds().unwrap();
                let restored = Interval::<Int>::from_bounds(bounds);

                #[cfg(feature = "singleton")]
                // skip the singleton interval as it
                // will be restored as a closed interval
                if let Interval::Singleton(_) = range {
                    prop_assert_eq!(range, restored);
                    return Ok(());
                }

                prop_assert_eq!(range, restored);
            }

            #[test]
            fn reversed_has_the_bounds_swapped(range: Interval<Int>) {
                match (range.into_bounds(), range.reverse().into_bounds()) {
                    (Ok((start, end)), Ok((rev_start, rev_end))) => {
                        prop_assert_eq!(start.into_bound(), rev_end.into_bound());
                        prop_assert_eq!(end.into_bound(), rev_start.into_bound());
                    }
                    // both bounds should be finite and complete a valid interval
                    (Ok((start, end)), Err(_)) |  (Err(_), Ok((start, end)))  => {
                        prop_assert!(!matches!(start, Infinite));
                        prop_assert!(!matches!(end, Infinite));
                        prop_assert!(start.bound_val().unwrap() <= end.bound_val().unwrap());
                    }
                    (Err(_), Err(_)) => {
                        prop_assert_eq!(range.len().into_diff().unwrap(), 0);
                        prop_assert_eq!(range.reverse().len().into_diff().unwrap(), 0);
                    }
                }
            }

            #[test]
            fn closure_has_no_exclusive_bounds(range: Interval<Int>) {
                use crate::bounds::Endpoint;

                let (start, end) = range.into_closure().into_bounds().unwrap_or((Infinite, Infinite));
                prop_assert!(!matches!(start, Endpoint::Excluded(_)));
                prop_assert!(!matches!(end, Endpoint::Excluded(_)));
            }

            #[test]
            fn interior_has_no_inclusive_bounds(range: Interval<Int>) {
                use crate::bounds::Endpoint;

                let (start, end) = range.into_interior().into_bounds().unwrap_or((Infinite, Infinite));
                prop_assert!(!matches!(start, Endpoint::Included(_)));
                prop_assert!(!matches!(end, Endpoint::Included(_)));
            }

            #[test]
            fn difference_on_err_returns_original(range1: Interval<Int>, range2: Interval<Int>) {
                if let Err((a, b)) = range1.difference(range2) {
                    prop_assert_eq!(a, range1);
                    prop_assert_eq!(b, range2);
                }
            }

            #[test]
            fn intersect_on_err_returns_original(range1: Interval<Int>, range2: Interval<Int>) {
                if let Err(fail) = range1.intersect(range2) {
                    prop_assert_eq!(fail, range2);
                }
            }

            #[test]
            fn symmetric_difference_is_reflexive(range1: Interval<Int>, range2: Interval<Int>) {
                let a = range1.symmetric_difference(range2).unwrap_or_else(crate::OneOrPair::One);
                let b = range2.symmetric_difference(range1).unwrap_or_else(crate::OneOrPair::One);
                prop_assert_eq!(a, b);
            }

            #[test]
            fn intersect_is_reflexive(range1: Interval<Int>, range2: Interval<Int>) {
                let a = range1.intersect(range2).unwrap_or_else(|err| err);
                let b = range2.intersect(range1).unwrap_or_else(|err| err);
                prop_assert_eq!(a, b);
            }

            #[test]
            fn union_is_reflexive(range1: Interval<Int>, range2: Interval<Int>) {
                let a = range1.union(range2);
                let b = range2.union(range1);
                prop_assert_eq!(a, b);
            }

            #[test]
            fn enclosure_is_reflexive(range1: Interval<Int>, range2: Interval<Int>) {
                let a = range1.enclosure(range2);
                let b = range2.enclosure(range1);
                prop_assert_eq!(a, b);
            }

            #[test]
            fn union_is_enclosure_when_intersects(range1: Interval<Int>, range2: Interval<Int>) {
                use crate::OneOrPair;

                let is_disjoint = range1.is_disjoint(&range2);
                // check the `is_disjoint` is reflexive
                prop_assert_eq!(is_disjoint, range2.is_disjoint(&range1));

                let inter = range1 & range2;
                let enclosed = range1.enclosure(range2);

                match range1.union(range2) {
                    OneOrPair::One(i) => {
                        prop_assert!(!is_disjoint);
                        prop_assert!(!i.is_empty() || (range1.is_empty() && range2.is_empty()));
                        prop_assert_eq!(i, enclosed);
                    }
                    OneOrPair::Pair((a, b)) => {
                        prop_assert!(is_disjoint);
                        prop_assert!(inter.is_empty());
                        let restored = Interval::from_bounds((
                            a.into_bounds().unwrap().0,
                            b.into_bounds().unwrap().1,
                        ));
                        prop_assert_eq!(restored, enclosed);
                    }
                }
            }

            #[test]
            fn complement_union_and_intersect(range in one_or_zero_endpoints()) {
                let complement = (!range).into_single().unwrap();

                prop_assert!((range & complement).is_empty());
                prop_assert_eq!(
                    range.union(complement).into_single().unwrap(),
                    Interval::Full,
                );
            }

            #[test]
            fn complement_union_and_intersect_with_closures(range in single_endpoint()) {
                let range = range.into_closure();
                let complement = (!range).into_single().unwrap().into_closure();

                let inter = range & complement;
                prop_assert!(!inter.is_empty());
                prop_assert_eq!(inter.len().into_diff().unwrap(), 0);

                prop_assert_eq!(
                    (range | complement).into_single().unwrap(),
                    Interval::Full
                );
            }

            #[test]
            fn complement_union_with_interior(range in single_endpoint()) {
                use crate::OneOrPair;

                let range = range.into_interior();
                let complement = (!range).into_single().unwrap().into_interior();

                let inter = range & complement;
                prop_assert!(inter.is_empty());
                prop_assert!(matches!(range | complement, OneOrPair::Pair(_)));
            }

            #[test]
            fn reflexive_set_operations(range: Interval<Int>) {
                let diff = range.difference(range);
                if range.is_empty() {
                    prop_assert_eq!(diff.unwrap().into_single().unwrap(), range);
                } else {
                    prop_assert!(diff.is_err());
                }
                let diff = range.symmetric_difference(range);
                if range.is_empty() {
                    prop_assert_eq!(diff.unwrap().into_single().unwrap(), range);
                } else {
                    prop_assert_eq!(diff.unwrap_err(), range);
                }
                prop_assert_eq!(range & range, range);
                prop_assert_eq!((range | range).into_single().unwrap(), range);
                prop_assert_eq!(range.enclosure(range), range);
                prop_assert!(range.is_super(&range));
                prop_assert!(range.is_sub(&range));
                prop_assert!(!range.is_disjoint(&range));
            }

            #[test]
            fn difference_from_universe_is_complement(range: Interval<Int>) {
                let diff = Interval::<Int>::Full.difference(range).unwrap_or_else(|(a, b)| {
                    assert!(a.is_full());
                    assert!(b.is_full());
                    Interval::Empty.into()
                });
                prop_assert_eq!(diff, !range);
            }

            #[test]
            fn difference_is_equivalent_intersect_with_complement(range1: Interval<Int>, range2: Interval<Int>) {
                use crate::OneOrPair;
                match !range2 {
                    OneOrPair::One(not_range2) => {
                        let inter = range1 & not_range2;
                        let diff = range1.difference(range2).map_or(
                            Interval::Empty, |diff| diff.into_single().unwrap());
                        prop_assert!(diff.is_sub(&range1));
                        prop_assert_eq!(diff, inter);
                    }
                    OneOrPair::Pair((a, b)) => {
                        let inter1 = range1 & a;
                        let inter2 = range1 & b;
                        match range1.difference(range2).unwrap_or_else(|_| Interval::Empty.into()) {
                            OneOrPair::One(x) => {
                                prop_assert!(x.is_sub(&range1));

                                if inter1.is_empty() {
                                    prop_assert_eq!(x, inter2);
                                } else if inter2.is_empty() {
                                    prop_assert_eq!(x, inter1);
                                } else {
                                    panic!("Expected a pair difference, got a single interval: {x:?}");
                                }
                            }
                            OneOrPair::Pair((diff1, diff2)) => {
                                prop_assert!(diff1.is_sub(&range1));
                                prop_assert!(diff2.is_sub(&range1));

                                prop_assert!(range2.is_sub(&range1));
                                prop_assert_eq!(diff1, inter1);
                                prop_assert_eq!(diff2, inter2);
                            }
                        }
                    }
                }
            }

            #[test]
            fn symmetric_difference_with_empty_is_preserving(range: Interval<Int>) {
                let e = Interval::<Int>::Empty;

                let diff = e.symmetric_difference(range);
                prop_assert_eq!(diff.unwrap().into_single().unwrap(), range);

                let diff = range.symmetric_difference(e);
                prop_assert_eq!(diff.unwrap().into_single().unwrap(), range);
            }

            #[test]
            fn symmetric_difference_with_universe_is_complement(range: Interval<Int>) {
                let u = Interval::<Int>::Full;
                let diff = u.symmetric_difference(range).unwrap_or_else(|r| {
                    assert!(r.is_empty() || r.is_full());
                    !r
                });
                prop_assert_eq!(diff, !range);
            }

            #[test]
            fn symmetric_difference_is_equivalent_to_union_of_differences(range1: Interval<Int>, range2: Interval<Int>) {
                use crate::OneOrPair;

                let diff1 = range1.difference(range2).unwrap_or_else(|_| Interval::Empty.into());
                let diff2 = range2.difference(range1).unwrap_or_else(|_| Interval::Empty.into());

                let un_diff = match (diff1, diff2) {
                    (OneOrPair::One(a), OneOrPair::One(b)) => a.union(b),
                    (OneOrPair::One(e), pair @ OneOrPair::Pair(_)) | (pair @ OneOrPair::Pair(_), OneOrPair::One(e)) => {
                        prop_assert!(e.is_empty());
                        pair
                    }
                    (OneOrPair::Pair(_), OneOrPair::Pair(_)) => {
                        panic!("Both differences are pairs");
                    }
                };

                let symm_diff = range1.symmetric_difference(range2).unwrap_or_else(|_| Interval::Empty.into());
                match symm_diff {
                    single @ OneOrPair::One(x) => {
                        prop_assert_eq!(single, un_diff);
                        let un = range1.union(range2).into_single().unwrap();
                        let inter = range1.intersect(range2).unwrap_or(Interval::Empty);
                        let un_inter_diff = un.difference(inter).unwrap_or_else(|_| Interval::Empty.into());
                        prop_assert_eq!(un_inter_diff, OneOrPair::One(x));
                    }
                    pair @ OneOrPair::Pair(_) => {
                        prop_assert_eq!(pair, un_diff);
                    }
                }
            }

            #[test]
            fn symmetric_difference_is_err_when_equals_and_non_empty(range1: Interval<Int>, range2: Interval<Int>) {
                let diff1 = range1.symmetric_difference(range2);
                let diff2 = range2.symmetric_difference(range1);

                if range1 == range2 && !range1.is_empty() {
                    prop_assert_eq!(diff1.unwrap_err(), range2);
                    prop_assert_eq!(diff2.unwrap_err(), range1);
                } else {
                    prop_assert_eq!(diff1.is_ok(), diff2.is_ok());
                }
            }

            #[test]
            fn sub_super_union_intersects(range1: Interval<Int>, range2: Interval<Int>) {
                let first_super = range1.is_super(&range2);
                let first_sub = range1.is_sub(&range2);
                let second_super = range2.is_super(&range1);
                let second_sub = range2.is_sub(&range1);
                let inter = range1 & range2;
                let uni = range1 | range2;

                if first_sub {
                    prop_assert!(second_super);
                }
                if second_sub {
                    prop_assert!(first_super);
                }

                match (first_super, second_super) {
                    (true, true) => {
                        prop_assert_eq!(range1, range2);
                        prop_assert!(first_sub);
                        prop_assert!(second_sub);
                        prop_assert_eq!(inter, range1);
                        prop_assert_eq!(uni.into_single().unwrap(), range1);
                    }
                    (true, false) => {
                        prop_assert!(!first_sub);
                        prop_assert!(second_sub);
                        prop_assert_eq!(inter, range2);
                        prop_assert_eq!(uni.into_single().unwrap(), range1);
                    }
                    (false, true) => {
                        prop_assert!(first_sub);
                        prop_assert!(!second_sub);
                        prop_assert_eq!(inter, range1);
                        prop_assert_eq!(uni.into_single().unwrap(), range2);
                    }
                    (false, false) => {
                        prop_assert!(!first_sub);
                        prop_assert!(!second_sub);

                        prop_assert!(range1.is_super(&inter));
                        prop_assert!(!range1.is_sub(&inter));
                        prop_assert!(range2.is_super(&inter));
                        prop_assert!(!range2.is_sub(&inter));
                    }
                }
            }
        }
    }
}
