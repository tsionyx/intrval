use core::ops;

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
        T: core::fmt::Debug + Clone + 'static,
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
    T: PartialOrd + Clone + Arbitrary<Strategy: 'static> + 'static,
    ops::Range<T>: Strategy<Value = T>,
    ops::RangeFrom<T>: Strategy<Value = T>,
    ops::RangeTo<T>: Strategy<Value = T>,
    ops::RangeInclusive<T>: Strategy<Value = T>,
    ops::RangeToInclusive<T>: Strategy<Value = T>,
{
    type Parameters = Self;
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(input_interval: Self::Parameters) -> Self::Strategy {
        let interval_strategy = input_interval.0.get_strategy();
        Interval::arbitrary_with_bounds_strategy(interval_strategy)
            .prop_map_into()
            .boxed()
    }
}

#[cfg(test)]
mod prop_test {
    use crate::interval;

    use super::*;

    type Int = i8;
    const PARAM_BOUND: Int = 100;

    fn params_range() -> impl Strategy<Value = Int> {
        #[allow(trivial_numeric_casts)]
        (-PARAM_BOUND..=PARAM_BOUND).boxed()
    }

    fn mul_range() -> ops::RangeInclusive<Int> {
        #[allow(trivial_numeric_casts)]
        (-11 as Int..=11)
    }

    fn mul_interval() -> Interval<Int> {
        mul_range().into()
    }

    // consider the `PARAM_BOUND` to prevent any case of addition overflow
    fn no_addition_overflow_interval() -> BoundedInterval<Int> {
        let min = Int::MIN + PARAM_BOUND;
        BoundedInterval(interval!((=min, -min)))
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
            range in BoundedInterval::arbitrary_with(no_addition_overflow_interval()),
            delta in params_range(),
        ) {
            let sub = range.0 - delta;
            let add_neg = range.0 + (-delta);
            prop_assert_eq!(sub, add_neg);
        }

        #[test]
        // i + x - x == i
        // i - x + x == i
        fn add_sub_roundtrip(
            range in BoundedInterval::arbitrary_with(no_addition_overflow_interval()),
            delta in params_range(),
        ) {
            let add_sub = (range.0 + delta) - delta;
            prop_assert_eq!(range.0, add_sub);
            let sub_add = (range.0 - delta) + delta;
            prop_assert_eq!(range.0, sub_add);
        }

        #[test]
        fn empty_and_full_are_preserving_under_add(increment in params_range()) {
            let empty = interval!(_: Int);
            prop_assert_eq!(empty + increment, empty);

            let full = interval!(..: Int);
            prop_assert_eq!(full + increment, full);
        }

        #[test]
        fn empty_is_reduced_and_not_clamped(range: Interval<Int>, x in params_range()) {
            let inv1 = range.is_empty();
            let inv2 = range.reduce() == interval!(_);
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
        fn contains_implies_clamp_preserving(range: Interval<Int>, x in params_range()) {
            use core::cmp::Ordering;
            use crate::bounds::{Endpoint, Bounded as _};

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
            } else if !range.is_empty() {
                let (a, b) = range.into_bounds().unwrap();
                let (ordering, clamped) = range.clamp(x).unwrap();
                match ordering {
                    Ordering::Less => {
                        prop_assert_eq!(b, Endpoint::Excluded(clamped));
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
                        prop_assert_eq!(a, Endpoint::Excluded(clamped));
                    }
                }
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
            fn mul_minus_one_is_neg(range in BoundedInterval::arbitrary_with(interval!(> Int::MIN).into())) {
                let mul_neg = range.0 * -1;
                prop_assert_eq!(mul_neg, -range.0);
            }

            #[test]
            fn empty_and_full_are_preserving(factor in params_range()) {
                let empty = interval!(_: Int);
                prop_assert_eq!(empty * factor, empty);

                let full = interval!(..: Int);
                if factor == 0 {
                    prop_assert_eq!(full * factor, interval!([0, 0]));
                }
                else {
                    prop_assert_eq!(full * factor, full);
                }
            }

            #[test]
            fn with_zero_is_always_zero(
                range in BoundedInterval::arbitrary_with(mul_interval().into())
            ) {
                let left = interval!(== 0);
                let right = range.0;

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
                range in BoundedInterval::arbitrary_with(mul_interval().into())
            ) {
                let left = interval!(== x);
                let right = range.0;
                prop_assert_eq!(left * right, right * left);
            }

            #[test]
            fn commutative(
                range1 in BoundedInterval::arbitrary_with(mul_interval().into()),
                range2 in BoundedInterval::arbitrary_with(mul_interval().into())
            ) {
                let (left, right) = (range1.0, range2.0);
                prop_assert_eq!(left * right, right * left);
            }
        }
    }

    mod bounds {
        use super::*;
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(8000))]

            #[test]
            fn roundtrip_into_bounds(range: Interval<Int>) {
                use crate::Bounded as _;

            let Ok(bounds) = range.into_bounds() else {
                // skip the empty interval as its bounds are not well-defined
                return Ok(());
            };

            let restored = Interval::<Int>::from_bounds(bounds);

                #[cfg(feature = "singleton")]
                // skip the singleton interval as it
                // will be restored as a closed interval
                if let Interval::Singleton(_) = range {
                    prop_assert_eq!(range, restored.reduce());
                    return Ok(());
                }

                prop_assert_eq!(range, restored);
            }

            #[test]
            fn reversed_has_the_bounds_swapped(range: Interval<Int>) {
                use crate::bounds::{Bounded as _, Endpoint};

                let (start, end) = range.into_bounds()
                    .unwrap_or((Endpoint::Infinite, Endpoint::Infinite));
                let (rev_start, rev_end) = range.reverse().into_bounds()
                    .unwrap_or((Endpoint::Infinite, Endpoint::Infinite));

                prop_assert_eq!(start.into_bound(), rev_end.into_bound());
                prop_assert_eq!(end.into_bound(), rev_start.into_bound());
            }

            #[test]
            fn closure_has_no_exclusive_bounds(range: Interval<Int>) {
                use crate::bounds::{Bounded as _, Endpoint};

                let (start, end) = range.into_closure().into_bounds()
                    .unwrap_or((Endpoint::Infinite, Endpoint::Infinite));
                prop_assert!(!matches!(start, Endpoint::Excluded(_)));
                prop_assert!(!matches!(end, Endpoint::Excluded(_)));
            }

            #[test]
            fn interior_has_no_inclusive_bounds(range: Interval<Int>) {
                use crate::bounds::{Bounded as _, Endpoint};

                let (start, end) = range.into_interior().into_bounds()
                .unwrap_or((Endpoint::Infinite, Endpoint::Infinite));
                prop_assert!(!matches!(start, Endpoint::Included(_)));
                prop_assert!(!matches!(end, Endpoint::Included(_)));
            }

            #[test]
            fn intersect_on_err_returns_original(range1: Interval<Int>, range2: Interval<Int>) {
                use crate::Bounded as _;

                if let Err((a, b)) = range1.intersect(range2) {
                    prop_assert_eq!(a.reduce(), range1.reduce());
                    prop_assert_eq!(b.reduce(), range2.reduce());
                }
            }

            #[test]
            fn union_on_err_returns_original(range1: Interval<Int>, range2: Interval<Int>) {
                use crate::Bounded as _;

                if let Err((a, b)) = range1.union(range2) {
                    prop_assert_eq!(a.reduce(), range1.reduce());
                    prop_assert_eq!(b.reduce(), range2.reduce());
                }
            }

            #[test]
            fn enclosure_on_err_returns_original(range1: Interval<Int>, range2: Interval<Int>) {
                use crate::Bounded as _;

                if let Err((a, b)) = range1.enclosure(range2) {
                    prop_assert_eq!(a.reduce(), range1.reduce());
                    prop_assert_eq!(b.reduce(), range2.reduce());
                }
            }

            #[test]
            fn union_is_enclosure_when_intersects(range1: Interval<Int>, range2: Interval<Int>) {
                use crate::{bounds::{Bounded as _, Endpoint}, OneOrPair};

                let fallback = (Endpoint::Infinite, Endpoint::Infinite);

                let inter_bounds = range1.intersect(range2)
                    .unwrap_or(fallback);
                let enc_bounds = range1.enclosure(range2)
                    .unwrap_or(fallback);

                match range1.union(range2)
                    .unwrap_or(OneOrPair::One(fallback)) {
                    OneOrPair::One(union_bounds) => {
                        prop_assert!(!Interval::from_bounds(inter_bounds).is_empty());
                        prop_assert_eq!(union_bounds, enc_bounds);
                    }
                    OneOrPair::Pair((a, b)) => {
                        prop_assert!(Interval::from_bounds(inter_bounds).is_empty());
                        prop_assert_eq!((a.0, b.1), enc_bounds);
                    }
                }
            }
        }
    }
}
