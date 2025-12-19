use core::{
    cmp::Ordering,
    ops::{Bound, Not},
};

use crate::{IntoBounds, Pair};

const LOWER: bool = false;
const UPPER: bool = true;

#[allow(unnameable_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SidedBound<const SIDE: bool, T>(Bound<T>);

pub fn sided_bounds<I: IntoBounds<T>, T>(
    interval: I,
) -> (SidedBound<LOWER, T>, SidedBound<UPPER, T>) {
    let (start, end) = interval.into_bounds();
    (SidedBound::<LOWER, T>(start), SidedBound::<UPPER, T>(end))
}

impl<const SIDE: bool, T> From<SidedBound<SIDE, T>> for Bound<T> {
    fn from(value: SidedBound<SIDE, T>) -> Self {
        value.0
    }
}

impl<const SIDE: bool, T: PartialOrd> PartialOrd for SidedBound<SIDE, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        use Bound::{Excluded, Included, Unbounded};

        #[allow(clippy::match_bool)]
        let inc_to_exc = match SIDE {
            // `(a, ...)` can also be represented as `[a + epsilon, ...)`
            // which leads to `[a, ...) < [a + epsilon, ...)`,
            // so `Included(a) < Included(a + epsilon) ~= Excluded(a)`
            LOWER => Ordering::Less,
            // `(..., b)` can also be represented as `(..., b - epsilon]`
            // which leads to `(..., b] > (..., b - epsilon]`
            // so `Included(b) > Included(b - epsilon) ~= Excluded(b)`
            UPPER => Ordering::Greater,
        };
        let exc_to_inc = inc_to_exc.reverse();

        match (&self.0, &other.0) {
            (Unbounded, Unbounded) => Some(Ordering::Equal),
            (Unbounded, _) => Some(inc_to_exc),
            (_, Unbounded) => Some(exc_to_inc),

            (Included(a), Included(b)) | (Excluded(a), Excluded(b)) => a.partial_cmp(b),
            (Included(i), Excluded(e)) => i.partial_cmp(e).map(|x| x.then(inc_to_exc)),
            (Excluded(e), Included(i)) => e.partial_cmp(i).map(|x| x.then(exc_to_inc)),
        }
    }
}

impl<const SIDE: bool, T: Ord> Ord for SidedBound<SIDE, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other)
            .expect("comparison between Ord values failed")
    }
}

fn bound_negate_inclusion<T>(bound: Bound<T>) -> Bound<T> {
    match bound {
        Bound::Included(v) => Bound::Excluded(v),
        Bound::Excluded(v) => Bound::Included(v),
        Bound::Unbounded => Bound::Unbounded,
    }
}

impl<T> Not for SidedBound<LOWER, T> {
    type Output = SidedBound<UPPER, T>;

    fn not(self) -> Self::Output {
        SidedBound(bound_negate_inclusion(self.0))
    }
}

impl<T> Not for SidedBound<UPPER, T> {
    type Output = SidedBound<LOWER, T>;

    fn not(self) -> Self::Output {
        SidedBound(bound_negate_inclusion(self.0))
    }
}

impl<const SIDE: bool, T> From<SidedBound<SIDE, T>> for Pair<Bound<T>> {
    fn from(value: SidedBound<SIDE, T>) -> Self {
        #[allow(clippy::match_bool)]
        match SIDE {
            LOWER => (value.0, Bound::Unbounded),
            UPPER => (Bound::Unbounded, value.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use Bound::{Excluded, Included, Unbounded};

    use super::*;

    fn lower(b: Bound<i32>) -> SidedBound<LOWER, i32> {
        SidedBound(b)
    }

    fn upper(b: Bound<i32>) -> SidedBound<UPPER, i32> {
        SidedBound(b)
    }

    #[test]
    fn unbounded_infimum() {
        assert!(lower(Unbounded) == lower(Unbounded));
        assert!(lower(Unbounded) < lower(Included(i32::MIN)));
        assert!(lower(Unbounded) < lower(Excluded(i32::MIN)));
        assert!(lower(Unbounded) < lower(Included(0)));
        assert!(lower(Unbounded) < lower(Excluded(0)));
        assert!(lower(Unbounded) < lower(Included(0)));
        assert!(lower(Unbounded) < lower(Excluded(0)));
    }

    #[test]
    fn forward_inner_inequality_for_lower() {
        assert!(lower(Included(i32::MIN)) < lower(Included(-1_000)));
        assert!(lower(Included(i32::MIN)) < lower(Excluded(-1_000)));

        assert!(lower(Excluded(0)) < lower(Excluded(1)));
        assert!(lower(Excluded(-1)) < lower(Included(1)));
        assert!(lower(Included(0)) < lower(Excluded(1)));
        assert!(lower(Included(0)) < lower(Included(1)));

        assert!(lower(Excluded(5)) > lower(Excluded(1)));
        assert!(lower(Excluded(8)) > lower(Included(7)));
        assert!(lower(Included(0)) > lower(Excluded(-1)));
        assert!(lower(Included(2)) > lower(Included(1)));

        assert!(lower(Included(i32::MAX)) > lower(Included(1_000)));
        assert!(lower(Included(i32::MAX)) > lower(Excluded(1_000)));
    }

    #[test]
    fn resolve_equal_included_excluded_lower() {
        // '>=5' < '>5'
        assert!(lower(Included(5)) < lower(Excluded(5)));
        // '>100' > '>=100'
        assert!(lower(Excluded(100)) > lower(Included(100)));
    }

    #[test]
    fn unbounded_supremum() {
        assert!(upper(Unbounded) == upper(Unbounded));
        assert!(upper(Unbounded) > upper(Included(i32::MIN)));
        assert!(upper(Unbounded) > upper(Excluded(i32::MIN)));
        assert!(upper(Unbounded) > upper(Included(0)));
        assert!(upper(Unbounded) > upper(Excluded(0)));
        assert!(upper(Unbounded) > upper(Included(0)));
        assert!(upper(Unbounded) > upper(Excluded(0)));
    }

    #[test]
    fn forward_inner_inequality_for_upper() {
        assert!(upper(Included(i32::MIN)) < upper(Included(-1_000)));
        assert!(upper(Included(i32::MIN)) < upper(Excluded(-1_000)));

        assert!(upper(Excluded(0)) < upper(Excluded(1)));
        assert!(upper(Excluded(-1)) < upper(Included(1)));
        assert!(upper(Included(0)) < upper(Excluded(1)));
        assert!(upper(Included(0)) < upper(Included(1)));

        assert!(upper(Excluded(5)) > upper(Excluded(1)));
        assert!(upper(Excluded(8)) > upper(Included(7)));
        assert!(upper(Included(0)) > upper(Excluded(-1)));
        assert!(upper(Included(2)) > upper(Included(1)));

        assert!(upper(Included(i32::MAX)) > upper(Included(1_000)));
        assert!(upper(Included(i32::MAX)) > upper(Excluded(1_000)));
    }

    #[test]
    fn resolve_equal_included_excluded_upper() {
        // '<=5' > '<5'
        assert!(upper(Included(5)) > upper(Excluded(5)));
        // '<100' < '<=100'
        assert!(upper(Excluded(100)) < upper(Included(100)));
    }
}
