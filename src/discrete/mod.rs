//! Implementation of discrete intervals, i.e. intervals over types
//! with discrete values (e.g. integers).
mod impls;
pub mod rounding;

use crate::helper::{OneOrPair, ValOrInf};

pub use self::impls::linear::LinearSpace;

/// Trait representing a discrete set of ordered points.
pub trait DiscreteOrdSet {
    /// The type of points in the set.
    type Point;

    /// Check if the set is empty i.e., contains no points.
    fn is_empty(&self) -> bool {
        self.get_min().is_none() && self.get_max().is_none()
    }

    /// Check if the set contains the given point.
    fn contains(&self, point: &Self::Point) -> bool;

    /// Get the minimum point of the set, if it is not empty.
    ///
    /// The result is wrapped in [`ValOrInf`] to allow representing an
    /// unbounded lower endpoint:
    ///
    /// - `None` means the set is empty;
    /// - `Some(ValOrInf::Val(l))` means `l` is the least valid point in the set;
    /// - `Some(ValOrInf::Inf)` represents negative infinity, i.e. the set
    ///   is unbounded below.
    fn get_min(&self) -> Option<ValOrInf<Self::Point>>;

    /// Get the maximum point of the set, if it is not empty.
    ///
    /// The result is wrapped in [`ValOrInf`] to allow representing an
    /// unbounded upper endpoint:
    ///
    /// - `None` means the set is empty;
    /// - `Some(ValOrInf::Val(u))` means `u` is the greatest valid point in the set;
    /// - `Some(ValOrInf::Inf)` represents positive infinity, i.e. the set
    ///   is unbounded above.
    fn get_max(&self) -> Option<ValOrInf<Self::Point>>;

    /// Find the point(s) in the set that are closest to the given point
    /// from either side, i.e.:
    /// - lower bound: `max({x in set | x <= point})`;
    /// - upper bound: `min({x in set | x >= point})`;
    ///
    /// # Returns
    ///
    /// - [`OneOrPair::Pair`] when both such points exist
    ///   (the two points may be equal, e.g. when
    ///   the given point itself is already in the set);
    /// - [`OneOrPair::One`] when only one such point exists, e.g.:
    ///   - the given point itself is in the set, and the implementation
    ///     chooses to return a single point rather than a `(x, x)` pair;
    ///   - the given point is out of the set's bounds;
    /// - [`None`] when no such points exist (e.g., the set is empty).
    ///
    /// Implementations are therefore allowed to return either
    /// `One(x)` or `Pair((x, x))` when the given point is in the set.
    fn get_nearest(&self, point: &Self::Point) -> Option<OneOrPair<Self::Point>>;

    /// Find the least point in the set greater than the given one.
    fn get_next(&self, point: &Self::Point) -> Option<Self::Point>;

    /// Find the greatest point in the set less than the given one.
    fn get_prev(&self, point: &Self::Point) -> Option<Self::Point>;
}
