use core::cmp::Ordering;

use crate::{
    bounds::{Bounded, Endpoint},
    helper::{Pair, Size, Zero},
    singleton::SingletonBounds,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "SCREAMING_SNAKE_CASE"))]
/// Represent the subset of domain values for some `T: PartialOrd`.
///
/// <https://en.wikipedia.org/wiki/Interval_(mathematics)>
pub enum Interval<T> {
    /// The empty set of values.
    ///
    /// <https://en.wikipedia.org/wiki/Empty_set>
    Empty,

    /// The set of values strictly less than the boundary.
    #[cfg_attr(feature = "serde", serde(alias = "<"))]
    LessThan(T),

    /// The set of values less or equal than the boundary.
    #[cfg_attr(feature = "serde", serde(alias = "<="))]
    LessThanOrEqual(T),

    #[cfg(feature = "singleton")]
    /// The set of values equal to the specified one.
    ///
    /// <https://en.wikipedia.org/wiki/Singleton_set>
    #[cfg_attr(feature = "serde", serde(alias = "=="))]
    Singleton(T),

    /// The set of values greater or equal than the boundary.
    #[cfg_attr(feature = "serde", serde(alias = ">="))]
    GreaterThanOrEqual(T),

    /// The set of values strictly greater than the boundary.
    #[cfg_attr(feature = "serde", serde(alias = ">"))]
    GreaterThan(T),

    /// The set of values in open range `(a, b)` (both boundaries are _excluded_).
    #[cfg_attr(
        feature = "serde",
        serde(rename = "BETWEEN_EXCLUSIVE", alias = "(,)", alias = "],[")
    )]
    Open(Pair<T>),

    /// The set of values in half-open `(a, b]` range bounded _exclusively_ below and _inclusively_ above.
    #[cfg_attr(feature = "serde", serde(alias = "(,]", alias = "],]"))]
    LeftOpen(Pair<T>),

    /// The set of values in half-open `[a, b)` range bounded _inclusively_ below and _exclusively_ above.
    #[cfg_attr(
        feature = "serde",
        serde(rename = "BETWEEN", alias = "[,)", alias = "[,[")
    )]
    RightOpen(Pair<T>),

    /// The set of values in closed range `[a, b]` (both boundaries are _included_).
    #[cfg_attr(feature = "serde", serde(rename = "BETWEEN_INCLUSIVE", alias = "[,]"))]
    Closed(Pair<T>),

    /// Interval containing any value.
    ///
    /// <https://en.wikipedia.org/wiki/Universe_(mathematics)>
    Full,
}

macro_rules! map_variants {
    ($value:expr, $single:pat => $single_result:expr, ($a: pat, $b: pat) => ($a_result: expr, $b_result: expr)) => {
        match $value {
            Interval::Empty => Interval::Empty,
            Interval::LessThan($single) => Interval::LessThan($single_result),
            Interval::LessThanOrEqual($single) => Interval::LessThanOrEqual($single_result),
            #[cfg(feature = "singleton")]
            Interval::Singleton($single) => Interval::Singleton($single_result),
            Interval::GreaterThanOrEqual($single) => Interval::GreaterThanOrEqual($single_result),
            Interval::GreaterThan($single) => Interval::GreaterThan($single_result),
            Interval::Open(($a, $b)) => Interval::Open(($a_result, $b_result)),
            Interval::LeftOpen(($a, $b)) => Interval::LeftOpen(($a_result, $b_result)),
            Interval::RightOpen(($a, $b)) => Interval::RightOpen(($a_result, $b_result)),
            Interval::Closed(($a, $b)) => Interval::Closed(($a_result, $b_result)),
            Interval::Full => Interval::Full,
        }
    };
}

impl<T> Interval<T> {
    /// Apply a transformation to the borders of an [`Interval`],
    /// preserving its structure.
    pub fn map<U, F>(self, mut f: F) -> Interval<U>
    where
        F: FnMut(T) -> U,
    {
        map_variants!(self,
            x => f(x),
            (a, b) => (f(a), f(b))
        )
    }

    /// Get an [`Interval`] with referenced bound values.
    pub const fn as_ref(&self) -> Interval<&T> {
        map_variants!(self,
            x => &x,
            (a, b) => (&a, &b)
        )
    }

    #[must_use = "this method consume the Interval and return a new one"]
    /// Swap the [`Interval`]'s bounds.
    pub fn reverse(self) -> Self {
        match self {
            Self::Empty => Self::Empty,
            Self::LessThan(x) => Self::GreaterThan(x),
            Self::LessThanOrEqual(x) => Self::GreaterThanOrEqual(x),
            #[cfg(feature = "singleton")]
            Self::Singleton(x) => Self::Singleton(x),
            Self::GreaterThanOrEqual(x) => Self::LessThanOrEqual(x),
            Self::GreaterThan(x) => Self::LessThan(x),
            Self::Open((a, b)) => Self::Open((b, a)),
            Self::LeftOpen((a, b)) => Self::RightOpen((b, a)),
            Self::RightOpen((a, b)) => Self::LeftOpen((b, a)),
            Self::Closed((a, b)) => Self::Closed((b, a)),
            Self::Full => Self::Full,
        }
    }

    #[must_use = "this method consume the Interval and return a new one"]
    /// Transform the [`Interval`] into a _closed_ one
    /// by _*augmenting*_ it with its endpoints.
    pub fn into_closure(self) -> Self {
        #[allow(clippy::match_same_arms)]
        match self {
            Self::Empty => Self::Empty,
            Self::LessThan(x) | Self::LessThanOrEqual(x) => Self::LessThanOrEqual(x),
            #[cfg(feature = "singleton")]
            Self::Singleton(x) => Self::Singleton(x),
            Self::GreaterThanOrEqual(x) | Self::GreaterThan(x) => Self::GreaterThanOrEqual(x),
            Self::Open((a, b))
            | Self::LeftOpen((a, b))
            | Self::RightOpen((a, b))
            | Self::Closed((a, b)) => Self::Closed((a, b)),
            Self::Full => Self::Full,
        }
    }

    #[must_use = "this method consume the Interval and return a new one"]
    /// Transform the [`Interval`] into an _open_ one
    /// by _*excluding*_ its endpoint values.
    pub fn into_interior(self) -> Self
    where
        Self: Bounded<T> + SingletonBounds<T>,
    {
        #[allow(clippy::match_same_arms)]
        match self {
            Self::Empty => Self::Empty,
            Self::LessThan(x) | Self::LessThanOrEqual(x) => Self::LessThan(x),
            #[cfg(feature = "singleton")]
            Self::Singleton(a) => {
                Self::from_bounds(<Self as SingletonBounds<T>>::value_into_bounds(a))
                    .into_interior()
            }
            Self::GreaterThanOrEqual(x) | Self::GreaterThan(x) => Self::GreaterThan(x),
            Self::Open((a, b))
            | Self::LeftOpen((a, b))
            | Self::RightOpen((a, b))
            | Self::Closed((a, b)) => Self::Open((a, b)),
            Self::Full => Self::Full,
        }
    }

    #[must_use = "this method consume the Interval and return a new one"]
    /// Reduce the [`Interval`] to its minimal form.
    ///
    /// ```
    /// # use intrval::{Interval, Singleton as _};
    /// assert_eq!(Interval::Open((5, 5)).reduce(), Interval::Empty);
    /// assert_eq!(Interval::Closed((5, 5)).reduce(), Interval::singleton(5));
    /// assert_eq!(Interval::Closed((7, 5)).reduce(), Interval::Empty);
    /// ```
    pub fn reduce(self) -> Self
    where
        T: PartialOrd,
    {
        match self {
            Self::Empty
            | Self::LessThan(_)
            | Self::LessThanOrEqual(_)
            | Self::GreaterThanOrEqual(_)
            | Self::GreaterThan(_)
            | Self::Full => self,
            #[cfg(feature = "singleton")]
            Self::Singleton(_) => self,
            Self::Open((ref a, ref b))
            | Self::LeftOpen((ref a, ref b))
            | Self::RightOpen((ref a, ref b)) => {
                if a >= b {
                    Self::Empty
                } else {
                    self
                }
            }
            Self::Closed((ref a, ref b)) => {
                #[cfg(feature = "singleton")]
                {
                    match a.partial_cmp(b) {
                        None | Some(Ordering::Less) => self,
                        Some(Ordering::Equal) => {
                            // match again to avoid cloning T
                            if let Self::Closed((a, _)) = self {
                                Self::Singleton(a)
                            } else {
                                self
                            }
                        }
                        Some(Ordering::Greater) => Self::Empty,
                    }
                }

                #[cfg(not(feature = "singleton"))]
                {
                    if a > b {
                        Self::Empty
                    } else {
                        self
                    }
                }
            }
        }
    }

    /// Whether the [`Interval`] contains no valid values,
    /// i.e. is [`Self::Empty`] or have invalid bounds (a > b).
    ///
    /// ```
    /// # use intrval::{Interval, interval};
    /// assert!(Interval::<u32>::Empty.is_empty());
    /// assert!(interval!((5, 5)).is_empty());
    /// assert!(!interval!([5, 5]).is_empty());
    /// assert!(interval!([7, 5]).is_empty());
    /// ```
    ///
    /// TODO: consider moving into `RangeBounds::is_empty` when
    /// the `feature = "range_bounds_is_empty"` gets stabilized.
    pub fn is_empty(&self) -> bool
    where
        T: PartialOrd,
    {
        self.as_ref().into_bounds().is_err()
    }

    /// Whether the [`Interval`] contains every possible value.
    pub const fn is_full(&self) -> bool {
        matches!(self, Self::Full)
    }

    /// Whether the [`Interval`] contains a given point.
    pub fn contains<U>(&self, point: &U) -> bool
    where
        T: PartialOrd + PartialOrd<U>,
        U: ?Sized + PartialOrd<T>,
    {
        use Endpoint::{Excluded, Included, Infinite};

        let Ok((a, b)) = self.as_ref().into_bounds() else {
            // an empty interval does not contain any point
            return false;
        };

        (match a {
            Included(start) => start <= point,
            Excluded(start) => start < point,
            Infinite => true,
        }) && (match b {
            Included(end) => point <= end,
            Excluded(end) => point < end,
            Infinite => true,
        })
    }

    /// Returns the length of the [`Interval`] if both bounds are finite.
    ///
    /// ```
    /// # use intrval::interval;
    ///
    /// assert_eq!(interval!(_: u8).len().into_diff().unwrap(), 0);
    /// assert!(interval!(< 5).len().into_diff().is_none());
    /// assert!(interval!(<= 3.2).len().into_diff().is_none());
    /// assert!(interval!(> -10).len().into_diff().is_none());
    /// assert!(interval!(>= 0u32).len().into_diff().is_none());
    /// assert_eq!(interval!(= 42).len().into_diff().unwrap(), 0);
    /// assert_eq!(interval!(== 42).len().into_diff().unwrap(), 0);
    ///
    /// assert_eq!(interval!((1, 10)).len().into_diff().unwrap(), 9);
    /// assert_eq!(interval!((0, =100)).len().into_diff().unwrap(), 100);
    /// assert_eq!(interval!((=0, 100)).len().into_diff().unwrap(), 100);
    /// assert_eq!(interval!((=6, 5)).len().into_diff().unwrap(), 0);
    /// assert_eq!(interval!([0.0, 1.0]).len().into_diff().unwrap(), 1.0);
    /// assert_eq!(interval!([5.0, 1.0]).len().into_diff().unwrap(), 0.0);
    /// assert_eq!(interval!((=0.0, =1.0)).len().into_diff().unwrap(), 1.0);
    /// assert!(interval!(..: i32).len().into_diff().is_none());
    /// ```
    pub fn len(&self) -> Size<<T as core::ops::Sub>::Output>
    where
        T: Clone + PartialOrd + core::ops::Sub<Output: Zero>,
    {
        use Endpoint::{Excluded, Included, Infinite};

        let Ok((a, b)) = self.as_ref().into_bounds() else {
            return Size::Empty;
        };

        let start = match a {
            Included(v) | Excluded(v) => v,
            Infinite => return Size::Infinite,
        };

        let end = match b {
            Included(v) | Excluded(v) => v,
            Infinite => return Size::Infinite,
        };

        if end > start {
            Size::Finite(end.clone() - start.clone())
        } else {
            Size::Empty
        }
    }

    /// Restrict a given value to the [`Interval`].
    ///
    /// # Returns
    ///
    /// - `Ok((Less, upper))` if the resulted value should be less than the `upper`;
    /// - `Ok((Equal, x))` if the resulted value is exactly `x`;
    /// - `Ok((Greater, lower))` if the resulted value should be greater than the `lower`.
    ///
    /// # Errors
    ///
    /// Returns wrapped original value if the interval is [empty][Self::is_empty];
    pub fn clamp(&self, x: T) -> Result<(Ordering, T), T>
    where
        T: Clone + PartialOrd,
    {
        if self.is_empty() {
            return Err(x);
        }

        let Ok((a, b)) = self.as_ref().into_bounds() else {
            return Err(x);
        };

        let res = {
            match a {
                Endpoint::Included(start) if &x < start => (Ordering::Equal, start.clone()),
                Endpoint::Excluded(start) if &x <= start => (Ordering::Greater, start.clone()),
                _ => match b {
                    Endpoint::Included(end) if &x > end => (Ordering::Equal, end.clone()),
                    Endpoint::Excluded(end) if &x >= end => (Ordering::Less, end.clone()),
                    _ => (Ordering::Equal, x),
                },
            }
        };

        Ok(res)
    }
}

#[macro_export]
/// Create an [`Interval`] using a concise syntax.
///
/// ```
/// # use intrval::{interval, Interval::{self, *}, Singleton as _};
///
/// assert_eq!(interval!(_), Empty::<i32>);
/// assert_eq!(interval!(_: u8), Empty);
///
/// assert_eq!(interval!(< 5), LessThan(5));
/// assert_eq!(interval!(<= 3.2), LessThanOrEqual(3.2));
/// assert_eq!(interval!(> -10), GreaterThan(-10));
/// assert_eq!(interval!(>= 0u32), GreaterThanOrEqual(0u32));
/// assert_eq!(interval!(= 42), Interval::singleton(42));
/// assert_eq!(interval!(== 42), Interval::singleton(42));
///
/// assert_eq!(interval!((1, 10)), Open((1, 10)));
/// assert_eq!(interval!((0, =100)), LeftOpen((0, 100)));
/// assert_eq!(interval!((=0, 100)), RightOpen((0, 100)));
/// assert_eq!(interval!([0.0, 1.0]), Closed((0.0, 1.0)));
/// assert_eq!(interval!((=0.0, =1.0)), Closed((0.0, 1.0)));
///
/// assert_eq!(interval!(..), Full::<f64>);
/// assert_eq!(interval!(..: f32), Full);
/// ```
macro_rules! interval {
    (_ $(:$t:ty)?) => {
        $crate::Interval $(::<$t>)? ::Empty
    };
    (< $x:expr) => {
        $crate::Interval::LessThan($x)
    };
    (<= $x:expr) => {
        $crate::Interval::LessThanOrEqual($x)
    };
    (> $x:expr) => {
        $crate::Interval::GreaterThan($x)
    };
    (>= $x:expr) => {
        $crate::Interval::GreaterThanOrEqual($x)
    };
    (= $x:expr) => {{
        use $crate::Singleton as _;
        $crate::Interval::singleton($x)
    }};
    (== $x:expr) => {{
        use $crate::Singleton as _;
        $crate::Interval::singleton($x)
    }};
    ( ( $a:expr , $b:expr ) ) => {
        $crate::Interval::Open(($a, $b))
    };

    // unbalanced [ and ) are not supported in macros to avoid confusion
    ( ($a:expr , =$b:expr) ) => {
       $crate::Interval::LeftOpen(($a, $b))
    };
    ( ( =$a:expr , $b:expr ) ) => {
       $crate::Interval::RightOpen(($a, $b))
    };
    ( ( =$a:expr , =$b:expr ) ) => {
        $crate::Interval::Closed(($a, $b))
    };

    ( [ $a:expr , $b:expr ] ) => {
        $crate::Interval::Closed(($a, $b))
    };
    (.. $(:$t:ty)? ) => {
        $crate::Interval $(::<$t>)? ::Full
    };
}
