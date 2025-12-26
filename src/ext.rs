//! Generalization
//! of [extended real numbers](https://en.wikipedia.org/wiki/Extended_real_number_line)
//! but for any single-dimensioned type for which infinite values make sense.

use core::{
    cmp::Ordering,
    fmt,
    ops::{Add, Div, Mul, Sub},
};

use crate::Zero;

/// Extended number type which can represent
/// either infinite value (negative or positive) or a finite value of type `T`.
#[derive(Debug, Clone, Copy)]
pub enum ExtNum<T> {
    /// A negative infinity value.
    NegInf,

    /// A finite value of type `T`.
    Finite(T),

    /// A positive infinity value.
    PosInf,

    /// Represents an indeterminate form, e.g., `∞ - ∞` or `0/0`.
    /// This value is contagious in arithmetic operations, i.e.
    /// if any operand is `Indeterminate`, the result is also `Indeterminate`.
    ///
    /// <https://en.wikipedia.org/wiki/Indeterminate_form>
    Indeterminate(IndeterminanceReason),
}

impl<T> From<T> for ExtNum<T> {
    fn from(value: T) -> Self {
        Self::Finite(value)
    }
}

impl<T> ExtNum<T> {
    #[must_use]
    /// Create an infinite value with the given polarity.
    pub const fn inf(positive: bool) -> Self {
        if positive {
            Self::PosInf
        } else {
            Self::NegInf
        }
    }
}

impl<T> PartialEq for ExtNum<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::NegInf, Self::NegInf) | (Self::PosInf, Self::PosInf) => true,
            (Self::Finite(a), Self::Finite(b)) => a == b,
            _ => false,
        }
    }
}

impl<T> PartialOrd for ExtNum<T>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        use ExtNum::{Finite, Indeterminate, NegInf, PosInf};

        match (self, other) {
            (Indeterminate(_), _) | (_, Indeterminate(_)) => None,
            (NegInf, NegInf) | (PosInf, PosInf) => Some(Ordering::Equal),
            (NegInf, _) | (_, PosInf) => Some(Ordering::Less),
            (_, NegInf) | (PosInf, _) => Some(Ordering::Greater),
            (Finite(a), Finite(b)) => a.partial_cmp(b),
        }
    }
}

impl<T, U, S> Add<ExtNum<U>> for ExtNum<T>
where
    T: Add<U, Output = S>,
{
    type Output = ExtNum<S>;

    fn add(self, rhs: ExtNum<U>) -> Self::Output {
        use ExtNum::{Finite, Indeterminate, NegInf, PosInf};

        match (self, rhs) {
            (Indeterminate(reason), _) | (_, Indeterminate(reason)) => Indeterminate(reason),
            (NegInf, PosInf) | (PosInf, NegInf) => {
                Indeterminate(IndeterminanceReason::SumOppositeInfinities)
            }
            (NegInf, _) | (_, NegInf) => NegInf,
            (PosInf, _) | (_, PosInf) => PosInf,
            (Finite(a), Finite(b)) => (a + b).into(),
        }
    }
}

impl<T, U, D> Sub<ExtNum<U>> for ExtNum<T>
where
    T: Sub<U, Output = D>,
{
    type Output = ExtNum<D>;

    fn sub(self, rhs: ExtNum<U>) -> Self::Output {
        use ExtNum::{Finite, Indeterminate, NegInf, PosInf};

        match (self, rhs) {
            (Indeterminate(reason), _) | (_, Indeterminate(reason)) => Indeterminate(reason),
            (NegInf, NegInf) | (PosInf, PosInf) => {
                Indeterminate(IndeterminanceReason::SumOppositeInfinities)
            }
            (NegInf, _) | (_, PosInf) => NegInf,
            (PosInf, _) | (_, NegInf) => PosInf,
            (Finite(a), Finite(b)) => (a - b).into(),
        }
    }
}

impl<T, U, P> Mul<ExtNum<U>> for ExtNum<T>
where
    T: Zero + Mul<U, Output = P>,
    U: Zero,
{
    type Output = ExtNum<P>;

    fn mul(self, rhs: ExtNum<U>) -> Self::Output {
        use ExtNum::{Finite, Indeterminate, NegInf, PosInf};

        let zero_mul_inf = IndeterminanceReason::ZeroInfProduct;
        match (self, rhs) {
            (Indeterminate(reason), _) | (_, Indeterminate(reason)) => Indeterminate(reason),
            (NegInf, NegInf) | (PosInf, PosInf) => PosInf,
            (NegInf, PosInf) | (PosInf, NegInf) => NegInf,
            (Finite(a), NegInf) => inf_with_signum(false, &a, zero_mul_inf),
            (Finite(a), PosInf) => inf_with_signum(true, &a, zero_mul_inf),
            (NegInf, Finite(b)) => inf_with_signum(false, &b, zero_mul_inf),
            (PosInf, Finite(b)) => inf_with_signum(true, &b, zero_mul_inf),
            (Finite(a), Finite(b)) => (a * b).into(),
        }
    }
}

fn inf_with_signum<S, T>(positive: bool, value: &S, on_zero: IndeterminanceReason) -> ExtNum<T>
where
    S: Zero,
{
    use ExtNum::Indeterminate;

    match value.cmp_zero() {
        Some(Ordering::Less) => ExtNum::inf(!positive),
        Some(Ordering::Equal) => Indeterminate(on_zero),
        Some(Ordering::Greater) => ExtNum::inf(positive),
        None => Indeterminate(IndeterminanceReason::NotComparableWithZero),
    }
}

impl<T, U, Q> Div<ExtNum<U>> for ExtNum<T>
where
    T: Div<U, Output = Q>,
    U: Zero,
    Q: Zero,
{
    type Output = ExtNum<Q>;

    fn div(self, rhs: ExtNum<U>) -> Self::Output {
        use ExtNum::{Finite, Indeterminate, NegInf, PosInf};

        let zero_div = IndeterminanceReason::DivisionByZero;
        match (self, rhs) {
            (Indeterminate(reason), _) | (_, Indeterminate(reason)) => Indeterminate(reason),
            (NegInf | PosInf, NegInf | PosInf) => {
                Indeterminate(IndeterminanceReason::DivideInfinities)
            }
            (Finite(_), NegInf | PosInf) => Q::zero().into(),
            (NegInf, Finite(b)) => inf_with_signum(false, &b, zero_div),
            (PosInf, Finite(b)) => inf_with_signum(true, &b, zero_div),
            (Finite(a), Finite(b)) => match b.cmp_zero() {
                Some(Ordering::Equal) => Indeterminate(IndeterminanceReason::DivisionByZero),
                Some(_) => (a / b).into(),
                None => Indeterminate(IndeterminanceReason::NotComparableWithZero),
            },
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
/// The reason why some operation considered indetermined.
pub enum IndeterminanceReason {
    /// The result of operations like:
    ///  - `∞ + (-∞)`;
    ///  - `(-∞) + ∞`;
    ///  - `∞ - ∞`;
    ///  - `(-∞) - (-∞)`;
    ///
    /// Keep in mind, though, that adding two same-side infinities
    /// is (an absorbing) fully determined operation:
    ///  - `∞ + ∞ == ∞`;
    ///  - `(-∞) - ∞ == -∞`;
    SumOppositeInfinities,

    /// The result of operations like:
    /// - `0 * ∞`;
    /// - `∞ * 0`;
    /// - `0 * (-∞)`;
    /// - `(-∞) * 0`;
    ZeroInfProduct,

    /// The result of any division by 0.
    DivisionByZero,

    /// The value cannot be compared with 0 to determine the sign.
    NotComparableWithZero,

    /// The result of operations like:
    ///  - `∞ / ∞`;
    ///  - `∞ / (-∞)`;
    ///  - `(-∞) / ∞`;
    ///  - `(-∞) / (-∞)`;
    DivideInfinities,

    /// Other uncommon reason.
    Other(&'static str),
}

impl fmt::Display for IndeterminanceReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SumOppositeInfinities => write!(f, "∞ - ∞"),
            Self::ZeroInfProduct => write!(f, "∞ * 0"),
            Self::DivisionByZero => write!(f, "a / 0"),
            Self::NotComparableWithZero => write!(f, "a ? 0"),
            Self::DivideInfinities => write!(f, "∞ / ∞"),
            Self::Other(reason) => write!(f, "{reason}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inf_are_strict_bounds() {
        assert!(ExtNum::NegInf < ExtNum::Finite(i32::MIN));
        assert!(ExtNum::PosInf > ExtNum::Finite(i32::MAX));
    }
}
