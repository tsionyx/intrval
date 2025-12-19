use core::fmt;

use crate::{Interval, OneOrPair};

#[derive(Debug)]
enum OperationNotation<S, T> {
    Zero(S),
    More(OneOrPair<(S, T)>),
}

impl<T> Interval<T> {
    const fn notation(&self) -> OperationNotation<&str, &T> {
        use OneOrPair::{One, Pair};
        use OperationNotation::{More, Zero};

        let notation = match self {
            Self::Empty => return Zero("{}"),
            Self::LessThan(x) => One(("< ", x)),
            Self::LessThanOrEqual(x) => One(("<= ", x)),
            #[cfg(feature = "singleton")]
            Self::Singleton(x) => One(("== ", x)),
            Self::GreaterThanOrEqual(x) => One((">= ", x)),
            Self::GreaterThan(x) => One(("> ", x)),
            Self::Open((a, b)) => Pair((("(", a), (")", b))),
            Self::LeftOpen((a, b)) => Pair((("(", a), ("]", b))),
            Self::RightOpen((a, b)) => Pair((("[", a), (")", b))),
            Self::Closed((a, b)) => Pair((("[", a), ("]", b))),
            Self::Full => return Zero("(..)"),
        };

        More(notation)
    }
}

impl<T: fmt::Display> fmt::Display for Interval<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.notation() {
            OperationNotation::More(OneOrPair::One((op, v))) => {
                f.write_str(op).and_then(|()| v.fmt(f))
            }
            OperationNotation::More(OneOrPair::Pair(((left_op, a), (right_op, b)))) => {
                f.write_str(left_op)?;
                a.fmt(f)?;
                f.write_str(", ")?;
                b.fmt(f)?;
                f.write_str(right_op)
            }
            OperationNotation::Zero(s) => f.write_str(s),
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::string::ToString as _;

    use crate::interval;

    use super::*;

    #[test]
    fn empty() {
        let i: Interval<f64> = interval!(_);
        assert_eq!(i.to_string(), "{}");
    }

    #[test]
    fn less() {
        let i: Interval<f64> = interval!(< -5.0);
        assert_eq!(i.to_string(), "< -5");
    }

    #[test]
    fn less_eq() {
        let i: Interval<f64> = interval!(<= -5.0);
        assert_eq!(i.to_string(), "<= -5");
    }

    #[test]
    fn eq() {
        let i: Interval<f64> = interval!(= -5.0);
        #[cfg(feature = "singleton")]
        assert_eq!(i.to_string(), "== -5");

        #[cfg(not(feature = "singleton"))]
        assert_eq!(i.to_string(), "[-5, -5]");
    }

    #[test]
    fn greater_eq() {
        let i: Interval<f64> = interval!(>= 5.0);
        assert_eq!(i.to_string(), ">= 5");
    }

    #[test]
    fn greater() {
        let i: Interval<f64> = interval!(> 5.0);
        assert_eq!(i.to_string(), "> 5");
    }

    #[test]
    fn open() {
        let i: Interval<f64> = interval!((5.0, 7.0));
        assert_eq!(i.to_string(), "(5, 7)");
    }

    #[test]
    fn left_open() {
        let i: Interval<f64> = interval!((5.0, =7.0));
        assert_eq!(i.to_string(), "(5, 7]");
    }

    #[test]
    fn right_open() {
        let i: Interval<f64> = interval!((=5.0, 7.0));
        assert_eq!(i.to_string(), "[5, 7)");
    }

    #[test]
    fn closed() {
        let i: Interval<f64> = interval!([5.0, 7.0]);
        assert_eq!(i.to_string(), "[5, 7]");
    }

    #[test]
    fn full() {
        let i: Interval<f64> = interval!(..);
        assert_eq!(i.to_string(), "(..)");
    }
}

#[cfg(all(feature = "serde", test))]
mod deser_tests {
    use serde_json::json;

    use crate::interval;

    use super::*;

    #[test]
    fn empty() {
        let j = json!("EMPTY");
        let i: Interval<f64> = serde_json::from_value(j).unwrap();
        assert_eq!(i, interval!(_));
    }

    #[test]
    fn less() {
        let expected = interval!(< -5.0);

        let j = json!({
            "LESS_THAN": -5.0
        });
        let i: Interval<f64> = serde_json::from_value(j).unwrap();
        assert_eq!(i, expected);

        let j = json!({
            "<": -5.0
        });
        let i: Interval<f64> = serde_json::from_value(j).unwrap();
        assert_eq!(i, expected);
    }

    #[test]
    fn less_eq() {
        let expected = interval!(<= -5.0);

        let j = json!({
            "LESS_THAN_OR_EQUAL": -5.0
        });
        let i: Interval<f64> = serde_json::from_value(j).unwrap();
        assert_eq!(i, expected);

        let j = json!({
            "<=": -5.0
        });
        let i: Interval<f64> = serde_json::from_value(j).unwrap();
        assert_eq!(i, expected);
    }

    #[cfg(feature = "singleton")]
    #[test]
    fn eq() {
        let expected = interval!(== 5.0);

        let j = json!({
            "SINGLETON": 5.0
        });
        let i: Interval<f64> = serde_json::from_value(j).unwrap();
        assert_eq!(i, expected);

        let j = json!({
            "==": 5.0
        });
        let i: Interval<f64> = serde_json::from_value(j).unwrap();
        assert_eq!(i, expected);
    }

    #[test]
    fn greater_eq() {
        let expected = interval!(>= 5.0);

        let j = json!({
            "GREATER_THAN_OR_EQUAL": 5.0
        });
        let i: Interval<f64> = serde_json::from_value(j).unwrap();
        assert_eq!(i, expected);

        let j = json!({
            ">=": 5.0
        });
        let i: Interval<f64> = serde_json::from_value(j).unwrap();
        assert_eq!(i, expected);
    }

    #[test]
    fn greater() {
        let expected = interval!(> 5.0);

        let j = json!({
            "GREATER_THAN": 5.0
        });
        let i: Interval<f64> = serde_json::from_value(j).unwrap();
        assert_eq!(i, expected);

        let j = json!({
            ">": 5.0
        });
        let i: Interval<f64> = serde_json::from_value(j).unwrap();
        assert_eq!(i, expected);
    }

    #[test]
    fn open() {
        let expected = interval!((5.0, 7.0));

        let j = json!({
            "BETWEEN_EXCLUSIVE": [5.0, 7.0],
        });
        let i: Interval<f64> = serde_json::from_value(j).unwrap();
        assert_eq!(i, expected);

        let j = json!({
            "(,)": [5.0, 7.0],
        });
        let i: Interval<f64> = serde_json::from_value(j).unwrap();
        assert_eq!(i, expected);

        let j = json!({
            "],[": [5.0, 7.0],
        });
        let i: Interval<f64> = serde_json::from_value(j).unwrap();
        assert_eq!(i, expected);
    }

    #[test]
    fn left_open() {
        let expected = interval!((5.0, =7.0));

        let j = json!({
            "LEFT_OPEN": [5.0, 7.0],
        });
        let i: Interval<f64> = serde_json::from_value(j).unwrap();
        assert_eq!(i, expected);

        let j = json!({
            "(,]": [5.0, 7.0],
        });
        let i: Interval<f64> = serde_json::from_value(j).unwrap();
        assert_eq!(i, expected);

        let j = json!({
            "],]": [5.0, 7.0],
        });
        let i: Interval<f64> = serde_json::from_value(j).unwrap();
        assert_eq!(i, expected);
    }

    #[test]
    fn right_open() {
        let expected = interval!((=5.0, 7.0));

        let j = json!({
            "BETWEEN": [5.0, 7.0],
        });
        let i: Interval<f64> = serde_json::from_value(j).unwrap();
        assert_eq!(i, expected);

        let j = json!({
            "[,)": [5.0, 7.0],
        });
        let i: Interval<f64> = serde_json::from_value(j).unwrap();
        assert_eq!(i, expected);

        let j = json!({
            "[,[": [5.0, 7.0],
        });
        let i: Interval<f64> = serde_json::from_value(j).unwrap();
        assert_eq!(i, expected);
    }

    #[test]
    fn closed() {
        let expected = interval!([5.0, 7.0]);

        let j = json!({
            "BETWEEN_INCLUSIVE": [5.0, 7.0],
        });
        let i: Interval<f64> = serde_json::from_value(j).unwrap();
        assert_eq!(i, expected);

        let j = json!({
            "[,]": [5.0, 7.0],
        });
        let i: Interval<f64> = serde_json::from_value(j).unwrap();
        assert_eq!(i, expected);
    }

    #[test]
    fn full() {
        let j = json!("FULL");
        let i: Interval<f64> = serde_json::from_value(j).unwrap();
        assert_eq!(i, interval!(..));
    }
}
