//! `repr(float)`-compatible f64 formatting.
//!
//! Python switches to scientific notation when the decimal exponent is outside
//! [-4, 15]; exponents render with a sign and at least two digits. Integral
//! floats keep one fractional digit ("100.0"). Shortest-roundtrip digits come
//! from Rust's `{:e}` formatting, which like Python produces the shortest
//! digit string that round-trips.

/// Format an f64 the way Python `repr`/`str` (and `json.dumps`) would.
/// Non-finite inputs are a caller error (writers reject them first).
pub fn py_repr(value: f64) -> String {
    debug_assert!(value.is_finite(), "callers must reject non-finite floats");
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("e-format always has an exponent");
    let exp10: i32 = exponent.parse().expect("e-format exponent");
    let negative = mantissa.starts_with('-');
    let digits: String = mantissa.chars().filter(|c| c.is_ascii_digit()).collect();

    let body = if (-4..16).contains(&exp10) {
        positional(&digits, exp10)
    } else {
        scientific_form(&digits, exp10)
    };
    if negative { format!("-{body}") } else { body }
}

fn positional(digits: &str, exp10: i32) -> String {
    let digits: Vec<char> = digits.chars().collect();
    let point = exp10 + 1;
    let mut out = String::new();
    if point <= 0 {
        out.push_str("0.");
        for _ in 0..(-point) {
            out.push('0');
        }
        out.extend(&digits);
    } else {
        let int_len = (point as usize).min(digits.len());
        out.extend(&digits[..int_len]);
        for _ in int_len..point as usize {
            out.push('0');
        }
        let fraction: String = digits[int_len..].iter().collect();
        if fraction.is_empty() {
            out.push_str(".0");
        } else {
            out.push('.');
            out.push_str(&fraction);
        }
    }
    out
}

fn scientific_form(digits: &str, exp10: i32) -> String {
    let mut out = String::new();
    let mut chars = digits.chars();
    out.push(chars.next().expect("at least one digit"));
    let rest: String = chars.collect();
    if !rest.is_empty() {
        out.push('.');
        out.push_str(&rest);
    }
    out.push('e');
    if exp10 < 0 {
        out.push('-');
    } else {
        out.push('+');
    }
    let magnitude = exp10.unsigned_abs();
    if magnitude < 10 {
        out.push('0');
    }
    out.push_str(&magnitude.to_string());
    out
}

#[cfg(test)]
mod tests {
    use super::py_repr;

    #[test]
    fn zero_forms() {
        assert_eq!(py_repr(0.0), "0.0");
        assert_eq!(py_repr(-0.0), "-0.0");
    }

    #[test]
    fn positional_forms() {
        assert_eq!(py_repr(1.5), "1.5");
        assert_eq!(py_repr(150.0), "150.0");
        assert_eq!(py_repr(0.1), "0.1");
        assert_eq!(py_repr(0.0001), "0.0001");
        assert_eq!(py_repr(1234567890123456800.0), "1.2345678901234568e+18");
        assert_eq!(py_repr(123456789012345.67), "123456789012345.67");
        assert_eq!(py_repr(1e15), "1000000000000000.0");
        assert_eq!(py_repr(-2.5), "-2.5");
    }

    #[test]
    fn scientific_forms() {
        assert_eq!(py_repr(1e16), "1e+16");
        assert_eq!(py_repr(1.5e-5), "1.5e-05");
        assert_eq!(py_repr(5e-324), "5e-324");
        assert_eq!(py_repr(1.7976931348623157e308), "1.7976931348623157e+308");
    }
}
