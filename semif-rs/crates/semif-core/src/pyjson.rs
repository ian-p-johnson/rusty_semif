//! The two Python-`json.dumps` writer profiles.
//!
//! Payload profile: `json.dumps(payload, ensure_ascii=False)` — raw unicode,
//! default `", "` / `": "` separators, insertion order preserved.
//! Output profile: `json.dumps(..., allow_nan=False)` — `ensure_ascii=True`
//! escapes, and non-finite floats raise with Python's exact message.

use crate::pyfloat::py_repr;
use semif_types::PyValue;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum JsonError {
    #[error("Out of range float values are not JSON compliant")]
    NonFinite,
}

pub fn dumps_payload(value: &PyValue) -> Result<String, JsonError> {
    let mut out = String::new();
    write_value(value, false, &mut out)?;
    Ok(out)
}

pub fn dumps_output(value: &PyValue) -> Result<String, JsonError> {
    let mut out = String::new();
    write_value(value, true, &mut out)?;
    Ok(out)
}

fn write_value(value: &PyValue, ensure_ascii: bool, out: &mut String) -> Result<(), JsonError> {
    match value {
        PyValue::Null => out.push_str("null"),
        PyValue::Bool(true) => out.push_str("true"),
        PyValue::Bool(false) => out.push_str("false"),
        PyValue::Int(digits) => out.push_str(digits),
        PyValue::Float(f) => {
            if !f.is_finite() {
                return Err(JsonError::NonFinite);
            }
            out.push_str(&py_repr(*f));
        }
        PyValue::Str(text) => write_string(text, ensure_ascii, out),
        PyValue::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_value(item, ensure_ascii, out)?;
            }
            out.push(']');
        }
        PyValue::Object(entries) => {
            out.push('{');
            for (index, (key, item)) in entries.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_string(key, ensure_ascii, out);
                out.push_str(": ");
                write_value(item, ensure_ascii, out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

fn write_string(text: &str, ensure_ascii: bool, out: &mut String) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c if ensure_ascii && (c as u32) > 0x7E => {
                let code = c as u32;
                if code > 0xFFFF {
                    let high = 0xD800 + ((code - 0x10000) >> 10);
                    let low = 0xDC00 + ((code - 0x10000) & 0x3FF);
                    out.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
                } else {
                    out.push_str(&format!("\\u{code:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::{dumps_output, dumps_payload};
    use crate::parser::parse;

    #[test]
    fn payload_profile_keeps_unicode_raw() {
        let value = parse("{\"k\": \"中文\"}").unwrap();
        assert_eq!(dumps_payload(&value).unwrap(), "{\"k\": \"中文\"}");
    }

    #[test]
    fn output_profile_escapes_non_ascii() {
        let value = parse("{\"k\": \"中文\"}").unwrap();
        assert_eq!(dumps_output(&value).unwrap(), "{\"k\": \"\\u4e2d\\u6587\"}");
    }

    #[test]
    fn output_profile_uses_surrogate_pairs() {
        let value = parse("\"🔥\"").unwrap();
        assert_eq!(dumps_output(&value).unwrap(), "\"\\ud83d\\udd25\"");
    }

    #[test]
    fn output_profile_rejects_non_finite() {
        let value = parse("{\"x\": NaN}").unwrap();
        assert_eq!(
            dumps_output(&value).unwrap_err().to_string(),
            "Out of range float values are not JSON compliant"
        );
    }

    #[test]
    fn separators_and_order_match_python_defaults() {
        let value = parse("{\"a\": 1, \"b\": [true, null], \"c\": \"x y\"}").unwrap();
        assert_eq!(
            dumps_payload(&value).unwrap(),
            "{\"a\": 1, \"b\": [true, null], \"c\": \"x y\"}"
        );
    }
}
