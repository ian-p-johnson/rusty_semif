//! `json.loads`-compatible recursive-descent parser.
//!
//! Divergences (documented, corpus-bounded): lone `\uD800`-style surrogates map
//! to U+FFFD where Python would carry them until a later encode failure with
//! the same exit code; `str.splitlines()`'s exotic separators beyond \r\n are
//! not split boundaries here.

use semif_types::PyValue;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("{message}")]
pub struct ParseError {
    pub message: String,
}

fn err<T>(message: impl Into<String>) -> Result<T, ParseError> {
    Err(ParseError {
        message: message.into(),
    })
}

struct Parser<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let byte = self.peek();
        if byte.is_some() {
            self.position += 1;
        }
        byte
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.position += 1;
        }
    }

    fn expect(&mut self, literal: &str) -> Result<(), ParseError> {
        if self.bytes[self.position..].starts_with(literal.as_bytes()) {
            self.position += literal.len();
            Ok(())
        } else {
            err("Expecting value")
        }
    }

    fn parse_value(&mut self) -> Result<PyValue, ParseError> {
        match self.peek() {
            None => err("Expecting value"),
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => Ok(PyValue::Str(self.parse_string()?)),
            Some(b't') => self.expect("true").map(|_| PyValue::Bool(true)),
            Some(b'f') => self.expect("false").map(|_| PyValue::Bool(false)),
            Some(b'n') => self.expect("null").map(|_| PyValue::Null),
            Some(b'N') => self.expect("NaN").map(|_| PyValue::Float(f64::NAN)),
            Some(b'I') => self
                .expect("Infinity")
                .map(|_| PyValue::Float(f64::INFINITY)),
            Some(b'-') if self.bytes[self.position..].starts_with(b"-Infinity") => self
                .expect("-Infinity")
                .map(|_| PyValue::Float(f64::NEG_INFINITY)),
            Some(byte) if byte == b'-' || byte.is_ascii_digit() => self.parse_number(),
            _ => err("Expecting value"),
        }
    }

    fn parse_object(&mut self) -> Result<PyValue, ParseError> {
        self.bump();
        let mut entries: Vec<(String, PyValue)> = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.bump();
            return Ok(PyValue::Object(entries));
        }
        loop {
            self.skip_whitespace();
            if self.peek() != Some(b'"') {
                return err("Expecting property name enclosed in double quotes");
            }
            let key = self.parse_string()?;
            self.skip_whitespace();
            if self.bump() != Some(b':') {
                return err("Expecting ':' delimiter");
            }
            self.skip_whitespace();
            let value = self.parse_value()?;
            if let Some(slot) = entries.iter_mut().find(|(name, _)| *name == key) {
                slot.1 = value;
            } else {
                entries.push((key, value));
            }
            self.skip_whitespace();
            match self.bump() {
                Some(b',') => continue,
                Some(b'}') => return Ok(PyValue::Object(entries)),
                _ => return err("Expecting ',' delimiter"),
            }
        }
    }

    fn parse_array(&mut self) -> Result<PyValue, ParseError> {
        self.bump();
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.bump();
            return Ok(PyValue::Array(items));
        }
        loop {
            self.skip_whitespace();
            items.push(self.parse_value()?);
            self.skip_whitespace();
            match self.bump() {
                Some(b',') => continue,
                Some(b']') => return Ok(PyValue::Array(items)),
                _ => return err("Expecting ',' delimiter"),
            }
        }
    }

    fn parse_string(&mut self) -> Result<String, ParseError> {
        self.bump();
        let mut text = String::new();
        loop {
            match self.bump() {
                None => return err("Unterminated string starting at"),
                Some(b'"') => return Ok(text),
                Some(b'\\') => match self.bump() {
                    Some(b'"') => text.push('"'),
                    Some(b'\\') => text.push('\\'),
                    Some(b'/') => text.push('/'),
                    Some(b'b') => text.push('\u{8}'),
                    Some(b'f') => text.push('\u{c}'),
                    Some(b'n') => text.push('\n'),
                    Some(b'r') => text.push('\r'),
                    Some(b't') => text.push('\t'),
                    Some(b'u') => {
                        let first = self.parse_hex4()?;
                        let code = if (0xD800..0xDC00).contains(&first) {
                            if self.bump() != Some(b'\\') || self.bump() != Some(b'u') {
                                return err("Invalid \\uXXXX escape");
                            }
                            let second = self.parse_hex4()?;
                            if !(0xDC00..0xE000).contains(&second) {
                                return err("Invalid \\uXXXX escape");
                            }
                            0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
                        } else if (0xDC00..0xE000).contains(&first) {
                            0xFFFD
                        } else {
                            first
                        };
                        text.push(char::from_u32(code).unwrap_or('\u{FFFD}'));
                    }
                    _ => return err("Invalid \\escape"),
                },
                Some(byte) if byte < 0x20 => return err("Invalid control character"),
                Some(byte) => {
                    let width = utf8_width(byte);
                    let start = self.position - 1;
                    self.position += width - 1;
                    let slice = self
                        .bytes
                        .get(start..self.position)
                        .ok_or_else(|| ParseError {
                            message: "Unterminated string".into(),
                        })?;
                    let chunk = std::str::from_utf8(slice).map_err(|_| ParseError {
                        message: "Invalid UTF-8 in input".into(),
                    })?;
                    text.push_str(chunk);
                }
            }
        }
    }

    fn parse_hex4(&mut self) -> Result<u32, ParseError> {
        let slice = self
            .bytes
            .get(self.position..self.position + 4)
            .ok_or_else(|| ParseError {
                message: "Invalid \\uXXXX escape".into(),
            })?;
        let text = std::str::from_utf8(slice).map_err(|_| ParseError {
            message: "Invalid \\uXXXX escape".into(),
        })?;
        let value = u32::from_str_radix(text, 16).map_err(|_| ParseError {
            message: "Invalid \\uXXXX escape".into(),
        })?;
        self.position += 4;
        Ok(value)
    }

    fn parse_number(&mut self) -> Result<PyValue, ParseError> {
        let start = self.position;
        if self.peek() == Some(b'-') {
            self.bump();
        }
        match self.peek() {
            Some(b'0') => {
                self.bump();
            }
            Some(byte) if byte.is_ascii_digit() => {
                while self.peek().is_some_and(|b| b.is_ascii_digit()) {
                    self.bump();
                }
            }
            _ => return err("Expecting value"),
        }
        let mut is_float = false;
        if self.peek() == Some(b'.') {
            is_float = true;
            self.bump();
            if !self.peek().is_some_and(|b| b.is_ascii_digit()) {
                return err("Expecting value");
            }
            while self.peek().is_some_and(|b| b.is_ascii_digit()) {
                self.bump();
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            is_float = true;
            self.bump();
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.bump();
            }
            if !self.peek().is_some_and(|b| b.is_ascii_digit()) {
                return err("Expecting value");
            }
            while self.peek().is_some_and(|b| b.is_ascii_digit()) {
                self.bump();
            }
        }
        let text =
            std::str::from_utf8(&self.bytes[start..self.position]).map_err(|_| ParseError {
                message: "Invalid UTF-8 in input".into(),
            })?;
        if is_float {
            let value: f64 = text.parse().map_err(|_| ParseError {
                message: "Invalid number".into(),
            })?;
            Ok(PyValue::Float(value))
        } else if text == "-0" {
            Ok(PyValue::Int("0".into()))
        } else {
            Ok(PyValue::Int(text.into()))
        }
    }
}

fn utf8_width(byte: u8) -> usize {
    match byte {
        0x00..=0x7F => 1,
        0xC2..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF4 => 4,
        _ => 1,
    }
}

/// Parse one JSON document with Python `json.loads` semantics.
pub fn parse(text: &str) -> Result<PyValue, ParseError> {
    let mut parser = Parser {
        bytes: text.as_bytes(),
        position: 0,
    };
    parser.skip_whitespace();
    let value = parser.parse_value()?;
    parser.skip_whitespace();
    if parser.position != parser.bytes.len() {
        return err("Extra data");
    }
    Ok(value)
}
