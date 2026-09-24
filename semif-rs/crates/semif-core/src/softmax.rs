//! Option-probability softmax, operation-for-operation with
//! `semif_phase1.core.softmax` (f64, max-shift, left-to-right sum).

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("Need at least two finite scores")]
pub struct SoftmaxError;

pub fn softmax(values: &[f64]) -> Result<Vec<f64>, SoftmaxError> {
    if values.len() < 2 || values.iter().any(|value| !value.is_finite()) {
        return Err(SoftmaxError);
    }
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = values.iter().map(|value| (value - maximum).exp()).collect();
    let mut total = 0.0;
    for weight in &weights {
        total += weight;
    }
    Ok(weights.iter().map(|weight| weight / total).collect())
}

#[cfg(test)]
mod tests {
    use super::softmax;

    #[test]
    fn extreme_spread_stays_finite() {
        let probabilities = softmax(&[1000.0, 999.0, -1000.0]).unwrap();
        assert!(probabilities.iter().all(|p| p.is_finite()));
        assert!(probabilities[0] > probabilities[1] && probabilities[1] > probabilities[2]);
    }

    #[test]
    fn ties_split_evenly() {
        let probabilities = softmax(&[0.0, 0.0]).unwrap();
        assert_eq!(probabilities, vec![0.5, 0.5]);
    }

    #[test]
    fn rejects_short_and_non_finite() {
        assert!(softmax(&[1.0]).is_err());
        assert!(softmax(&[1.0, f64::NAN]).is_err());
    }
}
