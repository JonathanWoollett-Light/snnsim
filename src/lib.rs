pub mod cuda;
pub mod encode;
pub mod net;

const DEFAULT_DECAY: f32 = 0.8f32;
const DEFAULT_THRESHOLD: f32 = 1f32;

/// `ndarray` is row-major and Cuda is column major, so we need conversion.
pub fn to_column_major_slice(n: ndarray::Array2<f32>) -> Vec<f32> {
    n.columns().into_iter().fold(Vec::new(), |mut x, y| {
        x.extend(y.into_iter());
        x
    })
}
