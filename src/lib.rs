use ndarray::{Data, Ix2, RawData};

pub mod cuda;
pub mod encode;
pub mod net;

const DEFAULT_DECAY: f32 = 0.8f32;
const DEFAULT_THRESHOLD: f32 = 1f32;

/// Converts `slice` (which is a column-major representation of a matrix) into `ndarray::Array2`.
pub fn from_column_major_slice(
    rows: usize,
    columns: usize,
    slice: &[f32],
) -> Option<ndarray::Array2<f32>> {
    if rows * columns == slice.len() {
        Some(ndarray::Array2::from_shape_fn(
            [rows, columns],
            |(row, column)| slice[column * rows + row],
        ))
    } else {
        None
    }
}
/// Converts `n` into a column-major slice.
///
/// `ndarray` is row-major and CUDA is column major, so we need conversion.
pub fn to_column_major_slice<T: RawData + Data>(n: ndarray::ArrayBase<T, Ix2>) -> Vec<f32>
where
    f32: From<<T as RawData>::Elem>,
    <T as RawData>::Elem: Clone,
{
    n.columns().into_iter().fold(Vec::new(), |mut x, y| {
        x.extend(y.iter().cloned().map(f32::from));
        x
    })
}
