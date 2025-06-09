use ndarray::{Data, Ix2, RawData};

pub mod cuda;
pub mod encode;
#[cfg(feature = "mnist")]
pub mod mnist;
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

pub enum PollingResult<A, B> {
    Incomplete(A),
    Complete(B),
}

pub trait PollingIterator {
    type Item;
    fn next(self) -> PollingResult<Self, Self::Item>
    where
        Self: Sized;
    fn zip<T: PollingIterator>(self, other: T) -> ZippedPollingIterator<Self, T>
    where
        Self: Sized,
    {
        ZippedPollingIterator { a: self, b: other }
    }
    fn finish(mut self) -> Self::Item
    where
        Self: Sized,
    {
        loop {
            self = match self.next() {
                PollingResult::Complete(x) => break x,
                PollingResult::Incomplete(x) => x,
            };
        }
    }
}

pub struct ZippedPollingIterator<A: PollingIterator, B: PollingIterator> {
    pub a: A,
    pub b: B,
}
impl<A: PollingIterator, B: PollingIterator> PollingIterator for ZippedPollingIterator<A, B> {
    type Item = (<A as PollingIterator>::Item, <B as PollingIterator>::Item);
    fn next(self) -> PollingResult<Self, Self::Item>
    where
        Self: Sized,
    {
        match (self.a.next(), self.b.next()) {
            (PollingResult::Incomplete(a), PollingResult::Incomplete(b)) => {
                PollingResult::Incomplete(Self { a, b })
            }
            (PollingResult::Complete(a), PollingResult::Complete(b)) => {
                PollingResult::Complete((a, b))
            }
            _ => unreachable!(),
        }
    }
}
