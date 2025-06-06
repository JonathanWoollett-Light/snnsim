use std::sync::Arc;

use cudarc::{
    cublas::{CudaBlas, Gemm, GemmConfig, sys::cublasOperation_t},
    driver::{CudaSlice, CudaStream},
};
use ndarray::{Data, Ix2, RawData};

use crate::{from_column_major_slice, to_column_major_slice};

pub mod kernels;
pub mod net;

#[derive(Debug, Clone)]
pub struct CudaMatrix {
    slice: CudaSlice<f32>,
    rows: usize,
    columns: usize,
}
impl CudaMatrix {
    pub fn len(&self) -> usize {
        self.rows * self.columns
    }
    pub fn zeros(stream: Arc<CudaStream>, rows: usize, columns: usize) -> Self {
        Self {
            slice: stream.alloc_zeros(rows * columns).unwrap(),
            rows,
            columns,
        }
    }
    pub fn from_ndarray<T: RawData + Data>(
        stream: Arc<CudaStream>,
        ndarray: ndarray::ArrayBase<T, Ix2>,
    ) -> CudaMatrix
    where
        f32: From<<T as RawData>::Elem>,
        <T as RawData>::Elem: Clone,
    {
        CudaMatrix {
            slice: stream
                .memcpy_stod(&to_column_major_slice(ndarray.view()))
                .unwrap(),
            rows: ndarray.nrows(),
            columns: ndarray.ncols(),
        }
    }
    pub fn to_ndarray(&self, stream: Arc<CudaStream>) -> ndarray::Array2<f32> {
        from_column_major_slice(
            self.rows,
            self.columns,
            &stream.memcpy_dtov(&self.slice).unwrap(),
        )
        .unwrap()
    }
}

pub fn matmul(
    cublas: &CudaBlas,
    a: &CudaMatrix,
    b: &CudaMatrix,
    c: &mut CudaMatrix,
    trans_a: bool,
    trans_b: bool,
) {
    gemm(cublas, a, b, c, trans_a, trans_b, 0f32)
}
/// Performs the BLAS matrix-matrix multiplication operation.
///
/// See reference https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2023-2/gemm.html
pub fn gemm(
    cublas: &CudaBlas,
    a: &CudaMatrix,
    b: &CudaMatrix,
    c: &mut CudaMatrix,
    trans_a: bool,
    trans_b: bool,
    beta: f32,
) {
    let m = c.rows as i32; // or `op(a.rows)`
    let n = c.columns as i32; // or `op(b.columns)`
    let k = if trans_a { a.rows } else { a.columns } as i32; // or the same for `b`
    unsafe {
        cublas
            .gemm(
                GemmConfig {
                    transa: if trans_a {
                        cublasOperation_t::CUBLAS_OP_T
                    } else {
                        cublasOperation_t::CUBLAS_OP_N
                    },
                    transb: if trans_b {
                        cublasOperation_t::CUBLAS_OP_T
                    } else {
                        cublasOperation_t::CUBLAS_OP_N
                    },
                    m,
                    n,
                    k,
                    alpha: 1f32,
                    lda: if trans_a { k } else { m },
                    ldb: if trans_b { n } else { k },
                    beta,
                    ldc: m,
                },
                &a.slice,
                &b.slice,
                &mut c.slice,
            )
            .unwrap()
    }
}
