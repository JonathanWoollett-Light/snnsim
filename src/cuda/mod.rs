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
    let m = a.rows as i32; // or c.rows
    let n = b.columns as i32; // or c.columns
    let k = a.columns as i32; // or b.rows
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
                    lda: m,
                    ldb: k,
                    beta: 0f32,
                    ldc: m,
                },
                &a.slice,
                &b.slice,
                &mut c.slice,
            )
            .unwrap()
    }
}
