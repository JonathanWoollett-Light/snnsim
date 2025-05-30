//! `ndarray` is row-major and Cuda is column major, so we need to conversions
//! to make the maths work out.

use cudarc::cublas::Gemm;
use ndarray::{Array2, Axis};

fn main() {
    let a_ndarray =
        Array2::<f32>::from_shape_vec([4, 2], vec![1.0f32, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
            .unwrap();
    let b_ndarray =
        Array2::<f32>::from_shape_vec([2, 3], vec![2.0f32, 0.21, 0.0, 0.0, 0.21, 2.0]).unwrap();
    println!("a_ndarray: {:?}", a_ndarray.as_slice().unwrap());
    println!("b_ndarray: {:?}", b_ndarray.as_slice().unwrap());

    let c_ndarray = a_ndarray.dot(&b_ndarray);
    // let d_ndarray = c_ndarray.as_slice().unwrap();

    let context = cudarc::driver::CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let cublas = cudarc::cublas::CudaBlas::new(stream.clone()).unwrap();

    // Convert from ndarray row-major to cuda column-major
    let a_column_major = a_ndarray
        .columns()
        .into_iter()
        .fold(Array2::zeros([0, a_ndarray.nrows()]), |mut x, y| {
            x.append(Axis(0), y.insert_axis(Axis(0))).unwrap();
            x
        })
        .into_raw_vec_and_offset()
        .0;
    let b_column_major = b_ndarray
        .columns()
        .into_iter()
        .fold(Array2::zeros([0, b_ndarray.nrows()]), |mut x, y| {
            x.append(Axis(0), y.insert_axis(Axis(0))).unwrap();
            x
        })
        .into_raw_vec_and_offset()
        .0;

    let a_cuda = stream.memcpy_stod(&a_column_major).unwrap();
    let b_cuda = stream.memcpy_stod(&b_column_major).unwrap();
    let mut c_cuda = stream.alloc_zeros::<f32>(c_ndarray.len()).unwrap();

    let m = a_ndarray.nrows() as i32; // or c.nrows()
    let n = b_ndarray.ncols() as i32; // or c.ncols()
    let k = a_ndarray.ncols() as i32; // or b.nrows()
    unsafe {
        cublas
            .gemm(
                cudarc::cublas::GemmConfig {
                    transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                    transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                    m,
                    n,
                    k,
                    alpha: 1f32,
                    lda: m,
                    ldb: k,
                    beta: 0f32,
                    ldc: m,
                },
                &a_cuda,
                &b_cuda,
                &mut c_cuda,
            )
            .unwrap();
    }
    let d_cuda = stream.memcpy_dtov(&c_cuda).unwrap();

    // Convert cuda array back to row-major ndarray.
    let e_cuda = Array2::from_shape_fn([c_ndarray.nrows(), c_ndarray.ncols()], |(row, column)| {
        d_cuda[column * c_ndarray.nrows() + row]
    });
    assert_eq!(c_ndarray, e_cuda);
}
