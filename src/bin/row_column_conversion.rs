//! `ndarray` is row-major and Cuda is column major, so we need to conversions
//! to make the maths work out.

use cudarc::cublas::Gemm;
use ndarray::Array2;

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
    let a_column_major = snnsim::to_column_major_slice(a_ndarray.view());
    let b_column_major = snnsim::to_column_major_slice(b_ndarray.view());

    let a_cuda = stream.memcpy_stod(&a_column_major).unwrap();
    let b_cuda = stream.memcpy_stod(&b_column_major).unwrap();
    let mut c_cuda = stream.alloc_zeros::<f32>(c_ndarray.len()).unwrap();

    let m = a_ndarray.nrows() as i32; // or c.nrows() or batch size
    let n = b_ndarray.ncols() as i32; // or c.ncols() or neurons
    let k = a_ndarray.ncols() as i32; // or b.nrows() or input size
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
    let e_cuda =
        snnsim::from_column_major_slice(c_ndarray.nrows(), c_ndarray.ncols(), &d_cuda).unwrap();
    assert_eq!(c_ndarray, e_cuda);
}
