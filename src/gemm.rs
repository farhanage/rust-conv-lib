use ndarray::{ArrayView2, ArrayViewMut2};

/// CPU matrix multiplication implementation using optimized approach
fn gemm_cpu_fallback(a: &ArrayView2<f32>, b: &ArrayView2<f32>, mut c: ArrayViewMut2<f32>) {
    let (m, k) = a.dim();
    let (_k2, n) = b.dim();
    
    // Zero out the output matrix first
    c.fill(0.0);
    
    // Optimized CPU GEMM with loop reordering for better cache locality
    for i in 0..m {
        for ki in 0..k {
            let a_ik = a[[i, ki]];
            for j in 0..n {
                c[[i, j]] += a_ik * b[[ki, j]];
            }
        }
    }
}

/// CUDA GEMM implementation (when cuda feature is enabled)
#[cfg(feature = "cuda")]
fn try_cuda_gemm(a: &ArrayView2<f32>, b: &ArrayView2<f32>, mut c: ArrayViewMut2<f32>) -> Result<(), Box<dyn std::error::Error>> {
    use cudarc::driver::{CudaDevice, DevicePtr, DevicePtrMut};
    use cudarc::cublas::CudaBlas;
    
    let (m, k) = a.dim();
    let (_k2, n) = b.dim();

    // Initialize CUDA device and CUBLAS
    let device = CudaDevice::new(0)?;
    let cublas = CudaBlas::new(device.clone())?;

    // Copy data to GPU
    let a_vec: Vec<f32> = a.iter().cloned().collect();
    let b_vec: Vec<f32> = b.iter().cloned().collect();
    
    let a_gpu = device.htod_copy(a_vec)?;
    let b_gpu = device.htod_copy(b_vec)?;
    let mut c_gpu = device.alloc_zeros::<f32>(m * n)?;

    // Use cudarc's raw sgemm function
    let result = unsafe {
        cudarc::cublas::result::sgemm(
            cublas.handle().clone(),
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N, // transa
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N, // transb
            n as i32,    // n
            m as i32,    // m
            k as i32,    // k
            &1.0f32,     // alpha
            *b_gpu.device_ptr() as *const f32, // B
            n as i32,    // ldb
            *a_gpu.device_ptr() as *const f32, // A
            k as i32,    // lda
            &0.0f32,     // beta
            *c_gpu.device_ptr_mut() as *mut f32, // C
            n as i32,    // ldc
        )
    };

    match result {
        Ok(_) => {
            // Copy result back to CPU
            let c_result = device.dtoh_sync_copy(&c_gpu)?;
            c.as_slice_mut().unwrap().copy_from_slice(&c_result);
            Ok(())
        }
        Err(e) => Err(e.into()),
    }
}

/// Matrix multiplication implementation with optional CUDA acceleration and CPU fallback
pub fn gemm(a: &ArrayView2<f32>, b: &ArrayView2<f32>, mut c: ArrayViewMut2<f32>) -> Result<(), Box<dyn std::error::Error>> {
    let (m, k) = a.dim();
    let (k2, n) = b.dim();
    assert_eq!(k, k2, "Matrix dimensions don't match for multiplication");
    assert_eq!(c.dim(), (m, n), "Output matrix has wrong dimensions");

    // Try CUDA first if available and enabled
    #[cfg(feature = "cuda")]
    {
        if let Ok(_) = try_cuda_gemm(a, b, c.view_mut()) {
            return Ok(());
        }
        // If CUDA fails, fallback to CPU (no error reporting, silent fallback)
    }
    
    // Use CPU implementation (either because CUDA is not available or failed)
    gemm_cpu_fallback(a, b, c.view_mut());
    Ok(())
}
