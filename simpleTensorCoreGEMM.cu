#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
  if (stat != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
  }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
  }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
  if (stat != CURAND_STATUS_SUCCESS) {
    fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
  }
}

#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/wmma_gemm_traits.h>

#include <mma.h>
using namespace nvcuda;

// Must be multiples of 16 for wmma code to work
#define MATRIX_M 16 // 12 x 4
#define MATRIX_N 32*12*12*12 // 16x12x12x12
//#define MATRIX_N 16 // 16x12x12x12
#define MATRIX_K 16 // MATRIX_M

// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

//template< class hf >
//__forceinline __device__ void crop_tile(hf* to, hf* from,
//                          const int& col,
//                          const int& t_row,         const int& t_col, 
//                          cosnt int& row_dim,       const int& col_dim // column major
//                          const int& wmma_row_dim,  const int& wmma_col_dim)
//{
////  for(int col = t_col*wmma_col_dim; col < (t_col+1)*wmma_col_dim; col++){
//    memcpy( to+(col-t_col*wmma_col_dim)*wmma_row_dim, 
//            from+col*row_dim+t_row*wmma_row_dim, 
//            wmma_row_dim*sizeof(hf) );
////  }
//}

// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16. 
//  3) Neither A nor B are transposed.
// Note: This is NOT a high performance example but is for demonstration purposes only
//       For a high performance code please use the GEMM provided in cuBLAS.
__global__ void wmma_example(half *a, half *b, float *c, int M, int N, int K, float alpha, float beta) {

  // M, N, K are the block-wise matrix dim.
  int tm_dim = M/WMMA_M; // tile dimension in m.
  int tn_dim = N/WMMA_N;
  int tk_dim = K/WMMA_K;

  // Declare all shared memory.
  extern __shared__ float4 sm[];

  half* sm_a = (half*)sm;
  half* sm_b = sm_a + M*K;
  float* sm_c = (float*)(sm_b + K*N);

//  half* wmma_a = (half*)(sm_c + M*N);
//  half* wmma_b = wmma_a + WMMA_M*WMMA_K;
//  float* wmma_c = (float*)(wmma_b + WMMA_K*WMMA_N);

  // Copy stuff from global memory to shared memory.
  int global_n = blockIdx.x*blockDim.x+threadIdx.x;

  if(threadIdx.x == 0){
    for(int k = 0; k < blockDim.y; k++){
      sm_a[threadIdx.y*blockDim.y+k] = a[threadIdx.y*blockDim.y+k];
    }
  }

  __syncthreads();

  //  sm_b[threadIdx.y*blockDim.x+threadIdx.x] = b[threadIdx.y*blockDim.x*gridDim.x+global_n];
  sm_b[threadIdx.x*blockDim.y+threadIdx.y] = b[global_n*blockDim.y+threadIdx.y];

  __syncthreads();

// Retard version

//  sm_c[threadIdx.x*blockDim.y+threadIdx.y] = 0.;
//  for(int k = 0; k < blockDim.y; k++){
//    sm_c[threadIdx.x*blockDim.y+threadIdx.y] += float(sm_a[k*blockDim.y+threadIdx.y]*sm_b[threadIdx.x*blockDim.y+k]);
//  }

  __syncthreads();

// CUTLASS version. Broken.
/*
  typedef cutlass::gemm::WmmaGemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor, 
    cutlass::Shape<32, 32, 32> // K by N by M
  > WmmaGemmTraits;

  typedef cutlass::gemm::Gemm<WmmaGemmTraits> Gemm;
  __shared__ typename Gemm::SharedStorage shared_storage;

  typename Gemm::Params params;
  
  params.initialize( 32, 32, 32, alpha, sm_a, 32, sm_b, 32, beta, sm_c, 32, sm_c, 32 );
  
  Gemm gemm(params, shared_storage);
  gemm.multiply_add();
*/

// WMMA version.

  // The actual/physical warp assigned to each thread in this block
  int phys_warp_n_dim = blockDim.x/warpSize;
  int phys_warp_m_dim = blockDim.y;

  int phys_warp_n = threadIdx.x/warpSize;
  int phys_warp_m = threadIdx.y; 

  int phys_warp_index = phys_warp_n*phys_warp_m_dim+phys_warp_m;

  // The logical warp assigned to each part of the matrix.
  int warp_n = phys_warp_index / tm_dim;
  int warp_m = phys_warp_index % tm_dim;

  // Set up the wmma stuff
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  // Zero the initial acc.
  wmma::fill_fragment(c_frag, 0.0f);
  
  for( int k = 0; k < K; k+=WMMA_K ){
    int a_row = warp_m*WMMA_M;
    int a_col = k;

    int b_row = k;
    int b_col = warp_n*WMMA_N;
    
    if(a_row < M && a_col < K && b_row < K && b_col < N) {    
      // Load Matrix
      wmma::load_matrix_sync(a_frag, sm_a+a_row+a_col*M, M);
      wmma::load_matrix_sync(b_frag, sm_b+b_row+b_col*K, K);
      // Perform the matrix multiplication
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    __syncthreads();
  } 
  
  __syncthreads();

  int c_row = warp_m*WMMA_M;
  int c_col = warp_n*WMMA_N;
  
  if(c_row < M && c_col < N){ 
    wmma::store_matrix_sync(sm_c+c_row+c_col*M, c_frag, M, wmma::mem_col_major);
  }

  __syncthreads();
  // Store result to global memory
  //  c[threadIdx.y*blockDim.x*gridDim.x+global_n] = sm_c[threadIdx.y*blockDim.x+threadIdx.x];
  c[global_n*blockDim.y+threadIdx.y] = sm_c[threadIdx.x*blockDim.y+threadIdx.y];

}

__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx];
  }
}

int main(int argc, char* argv[]) {
  float *a_fp32;
  float *b_fp32;
  half *a_fp16;
  half *b_fp16;

  float *c;
  float *c_cublas;
  float *c_wmma;

  float *c_host_cublas;
  float *c_host_wmma;

  curandGenerator_t gen;
  cublasHandle_t cublasHandle;

  cudaEvent_t startWMMA;
  cudaEvent_t stopWMMA;

  cudaEvent_t startcublas;
  cudaEvent_t stopcublas;

  cudaErrCheck(cudaEventCreate(&startWMMA));
  cudaErrCheck(cudaEventCreate(&stopWMMA));

  cudaErrCheck(cudaEventCreate(&startcublas));
  cudaErrCheck(cudaEventCreate(&stopcublas));


  cublasErrCheck(cublasCreate(&cublasHandle));

  // Use tensor cores
  cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));

  cudaErrCheck(cudaMalloc((void**)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&b_fp32, MATRIX_K * MATRIX_N * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
  cudaErrCheck(cudaMalloc((void**)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));

  cudaErrCheck(cudaMalloc((void**)&c, MATRIX_M * MATRIX_N * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&c_cublas, MATRIX_M * MATRIX_N * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&c_wmma, MATRIX_M * MATRIX_N * sizeof(float)));

  c_host_cublas = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
  c_host_wmma = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));

  curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

  curandErrCheck(curandGenerateUniform(gen, a_fp32, MATRIX_M * MATRIX_K));
  curandErrCheck(curandGenerateUniform(gen, b_fp32, MATRIX_K * MATRIX_N));

  // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
  convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_fp16, a_fp32, MATRIX_M * MATRIX_K);
  convertFp32ToFp16 <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (b_fp16, b_fp32, MATRIX_K * MATRIX_N);

  curandErrCheck(curandGenerateUniform(gen, c, MATRIX_M * MATRIX_N));

  curandErrCheck(curandDestroyGenerator(gen));

  cudaErrCheck(cudaMemcpy(c_cublas, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));
  cudaErrCheck(cudaMemcpy(c_wmma, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));

  float alpha = 1.0f;
  float beta = 0.0f;


  printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);

  // First: using WMMA
  dim3 gridDim;
  dim3 blockDim;

  // blockDim.x must be a multple of warpSize
  // 128x4 means we have 16 warps and a block computes a 64x64 output tile
  blockDim.x = 32;
  blockDim.y = MATRIX_M;

  gridDim.x = (MATRIX_N + blockDim.x-1) / blockDim.x;
  gridDim.y = 1;

  printf("Running with wmma...\n");
  cudaErrCheck(cudaEventRecord(startWMMA));
  int shared_memory_size = blockDim.y*blockDim.y*2+blockDim.y*blockDim.x*2+blockDim.y*blockDim.x*4;
  wmma_example <<< gridDim, blockDim, shared_memory_size*2 >>> (a_fp16, b_fp16, c_wmma, blockDim.y, blockDim.x, blockDim.y, alpha, beta);
  cudaErrCheck(cudaEventRecord(stopWMMA));



  // Now using cuBLAS
  printf("Running with cuBLAS...\n");
  cudaErrCheck(cudaEventRecord(startcublas));
  cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
        MATRIX_M, MATRIX_N, MATRIX_K, 
        &alpha,
        a_fp16, CUDA_R_16F, MATRIX_M,
        b_fp16, CUDA_R_16F, MATRIX_K,
        &beta, 
        c_cublas, CUDA_R_32F, MATRIX_M,
        CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
  cudaErrCheck(cudaEventRecord(stopcublas));

  // Error checking
  printf("\nChecking results...\n");
  cudaErrCheck(cudaMemcpy(c_host_wmma, c_wmma, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(c_host_cublas, c_cublas, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));

  // 0.01% relative tolerance. 1e-5 absolute tolerance.
  int errors = 0;
  for (int i = 0; i < MATRIX_M * MATRIX_N; i++) {
    float v1 = c_host_wmma[i];
    float v2 = c_host_cublas[i];
    if (v1 / v2 > 1.0001 || v2 / v1 > 1.0001 || abs(v1 - v2) > 1e-5) {
      errors++;
      if (errors < 64) printf("%06d %f %f\n", i, v1, v2);
    }
  }

  if (errors > 0) {
    printf("WMMA does not agree with cuBLAS! %d errors!\n", errors);
  }
  else {
    printf("Results verified: cublas and WMMA agree.\n\n");
    float wmmaTime;
    float cublasTime;
    cudaErrCheck(cudaEventSynchronize(stopWMMA));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWMMA, stopWMMA));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    printf("wmma took %fms\n", wmmaTime);
    printf("cublas took %fms\n", cublasTime);

    printf("\nFor a faster code using wmma you should check out the cudaTensorCoreGemm sample in the CUDA Toolkit.\nThis code was written as a demo only!\n\n");
  }


  cudaErrCheck(cudaEventDestroy(startWMMA));
  cudaErrCheck(cudaEventDestroy(stopWMMA));

  cudaErrCheck(cudaEventDestroy(startcublas));             
  cudaErrCheck(cudaEventDestroy(stopcublas));

  cudaErrCheck(cudaFree(a_fp32));
  cudaErrCheck(cudaFree(b_fp32));
  cudaErrCheck(cudaFree(a_fp16));
  cudaErrCheck(cudaFree(b_fp16));

  cudaErrCheck(cudaFree(c));
  cudaErrCheck(cudaFree(c_cublas));
  cudaErrCheck(cudaFree(c_wmma));

  free(c_host_cublas);
  free(c_host_wmma);

  cudaErrCheck(cudaDeviceReset());
  return 0;
}


