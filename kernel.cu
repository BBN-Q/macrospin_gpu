#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#endif

#include <stdio.h>
#include <pycuda-complex.hpp>

// ============================================================
//                      Helper Functions
// ============================================================

inline __device__ float4 cross(float4 left, float4 right) {
  return make_float4(left.y*right.z - left.z*right.y,
		     left.z*right.x - left.x*right.z,
		     left.x*right.y - left.y*right.x,
		     0.0f);
}
inline __device__ float4 xcross(float left, float4 right) {
  // left only has an x component
  return make_float4(0.0f, -left*right.z, left*right.y, 0.0f);
}
inline __device__ float4 operator+(float4 a, float4 b) {
  return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, 0.0f);
}
inline __device__ float4 mult(float4 a, float4 b) {
  return make_float4(a.x*b.x, a.y*b.y, a.z*b.z, 0.0f);
}
inline __device__ float4 operator-(float4 a, float4 b) {
  return make_float4(a.x-b.x, a.y-b.y, a.z-b.z, 0.0f);
}
inline __device__ float4 operator*(float b, float4 a) {
  return make_float4(a.x*b, a.y*b, a.z*b, 0.0f);
}
inline __device__ float magInv(float4 a) {
  return rsqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
}

// ============================================================
//                      Physics Code
// ============================================================

__global__ void evolve_current_vs_tilt(float4 *m,
                                       float4 *dW,
                                       float simulationTime,
                                       float pulseEnvelope
                                      ) {

  // Where are we? What are the duration and sigma of the pulse here?
  const int i      = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x)*blockDim.x); 
  float current    = {{minCurrent}} + float(blockIdx.x)*(({{currentRange}})/float(gridDim.x-1.0));
  float tilt       = {{minTilt}} + float(blockIdx.y)*(({{tiltRange}})/float(gridDim.y-1.0));

  // Demag tensor
  const float4 N = make_float4({{nxx}}, {{nyy}}, {{nzz}}, 0.0f);

  // Where we are...
  float4 mloc = m[i];

  // Controlling the current pulses
  float sttmag = pulseEnvelope*current*{{stt}};
  float fltmag = pulseEnvelope*current*{{flt}};

  float cosf_tilt, sinf_tilt;
  sincosf(tilt, &sinf_tilt, &cosf_tilt);

  // Effective field including demag
  float4 heff = make_float4({{hx}} - fltmag*cosf_tilt, {{hy}} - fltmag*sinf_tilt, {{hz}}, 0.0f) 
                            - mult(mloc, N);

  // Spin transfer torque
  const float4 stt = make_float4(-sttmag*cosf_tilt, sttmag*sinf_tilt, 0.0f, 0.0f);
  
  // Now for calculating the torques
  float4 hxm, pxm, mxhxm, nudWxm, numxdWxm;
  
  hxm      =  cross(heff, mloc);
  pxm      =  cross(stt,  mloc);
  mxhxm    =  cross(mloc, hxm + pxm);
  nudWxm   =  {{nuSqrtDt}}*cross(dW[threadIdx.x],mloc);
  numxdWxm =  cross(mloc, nudWxm);

  m[i] = mloc + {{dt}}*(hxm + {{alpha}}*mxhxm - {{nu2}}*mloc) + (nudWxm + {{alpha}}*numxdWxm);

}

__global__ void reduceM(float4 *m, float *phaseDiagram, int threadsPerBlock) {
  // Each thread in this kernel reduces an entire block as
  // run in the actual simulation kernel. 

  // The thread for this kernel
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  // This blockIdx.x corresponds to blockIdx.y in the main kernel
  // This threadIdx.x correspondings to blockIdx.x in the main kernel

  float sum = 0.0;

  // Loop over threads blockwise as used by the main kernel
  //int ref = (((blockDim.x * blockIdx.x) + threadIdx.x)*threadsPerBlock);
  int ref = i*threadsPerBlock;
  for (int t=0; t<threadsPerBlock; t++) {
    int ind = t + ref;
    float4 mloc = m[ind];
    sum += (mloc.z < 0.0f) ? 1.0f : 0.0f;
    // sum += mloc.z;
  }

  // Average the results
  phaseDiagram[i] = sum/float(threadsPerBlock);

}

__global__ void normalizeM(float4 *m) {
  //const int i = blockIdx.x*blockDim.x + threadIdx.x;
  const int i = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x)*blockDim.x); 
  
  float4 mloc = m[i];
  m[i] = rsqrt(mloc*mloc)*mloc;
}

__global__ void resetM(float4 *m, float mx, float my, float mz) {
  //const int i = blockIdx.x*blockDim.x + threadIdx.x;
  const int i = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x)*blockDim.x); 
  m[i] = make_float4(mx, my, mz, 0.0f);
}