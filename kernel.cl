// OpenCL Kernel for stoachstic macrospin simulations

__constant float4 N = (float4)({{nxx}}f, {{nyy}}f, {{nzz}}f, 0.0f);
__constant float alpha = {{alpha}}f;
__constant float dt = {{dt}}f;
__constant float nu2 = {{nu2}}f;
__constant float nuSqrtDt = {{nuSqrtDt}}f;
__constant float stt = {{stt}}f;
__constant float flt = {{flt}}f;
__constant float hx = {{hx}}f;
__constant float hy = {{hy}}f;
__constant float hz = {{hz}}f;


__kernel void evolve(__global float4 *m, __global float4 *dW, __global float *currents, __global float *tilts, float amplitude) {

	// Groups organized such that current changes in 0th dimension, tilt in 1st dimension
	int i = get_global_id(1)*get_global_size(0) + get_global_id(0);

	float4 m_loc  = m[i];
	float4 dW_loc = dW[i];

	__local float current;
	__local float tilt;

	if (get_local_id(0) == 0) {
		current = currents[get_group_id(0)];
		tilt    = tilts[get_group_id(1)];
	}
    barrier(CLK_GLOBAL_MEM_FENCE);

	// Controlling the current pulses
	float sttmag = amplitude*current*stt;
	float fltmag = amplitude*current*flt;

	float cos_tilt, sin_tilt;
	sin_tilt = sincos(tilt, &cos_tilt);

	// Effective field 
	float4 heff = (float4)(hx, hy, hz, 0.0f) ;

	// Subtract the demag field
	heff = heff - m_loc*N;

	// // Spin transfer torque
	float4 stt = (float4)(-sttmag*cos_tilt, sttmag*sin_tilt, 0.0f, 0.0f);

	// Now for calculating the torques
	float4 hxm, pxm, mxhxm, nudWxm, numxdWxm;

	hxm      =  cross(heff, m_loc);
	pxm      =  cross(stt,  m_loc);
	mxhxm    =  cross(m_loc, hxm + pxm);
	nudWxm   =  nuSqrtDt*cross(dW_loc,m_loc);
	numxdWxm =  cross(m_loc, nudWxm);

	float4 deterministic_part = hxm + fma(alpha, mxhxm, -nu2*m_loc);
	float4 stochastic_part    = fma(alpha, numxdWxm, nudWxm); 
	m[i] = m_loc + dt*deterministic_part + stochastic_part;
}

__kernel void normalize_m(__global float4 *m) {
  //const int i = blockIdx.x*blockDim.x + threadIdx.x;
  int i = get_global_id(1)*get_global_size(0) + get_global_id(0);

  float4 mloc = m[i];
  m[i] = rsqrt(dot(mloc,mloc))*mloc;
}

__kernel void reduce_m(__global float4 *m, __global float *phase_diagram, int realizations) {
	// In this case get_global_size(0) = total number of current steps
	//              get_group_id(0) = current current step
	//              get_global_size(1) = total number of tilt steps
	//              get_group_id(1) = current tilt step

	// Realizations are manually looped over
	// We grab elements of m: m[r + get_group_id(0)*warp_x + get_group_id(1)*warp_y]

	int warp_x = realizations;
	int warp_y = warp_x*get_global_size(0);

	float sum = 0.0f;
	for (int r=0; r<realizations; r++) {
		sum += m[r + get_group_id(0)*warp_x + get_group_id(1)*warp_y].x ;
	}

    phase_diagram[get_group_id(0) + get_group_id(1)*get_global_size(0)] = sum/float(realizations);

}
