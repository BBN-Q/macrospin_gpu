// OpenCL Kernel for stoachstic macrospin simulations

__constant float4 N = (float4)({{nxx}}, {{nyy}}, {{nzz}}, 0.0f);
__constant float alpha = {{alpha}};
__constant float dt = {{dt}};
__constant float nu2 = {{nu2}};
__constant float nuSqrtDt = {{nuSqrtDt}};
__constant float stt = {{stt}};
__constant float flt = {{flt}};
__constant float hx = {{hx}};
__constant float hy = {{hy}};
__constant float hz = {{hz}};


__kernel void evolve(__global float4 *m, __global float4 *dW, __global float *currents, __global float *tilts, float amplitude) {

	// Groups organized such that current changes in 0th dimension, tilt in 1st dimension
	int i = get_global_id(1)*get_global_size(0) + get_global_id(0);
	// float current    = {{minCurrent}} + float(get_group_id(0))*{{currentRange}}/float(get_num_groups(0)-1.0);
	// float tilt       = {{minTilt}} + float(get_group_id(1))*{{tiltRange}}/float(get_num_groups(1)-1.0);

	float4 m_loc  = m[i];
	float4 dW_loc = dW[i];

	__local float current;
	__local float tilt;

	if (get_local_id(0) == 0) {
		current = currents[get_group_id(0)];
		tilt    = tilts[get_group_id(1)];

		// printf("Setting current to %2.2g\n", current);
		// printf("Setting tilt to %2.2g\n", tilt);
	}
    barrier(CLK_GLOBAL_MEM_FENCE);

	// printf("GlobalID: %d, (%d of %d, %d of %d, %d) - tilt = %2.4f - current - %2.4g\n", i, get_group_id(0), get_num_groups(0), 
	// 													   get_group_id(1), get_num_groups(1), get_local_id(0), 
	// 													   tilt, current);
		// printf("f4 = %2.2v4hlf\n", f);

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

	// m[i] = m_loc + {{dt}}*(hxm + {{alpha}}*mxhxm - {{nu2}}*mloc) + (nudWxm + {{alpha}}*numxdWxm);

	float4 deterministic_part = hxm + fma(alpha, mxhxm, -nu2*m_loc);
	float4 stochastic_part    = fma(alpha, numxdWxm, nudWxm); 
	m[i] = m_loc + dt*deterministic_part + stochastic_part;
	// m[i] = (float4)(current+tilt*1e8, 0, 0, 0);
}

__kernel void normalize_m(__global float4 *m) {
  //const int i = blockIdx.x*blockDim.x + threadIdx.x;
  int i = get_global_id(0);

  float4 mloc = m[i];
  m[i] = rsqrt(dot(mloc,mloc))*mloc;
}

__kernel void reduce_m(__global float4 *m, __global float *phase_diagram, int realizations) {
	int i = get_global_id(1)*get_global_size(0) + get_global_id(0);

	if (get_local_id(0) == 0) {
		float sum = 0.0;
		for (int r=0; r<realizations; r++) {
			sum += m[i+r].x ;
		}
			// printf("(%d of %d, %d of %d, %d) \n", get_group_id(0), get_num_groups(0), 
			// 											   get_group_id(1), get_num_groups(1), get_local_id(0));

	    phase_diagram[get_group_id(1)*get_local_size(0)+get_group_id(0)] = sum/float(realizations);

	}
}