// OpenCL Kernel for stochastic macrospin simulations

__constant float4 N                = (float4)({{nxx}}f, {{nyy}}f, {{nzz}}f, 0.0f);
__constant float alpha             = {{alpha}}f;
__constant float dt                = {{dt}}f;
__constant float nu2               = {{nu2}}f;
__constant float nuSqrtDt          = {{nuSqrtDt}}f;
__constant float stt_op            = {{stt_op}}f;
__constant float stt_ip            = {{stt_ip}}f;
__constant float lambda2_plus1_op  = {{lambda2_plus1_op}}f;
__constant float lambda2_plus1_ip  = {{lambda2_plus1_ip}}f;
__constant float lambda2_minus1_op = {{lambda2_minus1_op}}f;
__constant float lambda2_minus1_ip = {{lambda2_minus1_ip}}f;
__constant float hx                = {{hx}}f;
__constant float hy                = {{hy}}f;
__constant float hz                = {{hz}}f;
__constant float rise_time         = {{rise_time}}f;
__constant float fall_time         = {{fall_time}}f;
__constant float pause_before      = {{pause_before}}f;
__constant float pause_after       = {{pause_after}}f;

__kernel void evolve(__global float4 *m, __global float4 *dW, __global float *currents, __global float *durations, float real_time) {

	// Groups organized such that duration changes in 0th dimension, tilt in 1st dimension
	int i = get_global_id(1)*get_global_size(0) + get_global_id(0);

	float4 m_loc  = m[i];
	float4 dW_loc = dW[i];

	__local float current;
	__local float duration;
	__local float envelope; 

	// Only need to update these parameters once per local work group
	if (get_local_id(0) == 0) {

		// Read the array values into local memory
		duration = durations[get_group_id(1)];
		current  = currents[get_group_id(0)];
		envelope = 0.0f;

		// Calculate the pulse shape
		// if (real_time <= pause_before) {
		// 	envelope = exp(-pown(real_time-pause_before,2)/(2.0f*pown(rise_time,2)));
		// } else 
		if (real_time < pause_before) {
			envelope = 0.0f;
		} else if (real_time < (pause_before + duration) ) {
			envelope = 1.0f;
		}
		// } else if (real_time > (pause_before + duration) ) {
		// 	envelope = exp(-pown(real_time-pause_before-duration,2)/(2.0f*pown(fall_time,2)));
		// } 
	}
	// Ensure thread execution doesn't continue until local variables are set
	barrier(CLK_GLOBAL_MEM_FENCE);

	// Spin torque pulse currents
	float stt_op_mag = current*envelope*stt_op/fma(lambda2_minus1_op, m_loc[2], lambda2_plus1_op);
	float stt_ip_mag = current*envelope*stt_ip/fma(lambda2_minus1_ip, m_loc[0], lambda2_plus1_ip);

	// Effective field 
	float4 heff = (float4)(hx, hy, hz, 0.0f) ;

	// Subtract the demag field
	heff = heff - m_loc*N;

	// // Spin transfer torque
	float4 stt = (float4)(-stt_ip_mag, 0.0f, stt_op_mag, 0.0f);

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
	// In this case get_global_size(0) = total number of duration steps
	//              get_group_id(0) = duration duration step
	//              get_global_size(1) = total number of tilt steps
	//              get_group_id(1) = duration tilt step

	// Realizations are manually looped over
	// We grab elements of m: m[r + get_group_id(0)*warp_x + get_group_id(1)*warp_y]
	int i = get_group_id(0) + get_group_id(1)*get_global_size(0);

	float sum = 0.0f;
	for (int r=0; r<realizations; r++) {
		sum += m[r + i*realizations].x > 0.0f ? 0.0f : 1.0f;
	}
	phase_diagram[i] = sum/float(realizations);

}
