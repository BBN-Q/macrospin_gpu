// Declare all of the global constants that we don't want to compute later
{% for fc in float_constants -%}
__constant float {{fc.name}} = {{fc.value}}f;
{% endfor %}
{% for fc in float3_constants -%}
__constant float3 {{fc.name}} = (float3)({{fc.x}}f, {{fc.y}}f, {{fc.z}}f);
{% endfor %}
{% for fc in float4_constants -%}
__constant float4 {{fc.name}} = (float4)({{fc.x}}f, {{fc.y}}f, {{fc.z}}f, {{fc.w}}f);
{% endfor %}

{%- if stt %}
float get_envelope(float real_time, float duration);

float get_envelope(float real_time, float duration) {
    {% if square_pulse %}
    if (real_time < initial_pause) {
        return 0.0f;
    } else if (real_time < (initial_pause + duration) ) {
        return 1.0f;
    } else {
        return 0.0f;
    }
    {% else %}
    if (real_time < initial_pause) {
        return exp(-pown(real_time-initial_pause,2)/(2.0f*pown({{rise_time}},2)));
    } else if (real_time < (initial_pause + duration) ) {
        return 1.0f;
    } else {
        return exp(-pown(real_time-initial_pause-duration,2)/(2.0f*pown({{fall_time}},2)));
    }
    {% endif %}
}
{% endif %}

// void dipole_field(float4 m1, float4 m2, float4 *h1, float4 *h2 );
// void dipole_field(float4 m1, float4 m2, float4 *h1, float4 *h2 ) {
//     // Calculate fields from m1 on m2, m2 on m1 and store these
//     // values to the variables h1, h2. We can use a crude version here
//     // and use the mutual demag tensor at some later point.

//     // dipole_prefac_21 = - (Ms_2*vol2/Ms_1)/4*pi*|r|^3

//     h1 = dipole_prefac_21*(m2 - 3.0f*dipole_r12_hat*dot(m2, dipole_r12_hat) );
//     h2 = dipole_prefac_12*(m1 - 3.0f*dipole_r12_hat*dot(m2, dipole_r12_hat) );
// }

__kernel void evolve(__global float4 *m,
                     __global float4 *m_ref,
                     {%- if thermal -%}__global float4 *dW,{% endif %}
                     __global float *first_loop_values,
                     __global float *second_loop_values,
                     float real_time) {

    // Get the global index of the current thread
    int i = get_global_id(1)*get_global_size(0) + get_global_id(0);

    float4 m_loc     = m[i];
    float4 m_ref_loc = m_ref[i];

    {%- if thermal -%}float4 dW_loc = dW[i];{% endif %}
    
    __local float {{first_loop_var}};
    __local float {{second_loop_var}};
    __local float envelope; 

    // Only need to update these parameters once per local work group
    if (get_local_id(0) == 0) {
        // Read the array values into local memory
        {{first_loop_var}}  = first_loop_values[get_group_id(0)];
        {{second_loop_var}} = second_loop_values[get_group_id(1)];
        envelope = get_envelope(real_time, pulse_duration);
    }
    // Ensure thread execution doesn't continue until local variables are set
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Effective field 
    float4 heff     = hext;
    float4 heff_ref = hext;

    // Subtract the demag field, add the dipolar field
    float4 dip     = -0.3f*m_ref_loc;
    float4 dip_ref = -0.3f*m_loc;

    // dipole_field(m_loc, m_ref_loc, dip, dip_ref);
    heff     = heff - m_loc*demag_tensor + dip;
    heff_ref = heff_ref - m_ref_loc*demag_tensor_ref + dip_ref;

    // Any other effective fields added here
    {% for eff in effective_fields %}
        heff = heff + {{eff}};
    {% endfor -%}

    // Any non-conservative torques added here
    {% if stt -%}
    float4 stt = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float4 p;
    float m_dot_p;
        {% for t in stt_torques %}
    p = (float4)({{t.pol_x}}f, {{t.pol_y}}f, {{t.pol_z}}f, 0.0f);
    m_dot_p = dot(p, m_loc);
    stt = stt + p*{{t.prefac}}f*current_density*envelope/fma({{t.l2_m1}}f, m_dot_p, {{t.l2_p1}}f);
        {% endfor -%}
    {% endif %}

    // Define torque variables
    float4 hxm, pxm, mxhxm;
    {% if thermal %}float4 nudWxm, numxdWxm;{% endif -%}

    // Deterministic terms (not including deterministic drift)
    hxm      =  cross(heff, m_loc);
    {% if stt -%}
    pxm      =  cross(stt,  m_loc);
    mxhxm    =  cross(m_loc, hxm + pxm);
    {% else -%}
    mxhxm    =  cross(m_loc, hxm);
    {% endif %}

    // Only deterministic terms for reference layer
    float4 hxm_ref, mxhxm_ref;
    hxm_ref   =  cross(heff_ref, m_ref_loc);
    mxhxm_ref =  cross(m_ref_loc, hxm_ref);

    {%- if thermal -%}
    // Stochastic terms
    nudWxm   =  nuSqrtDt*cross(dW_loc,m_loc);
    numxdWxm =  cross(m_loc, nudWxm);
    {% endif -%}

    // Assemble the torques and perform integration step for the free layer
    {% if thermal -%}
    float4 deterministic_part = hxm + fma(alpha, mxhxm, -nu2*m_loc);
    float4 stochastic_part    = fma(alpha, numxdWxm, nudWxm); 
    m[i] = m_loc + dt*deterministic_part + stochastic_part;
    {% else %}
    m[i] = m_loc + dt*fma(alpha, mxhxm, hxm);
    {% endif %}

    // Now perform integration step for the fixed layer
    m_ref[i] = m_ref_loc + dt*fma(alpha, mxhxm_ref, hxm_ref);
}

__kernel void normalize_m(__global float4 *m, __global float4 *m_ref) {
  //const int i = blockIdx.x*blockDim.x + threadIdx.x;
  int i = get_global_id(1)*get_global_size(0) + get_global_id(0);
  m[i]     = normalize(m[i]);
  m_ref[i] = normalize(m_ref[i]);
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
    phase_diagram[i] = sum/(float)realizations;

}