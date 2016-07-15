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

float4 eval_torques(float4 m,
                    {%- if thermal -%}float4 dW,{% endif %}
                    float {{first_loop_var}},
                    float {{second_loop_var}},
                    float envelope) {

    // Effective field
    float4 heff = hext;

    // Subtract the demag field
    heff = heff - m*demag_tensor;

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
    m_dot_p = dot(p, m);
    stt = stt + p*{{t.prefac}}f*current_density*envelope/fma({{t.l2_m1}}f, m_dot_p, {{t.l2_p1}}f);
        {% endfor -%}
    {% endif %}

    // Add Oersted Field
    {% if oersted -%}
    float4 h_oe = current_density*{{h_oe_prefac}}f*(float4)({{h_oe_x}}f, {{h_oe_y}}f, {{h_oe_z}}f, 0.0f);
    heff = heff+ h_oe;
    {% endif %}

    // Define torque variables
    float4 mxh, mxp, mxmxh;
    {% if thermal %}float4 nudWxm, numxdWxm;{% endif -%}

    // Deterministic terms (not including deterministic drift)
    mxh      =  cross(m, heff);
    {% if stt -%}
    mxp      =  cross(m, stt);
    mxmxh    =  cross(m, mxh - mxp );
    {% else -%}
    mxmxh    =  cross(m, mxh);
    {% endif %}

    {%- if thermal -%}
    // Stochastic terms
    nudWxm   =  nuSqrtDt*cross(dW,m);
    numxdWxm =  cross(m, nudWxm);
    {% endif -%}

    // Assemble the torques and perform integration step
    {% if thermal -%}
    // In the Stratanovich sense, i.e. no deterministic drift term
    float4 deterministic_part = -fma(alpha, mxmxh, mxh);
    float4 stochastic_part    =  fma(alpha, numxdWxm, nudWxm);
    return dt*deterministic_part + stochastic_part;
    {% else %}
    return -dt*fma(alpha, mxmxh, mxh);
    {% endif %}

}

__kernel void evolve(__global float4 *m,
                     {%- if thermal -%}__global float4 *dW,{% endif %}
                     __global float *first_loop_values,
                     __global float *second_loop_values,
                     float real_time) {

    // Get the global index of the current thread
    int i = get_global_id(1)*get_global_size(0) + get_global_id(0);

    float4 m_loc  = m[i];
    {%- if thermal -%}float4 dW_loc = dW[i];{% endif %}

    __local float {{first_loop_var}};
    __local float {{second_loop_var}};
    __local float envelope, envelope_end;

    // Only need to update these parameters once per local work group
    if (get_local_id(0) == 0) {
        // Read the array values into local memory
        {{first_loop_var}}  = first_loop_values[get_group_id(0)];
        {{second_loop_var}} = second_loop_values[get_group_id(1)];
        {% if stt %}
        envelope     = get_envelope(real_time, pulse_duration);
        envelope_end = get_envelope(real_time + real_dt, pulse_duration);
        {% endif %}
    }
    // Ensure thread execution doesn't continue until local variables are set
    barrier(CLK_GLOBAL_MEM_FENCE);

    // The Heun integration scheme is given as follows
    // m_(t+dt) = m_n + 1/2 [ determ(m_bar, t+dt) + determ(m, t) ]*dt
    //                + 1/2 [ stoch(m_bar, t+dt)  + stoch(m, t)  ]*dW
    // m_bar    = m_n + determ(m_n, t)*dt + stoch(m_n, t)*dW

    // m_bar = m + eval_torques(m)
    float4 torque_pred = eval_torques(m_loc,
                        {%- if thermal -%}dW_loc,{% endif %}
                        {{first_loop_var}},
                        {{second_loop_var}},
                        envelope);
    float4 torque_corr = eval_torques(m_loc + torque_pred,
                        {%- if thermal -%}dW_loc,{% endif %}
                        {{first_loop_var}},
                        {{second_loop_var}},
                        envelope_end);
    m[i] = m_loc + 0.5f*(torque_pred + torque_corr);
    // m[i] = m_loc + torque_pred;

}

__kernel void normalize_m(__global float4 *m) {
  //const int i = blockIdx.x*blockDim.x + threadIdx.x;
  int i = get_global_id(1)*get_global_size(0) + get_global_id(0);
  m[i] = normalize(m[i]);
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

__kernel void update_m_of_t(__global float4 *m, __global float4 *m_of_t, int time_points, int realizations, int current_index) {
  int i = get_group_id(0) + get_group_id(1)*get_global_size(0);

    float4 mloc = m[i*realizations];
    mloc.w = 1.0f;
    m_of_t[i*time_points + current_index] = mloc;

}
