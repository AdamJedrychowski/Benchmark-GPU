#define TYCHE_I_FLOAT_MULTI 5.4210108624275221700372640e-20f

typedef union{
	struct{
		uint a,b,c,d;
	};
	ulong res;
} tyche_i_state;

#define TYCHE_I_ROT(a,b) (((a) >> (b)) | ((a) << (32 - (b))))

#define tyche_i_ulong(state) (tyche_i_advance(&state), state.res)

void tyche_i_advance(tyche_i_state* state){
	state->b = TYCHE_I_ROT(state->b, 7) ^ state->c;
	state->c -= state->d;
	state->d = TYCHE_I_ROT(state->d, 8) ^ state->a;
	state->a -= state->b;
	state->b = TYCHE_I_ROT(state->b, 12) ^ state->c;
	state->c -= state->d;
	state->d = TYCHE_I_ROT(state->d, 16) ^ state->a;
	state->a -= state->b;
}

#define tyche_i_float(state) (tyche_i_ulong(state)*TYCHE_I_FLOAT_MULTI)


int ksi(int i, int j, __global int *x, __global int *a, const int N)
{
    int s = 0;
    for (int k = 0; k < N; ++k)
    {
        s += a[i * N + k] * x[i * N + k] * a[k * N + j] * x[k * N + j];
    }
    return s;
}

__kernel void startSyncSim(__global int *x, __global int *x_new, __global int *a, int N, __global float *probability, __global tyche_i_state *seed)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (j > i && a[i * N + j] != 0) {
        int new_val;

        float prob = probability[ksi(i , j, x, a, N) + N];
        float random = tyche_i_float(seed[i * N + j]);
        if (random < prob) {
            new_val = 1;
        } else {
            new_val = -1;
        }
        x_new[i * N + j] = x_new[j * N + i] = new_val;
    }
}

__kernel void calculateProperties(__global int *x, __global int *a, int N, __local int *energy, __local int *xmean, __global int *out_energy, __global int *out_xmean)
{
    uint l = get_global_id(0), local_id = get_local_id(0), group_size = get_local_size(0);
    uint i = l / N, j = l % N;
    xmean[local_id] = 0;
    energy[local_id] = 0;
    if(i < j)
    {
        xmean[local_id] = x[l];
        for (int k = j + 1; k < N; ++k)
        {
            energy[local_id] += a[l] * x[l] * a[j * N + k] * x[j * N + k] * a[k * N + i] * x[k * N + i];
        }
    }

    uint prev_stride = group_size;
    for(uint stride = group_size / 2; stride > 0; stride /= 2)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if(local_id < stride) {
            if(local_id == stride - 1 && stride*2 != prev_stride) {
                xmean[local_id] += xmean[local_id + stride + 1];
                energy[local_id] += energy[local_id + stride + 1];
            }
            xmean[local_id] += xmean[local_id + stride];
            energy[local_id] += energy[local_id + stride];
            prev_stride = stride;
        }
    }

    if (local_id == 0) {
        out_xmean[get_group_id(0)] = xmean[0];
        out_energy[get_group_id(0)] = energy[0];
    }
}