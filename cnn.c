#include <math.h>
#include <string.h>
#include <stdio.h>

#define RELU(x) ((x) > 0 ? (x) : 0)

void conv2d(
    const float *input,
    const float *weights,
    const float *bias,
    float *output,
    int in_c, int out_c,
    int h, int w,
    int k
) {
    int pad = k / 2;

    for (int oc = 0; oc < out_c; oc++) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {

                float acc = bias[oc];

                for (int ic = 0; ic < in_c; ic++) {
                    for (int ky = 0; ky < k; ky++) {
                        for (int kx = 0; kx < k; kx++) {
                            int iy = y + ky - pad;
                            int ix = x + kx - pad;
                            if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
                                int in_idx = ic*h*w + iy*w + ix;
                                int w_idx =
                                    oc*(in_c*k*k) +
                                    ic*(k*k) +
                                    ky*k + kx;
                                acc += input[in_idx] * weights[w_idx];
                            }
                        }
                    }
                }

                output[oc*h*w + y*w + x] = acc;
            }
        }
    }
}


void relu(float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = RELU(x[i]);
}

float relu1(float x) {
    return x > 0.0f ? x : 0.0f;
}

void maxpool2x2(const float *in, float *out, int c, int h, int w) {
    int oh = h / 2, ow = w / 2;
    for (int ch = 0; ch < c; ch++) {
        for (int y = 0; y < oh; y++) {
            for (int x = 0; x < ow; x++) {
                float m = -1e9;
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++) {
                        int iy = y*2 + dy;
                        int ix = x*2 + dx;
                        float v = in[ch*h*w + iy*w + ix];
                        if (v > m) m = v;
                    }
                out[ch*oh*ow + y*ow + x] = m;
            }
        }
    }
}

float fc(
    const float *x,
    const float *w,
    const float *b,
    int n
) {
    float sum = *b;
    for (int i = 0; i < n; i++)
        sum += x[i] * w[i];
    return sum;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}
