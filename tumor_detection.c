#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "header/stb_image.h"

#include "cnn_weights/c_arrays/conv1_weight.h"
#include "cnn_weights/c_arrays/conv1_bias.h"
#include "cnn_weights/c_arrays/conv2_weight.h"
#include "cnn_weights/c_arrays/conv2_bias.h"
#include "cnn_weights/c_arrays/fc1_weight.h"
#include "cnn_weights/c_arrays/fc1_bias.h"
#include "cnn_weights/c_arrays/fc2_weight.h"
#include "cnn_weights/c_arrays/fc2_bias.h"

extern void fast_convolution(uint8_t *in, uint8_t *out, int w, int h, float* kernel);
extern void fast_maxpool(uint8_t *in, uint8_t *out, int w, int h);

void conv2d(const float*, const float*, const float*, float*, int, int, int, int, int);
void relu(float*, int);
void maxpool2x2(const float*, float*, int, int, int);
float fc(const float*, const float*, const float*, int);
float sigmoid(float);

static inline float relu1(float x) {
    return x > 0.0f ? x : 0.0f;
}

int main(int argc, char **argv) {

    if (argc != 2) { //بررسی وود عکس ورودی
        printf("Usage: %s <image_path>\n", argv[0]);
        return 1;
    }

    int w, h, c;
    uint8_t *img = stbi_load(argv[1], &w, &h, &c, 1);
    if (!img) {
        printf("Failed to load image\n");
        return 1;
    }

    if (w != 128 || h != 128) { //بررسی سایز عکس
        printf("Image must be 128x128 (got %dx%d)\n", w, h);
        stbi_image_free(img);
        return 1;
    }

    //شروع پردازش تصویر با شبکه عصبی

    static uint8_t conv1_out_u8[8][128 * 128];
    for (int oc = 0; oc < 8; oc++) {
        fast_convolution(
            img,
            conv1_out_u8[oc],
            128, 128,
            (float *)(conv1_weight + oc * 9)
        );
        for (int i = 0; i < 128 * 128; i++) {
            float v = conv1_out_u8[oc][i] / 255.0f + conv1_bias[oc];
            conv1_out_u8[oc][i] = (uint8_t)(relu1(v) * 255.0f);
        }
    }
    stbi_image_free(img);
    static uint8_t pool1[8][64 * 64];
    
    for (int c = 0; c < 8; c++)
        fast_maxpool(conv1_out_u8[c], pool1[c], 128, 128);
    static uint8_t conv2_out_u8[16][64 * 64];
    for (int oc = 0; oc < 16; oc++) {
        memset(conv2_out_u8[oc], 0, 64 * 64);

        for (int ic = 0; ic < 8; ic++) {
            fast_convolution(
                pool1[ic],
                conv2_out_u8[oc],
                64, 64,
                (float *)(conv2_weight + oc * 8 * 9 + ic * 9)
            );
        }

        for (int i = 0; i < 64 * 64; i++) {
            float v = conv2_out_u8[oc][i] / 255.0f + conv2_bias[oc];
            conv2_out_u8[oc][i] = (uint8_t)(relu1(v) * 255.0f);
        }
    }
    
    static uint8_t pool2[16][32 * 32];
    for (int c = 0; c < 16; c++)
        fast_maxpool(conv2_out_u8[c], pool2[c], 64, 64);

    static float flat[16 * 32 * 32];
    int idx = 0;
    for (int c = 0; c < 16; c++)
        for (int i = 0; i < 32 * 32; i++)
            flat[idx++] = pool2[c][i] / 255.0f;

    static float fc1_out[64];
    for (int i = 0; i < 64; i++)
        fc1_out[i] = relu1(fc(flat, fc1_weight + i * (16 * 32 * 32),
                             fc1_bias + i, 16 * 32 * 32));


    float logit = fc(fc1_out, fc2_weight, fc2_bias, 64);
    float prob = sigmoid(logit) * 100;
    printf("Probability: %.3f\n", prob);

    if (prob > 40)
        printf("Diagnosis: TUMOR DETECTED!!!\n");
    else
        printf("Diagnosis: NO TUMOR\n");

    return 0;
}
