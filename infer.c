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
    unsigned char *img_u8 = stbi_load(argv[1], &w, &h, &c, 1);
    if (!img_u8) {
        printf("Failed to load image\n");
        return 1;
    }

    if (w != 128 || h != 128) { //بررسی سایز عکس
        printf("❌ Image must be 128x128 (got %dx%d)\n", w, h);
        stbi_image_free(img_u8);
        return 1;
    }

    float input_image[128 * 128];
    for (int i = 0; i < 128 * 128; i++)
        input_image[i] = img_u8[i] / 255.0f;

    stbi_image_free(img_u8);

    static float x1[8 * 128 * 128];
    static float p1[8 * 64 * 64];

    static float x2[16 * 64 * 64];
    static float p2[16 * 32 * 32];

    static float fc1_out[64];

    conv2d(input_image, conv1_weight, conv1_bias,
           x1, 1, 8, 128, 128, 3);
    relu(x1, 8 * 128 * 128);
    maxpool2x2(x1, p1, 8, 128, 128);

    conv2d(p1, conv2_weight, conv2_bias,
           x2, 8, 16, 64, 64, 3);
    relu(x2, 16 * 64 * 64);
    maxpool2x2(x2, p2, 16, 64, 64);

    /* ---------- FC1 ---------- */
    for (int i = 0; i < 64; i++) {
        fc1_out[i] = relu1(
            fc(p2,
               fc1_weight + i * (16 * 32 * 32),
               fc1_bias + i,
               16 * 32 * 32)
        );
    }

    /* ---------- FC2 ---------- */
    float logit = fc(fc1_out, fc2_weight, fc2_bias, 64);
    int percentage = (int)(sigmoid(logit) * 100);
    printf("Logit score: %.3f\n", logit);

    if (percentage > 25)
        printf("Diagnosis: TUMOR DETECTED!!!\n");
    else
        printf("Diagnosis: NO TUMOR\n");

    return 0;
}
