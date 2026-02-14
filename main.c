#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "header/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "header/stb_image_write.h"

extern void fast_convolution(uint8_t *in, uint8_t *out, int w, int h, float* kernel);


void convoloution_c(uint8_t* input, uint8_t* output, int w, int h, float* kernel) {
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            float sum = 0.0f;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int pixel = input[(y + ky) * w + (x + kx)];
                    sum += pixel * kernel[(ky + 1) * 3 + (kx + 1)];
                }
            }
            if (sum < 0) sum = 0;
            if (sum > 255) sum = 255;
            output[y * w + x] = (uint8_t)sum;
        }
    }
}

int main() {
    int width, height, channels;
    uint8_t* input_image = stbi_load("images/input_image.png", &width, &height, &channels, 3); 
    if (!input_image) {
        printf("Error: image loading failed!");
        return 1;
    }
    int size = width * height;
    uint8_t *r_in = malloc(size), *g_in = malloc(size), *b_in = malloc(size);
    uint8_t *r_out = malloc(size), *g_out = malloc(size), *b_out = malloc(size);
    uint8_t *final_output_asm = malloc(size * 3);
    uint8_t *final_output_c = malloc(size * 3);
    for (int i = 0; i < size; i++) {
        r_in[i] = input_image[i * 3 + 0];
        g_in[i] = input_image[i * 3 + 1];
        b_in[i] = input_image[i * 3 + 2];
    }

    //float kernel[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0}; // Sharpen Filter
    float kernel[9] = {
        -1.0f, -1.0f, -1.0f,
        -1.0f,  8.0f, -1.0f,
        -1.0f, -1.0f, -1.0f
    }; //edge detection
    
    FILE* fp = fopen("plot/benchmark_results.csv", "w");
    fprintf(fp, "iterations,c_time,asm_time,speedup\n");
    printf("Running benchmarks...\n");
    printf("Image size: %dx%d pixels\n\n", width, height);

    int test_iterations[] = {1, 5, 10, 20, 50, 100};
    int num_tests = sizeof(test_iterations) / sizeof(test_iterations[0]);

    for (int i = 0; i < num_tests; i++) {
        int iterations = test_iterations[i];
        clock_t start = clock();
        for (int j = 0; j < iterations; j++) {
            convoloution_c(r_in, r_out, width, height, kernel);
            convoloution_c(g_in, g_out, width, height, kernel);
            convoloution_c(b_in, b_out, width, height, kernel);
        }
        double time_c = (double)(clock() - start) / CLOCKS_PER_SEC;
    
        start = clock();
        for (int i = 0; i < iterations; i++) {
            fast_convolution(r_in, r_out, width, height, kernel);
            fast_convolution(g_in, g_out, width, height, kernel);
            fast_convolution(b_in, b_out, width, height, kernel);
        }
        double time_asm = (double)(clock() - start) / CLOCKS_PER_SEC;
        double speedup = time_c / time_asm;

        printf("Iterations: %3d | C: %.4f sec | ASM: %.4f sec | Speedup: %.2fx\n", 
                iterations, time_c, time_asm, speedup);
            
        fprintf(fp, "%d,%.6f,%.6f,%.4f\n", iterations, time_c, time_asm, speedup);
    }

    fclose(fp);
    printf("\nGenerating output images...\n");
    convoloution_c(r_in, r_out, width, height, kernel);
    convoloution_c(g_in, g_out, width, height, kernel);
    convoloution_c(b_in, b_out, width, height, kernel);


    for (int i = 0; i < size; i++) {
        final_output_c[i * 3 + 0] = r_out[i];
        final_output_c[i * 3 + 1] = g_out[i];
        final_output_c[i * 3 + 2] = b_out[i];
    }
    stbi_write_jpg("images/output_c_color.jpg", width, height, 3, final_output_c, 90);


    fast_convolution(r_in, r_out, width, height, kernel);
    fast_convolution(g_in, g_out, width, height, kernel);
    fast_convolution(b_in, b_out, width, height, kernel);
    
    for (int i = 0; i < size; i++) {
        final_output_asm[i * 3 + 0] = r_out[i];
        final_output_asm[i * 3 + 1] = g_out[i];
        final_output_asm[i * 3 + 2] = b_out[i];
    }

    stbi_write_jpg("images/output_asm_color.jpg", width, height, 3, final_output_asm, 90);
    printf("Benchmark data saved to benchmark_results.csv\n");
    printf("Output images saved.\n");

    stbi_image_free(input_image);
    free(r_in);
    free(g_in);
    free(b_in);
    free(r_out);
    free(g_out);
    free(b_out);
    free(final_output_asm);
    free(final_output_c);
    return 0;
}
