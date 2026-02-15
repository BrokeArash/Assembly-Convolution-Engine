#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "header/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "header/stb_image_write.h"

extern void fast_convolution(uint8_t *in, uint8_t *out, int w, int h, float* kernel);


long long calculate_energy(uint8_t* image, int size) {
    long long total = 0;
    for (int i = 0; i < size; i++) {
        total += image[i];
    }
    return total;
}

int recognize_pattern(uint8_t* r, uint8_t* g, uint8_t* b, int w, int h, 
                      float* k_vert, float* k_horiz, 
                      uint8_t* buffer, int use_asm) {
    
    int size = w * h;
    long long energy_vert = 0;
    long long energy_horiz = 0;

    fast_convolution(r, buffer, w, h, k_vert);
    energy_vert = calculate_energy(buffer, size);
    fast_convolution(r, buffer, w, h, k_horiz);
    energy_horiz = calculate_energy(buffer, size);

    // Decision Logic
    if (energy_vert > energy_horiz) {
        return 0; // Vertical
    } else {
        return 1; // Horizontal
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <image_path>\n", argv[0]);
        return 1;
    }

    char* input_filename = argv[1];
    int w, h, c;

    uint8_t* img = stbi_load(input_filename, &w, &h, &c, 3); // Force 3 channels (RGB)
    if (!img) {
        printf("Error loading image: %s\n", input_filename);
        return 1;
    }
    printf("Loaded Image: %s (%dx%d)\n", input_filename, w, h);

    int size = w * h;
    uint8_t *gray_in = malloc(size);
    
    uint8_t *out_vert = malloc(size);
    uint8_t *out_horiz = malloc(size);
    uint8_t *final_rgb = malloc(size * 3);
    for (int i = 0; i < size; i++) {
        gray_in[i] = (uint8_t)(img[i*3]*0.299 + img[i*3+1]*0.587 + img[i*3+2]*0.114);
    }

    float k_vertical[9] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };

    float k_horizontal[9] = {
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1
    };

    fast_convolution(gray_in, out_vert, w, h, k_vertical);
    long long energy_v = calculate_energy(out_vert, size);
    fast_convolution(gray_in, out_horiz, w, h, k_horizontal);
    long long energy_h = calculate_energy(out_horiz, size);

    printf("\nAnalysis Results:\n");
    printf("Vertical Energy:   %lld\n", energy_v);
    printf("Horizontal Energy: %lld\n", energy_h);

    uint8_t* winner_img;
    char* detection_result;

    if (energy_v > energy_h) {
        detection_result = "VERTICAL";
        winner_img = out_vert;
    } else {
        detection_result = "HORIZONTAL";
        winner_img = out_horiz;
    }

    printf("--------------------------\n");
    printf("DETECTED PATTERN: %s\n", detection_result);
    printf("--------------------------\n");

    char out_filename[100];
    sprintf(out_filename, "images/output_%s_detected.jpg", detection_result);
    for(int i=0; i<size; i++) {
        if (strcmp(detection_result, "VERTICAL") == 0) {
            final_rgb[i*3+0] = winner_img[i]; // R
            final_rgb[i*3+1] = winner_img[i]; // G
            final_rgb[i*3+2] = winner_img[i]; // B
        } else {
            final_rgb[i*3+0] = winner_img[i];
            final_rgb[i*3+1] = winner_img[i];
            final_rgb[i*3+2] = winner_img[i];
        }
    }

    stbi_write_jpg(out_filename, w, h, 3, final_rgb, 90);
    printf("Saved visualization to: %s\n", out_filename);

    // Cleanup
    stbi_image_free(img);
    free(gray_in);
    free(out_vert);
    free(out_horiz);
    free(final_rgb);

    return 0;

    // int num_images = 47;
    // int correct_asm = 0;

    // double total_time_asm = 0;
    // printf("Starting Pattern Recognition on %d images...\n", num_images);
    // printf("--------------------------------------------------\n");

    // for (int i = 0; i < num_images; i++) {
    //     char filename[64];
    //     sprintf(filename, "dataset/test_%d.jpg", i);
    //     int w, h, c;
    //     uint8_t* img = stbi_load(filename, &w, &h, &c, 3);
    //     if (!img) continue;

    //     int size = w * h;
    //     uint8_t *r = malloc(size);
    //     uint8_t *g = malloc(size);
    //     uint8_t *b = malloc(size);
    //     uint8_t *scratch_buffer = malloc(size);
    
    //     for (int p = 0; p < size; p++) {
    //         r[p] = img[p * 3 + 0];
    //         g[p] = img[p * 3 + 1];
    //         b[p] = img[p * 3 + 2];
    //     }
    //     clock_t start = clock();
    //     int pred_asm = recognize_pattern(r, g, b, w, h, k_vertical, k_horizontal, scratch_buffer, 1);
    //     total_time_asm += (double)(clock() - start) / CLOCKS_PER_SEC;

    //     int expected = (i % 2);
    //     if (pred_asm == expected) correct_asm++;
    //     stbi_image_free(img);
    //     free(r); free(g); free(b); free(scratch_buffer);

    //     if (i % 10 == 0) printf("Processed %d images...\n", i);
    // }

    // printf("--------------------------------------------------\n");
    // printf("RESULTS:\n");
    // printf("Total Images Processed: %d\n", num_images);
    // printf("ASM Accuracy: %.2f%%\n", (float)correct_asm / num_images * 100.0f);
    // printf("\n");
    // printf("Total Time ASM: %.4f sec\n", total_time_asm);

    // FILE* fp = fopen("plot/recognition_results.csv", "w");
    // if (fp) {
    //     fprintf(fp, "type,accuracy,time\n");
    //     fprintf(fp, "ASM,%.2f,%.4f\n", (float)correct_asm / num_images * 100.0f, total_time_asm);
    //     fclose(fp);
    //     printf("Data saved to plot/recognition_results.csv\n");
    // }

    // return 0;
}
