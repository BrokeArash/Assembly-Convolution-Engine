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

typedef struct { //اطلاعات عکس
    double v_density;
    double h_density;
    int label;
    int id;
} ImageSignature;

float K_VERT[9] = { -1, 0, 1,
                    -2, 0, 2,
                    -1, 0, 1 
                };
float K_HORI[9] = { -1, -2, -1,
                     0,  0,  0, 
                     1,  2,  1 
                };

void to_grayscale(uint8_t* src, uint8_t* dest, int size) {
    for (int i = 0; i < size; i++) {
        dest[i] = (uint8_t)(src[i*3]*0.299 + src[i*3+1]*0.587 + src[i*3+2]*0.114);
    }
}

ImageSignature extract_features(uint8_t* rgb_img, int w, int h) {
    int size = w * h;
    uint8_t* gray = malloc(size);
    uint8_t* buffer = malloc(size);
    
    to_grayscale(rgb_img, gray, size); //تبدیل به حاکستری برای کانولوشن

    fast_convolution(gray, buffer, w, h, K_VERT);
    long long sum_v = 0;
    for(int i=0; i<size; i++) sum_v += buffer[i];

    fast_convolution(gray, buffer, w, h, K_HORI);
    long long sum_h = 0;
    for(int i=0; i<size; i++) sum_h += buffer[i];

    free(gray);
    free(buffer);

    ImageSignature sig; //خروجی به عنوان استراکت
    sig.v_density = (double)sum_v / size; 
    sig.h_density = (double)sum_h / size;
    return sig;
}


int main(int argc, char** argv) {
    if (argc < 2) { //چک کردن عکس ورودی
        printf("Usage: %s <image_path>\n", argv[0]);
        return 1;
    }

    int DATASET_SIZE = 100; //تعداد عکس های دیتاست
    ImageSignature database[DATASET_SIZE];
    printf("1. Training System (Loading Dataset)...\n");

    for (int i = 0; i < DATASET_SIZE; i++) { //لود کردن عکس ها در استراکت
        char filename[64];
        sprintf(filename, "dataset/test_%d.jpg", i);

        int w, h, c;
        uint8_t* img = stbi_load(filename, &w, &h, &c, 3);
        
        if (img) {

            database[i] = extract_features(img, w, h);
            database[i].id = i;
            
            database[i].label = (i % 2 == 0) ? 0 : 1; 

            stbi_image_free(img);
        } else {
            database[i].v_density = -1; 
            database[i].h_density = -1;
            printf("   - picture %d can't be trained", i);
        }
        
        if (i % 10 == 0) printf("   - Learned from %d images...\n", i);
    }
    printf("   - Training Complete.\n\n");

    printf("2. analyzing Input Image: %s\n", argv[1]);
    int w, h, c;
    uint8_t* input_img = stbi_load(argv[1], &w, &h, &c, 3); //لود عکس ورودی
    if (!input_img) {
        printf("Error: Could not load input image.\n");
        return 1;
    }

    ImageSignature input_sig = extract_features(input_img, w, h);
    printf("   - Input Signature: [V: %.2f, H: %.2f]\n", input_sig.v_density, input_sig.h_density);

    double min_distance = 999999999.0;
    int closest_match_index = -1;

    for (int i = 0; i < DATASET_SIZE; i++) { //تشخیص نزدیک ترین عکس
        if (database[i].v_density < 0) continue;

        double diff_v = input_sig.v_density - database[i].v_density;
        double diff_h = input_sig.h_density - database[i].h_density;
        double dist = sqrt( pow(diff_v, 2) + pow(diff_h, 2) );

        if (dist < min_distance) {
            min_distance = dist;
            closest_match_index = i;
        }
    }

    int detected_type = database[closest_match_index].label;
    char* type_str = (detected_type == 0) ? "VERTICAL" : "HORIZONTAL";

    printf("\n------------------------------------------------\n");
    printf("MATCH FOUND!\n");
    printf("Closest Dataset Image: test_%d.jpg\n", database[closest_match_index].id);
    printf("Similarity Distance:   %.4f (Lower is better)\n", min_distance);
    printf("DETECTED PATTERN:      %s\n", type_str);
    printf("------------------------------------------------\n");

    int size = w * h;
    uint8_t *gray = malloc(size);
    uint8_t *output_gray = malloc(size);
    uint8_t *final_rgb = malloc(size * 3);

    to_grayscale(input_img, gray, size); //تبدیل عکس خروجی

    if (detected_type == 0) {
        fast_convolution(gray, output_gray, w, h, K_VERT);
    } else {
        fast_convolution(gray, output_gray, w, h, K_HORI);
    }

    for (int i = 0; i < size; i++) {
        final_rgb[i*3+0] = output_gray[i];
        final_rgb[i*3+1] = output_gray[i];
        final_rgb[i*3+2] = output_gray[i];
    }

    char out_filename[100];
    sprintf(out_filename, "images/output_%s_matched.jpg", type_str);
    stbi_write_jpg(out_filename, w, h, 3, final_rgb, 90);
    printf("Output saved to: %s\n", out_filename);

    stbi_image_free(input_img);
    free(gray);
    free(output_gray);
    free(final_rgb);

    return 0;
}
