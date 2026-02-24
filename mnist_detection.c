#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "header/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "header/stb_image_write.h"
#include "templates/kernels.h"

#define DIGIT_SIZE    28
#define FEAT_DIM      (DIGIT_SIZE * DIGIT_SIZE)
#define ZONE_GRID     7
#define ZONE_CELL     (DIGIT_SIZE / ZONE_GRID)
#define ZONE_DIM      (ZONE_GRID * ZONE_GRID)
#define PROJ_DIM      (DIGIT_SIZE * 2)
#define HU_DIM        7
#define MAX_TEMPLATES 2000
#define MAX_DIGITS    64
#define TOP_K         7
#define LABEL_H       10
#define BORDER        2
#define OUTPUT_PATH   "images/mnist_output.png"

extern void fast_convolution(const uint8_t *input, uint8_t *output, int width, int height, const float *kernel);

typedef struct { //هرکدام ویژگی متفاوتی از هر رقم را ثبت می‌کند
    float conv[NUM_KERNELS][FEAT_DIM];
    float zones[ZONE_DIM];
    float proj[PROJ_DIM];
    float hu[HU_DIM];
    int   digit;
} Template;


static const uint8_t FONT_5X7[10][7] = { //بیت‌مپ ارقام روی تصویر خروجی
    {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}, //0
    {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}, //1
    {0x0E,0x11,0x01,0x06,0x08,0x10,0x1F}, //2
    {0x0E,0x11,0x01,0x06,0x01,0x11,0x0E}, //3
    {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}, //4
    {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E}, //5
    {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E}, //6
    {0x1F,0x01,0x02,0x04,0x08,0x08,0x08}, //7
    {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}, //8
    {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C}, //9
};

 //نوشتن عدد تشخیص داده شده
static void draw_digit_glyph(uint8_t *canvas, int cw, int ch, int px, int py, int digit, uint8_t r, uint8_t g, uint8_t b) {
    for (int row = 0; row < 7; row++) {
        for (int col = 0; col < 5; col++) {
            if (FONT_5X7[digit][row] & (0x10 >> col)) {
                int x = px + col;
                int y = py + row;
                if (x >= 0 && x < cw && y >= 0 && y < ch) {
                    canvas[(y*cw + x)*3 + 0] = r;
                    canvas[(y*cw + x)*3 + 1] = g;
                    canvas[(y*cw + x)*3 + 2] = b;
                }
            }
        }
    }
}

//رسم کادر دور عدد تشخیص داده شده
static void draw_rect(uint8_t *canvas, int cw, int ch, int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b) { 
    for (int i = 0; i < BORDER; i++) {
        for (int x = x0+i; x <= x1-i; x++) {
            int yt = y0 + i;
            int yb = y1 - i;
            if (yt >=0 && yt < ch && x >= 0 && x < cw) {
                canvas[(yt * cw + x) * 3 + 0] = r;
                canvas[(yt * cw + x) * 3 + 1] = g;
                canvas[(yt * cw + x) * 3 + 2] =b;
            }
            if (yb >= 0 && yb < ch && x >=0 && x < cw) {
                canvas[(yb * cw + x) * 3 + 0] = r;
                canvas[(yb * cw + x) * 3 + 1] =g;
                canvas[(yb * cw + x) * 3 + 2] = b;
            }
        }

        for (int y = y0 + i; y <= y1-i; y++) {
            int xl = x0+i, xr = x1-i;
            if (y>= 0 && y < ch && xl >= 0 && xl < cw) {
                canvas[(y * cw + xl) * 3 + 0] = r;
                canvas[(y * cw + xl) * 3 + 1] = g;
                canvas[(y * cw + xl) * 3 + 2] = b;
            }
            if (y >=0 && y < ch && xr >= 0 && xr < cw) {
                canvas[(y * cw + xr) * 3 + 0] = r;
                canvas[(y * cw + xr) * 3 + 1] = g;
                canvas[(y * cw + xr) * 3 + 2] = b;
            }
        }
    }
}

static void to_gray_float(const uint8_t *src, float *dst, int w, int h, int ch) {
    for (int i = 0; i < w*h; i++) {
        float g = (ch==1) ? src[i]
            : 0.299f*src[i*ch] + 0.587f*src[i*ch+1] + 0.114f*src[i*ch+2];
        dst[i] = (255.0f - g) / 255.0f;
    }
}

static void normalise(float *x, int n) {
    float mean=0;
    for (int i=0; i<n; i++) mean+=x[i];
    mean/=n;
    float var=0;
    for (int i=0; i<n; i++) { x[i]-=mean; var+=x[i]*x[i]; }
    float sd=sqrtf(var/n);
    if (sd<1e-6f) sd=1.0f;
    for (int i=0; i<n; i++) x[i]/=sd;
}

static void center_digit(float *img) { //مرکز جرم عکس رو به وسط منتقل می‌کند
    float cx=0,cy=0,mass=0;
    for (int y=0; y<DIGIT_SIZE; y++)
        for (int x=0; x<DIGIT_SIZE; x++) {
            float v=img[y*DIGIT_SIZE+x];
            if (v>0.05f) { cx+=x*v; cy+=y*v; mass+=v; }
        }
    if (mass<1e-6f) 
        return;
    cx/=mass;
    cy/=mass;
    int dx=(int)roundf(DIGIT_SIZE/2.0f-cx);
    int dy=(int)roundf(DIGIT_SIZE/2.0f-cy);
    float tmp[FEAT_DIM]={0};
    for (int y=0; y<DIGIT_SIZE; y++)
        for (int x=0; x<DIGIT_SIZE; x++) {
            int nx=x+dx, ny=y+dy;
            if (nx>=0&&nx<DIGIT_SIZE&&ny>=0&&ny<DIGIT_SIZE)
                tmp[ny*DIGIT_SIZE+nx]=img[y*DIGIT_SIZE+x];
        }
    memcpy(img,tmp,sizeof(tmp));
}

//تبدیل به دو بخش مثبت و منفی چون تابع اسمبلی خروجی رو منفی نمیکنه بین 0 تا 255 نگه میداره
//ورودی رو به تابع پاس میدیم ترکیب میکنیم و نرمالایز میکنیم
static void conv2d_asm(const float *src, float *dst, const float *kernel) {

    uint8_t src_u8[FEAT_DIM];
    for (int i = 0; i < FEAT_DIM; i++) {
        float v = src[i] * 255.0f;
        if (v < 0.0f)   v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        src_u8[i] = (uint8_t)v;
    }

    uint8_t out_pos[FEAT_DIM] = {0};
    fast_convolution(src_u8, out_pos, DIGIT_SIZE, DIGIT_SIZE, kernel);

    float neg_kernel[9];
    for (int i = 0; i < 9; i++) neg_kernel[i] = -kernel[i];

    uint8_t out_neg[FEAT_DIM] = {0};
    fast_convolution(src_u8, out_neg, DIGIT_SIZE, DIGIT_SIZE, neg_kernel);

    for (int i = 0; i < FEAT_DIM; i++)
        dst[i] = (out_pos[i] - out_neg[i]) / 255.0f;
}

static void extract_conv(const float *tile, float out[NUM_KERNELS][FEAT_DIM]) {
    for (int k=0; k<NUM_KERNELS; k++) {
        conv2d_asm(tile, out[k], ALL_KERNELS[k]);
        normalise(out[k], FEAT_DIM);
    }
}

static void compute_zones(const float *img, float *zones) { //عکس 28در28 رو به 49 بخش 4در4 تقسیم میکند
    memset(zones,0,sizeof(float)*ZONE_DIM);
    for (int zy=0; zy<ZONE_GRID; zy++)
        for (int zx=0; zx<ZONE_GRID; zx++) {
            float sum=0;
            for (int py=0; py<ZONE_CELL; py++)
                for (int px=0; px<ZONE_CELL; px++)
                    sum+=img[(zy*ZONE_CELL+py)*DIGIT_SIZE+(zx*ZONE_CELL+px)];
            zones[zy*ZONE_GRID+zx]=sum/(ZONE_CELL*ZONE_CELL);
        }
    normalise(zones,ZONE_DIM);
}

static void compute_projections(const float *img, float *proj) {
    memset(proj,0,sizeof(float)*PROJ_DIM);
    for (int y=0; y<DIGIT_SIZE; y++)
        for (int x=0; x<DIGIT_SIZE; x++) {
            float v=img[y*DIGIT_SIZE+x];
            proj[y]           +=v;
            proj[DIGIT_SIZE+x]+=v;
        }
    normalise(proj,PROJ_DIM);
}

static void compute_hu(const float *img, float *hu) { //ممان‌های هو از هر عکس 7 عدد استخراج می‌کند که با تعییر عکس تغییر نمیکند
    double m00=0,m10=0,m01=0;
    for (int y=0; y<DIGIT_SIZE; y++)
        for (int x=0; x<DIGIT_SIZE; x++) {
            double v=img[y*DIGIT_SIZE+x];
            m00+=v; m10+=x*v; m01+=y*v;
        }
    if (m00<1e-10) { memset(hu,0,HU_DIM*sizeof(float)); return; }
    double cx=m10/m00, cy=m01/m00;
    double mu[4][4]={};
    for (int y=0; y<DIGIT_SIZE; y++)
        for (int x=0; x<DIGIT_SIZE; x++) {
            double v=img[y*DIGIT_SIZE+x];
            double dx=x-cx, dy=y-cy;
            mu[2][0]+=dx*dx*v; mu[0][2]+=dy*dy*v; mu[1][1]+=dx*dy*v;
            mu[3][0]+=dx*dx*dx*v; mu[0][3]+=dy*dy*dy*v;
            mu[2][1]+=dx*dx*dy*v; mu[1][2]+=dx*dy*dy*v;
        }
    double n20=mu[2][0]/m00,n02=mu[0][2]/m00,n11=mu[1][1]/m00;
    double n30=mu[3][0]/m00,n03=mu[0][3]/m00;
    double n21=mu[2][1]/m00,n12=mu[1][2]/m00;
    hu[0]=(float)(n20+n02);
    hu[1]=(float)((n20-n02)*(n20-n02)+4*n11*n11);
    hu[2]=(float)((n30-3*n12)*(n30-3*n12)+(3*n21-n03)*(3*n21-n03));
    hu[3]=(float)((n30+n12)*(n30+n12)+(n21+n03)*(n21+n03));
    hu[4]=(float)((n30-3*n12)*(n30+n12)*((n30+n12)*(n30+n12)-3*(n21+n03)*(n21+n03))
                 +(3*n21-n03)*(n21+n03)*(3*(n30+n12)*(n30+n12)-(n21+n03)*(n21+n03)));
    hu[5]=(float)((n20-n02)*((n30+n12)*(n30+n12)-(n21+n03)*(n21+n03))
                 +4*n11*(n30+n12)*(n21+n03));
    hu[6]=(float)((3*n21-n03)*(n30+n12)*((n30+n12)*(n30+n12)-3*(n21+n03)*(n21+n03))
                 -(n30-3*n12)*(n21+n03)*(3*(n30+n12)*(n30+n12)-(n21+n03)*(n21+n03)));
    for (int i=0; i<HU_DIM; i++) {
        double v=hu[i];
        hu[i]=(float)(v==0?0.0:copysign(log10(fabs(v)+1e-10),v));
    }
    normalise(hu,HU_DIM);
}

static void build_descriptor(float *tile, Template *t) { //پایپلاین اصلی کد  تمام تابع هارا روی عکس اجرا می‌کند
    center_digit(tile);
    normalise(tile,FEAT_DIM);
    extract_conv(tile,t->conv);
    compute_zones(tile,t->zones);
    compute_projections(tile,t->proj);
    compute_hu(tile,t->hu);
}

static int load_template(const char *path, Template *t, int digit) {
    int w,h,c;
    uint8_t *img=stbi_load(path,&w,&h,&c,1);
    if (!img||w!=DIGIT_SIZE||h!=DIGIT_SIZE) { if(img) stbi_image_free(img); return 0; }
    float tile[FEAT_DIM];
    for (int i=0; i<FEAT_DIM; i++) tile[i]=img[i]/255.0f;
    stbi_image_free(img);
    build_descriptor(tile,t);
    t->digit=digit;
    return 1;
}

static float cosine(const float *a, const float *b, int n) {
    float dot=0,na=0,nb=0;
    for (int i=0; i<n; i++) {
        dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i];
    }
    float d=sqrtf(na)*sqrtf(nb);
    return (d>1e-8f)?dot/d:0.0f;
}

static float similarity(const Template *a, const Template *b) { //تشخیص شباهت با ضرایب مختلف برای توابع مختلف
    float conv_s=0;
    for (int k=0; k<NUM_KERNELS; k++)
        conv_s+=cosine(a->conv[k],b->conv[k],FEAT_DIM);
    conv_s/=NUM_KERNELS;
    float zone_s=cosine(a->zones,b->zones,ZONE_DIM);
    float proj_s=cosine(a->proj, b->proj, PROJ_DIM);
    float hu_s  =cosine(a->hu,   b->hu,   HU_DIM);
    return 0.50f*conv_s + 0.25f*zone_s + 0.15f*proj_s + 0.10f*hu_s;
}

static int top_k_vote(const Template *query,
                      const Template *templates, int n_tpl,
                      float *out_score) { //بیشتین شباهت بین تمپلیت ها
    float class_score[10]={0};
    for (int d=0; d<10; d++) {
        float scores[MAX_TEMPLATES]; int cnt=0;
        for (int t=0; t<n_tpl; t++) {
            if (templates[t].digit!=d) continue;
            scores[cnt++]=similarity(query,&templates[t]);
        }
        int k=(cnt<TOP_K)?cnt:TOP_K;
        for (int i=0; i<k; i++) {
            for (int j=i+1; j<cnt; j++)
                if (scores[j]>scores[i]) {
                    float tmp=scores[i]; scores[i]=scores[j]; scores[j]=tmp;
                }
            class_score[d]+=scores[i];
        }
        if (k>0) class_score[d]/=k;
    }
    int best=0;
    for (int d=1; d<10; d++)
        if (class_score[d]>class_score[best]) best=d;
    if (out_score) *out_score=class_score[best];
    return best;
}

static void save_annotated(const uint8_t *orig_gray, int orig_w, int orig_h,
                            const int *detected, int n_tiles) { //تولید عکس  حروجی
    int out_w = orig_w;
    int out_h = orig_h + LABEL_H;
    uint8_t *canvas = malloc(out_w * out_h * 3);
    memset(canvas, 255, out_w * out_h * 3);

    for (int y = 0; y < orig_h; y++)
        for (int x = 0; x < orig_w; x++) {
            uint8_t v = orig_gray[y * orig_w + x];
            int dst = ((y + LABEL_H) * out_w + x) * 3;
            canvas[dst+0] = v;
            canvas[dst+1] = v;
            canvas[dst+2] = v;
        }

    static const uint8_t COLOURS[10][3] = {
        {220,  50,  50},
        { 50, 160,  50},
        { 50,  50, 220},
        {200, 130,   0},
        {140,   0, 200},
        {  0, 170, 170},
        {200,   0, 120},
        { 80, 120,   0},
        {  0,  80, 160},
        {160,  80,   0},
    };

    for (int d = 0; d < n_tiles; d++) {
        int digit = detected[d];
        const uint8_t *col = COLOURS[d % 10];

        int x0 = d * DIGIT_SIZE;
        int x1 = x0 + DIGIT_SIZE - 1;
        int y0 = LABEL_H;
        int y1 = LABEL_H + DIGIT_SIZE - 1;

        draw_rect(canvas, out_w, out_h, x0, y0, x1, y1,
                  col[0], col[1], col[2]);
        int glyph_x = x0 + (DIGIT_SIZE - 5) / 2;
        int glyph_y = (LABEL_H - 7) / 2;
        draw_digit_glyph(canvas, out_w, out_h, glyph_x, glyph_y, digit, col[0], col[1], col[2]);
    }

    if (stbi_write_png(OUTPUT_PATH, out_w, out_h, 3, canvas, out_w*3)) {
        printf("\nAnnotated image saved -> %s\n", OUTPUT_PATH);
    } else {
        fprintf(stderr, "Warning: could not write %s\n", OUTPUT_PATH);
    }

    free(canvas);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <image.jpg> <templates/>\n", argv[0]);
        return 1;
    }

    int w, h, c;
    uint8_t *raw = stbi_load(argv[1], &w, &h, &c, 0);
    if (!raw) { fprintf(stderr, "Cannot open: %s\n", argv[1]); return 1; }

    uint8_t *gray8 = malloc(w * h);
    for (int i = 0; i < w*h; i++) {
        float g = (c==1) ? raw[i]
            : 0.299f*raw[i*c] + 0.587f*raw[i*c+1] + 0.114f*raw[i*c+2];
        gray8[i] = (uint8_t)g;
    }

    float *gray = malloc(sizeof(float)*w*h);
    for (int i = 0; i < w*h; i++) {
        float g = (c==1) ? raw[i]
            : 0.299f*raw[i*c] + 0.587f*raw[i*c+1] + 0.114f*raw[i*c+2];
        gray[i] = (255.0f - g) / 255.0f;
    }
    stbi_image_free(raw);

    Template *templates = malloc(sizeof(Template)*MAX_TEMPLATES);
    int n_tpl = 0;
    for (int d = 0; d < 10 && n_tpl < MAX_TEMPLATES; d++) {
        int loaded = 0;
        for (int i = 0; i < MAX_TEMPLATES && n_tpl < MAX_TEMPLATES; i++) {
            char path[512];
            snprintf(path, sizeof(path), "%s/digit_%d/template_%d.png", argv[2], d, i);
            if (load_template(path, &templates[n_tpl], d)) { n_tpl++; loaded++; }
            else if (i > 0) break;
        }
    }

    int n_tiles = w / DIGIT_SIZE;
    if (n_tiles > MAX_DIGITS) n_tiles = MAX_DIGITS;
    printf("%-6s %-6s %-10s\n", "Tile", "Digit", "Score");
    printf("%-6s %-6s %-10s\n", "----", "-----", "-----");

    float tile[FEAT_DIM];
    Template query;
    int detected[MAX_DIGITS];

    for (int d = 0; d < n_tiles; d++) {
        int x0 = d * DIGIT_SIZE;
        for (int y = 0; y < DIGIT_SIZE; y++)
            for (int x = 0; x < DIGIT_SIZE; x++)
                tile[y*DIGIT_SIZE+x] = gray[y*w + x0+x];
        build_descriptor(tile, &query);
        float score;
        detected[d] = top_k_vote(&query, templates, n_tpl, &score);
        printf("%-6d %-6d %.4f\n", d, detected[d], score);
    }

    printf("\nDetected sequence: ");
    for (int d = 0; d < n_tiles; d++) printf("%d", detected[d]);
    printf("\n");

    save_annotated(gray8, w, h, detected, n_tiles);

    free(gray);
    free(gray8);
    free(templates);
    return 0;
}