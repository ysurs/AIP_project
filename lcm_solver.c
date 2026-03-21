/*
 * lcm_solver.c
 * Kernels for local context matching: BGR conversions, dilation, resize,
 * texture map, and masked SSD template matching via Cooley-Tukey FFT.
 * Called from Python via ctypes.
 *
 * Build:
 *   gcc -O3 -march=native -shared -fPIC -o lcm_solver_lib.so lcm_solver.c -lm
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <complex.h>

/* ============================================================
 * Internal helper: next power of 2 >= n
 * ============================================================ */
static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

/* ============================================================
 * 1-D Cooley-Tukey radix-2 DIT FFT (in-place)
 * inv=0 => forward, inv=1 => inverse (normalises by 1/n)
 * n must be a power of 2.
 * ============================================================ */
static void fft1d(double complex *x, int n, int inv) {
    /* Bit-reversal permutation */
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) {
            double complex tmp = x[i]; x[i] = x[j]; x[j] = tmp;
        }
    }
    /* Butterfly stages */
    for (int length = 2; length <= n; length <<= 1) {
        int half   = length >> 1;
        double ang = (inv ? 2.0 : -2.0) * M_PI / length;
        double complex w_step = cos(ang) + sin(ang) * I;
        for (int start = 0; start < n; start += length) {
            double complex w = 1.0 + 0.0 * I;
            for (int k = 0; k < half; k++) {
                double complex u = x[start + k];
                double complex v = x[start + k + half] * w;
                x[start + k]        = u + v;
                x[start + k + half] = u - v;
                w *= w_step;
            }
        }
    }
    if (inv) {
        double inv_n = 1.0 / n;
        for (int i = 0; i < n; i++) x[i] *= inv_n;
    }
}

/* ============================================================
 * 2-D FFT (in-place) on a ph*pw flat array (row-major).
 * ph and pw must be powers of 2.
 * ============================================================ */
static void fft2d(double complex *grid, int ph, int pw, int inv) {
    /* Row FFTs */
    for (int r = 0; r < ph; r++)
        fft1d(grid + r * pw, pw, inv);
    /* Column FFTs — need a temporary column buffer */
    double complex *col = (double complex *)malloc(ph * sizeof(double complex));
    for (int c = 0; c < pw; c++) {
        for (int r = 0; r < ph; r++) col[r] = grid[r * pw + c];
        fft1d(col, ph, inv);
        for (int r = 0; r < ph; r++) grid[r * pw + c] = col[r];
    }
    free(col);
}

/* ============================================================
 * 2-D cross-correlation 'valid' mode via FFT.
 * out shape: (ah-bh+1) × (aw-bw+1)
 * ============================================================ */
static void correlate_valid(const float *a, int ah, int aw,
                             const float *b, int bh, int bw,
                             float *out) {
    int pad_h = ah + bh - 1;
    int pad_w = aw + bw - 1;
    int ph    = next_pow2(pad_h);
    int pw    = next_pow2(pad_w);
    int sz    = ph * pw;

    double complex *fa = (double complex *)calloc(sz, sizeof(double complex));
    double complex *fb = (double complex *)calloc(sz, sizeof(double complex));

    /* Fill fa with a (zero-padded) */
    for (int r = 0; r < ah; r++)
        for (int c = 0; c < aw; c++)
            fa[r * pw + c] = (double complex)a[r * aw + c];

    /* Fill fb with b spatially reversed (for cross-correlation) */
    for (int r = 0; r < bh; r++)
        for (int c = 0; c < bw; c++)
            fb[(bh - 1 - r) * pw + (bw - 1 - c)] = (double complex)b[r * bw + c];

    fft2d(fa, ph, pw, 0);
    fft2d(fb, ph, pw, 0);

    /* Pointwise multiply */
    for (int i = 0; i < sz; i++) fa[i] *= fb[i];

    fft2d(fa, ph, pw, 1);  /* IFFT in-place */

    /* Extract valid region starting at (bh-1, bw-1) */
    int out_h = ah - bh + 1;
    int out_w = aw - bw + 1;
    for (int r = 0; r < out_h; r++)
        for (int c = 0; c < out_w; c++)
            out[r * out_w + c] = (float)creal(fa[(bh - 1 + r) * pw + (bw - 1 + c)]);

    free(fa);
    free(fb);
}

/* ============================================================
 * PUBLIC API
 * ============================================================ */

/* ------------------------------------------------------------
 * lcm_bgr_to_gray
 * bgr uint8 [h×w×3] BGR → out float32 [h×w]
 * Weights: B=0.114, G=0.587, R=0.299
 * ------------------------------------------------------------ */
void lcm_bgr_to_gray(const uint8_t *bgr, int h, int w, float *out) {
    int n = h * w;
    for (int i = 0; i < n; i++) {
        const uint8_t *p = bgr + i * 3;
        out[i] = 0.114f * (float)p[0]   /* B */
               + 0.587f * (float)p[1]   /* G */
               + 0.299f * (float)p[2];  /* R */
    }
}

/* ------------------------------------------------------------
 * lcm_bgr_to_lab
 * bgr uint8 [h×w×3] BGR → out float32 [h×w×3] CIE L*a*b* (D65)
 * ------------------------------------------------------------ */
void lcm_bgr_to_lab(const uint8_t *bgr, int h, int w, float *out) {
    /* sRGB -> linear sRGB -> XYZ (D65) matrix */
    static const double M[3][3] = {
        {0.4124564, 0.3575761, 0.1804375},
        {0.2126729, 0.7151522, 0.0721750},
        {0.0193339, 0.1191920, 0.9503041}
    };
    const double eps = 0.008856;
    int n = h * w;

    for (int i = 0; i < n; i++) {
        const uint8_t *p = bgr + i * 3;
        double b_v = (double)p[0] / 255.0;
        double g_v = (double)p[1] / 255.0;
        double r_v = (double)p[2] / 255.0;

        /* Linearise sRGB */
        double r_l = (r_v > 0.04045) ? pow((r_v + 0.055) / 1.055, 2.4) : r_v / 12.92;
        double g_l = (g_v > 0.04045) ? pow((g_v + 0.055) / 1.055, 2.4) : g_v / 12.92;
        double b_l = (b_v > 0.04045) ? pow((b_v + 0.055) / 1.055, 2.4) : b_v / 12.92;

        /* Linear RGB -> XYZ (D65) */
        double X = M[0][0]*r_l + M[0][1]*g_l + M[0][2]*b_l;
        double Y = M[1][0]*r_l + M[1][1]*g_l + M[1][2]*b_l;
        double Z = M[2][0]*r_l + M[2][1]*g_l + M[2][2]*b_l;

        /* Normalise by D65 white point */
        X /= 0.95047;
        /* Y stays (white Y=1.0) */
        Z /= 1.08883;

        /* XYZ -> f(t) with cbrt / linear knee */
        double fx = (X > eps) ? cbrt(X) : (903.3 * X + 16.0) / 116.0;
        double fy = (Y > eps) ? cbrt(Y) : (903.3 * Y + 16.0) / 116.0;
        double fz = (Z > eps) ? cbrt(Z) : (903.3 * Z + 16.0) / 116.0;

        float *o = out + i * 3;
        o[0] = (float)(116.0 * fy - 16.0);   /* L */
        o[1] = (float)(500.0 * (fx - fy));    /* a */
        o[2] = (float)(200.0 * (fy - fz));    /* b */
    }
}

/* ------------------------------------------------------------
 * lcm_dilate
 * Binary morphological dilation with circular SE of given radius.
 * mask_in uint8 [h×w] → mask_out uint8 [h×w]
 * ------------------------------------------------------------ */
void lcm_dilate(const uint8_t *mask_in, int h, int w, int radius, uint8_t *mask_out) {
    memset(mask_out, 0, (size_t)h * w);
    int r2 = radius * radius;

    for (int py = 0; py < h; py++) {
        for (int px = 0; px < w; px++) {
            if (!mask_in[py * w + px]) continue;

            int y_lo = py - radius < 0     ? 0     : py - radius;
            int y_hi = py + radius >= h    ? h - 1 : py + radius;

            for (int ny = y_lo; ny <= y_hi; ny++) {
                int dy2    = (ny - py) * (ny - py);
                int max_dx = (int)sqrt((double)(r2 - dy2));
                int x_lo   = px - max_dx < 0     ? 0     : px - max_dx;
                int x_hi   = px + max_dx >= w    ? w - 1 : px + max_dx;
                memset(mask_out + ny * w + x_lo, 1, (size_t)(x_hi - x_lo + 1));
            }
        }
    }
}

/* ------------------------------------------------------------
 * lcm_resize
 * Bilinear resize. src [sh×sw×nc] HWC → dst [dh×dw×nc] HWC.
 * Coordinate mapping: src_coord = dst_i * (src_dim-1) / (dst_dim-1)
 * ------------------------------------------------------------ */
void lcm_resize(const float *src, int sh, int sw,
                float *dst,       int dh, int dw, int nc) {
    for (int dy = 0; dy < dh; dy++) {
        double ys  = (dh > 1) ? (double)dy * (sh - 1) / (dh - 1) : 0.0;
        int    y0  = (int)ys;
        int    y1  = (y0 + 1 < sh) ? y0 + 1 : sh - 1;
        double yf  = ys - y0;
        double y0f = 1.0 - yf;

        for (int dx = 0; dx < dw; dx++) {
            double xs  = (dw > 1) ? (double)dx * (sw - 1) / (dw - 1) : 0.0;
            int    x0  = (int)xs;
            int    x1  = (x0 + 1 < sw) ? x0 + 1 : sw - 1;
            double xf  = xs - x0;
            double x0f = 1.0 - xf;

            float *o = dst + (dy * dw + dx) * nc;
            const float *tl = src + (y0 * sw + x0) * nc;
            const float *tr = src + (y0 * sw + x1) * nc;
            const float *bl = src + (y1 * sw + x0) * nc;
            const float *br = src + (y1 * sw + x1) * nc;

            double w_tl = y0f * x0f;
            double w_tr = y0f * xf;
            double w_bl = yf  * x0f;
            double w_br = yf  * xf;

            for (int c = 0; c < nc; c++) {
                o[c] = (float)(w_tl * tl[c] + w_tr * tr[c] +
                               w_bl * bl[c] + w_br * br[c]);
            }
        }
    }
}

/* ------------------------------------------------------------
 * lcm_texture_map
 * Sobel gradient magnitude + 5×5 median blur + normalise [0,255].
 * gray float32 [h×w] → out float32 [h×w]
 * ------------------------------------------------------------ */

/* qsort comparator for float */
static int _cmp_float(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

void lcm_texture_map(const float *gray, int h, int w, float *out) {
    /* Sobel kernels:
     *   kx = [[-1,0,1],[-2,0,2],[-1,0,1]]
     *   ky = [[-1,-2,-1],[0,0,0],[1,2,1]]
     */

    /* --- Compute gradient magnitude with edge-clamped padding --- */
    float *mag = (float *)malloc((size_t)h * w * sizeof(float));

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            /* Sample with edge clamping */
#define CLAMP_GRAY(yy, xx) gray[ \
    ((yy) < 0 ? 0 : (yy) >= h ? h-1 : (yy)) * w + \
    ((xx) < 0 ? 0 : (xx) >= w ? w-1 : (xx)) ]

            float gx = (-1.0f * CLAMP_GRAY(y-1, x-1) + 0.0f * CLAMP_GRAY(y-1, x) + 1.0f * CLAMP_GRAY(y-1, x+1)
                      + -2.0f * CLAMP_GRAY(y,   x-1) + 0.0f * CLAMP_GRAY(y,   x) + 2.0f * CLAMP_GRAY(y,   x+1)
                      + -1.0f * CLAMP_GRAY(y+1, x-1) + 0.0f * CLAMP_GRAY(y+1, x) + 1.0f * CLAMP_GRAY(y+1, x+1));

            float gy = (-1.0f * CLAMP_GRAY(y-1, x-1) + -2.0f * CLAMP_GRAY(y-1, x) + -1.0f * CLAMP_GRAY(y-1, x+1)
                      +  0.0f * CLAMP_GRAY(y,   x-1) +  0.0f * CLAMP_GRAY(y,   x) +  0.0f * CLAMP_GRAY(y,   x+1)
                      +  1.0f * CLAMP_GRAY(y+1, x-1) +  2.0f * CLAMP_GRAY(y+1, x) +  1.0f * CLAMP_GRAY(y+1, x+1));
#undef CLAMP_GRAY

            mag[y * w + x] = sqrtf(gx * gx + gy * gy);
        }
    }

    /* --- 5×5 median blur with edge-clamped padding --- */
    float window[25];
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int k = 0;
            for (int dy = -2; dy <= 2; dy++) {
                int ny = y + dy;
                if (ny < 0)   ny = 0;
                if (ny >= h)  ny = h - 1;
                for (int dx = -2; dx <= 2; dx++) {
                    int nx = x + dx;
                    if (nx < 0)  nx = 0;
                    if (nx >= w) nx = w - 1;
                    window[k++] = mag[ny * w + nx];
                }
            }
            qsort(window, 25, sizeof(float), _cmp_float);
            out[y * w + x] = window[12];  /* median of 25 elements */
        }
    }
    free(mag);

    /* --- Normalise to [0, 255] --- */
    int n     = h * w;
    float mn  = out[0];
    float mx  = out[0];
    for (int i = 1; i < n; i++) {
        if (out[i] < mn) mn = out[i];
        if (out[i] > mx) mx = out[i];
    }
    if (mx > mn) {
        float scale = 255.0f / (mx - mn);
        for (int i = 0; i < n; i++)
            out[i] = (out[i] - mn) * scale;
    }
}

/* ------------------------------------------------------------
 * lcm_match_ssd
 * Masked SSD template matching via FFT.
 *
 * SSD(y,x) = sum_{i,j} M[i,j] * (S[y+i,x+j] - T[i,j])^2
 *           = corr(S^2, M) - 2*corr(S, M*T) + sum(M*T^2)
 *
 * search  float32 [sh×sw×nc] HWC
 * templ   float32 [th×tw×nc] HWC
 * mask    float32 [th×tw]    (single channel)
 * out_ssd float32 [(sh-th+1)×(sw-tw+1)]  — zeroed on entry, accumulated
 * ------------------------------------------------------------ */
void lcm_match_ssd(const float *search, int sh, int sw,
                   const float *templ,  int th, int tw,
                   const float *mask,   int nc,
                   float *out_ssd) {
    int out_h   = sh - th + 1;
    int out_w   = sw - tw + 1;
    int out_n   = out_h * out_w;
    int s_area  = sh * sw;
    int t_area  = th * tw;

    /* Initialise output to 0 */
    memset(out_ssd, 0, (size_t)out_n * sizeof(float));

    /* Temporary per-channel flat buffers */
    float *s_ch  = (float *)malloc(s_area * sizeof(float));
    float *t_ch  = (float *)malloc(t_area * sizeof(float));
    float *s_sq  = (float *)malloc(s_area * sizeof(float));
    float *mt    = (float *)malloc(t_area * sizeof(float));
    float *corr1 = (float *)malloc(out_n  * sizeof(float));
    float *corr2 = (float *)malloc(out_n  * sizeof(float));

    for (int ch = 0; ch < nc; ch++) {
        /* Extract channel from HWC */
        for (int i = 0; i < s_area; i++) s_ch[i] = search[i * nc + ch];
        for (int i = 0; i < t_area; i++) t_ch[i] = templ[i * nc + ch];

        /* s^2 and M*T */
        for (int i = 0; i < s_area; i++) s_sq[i] = s_ch[i] * s_ch[i];

        double sum_mt2 = 0.0;
        for (int i = 0; i < t_area; i++) {
            float m_val   = mask[i];
            float t_val   = t_ch[i];
            mt[i]         = m_val * t_val;
            sum_mt2      += (double)(m_val * t_val * t_val);
        }

        /* corr(S^2, M) */
        correlate_valid(s_sq, sh, sw, mask, th, tw, corr1);
        /* corr(S, M*T)  */
        correlate_valid(s_ch, sh, sw, mt,   th, tw, corr2);

        /* Accumulate: out_ssd += corr1 - 2*corr2 + sum_mt2 */
        float fsum_mt2 = (float)sum_mt2;
        for (int i = 0; i < out_n; i++)
            out_ssd[i] += corr1[i] - 2.0f * corr2[i] + fsum_mt2;
    }

    free(s_ch);
    free(t_ch);
    free(s_sq);
    free(mt);
    free(corr1);
    free(corr2);
}
