// rmbg-rs kernel family: the vision-model ops (Swin window assembly, token
// shuffles, deformable convolution, resampling, NCHW channel ops, token-layout
// attention) that only this engine calls.
//
// These 19 kernels used to live in the shared `lightgpu` toolkit, where they sat
// next to an LLM/q8 kernel set with which they share almost nothing.
// They are compiled as a SECOND MODULE (see build.rs / fatbin_modules) and
// loaded alongside the toolkit subset, so a name here cannot collide with a
// toolkit kernel and vice versa.
//
// The conventions are the toolkit's, because these were written to them:
//   * `extern "C" __global__` with no engine structs, raw pointers plus scalars
//     only, so the driver API can look them up by name;
//   * no fast math, no flush-to-zero - every kernel keeps the same order of
//     operations as this crate's CPU path (src/cuda_graph.rs), so a difference
//     between the backends is a real one, which is what `--cuda-selftest`
//     checks;
//   * reduction scratch is sized for the largest legal blockDim so no caller has
//     to pass a shared-memory size.
//
// The two `LA_DEVI` helpers below (la_bilinear_zero, la_window_index) are here
// because their only callers are in this file; la_erf stayed behind with
// lg_gelu_erf, and `lg_channel_affine`/`lg_channel_mean` have gone back to the
// toolkit, where the ops table advertised them all along.
//
// `la_bilinear_zero` is the one helper that is arguably engine-agnostic - it is
// a plain bilinear sample with zero padding, torchvision's grid_sample rule -
// and the answer is still that it stays HERE. It cannot be shared as a kernel,
// because a device helper is inlined into its caller's translation unit and the
// toolkit is compiled as its own source file that no consumer includes; sharing
// it would mean either a toolkit header for device code (a new mechanism, and
// the toolkit's whole contract is "one .cu, looked up by name") or duplicating
// the inside of `lg_deform_conv`, which is a swin-specific op with a
// torchvision-compatible offset/mask layout. See CONVENTIONS.md in the toolkit
// for the recorded decision.
//
// Because the affine and the mean are now the toolkit's, their comments moved
// with them; what remains below is the token/window/deform machinery that no
// other engine in this family has.

#include <cuda_runtime.h>
#include <cstdint>

#define LA_DEVI __device__ __forceinline__


// ===========================================================================
// 1. Elementwise (vision-model ops)
// ===========================================================================

// 2 * sigmoid(x), elementwise (the deformable-conv modulator).
extern "C" __global__ void lg_double_sigmoid(
    const float *__restrict__ x, float *__restrict__ y, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = 2.0f / (1.0f + __expf(-x[i]));
}

// Copy with a channel offset: dst[dst_c0 + c][p] = src[c][p]  (NCHW concat)
extern "C" __global__ void lg_channel_copy(
    const float *__restrict__ src, float *__restrict__ dst,
    int src_c, int dst_c0, int hw)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)src_c * hw;
    if (idx >= total) return;
    const int c = (int)(idx / hw);
    const long p = idx % hw;
    dst[(size_t)(dst_c0 + c) * hw + p] = src[idx];
}

// In-place multiply by a single-channel map: x[c][p] *= a[p]  (GDT attention)
extern "C" __global__ void lg_mul_broadcast(
    float *__restrict__ x, const float *__restrict__ a, int c, int hw)
{
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)c * hw) return;
    x[i] *= a[i % hw];
}

// ===========================================================================
// 2. Reductions over the channel/row axis
// ===========================================================================

// Per-row softmax over the last axis of a (rows, row_pitch) matrix, in place.
// The CPU twin's scan folds with f32::max starting from -inf; using -inf here
// keeps the same "every element participates" behaviour.
extern "C" __global__ void lg_softmax_rows(float *__restrict__ w, int n, int row_pitch) {
    extern __shared__ float sh[];
    const int row = blockIdx.x;
    float *r = w + (size_t)row * row_pitch;
    const int t = threadIdx.x;
    float mx = -__int_as_float(0x7f800000);
    for (int i = t; i < n; i += blockDim.x) mx = fmaxf(mx, r[i]);
    sh[t] = mx;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s) sh[t] = fmaxf(sh[t], sh[t + s]);
        __syncthreads();
    }
    const float gm = sh[0];
    __syncthreads();
    float sum = 0.0f;
    for (int i = t; i < n; i += blockDim.x) {
        const float e = __expf(r[i] - gm);
        r[i] = e;
        sum += e;
    }
    sh[t] = sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        __syncthreads();
    }
    const float inv = 1.0f / sh[0];
    __syncthreads();
    for (int i = t; i < n; i += blockDim.x) r[i] *= inv;
}

// ===========================================================================
// 3. Deformable convolution and resampling
// ===========================================================================

// Bilinear sampling with zero padding (deformable conv's inner resample).
LA_DEVI float la_bilinear_zero(const float *__restrict__ plane, int h, int wd, float y, float x) {
    const float y0f = floorf(y);
    const float x0f = floorf(x);
    const int y0 = (int)y0f;
    const int x0 = (int)x0f;
    const float ly = y - y0f;
    const float lx = x - x0f;
    float acc = 0.0f;
    for (int a = 0; a < 2; ++a) {
        const float wy = a == 0 ? (1.0f - ly) : ly;
        const int yy = y0 + a;
        if (yy < 0 || yy >= h || wy == 0.0f) continue;
        for (int b = 0; b < 2; ++b) {
            const float wx = b == 0 ? (1.0f - lx) : lx;
            const int xx = x0 + b;
            if (xx < 0 || xx >= wd || wx == 0.0f) continue;
            acc += wy * wx * plane[(size_t)yy * wd + xx];
        }
    }
    return acc;
}

// Modulated deformable convolution (DCNv2), torchvision-compatible. Per-output-
// channel thread; loops oy,ox -> ky,kx -> ci, skipping m == 0 and w == 0, and
// bilinearly resampling per ci.
//   offset layout: (2*k*k, H, W), pitch H*W, [2*kk] = dy, [2*kk+1] = dx
//   mask layout:   (k*k,   H, W), pitch H*W
extern "C" __global__ void lg_deform_conv(
    const float *__restrict__ in, const float *__restrict__ offset,
    const float *__restrict__ mask, const float *__restrict__ weight,
    float *__restrict__ out, int c_in, int c_out, int h, int wd, int k, int pad)
{
    const int co = blockIdx.y;
    if (co >= c_out) return;
    const int hw = h * wd;
    const int kk_total = k * k;
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= hw) return;
    const int oy = (int)(t / wd);
    const int ox = (int)(t % wd);
    const int p = oy * wd + ox;
    const float *wco = weight + (size_t)co * c_in * kk_total;
    float acc = 0.0f;
    for (int ky = 0; ky < k; ++ky) {
        for (int kx = 0; kx < k; ++kx) {
            const int kk = ky * k + kx;
            const float off_y = offset[(size_t)(2 * kk) * hw + p];
            const float off_x = offset[(size_t)(2 * kk + 1) * hw + p];
            const float m = mask[(size_t)kk * hw + p];
            if (m == 0.0f) continue;
            const float sy = (float)oy + (float)ky - (float)pad + off_y;
            const float sx = (float)ox + (float)kx - (float)pad + off_x;
            for (int ci = 0; ci < c_in; ++ci) {
                const float wv = wco[(size_t)ci * kk_total + kk];
                if (wv == 0.0f) continue;
                const float v = la_bilinear_zero(in + (size_t)ci * hw, h, wd, sy, sx);
                acc += wv * m * v;
            }
        }
    }
    out[(size_t)co * hw + p] = acc;
}

// Bilinear resize, NCHW, both torch align_corners modes. Same source-index
// computation and lerp order as the CPU twin:
//   top = v00 + (v01 - v00) * wx;  bot = v10 + (v11 - v10) * wx
//   out = top + (bot - top) * wy
extern "C" __global__ void lg_resize_bilinear(
    const float *__restrict__ in, float *__restrict__ out,
    int c, int ih, int iw, int oh, int ow, int align_corners)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)c * oh * ow;
    if (idx >= total) return;
    const int xo = (int)(idx % ow);
    const long t = idx / ow;
    const int yo = (int)(t % oh);
    const int ch = (int)(t / oh);

    float rh, rw;
    if (align_corners) {
        rh = (oh > 1) ? (float)(ih - 1) / (float)(oh - 1) : 0.0f;
        rw = (ow > 1) ? (float)(iw - 1) / (float)(ow - 1) : 0.0f;
    } else {
        rh = (float)ih / (float)oh;
        rw = (float)iw / (float)ow;
    }
    const float fy = align_corners ? (float)yo * rh : fmaxf(((float)yo + 0.5f) * rh - 0.5f, 0.0f);
    const float fx = align_corners ? (float)xo * rw : fmaxf(((float)xo + 0.5f) * rw - 0.5f, 0.0f);
    const float y0f = floorf(fy);
    const float x0f = floorf(fx);
    const int y0 = min((int)y0f, ih - 1);
    const int y1 = min(y0 + 1, ih - 1);
    const int x0 = min((int)x0f, iw - 1);
    const int x1 = min(x0 + 1, iw - 1);
    const float wy = fy - y0f;
    const float wx = fx - x0f;

    const float *src = in + (size_t)ch * ih * iw;
    const float v00 = src[(size_t)y0 * iw + x0];
    const float v01 = src[(size_t)y0 * iw + x1];
    const float v10 = src[(size_t)y1 * iw + x0];
    const float v11 = src[(size_t)y1 * iw + x1];
    const float top = v00 + (v01 - v00) * wx;
    const float bot = v10 + (v11 - v10) * wx;
    out[idx] = top + (bot - top) * wy;
}

// ===========================================================================
// 4. Layout transposes between NCHW and the token layout
// ===========================================================================

// NCHW (c*HW + p) -> tokens (p*C + c)
extern "C" __global__ void lg_nchw_to_tokens(
    const float *__restrict__ in, float *__restrict__ out, int c, int hw)
{
    const long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long)c * hw) return;
    const int ch = (int)(gid / hw);
    const long p = gid % hw;
    out[p * c + ch] = in[gid];
}

// tokens (p*C + c) -> NCHW (c*HW + p)
extern "C" __global__ void lg_tokens_to_nchw(
    const float *__restrict__ in, float *__restrict__ out, int c, int hw)
{
    const long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long)c * hw) return;
    const int ch = (int)(gid / hw);
    const long p = gid % hw;
    out[gid] = in[p * c + ch];
}

// ===========================================================================
// 5. Attention over the token layout (non-flash: scores / apply / repack)
// ===========================================================================

// Window attention, scores stage: qkv is (n_win, 3, heads, N, hd).
//   scores[win,h,i,j] = scale * dot(q_i, k_j) + bias_table[idx*heads + h]
//                       [+ mask[win,i,j] when non-null]
// `index` is the (N*N,) f32 copy of the relative-position index table.
extern "C" __global__ void lg_attn_scores(
    const float *__restrict__ qkv, const float *__restrict__ bias_table,
    const float *__restrict__ index, const float *__restrict__ mask,
    float *__restrict__ scores, int heads, int hd, int n, float scale)
{
    const long n_all = (long)heads * n * n;
    const long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_all) return;
    // blockIdx.y selects the window: the flat index must NOT be derived from
    // blockIdx.x alone, or every block in y would recompute window 0.
    const int win = (int)blockIdx.y;
    const int h = (int)(gid / ((long)n * n));
    const long rr = gid % ((long)n * n);
    const int i = (int)(rr / n);
    const int j = (int)(rr % n);

    const float *base = qkv + (((size_t)win * 3 + 0) * heads + h) * n * hd;
    const float *kb = qkv + (((size_t)win * 3 + 1) * heads + h) * n * hd;
    float acc = 0.0f;
    const float *qi = base + (size_t)i * hd;
    const float *kj = kb + (size_t)j * hd;
    for (int d = 0; d < hd; ++d) acc += qi[d] * kj[d];
    float s = acc * scale;
    const int idx = (int)index[i * n + j];
    s += bias_table[(size_t)idx * heads + h];
    if (mask) s += mask[((size_t)win * n + i) * n + j];
    scores[((size_t)win * heads + h) * n * n + (size_t)i * n + j] = s;
}

// attn (n_win, heads, N, N) x v -> out (rows, C) with out[i*C + h*hd + d]
extern "C" __global__ void lg_attn_apply(
    const float *__restrict__ attn, const float *__restrict__ qkv,
    float *__restrict__ out, int heads, int c, int hd, int n, int rows)
{
    const long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long)rows * c) return;
    const int i = (int)(gid / c);
    const int ch = (int)(gid % c);
    const int h = ch / hd;
    const int d = ch % hd;
    const int win = i / n;
    const int ii = i % n;
    const float *a = attn + ((size_t)win * heads + h) * n * n + (size_t)ii * n;
    const float *vb = qkv + (((size_t)win * 3 + 2) * heads + h) * n * hd;
    float acc = 0.0f;
    for (int j = 0; j < n; ++j) acc += a[j] * vb[(size_t)j * hd + d];
    out[(size_t)i * c + ch] = acc;
}

// Repack (rows, 3C) token qkv into (n_win, 3, heads, N, hd):
//   dst[((((w*3 + blk)*heads + h)*N + i)*hd) + d]
extern "C" __global__ void lg_repack_attn(
    const float *__restrict__ qkv, float *__restrict__ dst,
    int rows, int c, int heads, int hd, int n)
{
    const long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)rows * 3 * c;
    if (gid >= total) return;
    const int d = (int)(gid % hd);
    long t = gid / hd;
    const int ii = (int)(t % n);
    t /= n;
    const int h = (int)(t % heads);
    t /= heads;
    const int blk = (int)(t % 3);
    const int win = (int)(t / 3);
    const long src = ((long)win * n + ii) * (3 * c) + (long)blk * c + (long)h * hd + d;
    const long dd = ((((long)win * 3 + blk) * heads + h) * n + ii) * hd + d;
    dst[dd] = qkv[src];
}

// ===========================================================================
// 6. Window assembly and token shuffles (Swin)
// ===========================================================================

// Shared index computation for the window gather/scatter pair.
LA_DEVI void la_window_index(long gid, int hp, int wp, int win, int c, long *src, long *dst) {
    const int ch = (int)(gid % c);
    long t = gid / c;
    const int j = (int)(t % win);
    t /= win;
    const int i = (int)(t % win);
    t /= win;
    const int wx = (int)(t % (wp / win));
    const int wy = (int)(t / (wp / win));
    const int widx = wy * (wp / win) + wx;
    *src = (long)ch * hp * wp + (long)(wy * win + i) * wp + (wx * win + j);
    *dst = ((long)widx * win * win + (long)i * win + j) * c + ch;
}

extern "C" __global__ void lg_window_gather(
    const float *__restrict__ padded, float *__restrict__ out,
    int hp, int wp, int win, int c)
{
    const long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long)c * hp * wp) return;
    long s, d;
    la_window_index(gid, hp, wp, win, c, &s, &d);
    out[d] = padded[s];
}

extern "C" __global__ void lg_window_scatter(
    const float *__restrict__ wins, float *__restrict__ padded,
    int hp, int wp, int win, int c)
{
    const long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long)c * hp * wp) return;
    long s, d;
    la_window_index(gid, hp, wp, win, c, &s, &d);
    padded[s] = wins[d];
}

// pad: (C,1,h*w) -> (C,1,hp*wp) with the (h,w) block placed at the origin.
extern "C" __global__ void lg_pad_tokens(
    const float *__restrict__ in, float *__restrict__ out,
    int h, int wd, int hp, int wp, int c)
{
    const long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long)c * hp * wp) return;
    const int ch = (int)(gid / ((long)hp * wp));
    const long p = gid % ((long)hp * wp);
    const int y = (int)(p / wp);
    const int x = (int)(p % wp);
    float v = 0.0f;
    if (y < h && x < wd) v = in[((size_t)y * wd + x) * c + ch];
    out[gid] = v;
}

// Cyclic roll of a (C,1,hp*wp) buffer by (sh, sh) with wraparound.
extern "C" __global__ void lg_roll_tokens(
    const float *__restrict__ in, float *__restrict__ out,
    int hp, int wp, int sh, int c)
{
    const long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long)c * hp * wp) return;
    const int ch = (int)(gid / ((long)hp * wp));
    const long p = gid % ((long)hp * wp);
    const int y = (int)(p / wp);
    const int x = (int)(p % wp);
    const int sy = (y + sh) % hp;
    const int sx = (x + sh) % wp;
    out[gid] = in[(size_t)ch * hp * wp + (size_t)sy * wp + sx];
}

// residual + crop, NCHW: out = x + crop(buf); x is (C,1,h*w), buf (C,1,hp*wp)
extern "C" __global__ void lg_add_crop(
    const float *__restrict__ x, const float *__restrict__ buf, float *__restrict__ out,
    int h, int wd, int wp, int hp, int c)
{
    const long hpw = (long)hp * wp;
    const long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long)c * h * wd) return;
    const int ch = (int)(gid / ((long)h * wd));
    const long p = gid % ((long)h * wd);
    const int y = (int)(p / wd);
    const int xx = (int)(p % wd);
    out[gid] = x[gid] + buf[(size_t)ch * hpw + (size_t)y * wp + xx];
}

// residual + crop, token layout: out[p*C + c] = x[p*C + c] + buf[c*hp*wp + y*wp + x]
extern "C" __global__ void lg_add_crop_tokens(
    const float *__restrict__ x, const float *__restrict__ buf, float *__restrict__ out,
    int h, int wd, int wp, int hp, int c)
{
    const long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long)h * wd * c) return;
    const int ch = (int)(gid % c);
    const long p = gid / c;
    const int y = (int)(p / wd);
    const int xx = (int)(p % wd);
    out[gid] = x[gid] + buf[(size_t)ch * hp * wp + (size_t)y * wp + xx];
}

// 2x2 patch merge (SwinStage downsample): in (C,1,h*w) -> out (4C,1,oh*ow),
// taps in the order (y0,x0), (y1,x0), (y0,x1), (y1,x1), zero outside source.
//
// Note: this is the Swin variant of the merge. lg_merge_2x2 is the MoonViT
// variant, which uses a different tap order and a grid/gridwidth source index;
// they are not interchangeable, hence two kernels rather than one.
extern "C" __global__ void lg_patch_merge(
    const float *__restrict__ in, float *__restrict__ out,
    int h, int wd, int c, int oh, int ow)
{
    const long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long)4 * c * oh * ow) return;
    const int ch = (int)(gid % c);
    long t = gid / c;
    const int g = (int)(t % 4);
    const long p = t / 4;
    const int y = (int)(p / ow);
    const int x = (int)(p % ow);
    const int dy = (g == 1 || g == 3) ? 1 : 0;
    const int dx = (g == 2 || g == 3) ? 1 : 0;
    const int sy = y * 2 + dy;
    const int sx = x * 2 + dx;
    float v = 0.0f;
    if (sy < h && sx < wd) v = in[((size_t)sy * wd + sx) * c + ch];
    out[p * (4 * c) + (long)g * c + ch] = v;
}

// get_patches: tile (C,h,w) into ph x pw tiles, column-then-row ordered, image
// channels innermost: out channel = (cw*nrow + ch)*C + cy. Out-of-range is zero.
extern "C" __global__ void lg_tile_patches(
    const float *__restrict__ in, float *__restrict__ out,
    int c, int h, int wd, int ph, int pw)
{
    const int nrow = (h + ph - 1) / ph;
    const int ncol = (wd + pw - 1) / pw;
    const long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long)c * nrow * ncol * ph * pw) return;
    // Destination is row-major (co, y, x) with x innermost, so x must be taken
    // modulo pw FIRST; taking y first transposes every tile.
    const int x = (int)(gid % pw);
    long t = gid / pw;
    const int y = (int)(t % ph);
    t /= ph;
    const int co = (int)(t % (c * nrow * ncol));
    const int cy = co % c;
    const int tile = co / c;
    const int ch = tile % nrow;
    const int cw = tile / nrow;
    const int iy = ch * ph + y;
    const int ix = cw * pw + x;
    float v = 0.0f;
    if (iy < h && ix < wd) v = in[(size_t)cy * h * wd + (size_t)iy * wd + ix];
    out[gid] = v;
}
