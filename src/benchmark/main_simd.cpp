#include "raylib.h"
#include <arm_neon.h>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <iomanip>

// ============================================================
// Phase 3: ARM NEON SIMD implementation
// - Structure of Arrays (SoA) for cache-friendly vectorization
// - Inner j-loop processes 4 particles at a time with NEON
// - Same physics as main_improved.cpp (baseline)
// ============================================================

struct SPHSettings {
    float mass;
    float restDensity;
    float gasConstant;
    float viscosity;
    float h;
    float h2;
    float g;
    float poly6;
    float spikyGrad;
    float spikyLap;
    float massPoly6;   // mass * poly6 (precomputed for density kernel)

    SPHSettings(float mass, float restDensity, float gasConst,
                float viscosity, float h, float g)
        : mass(mass), restDensity(restDensity), gasConstant(gasConst),
          viscosity(viscosity), h(h), g(g)
    {
        h2 = h * h;
        poly6 = 315.0f / (64.0f * PI * powf(h, 9));
        spikyGrad = -45.0f / (PI * powf(h, 6));
        spikyLap = 45.0f / (PI * powf(h, 6));
        massPoly6 = mass * poly6;
    }
};

// ============================================================
// SoA particle data — one array per component
// ============================================================
struct ParticlesSoA {
    std::vector<float> pos_x, pos_y;
    std::vector<float> vel_x, vel_y;
    std::vector<float> force_x, force_y;
    std::vector<float> density, pressure;
    int n = 0;

    void resize(int count) {
        n = count;
        pos_x.resize(n); pos_y.resize(n);
        vel_x.resize(n); vel_y.resize(n);
        force_x.resize(n); force_y.resize(n);
        density.resize(n); pressure.resize(n);
    }
};

// ============================================================
// Initialization: dam break block spanning full screen width
// ============================================================
void initialize_dam_break(ParticlesSoA& p, const SPHSettings& s,
                          float windowWidth, float windowHeight) {
    std::srand(1024);

    const float margin = 20.0f;
    float sep = s.h * 0.5f;

    int cols = (int)((windowWidth - 2.0f * margin) / sep);
    int rows = 28;
    int n = cols * rows;
    p.resize(n);

    float startX = margin;
    float startY = margin;

    for (int idx = 0; idx < n; idx++) {
        int i = idx % cols;
        int j = idx / cols;
        float ranX = (float(std::rand()) / float(RAND_MAX) * 2.0f - 1.0f) * s.h * 0.05f;
        float ranY = (float(std::rand()) / float(RAND_MAX) * 2.0f - 1.0f) * s.h * 0.05f;
        p.pos_x[idx] = startX + i * sep + ranX;
        p.pos_y[idx] = startY + j * sep + ranY;
        p.vel_x[idx] = 0.0f;
        p.vel_y[idx] = 0.0f;
        p.force_x[idx] = 0.0f;
        p.force_y[idx] = 0.0f;
        p.density[idx] = 0.0f;
        p.pressure[idx] = 0.0f;
    }
}

// ============================================================
// STEP 1: Density — NEON vectorized inner loop
//   For each i, process j in chunks of 4
// ============================================================
void compute_density_simd(ParticlesSoA& p, const SPHSettings& s) {
    const int n = p.n;
    const float* px = p.pos_x.data();
    const float* py = p.pos_y.data();
    float* dens = p.density.data();
    float* pres = p.pressure.data();

    const float32x4_t vh2 = vdupq_n_f32(s.h2);
    const float32x4_t vmassPoly6 = vdupq_n_f32(s.massPoly6);
    const float32x4_t vzero = vdupq_n_f32(0.0f);

    for (int i = 0; i < n; i++) {
        float32x4_t vxi = vdupq_n_f32(px[i]);
        float32x4_t vyi = vdupq_n_f32(py[i]);
        float32x4_t vsum = vzero;

        int j = 0;
        for (; j + 4 <= n; j += 4) {
            float32x4_t vxj = vld1q_f32(px + j);
            float32x4_t vyj = vld1q_f32(py + j);

            float32x4_t rx = vsubq_f32(vxj, vxi);
            float32x4_t ry = vsubq_f32(vyj, vyi);
            float32x4_t r2 = vmlaq_f32(vmulq_f32(rx, rx), ry, ry);  // rx*rx + ry*ry

            // mask: r2 < h2
            uint32x4_t mask = vcltq_f32(r2, vh2);

            // diff = h2 - r2
            float32x4_t diff = vsubq_f32(vh2, r2);
            // diff^3 = diff * diff * diff
            float32x4_t d2 = vmulq_f32(diff, diff);
            float32x4_t d3 = vmulq_f32(d2, diff);
            // contrib = massPoly6 * diff^3
            float32x4_t contrib = vmulq_f32(vmassPoly6, d3);
            // Zero out contributions where r2 >= h2
            contrib = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(contrib), mask));

            vsum = vaddq_f32(vsum, contrib);
        }

        // Horizontal sum of 4 lanes
        float sum = vaddvq_f32(vsum);

        // Scalar remainder
        for (; j < n; j++) {
            float rx = px[j] - px[i];
            float ry = py[j] - py[i];
            float r2 = rx * rx + ry * ry;
            if (r2 < s.h2) {
                float diff = s.h2 - r2;
                sum += s.massPoly6 * diff * diff * diff;
            }
        }

        dens[i] = sum;
        pres[i] = s.gasConstant * (sum - s.restDensity);
    }

    // Clamp minimum density
    float minDens = s.restDensity * 0.1f;
    for (int i = 0; i < n; i++) {
        if (dens[i] < minDens) dens[i] = minDens;
    }
}

// ============================================================
// STEP 2: Force accumulation — NEON vectorized inner loop
//   Pressure + viscosity + gravity
// ============================================================
void compute_forces_simd(ParticlesSoA& p, const SPHSettings& s) {
    const int n = p.n;
    const float* px = p.pos_x.data();
    const float* py = p.pos_y.data();
    const float* vx = p.vel_x.data();
    const float* vy = p.vel_y.data();
    const float* dens = p.density.data();
    const float* pres = p.pressure.data();
    float* fx = p.force_x.data();
    float* fy = p.force_y.data();

    const float32x4_t vh2 = vdupq_n_f32(s.h2);
    const float32x4_t vh = vdupq_n_f32(s.h);
    const float32x4_t veps = vdupq_n_f32(0.0001f);
    const float32x4_t vzero = vdupq_n_f32(0.0f);
    const float32x4_t vSpikyGrad = vdupq_n_f32(s.spikyGrad);
    const float32x4_t vSpikyLap = vdupq_n_f32(s.spikyLap);
    const float32x4_t vMass = vdupq_n_f32(s.mass);
    const float32x4_t vVisc = vdupq_n_f32(s.viscosity);
    const float32x4_t vNegMass = vdupq_n_f32(-s.mass);

    for (int i = 0; i < n; i++) {
        float32x4_t vxi = vdupq_n_f32(px[i]);
        float32x4_t vyi = vdupq_n_f32(py[i]);
        float32x4_t vvxi = vdupq_n_f32(vx[i]);
        float32x4_t vvyi = vdupq_n_f32(vy[i]);

        float pi_pres = pres[i];
        float pi_dens = dens[i];
        float pi_pOverD2 = pi_pres / (pi_dens * pi_dens);
        float32x4_t vPiPD2 = vdupq_n_f32(pi_pOverD2);

        float32x4_t sumFx = vzero;
        float32x4_t sumFy = vzero;

        int j = 0;
        for (; j + 4 <= n; j += 4) {
            float32x4_t vxj = vld1q_f32(px + j);
            float32x4_t vyj = vld1q_f32(py + j);

            float32x4_t rx = vsubq_f32(vxj, vxi);
            float32x4_t ry = vsubq_f32(vyj, vyi);
            float32x4_t r2 = vmlaq_f32(vmulq_f32(rx, rx), ry, ry);

            // mask: r2 < h2 AND r2 > eps
            uint32x4_t maskIn = vcltq_f32(r2, vh2);
            uint32x4_t maskPos = vcgtq_f32(r2, veps);
            uint32x4_t mask = vandq_u32(maskIn, maskPos);

            // r_length = sqrt(r2), safe (masked later)
            float32x4_t rLen = vsqrtq_f32(r2);
            // Avoid division by zero: clamp rLen to at least eps
            rLen = vmaxq_f32(rLen, veps);

            // --- Pressure force ---
            // spikyGrad coefficient: spikyGrad * (h - rLen)^2 / rLen
            float32x4_t hMinusR = vsubq_f32(vh, rLen);
            float32x4_t hMinusR2 = vmulq_f32(hMinusR, hMinusR);
            // pressure gradient coefficient per-particle
            float32x4_t gradCoeff = vmulq_f32(vSpikyGrad, hMinusR2);
            // Divide by rLen to get directional scaling
            float32x4_t invRLen = vrecpeq_f32(rLen);
            invRLen = vmulq_f32(invRLen, vrecpsq_f32(rLen, invRLen));  // Newton-Raphson step
            gradCoeff = vmulq_f32(gradCoeff, invRLen);

            // gradX = gradCoeff * rx, gradY = gradCoeff * ry
            float32x4_t gradX = vmulq_f32(gradCoeff, rx);
            float32x4_t gradY = vmulq_f32(gradCoeff, ry);

            // pressureTerm = P_i/(rho_i^2) + P_j/(rho_j^2)
            float32x4_t vPj = vld1q_f32(pres + j);
            float32x4_t vDj = vld1q_f32(dens + j);
            float32x4_t vDj2 = vmulq_f32(vDj, vDj);
            float32x4_t invDj2 = vrecpeq_f32(vDj2);
            invDj2 = vmulq_f32(invDj2, vrecpsq_f32(vDj2, invDj2));
            float32x4_t pjOverDj2 = vmulq_f32(vPj, invDj2);
            float32x4_t pressTerm = vaddq_f32(vPiPD2, pjOverDj2);

            // F_pressure contribution: -mass * pressTerm * grad
            float32x4_t pfx = vmulq_f32(vNegMass, vmulq_f32(pressTerm, gradX));
            float32x4_t pfy = vmulq_f32(vNegMass, vmulq_f32(pressTerm, gradY));

            // --- Viscosity force ---
            // laplacian = spikyLap * (h - rLen)
            float32x4_t lap = vmulq_f32(vSpikyLap, hMinusR);

            // velocity diff
            float32x4_t vvxj = vld1q_f32(vx + j);
            float32x4_t vvyj = vld1q_f32(vy + j);
            float32x4_t dvx = vsubq_f32(vvxj, vvxi);
            float32x4_t dvy = vsubq_f32(vvyj, vvyi);

            // viscosity * mass * (v_j - v_i) / density_j * laplacian
            float32x4_t invDj = vrecpeq_f32(vDj);
            invDj = vmulq_f32(invDj, vrecpsq_f32(vDj, invDj));
            float32x4_t viscCoeff = vmulq_f32(vmulq_f32(vVisc, vMass), vmulq_f32(lap, invDj));
            float32x4_t vfx = vmulq_f32(viscCoeff, dvx);
            float32x4_t vfy = vmulq_f32(viscCoeff, dvy);

            // Combined and masked
            float32x4_t totalFx = vaddq_f32(pfx, vfx);
            float32x4_t totalFy = vaddq_f32(pfy, vfy);
            totalFx = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(totalFx), mask));
            totalFy = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(totalFy), mask));

            sumFx = vaddq_f32(sumFx, totalFx);
            sumFy = vaddq_f32(sumFy, totalFy);
        }

        // Horizontal sum
        float sfx = vaddvq_f32(sumFx);
        float sfy = vaddvq_f32(sumFy);

        // Scalar remainder
        for (; j < n; j++) {
            if (j == i) continue;
            float drx = px[j] - px[i];
            float dry = py[j] - py[i];
            float r2 = drx * drx + dry * dry;
            if (r2 < s.h2 && r2 > 0.0001f) {
                float rLen = std::sqrt(r2);
                if (rLen >= s.h) continue;

                float diff = s.h - rLen;
                float gradCoeff = s.spikyGrad * diff * diff / rLen;
                float gx = gradCoeff * drx;
                float gy = gradCoeff * dry;

                float pTerm = pi_pOverD2 + pres[j] / (dens[j] * dens[j]);
                sfx += -s.mass * pTerm * gx;
                sfy += -s.mass * pTerm * gy;

                float lap = s.spikyLap * diff;
                float viscCoeff = s.viscosity * s.mass * lap / dens[j];
                sfx += viscCoeff * (vx[j] - vx[i]);
                sfy += viscCoeff * (vy[j] - vy[i]);
            }
        }

        // Add gravity
        fx[i] = sfx;
        fy[i] = sfy + s.g;
    }
}

// ============================================================
// STEP 3: Integration + boundary handling
// ============================================================
void integrate_and_boundary(ParticlesSoA& p, const SPHSettings& s,
                            float dt, float width, float height) {
    const float damping = 0.5f;
    const float wallMin = 10.0f;
    const float wallMaxX = width - 10.0f;
    const float wallMaxY = height - 10.0f;

    for (int i = 0; i < p.n; i++) {
        if (p.density[i] > 0.0001f) {
            float invDens = 1.0f / p.density[i];
            p.vel_x[i] += p.force_x[i] * invDens * dt;
            p.vel_y[i] += p.force_y[i] * invDens * dt;
            p.pos_x[i] += p.vel_x[i] * dt;
            p.pos_y[i] += p.vel_y[i] * dt;
        }

        if (p.pos_y[i] > wallMaxY) { p.pos_y[i] = wallMaxY; p.vel_y[i] *= -damping; }
        if (p.pos_y[i] < wallMin)   { p.pos_y[i] = wallMin;  p.vel_y[i] *= -damping; }
        if (p.pos_x[i] < wallMin)   { p.pos_x[i] = wallMin;  p.vel_x[i] *= -damping; }
        if (p.pos_x[i] > wallMaxX)  { p.pos_x[i] = wallMaxX; p.vel_x[i] *= -damping; }
    }
}

// ============================================================
// Main
// ============================================================
int main() {
    const int WINDOW_WIDTH = 800;
    const int WINDOW_HEIGHT = 600;

    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "SPH Phase 3 - NEON SIMD");
    SetTargetFPS(60);

    SPHSettings settings(
        1.0f,      // mass
        1.0f,      // restDensity
        2000.0f,   // gasConstant
        250.0f,    // viscosity
        16.0f,     // h
        980.0f     // gravity
    );

    ParticlesSoA particles;
    initialize_dam_break(particles, settings, (float)WINDOW_WIDTH, (float)WINDOW_HEIGHT);
    int numParticles = particles.n;

    const float dt = 0.003f;

    int frameCount = 0;
    float fpsSum = 0.0f;
    int fpsSamples = 0;

    while (!WindowShouldClose()) {
        compute_density_simd(particles, settings);
        compute_forces_simd(particles, settings);
        integrate_and_boundary(particles, settings, dt, (float)WINDOW_WIDTH, (float)WINDOW_HEIGHT);

        BeginDrawing();
        ClearBackground(WHITE);

        for (int i = 0; i < numParticles; i++) {
            DrawCircle((int)particles.pos_x[i], (int)particles.pos_y[i], 2, PURPLE);
        }

        float fps = GetFPS();
        if (fps > 0.0f) { fpsSum += fps; fpsSamples++; }
        frameCount++;
        DrawFPS(10, 10);
        DrawText(TextFormat("Particles: %d  [SIMD]", numParticles), 10, 30, 20, BLACK);
        DrawText(TextFormat("Frames: %d", frameCount), 10, 50, 20, BLACK);

        EndDrawing();
    }

    // ----- Session log -----
    {
        system("mkdir -p benchmarks");
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        std::tm* tm = std::localtime(&t);
        std::ostringstream fname;
        fname << "benchmarks/sph_simd_" << std::put_time(tm, "%Y%m%d_%H%M%S") << ".log";

        float minX = 1e9f, maxX = -1e9f, minY = 1e9f, maxY = -1e9f;
        for (int i = 0; i < numParticles; i++) {
            if (particles.pos_x[i] < minX) minX = particles.pos_x[i];
            if (particles.pos_x[i] > maxX) maxX = particles.pos_x[i];
            if (particles.pos_y[i] < minY) minY = particles.pos_y[i];
            if (particles.pos_y[i] > maxY) maxY = particles.pos_y[i];
        }
        float avgFps = (fpsSamples > 0) ? (fpsSum / fpsSamples) : 0.0f;

        std::ofstream log(fname.str());
        if (log.is_open()) {
            log << "SPH Phase 3 - NEON SIMD Session Log\n";
            log << "====================================\n";
            log << "Timestamp: " << std::put_time(tm, "%Y-%m-%d %H:%M:%S") << "\n\n";
            log << "Config:\n";
            log << "  particles  = " << numParticles << "\n";
            log << "  h         = " << settings.h << "\n";
            log << "  restDensity = " << settings.restDensity << "\n";
            log << "  gasConstant = " << settings.gasConstant << "\n";
            log << "  viscosity = " << settings.viscosity << "\n";
            log << "  gravity   = " << settings.g << "\n";
            log << "  dt        = " << dt << "\n\n";
            log << "Run:\n";
            log << "  total_frames = " << frameCount << "\n";
            log << "  avg_fps      = " << std::fixed << std::setprecision(2) << avgFps << "\n\n";
            log << "Final particle bounds:\n";
            log << "  x_min = " << minX << ", x_max = " << maxX << "\n";
            log << "  y_min = " << minY << ", y_max = " << maxY << "\n";
            log.close();
        }
    }

    CloseWindow();
    return 0;
}
