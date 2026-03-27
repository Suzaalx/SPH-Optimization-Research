#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
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
// 3D SPH — Optimized with NEON SIMD + fast GL point rendering
// - Rotatable cube container (arrow keys)
// - Orbiting camera (mouse)
// - ~2000+ particles rendered as GL points (not spheres)
// ============================================================

struct SPHSettings {
    float mass;
    float restDensity;
    float gasConstant;
    float viscosity;
    float h, h2, g;
    float poly6, spikyGrad, spikyLap, massPoly6;

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

struct Particles3D {
    std::vector<float> pos_x, pos_y, pos_z;
    std::vector<float> vel_x, vel_y, vel_z;
    std::vector<float> force_x, force_y, force_z;
    std::vector<float> density, pressure;
    int n = 0;

    void resize(int count) {
        n = count;
        pos_x.resize(n); pos_y.resize(n); pos_z.resize(n);
        vel_x.resize(n); vel_y.resize(n); vel_z.resize(n);
        force_x.resize(n); force_y.resize(n); force_z.resize(n);
        density.resize(n); pressure.resize(n);
    }
};

struct CubeContainer {
    float halfSize;
    Vector3 center;
    float rotX, rotZ;

    Matrix getRotation() const {
        return MatrixMultiply(MatrixRotateX(rotX), MatrixRotateZ(rotZ));
    }
};

// Fill bottom half of cube with particles in a grid
void initialize_particles(Particles3D& p, const SPHSettings& s, const CubeContainer& cube) {
    std::srand(1024);

    float sep = s.h * 0.5f;
    float hs = cube.halfSize;

    // Fill a block: full x/z width, bottom half of y
    int nx = (int)(2.0f * hs * 0.9f / sep);
    int nz = nx;
    int ny = (int)(hs * 0.9f / sep);  // bottom half
    if (nx < 2) nx = 2;
    if (ny < 2) ny = 2;
    if (nz < 2) nz = 2;

    int total = nx * ny * nz;
    p.resize(total);

    float ox = -hs * 0.9f;
    float oy = -hs * 0.9f;  // start from bottom
    float oz = -hs * 0.9f;

    int idx = 0;
    for (int ix = 0; ix < nx && idx < total; ix++) {
        for (int iy = 0; iy < ny && idx < total; iy++) {
            for (int iz = 0; iz < nz && idx < total; iz++) {
                float jx = (float(std::rand()) / RAND_MAX * 2.0f - 1.0f) * s.h * 0.02f;
                float jy = (float(std::rand()) / RAND_MAX * 2.0f - 1.0f) * s.h * 0.02f;
                float jz = (float(std::rand()) / RAND_MAX * 2.0f - 1.0f) * s.h * 0.02f;

                p.pos_x[idx] = cube.center.x + ox + ix * sep + jx;
                p.pos_y[idx] = cube.center.y + oy + iy * sep + jy;
                p.pos_z[idx] = cube.center.z + oz + iz * sep + jz;
                p.vel_x[idx] = p.vel_y[idx] = p.vel_z[idx] = 0.0f;
                p.force_x[idx] = p.force_y[idx] = p.force_z[idx] = 0.0f;
                p.density[idx] = 0.0f;
                p.pressure[idx] = 0.0f;
                idx++;
            }
        }
    }
    p.n = idx;
}

// ============================================================
// Density — NEON vectorized (3D)
// ============================================================
void compute_density_simd(Particles3D& p, const SPHSettings& s) {
    const int n = p.n;
    const float* px = p.pos_x.data();
    const float* py = p.pos_y.data();
    const float* pz = p.pos_z.data();
    float* dens = p.density.data();
    float* pres = p.pressure.data();

    const float32x4_t vh2 = vdupq_n_f32(s.h2);
    const float32x4_t vmp = vdupq_n_f32(s.massPoly6);
    const float32x4_t vzero = vdupq_n_f32(0.0f);

    for (int i = 0; i < n; i++) {
        float32x4_t vxi = vdupq_n_f32(px[i]);
        float32x4_t vyi = vdupq_n_f32(py[i]);
        float32x4_t vzi = vdupq_n_f32(pz[i]);
        float32x4_t vsum = vzero;

        int j = 0;
        for (; j + 4 <= n; j += 4) {
            float32x4_t rx = vsubq_f32(vld1q_f32(px + j), vxi);
            float32x4_t ry = vsubq_f32(vld1q_f32(py + j), vyi);
            float32x4_t rz = vsubq_f32(vld1q_f32(pz + j), vzi);
            float32x4_t r2 = vmlaq_f32(vmlaq_f32(vmulq_f32(rx, rx), ry, ry), rz, rz);

            uint32x4_t mask = vcltq_f32(r2, vh2);
            float32x4_t diff = vsubq_f32(vh2, r2);
            float32x4_t d3 = vmulq_f32(vmulq_f32(diff, diff), diff);
            float32x4_t c = vmulq_f32(vmp, d3);
            c = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(c), mask));
            vsum = vaddq_f32(vsum, c);
        }

        float sum = vaddvq_f32(vsum);
        for (; j < n; j++) {
            float dx = px[j] - px[i], dy = py[j] - py[i], dz = pz[j] - pz[i];
            float r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < s.h2) { float d = s.h2 - r2; sum += s.massPoly6 * d * d * d; }
        }

        dens[i] = sum;
        pres[i] = s.gasConstant * (sum - s.restDensity);
    }

    float minD = s.restDensity * 0.1f;
    for (int i = 0; i < n; i++) if (dens[i] < minD) dens[i] = minD;
}

// ============================================================
// Forces — NEON vectorized (3D)
// ============================================================
void compute_forces_simd(Particles3D& p, const SPHSettings& s,
                         float gx, float gy, float gz) {
    const int n = p.n;
    const float* px = p.pos_x.data(); const float* py = p.pos_y.data(); const float* pz = p.pos_z.data();
    const float* vvx = p.vel_x.data(); const float* vvy = p.vel_y.data(); const float* vvz = p.vel_z.data();
    const float* dens = p.density.data(); const float* pres = p.pressure.data();
    float* fx = p.force_x.data(); float* fy = p.force_y.data(); float* fz = p.force_z.data();

    const float32x4_t vh2 = vdupq_n_f32(s.h2);
    const float32x4_t vh = vdupq_n_f32(s.h);
    const float32x4_t veps = vdupq_n_f32(0.0001f);
    const float32x4_t vzero = vdupq_n_f32(0.0f);
    const float32x4_t vSG = vdupq_n_f32(s.spikyGrad);
    const float32x4_t vSL = vdupq_n_f32(s.spikyLap);
    const float32x4_t vM = vdupq_n_f32(s.mass);
    const float32x4_t vNM = vdupq_n_f32(-s.mass);
    const float32x4_t vV = vdupq_n_f32(s.viscosity);

    for (int i = 0; i < n; i++) {
        float32x4_t vxi = vdupq_n_f32(px[i]);
        float32x4_t vyi = vdupq_n_f32(py[i]);
        float32x4_t vzi = vdupq_n_f32(pz[i]);
        float32x4_t vvxi = vdupq_n_f32(vvx[i]);
        float32x4_t vvyi = vdupq_n_f32(vvy[i]);
        float32x4_t vvzi = vdupq_n_f32(vvz[i]);

        float piPD2 = pres[i] / (dens[i] * dens[i]);
        float32x4_t vPiPD2 = vdupq_n_f32(piPD2);

        float32x4_t sFx = vzero, sFy = vzero, sFz = vzero;

        int j = 0;
        for (; j + 4 <= n; j += 4) {
            float32x4_t rx = vsubq_f32(vld1q_f32(px + j), vxi);
            float32x4_t ry = vsubq_f32(vld1q_f32(py + j), vyi);
            float32x4_t rz = vsubq_f32(vld1q_f32(pz + j), vzi);
            float32x4_t r2 = vmlaq_f32(vmlaq_f32(vmulq_f32(rx, rx), ry, ry), rz, rz);

            uint32x4_t mask = vandq_u32(vcltq_f32(r2, vh2), vcgtq_f32(r2, veps));

            float32x4_t rLen = vmaxq_f32(vsqrtq_f32(r2), veps);
            float32x4_t hMr = vsubq_f32(vh, rLen);
            float32x4_t hMr2 = vmulq_f32(hMr, hMr);

            // Pressure gradient coefficient
            float32x4_t invR = vrecpeq_f32(rLen);
            invR = vmulq_f32(invR, vrecpsq_f32(rLen, invR));
            float32x4_t gC = vmulq_f32(vmulq_f32(vSG, hMr2), invR);
            float32x4_t grX = vmulq_f32(gC, rx);
            float32x4_t grY = vmulq_f32(gC, ry);
            float32x4_t grZ = vmulq_f32(gC, rz);

            // Pressure term
            float32x4_t vDj = vld1q_f32(dens + j);
            float32x4_t vDj2 = vmulq_f32(vDj, vDj);
            float32x4_t iDj2 = vrecpeq_f32(vDj2);
            iDj2 = vmulq_f32(iDj2, vrecpsq_f32(vDj2, iDj2));
            float32x4_t pT = vaddq_f32(vPiPD2, vmulq_f32(vld1q_f32(pres + j), iDj2));

            float32x4_t pfx = vmulq_f32(vNM, vmulq_f32(pT, grX));
            float32x4_t pfy = vmulq_f32(vNM, vmulq_f32(pT, grY));
            float32x4_t pfz = vmulq_f32(vNM, vmulq_f32(pT, grZ));

            // Viscosity
            float32x4_t lap = vmulq_f32(vSL, hMr);
            float32x4_t iDj = vrecpeq_f32(vDj);
            iDj = vmulq_f32(iDj, vrecpsq_f32(vDj, iDj));
            float32x4_t vC = vmulq_f32(vmulq_f32(vV, vM), vmulq_f32(lap, iDj));

            float32x4_t vfx = vmulq_f32(vC, vsubq_f32(vld1q_f32(vvx + j), vvxi));
            float32x4_t vfy = vmulq_f32(vC, vsubq_f32(vld1q_f32(vvy + j), vvyi));
            float32x4_t vfz = vmulq_f32(vC, vsubq_f32(vld1q_f32(vvz + j), vvzi));

            float32x4_t tx = vaddq_f32(pfx, vfx);
            float32x4_t ty = vaddq_f32(pfy, vfy);
            float32x4_t tz = vaddq_f32(pfz, vfz);
            tx = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(tx), mask));
            ty = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ty), mask));
            tz = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(tz), mask));

            sFx = vaddq_f32(sFx, tx);
            sFy = vaddq_f32(sFy, ty);
            sFz = vaddq_f32(sFz, tz);
        }

        float sfx = vaddvq_f32(sFx), sfy = vaddvq_f32(sFy), sfz = vaddvq_f32(sFz);

        // Scalar remainder
        for (; j < n; j++) {
            if (j == i) continue;
            float dx = px[j]-px[i], dy = py[j]-py[i], dz = pz[j]-pz[i];
            float r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < s.h2 && r2 > 0.0001f) {
                float rL = std::sqrt(r2);
                if (rL >= s.h) continue;
                float d = s.h - rL;
                float gC = s.spikyGrad * d * d / rL;
                float pT = piPD2 + pres[j]/(dens[j]*dens[j]);
                sfx += -s.mass * pT * gC * dx;
                sfy += -s.mass * pT * gC * dy;
                sfz += -s.mass * pT * gC * dz;
                float lap = s.spikyLap * d;
                float vC = s.viscosity * s.mass * lap / dens[j];
                sfx += vC * (vvx[j]-vvx[i]);
                sfy += vC * (vvy[j]-vvy[i]);
                sfz += vC * (vvz[j]-vvz[i]);
            }
        }

        fx[i] = sfx + gx;
        fy[i] = sfy + gy;
        fz[i] = sfz + gz;
    }
}

// Integration + rotated cube boundary
void integrate(Particles3D& p, const SPHSettings& s, float dt,
               const CubeContainer& cube) {
    const float damping = 0.5f;

    Matrix rot = cube.getRotation();
    Vector3 normals[6];
    float offsets[6];
    Vector3 axes[3] = {{ 1,0,0 }, { 0,1,0 }, { 0,0,1 }};
    for (int a = 0; a < 3; a++) {
        Vector3 n = Vector3Transform(axes[a], rot);
        normals[a*2]   = n;
        normals[a*2+1] = { -n.x, -n.y, -n.z };
        offsets[a*2]   = cube.halfSize;
        offsets[a*2+1] = cube.halfSize;
    }

    for (int i = 0; i < p.n; i++) {
        if (p.density[i] > 0.0001f) {
            float invD = 1.0f / p.density[i];
            p.vel_x[i] += p.force_x[i] * invD * dt;
            p.vel_y[i] += p.force_y[i] * invD * dt;
            p.vel_z[i] += p.force_z[i] * invD * dt;
            p.pos_x[i] += p.vel_x[i] * dt;
            p.pos_y[i] += p.vel_y[i] * dt;
            p.pos_z[i] += p.vel_z[i] * dt;
        }

        float rx = p.pos_x[i] - cube.center.x;
        float ry = p.pos_y[i] - cube.center.y;
        float rz = p.pos_z[i] - cube.center.z;

        for (int f = 0; f < 6; f++) {
            Vector3 n = normals[f];
            float dist = rx*n.x + ry*n.y + rz*n.z;
            if (dist > offsets[f]) {
                float pen = dist - offsets[f];
                p.pos_x[i] -= pen * n.x;
                p.pos_y[i] -= pen * n.y;
                p.pos_z[i] -= pen * n.z;
                rx = p.pos_x[i] - cube.center.x;
                ry = p.pos_y[i] - cube.center.y;
                rz = p.pos_z[i] - cube.center.z;
                float vn = p.vel_x[i]*n.x + p.vel_y[i]*n.y + p.vel_z[i]*n.z;
                if (vn > 0.0f) {
                    p.vel_x[i] -= (1.0f + damping) * vn * n.x;
                    p.vel_y[i] -= (1.0f + damping) * vn * n.y;
                    p.vel_z[i] -= (1.0f + damping) * vn * n.z;
                }
            }
        }
    }
}

void draw_cube_wireframe(const CubeContainer& cube) {
    rlPushMatrix();
    rlTranslatef(cube.center.x, cube.center.y, cube.center.z);
    rlRotatef(cube.rotX * RAD2DEG, 1, 0, 0);
    rlRotatef(cube.rotZ * RAD2DEG, 0, 0, 1);
    DrawCubeWires({ 0, 0, 0 }, cube.halfSize*2, cube.halfSize*2, cube.halfSize*2, DARKGRAY);
    rlPopMatrix();
}

// Fast particle rendering using GL points (orders of magnitude faster than DrawSphere)
void draw_particles_points(const Particles3D& p, float pointSize, Color color) {
    rlSetLineWidth(pointSize);
    rlBegin(RL_LINES);
    for (int i = 0; i < p.n; i++) {
        rlColor4ub(color.r, color.g, color.b, color.a);
        rlVertex3f(p.pos_x[i], p.pos_y[i], p.pos_z[i]);
        // Draw a tiny line segment so the point is visible
        rlVertex3f(p.pos_x[i] + 0.02f, p.pos_y[i] + 0.02f, p.pos_z[i]);
    }
    rlEnd();
}

int main() {
    const int WINDOW_WIDTH = 1000;
    const int WINDOW_HEIGHT = 750;

    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "SPH 3D - SIMD Optimized");
    SetTargetFPS(60);
    DisableCursor();

    SPHSettings settings(
        1.0f,      // mass
        300.0f,    // restDensity
        500.0f,    // gasConstant
        200.0f,    // viscosity
        0.8f,      // h (smaller = more particles fit, fewer neighbors)
        9.8f       // gravity
    );

    CubeContainer cube;
    cube.halfSize = 3.5f;
    cube.center = { 0.0f, 0.0f, 0.0f };
    cube.rotX = 0.0f;
    cube.rotZ = 0.0f;

    Particles3D particles;
    initialize_particles(particles, settings, cube);
    int numParticles = particles.n;

    const float dt = 0.003f;

    Camera3D camera = { 0 };
    camera.position = { 10.0f, 7.0f, 10.0f };
    camera.target = { 0.0f, 0.0f, 0.0f };
    camera.up = { 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    int frameCount = 0;
    float fpsSum = 0.0f;
    int fpsSamples = 0;

    while (!WindowShouldClose()) {
        UpdateCamera(&camera, CAMERA_THIRD_PERSON);

        float rotSpeed = 1.5f * GetFrameTime();
        if (IsKeyDown(KEY_RIGHT)) cube.rotZ -= rotSpeed;
        if (IsKeyDown(KEY_LEFT))  cube.rotZ += rotSpeed;
        if (IsKeyDown(KEY_UP))    cube.rotX -= rotSpeed;
        if (IsKeyDown(KEY_DOWN))  cube.rotX += rotSpeed;
        if (IsKeyPressed(KEY_R))  { cube.rotX = 0; cube.rotZ = 0; }

        float gx = 0.0f, gy = -settings.g, gz = 0.0f;

        compute_density_simd(particles, settings);
        compute_forces_simd(particles, settings, gx, gy, gz);
        integrate(particles, settings, dt, cube);

        BeginDrawing();
        ClearBackground(RAYWHITE);
        BeginMode3D(camera);

        draw_cube_wireframe(cube);
        draw_particles_points(particles, 3.0f, (Color){ 100, 20, 200, 255 });
        DrawGrid(20, 1.0f);

        EndMode3D();

        float fps = GetFPS();
        if (fps > 0.0f) { fpsSum += fps; fpsSamples++; }
        frameCount++;

        DrawFPS(10, 10);
        DrawText(TextFormat("Particles: %d  [3D SIMD]", numParticles), 10, 30, 20, DARKGRAY);
        DrawText(TextFormat("Frames: %d", frameCount), 10, 50, 20, DARKGRAY);
        DrawText("Arrow keys: tilt cube  |  R: reset  |  Mouse: orbit", 10, WINDOW_HEIGHT - 25, 16, GRAY);

        EndDrawing();
    }

    // ----- Session log -----
    {
        system("mkdir -p benchmarks");
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        std::tm* tm = std::localtime(&t);
        std::ostringstream fname;
        fname << "benchmarks/sph_3d_" << std::put_time(tm, "%Y%m%d_%H%M%S") << ".log";

        float minX=1e9f,maxX=-1e9f,minY=1e9f,maxY=-1e9f,minZ=1e9f,maxZ=-1e9f;
        for (int i = 0; i < numParticles; i++) {
            if (particles.pos_x[i]<minX) minX=particles.pos_x[i];
            if (particles.pos_x[i]>maxX) maxX=particles.pos_x[i];
            if (particles.pos_y[i]<minY) minY=particles.pos_y[i];
            if (particles.pos_y[i]>maxY) maxY=particles.pos_y[i];
            if (particles.pos_z[i]<minZ) minZ=particles.pos_z[i];
            if (particles.pos_z[i]>maxZ) maxZ=particles.pos_z[i];
        }
        float avgFps = (fpsSamples > 0) ? (fpsSum / fpsSamples) : 0.0f;

        std::ofstream log(fname.str());
        if (log.is_open()) {
            log << "SPH 3D - SIMD Optimized Session Log\n";
            log << "====================================\n";
            log << "Timestamp: " << std::put_time(tm, "%Y-%m-%d %H:%M:%S") << "\n\n";
            log << "Config:\n";
            log << "  particles   = " << numParticles << "\n";
            log << "  h           = " << settings.h << "\n";
            log << "  restDensity = " << settings.restDensity << "\n";
            log << "  gasConstant = " << settings.gasConstant << "\n";
            log << "  viscosity   = " << settings.viscosity << "\n";
            log << "  gravity     = " << settings.g << "\n";
            log << "  dt          = " << dt << "\n";
            log << "  cubeHalfSize = " << cube.halfSize << "\n";
            log << "  cubeRotX    = " << cube.rotX << "\n";
            log << "  cubeRotZ    = " << cube.rotZ << "\n\n";
            log << "Run:\n";
            log << "  total_frames = " << frameCount << "\n";
            log << "  avg_fps      = " << std::fixed << std::setprecision(2) << avgFps << "\n\n";
            log << "Final particle bounds:\n";
            log << "  x: " << minX << " .. " << maxX << "\n";
            log << "  y: " << minY << " .. " << maxY << "\n";
            log << "  z: " << minZ << " .. " << maxZ << "\n";
            log.close();
        }
    }

    EnableCursor();
    CloseWindow();
    return 0;
}
