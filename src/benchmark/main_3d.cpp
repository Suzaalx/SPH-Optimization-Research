#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <iomanip>

// ============================================================
// 3D SPH Simulation with rotatable cube container
// - Full 3D physics (vec3 positions, velocities, forces)
// - Orbiting camera with mouse control
// - Rotatable cube: arrow keys tilt the container
// - Same SPH kernels (already 3D-normalized)
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
    float massPoly6;

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

// 3D SoA particle data
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

// Cube container: 6 planes defined by normals and center offsets
struct CubeContainer {
    float halfSize;       // half-width of cube
    Vector3 center;       // cube center in world space
    float rotX, rotZ;     // rotation angles (pitch, roll) in radians

    Matrix getRotation() const {
        return MatrixMultiply(MatrixRotateX(rotX), MatrixRotateZ(rotZ));
    }
};

// Initialize particles in a 3D block inside the cube
void initialize_particles(Particles3D& p, const SPHSettings& s, const CubeContainer& cube) {
    std::srand(1024);

    float sep = s.h * 0.55f;
    float extent = cube.halfSize * 0.8f; // fill 80% of cube
    int dim = (int)(2.0f * extent / sep);
    if (dim < 2) dim = 2;
    int total = dim * dim * dim;
    p.resize(total);

    float origin = -extent;

    int idx = 0;
    for (int ix = 0; ix < dim && idx < total; ix++) {
        for (int iy = 0; iy < dim && idx < total; iy++) {
            for (int iz = 0; iz < dim && idx < total; iz++) {
                float jx = (float(std::rand()) / RAND_MAX * 2.0f - 1.0f) * s.h * 0.03f;
                float jy = (float(std::rand()) / RAND_MAX * 2.0f - 1.0f) * s.h * 0.03f;
                float jz = (float(std::rand()) / RAND_MAX * 2.0f - 1.0f) * s.h * 0.03f;

                p.pos_x[idx] = cube.center.x + origin + ix * sep + jx;
                p.pos_y[idx] = cube.center.y + origin + iy * sep + jy;
                p.pos_z[idx] = cube.center.z + origin + iz * sep + jz;
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

// Density calculation (scalar, 3D)
void compute_density(Particles3D& p, const SPHSettings& s) {
    for (int i = 0; i < p.n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < p.n; j++) {
            float dx = p.pos_x[j] - p.pos_x[i];
            float dy = p.pos_y[j] - p.pos_y[i];
            float dz = p.pos_z[j] - p.pos_z[i];
            float r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < s.h2) {
                float diff = s.h2 - r2;
                sum += s.massPoly6 * diff * diff * diff;
            }
        }
        p.density[i] = sum;
        p.pressure[i] = s.gasConstant * (sum - s.restDensity);
    }

    float minDens = s.restDensity * 0.1f;
    for (int i = 0; i < p.n; i++) {
        if (p.density[i] < minDens) p.density[i] = minDens;
    }
}

// Force accumulation (scalar, 3D)
void compute_forces(Particles3D& p, const SPHSettings& s,
                    float gx, float gy, float gz) {
    for (int i = 0; i < p.n; i++) {
        float sfx = 0.0f, sfy = 0.0f, sfz = 0.0f;
        float pi_pOverD2 = p.pressure[i] / (p.density[i] * p.density[i]);

        for (int j = 0; j < p.n; j++) {
            if (j == i) continue;
            float dx = p.pos_x[j] - p.pos_x[i];
            float dy = p.pos_y[j] - p.pos_y[i];
            float dz = p.pos_z[j] - p.pos_z[i];
            float r2 = dx*dx + dy*dy + dz*dz;

            if (r2 < s.h2 && r2 > 0.0001f) {
                float rLen = std::sqrt(r2);
                if (rLen >= s.h) continue;

                float diff = s.h - rLen;

                // Pressure force
                float gradCoeff = s.spikyGrad * diff * diff / rLen;
                float pTerm = pi_pOverD2 + p.pressure[j] / (p.density[j] * p.density[j]);
                sfx += -s.mass * pTerm * gradCoeff * dx;
                sfy += -s.mass * pTerm * gradCoeff * dy;
                sfz += -s.mass * pTerm * gradCoeff * dz;

                // Viscosity force
                float lap = s.spikyLap * diff;
                float viscCoeff = s.viscosity * s.mass * lap / p.density[j];
                sfx += viscCoeff * (p.vel_x[j] - p.vel_x[i]);
                sfy += viscCoeff * (p.vel_y[j] - p.vel_y[i]);
                sfz += viscCoeff * (p.vel_z[j] - p.vel_z[i]);
            }
        }

        p.force_x[i] = sfx + gx;
        p.force_y[i] = sfy + gy;
        p.force_z[i] = sfz + gz;
    }
}

// Integration + rotated cube boundary
void integrate(Particles3D& p, const SPHSettings& s, float dt,
               const CubeContainer& cube) {
    const float damping = 0.5f;

    // 6 face normals and plane offsets in world space
    Matrix rot = cube.getRotation();
    Vector3 normals[6];
    float offsets[6]; // signed distance from cube center along normal

    Vector3 axes[3] = {
        { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }
    };
    for (int a = 0; a < 3; a++) {
        Vector3 n = Vector3Transform(axes[a], rot);
        normals[a * 2]     = n;                                   // +face
        normals[a * 2 + 1] = { -n.x, -n.y, -n.z };              // -face
        offsets[a * 2]     =  cube.halfSize;
        offsets[a * 2 + 1] =  cube.halfSize;
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

        // Boundary: check each face of the rotated cube
        float rx = p.pos_x[i] - cube.center.x;
        float ry = p.pos_y[i] - cube.center.y;
        float rz = p.pos_z[i] - cube.center.z;

        for (int f = 0; f < 6; f++) {
            Vector3 n = normals[f];
            float dist = rx * n.x + ry * n.y + rz * n.z;
            if (dist > offsets[f]) {
                float pen = dist - offsets[f];
                p.pos_x[i] -= pen * n.x;
                p.pos_y[i] -= pen * n.y;
                p.pos_z[i] -= pen * n.z;
                rx = p.pos_x[i] - cube.center.x;
                ry = p.pos_y[i] - cube.center.y;
                rz = p.pos_z[i] - cube.center.z;

                float vn = p.vel_x[i] * n.x + p.vel_y[i] * n.y + p.vel_z[i] * n.z;
                if (vn > 0.0f) {
                    p.vel_x[i] -= (1.0f + damping) * vn * n.x;
                    p.vel_y[i] -= (1.0f + damping) * vn * n.y;
                    p.vel_z[i] -= (1.0f + damping) * vn * n.z;
                }
            }
        }
    }
}

// Draw cube wireframe using rotation
void draw_cube_wireframe(const CubeContainer& cube) {
    rlPushMatrix();
    rlTranslatef(cube.center.x, cube.center.y, cube.center.z);
    rlRotatef(cube.rotX * RAD2DEG, 1, 0, 0);
    rlRotatef(cube.rotZ * RAD2DEG, 0, 0, 1);
    DrawCubeWires({ 0, 0, 0 }, cube.halfSize * 2, cube.halfSize * 2, cube.halfSize * 2, DARKGRAY);
    rlPopMatrix();
}

int main() {
    const int WINDOW_WIDTH = 1000;
    const int WINDOW_HEIGHT = 750;

    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "SPH 3D - Rotatable Cube");
    SetTargetFPS(60);
    DisableCursor();

    // Smaller h for 3D (many more neighbors per particle in 3D)
    SPHSettings settings(
        1.0f,      // mass
        300.0f,    // restDensity (higher for 3D to keep fluid coherent)
        500.0f,    // gasConstant
        200.0f,    // viscosity
        1.0f,      // h (world units, not pixels)
        9.8f       // gravity (world units)
    );

    CubeContainer cube;
    cube.halfSize = 4.0f;
    cube.center = { 0.0f, 0.0f, 0.0f };
    cube.rotX = 0.0f;
    cube.rotZ = 0.0f;

    Particles3D particles;
    initialize_particles(particles, settings, cube);
    int numParticles = particles.n;

    const float dt = 0.003f;

    Camera3D camera = { 0 };
    camera.position = { 12.0f, 8.0f, 12.0f };
    camera.target = { 0.0f, 0.0f, 0.0f };
    camera.up = { 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    int frameCount = 0;
    float fpsSum = 0.0f;
    int fpsSamples = 0;

    while (!WindowShouldClose()) {
        // Camera controls (mouse orbit)
        UpdateCamera(&camera, CAMERA_THIRD_PERSON);

        // Cube rotation controls
        float rotSpeed = 1.0f * GetFrameTime();
        if (IsKeyDown(KEY_RIGHT)) cube.rotZ -= rotSpeed;
        if (IsKeyDown(KEY_LEFT))  cube.rotZ += rotSpeed;
        if (IsKeyDown(KEY_UP))    cube.rotX -= rotSpeed;
        if (IsKeyDown(KEY_DOWN))  cube.rotX += rotSpeed;
        if (IsKeyPressed(KEY_R))  { cube.rotX = 0; cube.rotZ = 0; }

        // Gravity in world space (always points down)
        float gx = 0.0f, gy = -settings.g, gz = 0.0f;

        compute_density(particles, settings);
        compute_forces(particles, settings, gx, gy, gz);
        integrate(particles, settings, dt, cube);

        BeginDrawing();
        ClearBackground(RAYWHITE);

        BeginMode3D(camera);

        // Draw cube wireframe
        draw_cube_wireframe(cube);

        // Draw particles as small spheres
        for (int i = 0; i < numParticles; i++) {
            DrawSphere(
                { particles.pos_x[i], particles.pos_y[i], particles.pos_z[i] },
                settings.h * 0.3f,
                (Color){ 128, 0, 200, 220 }  // translucent purple
            );
        }

        DrawGrid(20, 1.0f);

        EndMode3D();

        // HUD
        float fps = GetFPS();
        if (fps > 0.0f) { fpsSum += fps; fpsSamples++; }
        frameCount++;

        DrawFPS(10, 10);
        DrawText(TextFormat("Particles: %d  [3D]", numParticles), 10, 30, 20, DARKGRAY);
        DrawText(TextFormat("Frames: %d", frameCount), 10, 50, 20, DARKGRAY);
        DrawText("Arrow keys: tilt cube  |  R: reset tilt  |  Mouse: orbit camera", 10, WINDOW_HEIGHT - 25, 16, GRAY);

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

        float minX = 1e9f, maxX = -1e9f;
        float minY = 1e9f, maxY = -1e9f;
        float minZ = 1e9f, maxZ = -1e9f;
        for (int i = 0; i < numParticles; i++) {
            if (particles.pos_x[i] < minX) minX = particles.pos_x[i];
            if (particles.pos_x[i] > maxX) maxX = particles.pos_x[i];
            if (particles.pos_y[i] < minY) minY = particles.pos_y[i];
            if (particles.pos_y[i] > maxY) maxY = particles.pos_y[i];
            if (particles.pos_z[i] < minZ) minZ = particles.pos_z[i];
            if (particles.pos_z[i] > maxZ) maxZ = particles.pos_z[i];
        }
        float avgFps = (fpsSamples > 0) ? (fpsSum / fpsSamples) : 0.0f;

        std::ofstream log(fname.str());
        if (log.is_open()) {
            log << "SPH 3D - Rotatable Cube Session Log\n";
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
