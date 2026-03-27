#include "raylib.h"
#include <glm/glm.hpp>
#include <omp.h>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <iomanip>

// ============================================================
// Phase 2: OpenMP multi-threaded implementation
// - Same AoS layout and physics as main_improved.cpp (baseline)
// - Outer i-loop of density and force O(n²) loops parallelized
// - Per-step timing to measure speedup over single-threaded
// ============================================================

struct Particle {
    glm::vec2 position;
    glm::vec2 velocity;
    glm::vec2 force;
    float density;
    float pressure;
};

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

    SPHSettings(float mass, float restDensity, float gasConst,
                float viscosity, float h, float g)
        : mass(mass), restDensity(restDensity), gasConstant(gasConst),
          viscosity(viscosity), h(h), g(g)
    {
        h2 = h * h;
        poly6 = 315.0f / (64.0f * PI * powf(h, 9));
        spikyGrad = -45.0f / (PI * powf(h, 6));
        spikyLap = 45.0f / (PI * powf(h, 6));
    }
};

void initialize_dam_break(std::vector<Particle>& particles, const SPHSettings& settings,
                          float windowWidth, float windowHeight) {
    std::srand(1024);

    const float margin = 20.0f;
    float particleSeparation = settings.h * 0.5f;

    int cols = (int)((windowWidth - 2.0f * margin) / particleSeparation);
    int rows = 28;
    size_t n = (size_t)cols * (size_t)rows;
    particles.resize(n);

    float startX = margin;
    float startY = margin;

    for (size_t idx = 0; idx < n; idx++) {
        int i = (int)(idx % cols);
        int j = (int)(idx / cols);

        float ranX = (float(std::rand()) / float(RAND_MAX) * 2.0f - 1.0f) * settings.h * 0.05f;
        float ranY = (float(std::rand()) / float(RAND_MAX) * 2.0f - 1.0f) * settings.h * 0.05f;

        particles[idx].position = glm::vec2(
            startX + i * particleSeparation + ranX,
            startY + j * particleSeparation + ranY
        );
        particles[idx].velocity = glm::vec2(0.0f);
        particles[idx].density = 0.0f;
        particles[idx].pressure = 0.0f;
        particles[idx].force = glm::vec2(0.0f);
    }
}

void handle_boundaries(Particle& p, float width, float height) {
    const float damping = 0.5f;
    if (p.position.y > height - 10) { p.position.y = height - 10; p.velocity.y *= -damping; }
    if (p.position.y < 10)          { p.position.y = 10;          p.velocity.y *= -damping; }
    if (p.position.x < 10)          { p.position.x = 10;          p.velocity.x *= -damping; }
    if (p.position.x > width - 10)  { p.position.x = width - 10;  p.velocity.x *= -damping; }
}

int main() {
    const int WINDOW_WIDTH = 800;
    const int WINDOW_HEIGHT = 600;

    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "SPH Phase 2 - OpenMP Multi-threaded");
    SetTargetFPS(60);

    int numThreads = omp_get_max_threads();

    SPHSettings settings(
        1.0f,      // mass
        1.0f,      // restDensity
        2000.0f,   // gasConstant
        250.0f,    // viscosity
        16.0f,     // h
        980.0f     // gravity
    );

    std::vector<Particle> particles(1);
    initialize_dam_break(particles, settings, (float)WINDOW_WIDTH, (float)WINDOW_HEIGHT);
    int numParticles = (int)particles.size();

    const float dt = 0.003f;

    int frameCount = 0;
    float fpsSum = 0.0f;
    int fpsSamples = 0;
    double densityTimeSum = 0.0;
    double forceTimeSum = 0.0;

    while (!WindowShouldClose()) {
        // ============================================
        // STEP 1: DENSITY (O(n²)) — parallelized outer loop
        // Each particle i's density depends only on reading all j positions,
        // so the outer loop over i is embarrassingly parallel.
        // ============================================
        double t0 = omp_get_wtime();

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numParticles; i++) {
            float density_i = 0.0f;

            for (int j = 0; j < numParticles; j++) {
                glm::vec2 r_vec = particles[j].position - particles[i].position;
                float r_squared = glm::dot(r_vec, r_vec);

                if (r_squared < settings.h2) {
                    float diff = settings.h2 - r_squared;
                    density_i += settings.mass * settings.poly6 * diff * diff * diff;
                }
            }

            particles[i].density = density_i;
            particles[i].pressure = settings.gasConstant * (density_i - settings.restDensity);
        }

        float minDens = settings.restDensity * 0.1f;
        for (int i = 0; i < numParticles; i++) {
            if (particles[i].density < minDens)
                particles[i].density = minDens;
        }

        double t1 = omp_get_wtime();
        double densityMs = (t1 - t0) * 1000.0;
        densityTimeSum += densityMs;

        // ============================================
        // STEP 2: FORCES (O(n²)) — parallelized outer loop
        // Each particle i accumulates forces from all j neighbors.
        // We write only to particles[i].force, so no data races on the outer loop.
        // ============================================
        double t2 = omp_get_wtime();

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numParticles; i++) {
            glm::vec2 f_i(0.0f, settings.g);

            for (int j = 0; j < numParticles; j++) {
                if (i == j) continue;

                glm::vec2 r_vec = particles[j].position - particles[i].position;
                float r_squared = glm::dot(r_vec, r_vec);

                if (r_squared < settings.h2 && r_squared > 0.0001f) {
                    float r_length = std::sqrt(r_squared);

                    if (r_length < settings.h) {
                        float diff = settings.h - r_length;

                        // Pressure force
                        float gradCoeff = settings.spikyGrad * diff * diff / r_length;
                        glm::vec2 pressGrad = gradCoeff * r_vec;
                        float pressTerm = (particles[i].pressure / (particles[i].density * particles[i].density) +
                                           particles[j].pressure / (particles[j].density * particles[j].density));
                        f_i += -settings.mass * pressTerm * pressGrad;

                        // Viscosity force
                        float lap = settings.spikyLap * diff;
                        glm::vec2 velDiff = particles[j].velocity - particles[i].velocity;
                        f_i += settings.viscosity * settings.mass * velDiff / particles[j].density * lap;
                    }
                }
            }

            particles[i].force = f_i;
        }

        double t3 = omp_get_wtime();
        double forceMs = (t3 - t2) * 1000.0;
        forceTimeSum += forceMs;

        // ============================================
        // STEP 3: INTEGRATION (O(n))
        // ============================================
        for (auto& p : particles) {
            if (p.density > 0.0001f) {
                p.velocity += (p.force / p.density) * dt;
                p.position += p.velocity * dt;
            }
            handle_boundaries(p, WINDOW_WIDTH, WINDOW_HEIGHT);
        }

        // ============================================
        // RENDERING
        // ============================================
        BeginDrawing();
        ClearBackground(WHITE);

        for (const auto& p : particles) {
            DrawCircle(p.position.x, p.position.y, 2, PURPLE);
        }

        float fps = GetFPS();
        if (fps > 0.0f) { fpsSum += fps; fpsSamples++; }
        frameCount++;
        DrawFPS(10, 10);
        DrawText(TextFormat("Particles: %d  [OpenMP %d threads]", numParticles, numThreads), 10, 30, 20, BLACK);
        DrawText(TextFormat("Density: %.1f ms  Force: %.1f ms", densityMs, forceMs), 10, 50, 20, BLACK);
        DrawText(TextFormat("Frames: %d", frameCount), 10, 70, 20, BLACK);

        EndDrawing();
    }

    // ----- Session log -----
    {
#if defined(_WIN32) || defined(_WIN64)
        system("mkdir benchmarks 2>nul");
#else
        system("mkdir -p benchmarks");
#endif
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        std::tm* tm = std::localtime(&t);
        std::ostringstream fname;
        fname << "benchmarks/sph_omp_" << std::put_time(tm, "%Y%m%d_%H%M%S") << ".log";

        float minX = 1e9f, maxX = -1e9f, minY = 1e9f, maxY = -1e9f;
        for (const auto& p : particles) {
            if (p.position.x < minX) minX = p.position.x;
            if (p.position.x > maxX) maxX = p.position.x;
            if (p.position.y < minY) minY = p.position.y;
            if (p.position.y > maxY) maxY = p.position.y;
        }
        float avgFps = (fpsSamples > 0) ? (fpsSum / fpsSamples) : 0.0f;
        double avgDensityMs = (frameCount > 0) ? (densityTimeSum / frameCount) : 0.0;
        double avgForceMs = (frameCount > 0) ? (forceTimeSum / frameCount) : 0.0;

        std::ofstream log(fname.str());
        if (log.is_open()) {
            log << "SPH Phase 2 - OpenMP Multi-threaded Session Log\n";
            log << "================================================\n";
            log << "Timestamp: " << std::put_time(tm, "%Y-%m-%d %H:%M:%S") << "\n\n";
            log << "Config:\n";
            log << "  particles   = " << numParticles << "\n";
            log << "  threads     = " << numThreads << "\n";
            log << "  h           = " << settings.h << "\n";
            log << "  restDensity = " << settings.restDensity << "\n";
            log << "  gasConstant = " << settings.gasConstant << "\n";
            log << "  viscosity   = " << settings.viscosity << "\n";
            log << "  gravity     = " << settings.g << "\n";
            log << "  dt          = " << dt << "\n\n";
            log << "Run:\n";
            log << "  total_frames     = " << frameCount << "\n";
            log << "  avg_fps          = " << std::fixed << std::setprecision(2) << avgFps << "\n";
            log << "  avg_density_ms   = " << std::fixed << std::setprecision(3) << avgDensityMs << "\n";
            log << "  avg_force_ms     = " << std::fixed << std::setprecision(3) << avgForceMs << "\n";
            log << "  avg_total_phys   = " << std::fixed << std::setprecision(3) << (avgDensityMs + avgForceMs) << "\n\n";
            log << "Final particle bounds:\n";
            log << "  x_min = " << minX << ", x_max = " << maxX << "\n";
            log << "  y_min = " << minY << ", y_max = " << maxY << "\n";
            log.close();
        }
    }

    CloseWindow();
    return 0;
}
