#include "raylib.h"
#include <glm/glm.hpp>
#include <vector>
#include <chrono>

// Using the structure from the reference repo
struct Particle {
    glm::vec2 position;
    glm::vec2 velocity;
    glm::vec2 force;
    float density;
    float pressure;
};

// SPH Constants derived from EG STAR 2014 & Repo Reference
const float H = 16.0f;
const float HSQ = H * H;
const float MASS = 1.0f;
const float VISCOSITY = 250.0f;
const float GAS_CONSTANT = 2000.0f;
const float REST_DENSITY = 1.0f;

int main() {
    InitWindow(800, 600, "sph-optimization-research | Reference-Based Baseline");
    SetConfigFlags(FLAG_FULLSCREEN_MODE);
    
    int numParticles = 1500;
    std::vector<Particle> particles(numParticles);

    // Initial State: Dam Break (Repo logic)
    for (int i = 0; i < numParticles; i++) {
        particles[i].position = glm::vec2(GetRandomValue(100, 300), GetRandomValue(100, 400));
        particles[i].velocity = glm::vec2(0, 0);
    }

    while (!WindowShouldClose()) {
        float dt = 0.003f;

        // --- STEP 1: DENSITY LOOP (The O(n^2) Research Focus) ---
        // -- The Math: For every particle i, we look at every other particle j. If they are closer than the distance H (the smoothing radius), particle j contributes to the density of particle i.
        //The Kernel (W): We use a "Weighting Function" (Poly6). Particles very close to each other increase density more than particles far away.
        //Pressure: Once we know how "crowded" (dense) an area is, we calculate pressure. If density is higher than the REST_DENSITY, the particles will push away from each other.  --
        auto start = std::chrono::high_resolution_clock::now();
        for (auto& pi : particles) {
            pi.density = 0;
            for (auto& pj : particles) {
                float distSq = glm::distance(pi.position, pj.position);
                if (distSq < HSQ) {
                    // Kernel calculation from reference
                    float diff = HSQ - distSq;
                    pi.density += MASS * (315.0f / (64.0f * PI * pow(H, 9))) * diff * diff * diff;
                }
            }
            pi.pressure = GAS_CONSTANT * (pi.density - REST_DENSITY);
        }
        auto end = std::chrono::high_resolution_clock::now();

        // --- STEP 2: FORCE/INTEGRATION (Scalar Baseline) ---
        for (auto& p : particles) {
            glm::vec2 gravity(0, 9.8f * 100.0f);
            p.force = gravity;
            
            // Integration logic
            p.velocity += (p.force / p.density) * dt;
            p.position += p.velocity * dt;

            // Boundary handling
            if (p.position.y > 580) { p.position.y = 580; p.velocity.y *= -0.5f; }
        }

        BeginDrawing();
        ClearBackground(BLACK);
        for (auto& p : particles) DrawCircle(p.position.x, p.position.y, 2, BLUE);
        DrawFPS(10, 10);
        EndDrawing();
    }
    CloseWindow();
    return 0;
}