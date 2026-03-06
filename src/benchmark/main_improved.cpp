#include "raylib.h"
#include <glm/glm.hpp>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <iomanip>

// Particle structure matching reference.c style
struct Particle {
    glm::vec2 position;
    glm::vec2 velocity;
    glm::vec2 force;
    float density;
    float pressure;
};

// SPH Settings structure (matching reference.c architecture)
struct SPHSettings {
    float mass;
    float restDensity;
    float gasConstant;
    float viscosity;
    float h;           // Smoothing radius
    float h2;          // h squared (precomputed)
    float g;           // Gravity magnitude
    
    // Precomputed kernel constants (calculated once, used many times)
    float poly6;       // Poly6 kernel coefficient: 315 / (64πh^9)
    float spikyGrad;   // Spiky gradient coefficient: -45 / (πh^6)
    float spikyLap;    // Spiky Laplacian coefficient: 45 / (πh^6)
    
    SPHSettings(float mass, float restDensity, float gasConst, 
                float viscosity, float h, float g)
        : mass(mass)
        , restDensity(restDensity)
        , gasConstant(gasConst)
        , viscosity(viscosity)
        , h(h)
        , g(g)
    {
        h2 = h * h;
        // Precompute kernel constants (matching reference.c); raylib defines PI
        poly6 = 315.0f / (64.0f * PI * powf(h, 9));
        spikyGrad = -45.0f / (PI * powf(h, 6));
        spikyLap = 45.0f / (PI * powf(h, 6));
    }
};

// Poly6 kernel: W_poly6(r², h) = (315/64πh⁹)(h² - r²)³
// Used for density calculation
float poly6_kernel(float r_squared, const SPHSettings& settings) {
    if (r_squared >= settings.h2) return 0.0f;
    float diff = settings.h2 - r_squared;
    return settings.mass * settings.poly6 * diff * diff * diff;
}

// Spiky gradient kernel: ∇W_spiky(r, h) = (-45/πh⁶)(h - r)² * (r_vec / r)
// Used for pressure force calculation
glm::vec2 spiky_gradient(const glm::vec2& r_vec, float r_length, const SPHSettings& settings) {
    if (r_length >= settings.h || r_length < 0.0001f) return glm::vec2(0.0f);
    float diff = settings.h - r_length;
    float coeff = settings.spikyGrad * diff * diff / r_length;
    return coeff * r_vec;
}

// Spiky Laplacian kernel: ∇²W_spiky(r, h) = (45/πh⁶)(h - r)
// Used for viscosity force calculation
float spiky_laplacian(float r_length, const SPHSettings& settings) {
    if (r_length >= settings.h) return 0.0f;
    return settings.spikyLap * (settings.h - r_length);
}

// Initialize particles in dam break: 2D block spanning full screen width
void initialize_dam_break(std::vector<Particle>& particles, const SPHSettings& settings,
                          float windowWidth, float windowHeight) {
    std::srand(1024);  // Reproducible seed
    
    const float margin = 20.0f;
    float particleSeparation = settings.h * 0.5f;  // Compact spacing
    
    // Columns to span full width; rows for a thick block
    int cols = (int)((windowWidth - 2.0f * margin) / particleSeparation);
    int rows = 28;  // Block height in particle rows
    size_t n = (size_t)cols * (size_t)rows;
    particles.resize(n);
    
    float startX = margin;
    float startY = margin;  // Top of screen
    
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

// Boundary handling: bounce particles off walls
void handle_boundaries(Particle& p, float width, float height) {
    const float damping = 0.5f;  // Velocity damping on collision
    
    // Floor
    if (p.position.y > height - 10) {
        p.position.y = height - 10;
        p.velocity.y *= -damping;
    }
    // Ceiling
    if (p.position.y < 10) {
        p.position.y = 10;
        p.velocity.y *= -damping;
    }
    // Left wall
    if (p.position.x < 10) {
        p.position.x = 10;
        p.velocity.x *= -damping;
    }
    // Right wall
    if (p.position.x > width - 10) {
        p.position.x = width - 10;
        p.velocity.x *= -damping;
    }
}

int main() {
    const int WINDOW_WIDTH = 800;
    const int WINDOW_HEIGHT = 600;
    
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "SPH Baseline - Improved");
    SetTargetFPS(60);
    
    // Initialize SPH settings (matching reference.c values)
    SPHSettings settings(
        1.0f,      // mass
        1.0f,      // restDensity
        2000.0f,   // gasConstant
        250.0f,    // viscosity
        16.0f,     // smoothing radius h
        980.0f     // gravity (9.8 * 100 for pixel scale)
    );
    
    int numParticles = 0;  // Set after init (init resizes to fill full width)
    std::vector<Particle> particles(1);  // Placeholder; init resizes
    
    // Initialize dam break scenario (block spans full width)
    initialize_dam_break(particles, settings, (float)WINDOW_WIDTH, (float)WINDOW_HEIGHT);
    numParticles = (int)particles.size();
    
    // Fixed timestep for stability (matching reference.c)
    const float dt = 0.003f;
    
    // Session stats for logging on exit
    int frameCount = 0;
    float fpsSum = 0.0f;
    int fpsSamples = 0;
    
    while (!WindowShouldClose()) {
        // ============================================
        // STEP 1: DENSITY CALCULATION (O(n²))
        // ============================================
        // For each particle i, sum contributions from all neighbors j
        // ρ_i = Σ m_j * W_poly6(|r_i - r_j|, h)
        for (size_t i = 0; i < particles.size(); i++) {
            particles[i].density = 0.0f;
            
            for (size_t j = 0; j < particles.size(); j++) {
                // CRITICAL FIX: Use squared distance, not glm::distance()!
                glm::vec2 r_vec = particles[j].position - particles[i].position;
                float r_squared = glm::dot(r_vec, r_vec);
                
                // Only process neighbors within smoothing radius
                if (r_squared < settings.h2) {
                    particles[i].density += poly6_kernel(r_squared, settings);
                }
            }
            
            // Calculate pressure using equation of state
            particles[i].pressure = settings.gasConstant * (particles[i].density - settings.restDensity);
        }
        
        // Clamp minimum density to avoid explosion from division by near-zero
        for (auto& p : particles) {
            if (p.density < settings.restDensity * 0.1f)
                p.density = settings.restDensity * 0.1f;
        }
        
        // ============================================
        // STEP 2: FORCE ACCUMULATION (O(n²))
        // ============================================
        // F_i = F_gravity + F_pressure + F_viscosity
        for (size_t i = 0; i < particles.size(); i++) {
            glm::vec2 gravity(0.0f, settings.g);
            particles[i].force = gravity;
            
            for (size_t j = 0; j < particles.size(); j++) {
                if (i == j) continue;  // Skip self-interaction
                
                glm::vec2 r_vec = particles[j].position - particles[i].position;
                float r_squared = glm::dot(r_vec, r_vec);
                
                if (r_squared < settings.h2 && r_squared > 0.0001f) {
                    float r_length = std::sqrt(r_squared);
                    
                    // Pressure force: F_pressure = -m * (P_i/ρ_i² + P_j/ρ_j²) * ∇W_spiky
                    glm::vec2 pressureGrad = spiky_gradient(r_vec, r_length, settings);
                    float pressureTerm = (particles[i].pressure / (particles[i].density * particles[i].density) +
                                         particles[j].pressure / (particles[j].density * particles[j].density));
                    particles[i].force += -settings.mass * pressureTerm * pressureGrad;
                    
                    // Viscosity force: F_viscosity = μ * m * (v_j - v_i) / ρ_j * ∇²W_spiky
                    float viscosityTerm = spiky_laplacian(r_length, settings);
                    glm::vec2 velocityDiff = particles[j].velocity - particles[i].velocity;
                    particles[i].force += settings.viscosity * settings.mass * velocityDiff / particles[j].density * viscosityTerm;
                }
            }
        }
        
        // ============================================
        // STEP 3: INTEGRATION (O(n))
        // ============================================
        // v_{t+dt} = v_t + (F / ρ) * dt
        // x_{t+dt} = x_t + v_{t+dt} * dt
        for (auto& p : particles) {
            if (p.density > 0.0001f) {  // Avoid division by zero
                p.velocity += (p.force / p.density) * dt;
                p.position += p.velocity * dt;
            }
            
            // Boundary collision handling
            handle_boundaries(p, WINDOW_WIDTH, WINDOW_HEIGHT);
        }
        
        // ============================================
        // RENDERING
        // ============================================
        BeginDrawing();
        ClearBackground(WHITE);
        
        // Draw particles
        for (const auto& p : particles) {
            DrawCircle(p.position.x, p.position.y, 2, PURPLE);
        }
        
        // Draw info
        float fps = GetFPS();
        if (fps > 0.0f) { fpsSum += fps; fpsSamples++; }
        frameCount++;
        DrawFPS(10, 10);
        DrawText(TextFormat("Particles: %d", numParticles), 10, 30, 20, BLACK);
        DrawText(TextFormat("Frames: %d", frameCount), 10, 50, 20, BLACK);
        
        EndDrawing();
    }
    
    // ----- Session log (written when window closes) -----
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
        fname << "benchmarks/sph_run_" << std::put_time(tm, "%Y%m%d_%H%M%S") << ".log";
        
        float minX = 1e9f, maxX = -1e9f, minY = 1e9f, maxY = -1e9f;
        for (const auto& p : particles) {
            if (p.position.x < minX) minX = p.position.x;
            if (p.position.x > maxX) maxX = p.position.x;
            if (p.position.y < minY) minY = p.position.y;
            if (p.position.y > maxY) maxY = p.position.y;
        }
        float avgFps = (fpsSamples > 0) ? (fpsSum / fpsSamples) : 0.0f;
        
        std::ofstream log(fname.str());
        if (log.is_open()) {
            log << "SPH Baseline Improved - Session Log\n";
            log << "================================\n";
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
