#pragma once

#include <cstdint>
#include <ctime>

namespace vc {

// Fast xorshift64* PRNG - much faster than mt19937
// Thread-local for thread safety and better performance
class Random {
public:
    // Get thread-local instance
    static Random& instance() {
        thread_local Random rng;
        return rng;
    }

    // Seed the generator (thread-local)
    void seed(uint32_t s) {
        state_ = s;
        if (state_ == 0) state_ = 1; // xorshift can't have zero state
    }

    // Generate random integer in range [0, max)
    int randInt(int max) {
        if (max <= 0) return 0;
        return static_cast<int>(next() % static_cast<uint64_t>(max));
    }

    // Generate random integer in range [min, max)
    int randInt(int min, int max) {
        if (min >= max) return min;
        return min + randInt(max - min);
    }

    // Generate random float in range [0.0, 1.0]
    float randFloat() {
        return static_cast<float>(next() & 0xFFFFFF) / 16777216.0f;
    }

    // Generate random float in range [min, max]
    float randFloat(float min, float max) {
        return min + randFloat() * (max - min);
    }

private:
    Random() : state_(static_cast<uint64_t>(time(nullptr)) ^ static_cast<uint64_t>(reinterpret_cast<uintptr_t>(this))) {
        if (state_ == 0) state_ = 1;
    }
    Random(const Random&) = delete;
    Random& operator=(const Random&) = delete;

    // xorshift64* algorithm - very fast, good statistical properties
    uint64_t next() {
        uint64_t x = state_;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        state_ = x;
        return x * 0x2545F4914F6CDD1DULL;
    }

    uint64_t state_;
};

// Convenience functions for common use cases
inline void randomSeed(uint32_t seed) {
    Random::instance().seed(seed);
}

inline int randomInt(int max) {
    return Random::instance().randInt(max);
}

inline int randomInt(int min, int max) {
    return Random::instance().randInt(min, max);
}

inline float randomFloat() {
    return Random::instance().randFloat();
}

inline float randomFloat(float min, float max) {
    return Random::instance().randFloat(min, max);
}

} // namespace vc
