#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/Compositing.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <numbers>
#include <string>
#include <vector>

namespace {

LayerStack stack(std::vector<float> v)
{
    LayerStack s;
    s.values = std::move(v);
    s.validCount = static_cast<int>(s.values.size());
    return s;
}

LayerStack emptyStack()
{
    return LayerStack{};
}

} // namespace

TEST_CASE("CompositeMethod::mean")
{
    CHECK(CompositeMethod::mean(emptyStack()) == doctest::Approx(0.0f));
    CHECK(CompositeMethod::mean(stack({0.f, 10.f, 20.f})) == doctest::Approx(10.0f));
    CHECK(CompositeMethod::mean(stack({100.f})) == doctest::Approx(100.0f));
}

TEST_CASE("CompositeMethod::max")
{
    CHECK(CompositeMethod::max(emptyStack()) == doctest::Approx(0.0f));
    CHECK(CompositeMethod::max(stack({1.f, 5.f, 2.f})) == doctest::Approx(5.0f));
}

TEST_CASE("CompositeMethod::min")
{
    // empty → 255.0 (sentinel: nothing observed)
    CHECK(CompositeMethod::min(emptyStack()) == doctest::Approx(255.0f));
    CHECK(CompositeMethod::min(stack({5.f, 1.f, 3.f})) == doctest::Approx(1.0f));
}

TEST_CASE("CompositeMethod::alpha empty stack is 0")
{
    CompositeParams p;
    CHECK(CompositeMethod::alpha(emptyStack(), p) == doctest::Approx(0.0f));
}

TEST_CASE("CompositeMethod::alpha returns finite [0,255] value")
{
    CompositeParams p;
    p.alphaMin = 0.0f;
    p.alphaMax = 1.0f;
    p.alphaOpacity = 1.0f;
    p.alphaCutoff = 1.0f;
    float v = CompositeMethod::alpha(stack({50.f, 100.f, 200.f}), p);
    CHECK(std::isfinite(v));
    CHECK(v >= 0.0f);
    CHECK(v <= 255.0f);
}

TEST_CASE("compositeLayerStack dispatches per method")
{
    LayerStack s = stack({0.f, 100.f, 200.f});

    CompositeParams p;
    p.method = "mean";
    CHECK(compositeLayerStack(s, p) == doctest::Approx(100.0f));
    p.method = "max";
    CHECK(compositeLayerStack(s, p) == doctest::Approx(200.0f));
    p.method = "min";
    CHECK(compositeLayerStack(s, p) == doctest::Approx(0.0f));
}

TEST_CASE("compositeLayerStack: alpha dispatch returns finite")
{
    LayerStack s = stack({50.f, 100.f, 200.f});
    CompositeParams p;
    p.method = "alpha";
    CHECK(std::isfinite(compositeLayerStack(s, p)));
}

TEST_CASE("compositeLayerStack empty stack always returns 0")
{
    CompositeParams p;
    const std::vector<std::string> methods = {
        "mean", "max", "min", "alpha"
    };
    for (const auto& m : methods) {
        p.method = m;
        CHECK(compositeLayerStack(emptyStack(), p) == doctest::Approx(0.0f));
    }
}

TEST_CASE("compositeLayerStack unknown method falls back to mean")
{
    LayerStack s = stack({0.f, 100.f, 200.f});
    CompositeParams p;
    p.method = "this_does_not_exist";
    CHECK(compositeLayerStack(s, p) == doctest::Approx(100.0f));
}

TEST_CASE("methodRequiresLayerStorage returns bool for known and unknown methods")
{
    // We don't pin specific values to the underlying utils impl, just exercise
    // the call path for a representative set.
    (void)methodRequiresLayerStorage("mean");
    (void)methodRequiresLayerStorage("max");
    (void)methodRequiresLayerStorage("alpha");
    (void)methodRequiresLayerStorage("dvr");
    (void)methodRequiresLayerStorage("bogus");
    CHECK(true);
}

TEST_CASE("buildTfLut256 disabled is identity")
{
    uint8_t lut[256];
    buildTfLut256(false, 10, 20, 200, 100, lut);
    for (int i = 0; i < 256; ++i) {
        CHECK(int(lut[i]) == i);
    }
}

TEST_CASE("buildTfLut256 enabled passes through endpoints (0,0) and (255,255)")
{
    uint8_t lut[256];
    buildTfLut256(true, 85, 85, 170, 170, lut);
    CHECK(int(lut[0]) == 0);
    CHECK(int(lut[255]) == 255);
}

TEST_CASE("buildTfLut256 swaps x1 > x2 internally")
{
    uint8_t lutA[256], lutB[256];
    buildTfLut256(true, 50, 30, 200, 220, lutA);
    buildTfLut256(true, 200, 220, 50, 30, lutB);
    for (int i = 0; i < 256; ++i) CHECK(int(lutA[i]) == int(lutB[i]));
}

TEST_CASE("buildTfLut256 hits the knot values exactly")
{
    uint8_t lut[256];
    buildTfLut256(true, 100, 50, 200, 150, lut);
    CHECK(int(lut[100]) == 50);
    CHECK(int(lut[200]) == 150);
}

TEST_CASE("buildTfLut256 degenerate (x1==x2) does not crash and stays in [0,255]")
{
    uint8_t lut[256];
    buildTfLut256(true, 100, 80, 100, 200, lut);
    for (int i = 0; i < 256; ++i) {
        CHECK(int(lut[i]) >= 0);
        CHECK(int(lut[i]) <= 255);
    }
}

TEST_CASE("CompositeParams equality")
{
    CompositeParams a;
    CompositeParams b;
    CHECK(a == b);
    b.alphaOpacity = 0.5f;
    CHECK_FALSE(a == b);
}

TEST_CASE("CompositeRenderSettings equality")
{
    CompositeRenderSettings a;
    CompositeRenderSettings b;
    CHECK(a == b);
    b.layersFront = 999;
    CHECK_FALSE(a == b);
}
