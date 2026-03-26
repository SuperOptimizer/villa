#include "vc4d/core/math.hpp"

#include <QTest>
#include <cmath>

using namespace vc4d;

class TestMath : public QObject {
    Q_OBJECT

private slots:
    void vec3_basics() {
        Vec3f a{1, 2, 3};
        Vec3f b{4, 5, 6};

        auto c = a + b;
        QCOMPARE(c.x, 5.f);
        QCOMPARE(c.y, 7.f);
        QCOMPARE(c.z, 9.f);

        auto d = a - b;
        QCOMPARE(d.x, -3.f);
        QCOMPARE(d.y, -3.f);
        QCOMPARE(d.z, -3.f);

        auto e = a * 2.f;
        QCOMPARE(e.x, 2.f);
        QCOMPARE(e.y, 4.f);
        QCOMPARE(e.z, 6.f);

        QCOMPARE(2.f * a, e);
    }

    void vec3_dot_cross() {
        Vec3f a{1, 0, 0};
        Vec3f b{0, 1, 0};

        QCOMPARE(a.dot(b), 0.f);
        QCOMPARE(a.dot(a), 1.f);

        auto c = a.cross(b);
        QCOMPARE(c.x, 0.f);
        QCOMPARE(c.y, 0.f);
        QCOMPARE(c.z, 1.f);
    }

    void vec3_length_normalize() {
        Vec3f a{3, 4, 0};
        QCOMPARE(a.length_sq(), 25.f);
        QCOMPARE(a.length(), 5.f);

        auto n = a.normalized();
        QVERIFY(std::abs(n.length() - 1.f) < 1e-6f);
        QVERIFY(std::abs(n.x - 0.6f) < 1e-6f);
        QVERIFY(std::abs(n.y - 0.8f) < 1e-6f);

        // Zero vector normalizes to zero
        Vec3f zero{};
        auto zn = zero.normalized();
        QCOMPARE(zn.x, 0.f);
    }

    void vec3_subscript() {
        Vec3f a{10, 20, 30};
        QCOMPARE(a[0], 10.f);
        QCOMPARE(a[1], 20.f);
        QCOMPARE(a[2], 30.f);

        a[1] = 99.f;
        QCOMPARE(a.y, 99.f);
    }

    void vec2_basics() {
        Vec2f a{3, 4};
        QCOMPARE(a.length(), 5.f);
        QCOMPARE(a.dot(a), 25.f);
    }

    void box3_basics() {
        Box3f box{{0, 0, 0}, {10, 10, 10}};

        QVERIFY(box.valid());
        QVERIFY(box.contains({5, 5, 5}));
        QVERIFY(!box.contains({-1, 5, 5}));

        auto c = box.center();
        QCOMPARE(c.x, 5.f);
        QCOMPARE(c.y, 5.f);
        QCOMPARE(c.z, 5.f);
    }

    void box3_intersect() {
        Box3f a{{0, 0, 0}, {10, 10, 10}};
        Box3f b{{5, 5, 5}, {15, 15, 15}};
        Box3f c{{20, 20, 20}, {30, 30, 30}};

        QVERIFY(a.intersects(b));
        QVERIFY(!a.intersects(c));
    }

    void box3_expand_merge() {
        auto box = Box3f::empty();
        box = box.expanded({1, 2, 3});
        box = box.expanded({-1, 5, 0});

        QCOMPARE(box.lo.x, -1.f);
        QCOMPARE(box.lo.y, 2.f);
        QCOMPARE(box.lo.z, 0.f);
        QCOMPARE(box.hi.x, 1.f);
        QCOMPARE(box.hi.y, 5.f);
        QCOMPARE(box.hi.z, 3.f);
    }

    void distance_functions() {
        Vec3f a{0, 0, 0};
        Vec3f b{3, 4, 0};

        QCOMPARE(distance(a, b), 5.f);
        QCOMPARE(distance_sq(a, b), 25.f);
    }

    void lerp_functions() {
        QCOMPARE(lerp(0.f, 10.f, 0.5f), 5.f);
        QCOMPARE(lerp(0.f, 10.f, 0.f), 0.f);
        QCOMPARE(lerp(0.f, 10.f, 1.f), 10.f);

        auto v = lerp(Vec3f{0, 0, 0}, Vec3f{10, 20, 30}, 0.5f);
        QCOMPARE(v.x, 5.f);
        QCOMPARE(v.y, 10.f);
        QCOMPARE(v.z, 15.f);
    }

    void constexpr_usage() {
        // Verify these work at compile time
        constexpr Vec3f a{1, 2, 3};
        constexpr Vec3f b{4, 5, 6};
        constexpr auto c = a + b;
        static_assert(c.x == 5.f);
        static_assert(c.y == 7.f);

        constexpr auto d = a.dot(b);
        static_assert(d == 32.f);

        constexpr Box3f box{{0,0,0}, {1,1,1}};
        static_assert(box.valid());
        static_assert(box.contains({0.5f, 0.5f, 0.5f}));
    }
};

QTEST_MAIN(TestMath)
#include "test_math.moc"
