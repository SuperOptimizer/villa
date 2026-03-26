#include "vc4d/core/grid.hpp"
#include "vc4d/core/math.hpp"

#include <QTest>

using namespace vc4d;

class TestGrid : public QObject {
    Q_OBJECT

private slots:
    void construction() {
        Grid<int> g(3, 4);
        QCOMPARE(g.rows(), 3);
        QCOMPARE(g.cols(), 4);
        QVERIFY(!g.has(0, 0));
        QCOMPARE(g.count_valid(), 0);
    }

    void construction_with_fill() {
        Grid<int> g(2, 2, 42);
        QVERIFY(g.has(0, 0));
        QCOMPARE(*g(0, 0), 42);
        QCOMPARE(g.count_valid(), 4);
    }

    void set_and_get() {
        Grid<Vec3f> g(10, 10);

        g.set(3, 4, {1.f, 2.f, 3.f});
        QVERIFY(g.has(3, 4));
        QCOMPARE(g(3, 4)->x, 1.f);

        g.clear(3, 4);
        QVERIFY(!g.has(3, 4));
    }

    void bounds_checking() {
        Grid<int> g(5, 5);

        QVERIFY(g.in_bounds(0, 0));
        QVERIFY(g.in_bounds(4, 4));
        QVERIFY(!g.in_bounds(-1, 0));
        QVERIFY(!g.in_bounds(0, 5));
        QVERIFY(!g.in_bounds(5, 0));

        // has() returns false for out-of-bounds
        QVERIFY(!g.has(-1, 0));
        QVERIFY(!g.has(5, 5));
    }

    void valid_point_iteration() {
        Grid<int> g(3, 3);
        g.set(0, 0, 1);
        g.set(1, 1, 2);
        g.set(2, 2, 3);

        int count = 0;
        int sum = 0;
        for (auto [r, c, v] : g.valid_points()) {
            ++count;
            sum += v;
            // Verify we can modify through the reference
            v *= 10;
        }

        QCOMPARE(count, 3);
        QCOMPARE(sum, 6);
        QCOMPARE(*g(0, 0), 10);
        QCOMPARE(*g(1, 1), 20);
        QCOMPARE(*g(2, 2), 30);
    }

    void const_valid_point_iteration() {
        Grid<int> g(3, 3);
        g.set(0, 1, 5);
        g.set(2, 0, 7);

        const auto& cg = g;
        int count = 0;
        for (auto [r, c, v] : cg.valid_points()) {
            ++count;
            (void)r; (void)c; (void)v;
        }
        QCOMPARE(count, 2);
    }

    void valid_quad_iteration() {
        // Create a 3x3 grid with all points set
        Grid<Vec3f> g(3, 3);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                g.set(r, c, Vec3f{float(r), float(c), 0.f});

        // Should have 4 valid quads (2x2 sub-grids)
        int count = 0;
        for (auto [r, c, p00, p01, p10, p11] : g.valid_quads()) {
            ++count;
            (void)r; (void)c;
            // Verify quad corners
            QCOMPARE(p00.x, float(r));
            QCOMPARE(p00.y, float(c));
            QCOMPARE(p11.x, float(r + 1));
            QCOMPARE(p11.y, float(c + 1));
        }
        QCOMPARE(count, 4);
    }

    void sparse_quad_iteration() {
        // Grid with a hole — some quads should be missing
        Grid<Vec3f> g(3, 3);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                g.set(r, c, Vec3f{float(r), float(c), 0.f});

        // Remove center point — this invalidates all 4 quads touching (1,1)
        g.clear(1, 1);

        int count = 0;
        for (auto [r, c, p00, p01, p10, p11] : g.valid_quads()) {
            ++count;
            (void)r; (void)c; (void)p00; (void)p01; (void)p10; (void)p11;
        }
        // Only quad (0,0)→(1,1) uses center, plus three others
        // With center removed: quads (0,0), (0,1), (1,0), (1,1) all touch center
        // So 0 quads remain
        QCOMPARE(count, 0);
    }

    void no_sentinel_values() {
        // This is the key improvement: we can store x=-1 as valid data
        Grid<Vec3f> g(2, 2);
        g.set(0, 0, Vec3f{-1.f, -1.f, -1.f});  // Would be "invalid" in vc3d!

        QVERIFY(g.has(0, 0));
        QCOMPARE(g(0, 0)->x, -1.f);

        // And empty cells are truly empty, not sentinel
        QVERIFY(!g.has(1, 1));
    }

    void empty_grid() {
        Grid<int> g;
        QCOMPARE(g.rows(), 0);
        QCOMPARE(g.cols(), 0);
        QVERIFY(g.empty());
        QCOMPARE(g.count_valid(), 0);

        // Iterating over empty grid should be safe
        for (auto [r, c, v] : g.valid_points()) {
            (void)r; (void)c; (void)v;
            QFAIL("Should not iterate over empty grid");
        }
    }
};

QTEST_MAIN(TestGrid)
#include "test_grid.moc"
