#include "test.hpp"
#include "vc/core/types/Tensor.hpp"

#include <cstring>

using namespace vc;

// --- Construction ---

TEST(Tensor, DefaultConstruct) {
    Tensor<float> t;
    EXPECT_EQ(t.size(), 0u);
    EXPECT_EQ(t.ndim(), 0u);
    EXPECT_EQ(t.data(), nullptr);
}

TEST(Tensor, ShapeConstruct) {
    Tensor<uint8_t> t({3, 4, 5});
    EXPECT_EQ(t.ndim(), 3u);
    EXPECT_EQ(t.shape(0), 3u);
    EXPECT_EQ(t.shape(1), 4u);
    EXPECT_EQ(t.shape(2), 5u);
    EXPECT_EQ(t.size(), 60u);
    EXPECT_NE(t.data(), nullptr);
}

TEST(Tensor, ZerosFactory) {
    auto t = zeros<float>({2, 3});
    EXPECT_EQ(t.size(), 6u);
    for (size_t i = 0; i < t.size(); i++)
        EXPECT_EQ(t.data()[i], 0.0f);
}

TEST(Tensor, EmptyFactory) {
    auto t = empty<uint16_t>({4, 4, 4});
    EXPECT_EQ(t.size(), 64u);
    EXPECT_EQ(t.ndim(), 3u);
}

TEST(Tensor, FullLike) {
    auto t = zeros<uint8_t>({3, 3, 3});
    auto f = full_like(t, uint8_t(42));
    EXPECT_EQ(f.shape(), t.shape());
    for (size_t i = 0; i < f.size(); i++)
        EXPECT_EQ(f.data()[i], 42);
}

TEST(Tensor, Fill) {
    auto t = empty<float>({2, 2});
    t.fill(3.14f);
    for (size_t i = 0; i < t.size(); i++)
        EXPECT_FLOAT_EQ(t.data()[i], 3.14f);
}

// --- Element access (RowMajor) ---

TEST(Tensor, RowMajorAccess) {
    auto t = zeros<int>({2, 3, 4}, Layout::RowMajor);
    // Row-major: last index varies fastest
    t(0, 0, 0) = 1;
    t(1, 2, 3) = 99;
    EXPECT_EQ(t(0, 0, 0), 1);
    EXPECT_EQ(t(1, 2, 3), 99);
    // Verify linear layout: (1,2,3) = 1*3*4 + 2*4 + 3 = 12+8+3 = 23
    EXPECT_EQ(t.data()[23], 99);
}

// --- Element access (ColumnMajor) ---

TEST(Tensor, ColumnMajorAccess) {
    auto t = Tensor<int>({3, 4, 5}, Layout::ColumnMajor, 0);
    t(0, 0, 0) = 1;
    t(2, 3, 4) = 77;
    EXPECT_EQ(t(0, 0, 0), 1);
    EXPECT_EQ(t(2, 3, 4), 77);
    // Column-major strides: {1, 3, 12}
    // (2,3,4) = 2*1 + 3*3 + 4*12 = 2+9+48 = 59
    EXPECT_EQ(t.data()[59], 77);
}

TEST(Tensor, ColumnMajorStrides) {
    auto t = Tensor<uint8_t>({10, 20, 30}, Layout::ColumnMajor);
    const auto& s = t.strides();
    EXPECT_EQ(s[0], 1u);
    EXPECT_EQ(s[1], 10u);
    EXPECT_EQ(s[2], 200u);
}

// --- Copy / Move ---

TEST(Tensor, CopySemantic) {
    auto a = zeros<float>({2, 3});
    a(0, 0) = 42.0f;
    auto b = a;  // copy
    EXPECT_EQ(b(0, 0), 42.0f);
    b(0, 0) = 99.0f;
    EXPECT_EQ(a(0, 0), 42.0f);  // original unchanged
}

TEST(Tensor, MoveSemantic) {
    auto a = zeros<float>({2, 3});
    a(0, 0) = 42.0f;
    auto b = std::move(a);
    EXPECT_EQ(b(0, 0), 42.0f);
    EXPECT_EQ(a.size(), 0u);
    EXPECT_EQ(a.data(), nullptr);
}

// --- View ---

TEST(TensorView, BasicSlice) {
    auto t = zeros<int>({4, 6, 8}, Layout::RowMajor);
    // Set a known value
    t(1, 2, 3) = 55;

    // View: [1:3, 1:4, 2:6]
    auto v = view(t, {range(1,3), range(1,4), range(2,6)});
    EXPECT_EQ(v.shape(0), 2u);
    EXPECT_EQ(v.shape(1), 3u);
    EXPECT_EQ(v.shape(2), 4u);

    // v(0,1,1) = t(1,2,3)
    EXPECT_EQ(v(0, 1, 1), 55);
}

TEST(TensorView, WriteThrough) {
    auto t = zeros<int>({4, 4});
    auto v = view(t, {range(1,3), range(1,3)});
    v(0, 0) = 100;
    EXPECT_EQ(t(1, 1), 100);  // write-through
}

TEST(TensorView, ColumnMajorView) {
    auto t = Tensor<int>({4, 4, 4}, Layout::ColumnMajor, 0);
    t(2, 1, 3) = 88;
    auto v = view(t, {range(2,4), range(0,4), range(2,4)});
    // v(0,1,1) maps to t(2,1,3)
    EXPECT_EQ(v(0, 1, 1), 88);
}

// --- Adaptor ---

TEST(TensorAdaptor, WrapPointer) {
    float data[12] = {0};
    auto a = adapt(data, {3, 4});
    a(1, 2) = 7.5f;
    // Row-major: (1,2) = 1*4+2 = 6
    EXPECT_FLOAT_EQ(data[6], 7.5f);
}

TEST(TensorAdaptor, ColumnMajor) {
    uint8_t data[8] = {0};
    auto a = adapt(data, {2, 2, 2}, Layout::ColumnMajor);
    a(1, 0, 0) = 42;
    // Column-major strides: {1, 2, 4}; (1,0,0) = 1
    EXPECT_EQ(data[1], 42);
}
