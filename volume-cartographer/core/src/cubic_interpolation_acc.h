// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2023 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)

#ifndef CERES_PUBLIC_CUBIC_INTERPOLATION_H_ACC_
#define CERES_PUBLIC_CUBIC_INTERPOLATION_H_ACC_

#include "Eigen/Core"

namespace ceres {


template <typename Grid3D>
class LinxBiCubicInterpolator {
 public:
  explicit LinxBiCubicInterpolator(Grid3D& grid) : grid_(grid) {
    // The + casts the enum into an int before doing the
    // comparison. It is needed to prevent
    // "-Wunnamed-type-template-args" related errors.
    CHECK_GE(+Grid3D::DATA_DIMENSION, 1);
  }

  void set_z(int z)
  {
    _z = z;
  }

  // Evaluate the interpolated function value and/or its
  // derivative. Uses the nearest point on the grid boundary if r or
  // c is out of bounds.
  void Evaluate(
      double r, double c, double* f, double* dfdr, double* dfdc) const {
    // BiCubic interpolation requires 16 values around the point being
    // evaluated.  We will use pij, to indicate the elements of the
    // 4x4 grid of values.
    //
    //          col
    //      p00 p01 p02 p03
    // row  p10 p11 p12 p13
    //      p20 p21 p22 p23
    //      p30 p31 p32 p33
    //
    // The point (r,c) being evaluated is assumed to lie in the square
    // defined by p11, p12, p22 and p21.

    const int row = std::floor(r);
    const int col = std::floor(c);

    Eigen::Matrix<double, Grid3D::DATA_DIMENSION, 1> p0, p1, p2, p3;

    // Interpolate along each of the four rows, evaluating the function
    // value and the horizontal derivative in each row.
    Eigen::Matrix<double, Grid3D::DATA_DIMENSION, 1> f0, f1, f2, f3;
    Eigen::Matrix<double, Grid3D::DATA_DIMENSION, 1> df0dc, df1dc, df2dc, df3dc;

    grid_.GetValue(_z, row - 1, col - 1, p0.data());
    grid_.GetValue(_z, row - 1, col, p1.data());
    grid_.GetValue(_z, row - 1, col + 1, p2.data());
    grid_.GetValue(_z, row - 1, col + 2, p3.data());
    CubicHermiteSpline<Grid3D::DATA_DIMENSION>(
        p0, p1, p2, p3, c - col, f0.data(), df0dc.data());

    grid_.GetValue(_z, row, col - 1, p0.data());
    grid_.GetValue(_z, row, col, p1.data());
    grid_.GetValue(_z, row, col + 1, p2.data());
    grid_.GetValue(_z, row, col + 2, p3.data());
    CubicHermiteSpline<Grid3D::DATA_DIMENSION>(
        p0, p1, p2, p3, c - col, f1.data(), df1dc.data());

    grid_.GetValue(_z, row + 1, col - 1, p0.data());
    grid_.GetValue(_z, row + 1, col, p1.data());
    grid_.GetValue(_z, row + 1, col + 1, p2.data());
    grid_.GetValue(_z, row + 1, col + 2, p3.data());
    CubicHermiteSpline<Grid3D::DATA_DIMENSION>(
        p0, p1, p2, p3, c - col, f2.data(), df2dc.data());

    grid_.GetValue(_z, row + 2, col - 1, p0.data());
    grid_.GetValue(_z, row + 2, col, p1.data());
    grid_.GetValue(_z, row + 2, col + 1, p2.data());
    grid_.GetValue(_z, row + 2, col + 2, p3.data());
    CubicHermiteSpline<Grid3D::DATA_DIMENSION>(
        p0, p1, p2, p3, c - col, f3.data(), df3dc.data());

    // Interpolate vertically the interpolated value from each row and
    // compute the derivative along the columns.
    CubicHermiteSpline<Grid3D::DATA_DIMENSION>(f0, f1, f2, f3, r - row, f, dfdr);
    if (dfdc != nullptr) {
      // Interpolate vertically the derivative along the columns.
      CubicHermiteSpline<Grid3D::DATA_DIMENSION>(
          df0dc, df1dc, df2dc, df3dc, r - row, dfdc, nullptr);
    }
  }

  // The following two Evaluate overloads are needed for interfacing
  // with automatic differentiation. The first is for when a scalar
  // evaluation is done, and the second one is for when Jets are used.
  void Evaluate(const double& r, const double& c, double* f) const {
    Evaluate(r, c, f, nullptr, nullptr);
  }

  template <typename JetT>
  void Evaluate(const JetT& r, const JetT& c, JetT* f) const {
    double frc[Grid3D::DATA_DIMENSION];
    double dfdr[Grid3D::DATA_DIMENSION];
    double dfdc[Grid3D::DATA_DIMENSION];
    Evaluate(r.a, c.a, frc, dfdr, dfdc);
    for (int i = 0; i < Grid3D::DATA_DIMENSION; ++i) {
      f[i].a = frc[i];
      f[i].v = dfdr[i] * r.v + dfdc[i] * c.v;
    }
  }

 private:
  Grid3D& grid_;
  int _z = 0;
};

}  // namespace ceres

#endif  // CERES_PUBLIC_CUBIC_INTERPOLATOR_H_
