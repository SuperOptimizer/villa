#pragma once

// Optimization/loss flags (bitmask)
inline constexpr int OPTIMIZE_ALL    = 1;
inline constexpr int SURF_LOSS       = 2;
inline constexpr int SPACE_LOSS      = 2;  // SURF and SPACE are never used together
inline constexpr int LOSS_3D_INDIRECT = 4;
inline constexpr int LOSS_ZLOC       = 8;
inline constexpr int FLAG_GEN0       = 16;
inline constexpr int LOSS_ON_SURF    = 32;
inline constexpr int LOSS_ON_NORMALS = 64;

// State flags (bitmask)
inline constexpr int STATE_UNUSED      = 0;
inline constexpr int STATE_LOC_VALID   = 1;
inline constexpr int STATE_PROCESSING  = 2;
inline constexpr int STATE_COORD_VALID = 4;
inline constexpr int STATE_FAIL        = 8;
inline constexpr int STATE_PHYS_ONLY   = 16;
