#pragma once

// App-wide UI/render constants. Single source of truth so values that must stay
// in sync (the global clock interval and the debounce deadlines expressed in
// ticks) can't drift apart.
namespace vc3d {

// Global render clock. ViewerManager owns the one QTimer firing every
// kGlobalTickMs; per-viewer debounce deadlines are expressed in whole ticks.
constexpr int kGlobalTickMs = 16;  // ~60fps

// How long after the last resize event before we do a full re-render.
constexpr int kResizeSettleMs = 140;

// How long to debounce intersection-overlay rebuilds.
constexpr int kIntersectionSettleMs = 50;

// Status-label refresh cadence (disk/RAM figures during background prefetch).
constexpr int kStatusRefreshMs = 1000;

// Whole ticks needed to cover `ms`, rounded up, min 0 for non-positive input.
constexpr int msToTicks(int ms) {
    return ms <= 0 ? 0 : (ms + kGlobalTickMs - 1) / kGlobalTickMs;
}

}  // namespace vc3d
