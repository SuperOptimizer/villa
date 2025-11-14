
add_library(vc_tracer GrowPatch.cpp
        GrowSurface.cpp
        GridStore.cpp
        NormalGridVolume.cpp
        normalgridtools.cpp
)
target_link_libraries(vc_tracer PUBLIC vc_core vc_ui Ceres::ceres Qt6::Core)
