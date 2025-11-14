
qt6_wrap_cpp(metrics_moc VCCollection.hpp)
add_library(vc_ui
        VCCollection.cpp
        UDataManipulateUtils.cpp
        surface_metrics.cpp ${metrics_moc})
target_link_libraries(vc_ui PUBLIC vc_core Qt6::Core Qt6::Gui Qt6::Widgets)
