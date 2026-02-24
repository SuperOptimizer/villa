# ── qt_utils ──────────────────────────────────────────────────────────────
find_package(Qt6 REQUIRED COMPONENTS Widgets Core Gui)

set(CMAKE_AUTOMOC ON)

add_library(vc_qt_utils STATIC
    src/RangeSlider.cpp
    src/CollapsibleSettingsGroup.cpp
    src/ProgressUtil.cpp
    src/ConsoleOutputWidget.cpp
    src/JsonProfileEditor.cpp
    src/DropdownChecklistButton.cpp
    src/Keybinds.cpp
    src/WindowRangeWidget.cpp
)

target_include_directories(vc_qt_utils PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(vc_qt_utils PUBLIC
    Qt6::Widgets
    Qt6::Core
    Qt6::Gui
)

target_compile_options(vc_qt_utils PRIVATE -w)
