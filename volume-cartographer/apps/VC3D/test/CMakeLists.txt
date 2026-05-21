find_package(Qt6 REQUIRED COMPONENTS Test)

# Helper: same shape as vc_add_test in core/test, scoped to this directory.
function(vc3d_add_test)
    cmake_parse_arguments(T "" "NAME" "SOURCES;LIBS" ${ARGN})
    if(NOT T_SOURCES)
        set(T_SOURCES ${T_NAME}.cpp)
    endif()
    if(VC_ENABLE_ASAN OR VC_ENABLE_UBSAN OR VC_ENABLE_TSAN
       OR VC_ENABLE_LSAN OR VC_ENABLE_TYSAN OR VC_ENABLE_NSAN)
        list(APPEND T_SOURCES ${CMAKE_SOURCE_DIR}/misc/sanitizer_config.cpp)
    endif()
    add_executable(${T_NAME} ${T_SOURCES})
    target_link_libraries(${T_NAME} PRIVATE ${T_LIBS})
    target_include_directories(${T_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
    add_test(NAME ${T_NAME} COMMAND ${T_NAME})
endfunction()

vc3d_add_test(NAME test_segmentation_autosave_state
    LIBS doctest::doctest vc_core)

vc3d_add_test(NAME test_segmentation_intersection_invalidation
    SOURCES test_segmentation_intersection_invalidation.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../VCCollection.cpp
    LIBS Qt6::Core Qt6::Gui Qt6::Test vc_core)
