option(VC_BUILD_Z5 "Build in-source z5 header only library" on)
message("SHOULD WE BUILD Z5" $VC_BUILD_Z5)
if(VC_BUILD_Z5)
  # Declare the project
  FetchContent_Declare(
      z5
      GIT_REPOSITORY https://github.com/spacegaier/z5.git
      GIT_TAG 60a5ec4
      # GIT_REPOSITORY https://github.com/constantinpape/z5.git
      # GIT_TAG f118d95
  )

  # Populate the project but exclude from all
  FetchContent_GetProperties(z5)
  if(NOT z5_POPULATED)
    FetchContent_Populate(z5)
  endif()
  add_library(z5 INTERFACE)
  find_package(Blosc REQUIRED)
  target_include_directories(z5 INTERFACE ${z5_SOURCE_DIR}/include)
  target_compile_definitions(z5 INTERFACE WITH_BLOSC)
  target_link_libraries(z5 INTERFACE Blosc::blosc)
  # install(TARGETS z5
  #       EXPORT z5
  #       LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  #       ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  #       RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  #       INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  #       )
  # install(EXPORT z5
  #       FILE z5Targets.cmake
  #       NAMESPACE z5::
  #       DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/z5
  # )
else()
  find_package(z5 REQUIRED)
endif()
