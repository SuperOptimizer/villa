option(VC_BUILD_Z5 "Build in-source z5 header only library" on)
message("SHOULD WE BUILD Z5" $VC_BUILD_Z5)
if(VC_BUILD_Z5)
  # Declare the project
  FetchContent_Declare(
      z5
      GIT_REPOSITORY https://github.com/constantinpape/z5.git
      GIT_TAG f118d95
  )

  # Populate the project but exclude from all
  FetchContent_GetProperties(z5)
  if(NOT z5_POPULATED)
    FetchContent_Populate(z5)
  endif()
  add_library(z5 INTERFACE)
  target_include_directories(z5 INTERFACE ${z5_SOURCE_DIR}/include)
else()
  find_package(z5 REQUIRED)
endif()
