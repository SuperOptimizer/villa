add_library(libs3 STATIC libs3.c)
target_include_directories(libs3 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(libs3 PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  C_STANDARD 23
  C_STANDARD_REQUIRED ON
  OUTPUT_NAME s3vc)   # avoid clashing with any system libs3
find_package(CURL REQUIRED)
target_link_libraries(libs3 PRIVATE CURL::libcurl Threads::Threads)
