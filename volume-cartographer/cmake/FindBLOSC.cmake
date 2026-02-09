# FindBLOSC.cmake
#
# Finds the Blosc library. This module defines:
#   - BLOSC_INCLUDE_DIR, directory containing headers
#   - BLOSC_LIBRARIES, the Blosc library path
#   - BLOSC_FOUND, whether Blosc has been found

# Find header files
if(BLOSC_SEARCH_HEADER_PATHS)
  find_path(
      BLOSC_INCLUDE_DIR blosc.h
      PATHS ${BLOSC_SEARCH_HEADER_PATHS}
      NO_DEFAULT_PATH
  )
else()
  find_path(BLOSC_INCLUDE_DIR blosc.h)
endif()

# Find library
if(BLOSC_SEARCH_LIB_PATH)
  find_library(
      BLOSC_LIBRARIES NAMES blosc
      PATHS ${BLOSC_SEARCH_LIB_PATH}$
      NO_DEFAULT_PATH
  )
else()
  find_library(BLOSC_LIBRARIES NAMES blosc)
endif()

if(BLOSC_INCLUDE_DIR AND BLOSC_LIBRARIES)
  message(STATUS "Found Blosc: ${BLOSC_LIBRARIES}")
  set(BLOSC_FOUND TRUE)
else()
  set(BLOSC_FOUND FALSE)
endif()

if(BLOSC_FIND_REQUIRED AND NOT BLOSC_FOUND)
  message(FATAL_ERROR "Could not find the Blosc library.")
endif()
