#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Benchtools::Core" for configuration "Release"
set_property(TARGET Benchtools::Core APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Benchtools::Core PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libCore.a"
  )

list(APPEND _cmake_import_check_targets Benchtools::Core )
list(APPEND _cmake_import_check_files_for_Benchtools::Core "${_IMPORT_PREFIX}/lib/libCore.a" )

# Import target "Benchtools::Logger" for configuration "Release"
set_property(TARGET Benchtools::Logger APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Benchtools::Logger PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLogger.a"
  )

list(APPEND _cmake_import_check_targets Benchtools::Logger )
list(APPEND _cmake_import_check_files_for_Benchtools::Logger "${_IMPORT_PREFIX}/lib/libLogger.a" )

# Import target "Benchtools::Timer" for configuration "Release"
set_property(TARGET Benchtools::Timer APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Benchtools::Timer PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libTimer.a"
  )

list(APPEND _cmake_import_check_targets Benchtools::Timer )
list(APPEND _cmake_import_check_files_for_Benchtools::Timer "${_IMPORT_PREFIX}/lib/libTimer.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
