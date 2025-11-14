
add_subdirectory(apps)
add_subdirectory(core)
add_subdirectory(misc)
add_subdirectory(tracer)
add_subdirectory(ui)



if(VC_INSTALL_LIBS)
    install(
            TARGETS vc_core
            COMPONENT "Libraries"
            EXPORT "${targets_export_name}"
            ARCHIVE DESTINATION "lib"
            LIBRARY DESTINATION "lib"
            INCLUDES DESTINATION "${include_install_dir}"
            RUNTIME DESTINATION "bin"
    )

    install(
            TARGETS vc_ui
            COMPONENT "Libraries"
            EXPORT "${targets_export_name}"
            ARCHIVE DESTINATION "lib"
            LIBRARY DESTINATION "lib"
            INCLUDES DESTINATION "${include_install_dir}"
            RUNTIME DESTINATION "bin"
    )

    install(
            DIRECTORY "include"
            DESTINATION "${include_install_dir}"
            COMPONENT "Libraries"
            FILES_MATCHING REGEX ".*\\.(h|hpp)$"
    )
endif()
