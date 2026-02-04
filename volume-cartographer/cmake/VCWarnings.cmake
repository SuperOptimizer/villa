# VCWarnings.cmake - Extensive compiler warnings for development
# Enable with: cmake -DVC_DEVELOPER_WARNINGS=ON ..

# Base warnings (GCC & Clang)
set(project_warnings
    -Wall
    -Wextra
    -pedantic
)

# Common useful warnings
list(APPEND project_warnings
    -Wattributes
    -Wcast-align
    -Wcast-qual
    -Wchar-subscripts
    -Wcomment
    -Wconversion
    -Wdelete-incomplete
    -Wdelete-non-virtual-dtor
    -Wenum-compare
    -Wmain
    -Wmissing-field-initializers
    -Wmissing-noreturn
    -Wold-style-cast
    -Woverloaded-virtual
    -Wpointer-arith
    -Wtautological-compare
    -Wundef
    -Wuninitialized
    -Wunreachable-code
    -Wunused
    -Wvla
    -Wunused-parameter
)

# Clang-specific warnings
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    # Enable -Weverything, then selectively disable problematic ones
    list(APPEND project_warnings -Weverything)

    # ========== Compatibility warnings (we use C++23) ==========
    list(APPEND project_warnings
        -Wno-c++98-compat
        -Wno-c++98-compat-pedantic
        -Wno-c++98-compat-local-type-template-args
        -Wno-c++98-compat-unnamed-type-template-args
        -Wno-c++98-compat-extra-semi
        -Wno-c++11-compat
        -Wno-c++14-compat
        -Wno-c++17-compat
        -Wno-c++20-compat
        -Wno-pre-c++14-compat
        -Wno-pre-c++17-compat
        -Wno-pre-c++20-compat
        -Wno-pre-c++23-compat
        -Wno-pre-c++26-compat
    )

    # ========== Noisy/unhelpful warnings ==========
    list(APPEND project_warnings
        # Structure layout (implementation detail, not a bug)
        -Wno-padded

        # Common C++ patterns that trigger these
        -Wno-global-constructors
        -Wno-exit-time-destructors

        # Switch statements - having default is fine
        -Wno-covered-switch-default
        -Wno-switch-enum

        # Float comparison - sometimes intentional (0.0f checks, etc)
        -Wno-float-equal

        # C-ism, not relevant in C++
        -Wno-missing-prototypes
        -Wno-missing-variable-declarations

        # Documentation - only useful if using Doxygen consistently
        -Wno-documentation
        -Wno-documentation-unknown-command

        # Shadow warnings - too noisy with lambdas and RAII patterns
        -Wno-shadow
        -Wno-shadow-field
        -Wno-shadow-field-in-constructor
        -Wno-shadow-uncaptured-local

        # Implicit conversions - extremely noisy, -Wconversion is enough
        -Wno-implicit-int-conversion
        -Wno-implicit-float-conversion
        -Wno-implicit-int-float-conversion
        -Wno-shorten-64-to-32
        -Wno-sign-conversion
        -Wno-double-promotion

        # Macro-related
        -Wno-disabled-macro-expansion
        -Wno-reserved-macro-identifier
        -Wno-reserved-identifier

        # Style nits
        -Wno-newline-eof
        -Wno-extra-semi-stmt
        -Wno-extra-semi

        # C++20 unsafe buffer warnings (very noisy, not practical)
        -Wno-unsafe-buffer-usage
        -Wno-unsafe-buffer-usage-in-libc-call
        -Wno-unsafe-buffer-usage-in-container

        # Weak vtables warning (header-only classes)
        -Wno-weak-vtables

        # Packed structs
        -Wno-packed

        # Suggest braces around init (style preference)
        -Wno-missing-braces

        # Unused templates (common in generic code)
        -Wno-unused-template
        -Wno-unused-member-function

        # CTAD warnings
        -Wno-ctad-maybe-unsupported

        # Undefined reinterpret_cast (low-level code needs this)
        -Wno-undefined-reinterpret-cast

        # Format strings with non-literal (we use runtime formats sometimes)
        -Wno-format-nonliteral

        # Anonymous structs/unions (C11/C++11 extension, widely supported)
        -Wno-nested-anon-types
        -Wno-gnu-anonymous-struct

        # NaN/infinity warnings (we use fast-math intentionally for performance)
        -Wno-nan-infinity-disabled

        # Switch without default is fine when cases are exhaustive
        -Wno-switch-default

        # Thread safety negative capabilities (complex annotation system)
        -Wno-thread-safety-negative

        # NRVO warnings (optimizer hint, not a bug)
        -Wno-return-std-move-in-c++11
        -Wno-nrvo

        # Unused lambda captures (common with [this] or [&] patterns)
        -Wno-unused-lambda-capture
    )

    # ========== Keep these useful Clang warnings ==========
    # (They're in -Weverything and we want them)
    # -Wthread-safety-analysis
    # -Wself-assign / -Wself-move
    # -Wmove / -Wpessimizing-move / -Wredundant-move
    # -Winfinite-recursion
    # -Wloop-analysis
    # -Wunreachable-code-*
endif()

# GCC-specific warnings
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND project_warnings
        -Wlogical-op
        -Wduplicated-cond
        -Wduplicated-branches
        -Wnull-dereference
        -Wuseless-cast
        -Wsuggest-override
    )
endif()

add_compile_options(${project_warnings})

message(STATUS "Developer warnings enabled (-Weverything with sensible exclusions)")
