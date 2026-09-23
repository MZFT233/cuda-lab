# cmake/msvc.cmake -- locate MSVC + Windows SDK without requiring vcvars64.bat.
#
# Used via CMakePresets.json so that both the terminal and the VS Code CMake Tools
# extension can configure the project with an empty environment.
# nvcc needs a host compiler (cl.exe) on PATH; CMake finds cl.exe by absolute path
# here, and we put its directory on PATH so nvcc's own lookup succeeds too.

set(_vswhere "$ENV{ProgramFiles\(x86\)}/Microsoft Visual Studio/Installer/vswhere.exe")

if(NOT EXISTS "${_vswhere}")
  message(FATAL_ERROR "vswhere.exe not found - is Visual Studio / Build Tools installed?")
endif()

execute_process(
  COMMAND "${_vswhere}" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  OUTPUT_VARIABLE _vs_root
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _vs_root)
  message(FATAL_ERROR "no Visual Studio installation with the C++ toolset was found")
endif()

file(GLOB _msvc_dirs "${_vs_root}/VC/Tools/MSVC/*")
list(SORT _msvc_dirs COMPARE NATURAL ORDER DESCENDING)
list(GET _msvc_dirs 0 _msvc)
if(NOT IS_DIRECTORY "${_msvc}")
  message(FATAL_ERROR "MSVC toolset directory not found under ${_vs_root}/VC/Tools/MSVC")
endif()
file(TO_CMAKE_PATH "${_msvc}" _msvc)

set(_host "Hostx64/x64")
set(CMAKE_C_COMPILER   "${_msvc}/bin/${_host}/cl.exe"   CACHE FILEPATH "" FORCE)
set(CMAKE_CXX_COMPILER "${_msvc}/bin/${_host}/cl.exe"   CACHE FILEPATH "" FORCE)
set(CMAKE_LINKER       "${_msvc}/bin/${_host}/link.exe" CACHE FILEPATH "" FORCE)

# --- Windows SDK -------------------------------------------------------------
if(DEFINED ENV{WindowsSdkDir})
  set(_sdk "$ENV{WindowsSdkDir}")
else()
  foreach(_cand "E:/Windows Kits/10" "C:/Program Files (x86)/Windows Kits/10")
    if(IS_DIRECTORY "${_cand}") 
      set(_sdk "${_cand}")
      break()
    endif()
  endforeach()
endif()
if(NOT _sdk)
  message(FATAL_ERROR "Windows SDK root not found (looked for WindowsSdkDir, E:/Windows Kits/10)")
endif()

file(GLOB _sdk_vers "${_sdk}/Include/10.*")
list(SORT _sdk_vers COMPARE NATURAL ORDER DESCENDING)
list(GET _sdk_vers 0 _sdk_inc_dir)
get_filename_component(_sdk_ver "${_sdk_inc_dir}" NAME)
set(_sdk_inc "${_sdk}/Include/${_sdk_ver}")
set(_sdk_lib "${_sdk}/Lib/${_sdk_ver}")
if(NOT IS_DIRECTORY "${_sdk_lib}")
  message(FATAL_ERROR "Windows SDK libraries not found: ${_sdk_lib}")
endif()

# --- headers / libs (keep any inherited values) ------------------------------
set(_inc "${_msvc}/include")
foreach(_p ucrt um shared winrt cppwinrt)
  if(IS_DIRECTORY "${_sdk_inc}/${_p}")
    list(APPEND _inc "${_sdk_inc}/${_p}")
  endif()
endforeach()
set(ENV{INCLUDE} "${_inc}")

set(_lib "${_msvc}/lib/x64")
foreach(_p "ucrt/x64" "um/x64")
  if(IS_DIRECTORY "${_sdk_lib}/${_p}")
    list(APPEND _lib "${_sdk_lib}/${_p}")
  endif()
endforeach()
set(ENV{LIB} "${_lib}")

# --- make cl.exe / rc.exe / mt.exe reachable for nvcc and CMake's RC step ----
# NB: setting ENV{PATH} here only affects configure-time, not the build step
# (CMake captured its own copy of PATH at startup). nvcc therefore gets the host
# compiler pinned explicitly via -ccbin, which is what actually matters.
set(_sdk_bin "${_sdk}/bin/${_sdk_ver}/x64")
if(IS_DIRECTORY "${_sdk_bin}")
  set(ENV{PATH} "${_msvc}/bin/${_host};${_sdk_bin};$ENV{PATH}")
else()
  set(ENV{PATH} "${_msvc}/bin/${_host};$ENV{PATH}")
endif()

# nvcc needs cl.exe on PATH, or -ccbin. Pin it so an empty environment works.
set(CMAKE_CUDA_FLAGS_INIT "-ccbin \"${_msvc}/bin/${_host}\" ${CMAKE_CUDA_FLAGS_INIT}")

# The LINK step also runs with an empty environment, so pass the SDK/CRT lib
# directories explicitly instead of relying on the LIB variable.
set(_libpath_args "")
foreach(_l IN LISTS _lib)
  file(TO_CMAKE_PATH "${_l}" _l_fwd)
  string(APPEND _libpath_args " \"/LIBPATH:${_l_fwd}\"")
endforeach()
set(CMAKE_EXE_LINKER_FLAGS_INIT    "${CMAKE_EXE_LINKER_FLAGS_INIT}${_libpath_args}")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${CMAKE_SHARED_LINKER_FLAGS_INIT}${_libpath_args}")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "${CMAKE_MODULE_LINKER_FLAGS_INIT}${_libpath_args}")

message(STATUS "toolchain: MSVC ${_msvc}")
message(STATUS "toolchain: WinSDK ${_sdk_ver}")
message(STATUS "toolchain: nvcc -ccbin ${_msvc}/bin/${_host}")
message(STATUS "toolchain: linker libpaths${_libpath_args}")
