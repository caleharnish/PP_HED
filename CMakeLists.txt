cmake_minimum_required(VERSION 3.17.3)

project (PP_HED LANGUAGES CXX)

# set up options
option(PP_DEBUG "enable debugging" ON)
set(PP_SANITIZE "" CACHE STRING "-fsanitize compiler options")

# handle the sanitize options
if (PP_SANITIZE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=${PP_SANITIZE}")
  set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=${PP_SANITIZE}")
endif (PP_SANITIZE)

# and add the directories we care about
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(applications)
