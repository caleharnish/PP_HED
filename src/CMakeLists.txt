target_compile_features(pp_hed PUBLIC cxx_std_23)
target_include_directories(pp_hed PUBLIC ${CMAKE_SOURCE_DIR}/include)

if (PP_DEBUG)
  target_compile_definitions(pp_hed PUBLIC -DPP_ENABLE_DEBUG)
endif (PP_DEBUG)
