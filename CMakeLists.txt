cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 99)
project(parallel-rsp LANGUAGES CXX C)

if (CMAKE_COMPILER_IS_GNUCXX OR (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang"))
    set(PARALLEL_RSP_CXX_FLAGS -Wall -Wextra -Wno-missing-field-initializers -Wno-empty-body -ffast-math -Wno-unused-parameter)
elseif (MSVC)
    set(PARALLEL_RSP_CXX_FLAGS /D_CRT_SECURE_NO_WARNINGS /wd4267 /wd4244 /wd4309 /wd4005 /MP /DNOMINMAX)
endif()

add_library(parallel-rsp STATIC
		main.cpp
		rsp/vfunctions.cpp
		rsp.cpp rsp.hpp
		rsp_jit.cpp rsp_jit.hpp
		debug_jit.cpp debug_jit.hpp
		rsp/ls.cpp rsp/pipeline.h
		rsp/reciprocal.cpp rsp/reciprocal.h
		rsp_1.1.h
		rsp/cp0.cpp rsp/cp2.cpp
		arch/x86_64/rsp/rsp_core.cpp
		arch/x86_64/rsp/clamp.h
		arch/x86_64/rsp/rsp.h
		arch/x86_64/rsp/rsp_impl.h
		arch/x86_64/rsp/vcr.h
		arch/x86_64/rsp/vabs.h
		arch/x86_64/rsp/vadd.h
		arch/x86_64/rsp/vaddc.h
		arch/x86_64/rsp/vand.h
		arch/x86_64/rsp/vch.h
		arch/x86_64/rsp/vcl.h
		arch/x86_64/rsp/vcr.h
		arch/x86_64/rsp/vcmp.h
		arch/x86_64/rsp/vdivh.h
		arch/x86_64/rsp/vmac.h
		arch/x86_64/rsp/vmov.h
		arch/x86_64/rsp/vmrg.h
		arch/x86_64/rsp/vmudh.h
		arch/x86_64/rsp/vmul.h
		arch/x86_64/rsp/vmull.h
		arch/x86_64/rsp/vmulh.h
		arch/x86_64/rsp/vmuln.h
		arch/x86_64/rsp/vor.h
		arch/x86_64/rsp/vrcpsq.h
		arch/x86_64/rsp/vrsq.h
		arch/x86_64/rsp/vsub.h
		arch/x86_64/rsp/vsubc.h
		arch/x86_64/rsp/vxor.h
		arch/x86_64/rsp/vmulm.h)

target_include_directories(parallel-rsp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(parallel-rsp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/arch/x86_64/rsp)
target_compile_options(parallel-rsp PRIVATE ${PARALLEL_RSP_CXX_FLAGS})

option(PARALLEL_RSP_BAKED_LIGHTNING "Use built-in Lightning." OFF)

target_compile_definitions(parallel-rsp PUBLIC PARALLEL_RSP_LIGHTNING)
if (PARALLEL_RSP_BAKED_LIGHTNING)
	add_library(lightning STATIC
			lightning/lib/jit_disasm.c
			lightning/lib/jit_memory.c
			lightning/lib/jit_names.c
			lightning/lib/jit_note.c
			lightning/lib/jit_print.c
			lightning/lib/jit_size.c
			lightning/lib/lightning.c)
	target_include_directories(lightning PUBLIC
			${CMAKE_CURRENT_SOURCE_DIR}/lightning/include)
endif()
target_link_libraries(parallel-rsp PUBLIC lightning)
target_compile_definitions(parallel-rsp PUBLIC DEBUG_JIT)

if (NOT WIN32)
	target_link_libraries(parallel-rsp PRIVATE dl)
endif()

add_executable(rsp-runner main.cpp)
target_link_libraries(rsp-runner PRIVATE parallel-rsp)
target_compile_options(rsp-runner PRIVATE ${PARALLEL_RSP_CXX_FLAGS})
set_target_properties(rsp-runner PROPERTIES LINK_FLAGS "-rdynamic")

option(PARALLEL_RSP_TESTS "Enable unit tests for Lightning CPU." ON)

if (PARALLEL_RSP_TESTS)
	enable_testing()
	add_custom_target(rsp-tests ALL)

	add_custom_target(rsp-test-crt
			COMMAND ${CMAKE_MAKE_PROGRAM} crt-obj -j1
			WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/debug-toolchain)

	function(rsp_add_test NAME)
		add_custom_target(rsp-test-binary-${NAME}
				COMMAND ${CMAKE_MAKE_PROGRAM} RSP_TEST=${NAME} -j1
				WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/debug-toolchain)
		add_dependencies(rsp-test-binary-${NAME} rsp-test-crt)
		add_dependencies(rsp-tests rsp-test-binary-${NAME})

		add_test(NAME rsp-test-${NAME}
				COMMAND $<TARGET_FILE:rsp-runner>
				${CMAKE_CURRENT_SOURCE_DIR}/debug-toolchain/${NAME}.global.bin
				${CMAKE_CURRENT_SOURCE_DIR}/debug-toolchain/${NAME}.bin)
	endfunction()

	rsp_add_test(add)
	rsp_add_test(sub)
	rsp_add_test(or)
	rsp_add_test(and)
	rsp_add_test(xor)
	rsp_add_test(nor)
	rsp_add_test(slt)
	rsp_add_test(sltu)
	rsp_add_test(addi)
	rsp_add_test(slti)
	rsp_add_test(sltiu)
	rsp_add_test(andi)
	rsp_add_test(ori)
	rsp_add_test(xori)
	rsp_add_test(lui)
endif()
