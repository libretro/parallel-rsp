#ifndef RSP_JIT_HPP__
#define RSP_JIT_HPP__

#include <memory>
#include <stdint.h>
#include <string.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "rsp_op.hpp"
#include "state.hpp"

extern "C"
{
#include <lightning.h>
}

namespace RSP
{
namespace JIT
{
using Func = jit_pointer_t;

enum ReturnMode
{
	MODE_ENTER = 0,
	MODE_CONTINUE = 1,
	MODE_BREAK = 2,
	MODE_DMA_READ = 3,
	MODE_CHECK_FLAGS = 4
};

class alignas(64) CPU
{
public:
	CPU();

	~CPU();

	CPU(CPU &&) = delete;

	void operator=(CPU &&) = delete;

	void set_dmem(uint32_t *dmem)
	{
		state.dmem = dmem;
	}

	void set_imem(uint32_t *imem)
	{
		state.imem = imem;
	}

	void set_rdram(uint32_t *rdram)
	{
		state.rdram = rdram;
	}

	void invalidate_imem();

	CPUState &get_state()
	{
		return state;
	}

	ReturnMode run();

	void print_registers();

	Func get_jit_block(uint32_t pc);

private:
	CPUState state;
	Func blocks[IMEM_WORDS] = {};

	void invalidate_code();

	uint64_t hash_imem(unsigned pc, unsigned count) const;

	alignas(64) uint32_t cached_imem[IMEM_WORDS] = {};

	std::unordered_map<uint64_t, Func> cached_blocks[IMEM_WORDS];

	Func jit_region(uint64_t hash, unsigned pc, unsigned count);

	int enter(uint32_t pc);

	std::vector<jit_state_t *> cleanup_jit_states;

	void init_jit_thunks();

	struct
	{
		int (*enter_frame)(void *self, void *state) = nullptr;

		Func enter_thunk = nullptr;
		Func return_thunk = nullptr;
	} thunks;

	unsigned analyze_static_end(unsigned pc, unsigned end);
};
} // namespace JIT
} // namespace RSP

#endif