#include "rsp_jit.hpp"
#include <utility>

using namespace std;

#define JIT_REGISTER_SELF JIT_V0
#define JIT_REGISTER_STATE JIT_V1
#define JIT_REGISTER_MODE JIT_R1
#define JIT_REGISTER_NEXT_PC JIT_R0
#define JIT_FRAME_SIZE 256

namespace RSP
{
namespace JIT
{
CPU::CPU()
{
	cleanup_jit_states.reserve(16 * 1024);
	init_jit("RSP");
	init_jit_thunks();
}

CPU::~CPU()
{
	for (auto *_jit : cleanup_jit_states)
	{
		jit_clear_state();
		jit_destroy_state();
	}
	finish_jit();
}

void CPU::invalidate_imem()
{
	for (unsigned i = 0; i < CODE_BLOCKS; i++)
		if (memcmp(cached_imem + i * CODE_BLOCK_WORDS, state.imem + i * CODE_BLOCK_WORDS, CODE_BLOCK_SIZE))
			state.dirty_blocks |= (0x3 << i) >> 1;
}

void CPU::invalidate_code()
{
	if (!state.dirty_blocks)
		return;

	for (unsigned i = 0; i < CODE_BLOCKS; i++)
	{
		if (state.dirty_blocks & (1 << i))
		{
			memset(blocks + i * CODE_BLOCK_WORDS, 0, CODE_BLOCK_WORDS * sizeof(blocks[0]));
			memcpy(cached_imem + i * CODE_BLOCK_WORDS, state.imem + i * CODE_BLOCK_WORDS, CODE_BLOCK_SIZE);
		}
	}

	state.dirty_blocks = 0;
}

// Need super-fast hash here.
uint64_t CPU::hash_imem(unsigned pc, unsigned count) const
{
	size_t size = count;

	// FNV-1.
	const auto *data = state.imem + pc;
	uint64_t h = 0xcbf29ce484222325ull;
	h = (h * 0x100000001b3ull) ^ pc;
	h = (h * 0x100000001b3ull) ^ count;
	for (size_t i = 0; i < size; i++)
		h = (h * 0x100000001b3ull) ^ data[i];
	return h;
}

unsigned CPU::analyze_static_end(unsigned pc, unsigned end)
{
	// Scans through IMEM and finds the logical "end" of the instruction stream.
	unsigned max_static_pc = pc;
	unsigned count = end - pc;

	for (unsigned i = 0; i < count; i++)
	{
		uint32_t instr = state.imem[pc + i];
		uint32_t type = instr >> 26;
		uint32_t target;

		bool forward_goto;
		if (pc + i + 1 >= max_static_pc)
		{
			forward_goto = false;
			max_static_pc = pc + i + 1;
		}
		else
			forward_goto = true;

		// VU
		if ((instr >> 25) == 0x25)
			continue;

		switch (type)
		{
		case 000:
			switch (instr & 63)
			{
			case 010:
				// JR always terminates either by returning or exiting.
				// We execute the next instruction via delay slot and exit.
				// Unless we can branch past the JR
				// (max_static_pc will be higher than expected),
				// this will be the static end.
				if (!forward_goto)
				{
					max_static_pc = max(pc + i + 2, max_static_pc);
					goto end;
				}
				break;

			case 015:
				// BREAK always terminates.
				if (!forward_goto)
					goto end;
				break;

			default:
				break;
			}
			break;

		case 001: // REGIMM
			switch ((instr >> 16) & 31)
			{
			case 000: // BLTZ
			case 001: // BGEZ
			case 021: // BGEZAL
			case 020: // BLTZAL
				target = (pc + i + 1 + instr) & 0x3ff;
				if (target >= pc && target < end) // goto
					max_static_pc = max(max_static_pc, target + 1);
				break;

			default:
				break;
			}
			break;

		case 002:
			// J is resolved by goto.
			target = instr & 0x3ff;
			if (target >= pc && target < end) // goto
			{
				// J is a static jump, so if we aren't branching
				// past this instruction and we're branching backwards,
				// we can end the block here.
				if (!forward_goto && target < end)
				{
					max_static_pc = max(pc + i + 2, max_static_pc);
					goto end;
				}
				else
					max_static_pc = max(max_static_pc, target + 1);
			}
			else if (!forward_goto)
			{
				// If we have static branch outside our block,
				// we terminate the block.
				max_static_pc = max(pc + i + 2, max_static_pc);
				goto end;
			}
			break;

		case 004: // BEQ
		case 005: // BNE
		case 006: // BLEZ
		case 007: // BGTZ
			target = (pc + i + 1 + instr) & 0x3ff;
			if (target >= pc && target < end) // goto
				max_static_pc = max(max_static_pc, target + 1);
			break;

		default:
			break;
		}
	}

end:
	unsigned ret = min(max_static_pc, end);
	return ret;
}

extern "C"
{
	static Func RSP_ENTER(void *cpu, unsigned pc)
	{
		return static_cast<CPU *>(cpu)->get_jit_block(pc);
	}
}

void CPU::init_jit_thunks()
{
	jit_state_t *_jit = jit_new_state();

	jit_prolog();

	// Saves registers from C++ code.
	jit_frame(JIT_FRAME_SIZE);
	auto *self = jit_arg();
	auto *state = jit_arg();

	// These registers remain fixed and all called thunks will poke into these registers as necessary.
	jit_getarg(JIT_REGISTER_SELF, self);
	jit_getarg(JIT_REGISTER_STATE, state);
	jit_ldxi_i(JIT_REGISTER_NEXT_PC, JIT_REGISTER_STATE, offsetof(CPUState, pc));
	jit_movi(JIT_REGISTER_MODE, MODE_ENTER);

	auto *entry_label = jit_indirect();

	jit_prepare();
	jit_pushargr(JIT_REGISTER_SELF);
	jit_pushargr(JIT_REGISTER_NEXT_PC);
	jit_finishi(reinterpret_cast<jit_pointer_t>(RSP_ENTER));
	jit_retval(JIT_REGISTER_NEXT_PC);

	// Jump to thunk.
	jit_jmpr(JIT_REGISTER_NEXT_PC);

	// Keep going.
	jit_patch_at(jit_bnei(JIT_REGISTER_MODE, MODE_ENTER), entry_label);

	// When we want to return, JIT thunks will jump here.
	auto *return_label = jit_indirect();

	// Save PC.
	jit_stxi_i(offsetof(CPUState, pc), JIT_REGISTER_STATE, JIT_REGISTER_NEXT_PC);

	// Return status. This register is considered common for all thunks.
	jit_retr(JIT_REGISTER_MODE);

	thunks.enter_frame = reinterpret_cast<int (*)(void *, void *)>(jit_emit());
	thunks.enter_thunk = jit_address(entry_label);
	thunks.return_thunk = jit_address(return_label);

	printf(" === DISASM ===\n");
	jit_disassemble();
	printf(" === END DISASM ===\n");
	cleanup_jit_states.push_back(_jit);
}

Func CPU::get_jit_block(uint32_t pc)
{
	pc &= IMEM_SIZE - 1;
	uint32_t word_pc = pc >> 2;
	auto &block = blocks[word_pc];

	if (!block)
	{
		unsigned end = (pc + (CODE_BLOCK_SIZE * 2)) >> CODE_BLOCK_SIZE_LOG2;
		end <<= CODE_BLOCK_SIZE_LOG2 - 2;
		end = min(end, unsigned(IMEM_SIZE >> 2));
		end = analyze_static_end(word_pc, end);

		uint64_t hash = hash_imem(word_pc, end - word_pc);
		auto itr = cached_blocks[word_pc].find(hash);
		if (itr != cached_blocks[word_pc].end())
			block = itr->second;
		else
			block = jit_region(hash, word_pc, end - word_pc);
	}
	return block;
}

int CPU::enter(uint32_t pc)
{
	// Top level enter.
	state.pc = pc;
	return thunks.enter_frame(this, &state);
}

Func CPU::jit_region(uint64_t hash, unsigned pc, unsigned count)
{
	jit_state_t *_jit = jit_new_state();

	jit_prolog();
	jit_tramp(JIT_FRAME_SIZE);

	jit_movi(JIT_R0, 10);
	jit_stxi_i(offsetof(CPUState, sr) + 4, JIT_REGISTER_STATE, JIT_R0);
	jit_movi(JIT_R0, 20);
	jit_stxi_i(offsetof(CPUState, sr) + 8, JIT_REGISTER_STATE, JIT_R0);
	jit_movi(JIT_R0, 30);
	jit_stxi_i(offsetof(CPUState, sr) + 12, JIT_REGISTER_STATE, JIT_R0);
	jit_movi(JIT_R0, 40);
	jit_stxi_i(offsetof(CPUState, sr) + 16, JIT_REGISTER_STATE, JIT_R0);
	jit_movi(JIT_REGISTER_MODE, MODE_BREAK);
	jit_movi(JIT_REGISTER_NEXT_PC, 4);
	auto *jmp = jit_jmpi();
	jit_patch_abs(jmp, thunks.return_thunk);

	auto ret = reinterpret_cast<Func>(jit_emit());

	printf(" === DISASM ===\n");
	jit_disassemble();
	printf(" === DISASM END ===\n\n");
	cleanup_jit_states.push_back(_jit);
	return ret;
}

ReturnMode CPU::run()
{
	for (;;)
	{
		invalidate_code();

		int ret = enter(state.pc);
		switch (ret)
		{
		case MODE_BREAK:
			*state.cp0.cr[CP0_REGISTER_SP_STATUS] |= SP_STATUS_BROKE | SP_STATUS_HALT;
			if (*state.cp0.cr[CP0_REGISTER_SP_STATUS] & SP_STATUS_INTR_BREAK)
				*state.cp0.irq |= 1;
#ifndef PARALLEL_INTEGRATION
			print_registers();
#endif
			return MODE_BREAK;

		case MODE_CHECK_FLAGS:
		case MODE_DMA_READ:
			return static_cast<ReturnMode>(ret);

		default:
			break;
		}
	}
}
} // namespace JIT
} // namespace RSP