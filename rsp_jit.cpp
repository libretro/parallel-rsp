#include "rsp_jit.hpp"
#include <utility>
#include <assert.h>

using namespace std;

// We're only guaranteed 3 V registers (x86).
#define JIT_REGISTER_SELF JIT_V0
#define JIT_REGISTER_STATE JIT_V1
#define JIT_REGISTER_DMEM JIT_V2

#define JIT_REGISTER_MODE JIT_R1
#define JIT_REGISTER_NEXT_PC JIT_R0

// Freely used to implement instructions.
#define JIT_REGISTER_TMP0 JIT_R0
#define JIT_REGISTER_TMP1 JIT_R1

// We're only guaranteed 3 R registers (x86).
#define JIT_REGISTER_COND_BRANCH_TAKEN JIT_R(JIT_R_NUM - 1)
#define JIT_FRAME_SIZE 256

namespace RSP
{
namespace JIT
{
static const char *reg_names[32] = {
	"zero", "at", "v0", "v1", "a0", "a1", "a2", "a3", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7",
	"s0",   "s1", "s2", "s3", "s4", "s5", "s6", "s7", "t8", "t9", "k0", "k1", "gp", "sp", "s8", "ra",
};
#define NAME(reg) reg_names[reg]

CPU::CPU()
{
	cleanup_jit_states.reserve(16 * 1024);
	init_jit("RSP");
	init_jit_thunks();
}

CPU::~CPU()
{
	for (auto *_jit : cleanup_jit_states)
		jit_destroy_state();
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
	// A logical end of the instruction stream is where execution must terminate.
	// If we have forward branches into this block, i.e. gotos, they extend the execution stream.
	// However, we cannot execute beyond end.
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
			case 011:
				// JR and JALR always terminate execution of the block.
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
				// TODO/Optimization: Handle static branch case where $0 is used.
				target = (pc + i + 1 + instr) & 0x3ff;
				if (target >= pc && target < end) // goto
					max_static_pc = max(max_static_pc, target + 1);
				break;

			default:
				break;
			}
			break;

		case 002: // J
		case 003: // JAL
			// J is resolved by goto. Same with JAL if call target happens to be inside the block.
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
			// TODO/Optimization: Handle static branch case where $0 is used.
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
	jit_ldxi(JIT_REGISTER_DMEM, JIT_REGISTER_STATE, offsetof(CPUState, dmem));
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
	jit_clear_state();
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

void CPU::jit_end_of_block(jit_state_t *_jit, uint32_t pc, const CPU::InstructionInfo &last_info)
{
	// If we run off the end of a block with a pending delay slot, we need to move it to CPUState.
	// We always branch to the next PC, and the delay slot will be handled after the first instruction in next block.
	auto *forward = jit_forward();
	if (last_info.branch)
	{
		if (last_info.conditional)
			jit_patch_at(jit_beqi(JIT_REGISTER_COND_BRANCH_TAKEN, 0), forward);

		if (last_info.indirect)
			jit_ldxi_i(JIT_REGISTER_TMP0, JIT_REGISTER_STATE, offsetof(CPUState, sr) + 4 * last_info.branch_target);
		else
			jit_movi(JIT_REGISTER_TMP0, last_info.branch_target);
		jit_stxi_i(offsetof(CPUState, branch_target), JIT_REGISTER_STATE, JIT_REGISTER_TMP0);
		jit_movi(JIT_REGISTER_TMP0, 1);
		jit_stxi_i(offsetof(CPUState, has_delay_slot), JIT_REGISTER_STATE, JIT_REGISTER_TMP0);
	}

	jit_link(forward);
	jit_movi(JIT_REGISTER_NEXT_PC, pc);
	jit_patch_abs(jit_jmpi(), thunks.enter_thunk);
}

void CPU::jit_handle_delay_slot(jit_state_t *_jit, const InstructionInfo &last_info,
                                jit_node_t **local_targets, uint32_t base_pc, uint32_t end_pc)
{
	if (last_info.conditional)
	{
		if (!last_info.indirect && last_info.branch_target >= base_pc && last_info.branch_target < end_pc)
		{
			jit_patch_at(jit_bnei(JIT_REGISTER_COND_BRANCH_TAKEN, 0), local_targets[(last_info.branch_target - base_pc) >> 2]);
		}
		else
		{
			auto *no_branch = jit_bnei(JIT_REGISTER_COND_BRANCH_TAKEN, 0);
			if (last_info.indirect)
				jit_ldxi_i(JIT_REGISTER_NEXT_PC, JIT_REGISTER_STATE, offsetof(CPUState, sr) + 4 * last_info.branch_target);
			else
				jit_movi(JIT_REGISTER_NEXT_PC, last_info.branch_target);
			jit_patch_abs(jit_jmpi(), thunks.enter_thunk);
			jit_patch(no_branch);
		}
	}
	else
	{
		if (!last_info.indirect && last_info.branch_target >= base_pc && last_info.branch_target < end_pc)
		{
			jit_patch_at(jit_jmpi(), local_targets[(last_info.branch_target - base_pc) >> 2]);
		}
		else
		{
			if (last_info.indirect)
				jit_ldxi_i(JIT_REGISTER_NEXT_PC, JIT_REGISTER_STATE,offsetof(CPUState, sr) + 4 * last_info.branch_target);
			else
				jit_movi(JIT_REGISTER_NEXT_PC, last_info.branch_target);
			jit_patch_abs(jit_jmpi(), thunks.enter_thunk);
		}
	}
}

void CPU::jit_exit(jit_state_t *_jit, uint32_t pc, const InstructionInfo &last_info, ReturnMode mode, bool first_instruction)
{
	if (first_instruction)
	{
		// Need to consider that we need to move delay slot to PC.
		jit_ldxi_i(JIT_REGISTER_TMP0, JIT_REGISTER_STATE, offsetof(CPUState, has_delay_slot));

		auto *latent_delay_slot = jit_forward();
		jit_patch_at(jit_bnei(JIT_REGISTER_TMP0, 0), latent_delay_slot);

		// Common case.
		// Immediately exit.
		jit_movi(JIT_REGISTER_MODE, mode);
		jit_movi(JIT_REGISTER_NEXT_PC, pc + 4);
		auto *jmp = jit_jmpi();
		jit_patch_abs(jmp, thunks.return_thunk);

		// If we had a latent delay slot, we handle it here.
		jit_link(latent_delay_slot);
		// We cannot execute a branch inside a delay slot, so just assume we do not have to chain together these.
		// We could technically handle it, but it gets messy (and it's illegal MIPS), so don't bother.
		jit_movi(JIT_REGISTER_NEXT_PC, 0);
		jit_stxi_i(offsetof(CPUState, has_delay_slot), JIT_REGISTER_STATE, JIT_REGISTER_NEXT_PC);
		jit_movi(JIT_REGISTER_MODE, mode);
		jit_ldxi_i(JIT_REGISTER_NEXT_PC, JIT_REGISTER_STATE, offsetof(CPUState, branch_target));
	}
	else if (!last_info.branch)
	{
		// Immediately exit.
		jit_movi(JIT_REGISTER_MODE, mode);
		jit_movi(JIT_REGISTER_NEXT_PC, pc + 4);
	}
	else if (!last_info.indirect && !last_info.conditional)
	{
		// Redirect PC to whatever value we were supposed to branch to.
		jit_movi(JIT_REGISTER_MODE, mode);
		jit_movi(JIT_REGISTER_NEXT_PC, last_info.branch_target);
	}
	else if (!last_info.conditional)
	{
		// We have an indirect branch, load that register into PC.
		jit_ldxi_i(JIT_REGISTER_NEXT_PC, JIT_REGISTER_STATE, offsetof(CPUState, sr) + 4 * last_info.branch_target);
		jit_movi(JIT_REGISTER_MODE, mode);
	}
	else if (last_info.indirect)
	{
		// Indirect conditional branch.
		auto *node = jit_beqi(JIT_REGISTER_COND_BRANCH_TAKEN, 0);
		jit_ldxi_i(JIT_REGISTER_NEXT_PC, JIT_REGISTER_STATE, offsetof(CPUState, sr) + 4 * last_info.branch_target);
		auto *to_end = jit_jmpi();
		jit_patch(node);
		jit_movi(JIT_REGISTER_NEXT_PC, pc + 4);
		jit_patch(to_end);
	}
	else
	{
		// Direct conditional branch.
		auto *node = jit_beqi(JIT_REGISTER_COND_BRANCH_TAKEN, 0);
		jit_movi(JIT_REGISTER_NEXT_PC, last_info.branch_target);
		auto *to_end = jit_jmpi();
		jit_patch(node);
		jit_movi(JIT_REGISTER_NEXT_PC, pc + 4);
		jit_patch(to_end);
	}

	auto *jmp = jit_jmpi();
	jit_patch_abs(jmp, thunks.return_thunk);
}

void CPU::jit_load_register(jit_state_t *_jit, unsigned jit_register, unsigned mips_register)
{
	if (mips_register == 0)
		jit_movi(jit_register, 0);
	else
		jit_ldxi_i(jit_register, JIT_REGISTER_STATE, offsetof(CPUState, sr) + 4 * mips_register);
}

void CPU::jit_store_register(jit_state_t *_jit, unsigned jit_register, unsigned mips_register)
{
	assert(mips_register != 0);
	jit_stxi_i(offsetof(CPUState, sr) + 4 * mips_register, JIT_REGISTER_STATE, jit_register);
}

#define DISASM(asmfmt, ...) do { \
    char buf[1024]; \
    sprintf(buf, "0x%03x   " asmfmt, pc, __VA_ARGS__); \
    mips_disasm += buf; \
} while(0)

#define DISASM_NOP() do { \
    char buf[1024]; \
    sprintf(buf, "0x%03x   nop\n", pc); \
    mips_disasm += buf; \
} while(0)

void CPU::jit_instruction(jit_state_t *_jit, uint32_t pc, uint32_t instr,
                          InstructionInfo &info, const InstructionInfo &last_info,
                          bool first_instruction)
{
	// VU
	if ((instr >> 25) == 0x25)
	{
		return;
	}

	// TODO: Meaningful register allocation.
	// For now, always flush register state to memory after an instruction for simplicity.
	// Should be red-hot in L1 cache, so probably won't be that bad.
	// On x86, we unfortunately have an anemic register bank to work with.

	uint32_t type = instr >> 26;

#define NOP_IF_RD_ZERO() if (rd == 0) { DISASM_NOP(); break; }
#define NOP_IF_RT_ZERO() if (rt == 0) { DISASM_NOP(); break; }

	switch (type)
	{
	case 000:
	{
		auto rd = (instr >> 11) & 31;
		auto rt = (instr >> 16) & 31;
		auto shift = (instr >> 6) & 31;
		auto rs = (instr >> 21) & 31;

		switch (instr & 63)
		{
		case 000: // SLL
		{
			NOP_IF_RD_ZERO();
			jit_load_register(_jit, JIT_REGISTER_TMP0, rt);
			jit_lshi(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, shift);
			jit_store_register(_jit, JIT_REGISTER_TMP0, rd);
			DISASM("sll %s, %s, %u\n", NAME(rd), NAME(rt), shift);
			break;
		}

		case 002: // SRL
		{
			NOP_IF_RD_ZERO();
			jit_load_register(_jit, JIT_REGISTER_TMP0, rt);
			jit_rshi_u(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, shift);
			jit_store_register(_jit, JIT_REGISTER_TMP0, rd);
			DISASM("srl %s, %s, %u\n", NAME(rd), NAME(rt), shift);
			break;
		}

		case 003: // SRA
		{
			NOP_IF_RD_ZERO();
			jit_load_register(_jit, JIT_REGISTER_TMP0, rt);
			jit_rshi(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, shift);
			jit_store_register(_jit, JIT_REGISTER_TMP0, rd);
			DISASM("sra %s, %s, %u\n", NAME(rd), NAME(rt), shift);
			break;
		}

		case 004: // SLLV
		{
			NOP_IF_RD_ZERO();
			jit_load_register(_jit, JIT_REGISTER_TMP0, rt);
			jit_load_register(_jit, JIT_REGISTER_TMP1, rs);
			jit_andi(JIT_REGISTER_TMP1, JIT_REGISTER_TMP1, 31);
			jit_lshr(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, JIT_REGISTER_TMP1);
			jit_store_register(_jit, JIT_REGISTER_TMP0, rd);
			DISASM("sllv %s, %s, %s\n", NAME(rd), NAME(rt), NAME(rs));
			break;
		}

		case 006: // SRLV
		{
			NOP_IF_RD_ZERO();
			jit_load_register(_jit, JIT_REGISTER_TMP0, rt);
			jit_load_register(_jit, JIT_REGISTER_TMP1, rs);
			jit_andi(JIT_REGISTER_TMP1, JIT_REGISTER_TMP1, 31);
			jit_rshr_u(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, JIT_REGISTER_TMP1);
			jit_store_register(_jit, JIT_REGISTER_TMP0, rd);
			DISASM("srlv %s, %s, %s\n", NAME(rd), NAME(rt), NAME(rs));
			break;
		}

		case 007: // SRAV
		{
			NOP_IF_RD_ZERO();
			jit_load_register(_jit, JIT_REGISTER_TMP0, rt);
			jit_load_register(_jit, JIT_REGISTER_TMP1, rs);
			jit_andi(JIT_REGISTER_TMP1, JIT_REGISTER_TMP1, 31);
			jit_rshr(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, JIT_REGISTER_TMP1);
			jit_store_register(_jit, JIT_REGISTER_TMP0, rd);
			DISASM("srav %s, %s, %s\n", NAME(rd), NAME(rt), NAME(rs));
			break;
		}

		case 010: // JR
		{
			info.branch = true;
			info.indirect = true;
			info.branch_target = rs;
			DISASM("jr %s\n", NAME(rs));
			break;
		}

		case 011: // JALR
		{
			if (rd != 0)
			{
				jit_movi(JIT_REGISTER_TMP0, pc + 8);
				jit_store_register(_jit, JIT_REGISTER_TMP0, rd);
			}
			info.branch = true;
			info.indirect = true;
			info.branch_target = rs;
			DISASM("jalr %s\n", NAME(rs));
			break;
		}

		case 015: // BREAK
		{
			jit_exit(_jit, pc, last_info, MODE_BREAK, first_instruction);
			info.handles_delay_slot = true;
			DISASM("break %u\n", 0);
			break;
		}

		case 040: // ADD
		case 041: // ADDU
		{
			NOP_IF_RD_ZERO();
			jit_load_register(_jit, JIT_REGISTER_TMP0, rt);
			jit_load_register(_jit, JIT_REGISTER_TMP1, rs);
			jit_addr(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, JIT_REGISTER_TMP1);
			jit_store_register(_jit, JIT_REGISTER_TMP0, rd);
			DISASM("addu %s, %s, %s\n", NAME(rd), NAME(rt), NAME(rs));
			break;
		}

		case 042: // SUB
		case 043: // SUBU
		{
			NOP_IF_RD_ZERO();
			jit_load_register(_jit, JIT_REGISTER_TMP0, rt);
			jit_load_register(_jit, JIT_REGISTER_TMP1, rs);
			jit_subr(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, JIT_REGISTER_TMP1);
			jit_store_register(_jit, JIT_REGISTER_TMP0, rd);
			DISASM("subu %s, %s, %s\n", NAME(rd), NAME(rt), NAME(rs));
			break;
		}

		case 044: // AND
		{
			NOP_IF_RD_ZERO();
			jit_load_register(_jit, JIT_REGISTER_TMP0, rt);
			jit_load_register(_jit, JIT_REGISTER_TMP1, rs);
			jit_andr(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, JIT_REGISTER_TMP1);
			jit_store_register(_jit, JIT_REGISTER_TMP0, rd);
			DISASM("and %s, %s, %s\n", NAME(rd), NAME(rt), NAME(rs));
			break;
		}

		case 045: // OR
		{
			NOP_IF_RD_ZERO();
			jit_load_register(_jit, JIT_REGISTER_TMP0, rt);
			jit_load_register(_jit, JIT_REGISTER_TMP1, rs);
			jit_orr(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, JIT_REGISTER_TMP1);
			jit_store_register(_jit, JIT_REGISTER_TMP0, rd);
			DISASM("or %s, %s, %s\n", NAME(rd), NAME(rt), NAME(rs));
			break;
		}

		case 046: // XOR
		{
			NOP_IF_RD_ZERO();
			jit_load_register(_jit, JIT_REGISTER_TMP0, rt);
			jit_load_register(_jit, JIT_REGISTER_TMP1, rs);
			jit_xorr(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, JIT_REGISTER_TMP1);
			jit_store_register(_jit, JIT_REGISTER_TMP0, rd);
			DISASM("xor %s, %s, %s\n", NAME(rd), NAME(rt), NAME(rs));
			break;
		}

		case 047: // NOR
		{
			NOP_IF_RD_ZERO();
			jit_load_register(_jit, JIT_REGISTER_TMP0, rt);
			jit_load_register(_jit, JIT_REGISTER_TMP1, rs);
			jit_orr(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, JIT_REGISTER_TMP1);
			jit_xori(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, jit_word_t(-1));
			jit_store_register(_jit, JIT_REGISTER_TMP0, rd);
			DISASM("nor %s, %s, %s\n", NAME(rd), NAME(rt), NAME(rs));
			break;
		}

		case 052: // SLT
		{
			NOP_IF_RD_ZERO();
			jit_load_register(_jit, JIT_REGISTER_TMP0, rs);
			jit_load_register(_jit, JIT_REGISTER_TMP1, rt);
			jit_ltr(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, JIT_REGISTER_TMP1);
			jit_store_register(_jit, JIT_REGISTER_TMP0, rd);
			DISASM("slt %s, %s, %s\n", NAME(rd), NAME(rt), NAME(rs));
			break;
		}

		case 053: // SLTU
		{
			NOP_IF_RD_ZERO();
			jit_load_register(_jit, JIT_REGISTER_TMP0, rs);
			jit_load_register(_jit, JIT_REGISTER_TMP1, rt);
			jit_ltr_u(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, JIT_REGISTER_TMP1);
			jit_store_register(_jit, JIT_REGISTER_TMP0, rd);
			DISASM("sltu %s, %s, %s\n", NAME(rd), NAME(rt), NAME(rs));
			break;
		}

		default:
			break;
		}
		break;
	}

	case 001: // REGIMM
	{
		//unsigned rs = (instr >> 21) & 31;
		unsigned rt = (instr >> 16) & 31;

		switch (rt)
		{
		case 020: // BLTZAL
			DISASM("bltzal %u\n", 0);
			break;

		case 000: // BLTZ
			DISASM("bltz %u\n", 0);
			break;

		case 021: // BGEZAL
			DISASM("bgezal %u\n", 0);
			break;

		case 001: // BGEZ
			DISASM("bgez %u\n", 0);
			break;
		}
		break;
	}

	case 003: // JAL
	{
		uint32_t target_pc = (instr & 0x3ffu) << 2;
		jit_movi(JIT_REGISTER_TMP0, pc + 8);
		jit_store_register(_jit, JIT_REGISTER_TMP0, 31);
		info.branch = true;
		info.branch_target = target_pc;
		DISASM("jal 0x%03x\n", target_pc);
		break;
	}

	case 002: // J
	{
		uint32_t target_pc = (instr & 0x3ffu) << 2;
		info.branch = true;
		info.branch_target = target_pc;
		DISASM("j 0x%03x\n", target_pc);
		break;
	}

	case 004: // BEQ
		DISASM("beq %u\n", 0);
		break;

	case 005: // BNE
		DISASM("bne %u\n", 0);
		break;

	case 006: // BLEZ
		DISASM("blez %u\n", 0);
		break;

	case 007: // BGTZ
		DISASM("bgtz %u\n", 0);
		break;

	case 010: // ADDI
	case 011:
	{
		unsigned rt = (instr >> 16) & 31;
		NOP_IF_RT_ZERO();
		int16_t simm = int16_t(instr);
		unsigned rs = (instr >> 21) & 31;

		jit_load_register(_jit, JIT_REGISTER_TMP0, rs);
		jit_addi(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, simm);
		jit_store_register(_jit, JIT_REGISTER_TMP0, rt);
		DISASM("addi %s, %s, %d\n", NAME(rt), NAME(rs), simm);
		break;
	}

	case 012: // SLTI
	{
		unsigned rt = (instr >> 16) & 31;
		NOP_IF_RT_ZERO();
		int16_t simm = int16_t(instr);
		unsigned rs = (instr >> 21) & 31;

		jit_load_register(_jit, JIT_REGISTER_TMP0, rs);
		jit_lti(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, simm);
		jit_store_register(_jit, JIT_REGISTER_TMP0, rt);
		DISASM("slti %s, %s, %d\n", NAME(rt), NAME(rs), simm);
		break;
	}

	case 013: // SLTIU
	{
		unsigned rt = (instr >> 16) & 31;
		NOP_IF_RT_ZERO();
		uint16_t imm = uint16_t(instr);
		unsigned rs = (instr >> 21) & 31;

		jit_load_register(_jit, JIT_REGISTER_TMP0, rs);
		jit_lti_u(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, imm);
		jit_store_register(_jit, JIT_REGISTER_TMP0, rt);
		DISASM("sltiu %s, %s, %u\n", NAME(rt), NAME(rs), imm);
		break;
	}

	case 014: // ANDI
	{
		unsigned rt = (instr >> 16) & 31;
		NOP_IF_RT_ZERO();
		unsigned rs = (instr >> 21) & 31;
		uint16_t imm = uint16_t(instr);
		jit_load_register(_jit, JIT_REGISTER_TMP0, rs);
		jit_andi(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, imm);
		jit_store_register(_jit, JIT_REGISTER_TMP0, rt);
		DISASM("andi %s, %s, %u\n", NAME(rt), NAME(rs), imm);
		break;
	}

	case 015: // ORI
	{
		unsigned rt = (instr >> 16) & 31;
		NOP_IF_RT_ZERO();
		unsigned rs = (instr >> 21) & 31;
		uint16_t imm = uint16_t(instr);
		jit_load_register(_jit, JIT_REGISTER_TMP0, rs);
		jit_ori(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, imm);
		jit_store_register(_jit, JIT_REGISTER_TMP0, rt);
		DISASM("ori %s, %s, %u\n", NAME(rt), NAME(rs), imm);
		break;
	}

	case 016: // XORI
	{
		unsigned rt = (instr >> 16) & 31;
		if (rt == 0)
			break;
		unsigned rs = (instr >> 21) & 31;
		uint16_t imm = uint16_t(instr);
		jit_load_register(_jit, JIT_REGISTER_TMP0, rs);
		jit_xori(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, imm);
		jit_store_register(_jit, JIT_REGISTER_TMP0, rt);
		DISASM("xori %s, %s, %u\n", NAME(rt), NAME(rs), imm);
		break;
	}

	case 017: // LUI
	{
		unsigned rt = (instr >> 16) & 31;
		NOP_IF_RT_ZERO();
		int16_t imm = int16_t(instr);
		jit_movi(JIT_REGISTER_TMP0, imm << 16);
		jit_store_register(_jit, JIT_REGISTER_TMP0, rt);
		DISASM("lui %s, %d\n", NAME(rt), imm);
		break;
	}

	case 020: // COP0
		DISASM("cop0 %u\n", 0);
		break;

	case 022: // COP2
		DISASM("cop2 %u\n", 0);
		break;

	case 040: // LB
	{
		unsigned rt = (instr >> 16) & 31;
		NOP_IF_RT_ZERO();
		int16_t simm = int16_t(instr);
		unsigned rs = (instr >> 21) & 31;

		jit_load_register(_jit, JIT_REGISTER_TMP0, rs);
		jit_addi(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, simm);
		jit_andi(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, 0xfffu);
		jit_xori(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, 3); // Endian-fixup.
		jit_ldxr_c(JIT_REGISTER_TMP1, JIT_REGISTER_DMEM, JIT_REGISTER_TMP0);
		jit_store_register(_jit, JIT_REGISTER_TMP1, rt);
		DISASM("lb %s, %d(%s)\n", NAME(rt), simm, NAME(rs));
		break;
	}

	case 041: // LH
	{
		unsigned rt = (instr >> 16) & 31;
		NOP_IF_RT_ZERO();
		int16_t simm = int16_t(instr);
		unsigned rs = (instr >> 21) & 31;

		// TODO: Handle unaligned reads?
		jit_load_register(_jit, JIT_REGISTER_TMP0, rs);
		jit_addi(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, simm);
		jit_andi(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, 0xffeu);
		jit_xori(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, 2); // Endian-fixup.
		jit_ldxr_s(JIT_REGISTER_TMP1, JIT_REGISTER_DMEM, JIT_REGISTER_TMP0);
		jit_store_register(_jit, JIT_REGISTER_TMP1, rt);
		DISASM("lh %s, %d(%s)\n", NAME(rt), simm, NAME(rs));
		break;
	}

	case 043: // LW
	{
		unsigned rt = (instr >> 16) & 31;
		NOP_IF_RT_ZERO();
		int16_t simm = int16_t(instr);
		unsigned rs = (instr >> 21) & 31;

		// TODO: Handle unaligned reads?
		jit_load_register(_jit, JIT_REGISTER_TMP0, rs);
		jit_addi(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, simm);
		jit_andi(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, 0xffcu);
		jit_ldxr(JIT_REGISTER_TMP1, JIT_REGISTER_DMEM, JIT_REGISTER_TMP0);
		jit_store_register(_jit, JIT_REGISTER_TMP1, rt);
		DISASM("lw %s, %d(%s)\n", NAME(rt), simm, NAME(rs));
		break;
	}

	case 044: // LBU
	{
		unsigned rt = (instr >> 16) & 31;
		NOP_IF_RT_ZERO();
		int16_t simm = int16_t(instr);
		unsigned rs = (instr >> 21) & 31;

		jit_load_register(_jit, JIT_REGISTER_TMP0, rs);
		jit_addi(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, simm);
		jit_andi(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, 0xfffu);
		jit_xori(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, 3); // Endian-fixup.
		jit_ldxr_uc(JIT_REGISTER_TMP1, JIT_REGISTER_DMEM, JIT_REGISTER_TMP0);
		jit_store_register(_jit, JIT_REGISTER_TMP1, rt);
		DISASM("lbu %s, %d(%s)\n", NAME(rt), simm, NAME(rs));
		break;
	}

	case 045: // LHU
	{
		unsigned rt = (instr >> 16) & 31;
		NOP_IF_RT_ZERO();
		int16_t simm = int16_t(instr);
		unsigned rs = (instr >> 21) & 31;

		// TODO: Handle unaligned reads?
		jit_load_register(_jit, JIT_REGISTER_TMP0, rs);
		jit_addi(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, simm);
		jit_andi(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, 0xffeu);
		jit_xori(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, 2); // Endian-fixup.
		jit_ldxr_us(JIT_REGISTER_TMP1, JIT_REGISTER_DMEM, JIT_REGISTER_TMP0);
		jit_store_register(_jit, JIT_REGISTER_TMP1, rt);
		DISASM("lhu %s, %d(%s)\n", NAME(rt), simm, NAME(rs));
		break;
	}

	case 050: // SB
	{
		unsigned rt = (instr >> 16) & 31;
		NOP_IF_RT_ZERO();
		int16_t simm = int16_t(instr);
		unsigned rs = (instr >> 21) & 31;

		// TODO: Handle unaligned stores?
		jit_load_register(_jit, JIT_REGISTER_TMP0, rs);
		jit_load_register(_jit, JIT_REGISTER_TMP1, rt);
		jit_addi(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, simm);
		jit_andi(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, 0xfffu);
		jit_stxr_c(JIT_REGISTER_TMP0, JIT_REGISTER_DMEM, JIT_REGISTER_TMP1);
		DISASM("sb %s, %d(%s)\n", NAME(rt), simm, NAME(rs));
		break;
	}

	case 051: // SH
	{
		unsigned rt = (instr >> 16) & 31;
		NOP_IF_RT_ZERO();
		int16_t simm = int16_t(instr);
		unsigned rs = (instr >> 21) & 31;

		// TODO: Handle unaligned stores?
		jit_load_register(_jit, JIT_REGISTER_TMP0, rs);
		jit_load_register(_jit, JIT_REGISTER_TMP1, rt);
		jit_addi(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, simm);
		jit_andi(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, 0xffeu);
		jit_stxr_s(JIT_REGISTER_TMP0, JIT_REGISTER_DMEM, JIT_REGISTER_TMP1);
		DISASM("sh %s, %d(%s)\n", NAME(rt), simm, NAME(rs));
		break;
	}

	case 053: // SW
	{
		unsigned rt = (instr >> 16) & 31;
		NOP_IF_RT_ZERO();
		int16_t simm = int16_t(instr);
		unsigned rs = (instr >> 21) & 31;

		// TODO: Handle unaligned stores?
		jit_load_register(_jit, JIT_REGISTER_TMP0, rs);
		jit_load_register(_jit, JIT_REGISTER_TMP1, rt);
		jit_addi(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, simm);
		jit_andi(JIT_REGISTER_TMP0, JIT_REGISTER_TMP0, 0xffcu);
		jit_stxr(JIT_REGISTER_TMP0, JIT_REGISTER_DMEM, JIT_REGISTER_TMP1);
		DISASM("sw %s, %d(%s)\n", NAME(rt), simm, NAME(rs));
		break;
	}

	case 062: // LWC2
		DISASM("lcw2 %u\n", 0);
		break;

	case 072: // SWC2
		DISASM("swc2 %u\n", 0);
		break;

	default:
		break;
	}
}

Func CPU::jit_region(uint64_t hash, unsigned pc_word, unsigned instruction_count)
{
	mips_disasm.clear();
	jit_state_t *_jit = jit_new_state();

	jit_prolog();
	jit_tramp(JIT_FRAME_SIZE);

	// We can potentially branch to every instruction in the block, so declare forward references to them here.
	jit_node_t *branch_targets[CODE_BLOCK_SIZE];
	for (unsigned i = 0; i < instruction_count; i++)
		branch_targets[i] = jit_forward();

	jit_node_t *latent_delay_slot = nullptr;

	InstructionInfo last_info = {};
	for (unsigned i = 0; i < instruction_count; i++)
	{
		jit_link(branch_targets[i]);

		uint32_t instr = state.imem[pc_word + i];
		InstructionInfo inst_info = {};
		jit_instruction(_jit, (pc_word + i) << 2, instr, inst_info, last_info, i == 0);

		if (i == 0 && !inst_info.handles_delay_slot)
		{
			// After the first instruction, we might need to resolve a latent delay slot.
			latent_delay_slot = jit_forward();
			jit_ldxi_i(JIT_REGISTER_TMP0, JIT_REGISTER_STATE, offsetof(CPUState, has_delay_slot));
			jit_patch_at(jit_bnei(JIT_REGISTER_TMP0, 0), latent_delay_slot);
		}
		else if (i != 0 && !inst_info.handles_delay_slot && last_info.branch)
		{
			// Normal handling of the delay slot.
			jit_handle_delay_slot(_jit, last_info, branch_targets,
			                      pc_word << 2,
			                      (pc_word + instruction_count) << 2);
		}
		last_info = inst_info;
	}

	// Jump to another block.
	jit_end_of_block(_jit, (pc_word + instruction_count) << 2, last_info);

	// If we had a latent delay slot, we handle it here.
	if (latent_delay_slot)
	{
		jit_link(latent_delay_slot);
		// We cannot execute a branch inside a delay slot, so just assume we do not have to chain together these.
		// We could technically handle it, but it gets messy (and it's illegal MIPS), so don't bother.
		jit_movi(JIT_REGISTER_NEXT_PC, 0);
		jit_stxi_i(offsetof(CPUState, has_delay_slot), JIT_REGISTER_STATE, JIT_REGISTER_NEXT_PC);
		jit_ldxi_i(JIT_REGISTER_NEXT_PC, JIT_REGISTER_STATE, offsetof(CPUState, branch_target));
		jit_patch_abs(jit_jmpi(), thunks.enter_thunk);
	}

	auto ret = reinterpret_cast<Func>(jit_emit());

	printf(" === DISASM ===\n");
	jit_disassemble();
	jit_clear_state();
	printf("%s\n", mips_disasm.c_str());
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

void CPU::print_registers()
{
	fprintf(stderr, "RSP state:\n");
	fprintf(stderr, "  PC: 0x%03x\n", state.pc);
	for (unsigned i = 1; i < 32; i++)
		fprintf(stderr, "  SR[%s] = 0x%08x\n", NAME(i), state.sr[i]);
	fprintf(stderr, "\n");
	for (unsigned i = 0; i < 32; i++)
	{
		fprintf(stderr, "  VR[%02u] = { 0x%04x, 0x%04x, 0x%04x, 0x%04x, 0x%04x, 0x%04x, 0x%04x, 0x%04x }\n", i,
		        state.cp2.regs[i].e[0], state.cp2.regs[i].e[1], state.cp2.regs[i].e[2], state.cp2.regs[i].e[3],
		        state.cp2.regs[i].e[4], state.cp2.regs[i].e[5], state.cp2.regs[i].e[6], state.cp2.regs[i].e[7]);
	}

	fprintf(stderr, "\n");

	for (unsigned i = 0; i < 3; i++)
	{
		static const char *strings[] = { "ACC_HI", "ACC_MD", "ACC_LO" };
		fprintf(stderr, "  %s = { 0x%04x, 0x%04x, 0x%04x, 0x%04x, 0x%04x, 0x%04x, 0x%04x, 0x%04x }\n", strings[i],
		        state.cp2.acc.e[8 * i + 0], state.cp2.acc.e[8 * i + 1], state.cp2.acc.e[8 * i + 2],
		        state.cp2.acc.e[8 * i + 3], state.cp2.acc.e[8 * i + 4], state.cp2.acc.e[8 * i + 5],
		        state.cp2.acc.e[8 * i + 6], state.cp2.acc.e[8 * i + 7]);
	}

	fprintf(stderr, "\n");

	for (unsigned i = 0; i < 3; i++)
	{
		static const char *strings[] = { "VCO", "VCC", "VCE" };
		uint16_t flags = rsp_get_flags(state.cp2.flags[i].e);
		fprintf(stderr, "  %s = 0x%04x\n", strings[i], flags);
	}

	fprintf(stderr, "\n");
	fprintf(stderr, "  Div Out = 0x%04x\n", state.cp2.div_out);
	fprintf(stderr, "  Div In  = 0x%04x\n", state.cp2.div_in);
	fprintf(stderr, "  DP flag = 0x%04x\n", state.cp2.dp_flag);
}

} // namespace JIT
} // namespace RSP