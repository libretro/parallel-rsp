#include "rsp-mips.h"

u32 data[4] = { 0x10, 0x20, 0x30, 0x40 };

int main(void)
{
	rsp_debug_break(data[0], data[1], data[2], data[3]);
}
