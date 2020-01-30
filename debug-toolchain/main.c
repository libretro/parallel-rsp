#include "rsp-mips.h"

u8 data[4] = { 0x10, 0x20, 0x30, 0x40 };

__attribute__((noinline))
int load_byte(int i)
{
	return data[i];
}

int main(void)
{
	int a = load_byte(0);
	int b = load_byte(1);
	int c = load_byte(2);
	int d = load_byte(3);
	rsp_debug_break(a, b, c, d);
}
