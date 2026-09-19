#pragma once

#include "common/abi.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>

#if defined(__x86_64__) || defined(_M_X64)
#ifndef XBYAK_NO_EXCEPTION
#define XBYAK_NO_EXCEPTION
#endif
#include <xbyak/xbyak.h>
#endif

namespace Loader {
struct Program;
namespace DemonsSoulsCulling {
void Install(Program* program);
void Clear();

inline constexpr uint64_t SortOffset       = 0x00ab2e50;
inline constexpr uint64_t ContinueOffset   = 0x00ab3360;
inline constexpr uint64_t ComparatorOffset = 0x00ab52e0;
inline constexpr uint64_t CaveOffset       = 0x08004000;
inline constexpr uint64_t PageSize         = 0x00004000;

inline constexpr std::array<uint8_t, 17> PrologueBytes {
    0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x28
};

inline bool MatchesPrologue(std::span<const uint8_t> bytes) {
	return std::ranges::equal(bytes, PrologueBytes);
}

struct alignas(8) DrawPacket {
	uint64_t key;
	uint64_t data[3];

	bool operator==(const DrawPacket& other) const {
		return key == other.key && data[0] == other.data[0] && data[1] == other.data[1] &&
		       data[2] == other.data[2];
	}
};
static_assert(sizeof(DrawPacket) == 32);

using Comparator = KYTY_SYSV_ABI bool (*)(const DrawPacket* a, const DrawPacket* b);

void KYTY_SYSV_ABI NativeSortDrawPackets(uint64_t job_context, uint64_t packet_list, uint64_t comp_fn);

#if defined(__x86_64__) || defined(_M_X64)
inline void EmitCullingSortThunk(Xbyak::CodeGenerator& c, const void* native_sort, uint64_t comp_fn,
                                uint64_t continue_site) {
	using namespace Xbyak::util;
	c.push(rbp);
	c.mov(rbp, rsp);
	c.push(rdi);
	c.push(rsi);
	c.mov(rdx, comp_fn);
	c.mov(rax, reinterpret_cast<uint64_t>(native_sort));
	c.call(rax);
	c.pop(rsi);
	c.pop(rdi);
	c.pop(rbp);
	c.mov(rax, continue_site);
	c.jmp(rax);
}
#endif

} // namespace DemonsSoulsCulling
} // namespace Loader
