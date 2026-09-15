#pragma once
#include "common/abi.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace Loader {
struct Program;
namespace DemonsSoulsCopy {
void Install(Program* program);
void Clear();
using PrepareWrite = bool (*)(uint64_t, uint64_t);

constexpr size_t CoherentThreshold = 65536;

#if defined(__clang__) || defined(__GNUC__)
#define KYTY_STREAMING_TARGET __attribute__((__target__("avx2")))
#else
#define KYTY_STREAMING_TARGET
#endif

inline KYTY_STREAMING_TARGET void CopyGpuStreaming(void* dst, const void* src, size_t size) {
	auto*       d = static_cast<uint8_t*>(dst);
	const auto* s = static_cast<const uint8_t*>(src);

	// If buffers overlap in a way that forward copy corrupts source bytes, fall back to std::memmove
	if (d > s && d < s + size) {
		std::memmove(d, s, size);
		return;
	}

	size_t offset = 0;

	// Peel unaligned leading bytes until destination is 32-byte aligned
	const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(d);
	const size_t    prefix   = (32 - (dst_addr & 31)) & 31;
	if (prefix > 0 && size >= prefix) {
		std::memcpy(d, s, prefix);
		offset += prefix;
	}

	// 32-byte aligned streaming body (bypasses host CPU cache pollution)
	for (; offset + 32 <= size; offset += 32) {
		__m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + offset));
		_mm256_stream_si256(reinterpret_cast<__m256i*>(d + offset), chunk);
	}
	_mm_sfence();

	// Handle trailing remainder bytes
	if (offset < size) {
		std::memcpy(d + offset, s + offset, size - offset);
	}
}

inline void* Move(void* destination, const void* source, size_t size, PrepareWrite prepare) {
	if (size == 0 || destination == source) return destination;
	if (size >= CoherentThreshold) {
		if (prepare) (void)prepare(reinterpret_cast<uint64_t>(destination), size);
		CopyGpuStreaming(destination, source, size);
		return destination;
	}
	return std::memmove(destination, source, size);
}
inline std::array<uint8_t, 13> Tailcall(uint64_t function) {
	std::array<uint8_t, 13> code {0x5d, 0x48, 0xb8}; // pop rbp; mov rax, function; jmp rax
	std::memcpy(code.data() + 3, &function, sizeof(function));
	code[11] = 0xff;
	code[12] = 0xe0;
	return code;
}
} // namespace DemonsSoulsCopy
} // namespace Loader
