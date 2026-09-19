#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#include "loader/demonsSoulsCulling.h"
#include "common/virtualMemory.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace Loader {
class Elf64 {};
} // namespace Loader

#include "loader/runtimeLinker.h"

namespace Loader {
ThreadLocalStorage::~ThreadLocalStorage() = default;
Program::Program() = default;
Program::~Program() = default;
} // namespace Loader

namespace {
bool     g_mock_is_supported = false;
bool     g_mock_readable = false;
bool     g_mock_protect_fail = false;
bool     g_mock_protect_fail_on_sort = false;
uint64_t g_test_sort_addr = 0;

void Check(bool pass, const char* text) {
	if (!pass) {
		std::fprintf(stderr, "demons souls culling: %s\n", text);
		std::abort();
	}
}

struct MockPacketList {
	uint8_t padding[0x150];
	Loader::DemonsSoulsCulling::DrawPacket* first;
	Loader::DemonsSoulsCulling::DrawPacket* last;
};
static_assert(offsetof(MockPacketList, first) == 0x150);
static_assert(offsetof(MockPacketList, last) == 0x158);

uint64_t g_thunk_received_job_ctx = 0;
uint64_t g_thunk_received_packet_list = 0;
uint64_t g_thunk_received_comp_fn = 0;
bool     g_thunk_stack_aligned = false;

void KYTY_SYSV_ABI MockNativeSortThunkCallback(uint64_t job_ctx, uint64_t packet_list, uint64_t comp_fn) {
	g_thunk_received_job_ctx = job_ctx;
	g_thunk_received_packet_list = packet_list;
	g_thunk_received_comp_fn = comp_fn;

	// In System V ABI, at the entry of a function called via 'call',
	// rsp + 8 is 16-byte aligned (i.e. rsp % 16 == 8).
	uintptr_t current_rsp = 0;
#if defined(__x86_64__) || defined(_M_X64)
	#if defined(_MSC_VER) && !defined(__clang__)
		current_rsp = reinterpret_cast<uintptr_t>(_AddressOfReturnAddress()) + 8;
	#else
		__asm__ volatile("mov %%rsp, %0" : "=r"(current_rsp));
	#endif
#endif
	g_thunk_stack_aligned = (current_rsp % 16) == 8;
}

KYTY_SYSV_ABI bool MockComparator(const Loader::DemonsSoulsCulling::DrawPacket* a,
                                  const Loader::DemonsSoulsCulling::DrawPacket* b) {
	return a->data[0] < b->data[0];
}

} // namespace

namespace Libs {
namespace Graphics {
namespace DemonsSouls {
bool IsSupportedGame() { return g_mock_is_supported; }
} // namespace DemonsSouls
bool HostMemoryRangeIsReadable(uint64_t vaddr, uint64_t) {
	if (!g_mock_readable) return false;
	return vaddr != 0;
}
} // namespace Graphics
namespace LibKernel::Memory {
uint64_t AllocateRuntimeMemory(uint64_t requested, uint64_t, Common::VirtualMemory::Mode, const char*, bool) {
	return requested;
}
bool ProtectGuestMemory(uint64_t vaddr, uint64_t, Common::VirtualMemory::Mode, Common::VirtualMemory::Mode* old_mode) {
	if (g_mock_protect_fail) return false;
	if (g_mock_protect_fail_on_sort && vaddr == g_test_sort_addr) return false;
	if (old_mode) *old_mode = Common::VirtualMemory::Mode::ExecuteRead;
	return true;
}
bool FreeGuestMemory(uint64_t, uint64_t) { return true; }
} // namespace LibKernel::Memory
} // namespace Libs

int main() {
	using namespace Loader::DemonsSoulsCulling;

	// 1. Structure layout and mathematical ABI safety assertions
	static_assert(sizeof(DrawPacket) == 32);
	static_assert(alignof(DrawPacket) >= 8);
	static_assert(offsetof(DrawPacket, key) == 0);
	static_assert(offsetof(DrawPacket, data) == 8);
	static_assert(offsetof(DrawPacket, data[0]) == 8);
	static_assert(offsetof(DrawPacket, data[1]) == 16);
	static_assert(offsetof(DrawPacket, data[2]) == 24);

	static_assert(SortOffset == 0x00ab2e50);
	static_assert(ContinueOffset == 0x00ab3360);
	static_assert(ComparatorOffset == 0x00ab52e0);
	static_assert(CaveOffset == 0x08004000);
	static_assert(PageSize == 0x00004000);
	static_assert(PrologueBytes.size() == 17);

	// Displacement calculation assertion
	// 5-byte near jmp from SortOffset to CaveOffset:
	// displacement = CaveOffset - (SortOffset + 5) = 0x08004000 - 0x00ab2e55 = 0x075511ab
	constexpr int32_t expected_disp = static_cast<int32_t>(CaveOffset - (SortOffset + 5));
	static_assert(expected_disp == 0x075511ab);

	// Verify cave non-overlap with DemonsSoulsIdle (0x08000000) and guest image
	constexpr uint64_t IdleCaveOffset = 0x08000000;
	static_assert(CaveOffset >= IdleCaveOffset + PageSize, "Culling cave sits cleanly past DemonsSoulsIdle cave");

	// 2. Prologue byte signature verification
	Check(MatchesPrologue(PrologueBytes), "PrologueBytes exact match");
	auto changed_prologue = PrologueBytes;
	changed_prologue[0] ^= 1;
	Check(!MatchesPrologue(changed_prologue), "modified prologue first byte rejected");
	changed_prologue = PrologueBytes;
	changed_prologue[16] ^= 1;
	Check(!MatchesPrologue(changed_prologue), "modified prologue last byte rejected");
	changed_prologue = PrologueBytes;
	changed_prologue[8] ^= 1;
	Check(!MatchesPrologue(changed_prologue), "modified prologue middle byte rejected");
	Check(!MatchesPrologue(std::span(PrologueBytes).first(10)), "short prologue rejected");
	Check(!MatchesPrologue({}), "empty prologue rejected");

	// 3. Pointer safety and bounds checking in NativeSortDrawPackets
	NativeSortDrawPackets(0, 0, 0);
	NativeSortDrawPackets(0, 0x100, 0);
	NativeSortDrawPackets(0, 0xfff0, 0);

	MockPacketList mock_list {};
	mock_list.first = nullptr;
	mock_list.last = nullptr;
	NativeSortDrawPackets(0, reinterpret_cast<uint64_t>(&mock_list), 0);

	DrawPacket single_packet {10, {1, 2, 3}};
	mock_list.first = &single_packet;
	mock_list.last = nullptr;
	NativeSortDrawPackets(0, reinterpret_cast<uint64_t>(&mock_list), 0);
	Check(single_packet.key == 10, "single packet untouched when last_ptr is null");

	mock_list.first = nullptr;
	mock_list.last = &single_packet;
	NativeSortDrawPackets(0, reinterpret_cast<uint64_t>(&mock_list), 0);

	// Zero elements: first == last
	mock_list.first = &single_packet;
	mock_list.last = &single_packet;
	NativeSortDrawPackets(0, reinterpret_cast<uint64_t>(&mock_list), 0);

	// Inverted range: first > last
	std::array<DrawPacket, 2> inv_packets {{{20, {0, 0, 0}}, {10, {0, 0, 0}}}};
	mock_list.first = &inv_packets[1];
	mock_list.last = &inv_packets[0];
	NativeSortDrawPackets(0, reinterpret_cast<uint64_t>(&mock_list), 0);
	Check(inv_packets[0].key == 20 && inv_packets[1].key == 10, "inverted range does not sort");

	// Misaligned packet_list (& 7 != 0) rejected safely
	NativeSortDrawPackets(0, 0x20001, 0);
	NativeSortDrawPackets(0, 0x20003, 0);
	// Canonical user-space upper bound rejected safely
	NativeSortDrawPackets(0, 0x0000800000000000ull, 0);
	NativeSortDrawPackets(0, 0xffffffffffffffffull, 0);

	// Misaligned first_ptr or last_ptr rejected safely
	mock_list.first = reinterpret_cast<DrawPacket*>(0x20001);
	mock_list.last = reinterpret_cast<DrawPacket*>(0x20021);
	NativeSortDrawPackets(0, reinterpret_cast<uint64_t>(&mock_list), 0);

	mock_list.first = reinterpret_cast<DrawPacket*>(0x20000);
	mock_list.last = reinterpret_cast<DrawPacket*>(0x20005);
	NativeSortDrawPackets(0, reinterpret_cast<uint64_t>(&mock_list), 0);

	// Diff overflow (> 64 MB / 2M packets) rejected safely
	mock_list.first = reinterpret_cast<DrawPacket*>(0x20000);
	mock_list.last = reinterpret_cast<DrawPacket*>(0x20000 + 0x05000000);
	NativeSortDrawPackets(0, reinterpret_cast<uint64_t>(&mock_list), 0);

	// Invalid comp_fn fallback to key-only sort
	std::array<DrawPacket, 3> fallback_packets {{{30, {0, 0, 0}}, {10, {0, 0, 0}}, {20, {0, 0, 0}}}};
	mock_list.first = fallback_packets.data();
	mock_list.last = fallback_packets.data() + fallback_packets.size();
	NativeSortDrawPackets(0, reinterpret_cast<uint64_t>(&mock_list), 0x123);
	Check(fallback_packets[0].key == 10 && fallback_packets[1].key == 20 && fallback_packets[2].key == 30,
	      "invalid comp_fn safely falls back to key sort");

	// 4. Native sorting tests: basic keys
	{
		std::array<DrawPacket, 6> packets {{{50, {5, 0, 0}},
		                                    {10, {1, 0, 0}},
		                                    {40, {4, 0, 0}},
		                                    {20, {2, 0, 0}},
		                                    {60, {6, 0, 0}},
		                                    {30, {3, 0, 0}}}};
		mock_list.first = packets.data();
		mock_list.last = packets.data() + packets.size();
		NativeSortDrawPackets(0, reinterpret_cast<uint64_t>(&mock_list), 0);

		for (size_t i = 0; i < packets.size(); ++i) {
			const uint64_t expected_key = (i + 1) * 10;
			Check(packets[i].key == expected_key, "keys sorted ascending");
			Check(packets[i].data[0] == (i + 1), "payload data preserved with key");
		}
	}

	// 5. Native sorting with tie-breaker comparator
	{
		std::array<DrawPacket, 5> packets {{{20, {100, 0, 0}},
		                                    {10, {200, 0, 0}},
		                                    {20, {50, 0, 0}},
		                                    {10, {150, 0, 0}},
		                                    {20, {75, 0, 0}}}};
		mock_list.first = packets.data();
		mock_list.last = packets.data() + packets.size();
		NativeSortDrawPackets(0, reinterpret_cast<uint64_t>(&mock_list),
		                     reinterpret_cast<uint64_t>(&MockComparator));

		// Expect key 10 sorted with data[0]: 150, 200
		Check(packets[0].key == 10 && packets[0].data[0] == 150, "tie-break 10 #1");
		Check(packets[1].key == 10 && packets[1].data[0] == 200, "tie-break 10 #2");
		// Expect key 20 sorted with data[0]: 50, 75, 100
		Check(packets[2].key == 20 && packets[2].data[0] == 50, "tie-break 20 #1");
		Check(packets[3].key == 20 && packets[3].data[0] == 75, "tie-break 20 #2");
		Check(packets[4].key == 20 && packets[4].data[0] == 100, "tie-break 20 #3");
	}

	// 6. Scale stress test with 2048 packets
	{
		constexpr size_t N = 2048;
		std::vector<DrawPacket> large_packets(N);
		for (size_t i = 0; i < N; ++i) {
			large_packets[i].key = (i * 1103515245ull + 12345ull) % 500;
			large_packets[i].data[0] = i;
			large_packets[i].data[1] = i * 2;
			large_packets[i].data[2] = i * 3;
		}
		mock_list.first = large_packets.data();
		mock_list.last = large_packets.data() + N;
		NativeSortDrawPackets(0, reinterpret_cast<uint64_t>(&mock_list),
		                     reinterpret_cast<uint64_t>(&MockComparator));

		for (size_t i = 1; i < N; ++i) {
			const auto& prev = large_packets[i - 1];
			const auto& curr = large_packets[i];
			Check(prev.key <= curr.key, "stress test key order");
			if (prev.key == curr.key) {
				Check(prev.data[0] <= curr.data[0], "stress test tie-break order");
			}
			Check(curr.data[1] == curr.data[0] * 2, "stress test data[1] integrity");
			Check(curr.data[2] == curr.data[0] * 3, "stress test data[2] integrity");
		}
	}

	// 6a. Strict weak ordering test: duplicate identical packets
	{
		std::array<DrawPacket, 5> dup_packets {{{20, {5, 0, 0}},
		                                        {10, {5, 0, 0}},
		                                        {10, {5, 0, 0}},
		                                        {10, {1, 0, 0}},
		                                        {20, {5, 0, 0}}}};
		mock_list.first = dup_packets.data();
		mock_list.last = dup_packets.data() + dup_packets.size();
		NativeSortDrawPackets(0, reinterpret_cast<uint64_t>(&mock_list),
		                     reinterpret_cast<uint64_t>(&MockComparator));

		Check(dup_packets[0].key == 10 && dup_packets[0].data[0] == 1, "dup test 0");
		Check(dup_packets[1].key == 10 && dup_packets[1].data[0] == 5, "dup test 1");
		Check(dup_packets[2].key == 10 && dup_packets[2].data[0] == 5, "dup test 2");
		Check(dup_packets[3].key == 20 && dup_packets[3].data[0] == 5, "dup test 3");
		Check(dup_packets[4].key == 20 && dup_packets[4].data[0] == 5, "dup test 4");
	}

	// 6b. Concurrent multi-threaded stress test simulating parallel shadow cascades & pre-cull workers
	{
		constexpr size_t NumThreads = 8;
		constexpr size_t PacketsPerThread = 512;
		std::vector<std::thread> workers;
		std::vector<std::vector<DrawPacket>> thread_packets(NumThreads, std::vector<DrawPacket>(PacketsPerThread));
		std::vector<MockPacketList> thread_lists(NumThreads);

		for (size_t t = 0; t < NumThreads; ++t) {
			for (size_t i = 0; i < PacketsPerThread; ++i) {
				thread_packets[t][i].key = (i * 31337ull + t * 7919ull) % 256;
				thread_packets[t][i].data[0] = i;
				thread_packets[t][i].data[1] = t;
				thread_packets[t][i].data[2] = 0;
			}
			thread_lists[t].first = thread_packets[t].data();
			thread_lists[t].last = thread_packets[t].data() + PacketsPerThread;
		}

		for (size_t t = 0; t < NumThreads; ++t) {
			workers.emplace_back([&thread_lists, t]() {
				NativeSortDrawPackets(t, reinterpret_cast<uint64_t>(&thread_lists[t]),
				                     reinterpret_cast<uint64_t>(&MockComparator));
			});
		}

		for (auto& w : workers) {
			w.join();
		}

		for (size_t t = 0; t < NumThreads; ++t) {
			for (size_t i = 1; i < PacketsPerThread; ++i) {
				const auto& prev = thread_packets[t][i - 1];
				const auto& curr = thread_packets[t][i];
				Check(prev.key <= curr.key, "concurrent thread packet order");
				if (prev.key == curr.key) {
					Check(prev.data[0] <= curr.data[0], "concurrent thread tie-break order");
				}
				Check(curr.data[1] == t, "concurrent thread data[1] integrity");
			}
		}
	}

#if defined(__x86_64__) || defined(_M_X64)
	// 7. Thunk generation and ABI verification (using Xbyak)
	{
		using namespace Xbyak::util;
		Xbyak::CodeGenerator code(4096);

		const auto* continue_landing = code.getCurr();
		code.mov(rax, 0x1234567890abcdefull); // Landing marker
		code.ret();

		const auto* thunk = code.getCurr();
		EmitCullingSortThunk(code, reinterpret_cast<const void*>(&MockNativeSortThunkCallback),
		                    0xca7f00d5a110c001ull,
		                    reinterpret_cast<uint64_t>(continue_landing));

		// Test runner SysV ABI: (job_ctx: rdi, packet_list: rsi, out_preserved_regs: rdx)
		const auto* run_thunk = code.getCurr();
		code.push(r12);
		code.mov(r12, rdx); // Save output buffer pointer in callee-saved r12
		code.call(thunk);
		code.mov(qword[r12 + 0], rdi); // Preserved rdi
		code.mov(qword[r12 + 8], rsi); // Preserved rsi
		code.mov(qword[r12 + 16], rax); // Return value from continue_landing
		code.pop(r12);
		code.ret();

		code.ready();
		Check(!Xbyak::GetError(), "Xbyak thunk generation clean");

		using ThunkRunner = KYTY_SYSV_ABI void (*)(uint64_t, uint64_t, void*);
		const auto runner = reinterpret_cast<ThunkRunner>(reinterpret_cast<uintptr_t>(run_thunk));

		uint64_t preserved[3] = {0, 0, 0};
		const uint64_t test_job_ctx = 0xdeadbeef11223344ull;
		const uint64_t test_packet_list = 0xfeedface55667788ull;

		g_thunk_received_job_ctx = 0;
		g_thunk_received_packet_list = 0;
		g_thunk_received_comp_fn = 0;
		g_thunk_stack_aligned = false;

		runner(test_job_ctx, test_packet_list, preserved);

		Check(g_thunk_received_job_ctx == test_job_ctx, "thunk forwarded job_context in rdi");
		Check(g_thunk_received_packet_list == test_packet_list, "thunk forwarded packet_list in rsi");
		Check(g_thunk_received_comp_fn == 0xca7f00d5a110c001ull, "thunk forwarded comp_fn in rdx");
		Check(g_thunk_stack_aligned, "thunk maintained 16-byte stack alignment at call");
		Check(preserved[0] == test_job_ctx, "thunk preserved rdi across call");
		Check(preserved[1] == test_packet_list, "thunk preserved rsi across call");
		Check(preserved[2] == 0x1234567890abcdefull, "thunk successfully jumped to continue_landing");
	}

	// 8. End-to-end Detour Installation, Execution, Clear, and Rollback test
	{
		constexpr uint64_t TestTotalSize = CaveOffset + PageSize;
		const uint64_t base = Common::VirtualMemory::Alloc(0, TestTotalSize, Common::VirtualMemory::Mode::ExecuteReadWrite);
		Check(base != 0, "Alloc guest test virtual memory block");

		const uint64_t sort_addr = base + SortOffset;
		const uint64_t cont_addr = base + ContinueOffset;
		const uint64_t comp_addr = base + ComparatorOffset;

		std::memcpy(reinterpret_cast<void*>(sort_addr), PrologueBytes.data(), PrologueBytes.size());

		// Put a 'ret' instruction (0xc3) at ContinueOffset so the detour returns cleanly to the caller
		*reinterpret_cast<uint8_t*>(cont_addr) = 0xc3;

		// Put MockComparator at ComparatorOffset
		// In test virtual memory, we emit a SysV trampoline to MockComparator:
		// mov rax, &MockComparator; jmp rax
		uint8_t* comp_slot = reinterpret_cast<uint8_t*>(comp_addr);
		comp_slot[0] = 0x48; comp_slot[1] = 0xb8; // mov rax, imm64
		const uint64_t mock_comp_fn = reinterpret_cast<uint64_t>(&MockComparator);
		std::memcpy(comp_slot + 2, &mock_comp_fn, sizeof(mock_comp_fn));
		comp_slot[10] = 0xff; comp_slot[11] = 0xe0; // jmp rax

		Loader::Program program;
		program.file_name = "eboot.bin";
		program.mapped_size = CaveOffset;
		program.base_vaddr = base;

		g_mock_is_supported = true;
		g_mock_readable = true;

		// Initial installation
		Install(&program);

		const auto* sort_mem = reinterpret_cast<const uint8_t*>(sort_addr);
		Check(sort_mem[0] == 0xe9, "Install patched near jmp opcode 0xe9 at SortOffset");
		const int32_t patched_disp = *reinterpret_cast<const int32_t*>(sort_mem + 1);
		Check(sort_addr + 5 + patched_disp == base + CaveOffset, "detour near jmp targets CaveOffset");
		for (size_t k = 5; k < 17; ++k) {
			Check(sort_mem[k] == 0x90, "trailing byte is NOP");
		}

		// Execute the installed detour end-to-end!
		std::array<DrawPacket, 4> exec_packets {{{30, {0, 0, 0}}, {10, {0, 0, 0}}, {40, {0, 0, 0}}, {20, {0, 0, 0}}}};
		MockPacketList exec_list {};
		exec_list.first = exec_packets.data();
		exec_list.last = exec_packets.data() + exec_packets.size();

		using DetourEntry = KYTY_SYSV_ABI void (*)(uint64_t, uint64_t);
		const auto detour_caller = reinterpret_cast<DetourEntry>(sort_addr);
		detour_caller(0x11223344, reinterpret_cast<uint64_t>(&exec_list));

		Check(exec_packets[0].key == 10, "end-to-end detour sorted packet 0");
		Check(exec_packets[1].key == 20, "end-to-end detour sorted packet 1");
		Check(exec_packets[2].key == 30, "end-to-end detour sorted packet 2");
		Check(exec_packets[3].key == 40, "end-to-end detour sorted packet 3");

		// Idempotency: second Install does not re-hook or crash
		Install(&program);
		Check(sort_mem[0] == 0xe9, "idempotent Install retains jmp");

		// Clear uninstallation
		Clear();
		Check(std::memcmp(sort_mem, PrologueBytes.data(), PrologueBytes.size()) == 0,
		      "Clear cleanly restores PrologueBytes at SortOffset");

		// Clear idempotency
		Clear();
		Check(std::memcmp(sort_mem, PrologueBytes.data(), PrologueBytes.size()) == 0,
		      "Second Clear retains PrologueBytes");

		// Error Rollback: simulate ProtectGuestMemory failure
		g_test_sort_addr = sort_addr;
		g_mock_protect_fail_on_sort = true;

		Install(&program);
		Check(std::memcmp(sort_mem, PrologueBytes.data(), PrologueBytes.size()) == 0,
		      "Rollback keeps original PrologueBytes intact on ProtectGuestMemory failure");

		g_mock_protect_fail_on_sort = false;
		g_test_sort_addr = 0;

		// Re-install after rollback succeeds
		Install(&program);
		Check(sort_mem[0] == 0xe9, "Install succeeds after previous rollback");
		Clear();
		Check(std::memcmp(sort_mem, PrologueBytes.data(), PrologueBytes.size()) == 0,
		      "Clear cleanly restores PrologueBytes after re-installation");

		g_mock_is_supported = false;
		g_mock_readable = false;
		Common::VirtualMemory::Free(base);
	}
#endif

	std::puts("Demon's Souls culling sort tests passed");
	return 0;
}
