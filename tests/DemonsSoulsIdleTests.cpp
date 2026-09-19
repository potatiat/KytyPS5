#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#include "loader/demonsSoulsIdle.h"
#include "common/virtualMemory.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
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
bool     g_mock_protect_fail_on_timestep = false;
bool     g_mock_protect_fail_on_cvar = false;
uint64_t g_test_timestep_addr = 0;
uint64_t g_test_cvar_addr = 0;
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
	if (g_mock_protect_fail_on_timestep && vaddr == g_test_timestep_addr) return false;
	if (g_mock_protect_fail_on_cvar && vaddr == g_test_cvar_addr) return false;
	if (old_mode) *old_mode = Common::VirtualMemory::Mode::ExecuteRead;
	return true;
}
bool FreeGuestMemory(uint64_t, uint64_t) { return true; }
} // namespace LibKernel::Memory
} // namespace Libs

namespace {
void Check(bool pass, const char* text) {
	if (!pass) {
		std::fprintf(stderr, "idle wait: %s\n", text);
		std::abort();
	}
}

uint64_t captured_sync_ptr = 0;
uint64_t captured_expected_val = 0;
uint32_t captured_op = 0;
uint32_t captured_can_sleep = 0;

void KYTY_SYSV_ABI MockWaitCallback(uint64_t sync_ptr, uint64_t expected_val, uint32_t op, uint32_t can_sleep) {
	captured_sync_ptr = sync_ptr;
	captured_expected_val = expected_val;
	captured_op = op;
	captured_can_sleep = can_sleep;
}
} // namespace

int main() {
	using namespace Loader::DemonsSoulsIdle;
	Check(Matches(CallBytes, PollBytes), "known version signatures");
	auto changed = CallBytes;
	changed[20] ^= 1;
	Check(!Matches(changed, PollBytes), "modified code beyond the call must be rejected");
	Check(!Matches(CallBytes, std::span(PollBytes).first(5)), "short poll signature");

	// 1. JobSyncBytes matching verification
	Check(MatchesJobSync(JobSyncBytes), "JobSyncBytes exact match");
	auto changed_sync = JobSyncBytes;
	changed_sync[0] ^= 1;
	Check(!MatchesJobSync(changed_sync), "modified job sync first byte rejected");
	changed_sync = JobSyncBytes;
	changed_sync[13] ^= 1;
	Check(!MatchesJobSync(changed_sync), "modified job sync last byte rejected");
	Check(!MatchesJobSync(std::span(JobSyncBytes).first(5)), "short job sync signature rejected");
	Check(!MatchesJobSync({}), "empty job sync signature rejected");

	// 2. PhysicsTimestep and PhysicsCVar verification
	// PhysicsTimestepOriginalBytes and PhysicsTimestepPatchedBytes verification
	Check(MatchesPhysicsTimestep(PhysicsTimestepOriginalBytes), "PhysicsTimestepOriginalBytes exact match");
	auto changed_timestep = PhysicsTimestepOriginalBytes;
	changed_timestep[0] ^= 1;
	Check(!MatchesPhysicsTimestep(changed_timestep), "modified physics timestep first byte rejected");
	changed_timestep = PhysicsTimestepOriginalBytes;
	changed_timestep[3] = 0xeb; // jmp instead of jnz
	Check(!MatchesPhysicsTimestep(changed_timestep), "modified physics timestep opcode rejected");
	changed_timestep = PhysicsTimestepOriginalBytes;
	changed_timestep[4] ^= 1;
	Check(!MatchesPhysicsTimestep(changed_timestep), "modified physics timestep displacement rejected");
	Check(!MatchesPhysicsTimestep(std::span(PhysicsTimestepOriginalBytes).first(3)), "short physics timestep signature rejected");
	Check(!MatchesPhysicsTimestep({}), "empty physics timestep signature rejected");

	static_assert(PhysicsTimestepOriginalBytes.size() == 5);
	static_assert(PhysicsTimestepPatchedBytes.size() == 5);
	// cmp byte [rax], 0x01 (80 38 01) followed by jnz +0x2a (75 2a)
	Check(PhysicsTimestepOriginalBytes[0] == 0x80 && PhysicsTimestepOriginalBytes[1] == 0x38 &&
	      PhysicsTimestepOriginalBytes[2] == 0x01, "cmp byte [rax], 0x01");
	Check(PhysicsTimestepOriginalBytes[3] == 0x75, "original jnz opcode");
	const auto timestep_disp = static_cast<int8_t>(PhysicsTimestepOriginalBytes[4]);
	Check(timestep_disp == 0x2a, "displacement +0x2a matches skip to clamped frame delta vmovss");
	Check(PhysicsTimestepOffset + 5 + timestep_disp == 0x00bdda31, "displacement jumps to 0x00bdda31");

	// Patched: cmp byte [rax], 0x01 (80 38 01) followed by jmp +0x2a (eb 2a)
	Check(PhysicsTimestepPatchedBytes[0] == 0x80 && PhysicsTimestepPatchedBytes[1] == 0x38 &&
	      PhysicsTimestepPatchedBytes[2] == 0x01, "patched cmp byte [rax], 0x01");
	Check(PhysicsTimestepPatchedBytes[3] == 0xeb, "patched jmp short opcode");
	Check(static_cast<int8_t>(PhysicsTimestepPatchedBytes[4]) == 0x2a, "patched displacement is +0x2a");
	Check(PhysicsTimestepOffset + 5 + static_cast<int8_t>(PhysicsTimestepPatchedBytes[4]) == 0x00bdda31,
	      "patched jump targets 0x00bdda31");

	// Timestep buffer patch and rollback test
	std::array<uint8_t, 5> timestep_test_buffer = PhysicsTimestepOriginalBytes;
	Check(timestep_test_buffer[3] == 0x75, "timestep buffer initial jnz");
	std::copy(PhysicsTimestepPatchedBytes.begin(), PhysicsTimestepPatchedBytes.end(), timestep_test_buffer.begin());
	Check(timestep_test_buffer == PhysicsTimestepPatchedBytes, "timestep buffer matches patched");
	std::copy(PhysicsTimestepOriginalBytes.begin(), PhysicsTimestepOriginalBytes.end(), timestep_test_buffer.begin());
	Check(timestep_test_buffer == PhysicsTimestepOriginalBytes, "timestep buffer matches original after rollback");

	// PhysicsCVarOriginalBytes and PhysicsCVarPatchedBytes verification
	Check(MatchesPhysicsCVar(PhysicsCVarOriginalBytes), "PhysicsCVarOriginalBytes exact match");
	auto changed_cvar = PhysicsCVarOriginalBytes;
	changed_cvar[0] ^= 1;
	Check(!MatchesPhysicsCVar(changed_cvar), "modified physics cvar first byte rejected");
	changed_cvar = PhysicsCVarOriginalBytes;
	changed_cvar[6] = 0;
	Check(!MatchesPhysicsCVar(changed_cvar), "modified first physicsUseFixedTimestep rejected");
	changed_cvar = PhysicsCVarOriginalBytes;
	changed_cvar[13] = 0;
	Check(!MatchesPhysicsCVar(changed_cvar), "modified second physicsUseFixedTimestep rejected");
	changed_cvar = PhysicsCVarOriginalBytes;
	changed_cvar[13] ^= 1;
	Check(!MatchesPhysicsCVar(changed_cvar), "modified physics cvar last byte rejected");
	Check(!MatchesPhysicsCVar(std::span(PhysicsCVarOriginalBytes).first(7)), "short physics cvar signature rejected");
	Check(!MatchesPhysicsCVar({}), "empty physics cvar signature rejected");

	static_assert(PhysicsCVarOriginalBytes.size() == 14);
	static_assert(PhysicsCVarPatchedBytes.size() == 14);

	// Instruction 1: mov byte [rip+disp32], 0x01 (c6 05 5f 53 3a 02 01)
	Check(PhysicsCVarOriginalBytes[0] == 0xc6 && PhysicsCVarOriginalBytes[1] == 0x05, "first cvar mov byte [rip+disp]");
	const auto cvar_disp1 = *reinterpret_cast<const int32_t*>(&PhysicsCVarOriginalBytes[2]);
	Check(cvar_disp1 == 0x023a535f, "first cvar disp32 matches physicsUseFixedTimestep relative offset");
	Check(PhysicsCVarOffset + 7 + cvar_disp1 == 0x02f89918, "first cvar targets 0x02f89918");
	Check(PhysicsCVarOriginalBytes[6] == 1, "original first physicsUseFixedTimestep is 1");

	// Instruction 2: mov byte [rip+disp32], 0x01 (c6 05 d8 52 3a 02 01)
	Check(PhysicsCVarOriginalBytes[7] == 0xc6 && PhysicsCVarOriginalBytes[8] == 0x05, "second cvar mov byte [rip+disp]");
	const auto cvar_disp2 = *reinterpret_cast<const int32_t*>(&PhysicsCVarOriginalBytes[9]);
	Check(cvar_disp2 == 0x023a52d8, "second cvar disp32 matches secondary physicsUseFixedTimestep relative offset");
	Check(PhysicsCVarOffset + 14 + cvar_disp2 == 0x02f89898, "second cvar targets 0x02f89898");
	Check(PhysicsCVarOriginalBytes[13] == 1, "original second physicsUseFixedTimestep is 1");

	// Verify PhysicsCVarPatchedBytes modifies CVar values to 0 while preserving opcodes and displacements
	Check(PhysicsCVarPatchedBytes[0] == 0xc6 && PhysicsCVarPatchedBytes[1] == 0x05, "patched first cvar opcode");
	Check(*reinterpret_cast<const int32_t*>(&PhysicsCVarPatchedBytes[2]) == cvar_disp1, "patched first cvar disp32 preserved");
	Check(PhysicsCVarPatchedBytes[6] == 0, "patched first physicsUseFixedTimestep is 0");

	Check(PhysicsCVarPatchedBytes[7] == 0xc6 && PhysicsCVarPatchedBytes[8] == 0x05, "patched second cvar opcode");
	Check(*reinterpret_cast<const int32_t*>(&PhysicsCVarPatchedBytes[9]) == cvar_disp2, "patched second cvar disp32 preserved");
	Check(PhysicsCVarPatchedBytes[13] == 0, "patched second physicsUseFixedTimestep is 0");

	// CVar buffer patch and rollback test
	std::array<uint8_t, 14> cvar_test_buffer = PhysicsCVarOriginalBytes;
	Check(cvar_test_buffer[6] == 1 && cvar_test_buffer[13] == 1, "cvar buffer initial values 1");
	std::copy(PhysicsCVarPatchedBytes.begin(), PhysicsCVarPatchedBytes.end(), cvar_test_buffer.begin());
	Check(cvar_test_buffer == PhysicsCVarPatchedBytes, "cvar buffer matches patched");
	Check(cvar_test_buffer[6] == 0 && cvar_test_buffer[13] == 0, "cvar buffer patched values 0");
	std::copy(PhysicsCVarOriginalBytes.begin(), PhysicsCVarOriginalBytes.end(), cvar_test_buffer.begin());
	Check(cvar_test_buffer == PhysicsCVarOriginalBytes, "cvar buffer matches original after rollback");

	// 2. WaitForJobSync condition evaluations (all 5 operators)
	// Immediate return with sync_ptr == 0
	WaitForJobSync(0, 100, 0, 0);

	// Immediate return when conditions already satisfied:
	// op 0: cur == expected_val
	uint64_t cur0 = 42;
	WaitForJobSync(reinterpret_cast<uint64_t>(&cur0), 42, 0, 0);

	// op 1: cur <= expected_val
	uint64_t cur1_eq = 42;
	WaitForJobSync(reinterpret_cast<uint64_t>(&cur1_eq), 42, 1, 0);
	uint64_t cur1_lt = 40;
	WaitForJobSync(reinterpret_cast<uint64_t>(&cur1_lt), 42, 1, 0);

	// op 2: cur < expected_val
	uint64_t cur2 = 40;
	WaitForJobSync(reinterpret_cast<uint64_t>(&cur2), 42, 2, 0);

	// op 3: cur >= expected_val
	uint64_t cur3_eq = 42;
	WaitForJobSync(reinterpret_cast<uint64_t>(&cur3_eq), 42, 3, 0);
	uint64_t cur3_gt = 45;
	WaitForJobSync(reinterpret_cast<uint64_t>(&cur3_gt), 42, 3, 0);

	// op 4: cur > expected_val
	uint64_t cur4 = 45;
	WaitForJobSync(reinterpret_cast<uint64_t>(&cur4), 42, 4, 0);

	// op >= 5: invalid operators must return immediately without spinning
	uint64_t cur_invalid = 0;
	WaitForJobSync(reinterpret_cast<uint64_t>(&cur_invalid), 42, 5, 0);
	WaitForJobSync(reinterpret_cast<uint64_t>(&cur_invalid), 42, 999, 0);

	// Boundary and invalid sync_ptr addresses must return immediately without dereference
	WaitForJobSync(0x0f, 42, 0, 0);
	WaitForJobSync(0x1000, 42, 0, 0);
	WaitForJobSync(0x00007ffffffffff9ull, 42, 0, 0);
	WaitForJobSync(0x00007ffffffffffaull, 42, 0, 0);
	WaitForJobSync(0x00007fffffffffffull, 42, 0, 0);

	// Asynchronous condition evaluations for all 5 operators
	for (uint32_t op = 0; op < 5; ++op) {
		uint64_t initial_val = 0;
		uint64_t expected_val = 100;
		uint64_t final_val = 0;
		switch (op) {
			case 0: initial_val = 10; final_val = 100; break;  // == 100
			case 1: initial_val = 200; final_val = 90; break;  // <= 100
			case 2: initial_val = 200; final_val = 50; break;  // < 100
			case 3: initial_val = 50; final_val = 100; break;  // >= 100
			case 4: initial_val = 50; final_val = 120; break;  // > 100
			default: break;
		}
		alignas(8) volatile uint64_t sync_val = initial_val;
		std::thread t([&sync_val, final_val]() {
			std::this_thread::sleep_for(std::chrono::milliseconds(2));
			sync_val = final_val;
		});
		WaitForJobSync(reinterpret_cast<uint64_t>(&sync_val), expected_val, op, op % 2);
		Check(sync_val == final_val, "WaitForJobSync async condition fulfilled");
		t.join();
	}

	// Dedicated can_sleep test: verify sleep path (can_sleep = 1) and spin/yield path (can_sleep = 0)
	{
		alignas(8) volatile uint64_t sleep_sync = 0;
		std::thread t1([&sleep_sync]() {
			std::this_thread::sleep_for(std::chrono::milliseconds(3));
			sleep_sync = 1;
		});
		WaitForJobSync(reinterpret_cast<uint64_t>(&sleep_sync), 1, 0, 1);
		Check(sleep_sync == 1, "WaitForJobSync sleep backoff fulfilled with can_sleep = 1");
		t1.join();

		alignas(8) volatile uint64_t spin_sync = 0;
		std::thread t2([&spin_sync]() {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			spin_sync = 1;
		});
		WaitForJobSync(reinterpret_cast<uint64_t>(&spin_sync), 1, 0, 0);
		Check(spin_sync == 1, "WaitForJobSync yield backoff fulfilled with can_sleep = 0");
		t2.join();
	}

#if defined(__x86_64__) || defined(_M_X64)
	using namespace Xbyak::util;
	uint32_t             waits = 0;
	Xbyak::CodeGenerator code(8192);
	const auto*          sleep = code.getCurr();
	code.mov(rax, reinterpret_cast<uint64_t>(&waits));
	code.inc(dword[rax]);
	for (const auto& r: {rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11})
		code.xor_(r, r);
	code.pxor(xmm0, xmm0);
	code.pxor(xmm15, xmm15);
	code.ret();
	const auto* poll = code.getCurr();
	code.mov(rax, UINT64_C(0x1122334455667700));
	code.or_(rax, rdi); // The success value is AL, not the full return register.
	unsigned i = 1;
	for (const auto& r: {rcx, rdx, rsi, rdi, r8, r9, r10, r11})
		code.mov(r, 0xabcdef00u + i++);
	code.movq(xmm0, rax);
	code.pcmpeqd(xmm15, xmm15);
	code.ret();
	const auto* thunk = code.getCurr();
	EmitThunk(code, poll, sleep);
	// SysV runner: (output, busy, target). Save host callee-saved registers,
	// including Win64's nonvolatile XMM registers through the compiler bridge.
	const auto* run = code.getCurr();
	code.push(r12);
	code.mov(r12, rdi);
	code.mov(rdi, rsi);
	code.call(rdx);
	unsigned offset = 0;
	for (const auto& r: {rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11}) {
		code.mov(qword[r12 + offset], r);
		offset += 8;
	}
	code.movdqu(ptr[r12 + offset], xmm0);
	code.movdqu(ptr[r12 + offset + 16], xmm15);
	code.pop(r12);
	code.ret();

	// 3. EmitJobSyncThunk generation and execution test
	const auto* return_landing = code.getCurr();
	code.pop(rbx);
	code.pop(r13);
	code.pop(r14);
	code.pop(r15);
	code.ret();

	const auto* job_thunk = code.getCurr();
	EmitJobSyncThunk(code, reinterpret_cast<const void*>(&MockWaitCallback), return_landing);

	const auto* run_job_thunk = code.getCurr();
	// Caller in SysV ABI: (sync_ptr: rdi, expected_val: rsi, op: edx, can_sleep: ecx)
	// Preserve callee-saved registers:
	code.push(r15);
	code.push(r14);
	code.push(r13);
	code.push(rbx);
	code.mov(r15, rdi);
	code.mov(r14, rsi);
	code.mov(ebx, edx);
	code.mov(r13d, ecx);
	code.mov(rax, reinterpret_cast<uint64_t>(job_thunk));
	code.jmp(rax);

	code.ready();
	Check(!Xbyak::GetError(), "thunk generation");
	using Run         = KYTY_SYSV_ABI void (*)(void*, uint64_t, const void*);
	const auto runner = reinterpret_cast<Run>(reinterpret_cast<uintptr_t>(run));
	for (uint64_t busy: {0, 1}) {
		std::array<uint8_t, 104> expected {}, actual {};
		runner(expected.data(), busy, poll);
		const auto before = waits;
		runner(actual.data(), busy, thunk);
		Check(expected == actual, "poll GPR and SIMD state must survive the host callback");
		Check(waits == before + (busy == 0), "only an empty poll may wait");
	}

	using JobSyncRunner = KYTY_SYSV_ABI void (*)(uint64_t, uint64_t, uint32_t, uint32_t);
	const auto job_runner = reinterpret_cast<JobSyncRunner>(reinterpret_cast<uintptr_t>(run_job_thunk));
	captured_sync_ptr = 0;
	captured_expected_val = 0;
	captured_op = 0;
	captured_can_sleep = 0;
	job_runner(0x1122334455667788ull, 0x99aabbccddeeff00ull, 3, 1);
	Check(captured_sync_ptr == 0x1122334455667788ull, "JobSyncThunk correctly forwarded sync_ptr from r15");
	Check(captured_expected_val == 0x99aabbccddeeff00ull, "JobSyncThunk correctly forwarded expected_val from r14");
	Check(captured_op == 3, "JobSyncThunk correctly forwarded op from ebx");
	Check(captured_can_sleep == 1, "JobSyncThunk correctly forwarded can_sleep from r13d");

	// 4. End-to-end Install, Clear, and error rollback integration test for physics timestep and cvar
	constexpr uint64_t TestCaveOffset = 0x8000000;
	constexpr uint64_t TestPageSize = 0x4000;
	constexpr uint64_t TestTotalSize = TestCaveOffset + TestPageSize;
	const uint64_t base = Common::VirtualMemory::Alloc(0, TestTotalSize, Common::VirtualMemory::Mode::ExecuteReadWrite);
	Check(base != 0, "Alloc virtual memory block for guest test program");

	std::memcpy(reinterpret_cast<void*>(base + CallOffset), CallBytes.data(), CallBytes.size());
	std::memcpy(reinterpret_cast<void*>(base + PollOffset), PollBytes.data(), PollBytes.size());
	std::memcpy(reinterpret_cast<void*>(base + JobSyncOffset), JobSyncBytes.data(), JobSyncBytes.size());
	std::memcpy(reinterpret_cast<void*>(base + PhysicsTimestepOffset), PhysicsTimestepOriginalBytes.data(),
	            PhysicsTimestepOriginalBytes.size());
	std::memcpy(reinterpret_cast<void*>(base + PhysicsCVarOffset), PhysicsCVarOriginalBytes.data(),
	            PhysicsCVarOriginalBytes.size());

	Loader::Program program;
	program.file_name = "eboot.bin";
	program.mapped_size = TestCaveOffset;
	program.base_vaddr = base;

	g_mock_is_supported = true;
	g_mock_readable = true;

	// Perform initial Install
	Install(&program);

	const auto* timestep_mem = reinterpret_cast<const uint8_t*>(base + PhysicsTimestepOffset);
	const auto* cvar_mem = reinterpret_cast<const uint8_t*>(base + PhysicsCVarOffset);
	Check(std::memcmp(timestep_mem, PhysicsTimestepPatchedBytes.data(), PhysicsTimestepPatchedBytes.size()) == 0,
	      "Install patches physics timestep to PhysicsTimestepPatchedBytes");
	Check(timestep_mem[3] == 0xeb, "Install patches jnz to jmp short");

	Check(std::memcmp(cvar_mem, PhysicsCVarPatchedBytes.data(), PhysicsCVarPatchedBytes.size()) == 0,
	      "Install patches physics cvar to PhysicsCVarPatchedBytes");
	Check(cvar_mem[6] == 0 && cvar_mem[13] == 0, "Install sets physicsUseFixedTimestep to 0");

	// Perform Clear
	Clear();

	Check(std::memcmp(timestep_mem, PhysicsTimestepOriginalBytes.data(), PhysicsTimestepOriginalBytes.size()) == 0,
	      "Clear restores physics timestep memory cleanly to PhysicsTimestepOriginalBytes");
	Check(timestep_mem[3] == 0x75, "Clear restores jnz opcode");

	Check(std::memcmp(cvar_mem, PhysicsCVarOriginalBytes.data(), PhysicsCVarOriginalBytes.size()) == 0,
	      "Clear restores physics cvar memory cleanly to PhysicsCVarOriginalBytes");
	Check(cvar_mem[6] == 1 && cvar_mem[13] == 1, "Clear restores physicsUseFixedTimestep to 1");

	// Verify error rollback: simulate ProtectGuestMemory failure on physics_timestep
	g_test_timestep_addr = base + PhysicsTimestepOffset;
	g_mock_protect_fail_on_timestep = true;

	Install(&program);

	// Because protect failed on physics_timestep, all hooks must be rolled back:
	Check(std::memcmp(timestep_mem, PhysicsTimestepOriginalBytes.data(), PhysicsTimestepOriginalBytes.size()) == 0,
	      "Rollback keeps physics timestep memory intact as PhysicsTimestepOriginalBytes");
	const auto* call_mem = reinterpret_cast<const uint8_t*>(base + CallOffset);
	Check(std::memcmp(call_mem, CallBytes.data(), CallBytes.size()) == 0,
	      "Rollback restores call bytes cleanly on timestep failure");
	const auto* job_sync_mem = reinterpret_cast<const uint8_t*>(base + JobSyncOffset);
	Check(std::memcmp(job_sync_mem, JobSyncBytes.data(), JobSyncBytes.size()) == 0,
	      "Rollback restores job sync bytes cleanly on timestep failure");

	g_mock_protect_fail_on_timestep = false;
	g_test_timestep_addr = 0;

	// Verify error rollback: simulate ProtectGuestMemory failure on physics_cvar
	g_test_cvar_addr = base + PhysicsCVarOffset;
	g_mock_protect_fail_on_cvar = true;

	Install(&program);

	// Because protect failed on physics_cvar, all hooks (timestep, job_sync, call) must be rolled back:
	Check(std::memcmp(cvar_mem, PhysicsCVarOriginalBytes.data(), PhysicsCVarOriginalBytes.size()) == 0,
	      "Rollback keeps physics cvar memory intact as PhysicsCVarOriginalBytes");
	Check(std::memcmp(timestep_mem, PhysicsTimestepOriginalBytes.data(), PhysicsTimestepOriginalBytes.size()) == 0,
	      "Rollback restores physics timestep memory cleanly on cvar failure");
	Check(std::memcmp(call_mem, CallBytes.data(), CallBytes.size()) == 0,
	      "Rollback restores call bytes cleanly on cvar failure");
	Check(std::memcmp(job_sync_mem, JobSyncBytes.data(), JobSyncBytes.size()) == 0,
	      "Rollback restores job sync bytes cleanly on cvar failure");

	g_mock_protect_fail_on_cvar = false;
	g_test_cvar_addr = 0;

	// Verify we can cleanly install again after rollback
	Install(&program);
	Check(std::memcmp(timestep_mem, PhysicsTimestepPatchedBytes.data(), PhysicsTimestepPatchedBytes.size()) == 0,
	      "Install succeeds cleanly after previous rollbacks (timestep)");
	Check(std::memcmp(cvar_mem, PhysicsCVarPatchedBytes.data(), PhysicsCVarPatchedBytes.size()) == 0,
	      "Install succeeds cleanly after previous rollbacks (cvar)");

	Clear();
	Check(std::memcmp(timestep_mem, PhysicsTimestepOriginalBytes.data(), PhysicsTimestepOriginalBytes.size()) == 0,
	      "Clear restores physics timestep after re-installation");
	Check(std::memcmp(cvar_mem, PhysicsCVarOriginalBytes.data(), PhysicsCVarOriginalBytes.size()) == 0,
	      "Clear restores physics cvar after re-installation");

	// Verify Clear() handles ProtectGuestMemory failure safely without crash or memory corruption
	Install(&program);
	Check(std::memcmp(timestep_mem, PhysicsTimestepPatchedBytes.data(), PhysicsTimestepPatchedBytes.size()) == 0,
	      "Install succeeds before Clear protect fail test");
	g_mock_protect_fail = true;
	Clear();
	Check(std::memcmp(timestep_mem, PhysicsTimestepPatchedBytes.data(), PhysicsTimestepPatchedBytes.size()) == 0,
	      "Clear leaves timestep memory intact when protect fails");
	Check(std::memcmp(cvar_mem, PhysicsCVarPatchedBytes.data(), PhysicsCVarPatchedBytes.size()) == 0,
	      "Clear leaves cvar memory intact when protect fails");
	g_mock_protect_fail = false;

	// Reset mock buffers back to original
	std::memcpy(const_cast<uint8_t*>(timestep_mem), PhysicsTimestepOriginalBytes.data(),
	            PhysicsTimestepOriginalBytes.size());
	std::memcpy(const_cast<uint8_t*>(cvar_mem), PhysicsCVarOriginalBytes.data(),
	            PhysicsCVarOriginalBytes.size());

	// Verify idempotency of Clear() when already uninstalled / called multiple times
	Clear();
	Clear();

	// Verify mapped_size checks reject programs smaller than required offsets
	program.mapped_size = CallOffset + CallBytes.size() - 1;
	Install(&program);
	Check(std::memcmp(timestep_mem, PhysicsTimestepOriginalBytes.data(), PhysicsTimestepOriginalBytes.size()) == 0,
	      "Install rejected when mapped_size < CallOffset + CallBytes.size()");

	program.mapped_size = PollOffset + PollBytes.size() - 1;
	Install(&program);
	Check(std::memcmp(timestep_mem, PhysicsTimestepOriginalBytes.data(), PhysicsTimestepOriginalBytes.size()) == 0,
	      "Install rejected when mapped_size < PollOffset + PollBytes.size()");

	program.mapped_size = JobSyncOffset + JobSyncBytes.size() - 1;
	Install(&program);
	Check(std::memcmp(timestep_mem, PhysicsTimestepOriginalBytes.data(), PhysicsTimestepOriginalBytes.size()) == 0,
	      "Install rejected when mapped_size < JobSyncOffset + JobSyncBytes.size()");

	program.mapped_size = PhysicsTimestepOffset + PhysicsTimestepOriginalBytes.size() - 1;
	Install(&program);
	Check(std::memcmp(timestep_mem, PhysicsTimestepOriginalBytes.data(), PhysicsTimestepOriginalBytes.size()) == 0,
	      "Install rejected when mapped_size < PhysicsTimestepOffset + PhysicsTimestepOriginalBytes.size()");

	program.mapped_size = PhysicsCVarOffset + PhysicsCVarOriginalBytes.size() - 1;
	Install(&program);
	Check(std::memcmp(timestep_mem, PhysicsTimestepOriginalBytes.data(), PhysicsTimestepOriginalBytes.size()) == 0,
	      "Install rejected when mapped_size < PhysicsCVarOffset + PhysicsCVarOriginalBytes.size()");

	g_mock_is_supported = false;
	g_mock_readable = false;
	Common::VirtualMemory::Free(base);
#endif
	std::puts("Demon's Souls idle wait tests passed");
}
