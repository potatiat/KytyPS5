#include "loader/demonsSoulsIdle.h"

#include "common/logging/log.h"
#include "common/threads.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "loader/runtimeLinker.h"

#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include <thread>

#include <atomic>

namespace Loader::DemonsSoulsIdle {
namespace {
constexpr uint64_t     PageSize = 0x4000, CaveOffset = 0x8000000;
uint64_t               cave = 0, site = 0, job_sync_site = 0;
uint64_t               physics_timestep_site = 0, physics_cvar_site = 0;
std::array<uint8_t, 5> installed_call {};
std::array<uint8_t, JobSyncBytes.size()> installed_job_sync {};

void KYTY_SYSV_ABI WaitForWork() {
	Common::Thread::SleepMicroWithoutSpinning(50);
}

inline uint64_t ReadSyncValue(uint64_t sync_ptr) noexcept {
	if ((sync_ptr & 7u) == 0) {
		return std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t*>(sync_ptr))
		    .load(std::memory_order_acquire);
	}
	const auto val = *reinterpret_cast<const volatile uint64_t*>(sync_ptr);
	std::atomic_thread_fence(std::memory_order_acquire);
	return val;
}
} // namespace

void KYTY_SYSV_ABI WaitForJobSync(uint64_t sync_ptr, uint64_t expected_val, uint32_t op, uint32_t can_sleep) {
	if (sync_ptr < 0x10000 || sync_ptr > 0x00007ffffffffff8ull || op > 4) return;
	auto check = [op, expected_val](uint64_t cur) noexcept -> bool {
		const auto c = static_cast<int64_t>(cur);
		const auto e = static_cast<int64_t>(expected_val);
		switch (op) {
			case 0: return c == e;
			case 1: return c <= e;
			case 2: return c < e;
			case 3: return c >= e;
			case 4: return c > e;
			default: return false;
		}
	};
	if (check(ReadSyncValue(sync_ptr))) return;

	for (uint32_t spin = 0; !check(ReadSyncValue(sync_ptr)); ++spin) {
		if (spin < 64) {
			_mm_pause();
		} else if (can_sleep) {
			Common::Thread::SleepMicroWithoutSpinning(20);
		} else {
			std::this_thread::yield();
		}
	}
}

#if defined(__x86_64__) || defined(_M_X64)
void EmitJobSyncThunk(Xbyak::CodeGenerator& c, const void* native_wait, const void* return_site) {
	using namespace Xbyak::util;
	c.mov(rdi, r15);       // sync_ptr
	c.mov(rsi, r14);       // expected_val
	c.mov(edx, ebx);       // op
	c.mov(ecx, r13d);      // can_sleep
	c.mov(rax, reinterpret_cast<uint64_t>(native_wait));
	c.call(rax);
	c.mov(rax, reinterpret_cast<uint64_t>(return_site));
	c.jmp(rax);
}
#endif

void Install(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	if (cave != 0 || site != 0 || job_sync_site != 0 ||
	    physics_timestep_site != 0 || physics_cvar_site != 0 ||
	    program == nullptr ||
	    !Libs::Graphics::DemonsSouls::IsSupportedGame() ||
	    program->file_name.filename() != "eboot.bin" ||
	    program->mapped_size < CallOffset + CallBytes.size() ||
	    program->mapped_size < PollOffset + PollBytes.size() ||
	    program->mapped_size < JobSyncOffset + JobSyncBytes.size() ||
	    program->mapped_size < PhysicsTimestepOffset + PhysicsTimestepOriginalBytes.size() ||
	    program->mapped_size < PhysicsCVarOffset + PhysicsCVarOriginalBytes.size() ||
	    program->mapped_size > CaveOffset ||
	    program->base_vaddr > UINT64_MAX - CaveOffset - PageSize)
		return;
	const auto call = program->base_vaddr + CallOffset;
	const auto poll = program->base_vaddr + PollOffset;
	const auto job_sync = program->base_vaddr + JobSyncOffset;
	const auto job_sync_return = program->base_vaddr + JobSyncReturnOffset;
	const auto physics_timestep = program->base_vaddr + PhysicsTimestepOffset;
	const auto physics_cvar = program->base_vaddr + PhysicsCVarOffset;
	std::array<uint8_t, CallBytes.size()> call_bytes {};
	std::array<uint8_t, PollBytes.size()> poll_bytes {};
	std::array<uint8_t, JobSyncBytes.size()> job_sync_bytes {};
	std::array<uint8_t, PhysicsTimestepOriginalBytes.size()> physics_timestep_bytes {};
	std::array<uint8_t, PhysicsCVarOriginalBytes.size()> physics_cvar_bytes {};
	// Executable modules are private runtime allocations, not GPU backing aliases.
	if (!Libs::Graphics::HostMemoryRangeIsReadable(call, call_bytes.size()) ||
	    !Libs::Graphics::HostMemoryRangeIsReadable(poll, poll_bytes.size()) ||
	    !Libs::Graphics::HostMemoryRangeIsReadable(job_sync, job_sync_bytes.size()) ||
	    !Libs::Graphics::HostMemoryRangeIsReadable(physics_timestep, physics_timestep_bytes.size()) ||
	    !Libs::Graphics::HostMemoryRangeIsReadable(physics_cvar, physics_cvar_bytes.size()))
		return;
	std::memcpy(call_bytes.data(), reinterpret_cast<const void*>(call), call_bytes.size());
	std::memcpy(poll_bytes.data(), reinterpret_cast<const void*>(poll), poll_bytes.size());
	std::memcpy(job_sync_bytes.data(), reinterpret_cast<const void*>(job_sync), job_sync_bytes.size());
	std::memcpy(physics_timestep_bytes.data(), reinterpret_cast<const void*>(physics_timestep),
	            physics_timestep_bytes.size());
	std::memcpy(physics_cvar_bytes.data(), reinterpret_cast<const void*>(physics_cvar),
	            physics_cvar_bytes.size());
	if (!Matches(call_bytes, poll_bytes)) {
		LOGF("Demon's Souls idle wait: code signature differs; retaining guest code\n");
		return;
	}
	if (!MatchesJobSync(job_sync_bytes)) {
		LOGF("Demon's Souls job sync: code signature differs; retaining guest code\n");
		return;
	}
	if (!MatchesPhysicsTimestep(physics_timestep_bytes)) {
		LOGF("Demon's Souls physics timestep: code signature differs; retaining guest code\n");
		return;
	}
	if (!MatchesPhysicsCVar(physics_cvar_bytes)) {
		LOGF("Demon's Souls physics cvar: code signature differs; retaining guest code\n");
		return;
	}
	const auto requested = program->base_vaddr + CaveOffset;
	const auto allocated = Libs::LibKernel::Memory::AllocateRuntimeMemory(
	    requested, PageSize, Common::VirtualMemory::Mode::ExecuteReadWrite,
	    "demons_souls_idle_wait", true);
	if (allocated != requested) {
		if (allocated) Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}
	Xbyak::ClearError();
	Xbyak::CodeGenerator code(PageSize, reinterpret_cast<void*>(allocated));
	EmitThunk(code, reinterpret_cast<const void*>(poll),
	          reinterpret_cast<const void*>(&WaitForWork));
	const auto* job_sync_thunk = code.getCurr();
	EmitJobSyncThunk(code, reinterpret_cast<const void*>(&WaitForJobSync),
	                 reinterpret_cast<const void*>(job_sync_return));
	code.ready();
	if (Xbyak::GetError() ||
	    !Common::VirtualMemory::FlushInstructionCache(allocated, code.getSize()) ||
	    !Libs::LibKernel::Memory::ProtectGuestMemory(
	        allocated, PageSize, Common::VirtualMemory::Mode::ExecuteRead, nullptr)) {
		Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}

	auto restore_site = [](uint64_t vaddr, std::span<const uint8_t> orig, Common::VirtualMemory::Mode mode) {
		if (Libs::LibKernel::Memory::ProtectGuestMemory(
		        vaddr, orig.size(), Common::VirtualMemory::Mode::ExecuteReadWrite, nullptr)) {
			std::memcpy(reinterpret_cast<void*>(vaddr), orig.data(), orig.size());
			Libs::LibKernel::Memory::ProtectGuestMemory(vaddr, orig.size(), mode);
			Common::VirtualMemory::FlushInstructionCache(vaddr, orig.size());
		}
	};

	installed_call[0]       = 0xe8;
	const auto displacement = static_cast<int32_t>(CaveOffset - CallOffset - 5);
	std::memcpy(installed_call.data() + 1, &displacement, sizeof(displacement));

	Common::VirtualMemory::Mode old_call_mode {};
	if (!Libs::LibKernel::Memory::ProtectGuestMemory(
	        call, installed_call.size(),
	        Common::VirtualMemory::Mode::ExecuteReadWrite, &old_call_mode)) {
		Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}
	std::memcpy(reinterpret_cast<void*>(call), installed_call.data(), installed_call.size());
	const bool call_restore_ok = Libs::LibKernel::Memory::ProtectGuestMemory(
	    call, installed_call.size(), old_call_mode);
	const bool call_flush_ok = Common::VirtualMemory::FlushInstructionCache(call, installed_call.size());

	if (!call_restore_ok || !call_flush_ok) {
		restore_site(call, CallBytes, old_call_mode);
		Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}

	installed_job_sync[0] = 0xe9;
	const auto job_sync_displacement = static_cast<int32_t>(
	    reinterpret_cast<uintptr_t>(job_sync_thunk) - (job_sync + 5));
	std::memcpy(installed_job_sync.data() + 1, &job_sync_displacement, sizeof(job_sync_displacement));
	std::fill(installed_job_sync.begin() + 5, installed_job_sync.end(), uint8_t {0x90});

	Common::VirtualMemory::Mode old_job_sync_mode {};
	if (!Libs::LibKernel::Memory::ProtectGuestMemory(
	        job_sync, installed_job_sync.size(),
	        Common::VirtualMemory::Mode::ExecuteReadWrite, &old_job_sync_mode)) {
		restore_site(call, CallBytes, old_call_mode);
		Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}

	std::memcpy(reinterpret_cast<void*>(job_sync), installed_job_sync.data(), installed_job_sync.size());
	const bool job_sync_restore_ok = Libs::LibKernel::Memory::ProtectGuestMemory(
	    job_sync, installed_job_sync.size(), old_job_sync_mode);
	const bool job_sync_flush_ok =
	    Common::VirtualMemory::FlushInstructionCache(job_sync, installed_job_sync.size());

	if (!job_sync_restore_ok || !job_sync_flush_ok) {
		restore_site(job_sync, JobSyncBytes, old_job_sync_mode);
		restore_site(call, CallBytes, old_call_mode);
		Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}

	Common::VirtualMemory::Mode old_timestep_mode {};
	if (!Libs::LibKernel::Memory::ProtectGuestMemory(
	        physics_timestep, PhysicsTimestepPatchedBytes.size(),
	        Common::VirtualMemory::Mode::ExecuteReadWrite, &old_timestep_mode)) {
		restore_site(job_sync, JobSyncBytes, old_job_sync_mode);
		restore_site(call, CallBytes, old_call_mode);
		Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}

	std::memcpy(reinterpret_cast<void*>(physics_timestep), PhysicsTimestepPatchedBytes.data(),
	            PhysicsTimestepPatchedBytes.size());
	const bool timestep_restore_ok = Libs::LibKernel::Memory::ProtectGuestMemory(
	    physics_timestep, PhysicsTimestepPatchedBytes.size(), old_timestep_mode);
	const bool timestep_flush_ok =
	    Common::VirtualMemory::FlushInstructionCache(physics_timestep, PhysicsTimestepPatchedBytes.size());

	if (!timestep_restore_ok || !timestep_flush_ok) {
		restore_site(physics_timestep, PhysicsTimestepOriginalBytes, old_timestep_mode);
		restore_site(job_sync, JobSyncBytes, old_job_sync_mode);
		restore_site(call, CallBytes, old_call_mode);
		Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}

	Common::VirtualMemory::Mode old_cvar_mode {};
	if (!Libs::LibKernel::Memory::ProtectGuestMemory(
	        physics_cvar, PhysicsCVarPatchedBytes.size(),
	        Common::VirtualMemory::Mode::ExecuteReadWrite, &old_cvar_mode)) {
		restore_site(physics_timestep, PhysicsTimestepOriginalBytes, old_timestep_mode);
		restore_site(job_sync, JobSyncBytes, old_job_sync_mode);
		restore_site(call, CallBytes, old_call_mode);
		Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}

	std::memcpy(reinterpret_cast<void*>(physics_cvar), PhysicsCVarPatchedBytes.data(),
	            PhysicsCVarPatchedBytes.size());
	const bool cvar_restore_ok = Libs::LibKernel::Memory::ProtectGuestMemory(
	    physics_cvar, PhysicsCVarPatchedBytes.size(), old_cvar_mode);
	const bool cvar_flush_ok =
	    Common::VirtualMemory::FlushInstructionCache(physics_cvar, PhysicsCVarPatchedBytes.size());

	if (!cvar_restore_ok || !cvar_flush_ok) {
		restore_site(physics_cvar, PhysicsCVarOriginalBytes, old_cvar_mode);
		restore_site(physics_timestep, PhysicsTimestepOriginalBytes, old_timestep_mode);
		restore_site(job_sync, JobSyncBytes, old_job_sync_mode);
		restore_site(call, CallBytes, old_call_mode);
		Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}

	cave = allocated;
	site = call;
	job_sync_site = job_sync;
	physics_timestep_site = physics_timestep;
	physics_cvar_site = physics_cvar;
	LOGF("Demon's Souls idle wait: installed portable 50 us backoff, job sync hook, and physics timestep/cvar patches\n");
#endif
}

void Clear() {
	if (physics_cvar_site) {
		std::array<uint8_t, PhysicsCVarPatchedBytes.size()> bytes {};
		std::memcpy(bytes.data(), reinterpret_cast<const void*>(physics_cvar_site), bytes.size());
		if (bytes == PhysicsCVarPatchedBytes) {
			Common::VirtualMemory::Mode old_mode {};
			if (Libs::LibKernel::Memory::ProtectGuestMemory(
			        physics_cvar_site, PhysicsCVarOriginalBytes.size(),
			        Common::VirtualMemory::Mode::ExecuteReadWrite, &old_mode)) {
				std::memcpy(reinterpret_cast<void*>(physics_cvar_site),
				            PhysicsCVarOriginalBytes.data(),
				            PhysicsCVarOriginalBytes.size());
				Libs::LibKernel::Memory::ProtectGuestMemory(
				    physics_cvar_site, PhysicsCVarOriginalBytes.size(), old_mode);
				Common::VirtualMemory::FlushInstructionCache(
				    physics_cvar_site, PhysicsCVarOriginalBytes.size());
			}
		}
		physics_cvar_site = 0;
	}
	if (physics_timestep_site) {
		std::array<uint8_t, PhysicsTimestepPatchedBytes.size()> bytes {};
		std::memcpy(bytes.data(), reinterpret_cast<const void*>(physics_timestep_site), bytes.size());
		if (bytes == PhysicsTimestepPatchedBytes) {
			Common::VirtualMemory::Mode old_mode {};
			if (Libs::LibKernel::Memory::ProtectGuestMemory(
			        physics_timestep_site, PhysicsTimestepOriginalBytes.size(),
			        Common::VirtualMemory::Mode::ExecuteReadWrite, &old_mode)) {
				std::memcpy(reinterpret_cast<void*>(physics_timestep_site),
				            PhysicsTimestepOriginalBytes.data(),
				            PhysicsTimestepOriginalBytes.size());
				Libs::LibKernel::Memory::ProtectGuestMemory(
				    physics_timestep_site, PhysicsTimestepOriginalBytes.size(), old_mode);
				Common::VirtualMemory::FlushInstructionCache(
				    physics_timestep_site, PhysicsTimestepOriginalBytes.size());
			}
		}
		physics_timestep_site = 0;
	}
	if (job_sync_site) {
		std::array<uint8_t, JobSyncBytes.size()> bytes {};
		std::memcpy(bytes.data(), reinterpret_cast<const void*>(job_sync_site), bytes.size());
		if (bytes == installed_job_sync) {
			Common::VirtualMemory::Mode old_mode {};
			if (Libs::LibKernel::Memory::ProtectGuestMemory(
			        job_sync_site, JobSyncBytes.size(),
			        Common::VirtualMemory::Mode::ExecuteReadWrite, &old_mode)) {
				std::memcpy(reinterpret_cast<void*>(job_sync_site),
				            JobSyncBytes.data(), JobSyncBytes.size());
				Libs::LibKernel::Memory::ProtectGuestMemory(
				    job_sync_site, JobSyncBytes.size(), old_mode);
				Common::VirtualMemory::FlushInstructionCache(
				    job_sync_site, JobSyncBytes.size());
			}
		}
		job_sync_site = 0;
	}
	if (site) {
		std::array<uint8_t, 5> bytes {};
		std::memcpy(bytes.data(), reinterpret_cast<const void*>(site), bytes.size());
		if (bytes == installed_call) {
			Common::VirtualMemory::Mode old_mode {};
			if (Libs::LibKernel::Memory::ProtectGuestMemory(
			        site, bytes.size(),
			        Common::VirtualMemory::Mode::ExecuteReadWrite, &old_mode)) {
				std::memcpy(reinterpret_cast<void*>(site), CallBytes.data(), bytes.size());
				Libs::LibKernel::Memory::ProtectGuestMemory(
				    site, bytes.size(), old_mode);
				Common::VirtualMemory::FlushInstructionCache(site, bytes.size());
			}
		}
		site = 0;
	}
	if (cave) {
		Libs::LibKernel::Memory::FreeGuestMemory(cave, PageSize);
		cave = 0;
	}
}
} // namespace Loader::DemonsSoulsIdle
