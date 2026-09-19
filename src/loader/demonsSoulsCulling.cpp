#include "loader/demonsSoulsCulling.h"

#include "common/logging/log.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "loader/runtimeLinker.h"

#include <algorithm>
#include <cstring>

namespace Loader::DemonsSoulsCulling {
namespace {
uint64_t                cave = 0, site = 0;
std::array<uint8_t, 17> installed_patch {};
} // namespace

void KYTY_SYSV_ABI NativeSortDrawPackets([[maybe_unused]] uint64_t job_context, uint64_t packet_list,
                                        uint64_t comp_fn) {
	if (packet_list < 0x10000 || packet_list > 0x00007fffffffffffull - 0x160 || (packet_list & 7) != 0) {
		return;
	}
	auto** first_ptr = reinterpret_cast<DrawPacket**>(packet_list + 0x150);
	auto** last_ptr  = reinterpret_cast<DrawPacket**>(packet_list + 0x158);
	if (first_ptr && last_ptr && *first_ptr && *last_ptr && *first_ptr < *last_ptr) {
		const auto first_addr = reinterpret_cast<uintptr_t>(*first_ptr);
		const auto last_addr  = reinterpret_cast<uintptr_t>(*last_ptr);
		if (first_addr < 0x10000 || first_addr > 0x00007fffffffffffull ||
		    last_addr < 0x10000 || last_addr > 0x00007fffffffffffull ||
		    (first_addr & 7) != 0 || (last_addr & 7) != 0) {
			return;
		}
		const auto diff = last_addr - first_addr;
		if ((diff % sizeof(DrawPacket)) != 0 || diff > 0x04000000) {
			return;
		}
		if (comp_fn < 0x10000 || comp_fn > 0x00007fffffffffffull || (comp_fn & 1) != 0) {
			comp_fn = 0;
		}
		auto guest_comp = reinterpret_cast<Comparator>(comp_fn);
		std::sort(*first_ptr, *last_ptr, [guest_comp](const DrawPacket& a, const DrawPacket& b) {
			if (a.key != b.key) {
				return a.key < b.key;
			}
			if (&a == &b || a == b) {
				return false;
			}
			return guest_comp != nullptr ? guest_comp(&a, &b) : false;
		});
	}
}

void Install(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	if (cave != 0 || site != 0 || program == nullptr ||
	    !Libs::Graphics::DemonsSouls::IsSupportedGame() ||
	    program->file_name.filename() != "eboot.bin" ||
	    program->mapped_size < SortOffset + PrologueBytes.size() ||
	    program->mapped_size < ContinueOffset ||
	    program->mapped_size < ComparatorOffset ||
	    program->mapped_size > CaveOffset ||
	    program->base_vaddr > UINT64_MAX - CaveOffset - PageSize)
		return;

	const auto sort_site = program->base_vaddr + SortOffset;
	const auto cont_site = program->base_vaddr + ContinueOffset;
	const auto comp_site = program->base_vaddr + ComparatorOffset;

	if (!Libs::Graphics::HostMemoryRangeIsReadable(sort_site, PrologueBytes.size()) ||
	    !Libs::Graphics::HostMemoryRangeIsReadable(cont_site, 1) ||
	    !Libs::Graphics::HostMemoryRangeIsReadable(comp_site, 1))
		return;

	std::array<uint8_t, PrologueBytes.size()> sort_bytes {};
	std::memcpy(sort_bytes.data(), reinterpret_cast<const void*>(sort_site), sort_bytes.size());
	if (!MatchesPrologue(sort_bytes)) {
		LOGF("Demon's Souls culling sort: code signature differs; retaining guest code\n");
		return;
	}

	const auto requested = program->base_vaddr + CaveOffset;
	const auto allocated = Libs::LibKernel::Memory::AllocateRuntimeMemory(
	    requested, PageSize, Common::VirtualMemory::Mode::ExecuteReadWrite,
	    "demons_souls_culling_sort", true);
	if (allocated != requested) {
		if (allocated) Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}

	Xbyak::ClearError();
	Xbyak::CodeGenerator code(PageSize, reinterpret_cast<void*>(allocated));
	EmitCullingSortThunk(code, reinterpret_cast<const void*>(&NativeSortDrawPackets), comp_site,
	                    cont_site);
	code.ready();
	if (Xbyak::GetError() ||
	    !Common::VirtualMemory::FlushInstructionCache(allocated, code.getSize()) ||
	    !Libs::LibKernel::Memory::ProtectGuestMemory(
	        allocated, PageSize, Common::VirtualMemory::Mode::ExecuteRead, nullptr)) {
		Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}

	installed_patch[0] = 0xe9;
	const auto raw_displacement = static_cast<int64_t>(allocated - (sort_site + 5));
	if (raw_displacement < INT32_MIN || raw_displacement > INT32_MAX) {
		Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}
	const auto displacement = static_cast<int32_t>(raw_displacement);
	std::memcpy(installed_patch.data() + 1, &displacement, sizeof(displacement));
	std::fill(installed_patch.begin() + 5, installed_patch.end(), uint8_t {0x90});

	Common::VirtualMemory::Mode old_sort_mode {};
	if (!Libs::LibKernel::Memory::ProtectGuestMemory(
	        sort_site, installed_patch.size(),
	        Common::VirtualMemory::Mode::ExecuteReadWrite, &old_sort_mode)) {
		Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}

	std::memcpy(reinterpret_cast<void*>(sort_site), installed_patch.data(), installed_patch.size());
	const bool restore_mode_ok = Libs::LibKernel::Memory::ProtectGuestMemory(
	    sort_site, installed_patch.size(), old_sort_mode, nullptr);
	const bool flush_ok =
	    Common::VirtualMemory::FlushInstructionCache(sort_site, installed_patch.size());

	if (!restore_mode_ok || !flush_ok) {
		Libs::LibKernel::Memory::ProtectGuestMemory(
		    sort_site, PrologueBytes.size(), Common::VirtualMemory::Mode::ExecuteReadWrite, nullptr);
		std::memcpy(reinterpret_cast<void*>(sort_site), PrologueBytes.data(), PrologueBytes.size());
		Libs::LibKernel::Memory::ProtectGuestMemory(
		    sort_site, PrologueBytes.size(), old_sort_mode, nullptr);
		Common::VirtualMemory::FlushInstructionCache(sort_site, PrologueBytes.size());
		Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}

	cave = allocated;
	site = sort_site;
	LOGF("Demon's Souls culling sort: installed native draw packet introsort detour\n");
#endif
}

void Clear() {
	if (site) {
		std::array<uint8_t, PrologueBytes.size()> bytes {};
		std::memcpy(bytes.data(), reinterpret_cast<const void*>(site), bytes.size());
		if (bytes == installed_patch) {
			Common::VirtualMemory::Mode old_mode {};
			if (Libs::LibKernel::Memory::ProtectGuestMemory(
			        site, PrologueBytes.size(),
			        Common::VirtualMemory::Mode::ExecuteReadWrite, &old_mode)) {
				std::memcpy(reinterpret_cast<void*>(site), PrologueBytes.data(),
				            PrologueBytes.size());
				Libs::LibKernel::Memory::ProtectGuestMemory(site, PrologueBytes.size(), old_mode,
				                                           nullptr);
				Common::VirtualMemory::FlushInstructionCache(site, PrologueBytes.size());
			}
		}
		site = 0;
	}
	if (cave) {
		Libs::LibKernel::Memory::FreeGuestMemory(cave, PageSize);
		cave = 0;
	}
	installed_patch.fill(0);
}

} // namespace Loader::DemonsSoulsCulling
