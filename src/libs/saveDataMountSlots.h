#ifndef EMULATOR_INCLUDE_EMULATOR_LIBS_SAVEDATAMOUNTSLOTS_H_
#define EMULATOR_INCLUDE_EMULATOR_LIBS_SAVEDATAMOUNTSLOTS_H_

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace Libs::SaveData {

class SaveDataMountSlots {
public:
	static constexpr int    BUSY  = -2;
	static constexpr int    FULL  = -1;
	static constexpr size_t COUNT = 16;

	struct Slot {
		std::optional<std::string> directory;
		std::string                host_path;
		uint64_t                   allocated_blocks = 0;
	};

	[[nodiscard]] int FindAvailable(std::string_view directory) const {
		int available = FULL;
		for (size_t index = 0; index < m_slots.size(); index++) {
			const auto& mounted = m_slots[index].directory;
			if (mounted == directory) {
				return BUSY;
			}
			if (!mounted.has_value() && available == FULL) {
				available = static_cast<int>(index);
			}
		}
		return available;
	}

	void Mount(size_t slot, std::string_view directory, std::string_view host_path,
	           uint64_t allocated_blocks) {
		m_slots[slot].directory         = std::string(directory);
		m_slots[slot].host_path         = std::string(host_path);
		m_slots[slot].allocated_blocks  = allocated_blocks;
	}

	void Release(size_t slot) {
		if (slot < m_slots.size()) {
			m_slots[slot] = {};
		}
	}

	[[nodiscard]] int Find(std::string_view mount_point) const {
		for (size_t index = 0; index < m_slots.size(); index++) {
			if (m_slots[index].directory.has_value() && MountPoint(index) == mount_point) {
				return static_cast<int>(index);
			}
		}
		return FULL;
	}

	[[nodiscard]] const Slot* Get(size_t slot) const {
		if (slot >= m_slots.size() || !m_slots[slot].directory.has_value()) {
			return nullptr;
		}
		return &m_slots[slot];
	}

	[[nodiscard]] static std::string MountPoint(size_t slot) {
		return "/savedata" + std::to_string(slot);
	}

	[[nodiscard]] bool Empty() const {
		for (const auto& slot: m_slots) {
			if (slot.directory.has_value()) {
				return false;
			}
		}
		return true;
	}

private:
	std::array<Slot, COUNT> m_slots;
};

} // namespace Libs::SaveData

#endif /* EMULATOR_INCLUDE_EMULATOR_LIBS_SAVEDATAMOUNTSLOTS_H_ */
