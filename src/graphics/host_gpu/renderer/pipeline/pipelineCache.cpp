#include "graphics/host_gpu/renderer/pipeline/shaderReadObserver.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/timer.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "kytyGitVersion.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fmt/format.h>
#include <limits>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <thread>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// CxPrimitiveSetup::PolygonMode disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

std::string DriverCacheDriverSuffix(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	return fmt::format(":{:08x}:{:08x}:{:08x}:{}\n",
	                   properties.vendorID, properties.deviceID, properties.driverVersion, uuid);
}

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr const char* kCacheFormatVersion = "v2";
	return fmt::format("KytyPC1:{}{}", kCacheFormatVersion, DriverCacheDriverSuffix(properties));
}

bool IsDriverCacheSignatureCompatible(std::string_view cached_sig,
                                      const vk::PhysicalDeviceProperties& properties) {
	if (!cached_sig.starts_with("KytyPC1:")) {
		return false;
	}
	const auto suffix = DriverCacheDriverSuffix(properties);
	return cached_sig.ends_with(suffix);
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	if (Log::GetDirection() != Log::Direction::Console) {
		std::fwrite(message.data(), 1, message.size(), stdout);
		std::fflush(stdout);
	}
	Log::Write(message);
	Log::Flush();
}

// One preparation can evaluate the same SRT words through descriptor, predicate, and
// flattened-value evaluators. Snapshot clean blocks once for this preparation only.
// Dirty or partially mapped blocks fall back to the exact requested word so a read
// never forces an unrelated GPU resource to synchronize.
struct ShaderGuestReadCache {
	static constexpr uint64_t BlockSize = 256;
	static constexpr size_t   MaxBlocks = 4096;
	struct Block {
		uint64_t                       address;
		std::array<uint8_t, BlockSize> bytes;
	};
	struct Slot {
		uint32_t generation = 0;
		uint32_t index      = 0;
	};
	std::array<Slot, MaxBlocks * 2>        slots {};
	std::vector<Block>                     blocks;
	std::unordered_map<uint64_t, uint32_t> words;
	uint32_t                               generation = 0;

	void Reset() {
		// Retain storage, but never retain guest values between preparations.
		blocks.clear();
		words.clear();
		if (++generation == 0) {
			slots.fill({});
			generation = 1;
		}
	}

	Slot& FindSlot(uint64_t address) {
		auto index =
		    static_cast<size_t>(XXH3_64bits(&address, sizeof(address))) & (slots.size() - 1);
		while (slots[index].generation == generation &&
		       blocks[slots[index].index].address != address) {
			index = (index + 1) & (slots.size() - 1);
		}
		return slots[index];
	}
};

bool ReadShaderGuestMemory(void* userdata, uint64_t address, uint32_t* value) {
	if (value == nullptr) return false;
	if (userdata == nullptr) {
		return Libs::LibKernel::Memory::TryReadGpuCleanBacking(address, value, sizeof(*value));
	}
	auto&      cache  = *static_cast<ShaderGuestReadCache*>(userdata);
	const auto base   = address & ~(ShaderGuestReadCache::BlockSize - 1u);
	const auto offset = address - base;
	if (offset <= ShaderGuestReadCache::BlockSize - sizeof(*value)) {
		const auto& slot = cache.FindSlot(base);
		if (slot.generation == cache.generation) {
			std::memcpy(value, cache.blocks[slot.index].bytes.data() + offset, sizeof(*value));
			return true;
		}
	}
	if (const auto found = cache.words.find(address); found != cache.words.end()) {
		*value = found->second;
		return true;
	}
	if (offset <= ShaderGuestReadCache::BlockSize - sizeof(*value) &&
	    cache.blocks.size() < ShaderGuestReadCache::MaxBlocks) {
		std::array<uint8_t, ShaderGuestReadCache::BlockSize> block;
		if (Libs::LibKernel::Memory::TryReadGpuCleanBacking(base, block.data(), block.size())) {
			std::memcpy(value, block.data() + offset, sizeof(*value));
			auto& slot = cache.FindSlot(base);
			slot       = {cache.generation, static_cast<uint32_t>(cache.blocks.size())};
			cache.blocks.push_back({base, std::move(block)});
			return true;
		}
	}
	if (!Libs::LibKernel::Memory::TryReadGpuCleanBacking(address, value, sizeof(*value))) {
		return false;
	}
	cache.words.emplace(address, *value);
	return true;
}

bool SyncShaderGuestMemory(void*, uint64_t address, uint64_t size) {
	return Libs::LibKernel::Memory::SyncGpuCleanBacking(address, size);
}

bool ReadShaderGuestMemoryBlock(void* userdata, uint64_t address, uint32_t* words, uint32_t word_count) {
	if (words == nullptr || word_count == 0) return false;
	const uint64_t byte_size = uint64_t {word_count} * sizeof(uint32_t);
	if (userdata == nullptr) {
		return Libs::LibKernel::Memory::TryReadGpuCleanBacking(address, words, byte_size);
	}
	auto&      cache  = *static_cast<ShaderGuestReadCache*>(userdata);
	const auto base   = address & ~(ShaderGuestReadCache::BlockSize - 1u);
	const auto offset = address - base;
	if (offset + byte_size <= ShaderGuestReadCache::BlockSize) {
		const auto& slot = cache.FindSlot(base);
		if (slot.generation == cache.generation) {
			std::memcpy(words, cache.blocks[slot.index].bytes.data() + offset, byte_size);
			return true;
		}
		if (cache.blocks.size() < ShaderGuestReadCache::MaxBlocks) {
			std::array<uint8_t, ShaderGuestReadCache::BlockSize> block;
			if (Libs::LibKernel::Memory::TryReadGpuCleanBacking(base, block.data(), block.size())) {
				std::memcpy(words, block.data() + offset, byte_size);
				auto& new_slot = cache.FindSlot(base);
				new_slot       = {cache.generation, static_cast<uint32_t>(cache.blocks.size())};
				cache.blocks.push_back({base, std::move(block)});
				return true;
			}
		}
	}
	return Libs::LibKernel::Memory::TryReadGpuCleanBacking(address, words, byte_size);
}

void ReportMaterialization(const char* label, ShaderType stage, uint64_t hash,
                           const ShaderRecompiler::IR::MaterializeReport& report, bool ok) {
	if (!ok) {
		EXIT("shader resource materialization failed: stage=%u hash=0x%016" PRIx64 " reason=%s\n",
		     static_cast<uint32_t>(stage), hash, report.reason.c_str());
	}
	if (!report.dropped_summary.empty()) {
		LOGF("%s indirect image tables: hash=0x%016" PRIx64 " dropped=%" PRIu32 " shapes=%" PRIu32
		     "%s\n",
		     label, hash, report.dropped_candidates, report.dropped_shapes,
		     report.dropped_summary.c_str());
	}
}

bool ReadShaderRawGuestMemory(void*, uint64_t address, uint32_t* value) {
	// A GPU-written neighbour may protect a clean descriptor on the same page.
	// Reading its checked backing alias avoids an unnecessary GPU drain. Dirty,
	// unmapped and untracked addresses retain the original load/fault behavior.
	if (!Libs::LibKernel::Memory::TryReadGpuCleanBackingOnWatchedPage(address, value,
	                                                                  sizeof(*value)))
		std::memcpy(value, reinterpret_cast<const void*>(address), sizeof(*value));
	return true;
}

bool ReadShaderMemorySpan(void*, uint64_t address, uint32_t* values, uint32_t count, bool clean) {
	return count >= 2 && count <= 16 &&
	       Libs::LibKernel::Memory::TryReadGpuShaderSpan(address, values, count * 4u, clean);
}

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / fmt::format("{:04d}_new_shader_{}_{:016x}.spv",
	                                                             id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(spirv.data(), spirv.size() * sizeof(uint32_t));
}

void DumpShaderOriginal(const char* stage_name, uint64_t shader_hash,
                        std::span<const uint32_t> code, const std::string& decoded_dump) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	EXIT_IF(code.empty());
	static std::atomic_int id = 0;
	const auto base = Config::GetShaderLogFolder() / "original" /
	                  fmt::format("{:04d}_new_shader_{}_{:016x}", id++, stage_name, shader_hash);
	Common::File::CreateDirectories(base.parent_path());
	for (const auto& [suffix, data, size]: {
	         std::tuple {".bin", static_cast<const void*>(code.data()), code.size_bytes()},
	         std::tuple {".rdna2", static_cast<const void*>(decoded_dump.data()),
	                     decoded_dump.size()},
	     }) {
		if (size == 0) {
			continue;
		}
		auto path = base;
		path += suffix;
		Common::File file(path);
		if (file.IsInvalid()) {
			const auto path_text = Common::PathToString(path);
			LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		} else {
			file.Write(data, size);
		}
	}
}

bool ValidateShaderSpirv(const char* label, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv) {
	if (!Config::ShaderValidationEnabled()) {
		return true;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	if (tools.Validate(spirv)) {
		return true;
	}
	// Fatal validation diagnostics must survive silent shader/printf settings.
	std::fprintf(stderr, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s",
	             label, shader_hash, messages.c_str());
	std::fflush(stderr);
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF_COLOR(Log::Color::BrightRed, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s",
	           label, shader_hash, messages.c_str());
	LOGF("%s\n", text.c_str());
	return false;
}

} // namespace

struct PipelineCache::ProgramCache {
	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		uint32_t              code_size       = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	struct Permutation {
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		ShaderProgram                                handle;
	};

	struct CachedSnapshot {
		std::vector<uint32_t>                        user_data;
		uint64_t                                     shader_base = 0;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		bool                                         valid = false;
	};

	struct SourceEntry {
		explicit SourceEntry(ShaderRecompiler::IR::ResourcePlan plan)
		    : resource_plan(std::move(plan)) {}

		ShaderRecompiler::IR::ResourcePlan resource_plan;
		// ShaderStageRuntime keeps a pointer into a compiled permutation. Later
		// specializations of the same source must not invalidate an earlier draw.
		std::deque<Permutation> permutations;
		CachedSnapshot                     last_snapshot;
	};

	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const {
			std::size_t hash = static_cast<std::size_t>(key.stage);
			PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash));
			if constexpr (sizeof(std::size_t) < sizeof(uint64_t)) {
				PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash >> 32u));
			}
			PipelineKeyHash::Mix(hash, key.user_data_count);
			PipelineKeyHash::Mix(hash, key.code_size);
			PipelineKeyHash::Mix(hash, key.static_state.size());
			// Bucket same-shape static variants by source. ProgramKey equality performs the one
			// exact state comparison needed on a stable hit without hashing up to 429 words first.
			return hash;
		}
	};

	static constexpr std::size_t MaxStaticKeyWords = 13 + ShaderVertexInputInfo::RES_MAX * 13;

	Permutation CompilePermutation(const ShaderParams&                          params,
	                               const ShaderRecompiler::CompileOptions&      options,
	                               ShaderRecompiler::TranslateResult            translated,
	                               ShaderRecompiler::IR::ResourceSpecialization specialization,
	                               uint32_t push_data_start_dword) {
		const char* stage_name = nullptr;
		switch (options.stage) {
			case ShaderType::Vertex: stage_name = "vs"; break;
			case ShaderType::Mesh: stage_name = "ms"; break;
			case ShaderType::Pixel: stage_name = "ps"; break;
			case ShaderType::Compute: stage_name = "cs"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               specialization, push_data_start_dword);
		DumpShaderOriginal(stage_name, options.shader_hash, params.code, result.decoded_dump);
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, result.spirv)) {
			DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
			EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
			     options.shader_hash);
		}
		DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);

		vk::ShaderModuleCreateInfo create_info {};
		create_info.codeSize    = result.spirv.size() * sizeof(uint32_t);
		create_info.pCode       = result.spirv.data();
		vk::ShaderModule module = nullptr;
		RequireVulkanSuccess(device.createShaderModule(&create_info, nullptr, &module),
		                     "create recompiled shader module");
		EXIT_IF(module == nullptr);
		SetVulkanObjectNameF(device, module, "Kyty.Shader.{}[0x{:016x}]", stage_name,
		                     options.shader_hash);
		if (options.dump_ir) {
			if (!options.early_dump) {
				LOGF("%s decoded RDNA2:\n%s", options.dump_label, result.decoded_dump.c_str());
				LOGF("%s IR:\n%s", options.dump_label, result.ir_dump.c_str());
			}
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(result.spirv.size()), options.wave_size);
		}
		uint64_t shader_id =
		    XXH3_64bits(result.spirv.data(), result.spirv.size() * sizeof(uint32_t));
		shader_id ^= (static_cast<uint64_t>(options.stage) << 60);
		if (shader_id == 0) {
			shader_id = 1;
		}
		if (options.stage == ShaderType::Compute) {
			compute_spirv[shader_id] = result.spirv;
		}
		return {
		    .specialization = std::move(specialization),
		    .program        = std::move(result.program).TakeCompiledInfo(),
		    .handle         = {.id = shader_id, .module = module},
		};
	}

	template <typename InputInfo>
	ShaderProgram Get(const ShaderParams& params, InputInfo& input_info,
	                  uint32_t& push_data_cursor) {
		ShaderType stage;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage = input_info.mesh.threads_num[0] != 0 ? ShaderType::Mesh : ShaderType::Vertex;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage = ShaderType::Pixel;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			stage = ShaderType::Compute;
		}
		const char* label = nullptr;
		switch (stage) {
			case ShaderType::Vertex: label = "ShaderRecompiler VS"; break;
			case ShaderType::Mesh: label = "ShaderRecompiler MS"; break;
			case ShaderType::Pixel: label = "ShaderRecompiler PS"; break;
			case ShaderType::Compute: label = "ShaderRecompiler CS"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}

		lookup_key.stage           = stage;
		lookup_key.hash            = params.hash;
		lookup_key.user_data_count = static_cast<uint32_t>(params.user_data.size());
		lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, lookup_key.static_state);
		auto                                         entry = programs.find(lookup_key);
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		read_cache.Reset();
		const ShaderRecompiler::IR::SrtRuntime input_runtime {
		    .user_data                  = params.user_data,
		    .shader_base                = params.Base(),
		    .read_memory                = ReadShaderRawGuestMemory,
		    .userdata                   = &read_cache,
		    .read_specialization_memory = ReadShaderGuestMemory,
		    .read_specialization_block  = ReadShaderGuestMemoryBlock,
		    .sync_memory                = SyncShaderGuestMemory,
		    .try_read_memory_span       = ReadShaderMemorySpan,
		};
		ShaderReadObserver::Runtime observed_runtime(input_runtime);
		const auto& runtime = observed_runtime.Get();
		ShaderRecompiler::IR::MaterializeReport report;
		if (entry != programs.end()) {
			const bool can_memoize =
			    !entry->second.resource_plan.requires_specialization_memory &&
			    entry->second.resource_plan.memory_info.empty();
			if (entry->second.last_snapshot.valid && can_memoize &&
			    entry->second.last_snapshot.shader_base == params.Base() &&
			    entry->second.last_snapshot.user_data.size() == params.user_data.size() &&
			    std::memcmp(entry->second.last_snapshot.user_data.data(), params.user_data.data(),
			                params.user_data.size() * sizeof(uint32_t)) == 0) {
				resources      = entry->second.last_snapshot.resources;
				specialization = entry->second.last_snapshot.specialization;
			} else {
				ReportMaterialization(label, stage, params.hash, report,
				                      ShaderRecompiler::IR::MaterializeResources(
				                          entry->second.resource_plan, runtime, resources,
				                          specialization, &report));
				if (can_memoize) {
					entry->second.last_snapshot.user_data.assign(params.user_data.begin(),
					                                             params.user_data.end());
					entry->second.last_snapshot.shader_base    = params.Base();
					entry->second.last_snapshot.resources      = resources;
					entry->second.last_snapshot.specialization = specialization;
					entry->second.last_snapshot.valid          = true;
				}
			}
			if (const auto permutation = std::ranges::find_if(
			        entry->second.permutations, [&](const Permutation& candidate) {
				        const auto& layout = candidate.program.bindings;
				        return layout.push_data_start_dword ==
				                   ShaderRecompiler::IR::PushData::StartFor(
				                       push_data_cursor, layout.ShaderDataDwords()) &&
				               candidate.specialization == specialization;
			        });
			    permutation != entry->second.permutations.end()) {
				input_info.stage = {.program   = &permutation->program,
				                    .resources = std::move(resources)};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				return permutation->handle;
			}
		}

		ShaderStageInputInfo stage_input {};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage_input.vertex = &input_info;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage_input.pixel = &input_info;
		} else {
			stage_input.compute = &input_info;
		}
		ShaderRecompiler::CompileOptions options;
		options.stage       = stage;
		options.enable_lod_stats = true;
		options.shader_hash = params.hash;
		options.user_data   = params.user_data;
		options.back_code      = params.back_code;
		options.dump_ir     = Config::GetShaderLogDirection() != Config::ShaderLogDirection::Silent;
		options.early_dump  = options.dump_ir;
		options.dump_label  = label;
		options.input_info  = stage_input;
		options.scratch_dwords = input_info.scratch_size_dwords;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.user_data_base = 8;
			if (stage == ShaderType::Mesh) {
				options.user_data_base = 0;
				options.wave_size      = input_info.mesh.wave_size;
				options.scratch_dwords = input_info.mesh.scratch_size_dwords;
			}
		} else if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			options.wave_size = input_info.wave_size;
		}
		auto translated = ShaderRecompiler::TranslateProgram(params.code, options);
		if (entry == programs.end()) {
			auto resource_plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
			ReportMaterialization(label, stage, params.hash, report,
			                      ShaderRecompiler::IR::MaterializeResources(
			                          resource_plan, runtime, resources, specialization, &report));
			const bool can_memoize =
			    !resource_plan.requires_specialization_memory && resource_plan.memory_info.empty();
			entry = programs.try_emplace(lookup_key, std::move(resource_plan)).first;
			if (can_memoize) {
				entry->second.last_snapshot.user_data.assign(params.user_data.begin(),
				                                             params.user_data.end());
				entry->second.last_snapshot.shader_base    = params.Base();
				entry->second.last_snapshot.resources      = resources;
				entry->second.last_snapshot.specialization = specialization;
				entry->second.last_snapshot.valid          = true;
			}
		}
		entry->second.permutations.push_back(CompilePermutation(
		    params, options, std::move(translated), std::move(specialization), push_data_cursor));
		const auto& permutation = entry->second.permutations.back();
		input_info.stage = {.program = &permutation.program, .resources = std::move(resources)};
		permutation.program.bindings.AdvancePushData(push_data_cursor);

		std::array<size_t, static_cast<size_t>(ShaderType::Mesh) + 1> counts {};
		for (const auto& [key, source]: programs) {
			counts[static_cast<size_t>(key.stage)] += source.permutations.size();
		}
		// Guest geometry shaders are compiled through the host mesh stage.
		std::printf("Shaders: VS %zu | PS %zu | CS %zu | GS %zu\n",
		            counts[static_cast<size_t>(ShaderType::Vertex)],
		            counts[static_cast<size_t>(ShaderType::Pixel)],
		            counts[static_cast<size_t>(ShaderType::Compute)],
		            counts[static_cast<size_t>(ShaderType::Mesh)]);
		return permutation.handle;
	}

	explicit ProgramCache(vk::Device device): device(device) {
		lookup_key.static_state.reserve(MaxStaticKeyWords);
	}
	~ProgramCache() {
		for (const auto& [key, entry]: programs) {
			(void)key;
			for (const auto& permutation: entry.permutations) {
				device.destroyShaderModule(permutation.handle.module, nullptr);
			}
		}
	}

	std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash> programs;
	std::unordered_map<uint64_t, std::vector<uint32_t>>         compute_spirv;
	ProgramKey                                                  lookup_key;
	ShaderGuestReadCache                                        read_cache;
	vk::Device                                                  device;
	uint64_t                                                    next_shader_id = 0;

	[[nodiscard]] const std::vector<uint32_t>* GetSpirv(uint64_t shader_id) const {
		auto it = compute_spirv.find(shader_id);
		return it != compute_spirv.end() ? &it->second : nullptr;
	}
};

static std::atomic<uint64_t> s_next_pipeline_cache_instance_id {1};

PipelineCache::PipelineCache(GraphicContext& graphics)
    : m_graphics(graphics),
      m_instance_id(s_next_pipeline_cache_instance_id.fetch_add(1, std::memory_order_relaxed)),
      m_program_cache(std::make_unique<ProgramCache>(graphics.device)) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	InitializeDriverCache();
}

PipelineCache::~PipelineCache() {
	m_instance_id = 0;
	m_compiler_pool.WaitIdle();
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout, nullptr);
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	if (m_driver_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	}
}

static std::atomic<bool>     s_driver_cache_saving {false};
static std::atomic<uint64_t> s_last_saved_hash {0};
static std::atomic<uint64_t> s_last_save_time {0};

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}
	const std::string_view git_hash     = KYTY_GIT_HASH;
	const std::string_view git_revision = KYTY_GIT_REVISION;
	if (git_hash == "unknown" || git_revision == "unknown") {
		PipelineCacheLog("Vulkan pipeline cache: git revision unknown, using fallback signature");
	}

	m_driver_cache_path     = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
	m_manifest_path         = std::filesystem::path("_PipelineCache") / (title_id + ".manifest");
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	uint64_t             loaded_payload_hash = 0;
	if (cache_exists) {
		Common::File file(m_driver_cache_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   props     = m_graphics.GetPhysicalDeviceProperties();
		if (file_size > sizeof(uint64_t) && file_size <= std::numeric_limits<uint32_t>::max()) {
			std::string cached_signature;
			char        ch         = 0;
			uint32_t    read_bytes = 0;
			while (cached_signature.size() < 256) {
				file.Read(&ch, 1, &read_bytes);
				if (read_bytes != 1) {
					break;
				}
				cached_signature.push_back(ch);
				if (ch == '\n') {
					break;
				}
			}
			const auto sig_len = cached_signature.size();
			if (ch == '\n' && file_size >= sig_len + sizeof(uint64_t)) {
				uint64_t payload_hash = 0;
				uint32_t hash_read    = 0;
				uint32_t payload_read = 0;
				file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
				const auto payload_size = file_size - sig_len - sizeof(payload_hash);
				initial_data.resize(payload_size);
				if (payload_size > 0) {
					file.Read(initial_data.data(), static_cast<uint32_t>(payload_size), &payload_read);
				}
				file.Close();
				if (hash_read != sizeof(payload_hash) || payload_read != payload_size ||
				    !IsDriverCacheSignatureCompatible(cached_signature, props) ||
				    XXH3_64bits(initial_data.data(), initial_data.size()) != payload_hash) {
					initial_data.clear();
					PipelineCacheLog(
					    "Vulkan pipeline cache: invalidating {} (driver, emulator, or data mismatch)",
					    path);
				} else {
					loaded_payload_hash = payload_hash;
				}
			} else {
				file.Close();
				PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid header)", path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	if (!initial_data.empty()) {
		s_last_saved_hash.store(loaded_payload_hash, std::memory_order_relaxed);
		PipelineCacheLog("Vulkan pipeline cache: loaded {} bytes from {}", initial_data.size(),
		                 path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}

	PreloadPipelines();
}

void PipelineCache::SerializeManifest(
    const std::filesystem::path& path,
    const std::string& signature,
    const std::vector<ManifestComputeRecord>& records) {
	if (records.empty() || path.empty()) {
		return;
	}

	std::vector<uint8_t> payload;
	for (const auto& record: records) {
		const uint32_t binding_count = static_cast<uint32_t>(record.bindings.size());
		const uint32_t spirv_count   = static_cast<uint32_t>(record.spirv.size());

		const size_t prev_size = payload.size();
		const size_t needed    = sizeof(record.shader_id) + sizeof(record.wave_size) +
		                      sizeof(binding_count) +
		                      binding_count * sizeof(ManifestBinding) +
		                      sizeof(spirv_count) + spirv_count * sizeof(uint32_t);
		payload.resize(prev_size + needed);
		uint8_t* ptr = payload.data() + prev_size;

		std::memcpy(ptr, &record.shader_id, sizeof(record.shader_id));
		ptr += sizeof(record.shader_id);
		std::memcpy(ptr, &record.wave_size, sizeof(record.wave_size));
		ptr += sizeof(record.wave_size);
		std::memcpy(ptr, &binding_count, sizeof(binding_count));
		ptr += sizeof(binding_count);
		if (binding_count > 0) {
			std::memcpy(ptr, record.bindings.data(),
			            binding_count * sizeof(ManifestBinding));
			ptr += binding_count * sizeof(ManifestBinding);
		}
		std::memcpy(ptr, &spirv_count, sizeof(spirv_count));
		ptr += sizeof(spirv_count);
		if (spirv_count > 0) {
			std::memcpy(ptr, record.spirv.data(), spirv_count * sizeof(uint32_t));
			ptr += spirv_count * sizeof(uint32_t);
		}
	}

	const uint64_t payload_hash = XXH3_64bits(payload.data(), payload.size());

	const uint32_t magic   = 0x4d46504b;
	const uint32_t version = 1;
	const uint32_t sig_len = static_cast<uint32_t>(signature.size());
	const uint32_t count   = static_cast<uint32_t>(records.size());

	std::vector<uint8_t> header;
	header.reserve(sizeof(magic) + sizeof(version) + sizeof(sig_len) + sig_len +
	               sizeof(count) + sizeof(payload_hash));
	auto append = [&header](const void* src, size_t sz) {
		const auto* b = static_cast<const uint8_t*>(src);
		header.insert(header.end(), b, b + sz);
	};
	append(&magic, sizeof(magic));
	append(&version, sizeof(version));
	append(&sig_len, sizeof(sig_len));
	append(signature.data(), sig_len);
	append(&count, sizeof(count));
	append(&payload_hash, sizeof(payload_hash));

	if (!Common::File::CreateDirectories(path.parent_path())) {
		return;
	}

	auto temp_path = path;
	temp_path += ".tmp";

	Common::File file;
	uint32_t written_h = 0;
	uint32_t written_p = 0;
	if (file.Create(temp_path)) {
		file.Write(header.data(), static_cast<uint32_t>(header.size()), &written_h);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &written_p);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	if (written_h == header.size() && written_p == payload.size() && flushed &&
	    Common::File::RenameFile(temp_path, path)) {
		PipelineCacheLog("Vulkan pipeline cache: saved {} compute manifests to {}", count,
		                 Common::PathToString(path));
	}
}

void PipelineCache::Save() {
	while (s_driver_cache_saving.load(std::memory_order_acquire)) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
	Common::LockGuard lock(m_mutex);
	SaveDriverCacheLocked(true);
	SaveManifestLocked();
}

void PipelineCache::FlushDriverCache() {
	if (m_driver_cache == nullptr) {
		return;
	}

	const auto now  = Common::Timer::QueryPerformanceCounter();
	const auto freq = Common::Timer::QueryPerformanceFrequency();
	if ((now - s_last_save_time.load(std::memory_order_relaxed)) < freq * 30u) {
		return;
	}

	bool expected = false;
	if (!s_driver_cache_saving.compare_exchange_strong(expected, true)) {
		return;
	}

	s_last_save_time.store(now, std::memory_order_relaxed);
	m_new_pipelines_since_save.store(0, std::memory_order_relaxed);

	std::vector<ManifestComputeRecord> manifest_records;
	std::vector<uint8_t>               payload;
	size_t                             payload_size = 0;
	{
		Common::LockGuard lock(m_mutex);
		manifest_records.reserve(m_manifest_records.size());
		for (const auto& [id, rec]: m_manifest_records) {
			manifest_records.push_back(rec);
		}

		// Vulkan 1.3 spec Section 9.6: "Host access to pipelineCache must be externally synchronized"
		// for vkGetPipelineCacheData. Query data safely while holding m_mutex so no concurrent
		// pipeline creation occurs on driver_cache.
		size_t     size   = 0;
		vk::Result result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result == vk::Result::eSuccess && size > 0 &&
		    size <= std::numeric_limits<uint32_t>::max()) {
			payload.resize(size);
			result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
			if (result == vk::Result::eSuccess) {
				payload.resize(size);
				payload_size = size;
			} else {
				payload.clear();
			}
		}
	}

	if (payload.empty()) {
		s_driver_cache_saving.store(false, std::memory_order_release);
		return;
	}

	std::thread([dest = m_driver_cache_path,
	             manifest_dest = m_manifest_path,
	             records = std::move(manifest_records),
	             props = m_graphics.GetPhysicalDeviceProperties(),
	             payload = std::move(payload),
	             size = payload_size]() {
		const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
		if (payload_hash != s_last_saved_hash.load(std::memory_order_relaxed)) {
			auto prefix = DriverCacheSignature(props);
			prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
			if (!Common::File::CreateDirectories(dest.parent_path())) {
				PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
				s_driver_cache_saving.store(false, std::memory_order_release);
				return;
			}
			auto temp_path = dest;
			temp_path += ".tmp";

			Common::File file;
			uint32_t     prefix_written  = 0;
			uint32_t     payload_written = 0;
			if (file.Create(temp_path)) {
				file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
				file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
			}
			const bool flushed = !file.IsInvalid() && file.Flush();
			file.Close();
			if (prefix_written == prefix.size() && payload_written == payload.size() && flushed &&
			    Common::File::RenameFile(temp_path, dest)) {
				s_last_saved_hash.store(payload_hash, std::memory_order_relaxed);
				PipelineCacheLog("Vulkan pipeline cache: asynchronously saved {} bytes to {}", size,
				                 Common::PathToString(dest));
			}
		}

		if (!records.empty()) {
			SerializeManifest(manifest_dest, DriverCacheSignature(props), records);
		}

		s_driver_cache_saving.store(false, std::memory_order_release);
	}).detach();
}

void PipelineCache::SaveDriverCacheLocked(bool destroy_cache) {
	if (m_driver_cache == nullptr) {
		return;
	}

	size_t               size = 0;
	vk::Result           result = vk::Result::eSuccess;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {} bytes)",
		                 vk::to_string(result), size);
		if (destroy_cache) {
			m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
			m_driver_cache = nullptr;
		}
		return;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		if (destroy_cache) {
			m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
			m_driver_cache = nullptr;
		}
		return;
	}
	auto temp_path = m_driver_cache_path;
	temp_path += ".tmp";

	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	if (prefix_written != prefix.size() || payload_written != payload.size() || !flushed ||
	    !Common::File::RenameFile(temp_path, m_driver_cache_path)) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_path));
	} else {
		s_last_saved_hash.store(payload_hash, std::memory_order_relaxed);
		PipelineCacheLog("Vulkan pipeline cache: saved {} bytes to {}", payload.size(),
		                 Common::PathToString(m_driver_cache_path));
	}
	m_new_pipelines_since_save.store(0, std::memory_order_relaxed);
	if (destroy_cache) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
		m_driver_cache = nullptr;
	}
}

void PipelineCache::SaveManifestLocked() {
	if (m_manifest_records.empty() || m_manifest_path.empty()) {
		return;
	}
	std::vector<ManifestComputeRecord> records;
	records.reserve(m_manifest_records.size());
	for (const auto& [id, rec]: m_manifest_records) {
		records.push_back(rec);
	}
	SerializeManifest(m_manifest_path,
	                  DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties()), records);
}

void PipelineCache::PreloadPipelines() {
#if defined(_WIN32)
	if (GetModuleHandleA("Nvda.Graphics.Interception.dll") != nullptr ||
	    GetModuleHandleA("WarpVizTarget.dll") != nullptr ||
	    std::getenv("KYTY_DISABLE_PRELOAD_PIPELINES") != nullptr) {
		PipelineCacheLog("Vulkan pipeline cache: profiler/debugger detected; skipping startup preload to avoid stalls");
		return;
	}
#endif

	if (m_manifest_path.empty() || !Common::File::IsFileExisting(m_manifest_path)) {
		return;
	}

	Common::File file(m_manifest_path, Common::File::Mode::Read);
	const auto   file_size = file.IsInvalid() ? 0 : file.Size();
	if (file_size < sizeof(uint32_t) * 4 + sizeof(uint64_t)) {
		file.Close();
		return;
	}

	uint32_t magic      = 0;
	uint32_t version    = 0;
	uint32_t sig_len    = 0;
	uint32_t bytes_read = 0;

	file.Read(&magic, sizeof(magic), &bytes_read);
	file.Read(&version, sizeof(version), &bytes_read);
	file.Read(&sig_len, sizeof(sig_len), &bytes_read);

	if (magic != 0x4d46504b || version != 1 || sig_len > 4096 ||
	    file_size < sizeof(magic) + sizeof(version) + sizeof(sig_len) + sig_len +
	                    sizeof(uint32_t) + sizeof(uint64_t)) {
		file.Close();
		PipelineCacheLog("Vulkan pipeline cache: manifest invalid header or version in {}",
		                 Common::PathToString(m_manifest_path));
		return;
	}

	std::string signature(sig_len, '\0');
	file.Read(signature.data(), sig_len, &bytes_read);
	if (!IsDriverCacheSignatureCompatible(signature, m_graphics.GetPhysicalDeviceProperties())) {
		file.Close();
		PipelineCacheLog("Vulkan pipeline cache: manifest signature mismatch; skipping preload");
		return;
	}

	uint32_t pipeline_count = 0;
	uint64_t payload_hash   = 0;
	file.Read(&pipeline_count, sizeof(pipeline_count), &bytes_read);
	file.Read(&payload_hash, sizeof(payload_hash), &bytes_read);

	const size_t header_size = sizeof(magic) + sizeof(version) + sizeof(sig_len) + sig_len +
	                           sizeof(pipeline_count) + sizeof(payload_hash);
	const size_t payload_size = file_size - header_size;
	if (payload_size == 0 && pipeline_count > 0) {
		file.Close();
		return;
	}

	std::vector<uint8_t> payload(payload_size);
	if (payload_size > 0) {
		file.Read(payload.data(), static_cast<uint32_t>(payload_size), &bytes_read);
	}
	file.Close();

	if (bytes_read != payload_size || XXH3_64bits(payload.data(), payload.size()) != payload_hash) {
		PipelineCacheLog("Vulkan pipeline cache: manifest checksum mismatch; skipping preload");
		return;
	}

	std::vector<ManifestComputeRecord> records;
	records.reserve(pipeline_count);

	const uint8_t* ptr = payload.data();
	const uint8_t* end = ptr + payload.size();

	for (uint32_t i = 0; i < pipeline_count; ++i) {
		if (ptr + sizeof(uint64_t) + sizeof(uint32_t) * 2 > end) {
			break;
		}
		ManifestComputeRecord rec;
		std::memcpy(&rec.shader_id, ptr, sizeof(rec.shader_id));
		ptr += sizeof(rec.shader_id);
		std::memcpy(&rec.wave_size, ptr, sizeof(rec.wave_size));
		ptr += sizeof(rec.wave_size);
		uint32_t binding_count = 0;
		std::memcpy(&binding_count, ptr, sizeof(binding_count));
		ptr += sizeof(binding_count);

		if (binding_count > 256 ||
		    ptr + binding_count * sizeof(ManifestBinding) + sizeof(uint32_t) > end) {
			break;
		}
		rec.bindings.resize(binding_count);
		if (binding_count > 0) {
			std::memcpy(rec.bindings.data(), ptr, binding_count * sizeof(ManifestBinding));
			ptr += binding_count * sizeof(ManifestBinding);
		}

		uint32_t spirv_count = 0;
		std::memcpy(&spirv_count, ptr, sizeof(spirv_count));
		ptr += sizeof(spirv_count);

		if (spirv_count > 1024 * 1024 || ptr + spirv_count * sizeof(uint32_t) > end) {
			break;
		}
		rec.spirv.resize(spirv_count);
		if (spirv_count > 0) {
			std::memcpy(rec.spirv.data(), ptr, spirv_count * sizeof(uint32_t));
			ptr += spirv_count * sizeof(uint32_t);
		}
		records.push_back(std::move(rec));
	}

	if (records.empty()) {
		return;
	}

	PipelineCacheLog("Vulkan pipeline cache: pre-warming {} compute pipelines from {}...",
	                 records.size(), Common::PathToString(m_manifest_path));
	const auto start_time = Common::Timer::QueryPerformanceCounter();

	for (auto& rec: records) {
		m_compiler_pool.Enqueue([this, record = std::move(rec)]() {
			vk::ShaderModuleCreateInfo create_info {};
			create_info.codeSize    = record.spirv.size() * sizeof(uint32_t);
			create_info.pCode       = record.spirv.data();
			vk::ShaderModule module = nullptr;
			auto             res =
			    m_graphics.device.createShaderModule(&create_info, nullptr, &module);
			if (res != vk::Result::eSuccess || module == nullptr) {
				return;
			}

			auto                                        pipeline = std::make_shared<Pipeline>();
			std::vector<vk::DescriptorSetLayoutBinding> bindings;
			bindings.reserve(record.bindings.size());
			for (const auto& b: record.bindings) {
				bindings.push_back({
				    .binding         = b.binding,
				    .descriptorType  = static_cast<vk::DescriptorType>(b.descriptor_type),
				    .descriptorCount = b.descriptor_count,
				    .stageFlags      = static_cast<vk::ShaderStageFlags>(b.stage_flags),
				    .pImmutableSamplers = nullptr,
				});
			}

			CreateComputePipelineDirect(m_graphics, *pipeline, module, record.wave_size, bindings,
			                            m_driver_cache);
			m_graphics.device.destroyShaderModule(module, nullptr);

			if (pipeline->pipeline != nullptr) {
				Common::LockGuard lock(m_mutex);
				m_compute_pipelines.emplace(record.shader_id, std::move(pipeline));
				m_manifest_records.emplace(record.shader_id, std::move(record));
			}
		});
	}

	m_compiler_pool.WaitIdle();
	{
		Common::LockGuard lock(m_mutex);
		for (const auto& [id, pipeline]: m_compute_pipelines) {
			const uint32_t set =
			    static_cast<uint32_t>(((id * 0x9E3779B97F4A7C15ull) >> 32) &
			                          (COMPUTE_FAST_CACHE_SETS - 1u)) *
			    COMPUTE_FAST_CACHE_WAYS;
			for (size_t i = 0; i < COMPUTE_FAST_CACHE_WAYS; ++i) {
				const size_t slot = set + i;
				if (m_compute_fast_cache[slot].pipeline.load(std::memory_order_relaxed) == nullptr ||
				    m_compute_fast_cache[slot].id.load(std::memory_order_relaxed) == id) {
					m_compute_fast_cache[slot].pipeline.store(pipeline.get(), std::memory_order_relaxed);
					m_compute_fast_cache[slot].id.store(id, std::memory_order_release);
					break;
				}
			}
		}
	}
	const auto elapsed = Common::Timer::QueryPerformanceCounter() - start_time;
	const double elapsed_sec =
	    static_cast<double>(elapsed) / static_cast<double>(Common::Timer::QueryPerformanceFrequency());
	PipelineCacheLog("Vulkan pipeline cache: pre-warmed {} compute pipelines in {:.3f}s",
	                 m_compute_pipelines.size(), elapsed_sec);
}

void PipelineCache::RecordManifestComputeLocked(
    uint64_t shader_id, uint32_t wave_size,
    const ShaderRecompiler::IR::CompiledShaderInfo& program,
    std::span<const uint32_t> spirv) {
	std::vector<vk::DescriptorSetLayoutBinding> vk_bindings;
	AddLayoutBindings(vk_bindings, program, vk::ShaderStageFlagBits::eCompute);

	ManifestComputeRecord record {};
	record.shader_id = shader_id;
	record.wave_size = wave_size;
	record.bindings.reserve(vk_bindings.size());
	for (const auto& b: vk_bindings) {
		record.bindings.push_back({
		    .binding          = b.binding,
		    .descriptor_type  = static_cast<uint32_t>(b.descriptorType),
		    .descriptor_count = b.descriptorCount,
		    .stage_flags      = static_cast<uint32_t>(b.stageFlags),
		});
	}
	record.spirv.assign(spirv.begin(), spirv.end());

	m_manifest_records.emplace(shader_id, std::move(record));
}

void PipelineCache::SubmitSpeculativeComputeLocked(const ShaderProgram& handle,
                                                  const ShaderComputeInputInfo& input_info) {
	if (!handle || !input_info.stage || !input_info.stage.program) {
		return;
	}
	const uint64_t shader_id = handle.id;

	if (!m_manifest_records.contains(shader_id)) {
		if (const auto* spirv = m_program_cache->GetSpirv(shader_id)) {
			RecordManifestComputeLocked(shader_id, input_info.stage.program->wave_size,
			                            *input_info.stage.program, *spirv);
		}
	}

	if (m_compute_pipelines.contains(shader_id) || m_in_flight_compute.contains(shader_id)) {
		return;
	}

	std::vector<vk::DescriptorSetLayoutBinding> descriptor_bindings;
	AddLayoutBindings(descriptor_bindings, *input_info.stage.program,
	                  vk::ShaderStageFlagBits::eCompute);

	const uint32_t   wave_size = input_info.stage.program->wave_size;
	vk::ShaderModule module    = handle.module;

	auto future = m_compiler_pool.Enqueue([this, module, wave_size,
	                                       bindings = std::move(descriptor_bindings)]() {
		auto pipeline = std::make_shared<Pipeline>();
		CreateComputePipelineDirect(m_graphics, *pipeline, module, wave_size, bindings,
		                            m_driver_cache);
		return pipeline;
	});

	m_in_flight_compute.emplace(shader_id, std::move(future).share());
}

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    ShaderVertexInputInfo& vertex_info, ShaderPixelInputInfo& pixel_info) {
	const auto vertex_params = PrepareProgram(vertex_regs, context, user_config, vertex_info);
	const bool mesh_active   = vertex_info.mesh.threads_num[0] != 0;
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!m_graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info.mesh;
		mesh.host_subgroup_size = m_graphics.subgroup_size;
		const auto& limits      = m_graphics.mesh_shader_properties;
		const auto  logical_threads =
		    mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
		const auto host_threads = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) *
		                          std::min(mesh.host_subgroup_size, mesh.wave_size);
		if (host_threads > limits.maxMeshWorkGroupInvocations ||
		    host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			EXIT("mesh shader exceeds host limits: threads=%u vertices=%u primitives=%u LDS=%u\n",
			     host_threads, mesh.max_vertices, mesh.max_primitives, mesh.lds_size_dwords);
		}
	}
	ShaderParams pixel_params;
	if (pixel_active) {
		pixel_params = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
	}
	if (context.GetClipControl().clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = vertex_info.clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	}
	Common::LockGuard lock(m_mutex);
	uint32_t          push_data_cursor =
	    mesh_active ? ShaderRecompiler::IR::PushData::MeshDrawDwordCount : 0;
	GraphicsPrograms  result;
	if (pixel_active) {
		pixel_info.lod_stats_subgroup = m_graphics.fragment_subgroup_reduction;
		result.pixel = m_program_cache->Get(pixel_params, pixel_info, push_data_cursor);
	}
	result.vertex = m_program_cache->Get(vertex_params, vertex_info, push_data_cursor);
	return result;
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               const HW::ShaderRegisters&   sh,
                                               ShaderComputeInputInfo&      input_info) {
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto        params      = PrepareProgram(regs, sh, input_info);
	Common::LockGuard lock(m_mutex);
	uint32_t          push_data_cursor = 0;
	const auto program = m_program_cache->Get(params, input_info, push_data_cursor);
	SubmitSpeculativeComputeLocked(program, input_info);
	return program;
}

bool PipelineStaticParameters::operator==(const PipelineStaticParameters& other) const noexcept {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

PipelineCache::Pipeline& PipelineCache::CreateGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    const ShaderVertexInputInfo& vs_input_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const ShaderProgram& vertex_program,
    const ShaderProgram& pixel_program) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	EXIT_IF(colors.size() > RENDER_COLOR_ATTACHMENTS_MAX);
	EXIT_IF(!vertex_program);
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(ps_active && !pixel_program);
	const auto color_count = static_cast<uint32_t>(colors.size());
	auto&      ctx         = command.GetRegisters();

	const HW::ModeControl& mc = ctx.GetModeControl();

	const auto vs_id = vertex_program.id;
	const auto ps_id = ps_active ? pixel_program.id : 0;

	GraphicsPipelineKey key {};
	key.vs_shader_id            = vs_id;
	key.ps_shader_id            = ps_id;
	auto& static_params         = key.static_params;
	auto& rendering             = key.rendering;
	rendering.color_count       = color_count;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[i] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot));
		rendering.color_formats[i] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	if (ps_active && depth.depth_test_enable && ps_input_info->ps_execute_on_noop) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("Pipeline: temporary: accepting EXEC_ON_NOOP with depth test enabled\n");
		}
	}

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	static_params.topology                 = topology;
	static_params.primitive_restart_enable = primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info->ps_sample_shading;
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
	}
	static_params.depth_bounds_test_enable = depth.depth_bounds_test_enable;
	static_params.depth_min_bounds         = depth.depth_min_bounds;
	static_params.depth_max_bounds         = depth.depth_max_bounds;
	static_params.stencil_test_enable      = depth.stencil_test_enable;
	static_params.stencil_front            = depth.stencil_static_front;
	static_params.stencil_back             = depth.stencil_static_back;
	const bool rect_list     = topology == vk::PrimitiveTopology::ePatchList;
	static_params.cull_back  = !rect_list && mc.cull_back;
	static_params.cull_front = !rect_list && mc.cull_front;
	static_params.face       = mc.face;
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, static_params.cull_front, static_params.cull_back);

	for (uint32_t i = 0; i < color_count; i++) {
		const auto& rt                        = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].target_slot);
		static_params.color_srcblend[i]       = bc.color_srcblend;
		static_params.color_comb_fcn[i]       = bc.color_comb_fcn;
		static_params.color_destblend[i]      = bc.color_destblend;
		static_params.alpha_srcblend[i]       = bc.alpha_srcblend;
		static_params.alpha_comb_fcn[i]       = bc.alpha_comb_fcn;
		static_params.alpha_destblend[i]      = bc.alpha_destblend;
		static_params.separate_alpha_blend[i] = bc.separate_alpha_blend;
		static_params.blend_enable[i]         = bc.enable;
		static_params.blend_bypass[i]         = rt.info.blend_bypass;
	}
	if (vs_input_info.stage.program->stage != ShaderType::Mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		uint32_t attributes_num          = 0;
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			EXIT_IF(buffer.attr_num < 0 || buffer.attr_num > ShaderVertexInputBuffer::ATTR_MAX);
			attributes_num += static_cast<uint32_t>(buffer.attr_num);
			EXIT_IF(attributes_num > static_cast<uint32_t>(vs_input_info.resources_num));
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
			for (int attribute = 0; attribute < buffer.attr_num; attribute++) {
				const auto index = buffer.attr_indices[attribute];
				EXIT_IF(index < 0 || index >= vs_input_info.resources_num);
				uint32_t   attr_size     = 4;
				const auto registers_num = vs_input_info.resources_dst[index].registers_num;
				const auto compiled_components =
				    vs_input_info.stage.program->info.vertex_fetch_components[index];
				const auto used_components =
				    compiled_components > 0 ? static_cast<int>(compiled_components) : registers_num;
				vk::Format format = vk::Format::eUndefined;
				GetInputFormat(vs_input_info.resources[index], format, attr_size,
				               static_cast<uint32_t>(used_components));
				key.vertex_input.attributes[index] = {
				    .offset  = buffer.attr_offsets[attribute],
				    .binding = static_cast<uint8_t>(binding),
				    .format  = format,
				};
			}
		}
		EXIT_IF(attributes_num != static_cast<uint32_t>(vs_input_info.resources_num));
	}

	// Thread-local front cache: 8 entries (~4 KB, fits L1D, zero lock contention and zero data race)
	thread_local ThreadLocalGraphicsCache tl_graphics_cache;

	for (size_t i = 0; i < ThreadLocalGraphicsCache::SIZE; i++) {
		auto& entry = tl_graphics_cache.entries[i];
		if (entry.owner == this && entry.instance_id == m_instance_id &&
		    entry.vs_id == vs_id && entry.ps_id == ps_id && entry.pipeline != nullptr) {
			if (entry.key == key) {
				return *entry.pipeline;
			}
		}
	}

	Common::LockGuard lock(m_mutex);

	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		auto* res = iter->second.get();
		auto& slot = tl_graphics_cache.entries[tl_graphics_cache.victim++ % ThreadLocalGraphicsCache::SIZE];
		slot = {
		    .owner       = this,
		    .instance_id = m_instance_id,
		    .vs_id       = vs_id,
		    .ps_id       = ps_id,
		    .key         = iter->first,
		    .pipeline    = res,
		};
		return *res;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(*ps_input_info);
		}
		LOGF("PipelineTrace: shader modules VS=%" PRIu64 " module=%p PS=%" PRIu64 " module=%p\n",
		     vs_id, static_cast<void*>(vertex_program.module), ps_id,
		     static_cast<void*>(pixel_program.module));
	}

	auto cached = std::make_unique<Pipeline>();
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	CreatePipelineInternal(m_graphics, *cached, rendering, key.vertex_input, vs_input_info,
	                       vertex_program, ps_input_info, pixel_program, static_params,
	                       m_driver_cache);
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_graphics_pipelines.emplace(key, std::move(cached));
	EXIT_IF(!inserted);

	auto* res = iter->second.get();
	auto& slot = tl_graphics_cache.entries[tl_graphics_cache.victim++ % ThreadLocalGraphicsCache::SIZE];
	slot = {
	    .owner       = this,
	    .instance_id = m_instance_id,
	    .vs_id       = vs_id,
	    .ps_id       = ps_id,
	    .key         = iter->first,
	    .pipeline    = res,
	};

	m_new_pipelines_since_save.fetch_add(1, std::memory_order_relaxed);

	return *res;
}

PipelineCache::Pipeline&
PipelineCache::CreateComputePipeline(const ShaderComputeInputInfo& input_info,
                                     const ShaderProgram&          compute_program) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	EXIT_IF(!compute_program);

	const uint64_t id = compute_program.id;
	const uint32_t set =
	    static_cast<uint32_t>(((id * 0x9E3779B97F4A7C15ull) >> 32) &
	                          (COMPUTE_FAST_CACHE_SETS - 1u)) *
	    COMPUTE_FAST_CACHE_WAYS;
	for (size_t i = 0; i < COMPUTE_FAST_CACHE_WAYS; ++i) {
		auto* p = m_compute_fast_cache[set + i].pipeline.load(std::memory_order_acquire);
		if (p != nullptr && m_compute_fast_cache[set + i].id.load(std::memory_order_acquire) == id) {
			return *p;
		}
	}

	Common::LockGuard lock(m_mutex);

	auto insert_fast_cache = [this, set, id](Pipeline* pipeline) {
		for (size_t i = 0; i < COMPUTE_FAST_CACHE_WAYS; ++i) {
			const uint64_t entry_id = m_compute_fast_cache[set + i].id.load(std::memory_order_relaxed);
			if (m_compute_fast_cache[set + i].pipeline.load(std::memory_order_relaxed) == nullptr ||
			    entry_id == id) {
				m_compute_fast_cache[set + i].id.store(0, std::memory_order_relaxed);
				m_compute_fast_cache[set + i].pipeline.store(pipeline, std::memory_order_relaxed);
				m_compute_fast_cache[set + i].id.store(id, std::memory_order_release);
				return;
			}
		}
		static std::atomic<uint32_t> victim_counter {0};
		const size_t target_slot =
		    set + (victim_counter.fetch_add(1, std::memory_order_relaxed) % COMPUTE_FAST_CACHE_WAYS);
		m_compute_fast_cache[target_slot].id.store(0, std::memory_order_relaxed);
		m_compute_fast_cache[target_slot].pipeline.store(pipeline, std::memory_order_relaxed);
		m_compute_fast_cache[target_slot].id.store(id, std::memory_order_release);
	};

	if (auto iter = m_compute_pipelines.find(id); iter != m_compute_pipelines.end()) {
		auto* result = iter->second.get();
		insert_fast_cache(result);
		return *result;
	}

	std::shared_ptr<Pipeline> compiled_pipeline;

	auto in_flight = m_in_flight_compute.find(id);
	if (in_flight != m_in_flight_compute.end()) {
		auto future = in_flight->second;
		m_mutex.Unlock();
		compiled_pipeline = future.get();
		m_mutex.Lock();

		auto existing = m_compute_pipelines.find(id);
		if (existing != m_compute_pipelines.end()) {
			m_in_flight_compute.erase(id);
			auto* result_pipeline = existing->second.get();
			insert_fast_cache(result_pipeline);
			return *result_pipeline;
		}
		m_in_flight_compute.erase(id);
	} else {
		if (graphics_debug_dump_enabled()) {
			ShaderDbgDumpInputInfo(input_info);
		}
		compiled_pipeline = std::make_shared<Pipeline>();
		CreatePipelineInternal(m_graphics, *compiled_pipeline, input_info,
		                       compute_program.module, m_driver_cache);
		if (!m_manifest_records.contains(id) && input_info.stage &&
		    input_info.stage.program) {
			if (const auto* spirv = m_program_cache->GetSpirv(id)) {
				RecordManifestComputeLocked(id,
				                            input_info.stage.program->wave_size,
				                            *input_info.stage.program, *spirv);
			}
		}
	}

	EXIT_NOT_IMPLEMENTED(compiled_pipeline->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(compiled_pipeline->pipeline_layout == nullptr);

	auto [iter, inserted] =
	    m_compute_pipelines.emplace(id, std::move(compiled_pipeline));
	auto* result_pipeline = iter->second.get();

	insert_fast_cache(result_pipeline);

	m_new_pipelines_since_save.fetch_add(1, std::memory_order_relaxed);

	return *result_pipeline;
}
} // namespace Libs::Graphics
