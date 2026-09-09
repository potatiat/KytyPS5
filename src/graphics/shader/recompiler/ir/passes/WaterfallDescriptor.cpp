#include "graphics/shader/recompiler/ir/passes/WaterfallDescriptor.h"

#include <algorithm>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

bool ImmediateU32(Value value, uint32_t& result) {
	value = value.Resolve();
	if (!value.IsImmediate() || value.GetType() != Type::U32) {
		return false;
	}
	result = value.U32();
	return true;
}

const Inst* Match(Value value, ValueOpcode opcode, size_t args) {
	const auto* inst = value.Resolve().TryInstruction();
	return inst != nullptr && inst->GetOpcode() == opcode && inst->NumArgs() == args ? inst
	                                                                                : nullptr;
}

bool CommutativeImmediate(const Inst& inst, uint32_t& immediate, Value& other) {
	if (ImmediateU32(inst.Arg(1), immediate)) {
		other = inst.Arg(0);
		return true;
	}
	if (ImmediateU32(inst.Arg(0), immediate)) {
		other = inst.Arg(1);
		return true;
	}
	return false;
}

bool AcceptLaneAnd(uint32_t bits) {
	return bits == 31u || bits == 63u;
}

const Inst* MatchLowestSetBit(Value value, const Inst& mask_phi, uint32_t* lane_and = nullptr) {
	const auto* shift = Match(value, ValueOpcode::ShiftLeftLogical32, 2);
	uint32_t    one   = 0;
	if (shift == nullptr || !ImmediateU32(shift->Arg(0), one) || one != 1u) {
		return nullptr;
	}
	const auto* masked = Match(shift->Arg(1), ValueOpcode::BitwiseAnd32, 2);
	uint32_t    bits   = 0;
	Value       index;
	if (masked == nullptr || !CommutativeImmediate(*masked, bits, index) || !AcceptLaneAnd(bits)) {
		return nullptr;
	}
	const auto* lsb = Match(index, ValueOpcode::FindILsb32, 1);
	if (lsb == nullptr || lsb->Arg(0).Resolve().TryInstruction() != &mask_phi) {
		return nullptr;
	}
	if (lane_and != nullptr) {
		*lane_and = bits;
	}
	return lsb;
}

const Inst* MatchClearedMask(Value latch, const Inst& mask_phi, uint32_t* lane_and = nullptr) {
	if (const auto* cleared = Match(latch, ValueOpcode::BitwiseXor32, 2)) {
		for (size_t side = 0; side < 2u; side++) {
			if (cleared->Arg(side).Resolve().TryInstruction() != &mask_phi) {
				continue;
			}
			if (const auto* lsb =
			        MatchLowestSetBit(cleared->Arg(1u - side), mask_phi, lane_and)) {
				return lsb;
			}
		}
	}
	const auto* cleared = Match(latch, ValueOpcode::BitwiseAnd32, 2);
	if (cleared == nullptr) {
		return nullptr;
	}
	for (size_t side = 0; side < 2u; side++) {
		const auto* operand = cleared->Arg(side).Resolve().TryInstruction();
		const auto* inverted = Match(cleared->Arg(1u - side), ValueOpcode::BitwiseNot32, 1);
		if (inverted == nullptr) {
			continue;
		}
		if (operand == &mask_phi) {
			if (const auto* lsb = MatchLowestSetBit(inverted->Arg(0), mask_phi, lane_and)) {
				return lsb;
			}
		} else if (operand != nullptr && operand->GetOpcode() == ValueOpcode::Phi) {
			if (const auto* lsb = MatchLowestSetBit(inverted->Arg(0), *operand, lane_and)) {
				return lsb;
			}
		}
	}
	return nullptr;
}

bool Reaches(Value value, const Inst& target, std::vector<const Inst*>& visited) {
	const auto* inst = value.Resolve().TryInstruction();
	if (inst == nullptr) {
		return false;
	}
	if (inst == &target) {
		return true;
	}
	if (std::ranges::find(visited, inst) != visited.end()) {
		return false;
	}
	visited.push_back(inst);
	for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
		if (Reaches(inst->Arg(arg), target, visited)) {
			return true;
		}
	}
	return false;
}

bool MatchKey(const Inst& index, Value& key, const Inst*& compare_out) {
	const Inst* compare = nullptr;
	size_t      operand = 0;
	for (const auto& use: index.Uses()) {
		if (use.user == nullptr || use.user->GetOpcode() != ValueOpcode::IEqual32 ||
		    use.user->NumArgs() != 2u) {
			continue;
		}
		if (compare != nullptr) {
			return false;
		}
		compare = use.user;
		operand = use.operand;
	}
	if (compare == nullptr) {
		return false;
	}
	key         = compare->Arg(operand == 0u ? 1u : 0u);
	compare_out = compare;
	return true;
}

const Inst* MatchTableOffset(const Inst& index, uint32_t& stride_shift, uint32_t& table_offset,
                             const Inst*& scaled_out) {
	const Inst* scaled = nullptr;
	for (const auto& use: index.Uses()) {
		if (use.user == nullptr || use.user->GetOpcode() != ValueOpcode::ShiftLeftLogical32 ||
		    use.user->NumArgs() != 2u || use.operand != 0u) {
			continue;
		}
		if (scaled != nullptr) {
			return nullptr;
		}
		scaled = use.user;
	}
	if (scaled == nullptr || !ImmediateU32(scaled->Arg(1), stride_shift) || stride_shift == 0u ||
	    stride_shift > 31u) {
		return nullptr;
	}
	const Inst* based          = nullptr;
	bool        used_as_offset = false;
	for (const auto& use: scaled->Uses()) {
		uint32_t immediate = 0;
		Value    other;
		if (use.user != nullptr && use.user->GetOpcode() == ValueOpcode::IAdd32 &&
		    use.user->NumArgs() == 2u && CommutativeImmediate(*use.user, immediate, other)) {
			if (based != nullptr) {
				return nullptr;
			}
			scaled_out   = scaled;
			based        = use.user;
			table_offset = immediate;
			continue;
		}
		if (use.user != nullptr && use.operand == 1u &&
		    (use.user->GetOpcode() == ValueOpcode::LoadAddressU32 ||
		     use.user->GetOpcode() == ValueOpcode::ReadConstBuffer)) {
			used_as_offset = true;
		}
	}
	if (based != nullptr) {
		return based;
	}
	if (used_as_offset) {
		scaled_out   = scaled;
		table_offset = 0;
		return scaled;
	}
	return nullptr;
}

bool LaneIndexSource(Value value, Value& source, uint32_t& shift) {
	while (const auto* select = Match(value, ValueOpcode::SelectU32, 3)) {
		value = select->Arg(1);
	}
	if (const auto* extract = Match(value, ValueOpcode::BitFieldUExtract, 3)) {
		uint32_t width = 0;
		if (!ImmediateU32(extract->Arg(1), shift) || !ImmediateU32(extract->Arg(2), width) ||
		    shift + width != 32u) {
			return false;
		}
		source = extract->Arg(0);
		return true;
	}
	if (const auto* shifted = Match(value, ValueOpcode::ShiftRightLogical32, 2)) {
		if (!ImmediateU32(shifted->Arg(1), shift)) {
			return false;
		}
		source = shifted->Arg(0);
		return true;
	}
	source = value;
	shift  = 0;
	return true;
}

bool SameLaneIndex(const Program& program, Value left, Value right) {
	while (const auto* select = Match(left, ValueOpcode::SelectU32, 3)) {
		left = select->Arg(1);
	}
	while (const auto* select = Match(right, ValueOpcode::SelectU32, 3)) {
		right = select->Arg(1);
	}
	if (EquivalentValue(program, left, right)) {
		return true;
	}
	Value    left_source;
	Value    right_source;
	uint32_t left_shift  = 0;
	uint32_t right_shift = 0;
	return LaneIndexSource(left, left_source, left_shift) &&
	       LaneIndexSource(right, right_source, right_shift) && left_shift == right_shift &&
	       EquivalentValue(program, left_source, right_source);
}

bool IsLaneReduction(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::SelectU32:
		case ValueOpcode::DppMoveU32:
		case ValueOpcode::DppUpdateU32:
		case ValueOpcode::Permlane16U32:
		case ValueOpcode::ReadLane: return true;
		default: return false;
	}
}

const Inst* FindBallotBit(const Program& program, Value entry, Value key,
                          std::vector<const Inst*>& visited) {
	const auto* inst = entry.Resolve().TryInstruction();
	if (inst == nullptr || std::ranges::find(visited, inst) != visited.end()) {
		return nullptr;
	}
	visited.push_back(inst);
	if (inst->GetOpcode() == ValueOpcode::ShiftLeftLogical32 && inst->NumArgs() == 2u) {
		uint32_t    one    = 0;
		uint32_t    bits   = 0;
		Value       index;
		const auto* masked = Match(inst->Arg(1), ValueOpcode::BitwiseAnd32, 2);
		if (ImmediateU32(inst->Arg(0), one) && one == 1u && masked != nullptr &&
		    CommutativeImmediate(*masked, bits, index) && AcceptLaneAnd(bits) &&
		    SameLaneIndex(program, index, key)) {
			return inst;
		}
	}
	for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
		if (const auto* found = FindBallotBit(program, inst->Arg(arg), key, visited)) {
			return found;
		}
	}
	return nullptr;
}

bool ReducesInto(const Inst& node, const Inst& root, std::vector<const Inst*>& visited) {
	if (&node == &root) {
		return true;
	}
	if (std::ranges::find(visited, &node) != visited.end()) {
		return false;
	}
	visited.push_back(&node);
	for (const auto& use: node.Uses()) {
		if (use.user != nullptr && IsLaneReduction(use.user->GetOpcode()) &&
		    ReducesInto(*use.user, root, visited)) {
			return true;
		}
	}
	return false;
}
const Inst* MatchImageHandle(const Program& program, const Inst& based, Value& heap) {
	const Inst* handle = nullptr;
	for (const auto& use: based.Uses()) {
		if (use.user == nullptr ||
		    (use.user->GetOpcode() != ValueOpcode::LoadAddressU32 &&
		     use.user->GetOpcode() != ValueOpcode::ReadConstBuffer)) {
			continue;
		}
		for (const auto& consumer: use.user->Uses()) {
			if (consumer.user == nullptr ||
			    consumer.user->GetOpcode() != ValueOpcode::GetImageResource) {
				continue;
			}
			if (handle != nullptr && handle != consumer.user) {
				return nullptr;
			}
			handle = consumer.user;
		}
	}
	if (handle == nullptr || handle->NumArgs() != 8u) {
		return nullptr;
	}
	for (size_t dword = 0; dword < handle->NumArgs(); dword++) {
		const auto* load = Match(handle->Arg(dword), ValueOpcode::LoadAddressU32, 4);
		if (load == nullptr) {
			load = Match(handle->Arg(dword), ValueOpcode::ReadConstBuffer, 2);
		}
		if (load == nullptr || load->NumArgs() < 2u) {
			return nullptr;
		}
		if (load->Arg(1).Resolve().TryInstruction() != &based) {
			uint32_t    immediate = 0;
			Value       other;
			const auto* shifted = Match(load->Arg(1), ValueOpcode::IAdd32, 2);
			if (shifted == nullptr || !CommutativeImmediate(*shifted, immediate, other) ||
			    other.Resolve().TryInstruction() != &based) {
				return nullptr;
			}
		}
		if (heap.IsEmpty()) {
			heap = load->Arg(0);
		} else if (!EquivalentValue(program, heap, load->Arg(0))) {
			return nullptr;
		}
	}
	return handle;
}

bool ImageUsesMask(const Inst& mask_phi) {
	for (const auto& use: mask_phi.Uses()) {
		if (use.user != nullptr && use.user->GetOpcode() == ValueOpcode::GetImageResource) {
			return true;
		}
	}
	return false;
}

bool MatchReadFirstLaneLoop(const Program& program, Value latch_val, const Inst& mask_phi,
                            const Inst*& rfl_out, const Inst*& compare_out, Value& key_out) {
	const auto* latch = Match(latch_val, ValueOpcode::LogicalAnd, 2);
	if (latch == nullptr) {
		return false;
	}
	for (size_t side = 0; side < 2u; side++) {
		if (latch->Arg(side).Resolve().TryInstruction() != &mask_phi) {
			continue;
		}
		const auto* inverted = Match(latch->Arg(1u - side), ValueOpcode::LogicalNot, 1);
		if (inverted == nullptr) {
			continue;
		}
		const auto* inner = inverted->Arg(0).Resolve().TryInstruction();
		while (inner != nullptr && inner->GetOpcode() == ValueOpcode::LogicalAnd &&
		       inner->NumArgs() == 2u) {
			if (inner->Arg(0).Resolve().TryInstruction() == &mask_phi) {
				inner = inner->Arg(1).Resolve().TryInstruction();
			} else if (inner->Arg(1).Resolve().TryInstruction() == &mask_phi) {
				inner = inner->Arg(0).Resolve().TryInstruction();
			} else {
				break;
			}
		}
		if (inner == nullptr || inner->GetOpcode() != ValueOpcode::IEqual32 ||
		    inner->NumArgs() != 2u) {
			continue;
		}
		for (size_t comp_side = 0; comp_side < 2u; comp_side++) {
			const auto* rfl = Match(inner->Arg(comp_side), ValueOpcode::ReadFirstLane, 2);
			if (rfl == nullptr) {
				continue;
			}
			const auto key = inner->Arg(1u - comp_side);
			if (!EquivalentValue(program, rfl->Arg(0), key)) {
				continue;
			}
			rfl_out     = rfl;
			compare_out = inner;
			key_out     = key;
			return true;
		}
	}
	return false;
}

}

std::vector<WaterfallDescriptor> FindWaterfallDescriptors(const Program& program) {
	std::vector<WaterfallDescriptor> result;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() != ValueOpcode::Phi || inst.NumArgs() != 2u) {
				continue;
			}
			for (size_t latch = 0; latch < 2u; latch++) {
				uint32_t    lane_and = 31u;
				const auto* lsb      = MatchClearedMask(inst.Arg(latch), inst, &lane_and);
				if (lsb != nullptr) {
					const auto               entry = inst.Arg(1u - latch);
					std::vector<const Inst*> visited;
					if (Reaches(entry, inst, visited)) {
						continue;
					}
					WaterfallDescriptor found;
					found.mask_phi = &inst;
					found.index    = lsb;
					found.entry    = entry;
					found.latch    = latch;
					found.lane_and = lane_and;
					const bool used_as_image = ImageUsesMask(inst);
					if (MatchKey(*lsb, found.key, found.compare)) {
						std::vector<const Inst*> seen;
						const auto* bit = FindBallotBit(program, found.entry, found.key, seen);
						std::vector<const Inst*> reduction;
						const auto* root = found.entry.Resolve().TryInstruction();
						if (bit != nullptr && root != nullptr &&
						    ReducesInto(*bit, *root, reduction)) {
							const auto* based = MatchTableOffset(
							    *lsb, found.stride_shift, found.table_offset, found.scaled);
							if (based != nullptr) {
								found.handle = MatchImageHandle(program, *based, found.heap);
							}
						}
					}
					if (found.handle == nullptr && !used_as_image) {
						break;
					}
					result.push_back(found);
					break;
				}

				const Inst* rfl     = nullptr;
				const Inst* compare = nullptr;
				Value       key;
				if (MatchReadFirstLaneLoop(program, inst.Arg(latch), inst, rfl, compare, key)) {
					const auto               entry = inst.Arg(1u - latch);
					std::vector<const Inst*> visited;
					if (Reaches(entry, inst, visited)) {
						continue;
					}
					WaterfallDescriptor found;
					found.mask_phi = &inst;
					found.index    = rfl;
					found.entry    = entry;
					found.latch    = latch;
					found.lane_and = program.wave_size > 32u ? 63u : 31u;
					found.key      = key;
					found.compare  = compare;
					const auto* based = MatchTableOffset(
					    *rfl, found.stride_shift, found.table_offset, found.scaled);
					if (based != nullptr) {
						found.handle = MatchImageHandle(program, *based, found.heap);
					}
					if (found.handle == nullptr && !ImageUsesMask(inst)) {
						break;
					}
					result.push_back(found);
					break;
				}
			}
		}
	}
	return result;
}

uint32_t RewriteWaterfallDescriptors(Program& program) {
	uint32_t rewritten = 0;
	for (const auto& found: FindWaterfallDescriptors(program)) {
		if (found.scaled != nullptr && !found.key.IsEmpty()) {
			const_cast<Inst*>(found.scaled)->SetArg(0, found.key);
		}
		if (found.compare != nullptr && !found.key.IsEmpty()) {
			auto*      compare = const_cast<Inst*>(found.compare);
			auto*      block   = compare->Parent();
			const auto at      = std::ranges::find_if(
                *block, [&](const Inst& inst) { return &inst == compare; });
			const auto bound = block->PrependNewInst(
			    at, ValueOpcode::ULessThan32, {found.key, Value(found.lane_and + 1u)});
			compare->ReplaceUsesWith(Value(&*bound));
		}
		auto*      mask  = const_cast<Inst*>(found.mask_phi);
		const bool is_u1 = mask->GetType() == Type::U1;
		mask->SetArg(found.latch, is_u1 ? Value(false) : Value(0u));
		mask->SetArg(found.latch == 0u ? 1u : 0u, is_u1 ? Value(true) : Value(1u));
		rewritten++;
	}
	return rewritten;
}

}

