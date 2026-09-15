# KytyPS5 Performance Optimization & Native Hooking Architecture Plan

**Document Version:** 2.1.0 (Production-Ready Architecture)  
**Target Platform:** PC (Windows x64 / Linux x64) / Vulkan 1.3  
**Target Titles:** *Demon's Souls* (`PPSA01341`, `01.007.000`) & Universal PS5 Titles  
**Related Components:** Kernel FileSystem, RuntimeLinker, Host GPU Renderer, Shader Recompiler, Buffer/Texture Caches  

---

## 1. Executive Summary & Architectural Scope

Superficial game patches (such as uncapping framerate to 120 FPS or disabling graphic toggles like motion blur and shadows) adjust game variables in memory but fail to resolve the core architectural bottlenecks of 9th-generation console emulation. 

Modern console engines—specifically the **Bluepoint Engine** in *Demon's Souls* (`PPSA01341`)—are built on assumptions that fundamentally conflict with host PC operating systems and discrete GPU architectures:

1. **Dedicated Core Allocation**: The engine assumes exclusive, uninterrupted ownership of 7 Zen 2 hardware cores. When worker threads lack jobs, they spin in tight `PAUSE`/`REP NOP` loops, consuming 100% of host CPU cores and starving the host OS scheduler, Vulkan worker threads, and presentation queues.
2. **Unified Memory Architecture (UMA)**: The CPU writes dynamic per-frame data (bone matrices, particle instances, indirect draw arguments) directly into GDDR6 memory that the GPU immediately reads without explicit synchronization APIs.
3. **Synchronous File I/O Contention**: High-bandwidth asset streaming across multiple background threads gets serialized on emulated file descriptor mutexes and synchronous seeks.
4. **Fine-Grained Compute Utility Dispatches**: The engine frequently dispatches tiny compute shaders (clearing buffers, prefix sums, linear copies) that on PS5 execute with zero driver overhead, but in Vulkan require pipeline lookups, descriptor bindings, and pipeline barriers that stall the GPU pipeline.

This document presents the hardened, production-ready implementation plan for native hooking and architecture modernizations in KytyPS5, with mathematically verified alignment safety, lifecycle reference counting, and complete Vulkan API compliance.

---

## 2. Component 1: File I/O & Streaming Subsystem (Loading Speed & Zero Stutter)

### 2.1 Problem Analysis
During loading screens (e.g. Nexus to Boletarian Palace) and continuous world streaming, *Demon's Souls* spawns multiple background I/O threads issuing concurrent positional reads via POSIX `pread` (`KernelPread`) and PS5 asynchronous I/O (`sceKernelAioSubmitReadCommands`).

In [`src/kernel/fileSystem.cpp`](file:///C:/Users/piyon/Desktop/claudd/emu/KytyPS5/src/kernel/fileSystem.cpp#L734-L749):
```cpp
file->mutex.Lock();

bool       is_invalid = file->f.IsInvalid();
auto       pos        = file->f.Tell();
const auto file_size  = file->f.Size();
const auto remaining = static_cast<uint64_t>(offset) < file_size ? file_size - static_cast<uint64_t>(offset) : 0;

Memory::InvalidateMemory(reinterpret_cast<uint64_t>(buf), std::min<uint64_t>(nbytes, remaining));

uint32_t bytes_read = 0;
file->f.Seek(offset);
file->f.Read(buf, static_cast<uint32_t>(nbytes), &bytes_read);
file->f.Seek(pos);

file->mutex.Unlock();
```

#### Identified Bottlenecks:
1. **Exclusive Lock Contention**: All concurrent streaming threads lock `file->mutex`. Multiple background threads cannot read simultaneously from the same `.pak` file; they serialize into a single queue.
2. **Triple-Syscall Seek Overhead**: Each read issues `file->f.Tell()`, `file->f.Seek(offset)`, and `file->f.Seek(pos)`. On Windows, this translates to three kernel transitions (`SetFilePointerEx` + `ReadFile` + `SetFilePointerEx`) per 32 KB / 64 KB read chunk.
3. **Descriptor Lifecycle Race**: If a thread calls `GetFile(d)` and drops descriptor locks, a concurrent `KernelClose(d)` deleting the object creates a Use-After-Free window.
4. **Synchronous AIO Loop**: [`KernelAioSubmitReadCommands`](file:///C:/Users/piyon/Desktop/claudd/emu/KytyPS5/src/libs/libKernel.cpp#L2934-L2948) processes requests sequentially on the calling thread, collapsing NVMe hardware queue depths (QD=1 instead of QD=32/64).

### 2.2 Technical Implementation

#### Step 1: Reference-Counted Descriptor Lifecycle & `std::shared_mutex`
To guarantee 100% thread safety against Use-After-Free (UAF) during concurrent `KernelClose`:
1. In `FileDescriptors` ([`src/kernel/fileSystem.cpp`](file:///C:/Users/piyon/Desktop/claudd/emu/KytyPS5/src/kernel/fileSystem.cpp#L76-L86)):
   Store `std::vector<std::shared_ptr<File>> m_files;` instead of raw pointers.
   - `GetFile(int d)` returns `std::shared_ptr<File>`.
   - `DeleteDescriptor(int d)` sets `m_files[index] = nullptr;`. If a reader thread holds a `std::shared_ptr<File>`, the object remains valid until the read finishes.
2. In `struct File` ([`src/kernel/fileSystem.cpp`](file:///C:/Users/piyon/Desktop/claudd/emu/KytyPS5/src/kernel/fileSystem.cpp#L61-L74)):
   Replace `Common::Mutex mutex;` with:
   ```cpp
   std::shared_mutex rw_mutex;
   ```
   - In `KernelPread`: Acquire shared read lock: `std::shared_lock<std::shared_mutex> lock(file->rw_mutex);`.
   - Multiple streaming threads reading from the same file run **concurrently in parallel**.
   - In mutating operations (`KernelWrite`, `KernelPwrite`, `KernelFtruncate`): Acquire exclusive lock: `std::unique_lock<std::shared_mutex> lock(file->rw_mutex);`.

#### Step 2: Platform-Specific Positional Read Implementation
- **Windows ([`src/common/platform/sysWindowsFileIO.cpp`](file:///C:/Users/piyon/Desktop/claudd/emu/KytyPS5/src/common/platform/sysWindowsFileIO.cpp))**:
  1. Add `bool is_overlapped = false;` and atomic position tracking `std::atomic<uint64_t> pos = 0;` to `sys_file_t`.
  2. For files opened for read-only streaming access (`Mode::Read`), specify `FILE_FLAG_OVERLAPPED` in `CreateFileW` and set `f.is_overlapped = true`.
  3. In `SysFileRead` (sequential read compatibility):
     If `f.is_overlapped` is true, pass a stack `OVERLAPPED` initialized with `f.pos` and advance `f.pos += bytes_read`. This prevents Win32 `ERROR_INVALID_PARAMETER` when calling `ReadFile` on overlapped handles.
  4. Implement `SysFileReadAt` (positional read):
     ```cpp
     void SysFileReadAt(void* data, uint32_t size, uint64_t offset, sys_file_t& f, uint32_t* bytes_read) {
         if (f.type == SYS_FILE_FILE) {
             OVERLAPPED ov {};
             ov.Offset     = static_cast<DWORD>(offset & 0xffffffffu);
             ov.OffsetHigh = static_cast<DWORD>((offset >> 32) & 0xffffffffu);
             DWORD w = 0;
             if (ReadFile(f.handle, data, size, &w, &ov)) {
                 if (bytes_read != nullptr) *bytes_read = w;
             } else {
                 DWORD err = GetLastError();
                 if (err == ERROR_IO_PENDING) {
                     if (GetOverlappedResult(f.handle, &ov, &w, TRUE)) {
                         if (bytes_read != nullptr) *bytes_read = w;
                     } else {
                         if (bytes_read != nullptr) *bytes_read = 0;
                     }
                 } else {
                     if (bytes_read != nullptr) *bytes_read = 0;
                 }
             }
         } else if (f.type == SYS_FILE_MEMORY_STAT || f.type == SYS_FILE_MEMORY_DYN) {
             uint32_t s = 0;
             if (offset < f.buf->size) {
                 s = std::min<uint32_t>(size, static_cast<uint32_t>(f.buf->size - offset));
                 std::memcpy(data, f.buf->base + offset, s);
             }
             if (bytes_read != nullptr) *bytes_read = s;
         }
     }
     ```
- **Linux ([`src/common/platform/sysLinuxFileIO.cpp`](file:///C:/Users/piyon/Desktop/claudd/emu/KytyPS5/src/common/platform/sysLinuxFileIO.cpp))**:
  ```cpp
  void SysFileReadAt(void* data, uint32_t size, uint64_t offset, sys_file_t& f, uint32_t* bytes_read) {
      if (f.type == SYS_FILE_FILE) {
          ssize_t ret = pread64(f.fd, data, size, static_cast<off64_t>(offset));
          if (bytes_read != nullptr) *bytes_read = (ret > 0) ? static_cast<uint32_t>(ret) : 0;
      }
  }
  ```

#### Step 3: Common `File::ReadAt` & Optimized `KernelPread`
In [`src/kernel/fileSystem.cpp`](file:///C:/Users/piyon/Desktop/claudd/emu/KytyPS5/src/kernel/fileSystem.cpp#L698-L759):
```cpp
int64_t KYTY_SYSV_ABI KernelPread(int d, void* buf, size_t nbytes, int64_t offset) {
    auto file = g_files->GetFile(d); // Returns std::shared_ptr<File> (safe against UAF)
    if (file == nullptr) return KERNEL_ERROR_EBADF;

    uint32_t bytes_read = 0;
    bool is_invalid = false;
    {
        std::shared_lock<std::shared_mutex> lock(file->rw_mutex);
        is_invalid = file->f.IsInvalid();
        if (!is_invalid) {
            file->f.ReadAt(buf, static_cast<uint32_t>(nbytes), static_cast<uint64_t>(offset), &bytes_read);
        }
    }

    if (is_invalid) return KERNEL_ERROR_EIO;

    // Invalidate GPU memory caches AFTER releasing the file lock for the exact bytes read
    if (bytes_read > 0) {
        Memory::InvalidateMemory(reinterpret_cast<uint64_t>(buf), bytes_read);
    }
    return bytes_read;
}
```

---

## 3. Component 2: CPU Job/Fiber Scheduler Hooking (FPS & Frame Pacing)

### 3.1 Problem Analysis
The Bluepoint Engine employs a work-stealing job system across 7 available Zen 2 cores. While Kyty currently hooks the main thread poll site at `0x83b59b` ([`demonsSoulsIdle.cpp`](file:///C:/Users/piyon/Desktop/claudd/emu/KytyPS5/src/loader/demonsSoulsIdle.cpp)), the **6 worker threads** run an independent worker loop function (`WorkerThreadProc`).

When queues are empty between frames, the 6 worker threads spin in tight `PAUSE`/`REP NOP` loops on atomic queue counters. On Windows, this consumes 100% of 6 host CPU cores and forces the Windows OS scheduler to preempt the Vulkan command submission thread and presentation thread.

### 3.2 Technical Implementation

#### The Failure of Kernel Sleep:
Invoking kernel waits such as `SleepMicroWithoutSpinning(25)` or `WaitForSingleObject` causes a context switch and scheduler quantum delay of **0.5ms to 1.0ms**. In fine-grained job systems where tasks execute in 10µs–50µs, sleeping for 500µs stalls dependent task chains (Physics $\rightarrow$ Animation $\rightarrow$ Render Prep), dropping frames.

#### The Correct Approach: User-Space Adaptive Backoff
Guest worker threads must back off **purely in user mode** without entering a kernel wait state:

```
┌─────────────────────────────────────────────────────────────┐
│                 Guest Worker Polling Loop                   │
└───────────────────────────────┬─────────────────────────────┘
                                │
                    [Poll Worker Queue: empty?]
                                │
                    ┌───────────┴───────────┐
                   Yes                      No
                    │                       │
                    ▼                       ▼
      ┌───────────────────────────┐   [Execute Task]
      │   User-Mode Adaptive      │
      │   Backoff Detour          │
      │                           │
      │  Phase 1 (Immediate):     │
      │  - 64x _mm_pause()        │
      │                           │
      │  Phase 2 (Cooperative):   │
      │  - std::this_thread::     │
      │    yield()                │
      │  (Yields quantum to host  │
      │   Vulkan/render thread)   │
      └───────────────────────────┘
```

1. **Phase 1 (Microsecond Low Latency)**: 64 to 128 iterations of `_mm_pause()`. Handles sub-microsecond job handoffs between threads without thread descheduling.
2. **Phase 2 (Cooperative Host Yield)**: Call `std::this_thread::yield()` (`SwitchToThread()` on Windows). If other host threads (such as Kyty's Vulkan command submission thread or audio processing thread) are ready to execute, the host OS scheduler immediately gives them CPU time. If no other thread is waiting, the thread resumes immediately without the 1ms kernel wait penalty.

#### Implementation Steps:
1. Reverse-engineer the worker loop disassembly in `eboot.bin` to identify the empty queue branch instruction pattern.
2. Verify module hash and byte signature before patching.
3. Emit an Xbyak assembly thunk calling the user-space adaptive yield handler, preserving all caller-saved registers and FP/XMM state.

---

## 4. Component 3: Coherent Memory Transfers & UMA Emulation

### 4.1 Problem Analysis
In [`src/loader/demonsSoulsCopy.cpp`](file:///C:/Users/piyon/Desktop/claudd/emu/KytyPS5/src/loader/demonsSoulsCopy.cpp) and [`demonsSoulsCopy.h`](file:///C:/Users/piyon/Desktop/claudd/emu/KytyPS5/src/loader/demonsSoulsCopy.h):
```cpp
inline void* Move(void* destination, const void* source, size_t size, PrepareWrite prepare) {
    if (size == 0 || destination == source) return destination;
    if (size >= 256 * 1024 && prepare) (void)prepare(reinterpret_cast<uint64_t>(destination), size);
    return std::memmove(destination, source, size);
}
```

The `256 KB` threshold ignores per-frame dynamic updates (skeletal bone matrices, instance bounding spheres, camera uniforms) which are typically **4 KB to 64 KB**. These small writes bypass `TryPrepareHostWrite`, triggering slow page-fault handlers and cache re-uploads during draw call preparation.

However, lowering the threshold to 4 KB unconditionally for all copies floods `TryPrepareHostWrite` across thousands of non-GPU heap allocations (audio, UI, string copies), causing severe lock contention on `TextureCache::m_lock` and synchronous stalls via `BufferCache::ReadMemory` (`SendCommandSync`).

### 4.2 Technical Implementation

#### Step 1: Selective GPU-Aperture Filtering
Only invoke `prepare` for writes that target addresses inside the GPU direct memory range:
```cpp
inline void* Move(void* destination, const void* source, size_t size, PrepareWrite prepare) {
    if (size == 0 || destination == source) return destination;
    const auto dest_addr = reinterpret_cast<uint64_t>(destination);
    
    // Coherent notification for dynamic GPU staging buffers (>= 64 KB, or >= 4 KB if within GPU aperture)
    if ((size >= 65536 || (size >= 4096 && Libs::LibKernel::Memory::IsGpuAddressRange(dest_addr, size))) && prepare) {
        (void)prepare(dest_addr, size);
    }
    return std::memmove(destination, source, size);
}
```

#### Step 2: Alignment-Safe AVX2 Streaming Copy (`CopyGpuStreaming`)
Non-temporal stores (`_mm256_stream_si256` / `VMOVNTDQ`) **strictly require 32-byte alignment** on the destination address. Calling this on unaligned memory causes an immediate General Protection Exception (`#GP` / `0xC0000005`). In addition, `memmove` permits overlapping buffers.

Implement **overlap detection and alignment peeling**:
```cpp
inline void CopyGpuStreaming(void* dst, const void* src, size_t size) {
    auto* d = static_cast<uint8_t*>(dst);
    const auto* s = static_cast<const uint8_t*>(src);

    // If buffers overlap in a way that forward copy corrupts source bytes, fall back to std::memmove
    if (d > s && d < s + size) {
        std::memmove(d, s, size);
        return;
    }

    size_t offset = 0;

    // Peel unaligned leading bytes until destination is 32-byte aligned
    const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(d);
    const size_t prefix = (32 - (dst_addr & 31)) & 31;
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
```

---

## 5. Component 4: GPU Compute Shader Interception & Barrier Deferral

### 5.1 Technical Implementation

#### Step 1: Buffer Clear Interception via Existing `Buffer::Fill`
*Demon's Souls* repeatedly dispatches 1-workgroup compute shaders to zero indirect draw argument buffers and particle counters.

Instead of issuing raw `vk_buffer.fillBuffer` calls (which risk alignment violations and lack required pipeline barriers), leverage Kyty's existing [`Buffer::Fill`](file:///C:/Users/piyon/Desktop/claudd/emu/KytyPS5/src/graphics/host_gpu/renderer/cache/streamBuffer.cpp#L208-L227):
```cpp
bool DemonsSouls::TryBufferClear(const ShaderComputeInputInfo& input, CommandBuffer& command, ...) {
    // 1. Validate shader hash and descriptor parameters
    ...
    // 2. Validate 4-byte alignment required by Vulkan
    if ((offset & 3) != 0 || (size & 3) != 0 || size == 0) return false;

    // 3. Delegate to Kyty's robust Buffer::Fill
    // This automatically calls command.EndRendering(), inserts transfer barriers,
    // issues vkCmdFillBuffer, and inserts the post-transfer compute/draw barriers.
    target_buffer.Fill(offset, size, 0);
    return true;
}
```

#### Step 2: PM4 Compute Boundary Grouping
In [`src/graphics/host_gpu/renderer/context.cpp`](file:///C:/Users/piyon/Desktop/claudd/emu/KytyPS5/src/graphics/host_gpu/renderer/context.cpp#L36-L51), expand [`CommandBuffer::ChainHandle()`](file:///C:/Users/piyon/Desktop/claudd/emu/KytyPS5/src/graphics/host_gpu/renderer/context.cpp#L36) to defer barrier flushes across verified independent compute passes, keeping GPU Compute Units fully saturated.

---

## 6. Component 5: Universal Vulkan Pipeline Modernization

### 6.1 Multi-Draw Indirect Count (`vkCmdDrawIndexedIndirectCount`)
- [`WindowContext::RequiredVulkan12Features()`](file:///C:/Users/piyon/Desktop/claudd/emu/KytyPS5/src/graphics/presentation/window/vulkanWindow.cpp#L65-L76) already enables `features.drawIndirectCount = VK_TRUE;`.
- In [`src/graphics/host_gpu/renderer/renderDraw.cpp`](file:///C:/Users/piyon/Desktop/claudd/emu/KytyPS5/src/graphics/host_gpu/renderer/renderDraw.cpp#L1248), wire `vkCmdDrawIndexedIndirectCount` when the indirect draw command includes a GPU-written count buffer, eliminating CPU stalls.

### 6.2 Async Compute Queue: Strict Queue Family Ownership Rules
In Vulkan, buffers created with `VK_SHARING_MODE_EXCLUSIVE` (Kyty's default) **cannot** be accessed across queue families without explicit ownership transfer barriers:
1. When creating shared staging buffers, configure `vk::SharingMode::eConcurrent` across both graphics and compute queue family indices:
   ```cpp
   std::array<uint32_t, 2> queue_families {graphics_family, compute_family};
   buffer_info.sharingMode = vk::SharingMode::eConcurrent;
   buffer_info.queueFamilyIndexCount = 2;
   buffer_info.pQueueFamilyIndices = queue_families.data();
   ```
2. Alternatively, emit explicit queue family ownership release and acquire barriers (`VkBufferMemoryBarrier` with `srcQueueFamilyIndex` and `dstQueueFamilyIndex`) when transferring control.

### 6.3 Concurrency Architecture in `TextureCache`
`FindTexture` and `FindRenderTarget` ([`textureCache.cpp`](file:///C:/Users/piyon/Desktop/claudd/emu/KytyPS5/src/graphics/host_gpu/renderer/cache/textureCache.cpp#L1540-L1635)) mutate internal LRU lists (`TouchImage`), binding state, and image views (`FindView`).

**Do not place a naive `std::shared_lock` over `FindTexture`.**  
Instead:
1. Split `FindTexture` into a lock-free or read-locked hash table lookup for existing `ImageId`s.
2. Only acquire the exclusive lock `m_lock` when inserting new views, mutating LRU nodes, or allocating physical image backings.

---

## 7. Phased Implementation Roadmap & Verification Plan

| Phase | Subsystem | Focus | Primary Deliverable |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Kernel FileSystem & Platform I/O** | Loading & Streaming | `std::shared_ptr<File>` descriptor lifecycle; `std::shared_mutex` in `struct File`; `SysFileReadAt` with `FILE_FLAG_OVERLAPPED` on Windows; out-of-lock `Memory::InvalidateMemory`. |
| **Phase 2** | **Guest CPU Job Scheduler** | FPS & Frame Pacing | User-space adaptive backoff (`_mm_pause` + `std::this_thread::yield()`); eliminates host CPU pegging. |
| **Phase 3** | **Coherent Memory Transfers** | UMA Staging | Alignment-safe `CopyGpuStreaming` (peeling + backward overlap check); GPU-aperture selective write preparation. |
| **Phase 4** | **GPU Compute Interception** | Pipeline Efficiency | Buffer-clear interception using `Buffer::Fill`; compute boundary chaining. |
| **Phase 5** | **Vulkan Modernization** | Throughput | `vkCmdDrawIndexedIndirectCount`; concurrent sharing mode for async compute queues. |

### Verification Gate:
- **Build**: LLVM/Clang-cl + Ninja clean compilation on Windows.
- **Tests**: `ShaderRecompilerComputeTests.exe` (70+ compute tests pass), `SaveDataCapacityTests.exe` (file system integrity passes).
- **Safety**: Zero General Protection Faults (`#GP`), zero Vulkan validation layer warnings, zero thread sanitization races.
