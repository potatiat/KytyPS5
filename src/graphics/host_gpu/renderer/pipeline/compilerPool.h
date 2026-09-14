#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMPILERPOOL_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMPILERPOOL_H_

#include "common/common.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <vector>

namespace Libs::Graphics {

class CompilerThreadPool {
public:
	explicit CompilerThreadPool(size_t thread_count = 0) {
		if (thread_count == 0) {
			const auto hw = std::thread::hardware_concurrency();
			thread_count = hw > 2 ? hw - 2 : (hw > 0 ? hw : 2);
		}
		m_stop.store(false, std::memory_order_relaxed);
		m_workers.reserve(thread_count);
		for (size_t i = 0; i < thread_count; ++i) {
			m_workers.emplace_back([this] {
				while (true) {
					std::function<void()> task;
					{
						std::unique_lock<std::mutex> lock(m_queue_mutex);
						m_cv.wait(lock, [this] {
							return m_stop.load(std::memory_order_relaxed) || !m_tasks.empty();
						});
						if (m_stop.load(std::memory_order_relaxed) && m_tasks.empty()) {
							return;
						}
						task = std::move(m_tasks.front());
						m_tasks.pop();
					}
					task();
					{
						std::lock_guard<std::mutex> lock(m_queue_mutex);
						if (m_active_tasks.fetch_sub(1, std::memory_order_acq_rel) == 1) {
							m_idle_cv.notify_all();
						}
					}
				}
			});
		}
	}

	~CompilerThreadPool() {
		{
			std::lock_guard<std::mutex> lock(m_queue_mutex);
			m_stop.store(true, std::memory_order_relaxed);
		}
		m_cv.notify_all();
		for (auto& worker: m_workers) {
			if (worker.joinable()) {
				worker.join();
			}
		}
	}

	KYTY_CLASS_NO_COPY(CompilerThreadPool);

	template <class F, class... Args>
	auto Enqueue(F&& f, Args&&... args)
	    -> std::future<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>> {
		using return_type = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>;

		auto task = std::make_shared<std::packaged_task<return_type()>>(
		    [func = std::forward<F>(f),
		     args_tuple = std::make_tuple(std::forward<Args>(args)...)]() mutable {
			    return std::apply(std::move(func), std::move(args_tuple));
		    });

		std::future<return_type> res = task->get_future();
		{
			std::lock_guard<std::mutex> lock(m_queue_mutex);
			if (m_stop.load(std::memory_order_relaxed)) {
				return res;
			}
			m_active_tasks.fetch_add(1, std::memory_order_relaxed);
			m_tasks.emplace([task]() { (*task)(); });
		}
		m_cv.notify_one();
		return res;
	}

	void WaitIdle() {
		std::unique_lock<std::mutex> lock(m_queue_mutex);
		m_idle_cv.wait(lock, [this] {
			return m_tasks.empty() && m_active_tasks.load(std::memory_order_acquire) == 0;
		});
	}

	[[nodiscard]] size_t ThreadCount() const noexcept { return m_workers.size(); }

private:
	std::vector<std::thread>          m_workers;
	std::queue<std::function<void()>> m_tasks;
	std::mutex                        m_queue_mutex;
	std::condition_variable           m_cv;
	std::condition_variable           m_idle_cv;
	std::atomic<uint32_t>             m_active_tasks {0};
	std::atomic<bool>                 m_stop {false};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMPILERPOOL_H_
