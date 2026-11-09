#include <array>
#include <iostream>

#include "cxx_api.h"
#include "threadpool.h"

using Pool = annc::threadpool::AnncThreadPool;
extern "C" void annc_set_current_threadpool(Pool*);
extern "C" Pool* annc_get_current_threadpool();
extern "C" int RunPoolKernel(int total, int cost, int grain, int* visits,
                             bool throw_task);

namespace {
struct HostPool {
  int64_t total = -1, block = -1, cost = -1;
  bool fail = false, returned = false;
};

int Schedule(void* ctx, int64_t total, int64_t block, int64_t cost,
             AnncThreadPoolTask task, void* data) noexcept {
  auto& host = *static_cast<HostPool*>(ctx);
  host.total = total;
  host.block = block;
  host.cost = cost;
  if (host.fail) return -1;
  task(data, 0, total / 2);
  task(data, total / 2, total);
  host.returned = true;
  return 0;
}

annc::tensorflow::ThreadPoolAdapterPtr Adapter(HostPool& host) {
  const AnncThreadPoolCallbacks callbacks{&host, [](void*) { return 2; },
                                          [](void*) { return 0; }, Schedule};
  return annc::tensorflow::ThreadPoolAdapterPtr(
      anncCreateThreadPoolAdapter(&callbacks));
}

#define CHECK(condition)                          \
  do {                                            \
    if (!(condition)) {                           \
      std::cerr << "Failed: " #condition << '\n'; \
      return false;                               \
    }                                             \
  } while (false)

bool CheckSchedules() {
  HostPool host;
  auto adapter = Adapter(host);
  CHECK(adapter);
  tensorflow::ScopedAnncThreadPool scope(reinterpret_cast<Pool*>(adapter.get()),
                                         annc_set_current_threadpool,
                                         annc_get_current_threadpool);
  std::array<int, 4> visits{};
  const std::array<int, 4> empty{};
  CHECK(RunPoolKernel(0, 7, 2, visits.data(), false) == 0);
  CHECK(visits == empty && host.total == -1);
  const std::array<int, 4> expected{{1, 1, 1, 1}};
  struct ScheduleCase {
    int cost, grain, expected_block, expected_cost;
  };
  const ScheduleCase cases[] = {
      {7, -1, 0, 7}, {7, 2, 2, 7}, {-1, -1, 0, 1}, {0, 0, 1, 1}};
  for (const auto& test : cases) {
    visits.fill(0);
    CHECK(RunPoolKernel(4, test.cost, test.grain, visits.data(), false) == 0);
    CHECK(host.total == 4 && host.block == test.expected_block &&
          host.cost == test.expected_cost);
    CHECK(visits == expected);
  }
  return true;
}

bool CheckRecovery() {
  HostPool outer, inner;
  auto first = Adapter(outer);
  auto second = Adapter(inner);
  CHECK(first && second);
  auto* previous = reinterpret_cast<Pool*>(first.get());
  tensorflow::ScopedAnncThreadPool outer_scope(
      previous, annc_set_current_threadpool, annc_get_current_threadpool);
  std::array<int, 4> visits{};
  {
    tensorflow::ScopedAnncThreadPool scope(
        reinterpret_cast<Pool*>(second.get()), annc_set_current_threadpool,
        annc_get_current_threadpool);
    inner.fail = true;
    CHECK(RunPoolKernel(4, 1, 2, visits.data(), false) == 2);
    inner.fail = false;
    CHECK(RunPoolKernel(4, 1, 2, visits.data(), true) == 1);
    CHECK(inner.returned);
  }
  CHECK(annc_get_current_threadpool() == previous);
  {
    tensorflow::ScopedAnncThreadPool scope(nullptr, annc_set_current_threadpool,
                                           annc_get_current_threadpool);
    CHECK(annc_get_current_threadpool() == nullptr);
    scope.Restore();
    CHECK(annc_get_current_threadpool() == previous);
    tensorflow::ScopedAnncThreadPool nested(
        reinterpret_cast<Pool*>(second.get()), annc_set_current_threadpool,
        annc_get_current_threadpool);
    scope.Restore();
    CHECK(annc_get_current_threadpool() ==
          reinterpret_cast<Pool*>(second.get()));
  }
  const std::array<int, 4> expected{{1, 1, 1, 1}};
  CHECK(annc_get_current_threadpool() == previous);
  CHECK(RunPoolKernel(4, 1, 2, visits.data(), false) == 0);
  CHECK(outer.returned && visits == expected);
  return true;
}
}  // namespace

int main() {
  return CheckSchedules() && CheckRecovery() && !annc_get_current_threadpool()
             ? 0
             : 1;
}
