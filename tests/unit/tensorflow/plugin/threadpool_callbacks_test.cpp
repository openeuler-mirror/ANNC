#include <algorithm>
#include <array>
#include <atomic>
#include <iostream>

#include "threadpool_callbacks.h"
#include "tensorflow/core/platform/env.h"

namespace {

struct Work {
  std::array<int, 64> visits{};
  std::atomic<int> calls{0};
};

void Visit(void* context, int64_t begin, int64_t end) {
  auto& work = *static_cast<Work*>(context);
  ++work.calls;
  for (int64_t i = begin; i < end; ++i) ++work.visits[i];
}

}  // namespace

int main() {
  tensorflow::thread::ThreadPool pool(tensorflow::Env::Default(),
                                      "annc-shard-limit", 4);
  const auto callbacks = annc::tensorflow::MakeThreadPoolCallbacks(pool);
  for (int limit : {1, 2}) {
    for (int64_t block : {0, 1}) {
      Work work;
      // A large cost prevents the cost model from hiding a lost limit by
      // running this small fixture serially. The limit overrides fixed blocks.
      const int status = callbacks.parallel_for(
          callbacks.context, work.visits.size(), block, 1000000, limit, Visit,
          &work);
      if (status || work.calls.load() < 1 || work.calls.load() > limit ||
          !std::all_of(work.visits.begin(), work.visits.end(),
                       [](int count) { return count == 1; })) {
        std::cerr << "TF shard limit or task coverage failed: limit=" << limit
                  << " block=" << block << " calls=" << work.calls.load()
                  << '\n';
        return 1;
      }
    }
  }
  return 0;
}
