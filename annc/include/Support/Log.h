#ifndef ANNC_SUPPORT_LOG_H
#define ANNC_SUPPORT_LOG_H

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <strings.h>
#include <utility>

// ANNC 极简分级日志（单头文件，无新库依赖）。
//
// 环境变量 ANNC_LOG_LEVEL 控制输出级别（大小写不敏感）：
//   off | error | warn | info | debug，默认 error（库安静原则：未被
// 请求时保持静默；用户面向的警告应走 MLIR emitWarning 诊断而非本宏）。
// 非空但无法识别的取值（如 warnn/verbose）回退 error，并向 stderr
// 输出一次性提示——本变量是唯一控制入口，静默回退会让排查方向误入
// 日志调用点。unset/空值是正常默认路径，不提示。
// 级别在进程首次使用时读取并缓存（函数局部 static：静态初始化顺序
// 安全、线程安全），因此 main 之前的静态初始化阶段（如 kernel
// registrar）同样生效——这是不使用 CLI flag 的原因。
//
// 输出统一走 std::cerr（stderr），绝不写 stdout（stdout 是 MLIR 输出
// 通道）。不用 llvm::errs()：libANNCKernel 等库会被无 LLVM 链接的
// 单测直接链接，日志头文件不得引入 LLVM Support 符号依赖。
//
// 线程安全：整条日志由 RAII 临时对象（detail::LogStream）缓冲，在
// 完整语句结束时经单次 std::cerr.write() 一次性写出，多线程下（如
// annc-opt --split-input-file 的 chunk 级并行）同一行的 << 片段不会
// 交错；行尾自动补 '\n'（已有换行不重复），遗忘 "\n" 不会粘连。
//
// 用法（语句式；宏展开自带 else 分支，dangling-else 安全，外层
// if/else 会正确绑定到调用处）：
//   ANNC_LOG_INFO("kernel") << "Registered " << name << "\n";

namespace annc {
namespace log {

enum class Level : int { Off = -1, Error = 0, Warn = 1, Info = 2, Debug = 3 };

namespace detail {

inline Level parseLevel(const char *value) {
  if (!value || !*value) return Level::Error;
  if (strcasecmp(value, "off") == 0) return Level::Off;
  if (strcasecmp(value, "error") == 0) return Level::Error;
  if (strcasecmp(value, "warn") == 0) return Level::Warn;
  if (strcasecmp(value, "info") == 0) return Level::Info;
  if (strcasecmp(value, "debug") == 0) return Level::Debug;
  // 非空但无法识别：提示后回退。直接写 std::cerr 而非分级宏——
  // 回退级别为 error，WARN 级提示会被门控吞掉。本函数仅经
  // cachedLevel 的函数局部 static 调用一次（C++11 magic statics
  // 保证并发下也只执行一次），因此提示天然是一次性的。
  std::cerr << "ANNC_LOG_LEVEL '" << value
            << "' unrecognized, falling back to 'error'\n";
  return Level::Error;
}

inline Level cachedLevel() {
  static const Level level = parseLevel(std::getenv("ANNC_LOG_LEVEL"));
  return level;
}

inline bool enabled(Level msgLevel) { return msgLevel <= cachedLevel(); }

// RAII 日志流：仅经 ANNC_LOG_* 宏创建临时对象。整条日志先拼入内部
// buffer，完整语句结束时（临时对象析构）经单次 std::cerr.write()
// 一次性写出，避免多线程下同一行的 << 片段交错；行尾自动补 '\n'
// （已有换行不重复），调用方遗忘 "\n" 不会导致日志粘连。
class LogStream {
 public:
  explicit LogStream(const char *prefix) { buffer_ << prefix; }

  LogStream(const LogStream &) = delete;
  LogStream &operator=(const LogStream &) = delete;

  ~LogStream() {
    std::string message = buffer_.str();
    if (message.empty() || message.back() != '\n') message.push_back('\n');
    std::cerr.write(message.data(),
                    static_cast<std::streamsize>(message.size()));
  }

  template <typename T>
  LogStream &operator<<(T &&value) {
    buffer_ << std::forward<T>(value);
    return *this;
  }

  // 兼容 std::endl 等 ostream 操纵符（换行与写出时机由析构统一处理，
  // 一般直接用 "\n" 即可）。
  LogStream &operator<<(std::ostream &(*manip)(std::ostream &)) {
    manip(buffer_);
    return *this;
  }

 private:
  std::ostringstream buffer_;
};

}  // namespace detail

// 供宏之外的复杂打印使用（如需跨多个语句组织一条日志时）。
inline bool levelEnabled(Level msgLevel) { return detail::enabled(msgLevel); }

}  // namespace log
}  // namespace annc

#define ANNC_LOG_IMPL(subsystem, levelTag, level) \
  if (!::annc::log::detail::enabled((level))) {  \
  } else                                         \
    ::annc::log::detail::LogStream("[annc:" subsystem "][" levelTag "] ")

#define ANNC_LOG_ERROR(subsystem) \
  ANNC_LOG_IMPL(subsystem, "error", ::annc::log::Level::Error)
#define ANNC_LOG_WARN(subsystem) \
  ANNC_LOG_IMPL(subsystem, "warn", ::annc::log::Level::Warn)
#define ANNC_LOG_INFO(subsystem) \
  ANNC_LOG_IMPL(subsystem, "info", ::annc::log::Level::Info)
#define ANNC_LOG_DEBUG(subsystem) \
  ANNC_LOG_IMPL(subsystem, "debug", ::annc::log::Level::Debug)

#endif  // ANNC_SUPPORT_LOG_H
