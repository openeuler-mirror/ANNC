#include <absl/log/internal/log_message.h>

namespace absl {
ABSL_NAMESPACE_BEGIN
namespace log_internal {

template LogMessage& LogMessage::operator<<(const unsigned long& v);

}
ABSL_NAMESPACE_END
}