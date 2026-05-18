#ifndef ANNC_PATTERNREGISTRYMACROS_H
#define ANNC_PATTERNREGISTRYMACROS_H

#include "PatternRegistry.h"

/// New Automated Registration Macro
/// Usage:
///   struct MyPattern : public CustomFusionPatternBase<AnchorOp> { ... };
///   REGISTER_CUSTOM_PATTERN(MyPattern)
#define REGISTER_CUSTOM_PATTERN(PatternType) \
  namespace { \
    static const bool __registered_##PatternType = \
      atir::PatternRegistry::instance().registerPattern<PatternType>(#PatternType); \
  }

/// Legacy Registration Macro
#define REGISTER_CUSTOM_FUSION_PATTERN(PatternType) \
  REGISTER_CUSTOM_PATTERN(PatternType)

/// Another legacy alias
#define ANNC_REGISTER_PATTERN(PatternType) \
  REGISTER_CUSTOM_PATTERN(PatternType)

#endif  // ANNC_PATTERNREGISTRYMACROS_H
