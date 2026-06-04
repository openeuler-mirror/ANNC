#ifndef ANNC_PATTERNREGISTRYMACROS_H
#define ANNC_PATTERNREGISTRYMACROS_H

#include "PatternRegistry.h"

/// Register a custom fusion pattern.
/// Usage:
///   struct MyPattern : public CustomFusionPatternBase<AnchorOp> { ... };
///   REGISTER_CUSTOM_PATTERN(MyPattern)
///
/// For patterns that only make sense when a specific kernel library is
/// compiled in, wrap the registration with the corresponding macro guard:
///
///   #ifdef ANNC_ENABLE_KDNN_ADAPTOR
///   REGISTER_CUSTOM_PATTERN(KdnnBatchMatmulRewrite)
///   #endif
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
