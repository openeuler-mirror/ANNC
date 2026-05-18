#ifndef CODEGEN_PASSES_H
#define CODEGEN_PASSES_H

#include "Target/aarch64/Passes.h"

namespace annc {
inline void registerAllTargetPasses()
{
    registerAArch64CodeGenPasses();
}
}

#endif // CODEGEN_PASSES_H
