cc_library(
	name ="bisheng-cpu",
	srcs = glob([".cc"]),
	hdrs = glob(["*.h"]),
	linkstatic = True,
	alwayslink = True,
	visibility = ["//visibility:public"],
	deps = [
		"@xla//xla/hlo/ir:hlo",
		"@xla//xla/service:hlo_pass",
		"@xla//xla/service:custom_call_target_registry",
		"@xla//xla/service:pattern_matcher",
	],
)
