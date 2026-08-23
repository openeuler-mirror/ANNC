ANNC_KERNEL(Name("KPFusedGather")
                .Backend("aarch64")
                .Abi("annc_execution_v2")
                .TypeConstraint<float>("T")
                .TypeConstraint<int64_t>("Tkeys")
                .TypeConstraint<int32_t>("Tbegin")
                .TypeConstraint<int32_t>("Tindices"),
            (const AnncExecutionContext* execution, AnncMemRef2DF32* data,
             AnncMemRef2DI64* keys, AnncMemRef1DI32* begin),
            {
              const auto status =
                  kpFusedGatherImpl(::annc::threadpool::getCurrentThreadPool(),
                                    execution, data, keys, begin);
              return ::annc::kernels::statusCode(status);
            })
