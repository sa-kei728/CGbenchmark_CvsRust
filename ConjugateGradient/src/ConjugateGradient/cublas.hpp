#ifndef CUBLAS_HPP_INCLUDED
#define CUBLAS_HPP_INCLUDED

#include <stdexcept>
#include <sstream>
#include <cublas_v2.h>

namespace cublas
{
	// エラーチェック
	static inline void CheckCublasError(const cublasStatus_t status)
	{
		if(status != CUBLAS_STATUS_SUCCESS)
		{
			std::stringstream ss;
			ss << "error#" << status;
			throw std::runtime_error(ss.str());
		}
	}

	// ハンドル
	class Handle final
	{
	private:
		cublasHandle_t handle;

	public:
		Handle()
		{
			CheckCublasError(cublasCreate(&handle));
		}
		Handle(const Handle& src) : handle(src.handle) {}

		~Handle()
		{
			// CheckCublasError(cublasDestroy(handle));
		}

		auto operator()() const
		{
			return handle;
		}
	};

	// y += αx 
	static inline void Daxpy(
		Handle handle,
		double* const y,
		const double alpha, const double* const x,
		const int n,
		const int strideX = 1, const int strideY = 1)
	{
		CheckCublasError(cublasDaxpy(handle(), n, &alpha, x, strideX, y, strideY));
	}

	// y *= α
	static inline void Dscal(Handle handle,
		const double alpha, double* const x,
		const int n,
		const int stride = 1)
	{
		CheckCublasError(cublasDscal(handle(), n, &alpha, x, stride));
	}

	// y = x
	static inline void Dcopy(Handle handle,
		double* const y, const double* const x,
		const int n,
		const int strideX = 1, const int strideY = 1)
	{
		CheckCublasError(cublasDcopy(handle(), n, x, strideX, y, strideY));
	}

	// r = y・x
	static inline auto Ddot(Handle handle,
		const double* const x, const double* const y,
		const int n,
		const int strideX = 1, const int strideY = 1)
	{
		double ret;
		CheckCublasError(cublasDdot(handle(), n, x, strideX, y, strideY, &ret));
		return ret;
	}
}

#endif
