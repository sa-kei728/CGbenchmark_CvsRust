#ifndef CUDA_RUNTIME_HPP_INCLUDED
#define CUDA_RUNTIME_HPP_INCLUDED

#ifndef __CUDACC__
#include <memory>
#endif

#include <stdexcept>
#include <cuda_runtime.h>

namespace cuda
{
	// エラーチェック
	static inline void CheckCudaError(const cudaError_t error)
	{
		if(error != cudaSuccess)
		{
			throw std::runtime_error(cudaGetErrorString(error));
		}
	}

	template<class T>
	static inline void Malloc(T** ptr, const std::size_t size)
	{
		CheckCudaError(cudaMalloc(ptr, size));
	}
	static inline void Free(void *ptr)
	{
		CheckCudaError(cudaFree(ptr));
	}
	static inline void Memcpy(void *dst, const void *src, const std::size_t size, const enum cudaMemcpyKind kind)
	{
		CheckCudaError(cudaMemcpy(dst, src, size, kind));
	}
	static inline void Memset(void *ptr, const int value, const std::size_t size)
	{
		CheckCudaError(cudaMemset(ptr, value, size));
	}
	static inline void DeviceSynchronize()
	{
		CheckCudaError(cudaDeviceSynchronize());
	}
	static inline void GetLastError()
	{
		CheckCudaError(cudaGetLastError());
	}
	static inline void GetDeviceProperties(cudaDeviceProp *prop, const int device)
	{
		CheckCudaError(cudaGetDeviceProperties(prop, device));
	}

#ifndef __CUDACC__
	template<typename T>
	struct deleter
	{
		void operator()(T* ptr) const
		{
			Free(ptr);
		}
	};

	template<typename T>
	using unique_ptr = std::unique_ptr<T, deleter<std::remove_extent_t<T>>>;

	template<typename T>
	static inline auto make_unique(const std::size_t n)
	{
		using Type = std::remove_extent_t<T>;
		Type* ptr = nullptr;
		Malloc(&ptr, sizeof(Type)* n);
		return unique_ptr<T>(ptr);
	}
#endif
}

#endif
