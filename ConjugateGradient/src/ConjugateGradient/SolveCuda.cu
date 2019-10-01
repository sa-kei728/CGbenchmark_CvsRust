#include "SolveCuda.cuhpp"
#include "cuda_runtime.hpp"

#include <device_launch_parameters.h>

#include <cmath>
#include <algorithm>

namespace ConjugateGradient
{
	namespace SolveCuda
	{
		template<typename T>
		__device__
		static T RoundUp(const T n, const T d)
		{
			const auto r = n % d;
			return (r == 0) ? n : (n + d - r);
		}

		__global__
		static void SpMV(
			double* const __restrict__ y,
			const double* const __restrict__ Adata,
			const std::size_t* const __restrict__ Acolumn,
			const std::size_t* const __restrict__ Anonzero,
			const double* const __restrict__ x,
			const std::size_t n)
		{
			const auto i = blockIdx.y * gridDim.x + blockIdx.x;

			if(i < n)
			{
				const auto idx = threadIdx.x;
				const auto maxNonzero = blockDim.x;

				const auto nnz = Anonzero[i];

				extern __shared__ double buffer[];
				double sum = 0;

				if(idx < nnz)
				{
					const auto offset = i*maxNonzero + idx;
					const auto a_ij = Adata[offset];
					const auto j = Acolumn[offset];
					const auto x_j = x[j];
					const auto ax = a_ij * x_j;

					sum = ax;
				}

#if __CUDA_ARCH__ < 350 // CC 3.5
				buffer[idx] = sum;
				__syncthreads();
#endif

#if __CUDA_ARCH__ < 350 // CC 3.5
				for(auto count = nnz; count > 1; count /= 2)
				{
					const auto offset = count / 2;
					if(idx < offset)
					{
						const auto right = buffer[idx + offset];
						sum += right;

						if((count % 2 != 0) && (idx + 1 == offset))
						{
							const auto rightend = buffer[count - 1];
							sum += rightend;
						}

						buffer[idx] = sum;
					}
					__syncthreads();
				}
#else
				constexpr std::size_t WARP = 32;

				const auto roundupNnz = RoundUp(nnz, WARP);
				for(auto count = roundupNnz; count > WARP; count = RoundUp(count/WARP, WARP))
				{
					if(idx < count)
					{
						// warp単位でリダクション
						sum += __shfl_down(sum, 1);
						sum += __shfl_down(sum, 2);
						sum += __shfl_down(sum, 4);
						sum += __shfl_down(sum, 8);
						sum += __shfl_down(sum, 16);

						// 前の方に詰める
						if(idx % WARP == 0)
						{
							buffer[idx/WARP] = sum;
						}
						sum = (idx < count/WARP) ? buffer[idx] : 0;
					}
				}

				// 最後の1回
				if(idx < WARP)
				{
					sum += __shfl_down(sum, 1);
					sum += __shfl_down(sum, 2);
					sum += __shfl_down(sum, 4);
					sum += __shfl_down(sum, 8);
					sum += __shfl_down(sum, 16);
				}
#endif

				if(idx == 0)
				{
					y[i] = sum;
				}
			}
		}

		__global__
		static void Add(double y[], const double alpha, const double x[], const std::size_t n)
		{
			const auto i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

			if(i < n)
			{
				const auto x_i = x[i];
				auto y_i = y[i];

				y_i += alpha * x_i;
				y[i] = y_i;
			}
		}
		__global__
		static void Add(double y[], const double x[], const double beta, const std::size_t n)
		{
			const auto i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

			if(i < n)
			{
				const auto x_i = x[i];
				const auto y_i0 = y[i];

				const auto y_i = x_i + beta * y_i0;
				y[i] = y_i;
			}
		}
		__global__
		static void Sub(double z[], const double x[], const double y[], const std::size_t n)
		{
			const auto i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

			if(i < n)
			{
				const auto x_i = x[i];
				const auto y_i = y[i];

				const auto z_i = x_i - y_i;
				z[i] = z_i;
			}
		}
		__global__
		static void Dot(const double x[], const double y[], double buffer[], const std::size_t n)
		{
			const auto i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

			if(i < n)
			{
				const auto x_i = x[i];
				const auto y_i = y[i];

				const auto xy = x_i * y_i;
				buffer[i] = xy;
			}
		}
		__global__
		static void Dot(const double x[], double buffer[], const std::size_t n)
		{
			const auto i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

			if(i < n)
			{
				const auto x_i = x[i];

				const auto xx = x_i * x_i;
				buffer[i] = xx;
			}
		}
		__global__
		static void Reduce(double buffer[], const std::size_t n)
		{
			const auto i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

			const auto offset = n / 2;
			if(i < offset)
			{
				auto left = buffer[i];

				const auto right = buffer[i + offset];
				left += right;

				if((n % 2 != 0) && (i + 1 == offset))
				{
					const auto rightend = buffer[n - 1];
					left += rightend;
				}
				
				buffer[i] = left;
			}
		}
	}

	namespace SolveCudaMain
	{
		static constexpr unsigned int THREADS = 32;
		
		__host__
		static decltype(THREADS) GetTotalBlock(const std::size_t n)
		{
			return static_cast<decltype(THREADS)>(std::ceil(static_cast<long double>(n) / THREADS));
		}

		__host__
		static dim3 GetGrid(const std::size_t n)
		{
			constexpr std::size_t MAX_BLOCK = 65535;
			const auto gridX = static_cast<unsigned int>(std::min(n, MAX_BLOCK));
			const auto gridY = static_cast<unsigned int>(n / MAX_BLOCK + 1);
			return dim3(gridX, gridY);
		}


		__host__
		void SpMV(double y[], const double Adata[], const std::size_t Acolumn[], const std::size_t Anonzero[], const double x[], const std::size_t n, const std::size_t maxNonzero)
		{
			const auto grid = GetGrid(n);
			SolveCuda::SpMV<<<grid, maxNonzero, sizeof(double)*maxNonzero>>>(y, Adata, Acolumn, Anonzero, x, n);
			cuda::GetLastError();
		}

		__host__
		void Add(double y[], const double alpha, const double x[], const std::size_t n)
		{
			const auto blocks = GetTotalBlock(n);
			const auto grid = GetGrid(blocks);
			SolveCuda::Add<<<grid, THREADS>>>(y, alpha, x, n);
			cuda::GetLastError();
		}
		__host__
		void Add(double y[], const double x[], const double beta, const std::size_t n)
		{
			const auto blocks = GetTotalBlock(n);
			const auto grid = GetGrid(blocks);
			SolveCuda::Add<<<grid, THREADS>>>(y, x, beta, n);
			cuda::GetLastError();
		}
		__host__
		void Sub(double z[], const double x[], const double y[], const std::size_t n)
		{
			const auto blocks = GetTotalBlock(n);
			const auto grid = GetGrid(blocks);
			SolveCuda::Sub<<<grid, THREADS>>>(z, x, y, n);
			cuda::GetLastError();
		}

		__host__
		void Dot(const double x[], const double y[], double buffer[], const std::size_t n)
		{
			{
				const auto blocks = GetTotalBlock(n);
				const auto grid = GetGrid(blocks);
				SolveCuda::Dot<<<grid, THREADS>>>(x, y, buffer, n);
				cuda::GetLastError();
			}

			for(auto count = n; count > 1; count /= 2)
			{
				const auto blocks = GetTotalBlock(count);
				const auto grid = GetGrid(blocks);
				SolveCuda::Reduce<<<grid, THREADS>>>(buffer, count);
				cuda::GetLastError();
			}
		}
		__host__
		void Dot(const double x[], double buffer[], const std::size_t n)
		{
			{
				const auto blocks = GetTotalBlock(n);
				const auto grid = GetGrid(blocks);
				SolveCuda::Dot<<<grid, THREADS>>>(x, buffer, n);
				cuda::GetLastError();
			}

			for(auto count = n; count > 1; count /= 2)
			{
				const auto blocks = GetTotalBlock(count);
				const auto grid = GetGrid(blocks);
				SolveCuda::Reduce<<<grid, THREADS>>>(buffer, count);
				cuda::GetLastError();
			}
		}
	}
}
