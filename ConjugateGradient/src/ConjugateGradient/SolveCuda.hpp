#ifndef SOLVE_CUDA_INCLUDED
#define SOLVE_CUDA_INCLUDED

#include "Problem.hpp"
#include "cuda_runtime.hpp"
#include "SolveCuda.cuhpp"

namespace ConjugateGradient
{
	// CUDAで解く
	class SolveCUDA final
	{
	private:
		using Value = Problem::VectorT::Type;

		using Index = Problem::VectorT::Index;

		// 疎行列の値
		cuda::unique_ptr<Value[]> Adata;

		// 疎行列の各行の列番号
		cuda::unique_ptr<Index[]> Acolumn;

		// 疎行列の各行の非ゼロ要素数
		cuda::unique_ptr<Index[]> Anonzero;
		
		// 解ベクトル
		cuda::unique_ptr<Value[]> x;

		// 右辺ベクトル
		cuda::unique_ptr<Value[]> b;

		// 残差ベクトル
		cuda::unique_ptr<Value[]> r;

		// 探索ベクトル
		cuda::unique_ptr<Value[]> p;

		// 行列・ベクトル積の結果
		cuda::unique_ptr<Value[]> Ap;


		// 内積計算用バッファー
		cuda::unique_ptr<Value[]> dotBuffer;

		auto Dot(double xx[], const std::size_t n)
		{
			double ret;
			ConjugateGradient::SolveCudaMain::Dot(xx, dotBuffer.get(), n);
			cuda::Memcpy(&ret, dotBuffer.get(), sizeof(ret), cudaMemcpyDeviceToHost);
			return ret;
		}
		auto Dot(double xx[], double yy[], const std::size_t n)
		{
			double ret;
			ConjugateGradient::SolveCudaMain::Dot(xx, yy, dotBuffer.get(), n);
			cuda::Memcpy(&ret, dotBuffer.get(), sizeof(ret), cudaMemcpyDeviceToHost);
			return ret;
		}

	public:

		SolveCUDA() {}
		SolveCUDA(const SolveCUDA&) = delete;
		SolveCUDA(SolveCUDA&&) = delete;
		SolveCUDA& operator=(const SolveCUDA&) = delete;
		SolveCUDA& operator=(SolveCUDA&&) = delete;

		void PreProcess(const Problem& problem)
		{
			const auto n = problem.X().N;
			const auto nnz = problem.A().MaxNonzeroCount;

			Adata.reset(cuda::make_unique<Value[]>(n*nnz).release());
			Acolumn.reset(cuda::make_unique<Index[]>(n*nnz).release());
			Anonzero.reset(cuda::make_unique<Index[]>(n).release());

			x.reset(cuda::make_unique<Value[]>(n).release());
			b.reset(cuda::make_unique<Value[]>(n).release());
			r.reset(cuda::make_unique<Value[]>(n).release());
			p.reset(cuda::make_unique<Value[]>(n).release());
			Ap.reset(cuda::make_unique<Value[]>(n).release());
			dotBuffer.reset(cuda::make_unique<Value[]>(n).release());

			// 0初期化
			cuda::Memset(x.get(), 0, sizeof(Value) * n);

			// 行列と右辺ベクトルをデバイスへ転送
			cuda::Memcpy(Adata.get(), problem.A().Data(), sizeof(Value)*n*nnz, cudaMemcpyHostToDevice);
			cuda::Memcpy(Acolumn.get(), problem.A().Column(), sizeof(Index)*n*nnz, cudaMemcpyHostToDevice);
			cuda::Memcpy(Anonzero.get(), problem.A().Nonzero(), sizeof(Index)*n, cudaMemcpyHostToDevice);
			cuda::Memcpy(b.get(), problem.B()(), sizeof(Value)*n, cudaMemcpyHostToDevice);

			cuda::DeviceSynchronize();
		}

		void Solve(const Problem& problem)
		{
			using namespace ConjugateGradient::SolveCudaMain;

			const auto n = problem.X().N;
			const auto nnz = problem.A().MaxNonzeroCount;

			// 初期値を設定
			//  (Ap)_0 = A * x
			//  r_0 = b - Ap
			//  p_0 = r_0
			//  rr = r・r
			SpMV(Ap.get(), Adata.get(), Acolumn.get(), Anonzero.get(), x.get(), n, nnz);
			Sub(r.get(), b.get(), Ap.get(), n);
			cuda::Memcpy(p.get(), r.get(), sizeof(double)*n, cudaMemcpyDeviceToDevice);
			double rr = Dot(r.get(), n);

#ifdef OUTPUT_RESIDUAL
			std::cout << 0 << ", " << rr << std::endl;
#endif
			const auto loop = Problem::SOLVE_LOOP;
			for(auto i = decltype(loop)(0); i < loop; i++)
			{
				// 計算を実行
				//  Ap = A * p
				//  α = rr/(p・Ap)
				//  x' += αp
				//  r' -= αAp
				//  r'r' = r'・r'
				SpMV(Ap.get(), Adata.get(), Acolumn.get(), Anonzero.get(), p.get(), n, nnz);
				const auto pAp = Dot(p.get(), Ap.get(), n);
				const auto alpha = rr / pAp;
				Add(x.get(), alpha, p.get(), n);
				Add(r.get(), -alpha, Ap.get(), n);
				const auto rrNew = Dot(r.get(), n);

				// 収束判定

				// 残りの計算を実行
				//  β= r'r'/rr
				//  p = r' + βp
				//  rr = r'r'
				const auto beta = rrNew / rr;
				Add(p.get(), r.get(), beta, n);
				rr = rrNew;

#ifdef OUTPUT_RESIDUAL
				std::cout << (i+1) << ", " << rr << std::endl;
#endif
			}

			cuda::DeviceSynchronize();
		}

		void PostProcess(Vector<double>& result)
		{
			const auto n = result.N;
			cuda::Memcpy(result(), x.get(), sizeof(Value)*n, cudaMemcpyDeviceToHost);

			cuda::DeviceSynchronize();
		}
	};
}

#endif
