#ifndef SOLVE_CUBLASPARSE_INCLUDED
#define SOLVE_CUBLASPARSE_INCLUDED

#include "Problem.hpp"
#include "cuda_runtime.hpp"
#include "cusparse.hpp"
#include "cublas.hpp"

namespace ConjugateGradient
{
	// cuSPARSEとcublasで解く
	class SolveCublasparse final
	{
	private:
		using Value = Problem::VectorT::Type;

		using Index = Problem::VectorT::Index;

		// 疎行列の値
		cuda::unique_ptr<Value[]> Adata;

		// 疎行列の各行の列番号
		cuda::unique_ptr<int[]> Acolumn;

		// 疎行列の各行の先頭位置
		cuda::unique_ptr<int[]> ArowOffset;

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

		// cuSPARSE用
		cusparse::Handle sparse;

		// cuBLAS用
		cublas::Handle blas;

		// 行列形式
		cusparse::MatDescr descr;

		// 行列全体の非ゼロ要素数
		int totalNonzero;

#ifdef CUSPARSE_ELL
		cusparse::HybMat mat;

		void SpMV(double y[], const cusparse::HybMat& m, const double xx[])
		{
			cusparse::Dhybmv(sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
				descr, m, 1, xx, 0, y);
		}
#else
		void SpMV(double y[], const double data[], const int column[], const int rowOffset[], const double xx[], const std::size_t n)
		{
			const auto nn = static_cast<int>(n);
			cusparse::Dcsrmv(sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, nn, nn, totalNonzero,
				descr, data, column, rowOffset, 1, xx, 0, y);
		}
#endif

		// y += a*x
		void Add(double y[], const double alpha, const double xx[], const std::size_t n)
		{
			const auto nn = static_cast<int>(n);
			cublas::Daxpy(blas, y, alpha, xx, nn);
		}
		// y = x + b*y
		void Add(double y[], const double xx[], const double beta, const std::size_t n)
		{
			const auto nn = static_cast<int>(n);
			// y *= b
			// y += x
			cublas::Dscal(blas, beta, y, nn);
			cublas::Daxpy(blas, y, 1, xx, nn);
		}
		// z = x - y
		void Sub(double z[], const double xx[], const double y[], const std::size_t n)
		{
			const auto nn = static_cast<int>(n);
			// z = x
			// z += -y
			cublas::Dcopy(blas, z, xx, nn);
			cublas::Daxpy(blas, z, -1, y, nn);
		}

		auto Dot(double xx[], const std::size_t n)
		{
			const auto nn = static_cast<int>(n);
			return cublas::Ddot(blas, xx, xx, nn);
		}
		auto Dot(double xx[], double yy[], const std::size_t n)
		{
			const auto nn = static_cast<int>(n);
			return cublas::Ddot(blas, xx, yy, nn);
		}

	public:

		SolveCublasparse() {}
		SolveCublasparse(const SolveCublasparse&) = delete;
		SolveCublasparse(SolveCublasparse&&) = delete;
		SolveCublasparse& operator=(const SolveCublasparse&) = delete;
		SolveCublasparse& operator=(SolveCublasparse&&) = delete;

		void PreProcess(const Problem& problem)
		{
			const auto n = problem.X().N;
			const auto nnz = problem.A().MaxNonzeroCount;

			Adata.reset(cuda::make_unique<Value[]>(n*nnz).release());
			Acolumn.reset(cuda::make_unique<int[]>(n*nnz).release());
			ArowOffset.reset(cuda::make_unique<int[]>(n + 1).release());

			x.reset(cuda::make_unique<Value[]>(n).release());
			b.reset(cuda::make_unique<Value[]>(n).release());
			r.reset(cuda::make_unique<Value[]>(n).release());
			p.reset(cuda::make_unique<Value[]>(n).release());
			Ap.reset(cuda::make_unique<Value[]>(n).release());
			dotBuffer.reset(cuda::make_unique<Value[]>(n).release());

			// 0初期化
			cuda::Memset(x.get(), 0, sizeof(Value) * n);

			// CSR形式に変換
			Vector<double> data(n*nnz);
			Vector<int> column(n*nnz);
			Vector<int> rowOffset(n + 1);
			{
				rowOffset[0] = 0;
				totalNonzero = 0;
				for(auto i = decltype(n)(0); i < n; i++)
				{
					const auto nnz_i = problem.A().Nonzero()[i];
					for(auto idx = decltype(nnz_i)(0); idx < nnz_i; idx++)
					{
						data[totalNonzero + idx] = problem.A().Data(i, idx);
						column[totalNonzero + idx] = static_cast<int>(problem.A().Column(i, idx));
					}

					totalNonzero += static_cast<int>(nnz_i);
					rowOffset[i + 1] = totalNonzero;
				}
			}

			// 行列と右辺ベクトルをデバイスへ転送
			cuda::Memcpy(Adata.get(), data(), sizeof(Value)*n*nnz, cudaMemcpyHostToDevice);
			cuda::Memcpy(Acolumn.get(), column(), sizeof(int)*n*nnz, cudaMemcpyHostToDevice);
			cuda::Memcpy(ArowOffset.get(), rowOffset(), sizeof(int)*(n + 1), cudaMemcpyHostToDevice);
			cuda::Memcpy(b.get(), problem.B()(), sizeof(Value)*n, cudaMemcpyHostToDevice);

			// 行列形式を設定
			descr.Type(CUSPARSE_MATRIX_TYPE_GENERAL);
			descr.IndexBase(CUSPARSE_INDEX_BASE_ZERO);

#ifdef CUSPARSE_ELL
			// ELL形式に変換
			cusparse::Dcsr2Hyb(sparse, descr, static_cast<int>(n), static_cast<int>(n), Adata.get(), Acolumn.get(), ArowOffset.get(), mat, static_cast<int>(nnz), CUSPARSE_HYB_PARTITION_MAX);
#endif
			cuda::DeviceSynchronize();


		}

		void Solve(const Problem& problem)
		{
			const auto n = problem.X().N;

			// 初期値を設定
			//  (Ap)_0 = A * x
			//  r_0 = b - Ap
			//  p_0 = r_0
			//  rr = r・r
#ifdef CUSPARSE_ELL
			SpMV(Ap.get(), mat, x.get());
#else
			SpMV(Ap.get(), Adata.get(), Acolumn.get(), ArowOffset.get(), x.get(), n);
#endif
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
#ifdef CUSPARSE_ELL
				SpMV(Ap.get(), mat, p.get());
#else
				SpMV(Ap.get(), Adata.get(), Acolumn.get(), ArowOffset.get(), p.get(), n);
#endif
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
