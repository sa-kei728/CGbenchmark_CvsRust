#pragma once

#include "Problem.hpp"

#include "Problem.hpp"
extern "C" {
	void Add(const double*, const double*, std::size_t, double);
	void AddSelf(const double*, const double*, std::size_t, double);
	void Sub(double*, const double*, const double*, std::size_t);
}

namespace ConjugateGradient
{
	// Rust版
	class SolveRust final
	{
	private:
		// 疎行列・ベクトル積 y = A x
		static void SpMV(Problem::VectorT& y, const Problem::MatrixT& A, const Problem::VectorT& x)
		{
			const auto n = y.N;
			for(auto i = decltype(n)(0); i < n; i++)
			{
				double y_i = 0;

				const auto nnz = A.Nonzero(i);
				for(auto idx = decltype(nnz)(0); idx < nnz; idx++)
				{
					const auto a_ij = A.Data(i, idx);
					const auto j = A.Column(i, idx);
					const auto x_j = x[j];
					const auto ax = a_ij * x_j;
					y_i += ax;
				}

				y[i] = y_i;
			}
		}

		// ベクトルの加算y += αx
		static void Add(Problem::VectorT& y, const double alpha, const Problem::VectorT& x)
		{
			const auto n = y.N;
			::AddSelf(y(), x(), n, alpha);
		}
		// ベクトルの加算y = x + βy
		static void Add(Problem::VectorT& y, const Problem::VectorT& x, const double beta)
		{
			const auto n = y.N;
			::Add(y(), x(), n, beta);
		}
		// ベクトルの引き算z = x - y
		static void Sub(Problem::VectorT& z, const Problem::VectorT& x, const Problem::VectorT& y)
		{
			const auto n = y.N;
			::Sub(z(), x(), y(), n);
		}
		// ベクトルの内積r = x・y
		static auto Dot(const Problem::VectorT& x, const Problem::VectorT& y)
		{
			auto r = std::remove_reference_t<decltype(x)>::Type(0);

			const auto n = y.N;
			for(auto i = decltype(n)(0); i < n; i++)
			{
				const auto x_i = x[i];
				const auto y_i = y[i];

				const auto xy = x_i * y_i;
				r += xy;
			}
			return r;
		}
		// ベクトルの内積r = x・x
		static auto Dot(const Problem::VectorT& x)
		{
			auto r = std::remove_reference_t<decltype(x)>::Type(0);

			const auto n = x.N;
			for(auto i = decltype(n)(0); i < n; i++)
			{
				const auto x_i = x[i];

				const auto xx = x_i * x_i;
				r += xx;
			}
			return r;
		}
		// ベクトルの複製y = x
		static void Copy(Problem::VectorT& y, const Problem::VectorT& x)
		{
			const auto n = x.N;
			for(auto i = decltype(n)(0); i < n; i++)
			{
				const auto x_i = x[i];

				y[i] = x_i;
			}
		}

		// 解ベクトル
		Problem::VectorT x;

		// 残差ベクトル
		Problem::VectorT r;

		// 探索ベクトル
		Problem::VectorT p;

		// 行列・ベクトル積の結果
		Problem::VectorT Ap;

	public:

		SolveRust() {}
		SolveRust(const SolveRust&) = delete;
		SolveRust(SolveRust&&) = delete;
		SolveRust& operator=(const SolveRust&) = delete;
		SolveRust& operator=(SolveRust&&) = delete;

		void PreProcess(const Problem& problem)
		{
			const auto n = problem.X().N;

			x.Allocate(n);
			r.Allocate(n);
			p.Allocate(n);
			Ap.Allocate(n);

			// 0初期化
			std::fill_n(x(), n, 0);
		}

		void Solve(const Problem& problem)
		{
			const auto& A = problem.A();
			const auto& b = problem.B();

			// 初期値を設定
			//  (Ap)_0 = A * x
			//  r_0 = b - Ap
			//  p_0 = r_0
			//  rr = r・r
			SpMV(Ap, A, x);
			Sub(r, b, Ap);
			Copy(p, r);
			auto rr = Dot(r);

#ifdef OUTPUT_RESIDUAL
			std::cout << std::endl << 0 << ", " << rr << std::endl;
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
				SpMV(Ap, A, p);
				const auto pAp = Dot(p, Ap);
				const auto alpha = rr / pAp;
				Add(x, alpha, p);
				Add(r, -alpha, Ap);
				const auto rrNew = Dot(r);

				// 収束判定

				// 残りの計算を実行
				//  β= r'r'/rr
				//  p = r' + βp
				//  rr = r'r'
				const auto beta = rrNew / rr;
				Add(p, r, beta);
				rr = rrNew;

#ifdef OUTPUT_RESIDUAL
				std::cout << (i+1) << ", " << rr << std::endl;
#endif
			}
		}

		void PostProcess(Vector<double>& result)
		{
			const auto n = x.N;
			for(auto i = decltype(n)(0); i < n; i++)
			{
				result[i] = x[i];
			}
		}
	};
}
