#ifndef SOLVE_SEQUENTIAL_AVX2_INCLUDED
#define SOLVE_SEQUENTIAL_AVX2_INCLUDED

#include "Problem.hpp"

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <immintrin.h>
#endif

namespace ConjugateGradient
{
	// AVX2の逐次実行で解く
	class SolveSequentialAvx2 final
	{
	private:
		// 行列の値
		Array<Problem::MatrixT::Value> Adata;

		// 各行の列番号
		Array<Problem::MatrixT::Index> Acolumn;

		// 各行の非ゼロ要素数
		Array<Problem::MatrixT::Index> Anonzero;

		// 1行の最大非ゼロ要素数（アライメントを考慮）
		std::size_t maxNonzeroCount;

		// 右辺ベクトル
		Problem::VectorT b;

		// 解ベクトル
		Problem::VectorT x;

		// 残差ベクトル
		Problem::VectorT r;

		// 探索ベクトル
		Problem::VectorT p;

		// 行列・ベクトル積の結果
		Problem::VectorT Ap;

		// SIMD要素1つにdoubleが何個入っているか
		static constexpr auto SIMD_COUNT = sizeof(__m256d) / sizeof(double);


		// 疎行列・ベクトル積 y = A x
		static void SpMV(Problem::VectorT& y,
			const Array<Problem::MatrixT::Value>& data,
			const Array<Problem::MatrixT::Index>& column,
			const Array<Problem::MatrixT::Index>& nonzero,
			const Problem::VectorT& x,
			const std::size_t maxNonzero)
		{
			const auto n = y.N;

			const auto ZERO = _mm256_setzero_pd();

			const auto rest = n % SIMD_COUNT;
			for(auto i = decltype(n)(0); i < n - rest; i += SIMD_COUNT)
			{
				auto y_i = _mm256_setzero_pd();

				const auto nnz = _mm256_load_si256(reinterpret_cast<const __m256i*>(nonzero() + i));
				for(auto idx = decltype(maxNonzero)(0); idx < maxNonzero; idx++)
				{
					const auto a_ij = _mm256_load_pd(data() + i * maxNonzero + idx * SIMD_COUNT);
					const auto j = _mm256_load_si256(reinterpret_cast<const __m256i*>(column() + i * maxNonzero + idx * SIMD_COUNT));

					const auto iidex = static_cast<std::int64_t>(idx);
					const auto simdIdx = _mm256_set_epi64x(iidex, iidex, iidex, iidex);
					const auto mask = _mm256_cmpgt_epi64(nnz, simdIdx);
					
					const auto maskd = _mm256_castsi256_pd(mask); // 気にされるのは最上位ビットだけなので無理矢理で良い
					const auto x_j = _mm256_mask_i64gather_pd(ZERO, x(), j, maskd, sizeof(double));

					y_i = _mm256_fmadd_pd(a_ij, x_j, y_i);
				}

				_mm256_store_pd(y() + i, y_i);
			}

			// 端数処理
			for(auto i = n - rest; i < n; i++)
			{
				double y_i = 0;

				const auto nnz = nonzero(i);
				for(auto idx = decltype(nnz)(0); idx < nnz; idx++)
				{
					const auto a_ij = data[i * maxNonzero + idx];
					const auto j = column[i * maxNonzero + idx];
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
			const auto a = _mm256_broadcast_sd(&alpha);

			const auto rest = n % SIMD_COUNT;
			for(auto i = decltype(n)(0); i < n - rest; i += SIMD_COUNT)
			{
				const auto x_i = _mm256_load_pd(x() + i);
				const auto y_i0 = _mm256_load_pd(y() + i);

				const auto y_i = _mm256_fmadd_pd(a, x_i, y_i0);
				_mm256_store_pd(y() + i, y_i);
			}

			// 端数処理
			for(auto i = n - rest; i < n; i++)
			{
				const auto x_i = x[i];
				auto y_i = y[i];

				y_i += alpha * x_i;
				y[i] = y_i;
			}
		}
		// ベクトルの加算y = x + βy
		static void Add(Problem::VectorT& y, const Problem::VectorT& x, const double beta)
		{
			const auto n = y.N;
			const auto b = _mm256_broadcast_sd(&beta);

			const auto rest = n % SIMD_COUNT;
			for(auto i = decltype(n)(0); i < n - rest; i += SIMD_COUNT)
			{
				const auto x_i = _mm256_load_pd(x() + i);
				const auto y_i0 = _mm256_load_pd(y() + i);

				const auto y_i = _mm256_fmadd_pd(b, y_i0, x_i);
				_mm256_store_pd(y() + i, y_i);
			}

			// 端数処理
			for(auto i = n - rest; i < n; i++)
			{
				const auto x_i = x[i];
				const auto y_i0 = y[i];

				const auto y_i = x_i + beta * y_i0;
				y[i] = y_i;
			}
		}
		// ベクトルの引き算z = x - y
		static void Sub(Problem::VectorT& z, const Problem::VectorT& x, const Problem::VectorT& y)
		{
			const auto n = y.N;
			const auto rest = n % SIMD_COUNT;
			for(auto i = decltype(n)(0); i < n - rest; i += SIMD_COUNT)
			{
				const auto x_i = _mm256_load_pd(x() + i);
				const auto y_i = _mm256_load_pd(y() + i);

				const auto z_i = _mm256_sub_pd(x_i, y_i);
				_mm256_store_pd(z() + i, z_i);
			}

			// 端数処理
			for(auto i = n - rest; i < n; i++)
			{
				const auto x_i = x[i];
				const auto y_i = y[i];

				const auto z_i = x_i - y_i;
				z[i] = z_i;
			}
		}
		// ベクトルの内積r = x・y
		static auto Dot(const Problem::VectorT& x, const Problem::VectorT& y)
		{
			auto sum = _mm256_setzero_pd();

			const auto n = y.N;
			const auto rest = n % SIMD_COUNT;
			for(auto i = decltype(n)(0); i < n - rest; i += SIMD_COUNT)
			{
				const auto x_i = _mm256_load_pd(x() + i);
				const auto y_i = _mm256_load_pd(y() + i);

				sum = _mm256_fmadd_pd(x_i, y_i, sum);
			}

			alignas(256/8) double rr[SIMD_COUNT];
			_mm256_store_pd(rr, sum);
			auto r = rr[0] + rr[1] + rr[2] + rr[3];

			// 端数処理
			for(auto i = n - rest; i < n; i++)
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
			auto sum = _mm256_setzero_pd();

			const auto n = x.N;
			const auto rest = n % SIMD_COUNT;
			for(auto i = decltype(n)(0); i < n - rest; i += SIMD_COUNT)
			{
				const auto x_i = _mm256_load_pd(x() + i);
				sum = _mm256_fmadd_pd(x_i, x_i, sum);
			}

			alignas(256 / 8) double rr[SIMD_COUNT];
			_mm256_store_pd(rr, sum);
			auto r = rr[0] + rr[1] + rr[2] + rr[3];

			// 端数処理
			for(auto i = n - rest; i < n; i++)
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
		
		// nをm単位に切り上げる
		template<typename T>
		static auto RoundUp(const T n, const T m)
		{
			const auto r = n%m;
			return (r == 0) ? n : (n + m - r);
		}

	public:

		SolveSequentialAvx2() {}
		SolveSequentialAvx2(const SolveSequentialAvx2&) = delete;
		SolveSequentialAvx2(SolveSequentialAvx2&&) = delete;
		SolveSequentialAvx2& operator=(const SolveSequentialAvx2&) = delete;
		SolveSequentialAvx2& operator=(SolveSequentialAvx2&&) = delete;

		void PreProcess(const Problem& problem)
		{
			const auto n = problem.X().N;
			const auto maxNonzero = problem.A().MaxNonzeroCount;
			const auto maxNonzeroSimd = RoundUp(maxNonzero, SIMD_COUNT);
			maxNonzeroCount = maxNonzeroSimd;

			constexpr std::size_t ALIGN = sizeof(__m256d);

			Adata.Allocate(n*maxNonzeroSimd, ALIGN);
			Acolumn.Allocate(n*maxNonzeroSimd, ALIGN);
			Anonzero.Allocate(n, ALIGN);
			b.Allocate(n, ALIGN);
			x.Allocate(n, ALIGN);
			r.Allocate(n, ALIGN);
			p.Allocate(n, ALIGN);
			Ap.Allocate(n, ALIGN);

			// アラインされた領域にデータを複製
			std::copy_n(problem.A().Nonzero(), n, Anonzero());
			std::copy_n(problem.B()(), n, b());

			// Sliced-ELLに変換
			for(auto i = decltype(n)(0); i < n; i += SIMD_COUNT)
			{
				for(auto idx = decltype(maxNonzero)(0); idx < maxNonzero; idx++)
				{
					const auto srcOffset = i * maxNonzero + idx;
					const auto dstOffset = i* maxNonzeroSimd + SIMD_COUNT * idx;

					for(auto j = decltype(SIMD_COUNT)(0); j < SIMD_COUNT; j++)
					{
						// j行目をj列目へ
						Adata[dstOffset + j] = problem.A().Data()[srcOffset + j * maxNonzero];
						Acolumn[dstOffset + j] = problem.A().Column()[srcOffset + j * maxNonzero];
					}
				}
			}

			// 0初期化
			std::fill_n(x(), n, 0);
		}

		void Solve(const Problem&)
		{
			// 初期値を設定
			//  (Ap)_0 = A * x
			//  r_0 = b - Ap
			//  p_0 = r_0
			//  rr = r・r
			SpMV(Ap, Adata, Acolumn, Anonzero, x, maxNonzeroCount);
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
				SpMV(Ap, Adata, Acolumn, Anonzero, p, maxNonzeroCount);
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
				std::cout << (i + 1) << ", " << rr << std::endl;
#endif
			}
		}

		void PostProcess(Vector<double>& result)
		{
			const auto n = x.N;
			std::copy_n(x(), n, result());
		}
	};
}

#endif
