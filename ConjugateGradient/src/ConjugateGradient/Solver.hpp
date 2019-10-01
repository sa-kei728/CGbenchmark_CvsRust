#ifndef SOLVER_INCLUDED
#define SOLVER_INCLUDED

#include "Problem.hpp"
#include "Timer.hpp"

#include <iostream>
#include <cfloat>

namespace ConjugateGradient
{
	// 普通に逐次実行で解く
	class Solver final
	{
	public:

		template<typename SOLVER>
		static void Solve(const Problem& problem, SOLVER& solver)
		{
			const auto n = problem.X().N;

			Timer timer;

			// 前処理
			timer.Start();
			solver.PreProcess(problem);
			std::cout << "PreProcess, " << timer.Time() << ", " << std::flush;

			// 計算本体
			timer.Start();
			solver.Solve(problem);
			std::cout << "Solve, " << timer.Time() << ", " << std::flush;

			// 後処理（結果を格納する部分）
			Vector<double> x(n);
			timer.Start();
			solver.PostProcess(x);
			std::cout << "PostProcess, " << timer.Time() << std::endl;

#ifdef CHECK_RESULT
			// 理論解との一致具合を確認する
			for(auto i = decltype(n)(0); i < n; i++)
			{
				const auto expected = problem.X()[i];
				const auto actual = x[i];

				const auto r = std::abs((actual - expected)/expected);

				constexpr auto EPSILON = DBL_EPSILON;

				// TODO: 理論的（？）に求める
				if(r > EPSILON * 10)
				{
					std::cout << i << ": expected=" << expected << ", actual=" << actual << " -> residual=" << r << std::endl;
				}
			}
#endif
		}
	};
}

#endif
