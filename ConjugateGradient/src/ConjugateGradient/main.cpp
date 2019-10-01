/**************************************************/
// 実行するソルバーの種類などのコンパイルスイッチ //
/**************************************************/
#define SOLVER_SEQUENTIAL_NATIVE // 普通の逐次実行版
//#define SOLVER_OPENMP_NATIVE     // OpenMPで並列化しただけ

/**************/
// デバッグ用 //
/**************/
//#define OUTPUT_PROBLEM // 生成した行列を表示するかどうか
//#define CHECK_RESULT // 理論解（期待値）との比較をするかどうか
//#define OUTPUT_RESIDUAL // 収束状況（残差）を表示するかどうか


/**************************************************/
#include "Problem.hpp"

#include "Solver.hpp"
#ifdef SOLVER_SEQUENTIAL_NATIVE
#include "SolveSequentialNative.hpp"
#endif
#ifdef SOLVER_OPENMP_NATIVE
#include "SolveOpenMPNative.hpp"
#include <omp.h>
#endif

#include <iostream>

#ifdef OUTPUT_PROBLEM
static void OutputProblem(const ConjugateGradient::Problem& problem)
{
	const auto n = problem.X().N;

	auto a_ij = std::make_unique<double[]>(n);
	for(int i = 0; i < n; i++)
	{
		std::fill_n(a_ij.get(), n, 0);

		const auto nnz = problem.A().Nonzero(i);
		for(auto idx = decltype(nnz)(0); idx < nnz; idx++)
		{
			const auto a = problem.A().Data(i, idx);
			const auto j = problem.A().Column(i, idx);

			a_ij[j] = a;
		}

		for(int j = 0; j < n; j++)
		{
			std::cout << a_ij[j] << ", ";
		}
		std::cout << ", " << problem.X()[i] << ",, " << problem.B()[i] << std::endl;
	}
}
#endif

int main()
{
	const double r_e = 2.4;

	const std::size_t n = 1000000;
	const auto nnz = static_cast<std::size_t>(std::pow(std::ceil(r_e*2), 2));

	std::cout << "Generating Problem..." << std::endl;
	ConjugateGradient::Problem problem(n, nnz, r_e);

#ifdef OUTPUT_PROBLEM
	OutputProblem(problem);
#endif

#ifdef SOLVER_SEQUENTIAL_NATIVE
	{
		std::cout << "SequentialNative, ";
		ConjugateGradient::SolveSequentialNative solver;
		ConjugateGradient::Solver::Solve(problem, solver);
	}
#endif

#ifdef SOLVER_OPENMP_NATIVE
	{
		std::cout << "OpenMPNative(";
#pragma omp parallel
		{
#pragma omp master
			{
				std::cout << omp_get_num_threads();
			}
		}
		std::cout  << "), ";
		ConjugateGradient::SolveOpenMPNative solver;
		ConjugateGradient::Solver::Solve(problem, solver);
	}
#endif

	return 0;
}
