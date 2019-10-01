/**************************************************/
// 実行するソルバーの種類などのコンパイルスイッチ //
/**************************************************/
#define SOLVER_SEQUENTIAL_NATIVE // 普通の逐次実行版
//#define SOLVER_OPENMP_NATIVE     // OpenMPで並列化しただけ
//#define SOLVER_SEQUENTIAL_AVX2   // AVX2の逐次実行版
//#define SOLVER_OPENMP_AVX2       // AVX2のOpenMP版
//#define SOLVER_SEQUENTIAL_AVX512 // AVX2の逐次実行版
//#define SOLVER_OPENMP_AVX512     // AVX2のOpenMP版
//#define SOLVER_CUDA              // CUDA版
//#define SOLVER_CUSPARSE          // cuSPARSE版
//#define SOLVER_CUBLASPARSE       // cuSPARSEとcuBLAS併用版
//#define SOLVER_OPENCL            // OpenCL版
//#define SOLVER_PEZY              // PEZY版

#define CUSPARSE_ELL // cuSPARSEでELL形式を用いる（指定しない場合はCSR）

/**************/
// デバッグ用 //
/**************/
// #define OUTPUT_PROBLEM // 生成した行列を表示するかどうか
// #define CHECK_RESULT // 理論解（期待値）との比較をするかどうか
// #define OUTPUT_RESIDUAL // 収束状況（残差）を表示するかどうか


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
#ifdef SOLVER_SEQUENTIAL_AVX2
#include "SolveSequentialAvx2.hpp"
#endif
#ifdef SOLVER_OPENMP_AVX2
#include "SolveOpenMPAvx2.hpp"
#include <omp.h>
#endif
#ifdef SOLVER_SEQUENTIAL_AVX512
#include "SolveSequentialAvx512.hpp"
#endif
#ifdef SOLVER_OPENMP_AVX512
#include "SolveOpenMPAvx512.hpp"
#include <omp.h>
#endif
#ifdef SOLVER_CUDA
#include "SolveCuda.hpp"
#endif
#ifdef SOLVER_CUSPARSE
#include "SolveCusparse.hpp"
#endif
#ifdef SOLVER_CUBLASPARSE
#include "SolveCublasparse.hpp"
#endif
#ifdef SOLVER_OPENCL
#include "SolveOpencl.hpp"
#endif
#ifdef SOLVER_PEZY
#include "SolvePezy.hpp"
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

#ifdef SOLVER_SEQUENTIAL_AVX2
	{
		std::cout << "SequentialAvx2, ";
		ConjugateGradient::SolveSequentialAvx2 solver;
		ConjugateGradient::Solver::Solve(problem, solver);
	}
#endif

#ifdef SOLVER_OPENMP_AVX2
	{
		std::cout << "OpenMPAvx2(";
#pragma omp parallel
		{
#pragma omp master
			{
				std::cout << omp_get_num_threads();
			}
		}
		std::cout << "), ";
		ConjugateGradient::SolveOpenMPAvx2 solver;
		ConjugateGradient::Solver::Solve(problem, solver);
	}
#endif

#ifdef SOLVER_SEQUENTIAL_AVX512
        {
                std::cout << "SequentialAvx512, ";
                ConjugateGradient::SolveSequentialAvx512 solver;
                ConjugateGradient::Solver::Solve(problem, solver);
        }
#endif

#ifdef SOLVER_OPENMP_AVX512
        {
                std::cout << "OpenMPAvx512(";
#pragma omp parallel
                {
#pragma omp master
                        {
                                std::cout << omp_get_num_threads();
                        }
                }
                std::cout << "), ";
                ConjugateGradient::SolveOpenMPAvx512 solver;
                ConjugateGradient::Solver::Solve(problem, solver);
        }
#endif

#ifdef SOLVER_CUDA
	{
		std::cout << "CUDA(";

		cudaDeviceProp prop;
		cuda::GetDeviceProperties(&prop, 0);
		std::cout << prop.name << "), ";

		ConjugateGradient::SolveCUDA solver;
		ConjugateGradient::Solver::Solve(problem, solver);
	}
#endif

#ifdef SOLVER_CUSPARSE
	{
		std::cout << "cuSPARSE";
#ifdef CUSPARSE_ELL
		std::cout << "-ELL";
#endif
		std::cout << "(";

		cudaDeviceProp prop;
		cuda::GetDeviceProperties(&prop, 0);
		std::cout << prop.name << "), ";

		ConjugateGradient::SolveCusparse solver;
		ConjugateGradient::Solver::Solve(problem, solver);
	}
#endif

#ifdef SOLVER_CUBLASPARSE
	{
		std::cout << "cuSPARSE";
#ifdef CUSPARSE_ELL
		std::cout << "-ELL";
#endif
		std::cout << "&cuBLAS(";

		cudaDeviceProp prop;
		cuda::GetDeviceProperties(&prop, 0);
		std::cout << prop.name << "), ";

		ConjugateGradient::SolveCublasparse solver;
		ConjugateGradient::Solver::Solve(problem, solver);
	}
#endif

#ifdef SOLVER_OPENCL
	{
		std::cout << "OpenCL, ";
		ConjugateGradient::SolveOpenCL solver;
		ConjugateGradient::Solver::Solve(problem, solver);
	}
#endif

#ifdef SOLVER_PEZY
	{
		std::cout << "PEZY, ";
		ConjugateGradient::SolvePezy solver;
		ConjugateGradient::Solver::Solve(problem, solver);
	}
#endif

	return 0;
}
