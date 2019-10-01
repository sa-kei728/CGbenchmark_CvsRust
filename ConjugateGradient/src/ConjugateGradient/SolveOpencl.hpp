#ifndef SOLVE_OPENCL_INCLUDED
#define SOLVE_OPENCL_INCLUDED

#include <vector>
#include <stdexcept>
#include <sstream>

#include "Problem.hpp"
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>

namespace ConjugateGradient
{
	// OpenCLで解く
	class SolveOpenCL final
	{
	private:
		// 疎行列・ベクトル積 y = A x
		cl::Kernel spmv;
		void SpMV(cl::Buffer& y, const cl::Buffer& data, const cl::Buffer& column, const cl::Buffer& nonzero, const std::size_t maxNonzero, const cl::Buffer& x)
		{
			spmv.setArg(0, y);
			spmv.setArg(1, data);
			spmv.setArg(2, column);
			spmv.setArg(3, nonzero);
			spmv.setArg(4, maxNonzero);
			spmv.setArg(5, x);
			spmv.setArg(6, n);
			queue.enqueueNDRangeKernel(spmv, cl::NullRange, cl::NDRange(n), cl::NullRange);
		}

		// ベクトルの加算y += αx
		cl::Kernel addAlpha;
		void Add(cl::Buffer& y, const double alpha, const cl::Buffer& x)
		{
			addAlpha.setArg(0, y);
			addAlpha.setArg(1, alpha);
			addAlpha.setArg(2, x);
			addAlpha.setArg(3, n);
			queue.enqueueNDRangeKernel(addAlpha, cl::NullRange, cl::NDRange(n), cl::NullRange);
		}
		// ベクトルの加算y = x + βy
		cl::Kernel addBeta;
		void Add(cl::Buffer& y, const cl::Buffer& x, const double beta)
		{
			addBeta.setArg(0, y);
			addBeta.setArg(1, x);
			addBeta.setArg(2, beta);
			addBeta.setArg(3, n);
			queue.enqueueNDRangeKernel(addBeta, cl::NullRange, cl::NDRange(n), cl::NullRange);
		}
		// ベクトルの引き算z = x - y
		cl::Kernel sub;
		void Sub(cl::Buffer& z, const cl::Buffer& x, const cl::Buffer& y)
		{
			sub.setArg(0, z);
			sub.setArg(1, x);
			sub.setArg(2, y);
			sub.setArg(3, n);
			queue.enqueueNDRangeKernel(sub, cl::NullRange, cl::NDRange(n), cl::NullRange);
		}
		// 総和
		cl::Kernel sum;
		auto Sum(cl::Buffer z)
		{
			sum.setArg(0, z);
			for(auto m = n; m > 1;)
			{
				sum.setArg(1, m);
				m = static_cast<std::size_t>(std::ceil(m/2.0)); // 実際に起動するアイテム数（奇数の場合は1つ多め）
				queue.enqueueNDRangeKernel(sum, cl::NullRange, cl::NDRange(m), cl::NullRange);
			}

			cl_double ret;
			queue.enqueueReadBuffer(z, CL_TRUE, 0, sizeof(cl_double), &ret);
			return ret;
		}
		// ベクトルの内積r = x・y
		cl::Kernel mul;
		auto Dot(const cl::Buffer& x, const cl::Buffer& y)
		{
			mul.setArg(0, z);
			mul.setArg(1, x);
			mul.setArg(2, y);
			mul.setArg(3, n);
			queue.enqueueNDRangeKernel(mul, cl::NullRange, cl::NDRange(n), cl::NullRange);
			return Sum(z);
		}
		// ベクトルの内積r = x・x
		cl::Kernel square;
		auto Dot(const cl::Buffer& x)
		{
			square.setArg(0, z);
			square.setArg(1, x);
			square.setArg(2, n);
			queue.enqueueNDRangeKernel(square, cl::NullRange, cl::NDRange(n), cl::NullRange);
			return Sum(z);
		}
		// ベクトルの複製y = x
		void Copy(cl::Buffer& y, const cl::Buffer& x)
		{
			queue.enqueueCopyBuffer(x, y, 0, 0, sizeof(cl_double)*n);
		}

		struct
		{
			// 要素の値
			cl::Buffer data;

			// 列番号
			cl::Buffer column;
			
			// 非ゼロ要素数
			cl::Buffer nonzero;

		} A;

		// 解ベクトル
		cl::Buffer x;
		Problem::VectorT x_cpu;

		// 残差ベクトル
		cl::Buffer r;
		Problem::VectorT r_cpu;

		// 探索ベクトル
		cl::Buffer p;
		Problem::VectorT p_cpu;

		// 行列・ベクトル積の結果
		cl::Buffer Ap;
		Problem::VectorT Ap_cpu;

		// 右辺ベクトル
		cl::Buffer b;

		// 総和計算（リダクション）用バッファー
		cl::Buffer z;

		// プラットフォーム
		cl::Platform platform;

		// デバイス
		cl::Device device;

		// コンテキスト
		cl::Context context;

		// キュー
		cl::CommandQueue queue;

		// 要素数
		std::size_t n;

	public:

		SolveOpenCL() : n(0)
		{
			// プラットフォーム取得（複数ある場合は一番最後）
			std::vector<cl::Platform> platforms;
			cl::Platform::get(&platforms);
			if(platforms.size() == 0)
			{
				throw std::runtime_error{"No OpenCL platform"};
			}
			platform = *(platforms.rbegin());

			// デバイスを取得（複数ある場合は一番最後）
			std::vector<cl::Device> devices;
			platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
			if(devices.size() == 0)
			{
				throw std::runtime_error{"No OpenCL device"};
			}
			device = *(devices.rbegin());

			// コンテキスト作成
			context = cl::Context{device};

			// プログラムの作成＆ビルド
			#define OCL_EXTERNAL_INCLUDE(x) #x
			constexpr char src[] =
			#include "SolveOpencl.cl"
			;
			cl::Program program{context, src};
			try
			{
				program.build({device});
			}
			// OpenCL例外があった場合
			catch (cl::Error error)
			{
				std::stringstream msg;
				msg << "Build error #" << error.err() << " @ " << error.what() << std::endl;
				msg << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
				throw std::runtime_error{msg.str()};
			}

			// カーネルを作成
			addAlpha = cl::Kernel{program, "AddAlpha"};
			addBeta  = cl::Kernel{program, "AddBeta"};
			sub      = cl::Kernel{program, "Sub"};
			mul      = cl::Kernel{program, "Mul"};
			sum      = cl::Kernel{program, "Sum"};
			square   = cl::Kernel{program, "Square"};
			spmv     = cl::Kernel{program, "SpMV"};

			// キューを作成
			queue = cl::CommandQueue{context, device};
		}

		SolveOpenCL(const SolveOpenCL&) = delete;
		SolveOpenCL(SolveOpenCL&&) = delete;
		SolveOpenCL& operator=(const SolveOpenCL&) = delete;
		SolveOpenCL& operator=(SolveOpenCL&&) = delete;

		void PreProcess(const Problem& problem)
		{
			// バッファー足りない場合は再生成
			const auto lastN = n;
			const auto nnz = problem.A().MaxNonzeroCount;
			n = problem.X().N;
			if(lastN < n)
			{
				x  = cl::Buffer{context, CL_MEM_READ_WRITE, sizeof(cl_double)*n};
				p  = cl::Buffer{context, CL_MEM_READ_WRITE, sizeof(cl_double)*n};
				r  = cl::Buffer{context, CL_MEM_READ_WRITE, sizeof(cl_double)*n};
				Ap = cl::Buffer{context, CL_MEM_READ_WRITE, sizeof(cl_double)*n};
				b  = cl::Buffer{context, CL_MEM_READ_WRITE, sizeof(cl_double)*n};
				z  = cl::Buffer{context, CL_MEM_READ_WRITE, sizeof(cl_double)*n};

				A.data    = cl::Buffer{context, CL_MEM_READ_WRITE, sizeof(cl_double)*n*nnz};
				A.column  = cl::Buffer{context, CL_MEM_READ_WRITE, sizeof(cl_ulong) *n*nnz};
				A.nonzero = cl::Buffer{context, CL_MEM_READ_WRITE, sizeof(cl_ulong) *n};
			}

			x_cpu.Allocate(n);
			r_cpu.Allocate(n);
			p_cpu.Allocate(n);
			Ap_cpu.Allocate(n);

			// 0初期化
			std::fill_n(x_cpu(), n, 0);
			queue.enqueueFillBuffer<cl_double>(x, 0, 0, sizeof(double)*n);

			// 問題データ転送
			queue.enqueueWriteBuffer(A.data,    CL_FALSE, 0, sizeof(cl_double)*n*nnz, problem.A().Data());
			queue.enqueueWriteBuffer(A.column,  CL_FALSE, 0, sizeof(cl_ulong)*n*nnz,  problem.A().Column());
			queue.enqueueWriteBuffer(A.nonzero, CL_FALSE, 0, sizeof(cl_ulong)*n,      problem.A().Nonzero());
			queue.enqueueWriteBuffer(b, CL_TRUE, 0, sizeof(cl_double)*n, problem.B()());
		}

		void Solve(const Problem& problem)
		{
			const auto nnz = problem.A().MaxNonzeroCount;

			// 初期値を設定
			//  (Ap)_0 = A * x
			//  r_0 = b - Ap
			//  p_0 = r_0
			//  rr = r・r
			SpMV(Ap, A.data, A.column, A.nonzero, nnz, x);
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
				SpMV(Ap, A.data, A.column, A.nonzero, nnz, p);
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
			queue.enqueueWriteBuffer(x, CL_TRUE, 0, sizeof(cl_double)*n, result());
		}
	};
}

#endif
