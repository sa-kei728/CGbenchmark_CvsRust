#ifndef CUSPARSE_HPP_INCLUDED
#define CUSPARSE_HPP_INCLUDED

#include <stdexcept>
#include <sstream>
#include <cusparse.h>

namespace cusparse
{
	// エラーチェック
	static inline void CheckCusparseError(const cusparseStatus_t status)
	{
		if(status != CUSPARSE_STATUS_SUCCESS)
		{
			std::stringstream ss;
			ss << "error#" << status;
			throw std::runtime_error(ss.str());
		}
	}

	// ハンドル
	class Handle final
	{
	private:
		cusparseHandle_t handle;

	public:
		Handle()
		{
			CheckCusparseError(cusparseCreate(&handle));
		}
		Handle(const Handle& src) = delete;
		Handle(Handle&& src) : handle(src.handle) {}
		Handle& operator=(const Handle& src) = delete;

		~Handle()
		{
			CheckCusparseError(cusparseDestroy(handle));
		}

		auto operator()() const
		{
			return handle;
		}
	};

	// 行列形式
	class MatDescr final
	{
	private:
		cusparseMatDescr_t desc;

	public:
		MatDescr()
		{
			CheckCusparseError(cusparseCreateMatDescr(&desc));
		}
		MatDescr(const MatDescr& src) = delete;
		MatDescr(MatDescr&& src) : desc(src.desc) {}
		MatDescr& operator=(const MatDescr& src) = delete;

		~MatDescr()
		{
			CheckCusparseError(cusparseDestroyMatDescr(desc));
		}

		void Type(const cusparseMatrixType_t val)
		{
			CheckCusparseError(cusparseSetMatType(desc, val));
		}

		void IndexBase(const cusparseIndexBase_t val)
		{
			CheckCusparseError(cusparseSetMatIndexBase(desc, val));
		}

		auto operator()() const
		{
			return desc;
		}
	};

	// ハイブリッド形式の行列
	class HybMat final
	{
	private:
		cusparseHybMat_t mat;

	public:
		HybMat()
		{
			CheckCusparseError(cusparseCreateHybMat(&mat));
		}
		HybMat(const HybMat& src) = delete;
		HybMat(HybMat&& src) : mat(src.mat) {}
		HybMat& operator=(const HybMat& src) = delete;

		~HybMat()
		{
			CheckCusparseError(cusparseDestroyHybMat(mat));
		}

		auto operator()() const
		{
			return mat;
		}
	};

	// y = α*Ax + β*y
	static inline void Dcsrmv(
		Handle& handle,
		cusparseOperation_t op,
		const int m, const int n, const int nnz,
		const MatDescr& descr,
		const double* const Adata, const int* const Acolumn, const int* const ArowOffset,
		const double alpha, const double* const x,
		const double beta, double* const y)
	{
		CheckCusparseError(cusparseDcsrmv(handle(), op, m, n, nnz, &alpha, descr(), Adata, ArowOffset, Acolumn, x, &beta, y));
	}
	static inline void Dhybmv(
		Handle& handle,
		cusparseOperation_t op,
		const MatDescr& descr,
		const HybMat& mat,
		const double alpha, const double* const x,
		const double beta, double* const y)
	{
		CheckCusparseError(cusparseDhybmv(handle(), op, &alpha, descr(), mat(), x, &beta, y));
	}

	static inline void Dcsr2Hyb(
		Handle& handle,
		const MatDescr& descr,
		const int m, const int n,
		const double* const Adata, const int* const Acolumn, const int* const ArowOffset,
		HybMat& mat, const int ellWidth, const cusparseHybPartition_t partition)
	{
		CheckCusparseError(cusparseDcsr2hyb(handle(), m, n, descr(), Adata, ArowOffset, Acolumn, mat(), ellWidth, partition));
	}
}

#endif
