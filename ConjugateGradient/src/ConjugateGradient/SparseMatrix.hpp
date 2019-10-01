#ifndef SPARSE_MATRIX_HPP_INCLUDED
#define SPARSE_MATRIX_HPP_INCLUDED

#include "Array.hpp"

#include <type_traits>
#include <algorithm>

namespace ConjugateGradient
{
	template<typename VALUE, typename INDEX>
	class SparseMatrix final
	{
	public:
		using Value = VALUE;
		using Index = INDEX;

		template<typename T>
		using ConstPointer = std::add_pointer_t<std::add_const_t<std::remove_reference_t<T>>>;

	private:
		// 行列の値
		Array<Value> data;

		// 各行の列番号
		Array<Index> column;

		// 各行の非ゼロ要素数
		Array<Index> nonzero;

	public:
		// 行・列数
		const Index N;

		// 最大非ゼロ要素数
		const Index MaxNonzeroCount;

		SparseMatrix(const decltype(N) n, const decltype(MaxNonzeroCount) nnz)
			:
			data(n*nnz),
			column(n*nnz),
			nonzero(n),
			N(n), MaxNonzeroCount(nnz)
		{
			std::fill_n(nonzero(), n, 0);
		}

		SparseMatrix() = delete;
		SparseMatrix(const SparseMatrix&) = delete;
		SparseMatrix(SparseMatrix&&) = delete;
		SparseMatrix& operator=(const SparseMatrix&) = delete;
		SparseMatrix& operator=(SparseMatrix&&) = delete;

		void AddColumn(const Index i, const Index j, const Value value)
		{
			const auto idx = nonzero[i];
			nonzero[i]++;
			Data(i, idx) = value;
			Column(i, idx) = j;
		}

		decltype(auto) Data(const Index i, const Index index)
		{
			return data[i*MaxNonzeroCount + index];
		}
		decltype(auto) Data(const Index i, const Index index) const
		{
			return data[i*MaxNonzeroCount + index];
		}
		decltype(auto) Data()
		{
			return data();
		}
		ConstPointer<decltype(data[0])> Data() const
		{
			return data();
		}

		decltype(auto) Column(const Index i, const Index index)
		{
			return column[i*MaxNonzeroCount + index];
		}
		decltype(auto) Column(const Index i, const Index index) const
		{
			return column[i*MaxNonzeroCount + index];
		}
		decltype(auto) Column()
		{
			return column();
		}
		ConstPointer<decltype(column[0])> Column() const
		{
			return column();
		}

		decltype(auto) Nonzero(const Index i)
		{
			return nonzero[i];
		}
		decltype(auto) Nonzero(const Index i) const
		{
			return nonzero[i];
		}
		decltype(auto) Nonzero()
		{
			return nonzero();
		}
		ConstPointer<decltype(nonzero[0])> Nonzero() const
		{
			return nonzero();
		}
	};
}
#endif
