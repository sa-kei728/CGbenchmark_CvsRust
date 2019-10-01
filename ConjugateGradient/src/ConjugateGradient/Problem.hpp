#ifndef PROBLEM_INCLUDED
#define PROBLEM_INCLUDED

#include "SparseMatrix.hpp"
#include "Vector.hpp"

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>

namespace ConjugateGradient
{
	// 問題生成
	class Problem final
	{
	public:
		static constexpr std::size_t SOLVE_LOOP = 50;

		using MatrixT = SparseMatrix<double, std::size_t>;
		using VectorT = Vector<double>;

	private:
		// 係数行列
		MatrixT a;

		// 理論解（期待値）
		VectorT x;

		// 右辺ベクトル
		VectorT b;

		// 円周率
		template<typename T>
		struct PI
		{
			static constexpr auto Value = static_cast<T>(3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651L);
		};

		// 重み関数
		static auto W(const double r, const double r_e)
		{
			// r_e/r - 1
			return ((0 < r) && (r < r_e)) ? (r_e / r - 1) : 0;
		}

	public:

		Problem(const std::size_t n, const std::size_t nnz,
			const double r_e)
			: a(n, nnz), x(n), b(n)
		{
			constexpr auto L0 = decltype(r_e)(1);

			// 円周上に粒子の位置を生成
			struct X
			{
				double x;
				double y;

				X(const double xx, const double yy)
					: x(xx), y(yy) {}
			};
			auto particles = std::vector<X>();
			{
				// 中心の粒子
				particles.emplace_back(0, 0);

				double r = L0; // 現在の半径

				constexpr auto pi = PI<double>::Value;

				std::size_t nn = 1; // 何週目か
				std::size_t mm = 6; // 何角形か（最初は6角形）
				double theta0 = 0; // 位相ずれ

				auto j = decltype(n)(0);
				for(auto i = decltype(n)(1); i < n; i++)
				{
					const auto theta = theta0 + 2 * pi * j / mm; // 現在の角度

					particles.emplace_back(r * std::cos(theta), r * std::sin(theta));

					j++;

					// 次の周
					if(j == mm)
					{
						j = 0;

						nn++;
						mm = static_cast<decltype(mm)>(std::ceil(pi / std::asin(1.0 / (2 * nn))));
						theta0 = (nn % 4) * pi / 2;
						r = nn * L0;
					}
				}

				// 最終周
				for(; j < mm; j++)
				{
					const auto theta = theta0 + 2 * pi * j / mm; // 現在の角度
					particles.emplace_back(r * std::cos(theta), r * std::sin(theta));
				}
				// もう1周
				{
					nn++;
					mm = static_cast<decltype(mm)>(std::ceil(pi / std::asin(1.0 / (2 * nn))));
					theta0 = (nn % 4) * pi / 2;
					r = nn * L0;
					for(j = 0; j < mm; j++)
					{
						const auto theta = theta0 + 2 * pi * j / mm; // 現在の角度
						particles.emplace_back(r * std::cos(theta), r * std::sin(theta));
					}
				}
			}

			// 一番外側の半径
			const auto maxR = std::sqrt(particles.crbegin()->x * particles.crbegin()->x + particles.crbegin()->y * particles.crbegin()->y);

			// 探索用グリッドを生成
			const auto gridCount = 2 * static_cast<std::size_t>(std::ceil(maxR / r_e));
			std::unordered_map<std::size_t, std::vector<std::size_t>> grid;
			const auto m = particles.size();
			for(auto i = decltype(m)(0); i < m; i++)
			{
				const auto gridX = static_cast<std::size_t>(std::floor((maxR + particles[i].x) / r_e));
				const auto gridY = static_cast<std::size_t>(std::floor((maxR + particles[i].y) / r_e));
				const auto key = gridX*gridCount + gridY;

				grid[key].emplace_back(i);
			}

			// 係数行列と理論解（期待値）を生成
			for(auto i = decltype(particles.size())(0); i < n; i++)
			{
				const auto x_i = particles[i].x;
				const auto y_i = particles[i].y;

				double a_ii = 0;

				const auto gridX = static_cast<std::size_t>(std::floor((maxR + x_i) / r_e));
				const auto gridY = static_cast<std::size_t>(std::floor((maxR + y_i) / r_e));

				// 近傍粒子から
				for(auto gX = (gridX == 0 ? 0 : (gridX - 1)); gX <= gridX + 1; gX++)
				{
					for(auto gY = (gridY == 0 ? 0 : (gridY - 1)); gY <= gridY + 1; gY++)
					{
						const auto key = gX * gridCount + gY;
						auto block = grid.find(key);

						if(block != grid.end())
						{
							for(auto j : block->second)
							{
								const auto dx = particles[j].x - x_i;
								const auto dy = particles[j].y - y_i;
								const auto r_ij = std::sqrt(dx*dx + dy*dy);
								const auto w = W(r_ij, r_e);

								if(w > 0)
								{
									a_ii += w;

									// 外側の境界値は計算に含めない
									if(j < n)
									{
										a.AddColumn(i, j, w);
									}
								}
							}
						}
					}
				}
				a.AddColumn(i, i, -a_ii);

				// 理論解（期待値）は、中心が1、一番外側が0で線形に変化するものとする
				const auto r = std::sqrt(x_i*x_i + y_i*y_i);
				x[i] = 1 - r / maxR;
			}


			// 理論解（期待値）になるように右辺ベクトルを生成
			for(auto i = decltype(n)(0); i < n; i++)
			{
				double b_i = 0;

				const auto nonzero = a.Nonzero(i);
				for(auto idx = decltype(nonzero)(0); idx < nonzero; idx++)
				{
					const auto a_ij = a.Data(i, idx);
					const auto j = a.Column(i, idx);
					const auto x_j = x[j];
					const auto ax = a_ij * x_j;
					b_i += ax;
				}

				b[i] = b_i;
			}
		}

		Problem() = delete;
		Problem(const Problem&) = delete;
		Problem(Problem&&) = delete;
		Problem& operator=(const Problem&) = delete;
		Problem& operator=(Problem&&) = delete;

		const decltype(a)& A() const
		{
			return a;
		}
		const decltype(x)& X() const
		{
			return x;
		}
		const decltype(b)& B() const
		{
			return b;
		}
	};
}

#endif
