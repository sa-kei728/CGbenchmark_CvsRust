#ifndef TIMER_INCLUDED
#define TIMER_INCLUDED

#include <chrono>

namespace ConjugateGradient
{
	// 時間計測
	class Timer final
	{
	private:
		std::chrono::time_point<std::chrono::system_clock> begin;

	public:
		void Start()
		{
			this->begin = std::chrono::system_clock::now();
		}

		template<typename T = std::chrono::microseconds>
		auto Time()
		{
			const auto end = std::chrono::system_clock::now();
			return std::chrono::duration_cast<T>(end - begin).count() / 1000.0;
		}
	};
}

#endif
