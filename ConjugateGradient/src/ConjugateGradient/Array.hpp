#ifndef ARRAY_INCLUDED
#define ARRAY_INCLUDED

#include <memory>
#include <type_traits>

// iccはC++14の対応が遅れているので
#ifdef __INTEL_COMPILER
namespace std
{
	template<typename T>
	using add_pointer_t = typename add_pointer<T>::type;

	template<typename T>
	using add_const_t = typename add_const<T>::type;

	template<typename T>
	using remove_reference_t = typename remove_reference<T>::type;

	template<typename T>
	using remove_const_t = typename remove_const<T>::type;

	
	template<typename T, typename... ARGS>
	static typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type make_unique(ARGS... args)
	{
    	return std::unique_ptr<T>(new T(std::forward<ARGS>(args)...));
	}

	template<typename T>
	static typename std::enable_if<std::is_array<T>::value, std::unique_ptr<T>>::type make_unique(const std::size_t n)
	{
	    using E = typename std::remove_extent<T>::type;
	    return std::unique_ptr<T>(new E[n]);
	}


	static void* align(
		std::size_t alignment,
		std::size_t size,
		void* & ptr,
		std::size_t& space)
	{
		const auto p = reinterpret_cast<std::uintptr_t>(ptr);
		auto offset = static_cast<std::size_t>(p & (alignment - 1));

		if (0 < offset)
		{
			offset = alignment - offset;
		}

		void* ret = nullptr;
		if ((space >= offset) && (space >= size + offset))
		{
			ptr = reinterpret_cast<void*>(p + offset);
			space -= offset;
			ret = ptr;
		}

		return ret;
	}
}
#endif

namespace ConjugateGradient
{
	template<typename T>
	class Array final
	{
	public:
		using Type = std::remove_reference_t<T>;
		using Index = std::size_t;

		const Index N;

	private:
		std::unique_ptr<std::uint8_t[]> buffer;
		T* data;

	public:
		Array() noexcept
			: N(0), buffer(nullptr)
		{}

		Array(const std::size_t n, const std::size_t alignment = sizeof(T))
			: N(n), buffer(std::make_unique<std::uint8_t[]>(sizeof(T) * n + alignment))
		{
			void* ptr = buffer.get();
			auto size = sizeof(T) * n + alignment;
			const auto ret = std::align(alignment, sizeof(T) * n, ptr, size);
			if(ret != nullptr)
			{
				data = static_cast<T*>(ptr);
			}
			else
			{
				throw std::bad_alloc();
			}
		}

		Array(Array<Type>&& src) noexcept
			: N(src.N), buffer(std::move(src.data))
		{}

		Array(const Array<Type>& src) = delete;

		Type operator[](const Index i) const noexcept
		{
			return data[i];
		}
		Type& operator[](const Index i) noexcept
		{
			return data[i];
		}

		Array& operator=(const Array<Type>& src) = delete;
		Array& operator=(Array<Type>&& src) noexcept
		{
			this->data = std::move(src.data);
			const_cast<std::remove_const_t<decltype(src.N)>&>(this->N) = src.N;

			return *this;
		}

		Type* operator()() noexcept
		{
			return data;
		}
		const Type* operator()() const noexcept
		{
			return data;
		}
		Type operator()(const Index i) const noexcept
		{
			return operator[](i);
		}
		Type& operator()(const Index i) noexcept
		{
			return operator[](i);
		}

		void Allocate(const Index n, const std::size_t alignment = sizeof(T))
		{
			if(!(this->buffer))
			{
				auto next = std::make_unique<std::uint8_t[]>(sizeof(T) * n + alignment);
				this->buffer.reset(next.release());

				{
					void* ptr = buffer.get();
					auto size = sizeof(T) * n + alignment;
					const auto ret = std::align(alignment, sizeof(T) * n, ptr, size);
					if(ret != nullptr)
					{
						data = static_cast<T*>(ptr);
					}
					else
					{
						throw std::bad_alloc();
					}
				}

				const_cast<std::remove_const_t<decltype(n)>&>(this->N) = n;
			}
			else
			{
				throw std::bad_alloc();
			}
		}
	};
}
#endif
