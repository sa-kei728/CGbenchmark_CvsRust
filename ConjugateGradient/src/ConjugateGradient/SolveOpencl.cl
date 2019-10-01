#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(x) x
#endif
OCL_EXTERNAL_INCLUDE(

// 疎行列・ベクトル積 y = A x
__kernel void SpMV(
	__global double y[],
	__global const double data[],
	__global const ulong column[],
	__global const ulong nonzero[],
	         const ulong maxNonzero,
	__global const double x[],
	         const ulong n)
{
	const int i = get_global_id(0);
	if(i < n)
	{
		double y_i = 0;
		const ulong nnz = nonzero[i];
		for(ulong idx = 0; idx < nnz; ++idx)
		{
			const double a_ij = data[i*maxNonzero + idx];
			const ulong j = column[i*maxNonzero + idx];
			const double x_j = x[j];
			const double ax = a_ij * x_j;
			y_i += ax;
		}

		y[i] = y_i;
	}
}

// ベクトルの加算y += αx
__kernel void AddAlpha(__global double y[], const double alpha, __global const double x[], const ulong n)
{
	const int i = get_global_id(0);
	if(i < n)
	{
		const double x_i = x[i];
		      double y_i = y[i];
		y_i += alpha * x_i;
		y[i] = y_i;
	}
}

// ベクトルの加算y = x + βy
__kernel void AddBeta(__global double y[], __global const double x[], const double beta, const ulong n)
{
	const int i = get_global_id(0);
	if(i < n)
	{
		const double x_i = x[i];
		      double y_i = y[i];
		y_i = x_i + beta * y_i;
		y[i] = y_i;
	}
}

// ベクトルの引き算z = x - y
__kernel void Sub(__global double z[], __global const double x[], __global const double y[], const ulong n)
{
	const int i = get_global_id(0);
	if(i < n)
	{
		const double x_i = x[i];
		const double y_i = y[i];
		const double z_i = x_i - y_i;
		z[i] = z_i;
	}
}

// ベクトルの要素積z = x*y
__kernel void Mul(__global double z[], __global const double x[], __global const double y[], const ulong n)
{
	const int i = get_global_id(0);
	if(i < n)
	{
		const double x_i = x[i];
		const double y_i = y[i];
		const double z_i = x_i * y_i;
		z[i] = z_i;
	}
}

// ベクトルの要素二乗z = x*x
__kernel void Square(__global double z[], __global const double x[], const ulong n)
{
	const int i = get_global_id(0);
	if(i < n)
	{
		const double x_i = x[i];
		const double z_i = x_i * x_i;
		z[i] = z_i;
	}
}

// 総和計算（リダクション）
__kernel void Sum(__global double z[], const ulong n)
{
	const int i = get_global_id(0);
	const int m = get_global_size(0);

	const double left  = z[i];
	const double right = (i + m < n) ? z[i + m] : 0; // 範囲外の場合は0を足す
	const double z_i = left + right;
	z[i] = z_i;
}
)