#ifndef KERNELS_H
#define KERNELS_H

//c
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>
//cuda
#include <vector_types.h>
#include <vector_functions.h>
#include <cuda.h>
#include <cufft.h>
#include <vector_types.h>
#include <vector_functions.h>
#include "cutil.h"
#include "cutil_inline_runtime.h"

__global__
void zero_pad(int *dNe, int *dNx, int *dNy, int *dNz, double *ikx, double *iky, double *ikz, cufftDoubleComplex *u) {
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	while (ind < *dNe) {
		double wave = ikx[ind];
		if ((wave > *dNx/3.0 && wave <= (*dNx/2.0 - 1.0)) || (wave >= (1.0 - *dNx/2.0) && wave < -*dNx/3.0)) {
			u[ind].x = 0.0;
			u[ind].y = 0.0;
			continue;
		}
		wave = iky[ind];
		if ((wave > *dNy/3.0 && wave <= (*dNy/2.0 - 1.0)) || (wave >= (1.0 - *dNy/2.0) && wave < -*dNy/3.0)) {
			u[ind].x = 0.0;
			u[ind].y = 0.0;
			continue;
		}
		wave = ikz[ind];
		if ((wave > *dNz/3.0 && wave <= (*dNz/2.0 - 1.0)) || (wave >= (1.0 - *dNz/2.0) && wave < -*dNz/3.0)) {
			u[ind].x = 0.0;
			u[ind].y = 0.0;
		}
	}
};

__global__ 
void cuDC_init(int *dNe, cufftDoubleComplex* u) {
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	while (ind < *dNe) {
		u[ind] = (cufftDoubleComplex) make_double2(0.0,0.0);
		ind+= blockDim.x * gridDim.x;
	}
};

__global__
void kx_scal(int *dNe, cufftDoubleComplex* u, double* ikx, cufftDoubleComplex* ux) {
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	while (ind < *dNe) {		
		ux[ind].x = -u[ind].y * ikx[ind];
		ux[ind].y = u[ind].x * ikx[ind];
		ind+= blockDim.x * gridDim.x;
	}
};

__global__
void ky_scal(int *dNe, cufftDoubleComplex* u, double* iky, cufftDoubleComplex* uy)
{
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	while (ind < *dNe) {
		uy[ind].x = -u[ind].y * iky[ind];
		uy[ind].y = u[ind].x * iky[ind];
		ind+= blockDim.x * gridDim.x;
	}
};

__global__
void kz_scal(int *dNe, cufftDoubleComplex* u, double* ikz, cufftDoubleComplex* uz)
{
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	while (ind < *dNe) {
		uz[ind].x = -u[ind].y * ikz[ind];
		uz[ind].y = u[ind].x * ikz[ind];
		ind+= blockDim.x * gridDim.x;
	}
};

__global__
void full_scal(int *dNe, cufftDoubleComplex* u, cufftDoubleComplex *v, cufftDoubleComplex *r)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while (i < *dNe) {
		r[i].x = u[i].x * v[i].x - u[i].y * v[i].y;
		r[i].y = u[i].x * v[i].y + u[i].y * v[i].x;
		i+= blockDim.x * gridDim.x;
	}
};

__global__
void add(int *dNe, cufftDoubleComplex* u, cufftDoubleComplex *v, cufftDoubleComplex *r)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while (i < *dNe) {
		r[i].x = u[i].x + v[i].x;
		r[i].y = u[i].y + v[i].y;
		i+= blockDim.x * gridDim.x;
	}
};

__global__
void negate(int *dNe, cufftDoubleComplex *u)
{
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	while (ind < *dNe) {
		u[ind].x = -u[ind].x;
		u[ind].y = -u[ind].y;
		ind+= blockDim.x * gridDim.x;
	}
};

__global__
void direct_compute_1(  int *dNe, double *dRe, double *dPr, double *ddt,
						cufftDoubleComplex *bK_o, cufftDoubleComplex *bK_n,
						cufftDoubleComplex *nlbK_o, cufftDoubleComplex *nlbK_n,
						cufftDoubleComplex *wK_n, double* ikx, double* iky, double* ikz)
{
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	while (ind < *dNe) {
		double factor = (3.0/(2.0* *ddt)) - (ikx[ind]*ikx[ind] + iky[ind]*iky[ind] + ikz[ind]*ikz[ind]) / (*dRe * *dPr);
		bK_o[ind].x = ((4*bK_n[ind].x - bK_o[ind].x) / (2* *ddt) - 2*nlbK_n[ind].x + nlbK_o[ind].x - wK_n[ind].x)/factor;
		bK_o[ind].y = ((4*bK_n[ind].y - bK_o[ind].y) / (2* *ddt) - 2*nlbK_n[ind].y + nlbK_o[ind].y - wK_n[ind].y)/factor;
		ind+= blockDim.x * gridDim.x;
	}
};

__global__
void adjoint_compute_1( int *dNe, double *dRe, double *dPr, double *dRi, double *ddt,
						cufftDoubleComplex *bK_o, cufftDoubleComplex *bK_n,
						cufftDoubleComplex *nlbK_o, cufftDoubleComplex *nlbK_n,
						cufftDoubleComplex *wK_n, double* ikx, double* iky, double* ikz)
{
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	while (ind < *dNe) {
		double factor = (3.0/(2.0* *ddt)) + (ikx[ind]*ikx[ind] + iky[ind]*iky[ind] + ikz[ind]*ikz[ind]) / (*dRe * *dPr);
		bK_o[ind].x = ((4*bK_n[ind].x - bK_o[ind].x) / (2* *ddt) + 2*nlbK_n[ind].x - nlbK_o[ind].x - *dRi*wK_n[ind].x)/factor;
		bK_o[ind].y = ((4*bK_n[ind].y - bK_o[ind].y) / (2* *ddt) + 2*nlbK_n[ind].y - nlbK_o[ind].y - *dRi*wK_n[ind].y)/factor;
		ind+= blockDim.x * gridDim.x;
	}
};

/*
__global__
void direct_compute_2(  int *dNe, cufftDoubleComplex *nluK_n, cufftDoubleComplex *nluK_o,
						cufftDoubleComplex *nlvK_n, cufftDoubleComplex *nlvK_o,
						cufftDoubleComplex *nlwK_n, cufftDoubleComplex *nlwK_o,
						cufftDoubleComplex *bK_o, cufftDoubleComplex *pestK,
						double *ikx, double *iky, double *ikz)
{
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	while (ind < *dNe) {
		double factor = ikx[ind]*ikx[ind] + iky[ind]*iky[ind] + ikz[ind]*ikz[ind] + 0.1e13;
		pestK[ind].x = (-2.0*(-ikx[ind]*nluK_n[ind].y - iky[ind]*nlvK_n[ind].y - ikz[ind]*nlwK_n[ind].y)
						+ (-ikx[ind]*nluK_o[ind].y - iky[ind]*nlvK_o[ind].y - ikz[ind]*nlwK_o[ind].y)
						- (-ikz[ind]*bK_o[ind].y))/factor;
		pestK[ind].y = (-2.0*(ikx[ind]*nluK_n[ind].x + iky[ind]*nlvK_n[ind].x + ikz[ind]*nlwK_n[ind].x)
						+ (ikx[ind]*nluK_o[ind].x + iky[ind]*nlvK_o[ind].x + ikz[ind]*nlwK_o[ind].x)
						- (ikz[ind]*bK_o[ind].x))/factor;
		ind+= blockDim.x * gridDim.x;
	}
};
*/
//TODO: make sure this new direct_compute_2 is correct
__global__
void direct_compute_2(int *dNe, cufftDoubleComplex *pestK, cufftDoubleComplex *nlpK_n, cufftDoubleComplex *nlpK_o, 
						cufftDoubleComplex *bK_o, double *ikx, double *iky, double *ikz)
{
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	while (ind < *dNe) {
		double factor = ikx[ind]*ikx[ind] + iky[ind]*iky[ind] + ikz[ind]*ikz[ind] + 0.1e13;
		pestK[ind].x = (2.0 * nlpK_n[ind].x - nlpK_o[ind].x - (-ikz[ind] * bK_o[ind].y))/factor;
		pestK[ind].y = (2.0 * nlpK_n[ind].y - nlpK_o[ind].y - (ikz[ind] * bK_o[ind].x))/factor;
		ind+= blockDim.x * gridDim.x;
	}
};

__global__
void direct_compute_3(  int *dNe, double *dRe, double *dRi, double *ddt,
						cufftDoubleComplex *uK_n, cufftDoubleComplex *uK_o,
						cufftDoubleComplex *vK_n, cufftDoubleComplex *vK_o,
						cufftDoubleComplex *wK_n, cufftDoubleComplex *wK_o,
						cufftDoubleComplex *nluK_n, cufftDoubleComplex *nluK_o,
						cufftDoubleComplex *nlvK_n, cufftDoubleComplex *nlvK_o,
						cufftDoubleComplex *nlwK_n, cufftDoubleComplex *nlwK_o,
						cufftDoubleComplex *pestK, cufftDoubleComplex *bK_o,
						double *ikx, double *iky, double *ikz,
						cufftDoubleComplex *uestK, cufftDoubleComplex *vestK, cufftDoubleComplex *westK)
{
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	while (ind < *dNe) {
		double factor = (3.0/(2.0* *ddt)) - (ikx[ind]*ikx[ind] + iky[ind]*iky[ind] + ikz[ind]*ikz[ind]) / *dRe;
		uestK[ind].x = ((4*uK_n[ind].x - uK_o[ind].x) / (2* *ddt) - 2*nluK_n[ind].x + nluK_o[ind].x
						- (-ikx[ind]*pestK[ind].y))/factor;
		uestK[ind].y = ((4*uK_n[ind].y - uK_o[ind].y) / (2* *ddt) - 2*nluK_n[ind].y + nluK_o[ind].y
						- (ikx[ind]*pestK[ind].x))/factor;
		vestK[ind].x = ((4*vK_n[ind].x - vK_o[ind].x) / (2* *ddt) - 2*nlvK_n[ind].x + nlvK_o[ind].x
						- (-iky[ind]*pestK[ind].y))/factor;
		vestK[ind].y = ((4*vK_n[ind].y - vK_o[ind].y) / (2* *ddt) - 2*nlvK_n[ind].y + nlvK_o[ind].y
						- (iky[ind]*pestK[ind].x))/factor;
		westK[ind].x = ((4*wK_n[ind].x - wK_o[ind].x) / (2* *ddt) - 2*nlwK_n[ind].x + nlwK_o[ind].x
						- (-ikz[ind]*pestK[ind].y) - *dRi*bK_o[ind].x)/factor;
		westK[ind].y = ((4*wK_n[ind].y - wK_o[ind].y) / (2* *ddt) - 2*nlwK_n[ind].y + nlwK_o[ind].y
						- (ikz[ind]*pestK[ind].x) - *dRi*bK_o[ind].y)/factor;
		ind+= blockDim.x * gridDim.x;
	}
};

__global__
void adjoint_compute_3( int *dNe, double *dRe, double *ddt,
						cufftDoubleComplex *uK_n, cufftDoubleComplex *uK_o,
						cufftDoubleComplex *vK_n, cufftDoubleComplex *vK_o,
						cufftDoubleComplex *wK_n, cufftDoubleComplex *wK_o,
						cufftDoubleComplex *nluK_n, cufftDoubleComplex *nluK_o,
						cufftDoubleComplex *nlvK_n, cufftDoubleComplex *nlvK_o,
						cufftDoubleComplex *nlwK_n, cufftDoubleComplex *nlwK_o,
						cufftDoubleComplex *pestK, cufftDoubleComplex *bK_o,
						double *ikx, double *iky, double *ikz,
						cufftDoubleComplex *uestK, cufftDoubleComplex *vestK, cufftDoubleComplex *westK)
{
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	while (ind < *dNe) {
		double factor = (3.0/(2.0* *ddt)) + (ikx[ind]*ikx[ind] + iky[ind]*iky[ind] + ikz[ind]*ikz[ind]) / *dRe;
		uestK[ind].x = ((4*uK_n[ind].x - uK_o[ind].x) / (2* *ddt) + 2*nluK_n[ind].x - nluK_o[ind].x
						+ (-ikx[ind]*pestK[ind].y))/factor;
		uestK[ind].y = ((4*uK_n[ind].y - uK_o[ind].y) / (2* *ddt) + 2*nluK_n[ind].y - nluK_o[ind].y
						+ (ikx[ind]*pestK[ind].x))/factor;
		vestK[ind].x = ((4*vK_n[ind].x - vK_o[ind].x) / (2* *ddt) + 2*nlvK_n[ind].x - nlvK_o[ind].x
						+ (-iky[ind]*pestK[ind].y))/factor;
		vestK[ind].y = ((4*vK_n[ind].y - vK_o[ind].y) / (2* *ddt) + 2*nlvK_n[ind].y - nlvK_o[ind].y
						+ (iky[ind]*pestK[ind].x))/factor;
		westK[ind].x = ((4*wK_n[ind].x - wK_o[ind].x) / (2* *ddt) + 2*nlwK_n[ind].x - nlwK_o[ind].x
						+ (-ikz[ind]*pestK[ind].y) - bK_o[ind].x)/factor;
		westK[ind].y = ((4*wK_n[ind].y - wK_o[ind].y) / (2* *ddt) + 2*nlwK_n[ind].y - nlwK_o[ind].y
						+ (ikz[ind]*pestK[ind].x) + bK_o[ind].y)/factor;
		ind+= blockDim.x * gridDim.x;
	}
};

__global__ 
void direct_compute_4(  int *dNe, double *ddt,
						cufftDoubleComplex *uestK, cufftDoubleComplex *vestK, cufftDoubleComplex *westK,
						double *ikx, double *iky, double *ikz, cufftDoubleComplex *phiK)
{
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	while (ind < *dNe) {
		double factor = 3.0/(2.0 * *ddt * (ikx[ind]*ikx[ind] + iky[ind]*iky[ind] + ikz[ind]*ikz[ind] + 0.1e13));
		phiK[ind].x = (-ikx[ind]*uestK[ind].y - iky[ind]*vestK[ind].y - ikz[ind]*westK[ind].y)*factor;
		phiK[ind].y = (ikx[ind]*uestK[ind].x + iky[ind]*vestK[ind].x + ikz[ind]*westK[ind].x)*factor;
		ind+= blockDim.x * gridDim.x;
	}
};

__global__
void direct_compute_5(  int *dNe, double *ddt,
						cufftDoubleComplex *uestK, cufftDoubleComplex *vestK, cufftDoubleComplex *westK,
						cufftDoubleComplex *phiK, double *ikx, double *iky, double *ikz,
						cufftDoubleComplex *uK_o, cufftDoubleComplex *vK_o, cufftDoubleComplex *wK_o)
{
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	while (ind < *dNe) {
		uK_o[ind].x = uestK[ind].x - (2.0 * *ddt / 3.0 * (-ikx[ind]*phiK[ind].y));
		uK_o[ind].y = uestK[ind].y - (2.0 * *ddt / 3.0 * (ikx[ind]*phiK[ind].x));
		vK_o[ind].x = vestK[ind].x - (2.0 * *ddt / 3.0 * (-iky[ind]*phiK[ind].y));
		vK_o[ind].y = vestK[ind].y - (2.0 * *ddt / 3.0 * (iky[ind]*phiK[ind].x));
		wK_o[ind].x = westK[ind].x - (2.0 * *ddt / 3.0 * (-ikz[ind]*phiK[ind].y));
		wK_o[ind].y = westK[ind].y - (2.0 * *ddt / 3.0 * (ikz[ind]*phiK[ind].x));
		ind+= blockDim.x * gridDim.x;
	}
};

__global__
void scale_fft(int *dNe, double *dsNe, cufftDoubleComplex* u)
{
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	while (ind < *dNe) {
		u[ind].x /= *dsNe;
		u[ind].y /= *dsNe;
		ind+= blockDim.x * gridDim.x;
	}
};

__global__
void check_nan(int *dNe, cufftDoubleComplex *u, bool *danynan)
{
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	while (ind < *dNe) {
		if (isnan(u[ind].x) || isnan(u[ind].y)) {
			*danynan = true;
		}
		ind+= blockDim.x * gridDim.x;
	}
};

__global__
void set_wave_indicies( int *dNe, int *dNz, int *dNy,
						double *dkx, double* dky, double *dkz,
						double *ikx, double *iky, double *ikz)
{
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	int i,j,k;
	while (ind < *dNe) {
		k = ind % *dNz;
		j = ((ind - k)/ *dNz) % *dNy;
		i = (((ind - k)/ *dNz) - j)/ *dNy;
		ikx[ind] = dkx[i];
		iky[ind] = dky[j];
		ikz[ind] = dkz[k];
		ind+= blockDim.x * gridDim.x;
	}
};

__global__
void set_space_indicies(int *dNe, int *dNz, int *dNy,
						double *dx, double* dy, double *dz,
						double *ix, double *iy, double *iz)
{
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	int i,j,k;
	while (ind < *dNe) {
		k = ind % *dNz;
		j = ((ind - k)/ *dNz) % *dNy;
		i = (((ind - k)/ *dNz) - j)/ *dNy;
		ix[ind] = dx[i];
		iy[ind] = dy[j];
		iz[ind] = dz[k];
		ind+= blockDim.x * gridDim.x;
	}
};

__global__
void reduce_energy( int *dNe, double *dRi,
					cufftDoubleComplex *u, cufftDoubleComplex *v,
					cufftDoubleComplex *w, cufftDoubleComplex *b, double *d_energy_reduce)
{
	extern __shared__ double sdata[];
	int tid = threadIdx.x;
	int ind = threadIdx.x + blockIdx.x * blockDim.x * 2;
	if (ind < *dNe) {
		sdata[tid] = u[ind].x*u[ind].x + v[ind].x*v[ind].x + w[ind].x*w[ind].x + (b[ind].x*b[ind].x * *dRi);
		if (ind + blockDim.x < *dNe) {
			int ind2 = ind + blockDim.x;
			sdata[tid] += u[ind2].x*u[ind2].x + v[ind2].x*v[ind2].x + w[ind2].x*w[ind2].x + (b[ind2].x*b[ind2].x * *dRi);
		}
	}
	else sdata[tid] = 0.0;
	__syncthreads();

	for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
		if (tid < s) sdata[tid]+= sdata[tid + s];
		__syncthreads();
	}

	if (tid == 0)  {
		d_energy_reduce[blockIdx.x] = sdata[0];
	}
};

__global__
void reduce_kinetic(int *dNe, cufftDoubleComplex *u, cufftDoubleComplex *v,
					cufftDoubleComplex *w, double *reduce_array)
{
	extern __shared__ double sdata[];
	int tid = threadIdx.x;
	int ind = threadIdx.x + blockIdx.x * blockDim.x * 2;
	if (ind < *dNe) {
		sdata[tid] = u[ind].x * u[ind].x + v[ind].x * v[ind].x + w[ind].x * w[ind].x;
		if (ind + blockDim.x < *dNe) {
			int ind2 = ind + blockDim.x;
			sdata[tid] += u[ind2].x * u[ind2].x + v[ind2].x * v[ind2].x + w[ind2].x * w[ind2].x;
		}
	}
	else sdata[tid] = 0.0;
	__syncthreads();

	for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
		if (tid < s) sdata[tid]+= sdata[tid + s];
		__syncthreads();
	}

	if (tid == 0) {
		reduce_array[blockIdx.x] = sdata[0];
	}
};

__global__
void reduce_potential(int *dNe, cufftDoubleComplex *b, double *reduce_array)
{
	extern __shared__ double sdata[];
	int tid = threadIdx.x;
	int ind = threadIdx.x + blockIdx.x * blockDim.x * 2;
	if (ind < *dNe) {
		sdata[tid] = b[ind].x * b[ind].x;
		if (ind + blockDim.x < *dNe) {
			sdata[tid] += b[ind + blockDim.x].x * b[ind + blockDim.x].x;
		}
	}
	else sdata[tid] = 0.0;
	__syncthreads();

	for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
		if (tid < s) sdata[tid]+= sdata[tid + s];
		__syncthreads();
	}

	if (tid == 0) {
		reduce_array[blockIdx.x] = sdata[0];
	}
};

__global__
void reduce_uwUy(int *dNe, cufftDoubleComplex *u, cufftDoubleComplex *w, cufftDoubleComplex *Uy, double *reduce_array)
{
	extern __shared__ double sdata[];
	int tid = threadIdx.x;
	int ind = threadIdx.x + blockIdx.x * blockDim.x * 2;
	if (ind < *dNe) {
		sdata[tid] = u[ind].x * w[ind].x * Uy[ind].x;
		if (ind + blockDim.x < *dNe) {
			sdata[tid] += u[ind + blockDim.x].x * w[ind + blockDim.x].x * Uy[ind + blockDim.x].x;
		}
	}
	else sdata[tid] = 0.0;
	__syncthreads();

	for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
		if (tid < s) sdata[tid]+= sdata[tid + s];
		__syncthreads();
	}

	if (tid == 0) {
		reduce_array[blockIdx.x] = sdata[0];
	}
};

__global__
void reduce_wb(int *dNe, cufftDoubleComplex *w, cufftDoubleComplex *b, double *reduce_array)
{
	extern __shared__ double sdata[];
	int tid = threadIdx.x;
	int ind = threadIdx.x + blockIdx.x * blockDim.x * 2;
	if (ind < *dNe) {
		sdata[tid] = w[ind].x * b[ind].x;
		if (ind + blockDim.x < *dNe) {
			sdata[tid] += w[ind + blockDim.x].x * b[ind + blockDim.x].x;
		}
	}
	else sdata[tid] = 0.0;
	__syncthreads();

	for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
		if (tid < s) sdata[tid]+= sdata[tid + s];
		__syncthreads();
	}

	if (tid == 0) {
		reduce_array[blockIdx.x] = sdata[0];
	}
};

__global__
void normalize(int *dNe, double *dsqrtenergy, cufftDoubleComplex *u, cufftDoubleComplex *v, cufftDoubleComplex *w, cufftDoubleComplex *b)
{
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	while (ind < *dNe) {
		u[ind].x /= *dsqrtenergy;
		v[ind].x /= *dsqrtenergy;
		w[ind].x /= *dsqrtenergy;
		b[ind].x /= *dsqrtenergy;
		u[ind].y /= *dsqrtenergy;
		v[ind].y /= *dsqrtenergy;
		w[ind].y /= *dsqrtenergy;
		b[ind].y /= *dsqrtenergy;
		ind+= blockDim.x * gridDim.x;
	}
};

__global__
void refactor_b(int *dNe, double *dRi, cufftDoubleComplex *b)
{
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	while (ind < *dNe) {
		b[ind].x *= *dRi;
		b[ind].y *= *dRi;
		ind+= blockDim.x * gridDim.x;
	}
};

__global__
void reduce_residual(   int *dNe, double *dRi,
						cufftDoubleComplex *un, cufftDoubleComplex *uo,
						cufftDoubleComplex *vn, cufftDoubleComplex *vo,
						cufftDoubleComplex *wn, cufftDoubleComplex *wo,
						cufftDoubleComplex *bn, cufftDoubleComplex *bo,
						double *d_residual_reduce)
{
	extern __shared__ double sdata[];
	int tid = threadIdx.x;
	int ind = threadIdx.x + blockIdx.x * blockDim.x * 2;
	if (ind < *dNe) {
		double val = abs(un[ind].x-uo[ind].x)*abs(un[ind].x-uo[ind].x);
		val+= abs(vn[ind].x-vo[ind].x)*abs(vn[ind].x-vo[ind].x);
		val+= abs(wn[ind].x-wo[ind].x)*abs(wn[ind].x-wo[ind].x);
		val+= abs(bn[ind].x-bo[ind].x)*abs(bn[ind].x-bo[ind].x) * *dRi;
		sdata[tid] = val;
		if (ind + blockDim.x < *dNe) {
			int ind2 = ind + blockDim.x;
			val = abs(un[ind2].x-uo[ind2].x)*abs(un[ind2].x-uo[ind2].x);
			val+= abs(vn[ind2].x-vo[ind2].x)*abs(vn[ind2].x-vo[ind2].x);
			val+= abs(wn[ind2].x-wo[ind2].x)*abs(wn[ind2].x-wo[ind2].x);
			val+= abs(bn[ind2].x-bo[ind2].x)*abs(bn[ind2].x-bo[ind2].x) * *dRi;
			sdata[tid] += val;
		}
	}
	else sdata[tid] = 0.0;
	__syncthreads();

	for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
		if (tid < s) sdata[tid]+= sdata[tid + s];
		__syncthreads();
	}

	if (tid == 0) d_residual_reduce[blockIdx.x] = sdata[0];
};

__global__
void init_shear(int *dNe, double *dLy, double *iy, cufftDoubleComplex *U)
{
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	while (ind < *dNe) {
		U[ind] = (cufftDoubleComplex) make_double2(tanh(iy[ind]- *dLy*PI/2)-tanh(iy[ind]+ *dLy*PI/2)-1,0.0);
		ind+= blockDim.x * gridDim.x;
	}
};

__global__
void init_IC(int *dNe, double *ix, double *iy, double *iz,
				cufftDoubleComplex *u, cufftDoubleComplex *v, cufftDoubleComplex *w, cufftDoubleComplex *b)
{
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	while (ind < *dNe) {
		u[ind] = (cufftDoubleComplex) make_double2(2.0/sqrt(3.0)*sin(2.0*PI/3.0)*sin(ix[ind])*cos(iy[ind])*cos(iz[ind]),0.0);
		v[ind] = (cufftDoubleComplex) make_double2(2.0/sqrt(3.0)*sin(-2.0*PI/3.0)*cos(ix[ind])*sin(iy[ind])*cos(iz[ind]),0.0);
		w[ind] = (cufftDoubleComplex) make_double2(2.0/sqrt(3.0)*sin(0.0)*cos(ix[ind])*cos(iy[ind])*sin(iz[ind]),0.0);
		b[ind] = (cufftDoubleComplex) make_double2(1.0,0.0);
		ind+= blockDim.x * gridDim.x;
	}
};

__global__
void real_scale(int *dNe, double factor, cufftDoubleComplex *u)
{
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	while (ind < *dNe) {
		u[ind].x *= factor;
		u[ind].y *= factor;
		ind+= blockDim.x * gridDim.x;
	}
};


#endif //KERNELS_H
