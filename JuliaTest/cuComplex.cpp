#include "cuComplex.h"

cuComplex::cuComplex(float a, float b) : r(a), i(b){}

float cuComplex::magnitude2(void)
{
	return r * r + i * i;
}

cuComplex cuComplex::operator*(const cuComplex& a)
{
	return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
}

cuComplex cuComplex::operator+(const cuComplex& a)
{
	return cuComplex(r + a.r, i + a.i);
}
