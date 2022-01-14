// structure to represent complex numbers
struct cuComplex {
	float r;
	float i;
	__host__ __device__  cuComplex(float a, float b) : r(a), i(b) {}
	__host__ __device__  float magnitude2(void) { return r * r + i * i; }

	__host__ __device__  cuComplex operator*(const cuComplex& a) {
		return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
	}

	__host__ __device__ cuComplex operator+(const cuComplex& a) {
		return cuComplex(r + a.r, i + a.i);
	}
};