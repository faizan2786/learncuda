#include "fpBitmap.h"
#include <exception>

fpBitmap::fpBitmap(int width, int height) : _dim_x{width}, _dim_y{height}
{
	_image = new unsigned char[width * height * 3];   // 24-bit bitmap
}

fpBitmap::~fpBitmap()
{
	delete _image;
}

int fpBitmap::get_image_size()
{
	return _dim_x * _dim_y * 3;
}

unsigned char* fpBitmap::get_image_ptr()
{
	return _image;
}

uint8_t fpBitmap::operator[](int i) const
{
	return _image[i];
}

uint8_t& fpBitmap::operator[](int i)
{
	return _image[i];
}

uint8_t* fpBitmap::operator()(int col, int row) const
{
	if (col >= _dim_x || row >= _dim_y)
		throw std::runtime_error("pixel indices out of bound!") ;

	auto idx = (row * _dim_x + col) * 3;  // *3 because each pixel is 3-byte type 
	uint8_t arr[] = {_image[idx], _image[idx+1], _image[idx + 2] };
	return arr;
}
