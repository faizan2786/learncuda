#pragma once
#include<iostream>

// class to store 24-bit bitmap (8-bit RGB) image as a 1-D array 8-bit array
class fpBitmap {

public:
	fpBitmap(int width, int height);
	~fpBitmap();
	int get_image_size();  // return the bitmap size in bytes
	uint8_t* get_image_ptr();

	// overload [] operator to access underlying image array
	uint8_t operator [] (int i) const;    // for const objects
	uint8_t& operator [] (int i);

	uint8_t* operator()(int col, int row) const;  // returns RGB value at (x,y) location

private:
	int _dim_x, _dim_y;
	uint8_t* _image;

};