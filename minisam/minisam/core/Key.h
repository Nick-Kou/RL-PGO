/**
 * @file    Key.h
 * @brief   Index variable by interger
 * @author  Jing Dong
 * @date    Oct 15, 2017
 */

#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace minisam {

// Key is just size_type unsigned interger
//size_t is basically an unsigned integer type
//key is not an alias of that type
typedef size_t Key;

/**
 * formulate Key in char + uint
 */

// some const
//size of (unsigned char is 1 byte
//sizeof(size_t) or sizeof(key) = 64 bits or 8 bytes (returns in bytes)
//numeric limits max of unisgned char is 255 in ASCII shifted left by index bits
//index bits = (8 bytes-1byte) * 8 in bits
static constexpr size_t indexBits = (sizeof(Key) - sizeof(unsigned char)) * 8;
//from the 64 bits we want to allocate ther max size of the char all the way to the left and
//the rest of the (8byte-1byte)* 8 bits == 56 bits to a number like x,1 for example
// char mask moves the ones to the left (1111110000000 to remove the integer and returnthe char
// indexmask removes the char and returns the index
static constexpr Key charMask = Key(std::numeric_limits<unsigned char>::max())
                                << indexBits;
static constexpr Key indexMask = ~charMask;  // also max index

// convert char + uint form in Key
//*********************NOTE key takes form key = x1, key =x2 .etc********************************************
inline Key key(unsigned char c, size_t i) {
  // char saved first 8 bits
  // check index size
  if (i > indexMask) {
    throw std::invalid_argument("[key] index too large");
  }
  //this return line uses the 64 bits and allocates the
  //unsigned char in bit representation form on the far left and allocated the integer such as
  // "1" in (x,1) on the far right by | or statement
  //therefore result is a 64 bit with char and integer encoded in it
  return (static_cast<size_t>(c) << indexBits) | i;
}

// get char from key
inline unsigned char keyChar(Key key) {
  return static_cast<unsigned char>((key & charMask) >> indexBits);
}

// get index from key
inline size_t keyIndex(Key key) { return key & indexMask; }

// convert key to string to format "x0" for print
// only print char if char in 31 < char < 127
// if not in range, just output key itself as index
std::string keyString(Key key);

}  // namespace minisam
