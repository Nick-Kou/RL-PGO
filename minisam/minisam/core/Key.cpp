/**
 * @file    Key.cpp
 * @brief   Index variable by interger
 * @author  Jing Dong
 * @date    Sep 6, 2018
 */

#include <minisam/core/Key.h>

#include <sstream>

namespace minisam {

/* ************************************************************************** */
// convert key to string to format "x0" for print
// only print char if char in 31 < char < 127
// if not in range, just output key itself as index
std::string keyString(Key key) {
  std::stringstream ss;
  if (keyChar(key) >= 0x20 && keyChar(key) < 0x7F) {
    ss << keyChar(key) << keyIndex(key);
  } else {
    ss << key;
  }
  return ss.str();
}

}  // namespace minisam
