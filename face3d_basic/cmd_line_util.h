#pragma once
#include <dlib/cmd_line_parser.h>
#include <sstream>

namespace face3d {

template<class T>
bool get_arg(dlib::command_line_parser const& parser, const char* option, T& value, bool required)
{
  if (parser.option(option)) {
    std::stringstream ss( parser.option(option).argument() );
    ss >> value;
    return true;
  }
  if (required) {
    std::cerr << "Missing required argument: " << option << std::endl;
    parser.print_options();
  }
  return false;
}

template<class T>
bool get_optional_arg(dlib::command_line_parser const& parser, const char* option, T& value)
{
  return get_arg(parser, option, value, false);
}


template<class T>
bool get_required_arg(dlib::command_line_parser const& parser, const char* option, T& value)
{
  return get_arg(parser, option, value, true);
}



}
