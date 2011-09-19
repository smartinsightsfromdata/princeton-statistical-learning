// Copyright 2010 Sean Gerrish (sgerrish@cs.princeton.edu)
//
// Provides various logging utilities.

#include <string>
#include "strings.h"
#include <ext/hash_set>

#include "logging.h"

using std::string;
using __gnu_cxx::hash_set;

// This hash function for strings was taken from:
// http://www.partow.net/programming/hashfunctions/
namespace __gnu_cxx {
template<> struct hash< std::string > {
  size_t operator()( const std::string& x ) const {          
    return hash< const char* >()( x.c_str() );            
  }
};   
}

namespace slap_utils {

void LogOnce(const string& key, const string& value) {
  static hash_set<string> logged;
  hash_set<string>::const_iterator it = logged.find(key);
  if (it == logged.end()) {
    logged.insert(key);
    printf("%s\n", value.c_str());
  }
}

void LogFatal(const string& value) {
  printf("%s\n", value.c_str());
  exit(1);
}

void Log(const string& value) {
  printf("%s\n", value.c_str());
}

}  // namespace slap_utils
