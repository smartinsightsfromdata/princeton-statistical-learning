// Copyright 2010 Sean Gerrish (sgerrish@cs.princeton.edu)
//
// Provides various logging utilities.

#ifndef __UTIL_LOGGING_H__
#define __UTIL_LOGGING_H__

#include <string.h>
#include "strings.h"

using std::string;

namespace slap_utils {

void LogOnce(const string& key, const string& value);
void LogFatal(const string& value);
void Log(const string& value);

}  // namespace slap_utils

#endif  // __UTIL_LOGGING_H__
