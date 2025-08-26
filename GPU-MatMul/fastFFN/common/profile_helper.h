#pragma once

#include <string>
#include "Datatype/UnifiedType.h"

void nvtxPush(const std::string &name, Datatype::LOCATION backend);

void nvtxPop(const std::string &name, Datatype::LOCATION backend);

void printNVTXStats();