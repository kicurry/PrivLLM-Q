#include "profile_helper.h"
#include <chrono>
#include <iostream>
#include <unordered_map>
#ifdef USE_HE_GPU
#include <nvtx3/nvToolsExt.h>
#endif

using namespace std;
using namespace Datatype;

unordered_map<string, long long> nvtxDurations;

void nvtxPush(const string &name, LOCATION backend)
{
    if (backend == LOCATION::DEVICE)
    {
#ifdef USE_HE_GPU
        nvtxRangePush(name.c_str());
#endif
    }
    else
    {
        auto now = chrono::high_resolution_clock::now();
        nvtxDurations[name] -= chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()).count();
    }
}

void nvtxPop(const string &name, LOCATION backend)
{
    if (backend == DEVICE)
    {
#ifdef USE_HE_GPU
        nvtxRangePop();
#endif
    }
    else
    {
        auto now = chrono::high_resolution_clock::now();
        nvtxDurations[name] += chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()).count();
    }
}

void printNVTXStats()
{
    if (!nvtxDurations.empty())
    {
        cout << "\nTime Statistics (ms):\n";
        for (const auto &[name, duration] : nvtxDurations)
        {
            cout << name << ": " << duration << endl;
        }
    }
}