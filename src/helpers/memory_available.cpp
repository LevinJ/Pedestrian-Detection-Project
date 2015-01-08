#include "memory_available.hpp"

#include <sys/sysinfo.h>
#include <cassert>

/// Implementation based on
/// http://stackoverflow.com/questions/349889
/// http://linux.die.net/man/2/sysinfo
std::size_t get_memory_available()
{
    struct sysinfo info;
    const int ret = sysinfo(&info);
    assert(ret == 0);
    return info.freeram*info.mem_unit;
}

float get_memory_available_in_gigabytes()
{
    float memory_available = get_memory_available();
    const float gigabyte_in_bytes = 1024*1024*1024;
    memory_available /= gigabyte_in_bytes;
    return memory_available;
}
