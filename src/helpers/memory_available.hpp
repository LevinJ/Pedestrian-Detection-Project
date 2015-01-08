#ifndef MEMORY_AVAILABLE_HPP
#define MEMORY_AVAILABLE_HPP

#include <cstddef>

/// free system memory available in bytes
std::size_t get_memory_available();

float get_memory_available_in_gigabytes();

#endif // MEMORY_AVAILABLE_HPP
