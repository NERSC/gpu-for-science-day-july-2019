#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

#define aqsmtemp_size number_bands* ncouls
#define aqsntemp_size number_bands* ncouls
#define I_eps_array_size ngpown* ncouls
#define achtemp_size (nend - nstart)
#define achtemp_re_size (nend - nstart)
#define achtemp_im_size (nend - nstart)
#define vcoul_size ncouls
#define inv_igp_index_size ngpown
#define indinv_size (ncouls + 1)
#define wx_array_size (nend - nstart)
#define wtilde_array_size ngpown* ncouls

#define aqsmtemp(n1, ig) aqsmtemp[n1 * ncouls + ig]
#define aqsntemp(n1, ig) aqsntemp[n1 * ncouls + ig]
#define I_eps_array(my_igp, ig) I_eps_array[my_igp * ncouls + ig]
#define wtilde_array(my_igp, ig) wtilde_array[my_igp * ncouls + ig]

// these are performance analysis counters - see https://github.com/jrmadsen/TiMemory
#include "timemory/timemory.hpp"

namespace tim
{
// shorten the namespace for tim::component::real_clock, etc.
using namespace tim::component;
}

// this object starts recording when constructed, stops recording when destructed
// when TIMEMORY_USE_CUDA is not defined, tim::auto_tuple filters out that type
// and does not record the CUDA kernel wall-clock time
using auto_timer_t =
    tim::auto_tuple<tim::real_clock, tim::system_clock, tim::user_clock, tim::cpu_util,
                    tim::current_rss, tim::peak_rss, tim::cuda_event>;

using mem_usage_t = tim::auto_tuple<tim::current_rss, tim::peak_rss>;
