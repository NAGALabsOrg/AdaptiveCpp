#pragma once
#include "hipSYCL/sg/error.hpp"

#include <boost/test/unit_test.hpp>

#define SG_CHECK_ERRC(expr, errc_name)                                                             \
  BOOST_CHECK_EXCEPTION(expr, sg::error,                                                           \
                        [](const sg::error &e) { return e.code() == sg::errc::errc_name; })
