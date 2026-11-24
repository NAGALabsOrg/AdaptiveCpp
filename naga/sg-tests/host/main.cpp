#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API
#include <boost/test/unit_test.hpp>

#include "../common/device_report.hpp"

bool init_unit_test() { return true; }

int main(int argc, char **argv) {
  sgt::print_cell_line(false);
  return boost::unit_test::unit_test_main(&init_unit_test, argc, argv);
}
