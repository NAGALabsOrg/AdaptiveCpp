#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API
#include <boost/test/unit_test.hpp>

#include <cstdio>

bool init_unit_test() { return true; }

int main(int argc, char **argv) {
  std::printf("sg-cell backend=host platform=none device=none vendor=none\n");
  std::fflush(stdout);
  return boost::unit_test::unit_test_main(&init_unit_test, argc, argv);
}
