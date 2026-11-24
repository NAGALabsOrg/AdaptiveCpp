#include "../common/check.hpp"
#include "../common/graphs.hpp"

#include <vector>

using namespace sgt;

namespace {

void run_chain(int n) {
  const auto x = iota_i(64);
  const auto got = run_kernel<int, int>(x, 64, [n](auto &kh, auto xh, auto yh) { chain(kh, n, xh, yh); });
  const auto want = chain_reference(n, x);
  BOOST_REQUIRE_EQUAL(got.size(), want.size());
  for (int i = 0; i < 64; ++i)
    BOOST_REQUIRE_MESSAGE(got[i] == want[i], "chain " << n << ", element " << i << ": got " << got[i] << ", want "
                                                       << want[i]);
}

} // namespace

BOOST_AUTO_TEST_SUITE(chains)

BOOST_AUTO_TEST_CASE(chain_of_10_work_sets) { run_chain(10); }

BOOST_AUTO_TEST_CASE(chain_of_30_work_sets) { run_chain(30); }

BOOST_AUTO_TEST_SUITE_END()
