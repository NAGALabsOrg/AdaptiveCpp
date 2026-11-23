#include "../bench/bench_stats.hpp"

#include <boost/test/unit_test.hpp>

using namespace sgt::bench;

BOOST_AUTO_TEST_SUITE(bench_stats_suite)

BOOST_AUTO_TEST_CASE(median_cases) {
  BOOST_CHECK_EQUAL(median({3, 1, 2}), 2.0);
  BOOST_CHECK_EQUAL(median({4, 1, 3, 2}), 2.5);
  BOOST_CHECK_EQUAL(median({7}), 7.0);
  BOOST_CHECK_EQUAL(median({}), 0.0);
}

BOOST_AUTO_TEST_CASE(rejit_threshold) {
  BOOST_CHECK(rejit_detected({5, 6, 400}, 5));
  BOOST_CHECK(!rejit_detected({5, 6, 90}, 5));
  BOOST_CHECK(!rejit_detected({150}, 20));
  BOOST_CHECK(rejit_detected({250}, 20));
  BOOST_CHECK(!rejit_detected({}, 5));
}

BOOST_AUTO_TEST_CASE(format_line_exact) {
  BOOST_CHECK_EQUAL(format_result_line({"chain_30", true, 11043.2, 0.412, 3.87, false}),
                    "sg_jit_bench case=chain_30 ok=1 cold_ms=11043.2 warm_ms=0.412 big_ms=3.870 rejit=0");
  BOOST_CHECK_EQUAL(format_result_line({"a_to_f", false, 5.0, 1.0, std::nullopt, true}),
                    "sg_jit_bench case=a_to_f ok=0 cold_ms=5.0 warm_ms=1.000 big_ms=NA rejit=1");
  BOOST_CHECK_EQUAL(format_result_line({"a_to_f", true, 1234.56, 1.0, std::nullopt, false}),
                    "sg_jit_bench case=a_to_f ok=1 cold_ms=1234.6 warm_ms=1.000 big_ms=NA rejit=0");
}

BOOST_AUTO_TEST_CASE(expected_values) {
  BOOST_CHECK_EQUAL(expected_plain_item(4), 13);
  BOOST_CHECK_EQUAL(expected_ten_layers(0), 0);
  BOOST_CHECK_EQUAL(expected_ten_layers(3), 72);
  BOOST_CHECK_EQUAL(expected_ten_layers(15), 360);
  BOOST_CHECK_EQUAL(expected_layers_8_16_32(0), 56);
  BOOST_CHECK_EQUAL(expected_layers_8_16_32(1), 60);
  BOOST_CHECK_EQUAL(expected_layers_8_16_32(2), 64);
  BOOST_CHECK_EQUAL(expected_layers_8_16_32(3), 68);
  BOOST_CHECK_EQUAL(expected_layers_8_16_32(127), 68);
  BOOST_CHECK_EQUAL(expected_chain_30(5), 35);
  const auto v = expected_a_to_f(7);
  BOOST_CHECK_EQUAL(v.b, 8);
  BOOST_CHECK_EQUAL(v.c, 14);
  BOOST_CHECK_EQUAL(v.d, 22);
  BOOST_CHECK_EQUAL(expected_a_to_f(0).f, 1025);
  BOOST_CHECK_EQUAL(expected_a_to_f(5).f, 1019);
  BOOST_CHECK_EQUAL(expected_a_to_f(15).f, 1007);
}

BOOST_AUTO_TEST_CASE(parse_args_cases) {
  const auto d = parse_args({"--case", "chain_30"});
  BOOST_REQUIRE(d);
  BOOST_CHECK_EQUAL(d->case_name, "chain_30");
  BOOST_CHECK_EQUAL(d->warm_runs, 20);
  BOOST_CHECK_EQUAL(d->warmup_runs, 3);
  BOOST_CHECK_EQUAL(d->big_groups, 16384u);
  const auto w = parse_args({"--case", "a_to_f", "--warm-runs", "5"});
  BOOST_REQUIRE(w);
  BOOST_CHECK_EQUAL(w->warm_runs, 5);
  BOOST_CHECK(!parse_args({"--case", "nope"}));
  BOOST_CHECK(!parse_args({}));
  BOOST_CHECK(!parse_args({"--case"}));
  BOOST_CHECK(!parse_args({"--case", "a_to_f", "--bogus"}));
  BOOST_CHECK(!parse_args({"--case", "a_to_f", "--warm-runs", "x"}));
}

BOOST_AUTO_TEST_SUITE_END()
