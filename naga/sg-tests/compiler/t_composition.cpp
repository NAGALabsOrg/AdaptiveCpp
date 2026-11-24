#include "hipSYCL/compiler/llvm-to-backend/CompositionStep.hpp"

#include <boost/test/unit_test.hpp>

#include <llvm/AsmParser/Parser.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

using namespace hipsycl::compiler;
using namespace hipsycl::glue::sscp;

namespace {

std::string task_name(int k) { return "sg_task_" + std::to_string(k); }

std::string source(int tasks, int skip_definition = -1, bool wrong_type = false) {
  std::string s;
  s += "define void @__sg_ph_dispatch(ptr %a, i32 %b, i64 %c, ptr %d, ptr %e) noinline {\n  unreachable\n}\n";
  for (int k = 0; k < tasks; ++k) {
    const std::string n = "@" + task_name(k);
    if (k == skip_definition) {
      s += "declare void " + n + "(i32, ptr, i32, i64, ptr, ptr)\n";
      continue;
    }
    if (wrong_type && k == 0) {
      s += "define void " + n + "(i32 %self) {\n  ret void\n}\n";
      continue;
    }
    s += "define void " + n + "(i32 %self, ptr %a, i32 %b, i64 %c, ptr %d, ptr %e) {\n"
         "entry:\n"
         "  %other = icmp ne i32 %self, " + std::to_string(k) + "\n"
         "  br i1 %other, label %out, label %run\n"
         "run:\n"
         "  %p = getelementptr i32, ptr %e, i32 " + std::to_string(k) + "\n"
         "  store i32 " + std::to_string(k + 1) + ", ptr %p\n"
         "  br label %out\n"
         "out:\n"
         "  ret void\n"
         "}\n";
  }
  s += "define void @kernel(ptr %blk, ptr %mem) {\n"
       "  call void @__sg_ph_dispatch(ptr %blk, i32 0, i64 0, ptr null, ptr %mem)\n"
       "  ret void\n"
       "}\n";
  return s;
}

std::unique_ptr<llvm::Module> parse(llvm::LLVMContext &ctx, const std::string &text) {
  llvm::SMDiagnostic err;
  auto m = llvm::parseAssemblyString(text, err, ctx);
  BOOST_REQUIRE_MESSAGE(m != nullptr, "the test module does not parse");
  return m;
}

composition_config config_of(int tasks) {
  composition_config c;
  for (int k = 0; k < tasks; ++k)
    c.tasks.push_back(task_name(k));
  return c;
}

std::size_t line_count(const llvm::Module &m) {
  std::string text;
  llvm::raw_string_ostream os(text);
  m.print(os, nullptr);
  os.flush();
  return static_cast<std::size_t>(std::count(text.begin(), text.end(), '\n'));
}

int definitions_of(const llvm::Module &m, const std::string &name) {
  int n = 0;
  for (const llvm::Function &f : m)
    if (f.getName() == name && !f.isDeclaration())
      ++n;
  return n;
}

} // namespace

BOOST_AUTO_TEST_SUITE(composition)

BOOST_AUTO_TEST_CASE(every_task_function_keeps_one_definition_for_10_and_for_30_tasks) {
  for (int n : {10, 30}) {
    llvm::LLVMContext ctx;
    auto m = parse(ctx, source(n));
    std::vector<std::string> errors;
    BOOST_REQUIRE(applyComposition(*m, config_of(n), errors));
    BOOST_CHECK(errors.empty());
    for (int k = 0; k < n; ++k)
      BOOST_CHECK_EQUAL(definitions_of(*m, task_name(k)), 1);
    BOOST_CHECK_EQUAL(definitions_of(*m, compositionDispatchName()), 1);
  }
}

BOOST_AUTO_TEST_CASE(the_composed_function_calls_each_task_in_order_with_its_own_index) {
  llvm::LLVMContext ctx;
  auto m = parse(ctx, source(5));
  std::vector<std::string> errors;
  BOOST_REQUIRE(applyComposition(*m, config_of(5), errors));
  llvm::Function *composed = m->getFunction(compositionDispatchName());
  BOOST_REQUIRE(composed);
  int k = 0;
  for (llvm::Instruction &i : composed->getEntryBlock()) {
    auto *call = llvm::dyn_cast<llvm::CallInst>(&i);
    if (!call)
      continue;
    BOOST_REQUIRE(call->getCalledFunction());
    BOOST_CHECK_EQUAL(call->getCalledFunction()->getName().str(), task_name(k));
    auto *index = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(0));
    BOOST_REQUIRE(index);
    BOOST_CHECK_EQUAL(index->getZExtValue(), static_cast<std::uint64_t>(k));
    ++k;
  }
  BOOST_CHECK_EQUAL(k, 5);
}

BOOST_AUTO_TEST_CASE(the_kernel_calls_the_composed_function_and_no_other_call_of_the_placeholder_remains) {
  llvm::LLVMContext ctx;
  auto m = parse(ctx, source(3));
  std::vector<std::string> errors;
  BOOST_REQUIRE(applyComposition(*m, config_of(3), errors));
  llvm::Function *placeholder = m->getFunction(compositionPlaceholderName());
  BOOST_REQUIRE(placeholder);
  BOOST_CHECK(placeholder->use_empty());
  llvm::Function *kernel = m->getFunction("kernel");
  bool calls_composed = false;
  for (llvm::Instruction &i : kernel->getEntryBlock())
    if (auto *call = llvm::dyn_cast<llvm::CallInst>(&i))
      calls_composed = call->getCalledFunction() && call->getCalledFunction()->getName() == compositionDispatchName();
  BOOST_CHECK(calls_composed);
}

BOOST_AUTO_TEST_CASE(the_size_of_the_module_grows_linearly_with_the_task_count) {
  llvm::LLVMContext ctx10, ctx30;
  auto m10 = parse(ctx10, source(10));
  auto m30 = parse(ctx30, source(30));
  std::vector<std::string> errors;
  BOOST_REQUIRE(applyComposition(*m10, config_of(10), errors));
  BOOST_REQUIRE(applyComposition(*m30, config_of(30), errors));
  const double ratio = static_cast<double>(line_count(*m30)) / static_cast<double>(line_count(*m10));
  BOOST_CHECK_LE(ratio, 3.0);
}

BOOST_AUTO_TEST_CASE(a_task_without_a_definition_is_reported_by_name) {
  llvm::LLVMContext ctx;
  auto m = parse(ctx, source(4, 2));
  std::vector<std::string> errors;
  BOOST_CHECK(!applyComposition(*m, config_of(4), errors));
  BOOST_REQUIRE_EQUAL(errors.size(), 1u);
  BOOST_CHECK(errors[0].find(task_name(2)) != std::string::npos);
}

BOOST_AUTO_TEST_CASE(a_task_of_another_type_is_reported_by_name) {
  llvm::LLVMContext ctx;
  auto m = parse(ctx, source(2, -1, true));
  std::vector<std::string> errors;
  BOOST_CHECK(!applyComposition(*m, config_of(2), errors));
  BOOST_REQUIRE_EQUAL(errors.size(), 1u);
  BOOST_CHECK(errors[0].find(task_name(0)) != std::string::npos);
}

BOOST_AUTO_TEST_CASE(a_module_without_the_placeholder_is_reported) {
  llvm::LLVMContext ctx;
  auto m = parse(ctx, "define void @kernel() {\n  ret void\n}\n");
  std::vector<std::string> errors;
  BOOST_CHECK(!applyComposition(*m, config_of(1), errors));
  BOOST_REQUIRE_EQUAL(errors.size(), 1u);
  BOOST_CHECK(errors[0].find(compositionPlaceholderName()) != std::string::npos);
}

BOOST_AUTO_TEST_CASE(declaring_the_symbols_adds_one_declaration_per_distinct_task) {
  llvm::LLVMContext ctx;
  auto m = parse(ctx, "define void @__sg_ph_dispatch(ptr %a, i32 %b, i64 %c, ptr %d, ptr %e) {\n  unreachable\n}\n");
  composition_config c;
  c.tasks = {"sg_a", "sg_b", "sg_a"};
  declareCompositionSymbols(*m, c);
  llvm::Function *a = m->getFunction("sg_a");
  llvm::Function *b = m->getFunction("sg_b");
  BOOST_REQUIRE(a && b);
  BOOST_CHECK(a->isDeclaration() && b->isDeclaration());
  BOOST_CHECK(a->getFunctionType() == compositionTaskType(m->getFunction(compositionPlaceholderName())));
  BOOST_CHECK_EQUAL(c.imported_symbols().size(), 2u);
}

BOOST_AUTO_TEST_SUITE_END()
