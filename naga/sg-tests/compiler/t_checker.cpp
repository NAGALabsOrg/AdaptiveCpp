#include "hipSYCL/compiler/sg/BodyChecker.hpp"

#include <boost/test/unit_test.hpp>

#include <clang/Basic/Diagnostic.h>
#include <clang/Frontend/ASTUnit.h>
#include <clang/Tooling/Tooling.h>

#include <string>
#include <vector>

namespace {

struct diag {
  clang::DiagnosticsEngine::Level level;
  std::string text;
  unsigned line;
};

class collector : public clang::DiagnosticConsumer {
public:
  std::vector<diag> got;
  clang::SourceManager *sm = nullptr;
  void HandleDiagnostic(clang::DiagnosticsEngine::Level l, const clang::Diagnostic &i) override {
    clang::DiagnosticConsumer::HandleDiagnostic(l, i);
    if (l != clang::DiagnosticsEngine::Error && l != clang::DiagnosticsEngine::Note)
      return;
    llvm::SmallString<256> s;
    i.FormatDiagnostic(s);
    unsigned line = 0;
    if (sm && i.getLocation().isValid())
      line = sm->getSpellingLineNumber(i.getLocation());
    got.push_back({l, std::string(s.str()), line});
  }
};

const char *prelude = R"(
template <class C> [[clang::annotate("sg_body_entry")]] void entry(const C &c) { c(); }
)";

class entry_collector : public clang::RecursiveASTVisitor<entry_collector> {
public:
  std::vector<const clang::FunctionDecl *> entries;
  bool shouldVisitTemplateInstantiations() const { return true; }
  bool VisitFunctionDecl(clang::FunctionDecl *fd) {
    if (hipsycl::compiler::sg::is_body_entry(fd))
      entries.push_back(fd);
    return true;
  }
};

std::vector<diag> run_with(const std::string &body, const std::string &decls,
                           int repeat_entries) {
  const std::string code = std::string(prelude) + decls + "\nvoid user() {\n" + body + "\n}\n";
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      code, {"-std=c++20", "-fsyntax-only", "-nostdinc", "-nostdinc++"}, "t.cpp");
  BOOST_REQUIRE(ast);
  collector c;
  c.sm = &ast->getSourceManager();
  ast->getDiagnostics().setClient(&c, false);
  entry_collector ec;
  ec.TraverseDecl(ast->getASTContext().getTranslationUnitDecl());
  std::vector<const clang::FunctionDecl *> entries;
  for (int i = 0; i < repeat_entries; ++i)
    entries.insert(entries.end(), ec.entries.begin(), ec.entries.end());
  hipsycl::compiler::sg::check_entries(ast->getASTContext(), ast->getDiagnostics(), entries);
  return c.got;
}

std::vector<diag> run(const std::string &body, const std::string &decls = "") {
  return run_with(body, decls, 1);
}

std::vector<std::string> errors(const std::vector<diag> &d) {
  std::vector<std::string> out;
  for (const auto &x : d)
    if (x.level == clang::DiagnosticsEngine::Error)
      out.push_back(x.text);
  return out;
}
std::size_t count_id(const std::vector<diag> &d, const std::string &id) {
  std::size_t n = 0;
  for (const auto &x : d)
    if (x.level == clang::DiagnosticsEngine::Error && x.text.rfind("sg: " + id + ":", 0) == 0)
      ++n;
  return n;
}
std::size_t count_notes(const std::vector<diag> &d, const std::string &prefix) {
  std::size_t n = 0;
  for (const auto &x : d)
    if (x.level == clang::DiagnosticsEngine::Note && x.text.rfind(prefix, 0) == 0)
      ++n;
  return n;
}
void expect_one(const std::string &id, const std::string &body, const std::string &decls = "") {
  const auto d = run(body, decls);
  BOOST_CHECK_MESSAGE(count_id(d, id) >= 1, "expected '" + id + "' for: " + body);
  BOOST_CHECK_MESSAGE(count_notes(d, "in the body defined here") >= 1, "no body note for " + id);
  for (const auto &e : errors(d)) {
    BOOST_CHECK(e.find("__sg") == std::string::npos);
    BOOST_CHECK(e.find("entry") == std::string::npos);
  }
}
void expect_clean(const std::string &body, const std::string &decls = "") {
  const auto d = run(body, decls);
  BOOST_CHECK_MESSAGE(errors(d).empty(), "unexpected error for: " + body + " -> " +
                                             (errors(d).empty() ? "" : errors(d)[0]));
}

} // namespace

BOOST_AUTO_TEST_SUITE(body_checker_suite)

BOOST_AUTO_TEST_CASE(loops) {
  expect_one("loop", "entry([] { for (int i = 0; i < 3; ++i) {} });");
  expect_one("loop", "entry([] { int i = 0; while (i < 3) ++i; });");
  expect_one("loop", "entry([] { int i = 0; do { ++i; } while (i < 3); });");
  expect_one("loop", "entry([] { int a[3] = {1, 2, 3}; for (int x : a) { (void)x; } });");
}
BOOST_AUTO_TEST_CASE(loop_two_calls_deep_reports_chain) {
  const auto d = run("entry([] { return outer(1); });",
                     "int inner(int x) { for (int i = 0; i < x; ++i) {} return x; }\n"
                     "int outer(int x) { return inner(x); }\n");
  BOOST_CHECK_EQUAL(count_id(d, "loop"), 1u);
  BOOST_CHECK_EQUAL(count_notes(d, "called from"), 2u);
  // innermost first: the loop is in inner, called from outer, which the body calls
  std::vector<std::string> notes;
  for (const auto &x : d)
    if (x.level == clang::DiagnosticsEngine::Note && x.text.rfind("called from", 0) == 0)
      notes.push_back(x.text);
  BOOST_REQUIRE_EQUAL(notes.size(), 2u);
  BOOST_CHECK(notes[0].find("outer") != std::string::npos);
  BOOST_CHECK(notes[1].find("the body") != std::string::npos);
}
BOOST_AUTO_TEST_CASE(goto_stmt) {
  expect_one("goto", "entry([] { goto done; done: return; });");
}
BOOST_AUTO_TEST_CASE(recursion) {
  expect_one("call-cycle", "entry([] { return fact(3); });",
             "int fact(int n) { return n ? n * fact(n - 1) : 1; }");
  expect_one("call-cycle", "entry([] { return ping(3); });",
             "int pong(int n);\nint ping(int n) { return n ? pong(n - 1) : 0; }\n"
             "int pong(int n) { return n ? ping(n - 1) : 0; }");
}
BOOST_AUTO_TEST_CASE(closures) {
  expect_one("mutable-closure", "int k = 0; entry([k]() mutable { (void)k; });");
  expect_one("ref-capture", "int k = 0; entry([&k] { (void)k; });");
  expect_one("ref-capture", "int k = 0; entry([&] { (void)k; });");
}
BOOST_AUTO_TEST_CASE(writes) {
  expect_one("non-local-write", "entry([] { gval = 1; });", "int gval;");
  expect_one("non-local-write", "int v = 0; int *p = &v; entry([p] { *p = 1; });");
  expect_one("non-local-write", "entry([] { gval += 2; });", "int gval;");
  expect_one("non-local-write", "entry([] { ++gval; });", "int gval;");
  expect_one("non-local-write", "entry([] { gval++; });", "int gval;");
}
BOOST_AUTO_TEST_CASE(static_variable) {
  expect_one("non-const-static", "entry([] { return gval; });", "int gval;");
}
BOOST_AUTO_TEST_CASE(no_body_virtual_pointer) {
  expect_one("no-body", "entry([] { return ext(1); });", "int ext(int);");
  expect_one("virtual-call", "entry([] { B b; B *p = &b; return p->f(); });",
             "struct B { virtual int f() { return 1; } };");
  expect_one("function-pointer-call", "int (*fp)(int) = nullptr; entry([fp] { return fp(1); });");
  expect_one("function-pointer-call", "std::function<int(int)> f; entry([f] { return f(1); });",
             "namespace std { template <class T> struct function; template <class R, class... A> "
             "struct function<R(A...)> { R operator()(A...) const; }; }");
}
BOOST_AUTO_TEST_CASE(two_offences_two_errors) {
  const auto d = run("entry([] { gval = 1; for (;;) {} });", "int gval;");
  BOOST_CHECK(count_id(d, "non-local-write") >= 1);
  BOOST_CHECK(count_id(d, "loop") >= 1);
  BOOST_CHECK(errors(d).size() >= 2);
}
BOOST_AUTO_TEST_CASE(exempt_and_skipped) {
  expect_clean("entry([] { sg::loopy(); });", "namespace sg { inline void loopy() { for (;;) {} } }");
  expect_clean("entry([] { return picks<false>(); });",
               "template <bool B> int picks() { if constexpr (B) { for (;;) {} } return 0; }");
  expect_clean("entry([] { static_assert(sizeof(int) == 4); return int(sizeof(decltype(0))); });");
  expect_clean("entry([] { return cexpr(); });",
               "constexpr int cexpr() { return 3; }\nconstexpr int k = [] { int s = 0; for (int i = 0; i < 3; ++i) s += i; return s; }();");
}
BOOST_AUTO_TEST_CASE(unreached_code_not_checked) {
  expect_clean("entry([] { return 1; });", "void unused() { for (;;) {} }");
}
BOOST_AUTO_TEST_CASE(no_entries_no_diagnostics) {
  const auto d = run_with("int x = 0; (void)x;", "void bad() { for (;;) {} goto l; l:; }", 1);
  BOOST_CHECK(d.empty());
  const auto d0 = run_with("entry([] { for (;;) {} });", "", 0);
  BOOST_CHECK(d0.empty());
}
BOOST_AUTO_TEST_CASE(entry_instantiated_twice_is_checked_once) {
  const auto once = run_with("entry([] { for (;;) {} });", "", 1);
  const auto twice = run_with("entry([] { for (;;) {} });", "", 2);
  BOOST_CHECK_EQUAL(count_id(once, "loop"), 1u);
  BOOST_CHECK_EQUAL(count_id(twice, "loop"), 1u);
  BOOST_CHECK_EQUAL(errors(twice).size(), errors(once).size());
}
BOOST_AUTO_TEST_CASE(non_entry_is_not_a_body_entry) {
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      "void plain() {}\ntemplate <class T> void tpl(T) {}\nvoid use() { tpl(1); }",
      {"-std=c++20", "-fsyntax-only", "-nostdinc", "-nostdinc++"}, "t.cpp");
  BOOST_REQUIRE(ast);
  entry_collector ec;
  ec.TraverseDecl(ast->getASTContext().getTranslationUnitDecl());
  BOOST_CHECK(ec.entries.empty());
}
BOOST_AUTO_TEST_CASE(local_writes_are_fine) {
  expect_clean("entry([] { int x = 0; x += 3; ++x; int a[2] = {0, 0}; a[1] = x; return a[1]; });");
}

BOOST_AUTO_TEST_SUITE_END()
