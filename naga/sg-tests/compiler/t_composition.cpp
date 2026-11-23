#include "hipSYCL/compiler/llvm-to-backend/CompositionStep.hpp"

#include <boost/test/unit_test.hpp>

#include <llvm/AsmParser/Parser.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/SourceMgr.h>

#include <map>
#include <set>
#include <string>

using namespace hipsycl::compiler;
using namespace hipsycl::glue::sscp;

namespace {

struct ph_sig {
  const char *name;
  const char *params;
};
const ph_sig placeholders[] = {
    {"__sg_ph_positions", "ptr, i32, i32, i64, i32, ptr"},
    {"__sg_ph_item_value", "ptr, i32, i32, i64, i64, ptr, ptr"},
    {"__sg_ph_prep", "ptr, i32, i32, i64, ptr, ptr"},
    {"__sg_ph_run_source", "ptr, i32, i32, i64, ptr, ptr, ptr"},
    {"__sg_ph_resolve_tree", "ptr, i32, i32, i64, ptr, ptr"},
    {"__sg_ph_global_resolve", "ptr, i32, i32, i64, ptr, ptr"},
    {"__sg_ph_body", "ptr, i32, i64, i64, ptr, ptr"},
    {"__sg_ph_group_body", "ptr, i32, ptr, i64, ptr, ptr"},
    {"__sg_ph_write", "ptr, i32, ptr, ptr"},
    {"__sg_ph_run_layer", "ptr, i32, i64, ptr, ptr"},
};

std::string placeholder_defs(const std::string &skip = "") {
  std::string s;
  for (const auto &p : placeholders) {
    if (skip == p.name) {
      s += std::string("declare void @") + p.name + "(" + p.params + ")\n";
      continue;
    }
    std::string params = p.params, named;
    int i = 0;
    std::string cur;
    for (char c : params + ",") {
      if (c == ',') {
        named += (named.empty() ? "" : ", ") + cur + " %a" + std::to_string(i++);
        cur.clear();
      } else if (!(c == ' ' && cur.empty())) {
        cur += c;
      }
    }
    s += std::string("define void @") + p.name + "(" + named + ") noinline {\n  unreachable\n}\n";
  }
  return s;
}

const char *common_args = "ptr %c, i32 %n, i32 %l, i64 %g, ptr %nd, ptr %f";

std::string drivers(const std::string &skip = "") {
  std::string s;
  auto def = [&](const char *name, const std::string &body, bool out = false) {
    if (skip == name) {
      s += std::string("declare void @") + name +
           "(ptr, i32, i32, i64, ptr, ptr" + (out ? ", ptr" : "") + ")\n";
      return;
    }
    s += std::string("define void @") + name + "(" + common_args + (out ? ", ptr %out" : "") +
         ") noinline {\n" + body + "  ret void\n}\n";
  };
  def("__sg_drv_owner",
      "  call void @__sg_drv_compute(ptr %c, i32 %n, i32 %l, i64 %g, ptr %nd, ptr %f)\n");
  def("__sg_drv_compute",
      "  call void @__sg_drv_resolve_tree(ptr %c, i32 %n, i32 %l, i64 %g, ptr %nd, ptr %f)\n"
      "  call void @__sg_drv_prep(ptr %c, i32 %n, i32 %l, i64 %g, ptr %nd, ptr %f)\n"
      "  call void @__sg_drv_run_group(ptr %c, i32 %n, i32 %l, i64 %g, ptr %nd, ptr %f, ptr null)\n");
  def("__sg_drv_resolve_layer",
      "  call void @__sg_drv_compute(ptr %c, i32 %n, i32 %l, i64 %g, ptr %nd, ptr %f)\n");
  def("__sg_drv_resolve_tree",
      "  call void @__sg_ph_positions(ptr %c, i32 %n, i32 0, i64 %g, i32 0, ptr null)\n"
      "  call void @__sg_ph_resolve_tree(ptr %c, i32 %n, i32 0, i64 %g, ptr %nd, ptr %f)\n"
      "  call void @__sg_ph_global_resolve(ptr %c, i32 %n, i32 0, i64 %g, ptr %nd, ptr %f)\n");
  def("__sg_drv_prep",
      "  call void @__sg_ph_prep(ptr %c, i32 %n, i32 0, i64 %g, ptr %nd, ptr %f)\n"
      "  call void @__sg_ph_run_source(ptr %c, i32 %n, i32 0, i64 %g, ptr %nd, ptr %f, ptr null)\n");
  def("__sg_drv_run_group",
      "  call void @__sg_ph_body(ptr %c, i32 %n, i64 %g, i64 0, ptr %f, ptr null)\n"
      "  call void @__sg_ph_group_body(ptr %c, i32 %n, ptr %nd, i64 %g, ptr %f, ptr %out)\n"
      "  call void @__sg_ph_write(ptr %c, i32 %n, ptr %out, ptr null)\n",
      true);
  return s;
}

std::string entries(bool with_helper = true) {
  std::string s;
  s += "define void @helper(ptr %c) noinline {\n"
       "  call void @__sg_ph_item_value(ptr %c, i32 0, i32 0, i64 0, i64 0, ptr null, ptr null)\n"
       "  ret void\n}\n";
  for (const char *b : {"body_a", "body_b"})
    s += std::string("define void @") + b +
         "(ptr %c, i32 %n, i64 %g, i64 %l, ptr %f, ptr %d) {\n" +
         (with_helper ? "  call void @helper(ptr %c)\n" : "") + "  ret void\n}\n";
  for (const char *b : {"store_a", "store_b", "combine_a"})
    s += std::string("define void @") + b + "(ptr %c, i32 %n, ptr %d, ptr %s) {\n  ret void\n}\n";
  s += "define void @pos_a(ptr %c, i32 %n, i32 %l, i64 %g, i32 %s, ptr %o) {\n  ret void\n}\n";
  s += "define void @pos_b(ptr %c, i32 %n, i32 %l, i64 %g, i32 %s, ptr %o) {\n  ret void\n}\n";
  return s;
}

std::string kernel() {
  return "define void @kernel(ptr %c, i64 %g, ptr %nd, ptr %f) {\n"
         "  call void @__sg_ph_run_layer(ptr %c, i32 0, i64 %g, ptr %nd, ptr %f)\n  ret void\n}\n";
}

std::string module_text(const std::string &skip_ph = "", const std::string &skip_drv = "",
                        bool with_entries = true) {
  return placeholder_defs(skip_ph) + drivers(skip_drv) + (with_entries ? entries() : "") + kernel();
}

std::unique_ptr<llvm::Module> parse(llvm::LLVMContext &ctx, const std::string &text) {
  llvm::SMDiagnostic err;
  auto m = llvm::parseAssemblyString(text, err, ctx);
  BOOST_REQUIRE_MESSAGE(m, "IR parse failed: " + err.getMessage().str());
  return m;
}

std::set<std::string> callees(llvm::Module &m, const std::string &fn) {
  std::set<std::string> out;
  llvm::Function *f = m.getFunction(fn);
  if (!f)
    return out;
  for (auto &bb : *f)
    for (auto &i : bb)
      if (auto *cb = llvm::dyn_cast<llvm::CallBase>(&i))
        if (auto *c = cb->getCalledFunction())
          out.insert(c->getName().str());
  return out;
}

// 1: every call from fn to callee carries the call-site noinline attribute, 0: some does not,
// -1: there is no such call.
int call_mark(llvm::Module &m, const std::string &fn, const std::string &callee) {
  llvm::Function *f = m.getFunction(fn);
  if (!f)
    return -1;
  int found = 0, marked = 0;
  for (auto &bb : *f)
    for (auto &i : bb)
      if (auto *cb = llvm::dyn_cast<llvm::CallBase>(&i))
        if (auto *c = cb->getCalledFunction())
          if (c->getName() == callee) {
            ++found;
            if (cb->getAttributes().hasFnAttr(llvm::Attribute::NoInline))
              ++marked;
          }
  if (!found)
    return -1;
  return found == marked ? 1 : 0;
}

bool any_call_marked(llvm::Module &m) {
  for (auto &f : m)
    for (auto &bb : f)
      for (auto &i : bb)
        if (auto *cb = llvm::dyn_cast<llvm::CallBase>(&i))
          if (cb->getAttributes().hasFnAttr(llvm::Attribute::NoInline))
            return true;
  return false;
}

std::map<std::string, std::set<std::string>> call_graph(llvm::Module &m) {
  std::map<std::string, std::set<std::string>> out;
  for (auto &f : m)
    out[f.getName().str()] = callees(m, f.getName().str());
  return out;
}

bool apply(llvm::Module &m, const composition_config &c, std::vector<std::string> &errors) {
  const bool ok = applyComposition(m, c, errors);
  std::string all;
  for (const auto &e : errors)
    all += "\n  " + e;
  BOOST_CHECK_MESSAGE(ok, "applyComposition failed:" + all);
  return ok;
}

composition_config chain() {
  // layer 0: node 0 (body_a) with an item link to node 1 (body_b)
  composition_config c;
  c.unique_hash = 1;
  composition_node a, b;
  a.body_symbol = "body_a";
  a.write_symbol = "store_a";
  a.links.push_back({0, 1, "pos_a"});
  b.body_symbol = "body_b";
  b.write_symbol = "store_b";
  c.nodes = {a, b};
  c.layers = {{0}};
  return c;
}

composition_config two_layers() {
  // layer 0 = node 0 (body_b); layer 1 = node 1 (body_a) with a global link to layer 0
  composition_config c;
  c.unique_hash = 2;
  composition_node b, a;
  b.body_symbol = "body_b";
  b.write_symbol = "store_b";
  a.body_symbol = "body_a";
  a.write_symbol = "combine_a";
  a.links.push_back({2, 0, "pos_a"});
  c.nodes = {b, a};
  c.layers = {{0}, {1}};
  return c;
}

bool has(llvm::Module &m, const std::string &n) { return m.getFunction(n) != nullptr; }
bool contains(const std::set<std::string> &s, const std::string &n) { return s.count(n) != 0; }

bool calls_with_node_and_layer(llvm::Module &m, const std::string &fn, const std::string &callee,
                               std::uint64_t node, std::uint64_t layer) {
  llvm::Function *f = m.getFunction(fn);
  if (!f)
    return false;
  for (auto &bb : *f)
    for (auto &i : bb)
      if (auto *cb = llvm::dyn_cast<llvm::CallBase>(&i))
        if (auto *c = cb->getCalledFunction())
          if (c->getName() == callee && cb->arg_size() > 2) {
            auto *n = llvm::dyn_cast<llvm::ConstantInt>(cb->getArgOperand(1));
            auto *l = llvm::dyn_cast<llvm::ConstantInt>(cb->getArgOperand(2));
            if (n && l && n->getZExtValue() == node && l->getZExtValue() == layer)
              return true;
          }
  return false;
}

bool any_ph_call_outside_originals(llvm::Module &m) {
  for (auto &f : m) {
    if (f.isDeclaration())
      continue;
    const std::string name = f.getName().str();
    if (name.rfind("__sg_ph_", 0) == 0 || name.rfind("__sg_drv_", 0) == 0 && name.find('.') == std::string::npos ||
        name == "helper" || name == "body_a" || name == "body_b")
      continue;
    for (auto &bb : f)
      for (auto &i : bb)
        if (auto *cb = llvm::dyn_cast<llvm::CallBase>(&i))
          if (auto *c = cb->getCalledFunction())
            if (c->getName().starts_with("__sg_ph_") && name != "kernel")
              return true;
  }
  return false;
}

} // namespace

BOOST_AUTO_TEST_SUITE(composition_suite)

BOOST_AUTO_TEST_CASE(clones_exist_and_module_is_consistent) {
  llvm::LLVMContext ctx;
  auto m = parse(ctx, module_text());
  std::vector<std::string> errors;
  BOOST_REQUIRE(apply(*m, chain(), errors));
  BOOST_CHECK(errors.empty());
  for (const char *n : {"__sg_drv_owner.l0", "__sg_drv_compute.l0", "__sg_drv_resolve_layer.l0",
                        "__sg_drv_resolve_tree.n0", "__sg_drv_prep.n0", "__sg_drv_run_group.n0",
                        "__sg_drv_resolve_tree.n1", "__sg_drv_prep.n1", "__sg_drv_run_group.n1",
                        "body_a.n0", "body_b.n1", "helper.n0", "helper.n1"})
    BOOST_CHECK_MESSAGE(has(*m, n), std::string("missing ") + n);
  BOOST_CHECK(!any_ph_call_outside_originals(*m));
}

BOOST_AUTO_TEST_CASE(switches) {
  llvm::LLVMContext ctx;
  auto m = parse(ctx, module_text());
  std::vector<std::string> errors;
  BOOST_REQUIRE(apply(*m, chain(), errors));
  BOOST_CHECK(contains(callees(*m, "__sg_sw_positions.n0"), "pos_a"));
  BOOST_CHECK(contains(callees(*m, "__sg_sw_item_value.n0"), "body_b.n1"));
  BOOST_CHECK(contains(callees(*m, "__sg_sw_prep.n0"), "__sg_drv_prep.n1"));
  BOOST_CHECK(contains(callees(*m, "__sg_sw_resolve_tree.n0"), "__sg_drv_resolve_tree.n1"));
  BOOST_CHECK(contains(callees(*m, "__sg_drv_run_group.n0"), "body_a.n0"));
  BOOST_CHECK(contains(callees(*m, "__sg_drv_run_group.n0"), "store_a"));
  BOOST_CHECK(contains(callees(*m, "__sg_drv_run_group.n1"), "store_b"));
  BOOST_CHECK(contains(callees(*m, "__sg_sw_run_layer"), "__sg_drv_owner.l0"));
  BOOST_CHECK(contains(callees(*m, "kernel"), "__sg_sw_run_layer"));
  BOOST_CHECK(contains(callees(*m, "__sg_drv_compute.l0"), "__sg_drv_resolve_tree.n0"));
  BOOST_CHECK(contains(callees(*m, "__sg_drv_owner.l0"), "__sg_drv_compute.l0"));
}

BOOST_AUTO_TEST_CASE(local_link_run_source_switch) {
  llvm::LLVMContext ctx;
  auto m = parse(ctx, module_text());
  composition_config c = chain();
  c.nodes[0].links[0].kind = 1;
  std::vector<std::string> errors;
  BOOST_REQUIRE(apply(*m, c, errors));
  BOOST_REQUIRE(has(*m, "__sg_sw_run_source.n0"));
  BOOST_CHECK(calls_with_node_and_layer(*m, "__sg_sw_run_source.n0", "__sg_drv_run_group.n1", 1, 0xffffffffu));
  BOOST_CHECK(calls_with_node_and_layer(*m, "__sg_sw_prep.n0", "__sg_drv_prep.n1", 1, 0xffffffffu));
  BOOST_CHECK(contains(callees(*m, "__sg_drv_prep.n0"), "__sg_sw_run_source.n0"));
}

BOOST_AUTO_TEST_CASE(global_links_and_combine_write) {
  llvm::LLVMContext ctx;
  auto m = parse(ctx, module_text());
  std::vector<std::string> errors;
  BOOST_REQUIRE(apply(*m, two_layers(), errors));
  BOOST_CHECK(contains(callees(*m, "__sg_sw_global_resolve.n1"), "__sg_drv_resolve_layer.l0"));
  BOOST_CHECK(contains(callees(*m, "__sg_drv_run_group.n1"), "combine_a"));
  BOOST_CHECK(contains(callees(*m, "__sg_sw_run_layer"), "__sg_drv_owner.l0"));
  BOOST_CHECK(contains(callees(*m, "__sg_sw_run_layer"), "__sg_drv_owner.l1"));
  BOOST_CHECK(has(*m, "__sg_drv_resolve_layer.l0"));
}

BOOST_AUTO_TEST_CASE(same_body_two_nodes_two_clones) {
  llvm::LLVMContext ctx;
  auto m = parse(ctx, module_text());
  composition_config c = chain();
  c.nodes[1].body_symbol = "body_a";
  std::vector<std::string> errors;
  BOOST_REQUIRE(apply(*m, c, errors));
  BOOST_CHECK(has(*m, "body_a.n0"));
  BOOST_CHECK(has(*m, "body_a.n1"));
  BOOST_CHECK_EQUAL(c.imported_symbols().size(), 4u);
}

BOOST_AUTO_TEST_CASE(body_kind_selects_the_direct_call) {
  llvm::LLVMContext ctx;
  auto m = parse(ctx, module_text() +
                          "define void @gbody(ptr %c, i32 %n, ptr %nd, i64 %g, ptr %f, ptr %o) {\n  ret void\n}\n");
  composition_config c;
  c.unique_hash = 3;
  composition_node item, group;
  item.body_symbol = "body_a";
  item.write_symbol = "store_a";
  item.body_kind = 0;
  group.body_symbol = "gbody";
  group.write_symbol = "store_b";
  group.body_kind = 1;
  c.nodes = {item, group};
  c.layers = {{0}, {1}};
  std::vector<std::string> errors;
  BOOST_REQUIRE(apply(*m, c, errors));
  const std::string stub_body = "__sg_unreachable___sg_ph_body";
  const std::string stub_group = "__sg_unreachable___sg_ph_group_body";
  const auto item_calls = callees(*m, "__sg_drv_run_group.n0");
  BOOST_CHECK(contains(item_calls, "body_a.n0"));
  BOOST_CHECK(contains(item_calls, stub_group));
  BOOST_CHECK(!contains(item_calls, stub_body));
  const auto group_calls = callees(*m, "__sg_drv_run_group.n1");
  BOOST_CHECK(contains(group_calls, "gbody.n1"));
  BOOST_CHECK(contains(group_calls, stub_body));
  BOOST_CHECK(!contains(group_calls, stub_group));
  llvm::Function *sb = m->getFunction(stub_body);
  llvm::Function *sg = m->getFunction(stub_group);
  BOOST_REQUIRE(sb && sg);
  BOOST_CHECK(sb->getFunctionType() == m->getFunction("__sg_ph_body")->getFunctionType());
  BOOST_CHECK(sg->getFunctionType() == m->getFunction("__sg_ph_group_body")->getFunctionType());
  BOOST_REQUIRE(!sb->isDeclaration() && !sg->isDeclaration());
  BOOST_CHECK(llvm::isa<llvm::UnreachableInst>(sb->getEntryBlock().front()));
  BOOST_CHECK(llvm::isa<llvm::UnreachableInst>(sg->getEntryBlock().front()));
  BOOST_CHECK(!llvm::verifyModule(*m));
}

BOOST_AUTO_TEST_CASE(failures) {
  std::vector<std::string> errors;
  {
    llvm::LLVMContext ctx;
    auto m = parse(ctx, module_text("", "__sg_drv_prep"));
    BOOST_CHECK(!applyComposition(*m, chain(), errors));
    BOOST_REQUIRE(!errors.empty());
    BOOST_CHECK(errors[0].find("missing __sg_drv_prep") != std::string::npos);
  }
  errors.clear();
  {
    llvm::LLVMContext ctx;
    auto m = parse(ctx, module_text("__sg_ph_write"));
    BOOST_CHECK(!applyComposition(*m, chain(), errors));
    BOOST_CHECK(!errors.empty());
  }
  errors.clear();
  {
    llvm::LLVMContext ctx;
    auto m = parse(ctx, module_text());
    composition_config c = chain();
    c.nodes[1].body_symbol = "not_defined";
    BOOST_CHECK(!applyComposition(*m, c, errors));
    BOOST_CHECK(!errors.empty());
  }
  errors.clear();
  {
    llvm::LLVMContext ctx;
    auto m = parse(ctx, module_text() +
                            "define void @stray(ptr %c) {\n  call void @__sg_ph_write(ptr %c, i32 0, ptr null, ptr null)\n  ret void\n}\n"
                            "define void @body_c(ptr %c, i32 %n, i64 %g, i64 %l, ptr %f, ptr %d) {\n  call void @stray(ptr %c)\n  ret void\n}\n");
    composition_config c = chain();
    c.nodes[1].body_symbol = "body_c";
    // stray is cloned and rewritten; the composition itself succeeds
    BOOST_CHECK(apply(*m, c, errors));
  }
  errors.clear();
  {
    llvm::LLVMContext ctx;
    auto m = parse(ctx, module_text() +
                            "define void @loose(ptr %c) {\n  call void @__sg_ph_run_layer(ptr %c, i32 0, i64 0, ptr null, ptr null)\n  ret void\n}\n");
    // run_layer outside a clone is rewritten to the layer switch
    BOOST_CHECK(apply(*m, chain(), errors));
    BOOST_CHECK(contains(callees(*m, "loose"), "__sg_sw_run_layer"));
  }
}

BOOST_AUTO_TEST_CASE(declare_symbols) {
  llvm::LLVMContext ctx;
  auto m = parse(ctx, placeholder_defs() + "define void @store_a(ptr %c, i32 %n, ptr %d, ptr %s) {\n  ret void\n}\n");
  composition_config c = chain();
  declareCompositionSymbols(*m, c);
  llvm::Function *body = m->getFunction("body_a");
  llvm::Function *pos = m->getFunction("pos_a");
  llvm::Function *store = m->getFunction("store_a");
  BOOST_REQUIRE(body && pos && store);
  BOOST_CHECK(body->isDeclaration());
  BOOST_CHECK(pos->isDeclaration());
  BOOST_CHECK(!store->isDeclaration());
  BOOST_CHECK(body->getFunctionType() == m->getFunction("__sg_ph_body")->getFunctionType());
  BOOST_CHECK(pos->getFunctionType() == m->getFunction("__sg_ph_positions")->getFunctionType());
  const auto count = m->size();
  declareCompositionSymbols(*m, c);
  BOOST_CHECK_EQUAL(m->size(), count);
}

BOOST_AUTO_TEST_CASE(declare_without_placeholder_declares_nothing) {
  llvm::LLVMContext ctx;
  auto m = parse(ctx, "define void @x() {\n  ret void\n}\n");
  declareCompositionSymbols(*m, chain());
  BOOST_CHECK(!has(*m, "body_a"));
}

BOOST_AUTO_TEST_CASE(options_for_target) {
  BOOST_CHECK(composition_options::for_target(composition_target::ptx).keep_layer_calls);
  BOOST_CHECK(composition_options::for_target(composition_target::spirv).keep_layer_calls);
  BOOST_CHECK(!composition_options::for_target(composition_target::host).keep_layer_calls);
  BOOST_CHECK(!composition_options::for_target(composition_target::other).keep_layer_calls);
  BOOST_CHECK(!composition_options{}.keep_layer_calls);
}

BOOST_AUTO_TEST_CASE(layer_calls_marked_for_gpu_target) {
  llvm::LLVMContext ctx;
  auto m = parse(ctx, module_text());
  std::vector<std::string> errors;
  BOOST_REQUIRE(applyComposition(*m, two_layers(), errors,
                                 composition_options::for_target(composition_target::spirv)));
  BOOST_CHECK_EQUAL(call_mark(*m, "__sg_sw_run_layer", "__sg_drv_owner.l0"), 1);
  BOOST_CHECK_EQUAL(call_mark(*m, "__sg_sw_run_layer", "__sg_drv_owner.l1"), 1);
  BOOST_CHECK_EQUAL(call_mark(*m, "__sg_sw_global_resolve.n1", "__sg_drv_resolve_layer.l0"), 1);
  BOOST_CHECK_EQUAL(call_mark(*m, "__sg_drv_owner.l0", "__sg_drv_compute.l0"), 1);
  BOOST_CHECK_EQUAL(call_mark(*m, "__sg_drv_resolve_layer.l0", "__sg_drv_compute.l0"), 1);
}

BOOST_AUTO_TEST_CASE(non_layer_calls_unmarked_for_gpu_target) {
  llvm::LLVMContext ctx;
  auto m = parse(ctx, module_text());
  std::vector<std::string> errors;
  BOOST_REQUIRE(applyComposition(*m, two_layers(), errors,
                                 composition_options::for_target(composition_target::spirv)));
  BOOST_CHECK_EQUAL(call_mark(*m, "__sg_drv_compute.l0", "__sg_drv_resolve_tree.n0"), 0);
  BOOST_CHECK_EQUAL(call_mark(*m, "__sg_drv_compute.l0", "__sg_drv_prep.n0"), 0);
  BOOST_CHECK_EQUAL(call_mark(*m, "__sg_drv_compute.l0", "__sg_drv_run_group.n0"), 0);
  BOOST_CHECK_EQUAL(call_mark(*m, "__sg_drv_run_group.n1", "body_a.n1"), 0);
  BOOST_CHECK_EQUAL(call_mark(*m, "__sg_drv_run_group.n1", "combine_a"), 0);
  BOOST_CHECK_EQUAL(call_mark(*m, "__sg_sw_positions.n1", "pos_a"), 0);
  BOOST_CHECK_EQUAL(call_mark(*m, "__sg_drv_resolve_tree.n1", "__sg_sw_global_resolve.n1"), 0);
  BOOST_CHECK_EQUAL(call_mark(*m, "kernel", "__sg_sw_run_layer"), 0);
}

BOOST_AUTO_TEST_CASE(no_marks_by_default_and_for_host) {
  for (const composition_config &c : {chain(), two_layers()}) {
    llvm::LLVMContext cs;
    auto spirv = parse(cs, module_text());
    std::vector<std::string> errors;
    BOOST_REQUIRE(applyComposition(*spirv, c, errors,
                                   composition_options::for_target(composition_target::spirv)));
    const auto reference = call_graph(*spirv);
    for (int variant = 0; variant < 2; ++variant) {
      llvm::LLVMContext ctx;
      auto m = parse(ctx, module_text());
      errors.clear();
      if (variant == 0)
        BOOST_REQUIRE(applyComposition(*m, c, errors));
      else
        BOOST_REQUIRE(applyComposition(*m, c, errors,
                                       composition_options::for_target(composition_target::host)));
      BOOST_CHECK(!any_call_marked(*m));
      BOOST_CHECK(call_graph(*m) == reference);
    }
  }
}

BOOST_AUTO_TEST_CASE(kept_calls_module_verifies) {
  llvm::LLVMContext ctx;
  auto m = parse(ctx, module_text());
  std::vector<std::string> errors;
  BOOST_CHECK(applyComposition(*m, two_layers(), errors,
                               composition_options::for_target(composition_target::ptx)));
  BOOST_CHECK(errors.empty());
  BOOST_CHECK(!llvm::verifyModule(*m));
}

BOOST_AUTO_TEST_CASE(kept_calls_failures_unchanged) {
  const auto spirv = composition_options::for_target(composition_target::spirv);
  {
    llvm::LLVMContext ctx;
    auto m = parse(ctx, module_text("", "__sg_drv_prep"));
    std::vector<std::string> errors;
    BOOST_CHECK(!applyComposition(*m, chain(), errors, spirv));
    BOOST_REQUIRE(!errors.empty());
    BOOST_CHECK(errors[0].find("missing __sg_drv_prep") != std::string::npos);
  }
  {
    llvm::LLVMContext ctx;
    auto m = parse(ctx, module_text("__sg_ph_write"));
    std::vector<std::string> errors;
    BOOST_CHECK(!applyComposition(*m, chain(), errors, spirv));
    BOOST_CHECK(!errors.empty());
  }
  {
    llvm::LLVMContext ctx;
    auto m = parse(ctx, module_text());
    composition_config c = chain();
    c.nodes[1].body_symbol = "not_defined";
    std::vector<std::string> errors;
    BOOST_CHECK(!applyComposition(*m, c, errors, spirv));
    BOOST_CHECK(!errors.empty());
  }
}

BOOST_AUTO_TEST_CASE(release_swaps_hint_when_call_kept) {
  llvm::LLVMContext ctx;
  auto m = parse(ctx, "define void @callee() alwaysinline {\n  ret void\n}\n"
                      "define void @caller() {\n  call void @callee() #0\n  ret void\n}\n"
                      "attributes #0 = { noinline }\n");
  BOOST_CHECK_EQUAL(releaseKeptLayerCalls(*m), 1u);
  llvm::Function *f = m->getFunction("callee");
  BOOST_CHECK(f->hasFnAttribute(llvm::Attribute::NoInline));
  BOOST_CHECK(!f->hasFnAttribute(llvm::Attribute::AlwaysInline));
}

BOOST_AUTO_TEST_CASE(release_leaves_inlinable_callee) {
  llvm::LLVMContext ctx;
  auto m = parse(ctx, "define void @callee() alwaysinline {\n  ret void\n}\n"
                      "define void @caller() {\n  call void @callee()\n  ret void\n}\n");
  BOOST_CHECK_EQUAL(releaseKeptLayerCalls(*m), 0u);
  llvm::Function *f = m->getFunction("callee");
  BOOST_CHECK(f->hasFnAttribute(llvm::Attribute::AlwaysInline));
  BOOST_CHECK(!f->hasFnAttribute(llvm::Attribute::NoInline));
}

BOOST_AUTO_TEST_CASE(release_handles_callee_with_both_attributes) {
  llvm::LLVMContext ctx;
  auto m = parse(ctx, "define void @callee() {\n  ret void\n}\n"
                      "define void @caller() {\n  call void @callee() #0\n  ret void\n}\n"
                      "attributes #0 = { noinline }\n");
  llvm::Function *f = m->getFunction("callee");
  f->addFnAttr(llvm::Attribute::AlwaysInline);
  f->addFnAttr(llvm::Attribute::NoInline);
  BOOST_CHECK_EQUAL(releaseKeptLayerCalls(*m), 1u);
  BOOST_CHECK(f->hasFnAttribute(llvm::Attribute::NoInline));
  BOOST_CHECK(!f->hasFnAttribute(llvm::Attribute::AlwaysInline));
}

BOOST_AUTO_TEST_CASE(release_is_idempotent) {
  llvm::LLVMContext ctx;
  auto m = parse(ctx, "define void @callee() alwaysinline {\n  ret void\n}\n"
                      "define void @caller() {\n  call void @callee() #0\n  ret void\n}\n"
                      "attributes #0 = { noinline }\n");
  BOOST_CHECK_EQUAL(releaseKeptLayerCalls(*m), 1u);
  BOOST_CHECK_EQUAL(releaseKeptLayerCalls(*m), 0u);
}

BOOST_AUTO_TEST_CASE(release_ignores_declarations_and_uncalled_functions) {
  llvm::LLVMContext ctx;
  auto m = parse(ctx, "declare void @external()\n"
                      "define void @uncalled() alwaysinline {\n  ret void\n}\n"
                      "define void @caller() {\n  call void @external() #0\n  ret void\n}\n"
                      "attributes #0 = { noinline }\n");
  BOOST_CHECK_EQUAL(releaseKeptLayerCalls(*m), 0u);
  BOOST_CHECK(m->getFunction("uncalled")->hasFnAttribute(llvm::Attribute::AlwaysInline));
  BOOST_CHECK(!m->getFunction("uncalled")->hasFnAttribute(llvm::Attribute::NoInline));
  BOOST_CHECK(!m->getFunction("external")->hasFnAttribute(llvm::Attribute::AlwaysInline));
}

BOOST_AUTO_TEST_SUITE_END()
