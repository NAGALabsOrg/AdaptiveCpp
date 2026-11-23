/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
//
// The composition step of the sg graph toolchain: turns the generic drivers and
// the user's body entries into the code of one composed kernel, by cloning per
// layer and per node and rewriting placeholder calls into generated switches.
// Header-only so that unit tests can include it. No exceptions.
#ifndef HIPSYCL_COMPOSITION_STEP_HPP
#define HIPSYCL_COMPOSITION_STEP_HPP

#include "hipSYCL/glue/llvm-sscp/composition.hpp"

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace hipsycl {
namespace compiler {

enum class composition_target { host, ptx, spirv, other };

struct composition_options {
  bool keep_layer_calls = false;

  static composition_options for_target(composition_target T) {
    composition_options O;
    O.keep_layer_calls = (T == composition_target::ptx || T == composition_target::spirv);
    return O;
  }
};

namespace composition_detail {

inline const char *const DriverNames[] = {
    "__sg_drv_owner",        "__sg_drv_compute",   "__sg_drv_resolve_layer",
    "__sg_drv_resolve_tree", "__sg_drv_prep",      "__sg_drv_run_group"};
enum Drv { Owner, Compute, ResolveLayer, ResolveTree, Prep, RunGroup, NumDrv };

inline const char *const PlaceholderNames[] = {
    "__sg_ph_positions",  "__sg_ph_item_value",     "__sg_ph_prep",
    "__sg_ph_run_source", "__sg_ph_resolve_tree",   "__sg_ph_global_resolve",
    "__sg_ph_body",       "__sg_ph_group_body",     "__sg_ph_write",
    "__sg_ph_run_layer"};
enum Ph {
  Positions, ItemValue, PrepPh, RunSource, ResolveTreePh, GlobalResolve, Body, GroupBody,
  Write, RunLayer, NumPh
};

constexpr unsigned NoLayer = 0xffffffffu;

struct Composer {
  llvm::Module &M;
  const glue::sscp::composition_config &C;
  std::vector<std::string> &Errors;

  llvm::Function *Drv[NumDrv] = {};
  llvm::Function *PhF[NumPh] = {};

  std::map<std::tuple<llvm::Function *, char, int>, llvm::Function *> Memo;
  std::set<llvm::Function *> Originals;
  std::set<llvm::Function *> Generated;
  std::set<llvm::Function *> LayerClones;
  std::map<std::string, llvm::Function *> Switches;
  std::map<llvm::Function *, bool> NeedsClone;

  struct Ctx {
    char Kind; // 'n' node, 'l' layer
    int Id;
    int Node; // node whose placeholders are rewritten
  };

  Composer(llvm::Module &M, const glue::sscp::composition_config &C,
           std::vector<std::string> &E)
      : M(M), C(C), Errors(E) {}

  bool isPh(llvm::Function *F) const {
    for (auto *P : PhF)
      if (P == F)
        return true;
    return false;
  }
  int driverIndex(llvm::Function *F) const {
    for (int i = 0; i < NumDrv; ++i)
      if (Drv[i] == F)
        return i;
    return -1;
  }

  bool lookup() {
    bool Ok = true;
    for (int i = 0; i < NumDrv; ++i) {
      Drv[i] = M.getFunction(DriverNames[i]);
      if (!Drv[i] || Drv[i]->isDeclaration()) {
        Errors.push_back(std::string("composition: missing ") + DriverNames[i]);
        Ok = false;
      }
    }
    for (int i = 0; i < NumPh; ++i) {
      PhF[i] = M.getFunction(PlaceholderNames[i]);
      if (!PhF[i] || PhF[i]->isDeclaration()) {
        Errors.push_back(std::string("composition: missing ") + PlaceholderNames[i]);
        Ok = false;
      }
    }
    for (const auto &S : C.imported_symbols()) {
      llvm::Function *F = M.getFunction(S);
      if (!F || F->isDeclaration()) {
        Errors.push_back("composition: " + S + " is not defined");
        Ok = false;
      }
    }
    return Ok;
  }

  bool needsClone(llvm::Function *F) {
    if (F->isDeclaration() || isPh(F) || driverIndex(F) >= 0)
      return false;
    auto It = NeedsClone.find(F);
    if (It != NeedsClone.end())
      return It->second;
    NeedsClone[F] = false; // guards recursion
    bool R = false;
    for (llvm::BasicBlock &BB : *F)
      for (llvm::Instruction &I : BB)
        if (auto *CB = llvm::dyn_cast<llvm::CallBase>(&I))
          if (llvm::Function *Callee = CB->getCalledFunction())
            if (isPh(Callee) || driverIndex(Callee) >= 0 || needsClone(Callee))
              R = true;
    NeedsClone[F] = R;
    return R;
  }

  llvm::Function *clone(llvm::Function *F, const std::string &Name) {
    llvm::ValueToValueMapTy VMap;
    llvm::Function *Cl = llvm::CloneFunction(F, VMap);
    Cl->setName(Name);
    Cl->setLinkage(llvm::GlobalValue::InternalLinkage);
    Cl->setVisibility(llvm::GlobalValue::DefaultVisibility);
    Cl->setComdat(nullptr);
    Originals.insert(F);
    Generated.insert(Cl);
    return Cl;
  }

  llvm::Function *contextClone(llvm::Function *F, char Kind, int Id, int Node) {
    auto Key = std::make_tuple(F, Kind, Id);
    auto It = Memo.find(Key);
    if (It != Memo.end())
      return It->second;
    llvm::Function *Cl = clone(F, F->getName().str() + "." + Kind + std::to_string(Id));
    Memo[Key] = Cl;
    const int Idx = driverIndex(F);
    if (Kind == 'l' && (Idx == Owner || Idx == Compute || Idx == ResolveLayer))
      LayerClones.insert(Cl);
    rewriteCalls(Cl, Ctx{Kind, Id, Node});
    return Cl;
  }

  llvm::Function *driverClone(llvm::Function *D, const Ctx &X) {
    const int Idx = driverIndex(D);
    if (Idx == Owner || Idx == Compute || Idx == ResolveLayer) {
      if (X.Kind != 'l') {
        Errors.push_back("composition: " + D->getName().str() +
                         " is called outside of a layer");
        return nullptr;
      }
      return contextClone(D, 'l', X.Id, X.Node);
    }
    return contextClone(D, 'n', X.Node, X.Node);
  }

  const glue::sscp::composition_node &node(int N) const { return C.nodes[N]; }
  int rootOf(unsigned Layer) const { return static_cast<int>(C.layers[Layer].root_node); }

  llvm::Function *bodyClone(int N) {
    llvm::Function *B = M.getFunction(node(N).body_symbol);
    return contextClone(B, 'n', N, N);
  }

  struct Arg {
    int Param;      // pass-through parameter, or -1 for a constant
    uint64_t Value; // constant value
  };
  struct Case {
    uint32_t Key;
    llvm::Function *Callee;
    std::vector<Arg> Args;
  };

  llvm::Function *emitSwitch(const std::string &Name, llvm::Function *PH, unsigned Selector,
                             const std::vector<Case> &Cases) {
    auto It = Switches.find(Name);
    if (It != Switches.end())
      return It->second;
    llvm::FunctionType *FT = PH->getFunctionType();
    llvm::Function *SW =
        llvm::Function::Create(FT, llvm::GlobalValue::InternalLinkage, Name, &M);
    SW->setAttributes(PH->getAttributes());
    SW->setCallingConv(PH->getCallingConv());
    Switches[Name] = SW;
    Generated.insert(SW);
    llvm::LLVMContext &Ctx = M.getContext();
    llvm::BasicBlock *Entry = llvm::BasicBlock::Create(Ctx, "entry", SW);
    llvm::BasicBlock *Default = llvm::BasicBlock::Create(Ctx, "default", SW);
    llvm::IRBuilder<> B(Default);
    B.CreateUnreachable();
    B.SetInsertPoint(Entry);
    llvm::SwitchInst *Sw = B.CreateSwitch(SW->getArg(Selector), Default, Cases.size());
    llvm::Type *SelTy = SW->getArg(Selector)->getType();
    for (const Case &K : Cases) {
      llvm::BasicBlock *BB = llvm::BasicBlock::Create(Ctx, "case", SW);
      Sw->addCase(llvm::cast<llvm::ConstantInt>(llvm::ConstantInt::get(SelTy, K.Key)), BB);
      llvm::IRBuilder<> CB(BB);
      std::vector<llvm::Value *> Args;
      llvm::FunctionType *CFT = K.Callee->getFunctionType();
      for (size_t I = 0; I < K.Args.size(); ++I) {
        if (K.Args[I].Param >= 0)
          Args.push_back(SW->getArg(K.Args[I].Param));
        else
          Args.push_back(llvm::ConstantInt::get(CFT->getParamType(I), K.Args[I].Value));
      }
      llvm::CallInst *Call = CB.CreateCall(CFT, K.Callee, Args);
      Call->setCallingConv(K.Callee->getCallingConv());
      CB.CreateRetVoid();
    }
    return SW;
  }

  llvm::Function *unreachableStub(llvm::Function *PH) {
    const std::string Name = "__sg_unreachable_" + PH->getName().str();
    auto It = Switches.find(Name);
    if (It != Switches.end())
      return It->second;
    llvm::Function *F = llvm::Function::Create(
        PH->getFunctionType(), llvm::GlobalValue::InternalLinkage, Name, &M);
    F->setAttributes(PH->getAttributes());
    F->setCallingConv(PH->getCallingConv());
    Switches[Name] = F;
    Generated.insert(F);
    llvm::BasicBlock *BB = llvm::BasicBlock::Create(M.getContext(), "entry", F);
    llvm::IRBuilder<> B(BB);
    B.CreateUnreachable();
    return F;
  }

  static Arg P(int I) { return Arg{I, 0}; }
  static Arg K(uint64_t V) { return Arg{-1, V}; }

  llvm::Function *placeholderReplacement(int PhIdx, const Ctx &X) {
    const int N = X.Node;
    const auto &Nd = node(N);
    const std::string Suffix = ".n" + std::to_string(N);
    std::vector<Case> Cases;
    switch (PhIdx) {
    case Body:
    case GroupBody:
      if ((Nd.body_kind == 0) == (PhIdx == Body))
        return bodyClone(N);
      return unreachableStub(PhF[PhIdx]);
    case Write:
      return M.getFunction(Nd.write_symbol);
    case Positions: {
      for (size_t L = 0; L < Nd.links.size(); ++L)
        Cases.push_back({static_cast<uint32_t>(L), M.getFunction(Nd.links[L].positions_symbol),
                         {P(0), P(1), P(2), P(3), P(4), P(5)}});
      return emitSwitch("__sg_sw_positions" + Suffix, PhF[Positions], 2, Cases);
    }
    case ItemValue: {
      for (size_t L = 0; L < Nd.links.size(); ++L)
        if (Nd.links[L].kind == 0)
          Cases.push_back({static_cast<uint32_t>(L), bodyClone(Nd.links[L].source),
                           {P(0), K(Nd.links[L].source), P(3), P(4), P(5), P(6)}});
      return emitSwitch("__sg_sw_item_value" + Suffix, PhF[ItemValue], 2, Cases);
    }
    case PrepPh: {
      for (size_t L = 0; L < Nd.links.size(); ++L)
        if (Nd.links[L].kind != 2)
          Cases.push_back({static_cast<uint32_t>(L),
                           contextClone(Drv[Prep], 'n', Nd.links[L].source, Nd.links[L].source),
                           {P(0), K(Nd.links[L].source), K(NoLayer), P(3), P(4), P(5)}});
      return emitSwitch("__sg_sw_prep" + Suffix, PhF[PrepPh], 2, Cases);
    }
    case RunSource: {
      for (size_t L = 0; L < Nd.links.size(); ++L)
        if (Nd.links[L].kind == 1)
          Cases.push_back({static_cast<uint32_t>(L),
                           contextClone(Drv[RunGroup], 'n', Nd.links[L].source, Nd.links[L].source),
                           {P(0), K(Nd.links[L].source), K(NoLayer), P(3), P(4), P(5), P(6)}});
      return emitSwitch("__sg_sw_run_source" + Suffix, PhF[RunSource], 2, Cases);
    }
    case ResolveTreePh: {
      for (size_t L = 0; L < Nd.links.size(); ++L)
        if (Nd.links[L].kind != 2)
          Cases.push_back({static_cast<uint32_t>(L),
                           contextClone(Drv[ResolveTree], 'n', Nd.links[L].source,
                                        Nd.links[L].source),
                           {P(0), K(Nd.links[L].source), K(NoLayer), P(3), P(4), P(5)}});
      return emitSwitch("__sg_sw_resolve_tree" + Suffix, PhF[ResolveTreePh], 2, Cases);
    }
    case GlobalResolve: {
      for (size_t L = 0; L < Nd.links.size(); ++L)
        if (Nd.links[L].kind == 2) {
          const unsigned S = Nd.links[L].source;
          Cases.push_back({static_cast<uint32_t>(L),
                           contextClone(Drv[ResolveLayer], 'l', S, rootOf(S)),
                           {P(0), K(rootOf(S)), K(S), P(3), P(4), P(5)}});
        }
      return emitSwitch("__sg_sw_global_resolve" + Suffix, PhF[GlobalResolve], 2, Cases);
    }
    default:
      return nullptr;
    }
  }

  void rewriteCalls(llvm::Function *Cl, const Ctx &X) {
    std::vector<llvm::CallBase *> Calls;
    for (llvm::BasicBlock &BB : *Cl)
      for (llvm::Instruction &I : BB)
        if (auto *CB = llvm::dyn_cast<llvm::CallBase>(&I))
          Calls.push_back(CB);
    for (llvm::CallBase *CB : Calls) {
      llvm::Function *Callee = CB->getCalledFunction();
      if (!Callee)
        continue;
      llvm::Function *Repl = nullptr;
      if (isPh(Callee)) {
        int Idx = 0;
        while (PhF[Idx] != Callee)
          ++Idx;
        if (Idx == RunLayer) {
          Errors.push_back("composition: run_layer is called inside " + Cl->getName().str());
          continue;
        }
        Repl = placeholderReplacement(Idx, X);
      } else if (driverIndex(Callee) >= 0) {
        Repl = driverClone(Callee, X);
      } else if (needsClone(Callee)) {
        Repl = contextClone(Callee, X.Kind, X.Id, X.Node);
      }
      if (Repl)
        CB->setCalledFunction(Repl);
    }
  }

  void run() {
    for (size_t N = 0; N < C.nodes.size(); ++N) {
      bodyClone(static_cast<int>(N));
      contextClone(Drv[ResolveTree], 'n', N, N);
      contextClone(Drv[Prep], 'n', N, N);
      contextClone(Drv[RunGroup], 'n', N, N);
    }
    for (size_t L = 0; L < C.layers.size(); ++L) {
      const int R = rootOf(L);
      contextClone(Drv[Owner], 'l', L, R);
      contextClone(Drv[Compute], 'l', L, R);
      contextClone(Drv[ResolveLayer], 'l', L, R);
    }
    // the native group entry: a switch over layers
    std::vector<Case> Cases;
    for (size_t L = 0; L < C.layers.size(); ++L)
      Cases.push_back({static_cast<uint32_t>(L), contextClone(Drv[Owner], 'l', L, rootOf(L)),
                       {P(0), K(rootOf(L)), K(L), P(2), P(3), P(4)}});
    llvm::Function *RunLayerSw = emitSwitch("__sg_sw_run_layer", PhF[RunLayer], 1, Cases);
    std::vector<llvm::CallBase *> Calls;
    for (llvm::Function &F : M) {
      if (Generated.count(&F) || F.isDeclaration())
        continue;
      for (llvm::BasicBlock &BB : F)
        for (llvm::Instruction &I : BB)
          if (auto *CB = llvm::dyn_cast<llvm::CallBase>(&I))
            if (CB->getCalledFunction() == PhF[RunLayer] && !Originals.count(&F))
              Calls.push_back(CB);
    }
    for (llvm::CallBase *CB : Calls)
      CB->setCalledFunction(RunLayerSw);
  }

  void markLayerCalls() {
    for (llvm::Function &F : M) {
      if (F.isDeclaration())
        continue;
      for (llvm::BasicBlock &BB : F)
        for (llvm::Instruction &I : BB)
          if (auto *CB = llvm::dyn_cast<llvm::CallBase>(&I))
            if (LayerClones.count(CB->getCalledFunction()))
              CB->addFnAttr(llvm::Attribute::NoInline);
    }
  }

  bool checkNoStrayPlaceholders() {
    bool Ok = true;
    for (llvm::Function &F : M) {
      if (F.isDeclaration() || Originals.count(&F) || isPh(&F))
        continue;
      for (llvm::BasicBlock &BB : F)
        for (llvm::Instruction &I : BB)
          if (auto *CB = llvm::dyn_cast<llvm::CallBase>(&I))
            if (llvm::Function *Callee = CB->getCalledFunction())
              if (isPh(Callee)) {
                Errors.push_back("composition: a call to " + Callee->getName().str() +
                                 " remains in " + F.getName().str());
                Ok = false;
              }
    }
    return Ok;
  }

  bool checkAcyclic() {
    std::map<llvm::Function *, int> Color;
    bool Ok = true;
    std::vector<std::pair<llvm::Function *, std::vector<llvm::Function *>>> Stack;
    auto Callees = [&](llvm::Function *F) {
      std::vector<llvm::Function *> Out;
      for (llvm::BasicBlock &BB : *F)
        for (llvm::Instruction &I : BB)
          if (auto *CB = llvm::dyn_cast<llvm::CallBase>(&I))
            if (llvm::Function *Callee = CB->getCalledFunction())
              if (!Callee->isDeclaration() && !Originals.count(Callee) && !isPh(Callee))
                Out.push_back(Callee);
      return Out;
    };
    for (llvm::Function &Root : M) {
      if (Root.isDeclaration() || Originals.count(&Root) || isPh(&Root) || Color[&Root])
        continue;
      Stack.push_back({&Root, Callees(&Root)});
      Color[&Root] = 1;
      while (!Stack.empty()) {
        auto &Top = Stack.back();
        if (Top.second.empty()) {
          Color[Top.first] = 2;
          Stack.pop_back();
          continue;
        }
        llvm::Function *Next = Top.second.back();
        Top.second.pop_back();
        if (Color[Next] == 1) {
          Errors.push_back("composition: the call graph has a cycle through " +
                           Next->getName().str());
          Ok = false;
        } else if (Color[Next] == 0) {
          Color[Next] = 1;
          Stack.push_back({Next, Callees(Next)});
        }
      }
    }
    return Ok;
  }
};

} // namespace composition_detail

inline void declareCompositionSymbols(llvm::Module &M,
                                      const glue::sscp::composition_config &C) {
  using namespace composition_detail;
  auto Declare = [&](const std::string &Sym, const char *PlaceholderName) {
    if (Sym.empty() || M.getFunction(Sym))
      return;
    llvm::Function *PH = M.getFunction(PlaceholderName);
    if (!PH)
      return;
    llvm::Function *F = llvm::Function::Create(PH->getFunctionType(),
                                               llvm::GlobalValue::ExternalLinkage, Sym, &M);
    F->setCallingConv(PH->getCallingConv());
  };
  for (const auto &N : C.nodes) {
    Declare(N.body_symbol, N.body_kind == 0 ? "__sg_ph_body" : "__sg_ph_group_body");
    Declare(N.write_symbol, "__sg_ph_write");
    for (const auto &L : N.links)
      Declare(L.positions_symbol, "__sg_ph_positions");
  }
}

inline bool applyComposition(llvm::Module &M, const glue::sscp::composition_config &C,
                             std::vector<std::string> &Errors,
                             const composition_options &Options = composition_options{}) {
  composition_detail::Composer Cm(M, C, Errors);
  if (!Cm.lookup())
    return false;
  Cm.run();
  if (!Errors.empty())
    return false;
  if (Options.keep_layer_calls)
    Cm.markLayerCalls();
  bool Ok = Cm.checkNoStrayPlaceholders();
  Ok = Cm.checkAcyclic() && Ok;
  std::string Msg;
  llvm::raw_string_ostream OS(Msg);
  if (llvm::verifyModule(M, &OS)) {
    Errors.push_back("composition: the module does not verify: " + OS.str());
    Ok = false;
  }
  return Ok;
}

inline std::size_t releaseKeptLayerCalls(llvm::Module &M) {
  std::set<llvm::Function *> Kept;
  for (llvm::Function &F : M) {
    if (F.isDeclaration())
      continue;
    for (llvm::BasicBlock &BB : F)
      for (llvm::Instruction &I : BB)
        if (auto *CB = llvm::dyn_cast<llvm::CallBase>(&I))
          if (llvm::Function *Callee = CB->getCalledFunction())
            if (CB->getAttributes().hasFnAttr(llvm::Attribute::NoInline))
              Kept.insert(Callee);
  }
  std::size_t Changed = 0;
  for (llvm::Function *F : Kept) {
    if (F->isDeclaration() || !F->hasFnAttribute(llvm::Attribute::AlwaysInline))
      continue;
    F->removeFnAttr(llvm::Attribute::AlwaysInline);
    F->addFnAttr(llvm::Attribute::NoInline);
    ++Changed;
  }
  return Changed;
}

} // namespace compiler
} // namespace hipsycl

#endif
