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
// The composition step of the sg graph toolchain. The generic kernel calls one
// placeholder, __sg_ph_dispatch(block, record, aux, nd, memory). This step
// replaces it with a function that calls every task function of the composition
// in order, each with its own index as a constant first argument; a task
// function returns at once unless that index is the task being run. Every task
// function keeps its single definition. Header-only so that unit tests can
// include it. No exceptions.
#ifndef HIPSYCL_COMPOSITION_STEP_HPP
#define HIPSYCL_COMPOSITION_STEP_HPP

#include "hipSYCL/glue/llvm-sscp/composition.hpp"

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>

#include <string>
#include <vector>

namespace hipsycl {
namespace compiler {

inline const char *compositionPlaceholderName() { return "__sg_ph_dispatch"; }
inline const char *compositionDispatchName() { return "__sg_dispatch_composed"; }

// The type of a task function: the placeholder's parameters behind the task's own index.
inline llvm::FunctionType *compositionTaskType(llvm::Function *Placeholder) {
  llvm::LLVMContext &Ctx = Placeholder->getContext();
  std::vector<llvm::Type *> Params;
  Params.push_back(llvm::Type::getInt32Ty(Ctx));
  for (llvm::Type *T : Placeholder->getFunctionType()->params())
    Params.push_back(T);
  return llvm::FunctionType::get(Placeholder->getReturnType(), Params, false);
}

// Gives every task symbol a declaration so that symbol resolution can link its definition.
inline void declareCompositionSymbols(llvm::Module &M, const glue::sscp::composition_config &C) {
  llvm::Function *PH = M.getFunction(compositionPlaceholderName());
  if (!PH)
    return;
  llvm::FunctionType *TaskTy = compositionTaskType(PH);
  for (const std::string &Sym : C.imported_symbols()) {
    if (M.getFunction(Sym))
      continue;
    llvm::Function *F = llvm::Function::Create(TaskTy, llvm::GlobalValue::ExternalLinkage, Sym, &M);
    F->setCallingConv(PH->getCallingConv());
  }
}

inline bool applyComposition(llvm::Module &M, const glue::sscp::composition_config &C,
                             std::vector<std::string> &Errors) {
  llvm::Function *PH = M.getFunction(compositionPlaceholderName());
  if (!PH) {
    Errors.push_back(std::string("composition: missing ") + compositionPlaceholderName());
    return false;
  }
  llvm::FunctionType *TaskTy = compositionTaskType(PH);
  std::vector<llvm::Function *> Tasks;
  bool Ok = true;
  for (const std::string &Sym : C.tasks) {
    llvm::Function *F = M.getFunction(Sym);
    if (!F || F->isDeclaration()) {
      Errors.push_back("composition: " + Sym + " is not defined");
      Ok = false;
      continue;
    }
    if (F->getFunctionType() != TaskTy) {
      Errors.push_back("composition: " + Sym + " does not have the type of a task function");
      Ok = false;
      continue;
    }
    Tasks.push_back(F);
  }
  if (!Ok)
    return false;

  llvm::Function *Composed =
      llvm::Function::Create(PH->getFunctionType(), llvm::GlobalValue::InternalLinkage,
                             compositionDispatchName(), &M);
  Composed->setCallingConv(PH->getCallingConv());
  llvm::BasicBlock *Entry = llvm::BasicBlock::Create(M.getContext(), "entry", Composed);
  llvm::IRBuilder<> B(Entry);
  for (size_t K = 0; K < Tasks.size(); ++K) {
    std::vector<llvm::Value *> Args;
    Args.push_back(llvm::ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), K));
    for (llvm::Argument &A : Composed->args())
      Args.push_back(&A);
    llvm::CallInst *Call = B.CreateCall(TaskTy, Tasks[K], Args);
    Call->setCallingConv(Tasks[K]->getCallingConv());
  }
  B.CreateRetVoid();

  std::vector<llvm::CallBase *> Calls;
  for (llvm::Function &F : M) {
    if (&F == Composed || F.isDeclaration())
      continue;
    for (llvm::BasicBlock &BB : F)
      for (llvm::Instruction &I : BB)
        if (auto *CB = llvm::dyn_cast<llvm::CallBase>(&I))
          if (CB->getCalledFunction() == PH)
            Calls.push_back(CB);
  }
  for (llvm::CallBase *CB : Calls)
    CB->setCalledFunction(Composed);

  std::string Msg;
  llvm::raw_string_ostream OS(Msg);
  if (llvm::verifyModule(M, &OS)) {
    Errors.push_back("composition: the module does not verify: " + OS.str());
    return false;
  }
  return true;
}

} // namespace compiler
} // namespace hipsycl

#endif
