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
// Compile-time checker of the sg graph toolchain: everything reachable from a
// body, mapping function or positions function handed to the kernel builder
// must be a pure, loop-free function. Header-only so that unit tests can run it
// directly.
#ifndef HIPSYCL_SG_BODY_CHECKER_HPP
#define HIPSYCL_SG_BODY_CHECKER_HPP

#include <clang/AST/ASTContext.h>
#include <clang/AST/Attr.h>
#include <clang/AST/DeclCXX.h>
#include <clang/AST/Expr.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/AST/Stmt.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/SourceManager.h>

#include <set>
#include <string>
#include <vector>

namespace hipsycl {
namespace compiler {
namespace sg {

namespace detail {

inline bool inSgNamespace(const clang::Decl *D) {
  for (const clang::DeclContext *C = D->getDeclContext(); C; C = C->getParent())
    if (auto *NS = llvm::dyn_cast<clang::NamespaceDecl>(C))
      if (NS->getIdentifier() && NS->getName() == "sg")
        return true;
  return false;
}

inline bool exemptDecl(const clang::ASTContext &Ctx, const clang::Decl *D) {
  return inSgNamespace(D) || Ctx.getSourceManager().isInSystemHeader(D->getLocation());
}

class BodyWalker : public clang::RecursiveASTVisitor<BodyWalker> {
public:
  BodyWalker(clang::ASTContext &Ctx, clang::DiagnosticsEngine &D,
             clang::SourceLocation BodyLoc, const clang::FunctionDecl *Root)
      : Ctx(Ctx), D(D), BodyLoc(BodyLoc) {
    ErrId = D.getCustomDiagID(clang::DiagnosticsEngine::Error, "sg: %0: %1");
    NoteBody = D.getCustomDiagID(clang::DiagnosticsEngine::Note, "in the body defined here");
    NoteCall = D.getCustomDiagID(clang::DiagnosticsEngine::Note, "called from '%0' here");
    Stack.push_back({Root, clang::SourceLocation()});
  }

  bool shouldVisitTemplateInstantiations() const { return true; }
  bool shouldVisitImplicitCode() const { return false; }

  void run(const clang::FunctionDecl *Root) {
    const clang::FunctionDecl *Def = nullptr;
    if (Root->hasBody(Def))
      TraverseDecl(const_cast<clang::FunctionDecl *>(Def));
  }

  // unevaluated and constant-evaluated contexts are not walked
  bool TraverseConstantExpr(clang::ConstantExpr *) { return true; }
  bool TraverseUnaryExprOrTypeTraitExpr(clang::UnaryExprOrTypeTraitExpr *) { return true; }
  bool TraverseCXXNoexceptExpr(clang::CXXNoexceptExpr *) { return true; }
  bool TraverseDecltypeTypeLoc(clang::DecltypeTypeLoc, bool = true) { return true; }
  bool TraverseTypeOfExprTypeLoc(clang::TypeOfExprTypeLoc, bool = true) { return true; }
  bool TraverseStaticAssertDecl(clang::StaticAssertDecl *) { return true; }

  bool TraverseIfStmt(clang::IfStmt *S) {
    if (S->isConstexpr()) {
      if (auto Taken = S->getNondiscardedCase(Ctx)) {
        if (*Taken)
          return TraverseStmt(const_cast<clang::Stmt *>(*Taken));
        return true;
      }
      return true;
    }
    return clang::RecursiveASTVisitor<BodyWalker>::TraverseIfStmt(S);
  }

  bool VisitForStmt(clang::ForStmt *S) { return loop(S->getForLoc(), "for"); }
  bool VisitCXXForRangeStmt(clang::CXXForRangeStmt *S) { return loop(S->getForLoc(), "range-for"); }
  bool VisitWhileStmt(clang::WhileStmt *S) { return loop(S->getWhileLoc(), "while"); }
  bool VisitDoStmt(clang::DoStmt *S) { return loop(S->getDoLoc(), "do"); }
  bool VisitGotoStmt(clang::GotoStmt *S) {
    error(S->getGotoLoc(), "goto", "goto is not allowed in a device function");
    return true;
  }
  bool VisitIndirectGotoStmt(clang::IndirectGotoStmt *S) {
    error(S->getGotoLoc(), "goto", "goto is not allowed in a device function");
    return true;
  }

  bool VisitLambdaExpr(clang::LambdaExpr *L) {
    for (const clang::LambdaCapture &C : L->captures())
      if (C.getCaptureKind() == clang::LCK_ByRef)
        error(C.getLocation(), "ref-capture", "a body cannot capture by reference");
    return true;
  }

  bool VisitBinaryOperator(clang::BinaryOperator *B) {
    if (B->isAssignmentOp() && !isLocal(B->getLHS()))
      error(B->getOperatorLoc(), "write",
            "a device function cannot write to anything that is not local to it");
    return true;
  }
  bool VisitUnaryOperator(clang::UnaryOperator *U) {
    if (U->isIncrementDecrementOp() && !isLocal(U->getSubExpr()))
      error(U->getOperatorLoc(), "write",
            "a device function cannot write to anything that is not local to it");
    return true;
  }

  bool VisitDeclRefExpr(clang::DeclRefExpr *E) {
    if (auto *VD = llvm::dyn_cast<clang::VarDecl>(E->getDecl()))
      if (VD->hasGlobalStorage() && !VD->getType().isConstQualified() && !VD->isConstexpr() &&
          !exemptDecl(Ctx, VD))
        error(E->getLocation(), "non-const-static",
              "a body cannot refer to a non-const variable of static storage");
    return true;
  }

  bool VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr *E) {
    switch (E->getOperator()) {
    case clang::OO_Equal:
    case clang::OO_PlusEqual:
    case clang::OO_MinusEqual:
    case clang::OO_StarEqual:
    case clang::OO_SlashEqual:
    case clang::OO_PercentEqual:
    case clang::OO_AmpEqual:
    case clang::OO_PipeEqual:
    case clang::OO_CaretEqual:
    case clang::OO_LessLessEqual:
    case clang::OO_GreaterGreaterEqual:
    case clang::OO_PlusPlus:
    case clang::OO_MinusMinus:
      if (E->getNumArgs() > 0 && !isLocal(E->getArg(0)))
        error(E->getOperatorLoc(), "write",
              "a device function cannot write to anything that is not local to it");
      break;
    default:
      break;
    }
    return true;
  }

  bool VisitCallExpr(clang::CallExpr *CE) {
    if (llvm::isa<clang::CUDAKernelCallExpr>(CE))
      return true;
    const clang::FunctionDecl *Callee = CE->getDirectCallee();
    if (!Callee) {
      if (!llvm::isa<clang::CXXPseudoDestructorExpr>(CE->getCallee()->IgnoreParenImpCasts()) &&
          !CE->getCallee()->getType()->isDependentType())
        error(CE->getBeginLoc(), "function-pointer-call",
              "a body cannot call through a function pointer");
      return true;
    }
    if (auto *MC = llvm::dyn_cast<clang::CXXMemberCallExpr>(CE)) {
      const clang::CXXMethodDecl *M = MC->getMethodDecl();
      bool Qualified = false;
      if (auto *ME = llvm::dyn_cast<clang::MemberExpr>(MC->getCallee()->IgnoreParenImpCasts()))
        Qualified = ME->hasQualifier();
      if (M && M->isVirtual() && !Qualified) {
        error(CE->getBeginLoc(), "virtual-call", "a body cannot make a virtual call");
        return true;
      }
    }
    if (isStdFunctionCall(Callee)) {
      error(CE->getBeginLoc(), "function-pointer-call",
            "a body cannot call through std::function");
      return true;
    }
    visitCallee(Callee, CE->getBeginLoc());
    return true;
  }

  bool VisitCXXConstructExpr(clang::CXXConstructExpr *E) {
    if (E->getConstructor())
      visitCallee(E->getConstructor(), E->getBeginLoc());
    return true;
  }

  bool VisitCXXBindTemporaryExpr(clang::CXXBindTemporaryExpr *E) {
    if (const clang::CXXDestructorDecl *Dtor = E->getTemporary()->getDestructor())
      visitCallee(Dtor, E->getBeginLoc());
    return true;
  }

private:
  struct Frame {
    const clang::FunctionDecl *Fn;
    clang::SourceLocation CallLoc;
  };

  static bool isStdFunctionCall(const clang::FunctionDecl *F) {
    if (auto *M = llvm::dyn_cast<clang::CXXMethodDecl>(F))
      if (M->getOverloadedOperator() == clang::OO_Call) {
        const clang::CXXRecordDecl *R = M->getParent();
        if (R && R->getIdentifier() && R->getName() == "function")
          if (auto *NS = llvm::dyn_cast<clang::NamespaceDecl>(R->getDeclContext()->getRedeclContext()))
            return NS->getIdentifier() && NS->getName() == "std";
      }
    return false;
  }

  bool loop(clang::SourceLocation L, const char *Kind) {
    error(L, std::string("loop (") + Kind + ")",
          "loops are not allowed in a device function or in anything it calls");
    return true;
  }

  bool isLocal(const clang::Expr *E) {
    E = E->IgnoreParenImpCasts();
    if (auto *DR = llvm::dyn_cast<clang::DeclRefExpr>(E)) {
      if (auto *VD = llvm::dyn_cast<clang::VarDecl>(DR->getDecl()))
        return VD->hasLocalStorage() && !VD->getType()->isReferenceType();
      return false;
    }
    if (auto *ME = llvm::dyn_cast<clang::MemberExpr>(E))
      return !ME->isArrow() && isLocal(ME->getBase());
    if (auto *AS = llvm::dyn_cast<clang::ArraySubscriptExpr>(E))
      return AS->getBase()->IgnoreParenImpCasts()->getType()->isArrayType() &&
             isLocal(AS->getBase());
    return false;
  }

  void visitCallee(const clang::FunctionDecl *F, clang::SourceLocation Loc) {
    if (exemptDecl(Ctx, F))
      return;
    const clang::FunctionDecl *Def = nullptr;
    if (!F->hasBody(Def)) {
      if (F->getBuiltinID() == 0 && !F->isImplicit() && !F->isDefaulted() && !F->isTrivial())
        error(Loc, "no-body",
              "a body calls '" + F->getQualifiedNameAsString() + "', which has no visible definition");
      return;
    }
    for (const Frame &Fr : Stack)
      if (Fr.Fn->getCanonicalDecl() == Def->getCanonicalDecl()) {
        error(Loc, "recursion", "a device function cannot call a function that calls itself");
        return;
      }
    if (!Done.insert(Def->getCanonicalDecl()).second)
      return;
    Stack.push_back({Def, Loc});
    TraverseDecl(const_cast<clang::FunctionDecl *>(Def));
    Stack.pop_back();
  }

  void error(clang::SourceLocation L, const std::string &Id, const std::string &Text) {
    const clang::FunctionDecl *In = Stack.back().Fn;
    const std::string Where = Stack.size() == 1 ? std::string("the body") : In->getQualifiedNameAsString();
    D.Report(L, ErrId) << Id << (Text + " (in '" + Where + "')");
    D.Report(BodyLoc, NoteBody);
    for (size_t I = Stack.size(); I-- > 1;) {
      const Frame &Fr = Stack[I];
      const clang::FunctionDecl *Caller = Stack[I - 1].Fn;
      std::string Name = I == 1 ? std::string("the body") : Caller->getQualifiedNameAsString();
      D.Report(Fr.CallLoc, NoteCall) << Name;
    }
  }

  clang::ASTContext &Ctx;
  clang::DiagnosticsEngine &D;
  clang::SourceLocation BodyLoc;
  unsigned ErrId, NoteBody, NoteCall;
  std::vector<Frame> Stack;
  std::set<const clang::Decl *> Done;
};

inline const clang::CXXRecordDecl *entryClosureType(const clang::FunctionDecl *FD) {
  const clang::TemplateArgumentList *Args = FD->getTemplateSpecializationArgs();
  if (!Args || Args->size() == 0 || Args->get(0).getKind() != clang::TemplateArgument::Type)
    return nullptr;
  return Args->get(0).getAsType()->getAsCXXRecordDecl();
}

// Every defined call operator of the closure: the operator of a plain lambda or
// functor, or each instantiation of a generic one.
inline std::vector<const clang::CXXMethodDecl *> callOperators(const clang::CXXRecordDecl *R) {
  std::vector<const clang::CXXMethodDecl *> Out;
  if (!R->hasDefinition())
    return Out;
  auto addSpecializations = [&](const clang::FunctionTemplateDecl *FT) {
    for (const clang::FunctionDecl *Spec : FT->specializations())
      if (auto *M = llvm::dyn_cast<clang::CXXMethodDecl>(Spec))
        Out.push_back(M);
  };
  if (R->isLambda()) {
    if (const clang::FunctionTemplateDecl *FT = R->getDependentLambdaCallOperator())
      addSpecializations(FT);
    else if (const clang::CXXMethodDecl *M = R->getLambdaCallOperator())
      Out.push_back(M);
    return Out;
  }
  for (const clang::NamedDecl *ND :
       R->lookup(R->getASTContext().DeclarationNames.getCXXOperatorName(clang::OO_Call))) {
    const clang::NamedDecl *U = ND->getUnderlyingDecl();
    if (auto *FT = llvm::dyn_cast<clang::FunctionTemplateDecl>(U))
      addSpecializations(FT);
    else if (auto *M = llvm::dyn_cast<clang::CXXMethodDecl>(U))
      Out.push_back(M);
  }
  return Out;
}

} // namespace detail

inline bool is_body_entry(const clang::FunctionDecl *FD) {
  if (!FD || !FD->getTemplateSpecializationArgs())
    return false;
  for (const clang::AnnotateAttr *A : FD->specific_attrs<clang::AnnotateAttr>())
    if (A->getAnnotation() == "sg_body_entry")
      return true;
  return false;
}

inline void check_entries(clang::ASTContext &Ctx, clang::DiagnosticsEngine &Diags,
                          const std::vector<const clang::FunctionDecl *> &Entries) {
  if (Entries.empty())
    return;
  std::set<const clang::Decl *> Checked;
  for (const clang::FunctionDecl *Entry : Entries) {
    if (!is_body_entry(Entry))
      continue;
    const clang::CXXRecordDecl *R = detail::entryClosureType(Entry);
    if (!R)
      continue;
    unsigned ErrId = Diags.getCustomDiagID(clang::DiagnosticsEngine::Error, "sg: %0: %1");
    unsigned NoteBody =
        Diags.getCustomDiagID(clang::DiagnosticsEngine::Note, "in the body defined here");
    clang::SourceLocation BodyLoc = R->getLocation();
    bool ClosureChecked = false;
    for (const clang::CXXMethodDecl *Op : detail::callOperators(R)) {
      const clang::FunctionDecl *Def = nullptr;
      if (!Op->hasBody(Def))
        continue;
      if (!Checked.insert(Def->getCanonicalDecl()).second)
        continue;
      if (!ClosureChecked) {
        ClosureChecked = true;
        if (!Op->isConst()) {
          Diags.Report(Op->getLocation(), ErrId)
              << "mutable-closure" << "a device function must not be a mutable lambda";
          Diags.Report(BodyLoc, NoteBody);
        }
        if (R->isLambda())
          for (const clang::LambdaCapture &C : R->captures())
            if (C.getCaptureKind() == clang::LCK_ByRef) {
              Diags.Report(C.getLocation(), ErrId)
                  << "ref-capture" << "a device function cannot capture by reference";
              Diags.Report(BodyLoc, NoteBody);
            }
      }
      detail::BodyWalker W(Ctx, Diags, BodyLoc, Def);
      W.run(Def);
    }
  }
}

} // namespace sg
} // namespace compiler
} // namespace hipsycl

#endif
