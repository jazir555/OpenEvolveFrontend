// Lean compiler output
// Module: Mathlib.Lean.Meta.RefinedDiscrTree.Encode
// Imports: public import Init public import Mathlib.Lean.Meta.RefinedDiscrTree.Basic public import Lean.Meta.DiscrTree public import Lean.Meta.LazyDiscrTree import all Lean.Meta.DiscrTree
#include <lean/lean.h>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C" {
#endif
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Lean_Meta_RefinedDiscrTree_evalLazyEntry_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_findIdx_x3f___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop_reduce(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Meta_DiscrTree_hasNoindexAnnotation(lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint64_t l_Lean_Meta_Context_configKey(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_isIgnoredArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_evalLazyEntry___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_evalLazyEntry(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Lean_Meta_RefinedDiscrTree_evalLazyEntry_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint64_t lean_uint64_lor(uint64_t, uint64_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
uint8_t l_Lean_Expr_isApp(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_initializeLazyEntry___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvarId_x21(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEtaAux(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_back___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___redArg(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Lean_Meta_RefinedDiscrTree_evalLazyEntry_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__0;
static lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___closed__0;
lean_object* l_ReaderT_instMonad___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_reduce___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_expr_instantiate_rev_range(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux___closed__0;
uint8_t lean_expr_eqv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExpr(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadEST(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodeExprWithEta_go_fold(lean_object*, lean_object*, lean_object*);
lean_object* l_List_reverseAux___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEta___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_throwFunctionExpected___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_foldl___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isMVar(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___lam__1___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExprWithEta___closed__0;
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instInhabitedOfMonad___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instInhabitedForall___redArg___lam__0___boxed(lean_object*, lean_object*);
lean_object* lean_array_pop(lean_object*);
lean_object* l_Lean_Meta_LazyDiscrTree_MatchClone_toNatLit_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry___redArg(uint8_t, lean_object*);
uint8_t lean_is_out_param(lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_withLocalDecl___redArg(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnfD(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExprWithEta___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__3;
lean_object* l_Lean_Meta_DiscrTree_reduce(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_bind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_isStarWithArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_isStarWithArg___boxed(lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
lean_object* l_Lean_Meta_Context_config(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_initializeLazyEntry(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEtaAux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static uint64_t lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEta___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqFVarId_beq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
static lean_object* lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___redArg___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExprWithEta(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_set(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0_spec__0(lean_object*, lean_object*);
lean_object* l_Lean_Meta_instInhabitedMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_panic_fn(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEta(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodeExprWithEta_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__0;
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint64_t lean_uint64_shift_left(uint64_t, uint64_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_toList_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_isIgnoredArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_Meta_RefinedDiscrTree_encodeExprWithEta_spec__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_mk(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___closed__0;
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_mkExprInfo___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__2;
lean_object* l_Lean_Expr_fvar___override(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0_spec__0___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_panic___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_toList_spec__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_evalLazyEntry___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_toList_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_spec__0(lean_object*, lean_object*);
uint64_t l_Lean_Meta_TransparencyMode_toUInt64(uint8_t);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_StateT_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__4;
static lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodeExprWithEta_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_is_class(lean_object*, lean_object*);
uint64_t l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_evalLazyEntry___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Lean_Meta_RefinedDiscrTree_evalLazyEntry_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_expr_instantiate1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_Meta_RefinedDiscrTree_encodeExprWithEta_spec__0(lean_object*, lean_object*);
static lean_object* lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__1;
lean_object* l_Lean_Meta_isProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_pure(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop_reduce___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMCtxImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0_spec__0_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lean_array_get_size(x_1);
x_5 = lean_nat_dec_lt(x_3, x_4);
if (x_5 == 0)
{
lean_object* x_6; 
lean_dec(x_3);
x_6 = lean_box(0);
return x_6;
}
else
{
lean_object* x_7; uint8_t x_8; 
x_7 = lean_array_fget_borrowed(x_1, x_3);
x_8 = l_Lean_instBEqMVarId_beq(x_7, x_2);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; 
x_9 = lean_unsigned_to_nat(1u);
x_10 = lean_nat_add(x_3, x_9);
lean_dec(x_3);
x_3 = x_10;
goto _start;
}
else
{
lean_object* x_12; 
x_12 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_12, 0, x_3);
return x_12;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_unsigned_to_nat(0u);
x_4 = lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0_spec__0_spec__0(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0_spec__0(x_1, x_2);
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
else
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_3);
if (x_5 == 0)
{
return x_3;
}
else
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_ctor_get(x_3, 0);
lean_inc(x_6);
lean_dec(x_3);
x_7 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_7, 0, x_6);
return x_7;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_4; 
x_4 = lean_ctor_get(x_2, 3);
lean_inc(x_4);
if (lean_obj_tag(x_4) == 1)
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 1);
x_8 = lean_ctor_get(x_2, 2);
x_9 = lean_ctor_get(x_2, 4);
x_10 = lean_ctor_get(x_4, 0);
x_11 = lp_mathlib_Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0(x_10, x_1);
if (lean_obj_tag(x_11) == 0)
{
uint8_t x_12; 
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc(x_6);
x_12 = !lean_is_exclusive(x_2);
if (x_12 == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_13 = lean_ctor_get(x_2, 4);
lean_dec(x_13);
x_14 = lean_ctor_get(x_2, 3);
lean_dec(x_14);
x_15 = lean_ctor_get(x_2, 2);
lean_dec(x_15);
x_16 = lean_ctor_get(x_2, 1);
lean_dec(x_16);
x_17 = lean_ctor_get(x_2, 0);
lean_dec(x_17);
x_18 = lean_array_get_size(x_10);
x_19 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_19, 0, x_18);
x_20 = lean_array_push(x_10, x_1);
lean_ctor_set(x_4, 0, x_20);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_19);
lean_ctor_set(x_21, 1, x_2);
x_22 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_22, 0, x_21);
return x_22;
}
else
{
lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; 
lean_dec(x_2);
x_23 = lean_array_get_size(x_10);
x_24 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_24, 0, x_23);
x_25 = lean_array_push(x_10, x_1);
lean_ctor_set(x_4, 0, x_25);
x_26 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_26, 0, x_6);
lean_ctor_set(x_26, 1, x_7);
lean_ctor_set(x_26, 2, x_8);
lean_ctor_set(x_26, 3, x_4);
lean_ctor_set(x_26, 4, x_9);
x_27 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_27, 0, x_24);
lean_ctor_set(x_27, 1, x_26);
x_28 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_28, 0, x_27);
return x_28;
}
}
else
{
uint8_t x_29; 
lean_free_object(x_4);
lean_dec(x_10);
lean_dec(x_1);
x_29 = !lean_is_exclusive(x_11);
if (x_29 == 0)
{
lean_object* x_30; lean_object* x_31; 
x_30 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_30, 0, x_11);
lean_ctor_set(x_30, 1, x_2);
x_31 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_31, 0, x_30);
return x_31;
}
else
{
lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; 
x_32 = lean_ctor_get(x_11, 0);
lean_inc(x_32);
lean_dec(x_11);
x_33 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_33, 0, x_32);
x_34 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_34, 0, x_33);
lean_ctor_set(x_34, 1, x_2);
x_35 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_35, 0, x_34);
return x_35;
}
}
}
else
{
lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; 
x_36 = lean_ctor_get(x_2, 0);
x_37 = lean_ctor_get(x_2, 1);
x_38 = lean_ctor_get(x_2, 2);
x_39 = lean_ctor_get(x_2, 4);
x_40 = lean_ctor_get(x_4, 0);
lean_inc(x_40);
lean_dec(x_4);
x_41 = lp_mathlib_Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0(x_40, x_1);
if (lean_obj_tag(x_41) == 0)
{
lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; 
lean_inc(x_39);
lean_inc_ref(x_38);
lean_inc(x_37);
lean_inc(x_36);
if (lean_is_exclusive(x_2)) {
 lean_ctor_release(x_2, 0);
 lean_ctor_release(x_2, 1);
 lean_ctor_release(x_2, 2);
 lean_ctor_release(x_2, 3);
 lean_ctor_release(x_2, 4);
 x_42 = x_2;
} else {
 lean_dec_ref(x_2);
 x_42 = lean_box(0);
}
x_43 = lean_array_get_size(x_40);
x_44 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_44, 0, x_43);
x_45 = lean_array_push(x_40, x_1);
x_46 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_46, 0, x_45);
if (lean_is_scalar(x_42)) {
 x_47 = lean_alloc_ctor(0, 5, 0);
} else {
 x_47 = x_42;
}
lean_ctor_set(x_47, 0, x_36);
lean_ctor_set(x_47, 1, x_37);
lean_ctor_set(x_47, 2, x_38);
lean_ctor_set(x_47, 3, x_46);
lean_ctor_set(x_47, 4, x_39);
x_48 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_48, 0, x_44);
lean_ctor_set(x_48, 1, x_47);
x_49 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_49, 0, x_48);
return x_49;
}
else
{
lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; 
lean_dec(x_40);
lean_dec(x_1);
x_50 = lean_ctor_get(x_41, 0);
lean_inc(x_50);
if (lean_is_exclusive(x_41)) {
 lean_ctor_release(x_41, 0);
 x_51 = x_41;
} else {
 lean_dec_ref(x_41);
 x_51 = lean_box(0);
}
if (lean_is_scalar(x_51)) {
 x_52 = lean_alloc_ctor(1, 1, 0);
} else {
 x_52 = x_51;
}
lean_ctor_set(x_52, 0, x_50);
x_53 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_53, 0, x_52);
lean_ctor_set(x_53, 1, x_2);
x_54 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_54, 0, x_53);
return x_54;
}
}
}
else
{
lean_object* x_55; lean_object* x_56; lean_object* x_57; 
lean_dec(x_4);
lean_dec(x_1);
x_55 = lean_box(0);
x_56 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_56, 0, x_55);
lean_ctor_set(x_56, 1, x_2);
x_57 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_57, 0, x_56);
return x_57;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar___redArg(x_1, x_3);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0_spec__0(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0_spec__0_spec__0(x_1, x_2, x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar___redArg(x_1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_box(8);
x_4 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_9; lean_object* x_10; 
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_2);
lean_ctor_set(x_9, 1, x_3);
x_10 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_10, 0, x_9);
return x_10;
}
else
{
uint8_t x_11; 
x_11 = !lean_is_exclusive(x_1);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; uint8_t x_14; 
x_12 = lean_ctor_get(x_1, 1);
x_13 = lean_ctor_get(x_1, 0);
lean_dec(x_13);
x_14 = !lean_is_exclusive(x_3);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_15 = lean_ctor_get(x_3, 4);
lean_dec(x_15);
x_16 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg___lam__0___boxed), 2, 0);
x_17 = lean_box(0);
lean_ctor_set(x_1, 1, x_17);
lean_ctor_set(x_1, 0, x_2);
x_18 = l_List_foldl___redArg(x_16, x_1, x_12);
lean_ctor_set(x_3, 4, x_18);
x_19 = lean_box(8);
x_20 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_20, 0, x_19);
lean_ctor_set(x_20, 1, x_3);
x_21 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_21, 0, x_20);
return x_21;
}
else
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; 
x_22 = lean_ctor_get(x_3, 0);
x_23 = lean_ctor_get(x_3, 1);
x_24 = lean_ctor_get(x_3, 2);
x_25 = lean_ctor_get(x_3, 3);
lean_inc(x_25);
lean_inc(x_24);
lean_inc(x_23);
lean_inc(x_22);
lean_dec(x_3);
x_26 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg___lam__0___boxed), 2, 0);
x_27 = lean_box(0);
lean_ctor_set(x_1, 1, x_27);
lean_ctor_set(x_1, 0, x_2);
x_28 = l_List_foldl___redArg(x_26, x_1, x_12);
x_29 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_29, 0, x_22);
lean_ctor_set(x_29, 1, x_23);
lean_ctor_set(x_29, 2, x_24);
lean_ctor_set(x_29, 3, x_25);
lean_ctor_set(x_29, 4, x_28);
x_30 = lean_box(8);
x_31 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_31, 0, x_30);
lean_ctor_set(x_31, 1, x_29);
x_32 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_32, 0, x_31);
return x_32;
}
}
else
{
lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; 
x_33 = lean_ctor_get(x_1, 1);
lean_inc(x_33);
lean_dec(x_1);
x_34 = lean_ctor_get(x_3, 0);
lean_inc(x_34);
x_35 = lean_ctor_get(x_3, 1);
lean_inc(x_35);
x_36 = lean_ctor_get(x_3, 2);
lean_inc_ref(x_36);
x_37 = lean_ctor_get(x_3, 3);
lean_inc(x_37);
if (lean_is_exclusive(x_3)) {
 lean_ctor_release(x_3, 0);
 lean_ctor_release(x_3, 1);
 lean_ctor_release(x_3, 2);
 lean_ctor_release(x_3, 3);
 lean_ctor_release(x_3, 4);
 x_38 = x_3;
} else {
 lean_dec_ref(x_3);
 x_38 = lean_box(0);
}
x_39 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg___lam__0___boxed), 2, 0);
x_40 = lean_box(0);
x_41 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_41, 0, x_2);
lean_ctor_set(x_41, 1, x_40);
x_42 = l_List_foldl___redArg(x_39, x_41, x_33);
if (lean_is_scalar(x_38)) {
 x_43 = lean_alloc_ctor(0, 5, 0);
} else {
 x_43 = x_38;
}
lean_ctor_set(x_43, 0, x_34);
lean_ctor_set(x_43, 1, x_35);
lean_ctor_set(x_43, 2, x_36);
lean_ctor_set(x_43, 3, x_37);
lean_ctor_set(x_43, 4, x_42);
x_44 = lean_box(8);
x_45 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_45, 0, x_44);
lean_ctor_set(x_45, 1, x_43);
x_46 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_46, 0, x_45);
return x_46;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_2);
lean_ctor_set(x_5, 1, x_3);
x_6 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_6, 0, x_5);
return x_6;
}
else
{
uint8_t x_7; 
x_7 = !lean_is_exclusive(x_1);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_8 = lean_ctor_get(x_1, 1);
x_9 = lean_ctor_get(x_1, 0);
lean_dec(x_9);
x_10 = !lean_is_exclusive(x_3);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_11 = lean_ctor_get(x_3, 4);
lean_dec(x_11);
x_12 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg___lam__0___boxed), 2, 0);
x_13 = lean_box(0);
lean_ctor_set(x_1, 1, x_13);
lean_ctor_set(x_1, 0, x_2);
x_14 = l_List_foldl___redArg(x_12, x_1, x_8);
lean_ctor_set(x_3, 4, x_14);
x_15 = lean_box(8);
x_16 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_16, 0, x_15);
lean_ctor_set(x_16, 1, x_3);
x_17 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_17, 0, x_16);
return x_17;
}
else
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; 
x_18 = lean_ctor_get(x_3, 0);
x_19 = lean_ctor_get(x_3, 1);
x_20 = lean_ctor_get(x_3, 2);
x_21 = lean_ctor_get(x_3, 3);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_dec(x_3);
x_22 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg___lam__0___boxed), 2, 0);
x_23 = lean_box(0);
lean_ctor_set(x_1, 1, x_23);
lean_ctor_set(x_1, 0, x_2);
x_24 = l_List_foldl___redArg(x_22, x_1, x_8);
x_25 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_25, 0, x_18);
lean_ctor_set(x_25, 1, x_19);
lean_ctor_set(x_25, 2, x_20);
lean_ctor_set(x_25, 3, x_21);
lean_ctor_set(x_25, 4, x_24);
x_26 = lean_box(8);
x_27 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_27, 0, x_26);
lean_ctor_set(x_27, 1, x_25);
x_28 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_28, 0, x_27);
return x_28;
}
}
else
{
lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; 
x_29 = lean_ctor_get(x_1, 1);
lean_inc(x_29);
lean_dec(x_1);
x_30 = lean_ctor_get(x_3, 0);
lean_inc(x_30);
x_31 = lean_ctor_get(x_3, 1);
lean_inc(x_31);
x_32 = lean_ctor_get(x_3, 2);
lean_inc_ref(x_32);
x_33 = lean_ctor_get(x_3, 3);
lean_inc(x_33);
if (lean_is_exclusive(x_3)) {
 lean_ctor_release(x_3, 0);
 lean_ctor_release(x_3, 1);
 lean_ctor_release(x_3, 2);
 lean_ctor_release(x_3, 3);
 lean_ctor_release(x_3, 4);
 x_34 = x_3;
} else {
 lean_dec_ref(x_3);
 x_34 = lean_box(0);
}
x_35 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg___lam__0___boxed), 2, 0);
x_36 = lean_box(0);
x_37 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_37, 0, x_2);
lean_ctor_set(x_37, 1, x_36);
x_38 = l_List_foldl___redArg(x_35, x_37, x_29);
if (lean_is_scalar(x_34)) {
 x_39 = lean_alloc_ctor(0, 5, 0);
} else {
 x_39 = x_34;
}
lean_ctor_set(x_39, 0, x_30);
lean_ctor_set(x_39, 1, x_31);
lean_ctor_set(x_39, 2, x_32);
lean_ctor_set(x_39, 3, x_33);
lean_ctor_set(x_39, 4, x_38);
x_40 = lean_box(8);
x_41 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_41, 0, x_40);
lean_ctor_set(x_41, 1, x_39);
x_42 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_42, 0, x_41);
return x_42;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg(x_1, x_2, x_3);
return x_5;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = l_Lean_instBEqFVarId_beq(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; lean_object* x_12; 
x_11 = l_List_appendTR___redArg(x_1, x_3);
x_12 = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkExprInfo___redArg(x_2, x_11, x_6);
if (lean_obj_tag(x_12) == 0)
{
uint8_t x_13; 
x_13 = !lean_is_exclusive(x_12);
if (x_13 == 0)
{
uint8_t x_14; 
x_14 = !lean_is_exclusive(x_5);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_15 = lean_ctor_get(x_12, 0);
x_16 = lean_ctor_get(x_5, 0);
lean_dec(x_16);
x_17 = lean_box(0);
x_18 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_18, 0, x_15);
lean_ctor_set(x_5, 0, x_18);
x_19 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_19, 0, x_17);
lean_ctor_set(x_19, 1, x_5);
lean_ctor_set(x_12, 0, x_19);
return x_12;
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; 
x_20 = lean_ctor_get(x_12, 0);
x_21 = lean_ctor_get(x_5, 1);
x_22 = lean_ctor_get(x_5, 2);
x_23 = lean_ctor_get(x_5, 3);
x_24 = lean_ctor_get(x_5, 4);
lean_inc(x_24);
lean_inc(x_23);
lean_inc(x_22);
lean_inc(x_21);
lean_dec(x_5);
x_25 = lean_box(0);
x_26 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_26, 0, x_20);
x_27 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_27, 0, x_26);
lean_ctor_set(x_27, 1, x_21);
lean_ctor_set(x_27, 2, x_22);
lean_ctor_set(x_27, 3, x_23);
lean_ctor_set(x_27, 4, x_24);
x_28 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_28, 0, x_25);
lean_ctor_set(x_28, 1, x_27);
lean_ctor_set(x_12, 0, x_28);
return x_12;
}
}
else
{
lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; 
x_29 = lean_ctor_get(x_12, 0);
lean_inc(x_29);
lean_dec(x_12);
x_30 = lean_ctor_get(x_5, 1);
lean_inc(x_30);
x_31 = lean_ctor_get(x_5, 2);
lean_inc_ref(x_31);
x_32 = lean_ctor_get(x_5, 3);
lean_inc(x_32);
x_33 = lean_ctor_get(x_5, 4);
lean_inc(x_33);
if (lean_is_exclusive(x_5)) {
 lean_ctor_release(x_5, 0);
 lean_ctor_release(x_5, 1);
 lean_ctor_release(x_5, 2);
 lean_ctor_release(x_5, 3);
 lean_ctor_release(x_5, 4);
 x_34 = x_5;
} else {
 lean_dec_ref(x_5);
 x_34 = lean_box(0);
}
x_35 = lean_box(0);
x_36 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_36, 0, x_29);
if (lean_is_scalar(x_34)) {
 x_37 = lean_alloc_ctor(0, 5, 0);
} else {
 x_37 = x_34;
}
lean_ctor_set(x_37, 0, x_36);
lean_ctor_set(x_37, 1, x_30);
lean_ctor_set(x_37, 2, x_31);
lean_ctor_set(x_37, 3, x_32);
lean_ctor_set(x_37, 4, x_33);
x_38 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_38, 0, x_35);
lean_ctor_set(x_38, 1, x_37);
x_39 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_39, 0, x_38);
return x_39;
}
}
else
{
uint8_t x_40; 
lean_dec_ref(x_5);
x_40 = !lean_is_exclusive(x_12);
if (x_40 == 0)
{
return x_12;
}
else
{
lean_object* x_41; lean_object* x_42; 
x_41 = lean_ctor_get(x_12, 0);
lean_inc(x_41);
lean_dec(x_12);
x_42 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_42, 0, x_41);
return x_42;
}
}
}
}
static lean_object* _init_lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = l_instMonadEST(lean_box(0), lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_Core_instMonadCoreM___lam__0___boxed), 5, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_Core_instMonadCoreM___lam__1___boxed), 7, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_Meta_instMonadMetaM___lam__0___boxed), 7, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_Meta_instMonadMetaM___lam__1___boxed), 9, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_9 = lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__0;
x_10 = l_ReaderT_instMonad___redArg(x_9);
x_11 = !lean_is_exclusive(x_10);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; uint8_t x_14; 
x_12 = lean_ctor_get(x_10, 0);
x_13 = lean_ctor_get(x_10, 1);
lean_dec(x_13);
x_14 = !lean_is_exclusive(x_12);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; uint8_t x_29; 
x_15 = lean_ctor_get(x_12, 0);
x_16 = lean_ctor_get(x_12, 2);
x_17 = lean_ctor_get(x_12, 3);
x_18 = lean_ctor_get(x_12, 4);
x_19 = lean_ctor_get(x_12, 1);
lean_dec(x_19);
x_20 = lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__1;
x_21 = lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__2;
lean_inc_ref(x_15);
x_22 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_22, 0, x_15);
x_23 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_23, 0, x_15);
x_24 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_24, 0, x_22);
lean_ctor_set(x_24, 1, x_23);
x_25 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_25, 0, x_18);
x_26 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_26, 0, x_17);
x_27 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_27, 0, x_16);
lean_ctor_set(x_12, 4, x_25);
lean_ctor_set(x_12, 3, x_26);
lean_ctor_set(x_12, 2, x_27);
lean_ctor_set(x_12, 1, x_20);
lean_ctor_set(x_12, 0, x_24);
lean_ctor_set(x_10, 1, x_21);
x_28 = l_ReaderT_instMonad___redArg(x_10);
x_29 = !lean_is_exclusive(x_28);
if (x_29 == 0)
{
lean_object* x_30; lean_object* x_31; uint8_t x_32; 
x_30 = lean_ctor_get(x_28, 0);
x_31 = lean_ctor_get(x_28, 1);
lean_dec(x_31);
x_32 = !lean_is_exclusive(x_30);
if (x_32 == 0)
{
lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; 
x_33 = lean_ctor_get(x_30, 0);
x_34 = lean_ctor_get(x_30, 2);
x_35 = lean_ctor_get(x_30, 3);
x_36 = lean_ctor_get(x_30, 4);
x_37 = lean_ctor_get(x_30, 1);
lean_dec(x_37);
x_38 = lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__3;
x_39 = lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__4;
lean_inc_ref(x_33);
x_40 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_40, 0, x_33);
x_41 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_41, 0, x_33);
x_42 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_42, 0, x_40);
lean_ctor_set(x_42, 1, x_41);
x_43 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_43, 0, x_36);
x_44 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_44, 0, x_35);
x_45 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_45, 0, x_34);
lean_ctor_set(x_30, 4, x_43);
lean_ctor_set(x_30, 3, x_44);
lean_ctor_set(x_30, 2, x_45);
lean_ctor_set(x_30, 1, x_38);
lean_ctor_set(x_30, 0, x_42);
lean_ctor_set(x_28, 1, x_39);
lean_inc_ref(x_28);
x_46 = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_46, 0, x_28);
lean_inc_ref(x_28);
x_47 = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_47, 0, x_28);
lean_inc_ref(x_28);
x_48 = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__7), 6, 1);
lean_closure_set(x_48, 0, x_28);
lean_inc_ref(x_28);
x_49 = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__9), 6, 1);
lean_closure_set(x_49, 0, x_28);
lean_inc_ref(x_28);
x_50 = lean_alloc_closure((void*)(l_StateT_map), 8, 3);
lean_closure_set(x_50, 0, lean_box(0));
lean_closure_set(x_50, 1, lean_box(0));
lean_closure_set(x_50, 2, x_28);
x_51 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_51, 0, x_50);
lean_ctor_set(x_51, 1, x_46);
lean_inc_ref(x_28);
x_52 = lean_alloc_closure((void*)(l_StateT_pure), 6, 3);
lean_closure_set(x_52, 0, lean_box(0));
lean_closure_set(x_52, 1, lean_box(0));
lean_closure_set(x_52, 2, x_28);
x_53 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_53, 0, x_51);
lean_ctor_set(x_53, 1, x_52);
lean_ctor_set(x_53, 2, x_47);
lean_ctor_set(x_53, 3, x_48);
lean_ctor_set(x_53, 4, x_49);
x_54 = lean_alloc_closure((void*)(l_StateT_bind), 8, 3);
lean_closure_set(x_54, 0, lean_box(0));
lean_closure_set(x_54, 1, lean_box(0));
lean_closure_set(x_54, 2, x_28);
x_55 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_55, 0, x_53);
lean_ctor_set(x_55, 1, x_54);
x_56 = lean_box(0);
x_57 = l_instInhabitedOfMonad___redArg(x_55, x_56);
x_58 = lean_alloc_closure((void*)(l_instInhabitedForall___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_58, 0, x_57);
x_59 = lean_panic_fn(x_58, x_1);
x_60 = lean_apply_7(x_59, x_2, x_3, x_4, x_5, x_6, x_7, lean_box(0));
return x_60;
}
else
{
lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; 
x_61 = lean_ctor_get(x_30, 0);
x_62 = lean_ctor_get(x_30, 2);
x_63 = lean_ctor_get(x_30, 3);
x_64 = lean_ctor_get(x_30, 4);
lean_inc(x_64);
lean_inc(x_63);
lean_inc(x_62);
lean_inc(x_61);
lean_dec(x_30);
x_65 = lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__3;
x_66 = lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__4;
lean_inc_ref(x_61);
x_67 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_67, 0, x_61);
x_68 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_68, 0, x_61);
x_69 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_69, 0, x_67);
lean_ctor_set(x_69, 1, x_68);
x_70 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_70, 0, x_64);
x_71 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_71, 0, x_63);
x_72 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_72, 0, x_62);
x_73 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_73, 0, x_69);
lean_ctor_set(x_73, 1, x_65);
lean_ctor_set(x_73, 2, x_72);
lean_ctor_set(x_73, 3, x_71);
lean_ctor_set(x_73, 4, x_70);
lean_ctor_set(x_28, 1, x_66);
lean_ctor_set(x_28, 0, x_73);
lean_inc_ref(x_28);
x_74 = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_74, 0, x_28);
lean_inc_ref(x_28);
x_75 = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_75, 0, x_28);
lean_inc_ref(x_28);
x_76 = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__7), 6, 1);
lean_closure_set(x_76, 0, x_28);
lean_inc_ref(x_28);
x_77 = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__9), 6, 1);
lean_closure_set(x_77, 0, x_28);
lean_inc_ref(x_28);
x_78 = lean_alloc_closure((void*)(l_StateT_map), 8, 3);
lean_closure_set(x_78, 0, lean_box(0));
lean_closure_set(x_78, 1, lean_box(0));
lean_closure_set(x_78, 2, x_28);
x_79 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_79, 0, x_78);
lean_ctor_set(x_79, 1, x_74);
lean_inc_ref(x_28);
x_80 = lean_alloc_closure((void*)(l_StateT_pure), 6, 3);
lean_closure_set(x_80, 0, lean_box(0));
lean_closure_set(x_80, 1, lean_box(0));
lean_closure_set(x_80, 2, x_28);
x_81 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_81, 0, x_79);
lean_ctor_set(x_81, 1, x_80);
lean_ctor_set(x_81, 2, x_75);
lean_ctor_set(x_81, 3, x_76);
lean_ctor_set(x_81, 4, x_77);
x_82 = lean_alloc_closure((void*)(l_StateT_bind), 8, 3);
lean_closure_set(x_82, 0, lean_box(0));
lean_closure_set(x_82, 1, lean_box(0));
lean_closure_set(x_82, 2, x_28);
x_83 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_83, 0, x_81);
lean_ctor_set(x_83, 1, x_82);
x_84 = lean_box(0);
x_85 = l_instInhabitedOfMonad___redArg(x_83, x_84);
x_86 = lean_alloc_closure((void*)(l_instInhabitedForall___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_86, 0, x_85);
x_87 = lean_panic_fn(x_86, x_1);
x_88 = lean_apply_7(x_87, x_2, x_3, x_4, x_5, x_6, x_7, lean_box(0));
return x_88;
}
}
else
{
lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; lean_object* x_116; lean_object* x_117; lean_object* x_118; lean_object* x_119; 
x_89 = lean_ctor_get(x_28, 0);
lean_inc(x_89);
lean_dec(x_28);
x_90 = lean_ctor_get(x_89, 0);
lean_inc_ref(x_90);
x_91 = lean_ctor_get(x_89, 2);
lean_inc(x_91);
x_92 = lean_ctor_get(x_89, 3);
lean_inc(x_92);
x_93 = lean_ctor_get(x_89, 4);
lean_inc(x_93);
if (lean_is_exclusive(x_89)) {
 lean_ctor_release(x_89, 0);
 lean_ctor_release(x_89, 1);
 lean_ctor_release(x_89, 2);
 lean_ctor_release(x_89, 3);
 lean_ctor_release(x_89, 4);
 x_94 = x_89;
} else {
 lean_dec_ref(x_89);
 x_94 = lean_box(0);
}
x_95 = lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__3;
x_96 = lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__4;
lean_inc_ref(x_90);
x_97 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_97, 0, x_90);
x_98 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_98, 0, x_90);
x_99 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_99, 0, x_97);
lean_ctor_set(x_99, 1, x_98);
x_100 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_100, 0, x_93);
x_101 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_101, 0, x_92);
x_102 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_102, 0, x_91);
if (lean_is_scalar(x_94)) {
 x_103 = lean_alloc_ctor(0, 5, 0);
} else {
 x_103 = x_94;
}
lean_ctor_set(x_103, 0, x_99);
lean_ctor_set(x_103, 1, x_95);
lean_ctor_set(x_103, 2, x_102);
lean_ctor_set(x_103, 3, x_101);
lean_ctor_set(x_103, 4, x_100);
x_104 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_104, 0, x_103);
lean_ctor_set(x_104, 1, x_96);
lean_inc_ref(x_104);
x_105 = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_105, 0, x_104);
lean_inc_ref(x_104);
x_106 = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_106, 0, x_104);
lean_inc_ref(x_104);
x_107 = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__7), 6, 1);
lean_closure_set(x_107, 0, x_104);
lean_inc_ref(x_104);
x_108 = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__9), 6, 1);
lean_closure_set(x_108, 0, x_104);
lean_inc_ref(x_104);
x_109 = lean_alloc_closure((void*)(l_StateT_map), 8, 3);
lean_closure_set(x_109, 0, lean_box(0));
lean_closure_set(x_109, 1, lean_box(0));
lean_closure_set(x_109, 2, x_104);
x_110 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_110, 0, x_109);
lean_ctor_set(x_110, 1, x_105);
lean_inc_ref(x_104);
x_111 = lean_alloc_closure((void*)(l_StateT_pure), 6, 3);
lean_closure_set(x_111, 0, lean_box(0));
lean_closure_set(x_111, 1, lean_box(0));
lean_closure_set(x_111, 2, x_104);
x_112 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_112, 0, x_110);
lean_ctor_set(x_112, 1, x_111);
lean_ctor_set(x_112, 2, x_106);
lean_ctor_set(x_112, 3, x_107);
lean_ctor_set(x_112, 4, x_108);
x_113 = lean_alloc_closure((void*)(l_StateT_bind), 8, 3);
lean_closure_set(x_113, 0, lean_box(0));
lean_closure_set(x_113, 1, lean_box(0));
lean_closure_set(x_113, 2, x_104);
x_114 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_114, 0, x_112);
lean_ctor_set(x_114, 1, x_113);
x_115 = lean_box(0);
x_116 = l_instInhabitedOfMonad___redArg(x_114, x_115);
x_117 = lean_alloc_closure((void*)(l_instInhabitedForall___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_117, 0, x_116);
x_118 = lean_panic_fn(x_117, x_1);
x_119 = lean_apply_7(x_118, x_2, x_3, x_4, x_5, x_6, x_7, lean_box(0));
return x_119;
}
}
else
{
lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_127; lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; lean_object* x_132; lean_object* x_133; lean_object* x_134; lean_object* x_135; lean_object* x_136; lean_object* x_137; lean_object* x_138; lean_object* x_139; lean_object* x_140; lean_object* x_141; lean_object* x_142; lean_object* x_143; lean_object* x_144; lean_object* x_145; lean_object* x_146; lean_object* x_147; lean_object* x_148; lean_object* x_149; lean_object* x_150; lean_object* x_151; lean_object* x_152; lean_object* x_153; lean_object* x_154; lean_object* x_155; lean_object* x_156; lean_object* x_157; lean_object* x_158; lean_object* x_159; lean_object* x_160; lean_object* x_161; lean_object* x_162; lean_object* x_163; lean_object* x_164; lean_object* x_165; 
x_120 = lean_ctor_get(x_12, 0);
x_121 = lean_ctor_get(x_12, 2);
x_122 = lean_ctor_get(x_12, 3);
x_123 = lean_ctor_get(x_12, 4);
lean_inc(x_123);
lean_inc(x_122);
lean_inc(x_121);
lean_inc(x_120);
lean_dec(x_12);
x_124 = lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__1;
x_125 = lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__2;
lean_inc_ref(x_120);
x_126 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_126, 0, x_120);
x_127 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_127, 0, x_120);
x_128 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_128, 0, x_126);
lean_ctor_set(x_128, 1, x_127);
x_129 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_129, 0, x_123);
x_130 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_130, 0, x_122);
x_131 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_131, 0, x_121);
x_132 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_132, 0, x_128);
lean_ctor_set(x_132, 1, x_124);
lean_ctor_set(x_132, 2, x_131);
lean_ctor_set(x_132, 3, x_130);
lean_ctor_set(x_132, 4, x_129);
lean_ctor_set(x_10, 1, x_125);
lean_ctor_set(x_10, 0, x_132);
x_133 = l_ReaderT_instMonad___redArg(x_10);
x_134 = lean_ctor_get(x_133, 0);
lean_inc_ref(x_134);
if (lean_is_exclusive(x_133)) {
 lean_ctor_release(x_133, 0);
 lean_ctor_release(x_133, 1);
 x_135 = x_133;
} else {
 lean_dec_ref(x_133);
 x_135 = lean_box(0);
}
x_136 = lean_ctor_get(x_134, 0);
lean_inc_ref(x_136);
x_137 = lean_ctor_get(x_134, 2);
lean_inc(x_137);
x_138 = lean_ctor_get(x_134, 3);
lean_inc(x_138);
x_139 = lean_ctor_get(x_134, 4);
lean_inc(x_139);
if (lean_is_exclusive(x_134)) {
 lean_ctor_release(x_134, 0);
 lean_ctor_release(x_134, 1);
 lean_ctor_release(x_134, 2);
 lean_ctor_release(x_134, 3);
 lean_ctor_release(x_134, 4);
 x_140 = x_134;
} else {
 lean_dec_ref(x_134);
 x_140 = lean_box(0);
}
x_141 = lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__3;
x_142 = lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__4;
lean_inc_ref(x_136);
x_143 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_143, 0, x_136);
x_144 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_144, 0, x_136);
x_145 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_145, 0, x_143);
lean_ctor_set(x_145, 1, x_144);
x_146 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_146, 0, x_139);
x_147 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_147, 0, x_138);
x_148 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_148, 0, x_137);
if (lean_is_scalar(x_140)) {
 x_149 = lean_alloc_ctor(0, 5, 0);
} else {
 x_149 = x_140;
}
lean_ctor_set(x_149, 0, x_145);
lean_ctor_set(x_149, 1, x_141);
lean_ctor_set(x_149, 2, x_148);
lean_ctor_set(x_149, 3, x_147);
lean_ctor_set(x_149, 4, x_146);
if (lean_is_scalar(x_135)) {
 x_150 = lean_alloc_ctor(0, 2, 0);
} else {
 x_150 = x_135;
}
lean_ctor_set(x_150, 0, x_149);
lean_ctor_set(x_150, 1, x_142);
lean_inc_ref(x_150);
x_151 = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_151, 0, x_150);
lean_inc_ref(x_150);
x_152 = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_152, 0, x_150);
lean_inc_ref(x_150);
x_153 = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__7), 6, 1);
lean_closure_set(x_153, 0, x_150);
lean_inc_ref(x_150);
x_154 = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__9), 6, 1);
lean_closure_set(x_154, 0, x_150);
lean_inc_ref(x_150);
x_155 = lean_alloc_closure((void*)(l_StateT_map), 8, 3);
lean_closure_set(x_155, 0, lean_box(0));
lean_closure_set(x_155, 1, lean_box(0));
lean_closure_set(x_155, 2, x_150);
x_156 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_156, 0, x_155);
lean_ctor_set(x_156, 1, x_151);
lean_inc_ref(x_150);
x_157 = lean_alloc_closure((void*)(l_StateT_pure), 6, 3);
lean_closure_set(x_157, 0, lean_box(0));
lean_closure_set(x_157, 1, lean_box(0));
lean_closure_set(x_157, 2, x_150);
x_158 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_158, 0, x_156);
lean_ctor_set(x_158, 1, x_157);
lean_ctor_set(x_158, 2, x_152);
lean_ctor_set(x_158, 3, x_153);
lean_ctor_set(x_158, 4, x_154);
x_159 = lean_alloc_closure((void*)(l_StateT_bind), 8, 3);
lean_closure_set(x_159, 0, lean_box(0));
lean_closure_set(x_159, 1, lean_box(0));
lean_closure_set(x_159, 2, x_150);
x_160 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_160, 0, x_158);
lean_ctor_set(x_160, 1, x_159);
x_161 = lean_box(0);
x_162 = l_instInhabitedOfMonad___redArg(x_160, x_161);
x_163 = lean_alloc_closure((void*)(l_instInhabitedForall___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_163, 0, x_162);
x_164 = lean_panic_fn(x_163, x_1);
x_165 = lean_apply_7(x_164, x_2, x_3, x_4, x_5, x_6, x_7, lean_box(0));
return x_165;
}
}
else
{
lean_object* x_166; lean_object* x_167; lean_object* x_168; lean_object* x_169; lean_object* x_170; lean_object* x_171; lean_object* x_172; lean_object* x_173; lean_object* x_174; lean_object* x_175; lean_object* x_176; lean_object* x_177; lean_object* x_178; lean_object* x_179; lean_object* x_180; lean_object* x_181; lean_object* x_182; lean_object* x_183; lean_object* x_184; lean_object* x_185; lean_object* x_186; lean_object* x_187; lean_object* x_188; lean_object* x_189; lean_object* x_190; lean_object* x_191; lean_object* x_192; lean_object* x_193; lean_object* x_194; lean_object* x_195; lean_object* x_196; lean_object* x_197; lean_object* x_198; lean_object* x_199; lean_object* x_200; lean_object* x_201; lean_object* x_202; lean_object* x_203; lean_object* x_204; lean_object* x_205; lean_object* x_206; lean_object* x_207; lean_object* x_208; lean_object* x_209; lean_object* x_210; lean_object* x_211; lean_object* x_212; lean_object* x_213; lean_object* x_214; 
x_166 = lean_ctor_get(x_10, 0);
lean_inc(x_166);
lean_dec(x_10);
x_167 = lean_ctor_get(x_166, 0);
lean_inc_ref(x_167);
x_168 = lean_ctor_get(x_166, 2);
lean_inc(x_168);
x_169 = lean_ctor_get(x_166, 3);
lean_inc(x_169);
x_170 = lean_ctor_get(x_166, 4);
lean_inc(x_170);
if (lean_is_exclusive(x_166)) {
 lean_ctor_release(x_166, 0);
 lean_ctor_release(x_166, 1);
 lean_ctor_release(x_166, 2);
 lean_ctor_release(x_166, 3);
 lean_ctor_release(x_166, 4);
 x_171 = x_166;
} else {
 lean_dec_ref(x_166);
 x_171 = lean_box(0);
}
x_172 = lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__1;
x_173 = lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__2;
lean_inc_ref(x_167);
x_174 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_174, 0, x_167);
x_175 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_175, 0, x_167);
x_176 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_176, 0, x_174);
lean_ctor_set(x_176, 1, x_175);
x_177 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_177, 0, x_170);
x_178 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_178, 0, x_169);
x_179 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_179, 0, x_168);
if (lean_is_scalar(x_171)) {
 x_180 = lean_alloc_ctor(0, 5, 0);
} else {
 x_180 = x_171;
}
lean_ctor_set(x_180, 0, x_176);
lean_ctor_set(x_180, 1, x_172);
lean_ctor_set(x_180, 2, x_179);
lean_ctor_set(x_180, 3, x_178);
lean_ctor_set(x_180, 4, x_177);
x_181 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_181, 0, x_180);
lean_ctor_set(x_181, 1, x_173);
x_182 = l_ReaderT_instMonad___redArg(x_181);
x_183 = lean_ctor_get(x_182, 0);
lean_inc_ref(x_183);
if (lean_is_exclusive(x_182)) {
 lean_ctor_release(x_182, 0);
 lean_ctor_release(x_182, 1);
 x_184 = x_182;
} else {
 lean_dec_ref(x_182);
 x_184 = lean_box(0);
}
x_185 = lean_ctor_get(x_183, 0);
lean_inc_ref(x_185);
x_186 = lean_ctor_get(x_183, 2);
lean_inc(x_186);
x_187 = lean_ctor_get(x_183, 3);
lean_inc(x_187);
x_188 = lean_ctor_get(x_183, 4);
lean_inc(x_188);
if (lean_is_exclusive(x_183)) {
 lean_ctor_release(x_183, 0);
 lean_ctor_release(x_183, 1);
 lean_ctor_release(x_183, 2);
 lean_ctor_release(x_183, 3);
 lean_ctor_release(x_183, 4);
 x_189 = x_183;
} else {
 lean_dec_ref(x_183);
 x_189 = lean_box(0);
}
x_190 = lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__3;
x_191 = lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__4;
lean_inc_ref(x_185);
x_192 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(x_192, 0, x_185);
x_193 = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_193, 0, x_185);
x_194 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_194, 0, x_192);
lean_ctor_set(x_194, 1, x_193);
x_195 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_195, 0, x_188);
x_196 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(x_196, 0, x_187);
x_197 = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_197, 0, x_186);
if (lean_is_scalar(x_189)) {
 x_198 = lean_alloc_ctor(0, 5, 0);
} else {
 x_198 = x_189;
}
lean_ctor_set(x_198, 0, x_194);
lean_ctor_set(x_198, 1, x_190);
lean_ctor_set(x_198, 2, x_197);
lean_ctor_set(x_198, 3, x_196);
lean_ctor_set(x_198, 4, x_195);
if (lean_is_scalar(x_184)) {
 x_199 = lean_alloc_ctor(0, 2, 0);
} else {
 x_199 = x_184;
}
lean_ctor_set(x_199, 0, x_198);
lean_ctor_set(x_199, 1, x_191);
lean_inc_ref(x_199);
x_200 = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__1), 6, 1);
lean_closure_set(x_200, 0, x_199);
lean_inc_ref(x_199);
x_201 = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__4), 6, 1);
lean_closure_set(x_201, 0, x_199);
lean_inc_ref(x_199);
x_202 = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__7), 6, 1);
lean_closure_set(x_202, 0, x_199);
lean_inc_ref(x_199);
x_203 = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__9), 6, 1);
lean_closure_set(x_203, 0, x_199);
lean_inc_ref(x_199);
x_204 = lean_alloc_closure((void*)(l_StateT_map), 8, 3);
lean_closure_set(x_204, 0, lean_box(0));
lean_closure_set(x_204, 1, lean_box(0));
lean_closure_set(x_204, 2, x_199);
x_205 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_205, 0, x_204);
lean_ctor_set(x_205, 1, x_200);
lean_inc_ref(x_199);
x_206 = lean_alloc_closure((void*)(l_StateT_pure), 6, 3);
lean_closure_set(x_206, 0, lean_box(0));
lean_closure_set(x_206, 1, lean_box(0));
lean_closure_set(x_206, 2, x_199);
x_207 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_207, 0, x_205);
lean_ctor_set(x_207, 1, x_206);
lean_ctor_set(x_207, 2, x_201);
lean_ctor_set(x_207, 3, x_202);
lean_ctor_set(x_207, 4, x_203);
x_208 = lean_alloc_closure((void*)(l_StateT_bind), 8, 3);
lean_closure_set(x_208, 0, lean_box(0));
lean_closure_set(x_208, 1, lean_box(0));
lean_closure_set(x_208, 2, x_199);
x_209 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_209, 0, x_207);
lean_ctor_set(x_209, 1, x_208);
x_210 = lean_box(0);
x_211 = l_instInhabitedOfMonad___redArg(x_209, x_210);
x_212 = lean_alloc_closure((void*)(l_instInhabitedForall___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_212, 0, x_211);
x_213 = lean_panic_fn(x_212, x_1);
x_214 = lean_apply_7(x_213, x_2, x_3, x_4, x_5, x_6, x_7, lean_box(0));
return x_214;
}
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Mathlib.Lean.Meta.RefinedDiscrTree.Encode", 41, 41);
return x_1;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("_private.Mathlib.Lean.Meta.RefinedDiscrTree.Encode.0.Lean.Meta.RefinedDiscrTree.encodingStepAux.go", 98, 98);
return x_1;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("unreachable code has been reached", 33, 33);
return x_1;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_1 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__2;
x_2 = lean_unsigned_to_nat(19u);
x_3 = lean_unsigned_to_nat(126u);
x_4 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__1;
x_5 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__0;
x_6 = l_mkPanicMessageWithDecl(x_5, x_4, x_3, x_2, x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___lam__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___lam__1(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go(lean_object* x_1, lean_object* x_2, uint8_t x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; 
x_11 = l_Lean_Expr_getAppFn(x_1);
switch (lean_obj_tag(x_11)) {
case 4:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
lean_dec_ref(x_11);
if (x_3 == 0)
{
lean_object* x_43; 
lean_inc_ref(x_1);
x_43 = l_Lean_Meta_LazyDiscrTree_MatchClone_toNatLit_x3f(x_1);
if (lean_obj_tag(x_43) == 1)
{
uint8_t x_44; 
lean_dec(x_12);
lean_dec_ref(x_6);
lean_dec(x_4);
lean_dec(x_2);
lean_dec_ref(x_1);
x_44 = !lean_is_exclusive(x_43);
if (x_44 == 0)
{
lean_object* x_45; lean_object* x_46; 
lean_ctor_set_tag(x_43, 6);
x_45 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_45, 0, x_43);
lean_ctor_set(x_45, 1, x_5);
x_46 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_46, 0, x_45);
return x_46;
}
else
{
lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; 
x_47 = lean_ctor_get(x_43, 0);
lean_inc(x_47);
lean_dec(x_43);
x_48 = lean_alloc_ctor(6, 1, 0);
lean_ctor_set(x_48, 0, x_47);
x_49 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_49, 0, x_48);
lean_ctor_set(x_49, 1, x_5);
x_50 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_50, 0, x_49);
return x_50;
}
}
else
{
lean_dec(x_43);
x_20 = x_4;
x_21 = x_5;
x_22 = x_6;
x_23 = lean_box(0);
goto block_42;
}
}
else
{
x_20 = x_4;
x_21 = x_5;
x_22 = x_6;
x_23 = lean_box(0);
goto block_42;
}
block_19:
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_15 = l_Lean_Expr_getAppNumArgs(x_1);
lean_dec_ref(x_1);
x_16 = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(x_16, 0, x_12);
lean_ctor_set(x_16, 1, x_15);
x_17 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_17, 0, x_16);
lean_ctor_set(x_17, 1, x_13);
x_18 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_18, 0, x_17);
return x_18;
}
block_42:
{
lean_object* x_24; lean_object* x_25; uint8_t x_26; 
x_24 = l_Lean_Expr_getAppNumArgs(x_1);
x_25 = lean_unsigned_to_nat(0u);
x_26 = lean_nat_dec_eq(x_24, x_25);
lean_dec(x_24);
if (x_26 == 0)
{
lean_object* x_27; lean_object* x_28; 
x_27 = l_List_appendTR___redArg(x_2, x_20);
lean_inc_ref(x_1);
x_28 = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkExprInfo___redArg(x_1, x_27, x_22);
lean_dec_ref(x_22);
if (lean_obj_tag(x_28) == 0)
{
lean_object* x_29; uint8_t x_30; 
x_29 = lean_ctor_get(x_28, 0);
lean_inc(x_29);
lean_dec_ref(x_28);
x_30 = !lean_is_exclusive(x_21);
if (x_30 == 0)
{
lean_object* x_31; lean_object* x_32; 
x_31 = lean_ctor_get(x_21, 0);
lean_dec(x_31);
x_32 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_32, 0, x_29);
lean_ctor_set(x_21, 0, x_32);
x_13 = x_21;
x_14 = lean_box(0);
goto block_19;
}
else
{
lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; 
x_33 = lean_ctor_get(x_21, 1);
x_34 = lean_ctor_get(x_21, 2);
x_35 = lean_ctor_get(x_21, 3);
x_36 = lean_ctor_get(x_21, 4);
lean_inc(x_36);
lean_inc(x_35);
lean_inc(x_34);
lean_inc(x_33);
lean_dec(x_21);
x_37 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_37, 0, x_29);
x_38 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_38, 0, x_37);
lean_ctor_set(x_38, 1, x_33);
lean_ctor_set(x_38, 2, x_34);
lean_ctor_set(x_38, 3, x_35);
lean_ctor_set(x_38, 4, x_36);
x_13 = x_38;
x_14 = lean_box(0);
goto block_19;
}
}
else
{
uint8_t x_39; 
lean_dec_ref(x_21);
lean_dec(x_12);
lean_dec_ref(x_1);
x_39 = !lean_is_exclusive(x_28);
if (x_39 == 0)
{
return x_28;
}
else
{
lean_object* x_40; lean_object* x_41; 
x_40 = lean_ctor_get(x_28, 0);
lean_inc(x_40);
lean_dec(x_28);
x_41 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_41, 0, x_40);
return x_41;
}
}
}
else
{
lean_dec_ref(x_22);
lean_dec(x_20);
lean_dec(x_2);
x_13 = x_21;
x_14 = lean_box(0);
goto block_19;
}
}
}
case 11:
{
lean_object* x_51; lean_object* x_52; lean_object* x_53; 
x_51 = lean_ctor_get(x_11, 0);
lean_inc(x_51);
x_52 = lean_ctor_get(x_11, 1);
lean_inc(x_52);
lean_dec_ref(x_11);
lean_inc(x_4);
lean_inc_ref(x_1);
x_53 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___lam__0(x_2, x_1, x_4, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_4);
if (lean_obj_tag(x_53) == 0)
{
uint8_t x_54; 
x_54 = !lean_is_exclusive(x_53);
if (x_54 == 0)
{
lean_object* x_55; uint8_t x_56; 
x_55 = lean_ctor_get(x_53, 0);
x_56 = !lean_is_exclusive(x_55);
if (x_56 == 0)
{
lean_object* x_57; lean_object* x_58; lean_object* x_59; 
x_57 = lean_ctor_get(x_55, 0);
lean_dec(x_57);
x_58 = l_Lean_Expr_getAppNumArgs(x_1);
lean_dec_ref(x_1);
x_59 = lean_alloc_ctor(10, 3, 0);
lean_ctor_set(x_59, 0, x_51);
lean_ctor_set(x_59, 1, x_52);
lean_ctor_set(x_59, 2, x_58);
lean_ctor_set(x_55, 0, x_59);
return x_53;
}
else
{
lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; 
x_60 = lean_ctor_get(x_55, 1);
lean_inc(x_60);
lean_dec(x_55);
x_61 = l_Lean_Expr_getAppNumArgs(x_1);
lean_dec_ref(x_1);
x_62 = lean_alloc_ctor(10, 3, 0);
lean_ctor_set(x_62, 0, x_51);
lean_ctor_set(x_62, 1, x_52);
lean_ctor_set(x_62, 2, x_61);
x_63 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_63, 0, x_62);
lean_ctor_set(x_63, 1, x_60);
lean_ctor_set(x_53, 0, x_63);
return x_53;
}
}
else
{
lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; 
x_64 = lean_ctor_get(x_53, 0);
lean_inc(x_64);
lean_dec(x_53);
x_65 = lean_ctor_get(x_64, 1);
lean_inc(x_65);
if (lean_is_exclusive(x_64)) {
 lean_ctor_release(x_64, 0);
 lean_ctor_release(x_64, 1);
 x_66 = x_64;
} else {
 lean_dec_ref(x_64);
 x_66 = lean_box(0);
}
x_67 = l_Lean_Expr_getAppNumArgs(x_1);
lean_dec_ref(x_1);
x_68 = lean_alloc_ctor(10, 3, 0);
lean_ctor_set(x_68, 0, x_51);
lean_ctor_set(x_68, 1, x_52);
lean_ctor_set(x_68, 2, x_67);
if (lean_is_scalar(x_66)) {
 x_69 = lean_alloc_ctor(0, 2, 0);
} else {
 x_69 = x_66;
}
lean_ctor_set(x_69, 0, x_68);
lean_ctor_set(x_69, 1, x_65);
x_70 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_70, 0, x_69);
return x_70;
}
}
else
{
uint8_t x_71; 
lean_dec(x_52);
lean_dec(x_51);
lean_dec_ref(x_1);
x_71 = !lean_is_exclusive(x_53);
if (x_71 == 0)
{
return x_53;
}
else
{
lean_object* x_72; lean_object* x_73; 
x_72 = lean_ctor_get(x_53, 0);
lean_inc(x_72);
lean_dec(x_53);
x_73 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_73, 0, x_72);
return x_73;
}
}
}
case 1:
{
lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_95; lean_object* x_96; uint8_t x_97; 
x_74 = lean_ctor_get(x_11, 0);
lean_inc(x_74);
lean_dec_ref(x_11);
lean_inc(x_74);
x_75 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___lam__1___boxed), 2, 1);
lean_closure_set(x_75, 0, x_74);
lean_inc(x_4);
lean_inc(x_2);
x_76 = l_List_appendTR___redArg(x_2, x_4);
x_95 = l_Lean_Expr_getAppNumArgs(x_1);
x_96 = lean_unsigned_to_nat(0u);
x_97 = lean_nat_dec_eq(x_95, x_96);
lean_dec(x_95);
if (x_97 == 0)
{
lean_object* x_98; 
lean_inc(x_4);
lean_inc_ref(x_1);
x_98 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___lam__0(x_2, x_1, x_4, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_4);
if (lean_obj_tag(x_98) == 0)
{
lean_object* x_99; lean_object* x_100; 
x_99 = lean_ctor_get(x_98, 0);
lean_inc(x_99);
lean_dec_ref(x_98);
x_100 = lean_ctor_get(x_99, 1);
lean_inc(x_100);
lean_dec(x_99);
x_77 = x_100;
x_78 = lean_box(0);
goto block_94;
}
else
{
uint8_t x_101; 
lean_dec(x_76);
lean_dec_ref(x_75);
lean_dec(x_74);
lean_dec_ref(x_1);
x_101 = !lean_is_exclusive(x_98);
if (x_101 == 0)
{
return x_98;
}
else
{
lean_object* x_102; lean_object* x_103; 
x_102 = lean_ctor_get(x_98, 0);
lean_inc(x_102);
lean_dec(x_98);
x_103 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_103, 0, x_102);
return x_103;
}
}
}
else
{
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_4);
lean_dec(x_2);
x_77 = x_5;
x_78 = lean_box(0);
goto block_94;
}
block_94:
{
lean_object* x_79; 
x_79 = l_List_findIdx_x3f___redArg(x_75, x_76);
if (lean_obj_tag(x_79) == 1)
{
uint8_t x_80; 
lean_dec(x_74);
x_80 = !lean_is_exclusive(x_79);
if (x_80 == 0)
{
lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; 
x_81 = lean_ctor_get(x_79, 0);
x_82 = l_Lean_Expr_getAppNumArgs(x_1);
lean_dec_ref(x_1);
x_83 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_83, 0, x_81);
lean_ctor_set(x_83, 1, x_82);
x_84 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_84, 0, x_83);
lean_ctor_set(x_84, 1, x_77);
lean_ctor_set_tag(x_79, 0);
lean_ctor_set(x_79, 0, x_84);
return x_79;
}
else
{
lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; 
x_85 = lean_ctor_get(x_79, 0);
lean_inc(x_85);
lean_dec(x_79);
x_86 = l_Lean_Expr_getAppNumArgs(x_1);
lean_dec_ref(x_1);
x_87 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_87, 0, x_85);
lean_ctor_set(x_87, 1, x_86);
x_88 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_88, 0, x_87);
lean_ctor_set(x_88, 1, x_77);
x_89 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_89, 0, x_88);
return x_89;
}
}
else
{
lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; 
lean_dec(x_79);
x_90 = l_Lean_Expr_getAppNumArgs(x_1);
lean_dec_ref(x_1);
x_91 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_91, 0, x_74);
lean_ctor_set(x_91, 1, x_90);
x_92 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_92, 0, x_91);
lean_ctor_set(x_92, 1, x_77);
x_93 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_93, 0, x_92);
return x_93;
}
}
}
case 2:
{
lean_object* x_104; uint8_t x_105; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_4);
lean_dec(x_2);
x_104 = lean_ctor_get(x_11, 0);
lean_inc(x_104);
lean_dec_ref(x_11);
x_105 = l_Lean_Expr_isApp(x_1);
lean_dec_ref(x_1);
if (x_105 == 0)
{
lean_object* x_106; 
x_106 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar___redArg(x_104, x_5);
return x_106;
}
else
{
lean_object* x_107; lean_object* x_108; lean_object* x_109; 
lean_dec(x_104);
x_107 = lean_box(0);
x_108 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_108, 0, x_107);
lean_ctor_set(x_108, 1, x_5);
x_109 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_109, 0, x_108);
return x_109;
}
}
case 7:
{
lean_object* x_110; 
lean_dec_ref(x_11);
lean_inc(x_4);
x_110 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___lam__0(x_2, x_1, x_4, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_4);
if (lean_obj_tag(x_110) == 0)
{
uint8_t x_111; 
x_111 = !lean_is_exclusive(x_110);
if (x_111 == 0)
{
lean_object* x_112; uint8_t x_113; 
x_112 = lean_ctor_get(x_110, 0);
x_113 = !lean_is_exclusive(x_112);
if (x_113 == 0)
{
lean_object* x_114; lean_object* x_115; 
x_114 = lean_ctor_get(x_112, 0);
lean_dec(x_114);
x_115 = lean_box(9);
lean_ctor_set(x_112, 0, x_115);
return x_110;
}
else
{
lean_object* x_116; lean_object* x_117; lean_object* x_118; 
x_116 = lean_ctor_get(x_112, 1);
lean_inc(x_116);
lean_dec(x_112);
x_117 = lean_box(9);
x_118 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_118, 0, x_117);
lean_ctor_set(x_118, 1, x_116);
lean_ctor_set(x_110, 0, x_118);
return x_110;
}
}
else
{
lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; 
x_119 = lean_ctor_get(x_110, 0);
lean_inc(x_119);
lean_dec(x_110);
x_120 = lean_ctor_get(x_119, 1);
lean_inc(x_120);
if (lean_is_exclusive(x_119)) {
 lean_ctor_release(x_119, 0);
 lean_ctor_release(x_119, 1);
 x_121 = x_119;
} else {
 lean_dec_ref(x_119);
 x_121 = lean_box(0);
}
x_122 = lean_box(9);
if (lean_is_scalar(x_121)) {
 x_123 = lean_alloc_ctor(0, 2, 0);
} else {
 x_123 = x_121;
}
lean_ctor_set(x_123, 0, x_122);
lean_ctor_set(x_123, 1, x_120);
x_124 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_124, 0, x_123);
return x_124;
}
}
else
{
uint8_t x_125; 
x_125 = !lean_is_exclusive(x_110);
if (x_125 == 0)
{
return x_110;
}
else
{
lean_object* x_126; lean_object* x_127; 
x_126 = lean_ctor_get(x_110, 0);
lean_inc(x_126);
lean_dec(x_110);
x_127 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_127, 0, x_126);
return x_127;
}
}
}
case 9:
{
lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_4);
lean_dec(x_2);
lean_dec_ref(x_1);
x_128 = lean_ctor_get(x_11, 0);
lean_inc_ref(x_128);
lean_dec_ref(x_11);
x_129 = lean_alloc_ctor(6, 1, 0);
lean_ctor_set(x_129, 0, x_128);
x_130 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_130, 0, x_129);
lean_ctor_set(x_130, 1, x_5);
x_131 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_131, 0, x_130);
return x_131;
}
case 3:
{
lean_object* x_132; lean_object* x_133; lean_object* x_134; 
lean_dec_ref(x_11);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_4);
lean_dec(x_2);
lean_dec_ref(x_1);
x_132 = lean_box(7);
x_133 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_133, 0, x_132);
lean_ctor_set(x_133, 1, x_5);
x_134 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_134, 0, x_133);
return x_134;
}
case 8:
{
lean_object* x_135; lean_object* x_136; lean_object* x_137; 
lean_dec_ref(x_11);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_4);
lean_dec(x_2);
lean_dec_ref(x_1);
x_135 = lean_box(2);
x_136 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_136, 0, x_135);
lean_ctor_set(x_136, 1, x_5);
x_137 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_137, 0, x_136);
return x_137;
}
case 6:
{
lean_object* x_138; lean_object* x_139; lean_object* x_140; 
lean_dec_ref(x_11);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_4);
lean_dec(x_2);
lean_dec_ref(x_1);
x_138 = lean_box(2);
x_139 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_139, 0, x_138);
lean_ctor_set(x_139, 1, x_5);
x_140 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_140, 0, x_139);
return x_140;
}
default: 
{
lean_object* x_141; lean_object* x_142; 
lean_dec_ref(x_11);
lean_dec(x_2);
lean_dec_ref(x_1);
x_141 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__3;
x_142 = lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0(x_141, x_4, x_5, x_6, x_7, x_8, x_9);
return x_142;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_4);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
uint8_t x_11; lean_object* x_12; 
x_11 = lean_unbox(x_3);
x_12 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go(x_1, x_2, x_11, x_4, x_5, x_6, x_7, x_8, x_9);
return x_12;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg___lam__0___boxed), 2, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux(lean_object* x_1, lean_object* x_2, uint8_t x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; 
lean_inc(x_2);
x_11 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_11) == 0)
{
lean_object* x_12; 
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
if (lean_obj_tag(x_2) == 0)
{
lean_dec(x_12);
return x_11;
}
else
{
uint8_t x_13; 
x_13 = !lean_is_exclusive(x_11);
if (x_13 == 0)
{
lean_object* x_14; uint8_t x_15; 
x_14 = lean_ctor_get(x_11, 0);
lean_dec(x_14);
x_15 = !lean_is_exclusive(x_12);
if (x_15 == 0)
{
uint8_t x_16; 
x_16 = !lean_is_exclusive(x_2);
if (x_16 == 0)
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; uint8_t x_21; 
x_17 = lean_ctor_get(x_12, 1);
x_18 = lean_ctor_get(x_12, 0);
x_19 = lean_ctor_get(x_2, 1);
x_20 = lean_ctor_get(x_2, 0);
lean_dec(x_20);
x_21 = !lean_is_exclusive(x_17);
if (x_21 == 0)
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; 
x_22 = lean_ctor_get(x_17, 4);
lean_dec(x_22);
x_23 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux___closed__0;
x_24 = lean_box(0);
lean_ctor_set(x_2, 1, x_24);
lean_ctor_set(x_2, 0, x_18);
x_25 = l_List_foldl___redArg(x_23, x_2, x_19);
lean_ctor_set(x_17, 4, x_25);
x_26 = lean_box(8);
lean_ctor_set(x_12, 0, x_26);
return x_11;
}
else
{
lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; 
x_27 = lean_ctor_get(x_17, 0);
x_28 = lean_ctor_get(x_17, 1);
x_29 = lean_ctor_get(x_17, 2);
x_30 = lean_ctor_get(x_17, 3);
lean_inc(x_30);
lean_inc(x_29);
lean_inc(x_28);
lean_inc(x_27);
lean_dec(x_17);
x_31 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux___closed__0;
x_32 = lean_box(0);
lean_ctor_set(x_2, 1, x_32);
lean_ctor_set(x_2, 0, x_18);
x_33 = l_List_foldl___redArg(x_31, x_2, x_19);
x_34 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_34, 0, x_27);
lean_ctor_set(x_34, 1, x_28);
lean_ctor_set(x_34, 2, x_29);
lean_ctor_set(x_34, 3, x_30);
lean_ctor_set(x_34, 4, x_33);
x_35 = lean_box(8);
lean_ctor_set(x_12, 1, x_34);
lean_ctor_set(x_12, 0, x_35);
return x_11;
}
}
else
{
lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; 
x_36 = lean_ctor_get(x_12, 1);
x_37 = lean_ctor_get(x_12, 0);
x_38 = lean_ctor_get(x_2, 1);
lean_inc(x_38);
lean_dec(x_2);
x_39 = lean_ctor_get(x_36, 0);
lean_inc(x_39);
x_40 = lean_ctor_get(x_36, 1);
lean_inc(x_40);
x_41 = lean_ctor_get(x_36, 2);
lean_inc_ref(x_41);
x_42 = lean_ctor_get(x_36, 3);
lean_inc(x_42);
if (lean_is_exclusive(x_36)) {
 lean_ctor_release(x_36, 0);
 lean_ctor_release(x_36, 1);
 lean_ctor_release(x_36, 2);
 lean_ctor_release(x_36, 3);
 lean_ctor_release(x_36, 4);
 x_43 = x_36;
} else {
 lean_dec_ref(x_36);
 x_43 = lean_box(0);
}
x_44 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux___closed__0;
x_45 = lean_box(0);
x_46 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_46, 0, x_37);
lean_ctor_set(x_46, 1, x_45);
x_47 = l_List_foldl___redArg(x_44, x_46, x_38);
if (lean_is_scalar(x_43)) {
 x_48 = lean_alloc_ctor(0, 5, 0);
} else {
 x_48 = x_43;
}
lean_ctor_set(x_48, 0, x_39);
lean_ctor_set(x_48, 1, x_40);
lean_ctor_set(x_48, 2, x_41);
lean_ctor_set(x_48, 3, x_42);
lean_ctor_set(x_48, 4, x_47);
x_49 = lean_box(8);
lean_ctor_set(x_12, 1, x_48);
lean_ctor_set(x_12, 0, x_49);
return x_11;
}
}
else
{
lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; 
x_50 = lean_ctor_get(x_12, 1);
x_51 = lean_ctor_get(x_12, 0);
lean_inc(x_50);
lean_inc(x_51);
lean_dec(x_12);
x_52 = lean_ctor_get(x_2, 1);
lean_inc(x_52);
if (lean_is_exclusive(x_2)) {
 lean_ctor_release(x_2, 0);
 lean_ctor_release(x_2, 1);
 x_53 = x_2;
} else {
 lean_dec_ref(x_2);
 x_53 = lean_box(0);
}
x_54 = lean_ctor_get(x_50, 0);
lean_inc(x_54);
x_55 = lean_ctor_get(x_50, 1);
lean_inc(x_55);
x_56 = lean_ctor_get(x_50, 2);
lean_inc_ref(x_56);
x_57 = lean_ctor_get(x_50, 3);
lean_inc(x_57);
if (lean_is_exclusive(x_50)) {
 lean_ctor_release(x_50, 0);
 lean_ctor_release(x_50, 1);
 lean_ctor_release(x_50, 2);
 lean_ctor_release(x_50, 3);
 lean_ctor_release(x_50, 4);
 x_58 = x_50;
} else {
 lean_dec_ref(x_50);
 x_58 = lean_box(0);
}
x_59 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux___closed__0;
x_60 = lean_box(0);
if (lean_is_scalar(x_53)) {
 x_61 = lean_alloc_ctor(1, 2, 0);
} else {
 x_61 = x_53;
}
lean_ctor_set(x_61, 0, x_51);
lean_ctor_set(x_61, 1, x_60);
x_62 = l_List_foldl___redArg(x_59, x_61, x_52);
if (lean_is_scalar(x_58)) {
 x_63 = lean_alloc_ctor(0, 5, 0);
} else {
 x_63 = x_58;
}
lean_ctor_set(x_63, 0, x_54);
lean_ctor_set(x_63, 1, x_55);
lean_ctor_set(x_63, 2, x_56);
lean_ctor_set(x_63, 3, x_57);
lean_ctor_set(x_63, 4, x_62);
x_64 = lean_box(8);
x_65 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_65, 0, x_64);
lean_ctor_set(x_65, 1, x_63);
lean_ctor_set(x_11, 0, x_65);
return x_11;
}
}
else
{
lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; 
lean_dec(x_11);
x_66 = lean_ctor_get(x_12, 1);
lean_inc(x_66);
x_67 = lean_ctor_get(x_12, 0);
lean_inc(x_67);
if (lean_is_exclusive(x_12)) {
 lean_ctor_release(x_12, 0);
 lean_ctor_release(x_12, 1);
 x_68 = x_12;
} else {
 lean_dec_ref(x_12);
 x_68 = lean_box(0);
}
x_69 = lean_ctor_get(x_2, 1);
lean_inc(x_69);
if (lean_is_exclusive(x_2)) {
 lean_ctor_release(x_2, 0);
 lean_ctor_release(x_2, 1);
 x_70 = x_2;
} else {
 lean_dec_ref(x_2);
 x_70 = lean_box(0);
}
x_71 = lean_ctor_get(x_66, 0);
lean_inc(x_71);
x_72 = lean_ctor_get(x_66, 1);
lean_inc(x_72);
x_73 = lean_ctor_get(x_66, 2);
lean_inc_ref(x_73);
x_74 = lean_ctor_get(x_66, 3);
lean_inc(x_74);
if (lean_is_exclusive(x_66)) {
 lean_ctor_release(x_66, 0);
 lean_ctor_release(x_66, 1);
 lean_ctor_release(x_66, 2);
 lean_ctor_release(x_66, 3);
 lean_ctor_release(x_66, 4);
 x_75 = x_66;
} else {
 lean_dec_ref(x_66);
 x_75 = lean_box(0);
}
x_76 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux___closed__0;
x_77 = lean_box(0);
if (lean_is_scalar(x_70)) {
 x_78 = lean_alloc_ctor(1, 2, 0);
} else {
 x_78 = x_70;
}
lean_ctor_set(x_78, 0, x_67);
lean_ctor_set(x_78, 1, x_77);
x_79 = l_List_foldl___redArg(x_76, x_78, x_69);
if (lean_is_scalar(x_75)) {
 x_80 = lean_alloc_ctor(0, 5, 0);
} else {
 x_80 = x_75;
}
lean_ctor_set(x_80, 0, x_71);
lean_ctor_set(x_80, 1, x_72);
lean_ctor_set(x_80, 2, x_73);
lean_ctor_set(x_80, 3, x_74);
lean_ctor_set(x_80, 4, x_79);
x_81 = lean_box(8);
if (lean_is_scalar(x_68)) {
 x_82 = lean_alloc_ctor(0, 2, 0);
} else {
 x_82 = x_68;
}
lean_ctor_set(x_82, 0, x_81);
lean_ctor_set(x_82, 1, x_80);
x_83 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_83, 0, x_82);
return x_83;
}
}
}
else
{
lean_dec(x_2);
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
uint8_t x_11; lean_object* x_12; 
x_11 = lean_unbox(x_3);
x_12 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux(x_1, x_2, x_11, x_4, x_5, x_6, x_7, x_8, x_9);
return x_12;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_isStarWithArg(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_2) == 5)
{
lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_3 = lean_ctor_get(x_2, 0);
x_4 = lean_ctor_get(x_2, 1);
x_5 = lean_expr_eqv(x_4, x_1);
if (x_5 == 0)
{
x_2 = x_3;
goto _start;
}
else
{
lean_object* x_7; uint8_t x_8; 
x_7 = l_Lean_Expr_getAppFn(x_3);
x_8 = l_Lean_Expr_isMVar(x_7);
lean_dec_ref(x_7);
return x_8;
}
}
else
{
uint8_t x_9; 
x_9 = 0;
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_isStarWithArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_isStarWithArg(x_1, x_2);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_9; lean_object* x_10; 
x_9 = lean_box(0);
x_10 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_10, 0, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
return x_1;
}
else
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_ctor_get(x_2, 1);
x_5 = lean_ctor_get(x_2, 0);
lean_dec(x_5);
x_6 = lean_box(8);
lean_ctor_set(x_2, 1, x_1);
lean_ctor_set(x_2, 0, x_6);
x_1 = x_2;
x_2 = x_4;
goto _start;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_ctor_get(x_2, 1);
lean_inc(x_8);
lean_dec(x_2);
x_9 = lean_box(8);
x_10 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_10, 0, x_9);
lean_ctor_set(x_10, 1, x_1);
x_1 = x_10;
x_2 = x_8;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities(lean_object* x_1, lean_object* x_2, uint8_t x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_17; lean_object* x_18; lean_object* x_21; lean_object* x_22; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; uint8_t x_29; lean_object* x_32; lean_object* x_33; lean_object* x_45; lean_object* x_51; 
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc_ref(x_4);
lean_inc(x_5);
lean_inc(x_2);
lean_inc_ref(x_1);
x_51 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go(x_1, x_2, x_3, x_5, x_4, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_51) == 0)
{
lean_object* x_52; 
x_52 = lean_ctor_get(x_51, 0);
lean_inc(x_52);
if (lean_obj_tag(x_2) == 0)
{
lean_dec(x_52);
x_45 = x_51;
goto block_50;
}
else
{
uint8_t x_53; 
lean_dec_ref(x_51);
x_53 = !lean_is_exclusive(x_52);
if (x_53 == 0)
{
lean_object* x_54; uint8_t x_55; 
x_54 = lean_ctor_get(x_52, 1);
x_55 = !lean_is_exclusive(x_54);
if (x_55 == 0)
{
lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; 
x_56 = lean_ctor_get(x_52, 0);
x_57 = lean_ctor_get(x_2, 1);
x_58 = lean_ctor_get(x_54, 4);
lean_dec(x_58);
x_59 = lean_box(0);
x_60 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_60, 0, x_56);
lean_ctor_set(x_60, 1, x_59);
lean_inc(x_57);
x_61 = lp_mathlib_List_foldl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_spec__0(x_60, x_57);
lean_ctor_set(x_54, 4, x_61);
x_62 = lean_box(8);
lean_ctor_set(x_52, 0, x_62);
x_32 = x_52;
x_33 = lean_box(0);
goto block_44;
}
else
{
lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; 
x_63 = lean_ctor_get(x_52, 0);
x_64 = lean_ctor_get(x_2, 1);
x_65 = lean_ctor_get(x_54, 0);
x_66 = lean_ctor_get(x_54, 1);
x_67 = lean_ctor_get(x_54, 2);
x_68 = lean_ctor_get(x_54, 3);
lean_inc(x_68);
lean_inc(x_67);
lean_inc(x_66);
lean_inc(x_65);
lean_dec(x_54);
x_69 = lean_box(0);
x_70 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_70, 0, x_63);
lean_ctor_set(x_70, 1, x_69);
lean_inc(x_64);
x_71 = lp_mathlib_List_foldl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_spec__0(x_70, x_64);
x_72 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_72, 0, x_65);
lean_ctor_set(x_72, 1, x_66);
lean_ctor_set(x_72, 2, x_67);
lean_ctor_set(x_72, 3, x_68);
lean_ctor_set(x_72, 4, x_71);
x_73 = lean_box(8);
lean_ctor_set(x_52, 1, x_72);
lean_ctor_set(x_52, 0, x_73);
x_32 = x_52;
x_33 = lean_box(0);
goto block_44;
}
}
else
{
lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; 
x_74 = lean_ctor_get(x_52, 1);
x_75 = lean_ctor_get(x_52, 0);
lean_inc(x_74);
lean_inc(x_75);
lean_dec(x_52);
x_76 = lean_ctor_get(x_2, 1);
x_77 = lean_ctor_get(x_74, 0);
lean_inc(x_77);
x_78 = lean_ctor_get(x_74, 1);
lean_inc(x_78);
x_79 = lean_ctor_get(x_74, 2);
lean_inc_ref(x_79);
x_80 = lean_ctor_get(x_74, 3);
lean_inc(x_80);
if (lean_is_exclusive(x_74)) {
 lean_ctor_release(x_74, 0);
 lean_ctor_release(x_74, 1);
 lean_ctor_release(x_74, 2);
 lean_ctor_release(x_74, 3);
 lean_ctor_release(x_74, 4);
 x_81 = x_74;
} else {
 lean_dec_ref(x_74);
 x_81 = lean_box(0);
}
x_82 = lean_box(0);
x_83 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_83, 0, x_75);
lean_ctor_set(x_83, 1, x_82);
lean_inc(x_76);
x_84 = lp_mathlib_List_foldl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_spec__0(x_83, x_76);
if (lean_is_scalar(x_81)) {
 x_85 = lean_alloc_ctor(0, 5, 0);
} else {
 x_85 = x_81;
}
lean_ctor_set(x_85, 0, x_77);
lean_ctor_set(x_85, 1, x_78);
lean_ctor_set(x_85, 2, x_79);
lean_ctor_set(x_85, 3, x_80);
lean_ctor_set(x_85, 4, x_84);
x_86 = lean_box(8);
x_87 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_87, 0, x_86);
lean_ctor_set(x_87, 1, x_85);
x_32 = x_87;
x_33 = lean_box(0);
goto block_44;
}
}
}
else
{
x_45 = x_51;
goto block_50;
}
block_16:
{
lean_object* x_14; lean_object* x_15; 
x_14 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_14, 0, x_11);
lean_ctor_set(x_14, 1, x_12);
x_15 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_15, 0, x_14);
return x_15;
}
block_20:
{
lean_object* x_19; 
x_19 = lean_box(0);
x_11 = x_17;
x_12 = x_19;
x_13 = lean_box(0);
goto block_16;
}
block_24:
{
if (lean_obj_tag(x_22) == 0)
{
lean_object* x_23; 
x_23 = lean_ctor_get(x_22, 0);
lean_inc(x_23);
lean_dec_ref(x_22);
x_11 = x_21;
x_12 = x_23;
x_13 = lean_box(0);
goto block_16;
}
else
{
lean_dec_ref(x_21);
return x_22;
}
}
block_31:
{
if (x_29 == 0)
{
lean_dec(x_28);
lean_dec_ref(x_27);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
x_17 = x_26;
x_18 = lean_box(0);
goto block_20;
}
else
{
lean_object* x_30; 
x_30 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities(x_27, x_28, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
x_21 = x_26;
x_22 = x_30;
goto block_24;
}
}
block_44:
{
if (lean_obj_tag(x_1) == 5)
{
if (lean_obj_tag(x_2) == 1)
{
lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; uint8_t x_39; 
x_34 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_34);
x_35 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_35);
lean_dec_ref(x_1);
x_36 = lean_ctor_get(x_2, 0);
lean_inc(x_36);
x_37 = lean_ctor_get(x_2, 1);
lean_inc(x_37);
lean_dec_ref(x_2);
x_38 = l_Lean_Expr_fvar___override(x_36);
x_39 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_isStarWithArg(x_38, x_35);
lean_dec_ref(x_35);
lean_dec_ref(x_38);
if (x_39 == 0)
{
x_25 = lean_box(0);
x_26 = x_32;
x_27 = x_34;
x_28 = x_37;
x_29 = x_39;
goto block_31;
}
else
{
lean_object* x_40; uint8_t x_41; 
x_40 = l_Lean_Expr_getAppFn(x_34);
x_41 = l_Lean_Expr_isMVar(x_40);
lean_dec_ref(x_40);
if (x_41 == 0)
{
x_25 = lean_box(0);
x_26 = x_32;
x_27 = x_34;
x_28 = x_37;
x_29 = x_39;
goto block_31;
}
else
{
lean_dec(x_37);
lean_dec_ref(x_34);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
x_17 = x_32;
x_18 = lean_box(0);
goto block_20;
}
}
}
else
{
lean_object* x_42; 
lean_dec_ref(x_4);
x_42 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities___lam__0(x_1, x_2, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_2);
lean_dec_ref(x_1);
x_21 = x_32;
x_22 = x_42;
goto block_24;
}
}
else
{
lean_object* x_43; 
lean_dec_ref(x_4);
x_43 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities___lam__0(x_1, x_2, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_2);
lean_dec_ref(x_1);
x_21 = x_32;
x_22 = x_43;
goto block_24;
}
}
block_50:
{
if (lean_obj_tag(x_45) == 0)
{
lean_object* x_46; 
x_46 = lean_ctor_get(x_45, 0);
lean_inc(x_46);
lean_dec_ref(x_45);
x_32 = x_46;
x_33 = lean_box(0);
goto block_44;
}
else
{
uint8_t x_47; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_2);
lean_dec_ref(x_1);
x_47 = !lean_is_exclusive(x_45);
if (x_47 == 0)
{
return x_45;
}
else
{
lean_object* x_48; lean_object* x_49; 
x_48 = lean_ctor_get(x_45, 0);
lean_inc(x_48);
lean_dec(x_45);
x_49 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_49, 0, x_48);
return x_49;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
uint8_t x_11; lean_object* x_12; 
x_11 = lean_unbox(x_3);
x_12 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities(x_1, x_2, x_11, x_4, x_5, x_6, x_7, x_8, x_9);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lean_expr_instantiate1(x_1, x_8);
x_10 = l_Lean_Expr_fvarId_x21(x_8);
x_11 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_11, 0, x_10);
lean_ctor_set(x_11, 1, x_2);
x_12 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___redArg(x_3, x_4, x_5, x_9, x_11, x_6, x_7);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_8);
lean_dec_ref(x_1);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
if (lean_obj_tag(x_7) == 6)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; lean_object* x_12; uint8_t x_13; lean_object* x_14; 
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_7, 1);
lean_inc_ref(x_9);
x_10 = lean_ctor_get(x_7, 2);
lean_inc_ref(x_10);
x_11 = lean_ctor_get_uint8(x_7, sizeof(void*)*3 + 8);
lean_dec_ref(x_7);
lean_inc_ref(x_4);
lean_inc_ref(x_2);
x_12 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___redArg___lam__0___boxed), 8, 7);
lean_closure_set(x_12, 0, x_10);
lean_closure_set(x_12, 1, x_1);
lean_closure_set(x_12, 2, x_2);
lean_closure_set(x_12, 3, x_3);
lean_closure_set(x_12, 4, x_4);
lean_closure_set(x_12, 5, x_5);
lean_closure_set(x_12, 6, x_6);
x_13 = 0;
x_14 = l_Lean_Meta_withLocalDecl___redArg(x_4, x_2, x_8, x_11, x_9, x_12, x_13);
return x_14;
}
else
{
lean_object* x_15; 
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_15 = lean_apply_2(x_6, x_7, x_1);
return x_15;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
uint8_t x_8; 
x_8 = l_Lean_Meta_DiscrTree_hasNoindexAnnotation(x_4);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_9 = lean_ctor_get(x_1, 1);
lean_inc(x_9);
lean_inc(x_2);
x_10 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___redArg___lam__1), 7, 6);
lean_closure_set(x_10, 0, x_5);
lean_closure_set(x_10, 1, x_1);
lean_closure_set(x_10, 2, x_2);
lean_closure_set(x_10, 3, x_3);
lean_closure_set(x_10, 4, x_6);
lean_closure_set(x_10, 5, x_7);
x_11 = lean_alloc_closure((void*)(l_Lean_Meta_DiscrTree_reduce___boxed), 6, 1);
lean_closure_set(x_11, 0, x_4);
x_12 = lean_apply_2(x_2, lean_box(0), x_11);
x_13 = lean_apply_4(x_9, lean_box(0), lean_box(0), x_12, x_10);
return x_13;
}
else
{
lean_object* x_14; 
lean_dec(x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
x_14 = lean_apply_1(x_6, x_5);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___redArg(x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_9; 
x_9 = lean_apply_7(x_1, x_3, x_2, x_4, x_5, x_6, x_7, lean_box(0));
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___redArg(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, uint8_t x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; lean_object* x_13; 
x_12 = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___redArg___lam__0___boxed), 8, 2);
lean_closure_set(x_12, 0, x_4);
lean_closure_set(x_12, 1, x_6);
x_13 = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), x_1, x_2, x_3, x_12, x_5, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_13) == 0)
{
return x_13;
}
else
{
uint8_t x_14; 
x_14 = !lean_is_exclusive(x_13);
if (x_14 == 0)
{
return x_13;
}
else
{
lean_object* x_15; lean_object* x_16; 
x_15 = lean_ctor_get(x_13, 0);
lean_inc(x_15);
lean_dec(x_13);
x_16 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_16, 0, x_15);
return x_16;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0(lean_object* x_1, lean_object* x_2, uint8_t x_3, lean_object* x_4, lean_object* x_5, uint8_t x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, uint8_t x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_12 = lean_expr_instantiate1(x_1, x_5);
x_13 = l_Lean_Expr_fvarId_x21(x_5);
x_14 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_14, 0, x_13);
lean_ctor_set(x_14, 1, x_2);
x_15 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg(x_3, x_4, x_12, x_14, x_6, x_7, x_8, x_9, x_10);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
uint8_t x_12; lean_object* x_13; 
x_12 = lean_unbox(x_4);
x_13 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___lam__0(x_1, x_2, x_3, x_12, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_5);
lean_dec_ref(x_1);
return x_13;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lean_box(0);
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; lean_object* x_12; uint8_t x_17; 
x_17 = l_Lean_Meta_DiscrTree_hasNoindexAnnotation(x_3);
if (x_17 == 0)
{
lean_object* x_18; 
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
x_18 = l_Lean_Meta_DiscrTree_reduce(x_3, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_18) == 0)
{
lean_object* x_19; 
x_19 = lean_ctor_get(x_18, 0);
lean_inc(x_19);
lean_dec_ref(x_18);
if (lean_obj_tag(x_19) == 6)
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; uint8_t x_23; lean_object* x_24; lean_object* x_25; uint8_t x_26; lean_object* x_27; 
x_20 = lean_ctor_get(x_19, 0);
lean_inc(x_20);
x_21 = lean_ctor_get(x_19, 1);
lean_inc_ref(x_21);
x_22 = lean_ctor_get(x_19, 2);
lean_inc_ref(x_22);
x_23 = lean_ctor_get_uint8(x_19, sizeof(void*)*3 + 8);
lean_dec_ref(x_19);
x_24 = lean_box(x_2);
x_25 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___lam__0___boxed), 11, 4);
lean_closure_set(x_25, 0, x_22);
lean_closure_set(x_25, 1, x_4);
lean_closure_set(x_25, 2, x_1);
lean_closure_set(x_25, 3, x_24);
x_26 = 0;
x_27 = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___redArg(x_20, x_23, x_21, x_25, x_26, x_5, x_6, x_7, x_8, x_9);
return x_27;
}
else
{
lean_object* x_28; 
x_28 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities(x_19, x_4, x_2, x_1, x_5, x_6, x_7, x_8, x_9);
return x_28;
}
}
else
{
uint8_t x_29; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_1);
x_29 = !lean_is_exclusive(x_18);
if (x_29 == 0)
{
return x_18;
}
else
{
lean_object* x_30; lean_object* x_31; 
x_30 = lean_ctor_get(x_18, 0);
lean_inc(x_30);
lean_dec(x_18);
x_31 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_31, 0, x_30);
return x_31;
}
}
}
else
{
lean_object* x_32; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_3);
x_32 = lean_box(0);
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_33; 
x_33 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_33, 0, x_32);
lean_ctor_set(x_33, 1, x_1);
x_11 = x_33;
x_12 = lean_box(0);
goto block_16;
}
else
{
uint8_t x_34; 
x_34 = !lean_is_exclusive(x_4);
if (x_34 == 0)
{
lean_object* x_35; lean_object* x_36; uint8_t x_37; 
x_35 = lean_ctor_get(x_4, 1);
x_36 = lean_ctor_get(x_4, 0);
lean_dec(x_36);
x_37 = !lean_is_exclusive(x_1);
if (x_37 == 0)
{
lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; 
x_38 = lean_ctor_get(x_1, 4);
lean_dec(x_38);
x_39 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___closed__0;
x_40 = lp_mathlib_List_foldl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_spec__0(x_39, x_35);
lean_ctor_set(x_1, 4, x_40);
x_41 = lean_box(8);
lean_ctor_set_tag(x_4, 0);
lean_ctor_set(x_4, 1, x_1);
lean_ctor_set(x_4, 0, x_41);
x_11 = x_4;
x_12 = lean_box(0);
goto block_16;
}
else
{
lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; 
x_42 = lean_ctor_get(x_1, 0);
x_43 = lean_ctor_get(x_1, 1);
x_44 = lean_ctor_get(x_1, 2);
x_45 = lean_ctor_get(x_1, 3);
lean_inc(x_45);
lean_inc(x_44);
lean_inc(x_43);
lean_inc(x_42);
lean_dec(x_1);
x_46 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___closed__0;
x_47 = lp_mathlib_List_foldl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_spec__0(x_46, x_35);
x_48 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_48, 0, x_42);
lean_ctor_set(x_48, 1, x_43);
lean_ctor_set(x_48, 2, x_44);
lean_ctor_set(x_48, 3, x_45);
lean_ctor_set(x_48, 4, x_47);
x_49 = lean_box(8);
lean_ctor_set_tag(x_4, 0);
lean_ctor_set(x_4, 1, x_48);
lean_ctor_set(x_4, 0, x_49);
x_11 = x_4;
x_12 = lean_box(0);
goto block_16;
}
}
else
{
lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; 
x_50 = lean_ctor_get(x_4, 1);
lean_inc(x_50);
lean_dec(x_4);
x_51 = lean_ctor_get(x_1, 0);
lean_inc(x_51);
x_52 = lean_ctor_get(x_1, 1);
lean_inc(x_52);
x_53 = lean_ctor_get(x_1, 2);
lean_inc_ref(x_53);
x_54 = lean_ctor_get(x_1, 3);
lean_inc(x_54);
if (lean_is_exclusive(x_1)) {
 lean_ctor_release(x_1, 0);
 lean_ctor_release(x_1, 1);
 lean_ctor_release(x_1, 2);
 lean_ctor_release(x_1, 3);
 lean_ctor_release(x_1, 4);
 x_55 = x_1;
} else {
 lean_dec_ref(x_1);
 x_55 = lean_box(0);
}
x_56 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___closed__0;
x_57 = lp_mathlib_List_foldl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_spec__0(x_56, x_50);
if (lean_is_scalar(x_55)) {
 x_58 = lean_alloc_ctor(0, 5, 0);
} else {
 x_58 = x_55;
}
lean_ctor_set(x_58, 0, x_51);
lean_ctor_set(x_58, 1, x_52);
lean_ctor_set(x_58, 2, x_53);
lean_ctor_set(x_58, 3, x_54);
lean_ctor_set(x_58, 4, x_57);
x_59 = lean_box(8);
x_60 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_60, 0, x_59);
lean_ctor_set(x_60, 1, x_58);
x_11 = x_60;
x_12 = lean_box(0);
goto block_16;
}
}
}
block_16:
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_13 = lean_box(0);
x_14 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_14, 0, x_11);
lean_ctor_set(x_14, 1, x_13);
x_15 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_15, 0, x_14);
return x_15;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg(x_1, x_2, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; lean_object* x_11; 
x_10 = lean_box(0);
x_11 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg(x_3, x_2, x_1, x_10, x_4, x_5, x_6, x_7, x_8);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
uint8_t x_13; uint8_t x_14; lean_object* x_15; 
x_13 = lean_unbox(x_3);
x_14 = lean_unbox(x_6);
x_15 = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0(x_1, x_2, x_13, x_4, x_5, x_14, x_7, x_8, x_9, x_10, x_11);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
uint8_t x_12; lean_object* x_13; 
x_12 = lean_unbox(x_2);
x_13 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0(x_1, x_12, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
uint8_t x_10; lean_object* x_11; 
x_10 = lean_unbox(x_2);
x_11 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta(x_1, x_10, x_3, x_4, x_5, x_6, x_7, x_8);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
uint8_t x_12; uint8_t x_13; lean_object* x_14; 
x_12 = lean_unbox(x_2);
x_13 = lean_unbox(x_5);
x_14 = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___redArg(x_1, x_12, x_3, x_4, x_13, x_6, x_7, x_8, x_9, x_10);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
uint8_t x_11; lean_object* x_12; 
x_11 = lean_unbox(x_2);
x_12 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg(x_1, x_11, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; 
x_10 = lean_apply_8(x_1, x_4, x_2, x_3, x_5, x_6, x_7, x_8, lean_box(0));
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___redArg(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, uint8_t x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_13; lean_object* x_14; 
x_13 = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___redArg___lam__0___boxed), 9, 3);
lean_closure_set(x_13, 0, x_4);
lean_closure_set(x_13, 1, x_6);
lean_closure_set(x_13, 2, x_7);
x_14 = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), x_1, x_2, x_3, x_13, x_5, x_8, x_9, x_10, x_11);
if (lean_obj_tag(x_14) == 0)
{
uint8_t x_15; 
x_15 = !lean_is_exclusive(x_14);
if (x_15 == 0)
{
return x_14;
}
else
{
lean_object* x_16; lean_object* x_17; 
x_16 = lean_ctor_get(x_14, 0);
lean_inc(x_16);
lean_dec(x_14);
x_17 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_17, 0, x_16);
return x_17;
}
}
else
{
uint8_t x_18; 
x_18 = !lean_is_exclusive(x_14);
if (x_18 == 0)
{
return x_14;
}
else
{
lean_object* x_19; lean_object* x_20; 
x_19 = lean_ctor_get(x_14, 0);
lean_inc(x_19);
lean_dec(x_14);
x_20 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_20, 0, x_19);
return x_20;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0(lean_object* x_1, lean_object* x_2, uint8_t x_3, lean_object* x_4, lean_object* x_5, uint8_t x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___redArg___lam__0(lean_object* x_1, lean_object* x_2, uint8_t x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_12 = lean_expr_instantiate1(x_1, x_4);
x_13 = l_Lean_Expr_fvarId_x21(x_4);
x_14 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_14, 0, x_13);
lean_ctor_set(x_14, 1, x_2);
x_15 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___redArg(x_3, x_12, x_14, x_5, x_6, x_7, x_8, x_9, x_10);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
uint8_t x_12; lean_object* x_13; 
x_12 = lean_unbox(x_3);
x_13 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___redArg___lam__0(x_1, x_2, x_12, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_4);
lean_dec_ref(x_1);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___redArg(uint8_t x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
uint8_t x_11; 
x_11 = l_Lean_Meta_DiscrTree_hasNoindexAnnotation(x_2);
if (x_11 == 0)
{
lean_object* x_12; 
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
x_12 = l_Lean_Meta_DiscrTree_reduce(x_2, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_12) == 0)
{
lean_object* x_13; 
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
if (lean_obj_tag(x_13) == 6)
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; uint8_t x_17; lean_object* x_18; lean_object* x_19; uint8_t x_20; lean_object* x_21; 
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
x_15 = lean_ctor_get(x_13, 1);
lean_inc_ref(x_15);
x_16 = lean_ctor_get(x_13, 2);
lean_inc_ref(x_16);
x_17 = lean_ctor_get_uint8(x_13, sizeof(void*)*3 + 8);
lean_dec_ref(x_13);
x_18 = lean_box(x_1);
x_19 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___redArg___lam__0___boxed), 11, 3);
lean_closure_set(x_19, 0, x_16);
lean_closure_set(x_19, 1, x_3);
lean_closure_set(x_19, 2, x_18);
x_20 = 0;
x_21 = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___redArg(x_14, x_17, x_15, x_19, x_20, x_4, x_5, x_6, x_7, x_8, x_9);
return x_21;
}
else
{
lean_object* x_22; 
lean_inc(x_3);
x_22 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go(x_13, x_3, x_1, x_4, x_5, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_22) == 0)
{
lean_object* x_23; 
x_23 = lean_ctor_get(x_22, 0);
lean_inc(x_23);
if (lean_obj_tag(x_3) == 0)
{
lean_dec(x_23);
return x_22;
}
else
{
uint8_t x_24; 
x_24 = !lean_is_exclusive(x_22);
if (x_24 == 0)
{
lean_object* x_25; uint8_t x_26; 
x_25 = lean_ctor_get(x_22, 0);
lean_dec(x_25);
x_26 = !lean_is_exclusive(x_23);
if (x_26 == 0)
{
uint8_t x_27; 
x_27 = !lean_is_exclusive(x_3);
if (x_27 == 0)
{
lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; uint8_t x_32; 
x_28 = lean_ctor_get(x_23, 1);
x_29 = lean_ctor_get(x_23, 0);
x_30 = lean_ctor_get(x_3, 1);
x_31 = lean_ctor_get(x_3, 0);
lean_dec(x_31);
x_32 = !lean_is_exclusive(x_28);
if (x_32 == 0)
{
lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; 
x_33 = lean_ctor_get(x_28, 4);
lean_dec(x_33);
x_34 = lean_box(0);
lean_ctor_set(x_3, 1, x_34);
lean_ctor_set(x_3, 0, x_29);
x_35 = lp_mathlib_List_foldl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_spec__0(x_3, x_30);
lean_ctor_set(x_28, 4, x_35);
x_36 = lean_box(8);
lean_ctor_set(x_23, 0, x_36);
return x_22;
}
else
{
lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; 
x_37 = lean_ctor_get(x_28, 0);
x_38 = lean_ctor_get(x_28, 1);
x_39 = lean_ctor_get(x_28, 2);
x_40 = lean_ctor_get(x_28, 3);
lean_inc(x_40);
lean_inc(x_39);
lean_inc(x_38);
lean_inc(x_37);
lean_dec(x_28);
x_41 = lean_box(0);
lean_ctor_set(x_3, 1, x_41);
lean_ctor_set(x_3, 0, x_29);
x_42 = lp_mathlib_List_foldl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_spec__0(x_3, x_30);
x_43 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_43, 0, x_37);
lean_ctor_set(x_43, 1, x_38);
lean_ctor_set(x_43, 2, x_39);
lean_ctor_set(x_43, 3, x_40);
lean_ctor_set(x_43, 4, x_42);
x_44 = lean_box(8);
lean_ctor_set(x_23, 1, x_43);
lean_ctor_set(x_23, 0, x_44);
return x_22;
}
}
else
{
lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; 
x_45 = lean_ctor_get(x_23, 1);
x_46 = lean_ctor_get(x_23, 0);
x_47 = lean_ctor_get(x_3, 1);
lean_inc(x_47);
lean_dec(x_3);
x_48 = lean_ctor_get(x_45, 0);
lean_inc(x_48);
x_49 = lean_ctor_get(x_45, 1);
lean_inc(x_49);
x_50 = lean_ctor_get(x_45, 2);
lean_inc_ref(x_50);
x_51 = lean_ctor_get(x_45, 3);
lean_inc(x_51);
if (lean_is_exclusive(x_45)) {
 lean_ctor_release(x_45, 0);
 lean_ctor_release(x_45, 1);
 lean_ctor_release(x_45, 2);
 lean_ctor_release(x_45, 3);
 lean_ctor_release(x_45, 4);
 x_52 = x_45;
} else {
 lean_dec_ref(x_45);
 x_52 = lean_box(0);
}
x_53 = lean_box(0);
x_54 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_54, 0, x_46);
lean_ctor_set(x_54, 1, x_53);
x_55 = lp_mathlib_List_foldl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_spec__0(x_54, x_47);
if (lean_is_scalar(x_52)) {
 x_56 = lean_alloc_ctor(0, 5, 0);
} else {
 x_56 = x_52;
}
lean_ctor_set(x_56, 0, x_48);
lean_ctor_set(x_56, 1, x_49);
lean_ctor_set(x_56, 2, x_50);
lean_ctor_set(x_56, 3, x_51);
lean_ctor_set(x_56, 4, x_55);
x_57 = lean_box(8);
lean_ctor_set(x_23, 1, x_56);
lean_ctor_set(x_23, 0, x_57);
return x_22;
}
}
else
{
lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; 
x_58 = lean_ctor_get(x_23, 1);
x_59 = lean_ctor_get(x_23, 0);
lean_inc(x_58);
lean_inc(x_59);
lean_dec(x_23);
x_60 = lean_ctor_get(x_3, 1);
lean_inc(x_60);
if (lean_is_exclusive(x_3)) {
 lean_ctor_release(x_3, 0);
 lean_ctor_release(x_3, 1);
 x_61 = x_3;
} else {
 lean_dec_ref(x_3);
 x_61 = lean_box(0);
}
x_62 = lean_ctor_get(x_58, 0);
lean_inc(x_62);
x_63 = lean_ctor_get(x_58, 1);
lean_inc(x_63);
x_64 = lean_ctor_get(x_58, 2);
lean_inc_ref(x_64);
x_65 = lean_ctor_get(x_58, 3);
lean_inc(x_65);
if (lean_is_exclusive(x_58)) {
 lean_ctor_release(x_58, 0);
 lean_ctor_release(x_58, 1);
 lean_ctor_release(x_58, 2);
 lean_ctor_release(x_58, 3);
 lean_ctor_release(x_58, 4);
 x_66 = x_58;
} else {
 lean_dec_ref(x_58);
 x_66 = lean_box(0);
}
x_67 = lean_box(0);
if (lean_is_scalar(x_61)) {
 x_68 = lean_alloc_ctor(1, 2, 0);
} else {
 x_68 = x_61;
}
lean_ctor_set(x_68, 0, x_59);
lean_ctor_set(x_68, 1, x_67);
x_69 = lp_mathlib_List_foldl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_spec__0(x_68, x_60);
if (lean_is_scalar(x_66)) {
 x_70 = lean_alloc_ctor(0, 5, 0);
} else {
 x_70 = x_66;
}
lean_ctor_set(x_70, 0, x_62);
lean_ctor_set(x_70, 1, x_63);
lean_ctor_set(x_70, 2, x_64);
lean_ctor_set(x_70, 3, x_65);
lean_ctor_set(x_70, 4, x_69);
x_71 = lean_box(8);
x_72 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_72, 0, x_71);
lean_ctor_set(x_72, 1, x_70);
lean_ctor_set(x_22, 0, x_72);
return x_22;
}
}
else
{
lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; 
lean_dec(x_22);
x_73 = lean_ctor_get(x_23, 1);
lean_inc(x_73);
x_74 = lean_ctor_get(x_23, 0);
lean_inc(x_74);
if (lean_is_exclusive(x_23)) {
 lean_ctor_release(x_23, 0);
 lean_ctor_release(x_23, 1);
 x_75 = x_23;
} else {
 lean_dec_ref(x_23);
 x_75 = lean_box(0);
}
x_76 = lean_ctor_get(x_3, 1);
lean_inc(x_76);
if (lean_is_exclusive(x_3)) {
 lean_ctor_release(x_3, 0);
 lean_ctor_release(x_3, 1);
 x_77 = x_3;
} else {
 lean_dec_ref(x_3);
 x_77 = lean_box(0);
}
x_78 = lean_ctor_get(x_73, 0);
lean_inc(x_78);
x_79 = lean_ctor_get(x_73, 1);
lean_inc(x_79);
x_80 = lean_ctor_get(x_73, 2);
lean_inc_ref(x_80);
x_81 = lean_ctor_get(x_73, 3);
lean_inc(x_81);
if (lean_is_exclusive(x_73)) {
 lean_ctor_release(x_73, 0);
 lean_ctor_release(x_73, 1);
 lean_ctor_release(x_73, 2);
 lean_ctor_release(x_73, 3);
 lean_ctor_release(x_73, 4);
 x_82 = x_73;
} else {
 lean_dec_ref(x_73);
 x_82 = lean_box(0);
}
x_83 = lean_box(0);
if (lean_is_scalar(x_77)) {
 x_84 = lean_alloc_ctor(1, 2, 0);
} else {
 x_84 = x_77;
}
lean_ctor_set(x_84, 0, x_74);
lean_ctor_set(x_84, 1, x_83);
x_85 = lp_mathlib_List_foldl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_spec__0(x_84, x_76);
if (lean_is_scalar(x_82)) {
 x_86 = lean_alloc_ctor(0, 5, 0);
} else {
 x_86 = x_82;
}
lean_ctor_set(x_86, 0, x_78);
lean_ctor_set(x_86, 1, x_79);
lean_ctor_set(x_86, 2, x_80);
lean_ctor_set(x_86, 3, x_81);
lean_ctor_set(x_86, 4, x_85);
x_87 = lean_box(8);
if (lean_is_scalar(x_75)) {
 x_88 = lean_alloc_ctor(0, 2, 0);
} else {
 x_88 = x_75;
}
lean_ctor_set(x_88, 0, x_87);
lean_ctor_set(x_88, 1, x_86);
x_89 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_89, 0, x_88);
return x_89;
}
}
}
else
{
lean_dec(x_3);
return x_22;
}
}
}
else
{
uint8_t x_90; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec(x_3);
x_90 = !lean_is_exclusive(x_12);
if (x_90 == 0)
{
return x_12;
}
else
{
lean_object* x_91; lean_object* x_92; 
x_91 = lean_ctor_get(x_12, 0);
lean_inc(x_91);
lean_dec(x_12);
x_92 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_92, 0, x_91);
return x_92;
}
}
}
else
{
lean_object* x_93; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_4);
lean_dec_ref(x_2);
x_93 = lean_box(0);
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_94; lean_object* x_95; 
x_94 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_94, 0, x_93);
lean_ctor_set(x_94, 1, x_5);
x_95 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_95, 0, x_94);
return x_95;
}
else
{
lean_object* x_96; uint8_t x_97; 
x_96 = lean_ctor_get(x_3, 1);
lean_inc(x_96);
lean_dec_ref(x_3);
x_97 = !lean_is_exclusive(x_5);
if (x_97 == 0)
{
lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; 
x_98 = lean_ctor_get(x_5, 4);
lean_dec(x_98);
x_99 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___closed__0;
x_100 = lp_mathlib_List_foldl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_spec__0(x_99, x_96);
lean_ctor_set(x_5, 4, x_100);
x_101 = lean_box(8);
x_102 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_102, 0, x_101);
lean_ctor_set(x_102, 1, x_5);
x_103 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_103, 0, x_102);
return x_103;
}
else
{
lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; 
x_104 = lean_ctor_get(x_5, 0);
x_105 = lean_ctor_get(x_5, 1);
x_106 = lean_ctor_get(x_5, 2);
x_107 = lean_ctor_get(x_5, 3);
lean_inc(x_107);
lean_inc(x_106);
lean_inc(x_105);
lean_inc(x_104);
lean_dec(x_5);
x_108 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___closed__0;
x_109 = lp_mathlib_List_foldl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_spec__0(x_108, x_96);
x_110 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_110, 0, x_104);
lean_ctor_set(x_110, 1, x_105);
lean_ctor_set(x_110, 2, x_106);
lean_ctor_set(x_110, 3, x_107);
lean_ctor_set(x_110, 4, x_109);
x_111 = lean_box(8);
x_112 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_112, 0, x_111);
lean_ctor_set(x_112, 1, x_110);
x_113 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_113, 0, x_112);
return x_113;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0(uint8_t x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___redArg(x_1, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; lean_object* x_11; 
x_10 = lean_box(0);
x_11 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___redArg(x_2, x_1, x_10, x_3, x_4, x_5, x_6, x_7, x_8);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
uint8_t x_14; uint8_t x_15; lean_object* x_16; 
x_14 = lean_unbox(x_3);
x_15 = lean_unbox(x_6);
x_16 = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0(x_1, x_2, x_14, x_4, x_5, x_15, x_7, x_8, x_9, x_10, x_11, x_12);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
uint8_t x_12; lean_object* x_13; 
x_12 = lean_unbox(x_1);
x_13 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0(x_12, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
uint8_t x_10; lean_object* x_11; 
x_10 = lean_unbox(x_2);
x_11 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep(x_1, x_10, x_3, x_4, x_5, x_6, x_7, x_8);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
uint8_t x_13; uint8_t x_14; lean_object* x_15; 
x_13 = lean_unbox(x_2);
x_14 = lean_unbox(x_5);
x_15 = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___redArg(x_1, x_13, x_3, x_4, x_14, x_6, x_7, x_8, x_9, x_10, x_11);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
uint8_t x_11; lean_object* x_12; 
x_11 = lean_unbox(x_1);
x_12 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___redArg(x_11, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEtaAux(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry___redArg(x_2, x_4);
if (lean_obj_tag(x_8) == 0)
{
lean_object* x_9; uint8_t x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = 1;
x_11 = lean_box(0);
x_12 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta(x_1, x_10, x_9, x_11, x_3, x_4, x_5, x_6);
return x_12;
}
else
{
uint8_t x_13; 
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
x_13 = !lean_is_exclusive(x_8);
if (x_13 == 0)
{
return x_8;
}
else
{
lean_object* x_14; lean_object* x_15; 
x_14 = lean_ctor_get(x_8, 0);
lean_inc(x_14);
lean_dec(x_8);
x_15 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_15, 0, x_14);
return x_15;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEtaAux___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
uint8_t x_8; lean_object* x_9; 
x_8 = lean_unbox(x_2);
x_9 = lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEtaAux(x_1, x_8, x_3, x_4, x_5, x_6);
return x_9;
}
}
static uint64_t _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEta___closed__0() {
_start:
{
uint8_t x_1; uint64_t x_2; 
x_1 = 2;
x_2 = l_Lean_Meta_TransparencyMode_toUInt64(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEta(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; uint8_t x_9; 
x_8 = l_Lean_Meta_Context_config(x_3);
x_9 = !lean_is_exclusive(x_8);
if (x_9 == 0)
{
uint8_t x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; uint8_t x_17; uint8_t x_18; lean_object* x_19; 
x_10 = lean_ctor_get_uint8(x_3, sizeof(void*)*7);
x_11 = lean_ctor_get(x_3, 1);
lean_inc(x_11);
x_12 = lean_ctor_get(x_3, 2);
lean_inc_ref(x_12);
x_13 = lean_ctor_get(x_3, 3);
lean_inc_ref(x_13);
x_14 = lean_ctor_get(x_3, 4);
lean_inc(x_14);
x_15 = lean_ctor_get(x_3, 5);
lean_inc(x_15);
x_16 = lean_ctor_get(x_3, 6);
lean_inc(x_16);
x_17 = lean_ctor_get_uint8(x_3, sizeof(void*)*7 + 1);
x_18 = lean_ctor_get_uint8(x_3, sizeof(void*)*7 + 2);
x_19 = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry___redArg(x_2, x_4);
if (lean_obj_tag(x_19) == 0)
{
lean_object* x_20; uint8_t x_21; uint64_t x_22; uint8_t x_23; 
x_20 = lean_ctor_get(x_19, 0);
lean_inc(x_20);
lean_dec_ref(x_19);
x_21 = 2;
lean_ctor_set_uint8(x_8, 9, x_21);
x_22 = l_Lean_Meta_Context_configKey(x_3);
x_23 = !lean_is_exclusive(x_3);
if (x_23 == 0)
{
lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; uint64_t x_31; uint64_t x_32; uint64_t x_33; uint64_t x_34; uint64_t x_35; lean_object* x_36; uint8_t x_37; lean_object* x_38; lean_object* x_39; 
x_24 = lean_ctor_get(x_3, 6);
lean_dec(x_24);
x_25 = lean_ctor_get(x_3, 5);
lean_dec(x_25);
x_26 = lean_ctor_get(x_3, 4);
lean_dec(x_26);
x_27 = lean_ctor_get(x_3, 3);
lean_dec(x_27);
x_28 = lean_ctor_get(x_3, 2);
lean_dec(x_28);
x_29 = lean_ctor_get(x_3, 1);
lean_dec(x_29);
x_30 = lean_ctor_get(x_3, 0);
lean_dec(x_30);
x_31 = 2;
x_32 = lean_uint64_shift_right(x_22, x_31);
x_33 = lean_uint64_shift_left(x_32, x_31);
x_34 = lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEta___closed__0;
x_35 = lean_uint64_lor(x_33, x_34);
x_36 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_36, 0, x_8);
lean_ctor_set_uint64(x_36, sizeof(void*)*1, x_35);
lean_ctor_set(x_3, 0, x_36);
x_37 = 1;
x_38 = lean_box(0);
x_39 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta(x_1, x_37, x_20, x_38, x_3, x_4, x_5, x_6);
return x_39;
}
else
{
uint64_t x_40; uint64_t x_41; uint64_t x_42; uint64_t x_43; uint64_t x_44; lean_object* x_45; lean_object* x_46; uint8_t x_47; lean_object* x_48; lean_object* x_49; 
lean_dec(x_3);
x_40 = 2;
x_41 = lean_uint64_shift_right(x_22, x_40);
x_42 = lean_uint64_shift_left(x_41, x_40);
x_43 = lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEta___closed__0;
x_44 = lean_uint64_lor(x_42, x_43);
x_45 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_45, 0, x_8);
lean_ctor_set_uint64(x_45, sizeof(void*)*1, x_44);
x_46 = lean_alloc_ctor(0, 7, 3);
lean_ctor_set(x_46, 0, x_45);
lean_ctor_set(x_46, 1, x_11);
lean_ctor_set(x_46, 2, x_12);
lean_ctor_set(x_46, 3, x_13);
lean_ctor_set(x_46, 4, x_14);
lean_ctor_set(x_46, 5, x_15);
lean_ctor_set(x_46, 6, x_16);
lean_ctor_set_uint8(x_46, sizeof(void*)*7, x_10);
lean_ctor_set_uint8(x_46, sizeof(void*)*7 + 1, x_17);
lean_ctor_set_uint8(x_46, sizeof(void*)*7 + 2, x_18);
x_47 = 1;
x_48 = lean_box(0);
x_49 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta(x_1, x_47, x_20, x_48, x_46, x_4, x_5, x_6);
return x_49;
}
}
else
{
uint8_t x_50; 
lean_dec(x_16);
lean_dec(x_15);
lean_dec(x_14);
lean_dec_ref(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_free_object(x_8);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
x_50 = !lean_is_exclusive(x_19);
if (x_50 == 0)
{
return x_19;
}
else
{
lean_object* x_51; lean_object* x_52; 
x_51 = lean_ctor_get(x_19, 0);
lean_inc(x_51);
lean_dec(x_19);
x_52 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_52, 0, x_51);
return x_52;
}
}
}
else
{
uint8_t x_53; uint8_t x_54; uint8_t x_55; uint8_t x_56; uint8_t x_57; uint8_t x_58; uint8_t x_59; uint8_t x_60; uint8_t x_61; uint8_t x_62; uint8_t x_63; uint8_t x_64; uint8_t x_65; uint8_t x_66; uint8_t x_67; uint8_t x_68; uint8_t x_69; uint8_t x_70; uint8_t x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; uint8_t x_78; uint8_t x_79; lean_object* x_80; 
x_53 = lean_ctor_get_uint8(x_8, 0);
x_54 = lean_ctor_get_uint8(x_8, 1);
x_55 = lean_ctor_get_uint8(x_8, 2);
x_56 = lean_ctor_get_uint8(x_8, 3);
x_57 = lean_ctor_get_uint8(x_8, 4);
x_58 = lean_ctor_get_uint8(x_8, 5);
x_59 = lean_ctor_get_uint8(x_8, 6);
x_60 = lean_ctor_get_uint8(x_8, 7);
x_61 = lean_ctor_get_uint8(x_8, 8);
x_62 = lean_ctor_get_uint8(x_8, 10);
x_63 = lean_ctor_get_uint8(x_8, 11);
x_64 = lean_ctor_get_uint8(x_8, 12);
x_65 = lean_ctor_get_uint8(x_8, 13);
x_66 = lean_ctor_get_uint8(x_8, 14);
x_67 = lean_ctor_get_uint8(x_8, 15);
x_68 = lean_ctor_get_uint8(x_8, 16);
x_69 = lean_ctor_get_uint8(x_8, 17);
x_70 = lean_ctor_get_uint8(x_8, 18);
lean_dec(x_8);
x_71 = lean_ctor_get_uint8(x_3, sizeof(void*)*7);
x_72 = lean_ctor_get(x_3, 1);
lean_inc(x_72);
x_73 = lean_ctor_get(x_3, 2);
lean_inc_ref(x_73);
x_74 = lean_ctor_get(x_3, 3);
lean_inc_ref(x_74);
x_75 = lean_ctor_get(x_3, 4);
lean_inc(x_75);
x_76 = lean_ctor_get(x_3, 5);
lean_inc(x_76);
x_77 = lean_ctor_get(x_3, 6);
lean_inc(x_77);
x_78 = lean_ctor_get_uint8(x_3, sizeof(void*)*7 + 1);
x_79 = lean_ctor_get_uint8(x_3, sizeof(void*)*7 + 2);
x_80 = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry___redArg(x_2, x_4);
if (lean_obj_tag(x_80) == 0)
{
lean_object* x_81; uint8_t x_82; lean_object* x_83; uint64_t x_84; lean_object* x_85; uint64_t x_86; uint64_t x_87; uint64_t x_88; uint64_t x_89; uint64_t x_90; lean_object* x_91; lean_object* x_92; uint8_t x_93; lean_object* x_94; lean_object* x_95; 
x_81 = lean_ctor_get(x_80, 0);
lean_inc(x_81);
lean_dec_ref(x_80);
x_82 = 2;
x_83 = lean_alloc_ctor(0, 0, 19);
lean_ctor_set_uint8(x_83, 0, x_53);
lean_ctor_set_uint8(x_83, 1, x_54);
lean_ctor_set_uint8(x_83, 2, x_55);
lean_ctor_set_uint8(x_83, 3, x_56);
lean_ctor_set_uint8(x_83, 4, x_57);
lean_ctor_set_uint8(x_83, 5, x_58);
lean_ctor_set_uint8(x_83, 6, x_59);
lean_ctor_set_uint8(x_83, 7, x_60);
lean_ctor_set_uint8(x_83, 8, x_61);
lean_ctor_set_uint8(x_83, 9, x_82);
lean_ctor_set_uint8(x_83, 10, x_62);
lean_ctor_set_uint8(x_83, 11, x_63);
lean_ctor_set_uint8(x_83, 12, x_64);
lean_ctor_set_uint8(x_83, 13, x_65);
lean_ctor_set_uint8(x_83, 14, x_66);
lean_ctor_set_uint8(x_83, 15, x_67);
lean_ctor_set_uint8(x_83, 16, x_68);
lean_ctor_set_uint8(x_83, 17, x_69);
lean_ctor_set_uint8(x_83, 18, x_70);
x_84 = l_Lean_Meta_Context_configKey(x_3);
if (lean_is_exclusive(x_3)) {
 lean_ctor_release(x_3, 0);
 lean_ctor_release(x_3, 1);
 lean_ctor_release(x_3, 2);
 lean_ctor_release(x_3, 3);
 lean_ctor_release(x_3, 4);
 lean_ctor_release(x_3, 5);
 lean_ctor_release(x_3, 6);
 x_85 = x_3;
} else {
 lean_dec_ref(x_3);
 x_85 = lean_box(0);
}
x_86 = 2;
x_87 = lean_uint64_shift_right(x_84, x_86);
x_88 = lean_uint64_shift_left(x_87, x_86);
x_89 = lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEta___closed__0;
x_90 = lean_uint64_lor(x_88, x_89);
x_91 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_91, 0, x_83);
lean_ctor_set_uint64(x_91, sizeof(void*)*1, x_90);
if (lean_is_scalar(x_85)) {
 x_92 = lean_alloc_ctor(0, 7, 3);
} else {
 x_92 = x_85;
}
lean_ctor_set(x_92, 0, x_91);
lean_ctor_set(x_92, 1, x_72);
lean_ctor_set(x_92, 2, x_73);
lean_ctor_set(x_92, 3, x_74);
lean_ctor_set(x_92, 4, x_75);
lean_ctor_set(x_92, 5, x_76);
lean_ctor_set(x_92, 6, x_77);
lean_ctor_set_uint8(x_92, sizeof(void*)*7, x_71);
lean_ctor_set_uint8(x_92, sizeof(void*)*7 + 1, x_78);
lean_ctor_set_uint8(x_92, sizeof(void*)*7 + 2, x_79);
x_93 = 1;
x_94 = lean_box(0);
x_95 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta(x_1, x_93, x_81, x_94, x_92, x_4, x_5, x_6);
return x_95;
}
else
{
lean_object* x_96; lean_object* x_97; lean_object* x_98; 
lean_dec(x_77);
lean_dec(x_76);
lean_dec(x_75);
lean_dec_ref(x_74);
lean_dec_ref(x_73);
lean_dec(x_72);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
x_96 = lean_ctor_get(x_80, 0);
lean_inc(x_96);
if (lean_is_exclusive(x_80)) {
 lean_ctor_release(x_80, 0);
 x_97 = x_80;
} else {
 lean_dec_ref(x_80);
 x_97 = lean_box(0);
}
if (lean_is_scalar(x_97)) {
 x_98 = lean_alloc_ctor(1, 1, 0);
} else {
 x_98 = x_97;
}
lean_ctor_set(x_98, 0, x_96);
return x_98;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEta___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
uint8_t x_8; lean_object* x_9; 
x_8 = lean_unbox(x_2);
x_9 = lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEta(x_1, x_8, x_3, x_4, x_5, x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_initializeLazyEntry(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry___redArg(x_2, x_4);
if (lean_obj_tag(x_8) == 0)
{
lean_object* x_9; uint8_t x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = 1;
x_11 = lean_box(0);
x_12 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep(x_1, x_10, x_11, x_9, x_3, x_4, x_5, x_6);
return x_12;
}
else
{
uint8_t x_13; 
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
x_13 = !lean_is_exclusive(x_8);
if (x_13 == 0)
{
return x_8;
}
else
{
lean_object* x_14; lean_object* x_15; 
x_14 = lean_ctor_get(x_8, 0);
lean_inc(x_14);
lean_dec(x_8);
x_15 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_15, 0, x_14);
return x_15;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_initializeLazyEntry___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
uint8_t x_8; lean_object* x_9; 
x_8 = lean_unbox(x_2);
x_9 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_initializeLazyEntry(x_1, x_8, x_3, x_4, x_5, x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_9; 
x_9 = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_box(0), x_1, x_2, x_3, x_4, x_5, x_6, x_7);
if (lean_obj_tag(x_9) == 0)
{
uint8_t x_10; 
x_10 = !lean_is_exclusive(x_9);
if (x_10 == 0)
{
return x_9;
}
else
{
lean_object* x_11; lean_object* x_12; 
x_11 = lean_ctor_get(x_9, 0);
lean_inc(x_11);
lean_dec(x_9);
x_12 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_12, 0, x_11);
return x_12;
}
}
else
{
uint8_t x_13; 
x_13 = !lean_is_exclusive(x_9);
if (x_13 == 0)
{
return x_9;
}
else
{
lean_object* x_14; lean_object* x_15; 
x_14 = lean_ctor_get(x_9, 0);
lean_inc(x_14);
lean_dec(x_9);
x_15 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_15, 0, x_14);
return x_15;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux___lam__0(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
uint8_t x_11; 
x_11 = !lean_is_exclusive(x_6);
if (x_11 == 0)
{
lean_object* x_12; uint64_t x_13; lean_object* x_14; 
x_12 = lean_ctor_get(x_6, 0);
lean_dec(x_12);
x_13 = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(x_1);
x_14 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_14, 0, x_1);
lean_ctor_set_uint64(x_14, sizeof(void*)*1, x_13);
lean_ctor_set(x_6, 0, x_14);
if (x_2 == 0)
{
lean_object* x_15; 
x_15 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep(x_3, x_2, x_4, x_5, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_15) == 0)
{
uint8_t x_16; 
x_16 = !lean_is_exclusive(x_15);
if (x_16 == 0)
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_17 = lean_ctor_get(x_15, 0);
x_18 = lean_box(0);
x_19 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_19, 0, x_17);
lean_ctor_set(x_19, 1, x_18);
x_20 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_20, 0, x_19);
lean_ctor_set(x_15, 0, x_20);
return x_15;
}
else
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
x_21 = lean_ctor_get(x_15, 0);
lean_inc(x_21);
lean_dec(x_15);
x_22 = lean_box(0);
x_23 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_23, 0, x_21);
lean_ctor_set(x_23, 1, x_22);
x_24 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_24, 0, x_23);
x_25 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_25, 0, x_24);
return x_25;
}
}
else
{
uint8_t x_26; 
x_26 = !lean_is_exclusive(x_15);
if (x_26 == 0)
{
return x_15;
}
else
{
lean_object* x_27; lean_object* x_28; 
x_27 = lean_ctor_get(x_15, 0);
lean_inc(x_27);
lean_dec(x_15);
x_28 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_28, 0, x_27);
return x_28;
}
}
}
else
{
uint8_t x_29; lean_object* x_30; 
x_29 = 0;
x_30 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta(x_3, x_29, x_5, x_4, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_30) == 0)
{
uint8_t x_31; 
x_31 = !lean_is_exclusive(x_30);
if (x_31 == 0)
{
lean_object* x_32; lean_object* x_33; 
x_32 = lean_ctor_get(x_30, 0);
x_33 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_33, 0, x_32);
lean_ctor_set(x_30, 0, x_33);
return x_30;
}
else
{
lean_object* x_34; lean_object* x_35; lean_object* x_36; 
x_34 = lean_ctor_get(x_30, 0);
lean_inc(x_34);
lean_dec(x_30);
x_35 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_35, 0, x_34);
x_36 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_36, 0, x_35);
return x_36;
}
}
else
{
uint8_t x_37; 
x_37 = !lean_is_exclusive(x_30);
if (x_37 == 0)
{
return x_30;
}
else
{
lean_object* x_38; lean_object* x_39; 
x_38 = lean_ctor_get(x_30, 0);
lean_inc(x_38);
lean_dec(x_30);
x_39 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_39, 0, x_38);
return x_39;
}
}
}
}
else
{
uint8_t x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; uint8_t x_47; uint8_t x_48; uint64_t x_49; lean_object* x_50; lean_object* x_51; 
x_40 = lean_ctor_get_uint8(x_6, sizeof(void*)*7);
x_41 = lean_ctor_get(x_6, 1);
x_42 = lean_ctor_get(x_6, 2);
x_43 = lean_ctor_get(x_6, 3);
x_44 = lean_ctor_get(x_6, 4);
x_45 = lean_ctor_get(x_6, 5);
x_46 = lean_ctor_get(x_6, 6);
x_47 = lean_ctor_get_uint8(x_6, sizeof(void*)*7 + 1);
x_48 = lean_ctor_get_uint8(x_6, sizeof(void*)*7 + 2);
lean_inc(x_46);
lean_inc(x_45);
lean_inc(x_44);
lean_inc(x_43);
lean_inc(x_42);
lean_inc(x_41);
lean_dec(x_6);
x_49 = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(x_1);
x_50 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_50, 0, x_1);
lean_ctor_set_uint64(x_50, sizeof(void*)*1, x_49);
x_51 = lean_alloc_ctor(0, 7, 3);
lean_ctor_set(x_51, 0, x_50);
lean_ctor_set(x_51, 1, x_41);
lean_ctor_set(x_51, 2, x_42);
lean_ctor_set(x_51, 3, x_43);
lean_ctor_set(x_51, 4, x_44);
lean_ctor_set(x_51, 5, x_45);
lean_ctor_set(x_51, 6, x_46);
lean_ctor_set_uint8(x_51, sizeof(void*)*7, x_40);
lean_ctor_set_uint8(x_51, sizeof(void*)*7 + 1, x_47);
lean_ctor_set_uint8(x_51, sizeof(void*)*7 + 2, x_48);
if (x_2 == 0)
{
lean_object* x_52; 
x_52 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep(x_3, x_2, x_4, x_5, x_51, x_7, x_8, x_9);
if (lean_obj_tag(x_52) == 0)
{
lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; 
x_53 = lean_ctor_get(x_52, 0);
lean_inc(x_53);
if (lean_is_exclusive(x_52)) {
 lean_ctor_release(x_52, 0);
 x_54 = x_52;
} else {
 lean_dec_ref(x_52);
 x_54 = lean_box(0);
}
x_55 = lean_box(0);
x_56 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_56, 0, x_53);
lean_ctor_set(x_56, 1, x_55);
x_57 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_57, 0, x_56);
if (lean_is_scalar(x_54)) {
 x_58 = lean_alloc_ctor(0, 1, 0);
} else {
 x_58 = x_54;
}
lean_ctor_set(x_58, 0, x_57);
return x_58;
}
else
{
lean_object* x_59; lean_object* x_60; lean_object* x_61; 
x_59 = lean_ctor_get(x_52, 0);
lean_inc(x_59);
if (lean_is_exclusive(x_52)) {
 lean_ctor_release(x_52, 0);
 x_60 = x_52;
} else {
 lean_dec_ref(x_52);
 x_60 = lean_box(0);
}
if (lean_is_scalar(x_60)) {
 x_61 = lean_alloc_ctor(1, 1, 0);
} else {
 x_61 = x_60;
}
lean_ctor_set(x_61, 0, x_59);
return x_61;
}
}
else
{
uint8_t x_62; lean_object* x_63; 
x_62 = 0;
x_63 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta(x_3, x_62, x_5, x_4, x_51, x_7, x_8, x_9);
if (lean_obj_tag(x_63) == 0)
{
lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; 
x_64 = lean_ctor_get(x_63, 0);
lean_inc(x_64);
if (lean_is_exclusive(x_63)) {
 lean_ctor_release(x_63, 0);
 x_65 = x_63;
} else {
 lean_dec_ref(x_63);
 x_65 = lean_box(0);
}
x_66 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_66, 0, x_64);
if (lean_is_scalar(x_65)) {
 x_67 = lean_alloc_ctor(0, 1, 0);
} else {
 x_67 = x_65;
}
lean_ctor_set(x_67, 0, x_66);
return x_67;
}
else
{
lean_object* x_68; lean_object* x_69; lean_object* x_70; 
x_68 = lean_ctor_get(x_63, 0);
lean_inc(x_68);
if (lean_is_exclusive(x_63)) {
 lean_ctor_release(x_63, 0);
 x_69 = x_63;
} else {
 lean_dec_ref(x_63);
 x_69 = lean_box(0);
}
if (lean_is_scalar(x_69)) {
 x_70 = lean_alloc_ctor(1, 1, 0);
} else {
 x_70 = x_69;
}
lean_ctor_set(x_70, 0, x_68);
return x_70;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
uint8_t x_11; lean_object* x_12; 
x_11 = lean_unbox(x_2);
x_12 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux___lam__0(x_1, x_11, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; 
x_8 = lean_ctor_get(x_1, 1);
lean_inc(x_8);
if (lean_obj_tag(x_8) == 0)
{
lean_object* x_9; lean_object* x_10; 
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
x_9 = lean_box(0);
x_10 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_10, 0, x_9);
return x_10;
}
else
{
uint8_t x_11; 
x_11 = !lean_is_exclusive(x_1);
if (x_11 == 0)
{
lean_object* x_12; uint8_t x_13; 
x_12 = lean_ctor_get(x_1, 1);
lean_dec(x_12);
x_13 = !lean_is_exclusive(x_8);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; 
x_14 = lean_ctor_get(x_8, 0);
x_15 = lean_ctor_get(x_8, 1);
lean_ctor_set(x_1, 1, x_15);
if (lean_obj_tag(x_14) == 0)
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
x_16 = lean_box(0);
x_17 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_17, 0, x_16);
lean_ctor_set(x_17, 1, x_1);
x_18 = lean_box(0);
lean_ctor_set(x_8, 1, x_18);
lean_ctor_set(x_8, 0, x_17);
x_19 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_19, 0, x_8);
x_20 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_20, 0, x_19);
return x_20;
}
else
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
lean_free_object(x_8);
x_21 = lean_ctor_get(x_14, 0);
lean_inc_ref(x_21);
lean_dec_ref(x_14);
x_22 = lean_ctor_get(x_21, 0);
lean_inc_ref(x_22);
x_23 = lean_ctor_get(x_21, 1);
lean_inc(x_23);
x_24 = lean_ctor_get(x_21, 2);
lean_inc_ref(x_24);
x_25 = lean_ctor_get(x_21, 3);
lean_inc_ref(x_25);
x_26 = lean_ctor_get(x_21, 4);
lean_inc_ref(x_26);
lean_dec_ref(x_21);
x_27 = lean_box(x_2);
x_28 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux___lam__0___boxed), 10, 5);
lean_closure_set(x_28, 0, x_26);
lean_closure_set(x_28, 1, x_27);
lean_closure_set(x_28, 2, x_22);
lean_closure_set(x_28, 3, x_23);
lean_closure_set(x_28, 4, x_1);
x_29 = lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0___redArg(x_24, x_25, x_28, x_3, x_4, x_5, x_6);
return x_29;
}
}
else
{
lean_object* x_30; lean_object* x_31; 
x_30 = lean_ctor_get(x_8, 0);
x_31 = lean_ctor_get(x_8, 1);
lean_inc(x_31);
lean_inc(x_30);
lean_dec(x_8);
lean_ctor_set(x_1, 1, x_31);
if (lean_obj_tag(x_30) == 0)
{
lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; 
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
x_32 = lean_box(0);
x_33 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_33, 0, x_32);
lean_ctor_set(x_33, 1, x_1);
x_34 = lean_box(0);
x_35 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_35, 0, x_33);
lean_ctor_set(x_35, 1, x_34);
x_36 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_36, 0, x_35);
x_37 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_37, 0, x_36);
return x_37;
}
else
{
lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; 
x_38 = lean_ctor_get(x_30, 0);
lean_inc_ref(x_38);
lean_dec_ref(x_30);
x_39 = lean_ctor_get(x_38, 0);
lean_inc_ref(x_39);
x_40 = lean_ctor_get(x_38, 1);
lean_inc(x_40);
x_41 = lean_ctor_get(x_38, 2);
lean_inc_ref(x_41);
x_42 = lean_ctor_get(x_38, 3);
lean_inc_ref(x_42);
x_43 = lean_ctor_get(x_38, 4);
lean_inc_ref(x_43);
lean_dec_ref(x_38);
x_44 = lean_box(x_2);
x_45 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux___lam__0___boxed), 10, 5);
lean_closure_set(x_45, 0, x_43);
lean_closure_set(x_45, 1, x_44);
lean_closure_set(x_45, 2, x_39);
lean_closure_set(x_45, 3, x_40);
lean_closure_set(x_45, 4, x_1);
x_46 = lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0___redArg(x_41, x_42, x_45, x_3, x_4, x_5, x_6);
return x_46;
}
}
}
else
{
lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; 
x_47 = lean_ctor_get(x_1, 0);
x_48 = lean_ctor_get(x_1, 2);
x_49 = lean_ctor_get(x_1, 3);
x_50 = lean_ctor_get(x_1, 4);
lean_inc(x_50);
lean_inc(x_49);
lean_inc(x_48);
lean_inc(x_47);
lean_dec(x_1);
x_51 = lean_ctor_get(x_8, 0);
lean_inc(x_51);
x_52 = lean_ctor_get(x_8, 1);
lean_inc(x_52);
if (lean_is_exclusive(x_8)) {
 lean_ctor_release(x_8, 0);
 lean_ctor_release(x_8, 1);
 x_53 = x_8;
} else {
 lean_dec_ref(x_8);
 x_53 = lean_box(0);
}
x_54 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_54, 0, x_47);
lean_ctor_set(x_54, 1, x_52);
lean_ctor_set(x_54, 2, x_48);
lean_ctor_set(x_54, 3, x_49);
lean_ctor_set(x_54, 4, x_50);
if (lean_obj_tag(x_51) == 0)
{
lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; 
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
x_55 = lean_box(0);
x_56 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_56, 0, x_55);
lean_ctor_set(x_56, 1, x_54);
x_57 = lean_box(0);
if (lean_is_scalar(x_53)) {
 x_58 = lean_alloc_ctor(1, 2, 0);
} else {
 x_58 = x_53;
}
lean_ctor_set(x_58, 0, x_56);
lean_ctor_set(x_58, 1, x_57);
x_59 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_59, 0, x_58);
x_60 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_60, 0, x_59);
return x_60;
}
else
{
lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; 
lean_dec(x_53);
x_61 = lean_ctor_get(x_51, 0);
lean_inc_ref(x_61);
lean_dec_ref(x_51);
x_62 = lean_ctor_get(x_61, 0);
lean_inc_ref(x_62);
x_63 = lean_ctor_get(x_61, 1);
lean_inc(x_63);
x_64 = lean_ctor_get(x_61, 2);
lean_inc_ref(x_64);
x_65 = lean_ctor_get(x_61, 3);
lean_inc_ref(x_65);
x_66 = lean_ctor_get(x_61, 4);
lean_inc_ref(x_66);
lean_dec_ref(x_61);
x_67 = lean_box(x_2);
x_68 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux___lam__0___boxed), 10, 5);
lean_closure_set(x_68, 0, x_66);
lean_closure_set(x_68, 1, x_67);
lean_closure_set(x_68, 2, x_62);
lean_closure_set(x_68, 3, x_63);
lean_closure_set(x_68, 4, x_54);
x_69 = lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0___redArg(x_64, x_65, x_68, x_3, x_4, x_5, x_6);
return x_69;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
uint8_t x_8; lean_object* x_9; 
x_8 = lean_unbox(x_2);
x_9 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux(x_1, x_8, x_3, x_4, x_5, x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop_reduce(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; lean_object* x_12; 
x_11 = lean_expr_instantiate_rev_range(x_2, x_4, x_3, x_1);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
x_12 = l_Lean_Meta_whnfD(x_11, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_12) == 0)
{
lean_object* x_13; 
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
if (lean_obj_tag(x_13) == 7)
{
lean_object* x_14; lean_object* x_15; uint8_t x_16; lean_object* x_17; lean_object* x_18; 
x_14 = lean_ctor_get(x_13, 1);
lean_inc_ref(x_14);
x_15 = lean_ctor_get(x_13, 2);
lean_inc_ref(x_15);
x_16 = lean_ctor_get_uint8(x_13, sizeof(void*)*3 + 8);
lean_dec_ref(x_13);
x_17 = lean_box(x_16);
x_18 = lean_apply_9(x_5, x_3, x_14, x_15, x_17, x_6, x_7, x_8, x_9, lean_box(0));
return x_18;
}
else
{
lean_object* x_19; 
lean_dec_ref(x_5);
lean_dec(x_3);
x_19 = l_Lean_Meta_throwFunctionExpected___redArg(x_13, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
return x_19;
}
}
else
{
uint8_t x_20; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec(x_3);
x_20 = !lean_is_exclusive(x_12);
if (x_20 == 0)
{
return x_12;
}
else
{
lean_object* x_21; lean_object* x_22; 
x_21 = lean_ctor_get(x_12, 0);
lean_inc(x_21);
lean_dec(x_12);
x_22 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_22, 0, x_21);
return x_22;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop_reduce___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop_reduce(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_4);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_isIgnoredArg(lean_object* x_1, lean_object* x_2, uint8_t x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
uint8_t x_9; uint8_t x_10; 
x_9 = lean_is_out_param(x_2);
x_10 = 1;
if (x_9 == 0)
{
switch (x_3) {
case 0:
{
lean_object* x_11; 
x_11 = l_Lean_Meta_isProof(x_1, x_4, x_5, x_6, x_7);
return x_11;
}
case 3:
{
lean_object* x_12; lean_object* x_13; 
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_1);
x_12 = lean_box(x_10);
x_13 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_13, 0, x_12);
return x_13;
}
default: 
{
lean_object* x_14; 
x_14 = l_Lean_Meta_isType(x_1, x_4, x_5, x_6, x_7);
if (lean_obj_tag(x_14) == 0)
{
uint8_t x_15; 
x_15 = !lean_is_exclusive(x_14);
if (x_15 == 0)
{
lean_object* x_16; uint8_t x_17; 
x_16 = lean_ctor_get(x_14, 0);
x_17 = lean_unbox(x_16);
lean_dec(x_16);
if (x_17 == 0)
{
lean_object* x_18; 
x_18 = lean_box(x_10);
lean_ctor_set(x_14, 0, x_18);
return x_14;
}
else
{
lean_object* x_19; 
x_19 = lean_box(x_9);
lean_ctor_set(x_14, 0, x_19);
return x_14;
}
}
else
{
lean_object* x_20; uint8_t x_21; 
x_20 = lean_ctor_get(x_14, 0);
lean_inc(x_20);
lean_dec(x_14);
x_21 = lean_unbox(x_20);
lean_dec(x_20);
if (x_21 == 0)
{
lean_object* x_22; lean_object* x_23; 
x_22 = lean_box(x_10);
x_23 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_23, 0, x_22);
return x_23;
}
else
{
lean_object* x_24; lean_object* x_25; 
x_24 = lean_box(x_9);
x_25 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_25, 0, x_24);
return x_25;
}
}
}
else
{
return x_14;
}
}
}
}
else
{
lean_object* x_26; lean_object* x_27; 
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_1);
x_26 = lean_box(x_10);
x_27 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_27, 0, x_26);
return x_27;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_isIgnoredArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
uint8_t x_9; lean_object* x_10; 
x_9 = lean_unbox(x_3);
x_10 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_isIgnoredArg(x_1, x_2, x_9, x_4, x_5, x_6, x_7);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, uint8_t x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_15; 
lean_inc(x_13);
lean_inc_ref(x_12);
lean_inc(x_11);
lean_inc_ref(x_10);
lean_inc_ref(x_1);
x_15 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_isIgnoredArg(x_1, x_7, x_9, x_10, x_11, x_12, x_13);
if (lean_obj_tag(x_15) == 0)
{
lean_object* x_16; uint8_t x_17; 
x_16 = lean_ctor_get(x_15, 0);
lean_inc(x_16);
lean_dec_ref(x_15);
x_17 = lean_unbox(x_16);
lean_dec(x_16);
if (x_17 == 0)
{
lean_object* x_18; 
lean_inc(x_2);
x_18 = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkExprInfo___redArg(x_1, x_2, x_10);
if (lean_obj_tag(x_18) == 0)
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_19 = lean_ctor_get(x_18, 0);
lean_inc(x_19);
lean_dec_ref(x_18);
x_20 = lean_unsigned_to_nat(1u);
x_21 = lean_nat_add(x_3, x_20);
x_22 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_22, 0, x_19);
x_23 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_23, 0, x_22);
lean_ctor_set(x_23, 1, x_4);
x_24 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop(x_5, x_2, x_8, x_21, x_6, x_23, x_10, x_11, x_12, x_13);
return x_24;
}
else
{
uint8_t x_25; 
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_8);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec(x_2);
x_25 = !lean_is_exclusive(x_18);
if (x_25 == 0)
{
return x_18;
}
else
{
lean_object* x_26; lean_object* x_27; 
x_26 = lean_ctor_get(x_18, 0);
lean_inc(x_26);
lean_dec(x_18);
x_27 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_27, 0, x_26);
return x_27;
}
}
}
else
{
lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; 
lean_dec_ref(x_1);
x_28 = lean_unsigned_to_nat(1u);
x_29 = lean_nat_add(x_3, x_28);
x_30 = lean_box(0);
x_31 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_31, 0, x_30);
lean_ctor_set(x_31, 1, x_4);
x_32 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop(x_5, x_2, x_8, x_29, x_6, x_31, x_10, x_11, x_12, x_13);
return x_32;
}
}
else
{
uint8_t x_33; 
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_8);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec(x_2);
lean_dec_ref(x_1);
x_33 = !lean_is_exclusive(x_15);
if (x_33 == 0)
{
return x_15;
}
else
{
lean_object* x_34; lean_object* x_35; 
x_34 = lean_ctor_get(x_15, 0);
lean_inc(x_34);
lean_dec(x_15);
x_35 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_35, 0, x_34);
return x_35;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
uint8_t x_15; lean_object* x_16; 
x_15 = lean_unbox(x_9);
x_16 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_15, x_10, x_11, x_12, x_13);
lean_dec(x_6);
lean_dec(x_3);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; uint8_t x_13; 
x_12 = lean_array_get_size(x_1);
x_13 = lean_nat_dec_lt(x_4, x_12);
if (x_13 == 0)
{
lean_object* x_14; 
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
x_14 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_14, 0, x_6);
return x_14;
}
else
{
lean_object* x_15; lean_object* x_16; 
x_15 = lean_array_fget_borrowed(x_1, x_4);
lean_inc_ref(x_1);
lean_inc(x_6);
lean_inc(x_4);
lean_inc(x_2);
lean_inc(x_15);
x_16 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop___lam__0___boxed), 14, 5);
lean_closure_set(x_16, 0, x_15);
lean_closure_set(x_16, 1, x_2);
lean_closure_set(x_16, 2, x_4);
lean_closure_set(x_16, 3, x_6);
lean_closure_set(x_16, 4, x_1);
if (lean_obj_tag(x_3) == 7)
{
lean_object* x_17; lean_object* x_18; uint8_t x_19; lean_object* x_20; 
lean_dec_ref(x_16);
lean_inc(x_15);
x_17 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_17);
x_18 = lean_ctor_get(x_3, 2);
lean_inc_ref(x_18);
x_19 = lean_ctor_get_uint8(x_3, sizeof(void*)*3 + 8);
lean_dec_ref(x_3);
x_20 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop___lam__0(x_15, x_2, x_4, x_6, x_1, x_5, x_17, x_18, x_19, x_7, x_8, x_9, x_10);
lean_dec(x_4);
return x_20;
}
else
{
lean_object* x_21; 
lean_dec(x_6);
lean_dec(x_2);
x_21 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop_reduce(x_1, x_3, x_4, x_5, x_16, x_7, x_8, x_9, x_10);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
return x_21;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_5);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_9; 
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc_ref(x_4);
x_9 = lean_infer_type(x_1, x_4, x_5, x_6, x_7);
if (lean_obj_tag(x_9) == 0)
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
lean_dec_ref(x_9);
x_11 = lean_unsigned_to_nat(0u);
x_12 = lean_box(0);
x_13 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop(x_2, x_3, x_10, x_11, x_11, x_12, x_4, x_5, x_6, x_7);
return x_13;
}
else
{
uint8_t x_14; 
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_14 = !lean_is_exclusive(x_9);
if (x_14 == 0)
{
return x_9;
}
else
{
lean_object* x_15; lean_object* x_16; 
x_15 = lean_ctor_get(x_9, 0);
lean_inc(x_15);
lean_dec(x_9);
x_16 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_16, 0, x_15);
return x_16;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; 
x_8 = lean_apply_6(x_1, x_2, x_3, x_4, x_5, x_6, lean_box(0));
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___redArg(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, uint8_t x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; lean_object* x_12; 
x_11 = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___redArg___lam__0___boxed), 7, 1);
lean_closure_set(x_11, 0, x_4);
x_12 = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), x_1, x_2, x_3, x_11, x_5, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_12) == 0)
{
uint8_t x_13; 
x_13 = !lean_is_exclusive(x_12);
if (x_13 == 0)
{
return x_12;
}
else
{
lean_object* x_14; lean_object* x_15; 
x_14 = lean_ctor_get(x_12, 0);
lean_inc(x_14);
lean_dec(x_12);
x_15 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_15, 0, x_14);
return x_15;
}
}
else
{
uint8_t x_16; 
x_16 = !lean_is_exclusive(x_12);
if (x_16 == 0)
{
return x_12;
}
else
{
lean_object* x_17; lean_object* x_18; 
x_17 = lean_ctor_get(x_12, 0);
lean_inc(x_17);
lean_dec(x_12);
x_18 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_18, 0, x_17);
return x_18;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0(lean_object* x_1, lean_object* x_2, uint8_t x_3, lean_object* x_4, lean_object* x_5, uint8_t x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lean_expr_instantiate1(x_1, x_3);
x_10 = l_Lean_Expr_fvarId_x21(x_3);
x_11 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_11, 0, x_10);
lean_ctor_set(x_11, 1, x_2);
x_12 = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkExprInfo___redArg(x_9, x_11, x_4);
if (lean_obj_tag(x_12) == 0)
{
uint8_t x_13; 
x_13 = !lean_is_exclusive(x_12);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; 
x_14 = lean_ctor_get(x_12, 0);
x_15 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_15, 0, x_14);
lean_ctor_set(x_12, 0, x_15);
return x_12;
}
else
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_16 = lean_ctor_get(x_12, 0);
lean_inc(x_16);
lean_dec(x_12);
x_17 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_17, 0, x_16);
x_18 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_18, 0, x_17);
return x_18;
}
}
else
{
uint8_t x_19; 
x_19 = !lean_is_exclusive(x_12);
if (x_19 == 0)
{
return x_12;
}
else
{
lean_object* x_20; lean_object* x_21; 
x_20 = lean_ctor_get(x_12, 0);
lean_inc(x_20);
lean_dec(x_12);
x_21 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_21, 0, x_20);
return x_21;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___lam__1(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries(x_1, x_2, x_3, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_10) == 0)
{
uint8_t x_11; 
x_11 = !lean_is_exclusive(x_10);
if (x_11 == 0)
{
uint8_t x_12; 
x_12 = !lean_is_exclusive(x_4);
if (x_12 == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_13 = lean_ctor_get(x_10, 0);
x_14 = lean_ctor_get(x_4, 1);
x_15 = l_List_reverseAux___redArg(x_13, x_14);
lean_ctor_set(x_4, 1, x_15);
lean_ctor_set(x_10, 0, x_4);
return x_10;
}
else
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; 
x_16 = lean_ctor_get(x_10, 0);
x_17 = lean_ctor_get(x_4, 0);
x_18 = lean_ctor_get(x_4, 1);
x_19 = lean_ctor_get(x_4, 2);
x_20 = lean_ctor_get(x_4, 3);
x_21 = lean_ctor_get(x_4, 4);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_17);
lean_dec(x_4);
x_22 = l_List_reverseAux___redArg(x_16, x_18);
x_23 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_23, 0, x_17);
lean_ctor_set(x_23, 1, x_22);
lean_ctor_set(x_23, 2, x_19);
lean_ctor_set(x_23, 3, x_20);
lean_ctor_set(x_23, 4, x_21);
lean_ctor_set(x_10, 0, x_23);
return x_10;
}
}
else
{
lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; 
x_24 = lean_ctor_get(x_10, 0);
lean_inc(x_24);
lean_dec(x_10);
x_25 = lean_ctor_get(x_4, 0);
lean_inc(x_25);
x_26 = lean_ctor_get(x_4, 1);
lean_inc(x_26);
x_27 = lean_ctor_get(x_4, 2);
lean_inc_ref(x_27);
x_28 = lean_ctor_get(x_4, 3);
lean_inc(x_28);
x_29 = lean_ctor_get(x_4, 4);
lean_inc(x_29);
if (lean_is_exclusive(x_4)) {
 lean_ctor_release(x_4, 0);
 lean_ctor_release(x_4, 1);
 lean_ctor_release(x_4, 2);
 lean_ctor_release(x_4, 3);
 lean_ctor_release(x_4, 4);
 x_30 = x_4;
} else {
 lean_dec_ref(x_4);
 x_30 = lean_box(0);
}
x_31 = l_List_reverseAux___redArg(x_24, x_26);
if (lean_is_scalar(x_30)) {
 x_32 = lean_alloc_ctor(0, 5, 0);
} else {
 x_32 = x_30;
}
lean_ctor_set(x_32, 0, x_25);
lean_ctor_set(x_32, 1, x_31);
lean_ctor_set(x_32, 2, x_27);
lean_ctor_set(x_32, 3, x_28);
lean_ctor_set(x_32, 4, x_29);
x_33 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_33, 0, x_32);
return x_33;
}
}
else
{
uint8_t x_34; 
lean_dec_ref(x_4);
x_34 = !lean_is_exclusive(x_10);
if (x_34 == 0)
{
return x_10;
}
else
{
lean_object* x_35; lean_object* x_36; 
x_35 = lean_ctor_get(x_10, 0);
lean_inc(x_35);
lean_dec(x_10);
x_36 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_36, 0, x_35);
return x_36;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
if (lean_obj_tag(x_8) == 5)
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_16 = lean_ctor_get(x_8, 0);
lean_inc_ref(x_16);
x_17 = lean_ctor_get(x_8, 1);
lean_inc_ref(x_17);
lean_dec_ref(x_8);
x_18 = lean_array_set(x_9, x_10, x_17);
x_19 = lean_unsigned_to_nat(1u);
x_20 = lean_nat_sub(x_10, x_19);
lean_dec(x_10);
x_8 = x_16;
x_9 = x_18;
x_10 = x_20;
goto _start;
}
else
{
lean_dec(x_10);
switch (lean_obj_tag(x_8)) {
case 7:
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; uint8_t x_25; lean_object* x_26; 
lean_dec_ref(x_9);
lean_dec_ref(x_7);
x_22 = lean_ctor_get(x_8, 0);
lean_inc(x_22);
x_23 = lean_ctor_get(x_8, 1);
lean_inc_ref(x_23);
x_24 = lean_ctor_get(x_8, 2);
lean_inc_ref(x_24);
x_25 = lean_ctor_get_uint8(x_8, sizeof(void*)*3 + 8);
lean_dec_ref(x_8);
lean_inc(x_1);
lean_inc_ref(x_23);
x_26 = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkExprInfo___redArg(x_23, x_1, x_11);
if (lean_obj_tag(x_26) == 0)
{
lean_object* x_27; lean_object* x_28; uint8_t x_29; lean_object* x_30; 
x_27 = lean_ctor_get(x_26, 0);
lean_inc(x_27);
lean_dec_ref(x_26);
x_28 = lean_alloc_closure((void*)(lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___lam__1___boxed), 8, 2);
lean_closure_set(x_28, 0, x_24);
lean_closure_set(x_28, 1, x_1);
x_29 = 0;
x_30 = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___redArg(x_22, x_25, x_23, x_28, x_29, x_11, x_12, x_13, x_14);
if (lean_obj_tag(x_30) == 0)
{
uint8_t x_31; 
x_31 = !lean_is_exclusive(x_30);
if (x_31 == 0)
{
lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; 
x_32 = lean_ctor_get(x_30, 0);
x_33 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_33, 0, x_27);
x_34 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_34, 0, x_32);
lean_ctor_set(x_34, 1, x_2);
x_35 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_35, 0, x_33);
lean_ctor_set(x_35, 1, x_34);
x_36 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_36, 0, x_3);
lean_ctor_set(x_36, 1, x_35);
lean_ctor_set(x_36, 2, x_4);
lean_ctor_set(x_36, 3, x_5);
lean_ctor_set(x_36, 4, x_6);
lean_ctor_set(x_30, 0, x_36);
return x_30;
}
else
{
lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; 
x_37 = lean_ctor_get(x_30, 0);
lean_inc(x_37);
lean_dec(x_30);
x_38 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_38, 0, x_27);
x_39 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_39, 0, x_37);
lean_ctor_set(x_39, 1, x_2);
x_40 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_40, 0, x_38);
lean_ctor_set(x_40, 1, x_39);
x_41 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_41, 0, x_3);
lean_ctor_set(x_41, 1, x_40);
lean_ctor_set(x_41, 2, x_4);
lean_ctor_set(x_41, 3, x_5);
lean_ctor_set(x_41, 4, x_6);
x_42 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_42, 0, x_41);
return x_42;
}
}
else
{
uint8_t x_43; 
lean_dec(x_27);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec(x_2);
x_43 = !lean_is_exclusive(x_30);
if (x_43 == 0)
{
return x_30;
}
else
{
lean_object* x_44; lean_object* x_45; 
x_44 = lean_ctor_get(x_30, 0);
lean_inc(x_44);
lean_dec(x_30);
x_45 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_45, 0, x_44);
return x_45;
}
}
}
else
{
uint8_t x_46; 
lean_dec_ref(x_24);
lean_dec_ref(x_23);
lean_dec(x_22);
lean_dec(x_14);
lean_dec_ref(x_13);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec(x_2);
lean_dec(x_1);
x_46 = !lean_is_exclusive(x_26);
if (x_46 == 0)
{
return x_26;
}
else
{
lean_object* x_47; lean_object* x_48; 
x_47 = lean_ctor_get(x_26, 0);
lean_inc(x_47);
lean_dec(x_26);
x_48 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_48, 0, x_47);
return x_48;
}
}
}
case 11:
{
lean_object* x_49; lean_object* x_50; lean_object* x_51; 
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec(x_2);
x_49 = lean_ctor_get(x_8, 0);
lean_inc(x_49);
x_50 = lean_ctor_get(x_8, 2);
lean_inc_ref(x_50);
lean_inc(x_14);
lean_inc_ref(x_11);
lean_inc(x_1);
x_51 = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___lam__0(x_8, x_9, x_1, x_7, x_11, x_12, x_13, x_14);
if (lean_obj_tag(x_51) == 0)
{
uint8_t x_52; 
x_52 = !lean_is_exclusive(x_51);
if (x_52 == 0)
{
lean_object* x_53; lean_object* x_54; lean_object* x_55; uint8_t x_56; 
x_53 = lean_ctor_get(x_51, 0);
x_54 = lean_st_ref_get(x_14);
lean_dec(x_14);
x_55 = lean_ctor_get(x_54, 0);
lean_inc_ref(x_55);
lean_dec(x_54);
x_56 = lean_is_class(x_55, x_49);
if (x_56 == 0)
{
lean_object* x_57; 
lean_free_object(x_51);
x_57 = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkExprInfo___redArg(x_50, x_1, x_11);
lean_dec_ref(x_11);
if (lean_obj_tag(x_57) == 0)
{
uint8_t x_58; 
x_58 = !lean_is_exclusive(x_57);
if (x_58 == 0)
{
uint8_t x_59; 
x_59 = !lean_is_exclusive(x_53);
if (x_59 == 0)
{
lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; 
x_60 = lean_ctor_get(x_57, 0);
x_61 = lean_ctor_get(x_53, 1);
x_62 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_62, 0, x_60);
x_63 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_63, 0, x_62);
lean_ctor_set(x_63, 1, x_61);
lean_ctor_set(x_53, 1, x_63);
lean_ctor_set(x_57, 0, x_53);
return x_57;
}
else
{
lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; 
x_64 = lean_ctor_get(x_57, 0);
x_65 = lean_ctor_get(x_53, 0);
x_66 = lean_ctor_get(x_53, 1);
x_67 = lean_ctor_get(x_53, 2);
x_68 = lean_ctor_get(x_53, 3);
x_69 = lean_ctor_get(x_53, 4);
lean_inc(x_69);
lean_inc(x_68);
lean_inc(x_67);
lean_inc(x_66);
lean_inc(x_65);
lean_dec(x_53);
x_70 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_70, 0, x_64);
x_71 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_71, 0, x_70);
lean_ctor_set(x_71, 1, x_66);
x_72 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_72, 0, x_65);
lean_ctor_set(x_72, 1, x_71);
lean_ctor_set(x_72, 2, x_67);
lean_ctor_set(x_72, 3, x_68);
lean_ctor_set(x_72, 4, x_69);
lean_ctor_set(x_57, 0, x_72);
return x_57;
}
}
else
{
lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; 
x_73 = lean_ctor_get(x_57, 0);
lean_inc(x_73);
lean_dec(x_57);
x_74 = lean_ctor_get(x_53, 0);
lean_inc(x_74);
x_75 = lean_ctor_get(x_53, 1);
lean_inc(x_75);
x_76 = lean_ctor_get(x_53, 2);
lean_inc_ref(x_76);
x_77 = lean_ctor_get(x_53, 3);
lean_inc(x_77);
x_78 = lean_ctor_get(x_53, 4);
lean_inc(x_78);
if (lean_is_exclusive(x_53)) {
 lean_ctor_release(x_53, 0);
 lean_ctor_release(x_53, 1);
 lean_ctor_release(x_53, 2);
 lean_ctor_release(x_53, 3);
 lean_ctor_release(x_53, 4);
 x_79 = x_53;
} else {
 lean_dec_ref(x_53);
 x_79 = lean_box(0);
}
x_80 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_80, 0, x_73);
x_81 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_81, 0, x_80);
lean_ctor_set(x_81, 1, x_75);
if (lean_is_scalar(x_79)) {
 x_82 = lean_alloc_ctor(0, 5, 0);
} else {
 x_82 = x_79;
}
lean_ctor_set(x_82, 0, x_74);
lean_ctor_set(x_82, 1, x_81);
lean_ctor_set(x_82, 2, x_76);
lean_ctor_set(x_82, 3, x_77);
lean_ctor_set(x_82, 4, x_78);
x_83 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_83, 0, x_82);
return x_83;
}
}
else
{
uint8_t x_84; 
lean_dec(x_53);
x_84 = !lean_is_exclusive(x_57);
if (x_84 == 0)
{
return x_57;
}
else
{
lean_object* x_85; lean_object* x_86; 
x_85 = lean_ctor_get(x_57, 0);
lean_inc(x_85);
lean_dec(x_57);
x_86 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_86, 0, x_85);
return x_86;
}
}
}
else
{
uint8_t x_87; 
lean_dec_ref(x_50);
lean_dec_ref(x_11);
lean_dec(x_1);
x_87 = !lean_is_exclusive(x_53);
if (x_87 == 0)
{
lean_object* x_88; lean_object* x_89; lean_object* x_90; 
x_88 = lean_ctor_get(x_53, 1);
x_89 = lean_box(0);
x_90 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_90, 0, x_89);
lean_ctor_set(x_90, 1, x_88);
lean_ctor_set(x_53, 1, x_90);
return x_51;
}
else
{
lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; 
x_91 = lean_ctor_get(x_53, 0);
x_92 = lean_ctor_get(x_53, 1);
x_93 = lean_ctor_get(x_53, 2);
x_94 = lean_ctor_get(x_53, 3);
x_95 = lean_ctor_get(x_53, 4);
lean_inc(x_95);
lean_inc(x_94);
lean_inc(x_93);
lean_inc(x_92);
lean_inc(x_91);
lean_dec(x_53);
x_96 = lean_box(0);
x_97 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_97, 0, x_96);
lean_ctor_set(x_97, 1, x_92);
x_98 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_98, 0, x_91);
lean_ctor_set(x_98, 1, x_97);
lean_ctor_set(x_98, 2, x_93);
lean_ctor_set(x_98, 3, x_94);
lean_ctor_set(x_98, 4, x_95);
lean_ctor_set(x_51, 0, x_98);
return x_51;
}
}
}
else
{
lean_object* x_99; lean_object* x_100; lean_object* x_101; uint8_t x_102; 
x_99 = lean_ctor_get(x_51, 0);
lean_inc(x_99);
lean_dec(x_51);
x_100 = lean_st_ref_get(x_14);
lean_dec(x_14);
x_101 = lean_ctor_get(x_100, 0);
lean_inc_ref(x_101);
lean_dec(x_100);
x_102 = lean_is_class(x_101, x_49);
if (x_102 == 0)
{
lean_object* x_103; 
x_103 = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkExprInfo___redArg(x_50, x_1, x_11);
lean_dec_ref(x_11);
if (lean_obj_tag(x_103) == 0)
{
lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; 
x_104 = lean_ctor_get(x_103, 0);
lean_inc(x_104);
if (lean_is_exclusive(x_103)) {
 lean_ctor_release(x_103, 0);
 x_105 = x_103;
} else {
 lean_dec_ref(x_103);
 x_105 = lean_box(0);
}
x_106 = lean_ctor_get(x_99, 0);
lean_inc(x_106);
x_107 = lean_ctor_get(x_99, 1);
lean_inc(x_107);
x_108 = lean_ctor_get(x_99, 2);
lean_inc_ref(x_108);
x_109 = lean_ctor_get(x_99, 3);
lean_inc(x_109);
x_110 = lean_ctor_get(x_99, 4);
lean_inc(x_110);
if (lean_is_exclusive(x_99)) {
 lean_ctor_release(x_99, 0);
 lean_ctor_release(x_99, 1);
 lean_ctor_release(x_99, 2);
 lean_ctor_release(x_99, 3);
 lean_ctor_release(x_99, 4);
 x_111 = x_99;
} else {
 lean_dec_ref(x_99);
 x_111 = lean_box(0);
}
x_112 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_112, 0, x_104);
x_113 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_113, 0, x_112);
lean_ctor_set(x_113, 1, x_107);
if (lean_is_scalar(x_111)) {
 x_114 = lean_alloc_ctor(0, 5, 0);
} else {
 x_114 = x_111;
}
lean_ctor_set(x_114, 0, x_106);
lean_ctor_set(x_114, 1, x_113);
lean_ctor_set(x_114, 2, x_108);
lean_ctor_set(x_114, 3, x_109);
lean_ctor_set(x_114, 4, x_110);
if (lean_is_scalar(x_105)) {
 x_115 = lean_alloc_ctor(0, 1, 0);
} else {
 x_115 = x_105;
}
lean_ctor_set(x_115, 0, x_114);
return x_115;
}
else
{
lean_object* x_116; lean_object* x_117; lean_object* x_118; 
lean_dec(x_99);
x_116 = lean_ctor_get(x_103, 0);
lean_inc(x_116);
if (lean_is_exclusive(x_103)) {
 lean_ctor_release(x_103, 0);
 x_117 = x_103;
} else {
 lean_dec_ref(x_103);
 x_117 = lean_box(0);
}
if (lean_is_scalar(x_117)) {
 x_118 = lean_alloc_ctor(1, 1, 0);
} else {
 x_118 = x_117;
}
lean_ctor_set(x_118, 0, x_116);
return x_118;
}
}
else
{
lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_127; lean_object* x_128; 
lean_dec_ref(x_50);
lean_dec_ref(x_11);
lean_dec(x_1);
x_119 = lean_ctor_get(x_99, 0);
lean_inc(x_119);
x_120 = lean_ctor_get(x_99, 1);
lean_inc(x_120);
x_121 = lean_ctor_get(x_99, 2);
lean_inc_ref(x_121);
x_122 = lean_ctor_get(x_99, 3);
lean_inc(x_122);
x_123 = lean_ctor_get(x_99, 4);
lean_inc(x_123);
if (lean_is_exclusive(x_99)) {
 lean_ctor_release(x_99, 0);
 lean_ctor_release(x_99, 1);
 lean_ctor_release(x_99, 2);
 lean_ctor_release(x_99, 3);
 lean_ctor_release(x_99, 4);
 x_124 = x_99;
} else {
 lean_dec_ref(x_99);
 x_124 = lean_box(0);
}
x_125 = lean_box(0);
x_126 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_126, 0, x_125);
lean_ctor_set(x_126, 1, x_120);
if (lean_is_scalar(x_124)) {
 x_127 = lean_alloc_ctor(0, 5, 0);
} else {
 x_127 = x_124;
}
lean_ctor_set(x_127, 0, x_119);
lean_ctor_set(x_127, 1, x_126);
lean_ctor_set(x_127, 2, x_121);
lean_ctor_set(x_127, 3, x_122);
lean_ctor_set(x_127, 4, x_123);
x_128 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_128, 0, x_127);
return x_128;
}
}
}
else
{
lean_dec_ref(x_50);
lean_dec(x_49);
lean_dec(x_14);
lean_dec_ref(x_11);
lean_dec(x_1);
return x_51;
}
}
default: 
{
lean_object* x_129; 
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec(x_2);
x_129 = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___lam__0(x_8, x_9, x_1, x_7, x_11, x_12, x_13, x_14);
return x_129;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
uint8_t x_17; 
x_17 = !lean_is_exclusive(x_12);
if (x_17 == 0)
{
lean_object* x_18; uint64_t x_19; lean_object* x_20; lean_object* x_21; 
x_18 = lean_ctor_get(x_12, 0);
lean_dec(x_18);
x_19 = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(x_1);
x_20 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_20, 0, x_1);
lean_ctor_set_uint64(x_20, sizeof(void*)*1, x_19);
lean_ctor_set(x_12, 0, x_20);
x_21 = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
return x_21;
}
else
{
uint8_t x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; uint8_t x_29; uint8_t x_30; uint64_t x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; 
x_22 = lean_ctor_get_uint8(x_12, sizeof(void*)*7);
x_23 = lean_ctor_get(x_12, 1);
x_24 = lean_ctor_get(x_12, 2);
x_25 = lean_ctor_get(x_12, 3);
x_26 = lean_ctor_get(x_12, 4);
x_27 = lean_ctor_get(x_12, 5);
x_28 = lean_ctor_get(x_12, 6);
x_29 = lean_ctor_get_uint8(x_12, sizeof(void*)*7 + 1);
x_30 = lean_ctor_get_uint8(x_12, sizeof(void*)*7 + 2);
lean_inc(x_28);
lean_inc(x_27);
lean_inc(x_26);
lean_inc(x_25);
lean_inc(x_24);
lean_inc(x_23);
lean_dec(x_12);
x_31 = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(x_1);
x_32 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_32, 0, x_1);
lean_ctor_set_uint64(x_32, sizeof(void*)*1, x_31);
x_33 = lean_alloc_ctor(0, 7, 3);
lean_ctor_set(x_33, 0, x_32);
lean_ctor_set(x_33, 1, x_23);
lean_ctor_set(x_33, 2, x_24);
lean_ctor_set(x_33, 3, x_25);
lean_ctor_set(x_33, 4, x_26);
lean_ctor_set(x_33, 5, x_27);
lean_ctor_set(x_33, 6, x_28);
lean_ctor_set_uint8(x_33, sizeof(void*)*7, x_22);
lean_ctor_set_uint8(x_33, sizeof(void*)*7 + 1, x_29);
lean_ctor_set_uint8(x_33, sizeof(void*)*7 + 2, x_30);
x_34 = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_33, x_13, x_14, x_15);
return x_34;
}
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_box(0);
x_2 = l_Lean_Expr_sort___override(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
lean_object* x_17; 
x_17 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; 
x_7 = lean_ctor_get(x_1, 0);
if (lean_obj_tag(x_7) == 1)
{
lean_object* x_8; uint8_t x_9; 
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
x_9 = !lean_is_exclusive(x_1);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; 
x_10 = lean_ctor_get(x_1, 1);
x_11 = lean_ctor_get(x_1, 2);
x_12 = lean_ctor_get(x_1, 3);
x_13 = lean_ctor_get(x_1, 4);
x_14 = lean_ctor_get(x_1, 0);
lean_dec(x_14);
x_15 = lean_ctor_get(x_8, 0);
lean_inc_ref(x_15);
x_16 = lean_ctor_get(x_8, 1);
lean_inc(x_16);
x_17 = lean_ctor_get(x_8, 2);
lean_inc_ref(x_17);
x_18 = lean_ctor_get(x_8, 3);
lean_inc_ref(x_18);
x_19 = lean_ctor_get(x_8, 4);
lean_inc_ref(x_19);
lean_dec(x_8);
x_20 = lean_box(0);
lean_inc(x_13);
lean_inc(x_12);
lean_inc_ref(x_11);
lean_inc(x_10);
lean_ctor_set(x_1, 0, x_20);
x_21 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___closed__0;
x_22 = l_Lean_Expr_getAppNumArgs(x_15);
lean_inc(x_22);
x_23 = lean_mk_array(x_22, x_21);
x_24 = lean_unsigned_to_nat(1u);
x_25 = lean_nat_sub(x_22, x_24);
lean_dec(x_22);
x_26 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___lam__0___boxed), 16, 11);
lean_closure_set(x_26, 0, x_19);
lean_closure_set(x_26, 1, x_16);
lean_closure_set(x_26, 2, x_10);
lean_closure_set(x_26, 3, x_20);
lean_closure_set(x_26, 4, x_11);
lean_closure_set(x_26, 5, x_12);
lean_closure_set(x_26, 6, x_13);
lean_closure_set(x_26, 7, x_1);
lean_closure_set(x_26, 8, x_15);
lean_closure_set(x_26, 9, x_23);
lean_closure_set(x_26, 10, x_25);
x_27 = lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0___redArg(x_17, x_18, x_26, x_2, x_3, x_4, x_5);
return x_27;
}
else
{
lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; 
x_28 = lean_ctor_get(x_1, 1);
x_29 = lean_ctor_get(x_1, 2);
x_30 = lean_ctor_get(x_1, 3);
x_31 = lean_ctor_get(x_1, 4);
lean_inc(x_31);
lean_inc(x_30);
lean_inc(x_29);
lean_inc(x_28);
lean_dec(x_1);
x_32 = lean_ctor_get(x_8, 0);
lean_inc_ref(x_32);
x_33 = lean_ctor_get(x_8, 1);
lean_inc(x_33);
x_34 = lean_ctor_get(x_8, 2);
lean_inc_ref(x_34);
x_35 = lean_ctor_get(x_8, 3);
lean_inc_ref(x_35);
x_36 = lean_ctor_get(x_8, 4);
lean_inc_ref(x_36);
lean_dec(x_8);
x_37 = lean_box(0);
lean_inc(x_31);
lean_inc(x_30);
lean_inc_ref(x_29);
lean_inc(x_28);
x_38 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_38, 0, x_37);
lean_ctor_set(x_38, 1, x_28);
lean_ctor_set(x_38, 2, x_29);
lean_ctor_set(x_38, 3, x_30);
lean_ctor_set(x_38, 4, x_31);
x_39 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___closed__0;
x_40 = l_Lean_Expr_getAppNumArgs(x_32);
lean_inc(x_40);
x_41 = lean_mk_array(x_40, x_39);
x_42 = lean_unsigned_to_nat(1u);
x_43 = lean_nat_sub(x_40, x_42);
lean_dec(x_40);
x_44 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___lam__0___boxed), 16, 11);
lean_closure_set(x_44, 0, x_36);
lean_closure_set(x_44, 1, x_33);
lean_closure_set(x_44, 2, x_28);
lean_closure_set(x_44, 3, x_37);
lean_closure_set(x_44, 4, x_29);
lean_closure_set(x_44, 5, x_30);
lean_closure_set(x_44, 6, x_31);
lean_closure_set(x_44, 7, x_38);
lean_closure_set(x_44, 8, x_32);
lean_closure_set(x_44, 9, x_41);
lean_closure_set(x_44, 10, x_43);
x_45 = lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0___redArg(x_34, x_35, x_44, x_2, x_3, x_4, x_5);
return x_45;
}
}
else
{
lean_object* x_46; 
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_46 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_46, 0, x_1);
return x_46;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
uint8_t x_12; uint8_t x_13; lean_object* x_14; 
x_12 = lean_unbox(x_3);
x_13 = lean_unbox(x_6);
x_14 = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0(x_1, x_2, x_12, x_4, x_5, x_13, x_7, x_8, x_9, x_10);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
uint8_t x_11; uint8_t x_12; lean_object* x_13; 
x_11 = lean_unbox(x_2);
x_12 = lean_unbox(x_5);
x_13 = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___redArg(x_1, x_11, x_3, x_4, x_12, x_6, x_7, x_8, x_9);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious(x_1, x_2, x_3, x_4, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Lean_Meta_RefinedDiscrTree_evalLazyEntry_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; 
x_8 = l___private_Lean_Meta_Basic_0__Lean_Meta_withMCtxImp(lean_box(0), x_1, x_2, x_3, x_4, x_5, x_6);
if (lean_obj_tag(x_8) == 0)
{
uint8_t x_9; 
x_9 = !lean_is_exclusive(x_8);
if (x_9 == 0)
{
return x_8;
}
else
{
lean_object* x_10; lean_object* x_11; 
x_10 = lean_ctor_get(x_8, 0);
lean_inc(x_10);
lean_dec(x_8);
x_11 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_11, 0, x_10);
return x_11;
}
}
else
{
uint8_t x_12; 
x_12 = !lean_is_exclusive(x_8);
if (x_12 == 0)
{
return x_8;
}
else
{
lean_object* x_13; lean_object* x_14; 
x_13 = lean_ctor_get(x_8, 0);
lean_inc(x_13);
lean_dec(x_8);
x_14 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_14, 0, x_13);
return x_14;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Lean_Meta_RefinedDiscrTree_evalLazyEntry_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Lean_Meta_withMCtx___at___00Lean_Meta_RefinedDiscrTree_evalLazyEntry_spec__0___redArg(x_2, x_3, x_4, x_5, x_6, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_evalLazyEntry___lam__0(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; 
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
x_8 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious(x_1, x_3, x_4, x_5, x_6);
if (lean_obj_tag(x_8) == 0)
{
lean_object* x_9; lean_object* x_10; 
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux(x_9, x_2, x_3, x_4, x_5, x_6);
return x_10;
}
else
{
uint8_t x_11; 
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
x_11 = !lean_is_exclusive(x_8);
if (x_11 == 0)
{
return x_8;
}
else
{
lean_object* x_12; lean_object* x_13; 
x_12 = lean_ctor_get(x_8, 0);
lean_inc(x_12);
lean_dec(x_8);
x_13 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_13, 0, x_12);
return x_13;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_evalLazyEntry___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
uint8_t x_8; lean_object* x_9; 
x_8 = lean_unbox(x_2);
x_9 = lp_mathlib_Lean_Meta_RefinedDiscrTree_evalLazyEntry___lam__0(x_1, x_8, x_3, x_4, x_5, x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_evalLazyEntry(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; 
x_8 = lean_ctor_get(x_1, 4);
lean_inc(x_8);
if (lean_obj_tag(x_8) == 1)
{
uint8_t x_9; 
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
x_9 = !lean_is_exclusive(x_1);
if (x_9 == 0)
{
lean_object* x_10; uint8_t x_11; 
x_10 = lean_ctor_get(x_1, 4);
lean_dec(x_10);
x_11 = !lean_is_exclusive(x_8);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_12 = lean_ctor_get(x_8, 0);
x_13 = lean_ctor_get(x_8, 1);
lean_ctor_set(x_1, 4, x_13);
x_14 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_14, 0, x_12);
lean_ctor_set(x_14, 1, x_1);
x_15 = lean_box(0);
lean_ctor_set(x_8, 1, x_15);
lean_ctor_set(x_8, 0, x_14);
x_16 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_16, 0, x_8);
x_17 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_17, 0, x_16);
return x_17;
}
else
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_18 = lean_ctor_get(x_8, 0);
x_19 = lean_ctor_get(x_8, 1);
lean_inc(x_19);
lean_inc(x_18);
lean_dec(x_8);
lean_ctor_set(x_1, 4, x_19);
x_20 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_20, 0, x_18);
lean_ctor_set(x_20, 1, x_1);
x_21 = lean_box(0);
x_22 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_22, 0, x_20);
lean_ctor_set(x_22, 1, x_21);
x_23 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_23, 0, x_22);
x_24 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_24, 0, x_23);
return x_24;
}
}
else
{
lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; 
x_25 = lean_ctor_get(x_1, 0);
x_26 = lean_ctor_get(x_1, 1);
x_27 = lean_ctor_get(x_1, 2);
x_28 = lean_ctor_get(x_1, 3);
lean_inc(x_28);
lean_inc(x_27);
lean_inc(x_26);
lean_inc(x_25);
lean_dec(x_1);
x_29 = lean_ctor_get(x_8, 0);
lean_inc(x_29);
x_30 = lean_ctor_get(x_8, 1);
lean_inc(x_30);
if (lean_is_exclusive(x_8)) {
 lean_ctor_release(x_8, 0);
 lean_ctor_release(x_8, 1);
 x_31 = x_8;
} else {
 lean_dec_ref(x_8);
 x_31 = lean_box(0);
}
x_32 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_32, 0, x_25);
lean_ctor_set(x_32, 1, x_26);
lean_ctor_set(x_32, 2, x_27);
lean_ctor_set(x_32, 3, x_28);
lean_ctor_set(x_32, 4, x_30);
x_33 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_33, 0, x_29);
lean_ctor_set(x_33, 1, x_32);
x_34 = lean_box(0);
if (lean_is_scalar(x_31)) {
 x_35 = lean_alloc_ctor(1, 2, 0);
} else {
 x_35 = x_31;
}
lean_ctor_set(x_35, 0, x_33);
lean_ctor_set(x_35, 1, x_34);
x_36 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_36, 0, x_35);
x_37 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_37, 0, x_36);
return x_37;
}
}
else
{
lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; 
lean_dec(x_8);
x_38 = lean_ctor_get(x_1, 2);
lean_inc_ref(x_38);
x_39 = lean_box(x_2);
x_40 = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_evalLazyEntry___lam__0___boxed), 7, 2);
lean_closure_set(x_40, 0, x_1);
lean_closure_set(x_40, 1, x_39);
x_41 = lp_mathlib_Lean_Meta_withMCtx___at___00Lean_Meta_RefinedDiscrTree_evalLazyEntry_spec__0___redArg(x_38, x_40, x_3, x_4, x_5, x_6);
return x_41;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Lean_Meta_RefinedDiscrTree_evalLazyEntry_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Lean_Meta_withMCtx___at___00Lean_Meta_RefinedDiscrTree_evalLazyEntry_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Lean_Meta_RefinedDiscrTree_evalLazyEntry_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Lean_Meta_withMCtx___at___00Lean_Meta_RefinedDiscrTree_evalLazyEntry_spec__0___redArg(x_1, x_2, x_3, x_4, x_5, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_evalLazyEntry___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
uint8_t x_8; lean_object* x_9; 
x_8 = lean_unbox(x_2);
x_9 = lp_mathlib_Lean_Meta_RefinedDiscrTree_evalLazyEntry(x_1, x_8, x_3, x_4, x_5, x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodeExprWithEta_go_fold(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_dec_ref(x_1);
return x_3;
}
else
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_2, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_2, 1);
lean_inc(x_5);
lean_dec_ref(x_2);
if (lean_obj_tag(x_5) == 0)
{
uint8_t x_6; 
x_6 = !lean_is_exclusive(x_4);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lean_ctor_get(x_4, 0);
x_8 = lean_array_push(x_1, x_7);
lean_ctor_set(x_4, 0, x_8);
x_9 = lean_array_push(x_3, x_4);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_10 = lean_ctor_get(x_4, 0);
x_11 = lean_ctor_get(x_4, 1);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_4);
x_12 = lean_array_push(x_1, x_10);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_12);
lean_ctor_set(x_13, 1, x_11);
x_14 = lean_array_push(x_3, x_13);
return x_14;
}
}
else
{
uint8_t x_15; 
x_15 = !lean_is_exclusive(x_4);
if (x_15 == 0)
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_16 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_1);
x_17 = lean_array_push(x_1, x_16);
lean_ctor_set(x_4, 0, x_17);
x_18 = lean_array_push(x_3, x_4);
x_2 = x_5;
x_3 = x_18;
goto _start;
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_20 = lean_ctor_get(x_4, 0);
x_21 = lean_ctor_get(x_4, 1);
lean_inc(x_21);
lean_inc(x_20);
lean_dec(x_4);
lean_inc_ref(x_1);
x_22 = lean_array_push(x_1, x_20);
x_23 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_23, 0, x_22);
lean_ctor_set(x_23, 1, x_21);
x_24 = lean_array_push(x_3, x_23);
x_2 = x_5;
x_3 = x_24;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodeExprWithEta_go(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_8 = lean_array_get_size(x_1);
x_9 = lean_unsigned_to_nat(0u);
x_10 = lean_nat_dec_eq(x_8, x_9);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; lean_object* x_15; 
x_11 = l_Array_back___redArg(x_1);
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
x_13 = lean_ctor_get(x_11, 1);
lean_inc(x_13);
lean_dec(x_11);
x_14 = 1;
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
x_15 = lp_mathlib_Lean_Meta_RefinedDiscrTree_evalLazyEntry(x_13, x_14, x_3, x_4, x_5, x_6);
if (lean_obj_tag(x_15) == 0)
{
lean_object* x_16; lean_object* x_17; 
x_16 = lean_ctor_get(x_15, 0);
lean_inc(x_16);
lean_dec_ref(x_15);
x_17 = lean_array_pop(x_1);
if (lean_obj_tag(x_16) == 0)
{
lean_object* x_18; 
x_18 = lean_array_push(x_2, x_12);
x_1 = x_17;
x_2 = x_18;
goto _start;
}
else
{
lean_object* x_20; lean_object* x_21; 
x_20 = lean_ctor_get(x_16, 0);
lean_inc(x_20);
lean_dec_ref(x_16);
x_21 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodeExprWithEta_go_fold(x_12, x_20, x_17);
x_1 = x_21;
goto _start;
}
}
else
{
uint8_t x_23; 
lean_dec(x_12);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_23 = !lean_is_exclusive(x_15);
if (x_23 == 0)
{
return x_15;
}
else
{
lean_object* x_24; lean_object* x_25; 
x_24 = lean_ctor_get(x_15, 0);
lean_inc(x_24);
lean_dec(x_15);
x_25 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_25, 0, x_24);
return x_25;
}
}
}
else
{
lean_object* x_26; 
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
x_26 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_26, 0, x_2);
return x_26;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodeExprWithEta_go___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodeExprWithEta_go(x_1, x_2, x_3, x_4, x_5, x_6);
return x_8;
}
}
static lean_object* _init_lp_mathlib_List_mapTR_loop___at___00Lean_Meta_RefinedDiscrTree_encodeExprWithEta_spec__0___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_Meta_RefinedDiscrTree_encodeExprWithEta_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
x_3 = l_List_reverse___redArg(x_2);
return x_3;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; uint8_t x_6; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = !lean_is_exclusive(x_5);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_7 = lean_ctor_get(x_1, 1);
x_8 = lean_ctor_get(x_5, 0);
x_9 = lp_mathlib_List_mapTR_loop___at___00Lean_Meta_RefinedDiscrTree_encodeExprWithEta_spec__0___closed__0;
x_10 = lean_array_push(x_9, x_8);
lean_ctor_set(x_5, 0, x_10);
lean_ctor_set(x_1, 1, x_2);
{
lean_object* _tmp_0 = x_7;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_12 = lean_ctor_get(x_1, 1);
x_13 = lean_ctor_get(x_5, 0);
x_14 = lean_ctor_get(x_5, 1);
lean_inc(x_14);
lean_inc(x_13);
lean_dec(x_5);
x_15 = lp_mathlib_List_mapTR_loop___at___00Lean_Meta_RefinedDiscrTree_encodeExprWithEta_spec__0___closed__0;
x_16 = lean_array_push(x_15, x_13);
x_17 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_17, 0, x_16);
lean_ctor_set(x_17, 1, x_14);
lean_ctor_set(x_1, 1, x_2);
lean_ctor_set(x_1, 0, x_17);
{
lean_object* _tmp_0 = x_12;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
}
else
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; 
x_19 = lean_ctor_get(x_1, 0);
x_20 = lean_ctor_get(x_1, 1);
lean_inc(x_20);
lean_inc(x_19);
lean_dec(x_1);
x_21 = lean_ctor_get(x_19, 0);
lean_inc(x_21);
x_22 = lean_ctor_get(x_19, 1);
lean_inc(x_22);
if (lean_is_exclusive(x_19)) {
 lean_ctor_release(x_19, 0);
 lean_ctor_release(x_19, 1);
 x_23 = x_19;
} else {
 lean_dec_ref(x_19);
 x_23 = lean_box(0);
}
x_24 = lp_mathlib_List_mapTR_loop___at___00Lean_Meta_RefinedDiscrTree_encodeExprWithEta_spec__0___closed__0;
x_25 = lean_array_push(x_24, x_21);
if (lean_is_scalar(x_23)) {
 x_26 = lean_alloc_ctor(0, 2, 0);
} else {
 x_26 = x_23;
}
lean_ctor_set(x_26, 0, x_25);
lean_ctor_set(x_26, 1, x_22);
x_27 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_27, 0, x_26);
lean_ctor_set(x_27, 1, x_2);
x_1 = x_20;
x_2 = x_27;
goto _start;
}
}
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExprWithEta___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExprWithEta(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; uint8_t x_9; 
x_8 = l_Lean_Meta_Context_config(x_3);
x_9 = !lean_is_exclusive(x_8);
if (x_9 == 0)
{
uint8_t x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; uint8_t x_17; uint8_t x_18; lean_object* x_19; 
x_10 = lean_ctor_get_uint8(x_3, sizeof(void*)*7);
x_11 = lean_ctor_get(x_3, 1);
lean_inc(x_11);
x_12 = lean_ctor_get(x_3, 2);
lean_inc_ref(x_12);
x_13 = lean_ctor_get(x_3, 3);
lean_inc_ref(x_13);
x_14 = lean_ctor_get(x_3, 4);
lean_inc(x_14);
x_15 = lean_ctor_get(x_3, 5);
lean_inc(x_15);
x_16 = lean_ctor_get(x_3, 6);
lean_inc(x_16);
x_17 = lean_ctor_get_uint8(x_3, sizeof(void*)*7 + 1);
x_18 = lean_ctor_get_uint8(x_3, sizeof(void*)*7 + 2);
x_19 = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry___redArg(x_2, x_4);
if (lean_obj_tag(x_19) == 0)
{
lean_object* x_20; uint8_t x_21; uint64_t x_22; uint8_t x_23; 
x_20 = lean_ctor_get(x_19, 0);
lean_inc(x_20);
lean_dec_ref(x_19);
x_21 = 2;
lean_ctor_set_uint8(x_8, 9, x_21);
x_22 = l_Lean_Meta_Context_configKey(x_3);
x_23 = !lean_is_exclusive(x_3);
if (x_23 == 0)
{
lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; uint64_t x_31; uint64_t x_32; uint64_t x_33; uint64_t x_34; uint64_t x_35; lean_object* x_36; uint8_t x_37; lean_object* x_38; lean_object* x_39; 
x_24 = lean_ctor_get(x_3, 6);
lean_dec(x_24);
x_25 = lean_ctor_get(x_3, 5);
lean_dec(x_25);
x_26 = lean_ctor_get(x_3, 4);
lean_dec(x_26);
x_27 = lean_ctor_get(x_3, 3);
lean_dec(x_27);
x_28 = lean_ctor_get(x_3, 2);
lean_dec(x_28);
x_29 = lean_ctor_get(x_3, 1);
lean_dec(x_29);
x_30 = lean_ctor_get(x_3, 0);
lean_dec(x_30);
x_31 = 2;
x_32 = lean_uint64_shift_right(x_22, x_31);
x_33 = lean_uint64_shift_left(x_32, x_31);
x_34 = lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEta___closed__0;
x_35 = lean_uint64_lor(x_33, x_34);
x_36 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_36, 0, x_8);
lean_ctor_set_uint64(x_36, sizeof(void*)*1, x_35);
lean_ctor_set(x_3, 0, x_36);
x_37 = 1;
x_38 = lean_box(0);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
x_39 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta(x_1, x_37, x_20, x_38, x_3, x_4, x_5, x_6);
if (lean_obj_tag(x_39) == 0)
{
lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; 
x_40 = lean_ctor_get(x_39, 0);
lean_inc(x_40);
lean_dec_ref(x_39);
x_41 = lp_mathlib_List_mapTR_loop___at___00Lean_Meta_RefinedDiscrTree_encodeExprWithEta_spec__0(x_40, x_38);
x_42 = lean_array_mk(x_41);
x_43 = lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExprWithEta___closed__0;
x_44 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodeExprWithEta_go(x_42, x_43, x_3, x_4, x_5, x_6);
return x_44;
}
else
{
uint8_t x_45; 
lean_dec_ref(x_3);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
x_45 = !lean_is_exclusive(x_39);
if (x_45 == 0)
{
return x_39;
}
else
{
lean_object* x_46; lean_object* x_47; 
x_46 = lean_ctor_get(x_39, 0);
lean_inc(x_46);
lean_dec(x_39);
x_47 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_47, 0, x_46);
return x_47;
}
}
}
else
{
uint64_t x_48; uint64_t x_49; uint64_t x_50; uint64_t x_51; uint64_t x_52; lean_object* x_53; lean_object* x_54; uint8_t x_55; lean_object* x_56; lean_object* x_57; 
lean_dec(x_3);
x_48 = 2;
x_49 = lean_uint64_shift_right(x_22, x_48);
x_50 = lean_uint64_shift_left(x_49, x_48);
x_51 = lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEta___closed__0;
x_52 = lean_uint64_lor(x_50, x_51);
x_53 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_53, 0, x_8);
lean_ctor_set_uint64(x_53, sizeof(void*)*1, x_52);
x_54 = lean_alloc_ctor(0, 7, 3);
lean_ctor_set(x_54, 0, x_53);
lean_ctor_set(x_54, 1, x_11);
lean_ctor_set(x_54, 2, x_12);
lean_ctor_set(x_54, 3, x_13);
lean_ctor_set(x_54, 4, x_14);
lean_ctor_set(x_54, 5, x_15);
lean_ctor_set(x_54, 6, x_16);
lean_ctor_set_uint8(x_54, sizeof(void*)*7, x_10);
lean_ctor_set_uint8(x_54, sizeof(void*)*7 + 1, x_17);
lean_ctor_set_uint8(x_54, sizeof(void*)*7 + 2, x_18);
x_55 = 1;
x_56 = lean_box(0);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc_ref(x_54);
x_57 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta(x_1, x_55, x_20, x_56, x_54, x_4, x_5, x_6);
if (lean_obj_tag(x_57) == 0)
{
lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; 
x_58 = lean_ctor_get(x_57, 0);
lean_inc(x_58);
lean_dec_ref(x_57);
x_59 = lp_mathlib_List_mapTR_loop___at___00Lean_Meta_RefinedDiscrTree_encodeExprWithEta_spec__0(x_58, x_56);
x_60 = lean_array_mk(x_59);
x_61 = lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExprWithEta___closed__0;
x_62 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodeExprWithEta_go(x_60, x_61, x_54, x_4, x_5, x_6);
return x_62;
}
else
{
lean_object* x_63; lean_object* x_64; lean_object* x_65; 
lean_dec_ref(x_54);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
x_63 = lean_ctor_get(x_57, 0);
lean_inc(x_63);
if (lean_is_exclusive(x_57)) {
 lean_ctor_release(x_57, 0);
 x_64 = x_57;
} else {
 lean_dec_ref(x_57);
 x_64 = lean_box(0);
}
if (lean_is_scalar(x_64)) {
 x_65 = lean_alloc_ctor(1, 1, 0);
} else {
 x_65 = x_64;
}
lean_ctor_set(x_65, 0, x_63);
return x_65;
}
}
}
else
{
uint8_t x_66; 
lean_dec(x_16);
lean_dec(x_15);
lean_dec(x_14);
lean_dec_ref(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_free_object(x_8);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
x_66 = !lean_is_exclusive(x_19);
if (x_66 == 0)
{
return x_19;
}
else
{
lean_object* x_67; lean_object* x_68; 
x_67 = lean_ctor_get(x_19, 0);
lean_inc(x_67);
lean_dec(x_19);
x_68 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_68, 0, x_67);
return x_68;
}
}
}
else
{
uint8_t x_69; uint8_t x_70; uint8_t x_71; uint8_t x_72; uint8_t x_73; uint8_t x_74; uint8_t x_75; uint8_t x_76; uint8_t x_77; uint8_t x_78; uint8_t x_79; uint8_t x_80; uint8_t x_81; uint8_t x_82; uint8_t x_83; uint8_t x_84; uint8_t x_85; uint8_t x_86; uint8_t x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; uint8_t x_94; uint8_t x_95; lean_object* x_96; 
x_69 = lean_ctor_get_uint8(x_8, 0);
x_70 = lean_ctor_get_uint8(x_8, 1);
x_71 = lean_ctor_get_uint8(x_8, 2);
x_72 = lean_ctor_get_uint8(x_8, 3);
x_73 = lean_ctor_get_uint8(x_8, 4);
x_74 = lean_ctor_get_uint8(x_8, 5);
x_75 = lean_ctor_get_uint8(x_8, 6);
x_76 = lean_ctor_get_uint8(x_8, 7);
x_77 = lean_ctor_get_uint8(x_8, 8);
x_78 = lean_ctor_get_uint8(x_8, 10);
x_79 = lean_ctor_get_uint8(x_8, 11);
x_80 = lean_ctor_get_uint8(x_8, 12);
x_81 = lean_ctor_get_uint8(x_8, 13);
x_82 = lean_ctor_get_uint8(x_8, 14);
x_83 = lean_ctor_get_uint8(x_8, 15);
x_84 = lean_ctor_get_uint8(x_8, 16);
x_85 = lean_ctor_get_uint8(x_8, 17);
x_86 = lean_ctor_get_uint8(x_8, 18);
lean_dec(x_8);
x_87 = lean_ctor_get_uint8(x_3, sizeof(void*)*7);
x_88 = lean_ctor_get(x_3, 1);
lean_inc(x_88);
x_89 = lean_ctor_get(x_3, 2);
lean_inc_ref(x_89);
x_90 = lean_ctor_get(x_3, 3);
lean_inc_ref(x_90);
x_91 = lean_ctor_get(x_3, 4);
lean_inc(x_91);
x_92 = lean_ctor_get(x_3, 5);
lean_inc(x_92);
x_93 = lean_ctor_get(x_3, 6);
lean_inc(x_93);
x_94 = lean_ctor_get_uint8(x_3, sizeof(void*)*7 + 1);
x_95 = lean_ctor_get_uint8(x_3, sizeof(void*)*7 + 2);
x_96 = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry___redArg(x_2, x_4);
if (lean_obj_tag(x_96) == 0)
{
lean_object* x_97; uint8_t x_98; lean_object* x_99; uint64_t x_100; lean_object* x_101; uint64_t x_102; uint64_t x_103; uint64_t x_104; uint64_t x_105; uint64_t x_106; lean_object* x_107; lean_object* x_108; uint8_t x_109; lean_object* x_110; lean_object* x_111; 
x_97 = lean_ctor_get(x_96, 0);
lean_inc(x_97);
lean_dec_ref(x_96);
x_98 = 2;
x_99 = lean_alloc_ctor(0, 0, 19);
lean_ctor_set_uint8(x_99, 0, x_69);
lean_ctor_set_uint8(x_99, 1, x_70);
lean_ctor_set_uint8(x_99, 2, x_71);
lean_ctor_set_uint8(x_99, 3, x_72);
lean_ctor_set_uint8(x_99, 4, x_73);
lean_ctor_set_uint8(x_99, 5, x_74);
lean_ctor_set_uint8(x_99, 6, x_75);
lean_ctor_set_uint8(x_99, 7, x_76);
lean_ctor_set_uint8(x_99, 8, x_77);
lean_ctor_set_uint8(x_99, 9, x_98);
lean_ctor_set_uint8(x_99, 10, x_78);
lean_ctor_set_uint8(x_99, 11, x_79);
lean_ctor_set_uint8(x_99, 12, x_80);
lean_ctor_set_uint8(x_99, 13, x_81);
lean_ctor_set_uint8(x_99, 14, x_82);
lean_ctor_set_uint8(x_99, 15, x_83);
lean_ctor_set_uint8(x_99, 16, x_84);
lean_ctor_set_uint8(x_99, 17, x_85);
lean_ctor_set_uint8(x_99, 18, x_86);
x_100 = l_Lean_Meta_Context_configKey(x_3);
if (lean_is_exclusive(x_3)) {
 lean_ctor_release(x_3, 0);
 lean_ctor_release(x_3, 1);
 lean_ctor_release(x_3, 2);
 lean_ctor_release(x_3, 3);
 lean_ctor_release(x_3, 4);
 lean_ctor_release(x_3, 5);
 lean_ctor_release(x_3, 6);
 x_101 = x_3;
} else {
 lean_dec_ref(x_3);
 x_101 = lean_box(0);
}
x_102 = 2;
x_103 = lean_uint64_shift_right(x_100, x_102);
x_104 = lean_uint64_shift_left(x_103, x_102);
x_105 = lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEta___closed__0;
x_106 = lean_uint64_lor(x_104, x_105);
x_107 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_107, 0, x_99);
lean_ctor_set_uint64(x_107, sizeof(void*)*1, x_106);
if (lean_is_scalar(x_101)) {
 x_108 = lean_alloc_ctor(0, 7, 3);
} else {
 x_108 = x_101;
}
lean_ctor_set(x_108, 0, x_107);
lean_ctor_set(x_108, 1, x_88);
lean_ctor_set(x_108, 2, x_89);
lean_ctor_set(x_108, 3, x_90);
lean_ctor_set(x_108, 4, x_91);
lean_ctor_set(x_108, 5, x_92);
lean_ctor_set(x_108, 6, x_93);
lean_ctor_set_uint8(x_108, sizeof(void*)*7, x_87);
lean_ctor_set_uint8(x_108, sizeof(void*)*7 + 1, x_94);
lean_ctor_set_uint8(x_108, sizeof(void*)*7 + 2, x_95);
x_109 = 1;
x_110 = lean_box(0);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc_ref(x_108);
x_111 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta(x_1, x_109, x_97, x_110, x_108, x_4, x_5, x_6);
if (lean_obj_tag(x_111) == 0)
{
lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; lean_object* x_116; 
x_112 = lean_ctor_get(x_111, 0);
lean_inc(x_112);
lean_dec_ref(x_111);
x_113 = lp_mathlib_List_mapTR_loop___at___00Lean_Meta_RefinedDiscrTree_encodeExprWithEta_spec__0(x_112, x_110);
x_114 = lean_array_mk(x_113);
x_115 = lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExprWithEta___closed__0;
x_116 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodeExprWithEta_go(x_114, x_115, x_108, x_4, x_5, x_6);
return x_116;
}
else
{
lean_object* x_117; lean_object* x_118; lean_object* x_119; 
lean_dec_ref(x_108);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
x_117 = lean_ctor_get(x_111, 0);
lean_inc(x_117);
if (lean_is_exclusive(x_111)) {
 lean_ctor_release(x_111, 0);
 x_118 = x_111;
} else {
 lean_dec_ref(x_111);
 x_118 = lean_box(0);
}
if (lean_is_scalar(x_118)) {
 x_119 = lean_alloc_ctor(1, 1, 0);
} else {
 x_119 = x_118;
}
lean_ctor_set(x_119, 0, x_117);
return x_119;
}
}
else
{
lean_object* x_120; lean_object* x_121; lean_object* x_122; 
lean_dec(x_93);
lean_dec(x_92);
lean_dec(x_91);
lean_dec_ref(x_90);
lean_dec_ref(x_89);
lean_dec(x_88);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
x_120 = lean_ctor_get(x_96, 0);
lean_inc(x_120);
if (lean_is_exclusive(x_96)) {
 lean_ctor_release(x_96, 0);
 x_121 = x_96;
} else {
 lean_dec_ref(x_96);
 x_121 = lean_box(0);
}
if (lean_is_scalar(x_121)) {
 x_122 = lean_alloc_ctor(1, 1, 0);
} else {
 x_122 = x_121;
}
lean_ctor_set(x_122, 0, x_120);
return x_122;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExprWithEta___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
uint8_t x_8; lean_object* x_9; 
x_8 = lean_unbox(x_2);
x_9 = lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExprWithEta(x_1, x_8, x_3, x_4, x_5, x_6);
return x_9;
}
}
static lean_object* _init_lp_mathlib_panic___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_toList_spec__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_Meta_instInhabitedMetaM___lam__0___boxed), 5, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_toList_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lp_mathlib_panic___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_toList_spec__0___closed__0;
x_8 = lean_panic_fn(x_7, x_1);
x_9 = lean_apply_5(x_8, x_2, x_3, x_4, x_5, lean_box(0));
return x_9;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Lean.Meta.RefinedDiscrTree.LazyEntry.toList", 43, 43);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("`evalLazyEntry` with `eta := false` can only give a singleton list", 66, 66);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_1 = lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__1;
x_2 = lean_unsigned_to_nat(14u);
x_3 = lean_unsigned_to_nat(313u);
x_4 = lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__0;
x_5 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__0;
x_6 = l_mkPanicMessageWithDecl(x_5, x_4, x_3, x_2, x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_16; lean_object* x_17; 
x_16 = 0;
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
x_17 = lp_mathlib_Lean_Meta_RefinedDiscrTree_evalLazyEntry(x_1, x_16, x_3, x_4, x_5, x_6);
if (lean_obj_tag(x_17) == 0)
{
uint8_t x_18; 
x_18 = !lean_is_exclusive(x_17);
if (x_18 == 0)
{
lean_object* x_19; 
x_19 = lean_ctor_get(x_17, 0);
if (lean_obj_tag(x_19) == 0)
{
lean_object* x_20; 
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
x_20 = l_List_reverse___redArg(x_2);
lean_ctor_set(x_17, 0, x_20);
return x_17;
}
else
{
lean_object* x_21; 
lean_free_object(x_17);
x_21 = lean_ctor_get(x_19, 0);
lean_inc(x_21);
lean_dec_ref(x_19);
if (lean_obj_tag(x_21) == 1)
{
uint8_t x_22; 
x_22 = !lean_is_exclusive(x_21);
if (x_22 == 0)
{
lean_object* x_23; 
x_23 = lean_ctor_get(x_21, 1);
if (lean_obj_tag(x_23) == 0)
{
lean_object* x_24; lean_object* x_25; lean_object* x_26; 
x_24 = lean_ctor_get(x_21, 0);
x_25 = lean_ctor_get(x_24, 0);
lean_inc(x_25);
x_26 = lean_ctor_get(x_24, 1);
lean_inc(x_26);
lean_dec(x_24);
lean_ctor_set(x_21, 1, x_2);
lean_ctor_set(x_21, 0, x_25);
x_1 = x_26;
x_2 = x_21;
goto _start;
}
else
{
lean_object* x_28; 
x_28 = lean_ctor_get(x_21, 0);
lean_free_object(x_21);
lean_dec(x_23);
lean_dec(x_28);
lean_dec(x_2);
x_8 = x_3;
x_9 = x_4;
x_10 = x_5;
x_11 = x_6;
x_12 = lean_box(0);
goto block_15;
}
}
else
{
lean_object* x_29; lean_object* x_30; 
x_29 = lean_ctor_get(x_21, 0);
x_30 = lean_ctor_get(x_21, 1);
lean_inc(x_30);
lean_inc(x_29);
lean_dec(x_21);
if (lean_obj_tag(x_30) == 0)
{
lean_object* x_31; lean_object* x_32; lean_object* x_33; 
x_31 = lean_ctor_get(x_29, 0);
lean_inc(x_31);
x_32 = lean_ctor_get(x_29, 1);
lean_inc(x_32);
lean_dec(x_29);
x_33 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_33, 0, x_31);
lean_ctor_set(x_33, 1, x_2);
x_1 = x_32;
x_2 = x_33;
goto _start;
}
else
{
lean_dec(x_30);
lean_dec(x_29);
lean_dec(x_2);
x_8 = x_3;
x_9 = x_4;
x_10 = x_5;
x_11 = x_6;
x_12 = lean_box(0);
goto block_15;
}
}
}
else
{
lean_dec(x_21);
lean_dec(x_2);
x_8 = x_3;
x_9 = x_4;
x_10 = x_5;
x_11 = x_6;
x_12 = lean_box(0);
goto block_15;
}
}
}
else
{
lean_object* x_35; 
x_35 = lean_ctor_get(x_17, 0);
lean_inc(x_35);
lean_dec(x_17);
if (lean_obj_tag(x_35) == 0)
{
lean_object* x_36; lean_object* x_37; 
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
x_36 = l_List_reverse___redArg(x_2);
x_37 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_37, 0, x_36);
return x_37;
}
else
{
lean_object* x_38; 
x_38 = lean_ctor_get(x_35, 0);
lean_inc(x_38);
lean_dec_ref(x_35);
if (lean_obj_tag(x_38) == 1)
{
lean_object* x_39; lean_object* x_40; lean_object* x_41; 
x_39 = lean_ctor_get(x_38, 0);
lean_inc(x_39);
x_40 = lean_ctor_get(x_38, 1);
lean_inc(x_40);
if (lean_is_exclusive(x_38)) {
 lean_ctor_release(x_38, 0);
 lean_ctor_release(x_38, 1);
 x_41 = x_38;
} else {
 lean_dec_ref(x_38);
 x_41 = lean_box(0);
}
if (lean_obj_tag(x_40) == 0)
{
lean_object* x_42; lean_object* x_43; lean_object* x_44; 
x_42 = lean_ctor_get(x_39, 0);
lean_inc(x_42);
x_43 = lean_ctor_get(x_39, 1);
lean_inc(x_43);
lean_dec(x_39);
if (lean_is_scalar(x_41)) {
 x_44 = lean_alloc_ctor(1, 2, 0);
} else {
 x_44 = x_41;
}
lean_ctor_set(x_44, 0, x_42);
lean_ctor_set(x_44, 1, x_2);
x_1 = x_43;
x_2 = x_44;
goto _start;
}
else
{
lean_dec(x_41);
lean_dec(x_40);
lean_dec(x_39);
lean_dec(x_2);
x_8 = x_3;
x_9 = x_4;
x_10 = x_5;
x_11 = x_6;
x_12 = lean_box(0);
goto block_15;
}
}
else
{
lean_dec(x_38);
lean_dec(x_2);
x_8 = x_3;
x_9 = x_4;
x_10 = x_5;
x_11 = x_6;
x_12 = lean_box(0);
goto block_15;
}
}
}
}
else
{
uint8_t x_46; 
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
x_46 = !lean_is_exclusive(x_17);
if (x_46 == 0)
{
return x_17;
}
else
{
lean_object* x_47; lean_object* x_48; 
x_47 = lean_ctor_get(x_17, 0);
lean_inc(x_47);
lean_dec(x_17);
x_48 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_48, 0, x_47);
return x_48;
}
}
block_15:
{
lean_object* x_13; lean_object* x_14; 
x_13 = lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__2;
x_14 = lp_mathlib_panic___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_toList_spec__0(x_13, x_8, x_9, x_10, x_11);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_toList_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_panic___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_toList_spec__0(x_1, x_2, x_3, x_4, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList(x_1, x_2, x_3, x_4, x_5, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExpr(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; uint8_t x_9; 
x_8 = l_Lean_Meta_Context_config(x_3);
x_9 = !lean_is_exclusive(x_8);
if (x_9 == 0)
{
uint8_t x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; uint8_t x_17; uint8_t x_18; uint8_t x_19; uint64_t x_20; uint8_t x_21; 
x_10 = lean_ctor_get_uint8(x_3, sizeof(void*)*7);
x_11 = lean_ctor_get(x_3, 1);
lean_inc(x_11);
x_12 = lean_ctor_get(x_3, 2);
lean_inc_ref(x_12);
x_13 = lean_ctor_get(x_3, 3);
lean_inc_ref(x_13);
x_14 = lean_ctor_get(x_3, 4);
lean_inc(x_14);
x_15 = lean_ctor_get(x_3, 5);
lean_inc(x_15);
x_16 = lean_ctor_get(x_3, 6);
lean_inc(x_16);
x_17 = lean_ctor_get_uint8(x_3, sizeof(void*)*7 + 1);
x_18 = lean_ctor_get_uint8(x_3, sizeof(void*)*7 + 2);
x_19 = 2;
lean_ctor_set_uint8(x_8, 9, x_19);
x_20 = l_Lean_Meta_Context_configKey(x_3);
x_21 = !lean_is_exclusive(x_3);
if (x_21 == 0)
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; uint64_t x_29; uint64_t x_30; uint64_t x_31; uint64_t x_32; uint64_t x_33; lean_object* x_34; lean_object* x_35; 
x_22 = lean_ctor_get(x_3, 6);
lean_dec(x_22);
x_23 = lean_ctor_get(x_3, 5);
lean_dec(x_23);
x_24 = lean_ctor_get(x_3, 4);
lean_dec(x_24);
x_25 = lean_ctor_get(x_3, 3);
lean_dec(x_25);
x_26 = lean_ctor_get(x_3, 2);
lean_dec(x_26);
x_27 = lean_ctor_get(x_3, 1);
lean_dec(x_27);
x_28 = lean_ctor_get(x_3, 0);
lean_dec(x_28);
x_29 = 2;
x_30 = lean_uint64_shift_right(x_20, x_29);
x_31 = lean_uint64_shift_left(x_30, x_29);
x_32 = lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEta___closed__0;
x_33 = lean_uint64_lor(x_31, x_32);
x_34 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_34, 0, x_8);
lean_ctor_set_uint64(x_34, sizeof(void*)*1, x_33);
lean_ctor_set(x_3, 0, x_34);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
x_35 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_initializeLazyEntry(x_1, x_2, x_3, x_4, x_5, x_6);
if (lean_obj_tag(x_35) == 0)
{
lean_object* x_36; uint8_t x_37; 
x_36 = lean_ctor_get(x_35, 0);
lean_inc(x_36);
lean_dec_ref(x_35);
x_37 = !lean_is_exclusive(x_36);
if (x_37 == 0)
{
lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; 
x_38 = lean_ctor_get(x_36, 0);
x_39 = lean_ctor_get(x_36, 1);
x_40 = lean_box(0);
x_41 = lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList(x_39, x_40, x_3, x_4, x_5, x_6);
if (lean_obj_tag(x_41) == 0)
{
uint8_t x_42; 
x_42 = !lean_is_exclusive(x_41);
if (x_42 == 0)
{
lean_object* x_43; 
x_43 = lean_ctor_get(x_41, 0);
lean_ctor_set(x_36, 1, x_43);
lean_ctor_set(x_41, 0, x_36);
return x_41;
}
else
{
lean_object* x_44; lean_object* x_45; 
x_44 = lean_ctor_get(x_41, 0);
lean_inc(x_44);
lean_dec(x_41);
lean_ctor_set(x_36, 1, x_44);
x_45 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_45, 0, x_36);
return x_45;
}
}
else
{
uint8_t x_46; 
lean_free_object(x_36);
lean_dec(x_38);
x_46 = !lean_is_exclusive(x_41);
if (x_46 == 0)
{
return x_41;
}
else
{
lean_object* x_47; lean_object* x_48; 
x_47 = lean_ctor_get(x_41, 0);
lean_inc(x_47);
lean_dec(x_41);
x_48 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_48, 0, x_47);
return x_48;
}
}
}
else
{
lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; 
x_49 = lean_ctor_get(x_36, 0);
x_50 = lean_ctor_get(x_36, 1);
lean_inc(x_50);
lean_inc(x_49);
lean_dec(x_36);
x_51 = lean_box(0);
x_52 = lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList(x_50, x_51, x_3, x_4, x_5, x_6);
if (lean_obj_tag(x_52) == 0)
{
lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; 
x_53 = lean_ctor_get(x_52, 0);
lean_inc(x_53);
if (lean_is_exclusive(x_52)) {
 lean_ctor_release(x_52, 0);
 x_54 = x_52;
} else {
 lean_dec_ref(x_52);
 x_54 = lean_box(0);
}
x_55 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_55, 0, x_49);
lean_ctor_set(x_55, 1, x_53);
if (lean_is_scalar(x_54)) {
 x_56 = lean_alloc_ctor(0, 1, 0);
} else {
 x_56 = x_54;
}
lean_ctor_set(x_56, 0, x_55);
return x_56;
}
else
{
lean_object* x_57; lean_object* x_58; lean_object* x_59; 
lean_dec(x_49);
x_57 = lean_ctor_get(x_52, 0);
lean_inc(x_57);
if (lean_is_exclusive(x_52)) {
 lean_ctor_release(x_52, 0);
 x_58 = x_52;
} else {
 lean_dec_ref(x_52);
 x_58 = lean_box(0);
}
if (lean_is_scalar(x_58)) {
 x_59 = lean_alloc_ctor(1, 1, 0);
} else {
 x_59 = x_58;
}
lean_ctor_set(x_59, 0, x_57);
return x_59;
}
}
}
else
{
uint8_t x_60; 
lean_dec_ref(x_3);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
x_60 = !lean_is_exclusive(x_35);
if (x_60 == 0)
{
return x_35;
}
else
{
lean_object* x_61; lean_object* x_62; 
x_61 = lean_ctor_get(x_35, 0);
lean_inc(x_61);
lean_dec(x_35);
x_62 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_62, 0, x_61);
return x_62;
}
}
}
else
{
uint64_t x_63; uint64_t x_64; uint64_t x_65; uint64_t x_66; uint64_t x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; 
lean_dec(x_3);
x_63 = 2;
x_64 = lean_uint64_shift_right(x_20, x_63);
x_65 = lean_uint64_shift_left(x_64, x_63);
x_66 = lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEta___closed__0;
x_67 = lean_uint64_lor(x_65, x_66);
x_68 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_68, 0, x_8);
lean_ctor_set_uint64(x_68, sizeof(void*)*1, x_67);
x_69 = lean_alloc_ctor(0, 7, 3);
lean_ctor_set(x_69, 0, x_68);
lean_ctor_set(x_69, 1, x_11);
lean_ctor_set(x_69, 2, x_12);
lean_ctor_set(x_69, 3, x_13);
lean_ctor_set(x_69, 4, x_14);
lean_ctor_set(x_69, 5, x_15);
lean_ctor_set(x_69, 6, x_16);
lean_ctor_set_uint8(x_69, sizeof(void*)*7, x_10);
lean_ctor_set_uint8(x_69, sizeof(void*)*7 + 1, x_17);
lean_ctor_set_uint8(x_69, sizeof(void*)*7 + 2, x_18);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc_ref(x_69);
x_70 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_initializeLazyEntry(x_1, x_2, x_69, x_4, x_5, x_6);
if (lean_obj_tag(x_70) == 0)
{
lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; 
x_71 = lean_ctor_get(x_70, 0);
lean_inc(x_71);
lean_dec_ref(x_70);
x_72 = lean_ctor_get(x_71, 0);
lean_inc(x_72);
x_73 = lean_ctor_get(x_71, 1);
lean_inc(x_73);
if (lean_is_exclusive(x_71)) {
 lean_ctor_release(x_71, 0);
 lean_ctor_release(x_71, 1);
 x_74 = x_71;
} else {
 lean_dec_ref(x_71);
 x_74 = lean_box(0);
}
x_75 = lean_box(0);
x_76 = lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList(x_73, x_75, x_69, x_4, x_5, x_6);
if (lean_obj_tag(x_76) == 0)
{
lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; 
x_77 = lean_ctor_get(x_76, 0);
lean_inc(x_77);
if (lean_is_exclusive(x_76)) {
 lean_ctor_release(x_76, 0);
 x_78 = x_76;
} else {
 lean_dec_ref(x_76);
 x_78 = lean_box(0);
}
if (lean_is_scalar(x_74)) {
 x_79 = lean_alloc_ctor(0, 2, 0);
} else {
 x_79 = x_74;
}
lean_ctor_set(x_79, 0, x_72);
lean_ctor_set(x_79, 1, x_77);
if (lean_is_scalar(x_78)) {
 x_80 = lean_alloc_ctor(0, 1, 0);
} else {
 x_80 = x_78;
}
lean_ctor_set(x_80, 0, x_79);
return x_80;
}
else
{
lean_object* x_81; lean_object* x_82; lean_object* x_83; 
lean_dec(x_74);
lean_dec(x_72);
x_81 = lean_ctor_get(x_76, 0);
lean_inc(x_81);
if (lean_is_exclusive(x_76)) {
 lean_ctor_release(x_76, 0);
 x_82 = x_76;
} else {
 lean_dec_ref(x_76);
 x_82 = lean_box(0);
}
if (lean_is_scalar(x_82)) {
 x_83 = lean_alloc_ctor(1, 1, 0);
} else {
 x_83 = x_82;
}
lean_ctor_set(x_83, 0, x_81);
return x_83;
}
}
else
{
lean_object* x_84; lean_object* x_85; lean_object* x_86; 
lean_dec_ref(x_69);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
x_84 = lean_ctor_get(x_70, 0);
lean_inc(x_84);
if (lean_is_exclusive(x_70)) {
 lean_ctor_release(x_70, 0);
 x_85 = x_70;
} else {
 lean_dec_ref(x_70);
 x_85 = lean_box(0);
}
if (lean_is_scalar(x_85)) {
 x_86 = lean_alloc_ctor(1, 1, 0);
} else {
 x_86 = x_85;
}
lean_ctor_set(x_86, 0, x_84);
return x_86;
}
}
}
else
{
uint8_t x_87; uint8_t x_88; uint8_t x_89; uint8_t x_90; uint8_t x_91; uint8_t x_92; uint8_t x_93; uint8_t x_94; uint8_t x_95; uint8_t x_96; uint8_t x_97; uint8_t x_98; uint8_t x_99; uint8_t x_100; uint8_t x_101; uint8_t x_102; uint8_t x_103; uint8_t x_104; uint8_t x_105; lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; uint8_t x_112; uint8_t x_113; uint8_t x_114; lean_object* x_115; uint64_t x_116; lean_object* x_117; uint64_t x_118; uint64_t x_119; uint64_t x_120; uint64_t x_121; uint64_t x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; 
x_87 = lean_ctor_get_uint8(x_8, 0);
x_88 = lean_ctor_get_uint8(x_8, 1);
x_89 = lean_ctor_get_uint8(x_8, 2);
x_90 = lean_ctor_get_uint8(x_8, 3);
x_91 = lean_ctor_get_uint8(x_8, 4);
x_92 = lean_ctor_get_uint8(x_8, 5);
x_93 = lean_ctor_get_uint8(x_8, 6);
x_94 = lean_ctor_get_uint8(x_8, 7);
x_95 = lean_ctor_get_uint8(x_8, 8);
x_96 = lean_ctor_get_uint8(x_8, 10);
x_97 = lean_ctor_get_uint8(x_8, 11);
x_98 = lean_ctor_get_uint8(x_8, 12);
x_99 = lean_ctor_get_uint8(x_8, 13);
x_100 = lean_ctor_get_uint8(x_8, 14);
x_101 = lean_ctor_get_uint8(x_8, 15);
x_102 = lean_ctor_get_uint8(x_8, 16);
x_103 = lean_ctor_get_uint8(x_8, 17);
x_104 = lean_ctor_get_uint8(x_8, 18);
lean_dec(x_8);
x_105 = lean_ctor_get_uint8(x_3, sizeof(void*)*7);
x_106 = lean_ctor_get(x_3, 1);
lean_inc(x_106);
x_107 = lean_ctor_get(x_3, 2);
lean_inc_ref(x_107);
x_108 = lean_ctor_get(x_3, 3);
lean_inc_ref(x_108);
x_109 = lean_ctor_get(x_3, 4);
lean_inc(x_109);
x_110 = lean_ctor_get(x_3, 5);
lean_inc(x_110);
x_111 = lean_ctor_get(x_3, 6);
lean_inc(x_111);
x_112 = lean_ctor_get_uint8(x_3, sizeof(void*)*7 + 1);
x_113 = lean_ctor_get_uint8(x_3, sizeof(void*)*7 + 2);
x_114 = 2;
x_115 = lean_alloc_ctor(0, 0, 19);
lean_ctor_set_uint8(x_115, 0, x_87);
lean_ctor_set_uint8(x_115, 1, x_88);
lean_ctor_set_uint8(x_115, 2, x_89);
lean_ctor_set_uint8(x_115, 3, x_90);
lean_ctor_set_uint8(x_115, 4, x_91);
lean_ctor_set_uint8(x_115, 5, x_92);
lean_ctor_set_uint8(x_115, 6, x_93);
lean_ctor_set_uint8(x_115, 7, x_94);
lean_ctor_set_uint8(x_115, 8, x_95);
lean_ctor_set_uint8(x_115, 9, x_114);
lean_ctor_set_uint8(x_115, 10, x_96);
lean_ctor_set_uint8(x_115, 11, x_97);
lean_ctor_set_uint8(x_115, 12, x_98);
lean_ctor_set_uint8(x_115, 13, x_99);
lean_ctor_set_uint8(x_115, 14, x_100);
lean_ctor_set_uint8(x_115, 15, x_101);
lean_ctor_set_uint8(x_115, 16, x_102);
lean_ctor_set_uint8(x_115, 17, x_103);
lean_ctor_set_uint8(x_115, 18, x_104);
x_116 = l_Lean_Meta_Context_configKey(x_3);
if (lean_is_exclusive(x_3)) {
 lean_ctor_release(x_3, 0);
 lean_ctor_release(x_3, 1);
 lean_ctor_release(x_3, 2);
 lean_ctor_release(x_3, 3);
 lean_ctor_release(x_3, 4);
 lean_ctor_release(x_3, 5);
 lean_ctor_release(x_3, 6);
 x_117 = x_3;
} else {
 lean_dec_ref(x_3);
 x_117 = lean_box(0);
}
x_118 = 2;
x_119 = lean_uint64_shift_right(x_116, x_118);
x_120 = lean_uint64_shift_left(x_119, x_118);
x_121 = lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEta___closed__0;
x_122 = lean_uint64_lor(x_120, x_121);
x_123 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_123, 0, x_115);
lean_ctor_set_uint64(x_123, sizeof(void*)*1, x_122);
if (lean_is_scalar(x_117)) {
 x_124 = lean_alloc_ctor(0, 7, 3);
} else {
 x_124 = x_117;
}
lean_ctor_set(x_124, 0, x_123);
lean_ctor_set(x_124, 1, x_106);
lean_ctor_set(x_124, 2, x_107);
lean_ctor_set(x_124, 3, x_108);
lean_ctor_set(x_124, 4, x_109);
lean_ctor_set(x_124, 5, x_110);
lean_ctor_set(x_124, 6, x_111);
lean_ctor_set_uint8(x_124, sizeof(void*)*7, x_105);
lean_ctor_set_uint8(x_124, sizeof(void*)*7 + 1, x_112);
lean_ctor_set_uint8(x_124, sizeof(void*)*7 + 2, x_113);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc_ref(x_124);
x_125 = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_initializeLazyEntry(x_1, x_2, x_124, x_4, x_5, x_6);
if (lean_obj_tag(x_125) == 0)
{
lean_object* x_126; lean_object* x_127; lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; 
x_126 = lean_ctor_get(x_125, 0);
lean_inc(x_126);
lean_dec_ref(x_125);
x_127 = lean_ctor_get(x_126, 0);
lean_inc(x_127);
x_128 = lean_ctor_get(x_126, 1);
lean_inc(x_128);
if (lean_is_exclusive(x_126)) {
 lean_ctor_release(x_126, 0);
 lean_ctor_release(x_126, 1);
 x_129 = x_126;
} else {
 lean_dec_ref(x_126);
 x_129 = lean_box(0);
}
x_130 = lean_box(0);
x_131 = lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList(x_128, x_130, x_124, x_4, x_5, x_6);
if (lean_obj_tag(x_131) == 0)
{
lean_object* x_132; lean_object* x_133; lean_object* x_134; lean_object* x_135; 
x_132 = lean_ctor_get(x_131, 0);
lean_inc(x_132);
if (lean_is_exclusive(x_131)) {
 lean_ctor_release(x_131, 0);
 x_133 = x_131;
} else {
 lean_dec_ref(x_131);
 x_133 = lean_box(0);
}
if (lean_is_scalar(x_129)) {
 x_134 = lean_alloc_ctor(0, 2, 0);
} else {
 x_134 = x_129;
}
lean_ctor_set(x_134, 0, x_127);
lean_ctor_set(x_134, 1, x_132);
if (lean_is_scalar(x_133)) {
 x_135 = lean_alloc_ctor(0, 1, 0);
} else {
 x_135 = x_133;
}
lean_ctor_set(x_135, 0, x_134);
return x_135;
}
else
{
lean_object* x_136; lean_object* x_137; lean_object* x_138; 
lean_dec(x_129);
lean_dec(x_127);
x_136 = lean_ctor_get(x_131, 0);
lean_inc(x_136);
if (lean_is_exclusive(x_131)) {
 lean_ctor_release(x_131, 0);
 x_137 = x_131;
} else {
 lean_dec_ref(x_131);
 x_137 = lean_box(0);
}
if (lean_is_scalar(x_137)) {
 x_138 = lean_alloc_ctor(1, 1, 0);
} else {
 x_138 = x_137;
}
lean_ctor_set(x_138, 0, x_136);
return x_138;
}
}
else
{
lean_object* x_139; lean_object* x_140; lean_object* x_141; 
lean_dec_ref(x_124);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
x_139 = lean_ctor_get(x_125, 0);
lean_inc(x_139);
if (lean_is_exclusive(x_125)) {
 lean_ctor_release(x_125, 0);
 x_140 = x_125;
} else {
 lean_dec_ref(x_125);
 x_140 = lean_box(0);
}
if (lean_is_scalar(x_140)) {
 x_141 = lean_alloc_ctor(1, 1, 0);
} else {
 x_141 = x_140;
}
lean_ctor_set(x_141, 0, x_139);
return x_141;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExpr___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
uint8_t x_8; lean_object* x_9; 
x_8 = lean_unbox(x_2);
x_9 = lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExpr(x_1, x_8, x_3, x_4, x_5, x_6);
return x_9;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Basic(uint8_t builtin);
lean_object* initialize_Lean_Meta_DiscrTree(uint8_t builtin);
lean_object* initialize_Lean_Meta_LazyDiscrTree(uint8_t builtin);
lean_object* initialize_Lean_Meta_DiscrTree(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Encode(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_DiscrTree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_LazyDiscrTree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_DiscrTree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__0 = _init_lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__0();
lean_mark_persistent(lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__0);
lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__1 = _init_lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__1();
lean_mark_persistent(lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__1);
lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__2 = _init_lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__2();
lean_mark_persistent(lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__2);
lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__3 = _init_lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__3();
lean_mark_persistent(lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__3);
lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__4 = _init_lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__4();
lean_mark_persistent(lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___closed__4);
lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__0 = _init_lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__0();
lean_mark_persistent(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__0);
lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__1 = _init_lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__1();
lean_mark_persistent(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__1);
lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__2 = _init_lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__2();
lean_mark_persistent(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__2);
lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__3 = _init_lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__3();
lean_mark_persistent(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__3);
lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux___closed__0 = _init_lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux___closed__0();
lean_mark_persistent(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux___closed__0);
lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___closed__0 = _init_lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___closed__0();
lean_mark_persistent(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___closed__0);
lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEta___closed__0 = _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEta___closed__0();
lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___closed__0 = _init_lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___closed__0();
lean_mark_persistent(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___closed__0);
lp_mathlib_List_mapTR_loop___at___00Lean_Meta_RefinedDiscrTree_encodeExprWithEta_spec__0___closed__0 = _init_lp_mathlib_List_mapTR_loop___at___00Lean_Meta_RefinedDiscrTree_encodeExprWithEta_spec__0___closed__0();
lean_mark_persistent(lp_mathlib_List_mapTR_loop___at___00Lean_Meta_RefinedDiscrTree_encodeExprWithEta_spec__0___closed__0);
lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExprWithEta___closed__0 = _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExprWithEta___closed__0();
lean_mark_persistent(lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExprWithEta___closed__0);
lp_mathlib_panic___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_toList_spec__0___closed__0 = _init_lp_mathlib_panic___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_toList_spec__0___closed__0();
lean_mark_persistent(lp_mathlib_panic___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_toList_spec__0___closed__0);
lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__0 = _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__0();
lean_mark_persistent(lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__0);
lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__1 = _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__1();
lean_mark_persistent(lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__1);
lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__2 = _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__2();
lean_mark_persistent(lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__2);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
