// Lean compiler output
// Module: Mathlib.Tactic.Simproc.ExistsAndEq
// Imports: public import Init public import Mathlib.Init public meta import Qq
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
static lean_object* lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___closed__0;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__4;
static lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__1;
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_instInhabitedSimpM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__1;
lean_object* l_Lean_Meta_ppExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEq___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00ExistsAndEq_existsAndEq_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00ExistsAndEq_existsAndEq_spec__2___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__0___boxed(lean_object**);
static lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__3;
uint64_t l_Lean_Meta_Context_configKey(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__3;
lean_object* l_Lean_Meta_mkFreshLevelMVar(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00ExistsAndEq_existsAndEq_spec__2(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_ExistsAndEq_mkNestedExists___closed__2;
static lean_object* lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_ExistsAndEq_mkNestedExists___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__6;
static lean_object* lp_mathlib_ExistsAndEq_mkNestedExists___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__9;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_right_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withNestedExistsElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_left_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__2___boxed(lean_object**);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__8;
static lean_object* lp_mathlib_ExistsAndEq_existsAndEq___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static uint64_t lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00ExistsAndEq_existsAndEq_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_Qq_Qq_Impl_mkLambdaQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_ctorIdx___boxed(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
static lean_object* lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__3;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__11;
static lean_object* lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__0;
uint64_t lean_uint64_lor(uint64_t, uint64_t);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEq___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isApp(lean_object*);
lean_object* l_List_get_x3fInternal___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Level_succ___override(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_right_elim___redArg___boxed(lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__0;
lean_object* lean_array_push(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__3___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEq___lam__0___boxed(lean_object**);
lean_object* l_Lean_KVMap_find(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx_x27___at___00ExistsAndEq_existsAndEq_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
static lean_object* lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__1;
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_Expr_bvar___override(lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___redArg(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEq___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00ExistsAndEq_existsAndEq_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvarId_x21(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_right_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_ExistsAndEq_instBEqGoTo___closed__0;
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
static lean_object* lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__5;
static lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__6;
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_left_elim___redArg(lean_object*);
static lean_object* lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00ExistsAndEq_mkNestedExists_spec__0(lean_object*);
lean_object* l_Lean_Expr_cleanupAnnotations(lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
static lean_object* lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___closed__1;
static lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00ExistsAndEq_existsAndEq_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
static lean_object* lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__2;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__11;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEq___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedLocalDecl_default;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withExistsElimAlongPath___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__5;
uint8_t l_Lean_Expr_containsFVar(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withNestedExistsElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instantiateMVarsIfMVarApp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__1___boxed(lean_object**);
static lean_object* lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__2;
static lean_object* lp_mathlib_ExistsAndEq_mkNestedExists___closed__5;
static lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00ExistsAndEq_existsAndEq_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___lam__1___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withNestedExistsIntro___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__6;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_ExistsAndEq_instBEqGoTo_beq(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___lam__2(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_instInhabitedVarQ;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___lam__3(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__4;
uint8_t lean_expr_eqv(lean_object*, lean_object*);
static lean_object* lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__2;
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00ExistsAndEq_existsAndEq_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEq___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__5;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEqPath(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__2;
static lean_object* lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1___closed__0;
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_left_elim(lean_object*, uint8_t, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__7;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_instantiate_level_mvars(lean_object*, lean_object*);
static lean_object* lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__4;
lean_object* l_Lean_Expr_forallE___override(lean_object*, lean_object*, lean_object*, uint8_t);
static lean_object* lp_mathlib_ExistsAndEq_mkNestedExists___closed__0;
lean_object* lean_st_ref_get(lean_object*);
static lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_replaceFVar(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_ExistsAndEq_withNestedExistsIntro___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00ExistsAndEq_existsAndEq_spec__3___redArg(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_toCtorIdx___boxed(lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__5;
static lean_object* lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__1;
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_lambdaTelescopeImp(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withExistsElimAlongPath(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3___closed__0;
static lean_object* lp_mathlib_panic___at___00ExistsAndEq_mkNestedExists_spec__0___closed__0;
static lean_object* lp_mathlib_panic___at___00ExistsAndEq_existsAndEq_spec__1___closed__0;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__9;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00ExistsAndEq_existsAndEq_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_warningAsError;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_instBEqGoTo_beq___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__4;
static lean_object* lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__10;
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00ExistsAndEq_existsAndEq_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
extern lean_object* l_Std_Format_defWidth;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_ctorElim___redArg(lean_object*);
static lean_object* lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1___closed__1;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_ExistsAndEq_mkNestedExists___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00ExistsAndEq_existsAndEq_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx_x27___at___00ExistsAndEq_existsAndEq_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_userName(lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_ctorElim___redArg___boxed(lean_object*);
lean_object* l_Lean_Meta_Context_config(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_instInhabitedHypQ;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__14;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEqPath___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_constLevels_x21(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx_x27___at___00ExistsAndEq_existsAndEq_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__1;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__12;
lean_object* l_Lean_Expr_appFnCleanup___redArg(lean_object*);
static lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__4;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__10;
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_instBEqGoTo;
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___lam__3(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withNestedExistsIntro___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_ExistsAndEq_instInhabitedGoTo_default;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00ExistsAndEq_existsAndEq_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__3;
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_ctorIdx(uint8_t);
static lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__3;
static lean_object* lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1___closed__2;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_left_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withNestedExistsIntro___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_right_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instInhabitedMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkNestedExists___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__13;
lean_object* lean_panic_fn(lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__4___boxed(lean_object**);
static lean_object* lp_mathlib_ExistsAndEq_existsAndEq___lam__1___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00ExistsAndEq_existsAndEq_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint64_t lean_uint64_shift_left(uint64_t, uint64_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__16;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__12;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withNestedExistsIntro(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__15;
lean_object* lean_array_mk(lean_object*);
static lean_object* lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__4;
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_betaRev(lean_object*, lean_object*, uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEqPath___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEqPath___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__13;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__0;
lean_object* lean_string_append(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
static lean_object* lp_mathlib_ExistsAndEq_existsAndEq___lam__1___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLambdaFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__7;
uint64_t l_Lean_Meta_TransparencyMode_toUInt64(uint8_t);
lean_object* l_Lean_Expr_lam___override(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_toCtorIdx(uint8_t);
static lean_object* lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkNestedExists(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalContext_findFVar_x3f(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3;
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_ExistsAndEq_instInhabitedGoTo;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__1;
static lean_object* lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx_x27___at___00ExistsAndEq_existsAndEq_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_ctorIdx(uint8_t x_1) {
_start:
{
if (x_1 == 0)
{
lean_object* x_2; 
x_2 = lean_unsigned_to_nat(0u);
return x_2;
}
else
{
lean_object* x_3; 
x_3 = lean_unsigned_to_nat(1u);
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_ctorIdx___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lean_unbox(x_1);
x_3 = lp_mathlib_ExistsAndEq_GoTo_ctorIdx(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_toCtorIdx(uint8_t x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_ExistsAndEq_GoTo_ctorIdx(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_toCtorIdx___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lean_unbox(x_1);
x_3 = lp_mathlib_ExistsAndEq_GoTo_toCtorIdx(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_ctorElim(lean_object* x_1, lean_object* x_2, uint8_t x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_inc(x_5);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_ctorElim___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_ctorElim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; lean_object* x_7; 
x_6 = lean_unbox(x_3);
x_7 = lp_mathlib_ExistsAndEq_GoTo_ctorElim(x_1, x_2, x_6, x_4, x_5);
lean_dec(x_5);
lean_dec(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_ctorElim___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_ExistsAndEq_GoTo_ctorElim___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_left_elim(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_inc(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_left_elim___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_left_elim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lean_unbox(x_2);
x_6 = lp_mathlib_ExistsAndEq_GoTo_left_elim(x_1, x_5, x_3, x_4);
lean_dec(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_left_elim___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_ExistsAndEq_GoTo_left_elim___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_right_elim(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_inc(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_right_elim___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_right_elim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lean_unbox(x_2);
x_6 = lp_mathlib_ExistsAndEq_GoTo_right_elim(x_1, x_5, x_3, x_4);
lean_dec(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_right_elim___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_ExistsAndEq_GoTo_right_elim___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT uint8_t lp_mathlib_ExistsAndEq_instBEqGoTo_beq(uint8_t x_1, uint8_t x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_3 = lp_mathlib_ExistsAndEq_GoTo_ctorIdx(x_1);
x_4 = lp_mathlib_ExistsAndEq_GoTo_ctorIdx(x_2);
x_5 = lean_nat_dec_eq(x_3, x_4);
lean_dec(x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_instBEqGoTo_beq___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; uint8_t x_4; uint8_t x_5; lean_object* x_6; 
x_3 = lean_unbox(x_1);
x_4 = lean_unbox(x_2);
x_5 = lp_mathlib_ExistsAndEq_instBEqGoTo_beq(x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_instBEqGoTo___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_instBEqGoTo_beq___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_instBEqGoTo() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_ExistsAndEq_instBEqGoTo___closed__0;
return x_1;
}
}
static uint8_t _init_lp_mathlib_ExistsAndEq_instInhabitedGoTo_default() {
_start:
{
uint8_t x_1; 
x_1 = 0;
return x_1;
}
}
static uint8_t _init_lp_mathlib_ExistsAndEq_instInhabitedGoTo() {
_start:
{
uint8_t x_1; 
x_1 = 0;
return x_1;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("_inhabitedExprDummy", 19, 19);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__0;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__1;
x_3 = l_Lean_Expr_const___override(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__2;
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__3;
x_2 = lean_box(0);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_instInhabitedVarQ() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__4;
return x_1;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_instInhabitedHypQ() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__3;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_7 = lean_st_ref_get(x_5);
x_8 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_8);
lean_dec(x_7);
x_9 = lean_st_ref_get(x_3);
x_10 = lean_ctor_get(x_9, 0);
lean_inc_ref(x_10);
lean_dec(x_9);
x_11 = lean_ctor_get(x_2, 2);
x_12 = lean_ctor_get(x_4, 2);
lean_inc(x_12);
lean_inc_ref(x_11);
x_13 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_13, 0, x_8);
lean_ctor_set(x_13, 1, x_10);
lean_ctor_set(x_13, 2, x_11);
lean_ctor_set(x_13, 3, x_12);
x_14 = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(x_14, 0, x_13);
lean_ctor_set(x_14, 1, x_1);
x_15 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_15, 0, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__5___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_7 = lean_ctor_get(x_4, 5);
x_8 = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0_spec__0(x_1, x_2, x_3, x_4, x_5);
x_9 = !lean_is_exclusive(x_8);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; 
x_10 = lean_ctor_get(x_8, 0);
lean_inc(x_7);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_7);
lean_ctor_set(x_11, 1, x_10);
lean_ctor_set_tag(x_8, 1);
lean_ctor_set(x_8, 0, x_11);
return x_8;
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_12 = lean_ctor_get(x_8, 0);
lean_inc(x_12);
lean_dec(x_8);
lean_inc(x_7);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_7);
lean_ctor_set(x_13, 1, x_12);
x_14 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_14, 0, x_13);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__5(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__5___redArg(x_2, x_3, x_4, x_5, x_6);
return x_8;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("existsAndEq: internal error, unreachable case has occurred:\n", 60, 60);
return x_1;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(".", 1, 1);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("", 0, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Elab", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Tactic", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("unsolvedGoals", 13, 13);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("synthPlaceholder", 16, 16);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("lean", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("inductionWithNoAlts", 19, 19);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("_namedError", 11, 11);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__7() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("trace", 5, 5);
return x_1;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0(uint8_t x_1, uint8_t x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 1)
{
lean_object* x_4; 
x_4 = lean_ctor_get(x_3, 0);
switch (lean_obj_tag(x_4)) {
case 1:
{
lean_object* x_5; 
x_5 = lean_ctor_get(x_4, 0);
switch (lean_obj_tag(x_5)) {
case 0:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_6 = lean_ctor_get(x_3, 1);
x_7 = lean_ctor_get(x_4, 1);
x_8 = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__0;
x_9 = lean_string_dec_eq(x_7, x_8);
if (x_9 == 0)
{
lean_object* x_10; uint8_t x_11; 
x_10 = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__1;
x_11 = lean_string_dec_eq(x_7, x_10);
if (x_11 == 0)
{
return x_1;
}
else
{
lean_object* x_12; uint8_t x_13; 
x_12 = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__2;
x_13 = lean_string_dec_eq(x_6, x_12);
if (x_13 == 0)
{
return x_1;
}
else
{
return x_2;
}
}
}
else
{
lean_object* x_14; uint8_t x_15; 
x_14 = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__3;
x_15 = lean_string_dec_eq(x_6, x_14);
if (x_15 == 0)
{
return x_1;
}
else
{
return x_2;
}
}
}
case 1:
{
lean_object* x_16; 
x_16 = lean_ctor_get(x_5, 0);
if (lean_obj_tag(x_16) == 0)
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; uint8_t x_21; 
x_17 = lean_ctor_get(x_3, 1);
x_18 = lean_ctor_get(x_4, 1);
x_19 = lean_ctor_get(x_5, 1);
x_20 = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__4;
x_21 = lean_string_dec_eq(x_19, x_20);
if (x_21 == 0)
{
return x_1;
}
else
{
lean_object* x_22; uint8_t x_23; 
x_22 = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__5;
x_23 = lean_string_dec_eq(x_18, x_22);
if (x_23 == 0)
{
return x_1;
}
else
{
lean_object* x_24; uint8_t x_25; 
x_24 = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__6;
x_25 = lean_string_dec_eq(x_17, x_24);
if (x_25 == 0)
{
return x_1;
}
else
{
return x_2;
}
}
}
}
else
{
return x_1;
}
}
default: 
{
return x_1;
}
}
}
case 0:
{
lean_object* x_26; lean_object* x_27; uint8_t x_28; 
x_26 = lean_ctor_get(x_3, 1);
x_27 = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__7;
x_28 = lean_string_dec_eq(x_26, x_27);
if (x_28 == 0)
{
return x_1;
}
else
{
return x_2;
}
}
default: 
{
return x_1;
}
}
}
else
{
return x_1;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; uint8_t x_5; uint8_t x_6; lean_object* x_7; 
x_4 = lean_unbox(x_1);
x_5 = lean_unbox(x_2);
x_6 = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0(x_4, x_5, x_3);
lean_dec(x_3);
x_7 = lean_box(x_6);
return x_7;
}
}
static lean_object* _init_lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = l_Lean_warningAsError;
return x_1;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0_spec__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_2, 0);
x_4 = lean_ctor_get(x_2, 1);
x_5 = l_Lean_KVMap_find(x_1, x_3);
if (lean_obj_tag(x_5) == 0)
{
uint8_t x_6; 
x_6 = lean_unbox(x_4);
return x_6;
}
else
{
lean_object* x_7; 
x_7 = lean_ctor_get(x_5, 0);
lean_inc(x_7);
lean_dec_ref(x_5);
if (lean_obj_tag(x_7) == 1)
{
uint8_t x_8; 
x_8 = lean_ctor_get_uint8(x_7, 0);
lean_dec_ref(x_7);
return x_8;
}
else
{
uint8_t x_9; 
lean_dec(x_7);
x_9 = lean_unbox(x_4);
return x_9;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0(lean_object* x_1, lean_object* x_2, uint8_t x_3, uint8_t x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
uint8_t x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_50; lean_object* x_51; uint8_t x_52; lean_object* x_53; uint8_t x_54; uint8_t x_55; lean_object* x_56; lean_object* x_57; lean_object* x_77; lean_object* x_78; uint8_t x_79; lean_object* x_80; uint8_t x_81; uint8_t x_82; lean_object* x_83; lean_object* x_84; lean_object* x_88; lean_object* x_89; uint8_t x_90; uint8_t x_91; lean_object* x_92; lean_object* x_93; uint8_t x_94; uint8_t x_100; lean_object* x_101; uint8_t x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; uint8_t x_106; uint8_t x_107; uint8_t x_109; uint8_t x_125; 
x_100 = 2;
x_125 = l_Lean_instBEqMessageSeverity_beq(x_3, x_100);
if (x_125 == 0)
{
x_109 = x_125;
goto block_124;
}
else
{
uint8_t x_126; 
lean_inc_ref(x_2);
x_126 = l_Lean_MessageData_hasSyntheticSorry(x_2);
x_109 = x_126;
goto block_124;
}
block_49:
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; uint8_t x_23; 
x_20 = lean_st_ref_take(x_18);
x_21 = lean_ctor_get(x_17, 6);
lean_inc(x_21);
x_22 = lean_ctor_get(x_17, 7);
lean_inc(x_22);
lean_dec_ref(x_17);
x_23 = !lean_is_exclusive(x_20);
if (x_23 == 0)
{
lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; 
x_24 = lean_ctor_get(x_20, 6);
x_25 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_25, 0, x_21);
lean_ctor_set(x_25, 1, x_22);
x_26 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_26, 0, x_25);
lean_ctor_set(x_26, 1, x_16);
x_27 = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(x_27, 0, x_15);
lean_ctor_set(x_27, 1, x_11);
lean_ctor_set(x_27, 2, x_13);
lean_ctor_set(x_27, 3, x_12);
lean_ctor_set(x_27, 4, x_26);
lean_ctor_set_uint8(x_27, sizeof(void*)*5, x_14);
lean_ctor_set_uint8(x_27, sizeof(void*)*5 + 1, x_10);
lean_ctor_set_uint8(x_27, sizeof(void*)*5 + 2, x_4);
x_28 = l_Lean_MessageLog_add(x_27, x_24);
lean_ctor_set(x_20, 6, x_28);
x_29 = lean_st_ref_set(x_18, x_20);
x_30 = lean_box(0);
x_31 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_31, 0, x_30);
return x_31;
}
else
{
lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; 
x_32 = lean_ctor_get(x_20, 0);
x_33 = lean_ctor_get(x_20, 1);
x_34 = lean_ctor_get(x_20, 2);
x_35 = lean_ctor_get(x_20, 3);
x_36 = lean_ctor_get(x_20, 4);
x_37 = lean_ctor_get(x_20, 5);
x_38 = lean_ctor_get(x_20, 6);
x_39 = lean_ctor_get(x_20, 7);
x_40 = lean_ctor_get(x_20, 8);
lean_inc(x_40);
lean_inc(x_39);
lean_inc(x_38);
lean_inc(x_37);
lean_inc(x_36);
lean_inc(x_35);
lean_inc(x_34);
lean_inc(x_33);
lean_inc(x_32);
lean_dec(x_20);
x_41 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_41, 0, x_21);
lean_ctor_set(x_41, 1, x_22);
x_42 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_42, 0, x_41);
lean_ctor_set(x_42, 1, x_16);
x_43 = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(x_43, 0, x_15);
lean_ctor_set(x_43, 1, x_11);
lean_ctor_set(x_43, 2, x_13);
lean_ctor_set(x_43, 3, x_12);
lean_ctor_set(x_43, 4, x_42);
lean_ctor_set_uint8(x_43, sizeof(void*)*5, x_14);
lean_ctor_set_uint8(x_43, sizeof(void*)*5 + 1, x_10);
lean_ctor_set_uint8(x_43, sizeof(void*)*5 + 2, x_4);
x_44 = l_Lean_MessageLog_add(x_43, x_38);
x_45 = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(x_45, 0, x_32);
lean_ctor_set(x_45, 1, x_33);
lean_ctor_set(x_45, 2, x_34);
lean_ctor_set(x_45, 3, x_35);
lean_ctor_set(x_45, 4, x_36);
lean_ctor_set(x_45, 5, x_37);
lean_ctor_set(x_45, 6, x_44);
lean_ctor_set(x_45, 7, x_39);
lean_ctor_set(x_45, 8, x_40);
x_46 = lean_st_ref_set(x_18, x_45);
x_47 = lean_box(0);
x_48 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_48, 0, x_47);
return x_48;
}
}
block_76:
{
lean_object* x_58; lean_object* x_59; uint8_t x_60; 
x_58 = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(x_2);
x_59 = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0_spec__0(x_58, x_5, x_6, x_7, x_8);
x_60 = !lean_is_exclusive(x_59);
if (x_60 == 0)
{
lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; 
x_61 = lean_ctor_get(x_59, 0);
lean_inc_ref(x_51);
x_62 = l_Lean_FileMap_toPosition(x_51, x_53);
lean_dec(x_53);
x_63 = l_Lean_FileMap_toPosition(x_51, x_57);
lean_dec(x_57);
x_64 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_64, 0, x_63);
x_65 = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___closed__0;
if (x_55 == 0)
{
lean_free_object(x_59);
lean_dec_ref(x_50);
x_10 = x_52;
x_11 = x_62;
x_12 = x_65;
x_13 = x_64;
x_14 = x_54;
x_15 = x_56;
x_16 = x_61;
x_17 = x_7;
x_18 = x_8;
x_19 = lean_box(0);
goto block_49;
}
else
{
uint8_t x_66; 
lean_inc(x_61);
x_66 = l_Lean_MessageData_hasTag(x_50, x_61);
if (x_66 == 0)
{
lean_object* x_67; 
lean_dec_ref(x_64);
lean_dec_ref(x_62);
lean_dec(x_61);
lean_dec_ref(x_56);
lean_dec_ref(x_7);
x_67 = lean_box(0);
lean_ctor_set(x_59, 0, x_67);
return x_59;
}
else
{
lean_free_object(x_59);
x_10 = x_52;
x_11 = x_62;
x_12 = x_65;
x_13 = x_64;
x_14 = x_54;
x_15 = x_56;
x_16 = x_61;
x_17 = x_7;
x_18 = x_8;
x_19 = lean_box(0);
goto block_49;
}
}
}
else
{
lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; 
x_68 = lean_ctor_get(x_59, 0);
lean_inc(x_68);
lean_dec(x_59);
lean_inc_ref(x_51);
x_69 = l_Lean_FileMap_toPosition(x_51, x_53);
lean_dec(x_53);
x_70 = l_Lean_FileMap_toPosition(x_51, x_57);
lean_dec(x_57);
x_71 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_71, 0, x_70);
x_72 = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___closed__0;
if (x_55 == 0)
{
lean_dec_ref(x_50);
x_10 = x_52;
x_11 = x_69;
x_12 = x_72;
x_13 = x_71;
x_14 = x_54;
x_15 = x_56;
x_16 = x_68;
x_17 = x_7;
x_18 = x_8;
x_19 = lean_box(0);
goto block_49;
}
else
{
uint8_t x_73; 
lean_inc(x_68);
x_73 = l_Lean_MessageData_hasTag(x_50, x_68);
if (x_73 == 0)
{
lean_object* x_74; lean_object* x_75; 
lean_dec_ref(x_71);
lean_dec_ref(x_69);
lean_dec(x_68);
lean_dec_ref(x_56);
lean_dec_ref(x_7);
x_74 = lean_box(0);
x_75 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_75, 0, x_74);
return x_75;
}
else
{
x_10 = x_52;
x_11 = x_69;
x_12 = x_72;
x_13 = x_71;
x_14 = x_54;
x_15 = x_56;
x_16 = x_68;
x_17 = x_7;
x_18 = x_8;
x_19 = lean_box(0);
goto block_49;
}
}
}
}
block_87:
{
lean_object* x_85; 
x_85 = l_Lean_Syntax_getTailPos_x3f(x_80, x_81);
lean_dec(x_80);
if (lean_obj_tag(x_85) == 0)
{
lean_inc(x_84);
x_50 = x_77;
x_51 = x_78;
x_52 = x_79;
x_53 = x_84;
x_54 = x_81;
x_55 = x_82;
x_56 = x_83;
x_57 = x_84;
goto block_76;
}
else
{
lean_object* x_86; 
x_86 = lean_ctor_get(x_85, 0);
lean_inc(x_86);
lean_dec_ref(x_85);
x_50 = x_77;
x_51 = x_78;
x_52 = x_79;
x_53 = x_84;
x_54 = x_81;
x_55 = x_82;
x_56 = x_83;
x_57 = x_86;
goto block_76;
}
}
block_99:
{
lean_object* x_95; lean_object* x_96; 
x_95 = l_Lean_replaceRef(x_1, x_93);
lean_dec(x_93);
x_96 = l_Lean_Syntax_getPos_x3f(x_95, x_90);
if (lean_obj_tag(x_96) == 0)
{
lean_object* x_97; 
x_97 = lean_unsigned_to_nat(0u);
x_77 = x_88;
x_78 = x_89;
x_79 = x_94;
x_80 = x_95;
x_81 = x_90;
x_82 = x_91;
x_83 = x_92;
x_84 = x_97;
goto block_87;
}
else
{
lean_object* x_98; 
x_98 = lean_ctor_get(x_96, 0);
lean_inc(x_98);
lean_dec_ref(x_96);
x_77 = x_88;
x_78 = x_89;
x_79 = x_94;
x_80 = x_95;
x_81 = x_90;
x_82 = x_91;
x_83 = x_92;
x_84 = x_98;
goto block_87;
}
}
block_108:
{
if (x_107 == 0)
{
x_88 = x_105;
x_89 = x_101;
x_90 = x_106;
x_91 = x_102;
x_92 = x_103;
x_93 = x_104;
x_94 = x_3;
goto block_99;
}
else
{
x_88 = x_105;
x_89 = x_101;
x_90 = x_106;
x_91 = x_102;
x_92 = x_103;
x_93 = x_104;
x_94 = x_100;
goto block_99;
}
}
block_124:
{
if (x_109 == 0)
{
lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; uint8_t x_114; lean_object* x_115; lean_object* x_116; lean_object* x_117; uint8_t x_118; uint8_t x_119; 
x_110 = lean_ctor_get(x_7, 0);
x_111 = lean_ctor_get(x_7, 1);
x_112 = lean_ctor_get(x_7, 2);
x_113 = lean_ctor_get(x_7, 5);
x_114 = lean_ctor_get_uint8(x_7, sizeof(void*)*14 + 1);
x_115 = lean_box(x_109);
x_116 = lean_box(x_114);
x_117 = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___boxed), 3, 2);
lean_closure_set(x_117, 0, x_115);
lean_closure_set(x_117, 1, x_116);
x_118 = 1;
x_119 = l_Lean_instBEqMessageSeverity_beq(x_3, x_118);
if (x_119 == 0)
{
lean_inc(x_113);
lean_inc_ref(x_110);
lean_inc_ref(x_111);
x_101 = x_111;
x_102 = x_114;
x_103 = x_110;
x_104 = x_113;
x_105 = x_117;
x_106 = x_109;
x_107 = x_119;
goto block_108;
}
else
{
lean_object* x_120; uint8_t x_121; 
x_120 = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___closed__1;
x_121 = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0_spec__1(x_112, x_120);
lean_inc(x_113);
lean_inc_ref(x_110);
lean_inc_ref(x_111);
x_101 = x_111;
x_102 = x_114;
x_103 = x_110;
x_104 = x_113;
x_105 = x_117;
x_106 = x_109;
x_107 = x_121;
goto block_108;
}
}
else
{
lean_object* x_122; lean_object* x_123; 
lean_dec_ref(x_7);
lean_dec_ref(x_2);
x_122 = lean_box(0);
x_123 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_123, 0, x_122);
return x_123;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0(lean_object* x_1, uint8_t x_2, uint8_t x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_9; lean_object* x_10; 
x_9 = lean_ctor_get(x_6, 5);
lean_inc(x_9);
x_10 = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0(x_9, x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_7; uint8_t x_8; lean_object* x_9; 
x_7 = 2;
x_8 = 0;
x_9 = lp_mathlib_Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0(x_1, x_7, x_8, x_2, x_3, x_4, x_5);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_7 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg___closed__0;
x_8 = lean_string_append(x_7, x_1);
x_9 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg___closed__1;
x_10 = lean_string_append(x_8, x_9);
x_11 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_11, 0, x_10);
x_12 = l_Lean_MessageData_ofFormat(x_11);
lean_inc_ref(x_4);
lean_inc_ref(x_12);
x_13 = lp_mathlib_Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0(x_12, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_13) == 0)
{
lean_object* x_14; 
lean_dec_ref(x_13);
x_14 = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__5___redArg(x_12, x_2, x_3, x_4, x_5);
lean_dec_ref(x_4);
return x_14;
}
else
{
uint8_t x_15; 
lean_dec_ref(x_12);
lean_dec_ref(x_4);
x_15 = !lean_is_exclusive(x_13);
if (x_15 == 0)
{
return x_13;
}
else
{
lean_object* x_16; lean_object* x_17; 
x_16 = lean_ctor_get(x_13, 0);
lean_inc(x_16);
lean_dec(x_13);
x_17 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_17, 0, x_16);
return x_17;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg(x_2, x_3, x_4, x_5, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__5___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__5(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
uint8_t x_9; uint8_t x_10; lean_object* x_11; 
x_9 = lean_unbox(x_2);
x_10 = lean_unbox(x_3);
x_11 = lp_mathlib_Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0(x_1, x_9, x_10, x_4, x_5, x_6, x_7);
lean_dec(x_7);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0_spec__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0_spec__1(x_1, x_2);
lean_dec_ref(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__5___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__5___redArg(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0_spec__0(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
uint8_t x_10; uint8_t x_11; lean_object* x_12; 
x_10 = lean_unbox(x_3);
x_11 = lean_unbox(x_4);
x_12 = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0(x_1, x_2, x_10, x_11, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_1);
return x_12;
}
}
static lean_object* _init_lp_mathlib_panic___at___00ExistsAndEq_mkNestedExists_spec__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = l_Lean_instInhabitedLocalDecl_default;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00ExistsAndEq_mkNestedExists_spec__0(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_panic___at___00ExistsAndEq_mkNestedExists_spec__0___closed__0;
x_3 = lean_panic_fn(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_mkNestedExists___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Exists", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_mkNestedExists___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_ExistsAndEq_mkNestedExists___closed__0;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_mkNestedExists___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Init.Data.Option.BasicAux", 25, 25);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_mkNestedExists___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Option.get!", 11, 11);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_mkNestedExists___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("value is none", 13, 13);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_mkNestedExists___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_1 = lp_mathlib_ExistsAndEq_mkNestedExists___closed__4;
x_2 = lean_unsigned_to_nat(14u);
x_3 = lean_unsigned_to_nat(22u);
x_4 = lp_mathlib_ExistsAndEq_mkNestedExists___closed__3;
x_5 = lp_mathlib_ExistsAndEq_mkNestedExists___closed__2;
x_6 = l_mkPanicMessageWithDecl(x_5, x_4, x_3, x_2, x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkNestedExists(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_8; 
lean_dec_ref(x_3);
x_8 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_8, 0, x_2);
return x_8;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_9 = lean_ctor_get(x_1, 0);
lean_inc(x_9);
x_10 = lean_ctor_get(x_9, 1);
lean_inc(x_10);
x_11 = lean_ctor_get(x_1, 1);
lean_inc(x_11);
if (lean_is_exclusive(x_1)) {
 lean_ctor_release(x_1, 0);
 lean_ctor_release(x_1, 1);
 x_12 = x_1;
} else {
 lean_dec_ref(x_1);
 x_12 = lean_box(0);
}
x_13 = lean_ctor_get(x_9, 0);
lean_inc(x_13);
lean_dec(x_9);
x_14 = lean_ctor_get(x_10, 0);
lean_inc(x_14);
x_15 = lean_ctor_get(x_10, 1);
lean_inc(x_15);
lean_dec(x_10);
lean_inc_ref(x_3);
x_16 = lp_mathlib_ExistsAndEq_mkNestedExists(x_11, x_2, x_3, x_4, x_5, x_6);
if (lean_obj_tag(x_16) == 0)
{
lean_object* x_17; lean_object* x_18; lean_object* x_38; lean_object* x_39; 
x_17 = lean_ctor_get(x_16, 0);
lean_inc(x_17);
lean_dec_ref(x_16);
x_38 = lean_ctor_get(x_3, 2);
lean_inc_ref(x_38);
x_39 = l_Lean_LocalContext_findFVar_x3f(x_38, x_15);
if (lean_obj_tag(x_39) == 0)
{
lean_object* x_40; lean_object* x_41; 
x_40 = lp_mathlib_ExistsAndEq_mkNestedExists___closed__5;
x_41 = lp_mathlib_panic___at___00ExistsAndEq_mkNestedExists_spec__0(x_40);
x_18 = x_41;
goto block_37;
}
else
{
lean_object* x_42; 
x_42 = lean_ctor_get(x_39, 0);
lean_inc(x_42);
lean_dec_ref(x_39);
x_18 = x_42;
goto block_37;
}
block_37:
{
lean_object* x_19; lean_object* x_20; 
x_19 = l_Lean_LocalDecl_userName(x_18);
lean_dec_ref(x_18);
lean_inc(x_14);
x_20 = lp_Qq_Qq_Impl_mkLambdaQ___redArg(x_14, x_19, x_15, x_17, x_3, x_4, x_5, x_6);
lean_dec_ref(x_3);
if (lean_obj_tag(x_20) == 0)
{
uint8_t x_21; 
x_21 = !lean_is_exclusive(x_20);
if (x_21 == 0)
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; 
x_22 = lean_ctor_get(x_20, 0);
x_23 = lp_mathlib_ExistsAndEq_mkNestedExists___closed__1;
x_24 = lean_box(0);
if (lean_is_scalar(x_12)) {
 x_25 = lean_alloc_ctor(1, 2, 0);
} else {
 x_25 = x_12;
}
lean_ctor_set(x_25, 0, x_13);
lean_ctor_set(x_25, 1, x_24);
x_26 = l_Lean_Expr_const___override(x_23, x_25);
x_27 = l_Lean_Expr_app___override(x_26, x_14);
x_28 = l_Lean_Expr_app___override(x_27, x_22);
lean_ctor_set(x_20, 0, x_28);
return x_20;
}
else
{
lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; 
x_29 = lean_ctor_get(x_20, 0);
lean_inc(x_29);
lean_dec(x_20);
x_30 = lp_mathlib_ExistsAndEq_mkNestedExists___closed__1;
x_31 = lean_box(0);
if (lean_is_scalar(x_12)) {
 x_32 = lean_alloc_ctor(1, 2, 0);
} else {
 x_32 = x_12;
}
lean_ctor_set(x_32, 0, x_13);
lean_ctor_set(x_32, 1, x_31);
x_33 = l_Lean_Expr_const___override(x_30, x_32);
x_34 = l_Lean_Expr_app___override(x_33, x_14);
x_35 = l_Lean_Expr_app___override(x_34, x_29);
x_36 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_36, 0, x_35);
return x_36;
}
}
else
{
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_12);
return x_20;
}
}
}
else
{
lean_dec(x_15);
lean_dec(x_14);
lean_dec(x_13);
lean_dec(x_12);
lean_dec_ref(x_3);
return x_16;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkNestedExists___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_ExistsAndEq_mkNestedExists(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
return x_8;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_box(0);
x_2 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("And", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__1;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Eq", 2, 2);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__3;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEqPath___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_5; 
x_5 = l_Lean_Meta_instantiateMVarsIfMVarApp___redArg(x_2, x_3);
if (lean_obj_tag(x_5) == 0)
{
lean_object* x_6; lean_object* x_7; uint8_t x_14; lean_object* x_18; uint8_t x_19; 
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
if (lean_is_exclusive(x_5)) {
 lean_ctor_release(x_5, 0);
 x_7 = x_5;
} else {
 lean_dec_ref(x_5);
 x_7 = lean_box(0);
}
x_18 = l_Lean_Expr_cleanupAnnotations(x_6);
x_19 = l_Lean_Expr_isApp(x_18);
if (x_19 == 0)
{
lean_dec_ref(x_18);
goto block_10;
}
else
{
lean_object* x_20; uint8_t x_21; 
lean_inc_ref(x_18);
x_20 = l_Lean_Expr_appFnCleanup___redArg(x_18);
x_21 = l_Lean_Expr_isApp(x_20);
if (x_21 == 0)
{
lean_dec_ref(x_20);
lean_dec_ref(x_18);
goto block_10;
}
else
{
lean_object* x_22; lean_object* x_23; uint8_t x_24; lean_object* x_31; lean_object* x_32; uint8_t x_33; 
x_22 = lean_ctor_get(x_18, 1);
lean_inc_ref(x_22);
lean_dec_ref(x_18);
x_23 = lean_ctor_get(x_20, 1);
lean_inc_ref(x_23);
x_31 = l_Lean_Expr_appFnCleanup___redArg(x_20);
x_32 = lp_mathlib_ExistsAndEq_mkNestedExists___closed__1;
x_33 = l_Lean_Expr_isConstOf(x_31, x_32);
if (x_33 == 0)
{
lean_object* x_34; uint8_t x_35; 
x_34 = lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__2;
x_35 = l_Lean_Expr_isConstOf(x_31, x_34);
if (x_35 == 0)
{
uint8_t x_36; 
x_36 = l_Lean_Expr_isApp(x_31);
if (x_36 == 0)
{
lean_dec_ref(x_31);
lean_dec_ref(x_23);
lean_dec_ref(x_22);
goto block_10;
}
else
{
lean_object* x_37; lean_object* x_38; uint8_t x_39; 
x_37 = l_Lean_Expr_appFnCleanup___redArg(x_31);
x_38 = lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__4;
x_39 = l_Lean_Expr_isConstOf(x_37, x_38);
lean_dec_ref(x_37);
if (x_39 == 0)
{
lean_dec_ref(x_23);
lean_dec_ref(x_22);
goto block_10;
}
else
{
uint8_t x_40; 
lean_dec(x_7);
x_40 = lean_expr_eqv(x_1, x_23);
if (x_40 == 0)
{
x_24 = x_40;
goto block_30;
}
else
{
lean_object* x_41; uint8_t x_42; 
x_41 = l_Lean_Expr_fvarId_x21(x_1);
x_42 = l_Lean_Expr_containsFVar(x_22, x_41);
lean_dec(x_41);
if (x_42 == 0)
{
x_24 = x_40;
goto block_30;
}
else
{
x_24 = x_35;
goto block_30;
}
}
}
}
}
else
{
lean_object* x_43; 
lean_dec_ref(x_31);
lean_dec(x_7);
x_43 = lp_mathlib_ExistsAndEq_findEqPath___redArg(x_1, x_23, x_3);
if (lean_obj_tag(x_43) == 0)
{
uint8_t x_44; 
x_44 = !lean_is_exclusive(x_43);
if (x_44 == 0)
{
lean_object* x_45; 
x_45 = lean_ctor_get(x_43, 0);
if (lean_obj_tag(x_45) == 1)
{
uint8_t x_46; 
lean_dec_ref(x_22);
x_46 = !lean_is_exclusive(x_45);
if (x_46 == 0)
{
lean_object* x_47; uint8_t x_48; lean_object* x_49; lean_object* x_50; 
x_47 = lean_ctor_get(x_45, 0);
x_48 = 0;
x_49 = lean_box(x_48);
x_50 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_50, 0, x_49);
lean_ctor_set(x_50, 1, x_47);
lean_ctor_set(x_45, 0, x_50);
return x_43;
}
else
{
lean_object* x_51; uint8_t x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; 
x_51 = lean_ctor_get(x_45, 0);
lean_inc(x_51);
lean_dec(x_45);
x_52 = 0;
x_53 = lean_box(x_52);
x_54 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_54, 0, x_53);
lean_ctor_set(x_54, 1, x_51);
x_55 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_55, 0, x_54);
lean_ctor_set(x_43, 0, x_55);
return x_43;
}
}
else
{
lean_object* x_56; 
lean_free_object(x_43);
lean_dec(x_45);
x_56 = lp_mathlib_ExistsAndEq_findEqPath___redArg(x_1, x_22, x_3);
if (lean_obj_tag(x_56) == 0)
{
uint8_t x_57; 
x_57 = !lean_is_exclusive(x_56);
if (x_57 == 0)
{
lean_object* x_58; 
x_58 = lean_ctor_get(x_56, 0);
if (lean_obj_tag(x_58) == 1)
{
uint8_t x_59; 
x_59 = !lean_is_exclusive(x_58);
if (x_59 == 0)
{
lean_object* x_60; uint8_t x_61; lean_object* x_62; lean_object* x_63; 
x_60 = lean_ctor_get(x_58, 0);
x_61 = 1;
x_62 = lean_box(x_61);
x_63 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_63, 0, x_62);
lean_ctor_set(x_63, 1, x_60);
lean_ctor_set(x_58, 0, x_63);
return x_56;
}
else
{
lean_object* x_64; uint8_t x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; 
x_64 = lean_ctor_get(x_58, 0);
lean_inc(x_64);
lean_dec(x_58);
x_65 = 1;
x_66 = lean_box(x_65);
x_67 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_67, 0, x_66);
lean_ctor_set(x_67, 1, x_64);
x_68 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_68, 0, x_67);
lean_ctor_set(x_56, 0, x_68);
return x_56;
}
}
else
{
lean_object* x_69; 
lean_dec(x_58);
x_69 = lean_box(0);
lean_ctor_set(x_56, 0, x_69);
return x_56;
}
}
else
{
lean_object* x_70; 
x_70 = lean_ctor_get(x_56, 0);
lean_inc(x_70);
lean_dec(x_56);
if (lean_obj_tag(x_70) == 1)
{
lean_object* x_71; lean_object* x_72; uint8_t x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; 
x_71 = lean_ctor_get(x_70, 0);
lean_inc(x_71);
if (lean_is_exclusive(x_70)) {
 lean_ctor_release(x_70, 0);
 x_72 = x_70;
} else {
 lean_dec_ref(x_70);
 x_72 = lean_box(0);
}
x_73 = 1;
x_74 = lean_box(x_73);
x_75 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_75, 0, x_74);
lean_ctor_set(x_75, 1, x_71);
if (lean_is_scalar(x_72)) {
 x_76 = lean_alloc_ctor(1, 1, 0);
} else {
 x_76 = x_72;
}
lean_ctor_set(x_76, 0, x_75);
x_77 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_77, 0, x_76);
return x_77;
}
else
{
lean_object* x_78; lean_object* x_79; 
lean_dec(x_70);
x_78 = lean_box(0);
x_79 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_79, 0, x_78);
return x_79;
}
}
}
else
{
return x_56;
}
}
}
else
{
lean_object* x_80; 
x_80 = lean_ctor_get(x_43, 0);
lean_inc(x_80);
lean_dec(x_43);
if (lean_obj_tag(x_80) == 1)
{
lean_object* x_81; lean_object* x_82; uint8_t x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; 
lean_dec_ref(x_22);
x_81 = lean_ctor_get(x_80, 0);
lean_inc(x_81);
if (lean_is_exclusive(x_80)) {
 lean_ctor_release(x_80, 0);
 x_82 = x_80;
} else {
 lean_dec_ref(x_80);
 x_82 = lean_box(0);
}
x_83 = 0;
x_84 = lean_box(x_83);
x_85 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_85, 0, x_84);
lean_ctor_set(x_85, 1, x_81);
if (lean_is_scalar(x_82)) {
 x_86 = lean_alloc_ctor(1, 1, 0);
} else {
 x_86 = x_82;
}
lean_ctor_set(x_86, 0, x_85);
x_87 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_87, 0, x_86);
return x_87;
}
else
{
lean_object* x_88; 
lean_dec(x_80);
x_88 = lp_mathlib_ExistsAndEq_findEqPath___redArg(x_1, x_22, x_3);
if (lean_obj_tag(x_88) == 0)
{
lean_object* x_89; lean_object* x_90; 
x_89 = lean_ctor_get(x_88, 0);
lean_inc(x_89);
if (lean_is_exclusive(x_88)) {
 lean_ctor_release(x_88, 0);
 x_90 = x_88;
} else {
 lean_dec_ref(x_88);
 x_90 = lean_box(0);
}
if (lean_obj_tag(x_89) == 1)
{
lean_object* x_91; lean_object* x_92; uint8_t x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; 
x_91 = lean_ctor_get(x_89, 0);
lean_inc(x_91);
if (lean_is_exclusive(x_89)) {
 lean_ctor_release(x_89, 0);
 x_92 = x_89;
} else {
 lean_dec_ref(x_89);
 x_92 = lean_box(0);
}
x_93 = 1;
x_94 = lean_box(x_93);
x_95 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_95, 0, x_94);
lean_ctor_set(x_95, 1, x_91);
if (lean_is_scalar(x_92)) {
 x_96 = lean_alloc_ctor(1, 1, 0);
} else {
 x_96 = x_92;
}
lean_ctor_set(x_96, 0, x_95);
if (lean_is_scalar(x_90)) {
 x_97 = lean_alloc_ctor(0, 1, 0);
} else {
 x_97 = x_90;
}
lean_ctor_set(x_97, 0, x_96);
return x_97;
}
else
{
lean_object* x_98; lean_object* x_99; 
lean_dec(x_89);
x_98 = lean_box(0);
if (lean_is_scalar(x_90)) {
 x_99 = lean_alloc_ctor(0, 1, 0);
} else {
 x_99 = x_90;
}
lean_ctor_set(x_99, 0, x_98);
return x_99;
}
}
else
{
return x_88;
}
}
}
}
else
{
lean_dec_ref(x_22);
return x_43;
}
}
}
else
{
lean_object* x_100; uint8_t x_101; 
lean_dec_ref(x_31);
lean_dec(x_7);
x_100 = l_Lean_Expr_fvarId_x21(x_1);
x_101 = l_Lean_Expr_containsFVar(x_23, x_100);
lean_dec(x_100);
lean_dec_ref(x_23);
if (x_101 == 0)
{
if (lean_obj_tag(x_22) == 6)
{
lean_object* x_102; 
x_102 = lean_ctor_get(x_22, 2);
lean_inc_ref(x_102);
lean_dec_ref(x_22);
x_2 = x_102;
goto _start;
}
else
{
lean_object* x_104; lean_object* x_105; 
lean_dec_ref(x_22);
x_104 = lean_box(0);
x_105 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_105, 0, x_104);
return x_105;
}
}
else
{
lean_object* x_106; lean_object* x_107; 
lean_dec_ref(x_22);
x_106 = lean_box(0);
x_107 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_107, 0, x_106);
return x_107;
}
}
block_30:
{
if (x_24 == 0)
{
uint8_t x_25; 
x_25 = lean_expr_eqv(x_1, x_22);
lean_dec_ref(x_22);
if (x_25 == 0)
{
lean_dec_ref(x_23);
x_14 = x_25;
goto block_17;
}
else
{
lean_object* x_26; uint8_t x_27; 
x_26 = l_Lean_Expr_fvarId_x21(x_1);
x_27 = l_Lean_Expr_containsFVar(x_23, x_26);
lean_dec(x_26);
lean_dec_ref(x_23);
if (x_27 == 0)
{
x_14 = x_25;
goto block_17;
}
else
{
goto block_13;
}
}
}
else
{
lean_object* x_28; lean_object* x_29; 
lean_dec_ref(x_23);
lean_dec_ref(x_22);
x_28 = lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__0;
x_29 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_29, 0, x_28);
return x_29;
}
}
}
}
block_10:
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_box(0);
if (lean_is_scalar(x_7)) {
 x_9 = lean_alloc_ctor(0, 1, 0);
} else {
 x_9 = x_7;
}
lean_ctor_set(x_9, 0, x_8);
return x_9;
}
block_13:
{
lean_object* x_11; lean_object* x_12; 
x_11 = lean_box(0);
x_12 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_12, 0, x_11);
return x_12;
}
block_17:
{
if (x_14 == 0)
{
goto block_13;
}
else
{
lean_object* x_15; lean_object* x_16; 
x_15 = lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__0;
x_16 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_16, 0, x_15);
return x_16;
}
}
}
else
{
uint8_t x_108; 
x_108 = !lean_is_exclusive(x_5);
if (x_108 == 0)
{
return x_5;
}
else
{
lean_object* x_109; lean_object* x_110; 
x_109 = lean_ctor_get(x_5, 0);
lean_inc(x_109);
lean_dec(x_5);
x_110 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_110, 0, x_109);
return x_110;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEqPath(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_ExistsAndEq_findEqPath___redArg(x_3, x_4, x_6);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEqPath___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_ExistsAndEq_findEqPath(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEqPath___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ExistsAndEq_findEqPath___redArg(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec_ref(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_9; 
x_9 = lean_apply_7(x_1, x_2, x_3, x_4, x_5, x_6, x_7, lean_box(0));
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, uint8_t x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; uint8_t x_11; uint8_t x_12; lean_object* x_13; lean_object* x_14; 
x_10 = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___redArg___lam__0___boxed), 8, 1);
lean_closure_set(x_10, 0, x_3);
x_11 = 1;
x_12 = 0;
x_13 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_13, 0, x_2);
x_14 = l___private_Lean_Meta_Basic_0__Lean_Meta_lambdaTelescopeImp(lean_box(0), x_1, x_11, x_12, x_11, x_12, x_13, x_10, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_13);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, uint8_t x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; 
x_8 = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), x_2, x_1, x_3, x_4, x_5, x_6);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2(lean_object* x_1, lean_object* x_2, uint8_t x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(x_2, x_3, x_4, x_5, x_6, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_4 = lean_st_ref_get(x_2);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
lean_dec(x_4);
x_6 = lean_instantiate_level_mvars(x_5, x_1);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_6, 1);
lean_inc(x_8);
lean_dec_ref(x_6);
x_9 = lean_st_ref_take(x_2);
x_10 = !lean_is_exclusive(x_9);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_11 = lean_ctor_get(x_9, 0);
lean_dec(x_11);
lean_ctor_set(x_9, 0, x_7);
x_12 = lean_st_ref_set(x_2, x_9);
x_13 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_13, 0, x_8);
return x_13;
}
else
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_14 = lean_ctor_get(x_9, 1);
x_15 = lean_ctor_get(x_9, 2);
x_16 = lean_ctor_get(x_9, 3);
x_17 = lean_ctor_get(x_9, 4);
lean_inc(x_17);
lean_inc(x_16);
lean_inc(x_15);
lean_inc(x_14);
lean_dec(x_9);
x_18 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_18, 0, x_7);
lean_ctor_set(x_18, 1, x_14);
lean_ctor_set(x_18, 2, x_15);
lean_ctor_set(x_18, 3, x_16);
lean_ctor_set(x_18, 4, x_17);
x_19 = lean_st_ref_set(x_2, x_18);
x_20 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_20, 0, x_8);
return x_20;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg(x_1, x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_4; 
x_4 = l_Lean_Expr_hasMVar(x_1);
if (x_4 == 0)
{
lean_object* x_5; 
x_5 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_5, 0, x_1);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; uint8_t x_12; 
x_6 = lean_st_ref_get(x_2);
x_7 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_7);
lean_dec(x_6);
x_8 = l_Lean_instantiateMVarsCore(x_7, x_1);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
x_10 = lean_ctor_get(x_8, 1);
lean_inc(x_10);
lean_dec_ref(x_8);
x_11 = lean_st_ref_take(x_2);
x_12 = !lean_is_exclusive(x_11);
if (x_12 == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_13 = lean_ctor_get(x_11, 0);
lean_dec(x_13);
lean_ctor_set(x_11, 0, x_10);
x_14 = lean_st_ref_set(x_2, x_11);
x_15 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_15, 0, x_9);
return x_15;
}
else
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_16 = lean_ctor_get(x_11, 1);
x_17 = lean_ctor_get(x_11, 2);
x_18 = lean_ctor_get(x_11, 3);
x_19 = lean_ctor_get(x_11, 4);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_17);
lean_inc(x_16);
lean_dec(x_11);
x_20 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_20, 0, x_10);
lean_ctor_set(x_20, 1, x_16);
lean_ctor_set(x_20, 2, x_17);
lean_ctor_set(x_20, 3, x_18);
lean_ctor_set(x_20, 4, x_19);
x_21 = lean_st_ref_set(x_2, x_20);
x_22 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_22, 0, x_9);
return x_22;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_1, x_3);
return x_7;
}
}
static uint64_t _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1() {
_start:
{
uint8_t x_1; uint64_t x_2; 
x_1 = 2;
x_2 = l_Lean_Meta_TransparencyMode_toUInt64(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__0(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, uint8_t x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
lean_inc_ref(x_7);
lean_inc(x_3);
x_12 = l_Lean_Meta_mkFreshExprMVar(x_1, x_2, x_3, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_12) == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
lean_inc(x_13);
x_14 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_14, 0, x_13);
lean_inc_ref(x_7);
lean_inc(x_3);
lean_inc_ref(x_14);
x_15 = l_Lean_Meta_mkFreshExprMVar(x_14, x_2, x_3, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_15) == 0)
{
lean_object* x_16; lean_object* x_17; 
x_16 = lean_ctor_get(x_15, 0);
lean_inc(x_16);
lean_dec_ref(x_15);
lean_inc_ref(x_7);
x_17 = l_Lean_Meta_mkFreshExprMVar(x_14, x_2, x_3, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_17) == 0)
{
lean_object* x_18; lean_object* x_19; uint8_t x_20; 
x_18 = lean_ctor_get(x_17, 0);
lean_inc(x_18);
lean_dec_ref(x_17);
x_19 = l_Lean_Meta_Context_config(x_7);
x_20 = !lean_is_exclusive(x_19);
if (x_20 == 0)
{
uint8_t x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; uint8_t x_28; uint8_t x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; uint8_t x_37; uint64_t x_38; uint8_t x_39; 
x_21 = lean_ctor_get_uint8(x_7, sizeof(void*)*7);
x_22 = lean_ctor_get(x_7, 1);
lean_inc(x_22);
x_23 = lean_ctor_get(x_7, 2);
lean_inc_ref(x_23);
x_24 = lean_ctor_get(x_7, 3);
lean_inc_ref(x_24);
x_25 = lean_ctor_get(x_7, 4);
lean_inc(x_25);
x_26 = lean_ctor_get(x_7, 5);
lean_inc(x_26);
x_27 = lean_ctor_get(x_7, 6);
lean_inc(x_27);
x_28 = lean_ctor_get_uint8(x_7, sizeof(void*)*7 + 1);
x_29 = lean_ctor_get_uint8(x_7, sizeof(void*)*7 + 2);
x_30 = lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__4;
x_31 = lean_box(0);
x_32 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_32, 0, x_4);
lean_ctor_set(x_32, 1, x_31);
x_33 = l_Lean_Expr_const___override(x_30, x_32);
lean_inc(x_13);
x_34 = l_Lean_Expr_app___override(x_33, x_13);
lean_inc(x_16);
x_35 = l_Lean_Expr_app___override(x_34, x_16);
lean_inc(x_18);
x_36 = l_Lean_Expr_app___override(x_35, x_18);
x_37 = 2;
lean_ctor_set_uint8(x_19, 9, x_37);
x_38 = l_Lean_Meta_Context_configKey(x_7);
x_39 = !lean_is_exclusive(x_7);
if (x_39 == 0)
{
lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; uint64_t x_47; uint64_t x_48; uint64_t x_49; uint64_t x_50; uint64_t x_51; lean_object* x_52; lean_object* x_53; 
x_40 = lean_ctor_get(x_7, 6);
lean_dec(x_40);
x_41 = lean_ctor_get(x_7, 5);
lean_dec(x_41);
x_42 = lean_ctor_get(x_7, 4);
lean_dec(x_42);
x_43 = lean_ctor_get(x_7, 3);
lean_dec(x_43);
x_44 = lean_ctor_get(x_7, 2);
lean_dec(x_44);
x_45 = lean_ctor_get(x_7, 1);
lean_dec(x_45);
x_46 = lean_ctor_get(x_7, 0);
lean_dec(x_46);
x_47 = 2;
x_48 = lean_uint64_shift_right(x_38, x_47);
x_49 = lean_uint64_shift_left(x_48, x_47);
x_50 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_51 = lean_uint64_lor(x_49, x_50);
x_52 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_52, 0, x_19);
lean_ctor_set_uint64(x_52, sizeof(void*)*1, x_51);
lean_ctor_set(x_7, 0, x_52);
lean_inc(x_8);
x_53 = l_Lean_Meta_isExprDefEq(x_36, x_5, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_53) == 0)
{
uint8_t x_54; 
x_54 = !lean_is_exclusive(x_53);
if (x_54 == 0)
{
lean_object* x_55; uint8_t x_56; 
x_55 = lean_ctor_get(x_53, 0);
x_56 = lean_unbox(x_55);
if (x_56 == 0)
{
lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; 
lean_dec(x_55);
lean_dec(x_8);
x_57 = lean_box(x_6);
x_58 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_58, 0, x_18);
lean_ctor_set(x_58, 1, x_57);
x_59 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_59, 0, x_16);
lean_ctor_set(x_59, 1, x_58);
x_60 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_60, 0, x_13);
lean_ctor_set(x_60, 1, x_59);
lean_ctor_set(x_53, 0, x_60);
return x_53;
}
else
{
lean_object* x_61; 
lean_free_object(x_53);
x_61 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_13, x_8);
if (lean_obj_tag(x_61) == 0)
{
lean_object* x_62; lean_object* x_63; 
x_62 = lean_ctor_get(x_61, 0);
lean_inc(x_62);
lean_dec_ref(x_61);
x_63 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_16, x_8);
if (lean_obj_tag(x_63) == 0)
{
lean_object* x_64; lean_object* x_65; 
x_64 = lean_ctor_get(x_63, 0);
lean_inc(x_64);
lean_dec_ref(x_63);
x_65 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_18, x_8);
lean_dec(x_8);
if (lean_obj_tag(x_65) == 0)
{
uint8_t x_66; 
x_66 = !lean_is_exclusive(x_65);
if (x_66 == 0)
{
lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; 
x_67 = lean_ctor_get(x_65, 0);
x_68 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_68, 0, x_67);
lean_ctor_set(x_68, 1, x_55);
x_69 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_69, 0, x_64);
lean_ctor_set(x_69, 1, x_68);
x_70 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_70, 0, x_62);
lean_ctor_set(x_70, 1, x_69);
lean_ctor_set(x_65, 0, x_70);
return x_65;
}
else
{
lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; 
x_71 = lean_ctor_get(x_65, 0);
lean_inc(x_71);
lean_dec(x_65);
x_72 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_72, 0, x_71);
lean_ctor_set(x_72, 1, x_55);
x_73 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_73, 0, x_64);
lean_ctor_set(x_73, 1, x_72);
x_74 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_74, 0, x_62);
lean_ctor_set(x_74, 1, x_73);
x_75 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_75, 0, x_74);
return x_75;
}
}
else
{
uint8_t x_76; 
lean_dec(x_64);
lean_dec(x_62);
lean_dec(x_55);
x_76 = !lean_is_exclusive(x_65);
if (x_76 == 0)
{
return x_65;
}
else
{
lean_object* x_77; lean_object* x_78; 
x_77 = lean_ctor_get(x_65, 0);
lean_inc(x_77);
lean_dec(x_65);
x_78 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_78, 0, x_77);
return x_78;
}
}
}
else
{
uint8_t x_79; 
lean_dec(x_62);
lean_dec(x_55);
lean_dec(x_18);
lean_dec(x_8);
x_79 = !lean_is_exclusive(x_63);
if (x_79 == 0)
{
return x_63;
}
else
{
lean_object* x_80; lean_object* x_81; 
x_80 = lean_ctor_get(x_63, 0);
lean_inc(x_80);
lean_dec(x_63);
x_81 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_81, 0, x_80);
return x_81;
}
}
}
else
{
uint8_t x_82; 
lean_dec(x_55);
lean_dec(x_18);
lean_dec(x_16);
lean_dec(x_8);
x_82 = !lean_is_exclusive(x_61);
if (x_82 == 0)
{
return x_61;
}
else
{
lean_object* x_83; lean_object* x_84; 
x_83 = lean_ctor_get(x_61, 0);
lean_inc(x_83);
lean_dec(x_61);
x_84 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_84, 0, x_83);
return x_84;
}
}
}
}
else
{
lean_object* x_85; uint8_t x_86; 
x_85 = lean_ctor_get(x_53, 0);
lean_inc(x_85);
lean_dec(x_53);
x_86 = lean_unbox(x_85);
if (x_86 == 0)
{
lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; 
lean_dec(x_85);
lean_dec(x_8);
x_87 = lean_box(x_6);
x_88 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_88, 0, x_18);
lean_ctor_set(x_88, 1, x_87);
x_89 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_89, 0, x_16);
lean_ctor_set(x_89, 1, x_88);
x_90 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_90, 0, x_13);
lean_ctor_set(x_90, 1, x_89);
x_91 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_91, 0, x_90);
return x_91;
}
else
{
lean_object* x_92; 
x_92 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_13, x_8);
if (lean_obj_tag(x_92) == 0)
{
lean_object* x_93; lean_object* x_94; 
x_93 = lean_ctor_get(x_92, 0);
lean_inc(x_93);
lean_dec_ref(x_92);
x_94 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_16, x_8);
if (lean_obj_tag(x_94) == 0)
{
lean_object* x_95; lean_object* x_96; 
x_95 = lean_ctor_get(x_94, 0);
lean_inc(x_95);
lean_dec_ref(x_94);
x_96 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_18, x_8);
lean_dec(x_8);
if (lean_obj_tag(x_96) == 0)
{
lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; 
x_97 = lean_ctor_get(x_96, 0);
lean_inc(x_97);
if (lean_is_exclusive(x_96)) {
 lean_ctor_release(x_96, 0);
 x_98 = x_96;
} else {
 lean_dec_ref(x_96);
 x_98 = lean_box(0);
}
x_99 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_99, 0, x_97);
lean_ctor_set(x_99, 1, x_85);
x_100 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_100, 0, x_95);
lean_ctor_set(x_100, 1, x_99);
x_101 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_101, 0, x_93);
lean_ctor_set(x_101, 1, x_100);
if (lean_is_scalar(x_98)) {
 x_102 = lean_alloc_ctor(0, 1, 0);
} else {
 x_102 = x_98;
}
lean_ctor_set(x_102, 0, x_101);
return x_102;
}
else
{
lean_object* x_103; lean_object* x_104; lean_object* x_105; 
lean_dec(x_95);
lean_dec(x_93);
lean_dec(x_85);
x_103 = lean_ctor_get(x_96, 0);
lean_inc(x_103);
if (lean_is_exclusive(x_96)) {
 lean_ctor_release(x_96, 0);
 x_104 = x_96;
} else {
 lean_dec_ref(x_96);
 x_104 = lean_box(0);
}
if (lean_is_scalar(x_104)) {
 x_105 = lean_alloc_ctor(1, 1, 0);
} else {
 x_105 = x_104;
}
lean_ctor_set(x_105, 0, x_103);
return x_105;
}
}
else
{
lean_object* x_106; lean_object* x_107; lean_object* x_108; 
lean_dec(x_93);
lean_dec(x_85);
lean_dec(x_18);
lean_dec(x_8);
x_106 = lean_ctor_get(x_94, 0);
lean_inc(x_106);
if (lean_is_exclusive(x_94)) {
 lean_ctor_release(x_94, 0);
 x_107 = x_94;
} else {
 lean_dec_ref(x_94);
 x_107 = lean_box(0);
}
if (lean_is_scalar(x_107)) {
 x_108 = lean_alloc_ctor(1, 1, 0);
} else {
 x_108 = x_107;
}
lean_ctor_set(x_108, 0, x_106);
return x_108;
}
}
else
{
lean_object* x_109; lean_object* x_110; lean_object* x_111; 
lean_dec(x_85);
lean_dec(x_18);
lean_dec(x_16);
lean_dec(x_8);
x_109 = lean_ctor_get(x_92, 0);
lean_inc(x_109);
if (lean_is_exclusive(x_92)) {
 lean_ctor_release(x_92, 0);
 x_110 = x_92;
} else {
 lean_dec_ref(x_92);
 x_110 = lean_box(0);
}
if (lean_is_scalar(x_110)) {
 x_111 = lean_alloc_ctor(1, 1, 0);
} else {
 x_111 = x_110;
}
lean_ctor_set(x_111, 0, x_109);
return x_111;
}
}
}
}
else
{
uint8_t x_112; 
lean_dec(x_18);
lean_dec(x_16);
lean_dec(x_13);
lean_dec(x_8);
x_112 = !lean_is_exclusive(x_53);
if (x_112 == 0)
{
return x_53;
}
else
{
lean_object* x_113; lean_object* x_114; 
x_113 = lean_ctor_get(x_53, 0);
lean_inc(x_113);
lean_dec(x_53);
x_114 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_114, 0, x_113);
return x_114;
}
}
}
else
{
uint64_t x_115; uint64_t x_116; uint64_t x_117; uint64_t x_118; uint64_t x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; 
lean_dec(x_7);
x_115 = 2;
x_116 = lean_uint64_shift_right(x_38, x_115);
x_117 = lean_uint64_shift_left(x_116, x_115);
x_118 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_119 = lean_uint64_lor(x_117, x_118);
x_120 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_120, 0, x_19);
lean_ctor_set_uint64(x_120, sizeof(void*)*1, x_119);
x_121 = lean_alloc_ctor(0, 7, 3);
lean_ctor_set(x_121, 0, x_120);
lean_ctor_set(x_121, 1, x_22);
lean_ctor_set(x_121, 2, x_23);
lean_ctor_set(x_121, 3, x_24);
lean_ctor_set(x_121, 4, x_25);
lean_ctor_set(x_121, 5, x_26);
lean_ctor_set(x_121, 6, x_27);
lean_ctor_set_uint8(x_121, sizeof(void*)*7, x_21);
lean_ctor_set_uint8(x_121, sizeof(void*)*7 + 1, x_28);
lean_ctor_set_uint8(x_121, sizeof(void*)*7 + 2, x_29);
lean_inc(x_8);
x_122 = l_Lean_Meta_isExprDefEq(x_36, x_5, x_121, x_8, x_9, x_10);
if (lean_obj_tag(x_122) == 0)
{
lean_object* x_123; lean_object* x_124; uint8_t x_125; 
x_123 = lean_ctor_get(x_122, 0);
lean_inc(x_123);
if (lean_is_exclusive(x_122)) {
 lean_ctor_release(x_122, 0);
 x_124 = x_122;
} else {
 lean_dec_ref(x_122);
 x_124 = lean_box(0);
}
x_125 = lean_unbox(x_123);
if (x_125 == 0)
{
lean_object* x_126; lean_object* x_127; lean_object* x_128; lean_object* x_129; lean_object* x_130; 
lean_dec(x_123);
lean_dec(x_8);
x_126 = lean_box(x_6);
x_127 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_127, 0, x_18);
lean_ctor_set(x_127, 1, x_126);
x_128 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_128, 0, x_16);
lean_ctor_set(x_128, 1, x_127);
x_129 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_129, 0, x_13);
lean_ctor_set(x_129, 1, x_128);
if (lean_is_scalar(x_124)) {
 x_130 = lean_alloc_ctor(0, 1, 0);
} else {
 x_130 = x_124;
}
lean_ctor_set(x_130, 0, x_129);
return x_130;
}
else
{
lean_object* x_131; 
lean_dec(x_124);
x_131 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_13, x_8);
if (lean_obj_tag(x_131) == 0)
{
lean_object* x_132; lean_object* x_133; 
x_132 = lean_ctor_get(x_131, 0);
lean_inc(x_132);
lean_dec_ref(x_131);
x_133 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_16, x_8);
if (lean_obj_tag(x_133) == 0)
{
lean_object* x_134; lean_object* x_135; 
x_134 = lean_ctor_get(x_133, 0);
lean_inc(x_134);
lean_dec_ref(x_133);
x_135 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_18, x_8);
lean_dec(x_8);
if (lean_obj_tag(x_135) == 0)
{
lean_object* x_136; lean_object* x_137; lean_object* x_138; lean_object* x_139; lean_object* x_140; lean_object* x_141; 
x_136 = lean_ctor_get(x_135, 0);
lean_inc(x_136);
if (lean_is_exclusive(x_135)) {
 lean_ctor_release(x_135, 0);
 x_137 = x_135;
} else {
 lean_dec_ref(x_135);
 x_137 = lean_box(0);
}
x_138 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_138, 0, x_136);
lean_ctor_set(x_138, 1, x_123);
x_139 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_139, 0, x_134);
lean_ctor_set(x_139, 1, x_138);
x_140 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_140, 0, x_132);
lean_ctor_set(x_140, 1, x_139);
if (lean_is_scalar(x_137)) {
 x_141 = lean_alloc_ctor(0, 1, 0);
} else {
 x_141 = x_137;
}
lean_ctor_set(x_141, 0, x_140);
return x_141;
}
else
{
lean_object* x_142; lean_object* x_143; lean_object* x_144; 
lean_dec(x_134);
lean_dec(x_132);
lean_dec(x_123);
x_142 = lean_ctor_get(x_135, 0);
lean_inc(x_142);
if (lean_is_exclusive(x_135)) {
 lean_ctor_release(x_135, 0);
 x_143 = x_135;
} else {
 lean_dec_ref(x_135);
 x_143 = lean_box(0);
}
if (lean_is_scalar(x_143)) {
 x_144 = lean_alloc_ctor(1, 1, 0);
} else {
 x_144 = x_143;
}
lean_ctor_set(x_144, 0, x_142);
return x_144;
}
}
else
{
lean_object* x_145; lean_object* x_146; lean_object* x_147; 
lean_dec(x_132);
lean_dec(x_123);
lean_dec(x_18);
lean_dec(x_8);
x_145 = lean_ctor_get(x_133, 0);
lean_inc(x_145);
if (lean_is_exclusive(x_133)) {
 lean_ctor_release(x_133, 0);
 x_146 = x_133;
} else {
 lean_dec_ref(x_133);
 x_146 = lean_box(0);
}
if (lean_is_scalar(x_146)) {
 x_147 = lean_alloc_ctor(1, 1, 0);
} else {
 x_147 = x_146;
}
lean_ctor_set(x_147, 0, x_145);
return x_147;
}
}
else
{
lean_object* x_148; lean_object* x_149; lean_object* x_150; 
lean_dec(x_123);
lean_dec(x_18);
lean_dec(x_16);
lean_dec(x_8);
x_148 = lean_ctor_get(x_131, 0);
lean_inc(x_148);
if (lean_is_exclusive(x_131)) {
 lean_ctor_release(x_131, 0);
 x_149 = x_131;
} else {
 lean_dec_ref(x_131);
 x_149 = lean_box(0);
}
if (lean_is_scalar(x_149)) {
 x_150 = lean_alloc_ctor(1, 1, 0);
} else {
 x_150 = x_149;
}
lean_ctor_set(x_150, 0, x_148);
return x_150;
}
}
}
else
{
lean_object* x_151; lean_object* x_152; lean_object* x_153; 
lean_dec(x_18);
lean_dec(x_16);
lean_dec(x_13);
lean_dec(x_8);
x_151 = lean_ctor_get(x_122, 0);
lean_inc(x_151);
if (lean_is_exclusive(x_122)) {
 lean_ctor_release(x_122, 0);
 x_152 = x_122;
} else {
 lean_dec_ref(x_122);
 x_152 = lean_box(0);
}
if (lean_is_scalar(x_152)) {
 x_153 = lean_alloc_ctor(1, 1, 0);
} else {
 x_153 = x_152;
}
lean_ctor_set(x_153, 0, x_151);
return x_153;
}
}
}
else
{
uint8_t x_154; uint8_t x_155; uint8_t x_156; uint8_t x_157; uint8_t x_158; uint8_t x_159; uint8_t x_160; uint8_t x_161; uint8_t x_162; uint8_t x_163; uint8_t x_164; uint8_t x_165; uint8_t x_166; uint8_t x_167; uint8_t x_168; uint8_t x_169; uint8_t x_170; uint8_t x_171; uint8_t x_172; lean_object* x_173; lean_object* x_174; lean_object* x_175; lean_object* x_176; lean_object* x_177; lean_object* x_178; uint8_t x_179; uint8_t x_180; lean_object* x_181; lean_object* x_182; lean_object* x_183; lean_object* x_184; lean_object* x_185; lean_object* x_186; lean_object* x_187; uint8_t x_188; lean_object* x_189; uint64_t x_190; lean_object* x_191; uint64_t x_192; uint64_t x_193; uint64_t x_194; uint64_t x_195; uint64_t x_196; lean_object* x_197; lean_object* x_198; lean_object* x_199; 
x_154 = lean_ctor_get_uint8(x_19, 0);
x_155 = lean_ctor_get_uint8(x_19, 1);
x_156 = lean_ctor_get_uint8(x_19, 2);
x_157 = lean_ctor_get_uint8(x_19, 3);
x_158 = lean_ctor_get_uint8(x_19, 4);
x_159 = lean_ctor_get_uint8(x_19, 5);
x_160 = lean_ctor_get_uint8(x_19, 6);
x_161 = lean_ctor_get_uint8(x_19, 7);
x_162 = lean_ctor_get_uint8(x_19, 8);
x_163 = lean_ctor_get_uint8(x_19, 10);
x_164 = lean_ctor_get_uint8(x_19, 11);
x_165 = lean_ctor_get_uint8(x_19, 12);
x_166 = lean_ctor_get_uint8(x_19, 13);
x_167 = lean_ctor_get_uint8(x_19, 14);
x_168 = lean_ctor_get_uint8(x_19, 15);
x_169 = lean_ctor_get_uint8(x_19, 16);
x_170 = lean_ctor_get_uint8(x_19, 17);
x_171 = lean_ctor_get_uint8(x_19, 18);
lean_dec(x_19);
x_172 = lean_ctor_get_uint8(x_7, sizeof(void*)*7);
x_173 = lean_ctor_get(x_7, 1);
lean_inc(x_173);
x_174 = lean_ctor_get(x_7, 2);
lean_inc_ref(x_174);
x_175 = lean_ctor_get(x_7, 3);
lean_inc_ref(x_175);
x_176 = lean_ctor_get(x_7, 4);
lean_inc(x_176);
x_177 = lean_ctor_get(x_7, 5);
lean_inc(x_177);
x_178 = lean_ctor_get(x_7, 6);
lean_inc(x_178);
x_179 = lean_ctor_get_uint8(x_7, sizeof(void*)*7 + 1);
x_180 = lean_ctor_get_uint8(x_7, sizeof(void*)*7 + 2);
x_181 = lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__4;
x_182 = lean_box(0);
x_183 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_183, 0, x_4);
lean_ctor_set(x_183, 1, x_182);
x_184 = l_Lean_Expr_const___override(x_181, x_183);
lean_inc(x_13);
x_185 = l_Lean_Expr_app___override(x_184, x_13);
lean_inc(x_16);
x_186 = l_Lean_Expr_app___override(x_185, x_16);
lean_inc(x_18);
x_187 = l_Lean_Expr_app___override(x_186, x_18);
x_188 = 2;
x_189 = lean_alloc_ctor(0, 0, 19);
lean_ctor_set_uint8(x_189, 0, x_154);
lean_ctor_set_uint8(x_189, 1, x_155);
lean_ctor_set_uint8(x_189, 2, x_156);
lean_ctor_set_uint8(x_189, 3, x_157);
lean_ctor_set_uint8(x_189, 4, x_158);
lean_ctor_set_uint8(x_189, 5, x_159);
lean_ctor_set_uint8(x_189, 6, x_160);
lean_ctor_set_uint8(x_189, 7, x_161);
lean_ctor_set_uint8(x_189, 8, x_162);
lean_ctor_set_uint8(x_189, 9, x_188);
lean_ctor_set_uint8(x_189, 10, x_163);
lean_ctor_set_uint8(x_189, 11, x_164);
lean_ctor_set_uint8(x_189, 12, x_165);
lean_ctor_set_uint8(x_189, 13, x_166);
lean_ctor_set_uint8(x_189, 14, x_167);
lean_ctor_set_uint8(x_189, 15, x_168);
lean_ctor_set_uint8(x_189, 16, x_169);
lean_ctor_set_uint8(x_189, 17, x_170);
lean_ctor_set_uint8(x_189, 18, x_171);
x_190 = l_Lean_Meta_Context_configKey(x_7);
if (lean_is_exclusive(x_7)) {
 lean_ctor_release(x_7, 0);
 lean_ctor_release(x_7, 1);
 lean_ctor_release(x_7, 2);
 lean_ctor_release(x_7, 3);
 lean_ctor_release(x_7, 4);
 lean_ctor_release(x_7, 5);
 lean_ctor_release(x_7, 6);
 x_191 = x_7;
} else {
 lean_dec_ref(x_7);
 x_191 = lean_box(0);
}
x_192 = 2;
x_193 = lean_uint64_shift_right(x_190, x_192);
x_194 = lean_uint64_shift_left(x_193, x_192);
x_195 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_196 = lean_uint64_lor(x_194, x_195);
x_197 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_197, 0, x_189);
lean_ctor_set_uint64(x_197, sizeof(void*)*1, x_196);
if (lean_is_scalar(x_191)) {
 x_198 = lean_alloc_ctor(0, 7, 3);
} else {
 x_198 = x_191;
}
lean_ctor_set(x_198, 0, x_197);
lean_ctor_set(x_198, 1, x_173);
lean_ctor_set(x_198, 2, x_174);
lean_ctor_set(x_198, 3, x_175);
lean_ctor_set(x_198, 4, x_176);
lean_ctor_set(x_198, 5, x_177);
lean_ctor_set(x_198, 6, x_178);
lean_ctor_set_uint8(x_198, sizeof(void*)*7, x_172);
lean_ctor_set_uint8(x_198, sizeof(void*)*7 + 1, x_179);
lean_ctor_set_uint8(x_198, sizeof(void*)*7 + 2, x_180);
lean_inc(x_8);
x_199 = l_Lean_Meta_isExprDefEq(x_187, x_5, x_198, x_8, x_9, x_10);
if (lean_obj_tag(x_199) == 0)
{
lean_object* x_200; lean_object* x_201; uint8_t x_202; 
x_200 = lean_ctor_get(x_199, 0);
lean_inc(x_200);
if (lean_is_exclusive(x_199)) {
 lean_ctor_release(x_199, 0);
 x_201 = x_199;
} else {
 lean_dec_ref(x_199);
 x_201 = lean_box(0);
}
x_202 = lean_unbox(x_200);
if (x_202 == 0)
{
lean_object* x_203; lean_object* x_204; lean_object* x_205; lean_object* x_206; lean_object* x_207; 
lean_dec(x_200);
lean_dec(x_8);
x_203 = lean_box(x_6);
x_204 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_204, 0, x_18);
lean_ctor_set(x_204, 1, x_203);
x_205 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_205, 0, x_16);
lean_ctor_set(x_205, 1, x_204);
x_206 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_206, 0, x_13);
lean_ctor_set(x_206, 1, x_205);
if (lean_is_scalar(x_201)) {
 x_207 = lean_alloc_ctor(0, 1, 0);
} else {
 x_207 = x_201;
}
lean_ctor_set(x_207, 0, x_206);
return x_207;
}
else
{
lean_object* x_208; 
lean_dec(x_201);
x_208 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_13, x_8);
if (lean_obj_tag(x_208) == 0)
{
lean_object* x_209; lean_object* x_210; 
x_209 = lean_ctor_get(x_208, 0);
lean_inc(x_209);
lean_dec_ref(x_208);
x_210 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_16, x_8);
if (lean_obj_tag(x_210) == 0)
{
lean_object* x_211; lean_object* x_212; 
x_211 = lean_ctor_get(x_210, 0);
lean_inc(x_211);
lean_dec_ref(x_210);
x_212 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_18, x_8);
lean_dec(x_8);
if (lean_obj_tag(x_212) == 0)
{
lean_object* x_213; lean_object* x_214; lean_object* x_215; lean_object* x_216; lean_object* x_217; lean_object* x_218; 
x_213 = lean_ctor_get(x_212, 0);
lean_inc(x_213);
if (lean_is_exclusive(x_212)) {
 lean_ctor_release(x_212, 0);
 x_214 = x_212;
} else {
 lean_dec_ref(x_212);
 x_214 = lean_box(0);
}
x_215 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_215, 0, x_213);
lean_ctor_set(x_215, 1, x_200);
x_216 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_216, 0, x_211);
lean_ctor_set(x_216, 1, x_215);
x_217 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_217, 0, x_209);
lean_ctor_set(x_217, 1, x_216);
if (lean_is_scalar(x_214)) {
 x_218 = lean_alloc_ctor(0, 1, 0);
} else {
 x_218 = x_214;
}
lean_ctor_set(x_218, 0, x_217);
return x_218;
}
else
{
lean_object* x_219; lean_object* x_220; lean_object* x_221; 
lean_dec(x_211);
lean_dec(x_209);
lean_dec(x_200);
x_219 = lean_ctor_get(x_212, 0);
lean_inc(x_219);
if (lean_is_exclusive(x_212)) {
 lean_ctor_release(x_212, 0);
 x_220 = x_212;
} else {
 lean_dec_ref(x_212);
 x_220 = lean_box(0);
}
if (lean_is_scalar(x_220)) {
 x_221 = lean_alloc_ctor(1, 1, 0);
} else {
 x_221 = x_220;
}
lean_ctor_set(x_221, 0, x_219);
return x_221;
}
}
else
{
lean_object* x_222; lean_object* x_223; lean_object* x_224; 
lean_dec(x_209);
lean_dec(x_200);
lean_dec(x_18);
lean_dec(x_8);
x_222 = lean_ctor_get(x_210, 0);
lean_inc(x_222);
if (lean_is_exclusive(x_210)) {
 lean_ctor_release(x_210, 0);
 x_223 = x_210;
} else {
 lean_dec_ref(x_210);
 x_223 = lean_box(0);
}
if (lean_is_scalar(x_223)) {
 x_224 = lean_alloc_ctor(1, 1, 0);
} else {
 x_224 = x_223;
}
lean_ctor_set(x_224, 0, x_222);
return x_224;
}
}
else
{
lean_object* x_225; lean_object* x_226; lean_object* x_227; 
lean_dec(x_200);
lean_dec(x_18);
lean_dec(x_16);
lean_dec(x_8);
x_225 = lean_ctor_get(x_208, 0);
lean_inc(x_225);
if (lean_is_exclusive(x_208)) {
 lean_ctor_release(x_208, 0);
 x_226 = x_208;
} else {
 lean_dec_ref(x_208);
 x_226 = lean_box(0);
}
if (lean_is_scalar(x_226)) {
 x_227 = lean_alloc_ctor(1, 1, 0);
} else {
 x_227 = x_226;
}
lean_ctor_set(x_227, 0, x_225);
return x_227;
}
}
}
else
{
lean_object* x_228; lean_object* x_229; lean_object* x_230; 
lean_dec(x_18);
lean_dec(x_16);
lean_dec(x_13);
lean_dec(x_8);
x_228 = lean_ctor_get(x_199, 0);
lean_inc(x_228);
if (lean_is_exclusive(x_199)) {
 lean_ctor_release(x_199, 0);
 x_229 = x_199;
} else {
 lean_dec_ref(x_199);
 x_229 = lean_box(0);
}
if (lean_is_scalar(x_229)) {
 x_230 = lean_alloc_ctor(1, 1, 0);
} else {
 x_230 = x_229;
}
lean_ctor_set(x_230, 0, x_228);
return x_230;
}
}
}
else
{
uint8_t x_231; 
lean_dec(x_16);
lean_dec(x_13);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_5);
lean_dec(x_4);
x_231 = !lean_is_exclusive(x_17);
if (x_231 == 0)
{
return x_17;
}
else
{
lean_object* x_232; lean_object* x_233; 
x_232 = lean_ctor_get(x_17, 0);
lean_inc(x_232);
lean_dec(x_17);
x_233 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_233, 0, x_232);
return x_233;
}
}
}
else
{
uint8_t x_234; 
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec(x_3);
x_234 = !lean_is_exclusive(x_15);
if (x_234 == 0)
{
return x_15;
}
else
{
lean_object* x_235; lean_object* x_236; 
x_235 = lean_ctor_get(x_15, 0);
lean_inc(x_235);
lean_dec(x_15);
x_236 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_236, 0, x_235);
return x_236;
}
}
}
else
{
uint8_t x_237; 
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec(x_3);
x_237 = !lean_is_exclusive(x_12);
if (x_237 == 0)
{
return x_12;
}
else
{
lean_object* x_238; lean_object* x_239; 
x_238 = lean_ctor_get(x_12, 0);
lean_inc(x_238);
lean_dec(x_12);
x_239 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_239, 0, x_238);
return x_239;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
uint8_t x_12; uint8_t x_13; lean_object* x_14; 
x_12 = lean_unbox(x_2);
x_13 = lean_unbox(x_6);
x_14 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__0(x_1, x_12, x_3, x_4, x_5, x_13, x_7, x_8, x_9, x_10);
return x_14;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("findEq: some side of equality must be `a`, and the other must not depend on `a`", 79, 79);
return x_1;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_box(0);
x_2 = l_Lean_Expr_sort___override(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0;
x_2 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__2;
x_3 = l_Lean_Expr_const___override(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, uint8_t x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; 
lean_inc_ref(x_6);
lean_inc(x_3);
lean_inc(x_1);
x_11 = l_Lean_Meta_mkFreshExprMVar(x_1, x_2, x_3, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_11) == 0)
{
lean_object* x_12; lean_object* x_13; 
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
lean_dec_ref(x_11);
lean_inc_ref(x_6);
x_13 = l_Lean_Meta_mkFreshExprMVar(x_1, x_2, x_3, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_13) == 0)
{
lean_object* x_14; lean_object* x_15; uint8_t x_16; 
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
lean_dec_ref(x_13);
x_15 = l_Lean_Meta_Context_config(x_6);
x_16 = !lean_is_exclusive(x_15);
if (x_16 == 0)
{
uint8_t x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; uint8_t x_24; uint8_t x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; uint8_t x_29; uint64_t x_30; uint8_t x_31; 
x_17 = lean_ctor_get_uint8(x_6, sizeof(void*)*7);
x_18 = lean_ctor_get(x_6, 1);
lean_inc(x_18);
x_19 = lean_ctor_get(x_6, 2);
lean_inc_ref(x_19);
x_20 = lean_ctor_get(x_6, 3);
lean_inc_ref(x_20);
x_21 = lean_ctor_get(x_6, 4);
lean_inc(x_21);
x_22 = lean_ctor_get(x_6, 5);
lean_inc(x_22);
x_23 = lean_ctor_get(x_6, 6);
lean_inc(x_23);
x_24 = lean_ctor_get_uint8(x_6, sizeof(void*)*7 + 1);
x_25 = lean_ctor_get_uint8(x_6, sizeof(void*)*7 + 2);
x_26 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0;
lean_inc(x_12);
x_27 = l_Lean_Expr_app___override(x_26, x_12);
lean_inc(x_14);
x_28 = l_Lean_Expr_app___override(x_27, x_14);
x_29 = 2;
lean_ctor_set_uint8(x_15, 9, x_29);
x_30 = l_Lean_Meta_Context_configKey(x_6);
x_31 = !lean_is_exclusive(x_6);
if (x_31 == 0)
{
lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; uint64_t x_39; uint64_t x_40; uint64_t x_41; uint64_t x_42; uint64_t x_43; lean_object* x_44; lean_object* x_45; 
x_32 = lean_ctor_get(x_6, 6);
lean_dec(x_32);
x_33 = lean_ctor_get(x_6, 5);
lean_dec(x_33);
x_34 = lean_ctor_get(x_6, 4);
lean_dec(x_34);
x_35 = lean_ctor_get(x_6, 3);
lean_dec(x_35);
x_36 = lean_ctor_get(x_6, 2);
lean_dec(x_36);
x_37 = lean_ctor_get(x_6, 1);
lean_dec(x_37);
x_38 = lean_ctor_get(x_6, 0);
lean_dec(x_38);
x_39 = 2;
x_40 = lean_uint64_shift_right(x_30, x_39);
x_41 = lean_uint64_shift_left(x_40, x_39);
x_42 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_43 = lean_uint64_lor(x_41, x_42);
x_44 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_44, 0, x_15);
lean_ctor_set_uint64(x_44, sizeof(void*)*1, x_43);
lean_ctor_set(x_6, 0, x_44);
lean_inc(x_7);
x_45 = l_Lean_Meta_isExprDefEq(x_28, x_4, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_45) == 0)
{
uint8_t x_46; 
x_46 = !lean_is_exclusive(x_45);
if (x_46 == 0)
{
lean_object* x_47; uint8_t x_48; 
x_47 = lean_ctor_get(x_45, 0);
x_48 = lean_unbox(x_47);
if (x_48 == 0)
{
lean_object* x_49; lean_object* x_50; lean_object* x_51; 
lean_dec(x_47);
lean_dec(x_7);
x_49 = lean_box(x_5);
x_50 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_50, 0, x_14);
lean_ctor_set(x_50, 1, x_49);
x_51 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_51, 0, x_12);
lean_ctor_set(x_51, 1, x_50);
lean_ctor_set(x_45, 0, x_51);
return x_45;
}
else
{
lean_object* x_52; 
lean_free_object(x_45);
x_52 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_12, x_7);
if (lean_obj_tag(x_52) == 0)
{
lean_object* x_53; lean_object* x_54; 
x_53 = lean_ctor_get(x_52, 0);
lean_inc(x_53);
lean_dec_ref(x_52);
x_54 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_14, x_7);
lean_dec(x_7);
if (lean_obj_tag(x_54) == 0)
{
uint8_t x_55; 
x_55 = !lean_is_exclusive(x_54);
if (x_55 == 0)
{
lean_object* x_56; lean_object* x_57; lean_object* x_58; 
x_56 = lean_ctor_get(x_54, 0);
x_57 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_57, 0, x_56);
lean_ctor_set(x_57, 1, x_47);
x_58 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_58, 0, x_53);
lean_ctor_set(x_58, 1, x_57);
lean_ctor_set(x_54, 0, x_58);
return x_54;
}
else
{
lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; 
x_59 = lean_ctor_get(x_54, 0);
lean_inc(x_59);
lean_dec(x_54);
x_60 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_60, 0, x_59);
lean_ctor_set(x_60, 1, x_47);
x_61 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_61, 0, x_53);
lean_ctor_set(x_61, 1, x_60);
x_62 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_62, 0, x_61);
return x_62;
}
}
else
{
uint8_t x_63; 
lean_dec(x_53);
lean_dec(x_47);
x_63 = !lean_is_exclusive(x_54);
if (x_63 == 0)
{
return x_54;
}
else
{
lean_object* x_64; lean_object* x_65; 
x_64 = lean_ctor_get(x_54, 0);
lean_inc(x_64);
lean_dec(x_54);
x_65 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_65, 0, x_64);
return x_65;
}
}
}
else
{
uint8_t x_66; 
lean_dec(x_47);
lean_dec(x_14);
lean_dec(x_7);
x_66 = !lean_is_exclusive(x_52);
if (x_66 == 0)
{
return x_52;
}
else
{
lean_object* x_67; lean_object* x_68; 
x_67 = lean_ctor_get(x_52, 0);
lean_inc(x_67);
lean_dec(x_52);
x_68 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_68, 0, x_67);
return x_68;
}
}
}
}
else
{
lean_object* x_69; uint8_t x_70; 
x_69 = lean_ctor_get(x_45, 0);
lean_inc(x_69);
lean_dec(x_45);
x_70 = lean_unbox(x_69);
if (x_70 == 0)
{
lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; 
lean_dec(x_69);
lean_dec(x_7);
x_71 = lean_box(x_5);
x_72 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_72, 0, x_14);
lean_ctor_set(x_72, 1, x_71);
x_73 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_73, 0, x_12);
lean_ctor_set(x_73, 1, x_72);
x_74 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_74, 0, x_73);
return x_74;
}
else
{
lean_object* x_75; 
x_75 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_12, x_7);
if (lean_obj_tag(x_75) == 0)
{
lean_object* x_76; lean_object* x_77; 
x_76 = lean_ctor_get(x_75, 0);
lean_inc(x_76);
lean_dec_ref(x_75);
x_77 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_14, x_7);
lean_dec(x_7);
if (lean_obj_tag(x_77) == 0)
{
lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; 
x_78 = lean_ctor_get(x_77, 0);
lean_inc(x_78);
if (lean_is_exclusive(x_77)) {
 lean_ctor_release(x_77, 0);
 x_79 = x_77;
} else {
 lean_dec_ref(x_77);
 x_79 = lean_box(0);
}
x_80 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_80, 0, x_78);
lean_ctor_set(x_80, 1, x_69);
x_81 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_81, 0, x_76);
lean_ctor_set(x_81, 1, x_80);
if (lean_is_scalar(x_79)) {
 x_82 = lean_alloc_ctor(0, 1, 0);
} else {
 x_82 = x_79;
}
lean_ctor_set(x_82, 0, x_81);
return x_82;
}
else
{
lean_object* x_83; lean_object* x_84; lean_object* x_85; 
lean_dec(x_76);
lean_dec(x_69);
x_83 = lean_ctor_get(x_77, 0);
lean_inc(x_83);
if (lean_is_exclusive(x_77)) {
 lean_ctor_release(x_77, 0);
 x_84 = x_77;
} else {
 lean_dec_ref(x_77);
 x_84 = lean_box(0);
}
if (lean_is_scalar(x_84)) {
 x_85 = lean_alloc_ctor(1, 1, 0);
} else {
 x_85 = x_84;
}
lean_ctor_set(x_85, 0, x_83);
return x_85;
}
}
else
{
lean_object* x_86; lean_object* x_87; lean_object* x_88; 
lean_dec(x_69);
lean_dec(x_14);
lean_dec(x_7);
x_86 = lean_ctor_get(x_75, 0);
lean_inc(x_86);
if (lean_is_exclusive(x_75)) {
 lean_ctor_release(x_75, 0);
 x_87 = x_75;
} else {
 lean_dec_ref(x_75);
 x_87 = lean_box(0);
}
if (lean_is_scalar(x_87)) {
 x_88 = lean_alloc_ctor(1, 1, 0);
} else {
 x_88 = x_87;
}
lean_ctor_set(x_88, 0, x_86);
return x_88;
}
}
}
}
else
{
uint8_t x_89; 
lean_dec(x_14);
lean_dec(x_12);
lean_dec(x_7);
x_89 = !lean_is_exclusive(x_45);
if (x_89 == 0)
{
return x_45;
}
else
{
lean_object* x_90; lean_object* x_91; 
x_90 = lean_ctor_get(x_45, 0);
lean_inc(x_90);
lean_dec(x_45);
x_91 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_91, 0, x_90);
return x_91;
}
}
}
else
{
uint64_t x_92; uint64_t x_93; uint64_t x_94; uint64_t x_95; uint64_t x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; 
lean_dec(x_6);
x_92 = 2;
x_93 = lean_uint64_shift_right(x_30, x_92);
x_94 = lean_uint64_shift_left(x_93, x_92);
x_95 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_96 = lean_uint64_lor(x_94, x_95);
x_97 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_97, 0, x_15);
lean_ctor_set_uint64(x_97, sizeof(void*)*1, x_96);
x_98 = lean_alloc_ctor(0, 7, 3);
lean_ctor_set(x_98, 0, x_97);
lean_ctor_set(x_98, 1, x_18);
lean_ctor_set(x_98, 2, x_19);
lean_ctor_set(x_98, 3, x_20);
lean_ctor_set(x_98, 4, x_21);
lean_ctor_set(x_98, 5, x_22);
lean_ctor_set(x_98, 6, x_23);
lean_ctor_set_uint8(x_98, sizeof(void*)*7, x_17);
lean_ctor_set_uint8(x_98, sizeof(void*)*7 + 1, x_24);
lean_ctor_set_uint8(x_98, sizeof(void*)*7 + 2, x_25);
lean_inc(x_7);
x_99 = l_Lean_Meta_isExprDefEq(x_28, x_4, x_98, x_7, x_8, x_9);
if (lean_obj_tag(x_99) == 0)
{
lean_object* x_100; lean_object* x_101; uint8_t x_102; 
x_100 = lean_ctor_get(x_99, 0);
lean_inc(x_100);
if (lean_is_exclusive(x_99)) {
 lean_ctor_release(x_99, 0);
 x_101 = x_99;
} else {
 lean_dec_ref(x_99);
 x_101 = lean_box(0);
}
x_102 = lean_unbox(x_100);
if (x_102 == 0)
{
lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; 
lean_dec(x_100);
lean_dec(x_7);
x_103 = lean_box(x_5);
x_104 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_104, 0, x_14);
lean_ctor_set(x_104, 1, x_103);
x_105 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_105, 0, x_12);
lean_ctor_set(x_105, 1, x_104);
if (lean_is_scalar(x_101)) {
 x_106 = lean_alloc_ctor(0, 1, 0);
} else {
 x_106 = x_101;
}
lean_ctor_set(x_106, 0, x_105);
return x_106;
}
else
{
lean_object* x_107; 
lean_dec(x_101);
x_107 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_12, x_7);
if (lean_obj_tag(x_107) == 0)
{
lean_object* x_108; lean_object* x_109; 
x_108 = lean_ctor_get(x_107, 0);
lean_inc(x_108);
lean_dec_ref(x_107);
x_109 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_14, x_7);
lean_dec(x_7);
if (lean_obj_tag(x_109) == 0)
{
lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; 
x_110 = lean_ctor_get(x_109, 0);
lean_inc(x_110);
if (lean_is_exclusive(x_109)) {
 lean_ctor_release(x_109, 0);
 x_111 = x_109;
} else {
 lean_dec_ref(x_109);
 x_111 = lean_box(0);
}
x_112 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_112, 0, x_110);
lean_ctor_set(x_112, 1, x_100);
x_113 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_113, 0, x_108);
lean_ctor_set(x_113, 1, x_112);
if (lean_is_scalar(x_111)) {
 x_114 = lean_alloc_ctor(0, 1, 0);
} else {
 x_114 = x_111;
}
lean_ctor_set(x_114, 0, x_113);
return x_114;
}
else
{
lean_object* x_115; lean_object* x_116; lean_object* x_117; 
lean_dec(x_108);
lean_dec(x_100);
x_115 = lean_ctor_get(x_109, 0);
lean_inc(x_115);
if (lean_is_exclusive(x_109)) {
 lean_ctor_release(x_109, 0);
 x_116 = x_109;
} else {
 lean_dec_ref(x_109);
 x_116 = lean_box(0);
}
if (lean_is_scalar(x_116)) {
 x_117 = lean_alloc_ctor(1, 1, 0);
} else {
 x_117 = x_116;
}
lean_ctor_set(x_117, 0, x_115);
return x_117;
}
}
else
{
lean_object* x_118; lean_object* x_119; lean_object* x_120; 
lean_dec(x_100);
lean_dec(x_14);
lean_dec(x_7);
x_118 = lean_ctor_get(x_107, 0);
lean_inc(x_118);
if (lean_is_exclusive(x_107)) {
 lean_ctor_release(x_107, 0);
 x_119 = x_107;
} else {
 lean_dec_ref(x_107);
 x_119 = lean_box(0);
}
if (lean_is_scalar(x_119)) {
 x_120 = lean_alloc_ctor(1, 1, 0);
} else {
 x_120 = x_119;
}
lean_ctor_set(x_120, 0, x_118);
return x_120;
}
}
}
else
{
lean_object* x_121; lean_object* x_122; lean_object* x_123; 
lean_dec(x_14);
lean_dec(x_12);
lean_dec(x_7);
x_121 = lean_ctor_get(x_99, 0);
lean_inc(x_121);
if (lean_is_exclusive(x_99)) {
 lean_ctor_release(x_99, 0);
 x_122 = x_99;
} else {
 lean_dec_ref(x_99);
 x_122 = lean_box(0);
}
if (lean_is_scalar(x_122)) {
 x_123 = lean_alloc_ctor(1, 1, 0);
} else {
 x_123 = x_122;
}
lean_ctor_set(x_123, 0, x_121);
return x_123;
}
}
}
else
{
uint8_t x_124; uint8_t x_125; uint8_t x_126; uint8_t x_127; uint8_t x_128; uint8_t x_129; uint8_t x_130; uint8_t x_131; uint8_t x_132; uint8_t x_133; uint8_t x_134; uint8_t x_135; uint8_t x_136; uint8_t x_137; uint8_t x_138; uint8_t x_139; uint8_t x_140; uint8_t x_141; uint8_t x_142; lean_object* x_143; lean_object* x_144; lean_object* x_145; lean_object* x_146; lean_object* x_147; lean_object* x_148; uint8_t x_149; uint8_t x_150; lean_object* x_151; lean_object* x_152; lean_object* x_153; uint8_t x_154; lean_object* x_155; uint64_t x_156; lean_object* x_157; uint64_t x_158; uint64_t x_159; uint64_t x_160; uint64_t x_161; uint64_t x_162; lean_object* x_163; lean_object* x_164; lean_object* x_165; 
x_124 = lean_ctor_get_uint8(x_15, 0);
x_125 = lean_ctor_get_uint8(x_15, 1);
x_126 = lean_ctor_get_uint8(x_15, 2);
x_127 = lean_ctor_get_uint8(x_15, 3);
x_128 = lean_ctor_get_uint8(x_15, 4);
x_129 = lean_ctor_get_uint8(x_15, 5);
x_130 = lean_ctor_get_uint8(x_15, 6);
x_131 = lean_ctor_get_uint8(x_15, 7);
x_132 = lean_ctor_get_uint8(x_15, 8);
x_133 = lean_ctor_get_uint8(x_15, 10);
x_134 = lean_ctor_get_uint8(x_15, 11);
x_135 = lean_ctor_get_uint8(x_15, 12);
x_136 = lean_ctor_get_uint8(x_15, 13);
x_137 = lean_ctor_get_uint8(x_15, 14);
x_138 = lean_ctor_get_uint8(x_15, 15);
x_139 = lean_ctor_get_uint8(x_15, 16);
x_140 = lean_ctor_get_uint8(x_15, 17);
x_141 = lean_ctor_get_uint8(x_15, 18);
lean_dec(x_15);
x_142 = lean_ctor_get_uint8(x_6, sizeof(void*)*7);
x_143 = lean_ctor_get(x_6, 1);
lean_inc(x_143);
x_144 = lean_ctor_get(x_6, 2);
lean_inc_ref(x_144);
x_145 = lean_ctor_get(x_6, 3);
lean_inc_ref(x_145);
x_146 = lean_ctor_get(x_6, 4);
lean_inc(x_146);
x_147 = lean_ctor_get(x_6, 5);
lean_inc(x_147);
x_148 = lean_ctor_get(x_6, 6);
lean_inc(x_148);
x_149 = lean_ctor_get_uint8(x_6, sizeof(void*)*7 + 1);
x_150 = lean_ctor_get_uint8(x_6, sizeof(void*)*7 + 2);
x_151 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0;
lean_inc(x_12);
x_152 = l_Lean_Expr_app___override(x_151, x_12);
lean_inc(x_14);
x_153 = l_Lean_Expr_app___override(x_152, x_14);
x_154 = 2;
x_155 = lean_alloc_ctor(0, 0, 19);
lean_ctor_set_uint8(x_155, 0, x_124);
lean_ctor_set_uint8(x_155, 1, x_125);
lean_ctor_set_uint8(x_155, 2, x_126);
lean_ctor_set_uint8(x_155, 3, x_127);
lean_ctor_set_uint8(x_155, 4, x_128);
lean_ctor_set_uint8(x_155, 5, x_129);
lean_ctor_set_uint8(x_155, 6, x_130);
lean_ctor_set_uint8(x_155, 7, x_131);
lean_ctor_set_uint8(x_155, 8, x_132);
lean_ctor_set_uint8(x_155, 9, x_154);
lean_ctor_set_uint8(x_155, 10, x_133);
lean_ctor_set_uint8(x_155, 11, x_134);
lean_ctor_set_uint8(x_155, 12, x_135);
lean_ctor_set_uint8(x_155, 13, x_136);
lean_ctor_set_uint8(x_155, 14, x_137);
lean_ctor_set_uint8(x_155, 15, x_138);
lean_ctor_set_uint8(x_155, 16, x_139);
lean_ctor_set_uint8(x_155, 17, x_140);
lean_ctor_set_uint8(x_155, 18, x_141);
x_156 = l_Lean_Meta_Context_configKey(x_6);
if (lean_is_exclusive(x_6)) {
 lean_ctor_release(x_6, 0);
 lean_ctor_release(x_6, 1);
 lean_ctor_release(x_6, 2);
 lean_ctor_release(x_6, 3);
 lean_ctor_release(x_6, 4);
 lean_ctor_release(x_6, 5);
 lean_ctor_release(x_6, 6);
 x_157 = x_6;
} else {
 lean_dec_ref(x_6);
 x_157 = lean_box(0);
}
x_158 = 2;
x_159 = lean_uint64_shift_right(x_156, x_158);
x_160 = lean_uint64_shift_left(x_159, x_158);
x_161 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_162 = lean_uint64_lor(x_160, x_161);
x_163 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_163, 0, x_155);
lean_ctor_set_uint64(x_163, sizeof(void*)*1, x_162);
if (lean_is_scalar(x_157)) {
 x_164 = lean_alloc_ctor(0, 7, 3);
} else {
 x_164 = x_157;
}
lean_ctor_set(x_164, 0, x_163);
lean_ctor_set(x_164, 1, x_143);
lean_ctor_set(x_164, 2, x_144);
lean_ctor_set(x_164, 3, x_145);
lean_ctor_set(x_164, 4, x_146);
lean_ctor_set(x_164, 5, x_147);
lean_ctor_set(x_164, 6, x_148);
lean_ctor_set_uint8(x_164, sizeof(void*)*7, x_142);
lean_ctor_set_uint8(x_164, sizeof(void*)*7 + 1, x_149);
lean_ctor_set_uint8(x_164, sizeof(void*)*7 + 2, x_150);
lean_inc(x_7);
x_165 = l_Lean_Meta_isExprDefEq(x_153, x_4, x_164, x_7, x_8, x_9);
if (lean_obj_tag(x_165) == 0)
{
lean_object* x_166; lean_object* x_167; uint8_t x_168; 
x_166 = lean_ctor_get(x_165, 0);
lean_inc(x_166);
if (lean_is_exclusive(x_165)) {
 lean_ctor_release(x_165, 0);
 x_167 = x_165;
} else {
 lean_dec_ref(x_165);
 x_167 = lean_box(0);
}
x_168 = lean_unbox(x_166);
if (x_168 == 0)
{
lean_object* x_169; lean_object* x_170; lean_object* x_171; lean_object* x_172; 
lean_dec(x_166);
lean_dec(x_7);
x_169 = lean_box(x_5);
x_170 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_170, 0, x_14);
lean_ctor_set(x_170, 1, x_169);
x_171 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_171, 0, x_12);
lean_ctor_set(x_171, 1, x_170);
if (lean_is_scalar(x_167)) {
 x_172 = lean_alloc_ctor(0, 1, 0);
} else {
 x_172 = x_167;
}
lean_ctor_set(x_172, 0, x_171);
return x_172;
}
else
{
lean_object* x_173; 
lean_dec(x_167);
x_173 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_12, x_7);
if (lean_obj_tag(x_173) == 0)
{
lean_object* x_174; lean_object* x_175; 
x_174 = lean_ctor_get(x_173, 0);
lean_inc(x_174);
lean_dec_ref(x_173);
x_175 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_14, x_7);
lean_dec(x_7);
if (lean_obj_tag(x_175) == 0)
{
lean_object* x_176; lean_object* x_177; lean_object* x_178; lean_object* x_179; lean_object* x_180; 
x_176 = lean_ctor_get(x_175, 0);
lean_inc(x_176);
if (lean_is_exclusive(x_175)) {
 lean_ctor_release(x_175, 0);
 x_177 = x_175;
} else {
 lean_dec_ref(x_175);
 x_177 = lean_box(0);
}
x_178 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_178, 0, x_176);
lean_ctor_set(x_178, 1, x_166);
x_179 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_179, 0, x_174);
lean_ctor_set(x_179, 1, x_178);
if (lean_is_scalar(x_177)) {
 x_180 = lean_alloc_ctor(0, 1, 0);
} else {
 x_180 = x_177;
}
lean_ctor_set(x_180, 0, x_179);
return x_180;
}
else
{
lean_object* x_181; lean_object* x_182; lean_object* x_183; 
lean_dec(x_174);
lean_dec(x_166);
x_181 = lean_ctor_get(x_175, 0);
lean_inc(x_181);
if (lean_is_exclusive(x_175)) {
 lean_ctor_release(x_175, 0);
 x_182 = x_175;
} else {
 lean_dec_ref(x_175);
 x_182 = lean_box(0);
}
if (lean_is_scalar(x_182)) {
 x_183 = lean_alloc_ctor(1, 1, 0);
} else {
 x_183 = x_182;
}
lean_ctor_set(x_183, 0, x_181);
return x_183;
}
}
else
{
lean_object* x_184; lean_object* x_185; lean_object* x_186; 
lean_dec(x_166);
lean_dec(x_14);
lean_dec(x_7);
x_184 = lean_ctor_get(x_173, 0);
lean_inc(x_184);
if (lean_is_exclusive(x_173)) {
 lean_ctor_release(x_173, 0);
 x_185 = x_173;
} else {
 lean_dec_ref(x_173);
 x_185 = lean_box(0);
}
if (lean_is_scalar(x_185)) {
 x_186 = lean_alloc_ctor(1, 1, 0);
} else {
 x_186 = x_185;
}
lean_ctor_set(x_186, 0, x_184);
return x_186;
}
}
}
else
{
lean_object* x_187; lean_object* x_188; lean_object* x_189; 
lean_dec(x_14);
lean_dec(x_12);
lean_dec(x_7);
x_187 = lean_ctor_get(x_165, 0);
lean_inc(x_187);
if (lean_is_exclusive(x_165)) {
 lean_ctor_release(x_165, 0);
 x_188 = x_165;
} else {
 lean_dec_ref(x_165);
 x_188 = lean_box(0);
}
if (lean_is_scalar(x_188)) {
 x_189 = lean_alloc_ctor(1, 1, 0);
} else {
 x_189 = x_188;
}
lean_ctor_set(x_189, 0, x_187);
return x_189;
}
}
}
else
{
uint8_t x_190; 
lean_dec(x_12);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_4);
x_190 = !lean_is_exclusive(x_13);
if (x_190 == 0)
{
return x_13;
}
else
{
lean_object* x_191; lean_object* x_192; 
x_191 = lean_ctor_get(x_13, 0);
lean_inc(x_191);
lean_dec(x_13);
x_192 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_192, 0, x_191);
return x_192;
}
}
}
else
{
uint8_t x_193; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec(x_1);
x_193 = !lean_is_exclusive(x_11);
if (x_193 == 0)
{
return x_11;
}
else
{
lean_object* x_194; lean_object* x_195; 
x_194 = lean_ctor_get(x_11, 0);
lean_inc(x_194);
lean_dec(x_11);
x_195 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_195, 0, x_194);
return x_195;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
uint8_t x_11; uint8_t x_12; lean_object* x_13; 
x_11 = lean_unbox(x_2);
x_12 = lean_unbox(x_5);
x_13 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1(x_1, x_11, x_3, x_4, x_12, x_6, x_7, x_8, x_9);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; 
x_7 = l_Lean_Meta_mkFreshLevelMVar(x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_7) == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; lean_object* x_12; lean_object* x_13; 
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
lean_dec_ref(x_7);
lean_inc(x_8);
x_9 = l_Lean_Expr_sort___override(x_8);
x_10 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_10, 0, x_9);
x_11 = 0;
x_12 = lean_box(0);
lean_inc_ref(x_2);
x_13 = l_Lean_Meta_mkFreshExprMVar(x_10, x_11, x_12, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_13) == 0)
{
lean_object* x_14; lean_object* x_15; uint8_t x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
lean_dec_ref(x_13);
x_15 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0;
x_16 = 0;
lean_inc(x_14);
x_17 = l_Lean_Expr_forallE___override(x_12, x_14, x_15, x_16);
x_18 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_18, 0, x_17);
lean_inc_ref(x_2);
x_19 = l_Lean_Meta_mkFreshExprMVar(x_18, x_11, x_12, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_19) == 0)
{
lean_object* x_20; lean_object* x_21; uint8_t x_22; 
x_20 = lean_ctor_get(x_19, 0);
lean_inc(x_20);
lean_dec_ref(x_19);
x_21 = l_Lean_Meta_Context_config(x_2);
x_22 = !lean_is_exclusive(x_21);
if (x_22 == 0)
{
uint8_t x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; uint8_t x_30; uint8_t x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; uint8_t x_38; uint64_t x_39; uint8_t x_40; 
x_23 = lean_ctor_get_uint8(x_2, sizeof(void*)*7);
x_24 = lean_ctor_get(x_2, 1);
lean_inc(x_24);
x_25 = lean_ctor_get(x_2, 2);
lean_inc_ref(x_25);
x_26 = lean_ctor_get(x_2, 3);
lean_inc_ref(x_26);
x_27 = lean_ctor_get(x_2, 4);
lean_inc(x_27);
x_28 = lean_ctor_get(x_2, 5);
lean_inc(x_28);
x_29 = lean_ctor_get(x_2, 6);
lean_inc(x_29);
x_30 = lean_ctor_get_uint8(x_2, sizeof(void*)*7 + 1);
x_31 = lean_ctor_get_uint8(x_2, sizeof(void*)*7 + 2);
x_32 = lp_mathlib_ExistsAndEq_mkNestedExists___closed__1;
x_33 = lean_box(0);
lean_inc(x_8);
x_34 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_34, 0, x_8);
lean_ctor_set(x_34, 1, x_33);
x_35 = l_Lean_Expr_const___override(x_32, x_34);
lean_inc(x_14);
x_36 = l_Lean_Expr_app___override(x_35, x_14);
lean_inc(x_20);
x_37 = l_Lean_Expr_app___override(x_36, x_20);
x_38 = 2;
lean_ctor_set_uint8(x_21, 9, x_38);
x_39 = l_Lean_Meta_Context_configKey(x_2);
x_40 = !lean_is_exclusive(x_2);
if (x_40 == 0)
{
lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; uint64_t x_48; uint64_t x_49; uint64_t x_50; uint64_t x_51; uint64_t x_52; lean_object* x_53; lean_object* x_54; 
x_41 = lean_ctor_get(x_2, 6);
lean_dec(x_41);
x_42 = lean_ctor_get(x_2, 5);
lean_dec(x_42);
x_43 = lean_ctor_get(x_2, 4);
lean_dec(x_43);
x_44 = lean_ctor_get(x_2, 3);
lean_dec(x_44);
x_45 = lean_ctor_get(x_2, 2);
lean_dec(x_45);
x_46 = lean_ctor_get(x_2, 1);
lean_dec(x_46);
x_47 = lean_ctor_get(x_2, 0);
lean_dec(x_47);
x_48 = 2;
x_49 = lean_uint64_shift_right(x_39, x_48);
x_50 = lean_uint64_shift_left(x_49, x_48);
x_51 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_52 = lean_uint64_lor(x_50, x_51);
x_53 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_53, 0, x_21);
lean_ctor_set_uint64(x_53, sizeof(void*)*1, x_52);
lean_ctor_set(x_2, 0, x_53);
lean_inc(x_3);
x_54 = l_Lean_Meta_isExprDefEq(x_37, x_1, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_54) == 0)
{
uint8_t x_55; 
x_55 = !lean_is_exclusive(x_54);
if (x_55 == 0)
{
lean_object* x_56; uint8_t x_57; 
x_56 = lean_ctor_get(x_54, 0);
x_57 = lean_unbox(x_56);
if (x_57 == 0)
{
lean_object* x_58; lean_object* x_59; lean_object* x_60; 
lean_dec(x_3);
x_58 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_58, 0, x_20);
lean_ctor_set(x_58, 1, x_56);
x_59 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_59, 0, x_14);
lean_ctor_set(x_59, 1, x_58);
x_60 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_60, 0, x_8);
lean_ctor_set(x_60, 1, x_59);
lean_ctor_set(x_54, 0, x_60);
return x_54;
}
else
{
lean_object* x_61; 
lean_free_object(x_54);
x_61 = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg(x_8, x_3);
if (lean_obj_tag(x_61) == 0)
{
lean_object* x_62; lean_object* x_63; 
x_62 = lean_ctor_get(x_61, 0);
lean_inc(x_62);
lean_dec_ref(x_61);
x_63 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_14, x_3);
if (lean_obj_tag(x_63) == 0)
{
lean_object* x_64; lean_object* x_65; 
x_64 = lean_ctor_get(x_63, 0);
lean_inc(x_64);
lean_dec_ref(x_63);
x_65 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_20, x_3);
lean_dec(x_3);
if (lean_obj_tag(x_65) == 0)
{
uint8_t x_66; 
x_66 = !lean_is_exclusive(x_65);
if (x_66 == 0)
{
lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; 
x_67 = lean_ctor_get(x_65, 0);
x_68 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_68, 0, x_67);
lean_ctor_set(x_68, 1, x_56);
x_69 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_69, 0, x_64);
lean_ctor_set(x_69, 1, x_68);
x_70 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_70, 0, x_62);
lean_ctor_set(x_70, 1, x_69);
lean_ctor_set(x_65, 0, x_70);
return x_65;
}
else
{
lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; 
x_71 = lean_ctor_get(x_65, 0);
lean_inc(x_71);
lean_dec(x_65);
x_72 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_72, 0, x_71);
lean_ctor_set(x_72, 1, x_56);
x_73 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_73, 0, x_64);
lean_ctor_set(x_73, 1, x_72);
x_74 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_74, 0, x_62);
lean_ctor_set(x_74, 1, x_73);
x_75 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_75, 0, x_74);
return x_75;
}
}
else
{
uint8_t x_76; 
lean_dec(x_64);
lean_dec(x_62);
lean_dec(x_56);
x_76 = !lean_is_exclusive(x_65);
if (x_76 == 0)
{
return x_65;
}
else
{
lean_object* x_77; lean_object* x_78; 
x_77 = lean_ctor_get(x_65, 0);
lean_inc(x_77);
lean_dec(x_65);
x_78 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_78, 0, x_77);
return x_78;
}
}
}
else
{
uint8_t x_79; 
lean_dec(x_62);
lean_dec(x_56);
lean_dec(x_20);
lean_dec(x_3);
x_79 = !lean_is_exclusive(x_63);
if (x_79 == 0)
{
return x_63;
}
else
{
lean_object* x_80; lean_object* x_81; 
x_80 = lean_ctor_get(x_63, 0);
lean_inc(x_80);
lean_dec(x_63);
x_81 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_81, 0, x_80);
return x_81;
}
}
}
else
{
uint8_t x_82; 
lean_dec(x_56);
lean_dec(x_20);
lean_dec(x_14);
lean_dec(x_3);
x_82 = !lean_is_exclusive(x_61);
if (x_82 == 0)
{
return x_61;
}
else
{
lean_object* x_83; lean_object* x_84; 
x_83 = lean_ctor_get(x_61, 0);
lean_inc(x_83);
lean_dec(x_61);
x_84 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_84, 0, x_83);
return x_84;
}
}
}
}
else
{
lean_object* x_85; uint8_t x_86; 
x_85 = lean_ctor_get(x_54, 0);
lean_inc(x_85);
lean_dec(x_54);
x_86 = lean_unbox(x_85);
if (x_86 == 0)
{
lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; 
lean_dec(x_3);
x_87 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_87, 0, x_20);
lean_ctor_set(x_87, 1, x_85);
x_88 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_88, 0, x_14);
lean_ctor_set(x_88, 1, x_87);
x_89 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_89, 0, x_8);
lean_ctor_set(x_89, 1, x_88);
x_90 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_90, 0, x_89);
return x_90;
}
else
{
lean_object* x_91; 
x_91 = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg(x_8, x_3);
if (lean_obj_tag(x_91) == 0)
{
lean_object* x_92; lean_object* x_93; 
x_92 = lean_ctor_get(x_91, 0);
lean_inc(x_92);
lean_dec_ref(x_91);
x_93 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_14, x_3);
if (lean_obj_tag(x_93) == 0)
{
lean_object* x_94; lean_object* x_95; 
x_94 = lean_ctor_get(x_93, 0);
lean_inc(x_94);
lean_dec_ref(x_93);
x_95 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_20, x_3);
lean_dec(x_3);
if (lean_obj_tag(x_95) == 0)
{
lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; 
x_96 = lean_ctor_get(x_95, 0);
lean_inc(x_96);
if (lean_is_exclusive(x_95)) {
 lean_ctor_release(x_95, 0);
 x_97 = x_95;
} else {
 lean_dec_ref(x_95);
 x_97 = lean_box(0);
}
x_98 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_98, 0, x_96);
lean_ctor_set(x_98, 1, x_85);
x_99 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_99, 0, x_94);
lean_ctor_set(x_99, 1, x_98);
x_100 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_100, 0, x_92);
lean_ctor_set(x_100, 1, x_99);
if (lean_is_scalar(x_97)) {
 x_101 = lean_alloc_ctor(0, 1, 0);
} else {
 x_101 = x_97;
}
lean_ctor_set(x_101, 0, x_100);
return x_101;
}
else
{
lean_object* x_102; lean_object* x_103; lean_object* x_104; 
lean_dec(x_94);
lean_dec(x_92);
lean_dec(x_85);
x_102 = lean_ctor_get(x_95, 0);
lean_inc(x_102);
if (lean_is_exclusive(x_95)) {
 lean_ctor_release(x_95, 0);
 x_103 = x_95;
} else {
 lean_dec_ref(x_95);
 x_103 = lean_box(0);
}
if (lean_is_scalar(x_103)) {
 x_104 = lean_alloc_ctor(1, 1, 0);
} else {
 x_104 = x_103;
}
lean_ctor_set(x_104, 0, x_102);
return x_104;
}
}
else
{
lean_object* x_105; lean_object* x_106; lean_object* x_107; 
lean_dec(x_92);
lean_dec(x_85);
lean_dec(x_20);
lean_dec(x_3);
x_105 = lean_ctor_get(x_93, 0);
lean_inc(x_105);
if (lean_is_exclusive(x_93)) {
 lean_ctor_release(x_93, 0);
 x_106 = x_93;
} else {
 lean_dec_ref(x_93);
 x_106 = lean_box(0);
}
if (lean_is_scalar(x_106)) {
 x_107 = lean_alloc_ctor(1, 1, 0);
} else {
 x_107 = x_106;
}
lean_ctor_set(x_107, 0, x_105);
return x_107;
}
}
else
{
lean_object* x_108; lean_object* x_109; lean_object* x_110; 
lean_dec(x_85);
lean_dec(x_20);
lean_dec(x_14);
lean_dec(x_3);
x_108 = lean_ctor_get(x_91, 0);
lean_inc(x_108);
if (lean_is_exclusive(x_91)) {
 lean_ctor_release(x_91, 0);
 x_109 = x_91;
} else {
 lean_dec_ref(x_91);
 x_109 = lean_box(0);
}
if (lean_is_scalar(x_109)) {
 x_110 = lean_alloc_ctor(1, 1, 0);
} else {
 x_110 = x_109;
}
lean_ctor_set(x_110, 0, x_108);
return x_110;
}
}
}
}
else
{
uint8_t x_111; 
lean_dec(x_20);
lean_dec(x_14);
lean_dec(x_8);
lean_dec(x_3);
x_111 = !lean_is_exclusive(x_54);
if (x_111 == 0)
{
return x_54;
}
else
{
lean_object* x_112; lean_object* x_113; 
x_112 = lean_ctor_get(x_54, 0);
lean_inc(x_112);
lean_dec(x_54);
x_113 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_113, 0, x_112);
return x_113;
}
}
}
else
{
uint64_t x_114; uint64_t x_115; uint64_t x_116; uint64_t x_117; uint64_t x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; 
lean_dec(x_2);
x_114 = 2;
x_115 = lean_uint64_shift_right(x_39, x_114);
x_116 = lean_uint64_shift_left(x_115, x_114);
x_117 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_118 = lean_uint64_lor(x_116, x_117);
x_119 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_119, 0, x_21);
lean_ctor_set_uint64(x_119, sizeof(void*)*1, x_118);
x_120 = lean_alloc_ctor(0, 7, 3);
lean_ctor_set(x_120, 0, x_119);
lean_ctor_set(x_120, 1, x_24);
lean_ctor_set(x_120, 2, x_25);
lean_ctor_set(x_120, 3, x_26);
lean_ctor_set(x_120, 4, x_27);
lean_ctor_set(x_120, 5, x_28);
lean_ctor_set(x_120, 6, x_29);
lean_ctor_set_uint8(x_120, sizeof(void*)*7, x_23);
lean_ctor_set_uint8(x_120, sizeof(void*)*7 + 1, x_30);
lean_ctor_set_uint8(x_120, sizeof(void*)*7 + 2, x_31);
lean_inc(x_3);
x_121 = l_Lean_Meta_isExprDefEq(x_37, x_1, x_120, x_3, x_4, x_5);
if (lean_obj_tag(x_121) == 0)
{
lean_object* x_122; lean_object* x_123; uint8_t x_124; 
x_122 = lean_ctor_get(x_121, 0);
lean_inc(x_122);
if (lean_is_exclusive(x_121)) {
 lean_ctor_release(x_121, 0);
 x_123 = x_121;
} else {
 lean_dec_ref(x_121);
 x_123 = lean_box(0);
}
x_124 = lean_unbox(x_122);
if (x_124 == 0)
{
lean_object* x_125; lean_object* x_126; lean_object* x_127; lean_object* x_128; 
lean_dec(x_3);
x_125 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_125, 0, x_20);
lean_ctor_set(x_125, 1, x_122);
x_126 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_126, 0, x_14);
lean_ctor_set(x_126, 1, x_125);
x_127 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_127, 0, x_8);
lean_ctor_set(x_127, 1, x_126);
if (lean_is_scalar(x_123)) {
 x_128 = lean_alloc_ctor(0, 1, 0);
} else {
 x_128 = x_123;
}
lean_ctor_set(x_128, 0, x_127);
return x_128;
}
else
{
lean_object* x_129; 
lean_dec(x_123);
x_129 = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg(x_8, x_3);
if (lean_obj_tag(x_129) == 0)
{
lean_object* x_130; lean_object* x_131; 
x_130 = lean_ctor_get(x_129, 0);
lean_inc(x_130);
lean_dec_ref(x_129);
x_131 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_14, x_3);
if (lean_obj_tag(x_131) == 0)
{
lean_object* x_132; lean_object* x_133; 
x_132 = lean_ctor_get(x_131, 0);
lean_inc(x_132);
lean_dec_ref(x_131);
x_133 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_20, x_3);
lean_dec(x_3);
if (lean_obj_tag(x_133) == 0)
{
lean_object* x_134; lean_object* x_135; lean_object* x_136; lean_object* x_137; lean_object* x_138; lean_object* x_139; 
x_134 = lean_ctor_get(x_133, 0);
lean_inc(x_134);
if (lean_is_exclusive(x_133)) {
 lean_ctor_release(x_133, 0);
 x_135 = x_133;
} else {
 lean_dec_ref(x_133);
 x_135 = lean_box(0);
}
x_136 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_136, 0, x_134);
lean_ctor_set(x_136, 1, x_122);
x_137 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_137, 0, x_132);
lean_ctor_set(x_137, 1, x_136);
x_138 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_138, 0, x_130);
lean_ctor_set(x_138, 1, x_137);
if (lean_is_scalar(x_135)) {
 x_139 = lean_alloc_ctor(0, 1, 0);
} else {
 x_139 = x_135;
}
lean_ctor_set(x_139, 0, x_138);
return x_139;
}
else
{
lean_object* x_140; lean_object* x_141; lean_object* x_142; 
lean_dec(x_132);
lean_dec(x_130);
lean_dec(x_122);
x_140 = lean_ctor_get(x_133, 0);
lean_inc(x_140);
if (lean_is_exclusive(x_133)) {
 lean_ctor_release(x_133, 0);
 x_141 = x_133;
} else {
 lean_dec_ref(x_133);
 x_141 = lean_box(0);
}
if (lean_is_scalar(x_141)) {
 x_142 = lean_alloc_ctor(1, 1, 0);
} else {
 x_142 = x_141;
}
lean_ctor_set(x_142, 0, x_140);
return x_142;
}
}
else
{
lean_object* x_143; lean_object* x_144; lean_object* x_145; 
lean_dec(x_130);
lean_dec(x_122);
lean_dec(x_20);
lean_dec(x_3);
x_143 = lean_ctor_get(x_131, 0);
lean_inc(x_143);
if (lean_is_exclusive(x_131)) {
 lean_ctor_release(x_131, 0);
 x_144 = x_131;
} else {
 lean_dec_ref(x_131);
 x_144 = lean_box(0);
}
if (lean_is_scalar(x_144)) {
 x_145 = lean_alloc_ctor(1, 1, 0);
} else {
 x_145 = x_144;
}
lean_ctor_set(x_145, 0, x_143);
return x_145;
}
}
else
{
lean_object* x_146; lean_object* x_147; lean_object* x_148; 
lean_dec(x_122);
lean_dec(x_20);
lean_dec(x_14);
lean_dec(x_3);
x_146 = lean_ctor_get(x_129, 0);
lean_inc(x_146);
if (lean_is_exclusive(x_129)) {
 lean_ctor_release(x_129, 0);
 x_147 = x_129;
} else {
 lean_dec_ref(x_129);
 x_147 = lean_box(0);
}
if (lean_is_scalar(x_147)) {
 x_148 = lean_alloc_ctor(1, 1, 0);
} else {
 x_148 = x_147;
}
lean_ctor_set(x_148, 0, x_146);
return x_148;
}
}
}
else
{
lean_object* x_149; lean_object* x_150; lean_object* x_151; 
lean_dec(x_20);
lean_dec(x_14);
lean_dec(x_8);
lean_dec(x_3);
x_149 = lean_ctor_get(x_121, 0);
lean_inc(x_149);
if (lean_is_exclusive(x_121)) {
 lean_ctor_release(x_121, 0);
 x_150 = x_121;
} else {
 lean_dec_ref(x_121);
 x_150 = lean_box(0);
}
if (lean_is_scalar(x_150)) {
 x_151 = lean_alloc_ctor(1, 1, 0);
} else {
 x_151 = x_150;
}
lean_ctor_set(x_151, 0, x_149);
return x_151;
}
}
}
else
{
uint8_t x_152; uint8_t x_153; uint8_t x_154; uint8_t x_155; uint8_t x_156; uint8_t x_157; uint8_t x_158; uint8_t x_159; uint8_t x_160; uint8_t x_161; uint8_t x_162; uint8_t x_163; uint8_t x_164; uint8_t x_165; uint8_t x_166; uint8_t x_167; uint8_t x_168; uint8_t x_169; uint8_t x_170; lean_object* x_171; lean_object* x_172; lean_object* x_173; lean_object* x_174; lean_object* x_175; lean_object* x_176; uint8_t x_177; uint8_t x_178; lean_object* x_179; lean_object* x_180; lean_object* x_181; lean_object* x_182; lean_object* x_183; lean_object* x_184; uint8_t x_185; lean_object* x_186; uint64_t x_187; lean_object* x_188; uint64_t x_189; uint64_t x_190; uint64_t x_191; uint64_t x_192; uint64_t x_193; lean_object* x_194; lean_object* x_195; lean_object* x_196; 
x_152 = lean_ctor_get_uint8(x_21, 0);
x_153 = lean_ctor_get_uint8(x_21, 1);
x_154 = lean_ctor_get_uint8(x_21, 2);
x_155 = lean_ctor_get_uint8(x_21, 3);
x_156 = lean_ctor_get_uint8(x_21, 4);
x_157 = lean_ctor_get_uint8(x_21, 5);
x_158 = lean_ctor_get_uint8(x_21, 6);
x_159 = lean_ctor_get_uint8(x_21, 7);
x_160 = lean_ctor_get_uint8(x_21, 8);
x_161 = lean_ctor_get_uint8(x_21, 10);
x_162 = lean_ctor_get_uint8(x_21, 11);
x_163 = lean_ctor_get_uint8(x_21, 12);
x_164 = lean_ctor_get_uint8(x_21, 13);
x_165 = lean_ctor_get_uint8(x_21, 14);
x_166 = lean_ctor_get_uint8(x_21, 15);
x_167 = lean_ctor_get_uint8(x_21, 16);
x_168 = lean_ctor_get_uint8(x_21, 17);
x_169 = lean_ctor_get_uint8(x_21, 18);
lean_dec(x_21);
x_170 = lean_ctor_get_uint8(x_2, sizeof(void*)*7);
x_171 = lean_ctor_get(x_2, 1);
lean_inc(x_171);
x_172 = lean_ctor_get(x_2, 2);
lean_inc_ref(x_172);
x_173 = lean_ctor_get(x_2, 3);
lean_inc_ref(x_173);
x_174 = lean_ctor_get(x_2, 4);
lean_inc(x_174);
x_175 = lean_ctor_get(x_2, 5);
lean_inc(x_175);
x_176 = lean_ctor_get(x_2, 6);
lean_inc(x_176);
x_177 = lean_ctor_get_uint8(x_2, sizeof(void*)*7 + 1);
x_178 = lean_ctor_get_uint8(x_2, sizeof(void*)*7 + 2);
x_179 = lp_mathlib_ExistsAndEq_mkNestedExists___closed__1;
x_180 = lean_box(0);
lean_inc(x_8);
x_181 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_181, 0, x_8);
lean_ctor_set(x_181, 1, x_180);
x_182 = l_Lean_Expr_const___override(x_179, x_181);
lean_inc(x_14);
x_183 = l_Lean_Expr_app___override(x_182, x_14);
lean_inc(x_20);
x_184 = l_Lean_Expr_app___override(x_183, x_20);
x_185 = 2;
x_186 = lean_alloc_ctor(0, 0, 19);
lean_ctor_set_uint8(x_186, 0, x_152);
lean_ctor_set_uint8(x_186, 1, x_153);
lean_ctor_set_uint8(x_186, 2, x_154);
lean_ctor_set_uint8(x_186, 3, x_155);
lean_ctor_set_uint8(x_186, 4, x_156);
lean_ctor_set_uint8(x_186, 5, x_157);
lean_ctor_set_uint8(x_186, 6, x_158);
lean_ctor_set_uint8(x_186, 7, x_159);
lean_ctor_set_uint8(x_186, 8, x_160);
lean_ctor_set_uint8(x_186, 9, x_185);
lean_ctor_set_uint8(x_186, 10, x_161);
lean_ctor_set_uint8(x_186, 11, x_162);
lean_ctor_set_uint8(x_186, 12, x_163);
lean_ctor_set_uint8(x_186, 13, x_164);
lean_ctor_set_uint8(x_186, 14, x_165);
lean_ctor_set_uint8(x_186, 15, x_166);
lean_ctor_set_uint8(x_186, 16, x_167);
lean_ctor_set_uint8(x_186, 17, x_168);
lean_ctor_set_uint8(x_186, 18, x_169);
x_187 = l_Lean_Meta_Context_configKey(x_2);
if (lean_is_exclusive(x_2)) {
 lean_ctor_release(x_2, 0);
 lean_ctor_release(x_2, 1);
 lean_ctor_release(x_2, 2);
 lean_ctor_release(x_2, 3);
 lean_ctor_release(x_2, 4);
 lean_ctor_release(x_2, 5);
 lean_ctor_release(x_2, 6);
 x_188 = x_2;
} else {
 lean_dec_ref(x_2);
 x_188 = lean_box(0);
}
x_189 = 2;
x_190 = lean_uint64_shift_right(x_187, x_189);
x_191 = lean_uint64_shift_left(x_190, x_189);
x_192 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_193 = lean_uint64_lor(x_191, x_192);
x_194 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_194, 0, x_186);
lean_ctor_set_uint64(x_194, sizeof(void*)*1, x_193);
if (lean_is_scalar(x_188)) {
 x_195 = lean_alloc_ctor(0, 7, 3);
} else {
 x_195 = x_188;
}
lean_ctor_set(x_195, 0, x_194);
lean_ctor_set(x_195, 1, x_171);
lean_ctor_set(x_195, 2, x_172);
lean_ctor_set(x_195, 3, x_173);
lean_ctor_set(x_195, 4, x_174);
lean_ctor_set(x_195, 5, x_175);
lean_ctor_set(x_195, 6, x_176);
lean_ctor_set_uint8(x_195, sizeof(void*)*7, x_170);
lean_ctor_set_uint8(x_195, sizeof(void*)*7 + 1, x_177);
lean_ctor_set_uint8(x_195, sizeof(void*)*7 + 2, x_178);
lean_inc(x_3);
x_196 = l_Lean_Meta_isExprDefEq(x_184, x_1, x_195, x_3, x_4, x_5);
if (lean_obj_tag(x_196) == 0)
{
lean_object* x_197; lean_object* x_198; uint8_t x_199; 
x_197 = lean_ctor_get(x_196, 0);
lean_inc(x_197);
if (lean_is_exclusive(x_196)) {
 lean_ctor_release(x_196, 0);
 x_198 = x_196;
} else {
 lean_dec_ref(x_196);
 x_198 = lean_box(0);
}
x_199 = lean_unbox(x_197);
if (x_199 == 0)
{
lean_object* x_200; lean_object* x_201; lean_object* x_202; lean_object* x_203; 
lean_dec(x_3);
x_200 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_200, 0, x_20);
lean_ctor_set(x_200, 1, x_197);
x_201 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_201, 0, x_14);
lean_ctor_set(x_201, 1, x_200);
x_202 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_202, 0, x_8);
lean_ctor_set(x_202, 1, x_201);
if (lean_is_scalar(x_198)) {
 x_203 = lean_alloc_ctor(0, 1, 0);
} else {
 x_203 = x_198;
}
lean_ctor_set(x_203, 0, x_202);
return x_203;
}
else
{
lean_object* x_204; 
lean_dec(x_198);
x_204 = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg(x_8, x_3);
if (lean_obj_tag(x_204) == 0)
{
lean_object* x_205; lean_object* x_206; 
x_205 = lean_ctor_get(x_204, 0);
lean_inc(x_205);
lean_dec_ref(x_204);
x_206 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_14, x_3);
if (lean_obj_tag(x_206) == 0)
{
lean_object* x_207; lean_object* x_208; 
x_207 = lean_ctor_get(x_206, 0);
lean_inc(x_207);
lean_dec_ref(x_206);
x_208 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_20, x_3);
lean_dec(x_3);
if (lean_obj_tag(x_208) == 0)
{
lean_object* x_209; lean_object* x_210; lean_object* x_211; lean_object* x_212; lean_object* x_213; lean_object* x_214; 
x_209 = lean_ctor_get(x_208, 0);
lean_inc(x_209);
if (lean_is_exclusive(x_208)) {
 lean_ctor_release(x_208, 0);
 x_210 = x_208;
} else {
 lean_dec_ref(x_208);
 x_210 = lean_box(0);
}
x_211 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_211, 0, x_209);
lean_ctor_set(x_211, 1, x_197);
x_212 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_212, 0, x_207);
lean_ctor_set(x_212, 1, x_211);
x_213 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_213, 0, x_205);
lean_ctor_set(x_213, 1, x_212);
if (lean_is_scalar(x_210)) {
 x_214 = lean_alloc_ctor(0, 1, 0);
} else {
 x_214 = x_210;
}
lean_ctor_set(x_214, 0, x_213);
return x_214;
}
else
{
lean_object* x_215; lean_object* x_216; lean_object* x_217; 
lean_dec(x_207);
lean_dec(x_205);
lean_dec(x_197);
x_215 = lean_ctor_get(x_208, 0);
lean_inc(x_215);
if (lean_is_exclusive(x_208)) {
 lean_ctor_release(x_208, 0);
 x_216 = x_208;
} else {
 lean_dec_ref(x_208);
 x_216 = lean_box(0);
}
if (lean_is_scalar(x_216)) {
 x_217 = lean_alloc_ctor(1, 1, 0);
} else {
 x_217 = x_216;
}
lean_ctor_set(x_217, 0, x_215);
return x_217;
}
}
else
{
lean_object* x_218; lean_object* x_219; lean_object* x_220; 
lean_dec(x_205);
lean_dec(x_197);
lean_dec(x_20);
lean_dec(x_3);
x_218 = lean_ctor_get(x_206, 0);
lean_inc(x_218);
if (lean_is_exclusive(x_206)) {
 lean_ctor_release(x_206, 0);
 x_219 = x_206;
} else {
 lean_dec_ref(x_206);
 x_219 = lean_box(0);
}
if (lean_is_scalar(x_219)) {
 x_220 = lean_alloc_ctor(1, 1, 0);
} else {
 x_220 = x_219;
}
lean_ctor_set(x_220, 0, x_218);
return x_220;
}
}
else
{
lean_object* x_221; lean_object* x_222; lean_object* x_223; 
lean_dec(x_197);
lean_dec(x_20);
lean_dec(x_14);
lean_dec(x_3);
x_221 = lean_ctor_get(x_204, 0);
lean_inc(x_221);
if (lean_is_exclusive(x_204)) {
 lean_ctor_release(x_204, 0);
 x_222 = x_204;
} else {
 lean_dec_ref(x_204);
 x_222 = lean_box(0);
}
if (lean_is_scalar(x_222)) {
 x_223 = lean_alloc_ctor(1, 1, 0);
} else {
 x_223 = x_222;
}
lean_ctor_set(x_223, 0, x_221);
return x_223;
}
}
}
else
{
lean_object* x_224; lean_object* x_225; lean_object* x_226; 
lean_dec(x_20);
lean_dec(x_14);
lean_dec(x_8);
lean_dec(x_3);
x_224 = lean_ctor_get(x_196, 0);
lean_inc(x_224);
if (lean_is_exclusive(x_196)) {
 lean_ctor_release(x_196, 0);
 x_225 = x_196;
} else {
 lean_dec_ref(x_196);
 x_225 = lean_box(0);
}
if (lean_is_scalar(x_225)) {
 x_226 = lean_alloc_ctor(1, 1, 0);
} else {
 x_226 = x_225;
}
lean_ctor_set(x_226, 0, x_224);
return x_226;
}
}
}
else
{
uint8_t x_227; 
lean_dec(x_14);
lean_dec(x_8);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_227 = !lean_is_exclusive(x_19);
if (x_227 == 0)
{
return x_19;
}
else
{
lean_object* x_228; lean_object* x_229; 
x_228 = lean_ctor_get(x_19, 0);
lean_inc(x_228);
lean_dec(x_19);
x_229 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_229, 0, x_228);
return x_229;
}
}
}
else
{
uint8_t x_230; 
lean_dec(x_8);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_230 = !lean_is_exclusive(x_13);
if (x_230 == 0)
{
return x_13;
}
else
{
lean_object* x_231; lean_object* x_232; 
x_231 = lean_ctor_get(x_13, 0);
lean_inc(x_231);
lean_dec(x_13);
x_232 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_232, 0, x_231);
return x_232;
}
}
}
else
{
uint8_t x_233; 
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_233 = !lean_is_exclusive(x_7);
if (x_233 == 0)
{
return x_7;
}
else
{
lean_object* x_234; lean_object* x_235; 
x_234 = lean_ctor_get(x_7, 0);
lean_inc(x_234);
lean_dec(x_7);
x_235 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_235, 0, x_234);
return x_235;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2(x_1, x_2, x_3, x_4, x_5);
return x_7;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("findEq: unexpected P = ", 23, 23);
return x_1;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = l_Std_Format_defWidth;
return x_1;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("unreachable code has been reached", 33, 33);
return x_1;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("ExistsAndEq.findEq.go", 21, 21);
return x_1;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Mathlib.Tactic.Simproc.ExistsAndEq", 34, 34);
return x_1;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_1 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__2;
x_2 = lean_unsigned_to_nat(33u);
x_3 = lean_unsigned_to_nat(137u);
x_4 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__1;
x_5 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__0;
x_6 = l_mkPanicMessageWithDecl(x_5, x_4, x_3, x_2, x_1);
return x_6;
}
}
static lean_object* _init_lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_Meta_instInhabitedMetaM___lam__0___boxed), 5, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3___closed__0;
x_8 = lean_panic_fn(x_7, x_1);
x_9 = lean_apply_5(x_8, x_2, x_3, x_4, x_5, lean_box(0));
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_14; uint8_t x_15; 
x_14 = lean_array_get_size(x_7);
x_15 = lean_nat_dec_eq(x_14, x_1);
if (x_15 == 0)
{
lean_object* x_16; lean_object* x_17; 
lean_dec_ref(x_8);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
x_16 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__3;
x_17 = lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3(x_16, x_9, x_10, x_11, x_12);
return x_17;
}
else
{
lean_object* x_18; 
x_18 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg(x_2, x_3, x_8, x_4, x_9, x_10, x_11, x_12);
if (lean_obj_tag(x_18) == 0)
{
uint8_t x_19; 
x_19 = !lean_is_exclusive(x_18);
if (x_19 == 0)
{
lean_object* x_20; uint8_t x_21; 
x_20 = lean_ctor_get(x_18, 0);
x_21 = !lean_is_exclusive(x_20);
if (x_21 == 0)
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; 
x_22 = lean_ctor_get(x_20, 0);
x_23 = lean_unsigned_to_nat(0u);
x_24 = lean_array_fget_borrowed(x_7, x_23);
lean_inc(x_24);
x_25 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_25, 0, x_5);
lean_ctor_set(x_25, 1, x_24);
x_26 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_26, 0, x_6);
lean_ctor_set(x_26, 1, x_25);
x_27 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_27, 0, x_26);
lean_ctor_set(x_27, 1, x_22);
lean_ctor_set(x_20, 0, x_27);
return x_18;
}
else
{
lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; 
x_28 = lean_ctor_get(x_20, 0);
x_29 = lean_ctor_get(x_20, 1);
lean_inc(x_29);
lean_inc(x_28);
lean_dec(x_20);
x_30 = lean_unsigned_to_nat(0u);
x_31 = lean_array_fget_borrowed(x_7, x_30);
lean_inc(x_31);
x_32 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_32, 0, x_5);
lean_ctor_set(x_32, 1, x_31);
x_33 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_33, 0, x_6);
lean_ctor_set(x_33, 1, x_32);
x_34 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_34, 0, x_33);
lean_ctor_set(x_34, 1, x_28);
x_35 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_35, 0, x_34);
lean_ctor_set(x_35, 1, x_29);
lean_ctor_set(x_18, 0, x_35);
return x_18;
}
}
else
{
lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; 
x_36 = lean_ctor_get(x_18, 0);
lean_inc(x_36);
lean_dec(x_18);
x_37 = lean_ctor_get(x_36, 0);
lean_inc(x_37);
x_38 = lean_ctor_get(x_36, 1);
lean_inc(x_38);
if (lean_is_exclusive(x_36)) {
 lean_ctor_release(x_36, 0);
 lean_ctor_release(x_36, 1);
 x_39 = x_36;
} else {
 lean_dec_ref(x_36);
 x_39 = lean_box(0);
}
x_40 = lean_unsigned_to_nat(0u);
x_41 = lean_array_fget_borrowed(x_7, x_40);
lean_inc(x_41);
x_42 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_42, 0, x_5);
lean_ctor_set(x_42, 1, x_41);
x_43 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_43, 0, x_6);
lean_ctor_set(x_43, 1, x_42);
x_44 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_44, 0, x_43);
lean_ctor_set(x_44, 1, x_37);
if (lean_is_scalar(x_39)) {
 x_45 = lean_alloc_ctor(0, 2, 0);
} else {
 x_45 = x_39;
}
lean_ctor_set(x_45, 0, x_44);
lean_ctor_set(x_45, 1, x_38);
x_46 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_46, 0, x_45);
return x_46;
}
}
else
{
lean_dec(x_6);
lean_dec(x_5);
return x_18;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec_ref(x_7);
lean_dec(x_1);
return x_14;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("findEq: P is conjunction but path is empty", 42, 42);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
uint8_t x_10; uint8_t x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_10 = 0;
x_11 = 0;
x_12 = lean_box(0);
lean_inc(x_1);
x_13 = l_Lean_Expr_sort___override(x_1);
x_14 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_14, 0, x_13);
x_15 = lean_box(x_11);
x_16 = lean_box(x_10);
lean_inc_ref(x_3);
lean_inc(x_1);
x_17 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__0___boxed), 11, 6);
lean_closure_set(x_17, 0, x_14);
lean_closure_set(x_17, 1, x_15);
lean_closure_set(x_17, 2, x_12);
lean_closure_set(x_17, 3, x_1);
lean_closure_set(x_17, 4, x_3);
lean_closure_set(x_17, 5, x_16);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
x_18 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(x_17, x_10, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_18) == 0)
{
lean_object* x_19; lean_object* x_20; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; uint8_t x_42; 
x_19 = lean_ctor_get(x_18, 0);
lean_inc(x_19);
if (lean_is_exclusive(x_18)) {
 lean_ctor_release(x_18, 0);
 x_20 = x_18;
} else {
 lean_dec_ref(x_18);
 x_20 = lean_box(0);
}
x_24 = lean_ctor_get(x_19, 1);
lean_inc(x_24);
if (lean_is_exclusive(x_19)) {
 lean_ctor_release(x_19, 0);
 lean_ctor_release(x_19, 1);
 x_25 = x_19;
} else {
 lean_dec_ref(x_19);
 x_25 = lean_box(0);
}
x_26 = lean_ctor_get(x_24, 1);
lean_inc(x_26);
x_27 = lean_ctor_get(x_24, 0);
lean_inc(x_27);
if (lean_is_exclusive(x_24)) {
 lean_ctor_release(x_24, 0);
 lean_ctor_release(x_24, 1);
 x_28 = x_24;
} else {
 lean_dec_ref(x_24);
 x_28 = lean_box(0);
}
x_29 = lean_ctor_get(x_26, 0);
lean_inc(x_29);
x_30 = lean_ctor_get(x_26, 1);
lean_inc(x_30);
if (lean_is_exclusive(x_26)) {
 lean_ctor_release(x_26, 0);
 lean_ctor_release(x_26, 1);
 x_31 = x_26;
} else {
 lean_dec_ref(x_26);
 x_31 = lean_box(0);
}
x_42 = lean_unbox(x_30);
lean_dec(x_30);
if (x_42 == 0)
{
lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; 
lean_dec(x_31);
lean_dec(x_29);
lean_dec(x_28);
lean_dec(x_27);
lean_dec(x_25);
lean_dec(x_20);
x_43 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__1;
x_44 = lean_box(x_11);
x_45 = lean_box(x_10);
lean_inc_ref(x_3);
x_46 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___boxed), 10, 5);
lean_closure_set(x_46, 0, x_43);
lean_closure_set(x_46, 1, x_44);
lean_closure_set(x_46, 2, x_12);
lean_closure_set(x_46, 3, x_3);
lean_closure_set(x_46, 4, x_45);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
x_47 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(x_46, x_10, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_47) == 0)
{
lean_object* x_48; lean_object* x_49; lean_object* x_50; uint8_t x_51; 
x_48 = lean_ctor_get(x_47, 0);
lean_inc(x_48);
lean_dec_ref(x_47);
x_49 = lean_ctor_get(x_48, 1);
lean_inc(x_49);
x_50 = lean_ctor_get(x_49, 1);
x_51 = lean_unbox(x_50);
if (x_51 == 0)
{
lean_object* x_52; lean_object* x_53; 
lean_dec(x_49);
lean_dec(x_48);
lean_inc_ref(x_3);
x_52 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___boxed), 6, 1);
lean_closure_set(x_52, 0, x_3);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
x_53 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(x_52, x_10, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_53) == 0)
{
lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; uint8_t x_58; 
x_54 = lean_ctor_get(x_53, 0);
lean_inc(x_54);
lean_dec_ref(x_53);
x_55 = lean_ctor_get(x_54, 1);
lean_inc(x_55);
x_56 = lean_ctor_get(x_55, 1);
lean_inc(x_56);
x_57 = lean_ctor_get(x_56, 1);
x_58 = lean_unbox(x_57);
if (x_58 == 0)
{
lean_object* x_59; 
lean_dec(x_56);
lean_dec(x_55);
lean_dec(x_54);
lean_dec(x_4);
lean_dec_ref(x_2);
lean_dec(x_1);
x_59 = l_Lean_Meta_ppExpr(x_3, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_59) == 0)
{
lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; 
x_60 = lean_ctor_get(x_59, 0);
lean_inc(x_60);
lean_dec_ref(x_59);
x_61 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__2;
x_62 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3;
x_63 = lean_unsigned_to_nat(0u);
x_64 = l_Std_Format_pretty(x_60, x_62, x_63, x_63);
x_65 = lean_string_append(x_61, x_64);
lean_dec_ref(x_64);
x_66 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg(x_65, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_65);
return x_66;
}
else
{
uint8_t x_67; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
x_67 = !lean_is_exclusive(x_59);
if (x_67 == 0)
{
return x_59;
}
else
{
lean_object* x_68; lean_object* x_69; 
x_68 = lean_ctor_get(x_59, 0);
lean_inc(x_68);
lean_dec(x_59);
x_69 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_69, 0, x_68);
return x_69;
}
}
}
else
{
lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; 
lean_dec_ref(x_3);
x_70 = lean_ctor_get(x_54, 0);
lean_inc(x_70);
lean_dec(x_54);
x_71 = lean_ctor_get(x_55, 0);
lean_inc(x_71);
lean_dec(x_55);
x_72 = lean_ctor_get(x_56, 0);
lean_inc(x_72);
lean_dec(x_56);
x_73 = lean_unsigned_to_nat(1u);
x_74 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___boxed), 13, 6);
lean_closure_set(x_74, 0, x_73);
lean_closure_set(x_74, 1, x_1);
lean_closure_set(x_74, 2, x_2);
lean_closure_set(x_74, 3, x_4);
lean_closure_set(x_74, 4, x_71);
lean_closure_set(x_74, 5, x_70);
x_75 = lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___redArg(x_72, x_73, x_74, x_10, x_5, x_6, x_7, x_8);
return x_75;
}
}
else
{
uint8_t x_76; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_76 = !lean_is_exclusive(x_53);
if (x_76 == 0)
{
return x_53;
}
else
{
lean_object* x_77; lean_object* x_78; 
x_77 = lean_ctor_get(x_53, 0);
lean_inc(x_77);
lean_dec(x_53);
x_78 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_78, 0, x_77);
return x_78;
}
}
}
else
{
lean_dec_ref(x_3);
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_79; lean_object* x_80; 
lean_dec(x_49);
lean_dec(x_48);
lean_dec_ref(x_2);
lean_dec(x_1);
x_79 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__4;
x_80 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg(x_79, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec(x_6);
lean_dec_ref(x_5);
return x_80;
}
else
{
lean_object* x_81; uint8_t x_82; 
x_81 = lean_ctor_get(x_4, 0);
x_82 = lean_unbox(x_81);
if (x_82 == 0)
{
lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; 
x_83 = lean_ctor_get(x_48, 0);
lean_inc(x_83);
lean_dec(x_48);
x_84 = lean_ctor_get(x_49, 0);
lean_inc(x_84);
lean_dec(x_49);
x_85 = lean_ctor_get(x_4, 1);
lean_inc(x_85);
lean_dec_ref(x_4);
x_86 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg(x_1, x_2, x_83, x_85, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_86) == 0)
{
uint8_t x_87; 
x_87 = !lean_is_exclusive(x_86);
if (x_87 == 0)
{
lean_object* x_88; lean_object* x_89; lean_object* x_90; uint8_t x_91; 
x_88 = lean_ctor_get(x_86, 0);
x_89 = lean_ctor_get(x_88, 1);
lean_inc(x_89);
x_90 = lean_ctor_get(x_89, 1);
lean_inc(x_90);
x_91 = !lean_is_exclusive(x_88);
if (x_91 == 0)
{
lean_object* x_92; uint8_t x_93; 
x_92 = lean_ctor_get(x_88, 1);
lean_dec(x_92);
x_93 = !lean_is_exclusive(x_89);
if (x_93 == 0)
{
lean_object* x_94; uint8_t x_95; 
x_94 = lean_ctor_get(x_89, 1);
lean_dec(x_94);
x_95 = !lean_is_exclusive(x_90);
if (x_95 == 0)
{
lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; 
x_96 = lean_ctor_get(x_90, 0);
x_97 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0;
x_98 = l_Lean_Expr_app___override(x_97, x_96);
x_99 = l_Lean_Expr_app___override(x_98, x_84);
lean_ctor_set(x_90, 0, x_99);
return x_86;
}
else
{
lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; 
x_100 = lean_ctor_get(x_90, 0);
x_101 = lean_ctor_get(x_90, 1);
lean_inc(x_101);
lean_inc(x_100);
lean_dec(x_90);
x_102 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0;
x_103 = l_Lean_Expr_app___override(x_102, x_100);
x_104 = l_Lean_Expr_app___override(x_103, x_84);
x_105 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_105, 0, x_104);
lean_ctor_set(x_105, 1, x_101);
lean_ctor_set(x_89, 1, x_105);
return x_86;
}
}
else
{
lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; 
x_106 = lean_ctor_get(x_89, 0);
lean_inc(x_106);
lean_dec(x_89);
x_107 = lean_ctor_get(x_90, 0);
lean_inc(x_107);
x_108 = lean_ctor_get(x_90, 1);
lean_inc(x_108);
if (lean_is_exclusive(x_90)) {
 lean_ctor_release(x_90, 0);
 lean_ctor_release(x_90, 1);
 x_109 = x_90;
} else {
 lean_dec_ref(x_90);
 x_109 = lean_box(0);
}
x_110 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0;
x_111 = l_Lean_Expr_app___override(x_110, x_107);
x_112 = l_Lean_Expr_app___override(x_111, x_84);
if (lean_is_scalar(x_109)) {
 x_113 = lean_alloc_ctor(0, 2, 0);
} else {
 x_113 = x_109;
}
lean_ctor_set(x_113, 0, x_112);
lean_ctor_set(x_113, 1, x_108);
x_114 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_114, 0, x_106);
lean_ctor_set(x_114, 1, x_113);
lean_ctor_set(x_88, 1, x_114);
return x_86;
}
}
else
{
lean_object* x_115; lean_object* x_116; lean_object* x_117; lean_object* x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; lean_object* x_126; 
x_115 = lean_ctor_get(x_88, 0);
lean_inc(x_115);
lean_dec(x_88);
x_116 = lean_ctor_get(x_89, 0);
lean_inc(x_116);
if (lean_is_exclusive(x_89)) {
 lean_ctor_release(x_89, 0);
 lean_ctor_release(x_89, 1);
 x_117 = x_89;
} else {
 lean_dec_ref(x_89);
 x_117 = lean_box(0);
}
x_118 = lean_ctor_get(x_90, 0);
lean_inc(x_118);
x_119 = lean_ctor_get(x_90, 1);
lean_inc(x_119);
if (lean_is_exclusive(x_90)) {
 lean_ctor_release(x_90, 0);
 lean_ctor_release(x_90, 1);
 x_120 = x_90;
} else {
 lean_dec_ref(x_90);
 x_120 = lean_box(0);
}
x_121 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0;
x_122 = l_Lean_Expr_app___override(x_121, x_118);
x_123 = l_Lean_Expr_app___override(x_122, x_84);
if (lean_is_scalar(x_120)) {
 x_124 = lean_alloc_ctor(0, 2, 0);
} else {
 x_124 = x_120;
}
lean_ctor_set(x_124, 0, x_123);
lean_ctor_set(x_124, 1, x_119);
if (lean_is_scalar(x_117)) {
 x_125 = lean_alloc_ctor(0, 2, 0);
} else {
 x_125 = x_117;
}
lean_ctor_set(x_125, 0, x_116);
lean_ctor_set(x_125, 1, x_124);
x_126 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_126, 0, x_115);
lean_ctor_set(x_126, 1, x_125);
lean_ctor_set(x_86, 0, x_126);
return x_86;
}
}
else
{
lean_object* x_127; lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; lean_object* x_132; lean_object* x_133; lean_object* x_134; lean_object* x_135; lean_object* x_136; lean_object* x_137; lean_object* x_138; lean_object* x_139; lean_object* x_140; lean_object* x_141; lean_object* x_142; lean_object* x_143; 
x_127 = lean_ctor_get(x_86, 0);
lean_inc(x_127);
lean_dec(x_86);
x_128 = lean_ctor_get(x_127, 1);
lean_inc(x_128);
x_129 = lean_ctor_get(x_128, 1);
lean_inc(x_129);
x_130 = lean_ctor_get(x_127, 0);
lean_inc(x_130);
if (lean_is_exclusive(x_127)) {
 lean_ctor_release(x_127, 0);
 lean_ctor_release(x_127, 1);
 x_131 = x_127;
} else {
 lean_dec_ref(x_127);
 x_131 = lean_box(0);
}
x_132 = lean_ctor_get(x_128, 0);
lean_inc(x_132);
if (lean_is_exclusive(x_128)) {
 lean_ctor_release(x_128, 0);
 lean_ctor_release(x_128, 1);
 x_133 = x_128;
} else {
 lean_dec_ref(x_128);
 x_133 = lean_box(0);
}
x_134 = lean_ctor_get(x_129, 0);
lean_inc(x_134);
x_135 = lean_ctor_get(x_129, 1);
lean_inc(x_135);
if (lean_is_exclusive(x_129)) {
 lean_ctor_release(x_129, 0);
 lean_ctor_release(x_129, 1);
 x_136 = x_129;
} else {
 lean_dec_ref(x_129);
 x_136 = lean_box(0);
}
x_137 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0;
x_138 = l_Lean_Expr_app___override(x_137, x_134);
x_139 = l_Lean_Expr_app___override(x_138, x_84);
if (lean_is_scalar(x_136)) {
 x_140 = lean_alloc_ctor(0, 2, 0);
} else {
 x_140 = x_136;
}
lean_ctor_set(x_140, 0, x_139);
lean_ctor_set(x_140, 1, x_135);
if (lean_is_scalar(x_133)) {
 x_141 = lean_alloc_ctor(0, 2, 0);
} else {
 x_141 = x_133;
}
lean_ctor_set(x_141, 0, x_132);
lean_ctor_set(x_141, 1, x_140);
if (lean_is_scalar(x_131)) {
 x_142 = lean_alloc_ctor(0, 2, 0);
} else {
 x_142 = x_131;
}
lean_ctor_set(x_142, 0, x_130);
lean_ctor_set(x_142, 1, x_141);
x_143 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_143, 0, x_142);
return x_143;
}
}
else
{
lean_dec(x_84);
return x_86;
}
}
else
{
lean_object* x_144; lean_object* x_145; lean_object* x_146; lean_object* x_147; 
x_144 = lean_ctor_get(x_48, 0);
lean_inc(x_144);
lean_dec(x_48);
x_145 = lean_ctor_get(x_49, 0);
lean_inc(x_145);
lean_dec(x_49);
x_146 = lean_ctor_get(x_4, 1);
lean_inc(x_146);
lean_dec_ref(x_4);
x_147 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg(x_1, x_2, x_145, x_146, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_147) == 0)
{
uint8_t x_148; 
x_148 = !lean_is_exclusive(x_147);
if (x_148 == 0)
{
lean_object* x_149; lean_object* x_150; lean_object* x_151; uint8_t x_152; 
x_149 = lean_ctor_get(x_147, 0);
x_150 = lean_ctor_get(x_149, 1);
lean_inc(x_150);
x_151 = lean_ctor_get(x_150, 1);
lean_inc(x_151);
x_152 = !lean_is_exclusive(x_149);
if (x_152 == 0)
{
lean_object* x_153; uint8_t x_154; 
x_153 = lean_ctor_get(x_149, 1);
lean_dec(x_153);
x_154 = !lean_is_exclusive(x_150);
if (x_154 == 0)
{
lean_object* x_155; uint8_t x_156; 
x_155 = lean_ctor_get(x_150, 1);
lean_dec(x_155);
x_156 = !lean_is_exclusive(x_151);
if (x_156 == 0)
{
lean_object* x_157; lean_object* x_158; lean_object* x_159; lean_object* x_160; 
x_157 = lean_ctor_get(x_151, 0);
x_158 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0;
x_159 = l_Lean_Expr_app___override(x_158, x_144);
x_160 = l_Lean_Expr_app___override(x_159, x_157);
lean_ctor_set(x_151, 0, x_160);
return x_147;
}
else
{
lean_object* x_161; lean_object* x_162; lean_object* x_163; lean_object* x_164; lean_object* x_165; lean_object* x_166; 
x_161 = lean_ctor_get(x_151, 0);
x_162 = lean_ctor_get(x_151, 1);
lean_inc(x_162);
lean_inc(x_161);
lean_dec(x_151);
x_163 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0;
x_164 = l_Lean_Expr_app___override(x_163, x_144);
x_165 = l_Lean_Expr_app___override(x_164, x_161);
x_166 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_166, 0, x_165);
lean_ctor_set(x_166, 1, x_162);
lean_ctor_set(x_150, 1, x_166);
return x_147;
}
}
else
{
lean_object* x_167; lean_object* x_168; lean_object* x_169; lean_object* x_170; lean_object* x_171; lean_object* x_172; lean_object* x_173; lean_object* x_174; lean_object* x_175; 
x_167 = lean_ctor_get(x_150, 0);
lean_inc(x_167);
lean_dec(x_150);
x_168 = lean_ctor_get(x_151, 0);
lean_inc(x_168);
x_169 = lean_ctor_get(x_151, 1);
lean_inc(x_169);
if (lean_is_exclusive(x_151)) {
 lean_ctor_release(x_151, 0);
 lean_ctor_release(x_151, 1);
 x_170 = x_151;
} else {
 lean_dec_ref(x_151);
 x_170 = lean_box(0);
}
x_171 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0;
x_172 = l_Lean_Expr_app___override(x_171, x_144);
x_173 = l_Lean_Expr_app___override(x_172, x_168);
if (lean_is_scalar(x_170)) {
 x_174 = lean_alloc_ctor(0, 2, 0);
} else {
 x_174 = x_170;
}
lean_ctor_set(x_174, 0, x_173);
lean_ctor_set(x_174, 1, x_169);
x_175 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_175, 0, x_167);
lean_ctor_set(x_175, 1, x_174);
lean_ctor_set(x_149, 1, x_175);
return x_147;
}
}
else
{
lean_object* x_176; lean_object* x_177; lean_object* x_178; lean_object* x_179; lean_object* x_180; lean_object* x_181; lean_object* x_182; lean_object* x_183; lean_object* x_184; lean_object* x_185; lean_object* x_186; lean_object* x_187; 
x_176 = lean_ctor_get(x_149, 0);
lean_inc(x_176);
lean_dec(x_149);
x_177 = lean_ctor_get(x_150, 0);
lean_inc(x_177);
if (lean_is_exclusive(x_150)) {
 lean_ctor_release(x_150, 0);
 lean_ctor_release(x_150, 1);
 x_178 = x_150;
} else {
 lean_dec_ref(x_150);
 x_178 = lean_box(0);
}
x_179 = lean_ctor_get(x_151, 0);
lean_inc(x_179);
x_180 = lean_ctor_get(x_151, 1);
lean_inc(x_180);
if (lean_is_exclusive(x_151)) {
 lean_ctor_release(x_151, 0);
 lean_ctor_release(x_151, 1);
 x_181 = x_151;
} else {
 lean_dec_ref(x_151);
 x_181 = lean_box(0);
}
x_182 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0;
x_183 = l_Lean_Expr_app___override(x_182, x_144);
x_184 = l_Lean_Expr_app___override(x_183, x_179);
if (lean_is_scalar(x_181)) {
 x_185 = lean_alloc_ctor(0, 2, 0);
} else {
 x_185 = x_181;
}
lean_ctor_set(x_185, 0, x_184);
lean_ctor_set(x_185, 1, x_180);
if (lean_is_scalar(x_178)) {
 x_186 = lean_alloc_ctor(0, 2, 0);
} else {
 x_186 = x_178;
}
lean_ctor_set(x_186, 0, x_177);
lean_ctor_set(x_186, 1, x_185);
x_187 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_187, 0, x_176);
lean_ctor_set(x_187, 1, x_186);
lean_ctor_set(x_147, 0, x_187);
return x_147;
}
}
else
{
lean_object* x_188; lean_object* x_189; lean_object* x_190; lean_object* x_191; lean_object* x_192; lean_object* x_193; lean_object* x_194; lean_object* x_195; lean_object* x_196; lean_object* x_197; lean_object* x_198; lean_object* x_199; lean_object* x_200; lean_object* x_201; lean_object* x_202; lean_object* x_203; lean_object* x_204; 
x_188 = lean_ctor_get(x_147, 0);
lean_inc(x_188);
lean_dec(x_147);
x_189 = lean_ctor_get(x_188, 1);
lean_inc(x_189);
x_190 = lean_ctor_get(x_189, 1);
lean_inc(x_190);
x_191 = lean_ctor_get(x_188, 0);
lean_inc(x_191);
if (lean_is_exclusive(x_188)) {
 lean_ctor_release(x_188, 0);
 lean_ctor_release(x_188, 1);
 x_192 = x_188;
} else {
 lean_dec_ref(x_188);
 x_192 = lean_box(0);
}
x_193 = lean_ctor_get(x_189, 0);
lean_inc(x_193);
if (lean_is_exclusive(x_189)) {
 lean_ctor_release(x_189, 0);
 lean_ctor_release(x_189, 1);
 x_194 = x_189;
} else {
 lean_dec_ref(x_189);
 x_194 = lean_box(0);
}
x_195 = lean_ctor_get(x_190, 0);
lean_inc(x_195);
x_196 = lean_ctor_get(x_190, 1);
lean_inc(x_196);
if (lean_is_exclusive(x_190)) {
 lean_ctor_release(x_190, 0);
 lean_ctor_release(x_190, 1);
 x_197 = x_190;
} else {
 lean_dec_ref(x_190);
 x_197 = lean_box(0);
}
x_198 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0;
x_199 = l_Lean_Expr_app___override(x_198, x_144);
x_200 = l_Lean_Expr_app___override(x_199, x_195);
if (lean_is_scalar(x_197)) {
 x_201 = lean_alloc_ctor(0, 2, 0);
} else {
 x_201 = x_197;
}
lean_ctor_set(x_201, 0, x_200);
lean_ctor_set(x_201, 1, x_196);
if (lean_is_scalar(x_194)) {
 x_202 = lean_alloc_ctor(0, 2, 0);
} else {
 x_202 = x_194;
}
lean_ctor_set(x_202, 0, x_193);
lean_ctor_set(x_202, 1, x_201);
if (lean_is_scalar(x_192)) {
 x_203 = lean_alloc_ctor(0, 2, 0);
} else {
 x_203 = x_192;
}
lean_ctor_set(x_203, 0, x_191);
lean_ctor_set(x_203, 1, x_202);
x_204 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_204, 0, x_203);
return x_204;
}
}
else
{
lean_dec(x_144);
return x_147;
}
}
}
}
}
else
{
uint8_t x_205; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_205 = !lean_is_exclusive(x_47);
if (x_205 == 0)
{
return x_47;
}
else
{
lean_object* x_206; lean_object* x_207; 
x_206 = lean_ctor_get(x_47, 0);
lean_inc(x_206);
lean_dec(x_47);
x_207 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_207, 0, x_206);
return x_207;
}
}
}
else
{
uint8_t x_208; 
lean_dec(x_4);
lean_dec(x_1);
x_208 = lean_expr_eqv(x_2, x_27);
if (x_208 == 0)
{
goto block_41;
}
else
{
lean_object* x_209; uint8_t x_210; 
x_209 = l_Lean_Expr_fvarId_x21(x_2);
x_210 = l_Lean_Expr_containsFVar(x_29, x_209);
lean_dec(x_209);
if (x_210 == 0)
{
if (x_208 == 0)
{
goto block_41;
}
else
{
lean_object* x_211; lean_object* x_212; lean_object* x_213; lean_object* x_214; lean_object* x_215; lean_object* x_216; 
lean_dec(x_31);
lean_dec(x_28);
lean_dec(x_27);
lean_dec(x_25);
lean_dec(x_20);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_2);
x_211 = lean_ctor_get(x_5, 2);
lean_inc_ref(x_211);
lean_dec_ref(x_5);
x_212 = lean_box(0);
x_213 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_213, 0, x_3);
lean_ctor_set(x_213, 1, x_29);
x_214 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_214, 0, x_211);
lean_ctor_set(x_214, 1, x_213);
x_215 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_215, 0, x_212);
lean_ctor_set(x_215, 1, x_214);
x_216 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_216, 0, x_215);
return x_216;
}
}
else
{
goto block_41;
}
}
}
block_23:
{
lean_object* x_21; lean_object* x_22; 
x_21 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__0;
x_22 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg(x_21, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec(x_6);
lean_dec_ref(x_5);
return x_22;
}
block_41:
{
uint8_t x_32; 
x_32 = lean_expr_eqv(x_2, x_29);
lean_dec(x_29);
if (x_32 == 0)
{
lean_dec(x_31);
lean_dec(x_28);
lean_dec(x_27);
lean_dec(x_25);
lean_dec(x_20);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
goto block_23;
}
else
{
lean_object* x_33; uint8_t x_34; 
x_33 = l_Lean_Expr_fvarId_x21(x_2);
lean_dec_ref(x_2);
x_34 = l_Lean_Expr_containsFVar(x_27, x_33);
lean_dec(x_33);
if (x_34 == 0)
{
if (x_32 == 0)
{
lean_dec(x_31);
lean_dec(x_28);
lean_dec(x_27);
lean_dec(x_25);
lean_dec(x_20);
lean_dec_ref(x_3);
goto block_23;
}
else
{
lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
x_35 = lean_ctor_get(x_5, 2);
lean_inc_ref(x_35);
lean_dec_ref(x_5);
x_36 = lean_box(0);
if (lean_is_scalar(x_31)) {
 x_37 = lean_alloc_ctor(0, 2, 0);
} else {
 x_37 = x_31;
}
lean_ctor_set(x_37, 0, x_3);
lean_ctor_set(x_37, 1, x_27);
if (lean_is_scalar(x_28)) {
 x_38 = lean_alloc_ctor(0, 2, 0);
} else {
 x_38 = x_28;
}
lean_ctor_set(x_38, 0, x_35);
lean_ctor_set(x_38, 1, x_37);
if (lean_is_scalar(x_25)) {
 x_39 = lean_alloc_ctor(0, 2, 0);
} else {
 x_39 = x_25;
}
lean_ctor_set(x_39, 0, x_36);
lean_ctor_set(x_39, 1, x_38);
if (lean_is_scalar(x_20)) {
 x_40 = lean_alloc_ctor(0, 1, 0);
} else {
 x_40 = x_20;
}
lean_ctor_set(x_40, 0, x_39);
return x_40;
}
}
else
{
lean_dec(x_31);
lean_dec(x_28);
lean_dec(x_27);
lean_dec(x_25);
lean_dec(x_20);
lean_dec_ref(x_3);
goto block_23;
}
}
}
}
else
{
uint8_t x_217; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_217 = !lean_is_exclusive(x_18);
if (x_217 == 0)
{
return x_18;
}
else
{
lean_object* x_218; lean_object* x_219; 
x_218 = lean_ctor_get(x_18, 0);
lean_inc(x_218);
lean_dec(x_18);
x_219 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_219, 0, x_218);
return x_219;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg(x_1, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
uint8_t x_11; lean_object* x_12; 
x_11 = lean_unbox(x_5);
x_12 = lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4(x_1, x_2, x_3, x_4, x_11, x_6, x_7, x_8, x_9);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
uint8_t x_9; lean_object* x_10; 
x_9 = lean_unbox(x_3);
x_10 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2(x_1, x_2, x_9, x_4, x_5, x_6, x_7);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_2);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3(x_1, x_2, x_3, x_4, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
uint8_t x_8; lean_object* x_9; 
x_8 = lean_unbox(x_2);
x_9 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(x_1, x_8, x_3, x_4, x_5, x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
uint8_t x_10; lean_object* x_11; 
x_10 = lean_unbox(x_4);
x_11 = lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___redArg(x_1, x_2, x_3, x_10, x_5, x_6, x_7, x_8);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg(x_1, x_2);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_1, x_2);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEq(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg(x_1, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEq___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEq___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_ExistsAndEq_findEq(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_2);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEq___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_ExistsAndEq_findEq___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__0(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; 
lean_inc_ref(x_6);
lean_inc(x_3);
x_11 = l_Lean_Meta_mkFreshExprMVar(x_1, x_2, x_3, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_11) == 0)
{
lean_object* x_12; lean_object* x_13; uint8_t x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
lean_dec_ref(x_11);
x_13 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0;
x_14 = 0;
lean_inc(x_12);
lean_inc(x_3);
x_15 = l_Lean_Expr_forallE___override(x_3, x_12, x_13, x_14);
x_16 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_16, 0, x_15);
lean_inc_ref(x_6);
x_17 = l_Lean_Meta_mkFreshExprMVar(x_16, x_2, x_3, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_17) == 0)
{
lean_object* x_18; lean_object* x_19; uint8_t x_20; 
x_18 = lean_ctor_get(x_17, 0);
lean_inc(x_18);
lean_dec_ref(x_17);
x_19 = l_Lean_Meta_Context_config(x_6);
x_20 = !lean_is_exclusive(x_19);
if (x_20 == 0)
{
uint8_t x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; uint8_t x_28; uint8_t x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; uint8_t x_36; uint64_t x_37; uint8_t x_38; 
x_21 = lean_ctor_get_uint8(x_6, sizeof(void*)*7);
x_22 = lean_ctor_get(x_6, 1);
lean_inc(x_22);
x_23 = lean_ctor_get(x_6, 2);
lean_inc_ref(x_23);
x_24 = lean_ctor_get(x_6, 3);
lean_inc_ref(x_24);
x_25 = lean_ctor_get(x_6, 4);
lean_inc(x_25);
x_26 = lean_ctor_get(x_6, 5);
lean_inc(x_26);
x_27 = lean_ctor_get(x_6, 6);
lean_inc(x_27);
x_28 = lean_ctor_get_uint8(x_6, sizeof(void*)*7 + 1);
x_29 = lean_ctor_get_uint8(x_6, sizeof(void*)*7 + 2);
x_30 = lp_mathlib_ExistsAndEq_mkNestedExists___closed__1;
x_31 = lean_box(0);
x_32 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_32, 0, x_4);
lean_ctor_set(x_32, 1, x_31);
x_33 = l_Lean_Expr_const___override(x_30, x_32);
lean_inc(x_12);
x_34 = l_Lean_Expr_app___override(x_33, x_12);
lean_inc(x_18);
x_35 = l_Lean_Expr_app___override(x_34, x_18);
x_36 = 2;
lean_ctor_set_uint8(x_19, 9, x_36);
x_37 = l_Lean_Meta_Context_configKey(x_6);
x_38 = !lean_is_exclusive(x_6);
if (x_38 == 0)
{
lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; uint64_t x_46; uint64_t x_47; uint64_t x_48; uint64_t x_49; uint64_t x_50; lean_object* x_51; lean_object* x_52; 
x_39 = lean_ctor_get(x_6, 6);
lean_dec(x_39);
x_40 = lean_ctor_get(x_6, 5);
lean_dec(x_40);
x_41 = lean_ctor_get(x_6, 4);
lean_dec(x_41);
x_42 = lean_ctor_get(x_6, 3);
lean_dec(x_42);
x_43 = lean_ctor_get(x_6, 2);
lean_dec(x_43);
x_44 = lean_ctor_get(x_6, 1);
lean_dec(x_44);
x_45 = lean_ctor_get(x_6, 0);
lean_dec(x_45);
x_46 = 2;
x_47 = lean_uint64_shift_right(x_37, x_46);
x_48 = lean_uint64_shift_left(x_47, x_46);
x_49 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_50 = lean_uint64_lor(x_48, x_49);
x_51 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_51, 0, x_19);
lean_ctor_set_uint64(x_51, sizeof(void*)*1, x_50);
lean_ctor_set(x_6, 0, x_51);
lean_inc(x_7);
x_52 = l_Lean_Meta_isExprDefEq(x_35, x_5, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_52) == 0)
{
uint8_t x_53; 
x_53 = !lean_is_exclusive(x_52);
if (x_53 == 0)
{
lean_object* x_54; uint8_t x_55; 
x_54 = lean_ctor_get(x_52, 0);
x_55 = lean_unbox(x_54);
if (x_55 == 0)
{
lean_object* x_56; lean_object* x_57; 
lean_dec(x_7);
x_56 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_56, 0, x_18);
lean_ctor_set(x_56, 1, x_54);
x_57 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_57, 0, x_12);
lean_ctor_set(x_57, 1, x_56);
lean_ctor_set(x_52, 0, x_57);
return x_52;
}
else
{
lean_object* x_58; lean_object* x_59; lean_object* x_60; uint8_t x_61; 
lean_free_object(x_52);
x_58 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_12, x_7);
x_59 = lean_ctor_get(x_58, 0);
lean_inc(x_59);
lean_dec_ref(x_58);
x_60 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_18, x_7);
lean_dec(x_7);
x_61 = !lean_is_exclusive(x_60);
if (x_61 == 0)
{
lean_object* x_62; lean_object* x_63; lean_object* x_64; 
x_62 = lean_ctor_get(x_60, 0);
x_63 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_63, 0, x_62);
lean_ctor_set(x_63, 1, x_54);
x_64 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_64, 0, x_59);
lean_ctor_set(x_64, 1, x_63);
lean_ctor_set(x_60, 0, x_64);
return x_60;
}
else
{
lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; 
x_65 = lean_ctor_get(x_60, 0);
lean_inc(x_65);
lean_dec(x_60);
x_66 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_66, 0, x_65);
lean_ctor_set(x_66, 1, x_54);
x_67 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_67, 0, x_59);
lean_ctor_set(x_67, 1, x_66);
x_68 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_68, 0, x_67);
return x_68;
}
}
}
else
{
lean_object* x_69; uint8_t x_70; 
x_69 = lean_ctor_get(x_52, 0);
lean_inc(x_69);
lean_dec(x_52);
x_70 = lean_unbox(x_69);
if (x_70 == 0)
{
lean_object* x_71; lean_object* x_72; lean_object* x_73; 
lean_dec(x_7);
x_71 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_71, 0, x_18);
lean_ctor_set(x_71, 1, x_69);
x_72 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_72, 0, x_12);
lean_ctor_set(x_72, 1, x_71);
x_73 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_73, 0, x_72);
return x_73;
}
else
{
lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; 
x_74 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_12, x_7);
x_75 = lean_ctor_get(x_74, 0);
lean_inc(x_75);
lean_dec_ref(x_74);
x_76 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_18, x_7);
lean_dec(x_7);
x_77 = lean_ctor_get(x_76, 0);
lean_inc(x_77);
if (lean_is_exclusive(x_76)) {
 lean_ctor_release(x_76, 0);
 x_78 = x_76;
} else {
 lean_dec_ref(x_76);
 x_78 = lean_box(0);
}
x_79 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_79, 0, x_77);
lean_ctor_set(x_79, 1, x_69);
x_80 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_80, 0, x_75);
lean_ctor_set(x_80, 1, x_79);
if (lean_is_scalar(x_78)) {
 x_81 = lean_alloc_ctor(0, 1, 0);
} else {
 x_81 = x_78;
}
lean_ctor_set(x_81, 0, x_80);
return x_81;
}
}
}
else
{
uint8_t x_82; 
lean_dec(x_18);
lean_dec(x_12);
lean_dec(x_7);
x_82 = !lean_is_exclusive(x_52);
if (x_82 == 0)
{
return x_52;
}
else
{
lean_object* x_83; lean_object* x_84; 
x_83 = lean_ctor_get(x_52, 0);
lean_inc(x_83);
lean_dec(x_52);
x_84 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_84, 0, x_83);
return x_84;
}
}
}
else
{
uint64_t x_85; uint64_t x_86; uint64_t x_87; uint64_t x_88; uint64_t x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; 
lean_dec(x_6);
x_85 = 2;
x_86 = lean_uint64_shift_right(x_37, x_85);
x_87 = lean_uint64_shift_left(x_86, x_85);
x_88 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_89 = lean_uint64_lor(x_87, x_88);
x_90 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_90, 0, x_19);
lean_ctor_set_uint64(x_90, sizeof(void*)*1, x_89);
x_91 = lean_alloc_ctor(0, 7, 3);
lean_ctor_set(x_91, 0, x_90);
lean_ctor_set(x_91, 1, x_22);
lean_ctor_set(x_91, 2, x_23);
lean_ctor_set(x_91, 3, x_24);
lean_ctor_set(x_91, 4, x_25);
lean_ctor_set(x_91, 5, x_26);
lean_ctor_set(x_91, 6, x_27);
lean_ctor_set_uint8(x_91, sizeof(void*)*7, x_21);
lean_ctor_set_uint8(x_91, sizeof(void*)*7 + 1, x_28);
lean_ctor_set_uint8(x_91, sizeof(void*)*7 + 2, x_29);
lean_inc(x_7);
x_92 = l_Lean_Meta_isExprDefEq(x_35, x_5, x_91, x_7, x_8, x_9);
if (lean_obj_tag(x_92) == 0)
{
lean_object* x_93; lean_object* x_94; uint8_t x_95; 
x_93 = lean_ctor_get(x_92, 0);
lean_inc(x_93);
if (lean_is_exclusive(x_92)) {
 lean_ctor_release(x_92, 0);
 x_94 = x_92;
} else {
 lean_dec_ref(x_92);
 x_94 = lean_box(0);
}
x_95 = lean_unbox(x_93);
if (x_95 == 0)
{
lean_object* x_96; lean_object* x_97; lean_object* x_98; 
lean_dec(x_7);
x_96 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_96, 0, x_18);
lean_ctor_set(x_96, 1, x_93);
x_97 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_97, 0, x_12);
lean_ctor_set(x_97, 1, x_96);
if (lean_is_scalar(x_94)) {
 x_98 = lean_alloc_ctor(0, 1, 0);
} else {
 x_98 = x_94;
}
lean_ctor_set(x_98, 0, x_97);
return x_98;
}
else
{
lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; 
lean_dec(x_94);
x_99 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_12, x_7);
x_100 = lean_ctor_get(x_99, 0);
lean_inc(x_100);
lean_dec_ref(x_99);
x_101 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_18, x_7);
lean_dec(x_7);
x_102 = lean_ctor_get(x_101, 0);
lean_inc(x_102);
if (lean_is_exclusive(x_101)) {
 lean_ctor_release(x_101, 0);
 x_103 = x_101;
} else {
 lean_dec_ref(x_101);
 x_103 = lean_box(0);
}
x_104 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_104, 0, x_102);
lean_ctor_set(x_104, 1, x_93);
x_105 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_105, 0, x_100);
lean_ctor_set(x_105, 1, x_104);
if (lean_is_scalar(x_103)) {
 x_106 = lean_alloc_ctor(0, 1, 0);
} else {
 x_106 = x_103;
}
lean_ctor_set(x_106, 0, x_105);
return x_106;
}
}
else
{
lean_object* x_107; lean_object* x_108; lean_object* x_109; 
lean_dec(x_18);
lean_dec(x_12);
lean_dec(x_7);
x_107 = lean_ctor_get(x_92, 0);
lean_inc(x_107);
if (lean_is_exclusive(x_92)) {
 lean_ctor_release(x_92, 0);
 x_108 = x_92;
} else {
 lean_dec_ref(x_92);
 x_108 = lean_box(0);
}
if (lean_is_scalar(x_108)) {
 x_109 = lean_alloc_ctor(1, 1, 0);
} else {
 x_109 = x_108;
}
lean_ctor_set(x_109, 0, x_107);
return x_109;
}
}
}
else
{
uint8_t x_110; uint8_t x_111; uint8_t x_112; uint8_t x_113; uint8_t x_114; uint8_t x_115; uint8_t x_116; uint8_t x_117; uint8_t x_118; uint8_t x_119; uint8_t x_120; uint8_t x_121; uint8_t x_122; uint8_t x_123; uint8_t x_124; uint8_t x_125; uint8_t x_126; uint8_t x_127; uint8_t x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; lean_object* x_132; lean_object* x_133; lean_object* x_134; uint8_t x_135; uint8_t x_136; lean_object* x_137; lean_object* x_138; lean_object* x_139; lean_object* x_140; lean_object* x_141; lean_object* x_142; uint8_t x_143; lean_object* x_144; uint64_t x_145; lean_object* x_146; uint64_t x_147; uint64_t x_148; uint64_t x_149; uint64_t x_150; uint64_t x_151; lean_object* x_152; lean_object* x_153; lean_object* x_154; 
x_110 = lean_ctor_get_uint8(x_19, 0);
x_111 = lean_ctor_get_uint8(x_19, 1);
x_112 = lean_ctor_get_uint8(x_19, 2);
x_113 = lean_ctor_get_uint8(x_19, 3);
x_114 = lean_ctor_get_uint8(x_19, 4);
x_115 = lean_ctor_get_uint8(x_19, 5);
x_116 = lean_ctor_get_uint8(x_19, 6);
x_117 = lean_ctor_get_uint8(x_19, 7);
x_118 = lean_ctor_get_uint8(x_19, 8);
x_119 = lean_ctor_get_uint8(x_19, 10);
x_120 = lean_ctor_get_uint8(x_19, 11);
x_121 = lean_ctor_get_uint8(x_19, 12);
x_122 = lean_ctor_get_uint8(x_19, 13);
x_123 = lean_ctor_get_uint8(x_19, 14);
x_124 = lean_ctor_get_uint8(x_19, 15);
x_125 = lean_ctor_get_uint8(x_19, 16);
x_126 = lean_ctor_get_uint8(x_19, 17);
x_127 = lean_ctor_get_uint8(x_19, 18);
lean_dec(x_19);
x_128 = lean_ctor_get_uint8(x_6, sizeof(void*)*7);
x_129 = lean_ctor_get(x_6, 1);
lean_inc(x_129);
x_130 = lean_ctor_get(x_6, 2);
lean_inc_ref(x_130);
x_131 = lean_ctor_get(x_6, 3);
lean_inc_ref(x_131);
x_132 = lean_ctor_get(x_6, 4);
lean_inc(x_132);
x_133 = lean_ctor_get(x_6, 5);
lean_inc(x_133);
x_134 = lean_ctor_get(x_6, 6);
lean_inc(x_134);
x_135 = lean_ctor_get_uint8(x_6, sizeof(void*)*7 + 1);
x_136 = lean_ctor_get_uint8(x_6, sizeof(void*)*7 + 2);
x_137 = lp_mathlib_ExistsAndEq_mkNestedExists___closed__1;
x_138 = lean_box(0);
x_139 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_139, 0, x_4);
lean_ctor_set(x_139, 1, x_138);
x_140 = l_Lean_Expr_const___override(x_137, x_139);
lean_inc(x_12);
x_141 = l_Lean_Expr_app___override(x_140, x_12);
lean_inc(x_18);
x_142 = l_Lean_Expr_app___override(x_141, x_18);
x_143 = 2;
x_144 = lean_alloc_ctor(0, 0, 19);
lean_ctor_set_uint8(x_144, 0, x_110);
lean_ctor_set_uint8(x_144, 1, x_111);
lean_ctor_set_uint8(x_144, 2, x_112);
lean_ctor_set_uint8(x_144, 3, x_113);
lean_ctor_set_uint8(x_144, 4, x_114);
lean_ctor_set_uint8(x_144, 5, x_115);
lean_ctor_set_uint8(x_144, 6, x_116);
lean_ctor_set_uint8(x_144, 7, x_117);
lean_ctor_set_uint8(x_144, 8, x_118);
lean_ctor_set_uint8(x_144, 9, x_143);
lean_ctor_set_uint8(x_144, 10, x_119);
lean_ctor_set_uint8(x_144, 11, x_120);
lean_ctor_set_uint8(x_144, 12, x_121);
lean_ctor_set_uint8(x_144, 13, x_122);
lean_ctor_set_uint8(x_144, 14, x_123);
lean_ctor_set_uint8(x_144, 15, x_124);
lean_ctor_set_uint8(x_144, 16, x_125);
lean_ctor_set_uint8(x_144, 17, x_126);
lean_ctor_set_uint8(x_144, 18, x_127);
x_145 = l_Lean_Meta_Context_configKey(x_6);
if (lean_is_exclusive(x_6)) {
 lean_ctor_release(x_6, 0);
 lean_ctor_release(x_6, 1);
 lean_ctor_release(x_6, 2);
 lean_ctor_release(x_6, 3);
 lean_ctor_release(x_6, 4);
 lean_ctor_release(x_6, 5);
 lean_ctor_release(x_6, 6);
 x_146 = x_6;
} else {
 lean_dec_ref(x_6);
 x_146 = lean_box(0);
}
x_147 = 2;
x_148 = lean_uint64_shift_right(x_145, x_147);
x_149 = lean_uint64_shift_left(x_148, x_147);
x_150 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_151 = lean_uint64_lor(x_149, x_150);
x_152 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_152, 0, x_144);
lean_ctor_set_uint64(x_152, sizeof(void*)*1, x_151);
if (lean_is_scalar(x_146)) {
 x_153 = lean_alloc_ctor(0, 7, 3);
} else {
 x_153 = x_146;
}
lean_ctor_set(x_153, 0, x_152);
lean_ctor_set(x_153, 1, x_129);
lean_ctor_set(x_153, 2, x_130);
lean_ctor_set(x_153, 3, x_131);
lean_ctor_set(x_153, 4, x_132);
lean_ctor_set(x_153, 5, x_133);
lean_ctor_set(x_153, 6, x_134);
lean_ctor_set_uint8(x_153, sizeof(void*)*7, x_128);
lean_ctor_set_uint8(x_153, sizeof(void*)*7 + 1, x_135);
lean_ctor_set_uint8(x_153, sizeof(void*)*7 + 2, x_136);
lean_inc(x_7);
x_154 = l_Lean_Meta_isExprDefEq(x_142, x_5, x_153, x_7, x_8, x_9);
if (lean_obj_tag(x_154) == 0)
{
lean_object* x_155; lean_object* x_156; uint8_t x_157; 
x_155 = lean_ctor_get(x_154, 0);
lean_inc(x_155);
if (lean_is_exclusive(x_154)) {
 lean_ctor_release(x_154, 0);
 x_156 = x_154;
} else {
 lean_dec_ref(x_154);
 x_156 = lean_box(0);
}
x_157 = lean_unbox(x_155);
if (x_157 == 0)
{
lean_object* x_158; lean_object* x_159; lean_object* x_160; 
lean_dec(x_7);
x_158 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_158, 0, x_18);
lean_ctor_set(x_158, 1, x_155);
x_159 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_159, 0, x_12);
lean_ctor_set(x_159, 1, x_158);
if (lean_is_scalar(x_156)) {
 x_160 = lean_alloc_ctor(0, 1, 0);
} else {
 x_160 = x_156;
}
lean_ctor_set(x_160, 0, x_159);
return x_160;
}
else
{
lean_object* x_161; lean_object* x_162; lean_object* x_163; lean_object* x_164; lean_object* x_165; lean_object* x_166; lean_object* x_167; lean_object* x_168; 
lean_dec(x_156);
x_161 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_12, x_7);
x_162 = lean_ctor_get(x_161, 0);
lean_inc(x_162);
lean_dec_ref(x_161);
x_163 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_18, x_7);
lean_dec(x_7);
x_164 = lean_ctor_get(x_163, 0);
lean_inc(x_164);
if (lean_is_exclusive(x_163)) {
 lean_ctor_release(x_163, 0);
 x_165 = x_163;
} else {
 lean_dec_ref(x_163);
 x_165 = lean_box(0);
}
x_166 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_166, 0, x_164);
lean_ctor_set(x_166, 1, x_155);
x_167 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_167, 0, x_162);
lean_ctor_set(x_167, 1, x_166);
if (lean_is_scalar(x_165)) {
 x_168 = lean_alloc_ctor(0, 1, 0);
} else {
 x_168 = x_165;
}
lean_ctor_set(x_168, 0, x_167);
return x_168;
}
}
else
{
lean_object* x_169; lean_object* x_170; lean_object* x_171; 
lean_dec(x_18);
lean_dec(x_12);
lean_dec(x_7);
x_169 = lean_ctor_get(x_154, 0);
lean_inc(x_169);
if (lean_is_exclusive(x_154)) {
 lean_ctor_release(x_154, 0);
 x_170 = x_154;
} else {
 lean_dec_ref(x_154);
 x_170 = lean_box(0);
}
if (lean_is_scalar(x_170)) {
 x_171 = lean_alloc_ctor(1, 1, 0);
} else {
 x_171 = x_170;
}
lean_ctor_set(x_171, 0, x_169);
return x_171;
}
}
}
else
{
uint8_t x_172; 
lean_dec(x_12);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
x_172 = !lean_is_exclusive(x_17);
if (x_172 == 0)
{
return x_17;
}
else
{
lean_object* x_173; lean_object* x_174; 
x_173 = lean_ctor_get(x_17, 0);
lean_inc(x_173);
lean_dec(x_17);
x_174 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_174, 0, x_173);
return x_174;
}
}
}
else
{
uint8_t x_175; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec(x_3);
x_175 = !lean_is_exclusive(x_11);
if (x_175 == 0)
{
return x_11;
}
else
{
lean_object* x_176; lean_object* x_177; 
x_176 = lean_ctor_get(x_11, 0);
lean_inc(x_176);
lean_dec(x_11);
x_177 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_177, 0, x_176);
return x_177;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
uint8_t x_11; lean_object* x_12; 
x_11 = lean_unbox(x_2);
x_12 = lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__0(x_1, x_11, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_12;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("withNestedExistsElim: exs is not empty but P is not `Exists`.\n", 62, 62);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("P = ", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(2u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("elim", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1___closed__1;
x_2 = lp_mathlib_ExistsAndEq_mkNestedExists___closed__0;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, uint8_t x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
lean_object* x_18; 
lean_inc(x_16);
lean_inc_ref(x_15);
lean_inc(x_14);
lean_inc_ref(x_13);
lean_inc_ref(x_12);
lean_inc_ref(x_2);
x_18 = lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg(x_1, x_2, x_3, x_12, x_4, x_13, x_14, x_15, x_16);
if (lean_obj_tag(x_18) == 0)
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; uint8_t x_23; uint8_t x_24; uint8_t x_25; lean_object* x_26; 
x_19 = lean_ctor_get(x_18, 0);
lean_inc(x_19);
lean_dec_ref(x_18);
x_20 = lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1___closed__0;
x_21 = lean_array_push(x_20, x_5);
x_22 = lean_array_push(x_21, x_12);
x_23 = 1;
x_24 = lean_unbox(x_7);
x_25 = lean_unbox(x_7);
x_26 = l_Lean_Meta_mkLambdaFVars(x_22, x_19, x_6, x_24, x_6, x_25, x_23, x_13, x_14, x_15, x_16);
lean_dec(x_16);
lean_dec_ref(x_15);
lean_dec(x_14);
lean_dec_ref(x_13);
lean_dec_ref(x_22);
if (lean_obj_tag(x_26) == 0)
{
uint8_t x_27; 
x_27 = !lean_is_exclusive(x_26);
if (x_27 == 0)
{
lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; 
x_28 = lean_ctor_get(x_26, 0);
x_29 = lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1___closed__2;
x_30 = lean_box(0);
x_31 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_31, 0, x_8);
lean_ctor_set(x_31, 1, x_30);
x_32 = l_Lean_Expr_const___override(x_29, x_31);
x_33 = l_Lean_Expr_app___override(x_32, x_9);
x_34 = l_Lean_Expr_app___override(x_33, x_10);
x_35 = l_Lean_Expr_app___override(x_34, x_2);
x_36 = l_Lean_Expr_app___override(x_35, x_11);
x_37 = l_Lean_Expr_app___override(x_36, x_28);
lean_ctor_set(x_26, 0, x_37);
return x_26;
}
else
{
lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; 
x_38 = lean_ctor_get(x_26, 0);
lean_inc(x_38);
lean_dec(x_26);
x_39 = lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1___closed__2;
x_40 = lean_box(0);
x_41 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_41, 0, x_8);
lean_ctor_set(x_41, 1, x_40);
x_42 = l_Lean_Expr_const___override(x_39, x_41);
x_43 = l_Lean_Expr_app___override(x_42, x_9);
x_44 = l_Lean_Expr_app___override(x_43, x_10);
x_45 = l_Lean_Expr_app___override(x_44, x_2);
x_46 = l_Lean_Expr_app___override(x_45, x_11);
x_47 = l_Lean_Expr_app___override(x_46, x_38);
x_48 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_48, 0, x_47);
return x_48;
}
}
else
{
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_2);
return x_26;
}
}
else
{
lean_dec(x_16);
lean_dec_ref(x_15);
lean_dec(x_14);
lean_dec_ref(x_13);
lean_dec_ref(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
return x_18;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1___boxed(lean_object** _args) {
lean_object* x_1 = _args[0];
lean_object* x_2 = _args[1];
lean_object* x_3 = _args[2];
lean_object* x_4 = _args[3];
lean_object* x_5 = _args[4];
lean_object* x_6 = _args[5];
lean_object* x_7 = _args[6];
lean_object* x_8 = _args[7];
lean_object* x_9 = _args[8];
lean_object* x_10 = _args[9];
lean_object* x_11 = _args[10];
lean_object* x_12 = _args[11];
lean_object* x_13 = _args[12];
lean_object* x_14 = _args[13];
lean_object* x_15 = _args[14];
lean_object* x_16 = _args[15];
lean_object* x_17 = _args[16];
_start:
{
uint8_t x_18; lean_object* x_19; 
x_18 = lean_unbox(x_6);
x_19 = lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1(x_1, x_2, x_3, x_4, x_5, x_18, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16);
lean_dec(x_7);
return x_19;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0_spec__0___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; 
x_8 = lean_apply_6(x_1, x_2, x_3, x_4, x_5, x_6, lean_box(0));
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0_spec__0___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0_spec__0___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0_spec__0___redArg(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, uint8_t x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; lean_object* x_12; 
x_11 = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0_spec__0___redArg___lam__0___boxed), 7, 1);
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
LEAN_EXPORT lean_object* lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0___redArg(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
uint8_t x_10; lean_object* x_11; 
x_10 = 0;
x_11 = lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0_spec__0___redArg(x_1, x_2, x_3, x_4, x_10, x_5, x_6, x_7, x_8);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_11; 
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_11 = lean_apply_6(x_5, x_4, x_6, x_7, x_8, x_9, lean_box(0));
return x_11;
}
else
{
lean_object* x_12; lean_object* x_13; uint8_t x_14; 
x_12 = lean_ctor_get(x_3, 0);
lean_inc(x_12);
x_13 = lean_ctor_get(x_12, 1);
lean_inc(x_13);
x_14 = !lean_is_exclusive(x_3);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; uint8_t x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; uint8_t x_25; lean_object* x_26; 
x_15 = lean_ctor_get(x_3, 1);
x_16 = lean_ctor_get(x_3, 0);
lean_dec(x_16);
x_17 = lean_ctor_get(x_12, 0);
lean_inc(x_17);
lean_dec(x_12);
x_18 = lean_ctor_get(x_13, 1);
lean_inc(x_18);
lean_dec(x_13);
lean_inc(x_17);
x_19 = l_Lean_Expr_sort___override(x_17);
x_20 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_20, 0, x_19);
x_21 = 0;
x_22 = lean_box(0);
x_23 = lean_box(x_21);
lean_inc_ref(x_1);
lean_inc(x_17);
x_24 = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__0___boxed), 10, 5);
lean_closure_set(x_24, 0, x_20);
lean_closure_set(x_24, 1, x_23);
lean_closure_set(x_24, 2, x_22);
lean_closure_set(x_24, 3, x_17);
lean_closure_set(x_24, 4, x_1);
x_25 = 0;
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
x_26 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(x_24, x_25, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_26) == 0)
{
lean_object* x_27; lean_object* x_28; lean_object* x_29; uint8_t x_30; 
x_27 = lean_ctor_get(x_26, 0);
lean_inc(x_27);
lean_dec_ref(x_26);
x_28 = lean_ctor_get(x_27, 1);
lean_inc(x_28);
x_29 = lean_ctor_get(x_28, 1);
lean_inc(x_29);
x_30 = lean_unbox(x_29);
if (x_30 == 0)
{
lean_object* x_31; 
lean_dec(x_29);
lean_dec(x_28);
lean_dec(x_27);
lean_dec(x_18);
lean_dec(x_17);
lean_free_object(x_3);
lean_dec(x_15);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
x_31 = l_Lean_Meta_ppExpr(x_1, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_31) == 0)
{
lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; 
x_32 = lean_ctor_get(x_31, 0);
lean_inc(x_32);
lean_dec_ref(x_31);
x_33 = lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___closed__0;
x_34 = lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___closed__1;
x_35 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3;
x_36 = lean_unsigned_to_nat(0u);
x_37 = l_Std_Format_pretty(x_32, x_35, x_36, x_36);
x_38 = lean_string_append(x_34, x_37);
lean_dec_ref(x_37);
x_39 = lean_string_append(x_33, x_38);
lean_dec_ref(x_38);
x_40 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg(x_39, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_39);
return x_40;
}
else
{
uint8_t x_41; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
x_41 = !lean_is_exclusive(x_31);
if (x_41 == 0)
{
return x_31;
}
else
{
lean_object* x_42; lean_object* x_43; 
x_42 = lean_ctor_get(x_31, 0);
lean_inc(x_42);
lean_dec(x_31);
x_43 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_43, 0, x_42);
return x_43;
}
}
}
else
{
lean_object* x_44; lean_object* x_45; uint8_t x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; 
lean_dec_ref(x_1);
x_44 = lean_ctor_get(x_27, 0);
lean_inc(x_44);
lean_dec(x_27);
x_45 = lean_ctor_get(x_28, 0);
lean_inc(x_45);
lean_dec(x_28);
x_46 = 0;
x_47 = lean_box(0);
lean_inc(x_18);
lean_ctor_set(x_3, 1, x_47);
lean_ctor_set(x_3, 0, x_18);
x_48 = lean_array_mk(x_3);
lean_inc(x_45);
x_49 = l_Lean_Expr_betaRev(x_45, x_48, x_25, x_25);
lean_dec_ref(x_48);
x_50 = lean_box(x_25);
lean_inc_ref(x_49);
x_51 = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1___boxed), 17, 11);
lean_closure_set(x_51, 0, x_49);
lean_closure_set(x_51, 1, x_2);
lean_closure_set(x_51, 2, x_15);
lean_closure_set(x_51, 3, x_5);
lean_closure_set(x_51, 4, x_18);
lean_closure_set(x_51, 5, x_50);
lean_closure_set(x_51, 6, x_29);
lean_closure_set(x_51, 7, x_17);
lean_closure_set(x_51, 8, x_44);
lean_closure_set(x_51, 9, x_45);
lean_closure_set(x_51, 10, x_4);
x_52 = lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0___redArg(x_22, x_46, x_49, x_51, x_6, x_7, x_8, x_9);
return x_52;
}
}
else
{
uint8_t x_53; 
lean_dec(x_18);
lean_dec(x_17);
lean_free_object(x_3);
lean_dec(x_15);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_53 = !lean_is_exclusive(x_26);
if (x_53 == 0)
{
return x_26;
}
else
{
lean_object* x_54; lean_object* x_55; 
x_54 = lean_ctor_get(x_26, 0);
lean_inc(x_54);
lean_dec(x_26);
x_55 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_55, 0, x_54);
return x_55;
}
}
}
else
{
lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; uint8_t x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; uint8_t x_65; lean_object* x_66; 
x_56 = lean_ctor_get(x_3, 1);
lean_inc(x_56);
lean_dec(x_3);
x_57 = lean_ctor_get(x_12, 0);
lean_inc(x_57);
lean_dec(x_12);
x_58 = lean_ctor_get(x_13, 1);
lean_inc(x_58);
lean_dec(x_13);
lean_inc(x_57);
x_59 = l_Lean_Expr_sort___override(x_57);
x_60 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_60, 0, x_59);
x_61 = 0;
x_62 = lean_box(0);
x_63 = lean_box(x_61);
lean_inc_ref(x_1);
lean_inc(x_57);
x_64 = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__0___boxed), 10, 5);
lean_closure_set(x_64, 0, x_60);
lean_closure_set(x_64, 1, x_63);
lean_closure_set(x_64, 2, x_62);
lean_closure_set(x_64, 3, x_57);
lean_closure_set(x_64, 4, x_1);
x_65 = 0;
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
x_66 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(x_64, x_65, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_66) == 0)
{
lean_object* x_67; lean_object* x_68; lean_object* x_69; uint8_t x_70; 
x_67 = lean_ctor_get(x_66, 0);
lean_inc(x_67);
lean_dec_ref(x_66);
x_68 = lean_ctor_get(x_67, 1);
lean_inc(x_68);
x_69 = lean_ctor_get(x_68, 1);
lean_inc(x_69);
x_70 = lean_unbox(x_69);
if (x_70 == 0)
{
lean_object* x_71; 
lean_dec(x_69);
lean_dec(x_68);
lean_dec(x_67);
lean_dec(x_58);
lean_dec(x_57);
lean_dec(x_56);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
x_71 = l_Lean_Meta_ppExpr(x_1, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_71) == 0)
{
lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; 
x_72 = lean_ctor_get(x_71, 0);
lean_inc(x_72);
lean_dec_ref(x_71);
x_73 = lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___closed__0;
x_74 = lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___closed__1;
x_75 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3;
x_76 = lean_unsigned_to_nat(0u);
x_77 = l_Std_Format_pretty(x_72, x_75, x_76, x_76);
x_78 = lean_string_append(x_74, x_77);
lean_dec_ref(x_77);
x_79 = lean_string_append(x_73, x_78);
lean_dec_ref(x_78);
x_80 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg(x_79, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_79);
return x_80;
}
else
{
lean_object* x_81; lean_object* x_82; lean_object* x_83; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
x_81 = lean_ctor_get(x_71, 0);
lean_inc(x_81);
if (lean_is_exclusive(x_71)) {
 lean_ctor_release(x_71, 0);
 x_82 = x_71;
} else {
 lean_dec_ref(x_71);
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
lean_object* x_84; lean_object* x_85; uint8_t x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; 
lean_dec_ref(x_1);
x_84 = lean_ctor_get(x_67, 0);
lean_inc(x_84);
lean_dec(x_67);
x_85 = lean_ctor_get(x_68, 0);
lean_inc(x_85);
lean_dec(x_68);
x_86 = 0;
x_87 = lean_box(0);
lean_inc(x_58);
x_88 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_88, 0, x_58);
lean_ctor_set(x_88, 1, x_87);
x_89 = lean_array_mk(x_88);
lean_inc(x_85);
x_90 = l_Lean_Expr_betaRev(x_85, x_89, x_65, x_65);
lean_dec_ref(x_89);
x_91 = lean_box(x_65);
lean_inc_ref(x_90);
x_92 = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1___boxed), 17, 11);
lean_closure_set(x_92, 0, x_90);
lean_closure_set(x_92, 1, x_2);
lean_closure_set(x_92, 2, x_56);
lean_closure_set(x_92, 3, x_5);
lean_closure_set(x_92, 4, x_58);
lean_closure_set(x_92, 5, x_91);
lean_closure_set(x_92, 6, x_69);
lean_closure_set(x_92, 7, x_57);
lean_closure_set(x_92, 8, x_84);
lean_closure_set(x_92, 9, x_85);
lean_closure_set(x_92, 10, x_4);
x_93 = lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0___redArg(x_62, x_86, x_90, x_92, x_6, x_7, x_8, x_9);
return x_93;
}
}
else
{
lean_object* x_94; lean_object* x_95; lean_object* x_96; 
lean_dec(x_58);
lean_dec(x_57);
lean_dec(x_56);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_94 = lean_ctor_get(x_66, 0);
lean_inc(x_94);
if (lean_is_exclusive(x_66)) {
 lean_ctor_release(x_66, 0);
 x_95 = x_66;
} else {
 lean_dec_ref(x_66);
 x_95 = lean_box(0);
}
if (lean_is_scalar(x_95)) {
 x_96 = lean_alloc_ctor(1, 1, 0);
} else {
 x_96 = x_95;
}
lean_ctor_set(x_96, 0, x_94);
return x_96;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withNestedExistsElim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg(x_1, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0_spec__0(lean_object* x_1, lean_object* x_2, uint8_t x_3, lean_object* x_4, lean_object* x_5, uint8_t x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0_spec__0___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, uint8_t x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0___redArg(x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withNestedExistsElim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_ExistsAndEq_withNestedExistsElim(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec_ref(x_2);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
uint8_t x_12; uint8_t x_13; lean_object* x_14; 
x_12 = lean_unbox(x_3);
x_13 = lean_unbox(x_6);
x_14 = lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0_spec__0(x_1, x_2, x_12, x_4, x_5, x_13, x_7, x_8, x_9, x_10);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
uint8_t x_12; lean_object* x_13; 
x_12 = lean_unbox(x_4);
x_13 = lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0(x_1, x_2, x_3, x_12, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_1);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
uint8_t x_10; lean_object* x_11; 
x_10 = lean_unbox(x_2);
x_11 = lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0___redArg(x_1, x_10, x_3, x_4, x_5, x_6, x_7, x_8);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
uint8_t x_11; uint8_t x_12; lean_object* x_13; 
x_11 = lean_unbox(x_2);
x_12 = lean_unbox(x_5);
x_13 = lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0_spec__0___redArg(x_1, x_11, x_3, x_4, x_12, x_6, x_7, x_8, x_9);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___lam__1(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, uint8_t x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; 
lean_inc_ref(x_6);
lean_inc(x_3);
lean_inc(x_1);
x_11 = l_Lean_Meta_mkFreshExprMVar(x_1, x_2, x_3, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_11) == 0)
{
lean_object* x_12; lean_object* x_13; 
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
lean_dec_ref(x_11);
lean_inc_ref(x_6);
x_13 = l_Lean_Meta_mkFreshExprMVar(x_1, x_2, x_3, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_13) == 0)
{
lean_object* x_14; lean_object* x_15; uint8_t x_16; 
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
lean_dec_ref(x_13);
x_15 = l_Lean_Meta_Context_config(x_6);
x_16 = !lean_is_exclusive(x_15);
if (x_16 == 0)
{
uint8_t x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; uint8_t x_24; uint8_t x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; uint8_t x_29; uint64_t x_30; uint8_t x_31; 
x_17 = lean_ctor_get_uint8(x_6, sizeof(void*)*7);
x_18 = lean_ctor_get(x_6, 1);
lean_inc(x_18);
x_19 = lean_ctor_get(x_6, 2);
lean_inc_ref(x_19);
x_20 = lean_ctor_get(x_6, 3);
lean_inc_ref(x_20);
x_21 = lean_ctor_get(x_6, 4);
lean_inc(x_21);
x_22 = lean_ctor_get(x_6, 5);
lean_inc(x_22);
x_23 = lean_ctor_get(x_6, 6);
lean_inc(x_23);
x_24 = lean_ctor_get_uint8(x_6, sizeof(void*)*7 + 1);
x_25 = lean_ctor_get_uint8(x_6, sizeof(void*)*7 + 2);
x_26 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0;
lean_inc(x_12);
x_27 = l_Lean_Expr_app___override(x_26, x_12);
lean_inc(x_14);
x_28 = l_Lean_Expr_app___override(x_27, x_14);
x_29 = 2;
lean_ctor_set_uint8(x_15, 9, x_29);
x_30 = l_Lean_Meta_Context_configKey(x_6);
x_31 = !lean_is_exclusive(x_6);
if (x_31 == 0)
{
lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; uint64_t x_39; uint64_t x_40; uint64_t x_41; uint64_t x_42; uint64_t x_43; lean_object* x_44; lean_object* x_45; 
x_32 = lean_ctor_get(x_6, 6);
lean_dec(x_32);
x_33 = lean_ctor_get(x_6, 5);
lean_dec(x_33);
x_34 = lean_ctor_get(x_6, 4);
lean_dec(x_34);
x_35 = lean_ctor_get(x_6, 3);
lean_dec(x_35);
x_36 = lean_ctor_get(x_6, 2);
lean_dec(x_36);
x_37 = lean_ctor_get(x_6, 1);
lean_dec(x_37);
x_38 = lean_ctor_get(x_6, 0);
lean_dec(x_38);
x_39 = 2;
x_40 = lean_uint64_shift_right(x_30, x_39);
x_41 = lean_uint64_shift_left(x_40, x_39);
x_42 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_43 = lean_uint64_lor(x_41, x_42);
x_44 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_44, 0, x_15);
lean_ctor_set_uint64(x_44, sizeof(void*)*1, x_43);
lean_ctor_set(x_6, 0, x_44);
lean_inc(x_7);
x_45 = l_Lean_Meta_isExprDefEq(x_28, x_4, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_45) == 0)
{
uint8_t x_46; 
x_46 = !lean_is_exclusive(x_45);
if (x_46 == 0)
{
lean_object* x_47; uint8_t x_48; 
x_47 = lean_ctor_get(x_45, 0);
x_48 = lean_unbox(x_47);
if (x_48 == 0)
{
lean_object* x_49; lean_object* x_50; lean_object* x_51; 
lean_dec(x_47);
lean_dec(x_7);
x_49 = lean_box(x_5);
x_50 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_50, 0, x_14);
lean_ctor_set(x_50, 1, x_49);
x_51 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_51, 0, x_12);
lean_ctor_set(x_51, 1, x_50);
lean_ctor_set(x_45, 0, x_51);
return x_45;
}
else
{
lean_object* x_52; 
lean_free_object(x_45);
x_52 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_12, x_7);
if (lean_obj_tag(x_52) == 0)
{
lean_object* x_53; lean_object* x_54; 
x_53 = lean_ctor_get(x_52, 0);
lean_inc(x_53);
lean_dec_ref(x_52);
x_54 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_14, x_7);
lean_dec(x_7);
if (lean_obj_tag(x_54) == 0)
{
uint8_t x_55; 
x_55 = !lean_is_exclusive(x_54);
if (x_55 == 0)
{
lean_object* x_56; lean_object* x_57; lean_object* x_58; 
x_56 = lean_ctor_get(x_54, 0);
x_57 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_57, 0, x_56);
lean_ctor_set(x_57, 1, x_47);
x_58 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_58, 0, x_53);
lean_ctor_set(x_58, 1, x_57);
lean_ctor_set(x_54, 0, x_58);
return x_54;
}
else
{
lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; 
x_59 = lean_ctor_get(x_54, 0);
lean_inc(x_59);
lean_dec(x_54);
x_60 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_60, 0, x_59);
lean_ctor_set(x_60, 1, x_47);
x_61 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_61, 0, x_53);
lean_ctor_set(x_61, 1, x_60);
x_62 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_62, 0, x_61);
return x_62;
}
}
else
{
uint8_t x_63; 
lean_dec(x_53);
lean_dec(x_47);
x_63 = !lean_is_exclusive(x_54);
if (x_63 == 0)
{
return x_54;
}
else
{
lean_object* x_64; lean_object* x_65; 
x_64 = lean_ctor_get(x_54, 0);
lean_inc(x_64);
lean_dec(x_54);
x_65 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_65, 0, x_64);
return x_65;
}
}
}
else
{
uint8_t x_66; 
lean_dec(x_47);
lean_dec(x_14);
lean_dec(x_7);
x_66 = !lean_is_exclusive(x_52);
if (x_66 == 0)
{
return x_52;
}
else
{
lean_object* x_67; lean_object* x_68; 
x_67 = lean_ctor_get(x_52, 0);
lean_inc(x_67);
lean_dec(x_52);
x_68 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_68, 0, x_67);
return x_68;
}
}
}
}
else
{
lean_object* x_69; uint8_t x_70; 
x_69 = lean_ctor_get(x_45, 0);
lean_inc(x_69);
lean_dec(x_45);
x_70 = lean_unbox(x_69);
if (x_70 == 0)
{
lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; 
lean_dec(x_69);
lean_dec(x_7);
x_71 = lean_box(x_5);
x_72 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_72, 0, x_14);
lean_ctor_set(x_72, 1, x_71);
x_73 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_73, 0, x_12);
lean_ctor_set(x_73, 1, x_72);
x_74 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_74, 0, x_73);
return x_74;
}
else
{
lean_object* x_75; 
x_75 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_12, x_7);
if (lean_obj_tag(x_75) == 0)
{
lean_object* x_76; lean_object* x_77; 
x_76 = lean_ctor_get(x_75, 0);
lean_inc(x_76);
lean_dec_ref(x_75);
x_77 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_14, x_7);
lean_dec(x_7);
if (lean_obj_tag(x_77) == 0)
{
lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; 
x_78 = lean_ctor_get(x_77, 0);
lean_inc(x_78);
if (lean_is_exclusive(x_77)) {
 lean_ctor_release(x_77, 0);
 x_79 = x_77;
} else {
 lean_dec_ref(x_77);
 x_79 = lean_box(0);
}
x_80 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_80, 0, x_78);
lean_ctor_set(x_80, 1, x_69);
x_81 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_81, 0, x_76);
lean_ctor_set(x_81, 1, x_80);
if (lean_is_scalar(x_79)) {
 x_82 = lean_alloc_ctor(0, 1, 0);
} else {
 x_82 = x_79;
}
lean_ctor_set(x_82, 0, x_81);
return x_82;
}
else
{
lean_object* x_83; lean_object* x_84; lean_object* x_85; 
lean_dec(x_76);
lean_dec(x_69);
x_83 = lean_ctor_get(x_77, 0);
lean_inc(x_83);
if (lean_is_exclusive(x_77)) {
 lean_ctor_release(x_77, 0);
 x_84 = x_77;
} else {
 lean_dec_ref(x_77);
 x_84 = lean_box(0);
}
if (lean_is_scalar(x_84)) {
 x_85 = lean_alloc_ctor(1, 1, 0);
} else {
 x_85 = x_84;
}
lean_ctor_set(x_85, 0, x_83);
return x_85;
}
}
else
{
lean_object* x_86; lean_object* x_87; lean_object* x_88; 
lean_dec(x_69);
lean_dec(x_14);
lean_dec(x_7);
x_86 = lean_ctor_get(x_75, 0);
lean_inc(x_86);
if (lean_is_exclusive(x_75)) {
 lean_ctor_release(x_75, 0);
 x_87 = x_75;
} else {
 lean_dec_ref(x_75);
 x_87 = lean_box(0);
}
if (lean_is_scalar(x_87)) {
 x_88 = lean_alloc_ctor(1, 1, 0);
} else {
 x_88 = x_87;
}
lean_ctor_set(x_88, 0, x_86);
return x_88;
}
}
}
}
else
{
uint8_t x_89; 
lean_dec(x_14);
lean_dec(x_12);
lean_dec(x_7);
x_89 = !lean_is_exclusive(x_45);
if (x_89 == 0)
{
return x_45;
}
else
{
lean_object* x_90; lean_object* x_91; 
x_90 = lean_ctor_get(x_45, 0);
lean_inc(x_90);
lean_dec(x_45);
x_91 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_91, 0, x_90);
return x_91;
}
}
}
else
{
uint64_t x_92; uint64_t x_93; uint64_t x_94; uint64_t x_95; uint64_t x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; 
lean_dec(x_6);
x_92 = 2;
x_93 = lean_uint64_shift_right(x_30, x_92);
x_94 = lean_uint64_shift_left(x_93, x_92);
x_95 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_96 = lean_uint64_lor(x_94, x_95);
x_97 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_97, 0, x_15);
lean_ctor_set_uint64(x_97, sizeof(void*)*1, x_96);
x_98 = lean_alloc_ctor(0, 7, 3);
lean_ctor_set(x_98, 0, x_97);
lean_ctor_set(x_98, 1, x_18);
lean_ctor_set(x_98, 2, x_19);
lean_ctor_set(x_98, 3, x_20);
lean_ctor_set(x_98, 4, x_21);
lean_ctor_set(x_98, 5, x_22);
lean_ctor_set(x_98, 6, x_23);
lean_ctor_set_uint8(x_98, sizeof(void*)*7, x_17);
lean_ctor_set_uint8(x_98, sizeof(void*)*7 + 1, x_24);
lean_ctor_set_uint8(x_98, sizeof(void*)*7 + 2, x_25);
lean_inc(x_7);
x_99 = l_Lean_Meta_isExprDefEq(x_28, x_4, x_98, x_7, x_8, x_9);
if (lean_obj_tag(x_99) == 0)
{
lean_object* x_100; lean_object* x_101; uint8_t x_102; 
x_100 = lean_ctor_get(x_99, 0);
lean_inc(x_100);
if (lean_is_exclusive(x_99)) {
 lean_ctor_release(x_99, 0);
 x_101 = x_99;
} else {
 lean_dec_ref(x_99);
 x_101 = lean_box(0);
}
x_102 = lean_unbox(x_100);
if (x_102 == 0)
{
lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; 
lean_dec(x_100);
lean_dec(x_7);
x_103 = lean_box(x_5);
x_104 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_104, 0, x_14);
lean_ctor_set(x_104, 1, x_103);
x_105 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_105, 0, x_12);
lean_ctor_set(x_105, 1, x_104);
if (lean_is_scalar(x_101)) {
 x_106 = lean_alloc_ctor(0, 1, 0);
} else {
 x_106 = x_101;
}
lean_ctor_set(x_106, 0, x_105);
return x_106;
}
else
{
lean_object* x_107; 
lean_dec(x_101);
x_107 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_12, x_7);
if (lean_obj_tag(x_107) == 0)
{
lean_object* x_108; lean_object* x_109; 
x_108 = lean_ctor_get(x_107, 0);
lean_inc(x_108);
lean_dec_ref(x_107);
x_109 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_14, x_7);
lean_dec(x_7);
if (lean_obj_tag(x_109) == 0)
{
lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; 
x_110 = lean_ctor_get(x_109, 0);
lean_inc(x_110);
if (lean_is_exclusive(x_109)) {
 lean_ctor_release(x_109, 0);
 x_111 = x_109;
} else {
 lean_dec_ref(x_109);
 x_111 = lean_box(0);
}
x_112 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_112, 0, x_110);
lean_ctor_set(x_112, 1, x_100);
x_113 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_113, 0, x_108);
lean_ctor_set(x_113, 1, x_112);
if (lean_is_scalar(x_111)) {
 x_114 = lean_alloc_ctor(0, 1, 0);
} else {
 x_114 = x_111;
}
lean_ctor_set(x_114, 0, x_113);
return x_114;
}
else
{
lean_object* x_115; lean_object* x_116; lean_object* x_117; 
lean_dec(x_108);
lean_dec(x_100);
x_115 = lean_ctor_get(x_109, 0);
lean_inc(x_115);
if (lean_is_exclusive(x_109)) {
 lean_ctor_release(x_109, 0);
 x_116 = x_109;
} else {
 lean_dec_ref(x_109);
 x_116 = lean_box(0);
}
if (lean_is_scalar(x_116)) {
 x_117 = lean_alloc_ctor(1, 1, 0);
} else {
 x_117 = x_116;
}
lean_ctor_set(x_117, 0, x_115);
return x_117;
}
}
else
{
lean_object* x_118; lean_object* x_119; lean_object* x_120; 
lean_dec(x_100);
lean_dec(x_14);
lean_dec(x_7);
x_118 = lean_ctor_get(x_107, 0);
lean_inc(x_118);
if (lean_is_exclusive(x_107)) {
 lean_ctor_release(x_107, 0);
 x_119 = x_107;
} else {
 lean_dec_ref(x_107);
 x_119 = lean_box(0);
}
if (lean_is_scalar(x_119)) {
 x_120 = lean_alloc_ctor(1, 1, 0);
} else {
 x_120 = x_119;
}
lean_ctor_set(x_120, 0, x_118);
return x_120;
}
}
}
else
{
lean_object* x_121; lean_object* x_122; lean_object* x_123; 
lean_dec(x_14);
lean_dec(x_12);
lean_dec(x_7);
x_121 = lean_ctor_get(x_99, 0);
lean_inc(x_121);
if (lean_is_exclusive(x_99)) {
 lean_ctor_release(x_99, 0);
 x_122 = x_99;
} else {
 lean_dec_ref(x_99);
 x_122 = lean_box(0);
}
if (lean_is_scalar(x_122)) {
 x_123 = lean_alloc_ctor(1, 1, 0);
} else {
 x_123 = x_122;
}
lean_ctor_set(x_123, 0, x_121);
return x_123;
}
}
}
else
{
uint8_t x_124; uint8_t x_125; uint8_t x_126; uint8_t x_127; uint8_t x_128; uint8_t x_129; uint8_t x_130; uint8_t x_131; uint8_t x_132; uint8_t x_133; uint8_t x_134; uint8_t x_135; uint8_t x_136; uint8_t x_137; uint8_t x_138; uint8_t x_139; uint8_t x_140; uint8_t x_141; uint8_t x_142; lean_object* x_143; lean_object* x_144; lean_object* x_145; lean_object* x_146; lean_object* x_147; lean_object* x_148; uint8_t x_149; uint8_t x_150; lean_object* x_151; lean_object* x_152; lean_object* x_153; uint8_t x_154; lean_object* x_155; uint64_t x_156; lean_object* x_157; uint64_t x_158; uint64_t x_159; uint64_t x_160; uint64_t x_161; uint64_t x_162; lean_object* x_163; lean_object* x_164; lean_object* x_165; 
x_124 = lean_ctor_get_uint8(x_15, 0);
x_125 = lean_ctor_get_uint8(x_15, 1);
x_126 = lean_ctor_get_uint8(x_15, 2);
x_127 = lean_ctor_get_uint8(x_15, 3);
x_128 = lean_ctor_get_uint8(x_15, 4);
x_129 = lean_ctor_get_uint8(x_15, 5);
x_130 = lean_ctor_get_uint8(x_15, 6);
x_131 = lean_ctor_get_uint8(x_15, 7);
x_132 = lean_ctor_get_uint8(x_15, 8);
x_133 = lean_ctor_get_uint8(x_15, 10);
x_134 = lean_ctor_get_uint8(x_15, 11);
x_135 = lean_ctor_get_uint8(x_15, 12);
x_136 = lean_ctor_get_uint8(x_15, 13);
x_137 = lean_ctor_get_uint8(x_15, 14);
x_138 = lean_ctor_get_uint8(x_15, 15);
x_139 = lean_ctor_get_uint8(x_15, 16);
x_140 = lean_ctor_get_uint8(x_15, 17);
x_141 = lean_ctor_get_uint8(x_15, 18);
lean_dec(x_15);
x_142 = lean_ctor_get_uint8(x_6, sizeof(void*)*7);
x_143 = lean_ctor_get(x_6, 1);
lean_inc(x_143);
x_144 = lean_ctor_get(x_6, 2);
lean_inc_ref(x_144);
x_145 = lean_ctor_get(x_6, 3);
lean_inc_ref(x_145);
x_146 = lean_ctor_get(x_6, 4);
lean_inc(x_146);
x_147 = lean_ctor_get(x_6, 5);
lean_inc(x_147);
x_148 = lean_ctor_get(x_6, 6);
lean_inc(x_148);
x_149 = lean_ctor_get_uint8(x_6, sizeof(void*)*7 + 1);
x_150 = lean_ctor_get_uint8(x_6, sizeof(void*)*7 + 2);
x_151 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0;
lean_inc(x_12);
x_152 = l_Lean_Expr_app___override(x_151, x_12);
lean_inc(x_14);
x_153 = l_Lean_Expr_app___override(x_152, x_14);
x_154 = 2;
x_155 = lean_alloc_ctor(0, 0, 19);
lean_ctor_set_uint8(x_155, 0, x_124);
lean_ctor_set_uint8(x_155, 1, x_125);
lean_ctor_set_uint8(x_155, 2, x_126);
lean_ctor_set_uint8(x_155, 3, x_127);
lean_ctor_set_uint8(x_155, 4, x_128);
lean_ctor_set_uint8(x_155, 5, x_129);
lean_ctor_set_uint8(x_155, 6, x_130);
lean_ctor_set_uint8(x_155, 7, x_131);
lean_ctor_set_uint8(x_155, 8, x_132);
lean_ctor_set_uint8(x_155, 9, x_154);
lean_ctor_set_uint8(x_155, 10, x_133);
lean_ctor_set_uint8(x_155, 11, x_134);
lean_ctor_set_uint8(x_155, 12, x_135);
lean_ctor_set_uint8(x_155, 13, x_136);
lean_ctor_set_uint8(x_155, 14, x_137);
lean_ctor_set_uint8(x_155, 15, x_138);
lean_ctor_set_uint8(x_155, 16, x_139);
lean_ctor_set_uint8(x_155, 17, x_140);
lean_ctor_set_uint8(x_155, 18, x_141);
x_156 = l_Lean_Meta_Context_configKey(x_6);
if (lean_is_exclusive(x_6)) {
 lean_ctor_release(x_6, 0);
 lean_ctor_release(x_6, 1);
 lean_ctor_release(x_6, 2);
 lean_ctor_release(x_6, 3);
 lean_ctor_release(x_6, 4);
 lean_ctor_release(x_6, 5);
 lean_ctor_release(x_6, 6);
 x_157 = x_6;
} else {
 lean_dec_ref(x_6);
 x_157 = lean_box(0);
}
x_158 = 2;
x_159 = lean_uint64_shift_right(x_156, x_158);
x_160 = lean_uint64_shift_left(x_159, x_158);
x_161 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_162 = lean_uint64_lor(x_160, x_161);
x_163 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_163, 0, x_155);
lean_ctor_set_uint64(x_163, sizeof(void*)*1, x_162);
if (lean_is_scalar(x_157)) {
 x_164 = lean_alloc_ctor(0, 7, 3);
} else {
 x_164 = x_157;
}
lean_ctor_set(x_164, 0, x_163);
lean_ctor_set(x_164, 1, x_143);
lean_ctor_set(x_164, 2, x_144);
lean_ctor_set(x_164, 3, x_145);
lean_ctor_set(x_164, 4, x_146);
lean_ctor_set(x_164, 5, x_147);
lean_ctor_set(x_164, 6, x_148);
lean_ctor_set_uint8(x_164, sizeof(void*)*7, x_142);
lean_ctor_set_uint8(x_164, sizeof(void*)*7 + 1, x_149);
lean_ctor_set_uint8(x_164, sizeof(void*)*7 + 2, x_150);
lean_inc(x_7);
x_165 = l_Lean_Meta_isExprDefEq(x_153, x_4, x_164, x_7, x_8, x_9);
if (lean_obj_tag(x_165) == 0)
{
lean_object* x_166; lean_object* x_167; uint8_t x_168; 
x_166 = lean_ctor_get(x_165, 0);
lean_inc(x_166);
if (lean_is_exclusive(x_165)) {
 lean_ctor_release(x_165, 0);
 x_167 = x_165;
} else {
 lean_dec_ref(x_165);
 x_167 = lean_box(0);
}
x_168 = lean_unbox(x_166);
if (x_168 == 0)
{
lean_object* x_169; lean_object* x_170; lean_object* x_171; lean_object* x_172; 
lean_dec(x_166);
lean_dec(x_7);
x_169 = lean_box(x_5);
x_170 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_170, 0, x_14);
lean_ctor_set(x_170, 1, x_169);
x_171 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_171, 0, x_12);
lean_ctor_set(x_171, 1, x_170);
if (lean_is_scalar(x_167)) {
 x_172 = lean_alloc_ctor(0, 1, 0);
} else {
 x_172 = x_167;
}
lean_ctor_set(x_172, 0, x_171);
return x_172;
}
else
{
lean_object* x_173; 
lean_dec(x_167);
x_173 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_12, x_7);
if (lean_obj_tag(x_173) == 0)
{
lean_object* x_174; lean_object* x_175; 
x_174 = lean_ctor_get(x_173, 0);
lean_inc(x_174);
lean_dec_ref(x_173);
x_175 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_14, x_7);
lean_dec(x_7);
if (lean_obj_tag(x_175) == 0)
{
lean_object* x_176; lean_object* x_177; lean_object* x_178; lean_object* x_179; lean_object* x_180; 
x_176 = lean_ctor_get(x_175, 0);
lean_inc(x_176);
if (lean_is_exclusive(x_175)) {
 lean_ctor_release(x_175, 0);
 x_177 = x_175;
} else {
 lean_dec_ref(x_175);
 x_177 = lean_box(0);
}
x_178 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_178, 0, x_176);
lean_ctor_set(x_178, 1, x_166);
x_179 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_179, 0, x_174);
lean_ctor_set(x_179, 1, x_178);
if (lean_is_scalar(x_177)) {
 x_180 = lean_alloc_ctor(0, 1, 0);
} else {
 x_180 = x_177;
}
lean_ctor_set(x_180, 0, x_179);
return x_180;
}
else
{
lean_object* x_181; lean_object* x_182; lean_object* x_183; 
lean_dec(x_174);
lean_dec(x_166);
x_181 = lean_ctor_get(x_175, 0);
lean_inc(x_181);
if (lean_is_exclusive(x_175)) {
 lean_ctor_release(x_175, 0);
 x_182 = x_175;
} else {
 lean_dec_ref(x_175);
 x_182 = lean_box(0);
}
if (lean_is_scalar(x_182)) {
 x_183 = lean_alloc_ctor(1, 1, 0);
} else {
 x_183 = x_182;
}
lean_ctor_set(x_183, 0, x_181);
return x_183;
}
}
else
{
lean_object* x_184; lean_object* x_185; lean_object* x_186; 
lean_dec(x_166);
lean_dec(x_14);
lean_dec(x_7);
x_184 = lean_ctor_get(x_173, 0);
lean_inc(x_184);
if (lean_is_exclusive(x_173)) {
 lean_ctor_release(x_173, 0);
 x_185 = x_173;
} else {
 lean_dec_ref(x_173);
 x_185 = lean_box(0);
}
if (lean_is_scalar(x_185)) {
 x_186 = lean_alloc_ctor(1, 1, 0);
} else {
 x_186 = x_185;
}
lean_ctor_set(x_186, 0, x_184);
return x_186;
}
}
}
else
{
lean_object* x_187; lean_object* x_188; lean_object* x_189; 
lean_dec(x_14);
lean_dec(x_12);
lean_dec(x_7);
x_187 = lean_ctor_get(x_165, 0);
lean_inc(x_187);
if (lean_is_exclusive(x_165)) {
 lean_ctor_release(x_165, 0);
 x_188 = x_165;
} else {
 lean_dec_ref(x_165);
 x_188 = lean_box(0);
}
if (lean_is_scalar(x_188)) {
 x_189 = lean_alloc_ctor(1, 1, 0);
} else {
 x_189 = x_188;
}
lean_ctor_set(x_189, 0, x_187);
return x_189;
}
}
}
else
{
uint8_t x_190; 
lean_dec(x_12);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_4);
x_190 = !lean_is_exclusive(x_13);
if (x_190 == 0)
{
return x_13;
}
else
{
lean_object* x_191; lean_object* x_192; 
x_191 = lean_ctor_get(x_13, 0);
lean_inc(x_191);
lean_dec(x_13);
x_192 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_192, 0, x_191);
return x_192;
}
}
}
else
{
uint8_t x_193; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec(x_1);
x_193 = !lean_is_exclusive(x_11);
if (x_193 == 0)
{
return x_11;
}
else
{
lean_object* x_194; lean_object* x_195; 
x_194 = lean_ctor_get(x_11, 0);
lean_inc(x_194);
lean_dec(x_11);
x_195 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_195, 0, x_194);
return x_195;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___lam__3(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, uint8_t x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
lean_inc_ref(x_7);
lean_inc(x_3);
lean_inc(x_1);
x_12 = l_Lean_Meta_mkFreshExprMVar(x_1, x_2, x_3, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_12) == 0)
{
lean_object* x_13; lean_object* x_14; 
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
lean_inc_ref(x_7);
x_14 = l_Lean_Meta_mkFreshExprMVar(x_1, x_2, x_3, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_14) == 0)
{
lean_object* x_15; lean_object* x_16; uint8_t x_17; 
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
lean_dec_ref(x_14);
x_16 = l_Lean_Meta_Context_config(x_7);
x_17 = !lean_is_exclusive(x_16);
if (x_17 == 0)
{
uint8_t x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; uint8_t x_25; uint8_t x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; uint8_t x_30; uint64_t x_31; uint8_t x_32; 
x_18 = lean_ctor_get_uint8(x_7, sizeof(void*)*7);
x_19 = lean_ctor_get(x_7, 1);
lean_inc(x_19);
x_20 = lean_ctor_get(x_7, 2);
lean_inc_ref(x_20);
x_21 = lean_ctor_get(x_7, 3);
lean_inc_ref(x_21);
x_22 = lean_ctor_get(x_7, 4);
lean_inc(x_22);
x_23 = lean_ctor_get(x_7, 5);
lean_inc(x_23);
x_24 = lean_ctor_get(x_7, 6);
lean_inc(x_24);
x_25 = lean_ctor_get_uint8(x_7, sizeof(void*)*7 + 1);
x_26 = lean_ctor_get_uint8(x_7, sizeof(void*)*7 + 2);
x_27 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0;
lean_inc(x_13);
x_28 = l_Lean_Expr_app___override(x_27, x_13);
lean_inc(x_15);
x_29 = l_Lean_Expr_app___override(x_28, x_15);
x_30 = 2;
lean_ctor_set_uint8(x_16, 9, x_30);
x_31 = l_Lean_Meta_Context_configKey(x_7);
x_32 = !lean_is_exclusive(x_7);
if (x_32 == 0)
{
lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; uint64_t x_40; uint64_t x_41; uint64_t x_42; uint64_t x_43; uint64_t x_44; lean_object* x_45; lean_object* x_46; 
x_33 = lean_ctor_get(x_7, 6);
lean_dec(x_33);
x_34 = lean_ctor_get(x_7, 5);
lean_dec(x_34);
x_35 = lean_ctor_get(x_7, 4);
lean_dec(x_35);
x_36 = lean_ctor_get(x_7, 3);
lean_dec(x_36);
x_37 = lean_ctor_get(x_7, 2);
lean_dec(x_37);
x_38 = lean_ctor_get(x_7, 1);
lean_dec(x_38);
x_39 = lean_ctor_get(x_7, 0);
lean_dec(x_39);
x_40 = 2;
x_41 = lean_uint64_shift_right(x_31, x_40);
x_42 = lean_uint64_shift_left(x_41, x_40);
x_43 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_44 = lean_uint64_lor(x_42, x_43);
x_45 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_45, 0, x_16);
lean_ctor_set_uint64(x_45, sizeof(void*)*1, x_44);
lean_ctor_set(x_7, 0, x_45);
lean_inc(x_8);
x_46 = l_Lean_Meta_isExprDefEq(x_29, x_4, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_46) == 0)
{
uint8_t x_47; 
x_47 = !lean_is_exclusive(x_46);
if (x_47 == 0)
{
lean_object* x_48; uint8_t x_49; 
x_48 = lean_ctor_get(x_46, 0);
x_49 = lean_unbox(x_48);
lean_dec(x_48);
if (x_49 == 0)
{
lean_object* x_50; lean_object* x_51; lean_object* x_52; 
lean_dec(x_8);
lean_dec(x_6);
x_50 = lean_box(x_5);
x_51 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_51, 0, x_15);
lean_ctor_set(x_51, 1, x_50);
x_52 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_52, 0, x_13);
lean_ctor_set(x_52, 1, x_51);
lean_ctor_set(x_46, 0, x_52);
return x_46;
}
else
{
lean_object* x_53; 
lean_free_object(x_46);
x_53 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_13, x_8);
if (lean_obj_tag(x_53) == 0)
{
lean_object* x_54; lean_object* x_55; 
x_54 = lean_ctor_get(x_53, 0);
lean_inc(x_54);
lean_dec_ref(x_53);
x_55 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_15, x_8);
lean_dec(x_8);
if (lean_obj_tag(x_55) == 0)
{
uint8_t x_56; 
x_56 = !lean_is_exclusive(x_55);
if (x_56 == 0)
{
lean_object* x_57; lean_object* x_58; lean_object* x_59; 
x_57 = lean_ctor_get(x_55, 0);
x_58 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_58, 0, x_57);
lean_ctor_set(x_58, 1, x_6);
x_59 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_59, 0, x_54);
lean_ctor_set(x_59, 1, x_58);
lean_ctor_set(x_55, 0, x_59);
return x_55;
}
else
{
lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; 
x_60 = lean_ctor_get(x_55, 0);
lean_inc(x_60);
lean_dec(x_55);
x_61 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_61, 0, x_60);
lean_ctor_set(x_61, 1, x_6);
x_62 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_62, 0, x_54);
lean_ctor_set(x_62, 1, x_61);
x_63 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_63, 0, x_62);
return x_63;
}
}
else
{
uint8_t x_64; 
lean_dec(x_54);
lean_dec(x_6);
x_64 = !lean_is_exclusive(x_55);
if (x_64 == 0)
{
return x_55;
}
else
{
lean_object* x_65; lean_object* x_66; 
x_65 = lean_ctor_get(x_55, 0);
lean_inc(x_65);
lean_dec(x_55);
x_66 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_66, 0, x_65);
return x_66;
}
}
}
else
{
uint8_t x_67; 
lean_dec(x_15);
lean_dec(x_8);
lean_dec(x_6);
x_67 = !lean_is_exclusive(x_53);
if (x_67 == 0)
{
return x_53;
}
else
{
lean_object* x_68; lean_object* x_69; 
x_68 = lean_ctor_get(x_53, 0);
lean_inc(x_68);
lean_dec(x_53);
x_69 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_69, 0, x_68);
return x_69;
}
}
}
}
else
{
lean_object* x_70; uint8_t x_71; 
x_70 = lean_ctor_get(x_46, 0);
lean_inc(x_70);
lean_dec(x_46);
x_71 = lean_unbox(x_70);
lean_dec(x_70);
if (x_71 == 0)
{
lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; 
lean_dec(x_8);
lean_dec(x_6);
x_72 = lean_box(x_5);
x_73 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_73, 0, x_15);
lean_ctor_set(x_73, 1, x_72);
x_74 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_74, 0, x_13);
lean_ctor_set(x_74, 1, x_73);
x_75 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_75, 0, x_74);
return x_75;
}
else
{
lean_object* x_76; 
x_76 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_13, x_8);
if (lean_obj_tag(x_76) == 0)
{
lean_object* x_77; lean_object* x_78; 
x_77 = lean_ctor_get(x_76, 0);
lean_inc(x_77);
lean_dec_ref(x_76);
x_78 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_15, x_8);
lean_dec(x_8);
if (lean_obj_tag(x_78) == 0)
{
lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; 
x_79 = lean_ctor_get(x_78, 0);
lean_inc(x_79);
if (lean_is_exclusive(x_78)) {
 lean_ctor_release(x_78, 0);
 x_80 = x_78;
} else {
 lean_dec_ref(x_78);
 x_80 = lean_box(0);
}
x_81 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_81, 0, x_79);
lean_ctor_set(x_81, 1, x_6);
x_82 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_82, 0, x_77);
lean_ctor_set(x_82, 1, x_81);
if (lean_is_scalar(x_80)) {
 x_83 = lean_alloc_ctor(0, 1, 0);
} else {
 x_83 = x_80;
}
lean_ctor_set(x_83, 0, x_82);
return x_83;
}
else
{
lean_object* x_84; lean_object* x_85; lean_object* x_86; 
lean_dec(x_77);
lean_dec(x_6);
x_84 = lean_ctor_get(x_78, 0);
lean_inc(x_84);
if (lean_is_exclusive(x_78)) {
 lean_ctor_release(x_78, 0);
 x_85 = x_78;
} else {
 lean_dec_ref(x_78);
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
else
{
lean_object* x_87; lean_object* x_88; lean_object* x_89; 
lean_dec(x_15);
lean_dec(x_8);
lean_dec(x_6);
x_87 = lean_ctor_get(x_76, 0);
lean_inc(x_87);
if (lean_is_exclusive(x_76)) {
 lean_ctor_release(x_76, 0);
 x_88 = x_76;
} else {
 lean_dec_ref(x_76);
 x_88 = lean_box(0);
}
if (lean_is_scalar(x_88)) {
 x_89 = lean_alloc_ctor(1, 1, 0);
} else {
 x_89 = x_88;
}
lean_ctor_set(x_89, 0, x_87);
return x_89;
}
}
}
}
else
{
uint8_t x_90; 
lean_dec(x_15);
lean_dec(x_13);
lean_dec(x_8);
lean_dec(x_6);
x_90 = !lean_is_exclusive(x_46);
if (x_90 == 0)
{
return x_46;
}
else
{
lean_object* x_91; lean_object* x_92; 
x_91 = lean_ctor_get(x_46, 0);
lean_inc(x_91);
lean_dec(x_46);
x_92 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_92, 0, x_91);
return x_92;
}
}
}
else
{
uint64_t x_93; uint64_t x_94; uint64_t x_95; uint64_t x_96; uint64_t x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; 
lean_dec(x_7);
x_93 = 2;
x_94 = lean_uint64_shift_right(x_31, x_93);
x_95 = lean_uint64_shift_left(x_94, x_93);
x_96 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_97 = lean_uint64_lor(x_95, x_96);
x_98 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_98, 0, x_16);
lean_ctor_set_uint64(x_98, sizeof(void*)*1, x_97);
x_99 = lean_alloc_ctor(0, 7, 3);
lean_ctor_set(x_99, 0, x_98);
lean_ctor_set(x_99, 1, x_19);
lean_ctor_set(x_99, 2, x_20);
lean_ctor_set(x_99, 3, x_21);
lean_ctor_set(x_99, 4, x_22);
lean_ctor_set(x_99, 5, x_23);
lean_ctor_set(x_99, 6, x_24);
lean_ctor_set_uint8(x_99, sizeof(void*)*7, x_18);
lean_ctor_set_uint8(x_99, sizeof(void*)*7 + 1, x_25);
lean_ctor_set_uint8(x_99, sizeof(void*)*7 + 2, x_26);
lean_inc(x_8);
x_100 = l_Lean_Meta_isExprDefEq(x_29, x_4, x_99, x_8, x_9, x_10);
if (lean_obj_tag(x_100) == 0)
{
lean_object* x_101; lean_object* x_102; uint8_t x_103; 
x_101 = lean_ctor_get(x_100, 0);
lean_inc(x_101);
if (lean_is_exclusive(x_100)) {
 lean_ctor_release(x_100, 0);
 x_102 = x_100;
} else {
 lean_dec_ref(x_100);
 x_102 = lean_box(0);
}
x_103 = lean_unbox(x_101);
lean_dec(x_101);
if (x_103 == 0)
{
lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; 
lean_dec(x_8);
lean_dec(x_6);
x_104 = lean_box(x_5);
x_105 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_105, 0, x_15);
lean_ctor_set(x_105, 1, x_104);
x_106 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_106, 0, x_13);
lean_ctor_set(x_106, 1, x_105);
if (lean_is_scalar(x_102)) {
 x_107 = lean_alloc_ctor(0, 1, 0);
} else {
 x_107 = x_102;
}
lean_ctor_set(x_107, 0, x_106);
return x_107;
}
else
{
lean_object* x_108; 
lean_dec(x_102);
x_108 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_13, x_8);
if (lean_obj_tag(x_108) == 0)
{
lean_object* x_109; lean_object* x_110; 
x_109 = lean_ctor_get(x_108, 0);
lean_inc(x_109);
lean_dec_ref(x_108);
x_110 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_15, x_8);
lean_dec(x_8);
if (lean_obj_tag(x_110) == 0)
{
lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; 
x_111 = lean_ctor_get(x_110, 0);
lean_inc(x_111);
if (lean_is_exclusive(x_110)) {
 lean_ctor_release(x_110, 0);
 x_112 = x_110;
} else {
 lean_dec_ref(x_110);
 x_112 = lean_box(0);
}
x_113 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_113, 0, x_111);
lean_ctor_set(x_113, 1, x_6);
x_114 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_114, 0, x_109);
lean_ctor_set(x_114, 1, x_113);
if (lean_is_scalar(x_112)) {
 x_115 = lean_alloc_ctor(0, 1, 0);
} else {
 x_115 = x_112;
}
lean_ctor_set(x_115, 0, x_114);
return x_115;
}
else
{
lean_object* x_116; lean_object* x_117; lean_object* x_118; 
lean_dec(x_109);
lean_dec(x_6);
x_116 = lean_ctor_get(x_110, 0);
lean_inc(x_116);
if (lean_is_exclusive(x_110)) {
 lean_ctor_release(x_110, 0);
 x_117 = x_110;
} else {
 lean_dec_ref(x_110);
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
lean_object* x_119; lean_object* x_120; lean_object* x_121; 
lean_dec(x_15);
lean_dec(x_8);
lean_dec(x_6);
x_119 = lean_ctor_get(x_108, 0);
lean_inc(x_119);
if (lean_is_exclusive(x_108)) {
 lean_ctor_release(x_108, 0);
 x_120 = x_108;
} else {
 lean_dec_ref(x_108);
 x_120 = lean_box(0);
}
if (lean_is_scalar(x_120)) {
 x_121 = lean_alloc_ctor(1, 1, 0);
} else {
 x_121 = x_120;
}
lean_ctor_set(x_121, 0, x_119);
return x_121;
}
}
}
else
{
lean_object* x_122; lean_object* x_123; lean_object* x_124; 
lean_dec(x_15);
lean_dec(x_13);
lean_dec(x_8);
lean_dec(x_6);
x_122 = lean_ctor_get(x_100, 0);
lean_inc(x_122);
if (lean_is_exclusive(x_100)) {
 lean_ctor_release(x_100, 0);
 x_123 = x_100;
} else {
 lean_dec_ref(x_100);
 x_123 = lean_box(0);
}
if (lean_is_scalar(x_123)) {
 x_124 = lean_alloc_ctor(1, 1, 0);
} else {
 x_124 = x_123;
}
lean_ctor_set(x_124, 0, x_122);
return x_124;
}
}
}
else
{
uint8_t x_125; uint8_t x_126; uint8_t x_127; uint8_t x_128; uint8_t x_129; uint8_t x_130; uint8_t x_131; uint8_t x_132; uint8_t x_133; uint8_t x_134; uint8_t x_135; uint8_t x_136; uint8_t x_137; uint8_t x_138; uint8_t x_139; uint8_t x_140; uint8_t x_141; uint8_t x_142; uint8_t x_143; lean_object* x_144; lean_object* x_145; lean_object* x_146; lean_object* x_147; lean_object* x_148; lean_object* x_149; uint8_t x_150; uint8_t x_151; lean_object* x_152; lean_object* x_153; lean_object* x_154; uint8_t x_155; lean_object* x_156; uint64_t x_157; lean_object* x_158; uint64_t x_159; uint64_t x_160; uint64_t x_161; uint64_t x_162; uint64_t x_163; lean_object* x_164; lean_object* x_165; lean_object* x_166; 
x_125 = lean_ctor_get_uint8(x_16, 0);
x_126 = lean_ctor_get_uint8(x_16, 1);
x_127 = lean_ctor_get_uint8(x_16, 2);
x_128 = lean_ctor_get_uint8(x_16, 3);
x_129 = lean_ctor_get_uint8(x_16, 4);
x_130 = lean_ctor_get_uint8(x_16, 5);
x_131 = lean_ctor_get_uint8(x_16, 6);
x_132 = lean_ctor_get_uint8(x_16, 7);
x_133 = lean_ctor_get_uint8(x_16, 8);
x_134 = lean_ctor_get_uint8(x_16, 10);
x_135 = lean_ctor_get_uint8(x_16, 11);
x_136 = lean_ctor_get_uint8(x_16, 12);
x_137 = lean_ctor_get_uint8(x_16, 13);
x_138 = lean_ctor_get_uint8(x_16, 14);
x_139 = lean_ctor_get_uint8(x_16, 15);
x_140 = lean_ctor_get_uint8(x_16, 16);
x_141 = lean_ctor_get_uint8(x_16, 17);
x_142 = lean_ctor_get_uint8(x_16, 18);
lean_dec(x_16);
x_143 = lean_ctor_get_uint8(x_7, sizeof(void*)*7);
x_144 = lean_ctor_get(x_7, 1);
lean_inc(x_144);
x_145 = lean_ctor_get(x_7, 2);
lean_inc_ref(x_145);
x_146 = lean_ctor_get(x_7, 3);
lean_inc_ref(x_146);
x_147 = lean_ctor_get(x_7, 4);
lean_inc(x_147);
x_148 = lean_ctor_get(x_7, 5);
lean_inc(x_148);
x_149 = lean_ctor_get(x_7, 6);
lean_inc(x_149);
x_150 = lean_ctor_get_uint8(x_7, sizeof(void*)*7 + 1);
x_151 = lean_ctor_get_uint8(x_7, sizeof(void*)*7 + 2);
x_152 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0;
lean_inc(x_13);
x_153 = l_Lean_Expr_app___override(x_152, x_13);
lean_inc(x_15);
x_154 = l_Lean_Expr_app___override(x_153, x_15);
x_155 = 2;
x_156 = lean_alloc_ctor(0, 0, 19);
lean_ctor_set_uint8(x_156, 0, x_125);
lean_ctor_set_uint8(x_156, 1, x_126);
lean_ctor_set_uint8(x_156, 2, x_127);
lean_ctor_set_uint8(x_156, 3, x_128);
lean_ctor_set_uint8(x_156, 4, x_129);
lean_ctor_set_uint8(x_156, 5, x_130);
lean_ctor_set_uint8(x_156, 6, x_131);
lean_ctor_set_uint8(x_156, 7, x_132);
lean_ctor_set_uint8(x_156, 8, x_133);
lean_ctor_set_uint8(x_156, 9, x_155);
lean_ctor_set_uint8(x_156, 10, x_134);
lean_ctor_set_uint8(x_156, 11, x_135);
lean_ctor_set_uint8(x_156, 12, x_136);
lean_ctor_set_uint8(x_156, 13, x_137);
lean_ctor_set_uint8(x_156, 14, x_138);
lean_ctor_set_uint8(x_156, 15, x_139);
lean_ctor_set_uint8(x_156, 16, x_140);
lean_ctor_set_uint8(x_156, 17, x_141);
lean_ctor_set_uint8(x_156, 18, x_142);
x_157 = l_Lean_Meta_Context_configKey(x_7);
if (lean_is_exclusive(x_7)) {
 lean_ctor_release(x_7, 0);
 lean_ctor_release(x_7, 1);
 lean_ctor_release(x_7, 2);
 lean_ctor_release(x_7, 3);
 lean_ctor_release(x_7, 4);
 lean_ctor_release(x_7, 5);
 lean_ctor_release(x_7, 6);
 x_158 = x_7;
} else {
 lean_dec_ref(x_7);
 x_158 = lean_box(0);
}
x_159 = 2;
x_160 = lean_uint64_shift_right(x_157, x_159);
x_161 = lean_uint64_shift_left(x_160, x_159);
x_162 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_163 = lean_uint64_lor(x_161, x_162);
x_164 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_164, 0, x_156);
lean_ctor_set_uint64(x_164, sizeof(void*)*1, x_163);
if (lean_is_scalar(x_158)) {
 x_165 = lean_alloc_ctor(0, 7, 3);
} else {
 x_165 = x_158;
}
lean_ctor_set(x_165, 0, x_164);
lean_ctor_set(x_165, 1, x_144);
lean_ctor_set(x_165, 2, x_145);
lean_ctor_set(x_165, 3, x_146);
lean_ctor_set(x_165, 4, x_147);
lean_ctor_set(x_165, 5, x_148);
lean_ctor_set(x_165, 6, x_149);
lean_ctor_set_uint8(x_165, sizeof(void*)*7, x_143);
lean_ctor_set_uint8(x_165, sizeof(void*)*7 + 1, x_150);
lean_ctor_set_uint8(x_165, sizeof(void*)*7 + 2, x_151);
lean_inc(x_8);
x_166 = l_Lean_Meta_isExprDefEq(x_154, x_4, x_165, x_8, x_9, x_10);
if (lean_obj_tag(x_166) == 0)
{
lean_object* x_167; lean_object* x_168; uint8_t x_169; 
x_167 = lean_ctor_get(x_166, 0);
lean_inc(x_167);
if (lean_is_exclusive(x_166)) {
 lean_ctor_release(x_166, 0);
 x_168 = x_166;
} else {
 lean_dec_ref(x_166);
 x_168 = lean_box(0);
}
x_169 = lean_unbox(x_167);
lean_dec(x_167);
if (x_169 == 0)
{
lean_object* x_170; lean_object* x_171; lean_object* x_172; lean_object* x_173; 
lean_dec(x_8);
lean_dec(x_6);
x_170 = lean_box(x_5);
x_171 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_171, 0, x_15);
lean_ctor_set(x_171, 1, x_170);
x_172 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_172, 0, x_13);
lean_ctor_set(x_172, 1, x_171);
if (lean_is_scalar(x_168)) {
 x_173 = lean_alloc_ctor(0, 1, 0);
} else {
 x_173 = x_168;
}
lean_ctor_set(x_173, 0, x_172);
return x_173;
}
else
{
lean_object* x_174; 
lean_dec(x_168);
x_174 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_13, x_8);
if (lean_obj_tag(x_174) == 0)
{
lean_object* x_175; lean_object* x_176; 
x_175 = lean_ctor_get(x_174, 0);
lean_inc(x_175);
lean_dec_ref(x_174);
x_176 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_15, x_8);
lean_dec(x_8);
if (lean_obj_tag(x_176) == 0)
{
lean_object* x_177; lean_object* x_178; lean_object* x_179; lean_object* x_180; lean_object* x_181; 
x_177 = lean_ctor_get(x_176, 0);
lean_inc(x_177);
if (lean_is_exclusive(x_176)) {
 lean_ctor_release(x_176, 0);
 x_178 = x_176;
} else {
 lean_dec_ref(x_176);
 x_178 = lean_box(0);
}
x_179 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_179, 0, x_177);
lean_ctor_set(x_179, 1, x_6);
x_180 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_180, 0, x_175);
lean_ctor_set(x_180, 1, x_179);
if (lean_is_scalar(x_178)) {
 x_181 = lean_alloc_ctor(0, 1, 0);
} else {
 x_181 = x_178;
}
lean_ctor_set(x_181, 0, x_180);
return x_181;
}
else
{
lean_object* x_182; lean_object* x_183; lean_object* x_184; 
lean_dec(x_175);
lean_dec(x_6);
x_182 = lean_ctor_get(x_176, 0);
lean_inc(x_182);
if (lean_is_exclusive(x_176)) {
 lean_ctor_release(x_176, 0);
 x_183 = x_176;
} else {
 lean_dec_ref(x_176);
 x_183 = lean_box(0);
}
if (lean_is_scalar(x_183)) {
 x_184 = lean_alloc_ctor(1, 1, 0);
} else {
 x_184 = x_183;
}
lean_ctor_set(x_184, 0, x_182);
return x_184;
}
}
else
{
lean_object* x_185; lean_object* x_186; lean_object* x_187; 
lean_dec(x_15);
lean_dec(x_8);
lean_dec(x_6);
x_185 = lean_ctor_get(x_174, 0);
lean_inc(x_185);
if (lean_is_exclusive(x_174)) {
 lean_ctor_release(x_174, 0);
 x_186 = x_174;
} else {
 lean_dec_ref(x_174);
 x_186 = lean_box(0);
}
if (lean_is_scalar(x_186)) {
 x_187 = lean_alloc_ctor(1, 1, 0);
} else {
 x_187 = x_186;
}
lean_ctor_set(x_187, 0, x_185);
return x_187;
}
}
}
else
{
lean_object* x_188; lean_object* x_189; lean_object* x_190; 
lean_dec(x_15);
lean_dec(x_13);
lean_dec(x_8);
lean_dec(x_6);
x_188 = lean_ctor_get(x_166, 0);
lean_inc(x_188);
if (lean_is_exclusive(x_166)) {
 lean_ctor_release(x_166, 0);
 x_189 = x_166;
} else {
 lean_dec_ref(x_166);
 x_189 = lean_box(0);
}
if (lean_is_scalar(x_189)) {
 x_190 = lean_alloc_ctor(1, 1, 0);
} else {
 x_190 = x_189;
}
lean_ctor_set(x_190, 0, x_188);
return x_190;
}
}
}
else
{
uint8_t x_191; 
lean_dec(x_13);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_4);
x_191 = !lean_is_exclusive(x_14);
if (x_191 == 0)
{
return x_14;
}
else
{
lean_object* x_192; lean_object* x_193; 
x_192 = lean_ctor_get(x_14, 0);
lean_inc(x_192);
lean_dec(x_14);
x_193 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_193, 0, x_192);
return x_193;
}
}
}
else
{
uint8_t x_194; 
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec(x_1);
x_194 = !lean_is_exclusive(x_12);
if (x_194 == 0)
{
return x_12;
}
else
{
lean_object* x_195; lean_object* x_196; 
x_195 = lean_ctor_get(x_12, 0);
lean_inc(x_195);
lean_dec(x_12);
x_196 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_196, 0, x_195);
return x_196;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___lam__0(uint8_t x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, uint8_t x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; 
x_11 = l_Lean_Meta_mkFreshLevelMVar(x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_11) == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
lean_dec_ref(x_11);
lean_inc(x_12);
x_13 = l_Lean_Expr_sort___override(x_12);
x_14 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_14, 0, x_13);
lean_inc_ref(x_6);
lean_inc(x_2);
x_15 = l_Lean_Meta_mkFreshExprMVar(x_14, x_1, x_2, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_15) == 0)
{
lean_object* x_16; uint8_t x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_16 = lean_ctor_get(x_15, 0);
lean_inc(x_16);
lean_dec_ref(x_15);
x_17 = 0;
lean_inc(x_16);
lean_inc(x_2);
x_18 = l_Lean_Expr_forallE___override(x_2, x_16, x_3, x_17);
x_19 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_19, 0, x_18);
lean_inc_ref(x_6);
x_20 = l_Lean_Meta_mkFreshExprMVar(x_19, x_1, x_2, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_20) == 0)
{
lean_object* x_21; lean_object* x_22; uint8_t x_23; 
x_21 = lean_ctor_get(x_20, 0);
lean_inc(x_21);
lean_dec_ref(x_20);
x_22 = l_Lean_Meta_Context_config(x_6);
x_23 = !lean_is_exclusive(x_22);
if (x_23 == 0)
{
uint8_t x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; uint8_t x_31; uint8_t x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; uint8_t x_39; uint64_t x_40; uint8_t x_41; 
x_24 = lean_ctor_get_uint8(x_6, sizeof(void*)*7);
x_25 = lean_ctor_get(x_6, 1);
lean_inc(x_25);
x_26 = lean_ctor_get(x_6, 2);
lean_inc_ref(x_26);
x_27 = lean_ctor_get(x_6, 3);
lean_inc_ref(x_27);
x_28 = lean_ctor_get(x_6, 4);
lean_inc(x_28);
x_29 = lean_ctor_get(x_6, 5);
lean_inc(x_29);
x_30 = lean_ctor_get(x_6, 6);
lean_inc(x_30);
x_31 = lean_ctor_get_uint8(x_6, sizeof(void*)*7 + 1);
x_32 = lean_ctor_get_uint8(x_6, sizeof(void*)*7 + 2);
x_33 = lp_mathlib_ExistsAndEq_mkNestedExists___closed__1;
x_34 = lean_box(0);
lean_inc(x_12);
x_35 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_35, 0, x_12);
lean_ctor_set(x_35, 1, x_34);
x_36 = l_Lean_Expr_const___override(x_33, x_35);
lean_inc(x_16);
x_37 = l_Lean_Expr_app___override(x_36, x_16);
lean_inc(x_21);
x_38 = l_Lean_Expr_app___override(x_37, x_21);
x_39 = 2;
lean_ctor_set_uint8(x_22, 9, x_39);
x_40 = l_Lean_Meta_Context_configKey(x_6);
x_41 = !lean_is_exclusive(x_6);
if (x_41 == 0)
{
lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; uint64_t x_49; uint64_t x_50; uint64_t x_51; uint64_t x_52; uint64_t x_53; lean_object* x_54; lean_object* x_55; 
x_42 = lean_ctor_get(x_6, 6);
lean_dec(x_42);
x_43 = lean_ctor_get(x_6, 5);
lean_dec(x_43);
x_44 = lean_ctor_get(x_6, 4);
lean_dec(x_44);
x_45 = lean_ctor_get(x_6, 3);
lean_dec(x_45);
x_46 = lean_ctor_get(x_6, 2);
lean_dec(x_46);
x_47 = lean_ctor_get(x_6, 1);
lean_dec(x_47);
x_48 = lean_ctor_get(x_6, 0);
lean_dec(x_48);
x_49 = 2;
x_50 = lean_uint64_shift_right(x_40, x_49);
x_51 = lean_uint64_shift_left(x_50, x_49);
x_52 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_53 = lean_uint64_lor(x_51, x_52);
x_54 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_54, 0, x_22);
lean_ctor_set_uint64(x_54, sizeof(void*)*1, x_53);
lean_ctor_set(x_6, 0, x_54);
lean_inc(x_7);
x_55 = l_Lean_Meta_isExprDefEq(x_38, x_4, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_55) == 0)
{
uint8_t x_56; 
x_56 = !lean_is_exclusive(x_55);
if (x_56 == 0)
{
lean_object* x_57; uint8_t x_58; 
x_57 = lean_ctor_get(x_55, 0);
x_58 = lean_unbox(x_57);
if (x_58 == 0)
{
lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; 
lean_dec(x_57);
lean_dec(x_7);
x_59 = lean_box(x_5);
x_60 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_60, 0, x_21);
lean_ctor_set(x_60, 1, x_59);
x_61 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_61, 0, x_16);
lean_ctor_set(x_61, 1, x_60);
x_62 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_62, 0, x_12);
lean_ctor_set(x_62, 1, x_61);
lean_ctor_set(x_55, 0, x_62);
return x_55;
}
else
{
lean_object* x_63; 
lean_free_object(x_55);
x_63 = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg(x_12, x_7);
if (lean_obj_tag(x_63) == 0)
{
lean_object* x_64; lean_object* x_65; 
x_64 = lean_ctor_get(x_63, 0);
lean_inc(x_64);
lean_dec_ref(x_63);
x_65 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_16, x_7);
if (lean_obj_tag(x_65) == 0)
{
lean_object* x_66; lean_object* x_67; 
x_66 = lean_ctor_get(x_65, 0);
lean_inc(x_66);
lean_dec_ref(x_65);
x_67 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_21, x_7);
lean_dec(x_7);
if (lean_obj_tag(x_67) == 0)
{
uint8_t x_68; 
x_68 = !lean_is_exclusive(x_67);
if (x_68 == 0)
{
lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; 
x_69 = lean_ctor_get(x_67, 0);
x_70 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_70, 0, x_69);
lean_ctor_set(x_70, 1, x_57);
x_71 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_71, 0, x_66);
lean_ctor_set(x_71, 1, x_70);
x_72 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_72, 0, x_64);
lean_ctor_set(x_72, 1, x_71);
lean_ctor_set(x_67, 0, x_72);
return x_67;
}
else
{
lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; 
x_73 = lean_ctor_get(x_67, 0);
lean_inc(x_73);
lean_dec(x_67);
x_74 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_74, 0, x_73);
lean_ctor_set(x_74, 1, x_57);
x_75 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_75, 0, x_66);
lean_ctor_set(x_75, 1, x_74);
x_76 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_76, 0, x_64);
lean_ctor_set(x_76, 1, x_75);
x_77 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_77, 0, x_76);
return x_77;
}
}
else
{
uint8_t x_78; 
lean_dec(x_66);
lean_dec(x_64);
lean_dec(x_57);
x_78 = !lean_is_exclusive(x_67);
if (x_78 == 0)
{
return x_67;
}
else
{
lean_object* x_79; lean_object* x_80; 
x_79 = lean_ctor_get(x_67, 0);
lean_inc(x_79);
lean_dec(x_67);
x_80 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_80, 0, x_79);
return x_80;
}
}
}
else
{
uint8_t x_81; 
lean_dec(x_64);
lean_dec(x_57);
lean_dec(x_21);
lean_dec(x_7);
x_81 = !lean_is_exclusive(x_65);
if (x_81 == 0)
{
return x_65;
}
else
{
lean_object* x_82; lean_object* x_83; 
x_82 = lean_ctor_get(x_65, 0);
lean_inc(x_82);
lean_dec(x_65);
x_83 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_83, 0, x_82);
return x_83;
}
}
}
else
{
uint8_t x_84; 
lean_dec(x_57);
lean_dec(x_21);
lean_dec(x_16);
lean_dec(x_7);
x_84 = !lean_is_exclusive(x_63);
if (x_84 == 0)
{
return x_63;
}
else
{
lean_object* x_85; lean_object* x_86; 
x_85 = lean_ctor_get(x_63, 0);
lean_inc(x_85);
lean_dec(x_63);
x_86 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_86, 0, x_85);
return x_86;
}
}
}
}
else
{
lean_object* x_87; uint8_t x_88; 
x_87 = lean_ctor_get(x_55, 0);
lean_inc(x_87);
lean_dec(x_55);
x_88 = lean_unbox(x_87);
if (x_88 == 0)
{
lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; 
lean_dec(x_87);
lean_dec(x_7);
x_89 = lean_box(x_5);
x_90 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_90, 0, x_21);
lean_ctor_set(x_90, 1, x_89);
x_91 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_91, 0, x_16);
lean_ctor_set(x_91, 1, x_90);
x_92 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_92, 0, x_12);
lean_ctor_set(x_92, 1, x_91);
x_93 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_93, 0, x_92);
return x_93;
}
else
{
lean_object* x_94; 
x_94 = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg(x_12, x_7);
if (lean_obj_tag(x_94) == 0)
{
lean_object* x_95; lean_object* x_96; 
x_95 = lean_ctor_get(x_94, 0);
lean_inc(x_95);
lean_dec_ref(x_94);
x_96 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_16, x_7);
if (lean_obj_tag(x_96) == 0)
{
lean_object* x_97; lean_object* x_98; 
x_97 = lean_ctor_get(x_96, 0);
lean_inc(x_97);
lean_dec_ref(x_96);
x_98 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_21, x_7);
lean_dec(x_7);
if (lean_obj_tag(x_98) == 0)
{
lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; 
x_99 = lean_ctor_get(x_98, 0);
lean_inc(x_99);
if (lean_is_exclusive(x_98)) {
 lean_ctor_release(x_98, 0);
 x_100 = x_98;
} else {
 lean_dec_ref(x_98);
 x_100 = lean_box(0);
}
x_101 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_101, 0, x_99);
lean_ctor_set(x_101, 1, x_87);
x_102 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_102, 0, x_97);
lean_ctor_set(x_102, 1, x_101);
x_103 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_103, 0, x_95);
lean_ctor_set(x_103, 1, x_102);
if (lean_is_scalar(x_100)) {
 x_104 = lean_alloc_ctor(0, 1, 0);
} else {
 x_104 = x_100;
}
lean_ctor_set(x_104, 0, x_103);
return x_104;
}
else
{
lean_object* x_105; lean_object* x_106; lean_object* x_107; 
lean_dec(x_97);
lean_dec(x_95);
lean_dec(x_87);
x_105 = lean_ctor_get(x_98, 0);
lean_inc(x_105);
if (lean_is_exclusive(x_98)) {
 lean_ctor_release(x_98, 0);
 x_106 = x_98;
} else {
 lean_dec_ref(x_98);
 x_106 = lean_box(0);
}
if (lean_is_scalar(x_106)) {
 x_107 = lean_alloc_ctor(1, 1, 0);
} else {
 x_107 = x_106;
}
lean_ctor_set(x_107, 0, x_105);
return x_107;
}
}
else
{
lean_object* x_108; lean_object* x_109; lean_object* x_110; 
lean_dec(x_95);
lean_dec(x_87);
lean_dec(x_21);
lean_dec(x_7);
x_108 = lean_ctor_get(x_96, 0);
lean_inc(x_108);
if (lean_is_exclusive(x_96)) {
 lean_ctor_release(x_96, 0);
 x_109 = x_96;
} else {
 lean_dec_ref(x_96);
 x_109 = lean_box(0);
}
if (lean_is_scalar(x_109)) {
 x_110 = lean_alloc_ctor(1, 1, 0);
} else {
 x_110 = x_109;
}
lean_ctor_set(x_110, 0, x_108);
return x_110;
}
}
else
{
lean_object* x_111; lean_object* x_112; lean_object* x_113; 
lean_dec(x_87);
lean_dec(x_21);
lean_dec(x_16);
lean_dec(x_7);
x_111 = lean_ctor_get(x_94, 0);
lean_inc(x_111);
if (lean_is_exclusive(x_94)) {
 lean_ctor_release(x_94, 0);
 x_112 = x_94;
} else {
 lean_dec_ref(x_94);
 x_112 = lean_box(0);
}
if (lean_is_scalar(x_112)) {
 x_113 = lean_alloc_ctor(1, 1, 0);
} else {
 x_113 = x_112;
}
lean_ctor_set(x_113, 0, x_111);
return x_113;
}
}
}
}
else
{
uint8_t x_114; 
lean_dec(x_21);
lean_dec(x_16);
lean_dec(x_12);
lean_dec(x_7);
x_114 = !lean_is_exclusive(x_55);
if (x_114 == 0)
{
return x_55;
}
else
{
lean_object* x_115; lean_object* x_116; 
x_115 = lean_ctor_get(x_55, 0);
lean_inc(x_115);
lean_dec(x_55);
x_116 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_116, 0, x_115);
return x_116;
}
}
}
else
{
uint64_t x_117; uint64_t x_118; uint64_t x_119; uint64_t x_120; uint64_t x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; 
lean_dec(x_6);
x_117 = 2;
x_118 = lean_uint64_shift_right(x_40, x_117);
x_119 = lean_uint64_shift_left(x_118, x_117);
x_120 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_121 = lean_uint64_lor(x_119, x_120);
x_122 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_122, 0, x_22);
lean_ctor_set_uint64(x_122, sizeof(void*)*1, x_121);
x_123 = lean_alloc_ctor(0, 7, 3);
lean_ctor_set(x_123, 0, x_122);
lean_ctor_set(x_123, 1, x_25);
lean_ctor_set(x_123, 2, x_26);
lean_ctor_set(x_123, 3, x_27);
lean_ctor_set(x_123, 4, x_28);
lean_ctor_set(x_123, 5, x_29);
lean_ctor_set(x_123, 6, x_30);
lean_ctor_set_uint8(x_123, sizeof(void*)*7, x_24);
lean_ctor_set_uint8(x_123, sizeof(void*)*7 + 1, x_31);
lean_ctor_set_uint8(x_123, sizeof(void*)*7 + 2, x_32);
lean_inc(x_7);
x_124 = l_Lean_Meta_isExprDefEq(x_38, x_4, x_123, x_7, x_8, x_9);
if (lean_obj_tag(x_124) == 0)
{
lean_object* x_125; lean_object* x_126; uint8_t x_127; 
x_125 = lean_ctor_get(x_124, 0);
lean_inc(x_125);
if (lean_is_exclusive(x_124)) {
 lean_ctor_release(x_124, 0);
 x_126 = x_124;
} else {
 lean_dec_ref(x_124);
 x_126 = lean_box(0);
}
x_127 = lean_unbox(x_125);
if (x_127 == 0)
{
lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; lean_object* x_132; 
lean_dec(x_125);
lean_dec(x_7);
x_128 = lean_box(x_5);
x_129 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_129, 0, x_21);
lean_ctor_set(x_129, 1, x_128);
x_130 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_130, 0, x_16);
lean_ctor_set(x_130, 1, x_129);
x_131 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_131, 0, x_12);
lean_ctor_set(x_131, 1, x_130);
if (lean_is_scalar(x_126)) {
 x_132 = lean_alloc_ctor(0, 1, 0);
} else {
 x_132 = x_126;
}
lean_ctor_set(x_132, 0, x_131);
return x_132;
}
else
{
lean_object* x_133; 
lean_dec(x_126);
x_133 = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg(x_12, x_7);
if (lean_obj_tag(x_133) == 0)
{
lean_object* x_134; lean_object* x_135; 
x_134 = lean_ctor_get(x_133, 0);
lean_inc(x_134);
lean_dec_ref(x_133);
x_135 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_16, x_7);
if (lean_obj_tag(x_135) == 0)
{
lean_object* x_136; lean_object* x_137; 
x_136 = lean_ctor_get(x_135, 0);
lean_inc(x_136);
lean_dec_ref(x_135);
x_137 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_21, x_7);
lean_dec(x_7);
if (lean_obj_tag(x_137) == 0)
{
lean_object* x_138; lean_object* x_139; lean_object* x_140; lean_object* x_141; lean_object* x_142; lean_object* x_143; 
x_138 = lean_ctor_get(x_137, 0);
lean_inc(x_138);
if (lean_is_exclusive(x_137)) {
 lean_ctor_release(x_137, 0);
 x_139 = x_137;
} else {
 lean_dec_ref(x_137);
 x_139 = lean_box(0);
}
x_140 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_140, 0, x_138);
lean_ctor_set(x_140, 1, x_125);
x_141 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_141, 0, x_136);
lean_ctor_set(x_141, 1, x_140);
x_142 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_142, 0, x_134);
lean_ctor_set(x_142, 1, x_141);
if (lean_is_scalar(x_139)) {
 x_143 = lean_alloc_ctor(0, 1, 0);
} else {
 x_143 = x_139;
}
lean_ctor_set(x_143, 0, x_142);
return x_143;
}
else
{
lean_object* x_144; lean_object* x_145; lean_object* x_146; 
lean_dec(x_136);
lean_dec(x_134);
lean_dec(x_125);
x_144 = lean_ctor_get(x_137, 0);
lean_inc(x_144);
if (lean_is_exclusive(x_137)) {
 lean_ctor_release(x_137, 0);
 x_145 = x_137;
} else {
 lean_dec_ref(x_137);
 x_145 = lean_box(0);
}
if (lean_is_scalar(x_145)) {
 x_146 = lean_alloc_ctor(1, 1, 0);
} else {
 x_146 = x_145;
}
lean_ctor_set(x_146, 0, x_144);
return x_146;
}
}
else
{
lean_object* x_147; lean_object* x_148; lean_object* x_149; 
lean_dec(x_134);
lean_dec(x_125);
lean_dec(x_21);
lean_dec(x_7);
x_147 = lean_ctor_get(x_135, 0);
lean_inc(x_147);
if (lean_is_exclusive(x_135)) {
 lean_ctor_release(x_135, 0);
 x_148 = x_135;
} else {
 lean_dec_ref(x_135);
 x_148 = lean_box(0);
}
if (lean_is_scalar(x_148)) {
 x_149 = lean_alloc_ctor(1, 1, 0);
} else {
 x_149 = x_148;
}
lean_ctor_set(x_149, 0, x_147);
return x_149;
}
}
else
{
lean_object* x_150; lean_object* x_151; lean_object* x_152; 
lean_dec(x_125);
lean_dec(x_21);
lean_dec(x_16);
lean_dec(x_7);
x_150 = lean_ctor_get(x_133, 0);
lean_inc(x_150);
if (lean_is_exclusive(x_133)) {
 lean_ctor_release(x_133, 0);
 x_151 = x_133;
} else {
 lean_dec_ref(x_133);
 x_151 = lean_box(0);
}
if (lean_is_scalar(x_151)) {
 x_152 = lean_alloc_ctor(1, 1, 0);
} else {
 x_152 = x_151;
}
lean_ctor_set(x_152, 0, x_150);
return x_152;
}
}
}
else
{
lean_object* x_153; lean_object* x_154; lean_object* x_155; 
lean_dec(x_21);
lean_dec(x_16);
lean_dec(x_12);
lean_dec(x_7);
x_153 = lean_ctor_get(x_124, 0);
lean_inc(x_153);
if (lean_is_exclusive(x_124)) {
 lean_ctor_release(x_124, 0);
 x_154 = x_124;
} else {
 lean_dec_ref(x_124);
 x_154 = lean_box(0);
}
if (lean_is_scalar(x_154)) {
 x_155 = lean_alloc_ctor(1, 1, 0);
} else {
 x_155 = x_154;
}
lean_ctor_set(x_155, 0, x_153);
return x_155;
}
}
}
else
{
uint8_t x_156; uint8_t x_157; uint8_t x_158; uint8_t x_159; uint8_t x_160; uint8_t x_161; uint8_t x_162; uint8_t x_163; uint8_t x_164; uint8_t x_165; uint8_t x_166; uint8_t x_167; uint8_t x_168; uint8_t x_169; uint8_t x_170; uint8_t x_171; uint8_t x_172; uint8_t x_173; uint8_t x_174; lean_object* x_175; lean_object* x_176; lean_object* x_177; lean_object* x_178; lean_object* x_179; lean_object* x_180; uint8_t x_181; uint8_t x_182; lean_object* x_183; lean_object* x_184; lean_object* x_185; lean_object* x_186; lean_object* x_187; lean_object* x_188; uint8_t x_189; lean_object* x_190; uint64_t x_191; lean_object* x_192; uint64_t x_193; uint64_t x_194; uint64_t x_195; uint64_t x_196; uint64_t x_197; lean_object* x_198; lean_object* x_199; lean_object* x_200; 
x_156 = lean_ctor_get_uint8(x_22, 0);
x_157 = lean_ctor_get_uint8(x_22, 1);
x_158 = lean_ctor_get_uint8(x_22, 2);
x_159 = lean_ctor_get_uint8(x_22, 3);
x_160 = lean_ctor_get_uint8(x_22, 4);
x_161 = lean_ctor_get_uint8(x_22, 5);
x_162 = lean_ctor_get_uint8(x_22, 6);
x_163 = lean_ctor_get_uint8(x_22, 7);
x_164 = lean_ctor_get_uint8(x_22, 8);
x_165 = lean_ctor_get_uint8(x_22, 10);
x_166 = lean_ctor_get_uint8(x_22, 11);
x_167 = lean_ctor_get_uint8(x_22, 12);
x_168 = lean_ctor_get_uint8(x_22, 13);
x_169 = lean_ctor_get_uint8(x_22, 14);
x_170 = lean_ctor_get_uint8(x_22, 15);
x_171 = lean_ctor_get_uint8(x_22, 16);
x_172 = lean_ctor_get_uint8(x_22, 17);
x_173 = lean_ctor_get_uint8(x_22, 18);
lean_dec(x_22);
x_174 = lean_ctor_get_uint8(x_6, sizeof(void*)*7);
x_175 = lean_ctor_get(x_6, 1);
lean_inc(x_175);
x_176 = lean_ctor_get(x_6, 2);
lean_inc_ref(x_176);
x_177 = lean_ctor_get(x_6, 3);
lean_inc_ref(x_177);
x_178 = lean_ctor_get(x_6, 4);
lean_inc(x_178);
x_179 = lean_ctor_get(x_6, 5);
lean_inc(x_179);
x_180 = lean_ctor_get(x_6, 6);
lean_inc(x_180);
x_181 = lean_ctor_get_uint8(x_6, sizeof(void*)*7 + 1);
x_182 = lean_ctor_get_uint8(x_6, sizeof(void*)*7 + 2);
x_183 = lp_mathlib_ExistsAndEq_mkNestedExists___closed__1;
x_184 = lean_box(0);
lean_inc(x_12);
x_185 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_185, 0, x_12);
lean_ctor_set(x_185, 1, x_184);
x_186 = l_Lean_Expr_const___override(x_183, x_185);
lean_inc(x_16);
x_187 = l_Lean_Expr_app___override(x_186, x_16);
lean_inc(x_21);
x_188 = l_Lean_Expr_app___override(x_187, x_21);
x_189 = 2;
x_190 = lean_alloc_ctor(0, 0, 19);
lean_ctor_set_uint8(x_190, 0, x_156);
lean_ctor_set_uint8(x_190, 1, x_157);
lean_ctor_set_uint8(x_190, 2, x_158);
lean_ctor_set_uint8(x_190, 3, x_159);
lean_ctor_set_uint8(x_190, 4, x_160);
lean_ctor_set_uint8(x_190, 5, x_161);
lean_ctor_set_uint8(x_190, 6, x_162);
lean_ctor_set_uint8(x_190, 7, x_163);
lean_ctor_set_uint8(x_190, 8, x_164);
lean_ctor_set_uint8(x_190, 9, x_189);
lean_ctor_set_uint8(x_190, 10, x_165);
lean_ctor_set_uint8(x_190, 11, x_166);
lean_ctor_set_uint8(x_190, 12, x_167);
lean_ctor_set_uint8(x_190, 13, x_168);
lean_ctor_set_uint8(x_190, 14, x_169);
lean_ctor_set_uint8(x_190, 15, x_170);
lean_ctor_set_uint8(x_190, 16, x_171);
lean_ctor_set_uint8(x_190, 17, x_172);
lean_ctor_set_uint8(x_190, 18, x_173);
x_191 = l_Lean_Meta_Context_configKey(x_6);
if (lean_is_exclusive(x_6)) {
 lean_ctor_release(x_6, 0);
 lean_ctor_release(x_6, 1);
 lean_ctor_release(x_6, 2);
 lean_ctor_release(x_6, 3);
 lean_ctor_release(x_6, 4);
 lean_ctor_release(x_6, 5);
 lean_ctor_release(x_6, 6);
 x_192 = x_6;
} else {
 lean_dec_ref(x_6);
 x_192 = lean_box(0);
}
x_193 = 2;
x_194 = lean_uint64_shift_right(x_191, x_193);
x_195 = lean_uint64_shift_left(x_194, x_193);
x_196 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_197 = lean_uint64_lor(x_195, x_196);
x_198 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_198, 0, x_190);
lean_ctor_set_uint64(x_198, sizeof(void*)*1, x_197);
if (lean_is_scalar(x_192)) {
 x_199 = lean_alloc_ctor(0, 7, 3);
} else {
 x_199 = x_192;
}
lean_ctor_set(x_199, 0, x_198);
lean_ctor_set(x_199, 1, x_175);
lean_ctor_set(x_199, 2, x_176);
lean_ctor_set(x_199, 3, x_177);
lean_ctor_set(x_199, 4, x_178);
lean_ctor_set(x_199, 5, x_179);
lean_ctor_set(x_199, 6, x_180);
lean_ctor_set_uint8(x_199, sizeof(void*)*7, x_174);
lean_ctor_set_uint8(x_199, sizeof(void*)*7 + 1, x_181);
lean_ctor_set_uint8(x_199, sizeof(void*)*7 + 2, x_182);
lean_inc(x_7);
x_200 = l_Lean_Meta_isExprDefEq(x_188, x_4, x_199, x_7, x_8, x_9);
if (lean_obj_tag(x_200) == 0)
{
lean_object* x_201; lean_object* x_202; uint8_t x_203; 
x_201 = lean_ctor_get(x_200, 0);
lean_inc(x_201);
if (lean_is_exclusive(x_200)) {
 lean_ctor_release(x_200, 0);
 x_202 = x_200;
} else {
 lean_dec_ref(x_200);
 x_202 = lean_box(0);
}
x_203 = lean_unbox(x_201);
if (x_203 == 0)
{
lean_object* x_204; lean_object* x_205; lean_object* x_206; lean_object* x_207; lean_object* x_208; 
lean_dec(x_201);
lean_dec(x_7);
x_204 = lean_box(x_5);
x_205 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_205, 0, x_21);
lean_ctor_set(x_205, 1, x_204);
x_206 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_206, 0, x_16);
lean_ctor_set(x_206, 1, x_205);
x_207 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_207, 0, x_12);
lean_ctor_set(x_207, 1, x_206);
if (lean_is_scalar(x_202)) {
 x_208 = lean_alloc_ctor(0, 1, 0);
} else {
 x_208 = x_202;
}
lean_ctor_set(x_208, 0, x_207);
return x_208;
}
else
{
lean_object* x_209; 
lean_dec(x_202);
x_209 = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg(x_12, x_7);
if (lean_obj_tag(x_209) == 0)
{
lean_object* x_210; lean_object* x_211; 
x_210 = lean_ctor_get(x_209, 0);
lean_inc(x_210);
lean_dec_ref(x_209);
x_211 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_16, x_7);
if (lean_obj_tag(x_211) == 0)
{
lean_object* x_212; lean_object* x_213; 
x_212 = lean_ctor_get(x_211, 0);
lean_inc(x_212);
lean_dec_ref(x_211);
x_213 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_21, x_7);
lean_dec(x_7);
if (lean_obj_tag(x_213) == 0)
{
lean_object* x_214; lean_object* x_215; lean_object* x_216; lean_object* x_217; lean_object* x_218; lean_object* x_219; 
x_214 = lean_ctor_get(x_213, 0);
lean_inc(x_214);
if (lean_is_exclusive(x_213)) {
 lean_ctor_release(x_213, 0);
 x_215 = x_213;
} else {
 lean_dec_ref(x_213);
 x_215 = lean_box(0);
}
x_216 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_216, 0, x_214);
lean_ctor_set(x_216, 1, x_201);
x_217 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_217, 0, x_212);
lean_ctor_set(x_217, 1, x_216);
x_218 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_218, 0, x_210);
lean_ctor_set(x_218, 1, x_217);
if (lean_is_scalar(x_215)) {
 x_219 = lean_alloc_ctor(0, 1, 0);
} else {
 x_219 = x_215;
}
lean_ctor_set(x_219, 0, x_218);
return x_219;
}
else
{
lean_object* x_220; lean_object* x_221; lean_object* x_222; 
lean_dec(x_212);
lean_dec(x_210);
lean_dec(x_201);
x_220 = lean_ctor_get(x_213, 0);
lean_inc(x_220);
if (lean_is_exclusive(x_213)) {
 lean_ctor_release(x_213, 0);
 x_221 = x_213;
} else {
 lean_dec_ref(x_213);
 x_221 = lean_box(0);
}
if (lean_is_scalar(x_221)) {
 x_222 = lean_alloc_ctor(1, 1, 0);
} else {
 x_222 = x_221;
}
lean_ctor_set(x_222, 0, x_220);
return x_222;
}
}
else
{
lean_object* x_223; lean_object* x_224; lean_object* x_225; 
lean_dec(x_210);
lean_dec(x_201);
lean_dec(x_21);
lean_dec(x_7);
x_223 = lean_ctor_get(x_211, 0);
lean_inc(x_223);
if (lean_is_exclusive(x_211)) {
 lean_ctor_release(x_211, 0);
 x_224 = x_211;
} else {
 lean_dec_ref(x_211);
 x_224 = lean_box(0);
}
if (lean_is_scalar(x_224)) {
 x_225 = lean_alloc_ctor(1, 1, 0);
} else {
 x_225 = x_224;
}
lean_ctor_set(x_225, 0, x_223);
return x_225;
}
}
else
{
lean_object* x_226; lean_object* x_227; lean_object* x_228; 
lean_dec(x_201);
lean_dec(x_21);
lean_dec(x_16);
lean_dec(x_7);
x_226 = lean_ctor_get(x_209, 0);
lean_inc(x_226);
if (lean_is_exclusive(x_209)) {
 lean_ctor_release(x_209, 0);
 x_227 = x_209;
} else {
 lean_dec_ref(x_209);
 x_227 = lean_box(0);
}
if (lean_is_scalar(x_227)) {
 x_228 = lean_alloc_ctor(1, 1, 0);
} else {
 x_228 = x_227;
}
lean_ctor_set(x_228, 0, x_226);
return x_228;
}
}
}
else
{
lean_object* x_229; lean_object* x_230; lean_object* x_231; 
lean_dec(x_21);
lean_dec(x_16);
lean_dec(x_12);
lean_dec(x_7);
x_229 = lean_ctor_get(x_200, 0);
lean_inc(x_229);
if (lean_is_exclusive(x_200)) {
 lean_ctor_release(x_200, 0);
 x_230 = x_200;
} else {
 lean_dec_ref(x_200);
 x_230 = lean_box(0);
}
if (lean_is_scalar(x_230)) {
 x_231 = lean_alloc_ctor(1, 1, 0);
} else {
 x_231 = x_230;
}
lean_ctor_set(x_231, 0, x_229);
return x_231;
}
}
}
else
{
uint8_t x_232; 
lean_dec(x_16);
lean_dec(x_12);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_4);
x_232 = !lean_is_exclusive(x_20);
if (x_232 == 0)
{
return x_20;
}
else
{
lean_object* x_233; lean_object* x_234; 
x_233 = lean_ctor_get(x_20, 0);
lean_inc(x_233);
lean_dec(x_20);
x_234 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_234, 0, x_233);
return x_234;
}
}
}
else
{
uint8_t x_235; 
lean_dec(x_12);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
x_235 = !lean_is_exclusive(x_15);
if (x_235 == 0)
{
return x_15;
}
else
{
lean_object* x_236; lean_object* x_237; 
x_236 = lean_ctor_get(x_15, 0);
lean_inc(x_236);
lean_dec(x_15);
x_237 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_237, 0, x_236);
return x_237;
}
}
}
else
{
uint8_t x_238; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
x_238 = !lean_is_exclusive(x_11);
if (x_238 == 0)
{
return x_11;
}
else
{
lean_object* x_239; lean_object* x_240; 
x_239 = lean_ctor_get(x_11, 0);
lean_inc(x_239);
lean_dec(x_11);
x_240 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_240, 0, x_239);
return x_240;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; 
x_7 = l_Lean_Meta_mkFreshLevelMVar(x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_7) == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; lean_object* x_12; lean_object* x_13; 
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
lean_dec_ref(x_7);
lean_inc(x_8);
x_9 = l_Lean_Expr_sort___override(x_8);
x_10 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_10, 0, x_9);
x_11 = 0;
x_12 = lean_box(0);
lean_inc_ref(x_2);
x_13 = l_Lean_Meta_mkFreshExprMVar(x_10, x_11, x_12, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_13) == 0)
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
lean_dec_ref(x_13);
lean_inc(x_14);
x_15 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_15, 0, x_14);
lean_inc_ref(x_2);
lean_inc_ref(x_15);
x_16 = l_Lean_Meta_mkFreshExprMVar(x_15, x_11, x_12, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_16) == 0)
{
lean_object* x_17; lean_object* x_18; 
x_17 = lean_ctor_get(x_16, 0);
lean_inc(x_17);
lean_dec_ref(x_16);
lean_inc_ref(x_2);
x_18 = l_Lean_Meta_mkFreshExprMVar(x_15, x_11, x_12, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_18) == 0)
{
lean_object* x_19; lean_object* x_20; uint8_t x_21; 
x_19 = lean_ctor_get(x_18, 0);
lean_inc(x_19);
lean_dec_ref(x_18);
x_20 = l_Lean_Meta_Context_config(x_2);
x_21 = !lean_is_exclusive(x_20);
if (x_21 == 0)
{
uint8_t x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; uint8_t x_29; uint8_t x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; uint8_t x_38; uint64_t x_39; uint8_t x_40; 
x_22 = lean_ctor_get_uint8(x_2, sizeof(void*)*7);
x_23 = lean_ctor_get(x_2, 1);
lean_inc(x_23);
x_24 = lean_ctor_get(x_2, 2);
lean_inc_ref(x_24);
x_25 = lean_ctor_get(x_2, 3);
lean_inc_ref(x_25);
x_26 = lean_ctor_get(x_2, 4);
lean_inc(x_26);
x_27 = lean_ctor_get(x_2, 5);
lean_inc(x_27);
x_28 = lean_ctor_get(x_2, 6);
lean_inc(x_28);
x_29 = lean_ctor_get_uint8(x_2, sizeof(void*)*7 + 1);
x_30 = lean_ctor_get_uint8(x_2, sizeof(void*)*7 + 2);
x_31 = lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__4;
x_32 = lean_box(0);
lean_inc(x_8);
x_33 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_33, 0, x_8);
lean_ctor_set(x_33, 1, x_32);
x_34 = l_Lean_Expr_const___override(x_31, x_33);
lean_inc(x_14);
x_35 = l_Lean_Expr_app___override(x_34, x_14);
lean_inc(x_17);
x_36 = l_Lean_Expr_app___override(x_35, x_17);
lean_inc(x_19);
x_37 = l_Lean_Expr_app___override(x_36, x_19);
x_38 = 2;
lean_ctor_set_uint8(x_20, 9, x_38);
x_39 = l_Lean_Meta_Context_configKey(x_2);
x_40 = !lean_is_exclusive(x_2);
if (x_40 == 0)
{
lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; uint64_t x_48; uint64_t x_49; uint64_t x_50; uint64_t x_51; uint64_t x_52; lean_object* x_53; lean_object* x_54; 
x_41 = lean_ctor_get(x_2, 6);
lean_dec(x_41);
x_42 = lean_ctor_get(x_2, 5);
lean_dec(x_42);
x_43 = lean_ctor_get(x_2, 4);
lean_dec(x_43);
x_44 = lean_ctor_get(x_2, 3);
lean_dec(x_44);
x_45 = lean_ctor_get(x_2, 2);
lean_dec(x_45);
x_46 = lean_ctor_get(x_2, 1);
lean_dec(x_46);
x_47 = lean_ctor_get(x_2, 0);
lean_dec(x_47);
x_48 = 2;
x_49 = lean_uint64_shift_right(x_39, x_48);
x_50 = lean_uint64_shift_left(x_49, x_48);
x_51 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_52 = lean_uint64_lor(x_50, x_51);
x_53 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_53, 0, x_20);
lean_ctor_set_uint64(x_53, sizeof(void*)*1, x_52);
lean_ctor_set(x_2, 0, x_53);
lean_inc(x_3);
x_54 = l_Lean_Meta_isExprDefEq(x_37, x_1, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_54) == 0)
{
uint8_t x_55; 
x_55 = !lean_is_exclusive(x_54);
if (x_55 == 0)
{
lean_object* x_56; uint8_t x_57; 
x_56 = lean_ctor_get(x_54, 0);
x_57 = lean_unbox(x_56);
if (x_57 == 0)
{
lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; 
lean_dec(x_3);
x_58 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_58, 0, x_19);
lean_ctor_set(x_58, 1, x_56);
x_59 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_59, 0, x_17);
lean_ctor_set(x_59, 1, x_58);
x_60 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_60, 0, x_14);
lean_ctor_set(x_60, 1, x_59);
x_61 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_61, 0, x_8);
lean_ctor_set(x_61, 1, x_60);
lean_ctor_set(x_54, 0, x_61);
return x_54;
}
else
{
lean_object* x_62; 
lean_free_object(x_54);
x_62 = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg(x_8, x_3);
if (lean_obj_tag(x_62) == 0)
{
lean_object* x_63; lean_object* x_64; 
x_63 = lean_ctor_get(x_62, 0);
lean_inc(x_63);
lean_dec_ref(x_62);
x_64 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_14, x_3);
if (lean_obj_tag(x_64) == 0)
{
lean_object* x_65; lean_object* x_66; 
x_65 = lean_ctor_get(x_64, 0);
lean_inc(x_65);
lean_dec_ref(x_64);
x_66 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_17, x_3);
if (lean_obj_tag(x_66) == 0)
{
lean_object* x_67; lean_object* x_68; 
x_67 = lean_ctor_get(x_66, 0);
lean_inc(x_67);
lean_dec_ref(x_66);
x_68 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_19, x_3);
lean_dec(x_3);
if (lean_obj_tag(x_68) == 0)
{
uint8_t x_69; 
x_69 = !lean_is_exclusive(x_68);
if (x_69 == 0)
{
lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; 
x_70 = lean_ctor_get(x_68, 0);
x_71 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_71, 0, x_70);
lean_ctor_set(x_71, 1, x_56);
x_72 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_72, 0, x_67);
lean_ctor_set(x_72, 1, x_71);
x_73 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_73, 0, x_65);
lean_ctor_set(x_73, 1, x_72);
x_74 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_74, 0, x_63);
lean_ctor_set(x_74, 1, x_73);
lean_ctor_set(x_68, 0, x_74);
return x_68;
}
else
{
lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; 
x_75 = lean_ctor_get(x_68, 0);
lean_inc(x_75);
lean_dec(x_68);
x_76 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_76, 0, x_75);
lean_ctor_set(x_76, 1, x_56);
x_77 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_77, 0, x_67);
lean_ctor_set(x_77, 1, x_76);
x_78 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_78, 0, x_65);
lean_ctor_set(x_78, 1, x_77);
x_79 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_79, 0, x_63);
lean_ctor_set(x_79, 1, x_78);
x_80 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_80, 0, x_79);
return x_80;
}
}
else
{
uint8_t x_81; 
lean_dec(x_67);
lean_dec(x_65);
lean_dec(x_63);
lean_dec(x_56);
x_81 = !lean_is_exclusive(x_68);
if (x_81 == 0)
{
return x_68;
}
else
{
lean_object* x_82; lean_object* x_83; 
x_82 = lean_ctor_get(x_68, 0);
lean_inc(x_82);
lean_dec(x_68);
x_83 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_83, 0, x_82);
return x_83;
}
}
}
else
{
uint8_t x_84; 
lean_dec(x_65);
lean_dec(x_63);
lean_dec(x_56);
lean_dec(x_19);
lean_dec(x_3);
x_84 = !lean_is_exclusive(x_66);
if (x_84 == 0)
{
return x_66;
}
else
{
lean_object* x_85; lean_object* x_86; 
x_85 = lean_ctor_get(x_66, 0);
lean_inc(x_85);
lean_dec(x_66);
x_86 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_86, 0, x_85);
return x_86;
}
}
}
else
{
uint8_t x_87; 
lean_dec(x_63);
lean_dec(x_56);
lean_dec(x_19);
lean_dec(x_17);
lean_dec(x_3);
x_87 = !lean_is_exclusive(x_64);
if (x_87 == 0)
{
return x_64;
}
else
{
lean_object* x_88; lean_object* x_89; 
x_88 = lean_ctor_get(x_64, 0);
lean_inc(x_88);
lean_dec(x_64);
x_89 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_89, 0, x_88);
return x_89;
}
}
}
else
{
uint8_t x_90; 
lean_dec(x_56);
lean_dec(x_19);
lean_dec(x_17);
lean_dec(x_14);
lean_dec(x_3);
x_90 = !lean_is_exclusive(x_62);
if (x_90 == 0)
{
return x_62;
}
else
{
lean_object* x_91; lean_object* x_92; 
x_91 = lean_ctor_get(x_62, 0);
lean_inc(x_91);
lean_dec(x_62);
x_92 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_92, 0, x_91);
return x_92;
}
}
}
}
else
{
lean_object* x_93; uint8_t x_94; 
x_93 = lean_ctor_get(x_54, 0);
lean_inc(x_93);
lean_dec(x_54);
x_94 = lean_unbox(x_93);
if (x_94 == 0)
{
lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; 
lean_dec(x_3);
x_95 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_95, 0, x_19);
lean_ctor_set(x_95, 1, x_93);
x_96 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_96, 0, x_17);
lean_ctor_set(x_96, 1, x_95);
x_97 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_97, 0, x_14);
lean_ctor_set(x_97, 1, x_96);
x_98 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_98, 0, x_8);
lean_ctor_set(x_98, 1, x_97);
x_99 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_99, 0, x_98);
return x_99;
}
else
{
lean_object* x_100; 
x_100 = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg(x_8, x_3);
if (lean_obj_tag(x_100) == 0)
{
lean_object* x_101; lean_object* x_102; 
x_101 = lean_ctor_get(x_100, 0);
lean_inc(x_101);
lean_dec_ref(x_100);
x_102 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_14, x_3);
if (lean_obj_tag(x_102) == 0)
{
lean_object* x_103; lean_object* x_104; 
x_103 = lean_ctor_get(x_102, 0);
lean_inc(x_103);
lean_dec_ref(x_102);
x_104 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_17, x_3);
if (lean_obj_tag(x_104) == 0)
{
lean_object* x_105; lean_object* x_106; 
x_105 = lean_ctor_get(x_104, 0);
lean_inc(x_105);
lean_dec_ref(x_104);
x_106 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_19, x_3);
lean_dec(x_3);
if (lean_obj_tag(x_106) == 0)
{
lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; 
x_107 = lean_ctor_get(x_106, 0);
lean_inc(x_107);
if (lean_is_exclusive(x_106)) {
 lean_ctor_release(x_106, 0);
 x_108 = x_106;
} else {
 lean_dec_ref(x_106);
 x_108 = lean_box(0);
}
x_109 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_109, 0, x_107);
lean_ctor_set(x_109, 1, x_93);
x_110 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_110, 0, x_105);
lean_ctor_set(x_110, 1, x_109);
x_111 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_111, 0, x_103);
lean_ctor_set(x_111, 1, x_110);
x_112 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_112, 0, x_101);
lean_ctor_set(x_112, 1, x_111);
if (lean_is_scalar(x_108)) {
 x_113 = lean_alloc_ctor(0, 1, 0);
} else {
 x_113 = x_108;
}
lean_ctor_set(x_113, 0, x_112);
return x_113;
}
else
{
lean_object* x_114; lean_object* x_115; lean_object* x_116; 
lean_dec(x_105);
lean_dec(x_103);
lean_dec(x_101);
lean_dec(x_93);
x_114 = lean_ctor_get(x_106, 0);
lean_inc(x_114);
if (lean_is_exclusive(x_106)) {
 lean_ctor_release(x_106, 0);
 x_115 = x_106;
} else {
 lean_dec_ref(x_106);
 x_115 = lean_box(0);
}
if (lean_is_scalar(x_115)) {
 x_116 = lean_alloc_ctor(1, 1, 0);
} else {
 x_116 = x_115;
}
lean_ctor_set(x_116, 0, x_114);
return x_116;
}
}
else
{
lean_object* x_117; lean_object* x_118; lean_object* x_119; 
lean_dec(x_103);
lean_dec(x_101);
lean_dec(x_93);
lean_dec(x_19);
lean_dec(x_3);
x_117 = lean_ctor_get(x_104, 0);
lean_inc(x_117);
if (lean_is_exclusive(x_104)) {
 lean_ctor_release(x_104, 0);
 x_118 = x_104;
} else {
 lean_dec_ref(x_104);
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
lean_dec(x_101);
lean_dec(x_93);
lean_dec(x_19);
lean_dec(x_17);
lean_dec(x_3);
x_120 = lean_ctor_get(x_102, 0);
lean_inc(x_120);
if (lean_is_exclusive(x_102)) {
 lean_ctor_release(x_102, 0);
 x_121 = x_102;
} else {
 lean_dec_ref(x_102);
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
else
{
lean_object* x_123; lean_object* x_124; lean_object* x_125; 
lean_dec(x_93);
lean_dec(x_19);
lean_dec(x_17);
lean_dec(x_14);
lean_dec(x_3);
x_123 = lean_ctor_get(x_100, 0);
lean_inc(x_123);
if (lean_is_exclusive(x_100)) {
 lean_ctor_release(x_100, 0);
 x_124 = x_100;
} else {
 lean_dec_ref(x_100);
 x_124 = lean_box(0);
}
if (lean_is_scalar(x_124)) {
 x_125 = lean_alloc_ctor(1, 1, 0);
} else {
 x_125 = x_124;
}
lean_ctor_set(x_125, 0, x_123);
return x_125;
}
}
}
}
else
{
uint8_t x_126; 
lean_dec(x_19);
lean_dec(x_17);
lean_dec(x_14);
lean_dec(x_8);
lean_dec(x_3);
x_126 = !lean_is_exclusive(x_54);
if (x_126 == 0)
{
return x_54;
}
else
{
lean_object* x_127; lean_object* x_128; 
x_127 = lean_ctor_get(x_54, 0);
lean_inc(x_127);
lean_dec(x_54);
x_128 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_128, 0, x_127);
return x_128;
}
}
}
else
{
uint64_t x_129; uint64_t x_130; uint64_t x_131; uint64_t x_132; uint64_t x_133; lean_object* x_134; lean_object* x_135; lean_object* x_136; 
lean_dec(x_2);
x_129 = 2;
x_130 = lean_uint64_shift_right(x_39, x_129);
x_131 = lean_uint64_shift_left(x_130, x_129);
x_132 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_133 = lean_uint64_lor(x_131, x_132);
x_134 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_134, 0, x_20);
lean_ctor_set_uint64(x_134, sizeof(void*)*1, x_133);
x_135 = lean_alloc_ctor(0, 7, 3);
lean_ctor_set(x_135, 0, x_134);
lean_ctor_set(x_135, 1, x_23);
lean_ctor_set(x_135, 2, x_24);
lean_ctor_set(x_135, 3, x_25);
lean_ctor_set(x_135, 4, x_26);
lean_ctor_set(x_135, 5, x_27);
lean_ctor_set(x_135, 6, x_28);
lean_ctor_set_uint8(x_135, sizeof(void*)*7, x_22);
lean_ctor_set_uint8(x_135, sizeof(void*)*7 + 1, x_29);
lean_ctor_set_uint8(x_135, sizeof(void*)*7 + 2, x_30);
lean_inc(x_3);
x_136 = l_Lean_Meta_isExprDefEq(x_37, x_1, x_135, x_3, x_4, x_5);
if (lean_obj_tag(x_136) == 0)
{
lean_object* x_137; lean_object* x_138; uint8_t x_139; 
x_137 = lean_ctor_get(x_136, 0);
lean_inc(x_137);
if (lean_is_exclusive(x_136)) {
 lean_ctor_release(x_136, 0);
 x_138 = x_136;
} else {
 lean_dec_ref(x_136);
 x_138 = lean_box(0);
}
x_139 = lean_unbox(x_137);
if (x_139 == 0)
{
lean_object* x_140; lean_object* x_141; lean_object* x_142; lean_object* x_143; lean_object* x_144; 
lean_dec(x_3);
x_140 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_140, 0, x_19);
lean_ctor_set(x_140, 1, x_137);
x_141 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_141, 0, x_17);
lean_ctor_set(x_141, 1, x_140);
x_142 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_142, 0, x_14);
lean_ctor_set(x_142, 1, x_141);
x_143 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_143, 0, x_8);
lean_ctor_set(x_143, 1, x_142);
if (lean_is_scalar(x_138)) {
 x_144 = lean_alloc_ctor(0, 1, 0);
} else {
 x_144 = x_138;
}
lean_ctor_set(x_144, 0, x_143);
return x_144;
}
else
{
lean_object* x_145; 
lean_dec(x_138);
x_145 = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg(x_8, x_3);
if (lean_obj_tag(x_145) == 0)
{
lean_object* x_146; lean_object* x_147; 
x_146 = lean_ctor_get(x_145, 0);
lean_inc(x_146);
lean_dec_ref(x_145);
x_147 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_14, x_3);
if (lean_obj_tag(x_147) == 0)
{
lean_object* x_148; lean_object* x_149; 
x_148 = lean_ctor_get(x_147, 0);
lean_inc(x_148);
lean_dec_ref(x_147);
x_149 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_17, x_3);
if (lean_obj_tag(x_149) == 0)
{
lean_object* x_150; lean_object* x_151; 
x_150 = lean_ctor_get(x_149, 0);
lean_inc(x_150);
lean_dec_ref(x_149);
x_151 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_19, x_3);
lean_dec(x_3);
if (lean_obj_tag(x_151) == 0)
{
lean_object* x_152; lean_object* x_153; lean_object* x_154; lean_object* x_155; lean_object* x_156; lean_object* x_157; lean_object* x_158; 
x_152 = lean_ctor_get(x_151, 0);
lean_inc(x_152);
if (lean_is_exclusive(x_151)) {
 lean_ctor_release(x_151, 0);
 x_153 = x_151;
} else {
 lean_dec_ref(x_151);
 x_153 = lean_box(0);
}
x_154 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_154, 0, x_152);
lean_ctor_set(x_154, 1, x_137);
x_155 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_155, 0, x_150);
lean_ctor_set(x_155, 1, x_154);
x_156 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_156, 0, x_148);
lean_ctor_set(x_156, 1, x_155);
x_157 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_157, 0, x_146);
lean_ctor_set(x_157, 1, x_156);
if (lean_is_scalar(x_153)) {
 x_158 = lean_alloc_ctor(0, 1, 0);
} else {
 x_158 = x_153;
}
lean_ctor_set(x_158, 0, x_157);
return x_158;
}
else
{
lean_object* x_159; lean_object* x_160; lean_object* x_161; 
lean_dec(x_150);
lean_dec(x_148);
lean_dec(x_146);
lean_dec(x_137);
x_159 = lean_ctor_get(x_151, 0);
lean_inc(x_159);
if (lean_is_exclusive(x_151)) {
 lean_ctor_release(x_151, 0);
 x_160 = x_151;
} else {
 lean_dec_ref(x_151);
 x_160 = lean_box(0);
}
if (lean_is_scalar(x_160)) {
 x_161 = lean_alloc_ctor(1, 1, 0);
} else {
 x_161 = x_160;
}
lean_ctor_set(x_161, 0, x_159);
return x_161;
}
}
else
{
lean_object* x_162; lean_object* x_163; lean_object* x_164; 
lean_dec(x_148);
lean_dec(x_146);
lean_dec(x_137);
lean_dec(x_19);
lean_dec(x_3);
x_162 = lean_ctor_get(x_149, 0);
lean_inc(x_162);
if (lean_is_exclusive(x_149)) {
 lean_ctor_release(x_149, 0);
 x_163 = x_149;
} else {
 lean_dec_ref(x_149);
 x_163 = lean_box(0);
}
if (lean_is_scalar(x_163)) {
 x_164 = lean_alloc_ctor(1, 1, 0);
} else {
 x_164 = x_163;
}
lean_ctor_set(x_164, 0, x_162);
return x_164;
}
}
else
{
lean_object* x_165; lean_object* x_166; lean_object* x_167; 
lean_dec(x_146);
lean_dec(x_137);
lean_dec(x_19);
lean_dec(x_17);
lean_dec(x_3);
x_165 = lean_ctor_get(x_147, 0);
lean_inc(x_165);
if (lean_is_exclusive(x_147)) {
 lean_ctor_release(x_147, 0);
 x_166 = x_147;
} else {
 lean_dec_ref(x_147);
 x_166 = lean_box(0);
}
if (lean_is_scalar(x_166)) {
 x_167 = lean_alloc_ctor(1, 1, 0);
} else {
 x_167 = x_166;
}
lean_ctor_set(x_167, 0, x_165);
return x_167;
}
}
else
{
lean_object* x_168; lean_object* x_169; lean_object* x_170; 
lean_dec(x_137);
lean_dec(x_19);
lean_dec(x_17);
lean_dec(x_14);
lean_dec(x_3);
x_168 = lean_ctor_get(x_145, 0);
lean_inc(x_168);
if (lean_is_exclusive(x_145)) {
 lean_ctor_release(x_145, 0);
 x_169 = x_145;
} else {
 lean_dec_ref(x_145);
 x_169 = lean_box(0);
}
if (lean_is_scalar(x_169)) {
 x_170 = lean_alloc_ctor(1, 1, 0);
} else {
 x_170 = x_169;
}
lean_ctor_set(x_170, 0, x_168);
return x_170;
}
}
}
else
{
lean_object* x_171; lean_object* x_172; lean_object* x_173; 
lean_dec(x_19);
lean_dec(x_17);
lean_dec(x_14);
lean_dec(x_8);
lean_dec(x_3);
x_171 = lean_ctor_get(x_136, 0);
lean_inc(x_171);
if (lean_is_exclusive(x_136)) {
 lean_ctor_release(x_136, 0);
 x_172 = x_136;
} else {
 lean_dec_ref(x_136);
 x_172 = lean_box(0);
}
if (lean_is_scalar(x_172)) {
 x_173 = lean_alloc_ctor(1, 1, 0);
} else {
 x_173 = x_172;
}
lean_ctor_set(x_173, 0, x_171);
return x_173;
}
}
}
else
{
uint8_t x_174; uint8_t x_175; uint8_t x_176; uint8_t x_177; uint8_t x_178; uint8_t x_179; uint8_t x_180; uint8_t x_181; uint8_t x_182; uint8_t x_183; uint8_t x_184; uint8_t x_185; uint8_t x_186; uint8_t x_187; uint8_t x_188; uint8_t x_189; uint8_t x_190; uint8_t x_191; uint8_t x_192; lean_object* x_193; lean_object* x_194; lean_object* x_195; lean_object* x_196; lean_object* x_197; lean_object* x_198; uint8_t x_199; uint8_t x_200; lean_object* x_201; lean_object* x_202; lean_object* x_203; lean_object* x_204; lean_object* x_205; lean_object* x_206; lean_object* x_207; uint8_t x_208; lean_object* x_209; uint64_t x_210; lean_object* x_211; uint64_t x_212; uint64_t x_213; uint64_t x_214; uint64_t x_215; uint64_t x_216; lean_object* x_217; lean_object* x_218; lean_object* x_219; 
x_174 = lean_ctor_get_uint8(x_20, 0);
x_175 = lean_ctor_get_uint8(x_20, 1);
x_176 = lean_ctor_get_uint8(x_20, 2);
x_177 = lean_ctor_get_uint8(x_20, 3);
x_178 = lean_ctor_get_uint8(x_20, 4);
x_179 = lean_ctor_get_uint8(x_20, 5);
x_180 = lean_ctor_get_uint8(x_20, 6);
x_181 = lean_ctor_get_uint8(x_20, 7);
x_182 = lean_ctor_get_uint8(x_20, 8);
x_183 = lean_ctor_get_uint8(x_20, 10);
x_184 = lean_ctor_get_uint8(x_20, 11);
x_185 = lean_ctor_get_uint8(x_20, 12);
x_186 = lean_ctor_get_uint8(x_20, 13);
x_187 = lean_ctor_get_uint8(x_20, 14);
x_188 = lean_ctor_get_uint8(x_20, 15);
x_189 = lean_ctor_get_uint8(x_20, 16);
x_190 = lean_ctor_get_uint8(x_20, 17);
x_191 = lean_ctor_get_uint8(x_20, 18);
lean_dec(x_20);
x_192 = lean_ctor_get_uint8(x_2, sizeof(void*)*7);
x_193 = lean_ctor_get(x_2, 1);
lean_inc(x_193);
x_194 = lean_ctor_get(x_2, 2);
lean_inc_ref(x_194);
x_195 = lean_ctor_get(x_2, 3);
lean_inc_ref(x_195);
x_196 = lean_ctor_get(x_2, 4);
lean_inc(x_196);
x_197 = lean_ctor_get(x_2, 5);
lean_inc(x_197);
x_198 = lean_ctor_get(x_2, 6);
lean_inc(x_198);
x_199 = lean_ctor_get_uint8(x_2, sizeof(void*)*7 + 1);
x_200 = lean_ctor_get_uint8(x_2, sizeof(void*)*7 + 2);
x_201 = lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__4;
x_202 = lean_box(0);
lean_inc(x_8);
x_203 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_203, 0, x_8);
lean_ctor_set(x_203, 1, x_202);
x_204 = l_Lean_Expr_const___override(x_201, x_203);
lean_inc(x_14);
x_205 = l_Lean_Expr_app___override(x_204, x_14);
lean_inc(x_17);
x_206 = l_Lean_Expr_app___override(x_205, x_17);
lean_inc(x_19);
x_207 = l_Lean_Expr_app___override(x_206, x_19);
x_208 = 2;
x_209 = lean_alloc_ctor(0, 0, 19);
lean_ctor_set_uint8(x_209, 0, x_174);
lean_ctor_set_uint8(x_209, 1, x_175);
lean_ctor_set_uint8(x_209, 2, x_176);
lean_ctor_set_uint8(x_209, 3, x_177);
lean_ctor_set_uint8(x_209, 4, x_178);
lean_ctor_set_uint8(x_209, 5, x_179);
lean_ctor_set_uint8(x_209, 6, x_180);
lean_ctor_set_uint8(x_209, 7, x_181);
lean_ctor_set_uint8(x_209, 8, x_182);
lean_ctor_set_uint8(x_209, 9, x_208);
lean_ctor_set_uint8(x_209, 10, x_183);
lean_ctor_set_uint8(x_209, 11, x_184);
lean_ctor_set_uint8(x_209, 12, x_185);
lean_ctor_set_uint8(x_209, 13, x_186);
lean_ctor_set_uint8(x_209, 14, x_187);
lean_ctor_set_uint8(x_209, 15, x_188);
lean_ctor_set_uint8(x_209, 16, x_189);
lean_ctor_set_uint8(x_209, 17, x_190);
lean_ctor_set_uint8(x_209, 18, x_191);
x_210 = l_Lean_Meta_Context_configKey(x_2);
if (lean_is_exclusive(x_2)) {
 lean_ctor_release(x_2, 0);
 lean_ctor_release(x_2, 1);
 lean_ctor_release(x_2, 2);
 lean_ctor_release(x_2, 3);
 lean_ctor_release(x_2, 4);
 lean_ctor_release(x_2, 5);
 lean_ctor_release(x_2, 6);
 x_211 = x_2;
} else {
 lean_dec_ref(x_2);
 x_211 = lean_box(0);
}
x_212 = 2;
x_213 = lean_uint64_shift_right(x_210, x_212);
x_214 = lean_uint64_shift_left(x_213, x_212);
x_215 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_216 = lean_uint64_lor(x_214, x_215);
x_217 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_217, 0, x_209);
lean_ctor_set_uint64(x_217, sizeof(void*)*1, x_216);
if (lean_is_scalar(x_211)) {
 x_218 = lean_alloc_ctor(0, 7, 3);
} else {
 x_218 = x_211;
}
lean_ctor_set(x_218, 0, x_217);
lean_ctor_set(x_218, 1, x_193);
lean_ctor_set(x_218, 2, x_194);
lean_ctor_set(x_218, 3, x_195);
lean_ctor_set(x_218, 4, x_196);
lean_ctor_set(x_218, 5, x_197);
lean_ctor_set(x_218, 6, x_198);
lean_ctor_set_uint8(x_218, sizeof(void*)*7, x_192);
lean_ctor_set_uint8(x_218, sizeof(void*)*7 + 1, x_199);
lean_ctor_set_uint8(x_218, sizeof(void*)*7 + 2, x_200);
lean_inc(x_3);
x_219 = l_Lean_Meta_isExprDefEq(x_207, x_1, x_218, x_3, x_4, x_5);
if (lean_obj_tag(x_219) == 0)
{
lean_object* x_220; lean_object* x_221; uint8_t x_222; 
x_220 = lean_ctor_get(x_219, 0);
lean_inc(x_220);
if (lean_is_exclusive(x_219)) {
 lean_ctor_release(x_219, 0);
 x_221 = x_219;
} else {
 lean_dec_ref(x_219);
 x_221 = lean_box(0);
}
x_222 = lean_unbox(x_220);
if (x_222 == 0)
{
lean_object* x_223; lean_object* x_224; lean_object* x_225; lean_object* x_226; lean_object* x_227; 
lean_dec(x_3);
x_223 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_223, 0, x_19);
lean_ctor_set(x_223, 1, x_220);
x_224 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_224, 0, x_17);
lean_ctor_set(x_224, 1, x_223);
x_225 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_225, 0, x_14);
lean_ctor_set(x_225, 1, x_224);
x_226 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_226, 0, x_8);
lean_ctor_set(x_226, 1, x_225);
if (lean_is_scalar(x_221)) {
 x_227 = lean_alloc_ctor(0, 1, 0);
} else {
 x_227 = x_221;
}
lean_ctor_set(x_227, 0, x_226);
return x_227;
}
else
{
lean_object* x_228; 
lean_dec(x_221);
x_228 = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg(x_8, x_3);
if (lean_obj_tag(x_228) == 0)
{
lean_object* x_229; lean_object* x_230; 
x_229 = lean_ctor_get(x_228, 0);
lean_inc(x_229);
lean_dec_ref(x_228);
x_230 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_14, x_3);
if (lean_obj_tag(x_230) == 0)
{
lean_object* x_231; lean_object* x_232; 
x_231 = lean_ctor_get(x_230, 0);
lean_inc(x_231);
lean_dec_ref(x_230);
x_232 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_17, x_3);
if (lean_obj_tag(x_232) == 0)
{
lean_object* x_233; lean_object* x_234; 
x_233 = lean_ctor_get(x_232, 0);
lean_inc(x_233);
lean_dec_ref(x_232);
x_234 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_19, x_3);
lean_dec(x_3);
if (lean_obj_tag(x_234) == 0)
{
lean_object* x_235; lean_object* x_236; lean_object* x_237; lean_object* x_238; lean_object* x_239; lean_object* x_240; lean_object* x_241; 
x_235 = lean_ctor_get(x_234, 0);
lean_inc(x_235);
if (lean_is_exclusive(x_234)) {
 lean_ctor_release(x_234, 0);
 x_236 = x_234;
} else {
 lean_dec_ref(x_234);
 x_236 = lean_box(0);
}
x_237 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_237, 0, x_235);
lean_ctor_set(x_237, 1, x_220);
x_238 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_238, 0, x_233);
lean_ctor_set(x_238, 1, x_237);
x_239 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_239, 0, x_231);
lean_ctor_set(x_239, 1, x_238);
x_240 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_240, 0, x_229);
lean_ctor_set(x_240, 1, x_239);
if (lean_is_scalar(x_236)) {
 x_241 = lean_alloc_ctor(0, 1, 0);
} else {
 x_241 = x_236;
}
lean_ctor_set(x_241, 0, x_240);
return x_241;
}
else
{
lean_object* x_242; lean_object* x_243; lean_object* x_244; 
lean_dec(x_233);
lean_dec(x_231);
lean_dec(x_229);
lean_dec(x_220);
x_242 = lean_ctor_get(x_234, 0);
lean_inc(x_242);
if (lean_is_exclusive(x_234)) {
 lean_ctor_release(x_234, 0);
 x_243 = x_234;
} else {
 lean_dec_ref(x_234);
 x_243 = lean_box(0);
}
if (lean_is_scalar(x_243)) {
 x_244 = lean_alloc_ctor(1, 1, 0);
} else {
 x_244 = x_243;
}
lean_ctor_set(x_244, 0, x_242);
return x_244;
}
}
else
{
lean_object* x_245; lean_object* x_246; lean_object* x_247; 
lean_dec(x_231);
lean_dec(x_229);
lean_dec(x_220);
lean_dec(x_19);
lean_dec(x_3);
x_245 = lean_ctor_get(x_232, 0);
lean_inc(x_245);
if (lean_is_exclusive(x_232)) {
 lean_ctor_release(x_232, 0);
 x_246 = x_232;
} else {
 lean_dec_ref(x_232);
 x_246 = lean_box(0);
}
if (lean_is_scalar(x_246)) {
 x_247 = lean_alloc_ctor(1, 1, 0);
} else {
 x_247 = x_246;
}
lean_ctor_set(x_247, 0, x_245);
return x_247;
}
}
else
{
lean_object* x_248; lean_object* x_249; lean_object* x_250; 
lean_dec(x_229);
lean_dec(x_220);
lean_dec(x_19);
lean_dec(x_17);
lean_dec(x_3);
x_248 = lean_ctor_get(x_230, 0);
lean_inc(x_248);
if (lean_is_exclusive(x_230)) {
 lean_ctor_release(x_230, 0);
 x_249 = x_230;
} else {
 lean_dec_ref(x_230);
 x_249 = lean_box(0);
}
if (lean_is_scalar(x_249)) {
 x_250 = lean_alloc_ctor(1, 1, 0);
} else {
 x_250 = x_249;
}
lean_ctor_set(x_250, 0, x_248);
return x_250;
}
}
else
{
lean_object* x_251; lean_object* x_252; lean_object* x_253; 
lean_dec(x_220);
lean_dec(x_19);
lean_dec(x_17);
lean_dec(x_14);
lean_dec(x_3);
x_251 = lean_ctor_get(x_228, 0);
lean_inc(x_251);
if (lean_is_exclusive(x_228)) {
 lean_ctor_release(x_228, 0);
 x_252 = x_228;
} else {
 lean_dec_ref(x_228);
 x_252 = lean_box(0);
}
if (lean_is_scalar(x_252)) {
 x_253 = lean_alloc_ctor(1, 1, 0);
} else {
 x_253 = x_252;
}
lean_ctor_set(x_253, 0, x_251);
return x_253;
}
}
}
else
{
lean_object* x_254; lean_object* x_255; lean_object* x_256; 
lean_dec(x_19);
lean_dec(x_17);
lean_dec(x_14);
lean_dec(x_8);
lean_dec(x_3);
x_254 = lean_ctor_get(x_219, 0);
lean_inc(x_254);
if (lean_is_exclusive(x_219)) {
 lean_ctor_release(x_219, 0);
 x_255 = x_219;
} else {
 lean_dec_ref(x_219);
 x_255 = lean_box(0);
}
if (lean_is_scalar(x_255)) {
 x_256 = lean_alloc_ctor(1, 1, 0);
} else {
 x_256 = x_255;
}
lean_ctor_set(x_256, 0, x_254);
return x_256;
}
}
}
else
{
uint8_t x_257; 
lean_dec(x_17);
lean_dec(x_14);
lean_dec(x_8);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_257 = !lean_is_exclusive(x_18);
if (x_257 == 0)
{
return x_18;
}
else
{
lean_object* x_258; lean_object* x_259; 
x_258 = lean_ctor_get(x_18, 0);
lean_inc(x_258);
lean_dec(x_18);
x_259 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_259, 0, x_258);
return x_259;
}
}
}
else
{
uint8_t x_260; 
lean_dec_ref(x_15);
lean_dec(x_14);
lean_dec(x_8);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_260 = !lean_is_exclusive(x_16);
if (x_260 == 0)
{
return x_16;
}
else
{
lean_object* x_261; lean_object* x_262; 
x_261 = lean_ctor_get(x_16, 0);
lean_inc(x_261);
lean_dec(x_16);
x_262 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_262, 0, x_261);
return x_262;
}
}
}
else
{
uint8_t x_263; 
lean_dec(x_8);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_263 = !lean_is_exclusive(x_13);
if (x_263 == 0)
{
return x_13;
}
else
{
lean_object* x_264; lean_object* x_265; 
x_264 = lean_ctor_get(x_13, 0);
lean_inc(x_264);
lean_dec(x_13);
x_265 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_265, 0, x_264);
return x_265;
}
}
}
else
{
uint8_t x_266; 
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_266 = !lean_is_exclusive(x_7);
if (x_266 == 0)
{
return x_7;
}
else
{
lean_object* x_267; lean_object* x_268; 
x_267 = lean_ctor_get(x_7, 0);
lean_inc(x_267);
lean_dec(x_7);
x_268 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_268, 0, x_267);
return x_268;
}
}
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("rfl", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__0;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("mkAfterToBefore: unexpected goal: {← ppExpr goal}", 51, 49);
return x_1;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("mkAfterToBefore: `goal` is equality but `path` is not empty", 59, 59);
return x_1;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("mkAfterToBefore: goal is `And` but `P` is not `And`", 51, 51);
return x_1;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("mkAfterToBefore: goal is `And` but `exs` is empty", 49, 49);
return x_1;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("left", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__6;
x_2 = lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__1;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__8() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__7;
x_3 = l_Lean_Expr_const___override(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__9() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("right", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__10() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__9;
x_2 = lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__1;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__11() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__10;
x_3 = l_Lean_Expr_const___override(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__12() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("intro", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__13() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__12;
x_2 = lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__1;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__14() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__13;
x_3 = l_Lean_Expr_const___override(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__15() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("mkAfterToBefore: goal is `Exists` but `exs` is empty", 52, 52);
return x_1;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__16() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__12;
x_2 = lp_mathlib_ExistsAndEq_mkNestedExists___closed__0;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
uint8_t x_11; uint8_t x_12; lean_object* x_13; 
x_11 = lean_unbox(x_1);
x_12 = lean_unbox(x_5);
x_13 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___lam__0(x_11, x_2, x_3, x_4, x_12, x_6, x_7, x_8, x_9);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
uint8_t x_11; uint8_t x_12; lean_object* x_13; 
x_11 = lean_unbox(x_2);
x_12 = lean_unbox(x_5);
x_13 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___lam__1(x_1, x_11, x_3, x_4, x_12, x_6, x_7, x_8, x_9);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___lam__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___lam__2(x_1, x_2, x_3, x_4, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___lam__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
uint8_t x_12; uint8_t x_13; lean_object* x_14; 
x_12 = lean_unbox(x_2);
x_13 = lean_unbox(x_5);
x_14 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___lam__3(x_1, x_12, x_3, x_4, x_13, x_6, x_7, x_8, x_9, x_10);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
uint8_t x_11; lean_object* x_12; uint8_t x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_11 = 0;
x_12 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0;
x_13 = 0;
x_14 = lean_box(0);
x_15 = lean_box(x_13);
x_16 = lean_box(x_11);
lean_inc_ref(x_1);
x_17 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___lam__0___boxed), 10, 5);
lean_closure_set(x_17, 0, x_15);
lean_closure_set(x_17, 1, x_14);
lean_closure_set(x_17, 2, x_12);
lean_closure_set(x_17, 3, x_1);
lean_closure_set(x_17, 4, x_16);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
x_18 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(x_17, x_11, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_18) == 0)
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; uint8_t x_23; 
x_19 = lean_ctor_get(x_18, 0);
lean_inc(x_19);
lean_dec_ref(x_18);
x_20 = lean_ctor_get(x_19, 1);
lean_inc(x_20);
lean_dec(x_19);
x_21 = lean_ctor_get(x_20, 1);
lean_inc(x_21);
x_22 = lean_ctor_get(x_21, 1);
x_23 = lean_unbox(x_22);
if (x_23 == 0)
{
lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; 
lean_dec(x_21);
lean_dec(x_20);
x_24 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__1;
x_25 = lean_box(x_13);
x_26 = lean_box(x_11);
lean_inc_ref(x_1);
x_27 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___lam__1___boxed), 10, 5);
lean_closure_set(x_27, 0, x_24);
lean_closure_set(x_27, 1, x_25);
lean_closure_set(x_27, 2, x_14);
lean_closure_set(x_27, 3, x_1);
lean_closure_set(x_27, 4, x_26);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
x_28 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(x_27, x_11, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_28) == 0)
{
lean_object* x_29; lean_object* x_30; lean_object* x_31; uint8_t x_32; 
x_29 = lean_ctor_get(x_28, 0);
lean_inc(x_29);
lean_dec_ref(x_28);
x_30 = lean_ctor_get(x_29, 1);
lean_inc(x_30);
x_31 = lean_ctor_get(x_30, 1);
lean_inc(x_31);
x_32 = lean_unbox(x_31);
if (x_32 == 0)
{
lean_object* x_33; lean_object* x_34; 
lean_dec(x_31);
lean_dec(x_30);
lean_dec(x_29);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
x_33 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___lam__2___boxed), 6, 1);
lean_closure_set(x_33, 0, x_1);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
x_34 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(x_33, x_11, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_34) == 0)
{
lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_53; uint8_t x_54; 
x_35 = lean_ctor_get(x_34, 0);
lean_inc(x_35);
if (lean_is_exclusive(x_34)) {
 lean_ctor_release(x_34, 0);
 x_36 = x_34;
} else {
 lean_dec_ref(x_34);
 x_36 = lean_box(0);
}
x_37 = lean_ctor_get(x_35, 1);
lean_inc(x_37);
x_38 = lean_ctor_get(x_37, 1);
lean_inc(x_38);
x_39 = lean_ctor_get(x_35, 0);
lean_inc(x_39);
lean_dec(x_35);
x_40 = lean_ctor_get(x_37, 0);
lean_inc(x_40);
lean_dec(x_37);
x_41 = lean_ctor_get(x_38, 0);
lean_inc(x_41);
x_42 = lean_ctor_get(x_38, 1);
lean_inc(x_42);
if (lean_is_exclusive(x_38)) {
 lean_ctor_release(x_38, 0);
 lean_ctor_release(x_38, 1);
 x_43 = x_38;
} else {
 lean_dec_ref(x_38);
 x_43 = lean_box(0);
}
x_53 = lean_ctor_get(x_42, 1);
lean_inc(x_53);
lean_dec(x_42);
x_54 = lean_unbox(x_53);
lean_dec(x_53);
if (x_54 == 0)
{
lean_object* x_55; lean_object* x_56; 
lean_dec(x_43);
lean_dec(x_41);
lean_dec(x_40);
lean_dec(x_39);
lean_dec(x_36);
x_55 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__2;
x_56 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg(x_55, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
return x_56;
}
else
{
uint8_t x_57; 
x_57 = l_List_isEmpty___redArg(x_5);
if (x_57 == 0)
{
lean_object* x_58; lean_object* x_59; 
x_58 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__3;
x_59 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg(x_58, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
if (lean_obj_tag(x_59) == 0)
{
lean_dec_ref(x_59);
x_44 = lean_box(0);
goto block_52;
}
else
{
uint8_t x_60; 
lean_dec(x_43);
lean_dec(x_41);
lean_dec(x_40);
lean_dec(x_39);
lean_dec(x_36);
x_60 = !lean_is_exclusive(x_59);
if (x_60 == 0)
{
return x_59;
}
else
{
lean_object* x_61; lean_object* x_62; 
x_61 = lean_ctor_get(x_59, 0);
lean_inc(x_61);
lean_dec(x_59);
x_62 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_62, 0, x_61);
return x_62;
}
}
}
else
{
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
x_44 = lean_box(0);
goto block_52;
}
}
block_52:
{
lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; 
x_45 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__1;
x_46 = lean_box(0);
if (lean_is_scalar(x_43)) {
 x_47 = lean_alloc_ctor(1, 2, 0);
} else {
 x_47 = x_43;
 lean_ctor_set_tag(x_47, 1);
}
lean_ctor_set(x_47, 0, x_39);
lean_ctor_set(x_47, 1, x_46);
x_48 = l_Lean_Expr_const___override(x_45, x_47);
x_49 = l_Lean_Expr_app___override(x_48, x_40);
x_50 = l_Lean_Expr_app___override(x_49, x_41);
if (lean_is_scalar(x_36)) {
 x_51 = lean_alloc_ctor(0, 1, 0);
} else {
 x_51 = x_36;
}
lean_ctor_set(x_51, 0, x_50);
return x_51;
}
}
else
{
uint8_t x_63; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
x_63 = !lean_is_exclusive(x_34);
if (x_63 == 0)
{
return x_34;
}
else
{
lean_object* x_64; lean_object* x_65; 
x_64 = lean_ctor_get(x_34, 0);
lean_inc(x_64);
lean_dec(x_34);
x_65 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_65, 0, x_64);
return x_65;
}
}
}
else
{
lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; 
lean_dec_ref(x_1);
x_66 = lean_ctor_get(x_29, 0);
lean_inc(x_66);
lean_dec(x_29);
x_67 = lean_ctor_get(x_30, 0);
lean_inc(x_67);
lean_dec(x_30);
x_68 = lean_box(x_13);
x_69 = lean_box(x_11);
x_70 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___lam__3___boxed), 11, 6);
lean_closure_set(x_70, 0, x_24);
lean_closure_set(x_70, 1, x_68);
lean_closure_set(x_70, 2, x_14);
lean_closure_set(x_70, 3, x_2);
lean_closure_set(x_70, 4, x_69);
lean_closure_set(x_70, 5, x_31);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
x_71 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(x_70, x_11, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_71) == 0)
{
lean_object* x_72; lean_object* x_73; lean_object* x_74; uint8_t x_75; 
x_72 = lean_ctor_get(x_71, 0);
lean_inc(x_72);
lean_dec_ref(x_71);
x_73 = lean_ctor_get(x_72, 1);
lean_inc(x_73);
x_74 = lean_ctor_get(x_73, 1);
x_75 = lean_unbox(x_74);
if (x_75 == 0)
{
lean_object* x_76; lean_object* x_77; 
lean_dec(x_73);
lean_dec(x_72);
lean_dec(x_67);
lean_dec(x_66);
lean_dec(x_4);
lean_dec_ref(x_3);
x_76 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__4;
x_77 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg(x_76, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
return x_77;
}
else
{
if (lean_obj_tag(x_5) == 0)
{
lean_object* x_78; lean_object* x_79; 
lean_dec(x_73);
lean_dec(x_72);
lean_dec(x_67);
lean_dec(x_66);
lean_dec(x_4);
lean_dec_ref(x_3);
x_78 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__5;
x_79 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg(x_78, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
return x_79;
}
else
{
lean_object* x_80; uint8_t x_81; 
x_80 = lean_ctor_get(x_5, 0);
x_81 = lean_unbox(x_80);
if (x_81 == 0)
{
lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; 
x_82 = lean_ctor_get(x_72, 0);
lean_inc(x_82);
lean_dec(x_72);
x_83 = lean_ctor_get(x_73, 0);
lean_inc(x_83);
lean_dec(x_73);
x_84 = lean_ctor_get(x_5, 1);
x_85 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__8;
lean_inc(x_82);
x_86 = l_Lean_Expr_app___override(x_85, x_82);
x_87 = l_Lean_Expr_app___override(x_86, x_83);
lean_inc_ref(x_3);
x_88 = l_Lean_Expr_app___override(x_87, x_3);
lean_inc(x_82);
lean_inc(x_66);
x_89 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go(x_66, x_82, x_88, x_4, x_84, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_89) == 0)
{
uint8_t x_90; 
x_90 = !lean_is_exclusive(x_89);
if (x_90 == 0)
{
lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; 
x_91 = lean_ctor_get(x_89, 0);
x_92 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__11;
x_93 = l_Lean_Expr_app___override(x_92, x_82);
lean_inc(x_67);
x_94 = l_Lean_Expr_app___override(x_93, x_67);
x_95 = l_Lean_Expr_app___override(x_94, x_3);
x_96 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__14;
x_97 = l_Lean_Expr_app___override(x_96, x_66);
x_98 = l_Lean_Expr_app___override(x_97, x_67);
x_99 = l_Lean_Expr_app___override(x_98, x_91);
x_100 = l_Lean_Expr_app___override(x_99, x_95);
lean_ctor_set(x_89, 0, x_100);
return x_89;
}
else
{
lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; 
x_101 = lean_ctor_get(x_89, 0);
lean_inc(x_101);
lean_dec(x_89);
x_102 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__11;
x_103 = l_Lean_Expr_app___override(x_102, x_82);
lean_inc(x_67);
x_104 = l_Lean_Expr_app___override(x_103, x_67);
x_105 = l_Lean_Expr_app___override(x_104, x_3);
x_106 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__14;
x_107 = l_Lean_Expr_app___override(x_106, x_66);
x_108 = l_Lean_Expr_app___override(x_107, x_67);
x_109 = l_Lean_Expr_app___override(x_108, x_101);
x_110 = l_Lean_Expr_app___override(x_109, x_105);
x_111 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_111, 0, x_110);
return x_111;
}
}
else
{
lean_dec(x_82);
lean_dec(x_67);
lean_dec(x_66);
lean_dec_ref(x_3);
return x_89;
}
}
else
{
lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; lean_object* x_116; lean_object* x_117; lean_object* x_118; lean_object* x_119; 
x_112 = lean_ctor_get(x_72, 0);
lean_inc(x_112);
lean_dec(x_72);
x_113 = lean_ctor_get(x_73, 0);
lean_inc(x_113);
lean_dec(x_73);
x_114 = lean_ctor_get(x_5, 1);
x_115 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__11;
x_116 = l_Lean_Expr_app___override(x_115, x_112);
lean_inc(x_113);
x_117 = l_Lean_Expr_app___override(x_116, x_113);
lean_inc_ref(x_3);
x_118 = l_Lean_Expr_app___override(x_117, x_3);
lean_inc(x_113);
lean_inc(x_67);
x_119 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go(x_67, x_113, x_118, x_4, x_114, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_119) == 0)
{
uint8_t x_120; 
x_120 = !lean_is_exclusive(x_119);
if (x_120 == 0)
{
lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_127; lean_object* x_128; lean_object* x_129; lean_object* x_130; 
x_121 = lean_ctor_get(x_119, 0);
x_122 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__8;
lean_inc(x_66);
x_123 = l_Lean_Expr_app___override(x_122, x_66);
x_124 = l_Lean_Expr_app___override(x_123, x_113);
x_125 = l_Lean_Expr_app___override(x_124, x_3);
x_126 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__14;
x_127 = l_Lean_Expr_app___override(x_126, x_66);
x_128 = l_Lean_Expr_app___override(x_127, x_67);
x_129 = l_Lean_Expr_app___override(x_128, x_125);
x_130 = l_Lean_Expr_app___override(x_129, x_121);
lean_ctor_set(x_119, 0, x_130);
return x_119;
}
else
{
lean_object* x_131; lean_object* x_132; lean_object* x_133; lean_object* x_134; lean_object* x_135; lean_object* x_136; lean_object* x_137; lean_object* x_138; lean_object* x_139; lean_object* x_140; lean_object* x_141; 
x_131 = lean_ctor_get(x_119, 0);
lean_inc(x_131);
lean_dec(x_119);
x_132 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__8;
lean_inc(x_66);
x_133 = l_Lean_Expr_app___override(x_132, x_66);
x_134 = l_Lean_Expr_app___override(x_133, x_113);
x_135 = l_Lean_Expr_app___override(x_134, x_3);
x_136 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__14;
x_137 = l_Lean_Expr_app___override(x_136, x_66);
x_138 = l_Lean_Expr_app___override(x_137, x_67);
x_139 = l_Lean_Expr_app___override(x_138, x_135);
x_140 = l_Lean_Expr_app___override(x_139, x_131);
x_141 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_141, 0, x_140);
return x_141;
}
}
else
{
lean_dec(x_113);
lean_dec(x_67);
lean_dec(x_66);
lean_dec_ref(x_3);
return x_119;
}
}
}
}
}
else
{
uint8_t x_142; 
lean_dec(x_67);
lean_dec(x_66);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_4);
lean_dec_ref(x_3);
x_142 = !lean_is_exclusive(x_71);
if (x_142 == 0)
{
return x_71;
}
else
{
lean_object* x_143; lean_object* x_144; 
x_143 = lean_ctor_get(x_71, 0);
lean_inc(x_143);
lean_dec(x_71);
x_144 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_144, 0, x_143);
return x_144;
}
}
}
}
else
{
uint8_t x_145; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_145 = !lean_is_exclusive(x_28);
if (x_145 == 0)
{
return x_28;
}
else
{
lean_object* x_146; lean_object* x_147; 
x_146 = lean_ctor_get(x_28, 0);
lean_inc(x_146);
lean_dec(x_28);
x_147 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_147, 0, x_146);
return x_147;
}
}
}
else
{
lean_dec_ref(x_1);
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_148; lean_object* x_149; 
lean_dec(x_21);
lean_dec(x_20);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
x_148 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__15;
x_149 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg(x_148, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
return x_149;
}
else
{
lean_object* x_150; lean_object* x_151; lean_object* x_152; lean_object* x_153; uint8_t x_154; 
x_150 = lean_ctor_get(x_4, 0);
lean_inc(x_150);
x_151 = lean_ctor_get(x_150, 1);
lean_inc(x_151);
x_152 = lean_ctor_get(x_20, 0);
lean_inc(x_152);
lean_dec(x_20);
x_153 = lean_ctor_get(x_21, 0);
lean_inc(x_153);
lean_dec(x_21);
x_154 = !lean_is_exclusive(x_4);
if (x_154 == 0)
{
lean_object* x_155; lean_object* x_156; lean_object* x_157; uint8_t x_158; 
x_155 = lean_ctor_get(x_4, 1);
x_156 = lean_ctor_get(x_4, 0);
lean_dec(x_156);
x_157 = lean_ctor_get(x_150, 0);
lean_inc(x_157);
lean_dec(x_150);
x_158 = !lean_is_exclusive(x_151);
if (x_158 == 0)
{
lean_object* x_159; lean_object* x_160; lean_object* x_161; lean_object* x_162; lean_object* x_163; lean_object* x_164; 
x_159 = lean_ctor_get(x_151, 1);
x_160 = lean_ctor_get(x_151, 0);
lean_dec(x_160);
x_161 = lean_box(0);
lean_inc(x_159);
lean_ctor_set(x_4, 1, x_161);
lean_ctor_set(x_4, 0, x_159);
x_162 = lean_array_mk(x_4);
lean_inc(x_153);
x_163 = l_Lean_Expr_betaRev(x_153, x_162, x_11, x_11);
lean_dec_ref(x_162);
x_164 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go(x_163, x_2, x_3, x_155, x_5, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_164) == 0)
{
uint8_t x_165; 
x_165 = !lean_is_exclusive(x_164);
if (x_165 == 0)
{
lean_object* x_166; lean_object* x_167; lean_object* x_168; lean_object* x_169; lean_object* x_170; lean_object* x_171; lean_object* x_172; 
x_166 = lean_ctor_get(x_164, 0);
x_167 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__16;
lean_ctor_set_tag(x_151, 1);
lean_ctor_set(x_151, 1, x_161);
lean_ctor_set(x_151, 0, x_157);
x_168 = l_Lean_Expr_const___override(x_167, x_151);
x_169 = l_Lean_Expr_app___override(x_168, x_152);
x_170 = l_Lean_Expr_app___override(x_169, x_153);
x_171 = l_Lean_Expr_app___override(x_170, x_159);
x_172 = l_Lean_Expr_app___override(x_171, x_166);
lean_ctor_set(x_164, 0, x_172);
return x_164;
}
else
{
lean_object* x_173; lean_object* x_174; lean_object* x_175; lean_object* x_176; lean_object* x_177; lean_object* x_178; lean_object* x_179; lean_object* x_180; 
x_173 = lean_ctor_get(x_164, 0);
lean_inc(x_173);
lean_dec(x_164);
x_174 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__16;
lean_ctor_set_tag(x_151, 1);
lean_ctor_set(x_151, 1, x_161);
lean_ctor_set(x_151, 0, x_157);
x_175 = l_Lean_Expr_const___override(x_174, x_151);
x_176 = l_Lean_Expr_app___override(x_175, x_152);
x_177 = l_Lean_Expr_app___override(x_176, x_153);
x_178 = l_Lean_Expr_app___override(x_177, x_159);
x_179 = l_Lean_Expr_app___override(x_178, x_173);
x_180 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_180, 0, x_179);
return x_180;
}
}
else
{
lean_free_object(x_151);
lean_dec(x_159);
lean_dec(x_157);
lean_dec(x_153);
lean_dec(x_152);
return x_164;
}
}
else
{
lean_object* x_181; lean_object* x_182; lean_object* x_183; lean_object* x_184; lean_object* x_185; 
x_181 = lean_ctor_get(x_151, 1);
lean_inc(x_181);
lean_dec(x_151);
x_182 = lean_box(0);
lean_inc(x_181);
lean_ctor_set(x_4, 1, x_182);
lean_ctor_set(x_4, 0, x_181);
x_183 = lean_array_mk(x_4);
lean_inc(x_153);
x_184 = l_Lean_Expr_betaRev(x_153, x_183, x_11, x_11);
lean_dec_ref(x_183);
x_185 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go(x_184, x_2, x_3, x_155, x_5, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_185) == 0)
{
lean_object* x_186; lean_object* x_187; lean_object* x_188; lean_object* x_189; lean_object* x_190; lean_object* x_191; lean_object* x_192; lean_object* x_193; lean_object* x_194; lean_object* x_195; 
x_186 = lean_ctor_get(x_185, 0);
lean_inc(x_186);
if (lean_is_exclusive(x_185)) {
 lean_ctor_release(x_185, 0);
 x_187 = x_185;
} else {
 lean_dec_ref(x_185);
 x_187 = lean_box(0);
}
x_188 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__16;
x_189 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_189, 0, x_157);
lean_ctor_set(x_189, 1, x_182);
x_190 = l_Lean_Expr_const___override(x_188, x_189);
x_191 = l_Lean_Expr_app___override(x_190, x_152);
x_192 = l_Lean_Expr_app___override(x_191, x_153);
x_193 = l_Lean_Expr_app___override(x_192, x_181);
x_194 = l_Lean_Expr_app___override(x_193, x_186);
if (lean_is_scalar(x_187)) {
 x_195 = lean_alloc_ctor(0, 1, 0);
} else {
 x_195 = x_187;
}
lean_ctor_set(x_195, 0, x_194);
return x_195;
}
else
{
lean_dec(x_181);
lean_dec(x_157);
lean_dec(x_153);
lean_dec(x_152);
return x_185;
}
}
}
else
{
lean_object* x_196; lean_object* x_197; lean_object* x_198; lean_object* x_199; lean_object* x_200; lean_object* x_201; lean_object* x_202; lean_object* x_203; lean_object* x_204; 
x_196 = lean_ctor_get(x_4, 1);
lean_inc(x_196);
lean_dec(x_4);
x_197 = lean_ctor_get(x_150, 0);
lean_inc(x_197);
lean_dec(x_150);
x_198 = lean_ctor_get(x_151, 1);
lean_inc(x_198);
if (lean_is_exclusive(x_151)) {
 lean_ctor_release(x_151, 0);
 lean_ctor_release(x_151, 1);
 x_199 = x_151;
} else {
 lean_dec_ref(x_151);
 x_199 = lean_box(0);
}
x_200 = lean_box(0);
lean_inc(x_198);
x_201 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_201, 0, x_198);
lean_ctor_set(x_201, 1, x_200);
x_202 = lean_array_mk(x_201);
lean_inc(x_153);
x_203 = l_Lean_Expr_betaRev(x_153, x_202, x_11, x_11);
lean_dec_ref(x_202);
x_204 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go(x_203, x_2, x_3, x_196, x_5, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_204) == 0)
{
lean_object* x_205; lean_object* x_206; lean_object* x_207; lean_object* x_208; lean_object* x_209; lean_object* x_210; lean_object* x_211; lean_object* x_212; lean_object* x_213; lean_object* x_214; 
x_205 = lean_ctor_get(x_204, 0);
lean_inc(x_205);
if (lean_is_exclusive(x_204)) {
 lean_ctor_release(x_204, 0);
 x_206 = x_204;
} else {
 lean_dec_ref(x_204);
 x_206 = lean_box(0);
}
x_207 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__16;
if (lean_is_scalar(x_199)) {
 x_208 = lean_alloc_ctor(1, 2, 0);
} else {
 x_208 = x_199;
 lean_ctor_set_tag(x_208, 1);
}
lean_ctor_set(x_208, 0, x_197);
lean_ctor_set(x_208, 1, x_200);
x_209 = l_Lean_Expr_const___override(x_207, x_208);
x_210 = l_Lean_Expr_app___override(x_209, x_152);
x_211 = l_Lean_Expr_app___override(x_210, x_153);
x_212 = l_Lean_Expr_app___override(x_211, x_198);
x_213 = l_Lean_Expr_app___override(x_212, x_205);
if (lean_is_scalar(x_206)) {
 x_214 = lean_alloc_ctor(0, 1, 0);
} else {
 x_214 = x_206;
}
lean_ctor_set(x_214, 0, x_213);
return x_214;
}
else
{
lean_dec(x_199);
lean_dec(x_198);
lean_dec(x_197);
lean_dec(x_153);
lean_dec(x_152);
return x_204;
}
}
}
}
}
else
{
uint8_t x_215; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_215 = !lean_is_exclusive(x_18);
if (x_215 == 0)
{
return x_18;
}
else
{
lean_object* x_216; lean_object* x_217; 
x_216 = lean_ctor_get(x_18, 0);
lean_inc(x_216);
lean_dec(x_18);
x_217 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_217, 0, x_216);
return x_217;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_5);
return x_11;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("a", 1, 1);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__0;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = l_Lean_Expr_bvar___override(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__2;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__3;
x_2 = lean_array_mk(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, uint8_t x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
lean_inc_ref(x_1);
x_18 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_18, 0, x_1);
lean_ctor_set(x_18, 1, x_2);
x_19 = lean_array_mk(x_18);
x_20 = l_Lean_Expr_betaRev(x_3, x_19, x_4, x_4);
lean_dec_ref(x_19);
x_21 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go(x_20, x_5, x_12, x_6, x_7, x_13, x_14, x_15, x_16);
if (lean_obj_tag(x_21) == 0)
{
uint8_t x_22; 
x_22 = !lean_is_exclusive(x_21);
if (x_22 == 0)
{
lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; 
x_23 = lean_ctor_get(x_21, 0);
x_24 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__12;
x_25 = l_Lean_Name_mkStr2(x_8, x_24);
x_26 = l_Lean_Expr_const___override(x_25, x_9);
x_27 = l_Lean_Expr_app___override(x_26, x_10);
x_28 = l_Lean_Expr_app___override(x_27, x_11);
x_29 = l_Lean_Expr_app___override(x_28, x_1);
x_30 = l_Lean_Expr_app___override(x_29, x_23);
lean_ctor_set(x_21, 0, x_30);
return x_21;
}
else
{
lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; 
x_31 = lean_ctor_get(x_21, 0);
lean_inc(x_31);
lean_dec(x_21);
x_32 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__12;
x_33 = l_Lean_Name_mkStr2(x_8, x_32);
x_34 = l_Lean_Expr_const___override(x_33, x_9);
x_35 = l_Lean_Expr_app___override(x_34, x_10);
x_36 = l_Lean_Expr_app___override(x_35, x_11);
x_37 = l_Lean_Expr_app___override(x_36, x_1);
x_38 = l_Lean_Expr_app___override(x_37, x_31);
x_39 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_39, 0, x_38);
return x_39;
}
}
else
{
lean_dec_ref(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_1);
return x_21;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__0___boxed(lean_object** _args) {
lean_object* x_1 = _args[0];
lean_object* x_2 = _args[1];
lean_object* x_3 = _args[2];
lean_object* x_4 = _args[3];
lean_object* x_5 = _args[4];
lean_object* x_6 = _args[5];
lean_object* x_7 = _args[6];
lean_object* x_8 = _args[7];
lean_object* x_9 = _args[8];
lean_object* x_10 = _args[9];
lean_object* x_11 = _args[10];
lean_object* x_12 = _args[11];
lean_object* x_13 = _args[12];
lean_object* x_14 = _args[13];
lean_object* x_15 = _args[14];
lean_object* x_16 = _args[15];
lean_object* x_17 = _args[16];
_start:
{
uint8_t x_18; lean_object* x_19; 
x_18 = lean_unbox(x_4);
x_19 = lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__0(x_1, x_2, x_3, x_18, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16);
lean_dec(x_7);
return x_19;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, uint8_t x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; uint8_t x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; 
x_16 = lp_mathlib_ExistsAndEq_mkNestedExists___closed__0;
x_17 = lp_mathlib_ExistsAndEq_mkNestedExists___closed__1;
x_18 = lean_box(0);
x_19 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_19, 0, x_1);
lean_ctor_set(x_19, 1, x_18);
lean_inc_ref(x_19);
x_20 = l_Lean_Expr_const___override(x_17, x_19);
lean_inc_ref(x_2);
x_21 = l_Lean_Expr_app___override(x_20, x_2);
x_22 = lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__1;
x_23 = lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__4;
x_24 = 0;
lean_inc_ref(x_3);
x_25 = l_Lean_Expr_betaRev(x_3, x_23, x_24, x_24);
lean_inc_ref(x_2);
x_26 = l_Lean_Expr_lam___override(x_22, x_2, x_25, x_4);
x_27 = lean_box(x_24);
lean_inc_ref(x_26);
lean_inc(x_7);
x_28 = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__0___boxed), 17, 11);
lean_closure_set(x_28, 0, x_5);
lean_closure_set(x_28, 1, x_18);
lean_closure_set(x_28, 2, x_3);
lean_closure_set(x_28, 3, x_27);
lean_closure_set(x_28, 4, x_6);
lean_closure_set(x_28, 5, x_7);
lean_closure_set(x_28, 6, x_8);
lean_closure_set(x_28, 7, x_16);
lean_closure_set(x_28, 8, x_19);
lean_closure_set(x_28, 9, x_2);
lean_closure_set(x_28, 10, x_26);
x_29 = l_Lean_Expr_app___override(x_21, x_26);
lean_inc(x_14);
lean_inc_ref(x_13);
lean_inc(x_12);
lean_inc_ref(x_11);
lean_inc_ref(x_10);
x_30 = lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg(x_9, x_29, x_7, x_10, x_28, x_11, x_12, x_13, x_14);
if (lean_obj_tag(x_30) == 0)
{
lean_object* x_31; lean_object* x_32; lean_object* x_33; uint8_t x_34; uint8_t x_35; lean_object* x_36; 
x_31 = lean_ctor_get(x_30, 0);
lean_inc(x_31);
lean_dec_ref(x_30);
x_32 = lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__5;
x_33 = lean_array_push(x_32, x_10);
x_34 = 1;
x_35 = 1;
x_36 = l_Lean_Meta_mkLambdaFVars(x_33, x_31, x_24, x_34, x_24, x_34, x_35, x_11, x_12, x_13, x_14);
lean_dec(x_14);
lean_dec_ref(x_13);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec_ref(x_33);
return x_36;
}
else
{
lean_dec(x_14);
lean_dec_ref(x_13);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec_ref(x_10);
return x_30;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
uint8_t x_16; lean_object* x_17; 
x_16 = lean_unbox(x_4);
x_17 = lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1(x_1, x_2, x_3, x_16, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_14; uint8_t x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_14 = lean_box(0);
x_15 = 0;
x_16 = lean_box(x_15);
lean_inc_ref(x_4);
x_17 = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___boxed), 15, 9);
lean_closure_set(x_17, 0, x_1);
lean_closure_set(x_17, 1, x_2);
lean_closure_set(x_17, 2, x_3);
lean_closure_set(x_17, 3, x_16);
lean_closure_set(x_17, 4, x_5);
lean_closure_set(x_17, 5, x_6);
lean_closure_set(x_17, 6, x_7);
lean_closure_set(x_17, 7, x_8);
lean_closure_set(x_17, 8, x_4);
x_18 = lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0___redArg(x_14, x_15, x_4, x_17, x_9, x_10, x_11, x_12);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_ExistsAndEq_mkAfterToBefore(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___lam__0(uint8_t x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, uint8_t x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; 
x_11 = l_Lean_Meta_mkFreshLevelMVar(x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_11) == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
lean_dec_ref(x_11);
lean_inc(x_12);
x_13 = l_Lean_Expr_sort___override(x_12);
x_14 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_14, 0, x_13);
lean_inc_ref(x_6);
lean_inc(x_2);
x_15 = l_Lean_Meta_mkFreshExprMVar(x_14, x_1, x_2, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_15) == 0)
{
lean_object* x_16; uint8_t x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_16 = lean_ctor_get(x_15, 0);
lean_inc(x_16);
lean_dec_ref(x_15);
x_17 = 0;
lean_inc(x_16);
lean_inc(x_2);
x_18 = l_Lean_Expr_forallE___override(x_2, x_16, x_3, x_17);
x_19 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_19, 0, x_18);
lean_inc_ref(x_6);
x_20 = l_Lean_Meta_mkFreshExprMVar(x_19, x_1, x_2, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_20) == 0)
{
lean_object* x_21; lean_object* x_22; uint8_t x_23; 
x_21 = lean_ctor_get(x_20, 0);
lean_inc(x_21);
lean_dec_ref(x_20);
x_22 = l_Lean_Meta_Context_config(x_6);
x_23 = !lean_is_exclusive(x_22);
if (x_23 == 0)
{
uint8_t x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; uint8_t x_31; uint8_t x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; uint8_t x_39; uint64_t x_40; uint8_t x_41; 
x_24 = lean_ctor_get_uint8(x_6, sizeof(void*)*7);
x_25 = lean_ctor_get(x_6, 1);
lean_inc(x_25);
x_26 = lean_ctor_get(x_6, 2);
lean_inc_ref(x_26);
x_27 = lean_ctor_get(x_6, 3);
lean_inc_ref(x_27);
x_28 = lean_ctor_get(x_6, 4);
lean_inc(x_28);
x_29 = lean_ctor_get(x_6, 5);
lean_inc(x_29);
x_30 = lean_ctor_get(x_6, 6);
lean_inc(x_30);
x_31 = lean_ctor_get_uint8(x_6, sizeof(void*)*7 + 1);
x_32 = lean_ctor_get_uint8(x_6, sizeof(void*)*7 + 2);
x_33 = lp_mathlib_ExistsAndEq_mkNestedExists___closed__1;
x_34 = lean_box(0);
lean_inc(x_12);
x_35 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_35, 0, x_12);
lean_ctor_set(x_35, 1, x_34);
x_36 = l_Lean_Expr_const___override(x_33, x_35);
lean_inc(x_16);
x_37 = l_Lean_Expr_app___override(x_36, x_16);
lean_inc(x_21);
x_38 = l_Lean_Expr_app___override(x_37, x_21);
x_39 = 2;
lean_ctor_set_uint8(x_22, 9, x_39);
x_40 = l_Lean_Meta_Context_configKey(x_6);
x_41 = !lean_is_exclusive(x_6);
if (x_41 == 0)
{
lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; uint64_t x_49; uint64_t x_50; uint64_t x_51; uint64_t x_52; uint64_t x_53; lean_object* x_54; lean_object* x_55; 
x_42 = lean_ctor_get(x_6, 6);
lean_dec(x_42);
x_43 = lean_ctor_get(x_6, 5);
lean_dec(x_43);
x_44 = lean_ctor_get(x_6, 4);
lean_dec(x_44);
x_45 = lean_ctor_get(x_6, 3);
lean_dec(x_45);
x_46 = lean_ctor_get(x_6, 2);
lean_dec(x_46);
x_47 = lean_ctor_get(x_6, 1);
lean_dec(x_47);
x_48 = lean_ctor_get(x_6, 0);
lean_dec(x_48);
x_49 = 2;
x_50 = lean_uint64_shift_right(x_40, x_49);
x_51 = lean_uint64_shift_left(x_50, x_49);
x_52 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_53 = lean_uint64_lor(x_51, x_52);
x_54 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_54, 0, x_22);
lean_ctor_set_uint64(x_54, sizeof(void*)*1, x_53);
lean_ctor_set(x_6, 0, x_54);
lean_inc(x_7);
x_55 = l_Lean_Meta_isExprDefEq(x_38, x_4, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_55) == 0)
{
uint8_t x_56; 
x_56 = !lean_is_exclusive(x_55);
if (x_56 == 0)
{
lean_object* x_57; uint8_t x_58; 
x_57 = lean_ctor_get(x_55, 0);
x_58 = lean_unbox(x_57);
if (x_58 == 0)
{
lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; 
lean_dec(x_57);
lean_dec(x_7);
x_59 = lean_box(x_5);
x_60 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_60, 0, x_21);
lean_ctor_set(x_60, 1, x_59);
x_61 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_61, 0, x_16);
lean_ctor_set(x_61, 1, x_60);
x_62 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_62, 0, x_12);
lean_ctor_set(x_62, 1, x_61);
lean_ctor_set(x_55, 0, x_62);
return x_55;
}
else
{
lean_object* x_63; 
lean_free_object(x_55);
x_63 = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg(x_12, x_7);
if (lean_obj_tag(x_63) == 0)
{
lean_object* x_64; lean_object* x_65; 
x_64 = lean_ctor_get(x_63, 0);
lean_inc(x_64);
lean_dec_ref(x_63);
x_65 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_16, x_7);
if (lean_obj_tag(x_65) == 0)
{
lean_object* x_66; lean_object* x_67; 
x_66 = lean_ctor_get(x_65, 0);
lean_inc(x_66);
lean_dec_ref(x_65);
x_67 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_21, x_7);
lean_dec(x_7);
if (lean_obj_tag(x_67) == 0)
{
uint8_t x_68; 
x_68 = !lean_is_exclusive(x_67);
if (x_68 == 0)
{
lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; 
x_69 = lean_ctor_get(x_67, 0);
x_70 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_70, 0, x_69);
lean_ctor_set(x_70, 1, x_57);
x_71 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_71, 0, x_66);
lean_ctor_set(x_71, 1, x_70);
x_72 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_72, 0, x_64);
lean_ctor_set(x_72, 1, x_71);
lean_ctor_set(x_67, 0, x_72);
return x_67;
}
else
{
lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; 
x_73 = lean_ctor_get(x_67, 0);
lean_inc(x_73);
lean_dec(x_67);
x_74 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_74, 0, x_73);
lean_ctor_set(x_74, 1, x_57);
x_75 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_75, 0, x_66);
lean_ctor_set(x_75, 1, x_74);
x_76 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_76, 0, x_64);
lean_ctor_set(x_76, 1, x_75);
x_77 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_77, 0, x_76);
return x_77;
}
}
else
{
uint8_t x_78; 
lean_dec(x_66);
lean_dec(x_64);
lean_dec(x_57);
x_78 = !lean_is_exclusive(x_67);
if (x_78 == 0)
{
return x_67;
}
else
{
lean_object* x_79; lean_object* x_80; 
x_79 = lean_ctor_get(x_67, 0);
lean_inc(x_79);
lean_dec(x_67);
x_80 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_80, 0, x_79);
return x_80;
}
}
}
else
{
uint8_t x_81; 
lean_dec(x_64);
lean_dec(x_57);
lean_dec(x_21);
lean_dec(x_7);
x_81 = !lean_is_exclusive(x_65);
if (x_81 == 0)
{
return x_65;
}
else
{
lean_object* x_82; lean_object* x_83; 
x_82 = lean_ctor_get(x_65, 0);
lean_inc(x_82);
lean_dec(x_65);
x_83 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_83, 0, x_82);
return x_83;
}
}
}
else
{
uint8_t x_84; 
lean_dec(x_57);
lean_dec(x_21);
lean_dec(x_16);
lean_dec(x_7);
x_84 = !lean_is_exclusive(x_63);
if (x_84 == 0)
{
return x_63;
}
else
{
lean_object* x_85; lean_object* x_86; 
x_85 = lean_ctor_get(x_63, 0);
lean_inc(x_85);
lean_dec(x_63);
x_86 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_86, 0, x_85);
return x_86;
}
}
}
}
else
{
lean_object* x_87; uint8_t x_88; 
x_87 = lean_ctor_get(x_55, 0);
lean_inc(x_87);
lean_dec(x_55);
x_88 = lean_unbox(x_87);
if (x_88 == 0)
{
lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; 
lean_dec(x_87);
lean_dec(x_7);
x_89 = lean_box(x_5);
x_90 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_90, 0, x_21);
lean_ctor_set(x_90, 1, x_89);
x_91 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_91, 0, x_16);
lean_ctor_set(x_91, 1, x_90);
x_92 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_92, 0, x_12);
lean_ctor_set(x_92, 1, x_91);
x_93 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_93, 0, x_92);
return x_93;
}
else
{
lean_object* x_94; 
x_94 = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg(x_12, x_7);
if (lean_obj_tag(x_94) == 0)
{
lean_object* x_95; lean_object* x_96; 
x_95 = lean_ctor_get(x_94, 0);
lean_inc(x_95);
lean_dec_ref(x_94);
x_96 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_16, x_7);
if (lean_obj_tag(x_96) == 0)
{
lean_object* x_97; lean_object* x_98; 
x_97 = lean_ctor_get(x_96, 0);
lean_inc(x_97);
lean_dec_ref(x_96);
x_98 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_21, x_7);
lean_dec(x_7);
if (lean_obj_tag(x_98) == 0)
{
lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; 
x_99 = lean_ctor_get(x_98, 0);
lean_inc(x_99);
if (lean_is_exclusive(x_98)) {
 lean_ctor_release(x_98, 0);
 x_100 = x_98;
} else {
 lean_dec_ref(x_98);
 x_100 = lean_box(0);
}
x_101 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_101, 0, x_99);
lean_ctor_set(x_101, 1, x_87);
x_102 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_102, 0, x_97);
lean_ctor_set(x_102, 1, x_101);
x_103 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_103, 0, x_95);
lean_ctor_set(x_103, 1, x_102);
if (lean_is_scalar(x_100)) {
 x_104 = lean_alloc_ctor(0, 1, 0);
} else {
 x_104 = x_100;
}
lean_ctor_set(x_104, 0, x_103);
return x_104;
}
else
{
lean_object* x_105; lean_object* x_106; lean_object* x_107; 
lean_dec(x_97);
lean_dec(x_95);
lean_dec(x_87);
x_105 = lean_ctor_get(x_98, 0);
lean_inc(x_105);
if (lean_is_exclusive(x_98)) {
 lean_ctor_release(x_98, 0);
 x_106 = x_98;
} else {
 lean_dec_ref(x_98);
 x_106 = lean_box(0);
}
if (lean_is_scalar(x_106)) {
 x_107 = lean_alloc_ctor(1, 1, 0);
} else {
 x_107 = x_106;
}
lean_ctor_set(x_107, 0, x_105);
return x_107;
}
}
else
{
lean_object* x_108; lean_object* x_109; lean_object* x_110; 
lean_dec(x_95);
lean_dec(x_87);
lean_dec(x_21);
lean_dec(x_7);
x_108 = lean_ctor_get(x_96, 0);
lean_inc(x_108);
if (lean_is_exclusive(x_96)) {
 lean_ctor_release(x_96, 0);
 x_109 = x_96;
} else {
 lean_dec_ref(x_96);
 x_109 = lean_box(0);
}
if (lean_is_scalar(x_109)) {
 x_110 = lean_alloc_ctor(1, 1, 0);
} else {
 x_110 = x_109;
}
lean_ctor_set(x_110, 0, x_108);
return x_110;
}
}
else
{
lean_object* x_111; lean_object* x_112; lean_object* x_113; 
lean_dec(x_87);
lean_dec(x_21);
lean_dec(x_16);
lean_dec(x_7);
x_111 = lean_ctor_get(x_94, 0);
lean_inc(x_111);
if (lean_is_exclusive(x_94)) {
 lean_ctor_release(x_94, 0);
 x_112 = x_94;
} else {
 lean_dec_ref(x_94);
 x_112 = lean_box(0);
}
if (lean_is_scalar(x_112)) {
 x_113 = lean_alloc_ctor(1, 1, 0);
} else {
 x_113 = x_112;
}
lean_ctor_set(x_113, 0, x_111);
return x_113;
}
}
}
}
else
{
uint8_t x_114; 
lean_dec(x_21);
lean_dec(x_16);
lean_dec(x_12);
lean_dec(x_7);
x_114 = !lean_is_exclusive(x_55);
if (x_114 == 0)
{
return x_55;
}
else
{
lean_object* x_115; lean_object* x_116; 
x_115 = lean_ctor_get(x_55, 0);
lean_inc(x_115);
lean_dec(x_55);
x_116 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_116, 0, x_115);
return x_116;
}
}
}
else
{
uint64_t x_117; uint64_t x_118; uint64_t x_119; uint64_t x_120; uint64_t x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; 
lean_dec(x_6);
x_117 = 2;
x_118 = lean_uint64_shift_right(x_40, x_117);
x_119 = lean_uint64_shift_left(x_118, x_117);
x_120 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_121 = lean_uint64_lor(x_119, x_120);
x_122 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_122, 0, x_22);
lean_ctor_set_uint64(x_122, sizeof(void*)*1, x_121);
x_123 = lean_alloc_ctor(0, 7, 3);
lean_ctor_set(x_123, 0, x_122);
lean_ctor_set(x_123, 1, x_25);
lean_ctor_set(x_123, 2, x_26);
lean_ctor_set(x_123, 3, x_27);
lean_ctor_set(x_123, 4, x_28);
lean_ctor_set(x_123, 5, x_29);
lean_ctor_set(x_123, 6, x_30);
lean_ctor_set_uint8(x_123, sizeof(void*)*7, x_24);
lean_ctor_set_uint8(x_123, sizeof(void*)*7 + 1, x_31);
lean_ctor_set_uint8(x_123, sizeof(void*)*7 + 2, x_32);
lean_inc(x_7);
x_124 = l_Lean_Meta_isExprDefEq(x_38, x_4, x_123, x_7, x_8, x_9);
if (lean_obj_tag(x_124) == 0)
{
lean_object* x_125; lean_object* x_126; uint8_t x_127; 
x_125 = lean_ctor_get(x_124, 0);
lean_inc(x_125);
if (lean_is_exclusive(x_124)) {
 lean_ctor_release(x_124, 0);
 x_126 = x_124;
} else {
 lean_dec_ref(x_124);
 x_126 = lean_box(0);
}
x_127 = lean_unbox(x_125);
if (x_127 == 0)
{
lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; lean_object* x_132; 
lean_dec(x_125);
lean_dec(x_7);
x_128 = lean_box(x_5);
x_129 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_129, 0, x_21);
lean_ctor_set(x_129, 1, x_128);
x_130 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_130, 0, x_16);
lean_ctor_set(x_130, 1, x_129);
x_131 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_131, 0, x_12);
lean_ctor_set(x_131, 1, x_130);
if (lean_is_scalar(x_126)) {
 x_132 = lean_alloc_ctor(0, 1, 0);
} else {
 x_132 = x_126;
}
lean_ctor_set(x_132, 0, x_131);
return x_132;
}
else
{
lean_object* x_133; 
lean_dec(x_126);
x_133 = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg(x_12, x_7);
if (lean_obj_tag(x_133) == 0)
{
lean_object* x_134; lean_object* x_135; 
x_134 = lean_ctor_get(x_133, 0);
lean_inc(x_134);
lean_dec_ref(x_133);
x_135 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_16, x_7);
if (lean_obj_tag(x_135) == 0)
{
lean_object* x_136; lean_object* x_137; 
x_136 = lean_ctor_get(x_135, 0);
lean_inc(x_136);
lean_dec_ref(x_135);
x_137 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_21, x_7);
lean_dec(x_7);
if (lean_obj_tag(x_137) == 0)
{
lean_object* x_138; lean_object* x_139; lean_object* x_140; lean_object* x_141; lean_object* x_142; lean_object* x_143; 
x_138 = lean_ctor_get(x_137, 0);
lean_inc(x_138);
if (lean_is_exclusive(x_137)) {
 lean_ctor_release(x_137, 0);
 x_139 = x_137;
} else {
 lean_dec_ref(x_137);
 x_139 = lean_box(0);
}
x_140 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_140, 0, x_138);
lean_ctor_set(x_140, 1, x_125);
x_141 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_141, 0, x_136);
lean_ctor_set(x_141, 1, x_140);
x_142 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_142, 0, x_134);
lean_ctor_set(x_142, 1, x_141);
if (lean_is_scalar(x_139)) {
 x_143 = lean_alloc_ctor(0, 1, 0);
} else {
 x_143 = x_139;
}
lean_ctor_set(x_143, 0, x_142);
return x_143;
}
else
{
lean_object* x_144; lean_object* x_145; lean_object* x_146; 
lean_dec(x_136);
lean_dec(x_134);
lean_dec(x_125);
x_144 = lean_ctor_get(x_137, 0);
lean_inc(x_144);
if (lean_is_exclusive(x_137)) {
 lean_ctor_release(x_137, 0);
 x_145 = x_137;
} else {
 lean_dec_ref(x_137);
 x_145 = lean_box(0);
}
if (lean_is_scalar(x_145)) {
 x_146 = lean_alloc_ctor(1, 1, 0);
} else {
 x_146 = x_145;
}
lean_ctor_set(x_146, 0, x_144);
return x_146;
}
}
else
{
lean_object* x_147; lean_object* x_148; lean_object* x_149; 
lean_dec(x_134);
lean_dec(x_125);
lean_dec(x_21);
lean_dec(x_7);
x_147 = lean_ctor_get(x_135, 0);
lean_inc(x_147);
if (lean_is_exclusive(x_135)) {
 lean_ctor_release(x_135, 0);
 x_148 = x_135;
} else {
 lean_dec_ref(x_135);
 x_148 = lean_box(0);
}
if (lean_is_scalar(x_148)) {
 x_149 = lean_alloc_ctor(1, 1, 0);
} else {
 x_149 = x_148;
}
lean_ctor_set(x_149, 0, x_147);
return x_149;
}
}
else
{
lean_object* x_150; lean_object* x_151; lean_object* x_152; 
lean_dec(x_125);
lean_dec(x_21);
lean_dec(x_16);
lean_dec(x_7);
x_150 = lean_ctor_get(x_133, 0);
lean_inc(x_150);
if (lean_is_exclusive(x_133)) {
 lean_ctor_release(x_133, 0);
 x_151 = x_133;
} else {
 lean_dec_ref(x_133);
 x_151 = lean_box(0);
}
if (lean_is_scalar(x_151)) {
 x_152 = lean_alloc_ctor(1, 1, 0);
} else {
 x_152 = x_151;
}
lean_ctor_set(x_152, 0, x_150);
return x_152;
}
}
}
else
{
lean_object* x_153; lean_object* x_154; lean_object* x_155; 
lean_dec(x_21);
lean_dec(x_16);
lean_dec(x_12);
lean_dec(x_7);
x_153 = lean_ctor_get(x_124, 0);
lean_inc(x_153);
if (lean_is_exclusive(x_124)) {
 lean_ctor_release(x_124, 0);
 x_154 = x_124;
} else {
 lean_dec_ref(x_124);
 x_154 = lean_box(0);
}
if (lean_is_scalar(x_154)) {
 x_155 = lean_alloc_ctor(1, 1, 0);
} else {
 x_155 = x_154;
}
lean_ctor_set(x_155, 0, x_153);
return x_155;
}
}
}
else
{
uint8_t x_156; uint8_t x_157; uint8_t x_158; uint8_t x_159; uint8_t x_160; uint8_t x_161; uint8_t x_162; uint8_t x_163; uint8_t x_164; uint8_t x_165; uint8_t x_166; uint8_t x_167; uint8_t x_168; uint8_t x_169; uint8_t x_170; uint8_t x_171; uint8_t x_172; uint8_t x_173; uint8_t x_174; lean_object* x_175; lean_object* x_176; lean_object* x_177; lean_object* x_178; lean_object* x_179; lean_object* x_180; uint8_t x_181; uint8_t x_182; lean_object* x_183; lean_object* x_184; lean_object* x_185; lean_object* x_186; lean_object* x_187; lean_object* x_188; uint8_t x_189; lean_object* x_190; uint64_t x_191; lean_object* x_192; uint64_t x_193; uint64_t x_194; uint64_t x_195; uint64_t x_196; uint64_t x_197; lean_object* x_198; lean_object* x_199; lean_object* x_200; 
x_156 = lean_ctor_get_uint8(x_22, 0);
x_157 = lean_ctor_get_uint8(x_22, 1);
x_158 = lean_ctor_get_uint8(x_22, 2);
x_159 = lean_ctor_get_uint8(x_22, 3);
x_160 = lean_ctor_get_uint8(x_22, 4);
x_161 = lean_ctor_get_uint8(x_22, 5);
x_162 = lean_ctor_get_uint8(x_22, 6);
x_163 = lean_ctor_get_uint8(x_22, 7);
x_164 = lean_ctor_get_uint8(x_22, 8);
x_165 = lean_ctor_get_uint8(x_22, 10);
x_166 = lean_ctor_get_uint8(x_22, 11);
x_167 = lean_ctor_get_uint8(x_22, 12);
x_168 = lean_ctor_get_uint8(x_22, 13);
x_169 = lean_ctor_get_uint8(x_22, 14);
x_170 = lean_ctor_get_uint8(x_22, 15);
x_171 = lean_ctor_get_uint8(x_22, 16);
x_172 = lean_ctor_get_uint8(x_22, 17);
x_173 = lean_ctor_get_uint8(x_22, 18);
lean_dec(x_22);
x_174 = lean_ctor_get_uint8(x_6, sizeof(void*)*7);
x_175 = lean_ctor_get(x_6, 1);
lean_inc(x_175);
x_176 = lean_ctor_get(x_6, 2);
lean_inc_ref(x_176);
x_177 = lean_ctor_get(x_6, 3);
lean_inc_ref(x_177);
x_178 = lean_ctor_get(x_6, 4);
lean_inc(x_178);
x_179 = lean_ctor_get(x_6, 5);
lean_inc(x_179);
x_180 = lean_ctor_get(x_6, 6);
lean_inc(x_180);
x_181 = lean_ctor_get_uint8(x_6, sizeof(void*)*7 + 1);
x_182 = lean_ctor_get_uint8(x_6, sizeof(void*)*7 + 2);
x_183 = lp_mathlib_ExistsAndEq_mkNestedExists___closed__1;
x_184 = lean_box(0);
lean_inc(x_12);
x_185 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_185, 0, x_12);
lean_ctor_set(x_185, 1, x_184);
x_186 = l_Lean_Expr_const___override(x_183, x_185);
lean_inc(x_16);
x_187 = l_Lean_Expr_app___override(x_186, x_16);
lean_inc(x_21);
x_188 = l_Lean_Expr_app___override(x_187, x_21);
x_189 = 2;
x_190 = lean_alloc_ctor(0, 0, 19);
lean_ctor_set_uint8(x_190, 0, x_156);
lean_ctor_set_uint8(x_190, 1, x_157);
lean_ctor_set_uint8(x_190, 2, x_158);
lean_ctor_set_uint8(x_190, 3, x_159);
lean_ctor_set_uint8(x_190, 4, x_160);
lean_ctor_set_uint8(x_190, 5, x_161);
lean_ctor_set_uint8(x_190, 6, x_162);
lean_ctor_set_uint8(x_190, 7, x_163);
lean_ctor_set_uint8(x_190, 8, x_164);
lean_ctor_set_uint8(x_190, 9, x_189);
lean_ctor_set_uint8(x_190, 10, x_165);
lean_ctor_set_uint8(x_190, 11, x_166);
lean_ctor_set_uint8(x_190, 12, x_167);
lean_ctor_set_uint8(x_190, 13, x_168);
lean_ctor_set_uint8(x_190, 14, x_169);
lean_ctor_set_uint8(x_190, 15, x_170);
lean_ctor_set_uint8(x_190, 16, x_171);
lean_ctor_set_uint8(x_190, 17, x_172);
lean_ctor_set_uint8(x_190, 18, x_173);
x_191 = l_Lean_Meta_Context_configKey(x_6);
if (lean_is_exclusive(x_6)) {
 lean_ctor_release(x_6, 0);
 lean_ctor_release(x_6, 1);
 lean_ctor_release(x_6, 2);
 lean_ctor_release(x_6, 3);
 lean_ctor_release(x_6, 4);
 lean_ctor_release(x_6, 5);
 lean_ctor_release(x_6, 6);
 x_192 = x_6;
} else {
 lean_dec_ref(x_6);
 x_192 = lean_box(0);
}
x_193 = 2;
x_194 = lean_uint64_shift_right(x_191, x_193);
x_195 = lean_uint64_shift_left(x_194, x_193);
x_196 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_197 = lean_uint64_lor(x_195, x_196);
x_198 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_198, 0, x_190);
lean_ctor_set_uint64(x_198, sizeof(void*)*1, x_197);
if (lean_is_scalar(x_192)) {
 x_199 = lean_alloc_ctor(0, 7, 3);
} else {
 x_199 = x_192;
}
lean_ctor_set(x_199, 0, x_198);
lean_ctor_set(x_199, 1, x_175);
lean_ctor_set(x_199, 2, x_176);
lean_ctor_set(x_199, 3, x_177);
lean_ctor_set(x_199, 4, x_178);
lean_ctor_set(x_199, 5, x_179);
lean_ctor_set(x_199, 6, x_180);
lean_ctor_set_uint8(x_199, sizeof(void*)*7, x_174);
lean_ctor_set_uint8(x_199, sizeof(void*)*7 + 1, x_181);
lean_ctor_set_uint8(x_199, sizeof(void*)*7 + 2, x_182);
lean_inc(x_7);
x_200 = l_Lean_Meta_isExprDefEq(x_188, x_4, x_199, x_7, x_8, x_9);
if (lean_obj_tag(x_200) == 0)
{
lean_object* x_201; lean_object* x_202; uint8_t x_203; 
x_201 = lean_ctor_get(x_200, 0);
lean_inc(x_201);
if (lean_is_exclusive(x_200)) {
 lean_ctor_release(x_200, 0);
 x_202 = x_200;
} else {
 lean_dec_ref(x_200);
 x_202 = lean_box(0);
}
x_203 = lean_unbox(x_201);
if (x_203 == 0)
{
lean_object* x_204; lean_object* x_205; lean_object* x_206; lean_object* x_207; lean_object* x_208; 
lean_dec(x_201);
lean_dec(x_7);
x_204 = lean_box(x_5);
x_205 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_205, 0, x_21);
lean_ctor_set(x_205, 1, x_204);
x_206 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_206, 0, x_16);
lean_ctor_set(x_206, 1, x_205);
x_207 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_207, 0, x_12);
lean_ctor_set(x_207, 1, x_206);
if (lean_is_scalar(x_202)) {
 x_208 = lean_alloc_ctor(0, 1, 0);
} else {
 x_208 = x_202;
}
lean_ctor_set(x_208, 0, x_207);
return x_208;
}
else
{
lean_object* x_209; 
lean_dec(x_202);
x_209 = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg(x_12, x_7);
if (lean_obj_tag(x_209) == 0)
{
lean_object* x_210; lean_object* x_211; 
x_210 = lean_ctor_get(x_209, 0);
lean_inc(x_210);
lean_dec_ref(x_209);
x_211 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_16, x_7);
if (lean_obj_tag(x_211) == 0)
{
lean_object* x_212; lean_object* x_213; 
x_212 = lean_ctor_get(x_211, 0);
lean_inc(x_212);
lean_dec_ref(x_211);
x_213 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_21, x_7);
lean_dec(x_7);
if (lean_obj_tag(x_213) == 0)
{
lean_object* x_214; lean_object* x_215; lean_object* x_216; lean_object* x_217; lean_object* x_218; lean_object* x_219; 
x_214 = lean_ctor_get(x_213, 0);
lean_inc(x_214);
if (lean_is_exclusive(x_213)) {
 lean_ctor_release(x_213, 0);
 x_215 = x_213;
} else {
 lean_dec_ref(x_213);
 x_215 = lean_box(0);
}
x_216 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_216, 0, x_214);
lean_ctor_set(x_216, 1, x_201);
x_217 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_217, 0, x_212);
lean_ctor_set(x_217, 1, x_216);
x_218 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_218, 0, x_210);
lean_ctor_set(x_218, 1, x_217);
if (lean_is_scalar(x_215)) {
 x_219 = lean_alloc_ctor(0, 1, 0);
} else {
 x_219 = x_215;
}
lean_ctor_set(x_219, 0, x_218);
return x_219;
}
else
{
lean_object* x_220; lean_object* x_221; lean_object* x_222; 
lean_dec(x_212);
lean_dec(x_210);
lean_dec(x_201);
x_220 = lean_ctor_get(x_213, 0);
lean_inc(x_220);
if (lean_is_exclusive(x_213)) {
 lean_ctor_release(x_213, 0);
 x_221 = x_213;
} else {
 lean_dec_ref(x_213);
 x_221 = lean_box(0);
}
if (lean_is_scalar(x_221)) {
 x_222 = lean_alloc_ctor(1, 1, 0);
} else {
 x_222 = x_221;
}
lean_ctor_set(x_222, 0, x_220);
return x_222;
}
}
else
{
lean_object* x_223; lean_object* x_224; lean_object* x_225; 
lean_dec(x_210);
lean_dec(x_201);
lean_dec(x_21);
lean_dec(x_7);
x_223 = lean_ctor_get(x_211, 0);
lean_inc(x_223);
if (lean_is_exclusive(x_211)) {
 lean_ctor_release(x_211, 0);
 x_224 = x_211;
} else {
 lean_dec_ref(x_211);
 x_224 = lean_box(0);
}
if (lean_is_scalar(x_224)) {
 x_225 = lean_alloc_ctor(1, 1, 0);
} else {
 x_225 = x_224;
}
lean_ctor_set(x_225, 0, x_223);
return x_225;
}
}
else
{
lean_object* x_226; lean_object* x_227; lean_object* x_228; 
lean_dec(x_201);
lean_dec(x_21);
lean_dec(x_16);
lean_dec(x_7);
x_226 = lean_ctor_get(x_209, 0);
lean_inc(x_226);
if (lean_is_exclusive(x_209)) {
 lean_ctor_release(x_209, 0);
 x_227 = x_209;
} else {
 lean_dec_ref(x_209);
 x_227 = lean_box(0);
}
if (lean_is_scalar(x_227)) {
 x_228 = lean_alloc_ctor(1, 1, 0);
} else {
 x_228 = x_227;
}
lean_ctor_set(x_228, 0, x_226);
return x_228;
}
}
}
else
{
lean_object* x_229; lean_object* x_230; lean_object* x_231; 
lean_dec(x_21);
lean_dec(x_16);
lean_dec(x_12);
lean_dec(x_7);
x_229 = lean_ctor_get(x_200, 0);
lean_inc(x_229);
if (lean_is_exclusive(x_200)) {
 lean_ctor_release(x_200, 0);
 x_230 = x_200;
} else {
 lean_dec_ref(x_200);
 x_230 = lean_box(0);
}
if (lean_is_scalar(x_230)) {
 x_231 = lean_alloc_ctor(1, 1, 0);
} else {
 x_231 = x_230;
}
lean_ctor_set(x_231, 0, x_229);
return x_231;
}
}
}
else
{
uint8_t x_232; 
lean_dec(x_16);
lean_dec(x_12);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_4);
x_232 = !lean_is_exclusive(x_20);
if (x_232 == 0)
{
return x_20;
}
else
{
lean_object* x_233; lean_object* x_234; 
x_233 = lean_ctor_get(x_20, 0);
lean_inc(x_233);
lean_dec(x_20);
x_234 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_234, 0, x_233);
return x_234;
}
}
}
else
{
uint8_t x_235; 
lean_dec(x_12);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
x_235 = !lean_is_exclusive(x_15);
if (x_235 == 0)
{
return x_15;
}
else
{
lean_object* x_236; lean_object* x_237; 
x_236 = lean_ctor_get(x_15, 0);
lean_inc(x_236);
lean_dec(x_15);
x_237 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_237, 0, x_236);
return x_237;
}
}
}
else
{
uint8_t x_238; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
x_238 = !lean_is_exclusive(x_11);
if (x_238 == 0)
{
return x_11;
}
else
{
lean_object* x_239; lean_object* x_240; 
x_239 = lean_ctor_get(x_11, 0);
lean_inc(x_239);
lean_dec(x_11);
x_240 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_240, 0, x_239);
return x_240;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
uint8_t x_11; uint8_t x_12; lean_object* x_13; 
x_11 = lean_unbox(x_1);
x_12 = lean_unbox(x_5);
x_13 = lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___lam__0(x_11, x_2, x_3, x_4, x_12, x_6, x_7, x_8, x_9);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___lam__2(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; 
lean_inc_ref(x_6);
lean_inc(x_3);
x_11 = l_Lean_Meta_mkFreshExprMVar(x_1, x_2, x_3, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_11) == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
lean_dec_ref(x_11);
lean_inc(x_12);
x_13 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_13, 0, x_12);
lean_inc_ref(x_6);
lean_inc(x_3);
lean_inc_ref(x_13);
x_14 = l_Lean_Meta_mkFreshExprMVar(x_13, x_2, x_3, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_14) == 0)
{
lean_object* x_15; lean_object* x_16; 
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
lean_dec_ref(x_14);
lean_inc_ref(x_6);
x_16 = l_Lean_Meta_mkFreshExprMVar(x_13, x_2, x_3, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_16) == 0)
{
lean_object* x_17; lean_object* x_18; uint8_t x_19; 
x_17 = lean_ctor_get(x_16, 0);
lean_inc(x_17);
lean_dec_ref(x_16);
x_18 = l_Lean_Meta_Context_config(x_6);
x_19 = !lean_is_exclusive(x_18);
if (x_19 == 0)
{
uint8_t x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; uint8_t x_27; uint8_t x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; uint8_t x_36; uint64_t x_37; uint8_t x_38; 
x_20 = lean_ctor_get_uint8(x_6, sizeof(void*)*7);
x_21 = lean_ctor_get(x_6, 1);
lean_inc(x_21);
x_22 = lean_ctor_get(x_6, 2);
lean_inc_ref(x_22);
x_23 = lean_ctor_get(x_6, 3);
lean_inc_ref(x_23);
x_24 = lean_ctor_get(x_6, 4);
lean_inc(x_24);
x_25 = lean_ctor_get(x_6, 5);
lean_inc(x_25);
x_26 = lean_ctor_get(x_6, 6);
lean_inc(x_26);
x_27 = lean_ctor_get_uint8(x_6, sizeof(void*)*7 + 1);
x_28 = lean_ctor_get_uint8(x_6, sizeof(void*)*7 + 2);
x_29 = lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__4;
x_30 = lean_box(0);
x_31 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_31, 0, x_4);
lean_ctor_set(x_31, 1, x_30);
x_32 = l_Lean_Expr_const___override(x_29, x_31);
lean_inc(x_12);
x_33 = l_Lean_Expr_app___override(x_32, x_12);
lean_inc(x_15);
x_34 = l_Lean_Expr_app___override(x_33, x_15);
lean_inc(x_17);
x_35 = l_Lean_Expr_app___override(x_34, x_17);
x_36 = 2;
lean_ctor_set_uint8(x_18, 9, x_36);
x_37 = l_Lean_Meta_Context_configKey(x_6);
x_38 = !lean_is_exclusive(x_6);
if (x_38 == 0)
{
lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; uint64_t x_46; uint64_t x_47; uint64_t x_48; uint64_t x_49; uint64_t x_50; lean_object* x_51; lean_object* x_52; 
x_39 = lean_ctor_get(x_6, 6);
lean_dec(x_39);
x_40 = lean_ctor_get(x_6, 5);
lean_dec(x_40);
x_41 = lean_ctor_get(x_6, 4);
lean_dec(x_41);
x_42 = lean_ctor_get(x_6, 3);
lean_dec(x_42);
x_43 = lean_ctor_get(x_6, 2);
lean_dec(x_43);
x_44 = lean_ctor_get(x_6, 1);
lean_dec(x_44);
x_45 = lean_ctor_get(x_6, 0);
lean_dec(x_45);
x_46 = 2;
x_47 = lean_uint64_shift_right(x_37, x_46);
x_48 = lean_uint64_shift_left(x_47, x_46);
x_49 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_50 = lean_uint64_lor(x_48, x_49);
x_51 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_51, 0, x_18);
lean_ctor_set_uint64(x_51, sizeof(void*)*1, x_50);
lean_ctor_set(x_6, 0, x_51);
lean_inc(x_7);
x_52 = l_Lean_Meta_isExprDefEq(x_35, x_5, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_52) == 0)
{
uint8_t x_53; 
x_53 = !lean_is_exclusive(x_52);
if (x_53 == 0)
{
lean_object* x_54; uint8_t x_55; 
x_54 = lean_ctor_get(x_52, 0);
x_55 = lean_unbox(x_54);
if (x_55 == 0)
{
lean_object* x_56; lean_object* x_57; lean_object* x_58; 
lean_dec(x_7);
x_56 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_56, 0, x_17);
lean_ctor_set(x_56, 1, x_54);
x_57 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_57, 0, x_15);
lean_ctor_set(x_57, 1, x_56);
x_58 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_58, 0, x_12);
lean_ctor_set(x_58, 1, x_57);
lean_ctor_set(x_52, 0, x_58);
return x_52;
}
else
{
lean_object* x_59; 
lean_free_object(x_52);
x_59 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_12, x_7);
if (lean_obj_tag(x_59) == 0)
{
lean_object* x_60; lean_object* x_61; 
x_60 = lean_ctor_get(x_59, 0);
lean_inc(x_60);
lean_dec_ref(x_59);
x_61 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_15, x_7);
if (lean_obj_tag(x_61) == 0)
{
lean_object* x_62; lean_object* x_63; 
x_62 = lean_ctor_get(x_61, 0);
lean_inc(x_62);
lean_dec_ref(x_61);
x_63 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_17, x_7);
lean_dec(x_7);
if (lean_obj_tag(x_63) == 0)
{
uint8_t x_64; 
x_64 = !lean_is_exclusive(x_63);
if (x_64 == 0)
{
lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; 
x_65 = lean_ctor_get(x_63, 0);
x_66 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_66, 0, x_65);
lean_ctor_set(x_66, 1, x_54);
x_67 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_67, 0, x_62);
lean_ctor_set(x_67, 1, x_66);
x_68 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_68, 0, x_60);
lean_ctor_set(x_68, 1, x_67);
lean_ctor_set(x_63, 0, x_68);
return x_63;
}
else
{
lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; 
x_69 = lean_ctor_get(x_63, 0);
lean_inc(x_69);
lean_dec(x_63);
x_70 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_70, 0, x_69);
lean_ctor_set(x_70, 1, x_54);
x_71 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_71, 0, x_62);
lean_ctor_set(x_71, 1, x_70);
x_72 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_72, 0, x_60);
lean_ctor_set(x_72, 1, x_71);
x_73 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_73, 0, x_72);
return x_73;
}
}
else
{
uint8_t x_74; 
lean_dec(x_62);
lean_dec(x_60);
lean_dec(x_54);
x_74 = !lean_is_exclusive(x_63);
if (x_74 == 0)
{
return x_63;
}
else
{
lean_object* x_75; lean_object* x_76; 
x_75 = lean_ctor_get(x_63, 0);
lean_inc(x_75);
lean_dec(x_63);
x_76 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_76, 0, x_75);
return x_76;
}
}
}
else
{
uint8_t x_77; 
lean_dec(x_60);
lean_dec(x_54);
lean_dec(x_17);
lean_dec(x_7);
x_77 = !lean_is_exclusive(x_61);
if (x_77 == 0)
{
return x_61;
}
else
{
lean_object* x_78; lean_object* x_79; 
x_78 = lean_ctor_get(x_61, 0);
lean_inc(x_78);
lean_dec(x_61);
x_79 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_79, 0, x_78);
return x_79;
}
}
}
else
{
uint8_t x_80; 
lean_dec(x_54);
lean_dec(x_17);
lean_dec(x_15);
lean_dec(x_7);
x_80 = !lean_is_exclusive(x_59);
if (x_80 == 0)
{
return x_59;
}
else
{
lean_object* x_81; lean_object* x_82; 
x_81 = lean_ctor_get(x_59, 0);
lean_inc(x_81);
lean_dec(x_59);
x_82 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_82, 0, x_81);
return x_82;
}
}
}
}
else
{
lean_object* x_83; uint8_t x_84; 
x_83 = lean_ctor_get(x_52, 0);
lean_inc(x_83);
lean_dec(x_52);
x_84 = lean_unbox(x_83);
if (x_84 == 0)
{
lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; 
lean_dec(x_7);
x_85 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_85, 0, x_17);
lean_ctor_set(x_85, 1, x_83);
x_86 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_86, 0, x_15);
lean_ctor_set(x_86, 1, x_85);
x_87 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_87, 0, x_12);
lean_ctor_set(x_87, 1, x_86);
x_88 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_88, 0, x_87);
return x_88;
}
else
{
lean_object* x_89; 
x_89 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_12, x_7);
if (lean_obj_tag(x_89) == 0)
{
lean_object* x_90; lean_object* x_91; 
x_90 = lean_ctor_get(x_89, 0);
lean_inc(x_90);
lean_dec_ref(x_89);
x_91 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_15, x_7);
if (lean_obj_tag(x_91) == 0)
{
lean_object* x_92; lean_object* x_93; 
x_92 = lean_ctor_get(x_91, 0);
lean_inc(x_92);
lean_dec_ref(x_91);
x_93 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_17, x_7);
lean_dec(x_7);
if (lean_obj_tag(x_93) == 0)
{
lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; 
x_94 = lean_ctor_get(x_93, 0);
lean_inc(x_94);
if (lean_is_exclusive(x_93)) {
 lean_ctor_release(x_93, 0);
 x_95 = x_93;
} else {
 lean_dec_ref(x_93);
 x_95 = lean_box(0);
}
x_96 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_96, 0, x_94);
lean_ctor_set(x_96, 1, x_83);
x_97 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_97, 0, x_92);
lean_ctor_set(x_97, 1, x_96);
x_98 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_98, 0, x_90);
lean_ctor_set(x_98, 1, x_97);
if (lean_is_scalar(x_95)) {
 x_99 = lean_alloc_ctor(0, 1, 0);
} else {
 x_99 = x_95;
}
lean_ctor_set(x_99, 0, x_98);
return x_99;
}
else
{
lean_object* x_100; lean_object* x_101; lean_object* x_102; 
lean_dec(x_92);
lean_dec(x_90);
lean_dec(x_83);
x_100 = lean_ctor_get(x_93, 0);
lean_inc(x_100);
if (lean_is_exclusive(x_93)) {
 lean_ctor_release(x_93, 0);
 x_101 = x_93;
} else {
 lean_dec_ref(x_93);
 x_101 = lean_box(0);
}
if (lean_is_scalar(x_101)) {
 x_102 = lean_alloc_ctor(1, 1, 0);
} else {
 x_102 = x_101;
}
lean_ctor_set(x_102, 0, x_100);
return x_102;
}
}
else
{
lean_object* x_103; lean_object* x_104; lean_object* x_105; 
lean_dec(x_90);
lean_dec(x_83);
lean_dec(x_17);
lean_dec(x_7);
x_103 = lean_ctor_get(x_91, 0);
lean_inc(x_103);
if (lean_is_exclusive(x_91)) {
 lean_ctor_release(x_91, 0);
 x_104 = x_91;
} else {
 lean_dec_ref(x_91);
 x_104 = lean_box(0);
}
if (lean_is_scalar(x_104)) {
 x_105 = lean_alloc_ctor(1, 1, 0);
} else {
 x_105 = x_104;
}
lean_ctor_set(x_105, 0, x_103);
return x_105;
}
}
else
{
lean_object* x_106; lean_object* x_107; lean_object* x_108; 
lean_dec(x_83);
lean_dec(x_17);
lean_dec(x_15);
lean_dec(x_7);
x_106 = lean_ctor_get(x_89, 0);
lean_inc(x_106);
if (lean_is_exclusive(x_89)) {
 lean_ctor_release(x_89, 0);
 x_107 = x_89;
} else {
 lean_dec_ref(x_89);
 x_107 = lean_box(0);
}
if (lean_is_scalar(x_107)) {
 x_108 = lean_alloc_ctor(1, 1, 0);
} else {
 x_108 = x_107;
}
lean_ctor_set(x_108, 0, x_106);
return x_108;
}
}
}
}
else
{
uint8_t x_109; 
lean_dec(x_17);
lean_dec(x_15);
lean_dec(x_12);
lean_dec(x_7);
x_109 = !lean_is_exclusive(x_52);
if (x_109 == 0)
{
return x_52;
}
else
{
lean_object* x_110; lean_object* x_111; 
x_110 = lean_ctor_get(x_52, 0);
lean_inc(x_110);
lean_dec(x_52);
x_111 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_111, 0, x_110);
return x_111;
}
}
}
else
{
uint64_t x_112; uint64_t x_113; uint64_t x_114; uint64_t x_115; uint64_t x_116; lean_object* x_117; lean_object* x_118; lean_object* x_119; 
lean_dec(x_6);
x_112 = 2;
x_113 = lean_uint64_shift_right(x_37, x_112);
x_114 = lean_uint64_shift_left(x_113, x_112);
x_115 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_116 = lean_uint64_lor(x_114, x_115);
x_117 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_117, 0, x_18);
lean_ctor_set_uint64(x_117, sizeof(void*)*1, x_116);
x_118 = lean_alloc_ctor(0, 7, 3);
lean_ctor_set(x_118, 0, x_117);
lean_ctor_set(x_118, 1, x_21);
lean_ctor_set(x_118, 2, x_22);
lean_ctor_set(x_118, 3, x_23);
lean_ctor_set(x_118, 4, x_24);
lean_ctor_set(x_118, 5, x_25);
lean_ctor_set(x_118, 6, x_26);
lean_ctor_set_uint8(x_118, sizeof(void*)*7, x_20);
lean_ctor_set_uint8(x_118, sizeof(void*)*7 + 1, x_27);
lean_ctor_set_uint8(x_118, sizeof(void*)*7 + 2, x_28);
lean_inc(x_7);
x_119 = l_Lean_Meta_isExprDefEq(x_35, x_5, x_118, x_7, x_8, x_9);
if (lean_obj_tag(x_119) == 0)
{
lean_object* x_120; lean_object* x_121; uint8_t x_122; 
x_120 = lean_ctor_get(x_119, 0);
lean_inc(x_120);
if (lean_is_exclusive(x_119)) {
 lean_ctor_release(x_119, 0);
 x_121 = x_119;
} else {
 lean_dec_ref(x_119);
 x_121 = lean_box(0);
}
x_122 = lean_unbox(x_120);
if (x_122 == 0)
{
lean_object* x_123; lean_object* x_124; lean_object* x_125; lean_object* x_126; 
lean_dec(x_7);
x_123 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_123, 0, x_17);
lean_ctor_set(x_123, 1, x_120);
x_124 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_124, 0, x_15);
lean_ctor_set(x_124, 1, x_123);
x_125 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_125, 0, x_12);
lean_ctor_set(x_125, 1, x_124);
if (lean_is_scalar(x_121)) {
 x_126 = lean_alloc_ctor(0, 1, 0);
} else {
 x_126 = x_121;
}
lean_ctor_set(x_126, 0, x_125);
return x_126;
}
else
{
lean_object* x_127; 
lean_dec(x_121);
x_127 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_12, x_7);
if (lean_obj_tag(x_127) == 0)
{
lean_object* x_128; lean_object* x_129; 
x_128 = lean_ctor_get(x_127, 0);
lean_inc(x_128);
lean_dec_ref(x_127);
x_129 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_15, x_7);
if (lean_obj_tag(x_129) == 0)
{
lean_object* x_130; lean_object* x_131; 
x_130 = lean_ctor_get(x_129, 0);
lean_inc(x_130);
lean_dec_ref(x_129);
x_131 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_17, x_7);
lean_dec(x_7);
if (lean_obj_tag(x_131) == 0)
{
lean_object* x_132; lean_object* x_133; lean_object* x_134; lean_object* x_135; lean_object* x_136; lean_object* x_137; 
x_132 = lean_ctor_get(x_131, 0);
lean_inc(x_132);
if (lean_is_exclusive(x_131)) {
 lean_ctor_release(x_131, 0);
 x_133 = x_131;
} else {
 lean_dec_ref(x_131);
 x_133 = lean_box(0);
}
x_134 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_134, 0, x_132);
lean_ctor_set(x_134, 1, x_120);
x_135 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_135, 0, x_130);
lean_ctor_set(x_135, 1, x_134);
x_136 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_136, 0, x_128);
lean_ctor_set(x_136, 1, x_135);
if (lean_is_scalar(x_133)) {
 x_137 = lean_alloc_ctor(0, 1, 0);
} else {
 x_137 = x_133;
}
lean_ctor_set(x_137, 0, x_136);
return x_137;
}
else
{
lean_object* x_138; lean_object* x_139; lean_object* x_140; 
lean_dec(x_130);
lean_dec(x_128);
lean_dec(x_120);
x_138 = lean_ctor_get(x_131, 0);
lean_inc(x_138);
if (lean_is_exclusive(x_131)) {
 lean_ctor_release(x_131, 0);
 x_139 = x_131;
} else {
 lean_dec_ref(x_131);
 x_139 = lean_box(0);
}
if (lean_is_scalar(x_139)) {
 x_140 = lean_alloc_ctor(1, 1, 0);
} else {
 x_140 = x_139;
}
lean_ctor_set(x_140, 0, x_138);
return x_140;
}
}
else
{
lean_object* x_141; lean_object* x_142; lean_object* x_143; 
lean_dec(x_128);
lean_dec(x_120);
lean_dec(x_17);
lean_dec(x_7);
x_141 = lean_ctor_get(x_129, 0);
lean_inc(x_141);
if (lean_is_exclusive(x_129)) {
 lean_ctor_release(x_129, 0);
 x_142 = x_129;
} else {
 lean_dec_ref(x_129);
 x_142 = lean_box(0);
}
if (lean_is_scalar(x_142)) {
 x_143 = lean_alloc_ctor(1, 1, 0);
} else {
 x_143 = x_142;
}
lean_ctor_set(x_143, 0, x_141);
return x_143;
}
}
else
{
lean_object* x_144; lean_object* x_145; lean_object* x_146; 
lean_dec(x_120);
lean_dec(x_17);
lean_dec(x_15);
lean_dec(x_7);
x_144 = lean_ctor_get(x_127, 0);
lean_inc(x_144);
if (lean_is_exclusive(x_127)) {
 lean_ctor_release(x_127, 0);
 x_145 = x_127;
} else {
 lean_dec_ref(x_127);
 x_145 = lean_box(0);
}
if (lean_is_scalar(x_145)) {
 x_146 = lean_alloc_ctor(1, 1, 0);
} else {
 x_146 = x_145;
}
lean_ctor_set(x_146, 0, x_144);
return x_146;
}
}
}
else
{
lean_object* x_147; lean_object* x_148; lean_object* x_149; 
lean_dec(x_17);
lean_dec(x_15);
lean_dec(x_12);
lean_dec(x_7);
x_147 = lean_ctor_get(x_119, 0);
lean_inc(x_147);
if (lean_is_exclusive(x_119)) {
 lean_ctor_release(x_119, 0);
 x_148 = x_119;
} else {
 lean_dec_ref(x_119);
 x_148 = lean_box(0);
}
if (lean_is_scalar(x_148)) {
 x_149 = lean_alloc_ctor(1, 1, 0);
} else {
 x_149 = x_148;
}
lean_ctor_set(x_149, 0, x_147);
return x_149;
}
}
}
else
{
uint8_t x_150; uint8_t x_151; uint8_t x_152; uint8_t x_153; uint8_t x_154; uint8_t x_155; uint8_t x_156; uint8_t x_157; uint8_t x_158; uint8_t x_159; uint8_t x_160; uint8_t x_161; uint8_t x_162; uint8_t x_163; uint8_t x_164; uint8_t x_165; uint8_t x_166; uint8_t x_167; uint8_t x_168; lean_object* x_169; lean_object* x_170; lean_object* x_171; lean_object* x_172; lean_object* x_173; lean_object* x_174; uint8_t x_175; uint8_t x_176; lean_object* x_177; lean_object* x_178; lean_object* x_179; lean_object* x_180; lean_object* x_181; lean_object* x_182; lean_object* x_183; uint8_t x_184; lean_object* x_185; uint64_t x_186; lean_object* x_187; uint64_t x_188; uint64_t x_189; uint64_t x_190; uint64_t x_191; uint64_t x_192; lean_object* x_193; lean_object* x_194; lean_object* x_195; 
x_150 = lean_ctor_get_uint8(x_18, 0);
x_151 = lean_ctor_get_uint8(x_18, 1);
x_152 = lean_ctor_get_uint8(x_18, 2);
x_153 = lean_ctor_get_uint8(x_18, 3);
x_154 = lean_ctor_get_uint8(x_18, 4);
x_155 = lean_ctor_get_uint8(x_18, 5);
x_156 = lean_ctor_get_uint8(x_18, 6);
x_157 = lean_ctor_get_uint8(x_18, 7);
x_158 = lean_ctor_get_uint8(x_18, 8);
x_159 = lean_ctor_get_uint8(x_18, 10);
x_160 = lean_ctor_get_uint8(x_18, 11);
x_161 = lean_ctor_get_uint8(x_18, 12);
x_162 = lean_ctor_get_uint8(x_18, 13);
x_163 = lean_ctor_get_uint8(x_18, 14);
x_164 = lean_ctor_get_uint8(x_18, 15);
x_165 = lean_ctor_get_uint8(x_18, 16);
x_166 = lean_ctor_get_uint8(x_18, 17);
x_167 = lean_ctor_get_uint8(x_18, 18);
lean_dec(x_18);
x_168 = lean_ctor_get_uint8(x_6, sizeof(void*)*7);
x_169 = lean_ctor_get(x_6, 1);
lean_inc(x_169);
x_170 = lean_ctor_get(x_6, 2);
lean_inc_ref(x_170);
x_171 = lean_ctor_get(x_6, 3);
lean_inc_ref(x_171);
x_172 = lean_ctor_get(x_6, 4);
lean_inc(x_172);
x_173 = lean_ctor_get(x_6, 5);
lean_inc(x_173);
x_174 = lean_ctor_get(x_6, 6);
lean_inc(x_174);
x_175 = lean_ctor_get_uint8(x_6, sizeof(void*)*7 + 1);
x_176 = lean_ctor_get_uint8(x_6, sizeof(void*)*7 + 2);
x_177 = lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__4;
x_178 = lean_box(0);
x_179 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_179, 0, x_4);
lean_ctor_set(x_179, 1, x_178);
x_180 = l_Lean_Expr_const___override(x_177, x_179);
lean_inc(x_12);
x_181 = l_Lean_Expr_app___override(x_180, x_12);
lean_inc(x_15);
x_182 = l_Lean_Expr_app___override(x_181, x_15);
lean_inc(x_17);
x_183 = l_Lean_Expr_app___override(x_182, x_17);
x_184 = 2;
x_185 = lean_alloc_ctor(0, 0, 19);
lean_ctor_set_uint8(x_185, 0, x_150);
lean_ctor_set_uint8(x_185, 1, x_151);
lean_ctor_set_uint8(x_185, 2, x_152);
lean_ctor_set_uint8(x_185, 3, x_153);
lean_ctor_set_uint8(x_185, 4, x_154);
lean_ctor_set_uint8(x_185, 5, x_155);
lean_ctor_set_uint8(x_185, 6, x_156);
lean_ctor_set_uint8(x_185, 7, x_157);
lean_ctor_set_uint8(x_185, 8, x_158);
lean_ctor_set_uint8(x_185, 9, x_184);
lean_ctor_set_uint8(x_185, 10, x_159);
lean_ctor_set_uint8(x_185, 11, x_160);
lean_ctor_set_uint8(x_185, 12, x_161);
lean_ctor_set_uint8(x_185, 13, x_162);
lean_ctor_set_uint8(x_185, 14, x_163);
lean_ctor_set_uint8(x_185, 15, x_164);
lean_ctor_set_uint8(x_185, 16, x_165);
lean_ctor_set_uint8(x_185, 17, x_166);
lean_ctor_set_uint8(x_185, 18, x_167);
x_186 = l_Lean_Meta_Context_configKey(x_6);
if (lean_is_exclusive(x_6)) {
 lean_ctor_release(x_6, 0);
 lean_ctor_release(x_6, 1);
 lean_ctor_release(x_6, 2);
 lean_ctor_release(x_6, 3);
 lean_ctor_release(x_6, 4);
 lean_ctor_release(x_6, 5);
 lean_ctor_release(x_6, 6);
 x_187 = x_6;
} else {
 lean_dec_ref(x_6);
 x_187 = lean_box(0);
}
x_188 = 2;
x_189 = lean_uint64_shift_right(x_186, x_188);
x_190 = lean_uint64_shift_left(x_189, x_188);
x_191 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_192 = lean_uint64_lor(x_190, x_191);
x_193 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_193, 0, x_185);
lean_ctor_set_uint64(x_193, sizeof(void*)*1, x_192);
if (lean_is_scalar(x_187)) {
 x_194 = lean_alloc_ctor(0, 7, 3);
} else {
 x_194 = x_187;
}
lean_ctor_set(x_194, 0, x_193);
lean_ctor_set(x_194, 1, x_169);
lean_ctor_set(x_194, 2, x_170);
lean_ctor_set(x_194, 3, x_171);
lean_ctor_set(x_194, 4, x_172);
lean_ctor_set(x_194, 5, x_173);
lean_ctor_set(x_194, 6, x_174);
lean_ctor_set_uint8(x_194, sizeof(void*)*7, x_168);
lean_ctor_set_uint8(x_194, sizeof(void*)*7 + 1, x_175);
lean_ctor_set_uint8(x_194, sizeof(void*)*7 + 2, x_176);
lean_inc(x_7);
x_195 = l_Lean_Meta_isExprDefEq(x_183, x_5, x_194, x_7, x_8, x_9);
if (lean_obj_tag(x_195) == 0)
{
lean_object* x_196; lean_object* x_197; uint8_t x_198; 
x_196 = lean_ctor_get(x_195, 0);
lean_inc(x_196);
if (lean_is_exclusive(x_195)) {
 lean_ctor_release(x_195, 0);
 x_197 = x_195;
} else {
 lean_dec_ref(x_195);
 x_197 = lean_box(0);
}
x_198 = lean_unbox(x_196);
if (x_198 == 0)
{
lean_object* x_199; lean_object* x_200; lean_object* x_201; lean_object* x_202; 
lean_dec(x_7);
x_199 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_199, 0, x_17);
lean_ctor_set(x_199, 1, x_196);
x_200 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_200, 0, x_15);
lean_ctor_set(x_200, 1, x_199);
x_201 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_201, 0, x_12);
lean_ctor_set(x_201, 1, x_200);
if (lean_is_scalar(x_197)) {
 x_202 = lean_alloc_ctor(0, 1, 0);
} else {
 x_202 = x_197;
}
lean_ctor_set(x_202, 0, x_201);
return x_202;
}
else
{
lean_object* x_203; 
lean_dec(x_197);
x_203 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_12, x_7);
if (lean_obj_tag(x_203) == 0)
{
lean_object* x_204; lean_object* x_205; 
x_204 = lean_ctor_get(x_203, 0);
lean_inc(x_204);
lean_dec_ref(x_203);
x_205 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_15, x_7);
if (lean_obj_tag(x_205) == 0)
{
lean_object* x_206; lean_object* x_207; 
x_206 = lean_ctor_get(x_205, 0);
lean_inc(x_206);
lean_dec_ref(x_205);
x_207 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_17, x_7);
lean_dec(x_7);
if (lean_obj_tag(x_207) == 0)
{
lean_object* x_208; lean_object* x_209; lean_object* x_210; lean_object* x_211; lean_object* x_212; lean_object* x_213; 
x_208 = lean_ctor_get(x_207, 0);
lean_inc(x_208);
if (lean_is_exclusive(x_207)) {
 lean_ctor_release(x_207, 0);
 x_209 = x_207;
} else {
 lean_dec_ref(x_207);
 x_209 = lean_box(0);
}
x_210 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_210, 0, x_208);
lean_ctor_set(x_210, 1, x_196);
x_211 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_211, 0, x_206);
lean_ctor_set(x_211, 1, x_210);
x_212 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_212, 0, x_204);
lean_ctor_set(x_212, 1, x_211);
if (lean_is_scalar(x_209)) {
 x_213 = lean_alloc_ctor(0, 1, 0);
} else {
 x_213 = x_209;
}
lean_ctor_set(x_213, 0, x_212);
return x_213;
}
else
{
lean_object* x_214; lean_object* x_215; lean_object* x_216; 
lean_dec(x_206);
lean_dec(x_204);
lean_dec(x_196);
x_214 = lean_ctor_get(x_207, 0);
lean_inc(x_214);
if (lean_is_exclusive(x_207)) {
 lean_ctor_release(x_207, 0);
 x_215 = x_207;
} else {
 lean_dec_ref(x_207);
 x_215 = lean_box(0);
}
if (lean_is_scalar(x_215)) {
 x_216 = lean_alloc_ctor(1, 1, 0);
} else {
 x_216 = x_215;
}
lean_ctor_set(x_216, 0, x_214);
return x_216;
}
}
else
{
lean_object* x_217; lean_object* x_218; lean_object* x_219; 
lean_dec(x_204);
lean_dec(x_196);
lean_dec(x_17);
lean_dec(x_7);
x_217 = lean_ctor_get(x_205, 0);
lean_inc(x_217);
if (lean_is_exclusive(x_205)) {
 lean_ctor_release(x_205, 0);
 x_218 = x_205;
} else {
 lean_dec_ref(x_205);
 x_218 = lean_box(0);
}
if (lean_is_scalar(x_218)) {
 x_219 = lean_alloc_ctor(1, 1, 0);
} else {
 x_219 = x_218;
}
lean_ctor_set(x_219, 0, x_217);
return x_219;
}
}
else
{
lean_object* x_220; lean_object* x_221; lean_object* x_222; 
lean_dec(x_196);
lean_dec(x_17);
lean_dec(x_15);
lean_dec(x_7);
x_220 = lean_ctor_get(x_203, 0);
lean_inc(x_220);
if (lean_is_exclusive(x_203)) {
 lean_ctor_release(x_203, 0);
 x_221 = x_203;
} else {
 lean_dec_ref(x_203);
 x_221 = lean_box(0);
}
if (lean_is_scalar(x_221)) {
 x_222 = lean_alloc_ctor(1, 1, 0);
} else {
 x_222 = x_221;
}
lean_ctor_set(x_222, 0, x_220);
return x_222;
}
}
}
else
{
lean_object* x_223; lean_object* x_224; lean_object* x_225; 
lean_dec(x_17);
lean_dec(x_15);
lean_dec(x_12);
lean_dec(x_7);
x_223 = lean_ctor_get(x_195, 0);
lean_inc(x_223);
if (lean_is_exclusive(x_195)) {
 lean_ctor_release(x_195, 0);
 x_224 = x_195;
} else {
 lean_dec_ref(x_195);
 x_224 = lean_box(0);
}
if (lean_is_scalar(x_224)) {
 x_225 = lean_alloc_ctor(1, 1, 0);
} else {
 x_225 = x_224;
}
lean_ctor_set(x_225, 0, x_223);
return x_225;
}
}
}
else
{
uint8_t x_226; 
lean_dec(x_15);
lean_dec(x_12);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
x_226 = !lean_is_exclusive(x_16);
if (x_226 == 0)
{
return x_16;
}
else
{
lean_object* x_227; lean_object* x_228; 
x_227 = lean_ctor_get(x_16, 0);
lean_inc(x_227);
lean_dec(x_16);
x_228 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_228, 0, x_227);
return x_228;
}
}
}
else
{
uint8_t x_229; 
lean_dec_ref(x_13);
lean_dec(x_12);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec(x_3);
x_229 = !lean_is_exclusive(x_14);
if (x_229 == 0)
{
return x_14;
}
else
{
lean_object* x_230; lean_object* x_231; 
x_230 = lean_ctor_get(x_14, 0);
lean_inc(x_230);
lean_dec(x_14);
x_231 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_231, 0, x_230);
return x_231;
}
}
}
else
{
uint8_t x_232; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec(x_3);
x_232 = !lean_is_exclusive(x_11);
if (x_232 == 0)
{
return x_11;
}
else
{
lean_object* x_233; lean_object* x_234; 
x_233 = lean_ctor_get(x_11, 0);
lean_inc(x_233);
lean_dec(x_11);
x_234 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_234, 0, x_233);
return x_234;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___lam__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
uint8_t x_11; lean_object* x_12; 
x_11 = lean_unbox(x_2);
x_12 = lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___lam__2(x_1, x_11, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_12;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("withExistsElimAlongPathImp: `P` is equality but neither of sides is `a`", 71, 71);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("symm", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__1;
x_2 = lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__3;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("withExistsElimAlongPathImp: unexpected P = ", 43, 43);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("withExistsElimAlongPathImp: `P` is equality but `path` is not empty", 67, 67);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("withExistsElimAlongPathImp: `P` is `And` but `path` is empty", 60, 60);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("withExistsElimAlongPathImp: `P` is `Exists` but `exs` is empty", 62, 62);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___lam__1___boxed(lean_object** _args) {
lean_object* x_1 = _args[0];
lean_object* x_2 = _args[1];
lean_object* x_3 = _args[2];
lean_object* x_4 = _args[3];
lean_object* x_5 = _args[4];
lean_object* x_6 = _args[5];
lean_object* x_7 = _args[6];
lean_object* x_8 = _args[7];
lean_object* x_9 = _args[8];
lean_object* x_10 = _args[9];
lean_object* x_11 = _args[10];
lean_object* x_12 = _args[11];
lean_object* x_13 = _args[12];
lean_object* x_14 = _args[13];
lean_object* x_15 = _args[14];
lean_object* x_16 = _args[15];
lean_object* x_17 = _args[16];
lean_object* x_18 = _args[17];
lean_object* x_19 = _args[18];
lean_object* x_20 = _args[19];
lean_object* x_21 = _args[20];
lean_object* x_22 = _args[21];
lean_object* x_23 = _args[22];
_start:
{
uint8_t x_24; lean_object* x_25; 
x_24 = lean_unbox(x_12);
x_25 = lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___lam__1(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_24, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22);
lean_dec(x_13);
return x_25;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
uint8_t x_17; lean_object* x_18; uint8_t x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_17 = 0;
x_18 = lean_box(0);
x_19 = 0;
x_20 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0;
x_21 = lean_box(x_17);
x_22 = lean_box(x_19);
lean_inc_ref(x_3);
x_23 = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___lam__0___boxed), 10, 5);
lean_closure_set(x_23, 0, x_21);
lean_closure_set(x_23, 1, x_18);
lean_closure_set(x_23, 2, x_20);
lean_closure_set(x_23, 3, x_3);
lean_closure_set(x_23, 4, x_22);
lean_inc(x_15);
lean_inc_ref(x_14);
lean_inc(x_13);
lean_inc_ref(x_12);
x_24 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(x_23, x_19, x_12, x_13, x_14, x_15);
if (lean_obj_tag(x_24) == 0)
{
lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; uint8_t x_29; 
x_25 = lean_ctor_get(x_24, 0);
lean_inc(x_25);
lean_dec_ref(x_24);
x_26 = lean_ctor_get(x_25, 1);
lean_inc(x_26);
lean_dec(x_25);
x_27 = lean_ctor_get(x_26, 1);
lean_inc(x_27);
x_28 = lean_ctor_get(x_27, 1);
lean_inc(x_28);
x_29 = lean_unbox(x_28);
if (x_29 == 0)
{
lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; 
lean_dec(x_28);
lean_dec(x_27);
lean_dec(x_26);
x_30 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__1;
x_31 = lean_box(x_17);
x_32 = lean_box(x_19);
lean_inc_ref(x_3);
x_33 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___boxed), 10, 5);
lean_closure_set(x_33, 0, x_30);
lean_closure_set(x_33, 1, x_31);
lean_closure_set(x_33, 2, x_18);
lean_closure_set(x_33, 3, x_3);
lean_closure_set(x_33, 4, x_32);
lean_inc(x_15);
lean_inc_ref(x_14);
lean_inc(x_13);
lean_inc_ref(x_12);
x_34 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(x_33, x_19, x_12, x_13, x_14, x_15);
if (lean_obj_tag(x_34) == 0)
{
lean_object* x_35; lean_object* x_36; lean_object* x_37; uint8_t x_38; 
x_35 = lean_ctor_get(x_34, 0);
lean_inc(x_35);
lean_dec_ref(x_34);
x_36 = lean_ctor_get(x_35, 1);
lean_inc(x_36);
x_37 = lean_ctor_get(x_36, 1);
x_38 = lean_unbox(x_37);
if (x_38 == 0)
{
lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; 
lean_dec(x_36);
lean_dec(x_35);
lean_dec(x_8);
lean_dec_ref(x_4);
lean_inc(x_1);
x_39 = l_Lean_Expr_sort___override(x_1);
x_40 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_40, 0, x_39);
x_41 = lean_box(x_17);
lean_inc_ref(x_3);
lean_inc(x_1);
x_42 = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___lam__2___boxed), 10, 5);
lean_closure_set(x_42, 0, x_40);
lean_closure_set(x_42, 1, x_41);
lean_closure_set(x_42, 2, x_18);
lean_closure_set(x_42, 3, x_1);
lean_closure_set(x_42, 4, x_3);
lean_inc(x_15);
lean_inc_ref(x_14);
lean_inc(x_13);
lean_inc_ref(x_12);
x_43 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(x_42, x_19, x_12, x_13, x_14, x_15);
if (lean_obj_tag(x_43) == 0)
{
lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; uint8_t x_71; 
x_44 = lean_ctor_get(x_43, 0);
lean_inc(x_44);
lean_dec_ref(x_43);
x_45 = lean_ctor_get(x_44, 1);
lean_inc(x_45);
lean_dec(x_44);
x_46 = lean_ctor_get(x_45, 1);
lean_inc(x_46);
x_47 = lean_ctor_get(x_45, 0);
lean_inc(x_47);
lean_dec(x_45);
x_48 = lean_ctor_get(x_46, 0);
lean_inc(x_48);
x_49 = lean_ctor_get(x_46, 1);
lean_inc(x_49);
if (lean_is_exclusive(x_46)) {
 lean_ctor_release(x_46, 0);
 lean_ctor_release(x_46, 1);
 x_50 = x_46;
} else {
 lean_dec_ref(x_46);
 x_50 = lean_box(0);
}
x_71 = lean_unbox(x_49);
lean_dec(x_49);
if (x_71 == 0)
{
lean_object* x_72; 
lean_dec(x_50);
lean_dec(x_48);
lean_dec(x_47);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
lean_dec(x_1);
x_72 = l_Lean_Meta_ppExpr(x_3, x_12, x_13, x_14, x_15);
if (lean_obj_tag(x_72) == 0)
{
lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; 
x_73 = lean_ctor_get(x_72, 0);
lean_inc(x_73);
lean_dec_ref(x_72);
x_74 = lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__3;
x_75 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3;
x_76 = lean_unsigned_to_nat(0u);
x_77 = l_Std_Format_pretty(x_73, x_75, x_76, x_76);
x_78 = lean_string_append(x_74, x_77);
lean_dec_ref(x_77);
x_79 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg(x_78, x_12, x_13, x_14, x_15);
lean_dec(x_15);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec_ref(x_78);
return x_79;
}
else
{
uint8_t x_80; 
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
x_80 = !lean_is_exclusive(x_72);
if (x_80 == 0)
{
return x_72;
}
else
{
lean_object* x_81; lean_object* x_82; 
x_81 = lean_ctor_get(x_72, 0);
lean_inc(x_81);
lean_dec(x_72);
x_82 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_82, 0, x_81);
return x_82;
}
}
}
else
{
uint8_t x_83; 
lean_dec_ref(x_3);
x_83 = l_List_isEmpty___redArg(x_9);
lean_dec(x_9);
if (x_83 == 0)
{
lean_object* x_84; lean_object* x_85; 
x_84 = lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__4;
lean_inc_ref(x_14);
x_85 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg(x_84, x_12, x_13, x_14, x_15);
if (lean_obj_tag(x_85) == 0)
{
lean_dec_ref(x_85);
x_51 = x_12;
x_52 = x_13;
x_53 = x_14;
x_54 = x_15;
x_55 = lean_box(0);
goto block_70;
}
else
{
uint8_t x_86; 
lean_dec(x_50);
lean_dec(x_48);
lean_dec(x_47);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
lean_dec(x_1);
x_86 = !lean_is_exclusive(x_85);
if (x_86 == 0)
{
return x_85;
}
else
{
lean_object* x_87; lean_object* x_88; 
x_87 = lean_ctor_get(x_85, 0);
lean_inc(x_87);
lean_dec(x_85);
x_88 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_88, 0, x_87);
return x_88;
}
}
}
else
{
x_51 = x_12;
x_52 = x_13;
x_53 = x_14;
x_54 = x_15;
x_55 = lean_box(0);
goto block_70;
}
}
block_70:
{
uint8_t x_56; 
x_56 = lean_expr_eqv(x_6, x_47);
lean_dec(x_47);
if (x_56 == 0)
{
uint8_t x_57; 
x_57 = lean_expr_eqv(x_6, x_48);
lean_dec(x_48);
if (x_57 == 0)
{
lean_object* x_58; lean_object* x_59; 
lean_dec(x_50);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
lean_dec(x_1);
x_58 = lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__0;
x_59 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg(x_58, x_51, x_52, x_53, x_54);
lean_dec(x_54);
lean_dec(x_52);
lean_dec_ref(x_51);
return x_59;
}
else
{
lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; 
x_60 = lean_box(0);
if (lean_is_scalar(x_50)) {
 x_61 = lean_alloc_ctor(1, 2, 0);
} else {
 x_61 = x_50;
 lean_ctor_set_tag(x_61, 1);
}
lean_ctor_set(x_61, 0, x_1);
lean_ctor_set(x_61, 1, x_60);
x_62 = lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__2;
x_63 = l_Lean_Expr_const___override(x_62, x_61);
x_64 = l_Lean_Expr_app___override(x_63, x_2);
x_65 = l_Lean_Expr_app___override(x_64, x_7);
x_66 = l_Lean_Expr_app___override(x_65, x_6);
x_67 = l_Lean_Expr_app___override(x_66, x_5);
x_68 = lean_apply_7(x_11, x_67, x_10, x_51, x_52, x_53, x_54, lean_box(0));
return x_68;
}
}
else
{
lean_object* x_69; 
lean_dec(x_50);
lean_dec(x_48);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_2);
lean_dec(x_1);
x_69 = lean_apply_7(x_11, x_5, x_10, x_51, x_52, x_53, x_54, lean_box(0));
return x_69;
}
}
}
else
{
uint8_t x_89; 
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_89 = !lean_is_exclusive(x_43);
if (x_89 == 0)
{
return x_43;
}
else
{
lean_object* x_90; lean_object* x_91; 
x_90 = lean_ctor_get(x_43, 0);
lean_inc(x_90);
lean_dec(x_43);
x_91 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_91, 0, x_90);
return x_91;
}
}
}
else
{
lean_dec_ref(x_3);
if (lean_obj_tag(x_9) == 0)
{
lean_object* x_92; lean_object* x_93; 
lean_dec(x_36);
lean_dec(x_35);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
lean_dec(x_1);
x_92 = lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__5;
x_93 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg(x_92, x_12, x_13, x_14, x_15);
lean_dec(x_15);
lean_dec(x_13);
lean_dec_ref(x_12);
return x_93;
}
else
{
lean_object* x_94; uint8_t x_95; 
x_94 = lean_ctor_get(x_9, 0);
x_95 = lean_unbox(x_94);
if (x_95 == 0)
{
lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; 
x_96 = lean_ctor_get(x_35, 0);
lean_inc(x_96);
lean_dec(x_35);
x_97 = lean_ctor_get(x_36, 0);
lean_inc(x_97);
lean_dec(x_36);
x_98 = lean_ctor_get(x_9, 1);
lean_inc(x_98);
lean_dec_ref(x_9);
x_99 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__8;
lean_inc(x_96);
x_100 = l_Lean_Expr_app___override(x_99, x_96);
x_101 = l_Lean_Expr_app___override(x_100, x_97);
x_102 = l_Lean_Expr_app___override(x_101, x_5);
x_3 = x_96;
x_5 = x_102;
x_9 = x_98;
goto _start;
}
else
{
lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; 
x_104 = lean_ctor_get(x_35, 0);
lean_inc(x_104);
lean_dec(x_35);
x_105 = lean_ctor_get(x_36, 0);
lean_inc(x_105);
lean_dec(x_36);
x_106 = lean_ctor_get(x_9, 1);
lean_inc(x_106);
lean_dec_ref(x_9);
x_107 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__11;
x_108 = l_Lean_Expr_app___override(x_107, x_104);
lean_inc(x_105);
x_109 = l_Lean_Expr_app___override(x_108, x_105);
x_110 = l_Lean_Expr_app___override(x_109, x_5);
x_3 = x_105;
x_5 = x_110;
x_9 = x_106;
goto _start;
}
}
}
}
else
{
uint8_t x_112; 
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_112 = !lean_is_exclusive(x_34);
if (x_112 == 0)
{
return x_34;
}
else
{
lean_object* x_113; lean_object* x_114; 
x_113 = lean_ctor_get(x_34, 0);
lean_inc(x_113);
lean_dec(x_34);
x_114 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_114, 0, x_113);
return x_114;
}
}
}
else
{
lean_dec_ref(x_3);
if (lean_obj_tag(x_8) == 0)
{
lean_object* x_115; lean_object* x_116; 
lean_dec(x_28);
lean_dec(x_27);
lean_dec(x_26);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
lean_dec(x_1);
x_115 = lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__6;
x_116 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg(x_115, x_12, x_13, x_14, x_15);
lean_dec(x_15);
lean_dec(x_13);
lean_dec_ref(x_12);
return x_116;
}
else
{
lean_object* x_117; lean_object* x_118; lean_object* x_119; lean_object* x_120; uint8_t x_121; 
x_117 = lean_ctor_get(x_8, 0);
lean_inc(x_117);
x_118 = lean_ctor_get(x_117, 1);
lean_inc(x_118);
x_119 = lean_ctor_get(x_26, 0);
lean_inc(x_119);
lean_dec(x_26);
x_120 = lean_ctor_get(x_27, 0);
lean_inc(x_120);
lean_dec(x_27);
x_121 = !lean_is_exclusive(x_8);
if (x_121 == 0)
{
lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; uint8_t x_126; lean_object* x_127; lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; lean_object* x_132; 
x_122 = lean_ctor_get(x_8, 1);
x_123 = lean_ctor_get(x_8, 0);
lean_dec(x_123);
x_124 = lean_ctor_get(x_117, 0);
lean_inc(x_124);
lean_dec(x_117);
x_125 = lean_ctor_get(x_118, 1);
lean_inc(x_125);
lean_dec(x_118);
x_126 = 0;
x_127 = lean_box(0);
lean_inc(x_125);
lean_ctor_set(x_8, 1, x_127);
lean_ctor_set(x_8, 0, x_125);
x_128 = lean_array_mk(x_8);
lean_inc(x_120);
x_129 = l_Lean_Expr_betaRev(x_120, x_128, x_19, x_19);
lean_dec_ref(x_128);
x_130 = lean_box(x_19);
lean_inc_ref(x_129);
x_131 = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___lam__1___boxed), 23, 17);
lean_closure_set(x_131, 0, x_129);
lean_closure_set(x_131, 1, x_10);
lean_closure_set(x_131, 2, x_1);
lean_closure_set(x_131, 3, x_2);
lean_closure_set(x_131, 4, x_4);
lean_closure_set(x_131, 5, x_6);
lean_closure_set(x_131, 6, x_7);
lean_closure_set(x_131, 7, x_122);
lean_closure_set(x_131, 8, x_9);
lean_closure_set(x_131, 9, x_11);
lean_closure_set(x_131, 10, x_125);
lean_closure_set(x_131, 11, x_130);
lean_closure_set(x_131, 12, x_28);
lean_closure_set(x_131, 13, x_124);
lean_closure_set(x_131, 14, x_119);
lean_closure_set(x_131, 15, x_120);
lean_closure_set(x_131, 16, x_5);
x_132 = lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0___redArg(x_18, x_126, x_129, x_131, x_12, x_13, x_14, x_15);
return x_132;
}
else
{
lean_object* x_133; lean_object* x_134; lean_object* x_135; uint8_t x_136; lean_object* x_137; lean_object* x_138; lean_object* x_139; lean_object* x_140; lean_object* x_141; lean_object* x_142; lean_object* x_143; 
x_133 = lean_ctor_get(x_8, 1);
lean_inc(x_133);
lean_dec(x_8);
x_134 = lean_ctor_get(x_117, 0);
lean_inc(x_134);
lean_dec(x_117);
x_135 = lean_ctor_get(x_118, 1);
lean_inc(x_135);
lean_dec(x_118);
x_136 = 0;
x_137 = lean_box(0);
lean_inc(x_135);
x_138 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_138, 0, x_135);
lean_ctor_set(x_138, 1, x_137);
x_139 = lean_array_mk(x_138);
lean_inc(x_120);
x_140 = l_Lean_Expr_betaRev(x_120, x_139, x_19, x_19);
lean_dec_ref(x_139);
x_141 = lean_box(x_19);
lean_inc_ref(x_140);
x_142 = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___lam__1___boxed), 23, 17);
lean_closure_set(x_142, 0, x_140);
lean_closure_set(x_142, 1, x_10);
lean_closure_set(x_142, 2, x_1);
lean_closure_set(x_142, 3, x_2);
lean_closure_set(x_142, 4, x_4);
lean_closure_set(x_142, 5, x_6);
lean_closure_set(x_142, 6, x_7);
lean_closure_set(x_142, 7, x_133);
lean_closure_set(x_142, 8, x_9);
lean_closure_set(x_142, 9, x_11);
lean_closure_set(x_142, 10, x_135);
lean_closure_set(x_142, 11, x_141);
lean_closure_set(x_142, 12, x_28);
lean_closure_set(x_142, 13, x_134);
lean_closure_set(x_142, 14, x_119);
lean_closure_set(x_142, 15, x_120);
lean_closure_set(x_142, 16, x_5);
x_143 = lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0___redArg(x_18, x_136, x_140, x_142, x_12, x_13, x_14, x_15);
return x_143;
}
}
}
}
else
{
uint8_t x_144; 
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_144 = !lean_is_exclusive(x_24);
if (x_144 == 0)
{
return x_24;
}
else
{
lean_object* x_145; lean_object* x_146; 
x_145 = lean_ctor_get(x_24, 0);
lean_inc(x_145);
lean_dec(x_24);
x_146 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_146, 0, x_145);
return x_146;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, uint8_t x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19, lean_object* x_20, lean_object* x_21, lean_object* x_22) {
_start:
{
lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; 
lean_inc_ref(x_18);
lean_inc_ref(x_1);
x_24 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_24, 0, x_1);
lean_ctor_set(x_24, 1, x_18);
x_25 = lean_box(0);
x_26 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_26, 0, x_24);
lean_ctor_set(x_26, 1, x_25);
x_27 = l_List_appendTR___redArg(x_2, x_26);
lean_inc(x_22);
lean_inc_ref(x_21);
lean_inc(x_20);
lean_inc_ref(x_19);
lean_inc_ref(x_18);
lean_inc_ref(x_5);
x_28 = lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp(x_3, x_4, x_1, x_5, x_18, x_6, x_7, x_8, x_9, x_27, x_10, x_19, x_20, x_21, x_22);
if (lean_obj_tag(x_28) == 0)
{
lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; uint8_t x_33; uint8_t x_34; uint8_t x_35; lean_object* x_36; 
x_29 = lean_ctor_get(x_28, 0);
lean_inc(x_29);
lean_dec_ref(x_28);
x_30 = lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1___closed__0;
x_31 = lean_array_push(x_30, x_11);
x_32 = lean_array_push(x_31, x_18);
x_33 = 1;
x_34 = lean_unbox(x_13);
x_35 = lean_unbox(x_13);
x_36 = l_Lean_Meta_mkLambdaFVars(x_32, x_29, x_12, x_34, x_12, x_35, x_33, x_19, x_20, x_21, x_22);
lean_dec(x_22);
lean_dec_ref(x_21);
lean_dec(x_20);
lean_dec_ref(x_19);
lean_dec_ref(x_32);
if (lean_obj_tag(x_36) == 0)
{
uint8_t x_37; 
x_37 = !lean_is_exclusive(x_36);
if (x_37 == 0)
{
lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; 
x_38 = lean_ctor_get(x_36, 0);
x_39 = lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1___closed__2;
x_40 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_40, 0, x_14);
lean_ctor_set(x_40, 1, x_25);
x_41 = l_Lean_Expr_const___override(x_39, x_40);
x_42 = l_Lean_Expr_app___override(x_41, x_15);
x_43 = l_Lean_Expr_app___override(x_42, x_16);
x_44 = l_Lean_Expr_app___override(x_43, x_5);
x_45 = l_Lean_Expr_app___override(x_44, x_17);
x_46 = l_Lean_Expr_app___override(x_45, x_38);
lean_ctor_set(x_36, 0, x_46);
return x_36;
}
else
{
lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; 
x_47 = lean_ctor_get(x_36, 0);
lean_inc(x_47);
lean_dec(x_36);
x_48 = lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1___closed__2;
x_49 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_49, 0, x_14);
lean_ctor_set(x_49, 1, x_25);
x_50 = l_Lean_Expr_const___override(x_48, x_49);
x_51 = l_Lean_Expr_app___override(x_50, x_15);
x_52 = l_Lean_Expr_app___override(x_51, x_16);
x_53 = l_Lean_Expr_app___override(x_52, x_5);
x_54 = l_Lean_Expr_app___override(x_53, x_17);
x_55 = l_Lean_Expr_app___override(x_54, x_47);
x_56 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_56, 0, x_55);
return x_56;
}
}
else
{
lean_dec_ref(x_17);
lean_dec(x_16);
lean_dec(x_15);
lean_dec(x_14);
lean_dec_ref(x_5);
return x_36;
}
}
else
{
lean_dec(x_22);
lean_dec_ref(x_21);
lean_dec(x_20);
lean_dec_ref(x_19);
lean_dec_ref(x_18);
lean_dec_ref(x_17);
lean_dec(x_16);
lean_dec(x_15);
lean_dec(x_14);
lean_dec_ref(x_11);
lean_dec_ref(x_5);
return x_28;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
lean_object* x_17; 
x_17 = lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withExistsElimAlongPath(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_16; lean_object* x_17; 
x_16 = lean_box(0);
x_17 = lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_16, x_10, x_11, x_12, x_13, x_14);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withExistsElimAlongPath___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_16; 
x_16 = lp_mathlib_ExistsAndEq_withExistsElimAlongPath(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
return x_16;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_withNestedExistsIntro___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("withNestedExistsIntro: `exs` is not empty but `P` is not `Exists`", 65, 65);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withNestedExistsIntro___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_9; 
lean_dec_ref(x_1);
x_9 = lean_apply_5(x_3, x_4, x_5, x_6, x_7, lean_box(0));
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; uint8_t x_12; 
x_10 = lean_ctor_get(x_2, 0);
lean_inc(x_10);
x_11 = lean_ctor_get(x_10, 1);
lean_inc(x_11);
x_12 = !lean_is_exclusive(x_2);
if (x_12 == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; uint8_t x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; uint8_t x_23; lean_object* x_24; 
x_13 = lean_ctor_get(x_2, 1);
x_14 = lean_ctor_get(x_2, 0);
lean_dec(x_14);
x_15 = lean_ctor_get(x_10, 0);
lean_inc(x_15);
lean_dec(x_10);
x_16 = lean_ctor_get(x_11, 1);
lean_inc(x_16);
lean_dec(x_11);
lean_inc(x_15);
x_17 = l_Lean_Expr_sort___override(x_15);
x_18 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_18, 0, x_17);
x_19 = 0;
x_20 = lean_box(0);
x_21 = lean_box(x_19);
lean_inc(x_15);
x_22 = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__0___boxed), 10, 5);
lean_closure_set(x_22, 0, x_18);
lean_closure_set(x_22, 1, x_21);
lean_closure_set(x_22, 2, x_20);
lean_closure_set(x_22, 3, x_15);
lean_closure_set(x_22, 4, x_1);
x_23 = 0;
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc_ref(x_4);
x_24 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(x_22, x_23, x_4, x_5, x_6, x_7);
if (lean_obj_tag(x_24) == 0)
{
lean_object* x_25; lean_object* x_26; lean_object* x_27; uint8_t x_28; 
x_25 = lean_ctor_get(x_24, 0);
lean_inc(x_25);
lean_dec_ref(x_24);
x_26 = lean_ctor_get(x_25, 1);
lean_inc(x_26);
x_27 = lean_ctor_get(x_26, 1);
x_28 = lean_unbox(x_27);
if (x_28 == 0)
{
lean_object* x_29; lean_object* x_30; 
lean_dec(x_26);
lean_dec(x_25);
lean_dec(x_16);
lean_dec(x_15);
lean_free_object(x_2);
lean_dec(x_13);
lean_dec_ref(x_3);
x_29 = lp_mathlib_ExistsAndEq_withNestedExistsIntro___redArg___closed__0;
x_30 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg(x_29, x_4, x_5, x_6, x_7);
lean_dec(x_7);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_30;
}
else
{
lean_object* x_31; uint8_t x_32; 
x_31 = lean_ctor_get(x_25, 0);
lean_inc(x_31);
lean_dec(x_25);
x_32 = !lean_is_exclusive(x_26);
if (x_32 == 0)
{
lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; 
x_33 = lean_ctor_get(x_26, 0);
x_34 = lean_ctor_get(x_26, 1);
lean_dec(x_34);
x_35 = lean_box(0);
lean_inc(x_16);
lean_ctor_set(x_2, 1, x_35);
lean_ctor_set(x_2, 0, x_16);
x_36 = lean_array_mk(x_2);
lean_inc(x_33);
x_37 = l_Lean_Expr_betaRev(x_33, x_36, x_23, x_23);
lean_dec_ref(x_36);
x_38 = lp_mathlib_ExistsAndEq_withNestedExistsIntro___redArg(x_37, x_13, x_3, x_4, x_5, x_6, x_7);
if (lean_obj_tag(x_38) == 0)
{
uint8_t x_39; 
x_39 = !lean_is_exclusive(x_38);
if (x_39 == 0)
{
lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; 
x_40 = lean_ctor_get(x_38, 0);
x_41 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__16;
lean_ctor_set_tag(x_26, 1);
lean_ctor_set(x_26, 1, x_35);
lean_ctor_set(x_26, 0, x_15);
x_42 = l_Lean_Expr_const___override(x_41, x_26);
x_43 = l_Lean_Expr_app___override(x_42, x_31);
x_44 = l_Lean_Expr_app___override(x_43, x_33);
x_45 = l_Lean_Expr_app___override(x_44, x_16);
x_46 = l_Lean_Expr_app___override(x_45, x_40);
lean_ctor_set(x_38, 0, x_46);
return x_38;
}
else
{
lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; 
x_47 = lean_ctor_get(x_38, 0);
lean_inc(x_47);
lean_dec(x_38);
x_48 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__16;
lean_ctor_set_tag(x_26, 1);
lean_ctor_set(x_26, 1, x_35);
lean_ctor_set(x_26, 0, x_15);
x_49 = l_Lean_Expr_const___override(x_48, x_26);
x_50 = l_Lean_Expr_app___override(x_49, x_31);
x_51 = l_Lean_Expr_app___override(x_50, x_33);
x_52 = l_Lean_Expr_app___override(x_51, x_16);
x_53 = l_Lean_Expr_app___override(x_52, x_47);
x_54 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_54, 0, x_53);
return x_54;
}
}
else
{
lean_free_object(x_26);
lean_dec(x_33);
lean_dec(x_31);
lean_dec(x_16);
lean_dec(x_15);
return x_38;
}
}
else
{
lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; 
x_55 = lean_ctor_get(x_26, 0);
lean_inc(x_55);
lean_dec(x_26);
x_56 = lean_box(0);
lean_inc(x_16);
lean_ctor_set(x_2, 1, x_56);
lean_ctor_set(x_2, 0, x_16);
x_57 = lean_array_mk(x_2);
lean_inc(x_55);
x_58 = l_Lean_Expr_betaRev(x_55, x_57, x_23, x_23);
lean_dec_ref(x_57);
x_59 = lp_mathlib_ExistsAndEq_withNestedExistsIntro___redArg(x_58, x_13, x_3, x_4, x_5, x_6, x_7);
if (lean_obj_tag(x_59) == 0)
{
lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; 
x_60 = lean_ctor_get(x_59, 0);
lean_inc(x_60);
if (lean_is_exclusive(x_59)) {
 lean_ctor_release(x_59, 0);
 x_61 = x_59;
} else {
 lean_dec_ref(x_59);
 x_61 = lean_box(0);
}
x_62 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__16;
x_63 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_63, 0, x_15);
lean_ctor_set(x_63, 1, x_56);
x_64 = l_Lean_Expr_const___override(x_62, x_63);
x_65 = l_Lean_Expr_app___override(x_64, x_31);
x_66 = l_Lean_Expr_app___override(x_65, x_55);
x_67 = l_Lean_Expr_app___override(x_66, x_16);
x_68 = l_Lean_Expr_app___override(x_67, x_60);
if (lean_is_scalar(x_61)) {
 x_69 = lean_alloc_ctor(0, 1, 0);
} else {
 x_69 = x_61;
}
lean_ctor_set(x_69, 0, x_68);
return x_69;
}
else
{
lean_dec(x_55);
lean_dec(x_31);
lean_dec(x_16);
lean_dec(x_15);
return x_59;
}
}
}
}
else
{
uint8_t x_70; 
lean_dec(x_16);
lean_dec(x_15);
lean_free_object(x_2);
lean_dec(x_13);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
x_70 = !lean_is_exclusive(x_24);
if (x_70 == 0)
{
return x_24;
}
else
{
lean_object* x_71; lean_object* x_72; 
x_71 = lean_ctor_get(x_24, 0);
lean_inc(x_71);
lean_dec(x_24);
x_72 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_72, 0, x_71);
return x_72;
}
}
}
else
{
lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; uint8_t x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; uint8_t x_82; lean_object* x_83; 
x_73 = lean_ctor_get(x_2, 1);
lean_inc(x_73);
lean_dec(x_2);
x_74 = lean_ctor_get(x_10, 0);
lean_inc(x_74);
lean_dec(x_10);
x_75 = lean_ctor_get(x_11, 1);
lean_inc(x_75);
lean_dec(x_11);
lean_inc(x_74);
x_76 = l_Lean_Expr_sort___override(x_74);
x_77 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_77, 0, x_76);
x_78 = 0;
x_79 = lean_box(0);
x_80 = lean_box(x_78);
lean_inc(x_74);
x_81 = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__0___boxed), 10, 5);
lean_closure_set(x_81, 0, x_77);
lean_closure_set(x_81, 1, x_80);
lean_closure_set(x_81, 2, x_79);
lean_closure_set(x_81, 3, x_74);
lean_closure_set(x_81, 4, x_1);
x_82 = 0;
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc_ref(x_4);
x_83 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(x_81, x_82, x_4, x_5, x_6, x_7);
if (lean_obj_tag(x_83) == 0)
{
lean_object* x_84; lean_object* x_85; lean_object* x_86; uint8_t x_87; 
x_84 = lean_ctor_get(x_83, 0);
lean_inc(x_84);
lean_dec_ref(x_83);
x_85 = lean_ctor_get(x_84, 1);
lean_inc(x_85);
x_86 = lean_ctor_get(x_85, 1);
x_87 = lean_unbox(x_86);
if (x_87 == 0)
{
lean_object* x_88; lean_object* x_89; 
lean_dec(x_85);
lean_dec(x_84);
lean_dec(x_75);
lean_dec(x_74);
lean_dec(x_73);
lean_dec_ref(x_3);
x_88 = lp_mathlib_ExistsAndEq_withNestedExistsIntro___redArg___closed__0;
x_89 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg(x_88, x_4, x_5, x_6, x_7);
lean_dec(x_7);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_89;
}
else
{
lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; 
x_90 = lean_ctor_get(x_84, 0);
lean_inc(x_90);
lean_dec(x_84);
x_91 = lean_ctor_get(x_85, 0);
lean_inc(x_91);
if (lean_is_exclusive(x_85)) {
 lean_ctor_release(x_85, 0);
 lean_ctor_release(x_85, 1);
 x_92 = x_85;
} else {
 lean_dec_ref(x_85);
 x_92 = lean_box(0);
}
x_93 = lean_box(0);
lean_inc(x_75);
x_94 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_94, 0, x_75);
lean_ctor_set(x_94, 1, x_93);
x_95 = lean_array_mk(x_94);
lean_inc(x_91);
x_96 = l_Lean_Expr_betaRev(x_91, x_95, x_82, x_82);
lean_dec_ref(x_95);
x_97 = lp_mathlib_ExistsAndEq_withNestedExistsIntro___redArg(x_96, x_73, x_3, x_4, x_5, x_6, x_7);
if (lean_obj_tag(x_97) == 0)
{
lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; 
x_98 = lean_ctor_get(x_97, 0);
lean_inc(x_98);
if (lean_is_exclusive(x_97)) {
 lean_ctor_release(x_97, 0);
 x_99 = x_97;
} else {
 lean_dec_ref(x_97);
 x_99 = lean_box(0);
}
x_100 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__16;
if (lean_is_scalar(x_92)) {
 x_101 = lean_alloc_ctor(1, 2, 0);
} else {
 x_101 = x_92;
 lean_ctor_set_tag(x_101, 1);
}
lean_ctor_set(x_101, 0, x_74);
lean_ctor_set(x_101, 1, x_93);
x_102 = l_Lean_Expr_const___override(x_100, x_101);
x_103 = l_Lean_Expr_app___override(x_102, x_90);
x_104 = l_Lean_Expr_app___override(x_103, x_91);
x_105 = l_Lean_Expr_app___override(x_104, x_75);
x_106 = l_Lean_Expr_app___override(x_105, x_98);
if (lean_is_scalar(x_99)) {
 x_107 = lean_alloc_ctor(0, 1, 0);
} else {
 x_107 = x_99;
}
lean_ctor_set(x_107, 0, x_106);
return x_107;
}
else
{
lean_dec(x_92);
lean_dec(x_91);
lean_dec(x_90);
lean_dec(x_75);
lean_dec(x_74);
return x_97;
}
}
}
else
{
lean_object* x_108; lean_object* x_109; lean_object* x_110; 
lean_dec(x_75);
lean_dec(x_74);
lean_dec(x_73);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
x_108 = lean_ctor_get(x_83, 0);
lean_inc(x_108);
if (lean_is_exclusive(x_83)) {
 lean_ctor_release(x_83, 0);
 x_109 = x_83;
} else {
 lean_dec_ref(x_83);
 x_109 = lean_box(0);
}
if (lean_is_scalar(x_109)) {
 x_110 = lean_alloc_ctor(1, 1, 0);
} else {
 x_110 = x_109;
}
lean_ctor_set(x_110, 0, x_108);
return x_110;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withNestedExistsIntro(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_ExistsAndEq_withNestedExistsIntro___redArg(x_1, x_3, x_4, x_5, x_6, x_7, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withNestedExistsIntro___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_ExistsAndEq_withNestedExistsIntro(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_2);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withNestedExistsIntro___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_ExistsAndEq_withNestedExistsIntro___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___lam__3(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, uint8_t x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
lean_inc_ref(x_7);
lean_inc(x_3);
lean_inc(x_1);
x_12 = l_Lean_Meta_mkFreshExprMVar(x_1, x_2, x_3, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_12) == 0)
{
lean_object* x_13; lean_object* x_14; 
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
lean_inc_ref(x_7);
x_14 = l_Lean_Meta_mkFreshExprMVar(x_1, x_2, x_3, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_14) == 0)
{
lean_object* x_15; lean_object* x_16; uint8_t x_17; 
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
lean_dec_ref(x_14);
x_16 = l_Lean_Meta_Context_config(x_7);
x_17 = !lean_is_exclusive(x_16);
if (x_17 == 0)
{
uint8_t x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; uint8_t x_25; uint8_t x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; uint8_t x_30; uint64_t x_31; uint8_t x_32; 
x_18 = lean_ctor_get_uint8(x_7, sizeof(void*)*7);
x_19 = lean_ctor_get(x_7, 1);
lean_inc(x_19);
x_20 = lean_ctor_get(x_7, 2);
lean_inc_ref(x_20);
x_21 = lean_ctor_get(x_7, 3);
lean_inc_ref(x_21);
x_22 = lean_ctor_get(x_7, 4);
lean_inc(x_22);
x_23 = lean_ctor_get(x_7, 5);
lean_inc(x_23);
x_24 = lean_ctor_get(x_7, 6);
lean_inc(x_24);
x_25 = lean_ctor_get_uint8(x_7, sizeof(void*)*7 + 1);
x_26 = lean_ctor_get_uint8(x_7, sizeof(void*)*7 + 2);
x_27 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0;
lean_inc(x_13);
x_28 = l_Lean_Expr_app___override(x_27, x_13);
lean_inc(x_15);
x_29 = l_Lean_Expr_app___override(x_28, x_15);
x_30 = 2;
lean_ctor_set_uint8(x_16, 9, x_30);
x_31 = l_Lean_Meta_Context_configKey(x_7);
x_32 = !lean_is_exclusive(x_7);
if (x_32 == 0)
{
lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; uint64_t x_40; uint64_t x_41; uint64_t x_42; uint64_t x_43; uint64_t x_44; lean_object* x_45; lean_object* x_46; 
x_33 = lean_ctor_get(x_7, 6);
lean_dec(x_33);
x_34 = lean_ctor_get(x_7, 5);
lean_dec(x_34);
x_35 = lean_ctor_get(x_7, 4);
lean_dec(x_35);
x_36 = lean_ctor_get(x_7, 3);
lean_dec(x_36);
x_37 = lean_ctor_get(x_7, 2);
lean_dec(x_37);
x_38 = lean_ctor_get(x_7, 1);
lean_dec(x_38);
x_39 = lean_ctor_get(x_7, 0);
lean_dec(x_39);
x_40 = 2;
x_41 = lean_uint64_shift_right(x_31, x_40);
x_42 = lean_uint64_shift_left(x_41, x_40);
x_43 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_44 = lean_uint64_lor(x_42, x_43);
x_45 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_45, 0, x_16);
lean_ctor_set_uint64(x_45, sizeof(void*)*1, x_44);
lean_ctor_set(x_7, 0, x_45);
lean_inc(x_8);
x_46 = l_Lean_Meta_isExprDefEq(x_29, x_4, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_46) == 0)
{
uint8_t x_47; 
x_47 = !lean_is_exclusive(x_46);
if (x_47 == 0)
{
lean_object* x_48; uint8_t x_49; 
x_48 = lean_ctor_get(x_46, 0);
x_49 = lean_unbox(x_48);
lean_dec(x_48);
if (x_49 == 0)
{
lean_object* x_50; lean_object* x_51; lean_object* x_52; 
lean_dec(x_8);
lean_dec(x_6);
x_50 = lean_box(x_5);
x_51 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_51, 0, x_15);
lean_ctor_set(x_51, 1, x_50);
x_52 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_52, 0, x_13);
lean_ctor_set(x_52, 1, x_51);
lean_ctor_set(x_46, 0, x_52);
return x_46;
}
else
{
lean_object* x_53; 
lean_free_object(x_46);
x_53 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_13, x_8);
if (lean_obj_tag(x_53) == 0)
{
lean_object* x_54; lean_object* x_55; 
x_54 = lean_ctor_get(x_53, 0);
lean_inc(x_54);
lean_dec_ref(x_53);
x_55 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_15, x_8);
lean_dec(x_8);
if (lean_obj_tag(x_55) == 0)
{
uint8_t x_56; 
x_56 = !lean_is_exclusive(x_55);
if (x_56 == 0)
{
lean_object* x_57; lean_object* x_58; lean_object* x_59; 
x_57 = lean_ctor_get(x_55, 0);
x_58 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_58, 0, x_57);
lean_ctor_set(x_58, 1, x_6);
x_59 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_59, 0, x_54);
lean_ctor_set(x_59, 1, x_58);
lean_ctor_set(x_55, 0, x_59);
return x_55;
}
else
{
lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; 
x_60 = lean_ctor_get(x_55, 0);
lean_inc(x_60);
lean_dec(x_55);
x_61 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_61, 0, x_60);
lean_ctor_set(x_61, 1, x_6);
x_62 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_62, 0, x_54);
lean_ctor_set(x_62, 1, x_61);
x_63 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_63, 0, x_62);
return x_63;
}
}
else
{
uint8_t x_64; 
lean_dec(x_54);
lean_dec(x_6);
x_64 = !lean_is_exclusive(x_55);
if (x_64 == 0)
{
return x_55;
}
else
{
lean_object* x_65; lean_object* x_66; 
x_65 = lean_ctor_get(x_55, 0);
lean_inc(x_65);
lean_dec(x_55);
x_66 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_66, 0, x_65);
return x_66;
}
}
}
else
{
uint8_t x_67; 
lean_dec(x_15);
lean_dec(x_8);
lean_dec(x_6);
x_67 = !lean_is_exclusive(x_53);
if (x_67 == 0)
{
return x_53;
}
else
{
lean_object* x_68; lean_object* x_69; 
x_68 = lean_ctor_get(x_53, 0);
lean_inc(x_68);
lean_dec(x_53);
x_69 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_69, 0, x_68);
return x_69;
}
}
}
}
else
{
lean_object* x_70; uint8_t x_71; 
x_70 = lean_ctor_get(x_46, 0);
lean_inc(x_70);
lean_dec(x_46);
x_71 = lean_unbox(x_70);
lean_dec(x_70);
if (x_71 == 0)
{
lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; 
lean_dec(x_8);
lean_dec(x_6);
x_72 = lean_box(x_5);
x_73 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_73, 0, x_15);
lean_ctor_set(x_73, 1, x_72);
x_74 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_74, 0, x_13);
lean_ctor_set(x_74, 1, x_73);
x_75 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_75, 0, x_74);
return x_75;
}
else
{
lean_object* x_76; 
x_76 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_13, x_8);
if (lean_obj_tag(x_76) == 0)
{
lean_object* x_77; lean_object* x_78; 
x_77 = lean_ctor_get(x_76, 0);
lean_inc(x_77);
lean_dec_ref(x_76);
x_78 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_15, x_8);
lean_dec(x_8);
if (lean_obj_tag(x_78) == 0)
{
lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; 
x_79 = lean_ctor_get(x_78, 0);
lean_inc(x_79);
if (lean_is_exclusive(x_78)) {
 lean_ctor_release(x_78, 0);
 x_80 = x_78;
} else {
 lean_dec_ref(x_78);
 x_80 = lean_box(0);
}
x_81 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_81, 0, x_79);
lean_ctor_set(x_81, 1, x_6);
x_82 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_82, 0, x_77);
lean_ctor_set(x_82, 1, x_81);
if (lean_is_scalar(x_80)) {
 x_83 = lean_alloc_ctor(0, 1, 0);
} else {
 x_83 = x_80;
}
lean_ctor_set(x_83, 0, x_82);
return x_83;
}
else
{
lean_object* x_84; lean_object* x_85; lean_object* x_86; 
lean_dec(x_77);
lean_dec(x_6);
x_84 = lean_ctor_get(x_78, 0);
lean_inc(x_84);
if (lean_is_exclusive(x_78)) {
 lean_ctor_release(x_78, 0);
 x_85 = x_78;
} else {
 lean_dec_ref(x_78);
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
else
{
lean_object* x_87; lean_object* x_88; lean_object* x_89; 
lean_dec(x_15);
lean_dec(x_8);
lean_dec(x_6);
x_87 = lean_ctor_get(x_76, 0);
lean_inc(x_87);
if (lean_is_exclusive(x_76)) {
 lean_ctor_release(x_76, 0);
 x_88 = x_76;
} else {
 lean_dec_ref(x_76);
 x_88 = lean_box(0);
}
if (lean_is_scalar(x_88)) {
 x_89 = lean_alloc_ctor(1, 1, 0);
} else {
 x_89 = x_88;
}
lean_ctor_set(x_89, 0, x_87);
return x_89;
}
}
}
}
else
{
uint8_t x_90; 
lean_dec(x_15);
lean_dec(x_13);
lean_dec(x_8);
lean_dec(x_6);
x_90 = !lean_is_exclusive(x_46);
if (x_90 == 0)
{
return x_46;
}
else
{
lean_object* x_91; lean_object* x_92; 
x_91 = lean_ctor_get(x_46, 0);
lean_inc(x_91);
lean_dec(x_46);
x_92 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_92, 0, x_91);
return x_92;
}
}
}
else
{
uint64_t x_93; uint64_t x_94; uint64_t x_95; uint64_t x_96; uint64_t x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; 
lean_dec(x_7);
x_93 = 2;
x_94 = lean_uint64_shift_right(x_31, x_93);
x_95 = lean_uint64_shift_left(x_94, x_93);
x_96 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_97 = lean_uint64_lor(x_95, x_96);
x_98 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_98, 0, x_16);
lean_ctor_set_uint64(x_98, sizeof(void*)*1, x_97);
x_99 = lean_alloc_ctor(0, 7, 3);
lean_ctor_set(x_99, 0, x_98);
lean_ctor_set(x_99, 1, x_19);
lean_ctor_set(x_99, 2, x_20);
lean_ctor_set(x_99, 3, x_21);
lean_ctor_set(x_99, 4, x_22);
lean_ctor_set(x_99, 5, x_23);
lean_ctor_set(x_99, 6, x_24);
lean_ctor_set_uint8(x_99, sizeof(void*)*7, x_18);
lean_ctor_set_uint8(x_99, sizeof(void*)*7 + 1, x_25);
lean_ctor_set_uint8(x_99, sizeof(void*)*7 + 2, x_26);
lean_inc(x_8);
x_100 = l_Lean_Meta_isExprDefEq(x_29, x_4, x_99, x_8, x_9, x_10);
if (lean_obj_tag(x_100) == 0)
{
lean_object* x_101; lean_object* x_102; uint8_t x_103; 
x_101 = lean_ctor_get(x_100, 0);
lean_inc(x_101);
if (lean_is_exclusive(x_100)) {
 lean_ctor_release(x_100, 0);
 x_102 = x_100;
} else {
 lean_dec_ref(x_100);
 x_102 = lean_box(0);
}
x_103 = lean_unbox(x_101);
lean_dec(x_101);
if (x_103 == 0)
{
lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; 
lean_dec(x_8);
lean_dec(x_6);
x_104 = lean_box(x_5);
x_105 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_105, 0, x_15);
lean_ctor_set(x_105, 1, x_104);
x_106 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_106, 0, x_13);
lean_ctor_set(x_106, 1, x_105);
if (lean_is_scalar(x_102)) {
 x_107 = lean_alloc_ctor(0, 1, 0);
} else {
 x_107 = x_102;
}
lean_ctor_set(x_107, 0, x_106);
return x_107;
}
else
{
lean_object* x_108; 
lean_dec(x_102);
x_108 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_13, x_8);
if (lean_obj_tag(x_108) == 0)
{
lean_object* x_109; lean_object* x_110; 
x_109 = lean_ctor_get(x_108, 0);
lean_inc(x_109);
lean_dec_ref(x_108);
x_110 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_15, x_8);
lean_dec(x_8);
if (lean_obj_tag(x_110) == 0)
{
lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; 
x_111 = lean_ctor_get(x_110, 0);
lean_inc(x_111);
if (lean_is_exclusive(x_110)) {
 lean_ctor_release(x_110, 0);
 x_112 = x_110;
} else {
 lean_dec_ref(x_110);
 x_112 = lean_box(0);
}
x_113 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_113, 0, x_111);
lean_ctor_set(x_113, 1, x_6);
x_114 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_114, 0, x_109);
lean_ctor_set(x_114, 1, x_113);
if (lean_is_scalar(x_112)) {
 x_115 = lean_alloc_ctor(0, 1, 0);
} else {
 x_115 = x_112;
}
lean_ctor_set(x_115, 0, x_114);
return x_115;
}
else
{
lean_object* x_116; lean_object* x_117; lean_object* x_118; 
lean_dec(x_109);
lean_dec(x_6);
x_116 = lean_ctor_get(x_110, 0);
lean_inc(x_116);
if (lean_is_exclusive(x_110)) {
 lean_ctor_release(x_110, 0);
 x_117 = x_110;
} else {
 lean_dec_ref(x_110);
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
lean_object* x_119; lean_object* x_120; lean_object* x_121; 
lean_dec(x_15);
lean_dec(x_8);
lean_dec(x_6);
x_119 = lean_ctor_get(x_108, 0);
lean_inc(x_119);
if (lean_is_exclusive(x_108)) {
 lean_ctor_release(x_108, 0);
 x_120 = x_108;
} else {
 lean_dec_ref(x_108);
 x_120 = lean_box(0);
}
if (lean_is_scalar(x_120)) {
 x_121 = lean_alloc_ctor(1, 1, 0);
} else {
 x_121 = x_120;
}
lean_ctor_set(x_121, 0, x_119);
return x_121;
}
}
}
else
{
lean_object* x_122; lean_object* x_123; lean_object* x_124; 
lean_dec(x_15);
lean_dec(x_13);
lean_dec(x_8);
lean_dec(x_6);
x_122 = lean_ctor_get(x_100, 0);
lean_inc(x_122);
if (lean_is_exclusive(x_100)) {
 lean_ctor_release(x_100, 0);
 x_123 = x_100;
} else {
 lean_dec_ref(x_100);
 x_123 = lean_box(0);
}
if (lean_is_scalar(x_123)) {
 x_124 = lean_alloc_ctor(1, 1, 0);
} else {
 x_124 = x_123;
}
lean_ctor_set(x_124, 0, x_122);
return x_124;
}
}
}
else
{
uint8_t x_125; uint8_t x_126; uint8_t x_127; uint8_t x_128; uint8_t x_129; uint8_t x_130; uint8_t x_131; uint8_t x_132; uint8_t x_133; uint8_t x_134; uint8_t x_135; uint8_t x_136; uint8_t x_137; uint8_t x_138; uint8_t x_139; uint8_t x_140; uint8_t x_141; uint8_t x_142; uint8_t x_143; lean_object* x_144; lean_object* x_145; lean_object* x_146; lean_object* x_147; lean_object* x_148; lean_object* x_149; uint8_t x_150; uint8_t x_151; lean_object* x_152; lean_object* x_153; lean_object* x_154; uint8_t x_155; lean_object* x_156; uint64_t x_157; lean_object* x_158; uint64_t x_159; uint64_t x_160; uint64_t x_161; uint64_t x_162; uint64_t x_163; lean_object* x_164; lean_object* x_165; lean_object* x_166; 
x_125 = lean_ctor_get_uint8(x_16, 0);
x_126 = lean_ctor_get_uint8(x_16, 1);
x_127 = lean_ctor_get_uint8(x_16, 2);
x_128 = lean_ctor_get_uint8(x_16, 3);
x_129 = lean_ctor_get_uint8(x_16, 4);
x_130 = lean_ctor_get_uint8(x_16, 5);
x_131 = lean_ctor_get_uint8(x_16, 6);
x_132 = lean_ctor_get_uint8(x_16, 7);
x_133 = lean_ctor_get_uint8(x_16, 8);
x_134 = lean_ctor_get_uint8(x_16, 10);
x_135 = lean_ctor_get_uint8(x_16, 11);
x_136 = lean_ctor_get_uint8(x_16, 12);
x_137 = lean_ctor_get_uint8(x_16, 13);
x_138 = lean_ctor_get_uint8(x_16, 14);
x_139 = lean_ctor_get_uint8(x_16, 15);
x_140 = lean_ctor_get_uint8(x_16, 16);
x_141 = lean_ctor_get_uint8(x_16, 17);
x_142 = lean_ctor_get_uint8(x_16, 18);
lean_dec(x_16);
x_143 = lean_ctor_get_uint8(x_7, sizeof(void*)*7);
x_144 = lean_ctor_get(x_7, 1);
lean_inc(x_144);
x_145 = lean_ctor_get(x_7, 2);
lean_inc_ref(x_145);
x_146 = lean_ctor_get(x_7, 3);
lean_inc_ref(x_146);
x_147 = lean_ctor_get(x_7, 4);
lean_inc(x_147);
x_148 = lean_ctor_get(x_7, 5);
lean_inc(x_148);
x_149 = lean_ctor_get(x_7, 6);
lean_inc(x_149);
x_150 = lean_ctor_get_uint8(x_7, sizeof(void*)*7 + 1);
x_151 = lean_ctor_get_uint8(x_7, sizeof(void*)*7 + 2);
x_152 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0;
lean_inc(x_13);
x_153 = l_Lean_Expr_app___override(x_152, x_13);
lean_inc(x_15);
x_154 = l_Lean_Expr_app___override(x_153, x_15);
x_155 = 2;
x_156 = lean_alloc_ctor(0, 0, 19);
lean_ctor_set_uint8(x_156, 0, x_125);
lean_ctor_set_uint8(x_156, 1, x_126);
lean_ctor_set_uint8(x_156, 2, x_127);
lean_ctor_set_uint8(x_156, 3, x_128);
lean_ctor_set_uint8(x_156, 4, x_129);
lean_ctor_set_uint8(x_156, 5, x_130);
lean_ctor_set_uint8(x_156, 6, x_131);
lean_ctor_set_uint8(x_156, 7, x_132);
lean_ctor_set_uint8(x_156, 8, x_133);
lean_ctor_set_uint8(x_156, 9, x_155);
lean_ctor_set_uint8(x_156, 10, x_134);
lean_ctor_set_uint8(x_156, 11, x_135);
lean_ctor_set_uint8(x_156, 12, x_136);
lean_ctor_set_uint8(x_156, 13, x_137);
lean_ctor_set_uint8(x_156, 14, x_138);
lean_ctor_set_uint8(x_156, 15, x_139);
lean_ctor_set_uint8(x_156, 16, x_140);
lean_ctor_set_uint8(x_156, 17, x_141);
lean_ctor_set_uint8(x_156, 18, x_142);
x_157 = l_Lean_Meta_Context_configKey(x_7);
if (lean_is_exclusive(x_7)) {
 lean_ctor_release(x_7, 0);
 lean_ctor_release(x_7, 1);
 lean_ctor_release(x_7, 2);
 lean_ctor_release(x_7, 3);
 lean_ctor_release(x_7, 4);
 lean_ctor_release(x_7, 5);
 lean_ctor_release(x_7, 6);
 x_158 = x_7;
} else {
 lean_dec_ref(x_7);
 x_158 = lean_box(0);
}
x_159 = 2;
x_160 = lean_uint64_shift_right(x_157, x_159);
x_161 = lean_uint64_shift_left(x_160, x_159);
x_162 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1;
x_163 = lean_uint64_lor(x_161, x_162);
x_164 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_164, 0, x_156);
lean_ctor_set_uint64(x_164, sizeof(void*)*1, x_163);
if (lean_is_scalar(x_158)) {
 x_165 = lean_alloc_ctor(0, 7, 3);
} else {
 x_165 = x_158;
}
lean_ctor_set(x_165, 0, x_164);
lean_ctor_set(x_165, 1, x_144);
lean_ctor_set(x_165, 2, x_145);
lean_ctor_set(x_165, 3, x_146);
lean_ctor_set(x_165, 4, x_147);
lean_ctor_set(x_165, 5, x_148);
lean_ctor_set(x_165, 6, x_149);
lean_ctor_set_uint8(x_165, sizeof(void*)*7, x_143);
lean_ctor_set_uint8(x_165, sizeof(void*)*7 + 1, x_150);
lean_ctor_set_uint8(x_165, sizeof(void*)*7 + 2, x_151);
lean_inc(x_8);
x_166 = l_Lean_Meta_isExprDefEq(x_154, x_4, x_165, x_8, x_9, x_10);
if (lean_obj_tag(x_166) == 0)
{
lean_object* x_167; lean_object* x_168; uint8_t x_169; 
x_167 = lean_ctor_get(x_166, 0);
lean_inc(x_167);
if (lean_is_exclusive(x_166)) {
 lean_ctor_release(x_166, 0);
 x_168 = x_166;
} else {
 lean_dec_ref(x_166);
 x_168 = lean_box(0);
}
x_169 = lean_unbox(x_167);
lean_dec(x_167);
if (x_169 == 0)
{
lean_object* x_170; lean_object* x_171; lean_object* x_172; lean_object* x_173; 
lean_dec(x_8);
lean_dec(x_6);
x_170 = lean_box(x_5);
x_171 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_171, 0, x_15);
lean_ctor_set(x_171, 1, x_170);
x_172 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_172, 0, x_13);
lean_ctor_set(x_172, 1, x_171);
if (lean_is_scalar(x_168)) {
 x_173 = lean_alloc_ctor(0, 1, 0);
} else {
 x_173 = x_168;
}
lean_ctor_set(x_173, 0, x_172);
return x_173;
}
else
{
lean_object* x_174; 
lean_dec(x_168);
x_174 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_13, x_8);
if (lean_obj_tag(x_174) == 0)
{
lean_object* x_175; lean_object* x_176; 
x_175 = lean_ctor_get(x_174, 0);
lean_inc(x_175);
lean_dec_ref(x_174);
x_176 = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(x_15, x_8);
lean_dec(x_8);
if (lean_obj_tag(x_176) == 0)
{
lean_object* x_177; lean_object* x_178; lean_object* x_179; lean_object* x_180; lean_object* x_181; 
x_177 = lean_ctor_get(x_176, 0);
lean_inc(x_177);
if (lean_is_exclusive(x_176)) {
 lean_ctor_release(x_176, 0);
 x_178 = x_176;
} else {
 lean_dec_ref(x_176);
 x_178 = lean_box(0);
}
x_179 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_179, 0, x_177);
lean_ctor_set(x_179, 1, x_6);
x_180 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_180, 0, x_175);
lean_ctor_set(x_180, 1, x_179);
if (lean_is_scalar(x_178)) {
 x_181 = lean_alloc_ctor(0, 1, 0);
} else {
 x_181 = x_178;
}
lean_ctor_set(x_181, 0, x_180);
return x_181;
}
else
{
lean_object* x_182; lean_object* x_183; lean_object* x_184; 
lean_dec(x_175);
lean_dec(x_6);
x_182 = lean_ctor_get(x_176, 0);
lean_inc(x_182);
if (lean_is_exclusive(x_176)) {
 lean_ctor_release(x_176, 0);
 x_183 = x_176;
} else {
 lean_dec_ref(x_176);
 x_183 = lean_box(0);
}
if (lean_is_scalar(x_183)) {
 x_184 = lean_alloc_ctor(1, 1, 0);
} else {
 x_184 = x_183;
}
lean_ctor_set(x_184, 0, x_182);
return x_184;
}
}
else
{
lean_object* x_185; lean_object* x_186; lean_object* x_187; 
lean_dec(x_15);
lean_dec(x_8);
lean_dec(x_6);
x_185 = lean_ctor_get(x_174, 0);
lean_inc(x_185);
if (lean_is_exclusive(x_174)) {
 lean_ctor_release(x_174, 0);
 x_186 = x_174;
} else {
 lean_dec_ref(x_174);
 x_186 = lean_box(0);
}
if (lean_is_scalar(x_186)) {
 x_187 = lean_alloc_ctor(1, 1, 0);
} else {
 x_187 = x_186;
}
lean_ctor_set(x_187, 0, x_185);
return x_187;
}
}
}
else
{
lean_object* x_188; lean_object* x_189; lean_object* x_190; 
lean_dec(x_15);
lean_dec(x_13);
lean_dec(x_8);
lean_dec(x_6);
x_188 = lean_ctor_get(x_166, 0);
lean_inc(x_188);
if (lean_is_exclusive(x_166)) {
 lean_ctor_release(x_166, 0);
 x_189 = x_166;
} else {
 lean_dec_ref(x_166);
 x_189 = lean_box(0);
}
if (lean_is_scalar(x_189)) {
 x_190 = lean_alloc_ctor(1, 1, 0);
} else {
 x_190 = x_189;
}
lean_ctor_set(x_190, 0, x_188);
return x_190;
}
}
}
else
{
uint8_t x_191; 
lean_dec(x_13);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_4);
x_191 = !lean_is_exclusive(x_14);
if (x_191 == 0)
{
return x_14;
}
else
{
lean_object* x_192; lean_object* x_193; 
x_192 = lean_ctor_get(x_14, 0);
lean_inc(x_192);
lean_dec(x_14);
x_193 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_193, 0, x_192);
return x_193;
}
}
}
else
{
uint8_t x_194; 
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec(x_1);
x_194 = !lean_is_exclusive(x_12);
if (x_194 == 0)
{
return x_12;
}
else
{
lean_object* x_195; lean_object* x_196; 
x_195 = lean_ctor_get(x_12, 0);
lean_inc(x_195);
lean_dec(x_12);
x_196 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_196, 0, x_195);
return x_196;
}
}
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("mkBeforeToAfter: unexpected goal = ", 35, 35);
return x_1;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("mkBeforeToAfter: goal is equality but path is not empty", 55, 55);
return x_1;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("mkBeforeToAfter: `P` is `And` but `goal` is not `And`", 53, 53);
return x_1;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("mkBeforeToAfter: `P` is `And` but `path` is empty", 49, 49);
return x_1;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("mp", 2, 2);
return x_1;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__4;
x_2 = lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__3;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__6() {
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
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__6;
x_2 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__5;
x_3 = l_Lean_Expr_const___override(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__8() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("congrArg", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__8;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__10() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_box(0);
x_2 = l_Lean_Level_succ___override(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__11() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__10;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__12() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("mkBeforeToAfter: `P` is `Exists` but `exs` is empty", 51, 51);
return x_1;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__13() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("mkBeforeToAfter: `P` is `Exists` but `hs` is empty", 50, 50);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___lam__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
uint8_t x_12; uint8_t x_13; lean_object* x_14; 
x_12 = lean_unbox(x_2);
x_13 = lean_unbox(x_5);
x_14 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___lam__3(x_1, x_12, x_3, x_4, x_13, x_6, x_7, x_8, x_9, x_10);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
uint8_t x_17; lean_object* x_18; uint8_t x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_17 = 0;
x_18 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0;
x_19 = 0;
x_20 = lean_box(0);
x_21 = lean_box(x_19);
x_22 = lean_box(x_17);
lean_inc_ref(x_2);
x_23 = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___lam__0___boxed), 10, 5);
lean_closure_set(x_23, 0, x_21);
lean_closure_set(x_23, 1, x_20);
lean_closure_set(x_23, 2, x_18);
lean_closure_set(x_23, 3, x_2);
lean_closure_set(x_23, 4, x_22);
lean_inc(x_15);
lean_inc_ref(x_14);
lean_inc(x_13);
lean_inc_ref(x_12);
x_24 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(x_23, x_17, x_12, x_13, x_14, x_15);
if (lean_obj_tag(x_24) == 0)
{
lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; uint8_t x_29; 
x_25 = lean_ctor_get(x_24, 0);
lean_inc(x_25);
lean_dec_ref(x_24);
x_26 = lean_ctor_get(x_25, 1);
lean_inc(x_26);
lean_dec(x_25);
x_27 = lean_ctor_get(x_26, 1);
lean_inc(x_27);
lean_dec(x_26);
x_28 = lean_ctor_get(x_27, 1);
lean_inc(x_28);
lean_dec(x_27);
x_29 = lean_unbox(x_28);
if (x_29 == 0)
{
lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; 
x_30 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__1;
x_31 = lean_box(x_19);
x_32 = lean_box(x_17);
x_33 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___boxed), 10, 5);
lean_closure_set(x_33, 0, x_30);
lean_closure_set(x_33, 1, x_31);
lean_closure_set(x_33, 2, x_20);
lean_closure_set(x_33, 3, x_2);
lean_closure_set(x_33, 4, x_32);
lean_inc(x_15);
lean_inc_ref(x_14);
lean_inc(x_13);
lean_inc_ref(x_12);
x_34 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(x_33, x_17, x_12, x_13, x_14, x_15);
if (lean_obj_tag(x_34) == 0)
{
lean_object* x_35; lean_object* x_36; lean_object* x_37; uint8_t x_38; 
x_35 = lean_ctor_get(x_34, 0);
lean_inc(x_35);
lean_dec_ref(x_34);
x_36 = lean_ctor_get(x_35, 1);
lean_inc(x_36);
x_37 = lean_ctor_get(x_36, 1);
lean_inc(x_37);
x_38 = lean_unbox(x_37);
if (x_38 == 0)
{
lean_object* x_39; lean_object* x_40; 
lean_dec(x_37);
lean_dec(x_36);
lean_dec(x_35);
lean_dec(x_28);
lean_dec_ref(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_5);
lean_dec_ref(x_3);
lean_inc_ref(x_1);
x_39 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___lam__2___boxed), 6, 1);
lean_closure_set(x_39, 0, x_1);
lean_inc(x_15);
lean_inc_ref(x_14);
lean_inc(x_13);
lean_inc_ref(x_12);
x_40 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(x_39, x_17, x_12, x_13, x_14, x_15);
if (lean_obj_tag(x_40) == 0)
{
lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_59; uint8_t x_60; 
x_41 = lean_ctor_get(x_40, 0);
lean_inc(x_41);
if (lean_is_exclusive(x_40)) {
 lean_ctor_release(x_40, 0);
 x_42 = x_40;
} else {
 lean_dec_ref(x_40);
 x_42 = lean_box(0);
}
x_43 = lean_ctor_get(x_41, 1);
lean_inc(x_43);
x_44 = lean_ctor_get(x_43, 1);
lean_inc(x_44);
x_45 = lean_ctor_get(x_41, 0);
lean_inc(x_45);
lean_dec(x_41);
x_46 = lean_ctor_get(x_43, 0);
lean_inc(x_46);
lean_dec(x_43);
x_47 = lean_ctor_get(x_44, 0);
lean_inc(x_47);
x_48 = lean_ctor_get(x_44, 1);
lean_inc(x_48);
if (lean_is_exclusive(x_44)) {
 lean_ctor_release(x_44, 0);
 lean_ctor_release(x_44, 1);
 x_49 = x_44;
} else {
 lean_dec_ref(x_44);
 x_49 = lean_box(0);
}
x_59 = lean_ctor_get(x_48, 1);
lean_inc(x_59);
lean_dec(x_48);
x_60 = lean_unbox(x_59);
lean_dec(x_59);
if (x_60 == 0)
{
lean_object* x_61; 
lean_dec(x_49);
lean_dec(x_47);
lean_dec(x_46);
lean_dec(x_45);
lean_dec(x_42);
lean_dec(x_6);
x_61 = l_Lean_Meta_ppExpr(x_1, x_12, x_13, x_14, x_15);
if (lean_obj_tag(x_61) == 0)
{
lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; 
x_62 = lean_ctor_get(x_61, 0);
lean_inc(x_62);
lean_dec_ref(x_61);
x_63 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__0;
x_64 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3;
x_65 = lean_unsigned_to_nat(0u);
x_66 = l_Std_Format_pretty(x_62, x_64, x_65, x_65);
x_67 = lean_string_append(x_63, x_66);
lean_dec_ref(x_66);
x_68 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg(x_67, x_12, x_13, x_14, x_15);
lean_dec(x_15);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec_ref(x_67);
return x_68;
}
else
{
uint8_t x_69; 
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
x_69 = !lean_is_exclusive(x_61);
if (x_69 == 0)
{
return x_61;
}
else
{
lean_object* x_70; lean_object* x_71; 
x_70 = lean_ctor_get(x_61, 0);
lean_inc(x_70);
lean_dec(x_61);
x_71 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_71, 0, x_70);
return x_71;
}
}
}
else
{
uint8_t x_72; 
lean_dec_ref(x_1);
x_72 = l_List_isEmpty___redArg(x_6);
lean_dec(x_6);
if (x_72 == 0)
{
lean_object* x_73; lean_object* x_74; 
x_73 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__1;
x_74 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg(x_73, x_12, x_13, x_14, x_15);
lean_dec(x_15);
lean_dec(x_13);
lean_dec_ref(x_12);
if (lean_obj_tag(x_74) == 0)
{
lean_dec_ref(x_74);
x_50 = lean_box(0);
goto block_58;
}
else
{
uint8_t x_75; 
lean_dec(x_49);
lean_dec(x_47);
lean_dec(x_46);
lean_dec(x_45);
lean_dec(x_42);
x_75 = !lean_is_exclusive(x_74);
if (x_75 == 0)
{
return x_74;
}
else
{
lean_object* x_76; lean_object* x_77; 
x_76 = lean_ctor_get(x_74, 0);
lean_inc(x_76);
lean_dec(x_74);
x_77 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_77, 0, x_76);
return x_77;
}
}
}
else
{
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
x_50 = lean_box(0);
goto block_58;
}
}
block_58:
{
lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; 
x_51 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__1;
x_52 = lean_box(0);
if (lean_is_scalar(x_49)) {
 x_53 = lean_alloc_ctor(1, 2, 0);
} else {
 x_53 = x_49;
 lean_ctor_set_tag(x_53, 1);
}
lean_ctor_set(x_53, 0, x_45);
lean_ctor_set(x_53, 1, x_52);
x_54 = l_Lean_Expr_const___override(x_51, x_53);
x_55 = l_Lean_Expr_app___override(x_54, x_46);
x_56 = l_Lean_Expr_app___override(x_55, x_47);
if (lean_is_scalar(x_42)) {
 x_57 = lean_alloc_ctor(0, 1, 0);
} else {
 x_57 = x_42;
}
lean_ctor_set(x_57, 0, x_56);
return x_57;
}
}
else
{
uint8_t x_78; 
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_6);
lean_dec_ref(x_1);
x_78 = !lean_is_exclusive(x_40);
if (x_78 == 0)
{
return x_40;
}
else
{
lean_object* x_79; lean_object* x_80; 
x_79 = lean_ctor_get(x_40, 0);
lean_inc(x_79);
lean_dec(x_40);
x_80 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_80, 0, x_79);
return x_80;
}
}
}
else
{
lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; 
x_81 = lean_ctor_get(x_35, 0);
lean_inc(x_81);
lean_dec(x_35);
x_82 = lean_ctor_get(x_36, 0);
lean_inc(x_82);
lean_dec(x_36);
x_83 = lean_box(x_19);
x_84 = lean_box(x_17);
lean_inc(x_37);
x_85 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___lam__3___boxed), 11, 6);
lean_closure_set(x_85, 0, x_30);
lean_closure_set(x_85, 1, x_83);
lean_closure_set(x_85, 2, x_20);
lean_closure_set(x_85, 3, x_1);
lean_closure_set(x_85, 4, x_84);
lean_closure_set(x_85, 5, x_37);
lean_inc(x_15);
lean_inc_ref(x_14);
lean_inc(x_13);
lean_inc_ref(x_12);
x_86 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(x_85, x_17, x_12, x_13, x_14, x_15);
if (lean_obj_tag(x_86) == 0)
{
lean_object* x_87; lean_object* x_88; lean_object* x_89; uint8_t x_90; 
x_87 = lean_ctor_get(x_86, 0);
lean_inc(x_87);
lean_dec_ref(x_86);
x_88 = lean_ctor_get(x_87, 1);
lean_inc(x_88);
x_89 = lean_ctor_get(x_88, 1);
x_90 = lean_unbox(x_89);
if (x_90 == 0)
{
lean_object* x_91; lean_object* x_92; 
lean_dec(x_88);
lean_dec(x_87);
lean_dec(x_82);
lean_dec(x_81);
lean_dec(x_37);
lean_dec(x_28);
lean_dec_ref(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_3);
x_91 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__2;
x_92 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg(x_91, x_12, x_13, x_14, x_15);
lean_dec(x_15);
lean_dec(x_13);
lean_dec_ref(x_12);
return x_92;
}
else
{
if (lean_obj_tag(x_6) == 0)
{
lean_object* x_93; lean_object* x_94; 
lean_dec(x_88);
lean_dec(x_87);
lean_dec(x_82);
lean_dec(x_81);
lean_dec(x_37);
lean_dec(x_28);
lean_dec_ref(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_5);
lean_dec_ref(x_3);
x_93 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__3;
x_94 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg(x_93, x_12, x_13, x_14, x_15);
lean_dec(x_15);
lean_dec(x_13);
lean_dec_ref(x_12);
return x_94;
}
else
{
lean_object* x_95; uint8_t x_96; 
x_95 = lean_ctor_get(x_6, 0);
x_96 = lean_unbox(x_95);
if (x_96 == 0)
{
lean_object* x_97; uint8_t x_98; 
x_97 = lean_ctor_get(x_87, 0);
lean_inc(x_97);
lean_dec(x_87);
x_98 = !lean_is_exclusive(x_88);
if (x_98 == 0)
{
lean_object* x_99; lean_object* x_100; uint8_t x_101; 
x_99 = lean_ctor_get(x_88, 0);
x_100 = lean_ctor_get(x_88, 1);
lean_dec(x_100);
x_101 = !lean_is_exclusive(x_6);
if (x_101 == 0)
{
lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; uint8_t x_106; uint8_t x_107; uint8_t x_108; lean_object* x_109; 
x_102 = lean_ctor_get(x_6, 1);
x_103 = lean_ctor_get(x_6, 0);
lean_dec(x_103);
x_104 = lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__5;
lean_inc_ref(x_9);
x_105 = lean_array_push(x_104, x_9);
x_106 = 1;
x_107 = lean_unbox(x_37);
x_108 = lean_unbox(x_37);
lean_dec(x_37);
lean_inc(x_82);
x_109 = l_Lean_Meta_mkLambdaFVars(x_105, x_82, x_17, x_107, x_17, x_108, x_106, x_12, x_13, x_14, x_15);
lean_dec_ref(x_105);
if (lean_obj_tag(x_109) == 0)
{
lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; lean_object* x_116; 
x_110 = lean_ctor_get(x_109, 0);
lean_inc(x_110);
lean_dec_ref(x_109);
x_111 = lean_box(0);
x_112 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__8;
lean_inc(x_81);
x_113 = l_Lean_Expr_app___override(x_112, x_81);
lean_inc(x_82);
x_114 = l_Lean_Expr_app___override(x_113, x_82);
lean_inc_ref(x_3);
x_115 = l_Lean_Expr_app___override(x_114, x_3);
lean_inc_ref(x_11);
lean_inc_ref(x_10);
lean_inc_ref(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc(x_81);
lean_inc(x_97);
x_116 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go(x_97, x_81, x_115, x_4, x_5, x_102, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
if (lean_obj_tag(x_116) == 0)
{
uint8_t x_117; 
x_117 = !lean_is_exclusive(x_116);
if (x_117 == 0)
{
lean_object* x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; uint8_t x_125; uint8_t x_126; lean_object* x_127; lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; lean_object* x_132; lean_object* x_133; lean_object* x_134; lean_object* x_135; lean_object* x_136; lean_object* x_137; lean_object* x_138; lean_object* x_139; lean_object* x_140; lean_object* x_141; lean_object* x_142; lean_object* x_143; lean_object* x_144; lean_object* x_145; 
x_118 = lean_ctor_get(x_116, 0);
lean_inc_ref(x_9);
lean_ctor_set(x_6, 1, x_111);
lean_ctor_set(x_6, 0, x_9);
x_119 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__11;
x_120 = l_Lean_Expr_app___override(x_119, x_81);
x_121 = l_Lean_Expr_app___override(x_120, x_82);
x_122 = l_Lean_Expr_app___override(x_121, x_3);
x_123 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__7;
x_124 = lean_array_mk(x_6);
x_125 = lean_unbox(x_28);
x_126 = lean_unbox(x_28);
lean_dec(x_28);
lean_inc(x_110);
x_127 = l_Lean_Expr_betaRev(x_110, x_124, x_125, x_126);
lean_dec_ref(x_124);
x_128 = l_Lean_Expr_app___override(x_123, x_127);
lean_inc(x_99);
x_129 = l_Lean_Expr_app___override(x_128, x_99);
x_130 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__9;
x_131 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__11;
lean_ctor_set_tag(x_88, 1);
lean_ctor_set(x_88, 1, x_131);
lean_ctor_set(x_88, 0, x_7);
x_132 = l_Lean_Expr_const___override(x_130, x_88);
x_133 = l_Lean_Expr_app___override(x_132, x_8);
x_134 = l_Lean_Expr_app___override(x_133, x_18);
x_135 = l_Lean_Expr_app___override(x_134, x_9);
x_136 = l_Lean_Expr_app___override(x_135, x_10);
x_137 = l_Lean_Expr_app___override(x_136, x_110);
x_138 = l_Lean_Expr_app___override(x_137, x_11);
x_139 = l_Lean_Expr_app___override(x_129, x_138);
x_140 = l_Lean_Expr_app___override(x_139, x_122);
x_141 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__14;
x_142 = l_Lean_Expr_app___override(x_141, x_97);
x_143 = l_Lean_Expr_app___override(x_142, x_99);
x_144 = l_Lean_Expr_app___override(x_143, x_118);
x_145 = l_Lean_Expr_app___override(x_144, x_140);
lean_ctor_set(x_116, 0, x_145);
return x_116;
}
else
{
lean_object* x_146; lean_object* x_147; lean_object* x_148; lean_object* x_149; lean_object* x_150; lean_object* x_151; lean_object* x_152; uint8_t x_153; uint8_t x_154; lean_object* x_155; lean_object* x_156; lean_object* x_157; lean_object* x_158; lean_object* x_159; lean_object* x_160; lean_object* x_161; lean_object* x_162; lean_object* x_163; lean_object* x_164; lean_object* x_165; lean_object* x_166; lean_object* x_167; lean_object* x_168; lean_object* x_169; lean_object* x_170; lean_object* x_171; lean_object* x_172; lean_object* x_173; lean_object* x_174; 
x_146 = lean_ctor_get(x_116, 0);
lean_inc(x_146);
lean_dec(x_116);
lean_inc_ref(x_9);
lean_ctor_set(x_6, 1, x_111);
lean_ctor_set(x_6, 0, x_9);
x_147 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__11;
x_148 = l_Lean_Expr_app___override(x_147, x_81);
x_149 = l_Lean_Expr_app___override(x_148, x_82);
x_150 = l_Lean_Expr_app___override(x_149, x_3);
x_151 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__7;
x_152 = lean_array_mk(x_6);
x_153 = lean_unbox(x_28);
x_154 = lean_unbox(x_28);
lean_dec(x_28);
lean_inc(x_110);
x_155 = l_Lean_Expr_betaRev(x_110, x_152, x_153, x_154);
lean_dec_ref(x_152);
x_156 = l_Lean_Expr_app___override(x_151, x_155);
lean_inc(x_99);
x_157 = l_Lean_Expr_app___override(x_156, x_99);
x_158 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__9;
x_159 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__11;
lean_ctor_set_tag(x_88, 1);
lean_ctor_set(x_88, 1, x_159);
lean_ctor_set(x_88, 0, x_7);
x_160 = l_Lean_Expr_const___override(x_158, x_88);
x_161 = l_Lean_Expr_app___override(x_160, x_8);
x_162 = l_Lean_Expr_app___override(x_161, x_18);
x_163 = l_Lean_Expr_app___override(x_162, x_9);
x_164 = l_Lean_Expr_app___override(x_163, x_10);
x_165 = l_Lean_Expr_app___override(x_164, x_110);
x_166 = l_Lean_Expr_app___override(x_165, x_11);
x_167 = l_Lean_Expr_app___override(x_157, x_166);
x_168 = l_Lean_Expr_app___override(x_167, x_150);
x_169 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__14;
x_170 = l_Lean_Expr_app___override(x_169, x_97);
x_171 = l_Lean_Expr_app___override(x_170, x_99);
x_172 = l_Lean_Expr_app___override(x_171, x_146);
x_173 = l_Lean_Expr_app___override(x_172, x_168);
x_174 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_174, 0, x_173);
return x_174;
}
}
else
{
lean_dec(x_110);
lean_free_object(x_6);
lean_free_object(x_88);
lean_dec(x_99);
lean_dec(x_97);
lean_dec(x_82);
lean_dec(x_81);
lean_dec(x_28);
lean_dec_ref(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_3);
return x_116;
}
}
else
{
lean_free_object(x_6);
lean_dec(x_102);
lean_free_object(x_88);
lean_dec(x_99);
lean_dec(x_97);
lean_dec(x_82);
lean_dec(x_81);
lean_dec(x_28);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec_ref(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_5);
lean_dec_ref(x_3);
return x_109;
}
}
else
{
lean_object* x_175; lean_object* x_176; lean_object* x_177; uint8_t x_178; uint8_t x_179; uint8_t x_180; lean_object* x_181; 
x_175 = lean_ctor_get(x_6, 1);
lean_inc(x_175);
lean_dec(x_6);
x_176 = lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__5;
lean_inc_ref(x_9);
x_177 = lean_array_push(x_176, x_9);
x_178 = 1;
x_179 = lean_unbox(x_37);
x_180 = lean_unbox(x_37);
lean_dec(x_37);
lean_inc(x_82);
x_181 = l_Lean_Meta_mkLambdaFVars(x_177, x_82, x_17, x_179, x_17, x_180, x_178, x_12, x_13, x_14, x_15);
lean_dec_ref(x_177);
if (lean_obj_tag(x_181) == 0)
{
lean_object* x_182; lean_object* x_183; lean_object* x_184; lean_object* x_185; lean_object* x_186; lean_object* x_187; lean_object* x_188; 
x_182 = lean_ctor_get(x_181, 0);
lean_inc(x_182);
lean_dec_ref(x_181);
x_183 = lean_box(0);
x_184 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__8;
lean_inc(x_81);
x_185 = l_Lean_Expr_app___override(x_184, x_81);
lean_inc(x_82);
x_186 = l_Lean_Expr_app___override(x_185, x_82);
lean_inc_ref(x_3);
x_187 = l_Lean_Expr_app___override(x_186, x_3);
lean_inc_ref(x_11);
lean_inc_ref(x_10);
lean_inc_ref(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc(x_81);
lean_inc(x_97);
x_188 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go(x_97, x_81, x_187, x_4, x_5, x_175, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
if (lean_obj_tag(x_188) == 0)
{
lean_object* x_189; lean_object* x_190; lean_object* x_191; lean_object* x_192; lean_object* x_193; lean_object* x_194; lean_object* x_195; lean_object* x_196; lean_object* x_197; uint8_t x_198; uint8_t x_199; lean_object* x_200; lean_object* x_201; lean_object* x_202; lean_object* x_203; lean_object* x_204; lean_object* x_205; lean_object* x_206; lean_object* x_207; lean_object* x_208; lean_object* x_209; lean_object* x_210; lean_object* x_211; lean_object* x_212; lean_object* x_213; lean_object* x_214; lean_object* x_215; lean_object* x_216; lean_object* x_217; lean_object* x_218; lean_object* x_219; 
x_189 = lean_ctor_get(x_188, 0);
lean_inc(x_189);
if (lean_is_exclusive(x_188)) {
 lean_ctor_release(x_188, 0);
 x_190 = x_188;
} else {
 lean_dec_ref(x_188);
 x_190 = lean_box(0);
}
lean_inc_ref(x_9);
x_191 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_191, 0, x_9);
lean_ctor_set(x_191, 1, x_183);
x_192 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__11;
x_193 = l_Lean_Expr_app___override(x_192, x_81);
x_194 = l_Lean_Expr_app___override(x_193, x_82);
x_195 = l_Lean_Expr_app___override(x_194, x_3);
x_196 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__7;
x_197 = lean_array_mk(x_191);
x_198 = lean_unbox(x_28);
x_199 = lean_unbox(x_28);
lean_dec(x_28);
lean_inc(x_182);
x_200 = l_Lean_Expr_betaRev(x_182, x_197, x_198, x_199);
lean_dec_ref(x_197);
x_201 = l_Lean_Expr_app___override(x_196, x_200);
lean_inc(x_99);
x_202 = l_Lean_Expr_app___override(x_201, x_99);
x_203 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__9;
x_204 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__11;
lean_ctor_set_tag(x_88, 1);
lean_ctor_set(x_88, 1, x_204);
lean_ctor_set(x_88, 0, x_7);
x_205 = l_Lean_Expr_const___override(x_203, x_88);
x_206 = l_Lean_Expr_app___override(x_205, x_8);
x_207 = l_Lean_Expr_app___override(x_206, x_18);
x_208 = l_Lean_Expr_app___override(x_207, x_9);
x_209 = l_Lean_Expr_app___override(x_208, x_10);
x_210 = l_Lean_Expr_app___override(x_209, x_182);
x_211 = l_Lean_Expr_app___override(x_210, x_11);
x_212 = l_Lean_Expr_app___override(x_202, x_211);
x_213 = l_Lean_Expr_app___override(x_212, x_195);
x_214 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__14;
x_215 = l_Lean_Expr_app___override(x_214, x_97);
x_216 = l_Lean_Expr_app___override(x_215, x_99);
x_217 = l_Lean_Expr_app___override(x_216, x_189);
x_218 = l_Lean_Expr_app___override(x_217, x_213);
if (lean_is_scalar(x_190)) {
 x_219 = lean_alloc_ctor(0, 1, 0);
} else {
 x_219 = x_190;
}
lean_ctor_set(x_219, 0, x_218);
return x_219;
}
else
{
lean_dec(x_182);
lean_free_object(x_88);
lean_dec(x_99);
lean_dec(x_97);
lean_dec(x_82);
lean_dec(x_81);
lean_dec(x_28);
lean_dec_ref(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_3);
return x_188;
}
}
else
{
lean_dec(x_175);
lean_free_object(x_88);
lean_dec(x_99);
lean_dec(x_97);
lean_dec(x_82);
lean_dec(x_81);
lean_dec(x_28);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec_ref(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_5);
lean_dec_ref(x_3);
return x_181;
}
}
}
else
{
lean_object* x_220; lean_object* x_221; lean_object* x_222; lean_object* x_223; lean_object* x_224; uint8_t x_225; uint8_t x_226; uint8_t x_227; lean_object* x_228; 
x_220 = lean_ctor_get(x_88, 0);
lean_inc(x_220);
lean_dec(x_88);
x_221 = lean_ctor_get(x_6, 1);
lean_inc(x_221);
if (lean_is_exclusive(x_6)) {
 lean_ctor_release(x_6, 0);
 lean_ctor_release(x_6, 1);
 x_222 = x_6;
} else {
 lean_dec_ref(x_6);
 x_222 = lean_box(0);
}
x_223 = lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__5;
lean_inc_ref(x_9);
x_224 = lean_array_push(x_223, x_9);
x_225 = 1;
x_226 = lean_unbox(x_37);
x_227 = lean_unbox(x_37);
lean_dec(x_37);
lean_inc(x_82);
x_228 = l_Lean_Meta_mkLambdaFVars(x_224, x_82, x_17, x_226, x_17, x_227, x_225, x_12, x_13, x_14, x_15);
lean_dec_ref(x_224);
if (lean_obj_tag(x_228) == 0)
{
lean_object* x_229; lean_object* x_230; lean_object* x_231; lean_object* x_232; lean_object* x_233; lean_object* x_234; lean_object* x_235; 
x_229 = lean_ctor_get(x_228, 0);
lean_inc(x_229);
lean_dec_ref(x_228);
x_230 = lean_box(0);
x_231 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__8;
lean_inc(x_81);
x_232 = l_Lean_Expr_app___override(x_231, x_81);
lean_inc(x_82);
x_233 = l_Lean_Expr_app___override(x_232, x_82);
lean_inc_ref(x_3);
x_234 = l_Lean_Expr_app___override(x_233, x_3);
lean_inc_ref(x_11);
lean_inc_ref(x_10);
lean_inc_ref(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc(x_81);
lean_inc(x_97);
x_235 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go(x_97, x_81, x_234, x_4, x_5, x_221, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
if (lean_obj_tag(x_235) == 0)
{
lean_object* x_236; lean_object* x_237; lean_object* x_238; lean_object* x_239; lean_object* x_240; lean_object* x_241; lean_object* x_242; lean_object* x_243; lean_object* x_244; uint8_t x_245; uint8_t x_246; lean_object* x_247; lean_object* x_248; lean_object* x_249; lean_object* x_250; lean_object* x_251; lean_object* x_252; lean_object* x_253; lean_object* x_254; lean_object* x_255; lean_object* x_256; lean_object* x_257; lean_object* x_258; lean_object* x_259; lean_object* x_260; lean_object* x_261; lean_object* x_262; lean_object* x_263; lean_object* x_264; lean_object* x_265; lean_object* x_266; lean_object* x_267; 
x_236 = lean_ctor_get(x_235, 0);
lean_inc(x_236);
if (lean_is_exclusive(x_235)) {
 lean_ctor_release(x_235, 0);
 x_237 = x_235;
} else {
 lean_dec_ref(x_235);
 x_237 = lean_box(0);
}
lean_inc_ref(x_9);
if (lean_is_scalar(x_222)) {
 x_238 = lean_alloc_ctor(1, 2, 0);
} else {
 x_238 = x_222;
}
lean_ctor_set(x_238, 0, x_9);
lean_ctor_set(x_238, 1, x_230);
x_239 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__11;
x_240 = l_Lean_Expr_app___override(x_239, x_81);
x_241 = l_Lean_Expr_app___override(x_240, x_82);
x_242 = l_Lean_Expr_app___override(x_241, x_3);
x_243 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__7;
x_244 = lean_array_mk(x_238);
x_245 = lean_unbox(x_28);
x_246 = lean_unbox(x_28);
lean_dec(x_28);
lean_inc(x_229);
x_247 = l_Lean_Expr_betaRev(x_229, x_244, x_245, x_246);
lean_dec_ref(x_244);
x_248 = l_Lean_Expr_app___override(x_243, x_247);
lean_inc(x_220);
x_249 = l_Lean_Expr_app___override(x_248, x_220);
x_250 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__9;
x_251 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__11;
x_252 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_252, 0, x_7);
lean_ctor_set(x_252, 1, x_251);
x_253 = l_Lean_Expr_const___override(x_250, x_252);
x_254 = l_Lean_Expr_app___override(x_253, x_8);
x_255 = l_Lean_Expr_app___override(x_254, x_18);
x_256 = l_Lean_Expr_app___override(x_255, x_9);
x_257 = l_Lean_Expr_app___override(x_256, x_10);
x_258 = l_Lean_Expr_app___override(x_257, x_229);
x_259 = l_Lean_Expr_app___override(x_258, x_11);
x_260 = l_Lean_Expr_app___override(x_249, x_259);
x_261 = l_Lean_Expr_app___override(x_260, x_242);
x_262 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__14;
x_263 = l_Lean_Expr_app___override(x_262, x_97);
x_264 = l_Lean_Expr_app___override(x_263, x_220);
x_265 = l_Lean_Expr_app___override(x_264, x_236);
x_266 = l_Lean_Expr_app___override(x_265, x_261);
if (lean_is_scalar(x_237)) {
 x_267 = lean_alloc_ctor(0, 1, 0);
} else {
 x_267 = x_237;
}
lean_ctor_set(x_267, 0, x_266);
return x_267;
}
else
{
lean_dec(x_229);
lean_dec(x_222);
lean_dec(x_220);
lean_dec(x_97);
lean_dec(x_82);
lean_dec(x_81);
lean_dec(x_28);
lean_dec_ref(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_3);
return x_235;
}
}
else
{
lean_dec(x_222);
lean_dec(x_221);
lean_dec(x_220);
lean_dec(x_97);
lean_dec(x_82);
lean_dec(x_81);
lean_dec(x_28);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec_ref(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_5);
lean_dec_ref(x_3);
return x_228;
}
}
}
else
{
lean_object* x_268; uint8_t x_269; 
x_268 = lean_ctor_get(x_87, 0);
lean_inc(x_268);
lean_dec(x_87);
x_269 = !lean_is_exclusive(x_88);
if (x_269 == 0)
{
lean_object* x_270; lean_object* x_271; uint8_t x_272; 
x_270 = lean_ctor_get(x_88, 0);
x_271 = lean_ctor_get(x_88, 1);
lean_dec(x_271);
x_272 = !lean_is_exclusive(x_6);
if (x_272 == 0)
{
lean_object* x_273; lean_object* x_274; lean_object* x_275; lean_object* x_276; uint8_t x_277; uint8_t x_278; uint8_t x_279; lean_object* x_280; 
x_273 = lean_ctor_get(x_6, 1);
x_274 = lean_ctor_get(x_6, 0);
lean_dec(x_274);
x_275 = lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__5;
lean_inc_ref(x_9);
x_276 = lean_array_push(x_275, x_9);
x_277 = 1;
x_278 = lean_unbox(x_37);
x_279 = lean_unbox(x_37);
lean_dec(x_37);
lean_inc(x_81);
x_280 = l_Lean_Meta_mkLambdaFVars(x_276, x_81, x_17, x_278, x_17, x_279, x_277, x_12, x_13, x_14, x_15);
lean_dec_ref(x_276);
if (lean_obj_tag(x_280) == 0)
{
lean_object* x_281; lean_object* x_282; lean_object* x_283; lean_object* x_284; lean_object* x_285; lean_object* x_286; lean_object* x_287; 
x_281 = lean_ctor_get(x_280, 0);
lean_inc(x_281);
lean_dec_ref(x_280);
x_282 = lean_box(0);
x_283 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__11;
lean_inc(x_81);
x_284 = l_Lean_Expr_app___override(x_283, x_81);
lean_inc(x_82);
x_285 = l_Lean_Expr_app___override(x_284, x_82);
lean_inc_ref(x_3);
x_286 = l_Lean_Expr_app___override(x_285, x_3);
lean_inc_ref(x_11);
lean_inc_ref(x_10);
lean_inc_ref(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc(x_82);
lean_inc(x_270);
x_287 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go(x_270, x_82, x_286, x_4, x_5, x_273, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
if (lean_obj_tag(x_287) == 0)
{
uint8_t x_288; 
x_288 = !lean_is_exclusive(x_287);
if (x_288 == 0)
{
lean_object* x_289; lean_object* x_290; lean_object* x_291; lean_object* x_292; lean_object* x_293; lean_object* x_294; lean_object* x_295; uint8_t x_296; uint8_t x_297; lean_object* x_298; lean_object* x_299; lean_object* x_300; lean_object* x_301; lean_object* x_302; lean_object* x_303; lean_object* x_304; lean_object* x_305; lean_object* x_306; lean_object* x_307; lean_object* x_308; lean_object* x_309; lean_object* x_310; lean_object* x_311; lean_object* x_312; lean_object* x_313; lean_object* x_314; lean_object* x_315; lean_object* x_316; 
x_289 = lean_ctor_get(x_287, 0);
lean_inc_ref(x_9);
lean_ctor_set(x_6, 1, x_282);
lean_ctor_set(x_6, 0, x_9);
x_290 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__8;
x_291 = l_Lean_Expr_app___override(x_290, x_81);
x_292 = l_Lean_Expr_app___override(x_291, x_82);
x_293 = l_Lean_Expr_app___override(x_292, x_3);
x_294 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__7;
x_295 = lean_array_mk(x_6);
x_296 = lean_unbox(x_28);
x_297 = lean_unbox(x_28);
lean_dec(x_28);
lean_inc(x_281);
x_298 = l_Lean_Expr_betaRev(x_281, x_295, x_296, x_297);
lean_dec_ref(x_295);
x_299 = l_Lean_Expr_app___override(x_294, x_298);
lean_inc(x_268);
x_300 = l_Lean_Expr_app___override(x_299, x_268);
x_301 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__9;
x_302 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__11;
lean_ctor_set_tag(x_88, 1);
lean_ctor_set(x_88, 1, x_302);
lean_ctor_set(x_88, 0, x_7);
x_303 = l_Lean_Expr_const___override(x_301, x_88);
x_304 = l_Lean_Expr_app___override(x_303, x_8);
x_305 = l_Lean_Expr_app___override(x_304, x_18);
x_306 = l_Lean_Expr_app___override(x_305, x_9);
x_307 = l_Lean_Expr_app___override(x_306, x_10);
x_308 = l_Lean_Expr_app___override(x_307, x_281);
x_309 = l_Lean_Expr_app___override(x_308, x_11);
x_310 = l_Lean_Expr_app___override(x_300, x_309);
x_311 = l_Lean_Expr_app___override(x_310, x_293);
x_312 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__14;
x_313 = l_Lean_Expr_app___override(x_312, x_268);
x_314 = l_Lean_Expr_app___override(x_313, x_270);
x_315 = l_Lean_Expr_app___override(x_314, x_311);
x_316 = l_Lean_Expr_app___override(x_315, x_289);
lean_ctor_set(x_287, 0, x_316);
return x_287;
}
else
{
lean_object* x_317; lean_object* x_318; lean_object* x_319; lean_object* x_320; lean_object* x_321; lean_object* x_322; lean_object* x_323; uint8_t x_324; uint8_t x_325; lean_object* x_326; lean_object* x_327; lean_object* x_328; lean_object* x_329; lean_object* x_330; lean_object* x_331; lean_object* x_332; lean_object* x_333; lean_object* x_334; lean_object* x_335; lean_object* x_336; lean_object* x_337; lean_object* x_338; lean_object* x_339; lean_object* x_340; lean_object* x_341; lean_object* x_342; lean_object* x_343; lean_object* x_344; lean_object* x_345; 
x_317 = lean_ctor_get(x_287, 0);
lean_inc(x_317);
lean_dec(x_287);
lean_inc_ref(x_9);
lean_ctor_set(x_6, 1, x_282);
lean_ctor_set(x_6, 0, x_9);
x_318 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__8;
x_319 = l_Lean_Expr_app___override(x_318, x_81);
x_320 = l_Lean_Expr_app___override(x_319, x_82);
x_321 = l_Lean_Expr_app___override(x_320, x_3);
x_322 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__7;
x_323 = lean_array_mk(x_6);
x_324 = lean_unbox(x_28);
x_325 = lean_unbox(x_28);
lean_dec(x_28);
lean_inc(x_281);
x_326 = l_Lean_Expr_betaRev(x_281, x_323, x_324, x_325);
lean_dec_ref(x_323);
x_327 = l_Lean_Expr_app___override(x_322, x_326);
lean_inc(x_268);
x_328 = l_Lean_Expr_app___override(x_327, x_268);
x_329 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__9;
x_330 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__11;
lean_ctor_set_tag(x_88, 1);
lean_ctor_set(x_88, 1, x_330);
lean_ctor_set(x_88, 0, x_7);
x_331 = l_Lean_Expr_const___override(x_329, x_88);
x_332 = l_Lean_Expr_app___override(x_331, x_8);
x_333 = l_Lean_Expr_app___override(x_332, x_18);
x_334 = l_Lean_Expr_app___override(x_333, x_9);
x_335 = l_Lean_Expr_app___override(x_334, x_10);
x_336 = l_Lean_Expr_app___override(x_335, x_281);
x_337 = l_Lean_Expr_app___override(x_336, x_11);
x_338 = l_Lean_Expr_app___override(x_328, x_337);
x_339 = l_Lean_Expr_app___override(x_338, x_321);
x_340 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__14;
x_341 = l_Lean_Expr_app___override(x_340, x_268);
x_342 = l_Lean_Expr_app___override(x_341, x_270);
x_343 = l_Lean_Expr_app___override(x_342, x_339);
x_344 = l_Lean_Expr_app___override(x_343, x_317);
x_345 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_345, 0, x_344);
return x_345;
}
}
else
{
lean_dec(x_281);
lean_free_object(x_6);
lean_free_object(x_88);
lean_dec(x_270);
lean_dec(x_268);
lean_dec(x_82);
lean_dec(x_81);
lean_dec(x_28);
lean_dec_ref(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_3);
return x_287;
}
}
else
{
lean_free_object(x_6);
lean_dec(x_273);
lean_free_object(x_88);
lean_dec(x_270);
lean_dec(x_268);
lean_dec(x_82);
lean_dec(x_81);
lean_dec(x_28);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec_ref(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_5);
lean_dec_ref(x_3);
return x_280;
}
}
else
{
lean_object* x_346; lean_object* x_347; lean_object* x_348; uint8_t x_349; uint8_t x_350; uint8_t x_351; lean_object* x_352; 
x_346 = lean_ctor_get(x_6, 1);
lean_inc(x_346);
lean_dec(x_6);
x_347 = lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__5;
lean_inc_ref(x_9);
x_348 = lean_array_push(x_347, x_9);
x_349 = 1;
x_350 = lean_unbox(x_37);
x_351 = lean_unbox(x_37);
lean_dec(x_37);
lean_inc(x_81);
x_352 = l_Lean_Meta_mkLambdaFVars(x_348, x_81, x_17, x_350, x_17, x_351, x_349, x_12, x_13, x_14, x_15);
lean_dec_ref(x_348);
if (lean_obj_tag(x_352) == 0)
{
lean_object* x_353; lean_object* x_354; lean_object* x_355; lean_object* x_356; lean_object* x_357; lean_object* x_358; lean_object* x_359; 
x_353 = lean_ctor_get(x_352, 0);
lean_inc(x_353);
lean_dec_ref(x_352);
x_354 = lean_box(0);
x_355 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__11;
lean_inc(x_81);
x_356 = l_Lean_Expr_app___override(x_355, x_81);
lean_inc(x_82);
x_357 = l_Lean_Expr_app___override(x_356, x_82);
lean_inc_ref(x_3);
x_358 = l_Lean_Expr_app___override(x_357, x_3);
lean_inc_ref(x_11);
lean_inc_ref(x_10);
lean_inc_ref(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc(x_82);
lean_inc(x_270);
x_359 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go(x_270, x_82, x_358, x_4, x_5, x_346, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
if (lean_obj_tag(x_359) == 0)
{
lean_object* x_360; lean_object* x_361; lean_object* x_362; lean_object* x_363; lean_object* x_364; lean_object* x_365; lean_object* x_366; lean_object* x_367; lean_object* x_368; uint8_t x_369; uint8_t x_370; lean_object* x_371; lean_object* x_372; lean_object* x_373; lean_object* x_374; lean_object* x_375; lean_object* x_376; lean_object* x_377; lean_object* x_378; lean_object* x_379; lean_object* x_380; lean_object* x_381; lean_object* x_382; lean_object* x_383; lean_object* x_384; lean_object* x_385; lean_object* x_386; lean_object* x_387; lean_object* x_388; lean_object* x_389; lean_object* x_390; 
x_360 = lean_ctor_get(x_359, 0);
lean_inc(x_360);
if (lean_is_exclusive(x_359)) {
 lean_ctor_release(x_359, 0);
 x_361 = x_359;
} else {
 lean_dec_ref(x_359);
 x_361 = lean_box(0);
}
lean_inc_ref(x_9);
x_362 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_362, 0, x_9);
lean_ctor_set(x_362, 1, x_354);
x_363 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__8;
x_364 = l_Lean_Expr_app___override(x_363, x_81);
x_365 = l_Lean_Expr_app___override(x_364, x_82);
x_366 = l_Lean_Expr_app___override(x_365, x_3);
x_367 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__7;
x_368 = lean_array_mk(x_362);
x_369 = lean_unbox(x_28);
x_370 = lean_unbox(x_28);
lean_dec(x_28);
lean_inc(x_353);
x_371 = l_Lean_Expr_betaRev(x_353, x_368, x_369, x_370);
lean_dec_ref(x_368);
x_372 = l_Lean_Expr_app___override(x_367, x_371);
lean_inc(x_268);
x_373 = l_Lean_Expr_app___override(x_372, x_268);
x_374 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__9;
x_375 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__11;
lean_ctor_set_tag(x_88, 1);
lean_ctor_set(x_88, 1, x_375);
lean_ctor_set(x_88, 0, x_7);
x_376 = l_Lean_Expr_const___override(x_374, x_88);
x_377 = l_Lean_Expr_app___override(x_376, x_8);
x_378 = l_Lean_Expr_app___override(x_377, x_18);
x_379 = l_Lean_Expr_app___override(x_378, x_9);
x_380 = l_Lean_Expr_app___override(x_379, x_10);
x_381 = l_Lean_Expr_app___override(x_380, x_353);
x_382 = l_Lean_Expr_app___override(x_381, x_11);
x_383 = l_Lean_Expr_app___override(x_373, x_382);
x_384 = l_Lean_Expr_app___override(x_383, x_366);
x_385 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__14;
x_386 = l_Lean_Expr_app___override(x_385, x_268);
x_387 = l_Lean_Expr_app___override(x_386, x_270);
x_388 = l_Lean_Expr_app___override(x_387, x_384);
x_389 = l_Lean_Expr_app___override(x_388, x_360);
if (lean_is_scalar(x_361)) {
 x_390 = lean_alloc_ctor(0, 1, 0);
} else {
 x_390 = x_361;
}
lean_ctor_set(x_390, 0, x_389);
return x_390;
}
else
{
lean_dec(x_353);
lean_free_object(x_88);
lean_dec(x_270);
lean_dec(x_268);
lean_dec(x_82);
lean_dec(x_81);
lean_dec(x_28);
lean_dec_ref(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_3);
return x_359;
}
}
else
{
lean_dec(x_346);
lean_free_object(x_88);
lean_dec(x_270);
lean_dec(x_268);
lean_dec(x_82);
lean_dec(x_81);
lean_dec(x_28);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec_ref(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_5);
lean_dec_ref(x_3);
return x_352;
}
}
}
else
{
lean_object* x_391; lean_object* x_392; lean_object* x_393; lean_object* x_394; lean_object* x_395; uint8_t x_396; uint8_t x_397; uint8_t x_398; lean_object* x_399; 
x_391 = lean_ctor_get(x_88, 0);
lean_inc(x_391);
lean_dec(x_88);
x_392 = lean_ctor_get(x_6, 1);
lean_inc(x_392);
if (lean_is_exclusive(x_6)) {
 lean_ctor_release(x_6, 0);
 lean_ctor_release(x_6, 1);
 x_393 = x_6;
} else {
 lean_dec_ref(x_6);
 x_393 = lean_box(0);
}
x_394 = lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__5;
lean_inc_ref(x_9);
x_395 = lean_array_push(x_394, x_9);
x_396 = 1;
x_397 = lean_unbox(x_37);
x_398 = lean_unbox(x_37);
lean_dec(x_37);
lean_inc(x_81);
x_399 = l_Lean_Meta_mkLambdaFVars(x_395, x_81, x_17, x_397, x_17, x_398, x_396, x_12, x_13, x_14, x_15);
lean_dec_ref(x_395);
if (lean_obj_tag(x_399) == 0)
{
lean_object* x_400; lean_object* x_401; lean_object* x_402; lean_object* x_403; lean_object* x_404; lean_object* x_405; lean_object* x_406; 
x_400 = lean_ctor_get(x_399, 0);
lean_inc(x_400);
lean_dec_ref(x_399);
x_401 = lean_box(0);
x_402 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__11;
lean_inc(x_81);
x_403 = l_Lean_Expr_app___override(x_402, x_81);
lean_inc(x_82);
x_404 = l_Lean_Expr_app___override(x_403, x_82);
lean_inc_ref(x_3);
x_405 = l_Lean_Expr_app___override(x_404, x_3);
lean_inc_ref(x_11);
lean_inc_ref(x_10);
lean_inc_ref(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc(x_82);
lean_inc(x_391);
x_406 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go(x_391, x_82, x_405, x_4, x_5, x_392, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
if (lean_obj_tag(x_406) == 0)
{
lean_object* x_407; lean_object* x_408; lean_object* x_409; lean_object* x_410; lean_object* x_411; lean_object* x_412; lean_object* x_413; lean_object* x_414; lean_object* x_415; uint8_t x_416; uint8_t x_417; lean_object* x_418; lean_object* x_419; lean_object* x_420; lean_object* x_421; lean_object* x_422; lean_object* x_423; lean_object* x_424; lean_object* x_425; lean_object* x_426; lean_object* x_427; lean_object* x_428; lean_object* x_429; lean_object* x_430; lean_object* x_431; lean_object* x_432; lean_object* x_433; lean_object* x_434; lean_object* x_435; lean_object* x_436; lean_object* x_437; lean_object* x_438; 
x_407 = lean_ctor_get(x_406, 0);
lean_inc(x_407);
if (lean_is_exclusive(x_406)) {
 lean_ctor_release(x_406, 0);
 x_408 = x_406;
} else {
 lean_dec_ref(x_406);
 x_408 = lean_box(0);
}
lean_inc_ref(x_9);
if (lean_is_scalar(x_393)) {
 x_409 = lean_alloc_ctor(1, 2, 0);
} else {
 x_409 = x_393;
}
lean_ctor_set(x_409, 0, x_9);
lean_ctor_set(x_409, 1, x_401);
x_410 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__8;
x_411 = l_Lean_Expr_app___override(x_410, x_81);
x_412 = l_Lean_Expr_app___override(x_411, x_82);
x_413 = l_Lean_Expr_app___override(x_412, x_3);
x_414 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__7;
x_415 = lean_array_mk(x_409);
x_416 = lean_unbox(x_28);
x_417 = lean_unbox(x_28);
lean_dec(x_28);
lean_inc(x_400);
x_418 = l_Lean_Expr_betaRev(x_400, x_415, x_416, x_417);
lean_dec_ref(x_415);
x_419 = l_Lean_Expr_app___override(x_414, x_418);
lean_inc(x_268);
x_420 = l_Lean_Expr_app___override(x_419, x_268);
x_421 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__9;
x_422 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__11;
x_423 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_423, 0, x_7);
lean_ctor_set(x_423, 1, x_422);
x_424 = l_Lean_Expr_const___override(x_421, x_423);
x_425 = l_Lean_Expr_app___override(x_424, x_8);
x_426 = l_Lean_Expr_app___override(x_425, x_18);
x_427 = l_Lean_Expr_app___override(x_426, x_9);
x_428 = l_Lean_Expr_app___override(x_427, x_10);
x_429 = l_Lean_Expr_app___override(x_428, x_400);
x_430 = l_Lean_Expr_app___override(x_429, x_11);
x_431 = l_Lean_Expr_app___override(x_420, x_430);
x_432 = l_Lean_Expr_app___override(x_431, x_413);
x_433 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__14;
x_434 = l_Lean_Expr_app___override(x_433, x_268);
x_435 = l_Lean_Expr_app___override(x_434, x_391);
x_436 = l_Lean_Expr_app___override(x_435, x_432);
x_437 = l_Lean_Expr_app___override(x_436, x_407);
if (lean_is_scalar(x_408)) {
 x_438 = lean_alloc_ctor(0, 1, 0);
} else {
 x_438 = x_408;
}
lean_ctor_set(x_438, 0, x_437);
return x_438;
}
else
{
lean_dec(x_400);
lean_dec(x_393);
lean_dec(x_391);
lean_dec(x_268);
lean_dec(x_82);
lean_dec(x_81);
lean_dec(x_28);
lean_dec_ref(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_3);
return x_406;
}
}
else
{
lean_dec(x_393);
lean_dec(x_392);
lean_dec(x_391);
lean_dec(x_268);
lean_dec(x_82);
lean_dec(x_81);
lean_dec(x_28);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec_ref(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_5);
lean_dec_ref(x_3);
return x_399;
}
}
}
}
}
}
else
{
uint8_t x_439; 
lean_dec(x_82);
lean_dec(x_81);
lean_dec(x_37);
lean_dec(x_28);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec_ref(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_3);
x_439 = !lean_is_exclusive(x_86);
if (x_439 == 0)
{
return x_86;
}
else
{
lean_object* x_440; lean_object* x_441; 
x_440 = lean_ctor_get(x_86, 0);
lean_inc(x_440);
lean_dec(x_86);
x_441 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_441, 0, x_440);
return x_441;
}
}
}
}
else
{
uint8_t x_442; 
lean_dec(x_28);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec_ref(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
x_442 = !lean_is_exclusive(x_34);
if (x_442 == 0)
{
return x_34;
}
else
{
lean_object* x_443; lean_object* x_444; 
x_443 = lean_ctor_get(x_34, 0);
lean_inc(x_443);
lean_dec(x_34);
x_444 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_444, 0, x_443);
return x_444;
}
}
}
else
{
lean_dec(x_28);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_445; lean_object* x_446; 
lean_dec_ref(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_1);
x_445 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__12;
x_446 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg(x_445, x_12, x_13, x_14, x_15);
lean_dec(x_15);
lean_dec(x_13);
lean_dec_ref(x_12);
return x_446;
}
else
{
if (lean_obj_tag(x_5) == 0)
{
lean_object* x_447; lean_object* x_448; 
lean_dec_ref(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_1);
x_447 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__13;
x_448 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg(x_447, x_12, x_13, x_14, x_15);
lean_dec(x_15);
lean_dec(x_13);
lean_dec_ref(x_12);
return x_448;
}
else
{
lean_object* x_449; lean_object* x_450; lean_object* x_451; lean_object* x_452; lean_object* x_453; 
x_449 = lean_ctor_get(x_5, 0);
lean_inc(x_449);
x_450 = lean_ctor_get(x_4, 1);
x_451 = lean_ctor_get(x_5, 1);
lean_inc(x_451);
lean_dec_ref(x_5);
x_452 = lean_ctor_get(x_449, 0);
lean_inc(x_452);
x_453 = lean_ctor_get(x_449, 1);
lean_inc(x_453);
lean_dec(x_449);
x_2 = x_452;
x_3 = x_453;
x_4 = x_450;
x_5 = x_451;
goto _start;
}
}
}
}
else
{
uint8_t x_455; 
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec_ref(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_455 = !lean_is_exclusive(x_24);
if (x_455 == 0)
{
return x_24;
}
else
{
lean_object* x_456; lean_object* x_457; 
x_456 = lean_ctor_get(x_24, 0);
lean_inc(x_456);
lean_dec(x_24);
x_457 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_457, 0, x_456);
return x_457;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
lean_object* x_17; 
x_17 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
lean_dec(x_4);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
lean_object* x_17; 
x_17 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
lean_object* x_17; 
x_17 = lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
lean_dec(x_4);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
lean_object* x_18; lean_object* x_19; 
lean_inc(x_4);
x_18 = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__0___boxed), 16, 11);
lean_closure_set(x_18, 0, x_1);
lean_closure_set(x_18, 1, x_2);
lean_closure_set(x_18, 2, x_3);
lean_closure_set(x_18, 3, x_4);
lean_closure_set(x_18, 4, x_12);
lean_closure_set(x_18, 5, x_5);
lean_closure_set(x_18, 6, x_6);
lean_closure_set(x_18, 7, x_7);
lean_closure_set(x_18, 8, x_8);
lean_closure_set(x_18, 9, x_9);
lean_closure_set(x_18, 10, x_11);
x_19 = lp_mathlib_ExistsAndEq_withNestedExistsIntro___redArg(x_10, x_4, x_18, x_13, x_14, x_15, x_16);
return x_19;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__1___boxed(lean_object** _args) {
lean_object* x_1 = _args[0];
lean_object* x_2 = _args[1];
lean_object* x_3 = _args[2];
lean_object* x_4 = _args[3];
lean_object* x_5 = _args[4];
lean_object* x_6 = _args[5];
lean_object* x_7 = _args[6];
lean_object* x_8 = _args[7];
lean_object* x_9 = _args[8];
lean_object* x_10 = _args[9];
lean_object* x_11 = _args[10];
lean_object* x_12 = _args[11];
lean_object* x_13 = _args[12];
lean_object* x_14 = _args[13];
lean_object* x_15 = _args[14];
lean_object* x_16 = _args[15];
lean_object* x_17 = _args[16];
_start:
{
lean_object* x_18; 
x_18 = lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__1(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, uint8_t x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19) {
_start:
{
lean_object* x_21; lean_object* x_22; 
lean_inc_ref(x_9);
lean_inc_ref(x_8);
lean_inc_ref(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_3);
lean_inc_ref(x_15);
lean_inc_ref(x_2);
x_21 = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__1___boxed), 17, 10);
lean_closure_set(x_21, 0, x_1);
lean_closure_set(x_21, 1, x_2);
lean_closure_set(x_21, 2, x_15);
lean_closure_set(x_21, 3, x_3);
lean_closure_set(x_21, 4, x_4);
lean_closure_set(x_21, 5, x_5);
lean_closure_set(x_21, 6, x_6);
lean_closure_set(x_21, 7, x_7);
lean_closure_set(x_21, 8, x_8);
lean_closure_set(x_21, 9, x_9);
lean_inc(x_19);
lean_inc_ref(x_18);
lean_inc(x_17);
lean_inc_ref(x_16);
lean_inc_ref(x_7);
lean_inc_ref(x_15);
lean_inc_ref(x_9);
lean_inc_ref(x_6);
x_22 = lp_mathlib_ExistsAndEq_withExistsElimAlongPath(x_5, x_6, x_2, x_9, x_15, x_7, x_8, x_3, x_4, x_21, x_16, x_17, x_18, x_19);
if (lean_obj_tag(x_22) == 0)
{
lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; uint8_t x_27; uint8_t x_28; lean_object* x_29; 
x_23 = lean_ctor_get(x_22, 0);
lean_inc(x_23);
lean_dec_ref(x_22);
x_24 = lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1___closed__0;
x_25 = lean_array_push(x_24, x_7);
x_26 = lean_array_push(x_25, x_15);
x_27 = 1;
x_28 = 1;
x_29 = l_Lean_Meta_mkLambdaFVars(x_26, x_23, x_10, x_27, x_10, x_27, x_28, x_16, x_17, x_18, x_19);
lean_dec_ref(x_26);
if (lean_obj_tag(x_29) == 0)
{
lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; 
x_30 = lean_ctor_get(x_29, 0);
lean_inc(x_30);
lean_dec_ref(x_29);
x_31 = lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1___closed__1;
x_32 = l_Lean_Name_mkStr2(x_11, x_31);
x_33 = l_Lean_Expr_const___override(x_32, x_12);
x_34 = l_Lean_Expr_app___override(x_33, x_6);
x_35 = l_Lean_Expr_app___override(x_34, x_13);
x_36 = l_Lean_Expr_app___override(x_35, x_9);
lean_inc_ref(x_14);
x_37 = l_Lean_Expr_app___override(x_36, x_14);
x_38 = l_Lean_Expr_app___override(x_37, x_30);
x_39 = lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__5;
x_40 = lean_array_push(x_39, x_14);
x_41 = l_Lean_Meta_mkLambdaFVars(x_40, x_38, x_10, x_27, x_10, x_27, x_28, x_16, x_17, x_18, x_19);
lean_dec(x_19);
lean_dec_ref(x_18);
lean_dec(x_17);
lean_dec_ref(x_16);
lean_dec_ref(x_40);
return x_41;
}
else
{
lean_dec(x_19);
lean_dec_ref(x_18);
lean_dec(x_17);
lean_dec_ref(x_16);
lean_dec_ref(x_14);
lean_dec_ref(x_13);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec_ref(x_9);
lean_dec_ref(x_6);
return x_29;
}
}
else
{
lean_dec(x_19);
lean_dec_ref(x_18);
lean_dec(x_17);
lean_dec_ref(x_16);
lean_dec_ref(x_15);
lean_dec_ref(x_14);
lean_dec_ref(x_13);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec_ref(x_9);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
return x_22;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__2___boxed(lean_object** _args) {
lean_object* x_1 = _args[0];
lean_object* x_2 = _args[1];
lean_object* x_3 = _args[2];
lean_object* x_4 = _args[3];
lean_object* x_5 = _args[4];
lean_object* x_6 = _args[5];
lean_object* x_7 = _args[6];
lean_object* x_8 = _args[7];
lean_object* x_9 = _args[8];
lean_object* x_10 = _args[9];
lean_object* x_11 = _args[10];
lean_object* x_12 = _args[11];
lean_object* x_13 = _args[12];
lean_object* x_14 = _args[13];
lean_object* x_15 = _args[14];
lean_object* x_16 = _args[15];
lean_object* x_17 = _args[16];
lean_object* x_18 = _args[17];
lean_object* x_19 = _args[18];
lean_object* x_20 = _args[19];
_start:
{
uint8_t x_21; lean_object* x_22; 
x_21 = lean_unbox(x_10);
x_22 = lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__2(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_21, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19);
return x_22;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, uint8_t x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19) {
_start:
{
lean_object* x_21; lean_object* x_22; uint8_t x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; 
lean_inc_ref(x_15);
x_21 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_21, 0, x_15);
lean_ctor_set(x_21, 1, x_1);
x_22 = lean_array_mk(x_21);
x_23 = 0;
lean_inc_ref(x_2);
x_24 = l_Lean_Expr_betaRev(x_2, x_22, x_23, x_23);
lean_dec_ref(x_22);
x_25 = lean_box(x_23);
lean_inc_ref(x_24);
x_26 = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__2___boxed), 20, 14);
lean_closure_set(x_26, 0, x_3);
lean_closure_set(x_26, 1, x_24);
lean_closure_set(x_26, 2, x_4);
lean_closure_set(x_26, 3, x_5);
lean_closure_set(x_26, 4, x_6);
lean_closure_set(x_26, 5, x_7);
lean_closure_set(x_26, 6, x_15);
lean_closure_set(x_26, 7, x_8);
lean_closure_set(x_26, 8, x_9);
lean_closure_set(x_26, 9, x_25);
lean_closure_set(x_26, 10, x_10);
lean_closure_set(x_26, 11, x_11);
lean_closure_set(x_26, 12, x_2);
lean_closure_set(x_26, 13, x_12);
x_27 = lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0___redArg(x_13, x_14, x_24, x_26, x_16, x_17, x_18, x_19);
return x_27;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__3___boxed(lean_object** _args) {
lean_object* x_1 = _args[0];
lean_object* x_2 = _args[1];
lean_object* x_3 = _args[2];
lean_object* x_4 = _args[3];
lean_object* x_5 = _args[4];
lean_object* x_6 = _args[5];
lean_object* x_7 = _args[6];
lean_object* x_8 = _args[7];
lean_object* x_9 = _args[8];
lean_object* x_10 = _args[9];
lean_object* x_11 = _args[10];
lean_object* x_12 = _args[11];
lean_object* x_13 = _args[12];
lean_object* x_14 = _args[13];
lean_object* x_15 = _args[14];
lean_object* x_16 = _args[15];
lean_object* x_17 = _args[16];
lean_object* x_18 = _args[17];
lean_object* x_19 = _args[18];
lean_object* x_20 = _args[19];
_start:
{
uint8_t x_21; lean_object* x_22; 
x_21 = lean_unbox(x_14);
x_22 = lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__3(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_21, x_15, x_16, x_17, x_18, x_19);
return x_22;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__4(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, uint8_t x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18) {
_start:
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_20 = lean_box(x_13);
lean_inc(x_12);
lean_inc_ref(x_7);
x_21 = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__3___boxed), 20, 14);
lean_closure_set(x_21, 0, x_1);
lean_closure_set(x_21, 1, x_2);
lean_closure_set(x_21, 2, x_3);
lean_closure_set(x_21, 3, x_4);
lean_closure_set(x_21, 4, x_5);
lean_closure_set(x_21, 5, x_6);
lean_closure_set(x_21, 6, x_7);
lean_closure_set(x_21, 7, x_8);
lean_closure_set(x_21, 8, x_9);
lean_closure_set(x_21, 9, x_10);
lean_closure_set(x_21, 10, x_11);
lean_closure_set(x_21, 11, x_14);
lean_closure_set(x_21, 12, x_12);
lean_closure_set(x_21, 13, x_20);
x_22 = lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0___redArg(x_12, x_13, x_7, x_21, x_15, x_16, x_17, x_18);
return x_22;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__4___boxed(lean_object** _args) {
lean_object* x_1 = _args[0];
lean_object* x_2 = _args[1];
lean_object* x_3 = _args[2];
lean_object* x_4 = _args[3];
lean_object* x_5 = _args[4];
lean_object* x_6 = _args[5];
lean_object* x_7 = _args[6];
lean_object* x_8 = _args[7];
lean_object* x_9 = _args[8];
lean_object* x_10 = _args[9];
lean_object* x_11 = _args[10];
lean_object* x_12 = _args[11];
lean_object* x_13 = _args[12];
lean_object* x_14 = _args[13];
lean_object* x_15 = _args[14];
lean_object* x_16 = _args[15];
lean_object* x_17 = _args[16];
lean_object* x_18 = _args[17];
lean_object* x_19 = _args[18];
_start:
{
uint8_t x_20; lean_object* x_21; 
x_20 = lean_unbox(x_13);
x_21 = lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__4(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_20, x_14, x_15, x_16, x_17, x_18);
return x_21;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_14; uint8_t x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; uint8_t x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; 
x_14 = lean_box(0);
x_15 = 0;
x_16 = lp_mathlib_ExistsAndEq_mkNestedExists___closed__0;
x_17 = lp_mathlib_ExistsAndEq_mkNestedExists___closed__1;
x_18 = lean_box(0);
lean_inc(x_1);
x_19 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_19, 0, x_1);
lean_ctor_set(x_19, 1, x_18);
lean_inc_ref(x_19);
x_20 = l_Lean_Expr_const___override(x_17, x_19);
lean_inc_ref(x_2);
x_21 = l_Lean_Expr_app___override(x_20, x_2);
x_22 = lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__1;
x_23 = lean_box(x_15);
lean_inc_ref(x_2);
lean_inc_ref(x_3);
x_24 = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__4___boxed), 19, 13);
lean_closure_set(x_24, 0, x_18);
lean_closure_set(x_24, 1, x_3);
lean_closure_set(x_24, 2, x_6);
lean_closure_set(x_24, 3, x_7);
lean_closure_set(x_24, 4, x_8);
lean_closure_set(x_24, 5, x_1);
lean_closure_set(x_24, 6, x_2);
lean_closure_set(x_24, 7, x_5);
lean_closure_set(x_24, 8, x_4);
lean_closure_set(x_24, 9, x_16);
lean_closure_set(x_24, 10, x_19);
lean_closure_set(x_24, 11, x_14);
lean_closure_set(x_24, 12, x_23);
x_25 = lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__4;
x_26 = 0;
x_27 = l_Lean_Expr_betaRev(x_3, x_25, x_26, x_26);
x_28 = l_Lean_Expr_lam___override(x_22, x_2, x_27, x_15);
x_29 = l_Lean_Expr_app___override(x_21, x_28);
x_30 = lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_withNestedExistsElim_spec__0___redArg(x_14, x_15, x_29, x_24, x_9, x_10, x_11, x_12);
return x_30;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_ExistsAndEq_mkBeforeToAfter(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00ExistsAndEq_existsAndEq_spec__3___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lean_apply_10(x_1, x_5, x_6, x_2, x_3, x_4, x_7, x_8, x_9, x_10, lean_box(0));
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00ExistsAndEq_existsAndEq_spec__3___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00ExistsAndEq_existsAndEq_spec__3___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00ExistsAndEq_existsAndEq_spec__3___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, uint8_t x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_13; uint8_t x_14; uint8_t x_15; lean_object* x_16; lean_object* x_17; 
x_13 = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00ExistsAndEq_existsAndEq_spec__3___redArg___lam__0___boxed), 11, 4);
lean_closure_set(x_13, 0, x_3);
lean_closure_set(x_13, 1, x_5);
lean_closure_set(x_13, 2, x_6);
lean_closure_set(x_13, 3, x_7);
x_14 = 1;
x_15 = 0;
x_16 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_16, 0, x_2);
x_17 = l___private_Lean_Meta_Basic_0__Lean_Meta_lambdaTelescopeImp(lean_box(0), x_1, x_14, x_15, x_14, x_15, x_16, x_13, x_4, x_8, x_9, x_10, x_11);
lean_dec_ref(x_16);
if (lean_obj_tag(x_17) == 0)
{
return x_17;
}
else
{
uint8_t x_18; 
x_18 = !lean_is_exclusive(x_17);
if (x_18 == 0)
{
return x_17;
}
else
{
lean_object* x_19; lean_object* x_20; 
x_19 = lean_ctor_get(x_17, 0);
lean_inc(x_19);
lean_dec(x_17);
x_20 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_20, 0, x_19);
return x_20;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00ExistsAndEq_existsAndEq_spec__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, uint8_t x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00ExistsAndEq_existsAndEq_spec__3___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx_x27___at___00ExistsAndEq_existsAndEq_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
uint8_t x_11; 
x_11 = !lean_is_exclusive(x_6);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; 
x_12 = lean_ctor_get(x_6, 2);
lean_dec(x_12);
lean_ctor_set(x_6, 2, x_1);
x_13 = lean_apply_8(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, lean_box(0));
if (lean_obj_tag(x_13) == 0)
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
x_16 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_16, 0, x_15);
return x_16;
}
}
else
{
return x_13;
}
}
else
{
lean_object* x_17; uint8_t x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; uint8_t x_24; uint8_t x_25; lean_object* x_26; lean_object* x_27; 
x_17 = lean_ctor_get(x_6, 0);
x_18 = lean_ctor_get_uint8(x_6, sizeof(void*)*7);
x_19 = lean_ctor_get(x_6, 1);
x_20 = lean_ctor_get(x_6, 3);
x_21 = lean_ctor_get(x_6, 4);
x_22 = lean_ctor_get(x_6, 5);
x_23 = lean_ctor_get(x_6, 6);
x_24 = lean_ctor_get_uint8(x_6, sizeof(void*)*7 + 1);
x_25 = lean_ctor_get_uint8(x_6, sizeof(void*)*7 + 2);
lean_inc(x_23);
lean_inc(x_22);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_inc(x_17);
lean_dec(x_6);
x_26 = lean_alloc_ctor(0, 7, 3);
lean_ctor_set(x_26, 0, x_17);
lean_ctor_set(x_26, 1, x_19);
lean_ctor_set(x_26, 2, x_1);
lean_ctor_set(x_26, 3, x_20);
lean_ctor_set(x_26, 4, x_21);
lean_ctor_set(x_26, 5, x_22);
lean_ctor_set(x_26, 6, x_23);
lean_ctor_set_uint8(x_26, sizeof(void*)*7, x_18);
lean_ctor_set_uint8(x_26, sizeof(void*)*7 + 1, x_24);
lean_ctor_set_uint8(x_26, sizeof(void*)*7 + 2, x_25);
x_27 = lean_apply_8(x_2, x_3, x_4, x_5, x_26, x_7, x_8, x_9, lean_box(0));
if (lean_obj_tag(x_27) == 0)
{
lean_object* x_28; lean_object* x_29; lean_object* x_30; 
x_28 = lean_ctor_get(x_27, 0);
lean_inc(x_28);
if (lean_is_exclusive(x_27)) {
 lean_ctor_release(x_27, 0);
 x_29 = x_27;
} else {
 lean_dec_ref(x_27);
 x_29 = lean_box(0);
}
if (lean_is_scalar(x_29)) {
 x_30 = lean_alloc_ctor(0, 1, 0);
} else {
 x_30 = x_29;
}
lean_ctor_set(x_30, 0, x_28);
return x_30;
}
else
{
return x_27;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx_x27___at___00ExistsAndEq_existsAndEq_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Lean_Meta_withLCtx_x27___at___00ExistsAndEq_existsAndEq_spec__0___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00ExistsAndEq_existsAndEq_spec__2___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; 
x_10 = lean_apply_8(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, lean_box(0));
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00ExistsAndEq_existsAndEq_spec__2___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00ExistsAndEq_existsAndEq_spec__2___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00ExistsAndEq_existsAndEq_spec__2___redArg(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; lean_object* x_12; 
x_11 = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00ExistsAndEq_existsAndEq_spec__2___redArg___lam__0___boxed), 9, 4);
lean_closure_set(x_11, 0, x_1);
lean_closure_set(x_11, 1, x_3);
lean_closure_set(x_11, 2, x_4);
lean_closure_set(x_11, 3, x_5);
x_12 = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), x_2, x_11, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_12) == 0)
{
return x_12;
}
else
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
x_15 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_15, 0, x_14);
return x_15;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00ExistsAndEq_existsAndEq_spec__2(lean_object* x_1, lean_object* x_2, uint8_t x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00ExistsAndEq_existsAndEq_spec__2___redArg(x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
static lean_object* _init_lp_mathlib_panic___at___00ExistsAndEq_existsAndEq_spec__1___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_Meta_Simp_instInhabitedSimpM___lam__0___boxed), 8, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00ExistsAndEq_existsAndEq_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_10 = lp_mathlib_panic___at___00ExistsAndEq_existsAndEq_spec__1___closed__0;
x_11 = lean_panic_fn(x_10, x_1);
x_12 = lean_apply_8(x_11, x_2, x_3, x_4, x_5, x_6, x_7, x_8, lean_box(0));
return x_12;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_existsAndEq___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_box(0);
x_2 = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("propext", 7, 7);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__0;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__1;
x_3 = l_Lean_Expr_const___override(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Iff", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__12;
x_2 = lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__3;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__4;
x_3 = l_Lean_Expr_const___override(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEq___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, uint8_t x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17) {
_start:
{
lean_object* x_19; 
lean_inc_ref(x_14);
lean_inc_ref(x_2);
lean_inc(x_1);
x_19 = lp_mathlib_ExistsAndEq_mkNestedExists(x_1, x_2, x_14, x_15, x_16, x_17);
if (lean_obj_tag(x_19) == 0)
{
lean_object* x_20; lean_object* x_21; 
x_20 = lean_ctor_get(x_19, 0);
lean_inc(x_20);
lean_dec_ref(x_19);
lean_inc(x_17);
lean_inc_ref(x_16);
lean_inc(x_15);
lean_inc_ref(x_14);
lean_inc(x_7);
lean_inc(x_1);
lean_inc_ref(x_2);
lean_inc_ref(x_6);
lean_inc(x_20);
lean_inc_ref(x_5);
lean_inc_ref(x_4);
lean_inc(x_3);
x_21 = lp_mathlib_ExistsAndEq_mkBeforeToAfter(x_3, x_4, x_5, x_20, x_6, x_2, x_1, x_7, x_14, x_15, x_16, x_17);
if (lean_obj_tag(x_21) == 0)
{
lean_object* x_22; lean_object* x_23; 
x_22 = lean_ctor_get(x_21, 0);
lean_inc(x_22);
lean_dec_ref(x_21);
lean_inc(x_20);
lean_inc_ref(x_5);
lean_inc_ref(x_4);
lean_inc(x_3);
x_23 = lp_mathlib_ExistsAndEq_mkAfterToBefore(x_3, x_4, x_5, x_20, x_6, x_2, x_1, x_7, x_14, x_15, x_16, x_17);
if (lean_obj_tag(x_23) == 0)
{
uint8_t x_24; 
x_24 = !lean_is_exclusive(x_23);
if (x_24 == 0)
{
lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; uint8_t x_34; lean_object* x_35; uint8_t x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; 
x_25 = lean_ctor_get(x_23, 0);
x_26 = lean_box(0);
x_27 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_27, 0, x_3);
lean_ctor_set(x_27, 1, x_26);
x_28 = l_Lean_Expr_const___override(x_8, x_27);
lean_inc_ref(x_4);
x_29 = l_Lean_Expr_app___override(x_28, x_4);
x_30 = lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__1;
x_31 = l_Lean_Expr_bvar___override(x_9);
x_32 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_32, 0, x_31);
lean_ctor_set(x_32, 1, x_26);
x_33 = lean_array_mk(x_32);
x_34 = 0;
x_35 = l_Lean_Expr_betaRev(x_5, x_33, x_34, x_34);
lean_dec_ref(x_33);
x_36 = 0;
x_37 = l_Lean_Expr_lam___override(x_30, x_4, x_35, x_36);
x_38 = l_Lean_Expr_app___override(x_29, x_37);
x_39 = lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__2;
lean_inc_ref(x_38);
x_40 = l_Lean_Expr_app___override(x_39, x_38);
lean_inc(x_20);
x_41 = l_Lean_Expr_app___override(x_40, x_20);
x_42 = lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__5;
x_43 = l_Lean_Expr_app___override(x_42, x_38);
lean_inc(x_20);
x_44 = l_Lean_Expr_app___override(x_43, x_20);
x_45 = l_Lean_Expr_app___override(x_44, x_22);
x_46 = l_Lean_Expr_app___override(x_45, x_25);
x_47 = l_Lean_Expr_app___override(x_41, x_46);
x_48 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_48, 0, x_47);
x_49 = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(x_49, 0, x_20);
lean_ctor_set(x_49, 1, x_48);
lean_ctor_set_uint8(x_49, sizeof(void*)*2, x_10);
x_50 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_50, 0, x_49);
lean_ctor_set(x_23, 0, x_50);
return x_23;
}
else
{
lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; uint8_t x_60; lean_object* x_61; uint8_t x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; 
x_51 = lean_ctor_get(x_23, 0);
lean_inc(x_51);
lean_dec(x_23);
x_52 = lean_box(0);
x_53 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_53, 0, x_3);
lean_ctor_set(x_53, 1, x_52);
x_54 = l_Lean_Expr_const___override(x_8, x_53);
lean_inc_ref(x_4);
x_55 = l_Lean_Expr_app___override(x_54, x_4);
x_56 = lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__1;
x_57 = l_Lean_Expr_bvar___override(x_9);
x_58 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_58, 0, x_57);
lean_ctor_set(x_58, 1, x_52);
x_59 = lean_array_mk(x_58);
x_60 = 0;
x_61 = l_Lean_Expr_betaRev(x_5, x_59, x_60, x_60);
lean_dec_ref(x_59);
x_62 = 0;
x_63 = l_Lean_Expr_lam___override(x_56, x_4, x_61, x_62);
x_64 = l_Lean_Expr_app___override(x_55, x_63);
x_65 = lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__2;
lean_inc_ref(x_64);
x_66 = l_Lean_Expr_app___override(x_65, x_64);
lean_inc(x_20);
x_67 = l_Lean_Expr_app___override(x_66, x_20);
x_68 = lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__5;
x_69 = l_Lean_Expr_app___override(x_68, x_64);
lean_inc(x_20);
x_70 = l_Lean_Expr_app___override(x_69, x_20);
x_71 = l_Lean_Expr_app___override(x_70, x_22);
x_72 = l_Lean_Expr_app___override(x_71, x_51);
x_73 = l_Lean_Expr_app___override(x_67, x_72);
x_74 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_74, 0, x_73);
x_75 = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(x_75, 0, x_20);
lean_ctor_set(x_75, 1, x_74);
lean_ctor_set_uint8(x_75, sizeof(void*)*2, x_10);
x_76 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_76, 0, x_75);
x_77 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_77, 0, x_76);
return x_77;
}
}
else
{
uint8_t x_78; 
lean_dec(x_22);
lean_dec(x_20);
lean_dec(x_9);
lean_dec(x_8);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
x_78 = !lean_is_exclusive(x_23);
if (x_78 == 0)
{
return x_23;
}
else
{
lean_object* x_79; lean_object* x_80; 
x_79 = lean_ctor_get(x_23, 0);
lean_inc(x_79);
lean_dec(x_23);
x_80 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_80, 0, x_79);
return x_80;
}
}
}
else
{
uint8_t x_81; 
lean_dec(x_20);
lean_dec(x_17);
lean_dec_ref(x_16);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_9);
lean_dec(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_81 = !lean_is_exclusive(x_21);
if (x_81 == 0)
{
return x_21;
}
else
{
lean_object* x_82; lean_object* x_83; 
x_82 = lean_ctor_get(x_21, 0);
lean_inc(x_82);
lean_dec(x_21);
x_83 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_83, 0, x_82);
return x_83;
}
}
}
else
{
uint8_t x_84; 
lean_dec(x_17);
lean_dec_ref(x_16);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_9);
lean_dec(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_84 = !lean_is_exclusive(x_19);
if (x_84 == 0)
{
return x_19;
}
else
{
lean_object* x_85; lean_object* x_86; 
x_85 = lean_ctor_get(x_19, 0);
lean_inc(x_85);
lean_dec(x_19);
x_86 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_86, 0, x_85);
return x_86;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEq___lam__0___boxed(lean_object** _args) {
lean_object* x_1 = _args[0];
lean_object* x_2 = _args[1];
lean_object* x_3 = _args[2];
lean_object* x_4 = _args[3];
lean_object* x_5 = _args[4];
lean_object* x_6 = _args[5];
lean_object* x_7 = _args[6];
lean_object* x_8 = _args[7];
lean_object* x_9 = _args[8];
lean_object* x_10 = _args[9];
lean_object* x_11 = _args[10];
lean_object* x_12 = _args[11];
lean_object* x_13 = _args[12];
lean_object* x_14 = _args[13];
lean_object* x_15 = _args[14];
lean_object* x_16 = _args[15];
lean_object* x_17 = _args[16];
lean_object* x_18 = _args[17];
_start:
{
uint8_t x_19; lean_object* x_20; 
x_19 = lean_unbox(x_10);
x_20 = lp_mathlib_ExistsAndEq_existsAndEq___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_19, x_11, x_12, x_13, x_14, x_15, x_16, x_17);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
return x_20;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_existsAndEq___lam__1___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("ExistsAndEq.existsAndEq", 23, 23);
return x_1;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_existsAndEq___lam__1___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_1 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__2;
x_2 = lean_unsigned_to_nat(39u);
x_3 = lean_unsigned_to_nat(420u);
x_4 = lp_mathlib_ExistsAndEq_existsAndEq___lam__1___closed__0;
x_5 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__0;
x_6 = l_mkPanicMessageWithDecl(x_5, x_4, x_3, x_2, x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEq___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, uint8_t x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
if (lean_obj_tag(x_1) == 1)
{
uint8_t x_17; 
x_17 = !lean_is_exclusive(x_1);
if (x_17 == 0)
{
lean_object* x_18; lean_object* x_19; uint8_t x_20; 
x_18 = lean_ctor_get(x_1, 0);
x_19 = lean_array_get_size(x_2);
x_20 = lean_nat_dec_lt(x_3, x_19);
if (x_20 == 0)
{
lean_object* x_21; 
lean_dec(x_18);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
x_21 = lp_mathlib_ExistsAndEq_existsAndEq___closed__0;
lean_ctor_set_tag(x_1, 0);
lean_ctor_set(x_1, 0, x_21);
return x_1;
}
else
{
lean_object* x_22; lean_object* x_23; 
lean_free_object(x_1);
x_22 = lean_array_fget_borrowed(x_2, x_3);
lean_inc_ref(x_4);
x_23 = lp_mathlib_ExistsAndEq_findEqPath___redArg(x_22, x_4, x_13);
if (lean_obj_tag(x_23) == 0)
{
uint8_t x_24; 
x_24 = !lean_is_exclusive(x_23);
if (x_24 == 0)
{
lean_object* x_25; 
x_25 = lean_ctor_get(x_23, 0);
if (lean_obj_tag(x_25) == 1)
{
lean_object* x_26; lean_object* x_27; 
lean_free_object(x_23);
x_26 = lean_ctor_get(x_25, 0);
lean_inc(x_26);
lean_dec_ref(x_25);
lean_inc(x_15);
lean_inc_ref(x_14);
lean_inc(x_13);
lean_inc_ref(x_12);
lean_inc(x_26);
lean_inc(x_22);
lean_inc(x_18);
x_27 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg(x_18, x_22, x_4, x_26, x_12, x_13, x_14, x_15);
if (lean_obj_tag(x_27) == 0)
{
lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; 
x_28 = lean_ctor_get(x_27, 0);
lean_inc(x_28);
lean_dec_ref(x_27);
x_29 = lean_ctor_get(x_28, 1);
lean_inc(x_29);
x_30 = lean_ctor_get(x_29, 1);
lean_inc(x_30);
x_31 = lean_ctor_get(x_28, 0);
lean_inc(x_31);
lean_dec(x_28);
x_32 = lean_ctor_get(x_29, 0);
lean_inc(x_32);
lean_dec(x_29);
x_33 = lean_ctor_get(x_30, 0);
lean_inc(x_33);
x_34 = lean_ctor_get(x_30, 1);
lean_inc(x_34);
lean_dec(x_30);
lean_inc(x_22);
x_35 = l_Lean_Expr_replaceFVar(x_33, x_22, x_34);
lean_dec(x_33);
x_36 = lean_box(x_8);
x_37 = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_existsAndEq___lam__0___boxed), 18, 10);
lean_closure_set(x_37, 0, x_31);
lean_closure_set(x_37, 1, x_35);
lean_closure_set(x_37, 2, x_18);
lean_closure_set(x_37, 3, x_5);
lean_closure_set(x_37, 4, x_6);
lean_closure_set(x_37, 5, x_34);
lean_closure_set(x_37, 6, x_26);
lean_closure_set(x_37, 7, x_7);
lean_closure_set(x_37, 8, x_3);
lean_closure_set(x_37, 9, x_36);
x_38 = lp_mathlib_Lean_Meta_withLCtx_x27___at___00ExistsAndEq_existsAndEq_spec__0___redArg(x_32, x_37, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
return x_38;
}
else
{
uint8_t x_39; 
lean_dec(x_26);
lean_dec(x_18);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec(x_3);
x_39 = !lean_is_exclusive(x_27);
if (x_39 == 0)
{
return x_27;
}
else
{
lean_object* x_40; lean_object* x_41; 
x_40 = lean_ctor_get(x_27, 0);
lean_inc(x_40);
lean_dec(x_27);
x_41 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_41, 0, x_40);
return x_41;
}
}
}
else
{
lean_object* x_42; 
lean_dec(x_25);
lean_dec(x_18);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
x_42 = lp_mathlib_ExistsAndEq_existsAndEq___closed__0;
lean_ctor_set(x_23, 0, x_42);
return x_23;
}
}
else
{
lean_object* x_43; 
x_43 = lean_ctor_get(x_23, 0);
lean_inc(x_43);
lean_dec(x_23);
if (lean_obj_tag(x_43) == 1)
{
lean_object* x_44; lean_object* x_45; 
x_44 = lean_ctor_get(x_43, 0);
lean_inc(x_44);
lean_dec_ref(x_43);
lean_inc(x_15);
lean_inc_ref(x_14);
lean_inc(x_13);
lean_inc_ref(x_12);
lean_inc(x_44);
lean_inc(x_22);
lean_inc(x_18);
x_45 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg(x_18, x_22, x_4, x_44, x_12, x_13, x_14, x_15);
if (lean_obj_tag(x_45) == 0)
{
lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; 
x_46 = lean_ctor_get(x_45, 0);
lean_inc(x_46);
lean_dec_ref(x_45);
x_47 = lean_ctor_get(x_46, 1);
lean_inc(x_47);
x_48 = lean_ctor_get(x_47, 1);
lean_inc(x_48);
x_49 = lean_ctor_get(x_46, 0);
lean_inc(x_49);
lean_dec(x_46);
x_50 = lean_ctor_get(x_47, 0);
lean_inc(x_50);
lean_dec(x_47);
x_51 = lean_ctor_get(x_48, 0);
lean_inc(x_51);
x_52 = lean_ctor_get(x_48, 1);
lean_inc(x_52);
lean_dec(x_48);
lean_inc(x_22);
x_53 = l_Lean_Expr_replaceFVar(x_51, x_22, x_52);
lean_dec(x_51);
x_54 = lean_box(x_8);
x_55 = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_existsAndEq___lam__0___boxed), 18, 10);
lean_closure_set(x_55, 0, x_49);
lean_closure_set(x_55, 1, x_53);
lean_closure_set(x_55, 2, x_18);
lean_closure_set(x_55, 3, x_5);
lean_closure_set(x_55, 4, x_6);
lean_closure_set(x_55, 5, x_52);
lean_closure_set(x_55, 6, x_44);
lean_closure_set(x_55, 7, x_7);
lean_closure_set(x_55, 8, x_3);
lean_closure_set(x_55, 9, x_54);
x_56 = lp_mathlib_Lean_Meta_withLCtx_x27___at___00ExistsAndEq_existsAndEq_spec__0___redArg(x_50, x_55, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
return x_56;
}
else
{
lean_object* x_57; lean_object* x_58; lean_object* x_59; 
lean_dec(x_44);
lean_dec(x_18);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec(x_3);
x_57 = lean_ctor_get(x_45, 0);
lean_inc(x_57);
if (lean_is_exclusive(x_45)) {
 lean_ctor_release(x_45, 0);
 x_58 = x_45;
} else {
 lean_dec_ref(x_45);
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
else
{
lean_object* x_60; lean_object* x_61; 
lean_dec(x_43);
lean_dec(x_18);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
x_60 = lp_mathlib_ExistsAndEq_existsAndEq___closed__0;
x_61 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_61, 0, x_60);
return x_61;
}
}
}
else
{
uint8_t x_62; 
lean_dec(x_18);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
x_62 = !lean_is_exclusive(x_23);
if (x_62 == 0)
{
return x_23;
}
else
{
lean_object* x_63; lean_object* x_64; 
x_63 = lean_ctor_get(x_23, 0);
lean_inc(x_63);
lean_dec(x_23);
x_64 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_64, 0, x_63);
return x_64;
}
}
}
}
else
{
lean_object* x_65; lean_object* x_66; uint8_t x_67; 
x_65 = lean_ctor_get(x_1, 0);
lean_inc(x_65);
lean_dec(x_1);
x_66 = lean_array_get_size(x_2);
x_67 = lean_nat_dec_lt(x_3, x_66);
if (x_67 == 0)
{
lean_object* x_68; lean_object* x_69; 
lean_dec(x_65);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
x_68 = lp_mathlib_ExistsAndEq_existsAndEq___closed__0;
x_69 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_69, 0, x_68);
return x_69;
}
else
{
lean_object* x_70; lean_object* x_71; 
x_70 = lean_array_fget_borrowed(x_2, x_3);
lean_inc_ref(x_4);
x_71 = lp_mathlib_ExistsAndEq_findEqPath___redArg(x_70, x_4, x_13);
if (lean_obj_tag(x_71) == 0)
{
lean_object* x_72; lean_object* x_73; 
x_72 = lean_ctor_get(x_71, 0);
lean_inc(x_72);
if (lean_is_exclusive(x_71)) {
 lean_ctor_release(x_71, 0);
 x_73 = x_71;
} else {
 lean_dec_ref(x_71);
 x_73 = lean_box(0);
}
if (lean_obj_tag(x_72) == 1)
{
lean_object* x_74; lean_object* x_75; 
lean_dec(x_73);
x_74 = lean_ctor_get(x_72, 0);
lean_inc(x_74);
lean_dec_ref(x_72);
lean_inc(x_15);
lean_inc_ref(x_14);
lean_inc(x_13);
lean_inc_ref(x_12);
lean_inc(x_74);
lean_inc(x_70);
lean_inc(x_65);
x_75 = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg(x_65, x_70, x_4, x_74, x_12, x_13, x_14, x_15);
if (lean_obj_tag(x_75) == 0)
{
lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; 
x_76 = lean_ctor_get(x_75, 0);
lean_inc(x_76);
lean_dec_ref(x_75);
x_77 = lean_ctor_get(x_76, 1);
lean_inc(x_77);
x_78 = lean_ctor_get(x_77, 1);
lean_inc(x_78);
x_79 = lean_ctor_get(x_76, 0);
lean_inc(x_79);
lean_dec(x_76);
x_80 = lean_ctor_get(x_77, 0);
lean_inc(x_80);
lean_dec(x_77);
x_81 = lean_ctor_get(x_78, 0);
lean_inc(x_81);
x_82 = lean_ctor_get(x_78, 1);
lean_inc(x_82);
lean_dec(x_78);
lean_inc(x_70);
x_83 = l_Lean_Expr_replaceFVar(x_81, x_70, x_82);
lean_dec(x_81);
x_84 = lean_box(x_8);
x_85 = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_existsAndEq___lam__0___boxed), 18, 10);
lean_closure_set(x_85, 0, x_79);
lean_closure_set(x_85, 1, x_83);
lean_closure_set(x_85, 2, x_65);
lean_closure_set(x_85, 3, x_5);
lean_closure_set(x_85, 4, x_6);
lean_closure_set(x_85, 5, x_82);
lean_closure_set(x_85, 6, x_74);
lean_closure_set(x_85, 7, x_7);
lean_closure_set(x_85, 8, x_3);
lean_closure_set(x_85, 9, x_84);
x_86 = lp_mathlib_Lean_Meta_withLCtx_x27___at___00ExistsAndEq_existsAndEq_spec__0___redArg(x_80, x_85, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
return x_86;
}
else
{
lean_object* x_87; lean_object* x_88; lean_object* x_89; 
lean_dec(x_74);
lean_dec(x_65);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec(x_3);
x_87 = lean_ctor_get(x_75, 0);
lean_inc(x_87);
if (lean_is_exclusive(x_75)) {
 lean_ctor_release(x_75, 0);
 x_88 = x_75;
} else {
 lean_dec_ref(x_75);
 x_88 = lean_box(0);
}
if (lean_is_scalar(x_88)) {
 x_89 = lean_alloc_ctor(1, 1, 0);
} else {
 x_89 = x_88;
}
lean_ctor_set(x_89, 0, x_87);
return x_89;
}
}
else
{
lean_object* x_90; lean_object* x_91; 
lean_dec(x_72);
lean_dec(x_65);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
x_90 = lp_mathlib_ExistsAndEq_existsAndEq___closed__0;
if (lean_is_scalar(x_73)) {
 x_91 = lean_alloc_ctor(0, 1, 0);
} else {
 x_91 = x_73;
}
lean_ctor_set(x_91, 0, x_90);
return x_91;
}
}
else
{
lean_object* x_92; lean_object* x_93; lean_object* x_94; 
lean_dec(x_65);
lean_dec(x_15);
lean_dec_ref(x_14);
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
x_92 = lean_ctor_get(x_71, 0);
lean_inc(x_92);
if (lean_is_exclusive(x_71)) {
 lean_ctor_release(x_71, 0);
 x_93 = x_71;
} else {
 lean_dec_ref(x_71);
 x_93 = lean_box(0);
}
if (lean_is_scalar(x_93)) {
 x_94 = lean_alloc_ctor(1, 1, 0);
} else {
 x_94 = x_93;
}
lean_ctor_set(x_94, 0, x_92);
return x_94;
}
}
}
}
else
{
lean_object* x_95; lean_object* x_96; 
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec(x_1);
x_95 = lp_mathlib_ExistsAndEq_existsAndEq___lam__1___closed__1;
x_96 = lp_mathlib_panic___at___00ExistsAndEq_existsAndEq_spec__1(x_95, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
return x_96;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEq___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16) {
_start:
{
uint8_t x_17; lean_object* x_18; 
x_17 = lean_unbox(x_8);
x_18 = lp_mathlib_ExistsAndEq_existsAndEq___lam__1(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_17, x_9, x_10, x_11, x_12, x_13, x_14, x_15);
lean_dec_ref(x_2);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEq___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, uint8_t x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; uint8_t x_21; lean_object* x_22; 
x_16 = l_Lean_Expr_constLevels_x21(x_1);
x_17 = lean_unsigned_to_nat(0u);
x_18 = l_List_get_x3fInternal___redArg(x_16, x_17);
lean_dec(x_16);
x_19 = lean_box(x_5);
x_20 = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_existsAndEq___lam__1___boxed), 16, 8);
lean_closure_set(x_20, 0, x_18);
lean_closure_set(x_20, 1, x_6);
lean_closure_set(x_20, 2, x_17);
lean_closure_set(x_20, 3, x_7);
lean_closure_set(x_20, 4, x_2);
lean_closure_set(x_20, 5, x_3);
lean_closure_set(x_20, 6, x_4);
lean_closure_set(x_20, 7, x_19);
x_21 = 0;
x_22 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00ExistsAndEq_existsAndEq_spec__2___redArg(x_20, x_21, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
return x_22;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEq___lam__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
uint8_t x_16; lean_object* x_17; 
x_16 = lean_unbox(x_5);
x_17 = lp_mathlib_ExistsAndEq_existsAndEq___lam__2(x_1, x_2, x_3, x_4, x_16, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
lean_dec_ref(x_1);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEq(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; lean_object* x_14; uint8_t x_15; 
x_14 = l_Lean_Expr_cleanupAnnotations(x_1);
x_15 = l_Lean_Expr_isApp(x_14);
if (x_15 == 0)
{
lean_dec_ref(x_14);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
x_10 = lean_box(0);
goto block_13;
}
else
{
lean_object* x_16; uint8_t x_17; 
lean_inc_ref(x_14);
x_16 = l_Lean_Expr_appFnCleanup___redArg(x_14);
x_17 = l_Lean_Expr_isApp(x_16);
if (x_17 == 0)
{
lean_dec_ref(x_16);
lean_dec_ref(x_14);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
x_10 = lean_box(0);
goto block_13;
}
else
{
lean_object* x_18; lean_object* x_19; uint8_t x_20; 
lean_inc_ref(x_16);
x_18 = l_Lean_Expr_appFnCleanup___redArg(x_16);
x_19 = lp_mathlib_ExistsAndEq_mkNestedExists___closed__1;
x_20 = l_Lean_Expr_isConstOf(x_18, x_19);
if (x_20 == 0)
{
lean_dec_ref(x_18);
lean_dec_ref(x_16);
lean_dec_ref(x_14);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
x_10 = lean_box(0);
goto block_13;
}
else
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; uint8_t x_26; lean_object* x_27; 
x_21 = lean_ctor_get(x_14, 1);
lean_inc_ref(x_21);
lean_dec_ref(x_14);
x_22 = lean_ctor_get(x_16, 1);
lean_inc_ref(x_22);
lean_dec_ref(x_16);
x_23 = lean_box(x_20);
lean_inc_ref(x_21);
x_24 = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_existsAndEq___lam__2___boxed), 15, 5);
lean_closure_set(x_24, 0, x_18);
lean_closure_set(x_24, 1, x_22);
lean_closure_set(x_24, 2, x_21);
lean_closure_set(x_24, 3, x_19);
lean_closure_set(x_24, 4, x_23);
x_25 = lean_unsigned_to_nat(1u);
x_26 = 0;
x_27 = lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00ExistsAndEq_existsAndEq_spec__3___redArg(x_21, x_25, x_24, x_26, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
return x_27;
}
}
}
block_13:
{
lean_object* x_11; lean_object* x_12; 
x_11 = lp_mathlib_ExistsAndEq_existsAndEq___closed__0;
x_12 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_12, 0, x_11);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00ExistsAndEq_existsAndEq_spec__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
uint8_t x_14; lean_object* x_15; 
x_14 = lean_unbox(x_5);
x_15 = lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00ExistsAndEq_existsAndEq_spec__3(x_1, x_2, x_3, x_4, x_14, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx_x27___at___00ExistsAndEq_existsAndEq_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Lean_Meta_withLCtx_x27___at___00ExistsAndEq_existsAndEq_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00ExistsAndEq_existsAndEq_spec__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
uint8_t x_12; lean_object* x_13; 
x_12 = lean_unbox(x_3);
x_13 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00ExistsAndEq_existsAndEq_spec__2(x_1, x_2, x_12, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00ExistsAndEq_existsAndEq_spec__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_panic___at___00ExistsAndEq_existsAndEq_spec__1(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00ExistsAndEq_existsAndEq_spec__2___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
uint8_t x_11; lean_object* x_12; 
x_11 = lean_unbox(x_2);
x_12 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00ExistsAndEq_existsAndEq_spec__2___redArg(x_1, x_11, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx_x27___at___00ExistsAndEq_existsAndEq_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_Lean_Meta_withLCtx_x27___at___00ExistsAndEq_existsAndEq_spec__0___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00ExistsAndEq_existsAndEq_spec__3___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
uint8_t x_13; lean_object* x_14; 
x_13 = lean_unbox(x_4);
x_14 = lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00ExistsAndEq_existsAndEq_spec__3___redArg(x_1, x_2, x_3, x_13, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEq___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_ExistsAndEq_existsAndEq(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
return x_10;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Qq_Qq(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Simproc_ExistsAndEq(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Qq_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_ExistsAndEq_instBEqGoTo___closed__0 = _init_lp_mathlib_ExistsAndEq_instBEqGoTo___closed__0();
lean_mark_persistent(lp_mathlib_ExistsAndEq_instBEqGoTo___closed__0);
lp_mathlib_ExistsAndEq_instBEqGoTo = _init_lp_mathlib_ExistsAndEq_instBEqGoTo();
lean_mark_persistent(lp_mathlib_ExistsAndEq_instBEqGoTo);
lp_mathlib_ExistsAndEq_instInhabitedGoTo_default = _init_lp_mathlib_ExistsAndEq_instInhabitedGoTo_default();
lp_mathlib_ExistsAndEq_instInhabitedGoTo = _init_lp_mathlib_ExistsAndEq_instInhabitedGoTo();
lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__0 = _init_lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__0();
lean_mark_persistent(lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__0);
lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__1 = _init_lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__1();
lean_mark_persistent(lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__1);
lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__2 = _init_lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__2();
lean_mark_persistent(lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__2);
lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__3 = _init_lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__3();
lean_mark_persistent(lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__3);
lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__4 = _init_lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__4();
lean_mark_persistent(lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__4);
lp_mathlib_ExistsAndEq_instInhabitedVarQ = _init_lp_mathlib_ExistsAndEq_instInhabitedVarQ();
lean_mark_persistent(lp_mathlib_ExistsAndEq_instInhabitedVarQ);
lp_mathlib_ExistsAndEq_instInhabitedHypQ = _init_lp_mathlib_ExistsAndEq_instInhabitedHypQ();
lean_mark_persistent(lp_mathlib_ExistsAndEq_instInhabitedHypQ);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg___closed__0 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg___closed__0();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg___closed__0);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg___closed__1 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg___closed__1();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable___redArg___closed__1);
lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___closed__0 = _init_lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___closed__0();
lean_mark_persistent(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___closed__0);
lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__0 = _init_lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__0();
lean_mark_persistent(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__0);
lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__1 = _init_lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__1();
lean_mark_persistent(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__1);
lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__2 = _init_lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__2();
lean_mark_persistent(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__2);
lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__3 = _init_lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__3();
lean_mark_persistent(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__3);
lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__4 = _init_lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__4();
lean_mark_persistent(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__4);
lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__5 = _init_lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__5();
lean_mark_persistent(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__5);
lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__6 = _init_lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__6();
lean_mark_persistent(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__6);
lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__7 = _init_lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__7();
lean_mark_persistent(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___lam__0___closed__7);
lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___closed__1 = _init_lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___closed__1();
lean_mark_persistent(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_assertUnreachable_spec__0_spec__0_spec__0___closed__1);
lp_mathlib_panic___at___00ExistsAndEq_mkNestedExists_spec__0___closed__0 = _init_lp_mathlib_panic___at___00ExistsAndEq_mkNestedExists_spec__0___closed__0();
lean_mark_persistent(lp_mathlib_panic___at___00ExistsAndEq_mkNestedExists_spec__0___closed__0);
lp_mathlib_ExistsAndEq_mkNestedExists___closed__0 = _init_lp_mathlib_ExistsAndEq_mkNestedExists___closed__0();
lean_mark_persistent(lp_mathlib_ExistsAndEq_mkNestedExists___closed__0);
lp_mathlib_ExistsAndEq_mkNestedExists___closed__1 = _init_lp_mathlib_ExistsAndEq_mkNestedExists___closed__1();
lean_mark_persistent(lp_mathlib_ExistsAndEq_mkNestedExists___closed__1);
lp_mathlib_ExistsAndEq_mkNestedExists___closed__2 = _init_lp_mathlib_ExistsAndEq_mkNestedExists___closed__2();
lean_mark_persistent(lp_mathlib_ExistsAndEq_mkNestedExists___closed__2);
lp_mathlib_ExistsAndEq_mkNestedExists___closed__3 = _init_lp_mathlib_ExistsAndEq_mkNestedExists___closed__3();
lean_mark_persistent(lp_mathlib_ExistsAndEq_mkNestedExists___closed__3);
lp_mathlib_ExistsAndEq_mkNestedExists___closed__4 = _init_lp_mathlib_ExistsAndEq_mkNestedExists___closed__4();
lean_mark_persistent(lp_mathlib_ExistsAndEq_mkNestedExists___closed__4);
lp_mathlib_ExistsAndEq_mkNestedExists___closed__5 = _init_lp_mathlib_ExistsAndEq_mkNestedExists___closed__5();
lean_mark_persistent(lp_mathlib_ExistsAndEq_mkNestedExists___closed__5);
lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__0 = _init_lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__0();
lean_mark_persistent(lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__0);
lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__1 = _init_lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__1();
lean_mark_persistent(lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__1);
lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__2 = _init_lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__2();
lean_mark_persistent(lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__2);
lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__3 = _init_lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__3();
lean_mark_persistent(lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__3);
lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__4 = _init_lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__4();
lean_mark_persistent(lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__4);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__1();
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__0 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__0();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__0);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__1 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__1();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__1);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__2 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__2();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__2);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__2 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__2();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__2);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__1 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__1();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__1);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__0 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__0();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__0);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__3 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__3();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__3);
lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3___closed__0 = _init_lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3___closed__0();
lean_mark_persistent(lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3___closed__0);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__4 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__4();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__4);
lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___closed__0 = _init_lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___closed__0();
lean_mark_persistent(lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___closed__0);
lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___closed__1 = _init_lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___closed__1();
lean_mark_persistent(lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___closed__1);
lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1___closed__0 = _init_lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1___closed__0();
lean_mark_persistent(lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1___closed__0);
lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1___closed__1 = _init_lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1___closed__1();
lean_mark_persistent(lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1___closed__1);
lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1___closed__2 = _init_lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1___closed__2();
lean_mark_persistent(lp_mathlib_ExistsAndEq_withNestedExistsElim___redArg___lam__1___closed__2);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__0 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__0();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__0);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__1 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__1();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__1);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__2 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__2();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__2);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__3 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__3();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__3);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__4 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__4();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__4);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__5 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__5();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__5);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__6 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__6();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__6);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__7 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__7();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__7);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__8 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__8();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__8);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__9 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__9();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__9);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__10 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__10();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__10);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__11 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__11();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__11);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__12 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__12();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__12);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__13 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__13();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__13);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__14 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__14();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__14);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__15 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__15();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__15);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__16 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__16();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkAfterToBefore_go___closed__16);
lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__0 = _init_lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__0();
lean_mark_persistent(lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__0);
lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__1 = _init_lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__1();
lean_mark_persistent(lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__1);
lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__2 = _init_lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__2();
lean_mark_persistent(lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__2);
lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__3 = _init_lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__3();
lean_mark_persistent(lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__3);
lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__4 = _init_lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__4();
lean_mark_persistent(lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__4);
lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__5 = _init_lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__5();
lean_mark_persistent(lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___closed__5);
lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__0 = _init_lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__0();
lean_mark_persistent(lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__0);
lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__1 = _init_lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__1();
lean_mark_persistent(lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__1);
lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__2 = _init_lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__2();
lean_mark_persistent(lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__2);
lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__3 = _init_lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__3();
lean_mark_persistent(lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__3);
lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__4 = _init_lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__4();
lean_mark_persistent(lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__4);
lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__5 = _init_lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__5();
lean_mark_persistent(lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__5);
lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__6 = _init_lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__6();
lean_mark_persistent(lp_mathlib_ExistsAndEq_withExistsElimAlongPathImp___closed__6);
lp_mathlib_ExistsAndEq_withNestedExistsIntro___redArg___closed__0 = _init_lp_mathlib_ExistsAndEq_withNestedExistsIntro___redArg___closed__0();
lean_mark_persistent(lp_mathlib_ExistsAndEq_withNestedExistsIntro___redArg___closed__0);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__0 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__0();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__0);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__1 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__1();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__1);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__2 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__2();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__2);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__3 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__3();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__3);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__4 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__4();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__4);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__5 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__5();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__5);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__6 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__6();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__6);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__7 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__7();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__7);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__8 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__8();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__8);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__9 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__9();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__9);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__10 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__10();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__10);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__11 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__11();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__11);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__12 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__12();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__12);
lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__13 = _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__13();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_mkBeforeToAfter_go___closed__13);
lp_mathlib_panic___at___00ExistsAndEq_existsAndEq_spec__1___closed__0 = _init_lp_mathlib_panic___at___00ExistsAndEq_existsAndEq_spec__1___closed__0();
lean_mark_persistent(lp_mathlib_panic___at___00ExistsAndEq_existsAndEq_spec__1___closed__0);
lp_mathlib_ExistsAndEq_existsAndEq___closed__0 = _init_lp_mathlib_ExistsAndEq_existsAndEq___closed__0();
lean_mark_persistent(lp_mathlib_ExistsAndEq_existsAndEq___closed__0);
lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__0 = _init_lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__0();
lean_mark_persistent(lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__0);
lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__1 = _init_lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__1();
lean_mark_persistent(lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__1);
lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__2 = _init_lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__2();
lean_mark_persistent(lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__2);
lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__3 = _init_lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__3();
lean_mark_persistent(lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__3);
lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__4 = _init_lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__4();
lean_mark_persistent(lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__4);
lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__5 = _init_lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__5();
lean_mark_persistent(lp_mathlib_ExistsAndEq_existsAndEq___lam__0___closed__5);
lp_mathlib_ExistsAndEq_existsAndEq___lam__1___closed__0 = _init_lp_mathlib_ExistsAndEq_existsAndEq___lam__1___closed__0();
lean_mark_persistent(lp_mathlib_ExistsAndEq_existsAndEq___lam__1___closed__0);
lp_mathlib_ExistsAndEq_existsAndEq___lam__1___closed__1 = _init_lp_mathlib_ExistsAndEq_existsAndEq___lam__1___closed__1();
lean_mark_persistent(lp_mathlib_ExistsAndEq_existsAndEq___lam__1___closed__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
