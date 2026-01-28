// Lean compiler output
// Module: Mathlib.Tactic.CancelDenoms.Core
// Imports: public import Init public meta import Mathlib.Algebra.Field.Basic public meta import Mathlib.Algebra.Order.Ring.Defs public meta import Mathlib.Data.Tree.Basic public meta import Mathlib.Logic.Basic public meta import Mathlib.Tactic.NormNum.Core public meta import Mathlib.Util.SynthesizeUsing public meta import Mathlib.Util.Qq
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
lean_object* lean_nat_gcd(lean_object*, lean_object*);
lean_object* l_Nat_lcm(lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_findCancelFactor___closed__0;
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_cancelDenominators___closed__0;
static lean_object* lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__21;
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__16;
LEAN_EXPORT lean_object* lp_mathlib_initFn_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_();
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__9;
static lean_object* lp_mathlib_CancelDenoms_findCompLemma___closed__14;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__6;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__1;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__46;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__28;
uint64_t l_Lean_Meta_Context_configKey(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
static lean_object* lp_mathlib_CancelDenoms_findCompLemma___closed__15;
static lean_object* lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__6;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__55;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__8;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__49;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00CancelDenoms_synthesizeUsingNormNum_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_findCompLemma(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00cancelDenominators_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__14;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00cancelDenominatorsAt_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__18;
static lean_object* lp_mathlib_initFn___closed__10_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__1;
static lean_object* lp_mathlib_cancelDenoms___closed__9;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__90;
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__11;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__14;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__87;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__14;
static lean_object* lp_mathlib_initFn___closed__3_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__13;
static lean_object* lp_mathlib_initFn___closed__14_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__27;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__15;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__84;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__13;
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__7;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__76;
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
extern lean_object* l_Lean_Parser_Tactic_location;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00CancelDenoms_synthesizeUsingNormNum_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__4(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint64_t lean_uint64_lor(uint64_t, uint64_t);
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__22;
lean_object* l_Lean_Elab_Tactic_expandOptLocation(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_findCancelFactor(lean_object*);
static lean_object* lp_mathlib_CancelDenoms_deriveThms___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_findCompLemma___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_initFn___closed__16_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00CancelDenoms_synthesizeUsingNormNum_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Level_succ___override(lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__65;
lean_object* l_Lean_Expr_lit___override(lean_object*);
static lean_object* lp_mathlib_CancelDenoms_findCompLemma___closed__7;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__24;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__22;
static lean_object* lp_mathlib_CancelDenoms_findCancelFactor___closed__5;
lean_object* lean_array_push(lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__80;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__86;
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__12;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__48;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__0;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__29;
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__1;
static double lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3___closed__0;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__6;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__18;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__22;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__19;
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_replaceTargetEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__37;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__39;
static lean_object* lp_mathlib_CancelDenoms_findCancelFactor___closed__1;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__41;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__71;
static lean_object* lp_mathlib_CancelDenoms_derive___closed__9;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__10;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__16;
static lean_object* lp_mathlib_CancelDenoms_derive___closed__13;
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__3(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__36;
static lean_object* lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__11;
static lean_object* lp_mathlib_CancelDenoms_derive___closed__7;
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
static lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__12;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__52;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__58;
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_findCancelFactor___closed__9;
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__16;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__12;
lean_object* l_Lean_stringToMessageData(lean_object*);
static lean_object* lp_mathlib_initFn___closed__4_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__75;
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__25;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00CancelDenoms_synthesizeUsingNormNum_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__6;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__13;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Exception_toMessageData(lean_object*);
static lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__21;
static lean_object* lp_mathlib_CancelDenoms_findCancelFactor___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__5(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__28;
static lean_object* lp_mathlib_cancelDenoms___closed__8;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__15;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__4;
lean_object* l_Array_mkArray0(lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__0;
static lean_object* lp_mathlib_initFn___closed__13_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__8;
static lean_object* lp_mathlib_CancelDenoms_findCompLemma___closed__6;
static lean_object* lp_mathlib_CancelDenoms_findCompLemma___closed__1;
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__24;
static lean_object* lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__1;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__23;
static lean_object* lp_mathlib_CancelDenoms_derive___lam__0___closed__0;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__70;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__63;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00cancelDenominatorsAt_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__9;
uint8_t l_Lean_Expr_hasMVar(lean_object*);
static lean_object* lp_mathlib_CancelDenoms_deriveThms___closed__5;
static lean_object* lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__12;
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_deriveThms;
lean_object* l_Nat_reprFast(lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__5;
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__13;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__16;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__11;
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__0;
lean_object* l_Array_mkArray1___redArg(lean_object*);
static lean_object* lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__10;
static lean_object* lp_mathlib_CancelDenoms_derive___closed__20;
static lean_object* lp_mathlib_CancelDenoms_findCancelFactor___closed__11;
static lean_object* lp_mathlib_CancelDenoms_findCompLemma___closed__13;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_deriveThms___closed__1;
static lean_object* lp_mathlib_CancelDenoms_derive___closed__11;
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__23;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__18;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00cancelDenominatorsAt_spec__0___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__10;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__13;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__40;
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__6(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_derive___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__69;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__10;
static lean_object* lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3___closed__2;
lean_object* l_Lean_Elab_Tactic_withLocation(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__81;
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__5;
static lean_object* lp_mathlib_CancelDenoms_findCompLemma___closed__12;
extern lean_object* l_Lean_Meta_Simp_neutralConfig;
LEAN_EXPORT lean_object* lp_mathlib_cancelDenominators___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__3;
static lean_object* lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__13;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__15;
static lean_object* lp_mathlib_cancelDenominators___lam__0___closed__1;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__3;
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_cancelDenominatorsTarget___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Qq_inferTypeQ_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00cancelDenominatorsAt_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_findCancelFactor___closed__2;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__83;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__62;
static lean_object* lp_mathlib_CancelDenoms_findCompLemma___closed__16;
lean_object* lean_nat_div(lean_object*, lean_object*);
lean_object* l_Array_empty(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__2(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_cancelDenominatorsAt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__11;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__30;
lean_object* l_Lean_registerTraceClass(lean_object*, uint8_t, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_findCancelFactor___closed__3;
static lean_object* lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__0;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__1;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__25;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_initFn_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__73;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__92;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00cancelDenominators_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00CancelDenoms_synthesizeUsingNormNum_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_cancelDenominatorsTarget(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__23;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__14;
double lean_float_of_nat(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__1;
lean_object* l_Lean_Elab_Tactic_getMainTarget(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__19;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__54;
static lean_object* lp_mathlib_initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__2;
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
static lean_object* lp_mathlib_CancelDenoms_derive___closed__3;
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__26;
static lean_object* lp_mathlib_initFn___closed__7_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_checkTraceOption(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_findCompLemma___closed__5;
static lean_object* lp_mathlib_cancelDenoms___closed__11;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__57;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__27;
LEAN_EXPORT lean_object* lp_mathlib_Lean_isTracingEnabledFor___at___00CancelDenoms_mkProdPrf_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__82;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__5;
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__13;
static lean_object* lp_mathlib_CancelDenoms_findCancelFactor___closed__14;
static lean_object* lp_mathlib_CancelDenoms_findCompLemma___closed__9;
static lean_object* lp_mathlib_CancelDenoms_findCompLemma___closed__11;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__11;
static lean_object* lp_mathlib_tacticCancel__denoms___00__closed__1;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__24;
static lean_object* lp_mathlib_CancelDenoms_derive___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_synthesizeUsingTactic_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__9;
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__44;
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__43;
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__20;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__66;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__0;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__2;
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_derive___closed__2;
static lean_object* lp_mathlib_CancelDenoms_findCancelFactor___closed__13;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__8;
lean_object* l_Lean_mkOptionalNode(lean_object*);
static lean_object* lp_mathlib_CancelDenoms_derive___closed__8;
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__2;
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__6;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__3;
static lean_object* lp_mathlib_CancelDenoms_findCancelFactor___closed__4;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_isTracingEnabledFor___at___00CancelDenoms_mkProdPrf_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__93;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__47;
static lean_object* lp_mathlib_initFn___closed__12_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
static lean_object* lp_mathlib_CancelDenoms_findCompLemma___closed__10;
static lean_object* lp_mathlib_cancelDenoms___closed__0;
static lean_object* lp_mathlib_CancelDenoms_derive___closed__5;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__60;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__91;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__74;
static lean_object* lp_mathlib_tacticCancel__denoms___00__closed__0;
static lean_object* lp_mathlib_cancelDenoms___closed__1;
static lean_object* lp_mathlib_cancelDenoms___closed__3;
static lean_object* lp_mathlib_CancelDenoms_derive___closed__18;
static lean_object* lp_mathlib_cancelDenoms___closed__7;
static lean_object* lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__2;
lean_object* l_Lean_Meta_Simp_mkContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_findCompLemma___closed__4;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__8;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__21;
static lean_object* lp_mathlib_CancelDenoms_derive___closed__21;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__7;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__4;
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__17;
static lean_object* lp_mathlib_CancelDenoms_derive___closed__4;
static lean_object* lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__5;
static lean_object* lp_mathlib_CancelDenoms_derive___closed__14;
static lean_object* lp_mathlib_CancelDenoms_findCancelFactor___closed__12;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__1;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__68;
static lean_object* lp_mathlib_CancelDenoms_derive___closed__1;
lean_object* l_String_toRawSubstring_x27(lean_object*);
static lean_object* lp_mathlib_CancelDenoms_findCompLemma___closed__3;
static lean_object* lp_mathlib_CancelDenoms_findCompLemma___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_synthesizeUsingNormNum___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__10;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__59;
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__8;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__42;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__1;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__6;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__2;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__5;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__4;
static lean_object* lp_mathlib_cancelDenoms___closed__2;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__17;
static lean_object* lp_mathlib_initFn___closed__2_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__85;
lean_object* l_Lean_Meta_Context_config(lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__2;
static lean_object* lp_mathlib_initFn___closed__5_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
static lean_object* lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__12;
LEAN_EXPORT lean_object* lp_mathlib_cancelDenominators___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_cancelDenoms___closed__10;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__51;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__50;
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_synthesizeUsingNormNum(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__15;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__14;
lean_object* lean_array_fget(lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_derive___closed__17;
static lean_object* lp_mathlib_CancelDenoms_findCompLemma___closed__17;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__61;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__33;
LEAN_EXPORT lean_object* lp_mathlib_cancelDenominatorsAt___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_deriveThms___closed__3;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__79;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_initFn___closed__8_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
static lean_object* lp_mathlib_initFn___closed__15_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib_cancelDenominatorsTarget___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__16;
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__8;
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_pow(lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__0;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__18;
static lean_object* lp_mathlib_CancelDenoms_findCompLemma___closed__8;
static lean_object* lp_mathlib_initFn___closed__9_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__4;
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Meta_simpOnlyNames(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_findCompLemma___closed__0;
static lean_object* lp_mathlib_initFn___closed__11_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__10;
static lean_object* lp_mathlib_CancelDenoms_derive___closed__19;
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__4;
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_cancelDenominatorsAt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_derive___closed__15;
lean_object* l_Lean_mkRawNatLit(lean_object*);
static lean_object* lp_mathlib_CancelDenoms_deriveThms___closed__4;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__2;
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__15;
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
static lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__1;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_derive(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__15;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__77;
static lean_object* lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__8;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__11;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__72;
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_cancelDenoms___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__34;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__7;
static lean_object* lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__2;
static lean_object* lp_mathlib_CancelDenoms_findCancelFactor___closed__10;
static lean_object* lp_mathlib_initFn___closed__17_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
static lean_object* lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__9;
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__14;
lean_object* l_Lean_LocalDecl_type(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_derive___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__78;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__5;
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__2;
lean_object* lean_nat_mul(lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__4;
uint64_t lean_uint64_shift_left(uint64_t, uint64_t);
static lean_object* lp_mathlib_cancelDenominators___lam__0___closed__0;
static lean_object* lp_mathlib_CancelDenoms_derive___closed__0;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__5;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__2;
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__9;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__7;
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_cancelDenoms___closed__4;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__17;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__3;
static lean_object* lp_mathlib_CancelDenoms_findCancelFactor___closed__15;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__9;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__18;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1_spec__0___redArg();
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00CancelDenoms_mkProdPrf_spec__2___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__88;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__32;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__9;
extern lean_object* l_Lean_Meta_Simp_defaultMaxSteps;
static lean_object* lp_mathlib_CancelDenoms_derive___closed__12;
lean_object* l_Lean_FVarId_getDecl___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__35;
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__20;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_cancelDenominators(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_findCancelFactor___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Lean_isTracingEnabledFor___at___00CancelDenoms_mkProdPrf_spec__0___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__4;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00cancelDenominators_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__3;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__11;
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__19;
lean_object* lp_Qq_Qq_synthInstanceQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static uint64_t lp_mathlib_CancelDenoms_mkProdPrf___lam__4___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_tacticCancel__denoms__;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00CancelDenoms_mkProdPrf_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__5;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__12;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__6;
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1_spec__0___redArg___closed__0;
static lean_object* lp_mathlib_CancelDenoms_derive___closed__16;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1_spec__0___redArg___boxed(lean_object*);
static lean_object* lp_mathlib_CancelDenoms_derive___closed__10;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__17;
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__3;
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__12;
lean_object* l_Lean_Expr_getAppFnArgs(lean_object*);
static lean_object* lp_mathlib_CancelDenoms_derive___lam__0___closed__2;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__0;
static lean_object* lp_mathlib_CancelDenoms_derive___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_cancelDenoms;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__45;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00CancelDenoms_synthesizeUsingNormNum_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__6;
static lean_object* lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__4;
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00CancelDenoms_mkProdPrf_spec__2(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
static lean_object* lp_mathlib_CancelDenoms_deriveThms___closed__0;
lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveSimp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__38;
uint64_t l_Lean_Meta_TransparencyMode_toUInt64(uint8_t);
static lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__19;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__26;
static lean_object* lp_mathlib_CancelDenoms_findCancelFactor___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_derive___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__56;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__67;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__20;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__16;
LEAN_EXPORT lean_object* lp_mathlib_Lean_isTracingEnabledFor___at___00CancelDenoms_mkProdPrf_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Tactic_Replace_0__Lean_Meta_replaceLocalDeclCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__5;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__89;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__31;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__10;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00CancelDenoms_mkProdPrf_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Elab_Tactic_liftMetaTactic_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_nat_x3f(lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__1;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__64;
LEAN_EXPORT lean_object* lp_mathlib_cancelDenominators___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_tacticCancel__denoms___00__closed__2;
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___closed__53;
LEAN_EXPORT lean_object* lp_mathlib_cancelDenominatorsTarget___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00cancelDenominators_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1_spec__0___redArg___closed__1;
static lean_object* lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__3;
static lean_object* lp_mathlib_initFn___closed__1_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__9;
lean_object* l_Lean_Meta_whnfR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__17;
LEAN_EXPORT lean_object* lp_mathlib_cancelDenominatorsAt___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__7;
static lean_object* lp_mathlib_initFn___closed__18_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
static lean_object* lp_mathlib_cancelDenoms___closed__5;
static lean_object* _init_lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("CancelDenoms", 12, 12);
return x_1;
}
}
static lean_object* _init_lp_mathlib_initFn___closed__1_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_initFn___closed__2_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("initFn", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_initFn___closed__3_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_initFn___closed__2_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_2 = lean_box(0);
x_3 = l_Lean_Name_str___override(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_initFn___closed__4_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("_@", 2, 2);
return x_1;
}
}
static lean_object* _init_lp_mathlib_initFn___closed__5_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_initFn___closed__4_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_2 = lp_mathlib_initFn___closed__3_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_3 = l_Lean_Name_str___override(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Mathlib", 7, 7);
return x_1;
}
}
static lean_object* _init_lp_mathlib_initFn___closed__7_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_2 = lp_mathlib_initFn___closed__5_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_3 = l_Lean_Name_str___override(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_initFn___closed__8_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Tactic", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_initFn___closed__9_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_initFn___closed__8_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_2 = lp_mathlib_initFn___closed__7_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_3 = l_Lean_Name_str___override(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_initFn___closed__10_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_2 = lp_mathlib_initFn___closed__9_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_3 = l_Lean_Name_str___override(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_initFn___closed__11_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Core", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_initFn___closed__12_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_initFn___closed__11_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_2 = lp_mathlib_initFn___closed__10_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_3 = l_Lean_Name_str___override(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_initFn___closed__13_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_unsigned_to_nat(1602764063u);
x_2 = lp_mathlib_initFn___closed__12_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_3 = l_Lean_Name_num___override(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_initFn___closed__14_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("_hygCtx", 7, 7);
return x_1;
}
}
static lean_object* _init_lp_mathlib_initFn___closed__15_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_initFn___closed__14_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_2 = lp_mathlib_initFn___closed__13_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_3 = l_Lean_Name_str___override(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_initFn___closed__16_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("_hyg", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_initFn___closed__17_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_initFn___closed__16_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_2 = lp_mathlib_initFn___closed__15_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_3 = l_Lean_Name_str___override(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_initFn___closed__18_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_unsigned_to_nat(2u);
x_2 = lp_mathlib_initFn___closed__17_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_3 = l_Lean_Name_num___override(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_initFn_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_() {
_start:
{
lean_object* x_2; uint8_t x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_mathlib_initFn___closed__1_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_3 = 0;
x_4 = lp_mathlib_initFn___closed__18_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_5 = l_Lean_registerTraceClass(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_initFn_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2____boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_initFn_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_();
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lean_unsigned_to_nat(1u);
x_3 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
lean_ctor_set(x_3, 2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_findCancelFactor___closed__0;
x_2 = lean_unsigned_to_nat(1u);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("HAdd", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("HSub", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("HMul", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("HDiv", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Neg", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__7() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("HPow", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__8() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Inv", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__9() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("inv", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__10() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("hPow", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__11() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("neg", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__12() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("hDiv", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__13() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("hMul", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__14() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("hSub", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__15() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("hAdd", 4, 4);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_findCancelFactor(lean_object* x_1) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_21; lean_object* x_22; 
x_21 = l_Lean_Expr_getAppFnArgs(x_1);
x_22 = lean_ctor_get(x_21, 0);
lean_inc(x_22);
if (lean_obj_tag(x_22) == 1)
{
lean_object* x_23; 
x_23 = lean_ctor_get(x_22, 0);
lean_inc(x_23);
if (lean_obj_tag(x_23) == 1)
{
lean_object* x_24; 
x_24 = lean_ctor_get(x_23, 0);
if (lean_obj_tag(x_24) == 0)
{
uint8_t x_25; 
x_25 = !lean_is_exclusive(x_21);
if (x_25 == 0)
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; uint8_t x_31; 
x_26 = lean_ctor_get(x_21, 1);
x_27 = lean_ctor_get(x_21, 0);
lean_dec(x_27);
x_28 = lean_ctor_get(x_22, 1);
lean_inc_ref(x_28);
lean_dec_ref(x_22);
x_29 = lean_ctor_get(x_23, 1);
lean_inc_ref(x_29);
lean_dec_ref(x_23);
x_30 = lp_mathlib_CancelDenoms_findCancelFactor___closed__2;
x_31 = lean_string_dec_eq(x_29, x_30);
if (x_31 == 0)
{
lean_object* x_32; uint8_t x_33; 
x_32 = lp_mathlib_CancelDenoms_findCancelFactor___closed__3;
x_33 = lean_string_dec_eq(x_29, x_32);
if (x_33 == 0)
{
lean_object* x_34; uint8_t x_35; 
x_34 = lp_mathlib_CancelDenoms_findCancelFactor___closed__4;
x_35 = lean_string_dec_eq(x_29, x_34);
if (x_35 == 0)
{
lean_object* x_36; uint8_t x_37; 
x_36 = lp_mathlib_CancelDenoms_findCancelFactor___closed__5;
x_37 = lean_string_dec_eq(x_29, x_36);
if (x_37 == 0)
{
lean_object* x_38; uint8_t x_39; 
x_38 = lp_mathlib_CancelDenoms_findCancelFactor___closed__6;
x_39 = lean_string_dec_eq(x_29, x_38);
if (x_39 == 0)
{
lean_object* x_40; uint8_t x_41; 
x_40 = lp_mathlib_CancelDenoms_findCancelFactor___closed__7;
x_41 = lean_string_dec_eq(x_29, x_40);
if (x_41 == 0)
{
lean_object* x_42; uint8_t x_43; 
x_42 = lp_mathlib_CancelDenoms_findCancelFactor___closed__8;
x_43 = lean_string_dec_eq(x_29, x_42);
lean_dec_ref(x_29);
if (x_43 == 0)
{
lean_dec_ref(x_28);
lean_free_object(x_21);
lean_dec(x_26);
goto block_3;
}
else
{
lean_object* x_44; uint8_t x_45; 
x_44 = lp_mathlib_CancelDenoms_findCancelFactor___closed__9;
x_45 = lean_string_dec_eq(x_28, x_44);
lean_dec_ref(x_28);
if (x_45 == 0)
{
lean_free_object(x_21);
lean_dec(x_26);
goto block_3;
}
else
{
lean_object* x_46; lean_object* x_47; uint8_t x_48; 
x_46 = lean_array_get_size(x_26);
x_47 = lean_unsigned_to_nat(3u);
x_48 = lean_nat_dec_eq(x_46, x_47);
if (x_48 == 0)
{
lean_free_object(x_21);
lean_dec(x_26);
goto block_3;
}
else
{
lean_object* x_49; lean_object* x_50; lean_object* x_51; 
x_49 = lean_unsigned_to_nat(2u);
x_50 = lean_array_fget(x_26, x_49);
lean_dec(x_26);
x_51 = l_Lean_Expr_nat_x3f(x_50);
if (lean_obj_tag(x_51) == 0)
{
lean_object* x_52; 
lean_free_object(x_21);
x_52 = lp_mathlib_CancelDenoms_findCancelFactor___closed__1;
return x_52;
}
else
{
lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; 
x_53 = lean_ctor_get(x_51, 0);
lean_inc(x_53);
lean_dec_ref(x_51);
x_54 = lean_box(0);
lean_inc(x_53);
x_55 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_55, 0, x_53);
lean_ctor_set(x_55, 1, x_54);
lean_ctor_set(x_55, 2, x_54);
lean_inc(x_53);
x_56 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_56, 0, x_53);
lean_ctor_set(x_56, 1, x_54);
lean_ctor_set(x_56, 2, x_55);
lean_ctor_set(x_21, 1, x_56);
lean_ctor_set(x_21, 0, x_53);
return x_21;
}
}
}
}
}
else
{
lean_object* x_57; uint8_t x_58; 
lean_dec_ref(x_29);
lean_free_object(x_21);
x_57 = lp_mathlib_CancelDenoms_findCancelFactor___closed__10;
x_58 = lean_string_dec_eq(x_28, x_57);
lean_dec_ref(x_28);
if (x_58 == 0)
{
lean_dec(x_26);
goto block_3;
}
else
{
lean_object* x_59; lean_object* x_60; uint8_t x_61; 
x_59 = lean_array_get_size(x_26);
x_60 = lean_unsigned_to_nat(6u);
x_61 = lean_nat_dec_eq(x_59, x_60);
if (x_61 == 0)
{
lean_dec(x_26);
goto block_3;
}
else
{
lean_object* x_62; lean_object* x_63; lean_object* x_64; 
x_62 = lean_unsigned_to_nat(5u);
x_63 = lean_array_fget_borrowed(x_26, x_62);
lean_inc(x_63);
x_64 = l_Lean_Expr_nat_x3f(x_63);
if (lean_obj_tag(x_64) == 0)
{
lean_object* x_65; 
lean_dec(x_26);
x_65 = lp_mathlib_CancelDenoms_findCancelFactor___closed__1;
return x_65;
}
else
{
lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; uint8_t x_70; 
x_66 = lean_ctor_get(x_64, 0);
lean_inc(x_66);
lean_dec_ref(x_64);
x_67 = lean_unsigned_to_nat(4u);
x_68 = lean_array_fget(x_26, x_67);
lean_dec(x_26);
x_69 = lp_mathlib_CancelDenoms_findCancelFactor(x_68);
x_70 = !lean_is_exclusive(x_69);
if (x_70 == 0)
{
lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; 
x_71 = lean_ctor_get(x_69, 0);
x_72 = lean_ctor_get(x_69, 1);
x_73 = lean_nat_pow(x_71, x_66);
lean_dec(x_71);
x_74 = lean_box(0);
x_75 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_75, 0, x_66);
lean_ctor_set(x_75, 1, x_74);
lean_ctor_set(x_75, 2, x_74);
lean_inc(x_73);
x_76 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_76, 0, x_73);
lean_ctor_set(x_76, 1, x_72);
lean_ctor_set(x_76, 2, x_75);
lean_ctor_set(x_69, 1, x_76);
lean_ctor_set(x_69, 0, x_73);
return x_69;
}
else
{
lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; 
x_77 = lean_ctor_get(x_69, 0);
x_78 = lean_ctor_get(x_69, 1);
lean_inc(x_78);
lean_inc(x_77);
lean_dec(x_69);
x_79 = lean_nat_pow(x_77, x_66);
lean_dec(x_77);
x_80 = lean_box(0);
x_81 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_81, 0, x_66);
lean_ctor_set(x_81, 1, x_80);
lean_ctor_set(x_81, 2, x_80);
lean_inc(x_79);
x_82 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_82, 0, x_79);
lean_ctor_set(x_82, 1, x_78);
lean_ctor_set(x_82, 2, x_81);
x_83 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_83, 0, x_79);
lean_ctor_set(x_83, 1, x_82);
return x_83;
}
}
}
}
}
}
else
{
lean_object* x_84; uint8_t x_85; 
lean_dec_ref(x_29);
lean_free_object(x_21);
x_84 = lp_mathlib_CancelDenoms_findCancelFactor___closed__11;
x_85 = lean_string_dec_eq(x_28, x_84);
lean_dec_ref(x_28);
if (x_85 == 0)
{
lean_dec(x_26);
goto block_3;
}
else
{
lean_object* x_86; lean_object* x_87; uint8_t x_88; 
x_86 = lean_array_get_size(x_26);
x_87 = lean_unsigned_to_nat(3u);
x_88 = lean_nat_dec_eq(x_86, x_87);
if (x_88 == 0)
{
lean_dec(x_26);
goto block_3;
}
else
{
lean_object* x_89; lean_object* x_90; 
x_89 = lean_unsigned_to_nat(2u);
x_90 = lean_array_fget(x_26, x_89);
lean_dec(x_26);
x_1 = x_90;
goto _start;
}
}
}
}
else
{
lean_object* x_92; uint8_t x_93; 
lean_dec_ref(x_29);
lean_free_object(x_21);
x_92 = lp_mathlib_CancelDenoms_findCancelFactor___closed__12;
x_93 = lean_string_dec_eq(x_28, x_92);
lean_dec_ref(x_28);
if (x_93 == 0)
{
lean_dec(x_26);
goto block_3;
}
else
{
lean_object* x_94; lean_object* x_95; uint8_t x_96; 
x_94 = lean_array_get_size(x_26);
x_95 = lean_unsigned_to_nat(6u);
x_96 = lean_nat_dec_eq(x_94, x_95);
if (x_96 == 0)
{
lean_dec(x_26);
goto block_3;
}
else
{
lean_object* x_97; lean_object* x_98; lean_object* x_99; 
x_97 = lean_unsigned_to_nat(5u);
x_98 = lean_array_fget_borrowed(x_26, x_97);
lean_inc(x_98);
x_99 = l_Lean_Expr_nat_x3f(x_98);
if (lean_obj_tag(x_99) == 0)
{
lean_object* x_100; 
lean_dec(x_26);
x_100 = lp_mathlib_CancelDenoms_findCancelFactor___closed__1;
return x_100;
}
else
{
lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; uint8_t x_105; 
x_101 = lean_ctor_get(x_99, 0);
lean_inc(x_101);
lean_dec_ref(x_99);
x_102 = lean_unsigned_to_nat(4u);
x_103 = lean_array_fget(x_26, x_102);
lean_dec(x_26);
x_104 = lp_mathlib_CancelDenoms_findCancelFactor(x_103);
x_105 = !lean_is_exclusive(x_104);
if (x_105 == 0)
{
lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; 
x_106 = lean_ctor_get(x_104, 0);
x_107 = lean_ctor_get(x_104, 1);
x_108 = lean_nat_mul(x_106, x_101);
lean_dec(x_106);
x_109 = lean_box(0);
x_110 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_110, 0, x_101);
lean_ctor_set(x_110, 1, x_109);
lean_ctor_set(x_110, 2, x_109);
lean_inc(x_108);
x_111 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_111, 0, x_108);
lean_ctor_set(x_111, 1, x_107);
lean_ctor_set(x_111, 2, x_110);
lean_ctor_set(x_104, 1, x_111);
lean_ctor_set(x_104, 0, x_108);
return x_104;
}
else
{
lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; lean_object* x_116; lean_object* x_117; lean_object* x_118; 
x_112 = lean_ctor_get(x_104, 0);
x_113 = lean_ctor_get(x_104, 1);
lean_inc(x_113);
lean_inc(x_112);
lean_dec(x_104);
x_114 = lean_nat_mul(x_112, x_101);
lean_dec(x_112);
x_115 = lean_box(0);
x_116 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_116, 0, x_101);
lean_ctor_set(x_116, 1, x_115);
lean_ctor_set(x_116, 2, x_115);
lean_inc(x_114);
x_117 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_117, 0, x_114);
lean_ctor_set(x_117, 1, x_113);
lean_ctor_set(x_117, 2, x_116);
x_118 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_118, 0, x_114);
lean_ctor_set(x_118, 1, x_117);
return x_118;
}
}
}
}
}
}
else
{
lean_object* x_119; uint8_t x_120; 
lean_dec_ref(x_29);
lean_free_object(x_21);
x_119 = lp_mathlib_CancelDenoms_findCancelFactor___closed__13;
x_120 = lean_string_dec_eq(x_28, x_119);
lean_dec_ref(x_28);
if (x_120 == 0)
{
lean_dec(x_26);
goto block_3;
}
else
{
lean_object* x_121; lean_object* x_122; uint8_t x_123; 
x_121 = lean_array_get_size(x_26);
x_122 = lean_unsigned_to_nat(6u);
x_123 = lean_nat_dec_eq(x_121, x_122);
if (x_123 == 0)
{
lean_dec(x_26);
goto block_3;
}
else
{
lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_127; lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; uint8_t x_132; 
x_124 = lean_unsigned_to_nat(4u);
x_125 = lean_array_fget_borrowed(x_26, x_124);
lean_inc(x_125);
x_126 = lp_mathlib_CancelDenoms_findCancelFactor(x_125);
x_127 = lean_ctor_get(x_126, 0);
lean_inc(x_127);
x_128 = lean_ctor_get(x_126, 1);
lean_inc(x_128);
lean_dec_ref(x_126);
x_129 = lean_unsigned_to_nat(5u);
x_130 = lean_array_fget(x_26, x_129);
lean_dec(x_26);
x_131 = lp_mathlib_CancelDenoms_findCancelFactor(x_130);
x_132 = !lean_is_exclusive(x_131);
if (x_132 == 0)
{
lean_object* x_133; lean_object* x_134; lean_object* x_135; lean_object* x_136; 
x_133 = lean_ctor_get(x_131, 0);
x_134 = lean_ctor_get(x_131, 1);
x_135 = lean_nat_mul(x_127, x_133);
lean_dec(x_133);
lean_dec(x_127);
lean_inc(x_135);
x_136 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_136, 0, x_135);
lean_ctor_set(x_136, 1, x_128);
lean_ctor_set(x_136, 2, x_134);
lean_ctor_set(x_131, 1, x_136);
lean_ctor_set(x_131, 0, x_135);
return x_131;
}
else
{
lean_object* x_137; lean_object* x_138; lean_object* x_139; lean_object* x_140; lean_object* x_141; 
x_137 = lean_ctor_get(x_131, 0);
x_138 = lean_ctor_get(x_131, 1);
lean_inc(x_138);
lean_inc(x_137);
lean_dec(x_131);
x_139 = lean_nat_mul(x_127, x_137);
lean_dec(x_137);
lean_dec(x_127);
lean_inc(x_139);
x_140 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_140, 0, x_139);
lean_ctor_set(x_140, 1, x_128);
lean_ctor_set(x_140, 2, x_138);
x_141 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_141, 0, x_139);
lean_ctor_set(x_141, 1, x_140);
return x_141;
}
}
}
}
}
else
{
lean_object* x_142; uint8_t x_143; 
lean_dec_ref(x_29);
lean_free_object(x_21);
x_142 = lp_mathlib_CancelDenoms_findCancelFactor___closed__14;
x_143 = lean_string_dec_eq(x_28, x_142);
lean_dec_ref(x_28);
if (x_143 == 0)
{
lean_dec(x_26);
goto block_3;
}
else
{
lean_object* x_144; lean_object* x_145; uint8_t x_146; 
x_144 = lean_array_get_size(x_26);
x_145 = lean_unsigned_to_nat(6u);
x_146 = lean_nat_dec_eq(x_144, x_145);
if (x_146 == 0)
{
lean_dec(x_26);
goto block_3;
}
else
{
lean_object* x_147; lean_object* x_148; lean_object* x_149; lean_object* x_150; 
x_147 = lean_unsigned_to_nat(4u);
x_148 = lean_array_fget(x_26, x_147);
x_149 = lean_unsigned_to_nat(5u);
x_150 = lean_array_fget(x_26, x_149);
lean_dec(x_26);
x_4 = x_148;
x_5 = x_150;
goto block_20;
}
}
}
}
else
{
lean_object* x_151; uint8_t x_152; 
lean_dec_ref(x_29);
lean_free_object(x_21);
x_151 = lp_mathlib_CancelDenoms_findCancelFactor___closed__15;
x_152 = lean_string_dec_eq(x_28, x_151);
lean_dec_ref(x_28);
if (x_152 == 0)
{
lean_dec(x_26);
goto block_3;
}
else
{
lean_object* x_153; lean_object* x_154; uint8_t x_155; 
x_153 = lean_array_get_size(x_26);
x_154 = lean_unsigned_to_nat(6u);
x_155 = lean_nat_dec_eq(x_153, x_154);
if (x_155 == 0)
{
lean_dec(x_26);
goto block_3;
}
else
{
lean_object* x_156; lean_object* x_157; lean_object* x_158; lean_object* x_159; 
x_156 = lean_unsigned_to_nat(4u);
x_157 = lean_array_fget(x_26, x_156);
x_158 = lean_unsigned_to_nat(5u);
x_159 = lean_array_fget(x_26, x_158);
lean_dec(x_26);
x_4 = x_157;
x_5 = x_159;
goto block_20;
}
}
}
}
else
{
lean_object* x_160; lean_object* x_161; lean_object* x_162; lean_object* x_163; uint8_t x_164; 
x_160 = lean_ctor_get(x_21, 1);
lean_inc(x_160);
lean_dec(x_21);
x_161 = lean_ctor_get(x_22, 1);
lean_inc_ref(x_161);
lean_dec_ref(x_22);
x_162 = lean_ctor_get(x_23, 1);
lean_inc_ref(x_162);
lean_dec_ref(x_23);
x_163 = lp_mathlib_CancelDenoms_findCancelFactor___closed__2;
x_164 = lean_string_dec_eq(x_162, x_163);
if (x_164 == 0)
{
lean_object* x_165; uint8_t x_166; 
x_165 = lp_mathlib_CancelDenoms_findCancelFactor___closed__3;
x_166 = lean_string_dec_eq(x_162, x_165);
if (x_166 == 0)
{
lean_object* x_167; uint8_t x_168; 
x_167 = lp_mathlib_CancelDenoms_findCancelFactor___closed__4;
x_168 = lean_string_dec_eq(x_162, x_167);
if (x_168 == 0)
{
lean_object* x_169; uint8_t x_170; 
x_169 = lp_mathlib_CancelDenoms_findCancelFactor___closed__5;
x_170 = lean_string_dec_eq(x_162, x_169);
if (x_170 == 0)
{
lean_object* x_171; uint8_t x_172; 
x_171 = lp_mathlib_CancelDenoms_findCancelFactor___closed__6;
x_172 = lean_string_dec_eq(x_162, x_171);
if (x_172 == 0)
{
lean_object* x_173; uint8_t x_174; 
x_173 = lp_mathlib_CancelDenoms_findCancelFactor___closed__7;
x_174 = lean_string_dec_eq(x_162, x_173);
if (x_174 == 0)
{
lean_object* x_175; uint8_t x_176; 
x_175 = lp_mathlib_CancelDenoms_findCancelFactor___closed__8;
x_176 = lean_string_dec_eq(x_162, x_175);
lean_dec_ref(x_162);
if (x_176 == 0)
{
lean_dec_ref(x_161);
lean_dec(x_160);
goto block_3;
}
else
{
lean_object* x_177; uint8_t x_178; 
x_177 = lp_mathlib_CancelDenoms_findCancelFactor___closed__9;
x_178 = lean_string_dec_eq(x_161, x_177);
lean_dec_ref(x_161);
if (x_178 == 0)
{
lean_dec(x_160);
goto block_3;
}
else
{
lean_object* x_179; lean_object* x_180; uint8_t x_181; 
x_179 = lean_array_get_size(x_160);
x_180 = lean_unsigned_to_nat(3u);
x_181 = lean_nat_dec_eq(x_179, x_180);
if (x_181 == 0)
{
lean_dec(x_160);
goto block_3;
}
else
{
lean_object* x_182; lean_object* x_183; lean_object* x_184; 
x_182 = lean_unsigned_to_nat(2u);
x_183 = lean_array_fget(x_160, x_182);
lean_dec(x_160);
x_184 = l_Lean_Expr_nat_x3f(x_183);
if (lean_obj_tag(x_184) == 0)
{
lean_object* x_185; 
x_185 = lp_mathlib_CancelDenoms_findCancelFactor___closed__1;
return x_185;
}
else
{
lean_object* x_186; lean_object* x_187; lean_object* x_188; lean_object* x_189; lean_object* x_190; 
x_186 = lean_ctor_get(x_184, 0);
lean_inc(x_186);
lean_dec_ref(x_184);
x_187 = lean_box(0);
lean_inc(x_186);
x_188 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_188, 0, x_186);
lean_ctor_set(x_188, 1, x_187);
lean_ctor_set(x_188, 2, x_187);
lean_inc(x_186);
x_189 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_189, 0, x_186);
lean_ctor_set(x_189, 1, x_187);
lean_ctor_set(x_189, 2, x_188);
x_190 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_190, 0, x_186);
lean_ctor_set(x_190, 1, x_189);
return x_190;
}
}
}
}
}
else
{
lean_object* x_191; uint8_t x_192; 
lean_dec_ref(x_162);
x_191 = lp_mathlib_CancelDenoms_findCancelFactor___closed__10;
x_192 = lean_string_dec_eq(x_161, x_191);
lean_dec_ref(x_161);
if (x_192 == 0)
{
lean_dec(x_160);
goto block_3;
}
else
{
lean_object* x_193; lean_object* x_194; uint8_t x_195; 
x_193 = lean_array_get_size(x_160);
x_194 = lean_unsigned_to_nat(6u);
x_195 = lean_nat_dec_eq(x_193, x_194);
if (x_195 == 0)
{
lean_dec(x_160);
goto block_3;
}
else
{
lean_object* x_196; lean_object* x_197; lean_object* x_198; 
x_196 = lean_unsigned_to_nat(5u);
x_197 = lean_array_fget_borrowed(x_160, x_196);
lean_inc(x_197);
x_198 = l_Lean_Expr_nat_x3f(x_197);
if (lean_obj_tag(x_198) == 0)
{
lean_object* x_199; 
lean_dec(x_160);
x_199 = lp_mathlib_CancelDenoms_findCancelFactor___closed__1;
return x_199;
}
else
{
lean_object* x_200; lean_object* x_201; lean_object* x_202; lean_object* x_203; lean_object* x_204; lean_object* x_205; lean_object* x_206; lean_object* x_207; lean_object* x_208; lean_object* x_209; lean_object* x_210; lean_object* x_211; 
x_200 = lean_ctor_get(x_198, 0);
lean_inc(x_200);
lean_dec_ref(x_198);
x_201 = lean_unsigned_to_nat(4u);
x_202 = lean_array_fget(x_160, x_201);
lean_dec(x_160);
x_203 = lp_mathlib_CancelDenoms_findCancelFactor(x_202);
x_204 = lean_ctor_get(x_203, 0);
lean_inc(x_204);
x_205 = lean_ctor_get(x_203, 1);
lean_inc(x_205);
if (lean_is_exclusive(x_203)) {
 lean_ctor_release(x_203, 0);
 lean_ctor_release(x_203, 1);
 x_206 = x_203;
} else {
 lean_dec_ref(x_203);
 x_206 = lean_box(0);
}
x_207 = lean_nat_pow(x_204, x_200);
lean_dec(x_204);
x_208 = lean_box(0);
x_209 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_209, 0, x_200);
lean_ctor_set(x_209, 1, x_208);
lean_ctor_set(x_209, 2, x_208);
lean_inc(x_207);
x_210 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_210, 0, x_207);
lean_ctor_set(x_210, 1, x_205);
lean_ctor_set(x_210, 2, x_209);
if (lean_is_scalar(x_206)) {
 x_211 = lean_alloc_ctor(0, 2, 0);
} else {
 x_211 = x_206;
}
lean_ctor_set(x_211, 0, x_207);
lean_ctor_set(x_211, 1, x_210);
return x_211;
}
}
}
}
}
else
{
lean_object* x_212; uint8_t x_213; 
lean_dec_ref(x_162);
x_212 = lp_mathlib_CancelDenoms_findCancelFactor___closed__11;
x_213 = lean_string_dec_eq(x_161, x_212);
lean_dec_ref(x_161);
if (x_213 == 0)
{
lean_dec(x_160);
goto block_3;
}
else
{
lean_object* x_214; lean_object* x_215; uint8_t x_216; 
x_214 = lean_array_get_size(x_160);
x_215 = lean_unsigned_to_nat(3u);
x_216 = lean_nat_dec_eq(x_214, x_215);
if (x_216 == 0)
{
lean_dec(x_160);
goto block_3;
}
else
{
lean_object* x_217; lean_object* x_218; 
x_217 = lean_unsigned_to_nat(2u);
x_218 = lean_array_fget(x_160, x_217);
lean_dec(x_160);
x_1 = x_218;
goto _start;
}
}
}
}
else
{
lean_object* x_220; uint8_t x_221; 
lean_dec_ref(x_162);
x_220 = lp_mathlib_CancelDenoms_findCancelFactor___closed__12;
x_221 = lean_string_dec_eq(x_161, x_220);
lean_dec_ref(x_161);
if (x_221 == 0)
{
lean_dec(x_160);
goto block_3;
}
else
{
lean_object* x_222; lean_object* x_223; uint8_t x_224; 
x_222 = lean_array_get_size(x_160);
x_223 = lean_unsigned_to_nat(6u);
x_224 = lean_nat_dec_eq(x_222, x_223);
if (x_224 == 0)
{
lean_dec(x_160);
goto block_3;
}
else
{
lean_object* x_225; lean_object* x_226; lean_object* x_227; 
x_225 = lean_unsigned_to_nat(5u);
x_226 = lean_array_fget_borrowed(x_160, x_225);
lean_inc(x_226);
x_227 = l_Lean_Expr_nat_x3f(x_226);
if (lean_obj_tag(x_227) == 0)
{
lean_object* x_228; 
lean_dec(x_160);
x_228 = lp_mathlib_CancelDenoms_findCancelFactor___closed__1;
return x_228;
}
else
{
lean_object* x_229; lean_object* x_230; lean_object* x_231; lean_object* x_232; lean_object* x_233; lean_object* x_234; lean_object* x_235; lean_object* x_236; lean_object* x_237; lean_object* x_238; lean_object* x_239; lean_object* x_240; 
x_229 = lean_ctor_get(x_227, 0);
lean_inc(x_229);
lean_dec_ref(x_227);
x_230 = lean_unsigned_to_nat(4u);
x_231 = lean_array_fget(x_160, x_230);
lean_dec(x_160);
x_232 = lp_mathlib_CancelDenoms_findCancelFactor(x_231);
x_233 = lean_ctor_get(x_232, 0);
lean_inc(x_233);
x_234 = lean_ctor_get(x_232, 1);
lean_inc(x_234);
if (lean_is_exclusive(x_232)) {
 lean_ctor_release(x_232, 0);
 lean_ctor_release(x_232, 1);
 x_235 = x_232;
} else {
 lean_dec_ref(x_232);
 x_235 = lean_box(0);
}
x_236 = lean_nat_mul(x_233, x_229);
lean_dec(x_233);
x_237 = lean_box(0);
x_238 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_238, 0, x_229);
lean_ctor_set(x_238, 1, x_237);
lean_ctor_set(x_238, 2, x_237);
lean_inc(x_236);
x_239 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_239, 0, x_236);
lean_ctor_set(x_239, 1, x_234);
lean_ctor_set(x_239, 2, x_238);
if (lean_is_scalar(x_235)) {
 x_240 = lean_alloc_ctor(0, 2, 0);
} else {
 x_240 = x_235;
}
lean_ctor_set(x_240, 0, x_236);
lean_ctor_set(x_240, 1, x_239);
return x_240;
}
}
}
}
}
else
{
lean_object* x_241; uint8_t x_242; 
lean_dec_ref(x_162);
x_241 = lp_mathlib_CancelDenoms_findCancelFactor___closed__13;
x_242 = lean_string_dec_eq(x_161, x_241);
lean_dec_ref(x_161);
if (x_242 == 0)
{
lean_dec(x_160);
goto block_3;
}
else
{
lean_object* x_243; lean_object* x_244; uint8_t x_245; 
x_243 = lean_array_get_size(x_160);
x_244 = lean_unsigned_to_nat(6u);
x_245 = lean_nat_dec_eq(x_243, x_244);
if (x_245 == 0)
{
lean_dec(x_160);
goto block_3;
}
else
{
lean_object* x_246; lean_object* x_247; lean_object* x_248; lean_object* x_249; lean_object* x_250; lean_object* x_251; lean_object* x_252; lean_object* x_253; lean_object* x_254; lean_object* x_255; lean_object* x_256; lean_object* x_257; lean_object* x_258; lean_object* x_259; 
x_246 = lean_unsigned_to_nat(4u);
x_247 = lean_array_fget_borrowed(x_160, x_246);
lean_inc(x_247);
x_248 = lp_mathlib_CancelDenoms_findCancelFactor(x_247);
x_249 = lean_ctor_get(x_248, 0);
lean_inc(x_249);
x_250 = lean_ctor_get(x_248, 1);
lean_inc(x_250);
lean_dec_ref(x_248);
x_251 = lean_unsigned_to_nat(5u);
x_252 = lean_array_fget(x_160, x_251);
lean_dec(x_160);
x_253 = lp_mathlib_CancelDenoms_findCancelFactor(x_252);
x_254 = lean_ctor_get(x_253, 0);
lean_inc(x_254);
x_255 = lean_ctor_get(x_253, 1);
lean_inc(x_255);
if (lean_is_exclusive(x_253)) {
 lean_ctor_release(x_253, 0);
 lean_ctor_release(x_253, 1);
 x_256 = x_253;
} else {
 lean_dec_ref(x_253);
 x_256 = lean_box(0);
}
x_257 = lean_nat_mul(x_249, x_254);
lean_dec(x_254);
lean_dec(x_249);
lean_inc(x_257);
x_258 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_258, 0, x_257);
lean_ctor_set(x_258, 1, x_250);
lean_ctor_set(x_258, 2, x_255);
if (lean_is_scalar(x_256)) {
 x_259 = lean_alloc_ctor(0, 2, 0);
} else {
 x_259 = x_256;
}
lean_ctor_set(x_259, 0, x_257);
lean_ctor_set(x_259, 1, x_258);
return x_259;
}
}
}
}
else
{
lean_object* x_260; uint8_t x_261; 
lean_dec_ref(x_162);
x_260 = lp_mathlib_CancelDenoms_findCancelFactor___closed__14;
x_261 = lean_string_dec_eq(x_161, x_260);
lean_dec_ref(x_161);
if (x_261 == 0)
{
lean_dec(x_160);
goto block_3;
}
else
{
lean_object* x_262; lean_object* x_263; uint8_t x_264; 
x_262 = lean_array_get_size(x_160);
x_263 = lean_unsigned_to_nat(6u);
x_264 = lean_nat_dec_eq(x_262, x_263);
if (x_264 == 0)
{
lean_dec(x_160);
goto block_3;
}
else
{
lean_object* x_265; lean_object* x_266; lean_object* x_267; lean_object* x_268; 
x_265 = lean_unsigned_to_nat(4u);
x_266 = lean_array_fget(x_160, x_265);
x_267 = lean_unsigned_to_nat(5u);
x_268 = lean_array_fget(x_160, x_267);
lean_dec(x_160);
x_4 = x_266;
x_5 = x_268;
goto block_20;
}
}
}
}
else
{
lean_object* x_269; uint8_t x_270; 
lean_dec_ref(x_162);
x_269 = lp_mathlib_CancelDenoms_findCancelFactor___closed__15;
x_270 = lean_string_dec_eq(x_161, x_269);
lean_dec_ref(x_161);
if (x_270 == 0)
{
lean_dec(x_160);
goto block_3;
}
else
{
lean_object* x_271; lean_object* x_272; uint8_t x_273; 
x_271 = lean_array_get_size(x_160);
x_272 = lean_unsigned_to_nat(6u);
x_273 = lean_nat_dec_eq(x_271, x_272);
if (x_273 == 0)
{
lean_dec(x_160);
goto block_3;
}
else
{
lean_object* x_274; lean_object* x_275; lean_object* x_276; lean_object* x_277; 
x_274 = lean_unsigned_to_nat(4u);
x_275 = lean_array_fget(x_160, x_274);
x_276 = lean_unsigned_to_nat(5u);
x_277 = lean_array_fget(x_160, x_276);
lean_dec(x_160);
x_4 = x_275;
x_5 = x_277;
goto block_20;
}
}
}
}
}
else
{
lean_dec_ref(x_23);
lean_dec_ref(x_22);
lean_dec_ref(x_21);
goto block_3;
}
}
else
{
lean_dec(x_23);
lean_dec_ref(x_22);
lean_dec_ref(x_21);
goto block_3;
}
}
else
{
lean_dec(x_22);
lean_dec_ref(x_21);
goto block_3;
}
block_3:
{
lean_object* x_2; 
x_2 = lp_mathlib_CancelDenoms_findCancelFactor___closed__1;
return x_2;
}
block_20:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_6 = lp_mathlib_CancelDenoms_findCancelFactor(x_4);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_6, 1);
lean_inc(x_8);
lean_dec_ref(x_6);
x_9 = lp_mathlib_CancelDenoms_findCancelFactor(x_5);
x_10 = !lean_is_exclusive(x_9);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_11 = lean_ctor_get(x_9, 0);
x_12 = lean_ctor_get(x_9, 1);
x_13 = l_Nat_lcm(x_7, x_11);
lean_dec(x_11);
lean_dec(x_7);
lean_inc(x_13);
x_14 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_14, 0, x_13);
lean_ctor_set(x_14, 1, x_8);
lean_ctor_set(x_14, 2, x_12);
lean_ctor_set(x_9, 1, x_14);
lean_ctor_set(x_9, 0, x_13);
return x_9;
}
else
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_15 = lean_ctor_get(x_9, 0);
x_16 = lean_ctor_get(x_9, 1);
lean_inc(x_16);
lean_inc(x_15);
lean_dec(x_9);
x_17 = l_Nat_lcm(x_7, x_15);
lean_dec(x_15);
lean_dec(x_7);
lean_inc(x_17);
x_18 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_18, 0, x_17);
lean_ctor_set(x_18, 1, x_8);
lean_ctor_set(x_18, 2, x_16);
x_19 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_19, 0, x_17);
lean_ctor_set(x_19, 1, x_18);
return x_19;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00CancelDenoms_synthesizeUsingNormNum_spec__0_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00CancelDenoms_synthesizeUsingNormNum_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_7 = lean_ctor_get(x_4, 5);
x_8 = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00CancelDenoms_synthesizeUsingNormNum_spec__0_spec__0(x_1, x_2, x_3, x_4, x_5);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00CancelDenoms_synthesizeUsingNormNum_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Lean_throwError___at___00CancelDenoms_synthesizeUsingNormNum_spec__0___redArg(x_2, x_3, x_4, x_5, x_6);
return x_8;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("normNum", 7, 7);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__0;
x_2 = lp_mathlib_initFn___closed__8_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_3 = lp_mathlib_initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_4 = l_Lean_Name_mkStr3(x_3, x_2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("norm_num", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Lean", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Parser", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("optConfig", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__5;
x_2 = lp_mathlib_initFn___closed__8_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_3 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__4;
x_4 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__3;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__7() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("null", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__8() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__7;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__9() {
_start:
{
lean_object* x_1; 
x_1 = l_Array_mkArray0(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__10() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Could not prove ", 16, 16);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__11() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__10;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__12() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(" using norm_num. ", 17, 17);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__13() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__12;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_synthesizeUsingNormNum(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; uint8_t x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_7 = lean_ctor_get(x_4, 5);
x_8 = 0;
x_9 = l_Lean_SourceInfo_fromRef(x_7, x_8);
x_10 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__1;
x_11 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__2;
lean_inc(x_9);
x_12 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_12, 0, x_9);
lean_ctor_set(x_12, 1, x_11);
x_13 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__6;
x_14 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__8;
x_15 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__9;
lean_inc(x_9);
x_16 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_16, 0, x_9);
lean_ctor_set(x_16, 1, x_14);
lean_ctor_set(x_16, 2, x_15);
lean_inc_ref(x_16);
lean_inc(x_9);
x_17 = l_Lean_Syntax_node1(x_9, x_13, x_16);
lean_inc_ref_n(x_16, 2);
x_18 = l_Lean_Syntax_node5(x_9, x_10, x_12, x_17, x_16, x_16, x_16);
lean_inc(x_5);
lean_inc_ref(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_19 = lp_mathlib_synthesizeUsingTactic_x27___redArg(x_1, x_18, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_19) == 0)
{
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_19;
}
else
{
lean_object* x_20; uint8_t x_21; uint8_t x_31; 
x_20 = lean_ctor_get(x_19, 0);
lean_inc(x_20);
x_31 = l_Lean_Exception_isInterrupt(x_20);
if (x_31 == 0)
{
uint8_t x_32; 
lean_inc(x_20);
x_32 = l_Lean_Exception_isRuntime(x_20);
x_21 = x_32;
goto block_30;
}
else
{
x_21 = x_31;
goto block_30;
}
block_30:
{
if (x_21 == 0)
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
lean_dec_ref(x_19);
x_22 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__11;
x_23 = l_Lean_MessageData_ofExpr(x_1);
x_24 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_24, 0, x_22);
lean_ctor_set(x_24, 1, x_23);
x_25 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__13;
x_26 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_26, 0, x_24);
lean_ctor_set(x_26, 1, x_25);
x_27 = l_Lean_Exception_toMessageData(x_20);
x_28 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_28, 0, x_26);
lean_ctor_set(x_28, 1, x_27);
x_29 = lp_mathlib_Lean_throwError___at___00CancelDenoms_synthesizeUsingNormNum_spec__0___redArg(x_28, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_29;
}
else
{
lean_dec(x_20);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_19;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00CancelDenoms_synthesizeUsingNormNum_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Lean_throwError___at___00CancelDenoms_synthesizeUsingNormNum_spec__0(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00CancelDenoms_synthesizeUsingNormNum_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Lean_throwError___at___00CancelDenoms_synthesizeUsingNormNum_spec__0___redArg(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00CancelDenoms_synthesizeUsingNormNum_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00CancelDenoms_synthesizeUsingNormNum_spec__0_spec__0(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_synthesizeUsingNormNum___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum(x_1, x_2, x_3, x_4, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00CancelDenoms_mkProdPrf_spec__2___redArg(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00CancelDenoms_mkProdPrf_spec__2(lean_object* x_1, lean_object* x_2, uint8_t x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00CancelDenoms_mkProdPrf_spec__2___redArg(x_2, x_3, x_4, x_5, x_6, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(lean_object* x_1, lean_object* x_2) {
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_1, x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_isTracingEnabledFor___at___00CancelDenoms_mkProdPrf_spec__0___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_4; lean_object* x_5; uint8_t x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_ctor_get(x_2, 2);
x_5 = lean_ctor_get(x_2, 13);
x_6 = l_Lean_checkTraceOption(x_5, x_4, x_1);
x_7 = lean_box(x_6);
x_8 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_8, 0, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_isTracingEnabledFor___at___00CancelDenoms_mkProdPrf_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Lean_isTracingEnabledFor___at___00CancelDenoms_mkProdPrf_spec__0___redArg(x_1, x_4);
return x_7;
}
}
static double _init_lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3___closed__0() {
_start:
{
lean_object* x_1; double x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_float_of_nat(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("", 0, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_8 = lean_ctor_get(x_5, 5);
x_9 = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00CancelDenoms_synthesizeUsingNormNum_spec__0_spec__0(x_2, x_3, x_4, x_5, x_6);
x_10 = !lean_is_exclusive(x_9);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; uint8_t x_13; 
x_11 = lean_ctor_get(x_9, 0);
x_12 = lean_st_ref_take(x_6);
x_13 = !lean_is_exclusive(x_12);
if (x_13 == 0)
{
lean_object* x_14; uint8_t x_15; 
x_14 = lean_ctor_get(x_12, 4);
x_15 = !lean_is_exclusive(x_14);
if (x_15 == 0)
{
lean_object* x_16; double x_17; uint8_t x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; 
x_16 = lean_ctor_get(x_14, 0);
x_17 = lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3___closed__0;
x_18 = 0;
x_19 = lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3___closed__1;
x_20 = lean_alloc_ctor(0, 2, 17);
lean_ctor_set(x_20, 0, x_1);
lean_ctor_set(x_20, 1, x_19);
lean_ctor_set_float(x_20, sizeof(void*)*2, x_17);
lean_ctor_set_float(x_20, sizeof(void*)*2 + 8, x_17);
lean_ctor_set_uint8(x_20, sizeof(void*)*2 + 16, x_18);
x_21 = lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3___closed__2;
x_22 = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(x_22, 0, x_20);
lean_ctor_set(x_22, 1, x_11);
lean_ctor_set(x_22, 2, x_21);
lean_inc(x_8);
x_23 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_23, 0, x_8);
lean_ctor_set(x_23, 1, x_22);
x_24 = l_Lean_PersistentArray_push___redArg(x_16, x_23);
lean_ctor_set(x_14, 0, x_24);
x_25 = lean_st_ref_set(x_6, x_12);
x_26 = lean_box(0);
lean_ctor_set(x_9, 0, x_26);
return x_9;
}
else
{
uint64_t x_27; lean_object* x_28; double x_29; uint8_t x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; 
x_27 = lean_ctor_get_uint64(x_14, sizeof(void*)*1);
x_28 = lean_ctor_get(x_14, 0);
lean_inc(x_28);
lean_dec(x_14);
x_29 = lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3___closed__0;
x_30 = 0;
x_31 = lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3___closed__1;
x_32 = lean_alloc_ctor(0, 2, 17);
lean_ctor_set(x_32, 0, x_1);
lean_ctor_set(x_32, 1, x_31);
lean_ctor_set_float(x_32, sizeof(void*)*2, x_29);
lean_ctor_set_float(x_32, sizeof(void*)*2 + 8, x_29);
lean_ctor_set_uint8(x_32, sizeof(void*)*2 + 16, x_30);
x_33 = lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3___closed__2;
x_34 = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(x_34, 0, x_32);
lean_ctor_set(x_34, 1, x_11);
lean_ctor_set(x_34, 2, x_33);
lean_inc(x_8);
x_35 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_35, 0, x_8);
lean_ctor_set(x_35, 1, x_34);
x_36 = l_Lean_PersistentArray_push___redArg(x_28, x_35);
x_37 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_37, 0, x_36);
lean_ctor_set_uint64(x_37, sizeof(void*)*1, x_27);
lean_ctor_set(x_12, 4, x_37);
x_38 = lean_st_ref_set(x_6, x_12);
x_39 = lean_box(0);
lean_ctor_set(x_9, 0, x_39);
return x_9;
}
}
else
{
lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; uint64_t x_49; lean_object* x_50; lean_object* x_51; double x_52; uint8_t x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; 
x_40 = lean_ctor_get(x_12, 4);
x_41 = lean_ctor_get(x_12, 0);
x_42 = lean_ctor_get(x_12, 1);
x_43 = lean_ctor_get(x_12, 2);
x_44 = lean_ctor_get(x_12, 3);
x_45 = lean_ctor_get(x_12, 5);
x_46 = lean_ctor_get(x_12, 6);
x_47 = lean_ctor_get(x_12, 7);
x_48 = lean_ctor_get(x_12, 8);
lean_inc(x_48);
lean_inc(x_47);
lean_inc(x_46);
lean_inc(x_45);
lean_inc(x_40);
lean_inc(x_44);
lean_inc(x_43);
lean_inc(x_42);
lean_inc(x_41);
lean_dec(x_12);
x_49 = lean_ctor_get_uint64(x_40, sizeof(void*)*1);
x_50 = lean_ctor_get(x_40, 0);
lean_inc_ref(x_50);
if (lean_is_exclusive(x_40)) {
 lean_ctor_release(x_40, 0);
 x_51 = x_40;
} else {
 lean_dec_ref(x_40);
 x_51 = lean_box(0);
}
x_52 = lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3___closed__0;
x_53 = 0;
x_54 = lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3___closed__1;
x_55 = lean_alloc_ctor(0, 2, 17);
lean_ctor_set(x_55, 0, x_1);
lean_ctor_set(x_55, 1, x_54);
lean_ctor_set_float(x_55, sizeof(void*)*2, x_52);
lean_ctor_set_float(x_55, sizeof(void*)*2 + 8, x_52);
lean_ctor_set_uint8(x_55, sizeof(void*)*2 + 16, x_53);
x_56 = lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3___closed__2;
x_57 = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(x_57, 0, x_55);
lean_ctor_set(x_57, 1, x_11);
lean_ctor_set(x_57, 2, x_56);
lean_inc(x_8);
x_58 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_58, 0, x_8);
lean_ctor_set(x_58, 1, x_57);
x_59 = l_Lean_PersistentArray_push___redArg(x_50, x_58);
if (lean_is_scalar(x_51)) {
 x_60 = lean_alloc_ctor(0, 1, 8);
} else {
 x_60 = x_51;
}
lean_ctor_set(x_60, 0, x_59);
lean_ctor_set_uint64(x_60, sizeof(void*)*1, x_49);
x_61 = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(x_61, 0, x_41);
lean_ctor_set(x_61, 1, x_42);
lean_ctor_set(x_61, 2, x_43);
lean_ctor_set(x_61, 3, x_44);
lean_ctor_set(x_61, 4, x_60);
lean_ctor_set(x_61, 5, x_45);
lean_ctor_set(x_61, 6, x_46);
lean_ctor_set(x_61, 7, x_47);
lean_ctor_set(x_61, 8, x_48);
x_62 = lean_st_ref_set(x_6, x_61);
x_63 = lean_box(0);
lean_ctor_set(x_9, 0, x_63);
return x_9;
}
}
else
{
lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; uint64_t x_76; lean_object* x_77; lean_object* x_78; double x_79; uint8_t x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; 
x_64 = lean_ctor_get(x_9, 0);
lean_inc(x_64);
lean_dec(x_9);
x_65 = lean_st_ref_take(x_6);
x_66 = lean_ctor_get(x_65, 4);
lean_inc_ref(x_66);
x_67 = lean_ctor_get(x_65, 0);
lean_inc_ref(x_67);
x_68 = lean_ctor_get(x_65, 1);
lean_inc(x_68);
x_69 = lean_ctor_get(x_65, 2);
lean_inc_ref(x_69);
x_70 = lean_ctor_get(x_65, 3);
lean_inc_ref(x_70);
x_71 = lean_ctor_get(x_65, 5);
lean_inc_ref(x_71);
x_72 = lean_ctor_get(x_65, 6);
lean_inc_ref(x_72);
x_73 = lean_ctor_get(x_65, 7);
lean_inc_ref(x_73);
x_74 = lean_ctor_get(x_65, 8);
lean_inc_ref(x_74);
if (lean_is_exclusive(x_65)) {
 lean_ctor_release(x_65, 0);
 lean_ctor_release(x_65, 1);
 lean_ctor_release(x_65, 2);
 lean_ctor_release(x_65, 3);
 lean_ctor_release(x_65, 4);
 lean_ctor_release(x_65, 5);
 lean_ctor_release(x_65, 6);
 lean_ctor_release(x_65, 7);
 lean_ctor_release(x_65, 8);
 x_75 = x_65;
} else {
 lean_dec_ref(x_65);
 x_75 = lean_box(0);
}
x_76 = lean_ctor_get_uint64(x_66, sizeof(void*)*1);
x_77 = lean_ctor_get(x_66, 0);
lean_inc_ref(x_77);
if (lean_is_exclusive(x_66)) {
 lean_ctor_release(x_66, 0);
 x_78 = x_66;
} else {
 lean_dec_ref(x_66);
 x_78 = lean_box(0);
}
x_79 = lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3___closed__0;
x_80 = 0;
x_81 = lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3___closed__1;
x_82 = lean_alloc_ctor(0, 2, 17);
lean_ctor_set(x_82, 0, x_1);
lean_ctor_set(x_82, 1, x_81);
lean_ctor_set_float(x_82, sizeof(void*)*2, x_79);
lean_ctor_set_float(x_82, sizeof(void*)*2 + 8, x_79);
lean_ctor_set_uint8(x_82, sizeof(void*)*2 + 16, x_80);
x_83 = lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3___closed__2;
x_84 = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(x_84, 0, x_82);
lean_ctor_set(x_84, 1, x_64);
lean_ctor_set(x_84, 2, x_83);
lean_inc(x_8);
x_85 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_85, 0, x_8);
lean_ctor_set(x_85, 1, x_84);
x_86 = l_Lean_PersistentArray_push___redArg(x_77, x_85);
if (lean_is_scalar(x_78)) {
 x_87 = lean_alloc_ctor(0, 1, 8);
} else {
 x_87 = x_78;
}
lean_ctor_set(x_87, 0, x_86);
lean_ctor_set_uint64(x_87, sizeof(void*)*1, x_76);
if (lean_is_scalar(x_75)) {
 x_88 = lean_alloc_ctor(0, 9, 0);
} else {
 x_88 = x_75;
}
lean_ctor_set(x_88, 0, x_67);
lean_ctor_set(x_88, 1, x_68);
lean_ctor_set(x_88, 2, x_69);
lean_ctor_set(x_88, 3, x_70);
lean_ctor_set(x_88, 4, x_87);
lean_ctor_set(x_88, 5, x_71);
lean_ctor_set(x_88, 6, x_72);
lean_ctor_set(x_88, 7, x_73);
lean_ctor_set(x_88, 8, x_74);
x_89 = lean_st_ref_set(x_6, x_88);
x_90 = lean_box(0);
x_91 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_91, 0, x_90);
return x_91;
}
}
}
static uint64_t _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__4___closed__0() {
_start:
{
uint8_t x_1; uint64_t x_2; 
x_1 = 2;
x_2 = l_Lean_Meta_TransparencyMode_toUInt64(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__4(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, uint8_t x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
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
uint8_t x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; uint8_t x_25; uint8_t x_26; lean_object* x_27; lean_object* x_28; uint8_t x_29; uint64_t x_30; uint8_t x_31; 
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
lean_inc(x_13);
x_27 = l_Lean_Expr_app___override(x_4, x_13);
lean_inc(x_15);
x_28 = l_Lean_Expr_app___override(x_27, x_15);
x_29 = 2;
lean_ctor_set_uint8(x_16, 9, x_29);
x_30 = l_Lean_Meta_Context_configKey(x_7);
x_31 = !lean_is_exclusive(x_7);
if (x_31 == 0)
{
lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; uint64_t x_39; uint64_t x_40; uint64_t x_41; uint64_t x_42; uint64_t x_43; lean_object* x_44; lean_object* x_45; 
x_32 = lean_ctor_get(x_7, 6);
lean_dec(x_32);
x_33 = lean_ctor_get(x_7, 5);
lean_dec(x_33);
x_34 = lean_ctor_get(x_7, 4);
lean_dec(x_34);
x_35 = lean_ctor_get(x_7, 3);
lean_dec(x_35);
x_36 = lean_ctor_get(x_7, 2);
lean_dec(x_36);
x_37 = lean_ctor_get(x_7, 1);
lean_dec(x_37);
x_38 = lean_ctor_get(x_7, 0);
lean_dec(x_38);
x_39 = 2;
x_40 = lean_uint64_shift_right(x_30, x_39);
x_41 = lean_uint64_shift_left(x_40, x_39);
x_42 = lp_mathlib_CancelDenoms_mkProdPrf___lam__4___closed__0;
x_43 = lean_uint64_lor(x_41, x_42);
x_44 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_44, 0, x_16);
lean_ctor_set_uint64(x_44, sizeof(void*)*1, x_43);
lean_ctor_set(x_7, 0, x_44);
lean_inc(x_8);
x_45 = l_Lean_Meta_isExprDefEq(x_28, x_5, x_7, x_8, x_9, x_10);
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
lean_dec(x_8);
x_49 = lean_box(x_6);
x_50 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_50, 0, x_15);
lean_ctor_set(x_50, 1, x_49);
x_51 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_51, 0, x_13);
lean_ctor_set(x_51, 1, x_50);
lean_ctor_set(x_45, 0, x_51);
return x_45;
}
else
{
lean_object* x_52; 
lean_free_object(x_45);
x_52 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_13, x_8);
if (lean_obj_tag(x_52) == 0)
{
lean_object* x_53; lean_object* x_54; 
x_53 = lean_ctor_get(x_52, 0);
lean_inc(x_53);
lean_dec_ref(x_52);
x_54 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_15, x_8);
lean_dec(x_8);
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
lean_dec(x_15);
lean_dec(x_8);
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
lean_dec(x_8);
x_71 = lean_box(x_6);
x_72 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_72, 0, x_15);
lean_ctor_set(x_72, 1, x_71);
x_73 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_73, 0, x_13);
lean_ctor_set(x_73, 1, x_72);
x_74 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_74, 0, x_73);
return x_74;
}
else
{
lean_object* x_75; 
x_75 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_13, x_8);
if (lean_obj_tag(x_75) == 0)
{
lean_object* x_76; lean_object* x_77; 
x_76 = lean_ctor_get(x_75, 0);
lean_inc(x_76);
lean_dec_ref(x_75);
x_77 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_15, x_8);
lean_dec(x_8);
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
lean_dec(x_15);
lean_dec(x_8);
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
lean_dec(x_15);
lean_dec(x_13);
lean_dec(x_8);
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
lean_dec(x_7);
x_92 = 2;
x_93 = lean_uint64_shift_right(x_30, x_92);
x_94 = lean_uint64_shift_left(x_93, x_92);
x_95 = lp_mathlib_CancelDenoms_mkProdPrf___lam__4___closed__0;
x_96 = lean_uint64_lor(x_94, x_95);
x_97 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_97, 0, x_16);
lean_ctor_set_uint64(x_97, sizeof(void*)*1, x_96);
x_98 = lean_alloc_ctor(0, 7, 3);
lean_ctor_set(x_98, 0, x_97);
lean_ctor_set(x_98, 1, x_19);
lean_ctor_set(x_98, 2, x_20);
lean_ctor_set(x_98, 3, x_21);
lean_ctor_set(x_98, 4, x_22);
lean_ctor_set(x_98, 5, x_23);
lean_ctor_set(x_98, 6, x_24);
lean_ctor_set_uint8(x_98, sizeof(void*)*7, x_18);
lean_ctor_set_uint8(x_98, sizeof(void*)*7 + 1, x_25);
lean_ctor_set_uint8(x_98, sizeof(void*)*7 + 2, x_26);
lean_inc(x_8);
x_99 = l_Lean_Meta_isExprDefEq(x_28, x_5, x_98, x_8, x_9, x_10);
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
lean_dec(x_8);
x_103 = lean_box(x_6);
x_104 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_104, 0, x_15);
lean_ctor_set(x_104, 1, x_103);
x_105 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_105, 0, x_13);
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
x_107 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_13, x_8);
if (lean_obj_tag(x_107) == 0)
{
lean_object* x_108; lean_object* x_109; 
x_108 = lean_ctor_get(x_107, 0);
lean_inc(x_108);
lean_dec_ref(x_107);
x_109 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_15, x_8);
lean_dec(x_8);
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
lean_dec(x_15);
lean_dec(x_8);
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
lean_dec(x_15);
lean_dec(x_13);
lean_dec(x_8);
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
uint8_t x_124; uint8_t x_125; uint8_t x_126; uint8_t x_127; uint8_t x_128; uint8_t x_129; uint8_t x_130; uint8_t x_131; uint8_t x_132; uint8_t x_133; uint8_t x_134; uint8_t x_135; uint8_t x_136; uint8_t x_137; uint8_t x_138; uint8_t x_139; uint8_t x_140; uint8_t x_141; uint8_t x_142; lean_object* x_143; lean_object* x_144; lean_object* x_145; lean_object* x_146; lean_object* x_147; lean_object* x_148; uint8_t x_149; uint8_t x_150; lean_object* x_151; lean_object* x_152; uint8_t x_153; lean_object* x_154; uint64_t x_155; lean_object* x_156; uint64_t x_157; uint64_t x_158; uint64_t x_159; uint64_t x_160; uint64_t x_161; lean_object* x_162; lean_object* x_163; lean_object* x_164; 
x_124 = lean_ctor_get_uint8(x_16, 0);
x_125 = lean_ctor_get_uint8(x_16, 1);
x_126 = lean_ctor_get_uint8(x_16, 2);
x_127 = lean_ctor_get_uint8(x_16, 3);
x_128 = lean_ctor_get_uint8(x_16, 4);
x_129 = lean_ctor_get_uint8(x_16, 5);
x_130 = lean_ctor_get_uint8(x_16, 6);
x_131 = lean_ctor_get_uint8(x_16, 7);
x_132 = lean_ctor_get_uint8(x_16, 8);
x_133 = lean_ctor_get_uint8(x_16, 10);
x_134 = lean_ctor_get_uint8(x_16, 11);
x_135 = lean_ctor_get_uint8(x_16, 12);
x_136 = lean_ctor_get_uint8(x_16, 13);
x_137 = lean_ctor_get_uint8(x_16, 14);
x_138 = lean_ctor_get_uint8(x_16, 15);
x_139 = lean_ctor_get_uint8(x_16, 16);
x_140 = lean_ctor_get_uint8(x_16, 17);
x_141 = lean_ctor_get_uint8(x_16, 18);
lean_dec(x_16);
x_142 = lean_ctor_get_uint8(x_7, sizeof(void*)*7);
x_143 = lean_ctor_get(x_7, 1);
lean_inc(x_143);
x_144 = lean_ctor_get(x_7, 2);
lean_inc_ref(x_144);
x_145 = lean_ctor_get(x_7, 3);
lean_inc_ref(x_145);
x_146 = lean_ctor_get(x_7, 4);
lean_inc(x_146);
x_147 = lean_ctor_get(x_7, 5);
lean_inc(x_147);
x_148 = lean_ctor_get(x_7, 6);
lean_inc(x_148);
x_149 = lean_ctor_get_uint8(x_7, sizeof(void*)*7 + 1);
x_150 = lean_ctor_get_uint8(x_7, sizeof(void*)*7 + 2);
lean_inc(x_13);
x_151 = l_Lean_Expr_app___override(x_4, x_13);
lean_inc(x_15);
x_152 = l_Lean_Expr_app___override(x_151, x_15);
x_153 = 2;
x_154 = lean_alloc_ctor(0, 0, 19);
lean_ctor_set_uint8(x_154, 0, x_124);
lean_ctor_set_uint8(x_154, 1, x_125);
lean_ctor_set_uint8(x_154, 2, x_126);
lean_ctor_set_uint8(x_154, 3, x_127);
lean_ctor_set_uint8(x_154, 4, x_128);
lean_ctor_set_uint8(x_154, 5, x_129);
lean_ctor_set_uint8(x_154, 6, x_130);
lean_ctor_set_uint8(x_154, 7, x_131);
lean_ctor_set_uint8(x_154, 8, x_132);
lean_ctor_set_uint8(x_154, 9, x_153);
lean_ctor_set_uint8(x_154, 10, x_133);
lean_ctor_set_uint8(x_154, 11, x_134);
lean_ctor_set_uint8(x_154, 12, x_135);
lean_ctor_set_uint8(x_154, 13, x_136);
lean_ctor_set_uint8(x_154, 14, x_137);
lean_ctor_set_uint8(x_154, 15, x_138);
lean_ctor_set_uint8(x_154, 16, x_139);
lean_ctor_set_uint8(x_154, 17, x_140);
lean_ctor_set_uint8(x_154, 18, x_141);
x_155 = l_Lean_Meta_Context_configKey(x_7);
if (lean_is_exclusive(x_7)) {
 lean_ctor_release(x_7, 0);
 lean_ctor_release(x_7, 1);
 lean_ctor_release(x_7, 2);
 lean_ctor_release(x_7, 3);
 lean_ctor_release(x_7, 4);
 lean_ctor_release(x_7, 5);
 lean_ctor_release(x_7, 6);
 x_156 = x_7;
} else {
 lean_dec_ref(x_7);
 x_156 = lean_box(0);
}
x_157 = 2;
x_158 = lean_uint64_shift_right(x_155, x_157);
x_159 = lean_uint64_shift_left(x_158, x_157);
x_160 = lp_mathlib_CancelDenoms_mkProdPrf___lam__4___closed__0;
x_161 = lean_uint64_lor(x_159, x_160);
x_162 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_162, 0, x_154);
lean_ctor_set_uint64(x_162, sizeof(void*)*1, x_161);
if (lean_is_scalar(x_156)) {
 x_163 = lean_alloc_ctor(0, 7, 3);
} else {
 x_163 = x_156;
}
lean_ctor_set(x_163, 0, x_162);
lean_ctor_set(x_163, 1, x_143);
lean_ctor_set(x_163, 2, x_144);
lean_ctor_set(x_163, 3, x_145);
lean_ctor_set(x_163, 4, x_146);
lean_ctor_set(x_163, 5, x_147);
lean_ctor_set(x_163, 6, x_148);
lean_ctor_set_uint8(x_163, sizeof(void*)*7, x_142);
lean_ctor_set_uint8(x_163, sizeof(void*)*7 + 1, x_149);
lean_ctor_set_uint8(x_163, sizeof(void*)*7 + 2, x_150);
lean_inc(x_8);
x_164 = l_Lean_Meta_isExprDefEq(x_152, x_5, x_163, x_8, x_9, x_10);
if (lean_obj_tag(x_164) == 0)
{
lean_object* x_165; lean_object* x_166; uint8_t x_167; 
x_165 = lean_ctor_get(x_164, 0);
lean_inc(x_165);
if (lean_is_exclusive(x_164)) {
 lean_ctor_release(x_164, 0);
 x_166 = x_164;
} else {
 lean_dec_ref(x_164);
 x_166 = lean_box(0);
}
x_167 = lean_unbox(x_165);
if (x_167 == 0)
{
lean_object* x_168; lean_object* x_169; lean_object* x_170; lean_object* x_171; 
lean_dec(x_165);
lean_dec(x_8);
x_168 = lean_box(x_6);
x_169 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_169, 0, x_15);
lean_ctor_set(x_169, 1, x_168);
x_170 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_170, 0, x_13);
lean_ctor_set(x_170, 1, x_169);
if (lean_is_scalar(x_166)) {
 x_171 = lean_alloc_ctor(0, 1, 0);
} else {
 x_171 = x_166;
}
lean_ctor_set(x_171, 0, x_170);
return x_171;
}
else
{
lean_object* x_172; 
lean_dec(x_166);
x_172 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_13, x_8);
if (lean_obj_tag(x_172) == 0)
{
lean_object* x_173; lean_object* x_174; 
x_173 = lean_ctor_get(x_172, 0);
lean_inc(x_173);
lean_dec_ref(x_172);
x_174 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_15, x_8);
lean_dec(x_8);
if (lean_obj_tag(x_174) == 0)
{
lean_object* x_175; lean_object* x_176; lean_object* x_177; lean_object* x_178; lean_object* x_179; 
x_175 = lean_ctor_get(x_174, 0);
lean_inc(x_175);
if (lean_is_exclusive(x_174)) {
 lean_ctor_release(x_174, 0);
 x_176 = x_174;
} else {
 lean_dec_ref(x_174);
 x_176 = lean_box(0);
}
x_177 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_177, 0, x_175);
lean_ctor_set(x_177, 1, x_165);
x_178 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_178, 0, x_173);
lean_ctor_set(x_178, 1, x_177);
if (lean_is_scalar(x_176)) {
 x_179 = lean_alloc_ctor(0, 1, 0);
} else {
 x_179 = x_176;
}
lean_ctor_set(x_179, 0, x_178);
return x_179;
}
else
{
lean_object* x_180; lean_object* x_181; lean_object* x_182; 
lean_dec(x_173);
lean_dec(x_165);
x_180 = lean_ctor_get(x_174, 0);
lean_inc(x_180);
if (lean_is_exclusive(x_174)) {
 lean_ctor_release(x_174, 0);
 x_181 = x_174;
} else {
 lean_dec_ref(x_174);
 x_181 = lean_box(0);
}
if (lean_is_scalar(x_181)) {
 x_182 = lean_alloc_ctor(1, 1, 0);
} else {
 x_182 = x_181;
}
lean_ctor_set(x_182, 0, x_180);
return x_182;
}
}
else
{
lean_object* x_183; lean_object* x_184; lean_object* x_185; 
lean_dec(x_165);
lean_dec(x_15);
lean_dec(x_8);
x_183 = lean_ctor_get(x_172, 0);
lean_inc(x_183);
if (lean_is_exclusive(x_172)) {
 lean_ctor_release(x_172, 0);
 x_184 = x_172;
} else {
 lean_dec_ref(x_172);
 x_184 = lean_box(0);
}
if (lean_is_scalar(x_184)) {
 x_185 = lean_alloc_ctor(1, 1, 0);
} else {
 x_185 = x_184;
}
lean_ctor_set(x_185, 0, x_183);
return x_185;
}
}
}
else
{
lean_object* x_186; lean_object* x_187; lean_object* x_188; 
lean_dec(x_15);
lean_dec(x_13);
lean_dec(x_8);
x_186 = lean_ctor_get(x_164, 0);
lean_inc(x_186);
if (lean_is_exclusive(x_164)) {
 lean_ctor_release(x_164, 0);
 x_187 = x_164;
} else {
 lean_dec_ref(x_164);
 x_187 = lean_box(0);
}
if (lean_is_scalar(x_187)) {
 x_188 = lean_alloc_ctor(1, 1, 0);
} else {
 x_188 = x_187;
}
lean_ctor_set(x_188, 0, x_186);
return x_188;
}
}
}
else
{
uint8_t x_189; 
lean_dec(x_13);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
x_189 = !lean_is_exclusive(x_14);
if (x_189 == 0)
{
return x_14;
}
else
{
lean_object* x_190; lean_object* x_191; 
x_190 = lean_ctor_get(x_14, 0);
lean_inc(x_190);
lean_dec(x_14);
x_191 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_191, 0, x_190);
return x_191;
}
}
}
else
{
uint8_t x_192; 
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec(x_1);
x_192 = !lean_is_exclusive(x_12);
if (x_192 == 0)
{
return x_12;
}
else
{
lean_object* x_193; lean_object* x_194; 
x_193 = lean_ctor_get(x_12, 0);
lean_inc(x_193);
lean_dec(x_12);
x_194 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_194, 0, x_193);
return x_194;
}
}
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_findCancelFactor___closed__15;
x_2 = lp_mathlib_CancelDenoms_findCancelFactor___closed__2;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("instHAdd", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__1;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toAdd", 5, 5);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__6(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, uint8_t x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_16; 
lean_inc_ref(x_11);
lean_inc(x_3);
lean_inc(x_1);
x_16 = l_Lean_Meta_mkFreshExprMVar(x_1, x_2, x_3, x_11, x_12, x_13, x_14);
if (lean_obj_tag(x_16) == 0)
{
lean_object* x_17; lean_object* x_18; 
x_17 = lean_ctor_get(x_16, 0);
lean_inc(x_17);
lean_dec_ref(x_16);
lean_inc_ref(x_11);
x_18 = l_Lean_Meta_mkFreshExprMVar(x_1, x_2, x_3, x_11, x_12, x_13, x_14);
if (lean_obj_tag(x_18) == 0)
{
lean_object* x_19; lean_object* x_20; uint8_t x_21; 
x_19 = lean_ctor_get(x_18, 0);
lean_inc(x_19);
lean_dec_ref(x_18);
x_20 = l_Lean_Meta_Context_config(x_11);
x_21 = !lean_is_exclusive(x_20);
if (x_21 == 0)
{
uint8_t x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; uint8_t x_29; uint8_t x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; uint8_t x_48; uint64_t x_49; uint8_t x_50; 
x_22 = lean_ctor_get_uint8(x_11, sizeof(void*)*7);
x_23 = lean_ctor_get(x_11, 1);
lean_inc(x_23);
x_24 = lean_ctor_get(x_11, 2);
lean_inc_ref(x_24);
x_25 = lean_ctor_get(x_11, 3);
lean_inc_ref(x_25);
x_26 = lean_ctor_get(x_11, 4);
lean_inc(x_26);
x_27 = lean_ctor_get(x_11, 5);
lean_inc(x_27);
x_28 = lean_ctor_get(x_11, 6);
lean_inc(x_28);
x_29 = lean_ctor_get_uint8(x_11, sizeof(void*)*7 + 1);
x_30 = lean_ctor_get_uint8(x_11, sizeof(void*)*7 + 2);
x_31 = lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__0;
x_32 = l_Lean_Expr_const___override(x_31, x_4);
lean_inc_ref(x_5);
x_33 = l_Lean_Expr_app___override(x_32, x_5);
lean_inc_ref(x_5);
x_34 = l_Lean_Expr_app___override(x_33, x_5);
lean_inc_ref(x_5);
x_35 = l_Lean_Expr_app___override(x_34, x_5);
x_36 = lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__2;
lean_inc(x_6);
x_37 = l_Lean_Expr_const___override(x_36, x_6);
lean_inc_ref(x_5);
x_38 = l_Lean_Expr_app___override(x_37, x_5);
x_39 = lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__3;
x_40 = l_Lean_Name_mkStr2(x_7, x_39);
x_41 = l_Lean_Expr_const___override(x_40, x_6);
x_42 = l_Lean_Expr_app___override(x_41, x_5);
x_43 = l_Lean_Expr_app___override(x_42, x_8);
x_44 = l_Lean_Expr_app___override(x_38, x_43);
x_45 = l_Lean_Expr_app___override(x_35, x_44);
lean_inc(x_17);
x_46 = l_Lean_Expr_app___override(x_45, x_17);
lean_inc(x_19);
x_47 = l_Lean_Expr_app___override(x_46, x_19);
x_48 = 2;
lean_ctor_set_uint8(x_20, 9, x_48);
x_49 = l_Lean_Meta_Context_configKey(x_11);
x_50 = !lean_is_exclusive(x_11);
if (x_50 == 0)
{
lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; uint64_t x_58; uint64_t x_59; uint64_t x_60; uint64_t x_61; uint64_t x_62; lean_object* x_63; lean_object* x_64; 
x_51 = lean_ctor_get(x_11, 6);
lean_dec(x_51);
x_52 = lean_ctor_get(x_11, 5);
lean_dec(x_52);
x_53 = lean_ctor_get(x_11, 4);
lean_dec(x_53);
x_54 = lean_ctor_get(x_11, 3);
lean_dec(x_54);
x_55 = lean_ctor_get(x_11, 2);
lean_dec(x_55);
x_56 = lean_ctor_get(x_11, 1);
lean_dec(x_56);
x_57 = lean_ctor_get(x_11, 0);
lean_dec(x_57);
x_58 = 2;
x_59 = lean_uint64_shift_right(x_49, x_58);
x_60 = lean_uint64_shift_left(x_59, x_58);
x_61 = lp_mathlib_CancelDenoms_mkProdPrf___lam__4___closed__0;
x_62 = lean_uint64_lor(x_60, x_61);
x_63 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_63, 0, x_20);
lean_ctor_set_uint64(x_63, sizeof(void*)*1, x_62);
lean_ctor_set(x_11, 0, x_63);
lean_inc(x_12);
x_64 = l_Lean_Meta_isExprDefEq(x_47, x_9, x_11, x_12, x_13, x_14);
if (lean_obj_tag(x_64) == 0)
{
uint8_t x_65; 
x_65 = !lean_is_exclusive(x_64);
if (x_65 == 0)
{
lean_object* x_66; uint8_t x_67; 
x_66 = lean_ctor_get(x_64, 0);
x_67 = lean_unbox(x_66);
if (x_67 == 0)
{
lean_object* x_68; lean_object* x_69; lean_object* x_70; 
lean_dec(x_66);
lean_dec(x_12);
x_68 = lean_box(x_10);
x_69 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_69, 0, x_19);
lean_ctor_set(x_69, 1, x_68);
x_70 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_70, 0, x_17);
lean_ctor_set(x_70, 1, x_69);
lean_ctor_set(x_64, 0, x_70);
return x_64;
}
else
{
lean_object* x_71; 
lean_free_object(x_64);
x_71 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_17, x_12);
if (lean_obj_tag(x_71) == 0)
{
lean_object* x_72; lean_object* x_73; 
x_72 = lean_ctor_get(x_71, 0);
lean_inc(x_72);
lean_dec_ref(x_71);
x_73 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_19, x_12);
lean_dec(x_12);
if (lean_obj_tag(x_73) == 0)
{
uint8_t x_74; 
x_74 = !lean_is_exclusive(x_73);
if (x_74 == 0)
{
lean_object* x_75; lean_object* x_76; lean_object* x_77; 
x_75 = lean_ctor_get(x_73, 0);
x_76 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_76, 0, x_75);
lean_ctor_set(x_76, 1, x_66);
x_77 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_77, 0, x_72);
lean_ctor_set(x_77, 1, x_76);
lean_ctor_set(x_73, 0, x_77);
return x_73;
}
else
{
lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; 
x_78 = lean_ctor_get(x_73, 0);
lean_inc(x_78);
lean_dec(x_73);
x_79 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_79, 0, x_78);
lean_ctor_set(x_79, 1, x_66);
x_80 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_80, 0, x_72);
lean_ctor_set(x_80, 1, x_79);
x_81 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_81, 0, x_80);
return x_81;
}
}
else
{
uint8_t x_82; 
lean_dec(x_72);
lean_dec(x_66);
x_82 = !lean_is_exclusive(x_73);
if (x_82 == 0)
{
return x_73;
}
else
{
lean_object* x_83; lean_object* x_84; 
x_83 = lean_ctor_get(x_73, 0);
lean_inc(x_83);
lean_dec(x_73);
x_84 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_84, 0, x_83);
return x_84;
}
}
}
else
{
uint8_t x_85; 
lean_dec(x_66);
lean_dec(x_19);
lean_dec(x_12);
x_85 = !lean_is_exclusive(x_71);
if (x_85 == 0)
{
return x_71;
}
else
{
lean_object* x_86; lean_object* x_87; 
x_86 = lean_ctor_get(x_71, 0);
lean_inc(x_86);
lean_dec(x_71);
x_87 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_87, 0, x_86);
return x_87;
}
}
}
}
else
{
lean_object* x_88; uint8_t x_89; 
x_88 = lean_ctor_get(x_64, 0);
lean_inc(x_88);
lean_dec(x_64);
x_89 = lean_unbox(x_88);
if (x_89 == 0)
{
lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; 
lean_dec(x_88);
lean_dec(x_12);
x_90 = lean_box(x_10);
x_91 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_91, 0, x_19);
lean_ctor_set(x_91, 1, x_90);
x_92 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_92, 0, x_17);
lean_ctor_set(x_92, 1, x_91);
x_93 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_93, 0, x_92);
return x_93;
}
else
{
lean_object* x_94; 
x_94 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_17, x_12);
if (lean_obj_tag(x_94) == 0)
{
lean_object* x_95; lean_object* x_96; 
x_95 = lean_ctor_get(x_94, 0);
lean_inc(x_95);
lean_dec_ref(x_94);
x_96 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_19, x_12);
lean_dec(x_12);
if (lean_obj_tag(x_96) == 0)
{
lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; 
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
lean_ctor_set(x_99, 1, x_88);
x_100 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_100, 0, x_95);
lean_ctor_set(x_100, 1, x_99);
if (lean_is_scalar(x_98)) {
 x_101 = lean_alloc_ctor(0, 1, 0);
} else {
 x_101 = x_98;
}
lean_ctor_set(x_101, 0, x_100);
return x_101;
}
else
{
lean_object* x_102; lean_object* x_103; lean_object* x_104; 
lean_dec(x_95);
lean_dec(x_88);
x_102 = lean_ctor_get(x_96, 0);
lean_inc(x_102);
if (lean_is_exclusive(x_96)) {
 lean_ctor_release(x_96, 0);
 x_103 = x_96;
} else {
 lean_dec_ref(x_96);
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
lean_dec(x_88);
lean_dec(x_19);
lean_dec(x_12);
x_105 = lean_ctor_get(x_94, 0);
lean_inc(x_105);
if (lean_is_exclusive(x_94)) {
 lean_ctor_release(x_94, 0);
 x_106 = x_94;
} else {
 lean_dec_ref(x_94);
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
}
}
else
{
uint8_t x_108; 
lean_dec(x_19);
lean_dec(x_17);
lean_dec(x_12);
x_108 = !lean_is_exclusive(x_64);
if (x_108 == 0)
{
return x_64;
}
else
{
lean_object* x_109; lean_object* x_110; 
x_109 = lean_ctor_get(x_64, 0);
lean_inc(x_109);
lean_dec(x_64);
x_110 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_110, 0, x_109);
return x_110;
}
}
}
else
{
uint64_t x_111; uint64_t x_112; uint64_t x_113; uint64_t x_114; uint64_t x_115; lean_object* x_116; lean_object* x_117; lean_object* x_118; 
lean_dec(x_11);
x_111 = 2;
x_112 = lean_uint64_shift_right(x_49, x_111);
x_113 = lean_uint64_shift_left(x_112, x_111);
x_114 = lp_mathlib_CancelDenoms_mkProdPrf___lam__4___closed__0;
x_115 = lean_uint64_lor(x_113, x_114);
x_116 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_116, 0, x_20);
lean_ctor_set_uint64(x_116, sizeof(void*)*1, x_115);
x_117 = lean_alloc_ctor(0, 7, 3);
lean_ctor_set(x_117, 0, x_116);
lean_ctor_set(x_117, 1, x_23);
lean_ctor_set(x_117, 2, x_24);
lean_ctor_set(x_117, 3, x_25);
lean_ctor_set(x_117, 4, x_26);
lean_ctor_set(x_117, 5, x_27);
lean_ctor_set(x_117, 6, x_28);
lean_ctor_set_uint8(x_117, sizeof(void*)*7, x_22);
lean_ctor_set_uint8(x_117, sizeof(void*)*7 + 1, x_29);
lean_ctor_set_uint8(x_117, sizeof(void*)*7 + 2, x_30);
lean_inc(x_12);
x_118 = l_Lean_Meta_isExprDefEq(x_47, x_9, x_117, x_12, x_13, x_14);
if (lean_obj_tag(x_118) == 0)
{
lean_object* x_119; lean_object* x_120; uint8_t x_121; 
x_119 = lean_ctor_get(x_118, 0);
lean_inc(x_119);
if (lean_is_exclusive(x_118)) {
 lean_ctor_release(x_118, 0);
 x_120 = x_118;
} else {
 lean_dec_ref(x_118);
 x_120 = lean_box(0);
}
x_121 = lean_unbox(x_119);
if (x_121 == 0)
{
lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; 
lean_dec(x_119);
lean_dec(x_12);
x_122 = lean_box(x_10);
x_123 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_123, 0, x_19);
lean_ctor_set(x_123, 1, x_122);
x_124 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_124, 0, x_17);
lean_ctor_set(x_124, 1, x_123);
if (lean_is_scalar(x_120)) {
 x_125 = lean_alloc_ctor(0, 1, 0);
} else {
 x_125 = x_120;
}
lean_ctor_set(x_125, 0, x_124);
return x_125;
}
else
{
lean_object* x_126; 
lean_dec(x_120);
x_126 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_17, x_12);
if (lean_obj_tag(x_126) == 0)
{
lean_object* x_127; lean_object* x_128; 
x_127 = lean_ctor_get(x_126, 0);
lean_inc(x_127);
lean_dec_ref(x_126);
x_128 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_19, x_12);
lean_dec(x_12);
if (lean_obj_tag(x_128) == 0)
{
lean_object* x_129; lean_object* x_130; lean_object* x_131; lean_object* x_132; lean_object* x_133; 
x_129 = lean_ctor_get(x_128, 0);
lean_inc(x_129);
if (lean_is_exclusive(x_128)) {
 lean_ctor_release(x_128, 0);
 x_130 = x_128;
} else {
 lean_dec_ref(x_128);
 x_130 = lean_box(0);
}
x_131 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_131, 0, x_129);
lean_ctor_set(x_131, 1, x_119);
x_132 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_132, 0, x_127);
lean_ctor_set(x_132, 1, x_131);
if (lean_is_scalar(x_130)) {
 x_133 = lean_alloc_ctor(0, 1, 0);
} else {
 x_133 = x_130;
}
lean_ctor_set(x_133, 0, x_132);
return x_133;
}
else
{
lean_object* x_134; lean_object* x_135; lean_object* x_136; 
lean_dec(x_127);
lean_dec(x_119);
x_134 = lean_ctor_get(x_128, 0);
lean_inc(x_134);
if (lean_is_exclusive(x_128)) {
 lean_ctor_release(x_128, 0);
 x_135 = x_128;
} else {
 lean_dec_ref(x_128);
 x_135 = lean_box(0);
}
if (lean_is_scalar(x_135)) {
 x_136 = lean_alloc_ctor(1, 1, 0);
} else {
 x_136 = x_135;
}
lean_ctor_set(x_136, 0, x_134);
return x_136;
}
}
else
{
lean_object* x_137; lean_object* x_138; lean_object* x_139; 
lean_dec(x_119);
lean_dec(x_19);
lean_dec(x_12);
x_137 = lean_ctor_get(x_126, 0);
lean_inc(x_137);
if (lean_is_exclusive(x_126)) {
 lean_ctor_release(x_126, 0);
 x_138 = x_126;
} else {
 lean_dec_ref(x_126);
 x_138 = lean_box(0);
}
if (lean_is_scalar(x_138)) {
 x_139 = lean_alloc_ctor(1, 1, 0);
} else {
 x_139 = x_138;
}
lean_ctor_set(x_139, 0, x_137);
return x_139;
}
}
}
else
{
lean_object* x_140; lean_object* x_141; lean_object* x_142; 
lean_dec(x_19);
lean_dec(x_17);
lean_dec(x_12);
x_140 = lean_ctor_get(x_118, 0);
lean_inc(x_140);
if (lean_is_exclusive(x_118)) {
 lean_ctor_release(x_118, 0);
 x_141 = x_118;
} else {
 lean_dec_ref(x_118);
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
}
else
{
uint8_t x_143; uint8_t x_144; uint8_t x_145; uint8_t x_146; uint8_t x_147; uint8_t x_148; uint8_t x_149; uint8_t x_150; uint8_t x_151; uint8_t x_152; uint8_t x_153; uint8_t x_154; uint8_t x_155; uint8_t x_156; uint8_t x_157; uint8_t x_158; uint8_t x_159; uint8_t x_160; uint8_t x_161; lean_object* x_162; lean_object* x_163; lean_object* x_164; lean_object* x_165; lean_object* x_166; lean_object* x_167; uint8_t x_168; uint8_t x_169; lean_object* x_170; lean_object* x_171; lean_object* x_172; lean_object* x_173; lean_object* x_174; lean_object* x_175; lean_object* x_176; lean_object* x_177; lean_object* x_178; lean_object* x_179; lean_object* x_180; lean_object* x_181; lean_object* x_182; lean_object* x_183; lean_object* x_184; lean_object* x_185; lean_object* x_186; uint8_t x_187; lean_object* x_188; uint64_t x_189; lean_object* x_190; uint64_t x_191; uint64_t x_192; uint64_t x_193; uint64_t x_194; uint64_t x_195; lean_object* x_196; lean_object* x_197; lean_object* x_198; 
x_143 = lean_ctor_get_uint8(x_20, 0);
x_144 = lean_ctor_get_uint8(x_20, 1);
x_145 = lean_ctor_get_uint8(x_20, 2);
x_146 = lean_ctor_get_uint8(x_20, 3);
x_147 = lean_ctor_get_uint8(x_20, 4);
x_148 = lean_ctor_get_uint8(x_20, 5);
x_149 = lean_ctor_get_uint8(x_20, 6);
x_150 = lean_ctor_get_uint8(x_20, 7);
x_151 = lean_ctor_get_uint8(x_20, 8);
x_152 = lean_ctor_get_uint8(x_20, 10);
x_153 = lean_ctor_get_uint8(x_20, 11);
x_154 = lean_ctor_get_uint8(x_20, 12);
x_155 = lean_ctor_get_uint8(x_20, 13);
x_156 = lean_ctor_get_uint8(x_20, 14);
x_157 = lean_ctor_get_uint8(x_20, 15);
x_158 = lean_ctor_get_uint8(x_20, 16);
x_159 = lean_ctor_get_uint8(x_20, 17);
x_160 = lean_ctor_get_uint8(x_20, 18);
lean_dec(x_20);
x_161 = lean_ctor_get_uint8(x_11, sizeof(void*)*7);
x_162 = lean_ctor_get(x_11, 1);
lean_inc(x_162);
x_163 = lean_ctor_get(x_11, 2);
lean_inc_ref(x_163);
x_164 = lean_ctor_get(x_11, 3);
lean_inc_ref(x_164);
x_165 = lean_ctor_get(x_11, 4);
lean_inc(x_165);
x_166 = lean_ctor_get(x_11, 5);
lean_inc(x_166);
x_167 = lean_ctor_get(x_11, 6);
lean_inc(x_167);
x_168 = lean_ctor_get_uint8(x_11, sizeof(void*)*7 + 1);
x_169 = lean_ctor_get_uint8(x_11, sizeof(void*)*7 + 2);
x_170 = lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__0;
x_171 = l_Lean_Expr_const___override(x_170, x_4);
lean_inc_ref(x_5);
x_172 = l_Lean_Expr_app___override(x_171, x_5);
lean_inc_ref(x_5);
x_173 = l_Lean_Expr_app___override(x_172, x_5);
lean_inc_ref(x_5);
x_174 = l_Lean_Expr_app___override(x_173, x_5);
x_175 = lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__2;
lean_inc(x_6);
x_176 = l_Lean_Expr_const___override(x_175, x_6);
lean_inc_ref(x_5);
x_177 = l_Lean_Expr_app___override(x_176, x_5);
x_178 = lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__3;
x_179 = l_Lean_Name_mkStr2(x_7, x_178);
x_180 = l_Lean_Expr_const___override(x_179, x_6);
x_181 = l_Lean_Expr_app___override(x_180, x_5);
x_182 = l_Lean_Expr_app___override(x_181, x_8);
x_183 = l_Lean_Expr_app___override(x_177, x_182);
x_184 = l_Lean_Expr_app___override(x_174, x_183);
lean_inc(x_17);
x_185 = l_Lean_Expr_app___override(x_184, x_17);
lean_inc(x_19);
x_186 = l_Lean_Expr_app___override(x_185, x_19);
x_187 = 2;
x_188 = lean_alloc_ctor(0, 0, 19);
lean_ctor_set_uint8(x_188, 0, x_143);
lean_ctor_set_uint8(x_188, 1, x_144);
lean_ctor_set_uint8(x_188, 2, x_145);
lean_ctor_set_uint8(x_188, 3, x_146);
lean_ctor_set_uint8(x_188, 4, x_147);
lean_ctor_set_uint8(x_188, 5, x_148);
lean_ctor_set_uint8(x_188, 6, x_149);
lean_ctor_set_uint8(x_188, 7, x_150);
lean_ctor_set_uint8(x_188, 8, x_151);
lean_ctor_set_uint8(x_188, 9, x_187);
lean_ctor_set_uint8(x_188, 10, x_152);
lean_ctor_set_uint8(x_188, 11, x_153);
lean_ctor_set_uint8(x_188, 12, x_154);
lean_ctor_set_uint8(x_188, 13, x_155);
lean_ctor_set_uint8(x_188, 14, x_156);
lean_ctor_set_uint8(x_188, 15, x_157);
lean_ctor_set_uint8(x_188, 16, x_158);
lean_ctor_set_uint8(x_188, 17, x_159);
lean_ctor_set_uint8(x_188, 18, x_160);
x_189 = l_Lean_Meta_Context_configKey(x_11);
if (lean_is_exclusive(x_11)) {
 lean_ctor_release(x_11, 0);
 lean_ctor_release(x_11, 1);
 lean_ctor_release(x_11, 2);
 lean_ctor_release(x_11, 3);
 lean_ctor_release(x_11, 4);
 lean_ctor_release(x_11, 5);
 lean_ctor_release(x_11, 6);
 x_190 = x_11;
} else {
 lean_dec_ref(x_11);
 x_190 = lean_box(0);
}
x_191 = 2;
x_192 = lean_uint64_shift_right(x_189, x_191);
x_193 = lean_uint64_shift_left(x_192, x_191);
x_194 = lp_mathlib_CancelDenoms_mkProdPrf___lam__4___closed__0;
x_195 = lean_uint64_lor(x_193, x_194);
x_196 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_196, 0, x_188);
lean_ctor_set_uint64(x_196, sizeof(void*)*1, x_195);
if (lean_is_scalar(x_190)) {
 x_197 = lean_alloc_ctor(0, 7, 3);
} else {
 x_197 = x_190;
}
lean_ctor_set(x_197, 0, x_196);
lean_ctor_set(x_197, 1, x_162);
lean_ctor_set(x_197, 2, x_163);
lean_ctor_set(x_197, 3, x_164);
lean_ctor_set(x_197, 4, x_165);
lean_ctor_set(x_197, 5, x_166);
lean_ctor_set(x_197, 6, x_167);
lean_ctor_set_uint8(x_197, sizeof(void*)*7, x_161);
lean_ctor_set_uint8(x_197, sizeof(void*)*7 + 1, x_168);
lean_ctor_set_uint8(x_197, sizeof(void*)*7 + 2, x_169);
lean_inc(x_12);
x_198 = l_Lean_Meta_isExprDefEq(x_186, x_9, x_197, x_12, x_13, x_14);
if (lean_obj_tag(x_198) == 0)
{
lean_object* x_199; lean_object* x_200; uint8_t x_201; 
x_199 = lean_ctor_get(x_198, 0);
lean_inc(x_199);
if (lean_is_exclusive(x_198)) {
 lean_ctor_release(x_198, 0);
 x_200 = x_198;
} else {
 lean_dec_ref(x_198);
 x_200 = lean_box(0);
}
x_201 = lean_unbox(x_199);
if (x_201 == 0)
{
lean_object* x_202; lean_object* x_203; lean_object* x_204; lean_object* x_205; 
lean_dec(x_199);
lean_dec(x_12);
x_202 = lean_box(x_10);
x_203 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_203, 0, x_19);
lean_ctor_set(x_203, 1, x_202);
x_204 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_204, 0, x_17);
lean_ctor_set(x_204, 1, x_203);
if (lean_is_scalar(x_200)) {
 x_205 = lean_alloc_ctor(0, 1, 0);
} else {
 x_205 = x_200;
}
lean_ctor_set(x_205, 0, x_204);
return x_205;
}
else
{
lean_object* x_206; 
lean_dec(x_200);
x_206 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_17, x_12);
if (lean_obj_tag(x_206) == 0)
{
lean_object* x_207; lean_object* x_208; 
x_207 = lean_ctor_get(x_206, 0);
lean_inc(x_207);
lean_dec_ref(x_206);
x_208 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_19, x_12);
lean_dec(x_12);
if (lean_obj_tag(x_208) == 0)
{
lean_object* x_209; lean_object* x_210; lean_object* x_211; lean_object* x_212; lean_object* x_213; 
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
lean_ctor_set(x_211, 1, x_199);
x_212 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_212, 0, x_207);
lean_ctor_set(x_212, 1, x_211);
if (lean_is_scalar(x_210)) {
 x_213 = lean_alloc_ctor(0, 1, 0);
} else {
 x_213 = x_210;
}
lean_ctor_set(x_213, 0, x_212);
return x_213;
}
else
{
lean_object* x_214; lean_object* x_215; lean_object* x_216; 
lean_dec(x_207);
lean_dec(x_199);
x_214 = lean_ctor_get(x_208, 0);
lean_inc(x_214);
if (lean_is_exclusive(x_208)) {
 lean_ctor_release(x_208, 0);
 x_215 = x_208;
} else {
 lean_dec_ref(x_208);
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
lean_dec(x_199);
lean_dec(x_19);
lean_dec(x_12);
x_217 = lean_ctor_get(x_206, 0);
lean_inc(x_217);
if (lean_is_exclusive(x_206)) {
 lean_ctor_release(x_206, 0);
 x_218 = x_206;
} else {
 lean_dec_ref(x_206);
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
}
else
{
lean_object* x_220; lean_object* x_221; lean_object* x_222; 
lean_dec(x_19);
lean_dec(x_17);
lean_dec(x_12);
x_220 = lean_ctor_get(x_198, 0);
lean_inc(x_220);
if (lean_is_exclusive(x_198)) {
 lean_ctor_release(x_198, 0);
 x_221 = x_198;
} else {
 lean_dec_ref(x_198);
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
uint8_t x_223; 
lean_dec(x_17);
lean_dec(x_14);
lean_dec_ref(x_13);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
x_223 = !lean_is_exclusive(x_18);
if (x_223 == 0)
{
return x_18;
}
else
{
lean_object* x_224; lean_object* x_225; 
x_224 = lean_ctor_get(x_18, 0);
lean_inc(x_224);
lean_dec(x_18);
x_225 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_225, 0, x_224);
return x_225;
}
}
}
else
{
uint8_t x_226; 
lean_dec(x_14);
lean_dec_ref(x_13);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec(x_1);
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
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_findCancelFactor___closed__12;
x_2 = lp_mathlib_CancelDenoms_findCancelFactor___closed__5;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("instHDiv", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__1;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("DivInvMonoid", 12, 12);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toDiv", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__4;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__3;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toDivInvMonoid", 14, 14);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__3(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, uint8_t x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_16; 
lean_inc_ref(x_11);
lean_inc(x_3);
lean_inc(x_1);
x_16 = l_Lean_Meta_mkFreshExprMVar(x_1, x_2, x_3, x_11, x_12, x_13, x_14);
if (lean_obj_tag(x_16) == 0)
{
lean_object* x_17; lean_object* x_18; 
x_17 = lean_ctor_get(x_16, 0);
lean_inc(x_17);
lean_dec_ref(x_16);
lean_inc_ref(x_11);
x_18 = l_Lean_Meta_mkFreshExprMVar(x_1, x_2, x_3, x_11, x_12, x_13, x_14);
if (lean_obj_tag(x_18) == 0)
{
lean_object* x_19; lean_object* x_20; uint8_t x_21; 
x_19 = lean_ctor_get(x_18, 0);
lean_inc(x_19);
lean_dec_ref(x_18);
x_20 = l_Lean_Meta_Context_config(x_11);
x_21 = !lean_is_exclusive(x_20);
if (x_21 == 0)
{
uint8_t x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; uint8_t x_29; uint8_t x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; uint8_t x_52; uint64_t x_53; uint8_t x_54; 
x_22 = lean_ctor_get_uint8(x_11, sizeof(void*)*7);
x_23 = lean_ctor_get(x_11, 1);
lean_inc(x_23);
x_24 = lean_ctor_get(x_11, 2);
lean_inc_ref(x_24);
x_25 = lean_ctor_get(x_11, 3);
lean_inc_ref(x_25);
x_26 = lean_ctor_get(x_11, 4);
lean_inc(x_26);
x_27 = lean_ctor_get(x_11, 5);
lean_inc(x_27);
x_28 = lean_ctor_get(x_11, 6);
lean_inc(x_28);
x_29 = lean_ctor_get_uint8(x_11, sizeof(void*)*7 + 1);
x_30 = lean_ctor_get_uint8(x_11, sizeof(void*)*7 + 2);
x_31 = lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__0;
x_32 = l_Lean_Expr_const___override(x_31, x_4);
lean_inc_ref(x_5);
x_33 = l_Lean_Expr_app___override(x_32, x_5);
lean_inc_ref(x_5);
x_34 = l_Lean_Expr_app___override(x_33, x_5);
lean_inc_ref(x_5);
x_35 = l_Lean_Expr_app___override(x_34, x_5);
x_36 = lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__2;
lean_inc(x_6);
x_37 = l_Lean_Expr_const___override(x_36, x_6);
lean_inc_ref(x_5);
x_38 = l_Lean_Expr_app___override(x_37, x_5);
x_39 = lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__5;
lean_inc(x_6);
x_40 = l_Lean_Expr_const___override(x_39, x_6);
lean_inc_ref(x_5);
x_41 = l_Lean_Expr_app___override(x_40, x_5);
x_42 = lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__6;
x_43 = l_Lean_Name_mkStr2(x_7, x_42);
x_44 = l_Lean_Expr_const___override(x_43, x_6);
x_45 = l_Lean_Expr_app___override(x_44, x_5);
x_46 = l_Lean_Expr_app___override(x_45, x_8);
x_47 = l_Lean_Expr_app___override(x_41, x_46);
x_48 = l_Lean_Expr_app___override(x_38, x_47);
x_49 = l_Lean_Expr_app___override(x_35, x_48);
lean_inc(x_17);
x_50 = l_Lean_Expr_app___override(x_49, x_17);
lean_inc(x_19);
x_51 = l_Lean_Expr_app___override(x_50, x_19);
x_52 = 2;
lean_ctor_set_uint8(x_20, 9, x_52);
x_53 = l_Lean_Meta_Context_configKey(x_11);
x_54 = !lean_is_exclusive(x_11);
if (x_54 == 0)
{
lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; uint64_t x_62; uint64_t x_63; uint64_t x_64; uint64_t x_65; uint64_t x_66; lean_object* x_67; lean_object* x_68; 
x_55 = lean_ctor_get(x_11, 6);
lean_dec(x_55);
x_56 = lean_ctor_get(x_11, 5);
lean_dec(x_56);
x_57 = lean_ctor_get(x_11, 4);
lean_dec(x_57);
x_58 = lean_ctor_get(x_11, 3);
lean_dec(x_58);
x_59 = lean_ctor_get(x_11, 2);
lean_dec(x_59);
x_60 = lean_ctor_get(x_11, 1);
lean_dec(x_60);
x_61 = lean_ctor_get(x_11, 0);
lean_dec(x_61);
x_62 = 2;
x_63 = lean_uint64_shift_right(x_53, x_62);
x_64 = lean_uint64_shift_left(x_63, x_62);
x_65 = lp_mathlib_CancelDenoms_mkProdPrf___lam__4___closed__0;
x_66 = lean_uint64_lor(x_64, x_65);
x_67 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_67, 0, x_20);
lean_ctor_set_uint64(x_67, sizeof(void*)*1, x_66);
lean_ctor_set(x_11, 0, x_67);
lean_inc(x_12);
x_68 = l_Lean_Meta_isExprDefEq(x_51, x_9, x_11, x_12, x_13, x_14);
if (lean_obj_tag(x_68) == 0)
{
uint8_t x_69; 
x_69 = !lean_is_exclusive(x_68);
if (x_69 == 0)
{
lean_object* x_70; uint8_t x_71; 
x_70 = lean_ctor_get(x_68, 0);
x_71 = lean_unbox(x_70);
if (x_71 == 0)
{
lean_object* x_72; lean_object* x_73; lean_object* x_74; 
lean_dec(x_70);
lean_dec(x_12);
x_72 = lean_box(x_10);
x_73 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_73, 0, x_19);
lean_ctor_set(x_73, 1, x_72);
x_74 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_74, 0, x_17);
lean_ctor_set(x_74, 1, x_73);
lean_ctor_set(x_68, 0, x_74);
return x_68;
}
else
{
lean_object* x_75; 
lean_free_object(x_68);
x_75 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_17, x_12);
if (lean_obj_tag(x_75) == 0)
{
lean_object* x_76; lean_object* x_77; 
x_76 = lean_ctor_get(x_75, 0);
lean_inc(x_76);
lean_dec_ref(x_75);
x_77 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_19, x_12);
lean_dec(x_12);
if (lean_obj_tag(x_77) == 0)
{
uint8_t x_78; 
x_78 = !lean_is_exclusive(x_77);
if (x_78 == 0)
{
lean_object* x_79; lean_object* x_80; lean_object* x_81; 
x_79 = lean_ctor_get(x_77, 0);
x_80 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_80, 0, x_79);
lean_ctor_set(x_80, 1, x_70);
x_81 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_81, 0, x_76);
lean_ctor_set(x_81, 1, x_80);
lean_ctor_set(x_77, 0, x_81);
return x_77;
}
else
{
lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; 
x_82 = lean_ctor_get(x_77, 0);
lean_inc(x_82);
lean_dec(x_77);
x_83 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_83, 0, x_82);
lean_ctor_set(x_83, 1, x_70);
x_84 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_84, 0, x_76);
lean_ctor_set(x_84, 1, x_83);
x_85 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_85, 0, x_84);
return x_85;
}
}
else
{
uint8_t x_86; 
lean_dec(x_76);
lean_dec(x_70);
x_86 = !lean_is_exclusive(x_77);
if (x_86 == 0)
{
return x_77;
}
else
{
lean_object* x_87; lean_object* x_88; 
x_87 = lean_ctor_get(x_77, 0);
lean_inc(x_87);
lean_dec(x_77);
x_88 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_88, 0, x_87);
return x_88;
}
}
}
else
{
uint8_t x_89; 
lean_dec(x_70);
lean_dec(x_19);
lean_dec(x_12);
x_89 = !lean_is_exclusive(x_75);
if (x_89 == 0)
{
return x_75;
}
else
{
lean_object* x_90; lean_object* x_91; 
x_90 = lean_ctor_get(x_75, 0);
lean_inc(x_90);
lean_dec(x_75);
x_91 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_91, 0, x_90);
return x_91;
}
}
}
}
else
{
lean_object* x_92; uint8_t x_93; 
x_92 = lean_ctor_get(x_68, 0);
lean_inc(x_92);
lean_dec(x_68);
x_93 = lean_unbox(x_92);
if (x_93 == 0)
{
lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; 
lean_dec(x_92);
lean_dec(x_12);
x_94 = lean_box(x_10);
x_95 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_95, 0, x_19);
lean_ctor_set(x_95, 1, x_94);
x_96 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_96, 0, x_17);
lean_ctor_set(x_96, 1, x_95);
x_97 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_97, 0, x_96);
return x_97;
}
else
{
lean_object* x_98; 
x_98 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_17, x_12);
if (lean_obj_tag(x_98) == 0)
{
lean_object* x_99; lean_object* x_100; 
x_99 = lean_ctor_get(x_98, 0);
lean_inc(x_99);
lean_dec_ref(x_98);
x_100 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_19, x_12);
lean_dec(x_12);
if (lean_obj_tag(x_100) == 0)
{
lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; lean_object* x_105; 
x_101 = lean_ctor_get(x_100, 0);
lean_inc(x_101);
if (lean_is_exclusive(x_100)) {
 lean_ctor_release(x_100, 0);
 x_102 = x_100;
} else {
 lean_dec_ref(x_100);
 x_102 = lean_box(0);
}
x_103 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_103, 0, x_101);
lean_ctor_set(x_103, 1, x_92);
x_104 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_104, 0, x_99);
lean_ctor_set(x_104, 1, x_103);
if (lean_is_scalar(x_102)) {
 x_105 = lean_alloc_ctor(0, 1, 0);
} else {
 x_105 = x_102;
}
lean_ctor_set(x_105, 0, x_104);
return x_105;
}
else
{
lean_object* x_106; lean_object* x_107; lean_object* x_108; 
lean_dec(x_99);
lean_dec(x_92);
x_106 = lean_ctor_get(x_100, 0);
lean_inc(x_106);
if (lean_is_exclusive(x_100)) {
 lean_ctor_release(x_100, 0);
 x_107 = x_100;
} else {
 lean_dec_ref(x_100);
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
lean_dec(x_92);
lean_dec(x_19);
lean_dec(x_12);
x_109 = lean_ctor_get(x_98, 0);
lean_inc(x_109);
if (lean_is_exclusive(x_98)) {
 lean_ctor_release(x_98, 0);
 x_110 = x_98;
} else {
 lean_dec_ref(x_98);
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
lean_dec(x_19);
lean_dec(x_17);
lean_dec(x_12);
x_112 = !lean_is_exclusive(x_68);
if (x_112 == 0)
{
return x_68;
}
else
{
lean_object* x_113; lean_object* x_114; 
x_113 = lean_ctor_get(x_68, 0);
lean_inc(x_113);
lean_dec(x_68);
x_114 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_114, 0, x_113);
return x_114;
}
}
}
else
{
uint64_t x_115; uint64_t x_116; uint64_t x_117; uint64_t x_118; uint64_t x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; 
lean_dec(x_11);
x_115 = 2;
x_116 = lean_uint64_shift_right(x_53, x_115);
x_117 = lean_uint64_shift_left(x_116, x_115);
x_118 = lp_mathlib_CancelDenoms_mkProdPrf___lam__4___closed__0;
x_119 = lean_uint64_lor(x_117, x_118);
x_120 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_120, 0, x_20);
lean_ctor_set_uint64(x_120, sizeof(void*)*1, x_119);
x_121 = lean_alloc_ctor(0, 7, 3);
lean_ctor_set(x_121, 0, x_120);
lean_ctor_set(x_121, 1, x_23);
lean_ctor_set(x_121, 2, x_24);
lean_ctor_set(x_121, 3, x_25);
lean_ctor_set(x_121, 4, x_26);
lean_ctor_set(x_121, 5, x_27);
lean_ctor_set(x_121, 6, x_28);
lean_ctor_set_uint8(x_121, sizeof(void*)*7, x_22);
lean_ctor_set_uint8(x_121, sizeof(void*)*7 + 1, x_29);
lean_ctor_set_uint8(x_121, sizeof(void*)*7 + 2, x_30);
lean_inc(x_12);
x_122 = l_Lean_Meta_isExprDefEq(x_51, x_9, x_121, x_12, x_13, x_14);
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
lean_object* x_126; lean_object* x_127; lean_object* x_128; lean_object* x_129; 
lean_dec(x_123);
lean_dec(x_12);
x_126 = lean_box(x_10);
x_127 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_127, 0, x_19);
lean_ctor_set(x_127, 1, x_126);
x_128 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_128, 0, x_17);
lean_ctor_set(x_128, 1, x_127);
if (lean_is_scalar(x_124)) {
 x_129 = lean_alloc_ctor(0, 1, 0);
} else {
 x_129 = x_124;
}
lean_ctor_set(x_129, 0, x_128);
return x_129;
}
else
{
lean_object* x_130; 
lean_dec(x_124);
x_130 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_17, x_12);
if (lean_obj_tag(x_130) == 0)
{
lean_object* x_131; lean_object* x_132; 
x_131 = lean_ctor_get(x_130, 0);
lean_inc(x_131);
lean_dec_ref(x_130);
x_132 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_19, x_12);
lean_dec(x_12);
if (lean_obj_tag(x_132) == 0)
{
lean_object* x_133; lean_object* x_134; lean_object* x_135; lean_object* x_136; lean_object* x_137; 
x_133 = lean_ctor_get(x_132, 0);
lean_inc(x_133);
if (lean_is_exclusive(x_132)) {
 lean_ctor_release(x_132, 0);
 x_134 = x_132;
} else {
 lean_dec_ref(x_132);
 x_134 = lean_box(0);
}
x_135 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_135, 0, x_133);
lean_ctor_set(x_135, 1, x_123);
x_136 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_136, 0, x_131);
lean_ctor_set(x_136, 1, x_135);
if (lean_is_scalar(x_134)) {
 x_137 = lean_alloc_ctor(0, 1, 0);
} else {
 x_137 = x_134;
}
lean_ctor_set(x_137, 0, x_136);
return x_137;
}
else
{
lean_object* x_138; lean_object* x_139; lean_object* x_140; 
lean_dec(x_131);
lean_dec(x_123);
x_138 = lean_ctor_get(x_132, 0);
lean_inc(x_138);
if (lean_is_exclusive(x_132)) {
 lean_ctor_release(x_132, 0);
 x_139 = x_132;
} else {
 lean_dec_ref(x_132);
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
lean_dec(x_123);
lean_dec(x_19);
lean_dec(x_12);
x_141 = lean_ctor_get(x_130, 0);
lean_inc(x_141);
if (lean_is_exclusive(x_130)) {
 lean_ctor_release(x_130, 0);
 x_142 = x_130;
} else {
 lean_dec_ref(x_130);
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
}
else
{
lean_object* x_144; lean_object* x_145; lean_object* x_146; 
lean_dec(x_19);
lean_dec(x_17);
lean_dec(x_12);
x_144 = lean_ctor_get(x_122, 0);
lean_inc(x_144);
if (lean_is_exclusive(x_122)) {
 lean_ctor_release(x_122, 0);
 x_145 = x_122;
} else {
 lean_dec_ref(x_122);
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
uint8_t x_147; uint8_t x_148; uint8_t x_149; uint8_t x_150; uint8_t x_151; uint8_t x_152; uint8_t x_153; uint8_t x_154; uint8_t x_155; uint8_t x_156; uint8_t x_157; uint8_t x_158; uint8_t x_159; uint8_t x_160; uint8_t x_161; uint8_t x_162; uint8_t x_163; uint8_t x_164; uint8_t x_165; lean_object* x_166; lean_object* x_167; lean_object* x_168; lean_object* x_169; lean_object* x_170; lean_object* x_171; uint8_t x_172; uint8_t x_173; lean_object* x_174; lean_object* x_175; lean_object* x_176; lean_object* x_177; lean_object* x_178; lean_object* x_179; lean_object* x_180; lean_object* x_181; lean_object* x_182; lean_object* x_183; lean_object* x_184; lean_object* x_185; lean_object* x_186; lean_object* x_187; lean_object* x_188; lean_object* x_189; lean_object* x_190; lean_object* x_191; lean_object* x_192; lean_object* x_193; lean_object* x_194; uint8_t x_195; lean_object* x_196; uint64_t x_197; lean_object* x_198; uint64_t x_199; uint64_t x_200; uint64_t x_201; uint64_t x_202; uint64_t x_203; lean_object* x_204; lean_object* x_205; lean_object* x_206; 
x_147 = lean_ctor_get_uint8(x_20, 0);
x_148 = lean_ctor_get_uint8(x_20, 1);
x_149 = lean_ctor_get_uint8(x_20, 2);
x_150 = lean_ctor_get_uint8(x_20, 3);
x_151 = lean_ctor_get_uint8(x_20, 4);
x_152 = lean_ctor_get_uint8(x_20, 5);
x_153 = lean_ctor_get_uint8(x_20, 6);
x_154 = lean_ctor_get_uint8(x_20, 7);
x_155 = lean_ctor_get_uint8(x_20, 8);
x_156 = lean_ctor_get_uint8(x_20, 10);
x_157 = lean_ctor_get_uint8(x_20, 11);
x_158 = lean_ctor_get_uint8(x_20, 12);
x_159 = lean_ctor_get_uint8(x_20, 13);
x_160 = lean_ctor_get_uint8(x_20, 14);
x_161 = lean_ctor_get_uint8(x_20, 15);
x_162 = lean_ctor_get_uint8(x_20, 16);
x_163 = lean_ctor_get_uint8(x_20, 17);
x_164 = lean_ctor_get_uint8(x_20, 18);
lean_dec(x_20);
x_165 = lean_ctor_get_uint8(x_11, sizeof(void*)*7);
x_166 = lean_ctor_get(x_11, 1);
lean_inc(x_166);
x_167 = lean_ctor_get(x_11, 2);
lean_inc_ref(x_167);
x_168 = lean_ctor_get(x_11, 3);
lean_inc_ref(x_168);
x_169 = lean_ctor_get(x_11, 4);
lean_inc(x_169);
x_170 = lean_ctor_get(x_11, 5);
lean_inc(x_170);
x_171 = lean_ctor_get(x_11, 6);
lean_inc(x_171);
x_172 = lean_ctor_get_uint8(x_11, sizeof(void*)*7 + 1);
x_173 = lean_ctor_get_uint8(x_11, sizeof(void*)*7 + 2);
x_174 = lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__0;
x_175 = l_Lean_Expr_const___override(x_174, x_4);
lean_inc_ref(x_5);
x_176 = l_Lean_Expr_app___override(x_175, x_5);
lean_inc_ref(x_5);
x_177 = l_Lean_Expr_app___override(x_176, x_5);
lean_inc_ref(x_5);
x_178 = l_Lean_Expr_app___override(x_177, x_5);
x_179 = lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__2;
lean_inc(x_6);
x_180 = l_Lean_Expr_const___override(x_179, x_6);
lean_inc_ref(x_5);
x_181 = l_Lean_Expr_app___override(x_180, x_5);
x_182 = lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__5;
lean_inc(x_6);
x_183 = l_Lean_Expr_const___override(x_182, x_6);
lean_inc_ref(x_5);
x_184 = l_Lean_Expr_app___override(x_183, x_5);
x_185 = lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__6;
x_186 = l_Lean_Name_mkStr2(x_7, x_185);
x_187 = l_Lean_Expr_const___override(x_186, x_6);
x_188 = l_Lean_Expr_app___override(x_187, x_5);
x_189 = l_Lean_Expr_app___override(x_188, x_8);
x_190 = l_Lean_Expr_app___override(x_184, x_189);
x_191 = l_Lean_Expr_app___override(x_181, x_190);
x_192 = l_Lean_Expr_app___override(x_178, x_191);
lean_inc(x_17);
x_193 = l_Lean_Expr_app___override(x_192, x_17);
lean_inc(x_19);
x_194 = l_Lean_Expr_app___override(x_193, x_19);
x_195 = 2;
x_196 = lean_alloc_ctor(0, 0, 19);
lean_ctor_set_uint8(x_196, 0, x_147);
lean_ctor_set_uint8(x_196, 1, x_148);
lean_ctor_set_uint8(x_196, 2, x_149);
lean_ctor_set_uint8(x_196, 3, x_150);
lean_ctor_set_uint8(x_196, 4, x_151);
lean_ctor_set_uint8(x_196, 5, x_152);
lean_ctor_set_uint8(x_196, 6, x_153);
lean_ctor_set_uint8(x_196, 7, x_154);
lean_ctor_set_uint8(x_196, 8, x_155);
lean_ctor_set_uint8(x_196, 9, x_195);
lean_ctor_set_uint8(x_196, 10, x_156);
lean_ctor_set_uint8(x_196, 11, x_157);
lean_ctor_set_uint8(x_196, 12, x_158);
lean_ctor_set_uint8(x_196, 13, x_159);
lean_ctor_set_uint8(x_196, 14, x_160);
lean_ctor_set_uint8(x_196, 15, x_161);
lean_ctor_set_uint8(x_196, 16, x_162);
lean_ctor_set_uint8(x_196, 17, x_163);
lean_ctor_set_uint8(x_196, 18, x_164);
x_197 = l_Lean_Meta_Context_configKey(x_11);
if (lean_is_exclusive(x_11)) {
 lean_ctor_release(x_11, 0);
 lean_ctor_release(x_11, 1);
 lean_ctor_release(x_11, 2);
 lean_ctor_release(x_11, 3);
 lean_ctor_release(x_11, 4);
 lean_ctor_release(x_11, 5);
 lean_ctor_release(x_11, 6);
 x_198 = x_11;
} else {
 lean_dec_ref(x_11);
 x_198 = lean_box(0);
}
x_199 = 2;
x_200 = lean_uint64_shift_right(x_197, x_199);
x_201 = lean_uint64_shift_left(x_200, x_199);
x_202 = lp_mathlib_CancelDenoms_mkProdPrf___lam__4___closed__0;
x_203 = lean_uint64_lor(x_201, x_202);
x_204 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_204, 0, x_196);
lean_ctor_set_uint64(x_204, sizeof(void*)*1, x_203);
if (lean_is_scalar(x_198)) {
 x_205 = lean_alloc_ctor(0, 7, 3);
} else {
 x_205 = x_198;
}
lean_ctor_set(x_205, 0, x_204);
lean_ctor_set(x_205, 1, x_166);
lean_ctor_set(x_205, 2, x_167);
lean_ctor_set(x_205, 3, x_168);
lean_ctor_set(x_205, 4, x_169);
lean_ctor_set(x_205, 5, x_170);
lean_ctor_set(x_205, 6, x_171);
lean_ctor_set_uint8(x_205, sizeof(void*)*7, x_165);
lean_ctor_set_uint8(x_205, sizeof(void*)*7 + 1, x_172);
lean_ctor_set_uint8(x_205, sizeof(void*)*7 + 2, x_173);
lean_inc(x_12);
x_206 = l_Lean_Meta_isExprDefEq(x_194, x_9, x_205, x_12, x_13, x_14);
if (lean_obj_tag(x_206) == 0)
{
lean_object* x_207; lean_object* x_208; uint8_t x_209; 
x_207 = lean_ctor_get(x_206, 0);
lean_inc(x_207);
if (lean_is_exclusive(x_206)) {
 lean_ctor_release(x_206, 0);
 x_208 = x_206;
} else {
 lean_dec_ref(x_206);
 x_208 = lean_box(0);
}
x_209 = lean_unbox(x_207);
if (x_209 == 0)
{
lean_object* x_210; lean_object* x_211; lean_object* x_212; lean_object* x_213; 
lean_dec(x_207);
lean_dec(x_12);
x_210 = lean_box(x_10);
x_211 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_211, 0, x_19);
lean_ctor_set(x_211, 1, x_210);
x_212 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_212, 0, x_17);
lean_ctor_set(x_212, 1, x_211);
if (lean_is_scalar(x_208)) {
 x_213 = lean_alloc_ctor(0, 1, 0);
} else {
 x_213 = x_208;
}
lean_ctor_set(x_213, 0, x_212);
return x_213;
}
else
{
lean_object* x_214; 
lean_dec(x_208);
x_214 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_17, x_12);
if (lean_obj_tag(x_214) == 0)
{
lean_object* x_215; lean_object* x_216; 
x_215 = lean_ctor_get(x_214, 0);
lean_inc(x_215);
lean_dec_ref(x_214);
x_216 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_19, x_12);
lean_dec(x_12);
if (lean_obj_tag(x_216) == 0)
{
lean_object* x_217; lean_object* x_218; lean_object* x_219; lean_object* x_220; lean_object* x_221; 
x_217 = lean_ctor_get(x_216, 0);
lean_inc(x_217);
if (lean_is_exclusive(x_216)) {
 lean_ctor_release(x_216, 0);
 x_218 = x_216;
} else {
 lean_dec_ref(x_216);
 x_218 = lean_box(0);
}
x_219 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_219, 0, x_217);
lean_ctor_set(x_219, 1, x_207);
x_220 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_220, 0, x_215);
lean_ctor_set(x_220, 1, x_219);
if (lean_is_scalar(x_218)) {
 x_221 = lean_alloc_ctor(0, 1, 0);
} else {
 x_221 = x_218;
}
lean_ctor_set(x_221, 0, x_220);
return x_221;
}
else
{
lean_object* x_222; lean_object* x_223; lean_object* x_224; 
lean_dec(x_215);
lean_dec(x_207);
x_222 = lean_ctor_get(x_216, 0);
lean_inc(x_222);
if (lean_is_exclusive(x_216)) {
 lean_ctor_release(x_216, 0);
 x_223 = x_216;
} else {
 lean_dec_ref(x_216);
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
lean_dec(x_207);
lean_dec(x_19);
lean_dec(x_12);
x_225 = lean_ctor_get(x_214, 0);
lean_inc(x_225);
if (lean_is_exclusive(x_214)) {
 lean_ctor_release(x_214, 0);
 x_226 = x_214;
} else {
 lean_dec_ref(x_214);
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
lean_dec(x_19);
lean_dec(x_17);
lean_dec(x_12);
x_228 = lean_ctor_get(x_206, 0);
lean_inc(x_228);
if (lean_is_exclusive(x_206)) {
 lean_ctor_release(x_206, 0);
 x_229 = x_206;
} else {
 lean_dec_ref(x_206);
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
lean_dec(x_17);
lean_dec(x_14);
lean_dec_ref(x_13);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
x_231 = !lean_is_exclusive(x_18);
if (x_231 == 0)
{
return x_18;
}
else
{
lean_object* x_232; lean_object* x_233; 
x_232 = lean_ctor_get(x_18, 0);
lean_inc(x_232);
lean_dec(x_18);
x_233 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_233, 0, x_232);
return x_233;
}
}
}
else
{
uint8_t x_234; 
lean_dec(x_14);
lean_dec_ref(x_13);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec(x_1);
x_234 = !lean_is_exclusive(x_16);
if (x_234 == 0)
{
return x_16;
}
else
{
lean_object* x_235; lean_object* x_236; 
x_235 = lean_ctor_get(x_16, 0);
lean_inc(x_235);
lean_dec(x_16);
x_236 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_236, 0, x_235);
return x_236;
}
}
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_findCancelFactor___closed__11;
x_2 = lp_mathlib_CancelDenoms_findCancelFactor___closed__6;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("NegZeroClass", 12, 12);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toNeg", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__2;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__1;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("SubNegZeroMonoid", 16, 16);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toNegZeroClass", 14, 14);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__5;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__4;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__7() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("SubtractionMonoid", 17, 17);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__8() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toSubNegZeroMonoid", 18, 18);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__8;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__7;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__10() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("SubtractionCommMonoid", 21, 21);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__11() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toSubtractionMonoid", 19, 19);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__12() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__11;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__10;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__13() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("AddCommGroup", 12, 12);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__14() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toDivisionAddCommMonoid", 23, 23);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__15() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__14;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__13;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__16() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toAddCommGroup", 14, 14);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__2(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_14; 
lean_inc_ref(x_9);
x_14 = l_Lean_Meta_mkFreshExprMVar(x_1, x_2, x_3, x_9, x_10, x_11, x_12);
if (lean_obj_tag(x_14) == 0)
{
lean_object* x_15; lean_object* x_16; uint8_t x_17; 
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
lean_dec_ref(x_14);
x_16 = l_Lean_Meta_Context_config(x_9);
x_17 = !lean_is_exclusive(x_16);
if (x_17 == 0)
{
uint8_t x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; uint8_t x_25; uint8_t x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; uint8_t x_57; uint64_t x_58; uint8_t x_59; 
x_18 = lean_ctor_get_uint8(x_9, sizeof(void*)*7);
x_19 = lean_ctor_get(x_9, 1);
lean_inc(x_19);
x_20 = lean_ctor_get(x_9, 2);
lean_inc_ref(x_20);
x_21 = lean_ctor_get(x_9, 3);
lean_inc_ref(x_21);
x_22 = lean_ctor_get(x_9, 4);
lean_inc(x_22);
x_23 = lean_ctor_get(x_9, 5);
lean_inc(x_23);
x_24 = lean_ctor_get(x_9, 6);
lean_inc(x_24);
x_25 = lean_ctor_get_uint8(x_9, sizeof(void*)*7 + 1);
x_26 = lean_ctor_get_uint8(x_9, sizeof(void*)*7 + 2);
x_27 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__0;
lean_inc(x_4);
x_28 = l_Lean_Expr_const___override(x_27, x_4);
lean_inc_ref(x_5);
x_29 = l_Lean_Expr_app___override(x_28, x_5);
x_30 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__3;
lean_inc(x_4);
x_31 = l_Lean_Expr_const___override(x_30, x_4);
lean_inc_ref(x_5);
x_32 = l_Lean_Expr_app___override(x_31, x_5);
x_33 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__6;
lean_inc(x_4);
x_34 = l_Lean_Expr_const___override(x_33, x_4);
lean_inc_ref(x_5);
x_35 = l_Lean_Expr_app___override(x_34, x_5);
x_36 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__9;
lean_inc(x_4);
x_37 = l_Lean_Expr_const___override(x_36, x_4);
lean_inc_ref(x_5);
x_38 = l_Lean_Expr_app___override(x_37, x_5);
x_39 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__12;
lean_inc(x_4);
x_40 = l_Lean_Expr_const___override(x_39, x_4);
lean_inc_ref(x_5);
x_41 = l_Lean_Expr_app___override(x_40, x_5);
x_42 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__15;
lean_inc(x_4);
x_43 = l_Lean_Expr_const___override(x_42, x_4);
lean_inc_ref(x_5);
x_44 = l_Lean_Expr_app___override(x_43, x_5);
x_45 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__16;
x_46 = l_Lean_Name_mkStr2(x_6, x_45);
x_47 = l_Lean_Expr_const___override(x_46, x_4);
x_48 = l_Lean_Expr_app___override(x_47, x_5);
x_49 = l_Lean_Expr_app___override(x_48, x_7);
x_50 = l_Lean_Expr_app___override(x_44, x_49);
x_51 = l_Lean_Expr_app___override(x_41, x_50);
x_52 = l_Lean_Expr_app___override(x_38, x_51);
x_53 = l_Lean_Expr_app___override(x_35, x_52);
x_54 = l_Lean_Expr_app___override(x_32, x_53);
x_55 = l_Lean_Expr_app___override(x_29, x_54);
lean_inc(x_15);
x_56 = l_Lean_Expr_app___override(x_55, x_15);
x_57 = 2;
lean_ctor_set_uint8(x_16, 9, x_57);
x_58 = l_Lean_Meta_Context_configKey(x_9);
x_59 = !lean_is_exclusive(x_9);
if (x_59 == 0)
{
lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; uint64_t x_67; uint64_t x_68; uint64_t x_69; uint64_t x_70; uint64_t x_71; lean_object* x_72; lean_object* x_73; 
x_60 = lean_ctor_get(x_9, 6);
lean_dec(x_60);
x_61 = lean_ctor_get(x_9, 5);
lean_dec(x_61);
x_62 = lean_ctor_get(x_9, 4);
lean_dec(x_62);
x_63 = lean_ctor_get(x_9, 3);
lean_dec(x_63);
x_64 = lean_ctor_get(x_9, 2);
lean_dec(x_64);
x_65 = lean_ctor_get(x_9, 1);
lean_dec(x_65);
x_66 = lean_ctor_get(x_9, 0);
lean_dec(x_66);
x_67 = 2;
x_68 = lean_uint64_shift_right(x_58, x_67);
x_69 = lean_uint64_shift_left(x_68, x_67);
x_70 = lp_mathlib_CancelDenoms_mkProdPrf___lam__4___closed__0;
x_71 = lean_uint64_lor(x_69, x_70);
x_72 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_72, 0, x_16);
lean_ctor_set_uint64(x_72, sizeof(void*)*1, x_71);
lean_ctor_set(x_9, 0, x_72);
lean_inc(x_10);
x_73 = l_Lean_Meta_isExprDefEq(x_56, x_8, x_9, x_10, x_11, x_12);
if (lean_obj_tag(x_73) == 0)
{
uint8_t x_74; 
x_74 = !lean_is_exclusive(x_73);
if (x_74 == 0)
{
lean_object* x_75; uint8_t x_76; 
x_75 = lean_ctor_get(x_73, 0);
x_76 = lean_unbox(x_75);
if (x_76 == 0)
{
lean_object* x_77; 
lean_dec(x_10);
x_77 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_77, 0, x_15);
lean_ctor_set(x_77, 1, x_75);
lean_ctor_set(x_73, 0, x_77);
return x_73;
}
else
{
lean_object* x_78; 
lean_free_object(x_73);
x_78 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_15, x_10);
lean_dec(x_10);
if (lean_obj_tag(x_78) == 0)
{
uint8_t x_79; 
x_79 = !lean_is_exclusive(x_78);
if (x_79 == 0)
{
lean_object* x_80; lean_object* x_81; 
x_80 = lean_ctor_get(x_78, 0);
x_81 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_81, 0, x_80);
lean_ctor_set(x_81, 1, x_75);
lean_ctor_set(x_78, 0, x_81);
return x_78;
}
else
{
lean_object* x_82; lean_object* x_83; lean_object* x_84; 
x_82 = lean_ctor_get(x_78, 0);
lean_inc(x_82);
lean_dec(x_78);
x_83 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_83, 0, x_82);
lean_ctor_set(x_83, 1, x_75);
x_84 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_84, 0, x_83);
return x_84;
}
}
else
{
uint8_t x_85; 
lean_dec(x_75);
x_85 = !lean_is_exclusive(x_78);
if (x_85 == 0)
{
return x_78;
}
else
{
lean_object* x_86; lean_object* x_87; 
x_86 = lean_ctor_get(x_78, 0);
lean_inc(x_86);
lean_dec(x_78);
x_87 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_87, 0, x_86);
return x_87;
}
}
}
}
else
{
lean_object* x_88; uint8_t x_89; 
x_88 = lean_ctor_get(x_73, 0);
lean_inc(x_88);
lean_dec(x_73);
x_89 = lean_unbox(x_88);
if (x_89 == 0)
{
lean_object* x_90; lean_object* x_91; 
lean_dec(x_10);
x_90 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_90, 0, x_15);
lean_ctor_set(x_90, 1, x_88);
x_91 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_91, 0, x_90);
return x_91;
}
else
{
lean_object* x_92; 
x_92 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_15, x_10);
lean_dec(x_10);
if (lean_obj_tag(x_92) == 0)
{
lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; 
x_93 = lean_ctor_get(x_92, 0);
lean_inc(x_93);
if (lean_is_exclusive(x_92)) {
 lean_ctor_release(x_92, 0);
 x_94 = x_92;
} else {
 lean_dec_ref(x_92);
 x_94 = lean_box(0);
}
x_95 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_95, 0, x_93);
lean_ctor_set(x_95, 1, x_88);
if (lean_is_scalar(x_94)) {
 x_96 = lean_alloc_ctor(0, 1, 0);
} else {
 x_96 = x_94;
}
lean_ctor_set(x_96, 0, x_95);
return x_96;
}
else
{
lean_object* x_97; lean_object* x_98; lean_object* x_99; 
lean_dec(x_88);
x_97 = lean_ctor_get(x_92, 0);
lean_inc(x_97);
if (lean_is_exclusive(x_92)) {
 lean_ctor_release(x_92, 0);
 x_98 = x_92;
} else {
 lean_dec_ref(x_92);
 x_98 = lean_box(0);
}
if (lean_is_scalar(x_98)) {
 x_99 = lean_alloc_ctor(1, 1, 0);
} else {
 x_99 = x_98;
}
lean_ctor_set(x_99, 0, x_97);
return x_99;
}
}
}
}
else
{
uint8_t x_100; 
lean_dec(x_15);
lean_dec(x_10);
x_100 = !lean_is_exclusive(x_73);
if (x_100 == 0)
{
return x_73;
}
else
{
lean_object* x_101; lean_object* x_102; 
x_101 = lean_ctor_get(x_73, 0);
lean_inc(x_101);
lean_dec(x_73);
x_102 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_102, 0, x_101);
return x_102;
}
}
}
else
{
uint64_t x_103; uint64_t x_104; uint64_t x_105; uint64_t x_106; uint64_t x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; 
lean_dec(x_9);
x_103 = 2;
x_104 = lean_uint64_shift_right(x_58, x_103);
x_105 = lean_uint64_shift_left(x_104, x_103);
x_106 = lp_mathlib_CancelDenoms_mkProdPrf___lam__4___closed__0;
x_107 = lean_uint64_lor(x_105, x_106);
x_108 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_108, 0, x_16);
lean_ctor_set_uint64(x_108, sizeof(void*)*1, x_107);
x_109 = lean_alloc_ctor(0, 7, 3);
lean_ctor_set(x_109, 0, x_108);
lean_ctor_set(x_109, 1, x_19);
lean_ctor_set(x_109, 2, x_20);
lean_ctor_set(x_109, 3, x_21);
lean_ctor_set(x_109, 4, x_22);
lean_ctor_set(x_109, 5, x_23);
lean_ctor_set(x_109, 6, x_24);
lean_ctor_set_uint8(x_109, sizeof(void*)*7, x_18);
lean_ctor_set_uint8(x_109, sizeof(void*)*7 + 1, x_25);
lean_ctor_set_uint8(x_109, sizeof(void*)*7 + 2, x_26);
lean_inc(x_10);
x_110 = l_Lean_Meta_isExprDefEq(x_56, x_8, x_109, x_10, x_11, x_12);
if (lean_obj_tag(x_110) == 0)
{
lean_object* x_111; lean_object* x_112; uint8_t x_113; 
x_111 = lean_ctor_get(x_110, 0);
lean_inc(x_111);
if (lean_is_exclusive(x_110)) {
 lean_ctor_release(x_110, 0);
 x_112 = x_110;
} else {
 lean_dec_ref(x_110);
 x_112 = lean_box(0);
}
x_113 = lean_unbox(x_111);
if (x_113 == 0)
{
lean_object* x_114; lean_object* x_115; 
lean_dec(x_10);
x_114 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_114, 0, x_15);
lean_ctor_set(x_114, 1, x_111);
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
lean_object* x_116; 
lean_dec(x_112);
x_116 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_15, x_10);
lean_dec(x_10);
if (lean_obj_tag(x_116) == 0)
{
lean_object* x_117; lean_object* x_118; lean_object* x_119; lean_object* x_120; 
x_117 = lean_ctor_get(x_116, 0);
lean_inc(x_117);
if (lean_is_exclusive(x_116)) {
 lean_ctor_release(x_116, 0);
 x_118 = x_116;
} else {
 lean_dec_ref(x_116);
 x_118 = lean_box(0);
}
x_119 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_119, 0, x_117);
lean_ctor_set(x_119, 1, x_111);
if (lean_is_scalar(x_118)) {
 x_120 = lean_alloc_ctor(0, 1, 0);
} else {
 x_120 = x_118;
}
lean_ctor_set(x_120, 0, x_119);
return x_120;
}
else
{
lean_object* x_121; lean_object* x_122; lean_object* x_123; 
lean_dec(x_111);
x_121 = lean_ctor_get(x_116, 0);
lean_inc(x_121);
if (lean_is_exclusive(x_116)) {
 lean_ctor_release(x_116, 0);
 x_122 = x_116;
} else {
 lean_dec_ref(x_116);
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
lean_object* x_124; lean_object* x_125; lean_object* x_126; 
lean_dec(x_15);
lean_dec(x_10);
x_124 = lean_ctor_get(x_110, 0);
lean_inc(x_124);
if (lean_is_exclusive(x_110)) {
 lean_ctor_release(x_110, 0);
 x_125 = x_110;
} else {
 lean_dec_ref(x_110);
 x_125 = lean_box(0);
}
if (lean_is_scalar(x_125)) {
 x_126 = lean_alloc_ctor(1, 1, 0);
} else {
 x_126 = x_125;
}
lean_ctor_set(x_126, 0, x_124);
return x_126;
}
}
}
else
{
uint8_t x_127; uint8_t x_128; uint8_t x_129; uint8_t x_130; uint8_t x_131; uint8_t x_132; uint8_t x_133; uint8_t x_134; uint8_t x_135; uint8_t x_136; uint8_t x_137; uint8_t x_138; uint8_t x_139; uint8_t x_140; uint8_t x_141; uint8_t x_142; uint8_t x_143; uint8_t x_144; uint8_t x_145; lean_object* x_146; lean_object* x_147; lean_object* x_148; lean_object* x_149; lean_object* x_150; lean_object* x_151; uint8_t x_152; uint8_t x_153; lean_object* x_154; lean_object* x_155; lean_object* x_156; lean_object* x_157; lean_object* x_158; lean_object* x_159; lean_object* x_160; lean_object* x_161; lean_object* x_162; lean_object* x_163; lean_object* x_164; lean_object* x_165; lean_object* x_166; lean_object* x_167; lean_object* x_168; lean_object* x_169; lean_object* x_170; lean_object* x_171; lean_object* x_172; lean_object* x_173; lean_object* x_174; lean_object* x_175; lean_object* x_176; lean_object* x_177; lean_object* x_178; lean_object* x_179; lean_object* x_180; lean_object* x_181; lean_object* x_182; lean_object* x_183; uint8_t x_184; lean_object* x_185; uint64_t x_186; lean_object* x_187; uint64_t x_188; uint64_t x_189; uint64_t x_190; uint64_t x_191; uint64_t x_192; lean_object* x_193; lean_object* x_194; lean_object* x_195; 
x_127 = lean_ctor_get_uint8(x_16, 0);
x_128 = lean_ctor_get_uint8(x_16, 1);
x_129 = lean_ctor_get_uint8(x_16, 2);
x_130 = lean_ctor_get_uint8(x_16, 3);
x_131 = lean_ctor_get_uint8(x_16, 4);
x_132 = lean_ctor_get_uint8(x_16, 5);
x_133 = lean_ctor_get_uint8(x_16, 6);
x_134 = lean_ctor_get_uint8(x_16, 7);
x_135 = lean_ctor_get_uint8(x_16, 8);
x_136 = lean_ctor_get_uint8(x_16, 10);
x_137 = lean_ctor_get_uint8(x_16, 11);
x_138 = lean_ctor_get_uint8(x_16, 12);
x_139 = lean_ctor_get_uint8(x_16, 13);
x_140 = lean_ctor_get_uint8(x_16, 14);
x_141 = lean_ctor_get_uint8(x_16, 15);
x_142 = lean_ctor_get_uint8(x_16, 16);
x_143 = lean_ctor_get_uint8(x_16, 17);
x_144 = lean_ctor_get_uint8(x_16, 18);
lean_dec(x_16);
x_145 = lean_ctor_get_uint8(x_9, sizeof(void*)*7);
x_146 = lean_ctor_get(x_9, 1);
lean_inc(x_146);
x_147 = lean_ctor_get(x_9, 2);
lean_inc_ref(x_147);
x_148 = lean_ctor_get(x_9, 3);
lean_inc_ref(x_148);
x_149 = lean_ctor_get(x_9, 4);
lean_inc(x_149);
x_150 = lean_ctor_get(x_9, 5);
lean_inc(x_150);
x_151 = lean_ctor_get(x_9, 6);
lean_inc(x_151);
x_152 = lean_ctor_get_uint8(x_9, sizeof(void*)*7 + 1);
x_153 = lean_ctor_get_uint8(x_9, sizeof(void*)*7 + 2);
x_154 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__0;
lean_inc(x_4);
x_155 = l_Lean_Expr_const___override(x_154, x_4);
lean_inc_ref(x_5);
x_156 = l_Lean_Expr_app___override(x_155, x_5);
x_157 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__3;
lean_inc(x_4);
x_158 = l_Lean_Expr_const___override(x_157, x_4);
lean_inc_ref(x_5);
x_159 = l_Lean_Expr_app___override(x_158, x_5);
x_160 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__6;
lean_inc(x_4);
x_161 = l_Lean_Expr_const___override(x_160, x_4);
lean_inc_ref(x_5);
x_162 = l_Lean_Expr_app___override(x_161, x_5);
x_163 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__9;
lean_inc(x_4);
x_164 = l_Lean_Expr_const___override(x_163, x_4);
lean_inc_ref(x_5);
x_165 = l_Lean_Expr_app___override(x_164, x_5);
x_166 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__12;
lean_inc(x_4);
x_167 = l_Lean_Expr_const___override(x_166, x_4);
lean_inc_ref(x_5);
x_168 = l_Lean_Expr_app___override(x_167, x_5);
x_169 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__15;
lean_inc(x_4);
x_170 = l_Lean_Expr_const___override(x_169, x_4);
lean_inc_ref(x_5);
x_171 = l_Lean_Expr_app___override(x_170, x_5);
x_172 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__16;
x_173 = l_Lean_Name_mkStr2(x_6, x_172);
x_174 = l_Lean_Expr_const___override(x_173, x_4);
x_175 = l_Lean_Expr_app___override(x_174, x_5);
x_176 = l_Lean_Expr_app___override(x_175, x_7);
x_177 = l_Lean_Expr_app___override(x_171, x_176);
x_178 = l_Lean_Expr_app___override(x_168, x_177);
x_179 = l_Lean_Expr_app___override(x_165, x_178);
x_180 = l_Lean_Expr_app___override(x_162, x_179);
x_181 = l_Lean_Expr_app___override(x_159, x_180);
x_182 = l_Lean_Expr_app___override(x_156, x_181);
lean_inc(x_15);
x_183 = l_Lean_Expr_app___override(x_182, x_15);
x_184 = 2;
x_185 = lean_alloc_ctor(0, 0, 19);
lean_ctor_set_uint8(x_185, 0, x_127);
lean_ctor_set_uint8(x_185, 1, x_128);
lean_ctor_set_uint8(x_185, 2, x_129);
lean_ctor_set_uint8(x_185, 3, x_130);
lean_ctor_set_uint8(x_185, 4, x_131);
lean_ctor_set_uint8(x_185, 5, x_132);
lean_ctor_set_uint8(x_185, 6, x_133);
lean_ctor_set_uint8(x_185, 7, x_134);
lean_ctor_set_uint8(x_185, 8, x_135);
lean_ctor_set_uint8(x_185, 9, x_184);
lean_ctor_set_uint8(x_185, 10, x_136);
lean_ctor_set_uint8(x_185, 11, x_137);
lean_ctor_set_uint8(x_185, 12, x_138);
lean_ctor_set_uint8(x_185, 13, x_139);
lean_ctor_set_uint8(x_185, 14, x_140);
lean_ctor_set_uint8(x_185, 15, x_141);
lean_ctor_set_uint8(x_185, 16, x_142);
lean_ctor_set_uint8(x_185, 17, x_143);
lean_ctor_set_uint8(x_185, 18, x_144);
x_186 = l_Lean_Meta_Context_configKey(x_9);
if (lean_is_exclusive(x_9)) {
 lean_ctor_release(x_9, 0);
 lean_ctor_release(x_9, 1);
 lean_ctor_release(x_9, 2);
 lean_ctor_release(x_9, 3);
 lean_ctor_release(x_9, 4);
 lean_ctor_release(x_9, 5);
 lean_ctor_release(x_9, 6);
 x_187 = x_9;
} else {
 lean_dec_ref(x_9);
 x_187 = lean_box(0);
}
x_188 = 2;
x_189 = lean_uint64_shift_right(x_186, x_188);
x_190 = lean_uint64_shift_left(x_189, x_188);
x_191 = lp_mathlib_CancelDenoms_mkProdPrf___lam__4___closed__0;
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
lean_ctor_set(x_194, 1, x_146);
lean_ctor_set(x_194, 2, x_147);
lean_ctor_set(x_194, 3, x_148);
lean_ctor_set(x_194, 4, x_149);
lean_ctor_set(x_194, 5, x_150);
lean_ctor_set(x_194, 6, x_151);
lean_ctor_set_uint8(x_194, sizeof(void*)*7, x_145);
lean_ctor_set_uint8(x_194, sizeof(void*)*7 + 1, x_152);
lean_ctor_set_uint8(x_194, sizeof(void*)*7 + 2, x_153);
lean_inc(x_10);
x_195 = l_Lean_Meta_isExprDefEq(x_183, x_8, x_194, x_10, x_11, x_12);
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
lean_object* x_199; lean_object* x_200; 
lean_dec(x_10);
x_199 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_199, 0, x_15);
lean_ctor_set(x_199, 1, x_196);
if (lean_is_scalar(x_197)) {
 x_200 = lean_alloc_ctor(0, 1, 0);
} else {
 x_200 = x_197;
}
lean_ctor_set(x_200, 0, x_199);
return x_200;
}
else
{
lean_object* x_201; 
lean_dec(x_197);
x_201 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_15, x_10);
lean_dec(x_10);
if (lean_obj_tag(x_201) == 0)
{
lean_object* x_202; lean_object* x_203; lean_object* x_204; lean_object* x_205; 
x_202 = lean_ctor_get(x_201, 0);
lean_inc(x_202);
if (lean_is_exclusive(x_201)) {
 lean_ctor_release(x_201, 0);
 x_203 = x_201;
} else {
 lean_dec_ref(x_201);
 x_203 = lean_box(0);
}
x_204 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_204, 0, x_202);
lean_ctor_set(x_204, 1, x_196);
if (lean_is_scalar(x_203)) {
 x_205 = lean_alloc_ctor(0, 1, 0);
} else {
 x_205 = x_203;
}
lean_ctor_set(x_205, 0, x_204);
return x_205;
}
else
{
lean_object* x_206; lean_object* x_207; lean_object* x_208; 
lean_dec(x_196);
x_206 = lean_ctor_get(x_201, 0);
lean_inc(x_206);
if (lean_is_exclusive(x_201)) {
 lean_ctor_release(x_201, 0);
 x_207 = x_201;
} else {
 lean_dec_ref(x_201);
 x_207 = lean_box(0);
}
if (lean_is_scalar(x_207)) {
 x_208 = lean_alloc_ctor(1, 1, 0);
} else {
 x_208 = x_207;
}
lean_ctor_set(x_208, 0, x_206);
return x_208;
}
}
}
else
{
lean_object* x_209; lean_object* x_210; lean_object* x_211; 
lean_dec(x_15);
lean_dec(x_10);
x_209 = lean_ctor_get(x_195, 0);
lean_inc(x_209);
if (lean_is_exclusive(x_195)) {
 lean_ctor_release(x_195, 0);
 x_210 = x_195;
} else {
 lean_dec_ref(x_195);
 x_210 = lean_box(0);
}
if (lean_is_scalar(x_210)) {
 x_211 = lean_alloc_ctor(1, 1, 0);
} else {
 x_211 = x_210;
}
lean_ctor_set(x_211, 0, x_209);
return x_211;
}
}
}
else
{
uint8_t x_212; 
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
x_212 = !lean_is_exclusive(x_14);
if (x_212 == 0)
{
return x_14;
}
else
{
lean_object* x_213; lean_object* x_214; 
x_213 = lean_ctor_get(x_14, 0);
lean_inc(x_213);
lean_dec(x_14);
x_214 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_214, 0, x_213);
return x_214;
}
}
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_findCancelFactor___closed__14;
x_2 = lp_mathlib_CancelDenoms_findCancelFactor___closed__3;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("instHSub", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__1;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("SubNegMonoid", 12, 12);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toSub", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__4;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__3;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("AddGroup", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__7() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toSubNegMonoid", 14, 14);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__8() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__7;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__6;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__9() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toAddGroup", 10, 10);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__5(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, uint8_t x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_16; 
lean_inc_ref(x_11);
lean_inc(x_3);
lean_inc(x_1);
x_16 = l_Lean_Meta_mkFreshExprMVar(x_1, x_2, x_3, x_11, x_12, x_13, x_14);
if (lean_obj_tag(x_16) == 0)
{
lean_object* x_17; lean_object* x_18; 
x_17 = lean_ctor_get(x_16, 0);
lean_inc(x_17);
lean_dec_ref(x_16);
lean_inc_ref(x_11);
x_18 = l_Lean_Meta_mkFreshExprMVar(x_1, x_2, x_3, x_11, x_12, x_13, x_14);
if (lean_obj_tag(x_18) == 0)
{
lean_object* x_19; lean_object* x_20; uint8_t x_21; 
x_19 = lean_ctor_get(x_18, 0);
lean_inc(x_19);
lean_dec_ref(x_18);
x_20 = l_Lean_Meta_Context_config(x_11);
x_21 = !lean_is_exclusive(x_20);
if (x_21 == 0)
{
uint8_t x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; uint8_t x_29; uint8_t x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; uint8_t x_56; uint64_t x_57; uint8_t x_58; 
x_22 = lean_ctor_get_uint8(x_11, sizeof(void*)*7);
x_23 = lean_ctor_get(x_11, 1);
lean_inc(x_23);
x_24 = lean_ctor_get(x_11, 2);
lean_inc_ref(x_24);
x_25 = lean_ctor_get(x_11, 3);
lean_inc_ref(x_25);
x_26 = lean_ctor_get(x_11, 4);
lean_inc(x_26);
x_27 = lean_ctor_get(x_11, 5);
lean_inc(x_27);
x_28 = lean_ctor_get(x_11, 6);
lean_inc(x_28);
x_29 = lean_ctor_get_uint8(x_11, sizeof(void*)*7 + 1);
x_30 = lean_ctor_get_uint8(x_11, sizeof(void*)*7 + 2);
x_31 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__0;
x_32 = l_Lean_Expr_const___override(x_31, x_4);
lean_inc_ref(x_5);
x_33 = l_Lean_Expr_app___override(x_32, x_5);
lean_inc_ref(x_5);
x_34 = l_Lean_Expr_app___override(x_33, x_5);
lean_inc_ref(x_5);
x_35 = l_Lean_Expr_app___override(x_34, x_5);
x_36 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__2;
lean_inc(x_6);
x_37 = l_Lean_Expr_const___override(x_36, x_6);
lean_inc_ref(x_5);
x_38 = l_Lean_Expr_app___override(x_37, x_5);
x_39 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__5;
lean_inc(x_6);
x_40 = l_Lean_Expr_const___override(x_39, x_6);
lean_inc_ref(x_5);
x_41 = l_Lean_Expr_app___override(x_40, x_5);
x_42 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__8;
lean_inc(x_6);
x_43 = l_Lean_Expr_const___override(x_42, x_6);
lean_inc_ref(x_5);
x_44 = l_Lean_Expr_app___override(x_43, x_5);
x_45 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__9;
x_46 = l_Lean_Name_mkStr2(x_7, x_45);
x_47 = l_Lean_Expr_const___override(x_46, x_6);
x_48 = l_Lean_Expr_app___override(x_47, x_5);
x_49 = l_Lean_Expr_app___override(x_48, x_8);
x_50 = l_Lean_Expr_app___override(x_44, x_49);
x_51 = l_Lean_Expr_app___override(x_41, x_50);
x_52 = l_Lean_Expr_app___override(x_38, x_51);
x_53 = l_Lean_Expr_app___override(x_35, x_52);
lean_inc(x_17);
x_54 = l_Lean_Expr_app___override(x_53, x_17);
lean_inc(x_19);
x_55 = l_Lean_Expr_app___override(x_54, x_19);
x_56 = 2;
lean_ctor_set_uint8(x_20, 9, x_56);
x_57 = l_Lean_Meta_Context_configKey(x_11);
x_58 = !lean_is_exclusive(x_11);
if (x_58 == 0)
{
lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; uint64_t x_66; uint64_t x_67; uint64_t x_68; uint64_t x_69; uint64_t x_70; lean_object* x_71; lean_object* x_72; 
x_59 = lean_ctor_get(x_11, 6);
lean_dec(x_59);
x_60 = lean_ctor_get(x_11, 5);
lean_dec(x_60);
x_61 = lean_ctor_get(x_11, 4);
lean_dec(x_61);
x_62 = lean_ctor_get(x_11, 3);
lean_dec(x_62);
x_63 = lean_ctor_get(x_11, 2);
lean_dec(x_63);
x_64 = lean_ctor_get(x_11, 1);
lean_dec(x_64);
x_65 = lean_ctor_get(x_11, 0);
lean_dec(x_65);
x_66 = 2;
x_67 = lean_uint64_shift_right(x_57, x_66);
x_68 = lean_uint64_shift_left(x_67, x_66);
x_69 = lp_mathlib_CancelDenoms_mkProdPrf___lam__4___closed__0;
x_70 = lean_uint64_lor(x_68, x_69);
x_71 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_71, 0, x_20);
lean_ctor_set_uint64(x_71, sizeof(void*)*1, x_70);
lean_ctor_set(x_11, 0, x_71);
lean_inc(x_12);
x_72 = l_Lean_Meta_isExprDefEq(x_55, x_9, x_11, x_12, x_13, x_14);
if (lean_obj_tag(x_72) == 0)
{
uint8_t x_73; 
x_73 = !lean_is_exclusive(x_72);
if (x_73 == 0)
{
lean_object* x_74; uint8_t x_75; 
x_74 = lean_ctor_get(x_72, 0);
x_75 = lean_unbox(x_74);
if (x_75 == 0)
{
lean_object* x_76; lean_object* x_77; lean_object* x_78; 
lean_dec(x_74);
lean_dec(x_12);
x_76 = lean_box(x_10);
x_77 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_77, 0, x_19);
lean_ctor_set(x_77, 1, x_76);
x_78 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_78, 0, x_17);
lean_ctor_set(x_78, 1, x_77);
lean_ctor_set(x_72, 0, x_78);
return x_72;
}
else
{
lean_object* x_79; 
lean_free_object(x_72);
x_79 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_17, x_12);
if (lean_obj_tag(x_79) == 0)
{
lean_object* x_80; lean_object* x_81; 
x_80 = lean_ctor_get(x_79, 0);
lean_inc(x_80);
lean_dec_ref(x_79);
x_81 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_19, x_12);
lean_dec(x_12);
if (lean_obj_tag(x_81) == 0)
{
uint8_t x_82; 
x_82 = !lean_is_exclusive(x_81);
if (x_82 == 0)
{
lean_object* x_83; lean_object* x_84; lean_object* x_85; 
x_83 = lean_ctor_get(x_81, 0);
x_84 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_84, 0, x_83);
lean_ctor_set(x_84, 1, x_74);
x_85 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_85, 0, x_80);
lean_ctor_set(x_85, 1, x_84);
lean_ctor_set(x_81, 0, x_85);
return x_81;
}
else
{
lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; 
x_86 = lean_ctor_get(x_81, 0);
lean_inc(x_86);
lean_dec(x_81);
x_87 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_87, 0, x_86);
lean_ctor_set(x_87, 1, x_74);
x_88 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_88, 0, x_80);
lean_ctor_set(x_88, 1, x_87);
x_89 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_89, 0, x_88);
return x_89;
}
}
else
{
uint8_t x_90; 
lean_dec(x_80);
lean_dec(x_74);
x_90 = !lean_is_exclusive(x_81);
if (x_90 == 0)
{
return x_81;
}
else
{
lean_object* x_91; lean_object* x_92; 
x_91 = lean_ctor_get(x_81, 0);
lean_inc(x_91);
lean_dec(x_81);
x_92 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_92, 0, x_91);
return x_92;
}
}
}
else
{
uint8_t x_93; 
lean_dec(x_74);
lean_dec(x_19);
lean_dec(x_12);
x_93 = !lean_is_exclusive(x_79);
if (x_93 == 0)
{
return x_79;
}
else
{
lean_object* x_94; lean_object* x_95; 
x_94 = lean_ctor_get(x_79, 0);
lean_inc(x_94);
lean_dec(x_79);
x_95 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_95, 0, x_94);
return x_95;
}
}
}
}
else
{
lean_object* x_96; uint8_t x_97; 
x_96 = lean_ctor_get(x_72, 0);
lean_inc(x_96);
lean_dec(x_72);
x_97 = lean_unbox(x_96);
if (x_97 == 0)
{
lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; 
lean_dec(x_96);
lean_dec(x_12);
x_98 = lean_box(x_10);
x_99 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_99, 0, x_19);
lean_ctor_set(x_99, 1, x_98);
x_100 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_100, 0, x_17);
lean_ctor_set(x_100, 1, x_99);
x_101 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_101, 0, x_100);
return x_101;
}
else
{
lean_object* x_102; 
x_102 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_17, x_12);
if (lean_obj_tag(x_102) == 0)
{
lean_object* x_103; lean_object* x_104; 
x_103 = lean_ctor_get(x_102, 0);
lean_inc(x_103);
lean_dec_ref(x_102);
x_104 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_19, x_12);
lean_dec(x_12);
if (lean_obj_tag(x_104) == 0)
{
lean_object* x_105; lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; 
x_105 = lean_ctor_get(x_104, 0);
lean_inc(x_105);
if (lean_is_exclusive(x_104)) {
 lean_ctor_release(x_104, 0);
 x_106 = x_104;
} else {
 lean_dec_ref(x_104);
 x_106 = lean_box(0);
}
x_107 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_107, 0, x_105);
lean_ctor_set(x_107, 1, x_96);
x_108 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_108, 0, x_103);
lean_ctor_set(x_108, 1, x_107);
if (lean_is_scalar(x_106)) {
 x_109 = lean_alloc_ctor(0, 1, 0);
} else {
 x_109 = x_106;
}
lean_ctor_set(x_109, 0, x_108);
return x_109;
}
else
{
lean_object* x_110; lean_object* x_111; lean_object* x_112; 
lean_dec(x_103);
lean_dec(x_96);
x_110 = lean_ctor_get(x_104, 0);
lean_inc(x_110);
if (lean_is_exclusive(x_104)) {
 lean_ctor_release(x_104, 0);
 x_111 = x_104;
} else {
 lean_dec_ref(x_104);
 x_111 = lean_box(0);
}
if (lean_is_scalar(x_111)) {
 x_112 = lean_alloc_ctor(1, 1, 0);
} else {
 x_112 = x_111;
}
lean_ctor_set(x_112, 0, x_110);
return x_112;
}
}
else
{
lean_object* x_113; lean_object* x_114; lean_object* x_115; 
lean_dec(x_96);
lean_dec(x_19);
lean_dec(x_12);
x_113 = lean_ctor_get(x_102, 0);
lean_inc(x_113);
if (lean_is_exclusive(x_102)) {
 lean_ctor_release(x_102, 0);
 x_114 = x_102;
} else {
 lean_dec_ref(x_102);
 x_114 = lean_box(0);
}
if (lean_is_scalar(x_114)) {
 x_115 = lean_alloc_ctor(1, 1, 0);
} else {
 x_115 = x_114;
}
lean_ctor_set(x_115, 0, x_113);
return x_115;
}
}
}
}
else
{
uint8_t x_116; 
lean_dec(x_19);
lean_dec(x_17);
lean_dec(x_12);
x_116 = !lean_is_exclusive(x_72);
if (x_116 == 0)
{
return x_72;
}
else
{
lean_object* x_117; lean_object* x_118; 
x_117 = lean_ctor_get(x_72, 0);
lean_inc(x_117);
lean_dec(x_72);
x_118 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_118, 0, x_117);
return x_118;
}
}
}
else
{
uint64_t x_119; uint64_t x_120; uint64_t x_121; uint64_t x_122; uint64_t x_123; lean_object* x_124; lean_object* x_125; lean_object* x_126; 
lean_dec(x_11);
x_119 = 2;
x_120 = lean_uint64_shift_right(x_57, x_119);
x_121 = lean_uint64_shift_left(x_120, x_119);
x_122 = lp_mathlib_CancelDenoms_mkProdPrf___lam__4___closed__0;
x_123 = lean_uint64_lor(x_121, x_122);
x_124 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_124, 0, x_20);
lean_ctor_set_uint64(x_124, sizeof(void*)*1, x_123);
x_125 = lean_alloc_ctor(0, 7, 3);
lean_ctor_set(x_125, 0, x_124);
lean_ctor_set(x_125, 1, x_23);
lean_ctor_set(x_125, 2, x_24);
lean_ctor_set(x_125, 3, x_25);
lean_ctor_set(x_125, 4, x_26);
lean_ctor_set(x_125, 5, x_27);
lean_ctor_set(x_125, 6, x_28);
lean_ctor_set_uint8(x_125, sizeof(void*)*7, x_22);
lean_ctor_set_uint8(x_125, sizeof(void*)*7 + 1, x_29);
lean_ctor_set_uint8(x_125, sizeof(void*)*7 + 2, x_30);
lean_inc(x_12);
x_126 = l_Lean_Meta_isExprDefEq(x_55, x_9, x_125, x_12, x_13, x_14);
if (lean_obj_tag(x_126) == 0)
{
lean_object* x_127; lean_object* x_128; uint8_t x_129; 
x_127 = lean_ctor_get(x_126, 0);
lean_inc(x_127);
if (lean_is_exclusive(x_126)) {
 lean_ctor_release(x_126, 0);
 x_128 = x_126;
} else {
 lean_dec_ref(x_126);
 x_128 = lean_box(0);
}
x_129 = lean_unbox(x_127);
if (x_129 == 0)
{
lean_object* x_130; lean_object* x_131; lean_object* x_132; lean_object* x_133; 
lean_dec(x_127);
lean_dec(x_12);
x_130 = lean_box(x_10);
x_131 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_131, 0, x_19);
lean_ctor_set(x_131, 1, x_130);
x_132 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_132, 0, x_17);
lean_ctor_set(x_132, 1, x_131);
if (lean_is_scalar(x_128)) {
 x_133 = lean_alloc_ctor(0, 1, 0);
} else {
 x_133 = x_128;
}
lean_ctor_set(x_133, 0, x_132);
return x_133;
}
else
{
lean_object* x_134; 
lean_dec(x_128);
x_134 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_17, x_12);
if (lean_obj_tag(x_134) == 0)
{
lean_object* x_135; lean_object* x_136; 
x_135 = lean_ctor_get(x_134, 0);
lean_inc(x_135);
lean_dec_ref(x_134);
x_136 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_19, x_12);
lean_dec(x_12);
if (lean_obj_tag(x_136) == 0)
{
lean_object* x_137; lean_object* x_138; lean_object* x_139; lean_object* x_140; lean_object* x_141; 
x_137 = lean_ctor_get(x_136, 0);
lean_inc(x_137);
if (lean_is_exclusive(x_136)) {
 lean_ctor_release(x_136, 0);
 x_138 = x_136;
} else {
 lean_dec_ref(x_136);
 x_138 = lean_box(0);
}
x_139 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_139, 0, x_137);
lean_ctor_set(x_139, 1, x_127);
x_140 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_140, 0, x_135);
lean_ctor_set(x_140, 1, x_139);
if (lean_is_scalar(x_138)) {
 x_141 = lean_alloc_ctor(0, 1, 0);
} else {
 x_141 = x_138;
}
lean_ctor_set(x_141, 0, x_140);
return x_141;
}
else
{
lean_object* x_142; lean_object* x_143; lean_object* x_144; 
lean_dec(x_135);
lean_dec(x_127);
x_142 = lean_ctor_get(x_136, 0);
lean_inc(x_142);
if (lean_is_exclusive(x_136)) {
 lean_ctor_release(x_136, 0);
 x_143 = x_136;
} else {
 lean_dec_ref(x_136);
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
lean_dec(x_127);
lean_dec(x_19);
lean_dec(x_12);
x_145 = lean_ctor_get(x_134, 0);
lean_inc(x_145);
if (lean_is_exclusive(x_134)) {
 lean_ctor_release(x_134, 0);
 x_146 = x_134;
} else {
 lean_dec_ref(x_134);
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
}
else
{
lean_object* x_148; lean_object* x_149; lean_object* x_150; 
lean_dec(x_19);
lean_dec(x_17);
lean_dec(x_12);
x_148 = lean_ctor_get(x_126, 0);
lean_inc(x_148);
if (lean_is_exclusive(x_126)) {
 lean_ctor_release(x_126, 0);
 x_149 = x_126;
} else {
 lean_dec_ref(x_126);
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
uint8_t x_151; uint8_t x_152; uint8_t x_153; uint8_t x_154; uint8_t x_155; uint8_t x_156; uint8_t x_157; uint8_t x_158; uint8_t x_159; uint8_t x_160; uint8_t x_161; uint8_t x_162; uint8_t x_163; uint8_t x_164; uint8_t x_165; uint8_t x_166; uint8_t x_167; uint8_t x_168; uint8_t x_169; lean_object* x_170; lean_object* x_171; lean_object* x_172; lean_object* x_173; lean_object* x_174; lean_object* x_175; uint8_t x_176; uint8_t x_177; lean_object* x_178; lean_object* x_179; lean_object* x_180; lean_object* x_181; lean_object* x_182; lean_object* x_183; lean_object* x_184; lean_object* x_185; lean_object* x_186; lean_object* x_187; lean_object* x_188; lean_object* x_189; lean_object* x_190; lean_object* x_191; lean_object* x_192; lean_object* x_193; lean_object* x_194; lean_object* x_195; lean_object* x_196; lean_object* x_197; lean_object* x_198; lean_object* x_199; lean_object* x_200; lean_object* x_201; lean_object* x_202; uint8_t x_203; lean_object* x_204; uint64_t x_205; lean_object* x_206; uint64_t x_207; uint64_t x_208; uint64_t x_209; uint64_t x_210; uint64_t x_211; lean_object* x_212; lean_object* x_213; lean_object* x_214; 
x_151 = lean_ctor_get_uint8(x_20, 0);
x_152 = lean_ctor_get_uint8(x_20, 1);
x_153 = lean_ctor_get_uint8(x_20, 2);
x_154 = lean_ctor_get_uint8(x_20, 3);
x_155 = lean_ctor_get_uint8(x_20, 4);
x_156 = lean_ctor_get_uint8(x_20, 5);
x_157 = lean_ctor_get_uint8(x_20, 6);
x_158 = lean_ctor_get_uint8(x_20, 7);
x_159 = lean_ctor_get_uint8(x_20, 8);
x_160 = lean_ctor_get_uint8(x_20, 10);
x_161 = lean_ctor_get_uint8(x_20, 11);
x_162 = lean_ctor_get_uint8(x_20, 12);
x_163 = lean_ctor_get_uint8(x_20, 13);
x_164 = lean_ctor_get_uint8(x_20, 14);
x_165 = lean_ctor_get_uint8(x_20, 15);
x_166 = lean_ctor_get_uint8(x_20, 16);
x_167 = lean_ctor_get_uint8(x_20, 17);
x_168 = lean_ctor_get_uint8(x_20, 18);
lean_dec(x_20);
x_169 = lean_ctor_get_uint8(x_11, sizeof(void*)*7);
x_170 = lean_ctor_get(x_11, 1);
lean_inc(x_170);
x_171 = lean_ctor_get(x_11, 2);
lean_inc_ref(x_171);
x_172 = lean_ctor_get(x_11, 3);
lean_inc_ref(x_172);
x_173 = lean_ctor_get(x_11, 4);
lean_inc(x_173);
x_174 = lean_ctor_get(x_11, 5);
lean_inc(x_174);
x_175 = lean_ctor_get(x_11, 6);
lean_inc(x_175);
x_176 = lean_ctor_get_uint8(x_11, sizeof(void*)*7 + 1);
x_177 = lean_ctor_get_uint8(x_11, sizeof(void*)*7 + 2);
x_178 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__0;
x_179 = l_Lean_Expr_const___override(x_178, x_4);
lean_inc_ref(x_5);
x_180 = l_Lean_Expr_app___override(x_179, x_5);
lean_inc_ref(x_5);
x_181 = l_Lean_Expr_app___override(x_180, x_5);
lean_inc_ref(x_5);
x_182 = l_Lean_Expr_app___override(x_181, x_5);
x_183 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__2;
lean_inc(x_6);
x_184 = l_Lean_Expr_const___override(x_183, x_6);
lean_inc_ref(x_5);
x_185 = l_Lean_Expr_app___override(x_184, x_5);
x_186 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__5;
lean_inc(x_6);
x_187 = l_Lean_Expr_const___override(x_186, x_6);
lean_inc_ref(x_5);
x_188 = l_Lean_Expr_app___override(x_187, x_5);
x_189 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__8;
lean_inc(x_6);
x_190 = l_Lean_Expr_const___override(x_189, x_6);
lean_inc_ref(x_5);
x_191 = l_Lean_Expr_app___override(x_190, x_5);
x_192 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__9;
x_193 = l_Lean_Name_mkStr2(x_7, x_192);
x_194 = l_Lean_Expr_const___override(x_193, x_6);
x_195 = l_Lean_Expr_app___override(x_194, x_5);
x_196 = l_Lean_Expr_app___override(x_195, x_8);
x_197 = l_Lean_Expr_app___override(x_191, x_196);
x_198 = l_Lean_Expr_app___override(x_188, x_197);
x_199 = l_Lean_Expr_app___override(x_185, x_198);
x_200 = l_Lean_Expr_app___override(x_182, x_199);
lean_inc(x_17);
x_201 = l_Lean_Expr_app___override(x_200, x_17);
lean_inc(x_19);
x_202 = l_Lean_Expr_app___override(x_201, x_19);
x_203 = 2;
x_204 = lean_alloc_ctor(0, 0, 19);
lean_ctor_set_uint8(x_204, 0, x_151);
lean_ctor_set_uint8(x_204, 1, x_152);
lean_ctor_set_uint8(x_204, 2, x_153);
lean_ctor_set_uint8(x_204, 3, x_154);
lean_ctor_set_uint8(x_204, 4, x_155);
lean_ctor_set_uint8(x_204, 5, x_156);
lean_ctor_set_uint8(x_204, 6, x_157);
lean_ctor_set_uint8(x_204, 7, x_158);
lean_ctor_set_uint8(x_204, 8, x_159);
lean_ctor_set_uint8(x_204, 9, x_203);
lean_ctor_set_uint8(x_204, 10, x_160);
lean_ctor_set_uint8(x_204, 11, x_161);
lean_ctor_set_uint8(x_204, 12, x_162);
lean_ctor_set_uint8(x_204, 13, x_163);
lean_ctor_set_uint8(x_204, 14, x_164);
lean_ctor_set_uint8(x_204, 15, x_165);
lean_ctor_set_uint8(x_204, 16, x_166);
lean_ctor_set_uint8(x_204, 17, x_167);
lean_ctor_set_uint8(x_204, 18, x_168);
x_205 = l_Lean_Meta_Context_configKey(x_11);
if (lean_is_exclusive(x_11)) {
 lean_ctor_release(x_11, 0);
 lean_ctor_release(x_11, 1);
 lean_ctor_release(x_11, 2);
 lean_ctor_release(x_11, 3);
 lean_ctor_release(x_11, 4);
 lean_ctor_release(x_11, 5);
 lean_ctor_release(x_11, 6);
 x_206 = x_11;
} else {
 lean_dec_ref(x_11);
 x_206 = lean_box(0);
}
x_207 = 2;
x_208 = lean_uint64_shift_right(x_205, x_207);
x_209 = lean_uint64_shift_left(x_208, x_207);
x_210 = lp_mathlib_CancelDenoms_mkProdPrf___lam__4___closed__0;
x_211 = lean_uint64_lor(x_209, x_210);
x_212 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_212, 0, x_204);
lean_ctor_set_uint64(x_212, sizeof(void*)*1, x_211);
if (lean_is_scalar(x_206)) {
 x_213 = lean_alloc_ctor(0, 7, 3);
} else {
 x_213 = x_206;
}
lean_ctor_set(x_213, 0, x_212);
lean_ctor_set(x_213, 1, x_170);
lean_ctor_set(x_213, 2, x_171);
lean_ctor_set(x_213, 3, x_172);
lean_ctor_set(x_213, 4, x_173);
lean_ctor_set(x_213, 5, x_174);
lean_ctor_set(x_213, 6, x_175);
lean_ctor_set_uint8(x_213, sizeof(void*)*7, x_169);
lean_ctor_set_uint8(x_213, sizeof(void*)*7 + 1, x_176);
lean_ctor_set_uint8(x_213, sizeof(void*)*7 + 2, x_177);
lean_inc(x_12);
x_214 = l_Lean_Meta_isExprDefEq(x_202, x_9, x_213, x_12, x_13, x_14);
if (lean_obj_tag(x_214) == 0)
{
lean_object* x_215; lean_object* x_216; uint8_t x_217; 
x_215 = lean_ctor_get(x_214, 0);
lean_inc(x_215);
if (lean_is_exclusive(x_214)) {
 lean_ctor_release(x_214, 0);
 x_216 = x_214;
} else {
 lean_dec_ref(x_214);
 x_216 = lean_box(0);
}
x_217 = lean_unbox(x_215);
if (x_217 == 0)
{
lean_object* x_218; lean_object* x_219; lean_object* x_220; lean_object* x_221; 
lean_dec(x_215);
lean_dec(x_12);
x_218 = lean_box(x_10);
x_219 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_219, 0, x_19);
lean_ctor_set(x_219, 1, x_218);
x_220 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_220, 0, x_17);
lean_ctor_set(x_220, 1, x_219);
if (lean_is_scalar(x_216)) {
 x_221 = lean_alloc_ctor(0, 1, 0);
} else {
 x_221 = x_216;
}
lean_ctor_set(x_221, 0, x_220);
return x_221;
}
else
{
lean_object* x_222; 
lean_dec(x_216);
x_222 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_17, x_12);
if (lean_obj_tag(x_222) == 0)
{
lean_object* x_223; lean_object* x_224; 
x_223 = lean_ctor_get(x_222, 0);
lean_inc(x_223);
lean_dec_ref(x_222);
x_224 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_19, x_12);
lean_dec(x_12);
if (lean_obj_tag(x_224) == 0)
{
lean_object* x_225; lean_object* x_226; lean_object* x_227; lean_object* x_228; lean_object* x_229; 
x_225 = lean_ctor_get(x_224, 0);
lean_inc(x_225);
if (lean_is_exclusive(x_224)) {
 lean_ctor_release(x_224, 0);
 x_226 = x_224;
} else {
 lean_dec_ref(x_224);
 x_226 = lean_box(0);
}
x_227 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_227, 0, x_225);
lean_ctor_set(x_227, 1, x_215);
x_228 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_228, 0, x_223);
lean_ctor_set(x_228, 1, x_227);
if (lean_is_scalar(x_226)) {
 x_229 = lean_alloc_ctor(0, 1, 0);
} else {
 x_229 = x_226;
}
lean_ctor_set(x_229, 0, x_228);
return x_229;
}
else
{
lean_object* x_230; lean_object* x_231; lean_object* x_232; 
lean_dec(x_223);
lean_dec(x_215);
x_230 = lean_ctor_get(x_224, 0);
lean_inc(x_230);
if (lean_is_exclusive(x_224)) {
 lean_ctor_release(x_224, 0);
 x_231 = x_224;
} else {
 lean_dec_ref(x_224);
 x_231 = lean_box(0);
}
if (lean_is_scalar(x_231)) {
 x_232 = lean_alloc_ctor(1, 1, 0);
} else {
 x_232 = x_231;
}
lean_ctor_set(x_232, 0, x_230);
return x_232;
}
}
else
{
lean_object* x_233; lean_object* x_234; lean_object* x_235; 
lean_dec(x_215);
lean_dec(x_19);
lean_dec(x_12);
x_233 = lean_ctor_get(x_222, 0);
lean_inc(x_233);
if (lean_is_exclusive(x_222)) {
 lean_ctor_release(x_222, 0);
 x_234 = x_222;
} else {
 lean_dec_ref(x_222);
 x_234 = lean_box(0);
}
if (lean_is_scalar(x_234)) {
 x_235 = lean_alloc_ctor(1, 1, 0);
} else {
 x_235 = x_234;
}
lean_ctor_set(x_235, 0, x_233);
return x_235;
}
}
}
else
{
lean_object* x_236; lean_object* x_237; lean_object* x_238; 
lean_dec(x_19);
lean_dec(x_17);
lean_dec(x_12);
x_236 = lean_ctor_get(x_214, 0);
lean_inc(x_236);
if (lean_is_exclusive(x_214)) {
 lean_ctor_release(x_214, 0);
 x_237 = x_214;
} else {
 lean_dec_ref(x_214);
 x_237 = lean_box(0);
}
if (lean_is_scalar(x_237)) {
 x_238 = lean_alloc_ctor(1, 1, 0);
} else {
 x_238 = x_237;
}
lean_ctor_set(x_238, 0, x_236);
return x_238;
}
}
}
else
{
uint8_t x_239; 
lean_dec(x_17);
lean_dec(x_14);
lean_dec_ref(x_13);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
x_239 = !lean_is_exclusive(x_18);
if (x_239 == 0)
{
return x_18;
}
else
{
lean_object* x_240; lean_object* x_241; 
x_240 = lean_ctor_get(x_18, 0);
lean_inc(x_240);
lean_dec(x_18);
x_241 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_241, 0, x_240);
return x_241;
}
}
}
else
{
uint8_t x_242; 
lean_dec(x_14);
lean_dec_ref(x_13);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec(x_1);
x_242 = !lean_is_exclusive(x_16);
if (x_242 == 0)
{
return x_16;
}
else
{
lean_object* x_243; lean_object* x_244; 
x_243 = lean_ctor_get(x_16, 0);
lean_inc(x_243);
lean_dec(x_16);
x_244 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_244, 0, x_243);
return x_244;
}
}
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_findCancelFactor___closed__9;
x_2 = lp_mathlib_CancelDenoms_findCancelFactor___closed__8;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("InvOneClass", 11, 11);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toInv", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__2;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__1;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("DivInvOneMonoid", 15, 15);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toInvOneClass", 13, 13);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__5;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__4;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__7() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("DivisionMonoid", 14, 14);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__8() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toDivInvOneMonoid", 17, 17);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__8;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__7;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__10() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("DivisionCommMonoid", 18, 18);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__11() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toDivisionMonoid", 16, 16);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__12() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__11;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__10;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__13() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("CommGroupWithZero", 17, 17);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__14() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toDivisionCommMonoid", 20, 20);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__15() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__14;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__13;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__16() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Semifield", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__17() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toCommGroupWithZero", 19, 19);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__18() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__17;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__16;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__19() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toSemifield", 11, 11);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__0(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_14; 
lean_inc_ref(x_9);
x_14 = l_Lean_Meta_mkFreshExprMVar(x_1, x_2, x_3, x_9, x_10, x_11, x_12);
if (lean_obj_tag(x_14) == 0)
{
lean_object* x_15; lean_object* x_16; uint8_t x_17; 
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
lean_dec_ref(x_14);
x_16 = l_Lean_Meta_Context_config(x_9);
x_17 = !lean_is_exclusive(x_16);
if (x_17 == 0)
{
uint8_t x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; uint8_t x_25; uint8_t x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; uint8_t x_61; uint64_t x_62; uint8_t x_63; 
x_18 = lean_ctor_get_uint8(x_9, sizeof(void*)*7);
x_19 = lean_ctor_get(x_9, 1);
lean_inc(x_19);
x_20 = lean_ctor_get(x_9, 2);
lean_inc_ref(x_20);
x_21 = lean_ctor_get(x_9, 3);
lean_inc_ref(x_21);
x_22 = lean_ctor_get(x_9, 4);
lean_inc(x_22);
x_23 = lean_ctor_get(x_9, 5);
lean_inc(x_23);
x_24 = lean_ctor_get(x_9, 6);
lean_inc(x_24);
x_25 = lean_ctor_get_uint8(x_9, sizeof(void*)*7 + 1);
x_26 = lean_ctor_get_uint8(x_9, sizeof(void*)*7 + 2);
x_27 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__0;
lean_inc(x_4);
x_28 = l_Lean_Expr_const___override(x_27, x_4);
lean_inc_ref(x_5);
x_29 = l_Lean_Expr_app___override(x_28, x_5);
x_30 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__3;
lean_inc(x_4);
x_31 = l_Lean_Expr_const___override(x_30, x_4);
lean_inc_ref(x_5);
x_32 = l_Lean_Expr_app___override(x_31, x_5);
x_33 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__6;
lean_inc(x_4);
x_34 = l_Lean_Expr_const___override(x_33, x_4);
lean_inc_ref(x_5);
x_35 = l_Lean_Expr_app___override(x_34, x_5);
x_36 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__9;
lean_inc(x_4);
x_37 = l_Lean_Expr_const___override(x_36, x_4);
lean_inc_ref(x_5);
x_38 = l_Lean_Expr_app___override(x_37, x_5);
x_39 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__12;
lean_inc(x_4);
x_40 = l_Lean_Expr_const___override(x_39, x_4);
lean_inc_ref(x_5);
x_41 = l_Lean_Expr_app___override(x_40, x_5);
x_42 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__15;
lean_inc(x_4);
x_43 = l_Lean_Expr_const___override(x_42, x_4);
lean_inc_ref(x_5);
x_44 = l_Lean_Expr_app___override(x_43, x_5);
x_45 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__18;
lean_inc(x_4);
x_46 = l_Lean_Expr_const___override(x_45, x_4);
lean_inc_ref(x_5);
x_47 = l_Lean_Expr_app___override(x_46, x_5);
x_48 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__19;
x_49 = l_Lean_Name_mkStr2(x_6, x_48);
x_50 = l_Lean_Expr_const___override(x_49, x_4);
x_51 = l_Lean_Expr_app___override(x_50, x_5);
x_52 = l_Lean_Expr_app___override(x_51, x_7);
x_53 = l_Lean_Expr_app___override(x_47, x_52);
x_54 = l_Lean_Expr_app___override(x_44, x_53);
x_55 = l_Lean_Expr_app___override(x_41, x_54);
x_56 = l_Lean_Expr_app___override(x_38, x_55);
x_57 = l_Lean_Expr_app___override(x_35, x_56);
x_58 = l_Lean_Expr_app___override(x_32, x_57);
x_59 = l_Lean_Expr_app___override(x_29, x_58);
lean_inc(x_15);
x_60 = l_Lean_Expr_app___override(x_59, x_15);
x_61 = 2;
lean_ctor_set_uint8(x_16, 9, x_61);
x_62 = l_Lean_Meta_Context_configKey(x_9);
x_63 = !lean_is_exclusive(x_9);
if (x_63 == 0)
{
lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; uint64_t x_71; uint64_t x_72; uint64_t x_73; uint64_t x_74; uint64_t x_75; lean_object* x_76; lean_object* x_77; 
x_64 = lean_ctor_get(x_9, 6);
lean_dec(x_64);
x_65 = lean_ctor_get(x_9, 5);
lean_dec(x_65);
x_66 = lean_ctor_get(x_9, 4);
lean_dec(x_66);
x_67 = lean_ctor_get(x_9, 3);
lean_dec(x_67);
x_68 = lean_ctor_get(x_9, 2);
lean_dec(x_68);
x_69 = lean_ctor_get(x_9, 1);
lean_dec(x_69);
x_70 = lean_ctor_get(x_9, 0);
lean_dec(x_70);
x_71 = 2;
x_72 = lean_uint64_shift_right(x_62, x_71);
x_73 = lean_uint64_shift_left(x_72, x_71);
x_74 = lp_mathlib_CancelDenoms_mkProdPrf___lam__4___closed__0;
x_75 = lean_uint64_lor(x_73, x_74);
x_76 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_76, 0, x_16);
lean_ctor_set_uint64(x_76, sizeof(void*)*1, x_75);
lean_ctor_set(x_9, 0, x_76);
lean_inc(x_10);
x_77 = l_Lean_Meta_isExprDefEq(x_60, x_8, x_9, x_10, x_11, x_12);
if (lean_obj_tag(x_77) == 0)
{
uint8_t x_78; 
x_78 = !lean_is_exclusive(x_77);
if (x_78 == 0)
{
lean_object* x_79; uint8_t x_80; 
x_79 = lean_ctor_get(x_77, 0);
x_80 = lean_unbox(x_79);
if (x_80 == 0)
{
lean_object* x_81; 
lean_dec(x_10);
x_81 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_81, 0, x_15);
lean_ctor_set(x_81, 1, x_79);
lean_ctor_set(x_77, 0, x_81);
return x_77;
}
else
{
lean_object* x_82; 
lean_free_object(x_77);
x_82 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_15, x_10);
lean_dec(x_10);
if (lean_obj_tag(x_82) == 0)
{
uint8_t x_83; 
x_83 = !lean_is_exclusive(x_82);
if (x_83 == 0)
{
lean_object* x_84; lean_object* x_85; 
x_84 = lean_ctor_get(x_82, 0);
x_85 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_85, 0, x_84);
lean_ctor_set(x_85, 1, x_79);
lean_ctor_set(x_82, 0, x_85);
return x_82;
}
else
{
lean_object* x_86; lean_object* x_87; lean_object* x_88; 
x_86 = lean_ctor_get(x_82, 0);
lean_inc(x_86);
lean_dec(x_82);
x_87 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_87, 0, x_86);
lean_ctor_set(x_87, 1, x_79);
x_88 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_88, 0, x_87);
return x_88;
}
}
else
{
uint8_t x_89; 
lean_dec(x_79);
x_89 = !lean_is_exclusive(x_82);
if (x_89 == 0)
{
return x_82;
}
else
{
lean_object* x_90; lean_object* x_91; 
x_90 = lean_ctor_get(x_82, 0);
lean_inc(x_90);
lean_dec(x_82);
x_91 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_91, 0, x_90);
return x_91;
}
}
}
}
else
{
lean_object* x_92; uint8_t x_93; 
x_92 = lean_ctor_get(x_77, 0);
lean_inc(x_92);
lean_dec(x_77);
x_93 = lean_unbox(x_92);
if (x_93 == 0)
{
lean_object* x_94; lean_object* x_95; 
lean_dec(x_10);
x_94 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_94, 0, x_15);
lean_ctor_set(x_94, 1, x_92);
x_95 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_95, 0, x_94);
return x_95;
}
else
{
lean_object* x_96; 
x_96 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_15, x_10);
lean_dec(x_10);
if (lean_obj_tag(x_96) == 0)
{
lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; 
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
lean_ctor_set(x_99, 1, x_92);
if (lean_is_scalar(x_98)) {
 x_100 = lean_alloc_ctor(0, 1, 0);
} else {
 x_100 = x_98;
}
lean_ctor_set(x_100, 0, x_99);
return x_100;
}
else
{
lean_object* x_101; lean_object* x_102; lean_object* x_103; 
lean_dec(x_92);
x_101 = lean_ctor_get(x_96, 0);
lean_inc(x_101);
if (lean_is_exclusive(x_96)) {
 lean_ctor_release(x_96, 0);
 x_102 = x_96;
} else {
 lean_dec_ref(x_96);
 x_102 = lean_box(0);
}
if (lean_is_scalar(x_102)) {
 x_103 = lean_alloc_ctor(1, 1, 0);
} else {
 x_103 = x_102;
}
lean_ctor_set(x_103, 0, x_101);
return x_103;
}
}
}
}
else
{
uint8_t x_104; 
lean_dec(x_15);
lean_dec(x_10);
x_104 = !lean_is_exclusive(x_77);
if (x_104 == 0)
{
return x_77;
}
else
{
lean_object* x_105; lean_object* x_106; 
x_105 = lean_ctor_get(x_77, 0);
lean_inc(x_105);
lean_dec(x_77);
x_106 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_106, 0, x_105);
return x_106;
}
}
}
else
{
uint64_t x_107; uint64_t x_108; uint64_t x_109; uint64_t x_110; uint64_t x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; 
lean_dec(x_9);
x_107 = 2;
x_108 = lean_uint64_shift_right(x_62, x_107);
x_109 = lean_uint64_shift_left(x_108, x_107);
x_110 = lp_mathlib_CancelDenoms_mkProdPrf___lam__4___closed__0;
x_111 = lean_uint64_lor(x_109, x_110);
x_112 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_112, 0, x_16);
lean_ctor_set_uint64(x_112, sizeof(void*)*1, x_111);
x_113 = lean_alloc_ctor(0, 7, 3);
lean_ctor_set(x_113, 0, x_112);
lean_ctor_set(x_113, 1, x_19);
lean_ctor_set(x_113, 2, x_20);
lean_ctor_set(x_113, 3, x_21);
lean_ctor_set(x_113, 4, x_22);
lean_ctor_set(x_113, 5, x_23);
lean_ctor_set(x_113, 6, x_24);
lean_ctor_set_uint8(x_113, sizeof(void*)*7, x_18);
lean_ctor_set_uint8(x_113, sizeof(void*)*7 + 1, x_25);
lean_ctor_set_uint8(x_113, sizeof(void*)*7 + 2, x_26);
lean_inc(x_10);
x_114 = l_Lean_Meta_isExprDefEq(x_60, x_8, x_113, x_10, x_11, x_12);
if (lean_obj_tag(x_114) == 0)
{
lean_object* x_115; lean_object* x_116; uint8_t x_117; 
x_115 = lean_ctor_get(x_114, 0);
lean_inc(x_115);
if (lean_is_exclusive(x_114)) {
 lean_ctor_release(x_114, 0);
 x_116 = x_114;
} else {
 lean_dec_ref(x_114);
 x_116 = lean_box(0);
}
x_117 = lean_unbox(x_115);
if (x_117 == 0)
{
lean_object* x_118; lean_object* x_119; 
lean_dec(x_10);
x_118 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_118, 0, x_15);
lean_ctor_set(x_118, 1, x_115);
if (lean_is_scalar(x_116)) {
 x_119 = lean_alloc_ctor(0, 1, 0);
} else {
 x_119 = x_116;
}
lean_ctor_set(x_119, 0, x_118);
return x_119;
}
else
{
lean_object* x_120; 
lean_dec(x_116);
x_120 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_15, x_10);
lean_dec(x_10);
if (lean_obj_tag(x_120) == 0)
{
lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; 
x_121 = lean_ctor_get(x_120, 0);
lean_inc(x_121);
if (lean_is_exclusive(x_120)) {
 lean_ctor_release(x_120, 0);
 x_122 = x_120;
} else {
 lean_dec_ref(x_120);
 x_122 = lean_box(0);
}
x_123 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_123, 0, x_121);
lean_ctor_set(x_123, 1, x_115);
if (lean_is_scalar(x_122)) {
 x_124 = lean_alloc_ctor(0, 1, 0);
} else {
 x_124 = x_122;
}
lean_ctor_set(x_124, 0, x_123);
return x_124;
}
else
{
lean_object* x_125; lean_object* x_126; lean_object* x_127; 
lean_dec(x_115);
x_125 = lean_ctor_get(x_120, 0);
lean_inc(x_125);
if (lean_is_exclusive(x_120)) {
 lean_ctor_release(x_120, 0);
 x_126 = x_120;
} else {
 lean_dec_ref(x_120);
 x_126 = lean_box(0);
}
if (lean_is_scalar(x_126)) {
 x_127 = lean_alloc_ctor(1, 1, 0);
} else {
 x_127 = x_126;
}
lean_ctor_set(x_127, 0, x_125);
return x_127;
}
}
}
else
{
lean_object* x_128; lean_object* x_129; lean_object* x_130; 
lean_dec(x_15);
lean_dec(x_10);
x_128 = lean_ctor_get(x_114, 0);
lean_inc(x_128);
if (lean_is_exclusive(x_114)) {
 lean_ctor_release(x_114, 0);
 x_129 = x_114;
} else {
 lean_dec_ref(x_114);
 x_129 = lean_box(0);
}
if (lean_is_scalar(x_129)) {
 x_130 = lean_alloc_ctor(1, 1, 0);
} else {
 x_130 = x_129;
}
lean_ctor_set(x_130, 0, x_128);
return x_130;
}
}
}
else
{
uint8_t x_131; uint8_t x_132; uint8_t x_133; uint8_t x_134; uint8_t x_135; uint8_t x_136; uint8_t x_137; uint8_t x_138; uint8_t x_139; uint8_t x_140; uint8_t x_141; uint8_t x_142; uint8_t x_143; uint8_t x_144; uint8_t x_145; uint8_t x_146; uint8_t x_147; uint8_t x_148; uint8_t x_149; lean_object* x_150; lean_object* x_151; lean_object* x_152; lean_object* x_153; lean_object* x_154; lean_object* x_155; uint8_t x_156; uint8_t x_157; lean_object* x_158; lean_object* x_159; lean_object* x_160; lean_object* x_161; lean_object* x_162; lean_object* x_163; lean_object* x_164; lean_object* x_165; lean_object* x_166; lean_object* x_167; lean_object* x_168; lean_object* x_169; lean_object* x_170; lean_object* x_171; lean_object* x_172; lean_object* x_173; lean_object* x_174; lean_object* x_175; lean_object* x_176; lean_object* x_177; lean_object* x_178; lean_object* x_179; lean_object* x_180; lean_object* x_181; lean_object* x_182; lean_object* x_183; lean_object* x_184; lean_object* x_185; lean_object* x_186; lean_object* x_187; lean_object* x_188; lean_object* x_189; lean_object* x_190; lean_object* x_191; uint8_t x_192; lean_object* x_193; uint64_t x_194; lean_object* x_195; uint64_t x_196; uint64_t x_197; uint64_t x_198; uint64_t x_199; uint64_t x_200; lean_object* x_201; lean_object* x_202; lean_object* x_203; 
x_131 = lean_ctor_get_uint8(x_16, 0);
x_132 = lean_ctor_get_uint8(x_16, 1);
x_133 = lean_ctor_get_uint8(x_16, 2);
x_134 = lean_ctor_get_uint8(x_16, 3);
x_135 = lean_ctor_get_uint8(x_16, 4);
x_136 = lean_ctor_get_uint8(x_16, 5);
x_137 = lean_ctor_get_uint8(x_16, 6);
x_138 = lean_ctor_get_uint8(x_16, 7);
x_139 = lean_ctor_get_uint8(x_16, 8);
x_140 = lean_ctor_get_uint8(x_16, 10);
x_141 = lean_ctor_get_uint8(x_16, 11);
x_142 = lean_ctor_get_uint8(x_16, 12);
x_143 = lean_ctor_get_uint8(x_16, 13);
x_144 = lean_ctor_get_uint8(x_16, 14);
x_145 = lean_ctor_get_uint8(x_16, 15);
x_146 = lean_ctor_get_uint8(x_16, 16);
x_147 = lean_ctor_get_uint8(x_16, 17);
x_148 = lean_ctor_get_uint8(x_16, 18);
lean_dec(x_16);
x_149 = lean_ctor_get_uint8(x_9, sizeof(void*)*7);
x_150 = lean_ctor_get(x_9, 1);
lean_inc(x_150);
x_151 = lean_ctor_get(x_9, 2);
lean_inc_ref(x_151);
x_152 = lean_ctor_get(x_9, 3);
lean_inc_ref(x_152);
x_153 = lean_ctor_get(x_9, 4);
lean_inc(x_153);
x_154 = lean_ctor_get(x_9, 5);
lean_inc(x_154);
x_155 = lean_ctor_get(x_9, 6);
lean_inc(x_155);
x_156 = lean_ctor_get_uint8(x_9, sizeof(void*)*7 + 1);
x_157 = lean_ctor_get_uint8(x_9, sizeof(void*)*7 + 2);
x_158 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__0;
lean_inc(x_4);
x_159 = l_Lean_Expr_const___override(x_158, x_4);
lean_inc_ref(x_5);
x_160 = l_Lean_Expr_app___override(x_159, x_5);
x_161 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__3;
lean_inc(x_4);
x_162 = l_Lean_Expr_const___override(x_161, x_4);
lean_inc_ref(x_5);
x_163 = l_Lean_Expr_app___override(x_162, x_5);
x_164 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__6;
lean_inc(x_4);
x_165 = l_Lean_Expr_const___override(x_164, x_4);
lean_inc_ref(x_5);
x_166 = l_Lean_Expr_app___override(x_165, x_5);
x_167 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__9;
lean_inc(x_4);
x_168 = l_Lean_Expr_const___override(x_167, x_4);
lean_inc_ref(x_5);
x_169 = l_Lean_Expr_app___override(x_168, x_5);
x_170 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__12;
lean_inc(x_4);
x_171 = l_Lean_Expr_const___override(x_170, x_4);
lean_inc_ref(x_5);
x_172 = l_Lean_Expr_app___override(x_171, x_5);
x_173 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__15;
lean_inc(x_4);
x_174 = l_Lean_Expr_const___override(x_173, x_4);
lean_inc_ref(x_5);
x_175 = l_Lean_Expr_app___override(x_174, x_5);
x_176 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__18;
lean_inc(x_4);
x_177 = l_Lean_Expr_const___override(x_176, x_4);
lean_inc_ref(x_5);
x_178 = l_Lean_Expr_app___override(x_177, x_5);
x_179 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__19;
x_180 = l_Lean_Name_mkStr2(x_6, x_179);
x_181 = l_Lean_Expr_const___override(x_180, x_4);
x_182 = l_Lean_Expr_app___override(x_181, x_5);
x_183 = l_Lean_Expr_app___override(x_182, x_7);
x_184 = l_Lean_Expr_app___override(x_178, x_183);
x_185 = l_Lean_Expr_app___override(x_175, x_184);
x_186 = l_Lean_Expr_app___override(x_172, x_185);
x_187 = l_Lean_Expr_app___override(x_169, x_186);
x_188 = l_Lean_Expr_app___override(x_166, x_187);
x_189 = l_Lean_Expr_app___override(x_163, x_188);
x_190 = l_Lean_Expr_app___override(x_160, x_189);
lean_inc(x_15);
x_191 = l_Lean_Expr_app___override(x_190, x_15);
x_192 = 2;
x_193 = lean_alloc_ctor(0, 0, 19);
lean_ctor_set_uint8(x_193, 0, x_131);
lean_ctor_set_uint8(x_193, 1, x_132);
lean_ctor_set_uint8(x_193, 2, x_133);
lean_ctor_set_uint8(x_193, 3, x_134);
lean_ctor_set_uint8(x_193, 4, x_135);
lean_ctor_set_uint8(x_193, 5, x_136);
lean_ctor_set_uint8(x_193, 6, x_137);
lean_ctor_set_uint8(x_193, 7, x_138);
lean_ctor_set_uint8(x_193, 8, x_139);
lean_ctor_set_uint8(x_193, 9, x_192);
lean_ctor_set_uint8(x_193, 10, x_140);
lean_ctor_set_uint8(x_193, 11, x_141);
lean_ctor_set_uint8(x_193, 12, x_142);
lean_ctor_set_uint8(x_193, 13, x_143);
lean_ctor_set_uint8(x_193, 14, x_144);
lean_ctor_set_uint8(x_193, 15, x_145);
lean_ctor_set_uint8(x_193, 16, x_146);
lean_ctor_set_uint8(x_193, 17, x_147);
lean_ctor_set_uint8(x_193, 18, x_148);
x_194 = l_Lean_Meta_Context_configKey(x_9);
if (lean_is_exclusive(x_9)) {
 lean_ctor_release(x_9, 0);
 lean_ctor_release(x_9, 1);
 lean_ctor_release(x_9, 2);
 lean_ctor_release(x_9, 3);
 lean_ctor_release(x_9, 4);
 lean_ctor_release(x_9, 5);
 lean_ctor_release(x_9, 6);
 x_195 = x_9;
} else {
 lean_dec_ref(x_9);
 x_195 = lean_box(0);
}
x_196 = 2;
x_197 = lean_uint64_shift_right(x_194, x_196);
x_198 = lean_uint64_shift_left(x_197, x_196);
x_199 = lp_mathlib_CancelDenoms_mkProdPrf___lam__4___closed__0;
x_200 = lean_uint64_lor(x_198, x_199);
x_201 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_201, 0, x_193);
lean_ctor_set_uint64(x_201, sizeof(void*)*1, x_200);
if (lean_is_scalar(x_195)) {
 x_202 = lean_alloc_ctor(0, 7, 3);
} else {
 x_202 = x_195;
}
lean_ctor_set(x_202, 0, x_201);
lean_ctor_set(x_202, 1, x_150);
lean_ctor_set(x_202, 2, x_151);
lean_ctor_set(x_202, 3, x_152);
lean_ctor_set(x_202, 4, x_153);
lean_ctor_set(x_202, 5, x_154);
lean_ctor_set(x_202, 6, x_155);
lean_ctor_set_uint8(x_202, sizeof(void*)*7, x_149);
lean_ctor_set_uint8(x_202, sizeof(void*)*7 + 1, x_156);
lean_ctor_set_uint8(x_202, sizeof(void*)*7 + 2, x_157);
lean_inc(x_10);
x_203 = l_Lean_Meta_isExprDefEq(x_191, x_8, x_202, x_10, x_11, x_12);
if (lean_obj_tag(x_203) == 0)
{
lean_object* x_204; lean_object* x_205; uint8_t x_206; 
x_204 = lean_ctor_get(x_203, 0);
lean_inc(x_204);
if (lean_is_exclusive(x_203)) {
 lean_ctor_release(x_203, 0);
 x_205 = x_203;
} else {
 lean_dec_ref(x_203);
 x_205 = lean_box(0);
}
x_206 = lean_unbox(x_204);
if (x_206 == 0)
{
lean_object* x_207; lean_object* x_208; 
lean_dec(x_10);
x_207 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_207, 0, x_15);
lean_ctor_set(x_207, 1, x_204);
if (lean_is_scalar(x_205)) {
 x_208 = lean_alloc_ctor(0, 1, 0);
} else {
 x_208 = x_205;
}
lean_ctor_set(x_208, 0, x_207);
return x_208;
}
else
{
lean_object* x_209; 
lean_dec(x_205);
x_209 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_15, x_10);
lean_dec(x_10);
if (lean_obj_tag(x_209) == 0)
{
lean_object* x_210; lean_object* x_211; lean_object* x_212; lean_object* x_213; 
x_210 = lean_ctor_get(x_209, 0);
lean_inc(x_210);
if (lean_is_exclusive(x_209)) {
 lean_ctor_release(x_209, 0);
 x_211 = x_209;
} else {
 lean_dec_ref(x_209);
 x_211 = lean_box(0);
}
x_212 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_212, 0, x_210);
lean_ctor_set(x_212, 1, x_204);
if (lean_is_scalar(x_211)) {
 x_213 = lean_alloc_ctor(0, 1, 0);
} else {
 x_213 = x_211;
}
lean_ctor_set(x_213, 0, x_212);
return x_213;
}
else
{
lean_object* x_214; lean_object* x_215; lean_object* x_216; 
lean_dec(x_204);
x_214 = lean_ctor_get(x_209, 0);
lean_inc(x_214);
if (lean_is_exclusive(x_209)) {
 lean_ctor_release(x_209, 0);
 x_215 = x_209;
} else {
 lean_dec_ref(x_209);
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
}
else
{
lean_object* x_217; lean_object* x_218; lean_object* x_219; 
lean_dec(x_15);
lean_dec(x_10);
x_217 = lean_ctor_get(x_203, 0);
lean_inc(x_217);
if (lean_is_exclusive(x_203)) {
 lean_ctor_release(x_203, 0);
 x_218 = x_203;
} else {
 lean_dec_ref(x_203);
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
}
else
{
uint8_t x_220; 
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
x_220 = !lean_is_exclusive(x_14);
if (x_220 == 0)
{
return x_14;
}
else
{
lean_object* x_221; lean_object* x_222; 
x_221 = lean_ctor_get(x_14, 0);
lean_inc(x_221);
lean_dec(x_14);
x_222 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_222, 0, x_221);
return x_222;
}
}
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Nat", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__0;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_findCancelFactor___closed__10;
x_2 = lp_mathlib_CancelDenoms_findCancelFactor___closed__7;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("instHPow", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__3;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Monoid", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toNatPow", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__6;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__5;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__8() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("MonoidWithZero", 14, 14);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__9() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toMonoid", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__10() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__9;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__8;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__11() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Semiring", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__12() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toMonoidWithZero", 16, 16);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__13() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__12;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__11;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__14() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("DivisionSemiring", 16, 16);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__15() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toSemiring", 10, 10);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__16() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__15;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__14;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__17() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toDivisionSemiring", 18, 18);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__18() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__17;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__16;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__1(lean_object* x_1, uint8_t x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_16; 
lean_inc_ref(x_11);
lean_inc(x_3);
x_16 = l_Lean_Meta_mkFreshExprMVar(x_1, x_2, x_3, x_11, x_12, x_13, x_14);
if (lean_obj_tag(x_16) == 0)
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_17 = lean_ctor_get(x_16, 0);
lean_inc(x_17);
lean_dec_ref(x_16);
x_18 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__1;
lean_inc(x_4);
x_19 = l_Lean_Expr_const___override(x_18, x_4);
lean_inc_ref(x_19);
x_20 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_20, 0, x_19);
lean_inc_ref(x_11);
x_21 = l_Lean_Meta_mkFreshExprMVar(x_20, x_2, x_3, x_11, x_12, x_13, x_14);
if (lean_obj_tag(x_21) == 0)
{
lean_object* x_22; lean_object* x_23; uint8_t x_24; 
x_22 = lean_ctor_get(x_21, 0);
lean_inc(x_22);
lean_dec_ref(x_21);
x_23 = l_Lean_Meta_Context_config(x_11);
x_24 = !lean_is_exclusive(x_23);
if (x_24 == 0)
{
uint8_t x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; uint8_t x_32; uint8_t x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; uint8_t x_77; uint64_t x_78; uint8_t x_79; 
x_25 = lean_ctor_get_uint8(x_11, sizeof(void*)*7);
x_26 = lean_ctor_get(x_11, 1);
lean_inc(x_26);
x_27 = lean_ctor_get(x_11, 2);
lean_inc_ref(x_27);
x_28 = lean_ctor_get(x_11, 3);
lean_inc_ref(x_28);
x_29 = lean_ctor_get(x_11, 4);
lean_inc(x_29);
x_30 = lean_ctor_get(x_11, 5);
lean_inc(x_30);
x_31 = lean_ctor_get(x_11, 6);
lean_inc(x_31);
x_32 = lean_ctor_get_uint8(x_11, sizeof(void*)*7 + 1);
x_33 = lean_ctor_get_uint8(x_11, sizeof(void*)*7 + 2);
x_34 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__2;
x_35 = lean_box(0);
lean_inc(x_5);
x_36 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_36, 0, x_35);
lean_ctor_set(x_36, 1, x_5);
lean_inc(x_6);
x_37 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_37, 0, x_6);
lean_ctor_set(x_37, 1, x_36);
x_38 = l_Lean_Expr_const___override(x_34, x_37);
lean_inc_ref(x_7);
x_39 = l_Lean_Expr_app___override(x_38, x_7);
lean_inc_ref(x_19);
x_40 = l_Lean_Expr_app___override(x_39, x_19);
lean_inc_ref(x_7);
x_41 = l_Lean_Expr_app___override(x_40, x_7);
x_42 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__4;
x_43 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_43, 0, x_35);
lean_ctor_set(x_43, 1, x_4);
x_44 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_44, 0, x_6);
lean_ctor_set(x_44, 1, x_43);
x_45 = l_Lean_Expr_const___override(x_42, x_44);
lean_inc_ref(x_7);
x_46 = l_Lean_Expr_app___override(x_45, x_7);
x_47 = l_Lean_Expr_app___override(x_46, x_19);
x_48 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__7;
lean_inc(x_5);
x_49 = l_Lean_Expr_const___override(x_48, x_5);
lean_inc_ref(x_7);
x_50 = l_Lean_Expr_app___override(x_49, x_7);
x_51 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__10;
lean_inc(x_5);
x_52 = l_Lean_Expr_const___override(x_51, x_5);
lean_inc_ref(x_7);
x_53 = l_Lean_Expr_app___override(x_52, x_7);
x_54 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__13;
lean_inc(x_5);
x_55 = l_Lean_Expr_const___override(x_54, x_5);
lean_inc_ref(x_7);
x_56 = l_Lean_Expr_app___override(x_55, x_7);
x_57 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__16;
lean_inc(x_5);
x_58 = l_Lean_Expr_const___override(x_57, x_5);
lean_inc_ref(x_7);
x_59 = l_Lean_Expr_app___override(x_58, x_7);
x_60 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__18;
lean_inc(x_5);
x_61 = l_Lean_Expr_const___override(x_60, x_5);
lean_inc_ref(x_7);
x_62 = l_Lean_Expr_app___override(x_61, x_7);
x_63 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__19;
x_64 = l_Lean_Name_mkStr2(x_8, x_63);
x_65 = l_Lean_Expr_const___override(x_64, x_5);
x_66 = l_Lean_Expr_app___override(x_65, x_7);
x_67 = l_Lean_Expr_app___override(x_66, x_9);
x_68 = l_Lean_Expr_app___override(x_62, x_67);
x_69 = l_Lean_Expr_app___override(x_59, x_68);
x_70 = l_Lean_Expr_app___override(x_56, x_69);
x_71 = l_Lean_Expr_app___override(x_53, x_70);
x_72 = l_Lean_Expr_app___override(x_50, x_71);
x_73 = l_Lean_Expr_app___override(x_47, x_72);
x_74 = l_Lean_Expr_app___override(x_41, x_73);
lean_inc(x_17);
x_75 = l_Lean_Expr_app___override(x_74, x_17);
lean_inc(x_22);
x_76 = l_Lean_Expr_app___override(x_75, x_22);
x_77 = 2;
lean_ctor_set_uint8(x_23, 9, x_77);
x_78 = l_Lean_Meta_Context_configKey(x_11);
x_79 = !lean_is_exclusive(x_11);
if (x_79 == 0)
{
lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; uint64_t x_87; uint64_t x_88; uint64_t x_89; uint64_t x_90; uint64_t x_91; lean_object* x_92; lean_object* x_93; 
x_80 = lean_ctor_get(x_11, 6);
lean_dec(x_80);
x_81 = lean_ctor_get(x_11, 5);
lean_dec(x_81);
x_82 = lean_ctor_get(x_11, 4);
lean_dec(x_82);
x_83 = lean_ctor_get(x_11, 3);
lean_dec(x_83);
x_84 = lean_ctor_get(x_11, 2);
lean_dec(x_84);
x_85 = lean_ctor_get(x_11, 1);
lean_dec(x_85);
x_86 = lean_ctor_get(x_11, 0);
lean_dec(x_86);
x_87 = 2;
x_88 = lean_uint64_shift_right(x_78, x_87);
x_89 = lean_uint64_shift_left(x_88, x_87);
x_90 = lp_mathlib_CancelDenoms_mkProdPrf___lam__4___closed__0;
x_91 = lean_uint64_lor(x_89, x_90);
x_92 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_92, 0, x_23);
lean_ctor_set_uint64(x_92, sizeof(void*)*1, x_91);
lean_ctor_set(x_11, 0, x_92);
lean_inc(x_12);
x_93 = l_Lean_Meta_isExprDefEq(x_76, x_10, x_11, x_12, x_13, x_14);
if (lean_obj_tag(x_93) == 0)
{
uint8_t x_94; 
x_94 = !lean_is_exclusive(x_93);
if (x_94 == 0)
{
lean_object* x_95; uint8_t x_96; 
x_95 = lean_ctor_get(x_93, 0);
x_96 = lean_unbox(x_95);
if (x_96 == 0)
{
lean_object* x_97; lean_object* x_98; 
lean_dec(x_12);
x_97 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_97, 0, x_22);
lean_ctor_set(x_97, 1, x_95);
x_98 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_98, 0, x_17);
lean_ctor_set(x_98, 1, x_97);
lean_ctor_set(x_93, 0, x_98);
return x_93;
}
else
{
lean_object* x_99; 
lean_free_object(x_93);
x_99 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_17, x_12);
if (lean_obj_tag(x_99) == 0)
{
lean_object* x_100; lean_object* x_101; 
x_100 = lean_ctor_get(x_99, 0);
lean_inc(x_100);
lean_dec_ref(x_99);
x_101 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_22, x_12);
lean_dec(x_12);
if (lean_obj_tag(x_101) == 0)
{
uint8_t x_102; 
x_102 = !lean_is_exclusive(x_101);
if (x_102 == 0)
{
lean_object* x_103; lean_object* x_104; lean_object* x_105; 
x_103 = lean_ctor_get(x_101, 0);
x_104 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_104, 0, x_103);
lean_ctor_set(x_104, 1, x_95);
x_105 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_105, 0, x_100);
lean_ctor_set(x_105, 1, x_104);
lean_ctor_set(x_101, 0, x_105);
return x_101;
}
else
{
lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; 
x_106 = lean_ctor_get(x_101, 0);
lean_inc(x_106);
lean_dec(x_101);
x_107 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_107, 0, x_106);
lean_ctor_set(x_107, 1, x_95);
x_108 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_108, 0, x_100);
lean_ctor_set(x_108, 1, x_107);
x_109 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_109, 0, x_108);
return x_109;
}
}
else
{
uint8_t x_110; 
lean_dec(x_100);
lean_dec(x_95);
x_110 = !lean_is_exclusive(x_101);
if (x_110 == 0)
{
return x_101;
}
else
{
lean_object* x_111; lean_object* x_112; 
x_111 = lean_ctor_get(x_101, 0);
lean_inc(x_111);
lean_dec(x_101);
x_112 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_112, 0, x_111);
return x_112;
}
}
}
else
{
uint8_t x_113; 
lean_dec(x_95);
lean_dec(x_22);
lean_dec(x_12);
x_113 = !lean_is_exclusive(x_99);
if (x_113 == 0)
{
return x_99;
}
else
{
lean_object* x_114; lean_object* x_115; 
x_114 = lean_ctor_get(x_99, 0);
lean_inc(x_114);
lean_dec(x_99);
x_115 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_115, 0, x_114);
return x_115;
}
}
}
}
else
{
lean_object* x_116; uint8_t x_117; 
x_116 = lean_ctor_get(x_93, 0);
lean_inc(x_116);
lean_dec(x_93);
x_117 = lean_unbox(x_116);
if (x_117 == 0)
{
lean_object* x_118; lean_object* x_119; lean_object* x_120; 
lean_dec(x_12);
x_118 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_118, 0, x_22);
lean_ctor_set(x_118, 1, x_116);
x_119 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_119, 0, x_17);
lean_ctor_set(x_119, 1, x_118);
x_120 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_120, 0, x_119);
return x_120;
}
else
{
lean_object* x_121; 
x_121 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_17, x_12);
if (lean_obj_tag(x_121) == 0)
{
lean_object* x_122; lean_object* x_123; 
x_122 = lean_ctor_get(x_121, 0);
lean_inc(x_122);
lean_dec_ref(x_121);
x_123 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_22, x_12);
lean_dec(x_12);
if (lean_obj_tag(x_123) == 0)
{
lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_127; lean_object* x_128; 
x_124 = lean_ctor_get(x_123, 0);
lean_inc(x_124);
if (lean_is_exclusive(x_123)) {
 lean_ctor_release(x_123, 0);
 x_125 = x_123;
} else {
 lean_dec_ref(x_123);
 x_125 = lean_box(0);
}
x_126 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_126, 0, x_124);
lean_ctor_set(x_126, 1, x_116);
x_127 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_127, 0, x_122);
lean_ctor_set(x_127, 1, x_126);
if (lean_is_scalar(x_125)) {
 x_128 = lean_alloc_ctor(0, 1, 0);
} else {
 x_128 = x_125;
}
lean_ctor_set(x_128, 0, x_127);
return x_128;
}
else
{
lean_object* x_129; lean_object* x_130; lean_object* x_131; 
lean_dec(x_122);
lean_dec(x_116);
x_129 = lean_ctor_get(x_123, 0);
lean_inc(x_129);
if (lean_is_exclusive(x_123)) {
 lean_ctor_release(x_123, 0);
 x_130 = x_123;
} else {
 lean_dec_ref(x_123);
 x_130 = lean_box(0);
}
if (lean_is_scalar(x_130)) {
 x_131 = lean_alloc_ctor(1, 1, 0);
} else {
 x_131 = x_130;
}
lean_ctor_set(x_131, 0, x_129);
return x_131;
}
}
else
{
lean_object* x_132; lean_object* x_133; lean_object* x_134; 
lean_dec(x_116);
lean_dec(x_22);
lean_dec(x_12);
x_132 = lean_ctor_get(x_121, 0);
lean_inc(x_132);
if (lean_is_exclusive(x_121)) {
 lean_ctor_release(x_121, 0);
 x_133 = x_121;
} else {
 lean_dec_ref(x_121);
 x_133 = lean_box(0);
}
if (lean_is_scalar(x_133)) {
 x_134 = lean_alloc_ctor(1, 1, 0);
} else {
 x_134 = x_133;
}
lean_ctor_set(x_134, 0, x_132);
return x_134;
}
}
}
}
else
{
uint8_t x_135; 
lean_dec(x_22);
lean_dec(x_17);
lean_dec(x_12);
x_135 = !lean_is_exclusive(x_93);
if (x_135 == 0)
{
return x_93;
}
else
{
lean_object* x_136; lean_object* x_137; 
x_136 = lean_ctor_get(x_93, 0);
lean_inc(x_136);
lean_dec(x_93);
x_137 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_137, 0, x_136);
return x_137;
}
}
}
else
{
uint64_t x_138; uint64_t x_139; uint64_t x_140; uint64_t x_141; uint64_t x_142; lean_object* x_143; lean_object* x_144; lean_object* x_145; 
lean_dec(x_11);
x_138 = 2;
x_139 = lean_uint64_shift_right(x_78, x_138);
x_140 = lean_uint64_shift_left(x_139, x_138);
x_141 = lp_mathlib_CancelDenoms_mkProdPrf___lam__4___closed__0;
x_142 = lean_uint64_lor(x_140, x_141);
x_143 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_143, 0, x_23);
lean_ctor_set_uint64(x_143, sizeof(void*)*1, x_142);
x_144 = lean_alloc_ctor(0, 7, 3);
lean_ctor_set(x_144, 0, x_143);
lean_ctor_set(x_144, 1, x_26);
lean_ctor_set(x_144, 2, x_27);
lean_ctor_set(x_144, 3, x_28);
lean_ctor_set(x_144, 4, x_29);
lean_ctor_set(x_144, 5, x_30);
lean_ctor_set(x_144, 6, x_31);
lean_ctor_set_uint8(x_144, sizeof(void*)*7, x_25);
lean_ctor_set_uint8(x_144, sizeof(void*)*7 + 1, x_32);
lean_ctor_set_uint8(x_144, sizeof(void*)*7 + 2, x_33);
lean_inc(x_12);
x_145 = l_Lean_Meta_isExprDefEq(x_76, x_10, x_144, x_12, x_13, x_14);
if (lean_obj_tag(x_145) == 0)
{
lean_object* x_146; lean_object* x_147; uint8_t x_148; 
x_146 = lean_ctor_get(x_145, 0);
lean_inc(x_146);
if (lean_is_exclusive(x_145)) {
 lean_ctor_release(x_145, 0);
 x_147 = x_145;
} else {
 lean_dec_ref(x_145);
 x_147 = lean_box(0);
}
x_148 = lean_unbox(x_146);
if (x_148 == 0)
{
lean_object* x_149; lean_object* x_150; lean_object* x_151; 
lean_dec(x_12);
x_149 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_149, 0, x_22);
lean_ctor_set(x_149, 1, x_146);
x_150 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_150, 0, x_17);
lean_ctor_set(x_150, 1, x_149);
if (lean_is_scalar(x_147)) {
 x_151 = lean_alloc_ctor(0, 1, 0);
} else {
 x_151 = x_147;
}
lean_ctor_set(x_151, 0, x_150);
return x_151;
}
else
{
lean_object* x_152; 
lean_dec(x_147);
x_152 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_17, x_12);
if (lean_obj_tag(x_152) == 0)
{
lean_object* x_153; lean_object* x_154; 
x_153 = lean_ctor_get(x_152, 0);
lean_inc(x_153);
lean_dec_ref(x_152);
x_154 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_22, x_12);
lean_dec(x_12);
if (lean_obj_tag(x_154) == 0)
{
lean_object* x_155; lean_object* x_156; lean_object* x_157; lean_object* x_158; lean_object* x_159; 
x_155 = lean_ctor_get(x_154, 0);
lean_inc(x_155);
if (lean_is_exclusive(x_154)) {
 lean_ctor_release(x_154, 0);
 x_156 = x_154;
} else {
 lean_dec_ref(x_154);
 x_156 = lean_box(0);
}
x_157 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_157, 0, x_155);
lean_ctor_set(x_157, 1, x_146);
x_158 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_158, 0, x_153);
lean_ctor_set(x_158, 1, x_157);
if (lean_is_scalar(x_156)) {
 x_159 = lean_alloc_ctor(0, 1, 0);
} else {
 x_159 = x_156;
}
lean_ctor_set(x_159, 0, x_158);
return x_159;
}
else
{
lean_object* x_160; lean_object* x_161; lean_object* x_162; 
lean_dec(x_153);
lean_dec(x_146);
x_160 = lean_ctor_get(x_154, 0);
lean_inc(x_160);
if (lean_is_exclusive(x_154)) {
 lean_ctor_release(x_154, 0);
 x_161 = x_154;
} else {
 lean_dec_ref(x_154);
 x_161 = lean_box(0);
}
if (lean_is_scalar(x_161)) {
 x_162 = lean_alloc_ctor(1, 1, 0);
} else {
 x_162 = x_161;
}
lean_ctor_set(x_162, 0, x_160);
return x_162;
}
}
else
{
lean_object* x_163; lean_object* x_164; lean_object* x_165; 
lean_dec(x_146);
lean_dec(x_22);
lean_dec(x_12);
x_163 = lean_ctor_get(x_152, 0);
lean_inc(x_163);
if (lean_is_exclusive(x_152)) {
 lean_ctor_release(x_152, 0);
 x_164 = x_152;
} else {
 lean_dec_ref(x_152);
 x_164 = lean_box(0);
}
if (lean_is_scalar(x_164)) {
 x_165 = lean_alloc_ctor(1, 1, 0);
} else {
 x_165 = x_164;
}
lean_ctor_set(x_165, 0, x_163);
return x_165;
}
}
}
else
{
lean_object* x_166; lean_object* x_167; lean_object* x_168; 
lean_dec(x_22);
lean_dec(x_17);
lean_dec(x_12);
x_166 = lean_ctor_get(x_145, 0);
lean_inc(x_166);
if (lean_is_exclusive(x_145)) {
 lean_ctor_release(x_145, 0);
 x_167 = x_145;
} else {
 lean_dec_ref(x_145);
 x_167 = lean_box(0);
}
if (lean_is_scalar(x_167)) {
 x_168 = lean_alloc_ctor(1, 1, 0);
} else {
 x_168 = x_167;
}
lean_ctor_set(x_168, 0, x_166);
return x_168;
}
}
}
else
{
uint8_t x_169; uint8_t x_170; uint8_t x_171; uint8_t x_172; uint8_t x_173; uint8_t x_174; uint8_t x_175; uint8_t x_176; uint8_t x_177; uint8_t x_178; uint8_t x_179; uint8_t x_180; uint8_t x_181; uint8_t x_182; uint8_t x_183; uint8_t x_184; uint8_t x_185; uint8_t x_186; uint8_t x_187; lean_object* x_188; lean_object* x_189; lean_object* x_190; lean_object* x_191; lean_object* x_192; lean_object* x_193; uint8_t x_194; uint8_t x_195; lean_object* x_196; lean_object* x_197; lean_object* x_198; lean_object* x_199; lean_object* x_200; lean_object* x_201; lean_object* x_202; lean_object* x_203; lean_object* x_204; lean_object* x_205; lean_object* x_206; lean_object* x_207; lean_object* x_208; lean_object* x_209; lean_object* x_210; lean_object* x_211; lean_object* x_212; lean_object* x_213; lean_object* x_214; lean_object* x_215; lean_object* x_216; lean_object* x_217; lean_object* x_218; lean_object* x_219; lean_object* x_220; lean_object* x_221; lean_object* x_222; lean_object* x_223; lean_object* x_224; lean_object* x_225; lean_object* x_226; lean_object* x_227; lean_object* x_228; lean_object* x_229; lean_object* x_230; lean_object* x_231; lean_object* x_232; lean_object* x_233; lean_object* x_234; lean_object* x_235; lean_object* x_236; lean_object* x_237; lean_object* x_238; uint8_t x_239; lean_object* x_240; uint64_t x_241; lean_object* x_242; uint64_t x_243; uint64_t x_244; uint64_t x_245; uint64_t x_246; uint64_t x_247; lean_object* x_248; lean_object* x_249; lean_object* x_250; 
x_169 = lean_ctor_get_uint8(x_23, 0);
x_170 = lean_ctor_get_uint8(x_23, 1);
x_171 = lean_ctor_get_uint8(x_23, 2);
x_172 = lean_ctor_get_uint8(x_23, 3);
x_173 = lean_ctor_get_uint8(x_23, 4);
x_174 = lean_ctor_get_uint8(x_23, 5);
x_175 = lean_ctor_get_uint8(x_23, 6);
x_176 = lean_ctor_get_uint8(x_23, 7);
x_177 = lean_ctor_get_uint8(x_23, 8);
x_178 = lean_ctor_get_uint8(x_23, 10);
x_179 = lean_ctor_get_uint8(x_23, 11);
x_180 = lean_ctor_get_uint8(x_23, 12);
x_181 = lean_ctor_get_uint8(x_23, 13);
x_182 = lean_ctor_get_uint8(x_23, 14);
x_183 = lean_ctor_get_uint8(x_23, 15);
x_184 = lean_ctor_get_uint8(x_23, 16);
x_185 = lean_ctor_get_uint8(x_23, 17);
x_186 = lean_ctor_get_uint8(x_23, 18);
lean_dec(x_23);
x_187 = lean_ctor_get_uint8(x_11, sizeof(void*)*7);
x_188 = lean_ctor_get(x_11, 1);
lean_inc(x_188);
x_189 = lean_ctor_get(x_11, 2);
lean_inc_ref(x_189);
x_190 = lean_ctor_get(x_11, 3);
lean_inc_ref(x_190);
x_191 = lean_ctor_get(x_11, 4);
lean_inc(x_191);
x_192 = lean_ctor_get(x_11, 5);
lean_inc(x_192);
x_193 = lean_ctor_get(x_11, 6);
lean_inc(x_193);
x_194 = lean_ctor_get_uint8(x_11, sizeof(void*)*7 + 1);
x_195 = lean_ctor_get_uint8(x_11, sizeof(void*)*7 + 2);
x_196 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__2;
x_197 = lean_box(0);
lean_inc(x_5);
x_198 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_198, 0, x_197);
lean_ctor_set(x_198, 1, x_5);
lean_inc(x_6);
x_199 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_199, 0, x_6);
lean_ctor_set(x_199, 1, x_198);
x_200 = l_Lean_Expr_const___override(x_196, x_199);
lean_inc_ref(x_7);
x_201 = l_Lean_Expr_app___override(x_200, x_7);
lean_inc_ref(x_19);
x_202 = l_Lean_Expr_app___override(x_201, x_19);
lean_inc_ref(x_7);
x_203 = l_Lean_Expr_app___override(x_202, x_7);
x_204 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__4;
x_205 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_205, 0, x_197);
lean_ctor_set(x_205, 1, x_4);
x_206 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_206, 0, x_6);
lean_ctor_set(x_206, 1, x_205);
x_207 = l_Lean_Expr_const___override(x_204, x_206);
lean_inc_ref(x_7);
x_208 = l_Lean_Expr_app___override(x_207, x_7);
x_209 = l_Lean_Expr_app___override(x_208, x_19);
x_210 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__7;
lean_inc(x_5);
x_211 = l_Lean_Expr_const___override(x_210, x_5);
lean_inc_ref(x_7);
x_212 = l_Lean_Expr_app___override(x_211, x_7);
x_213 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__10;
lean_inc(x_5);
x_214 = l_Lean_Expr_const___override(x_213, x_5);
lean_inc_ref(x_7);
x_215 = l_Lean_Expr_app___override(x_214, x_7);
x_216 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__13;
lean_inc(x_5);
x_217 = l_Lean_Expr_const___override(x_216, x_5);
lean_inc_ref(x_7);
x_218 = l_Lean_Expr_app___override(x_217, x_7);
x_219 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__16;
lean_inc(x_5);
x_220 = l_Lean_Expr_const___override(x_219, x_5);
lean_inc_ref(x_7);
x_221 = l_Lean_Expr_app___override(x_220, x_7);
x_222 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__18;
lean_inc(x_5);
x_223 = l_Lean_Expr_const___override(x_222, x_5);
lean_inc_ref(x_7);
x_224 = l_Lean_Expr_app___override(x_223, x_7);
x_225 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__19;
x_226 = l_Lean_Name_mkStr2(x_8, x_225);
x_227 = l_Lean_Expr_const___override(x_226, x_5);
x_228 = l_Lean_Expr_app___override(x_227, x_7);
x_229 = l_Lean_Expr_app___override(x_228, x_9);
x_230 = l_Lean_Expr_app___override(x_224, x_229);
x_231 = l_Lean_Expr_app___override(x_221, x_230);
x_232 = l_Lean_Expr_app___override(x_218, x_231);
x_233 = l_Lean_Expr_app___override(x_215, x_232);
x_234 = l_Lean_Expr_app___override(x_212, x_233);
x_235 = l_Lean_Expr_app___override(x_209, x_234);
x_236 = l_Lean_Expr_app___override(x_203, x_235);
lean_inc(x_17);
x_237 = l_Lean_Expr_app___override(x_236, x_17);
lean_inc(x_22);
x_238 = l_Lean_Expr_app___override(x_237, x_22);
x_239 = 2;
x_240 = lean_alloc_ctor(0, 0, 19);
lean_ctor_set_uint8(x_240, 0, x_169);
lean_ctor_set_uint8(x_240, 1, x_170);
lean_ctor_set_uint8(x_240, 2, x_171);
lean_ctor_set_uint8(x_240, 3, x_172);
lean_ctor_set_uint8(x_240, 4, x_173);
lean_ctor_set_uint8(x_240, 5, x_174);
lean_ctor_set_uint8(x_240, 6, x_175);
lean_ctor_set_uint8(x_240, 7, x_176);
lean_ctor_set_uint8(x_240, 8, x_177);
lean_ctor_set_uint8(x_240, 9, x_239);
lean_ctor_set_uint8(x_240, 10, x_178);
lean_ctor_set_uint8(x_240, 11, x_179);
lean_ctor_set_uint8(x_240, 12, x_180);
lean_ctor_set_uint8(x_240, 13, x_181);
lean_ctor_set_uint8(x_240, 14, x_182);
lean_ctor_set_uint8(x_240, 15, x_183);
lean_ctor_set_uint8(x_240, 16, x_184);
lean_ctor_set_uint8(x_240, 17, x_185);
lean_ctor_set_uint8(x_240, 18, x_186);
x_241 = l_Lean_Meta_Context_configKey(x_11);
if (lean_is_exclusive(x_11)) {
 lean_ctor_release(x_11, 0);
 lean_ctor_release(x_11, 1);
 lean_ctor_release(x_11, 2);
 lean_ctor_release(x_11, 3);
 lean_ctor_release(x_11, 4);
 lean_ctor_release(x_11, 5);
 lean_ctor_release(x_11, 6);
 x_242 = x_11;
} else {
 lean_dec_ref(x_11);
 x_242 = lean_box(0);
}
x_243 = 2;
x_244 = lean_uint64_shift_right(x_241, x_243);
x_245 = lean_uint64_shift_left(x_244, x_243);
x_246 = lp_mathlib_CancelDenoms_mkProdPrf___lam__4___closed__0;
x_247 = lean_uint64_lor(x_245, x_246);
x_248 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_248, 0, x_240);
lean_ctor_set_uint64(x_248, sizeof(void*)*1, x_247);
if (lean_is_scalar(x_242)) {
 x_249 = lean_alloc_ctor(0, 7, 3);
} else {
 x_249 = x_242;
}
lean_ctor_set(x_249, 0, x_248);
lean_ctor_set(x_249, 1, x_188);
lean_ctor_set(x_249, 2, x_189);
lean_ctor_set(x_249, 3, x_190);
lean_ctor_set(x_249, 4, x_191);
lean_ctor_set(x_249, 5, x_192);
lean_ctor_set(x_249, 6, x_193);
lean_ctor_set_uint8(x_249, sizeof(void*)*7, x_187);
lean_ctor_set_uint8(x_249, sizeof(void*)*7 + 1, x_194);
lean_ctor_set_uint8(x_249, sizeof(void*)*7 + 2, x_195);
lean_inc(x_12);
x_250 = l_Lean_Meta_isExprDefEq(x_238, x_10, x_249, x_12, x_13, x_14);
if (lean_obj_tag(x_250) == 0)
{
lean_object* x_251; lean_object* x_252; uint8_t x_253; 
x_251 = lean_ctor_get(x_250, 0);
lean_inc(x_251);
if (lean_is_exclusive(x_250)) {
 lean_ctor_release(x_250, 0);
 x_252 = x_250;
} else {
 lean_dec_ref(x_250);
 x_252 = lean_box(0);
}
x_253 = lean_unbox(x_251);
if (x_253 == 0)
{
lean_object* x_254; lean_object* x_255; lean_object* x_256; 
lean_dec(x_12);
x_254 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_254, 0, x_22);
lean_ctor_set(x_254, 1, x_251);
x_255 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_255, 0, x_17);
lean_ctor_set(x_255, 1, x_254);
if (lean_is_scalar(x_252)) {
 x_256 = lean_alloc_ctor(0, 1, 0);
} else {
 x_256 = x_252;
}
lean_ctor_set(x_256, 0, x_255);
return x_256;
}
else
{
lean_object* x_257; 
lean_dec(x_252);
x_257 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_17, x_12);
if (lean_obj_tag(x_257) == 0)
{
lean_object* x_258; lean_object* x_259; 
x_258 = lean_ctor_get(x_257, 0);
lean_inc(x_258);
lean_dec_ref(x_257);
x_259 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_22, x_12);
lean_dec(x_12);
if (lean_obj_tag(x_259) == 0)
{
lean_object* x_260; lean_object* x_261; lean_object* x_262; lean_object* x_263; lean_object* x_264; 
x_260 = lean_ctor_get(x_259, 0);
lean_inc(x_260);
if (lean_is_exclusive(x_259)) {
 lean_ctor_release(x_259, 0);
 x_261 = x_259;
} else {
 lean_dec_ref(x_259);
 x_261 = lean_box(0);
}
x_262 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_262, 0, x_260);
lean_ctor_set(x_262, 1, x_251);
x_263 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_263, 0, x_258);
lean_ctor_set(x_263, 1, x_262);
if (lean_is_scalar(x_261)) {
 x_264 = lean_alloc_ctor(0, 1, 0);
} else {
 x_264 = x_261;
}
lean_ctor_set(x_264, 0, x_263);
return x_264;
}
else
{
lean_object* x_265; lean_object* x_266; lean_object* x_267; 
lean_dec(x_258);
lean_dec(x_251);
x_265 = lean_ctor_get(x_259, 0);
lean_inc(x_265);
if (lean_is_exclusive(x_259)) {
 lean_ctor_release(x_259, 0);
 x_266 = x_259;
} else {
 lean_dec_ref(x_259);
 x_266 = lean_box(0);
}
if (lean_is_scalar(x_266)) {
 x_267 = lean_alloc_ctor(1, 1, 0);
} else {
 x_267 = x_266;
}
lean_ctor_set(x_267, 0, x_265);
return x_267;
}
}
else
{
lean_object* x_268; lean_object* x_269; lean_object* x_270; 
lean_dec(x_251);
lean_dec(x_22);
lean_dec(x_12);
x_268 = lean_ctor_get(x_257, 0);
lean_inc(x_268);
if (lean_is_exclusive(x_257)) {
 lean_ctor_release(x_257, 0);
 x_269 = x_257;
} else {
 lean_dec_ref(x_257);
 x_269 = lean_box(0);
}
if (lean_is_scalar(x_269)) {
 x_270 = lean_alloc_ctor(1, 1, 0);
} else {
 x_270 = x_269;
}
lean_ctor_set(x_270, 0, x_268);
return x_270;
}
}
}
else
{
lean_object* x_271; lean_object* x_272; lean_object* x_273; 
lean_dec(x_22);
lean_dec(x_17);
lean_dec(x_12);
x_271 = lean_ctor_get(x_250, 0);
lean_inc(x_271);
if (lean_is_exclusive(x_250)) {
 lean_ctor_release(x_250, 0);
 x_272 = x_250;
} else {
 lean_dec_ref(x_250);
 x_272 = lean_box(0);
}
if (lean_is_scalar(x_272)) {
 x_273 = lean_alloc_ctor(1, 1, 0);
} else {
 x_273 = x_272;
}
lean_ctor_set(x_273, 0, x_271);
return x_273;
}
}
}
else
{
uint8_t x_274; 
lean_dec_ref(x_19);
lean_dec(x_17);
lean_dec(x_14);
lean_dec_ref(x_13);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
x_274 = !lean_is_exclusive(x_21);
if (x_274 == 0)
{
return x_21;
}
else
{
lean_object* x_275; lean_object* x_276; 
x_275 = lean_ctor_get(x_21, 0);
lean_inc(x_275);
lean_dec(x_21);
x_276 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_276, 0, x_275);
return x_276;
}
}
}
else
{
uint8_t x_277; 
lean_dec(x_14);
lean_dec_ref(x_13);
lean_dec(x_12);
lean_dec_ref(x_11);
lean_dec_ref(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
x_277 = !lean_is_exclusive(x_16);
if (x_277 == 0)
{
return x_16;
}
else
{
lean_object* x_278; lean_object* x_279; 
x_278 = lean_ctor_get(x_16, 0);
lean_inc(x_278);
lean_dec(x_16);
x_279 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_279, 0, x_278);
return x_279;
}
}
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("AddMonoidWithOne", 16, 16);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__0;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("inferInstance", 13, 13);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__2;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("AddGroupWithOne", 15, 15);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toAddMonoidWithOne", 18, 18);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__5;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___closed__4;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__7() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Ring", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__8() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toAddGroupWithOne", 17, 17);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__8;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___closed__7;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__10() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("DivisionRing", 12, 12);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__11() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toRing", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__12() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__11;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___closed__10;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__13() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Field", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__14() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toDivisionRing", 14, 14);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__15() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__14;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___closed__13;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__16() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Ne", 2, 2);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__17() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__16;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__18() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("OfNat", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__19() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("ofNat", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__20() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__19;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___closed__18;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__21() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__22() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__21;
x_2 = l_Lean_Expr_lit___override(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__23() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Zero", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__24() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toOfNat0", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__25() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__24;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___closed__23;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__26() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("MulZeroClass", 12, 12);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__27() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toZero", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__28() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__27;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___closed__26;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__29() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toMulZeroClass", 14, 14);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__30() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("inv_subst", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__31() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__30;
x_2 = lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__32() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__1;
x_3 = l_Lean_Expr_const___override(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__33() {
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
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__34() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__19;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___closed__13;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__35() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("pow_subst", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__36() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__35;
x_2 = lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__37() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__16;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___closed__7;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__38() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("neg_subst", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__39() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__38;
x_2 = lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__40() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__6;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___closed__10;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__41() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__42() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__41;
x_2 = l_Lean_Expr_lit___override(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__43() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("One", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__44() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toOfNat1", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__45() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__44;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___closed__43;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__46() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toOne", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__47() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__46;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___closed__0;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__48() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("div_subst", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__49() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__48;
x_2 = lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__50() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("mul_subst", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__51() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__50;
x_2 = lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__52() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("recursing into mul", 18, 18);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__53() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__52;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__54() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__9;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___closed__4;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__55() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("sub_subst", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__56() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__55;
x_2 = lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__57() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Mul", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__58() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__57;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__59() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Distrib", 7, 7);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__60() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toMul", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__61() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__60;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___closed__59;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__62() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("NonUnitalNonAssocSemiring", 25, 25);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__63() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toDistrib", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__64() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__63;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___closed__62;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__65() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("NonUnitalNonAssocRing", 21, 21);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__66() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toNonUnitalNonAssocSemiring", 27, 27);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__67() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__66;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___closed__65;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__68() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("NonUnitalNonAssocCommRing", 25, 25);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__69() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toNonUnitalNonAssocRing", 23, 23);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__70() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__69;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___closed__68;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__71() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("NonUnitalCommRing", 17, 17);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__72() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toNonUnitalNonAssocCommRing", 27, 27);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__73() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__72;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___closed__71;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__74() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("CommRing", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__75() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toNonUnitalCommRing", 19, 19);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__76() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__75;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___closed__74;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__77() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toCommRing", 10, 10);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__78() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__77;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___closed__13;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__79() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_findCancelFactor___closed__13;
x_2 = lp_mathlib_CancelDenoms_findCancelFactor___closed__4;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__80() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("instHMul", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__81() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__80;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__82() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Eq", 2, 2);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__83() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__82;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__84() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("rfl", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__85() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__84;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__86() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__3;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___closed__59;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__87() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("add_subst", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__88() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__87;
x_2 = lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__89() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__11;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___closed__74;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__90() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("mkProdPrf ", 10, 10);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__91() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__90;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__92() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(" ", 1, 1);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__93() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__92;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
uint8_t x_14; lean_object* x_15; 
x_14 = lean_unbox(x_2);
x_15 = lp_mathlib_CancelDenoms_mkProdPrf___lam__0(x_1, x_14, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
uint8_t x_16; lean_object* x_17; 
x_16 = lean_unbox(x_2);
x_17 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1(x_1, x_16, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
uint8_t x_14; lean_object* x_15; 
x_14 = lean_unbox(x_2);
x_15 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2(x_1, x_14, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
uint8_t x_16; uint8_t x_17; lean_object* x_18; 
x_16 = lean_unbox(x_2);
x_17 = lean_unbox(x_10);
x_18 = lp_mathlib_CancelDenoms_mkProdPrf___lam__3(x_1, x_16, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_17, x_11, x_12, x_13, x_14);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__4___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
uint8_t x_12; uint8_t x_13; lean_object* x_14; 
x_12 = lean_unbox(x_2);
x_13 = lean_unbox(x_6);
x_14 = lp_mathlib_CancelDenoms_mkProdPrf___lam__4(x_1, x_12, x_3, x_4, x_5, x_13, x_7, x_8, x_9, x_10);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__5___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
uint8_t x_16; uint8_t x_17; lean_object* x_18; 
x_16 = lean_unbox(x_2);
x_17 = lean_unbox(x_10);
x_18 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5(x_1, x_16, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_17, x_11, x_12, x_13, x_14);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___lam__6___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15) {
_start:
{
uint8_t x_16; uint8_t x_17; lean_object* x_18; 
x_16 = lean_unbox(x_2);
x_17 = lean_unbox(x_10);
x_18 = lp_mathlib_CancelDenoms_mkProdPrf___lam__6(x_1, x_16, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_17, x_11, x_12, x_13, x_14);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_13; lean_object* x_14; 
x_13 = lp_mathlib_initFn___closed__1_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_14 = lp_mathlib_Lean_isTracingEnabledFor___at___00CancelDenoms_mkProdPrf_spec__0___redArg(x_13, x_10);
if (lean_obj_tag(x_14) == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_244; lean_object* x_245; lean_object* x_246; lean_object* x_247; lean_object* x_248; lean_object* x_249; lean_object* x_250; lean_object* x_251; lean_object* x_252; lean_object* x_253; lean_object* x_254; lean_object* x_255; lean_object* x_256; lean_object* x_257; lean_object* x_821; lean_object* x_822; lean_object* x_823; lean_object* x_824; lean_object* x_825; lean_object* x_826; lean_object* x_827; lean_object* x_828; lean_object* x_829; lean_object* x_830; lean_object* x_838; uint8_t x_839; lean_object* x_840; lean_object* x_841; lean_object* x_842; lean_object* x_843; lean_object* x_844; lean_object* x_845; lean_object* x_846; lean_object* x_847; lean_object* x_848; lean_object* x_849; lean_object* x_850; lean_object* x_851; lean_object* x_852; lean_object* x_853; lean_object* x_854; uint8_t x_1174; lean_object* x_1175; lean_object* x_1176; lean_object* x_1177; lean_object* x_1178; lean_object* x_1179; lean_object* x_1180; lean_object* x_1181; lean_object* x_1182; lean_object* x_1183; lean_object* x_1184; lean_object* x_1185; lean_object* x_1186; lean_object* x_1328; uint8_t x_1329; lean_object* x_1330; lean_object* x_1331; lean_object* x_1332; lean_object* x_1333; lean_object* x_1334; lean_object* x_1335; lean_object* x_1336; lean_object* x_1337; lean_object* x_1338; lean_object* x_1339; lean_object* x_1340; lean_object* x_1341; lean_object* x_1342; lean_object* x_1347; lean_object* x_1348; lean_object* x_1349; lean_object* x_1350; lean_object* x_1351; lean_object* x_1352; lean_object* x_1353; lean_object* x_1354; lean_object* x_1355; lean_object* x_1356; lean_object* x_1357; lean_object* x_1358; lean_object* x_1359; lean_object* x_1459; uint8_t x_1460; lean_object* x_1461; lean_object* x_1462; lean_object* x_1463; lean_object* x_1464; lean_object* x_1465; lean_object* x_1466; lean_object* x_1467; lean_object* x_1468; lean_object* x_1469; lean_object* x_1470; lean_object* x_1471; lean_object* x_1472; lean_object* x_1473; lean_object* x_1474; lean_object* x_1475; lean_object* x_1476; lean_object* x_1477; lean_object* x_1503; lean_object* x_1504; lean_object* x_1505; uint8_t x_1506; lean_object* x_1507; lean_object* x_1508; lean_object* x_1509; lean_object* x_1510; lean_object* x_1511; lean_object* x_1512; lean_object* x_1513; lean_object* x_1514; lean_object* x_1515; lean_object* x_1516; lean_object* x_1517; lean_object* x_1518; lean_object* x_1519; lean_object* x_1520; lean_object* x_1521; lean_object* x_1660; lean_object* x_1661; lean_object* x_1662; lean_object* x_1663; lean_object* x_1664; uint8_t x_1874; 
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
if (lean_is_exclusive(x_14)) {
 lean_ctor_release(x_14, 0);
 x_16 = x_14;
} else {
 lean_dec_ref(x_14);
 x_16 = lean_box(0);
}
x_17 = lp_mathlib_CancelDenoms_mkProdPrf___closed__1;
x_18 = lean_box(0);
lean_inc(x_1);
x_19 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_19, 0, x_1);
lean_ctor_set(x_19, 1, x_18);
lean_inc_ref(x_19);
x_20 = l_Lean_Expr_const___override(x_17, x_19);
lean_inc_ref(x_2);
x_21 = l_Lean_Expr_app___override(x_20, x_2);
x_22 = lp_mathlib_CancelDenoms_mkProdPrf___closed__3;
lean_inc(x_1);
x_23 = l_Lean_Level_succ___override(x_1);
x_24 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_24, 0, x_23);
lean_ctor_set(x_24, 1, x_18);
lean_inc_ref(x_24);
x_25 = l_Lean_Expr_const___override(x_22, x_24);
lean_inc_ref(x_25);
x_26 = l_Lean_Expr_app___override(x_25, x_21);
x_27 = lp_mathlib_CancelDenoms_mkProdPrf___closed__4;
x_28 = lp_mathlib_CancelDenoms_mkProdPrf___closed__6;
lean_inc_ref(x_19);
x_29 = l_Lean_Expr_const___override(x_28, x_19);
lean_inc_ref(x_2);
x_30 = l_Lean_Expr_app___override(x_29, x_2);
x_31 = lp_mathlib_CancelDenoms_mkProdPrf___closed__7;
x_32 = lp_mathlib_CancelDenoms_mkProdPrf___closed__9;
lean_inc_ref(x_19);
x_33 = l_Lean_Expr_const___override(x_32, x_19);
lean_inc_ref(x_2);
x_34 = l_Lean_Expr_app___override(x_33, x_2);
x_35 = lp_mathlib_CancelDenoms_mkProdPrf___closed__10;
x_36 = lp_mathlib_CancelDenoms_mkProdPrf___closed__12;
lean_inc_ref(x_19);
x_37 = l_Lean_Expr_const___override(x_36, x_19);
lean_inc_ref(x_2);
x_38 = l_Lean_Expr_app___override(x_37, x_2);
x_39 = lp_mathlib_CancelDenoms_mkProdPrf___closed__13;
x_40 = lp_mathlib_CancelDenoms_mkProdPrf___closed__15;
lean_inc_ref(x_19);
x_41 = l_Lean_Expr_const___override(x_40, x_19);
lean_inc_ref(x_2);
x_42 = l_Lean_Expr_app___override(x_41, x_2);
lean_inc_ref(x_3);
x_43 = l_Lean_Expr_app___override(x_42, x_3);
lean_inc_ref(x_43);
x_44 = l_Lean_Expr_app___override(x_38, x_43);
lean_inc_ref(x_44);
x_45 = l_Lean_Expr_app___override(x_34, x_44);
lean_inc_ref(x_45);
x_46 = l_Lean_Expr_app___override(x_30, x_45);
x_47 = l_Lean_Expr_app___override(x_26, x_46);
x_1874 = lean_unbox(x_15);
lean_dec(x_15);
if (x_1874 == 0)
{
x_1660 = x_8;
x_1661 = x_9;
x_1662 = x_10;
x_1663 = x_11;
x_1664 = lean_box(0);
goto block_1873;
}
else
{
lean_object* x_1875; lean_object* x_1876; lean_object* x_1877; lean_object* x_1878; lean_object* x_1879; lean_object* x_1880; lean_object* x_1881; lean_object* x_1882; lean_object* x_1883; lean_object* x_1884; 
x_1875 = lp_mathlib_CancelDenoms_mkProdPrf___closed__91;
lean_inc_ref(x_7);
x_1876 = l_Lean_MessageData_ofExpr(x_7);
x_1877 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_1877, 0, x_1875);
lean_ctor_set(x_1877, 1, x_1876);
x_1878 = lp_mathlib_CancelDenoms_mkProdPrf___closed__93;
x_1879 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_1879, 0, x_1877);
lean_ctor_set(x_1879, 1, x_1878);
lean_inc(x_4);
x_1880 = l_Nat_reprFast(x_4);
x_1881 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_1881, 0, x_1880);
x_1882 = l_Lean_MessageData_ofFormat(x_1881);
x_1883 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_1883, 0, x_1879);
lean_ctor_set(x_1883, 1, x_1882);
x_1884 = lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3(x_13, x_1883, x_8, x_9, x_10, x_11);
if (lean_obj_tag(x_1884) == 0)
{
lean_dec_ref(x_1884);
x_1660 = x_8;
x_1661 = x_9;
x_1662 = x_10;
x_1663 = x_11;
x_1664 = lean_box(0);
goto block_1873;
}
else
{
uint8_t x_1885; 
lean_dec_ref(x_47);
lean_dec_ref(x_45);
lean_dec_ref(x_44);
lean_dec_ref(x_43);
lean_dec_ref(x_25);
lean_dec_ref(x_24);
lean_dec_ref(x_19);
lean_dec(x_16);
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_1885 = !lean_is_exclusive(x_1884);
if (x_1885 == 0)
{
return x_1884;
}
else
{
lean_object* x_1886; lean_object* x_1887; 
x_1886 = lean_ctor_get(x_1884, 0);
lean_inc(x_1886);
lean_dec(x_1884);
x_1887 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_1887, 0, x_1886);
return x_1887;
}
}
}
block_243:
{
lean_object* x_59; uint8_t x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; uint8_t x_64; lean_object* x_65; 
lean_inc_ref(x_2);
x_59 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_59, 0, x_2);
x_60 = 0;
x_61 = lean_box(0);
x_62 = lean_box(x_60);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_19);
x_63 = lean_alloc_closure((void*)(lp_mathlib_CancelDenoms_mkProdPrf___lam__0___boxed), 13, 8);
lean_closure_set(x_63, 0, x_59);
lean_closure_set(x_63, 1, x_62);
lean_closure_set(x_63, 2, x_61);
lean_closure_set(x_63, 3, x_19);
lean_closure_set(x_63, 4, x_2);
lean_closure_set(x_63, 5, x_39);
lean_closure_set(x_63, 6, x_3);
lean_closure_set(x_63, 7, x_7);
x_64 = 0;
lean_inc(x_57);
lean_inc_ref(x_56);
lean_inc(x_55);
lean_inc_ref(x_54);
x_65 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00CancelDenoms_mkProdPrf_spec__2___redArg(x_63, x_64, x_54, x_55, x_56, x_57);
if (lean_obj_tag(x_65) == 0)
{
uint8_t x_66; 
x_66 = !lean_is_exclusive(x_65);
if (x_66 == 0)
{
lean_object* x_67; lean_object* x_68; uint8_t x_69; 
x_67 = lean_ctor_get(x_65, 0);
x_68 = lean_ctor_get(x_67, 1);
x_69 = lean_unbox(x_68);
if (x_69 == 0)
{
lean_dec(x_67);
lean_dec(x_57);
lean_dec_ref(x_56);
lean_dec(x_55);
lean_dec_ref(x_54);
lean_dec(x_53);
lean_dec_ref(x_52);
lean_dec_ref(x_51);
lean_dec_ref(x_50);
lean_dec_ref(x_48);
lean_dec_ref(x_47);
lean_dec_ref(x_24);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
lean_ctor_set(x_65, 0, x_49);
return x_65;
}
else
{
lean_object* x_70; lean_object* x_71; lean_object* x_72; 
lean_free_object(x_65);
lean_dec_ref(x_49);
x_70 = lean_ctor_get(x_67, 0);
lean_inc(x_70);
lean_dec(x_67);
lean_inc(x_53);
x_71 = l_Lean_mkRawNatLit(x_53);
lean_inc_ref(x_47);
lean_inc_ref(x_2);
lean_inc(x_1);
x_72 = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(x_1, x_2, x_47, x_71, x_54, x_55, x_56, x_57);
if (lean_obj_tag(x_72) == 0)
{
lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; 
x_73 = lean_ctor_get(x_72, 0);
lean_inc(x_73);
lean_dec_ref(x_72);
x_74 = lean_nat_div(x_4, x_53);
lean_dec(x_53);
lean_dec(x_4);
x_75 = l_Lean_mkRawNatLit(x_74);
lean_inc_ref(x_2);
x_76 = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(x_1, x_2, x_47, x_75, x_54, x_55, x_56, x_57);
if (lean_obj_tag(x_76) == 0)
{
lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; 
x_77 = lean_ctor_get(x_76, 0);
lean_inc(x_77);
lean_dec_ref(x_76);
x_78 = lean_ctor_get(x_73, 0);
lean_inc(x_78);
lean_dec(x_73);
x_79 = lp_mathlib_CancelDenoms_mkProdPrf___closed__17;
x_80 = l_Lean_Expr_const___override(x_79, x_24);
lean_inc_ref(x_2);
x_81 = l_Lean_Expr_app___override(x_80, x_2);
lean_inc(x_78);
x_82 = l_Lean_Expr_app___override(x_81, x_78);
x_83 = lp_mathlib_CancelDenoms_mkProdPrf___closed__20;
lean_inc_ref(x_19);
x_84 = l_Lean_Expr_const___override(x_83, x_19);
lean_inc_ref(x_2);
x_85 = l_Lean_Expr_app___override(x_84, x_2);
x_86 = lp_mathlib_CancelDenoms_mkProdPrf___closed__22;
x_87 = l_Lean_Expr_app___override(x_85, x_86);
x_88 = lp_mathlib_CancelDenoms_mkProdPrf___closed__25;
lean_inc_ref(x_19);
x_89 = l_Lean_Expr_const___override(x_88, x_19);
lean_inc_ref(x_2);
x_90 = l_Lean_Expr_app___override(x_89, x_2);
x_91 = lp_mathlib_CancelDenoms_mkProdPrf___closed__28;
lean_inc_ref(x_19);
x_92 = l_Lean_Expr_const___override(x_91, x_19);
lean_inc_ref(x_2);
x_93 = l_Lean_Expr_app___override(x_92, x_2);
x_94 = lp_mathlib_CancelDenoms_mkProdPrf___closed__29;
x_95 = l_Lean_Name_mkStr2(x_52, x_94);
lean_inc_ref(x_19);
x_96 = l_Lean_Expr_const___override(x_95, x_19);
lean_inc_ref(x_2);
x_97 = l_Lean_Expr_app___override(x_96, x_2);
x_98 = l_Lean_Expr_app___override(x_97, x_51);
x_99 = l_Lean_Expr_app___override(x_93, x_98);
x_100 = l_Lean_Expr_app___override(x_90, x_99);
x_101 = l_Lean_Expr_app___override(x_87, x_100);
x_102 = l_Lean_Expr_app___override(x_82, x_101);
lean_inc(x_57);
lean_inc_ref(x_56);
lean_inc(x_55);
lean_inc_ref(x_54);
x_103 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum(x_102, x_54, x_55, x_56, x_57);
if (lean_obj_tag(x_103) == 0)
{
lean_object* x_104; uint8_t x_105; 
x_104 = lean_ctor_get(x_103, 0);
lean_inc(x_104);
lean_dec_ref(x_103);
x_105 = !lean_is_exclusive(x_77);
if (x_105 == 0)
{
lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; lean_object* x_112; 
x_106 = lean_ctor_get(x_77, 0);
x_107 = lean_ctor_get(x_77, 1);
lean_dec(x_107);
lean_inc(x_106);
x_108 = l_Lean_Expr_app___override(x_50, x_106);
x_109 = l_Lean_Expr_app___override(x_108, x_78);
x_110 = l_Lean_Expr_app___override(x_48, x_109);
lean_inc_ref(x_5);
x_111 = l_Lean_Expr_app___override(x_110, x_5);
x_112 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum(x_111, x_54, x_55, x_56, x_57);
if (lean_obj_tag(x_112) == 0)
{
uint8_t x_113; 
x_113 = !lean_is_exclusive(x_112);
if (x_113 == 0)
{
lean_object* x_114; lean_object* x_115; lean_object* x_116; lean_object* x_117; lean_object* x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; 
x_114 = lean_ctor_get(x_112, 0);
x_115 = lp_mathlib_CancelDenoms_mkProdPrf___closed__31;
x_116 = l_Lean_Expr_const___override(x_115, x_19);
x_117 = l_Lean_Expr_app___override(x_116, x_2);
x_118 = l_Lean_Expr_app___override(x_117, x_3);
lean_inc(x_106);
x_119 = l_Lean_Expr_app___override(x_118, x_106);
x_120 = l_Lean_Expr_app___override(x_119, x_5);
x_121 = l_Lean_Expr_app___override(x_120, x_70);
x_122 = l_Lean_Expr_app___override(x_121, x_104);
x_123 = l_Lean_Expr_app___override(x_122, x_114);
lean_ctor_set(x_77, 1, x_123);
lean_ctor_set(x_112, 0, x_77);
return x_112;
}
else
{
lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_127; lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; lean_object* x_132; lean_object* x_133; lean_object* x_134; 
x_124 = lean_ctor_get(x_112, 0);
lean_inc(x_124);
lean_dec(x_112);
x_125 = lp_mathlib_CancelDenoms_mkProdPrf___closed__31;
x_126 = l_Lean_Expr_const___override(x_125, x_19);
x_127 = l_Lean_Expr_app___override(x_126, x_2);
x_128 = l_Lean_Expr_app___override(x_127, x_3);
lean_inc(x_106);
x_129 = l_Lean_Expr_app___override(x_128, x_106);
x_130 = l_Lean_Expr_app___override(x_129, x_5);
x_131 = l_Lean_Expr_app___override(x_130, x_70);
x_132 = l_Lean_Expr_app___override(x_131, x_104);
x_133 = l_Lean_Expr_app___override(x_132, x_124);
lean_ctor_set(x_77, 1, x_133);
x_134 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_134, 0, x_77);
return x_134;
}
}
else
{
uint8_t x_135; 
lean_free_object(x_77);
lean_dec(x_106);
lean_dec(x_104);
lean_dec(x_70);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
x_135 = !lean_is_exclusive(x_112);
if (x_135 == 0)
{
return x_112;
}
else
{
lean_object* x_136; lean_object* x_137; 
x_136 = lean_ctor_get(x_112, 0);
lean_inc(x_136);
lean_dec(x_112);
x_137 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_137, 0, x_136);
return x_137;
}
}
}
else
{
lean_object* x_138; lean_object* x_139; lean_object* x_140; lean_object* x_141; lean_object* x_142; lean_object* x_143; 
x_138 = lean_ctor_get(x_77, 0);
lean_inc(x_138);
lean_dec(x_77);
lean_inc(x_138);
x_139 = l_Lean_Expr_app___override(x_50, x_138);
x_140 = l_Lean_Expr_app___override(x_139, x_78);
x_141 = l_Lean_Expr_app___override(x_48, x_140);
lean_inc_ref(x_5);
x_142 = l_Lean_Expr_app___override(x_141, x_5);
x_143 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum(x_142, x_54, x_55, x_56, x_57);
if (lean_obj_tag(x_143) == 0)
{
lean_object* x_144; lean_object* x_145; lean_object* x_146; lean_object* x_147; lean_object* x_148; lean_object* x_149; lean_object* x_150; lean_object* x_151; lean_object* x_152; lean_object* x_153; lean_object* x_154; lean_object* x_155; lean_object* x_156; 
x_144 = lean_ctor_get(x_143, 0);
lean_inc(x_144);
if (lean_is_exclusive(x_143)) {
 lean_ctor_release(x_143, 0);
 x_145 = x_143;
} else {
 lean_dec_ref(x_143);
 x_145 = lean_box(0);
}
x_146 = lp_mathlib_CancelDenoms_mkProdPrf___closed__31;
x_147 = l_Lean_Expr_const___override(x_146, x_19);
x_148 = l_Lean_Expr_app___override(x_147, x_2);
x_149 = l_Lean_Expr_app___override(x_148, x_3);
lean_inc(x_138);
x_150 = l_Lean_Expr_app___override(x_149, x_138);
x_151 = l_Lean_Expr_app___override(x_150, x_5);
x_152 = l_Lean_Expr_app___override(x_151, x_70);
x_153 = l_Lean_Expr_app___override(x_152, x_104);
x_154 = l_Lean_Expr_app___override(x_153, x_144);
x_155 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_155, 0, x_138);
lean_ctor_set(x_155, 1, x_154);
if (lean_is_scalar(x_145)) {
 x_156 = lean_alloc_ctor(0, 1, 0);
} else {
 x_156 = x_145;
}
lean_ctor_set(x_156, 0, x_155);
return x_156;
}
else
{
lean_object* x_157; lean_object* x_158; lean_object* x_159; 
lean_dec(x_138);
lean_dec(x_104);
lean_dec(x_70);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
x_157 = lean_ctor_get(x_143, 0);
lean_inc(x_157);
if (lean_is_exclusive(x_143)) {
 lean_ctor_release(x_143, 0);
 x_158 = x_143;
} else {
 lean_dec_ref(x_143);
 x_158 = lean_box(0);
}
if (lean_is_scalar(x_158)) {
 x_159 = lean_alloc_ctor(1, 1, 0);
} else {
 x_159 = x_158;
}
lean_ctor_set(x_159, 0, x_157);
return x_159;
}
}
}
else
{
uint8_t x_160; 
lean_dec(x_78);
lean_dec(x_77);
lean_dec(x_70);
lean_dec(x_57);
lean_dec_ref(x_56);
lean_dec(x_55);
lean_dec_ref(x_54);
lean_dec_ref(x_50);
lean_dec_ref(x_48);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
x_160 = !lean_is_exclusive(x_103);
if (x_160 == 0)
{
return x_103;
}
else
{
lean_object* x_161; lean_object* x_162; 
x_161 = lean_ctor_get(x_103, 0);
lean_inc(x_161);
lean_dec(x_103);
x_162 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_162, 0, x_161);
return x_162;
}
}
}
else
{
uint8_t x_163; 
lean_dec(x_73);
lean_dec(x_70);
lean_dec(x_57);
lean_dec_ref(x_56);
lean_dec(x_55);
lean_dec_ref(x_54);
lean_dec_ref(x_52);
lean_dec_ref(x_51);
lean_dec_ref(x_50);
lean_dec_ref(x_48);
lean_dec_ref(x_24);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
x_163 = !lean_is_exclusive(x_76);
if (x_163 == 0)
{
return x_76;
}
else
{
lean_object* x_164; lean_object* x_165; 
x_164 = lean_ctor_get(x_76, 0);
lean_inc(x_164);
lean_dec(x_76);
x_165 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_165, 0, x_164);
return x_165;
}
}
}
else
{
uint8_t x_166; 
lean_dec(x_70);
lean_dec(x_57);
lean_dec_ref(x_56);
lean_dec(x_55);
lean_dec_ref(x_54);
lean_dec(x_53);
lean_dec_ref(x_52);
lean_dec_ref(x_51);
lean_dec_ref(x_50);
lean_dec_ref(x_48);
lean_dec_ref(x_47);
lean_dec_ref(x_24);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_166 = !lean_is_exclusive(x_72);
if (x_166 == 0)
{
return x_72;
}
else
{
lean_object* x_167; lean_object* x_168; 
x_167 = lean_ctor_get(x_72, 0);
lean_inc(x_167);
lean_dec(x_72);
x_168 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_168, 0, x_167);
return x_168;
}
}
}
}
else
{
lean_object* x_169; lean_object* x_170; uint8_t x_171; 
x_169 = lean_ctor_get(x_65, 0);
lean_inc(x_169);
lean_dec(x_65);
x_170 = lean_ctor_get(x_169, 1);
x_171 = lean_unbox(x_170);
if (x_171 == 0)
{
lean_object* x_172; 
lean_dec(x_169);
lean_dec(x_57);
lean_dec_ref(x_56);
lean_dec(x_55);
lean_dec_ref(x_54);
lean_dec(x_53);
lean_dec_ref(x_52);
lean_dec_ref(x_51);
lean_dec_ref(x_50);
lean_dec_ref(x_48);
lean_dec_ref(x_47);
lean_dec_ref(x_24);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_172 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_172, 0, x_49);
return x_172;
}
else
{
lean_object* x_173; lean_object* x_174; lean_object* x_175; 
lean_dec_ref(x_49);
x_173 = lean_ctor_get(x_169, 0);
lean_inc(x_173);
lean_dec(x_169);
lean_inc(x_53);
x_174 = l_Lean_mkRawNatLit(x_53);
lean_inc_ref(x_47);
lean_inc_ref(x_2);
lean_inc(x_1);
x_175 = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(x_1, x_2, x_47, x_174, x_54, x_55, x_56, x_57);
if (lean_obj_tag(x_175) == 0)
{
lean_object* x_176; lean_object* x_177; lean_object* x_178; lean_object* x_179; 
x_176 = lean_ctor_get(x_175, 0);
lean_inc(x_176);
lean_dec_ref(x_175);
x_177 = lean_nat_div(x_4, x_53);
lean_dec(x_53);
lean_dec(x_4);
x_178 = l_Lean_mkRawNatLit(x_177);
lean_inc_ref(x_2);
x_179 = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(x_1, x_2, x_47, x_178, x_54, x_55, x_56, x_57);
if (lean_obj_tag(x_179) == 0)
{
lean_object* x_180; lean_object* x_181; lean_object* x_182; lean_object* x_183; lean_object* x_184; lean_object* x_185; lean_object* x_186; lean_object* x_187; lean_object* x_188; lean_object* x_189; lean_object* x_190; lean_object* x_191; lean_object* x_192; lean_object* x_193; lean_object* x_194; lean_object* x_195; lean_object* x_196; lean_object* x_197; lean_object* x_198; lean_object* x_199; lean_object* x_200; lean_object* x_201; lean_object* x_202; lean_object* x_203; lean_object* x_204; lean_object* x_205; lean_object* x_206; 
x_180 = lean_ctor_get(x_179, 0);
lean_inc(x_180);
lean_dec_ref(x_179);
x_181 = lean_ctor_get(x_176, 0);
lean_inc(x_181);
lean_dec(x_176);
x_182 = lp_mathlib_CancelDenoms_mkProdPrf___closed__17;
x_183 = l_Lean_Expr_const___override(x_182, x_24);
lean_inc_ref(x_2);
x_184 = l_Lean_Expr_app___override(x_183, x_2);
lean_inc(x_181);
x_185 = l_Lean_Expr_app___override(x_184, x_181);
x_186 = lp_mathlib_CancelDenoms_mkProdPrf___closed__20;
lean_inc_ref(x_19);
x_187 = l_Lean_Expr_const___override(x_186, x_19);
lean_inc_ref(x_2);
x_188 = l_Lean_Expr_app___override(x_187, x_2);
x_189 = lp_mathlib_CancelDenoms_mkProdPrf___closed__22;
x_190 = l_Lean_Expr_app___override(x_188, x_189);
x_191 = lp_mathlib_CancelDenoms_mkProdPrf___closed__25;
lean_inc_ref(x_19);
x_192 = l_Lean_Expr_const___override(x_191, x_19);
lean_inc_ref(x_2);
x_193 = l_Lean_Expr_app___override(x_192, x_2);
x_194 = lp_mathlib_CancelDenoms_mkProdPrf___closed__28;
lean_inc_ref(x_19);
x_195 = l_Lean_Expr_const___override(x_194, x_19);
lean_inc_ref(x_2);
x_196 = l_Lean_Expr_app___override(x_195, x_2);
x_197 = lp_mathlib_CancelDenoms_mkProdPrf___closed__29;
x_198 = l_Lean_Name_mkStr2(x_52, x_197);
lean_inc_ref(x_19);
x_199 = l_Lean_Expr_const___override(x_198, x_19);
lean_inc_ref(x_2);
x_200 = l_Lean_Expr_app___override(x_199, x_2);
x_201 = l_Lean_Expr_app___override(x_200, x_51);
x_202 = l_Lean_Expr_app___override(x_196, x_201);
x_203 = l_Lean_Expr_app___override(x_193, x_202);
x_204 = l_Lean_Expr_app___override(x_190, x_203);
x_205 = l_Lean_Expr_app___override(x_185, x_204);
lean_inc(x_57);
lean_inc_ref(x_56);
lean_inc(x_55);
lean_inc_ref(x_54);
x_206 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum(x_205, x_54, x_55, x_56, x_57);
if (lean_obj_tag(x_206) == 0)
{
lean_object* x_207; lean_object* x_208; lean_object* x_209; lean_object* x_210; lean_object* x_211; lean_object* x_212; lean_object* x_213; lean_object* x_214; 
x_207 = lean_ctor_get(x_206, 0);
lean_inc(x_207);
lean_dec_ref(x_206);
x_208 = lean_ctor_get(x_180, 0);
lean_inc(x_208);
if (lean_is_exclusive(x_180)) {
 lean_ctor_release(x_180, 0);
 lean_ctor_release(x_180, 1);
 x_209 = x_180;
} else {
 lean_dec_ref(x_180);
 x_209 = lean_box(0);
}
lean_inc(x_208);
x_210 = l_Lean_Expr_app___override(x_50, x_208);
x_211 = l_Lean_Expr_app___override(x_210, x_181);
x_212 = l_Lean_Expr_app___override(x_48, x_211);
lean_inc_ref(x_5);
x_213 = l_Lean_Expr_app___override(x_212, x_5);
x_214 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum(x_213, x_54, x_55, x_56, x_57);
if (lean_obj_tag(x_214) == 0)
{
lean_object* x_215; lean_object* x_216; lean_object* x_217; lean_object* x_218; lean_object* x_219; lean_object* x_220; lean_object* x_221; lean_object* x_222; lean_object* x_223; lean_object* x_224; lean_object* x_225; lean_object* x_226; lean_object* x_227; 
x_215 = lean_ctor_get(x_214, 0);
lean_inc(x_215);
if (lean_is_exclusive(x_214)) {
 lean_ctor_release(x_214, 0);
 x_216 = x_214;
} else {
 lean_dec_ref(x_214);
 x_216 = lean_box(0);
}
x_217 = lp_mathlib_CancelDenoms_mkProdPrf___closed__31;
x_218 = l_Lean_Expr_const___override(x_217, x_19);
x_219 = l_Lean_Expr_app___override(x_218, x_2);
x_220 = l_Lean_Expr_app___override(x_219, x_3);
lean_inc(x_208);
x_221 = l_Lean_Expr_app___override(x_220, x_208);
x_222 = l_Lean_Expr_app___override(x_221, x_5);
x_223 = l_Lean_Expr_app___override(x_222, x_173);
x_224 = l_Lean_Expr_app___override(x_223, x_207);
x_225 = l_Lean_Expr_app___override(x_224, x_215);
if (lean_is_scalar(x_209)) {
 x_226 = lean_alloc_ctor(0, 2, 0);
} else {
 x_226 = x_209;
}
lean_ctor_set(x_226, 0, x_208);
lean_ctor_set(x_226, 1, x_225);
if (lean_is_scalar(x_216)) {
 x_227 = lean_alloc_ctor(0, 1, 0);
} else {
 x_227 = x_216;
}
lean_ctor_set(x_227, 0, x_226);
return x_227;
}
else
{
lean_object* x_228; lean_object* x_229; lean_object* x_230; 
lean_dec(x_209);
lean_dec(x_208);
lean_dec(x_207);
lean_dec(x_173);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
x_228 = lean_ctor_get(x_214, 0);
lean_inc(x_228);
if (lean_is_exclusive(x_214)) {
 lean_ctor_release(x_214, 0);
 x_229 = x_214;
} else {
 lean_dec_ref(x_214);
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
else
{
lean_object* x_231; lean_object* x_232; lean_object* x_233; 
lean_dec(x_181);
lean_dec(x_180);
lean_dec(x_173);
lean_dec(x_57);
lean_dec_ref(x_56);
lean_dec(x_55);
lean_dec_ref(x_54);
lean_dec_ref(x_50);
lean_dec_ref(x_48);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
x_231 = lean_ctor_get(x_206, 0);
lean_inc(x_231);
if (lean_is_exclusive(x_206)) {
 lean_ctor_release(x_206, 0);
 x_232 = x_206;
} else {
 lean_dec_ref(x_206);
 x_232 = lean_box(0);
}
if (lean_is_scalar(x_232)) {
 x_233 = lean_alloc_ctor(1, 1, 0);
} else {
 x_233 = x_232;
}
lean_ctor_set(x_233, 0, x_231);
return x_233;
}
}
else
{
lean_object* x_234; lean_object* x_235; lean_object* x_236; 
lean_dec(x_176);
lean_dec(x_173);
lean_dec(x_57);
lean_dec_ref(x_56);
lean_dec(x_55);
lean_dec_ref(x_54);
lean_dec_ref(x_52);
lean_dec_ref(x_51);
lean_dec_ref(x_50);
lean_dec_ref(x_48);
lean_dec_ref(x_24);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
x_234 = lean_ctor_get(x_179, 0);
lean_inc(x_234);
if (lean_is_exclusive(x_179)) {
 lean_ctor_release(x_179, 0);
 x_235 = x_179;
} else {
 lean_dec_ref(x_179);
 x_235 = lean_box(0);
}
if (lean_is_scalar(x_235)) {
 x_236 = lean_alloc_ctor(1, 1, 0);
} else {
 x_236 = x_235;
}
lean_ctor_set(x_236, 0, x_234);
return x_236;
}
}
else
{
lean_object* x_237; lean_object* x_238; lean_object* x_239; 
lean_dec(x_173);
lean_dec(x_57);
lean_dec_ref(x_56);
lean_dec(x_55);
lean_dec_ref(x_54);
lean_dec(x_53);
lean_dec_ref(x_52);
lean_dec_ref(x_51);
lean_dec_ref(x_50);
lean_dec_ref(x_48);
lean_dec_ref(x_47);
lean_dec_ref(x_24);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_237 = lean_ctor_get(x_175, 0);
lean_inc(x_237);
if (lean_is_exclusive(x_175)) {
 lean_ctor_release(x_175, 0);
 x_238 = x_175;
} else {
 lean_dec_ref(x_175);
 x_238 = lean_box(0);
}
if (lean_is_scalar(x_238)) {
 x_239 = lean_alloc_ctor(1, 1, 0);
} else {
 x_239 = x_238;
}
lean_ctor_set(x_239, 0, x_237);
return x_239;
}
}
}
}
else
{
uint8_t x_240; 
lean_dec(x_57);
lean_dec_ref(x_56);
lean_dec(x_55);
lean_dec_ref(x_54);
lean_dec(x_53);
lean_dec_ref(x_52);
lean_dec_ref(x_51);
lean_dec_ref(x_50);
lean_dec_ref(x_49);
lean_dec_ref(x_48);
lean_dec_ref(x_47);
lean_dec_ref(x_24);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_240 = !lean_is_exclusive(x_65);
if (x_240 == 0)
{
return x_65;
}
else
{
lean_object* x_241; lean_object* x_242; 
x_241 = lean_ctor_get(x_65, 0);
lean_inc(x_241);
lean_dec(x_65);
x_242 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_242, 0, x_241);
return x_242;
}
}
}
block_820:
{
lean_object* x_258; uint8_t x_259; lean_object* x_260; lean_object* x_261; lean_object* x_262; uint8_t x_263; lean_object* x_264; 
lean_inc_ref(x_2);
x_258 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_258, 0, x_2);
x_259 = 0;
x_260 = lean_box(0);
x_261 = lean_box(x_259);
lean_inc_ref(x_7);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc(x_1);
lean_inc_ref(x_19);
x_262 = lean_alloc_closure((void*)(lp_mathlib_CancelDenoms_mkProdPrf___lam__1___boxed), 15, 10);
lean_closure_set(x_262, 0, x_258);
lean_closure_set(x_262, 1, x_261);
lean_closure_set(x_262, 2, x_260);
lean_closure_set(x_262, 3, x_18);
lean_closure_set(x_262, 4, x_19);
lean_closure_set(x_262, 5, x_1);
lean_closure_set(x_262, 6, x_2);
lean_closure_set(x_262, 7, x_39);
lean_closure_set(x_262, 8, x_3);
lean_closure_set(x_262, 9, x_7);
x_263 = 0;
lean_inc(x_256);
lean_inc_ref(x_255);
lean_inc(x_254);
lean_inc_ref(x_253);
x_264 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00CancelDenoms_mkProdPrf_spec__2___redArg(x_262, x_263, x_253, x_254, x_255, x_256);
if (lean_obj_tag(x_264) == 0)
{
uint8_t x_265; 
x_265 = !lean_is_exclusive(x_264);
if (x_265 == 0)
{
lean_object* x_266; lean_object* x_267; lean_object* x_268; uint8_t x_269; 
x_266 = lean_ctor_get(x_264, 0);
x_267 = lean_ctor_get(x_266, 1);
lean_inc(x_267);
x_268 = lean_ctor_get(x_267, 1);
x_269 = lean_unbox(x_268);
if (x_269 == 0)
{
lean_dec(x_267);
lean_dec(x_266);
lean_dec(x_252);
lean_dec(x_251);
lean_dec(x_250);
lean_dec_ref(x_246);
if (lean_obj_tag(x_6) == 1)
{
lean_object* x_270; 
x_270 = lean_ctor_get(x_6, 1);
if (lean_obj_tag(x_270) == 0)
{
lean_object* x_271; 
x_271 = lean_ctor_get(x_6, 2);
lean_inc(x_271);
lean_dec_ref(x_6);
if (lean_obj_tag(x_271) == 1)
{
lean_object* x_272; 
lean_free_object(x_264);
x_272 = lean_ctor_get(x_271, 0);
lean_inc(x_272);
lean_dec_ref(x_271);
x_48 = x_244;
x_49 = x_245;
x_50 = x_249;
x_51 = x_248;
x_52 = x_247;
x_53 = x_272;
x_54 = x_253;
x_55 = x_254;
x_56 = x_255;
x_57 = x_256;
x_58 = lean_box(0);
goto block_243;
}
else
{
lean_dec(x_271);
lean_dec(x_256);
lean_dec_ref(x_255);
lean_dec(x_254);
lean_dec_ref(x_253);
lean_dec_ref(x_249);
lean_dec_ref(x_248);
lean_dec_ref(x_247);
lean_dec_ref(x_244);
lean_dec_ref(x_47);
lean_dec_ref(x_24);
lean_dec_ref(x_19);
lean_dec_ref(x_7);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
lean_ctor_set(x_264, 0, x_245);
return x_264;
}
}
else
{
lean_dec(x_256);
lean_dec_ref(x_255);
lean_dec(x_254);
lean_dec_ref(x_253);
lean_dec_ref(x_249);
lean_dec_ref(x_248);
lean_dec_ref(x_247);
lean_dec_ref(x_244);
lean_dec_ref(x_47);
lean_dec_ref(x_24);
lean_dec_ref(x_19);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
lean_ctor_set(x_264, 0, x_245);
return x_264;
}
}
else
{
lean_dec(x_256);
lean_dec_ref(x_255);
lean_dec(x_254);
lean_dec_ref(x_253);
lean_dec_ref(x_249);
lean_dec_ref(x_248);
lean_dec_ref(x_247);
lean_dec_ref(x_244);
lean_dec_ref(x_47);
lean_dec_ref(x_24);
lean_dec_ref(x_19);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
lean_ctor_set(x_264, 0, x_245);
return x_264;
}
}
else
{
lean_object* x_273; uint8_t x_274; 
lean_free_object(x_264);
lean_dec_ref(x_248);
lean_dec_ref(x_247);
lean_dec_ref(x_245);
lean_dec_ref(x_24);
lean_dec_ref(x_7);
lean_dec(x_6);
x_273 = lean_ctor_get(x_266, 0);
lean_inc(x_273);
lean_dec(x_266);
x_274 = !lean_is_exclusive(x_267);
if (x_274 == 0)
{
lean_object* x_275; lean_object* x_276; lean_object* x_277; lean_object* x_278; 
x_275 = lean_ctor_get(x_267, 0);
x_276 = lean_ctor_get(x_267, 1);
lean_dec(x_276);
lean_inc(x_251);
x_277 = l_Lean_mkRawNatLit(x_251);
lean_inc_ref(x_47);
lean_inc_ref(x_2);
lean_inc(x_1);
x_278 = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(x_1, x_2, x_47, x_277, x_253, x_254, x_255, x_256);
if (lean_obj_tag(x_278) == 0)
{
lean_object* x_279; uint8_t x_280; 
x_279 = lean_ctor_get(x_278, 0);
lean_inc(x_279);
lean_dec_ref(x_278);
x_280 = !lean_is_exclusive(x_279);
if (x_280 == 0)
{
lean_object* x_281; lean_object* x_282; lean_object* x_283; 
x_281 = lean_ctor_get(x_279, 0);
x_282 = lean_ctor_get(x_279, 1);
lean_dec(x_282);
lean_inc(x_256);
lean_inc_ref(x_255);
lean_inc(x_254);
lean_inc_ref(x_253);
lean_inc(x_273);
lean_inc(x_281);
lean_inc(x_251);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc(x_1);
x_283 = lp_mathlib_CancelDenoms_mkProdPrf(x_1, x_2, x_3, x_251, x_281, x_250, x_273, x_253, x_254, x_255, x_256);
if (lean_obj_tag(x_283) == 0)
{
lean_object* x_284; uint8_t x_285; 
x_284 = lean_ctor_get(x_283, 0);
lean_inc(x_284);
lean_dec_ref(x_283);
x_285 = !lean_is_exclusive(x_284);
if (x_285 == 0)
{
lean_object* x_286; lean_object* x_287; lean_object* x_288; lean_object* x_289; lean_object* x_290; lean_object* x_291; 
x_286 = lean_ctor_get(x_284, 0);
x_287 = lean_ctor_get(x_284, 1);
x_288 = lean_nat_pow(x_251, x_252);
lean_dec(x_252);
lean_dec(x_251);
x_289 = lean_nat_div(x_4, x_288);
lean_dec(x_288);
lean_dec(x_4);
x_290 = l_Lean_mkRawNatLit(x_289);
lean_inc_ref(x_2);
lean_inc(x_1);
x_291 = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(x_1, x_2, x_47, x_290, x_253, x_254, x_255, x_256);
if (lean_obj_tag(x_291) == 0)
{
lean_object* x_292; uint8_t x_293; 
x_292 = lean_ctor_get(x_291, 0);
lean_inc(x_292);
lean_dec_ref(x_291);
x_293 = !lean_is_exclusive(x_292);
if (x_293 == 0)
{
lean_object* x_294; lean_object* x_295; lean_object* x_296; lean_object* x_297; lean_object* x_298; lean_object* x_299; lean_object* x_300; lean_object* x_301; lean_object* x_302; lean_object* x_303; lean_object* x_304; lean_object* x_305; lean_object* x_306; lean_object* x_307; lean_object* x_308; lean_object* x_309; lean_object* x_310; lean_object* x_311; lean_object* x_312; lean_object* x_313; lean_object* x_314; lean_object* x_315; lean_object* x_316; lean_object* x_317; lean_object* x_318; lean_object* x_319; lean_object* x_320; lean_object* x_321; lean_object* x_322; lean_object* x_323; lean_object* x_324; lean_object* x_325; lean_object* x_326; lean_object* x_327; lean_object* x_328; lean_object* x_329; lean_object* x_330; lean_object* x_331; lean_object* x_332; lean_object* x_333; lean_object* x_334; lean_object* x_335; lean_object* x_336; lean_object* x_337; lean_object* x_338; lean_object* x_339; lean_object* x_340; 
x_294 = lean_ctor_get(x_292, 0);
x_295 = lean_ctor_get(x_292, 1);
lean_dec(x_295);
x_296 = lean_box(0);
lean_inc(x_294);
x_297 = l_Lean_Expr_app___override(x_249, x_294);
x_298 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__2;
lean_inc_ref(x_19);
lean_ctor_set_tag(x_292, 1);
lean_ctor_set(x_292, 1, x_19);
lean_ctor_set(x_292, 0, x_296);
lean_inc(x_1);
lean_ctor_set_tag(x_279, 1);
lean_ctor_set(x_279, 1, x_292);
lean_ctor_set(x_279, 0, x_1);
x_299 = l_Lean_Expr_const___override(x_298, x_279);
lean_inc_ref(x_2);
x_300 = l_Lean_Expr_app___override(x_299, x_2);
x_301 = lp_mathlib_CancelDenoms_mkProdPrf___closed__32;
x_302 = l_Lean_Expr_app___override(x_300, x_301);
lean_inc_ref(x_2);
x_303 = l_Lean_Expr_app___override(x_302, x_2);
x_304 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__4;
x_305 = lp_mathlib_CancelDenoms_mkProdPrf___closed__33;
lean_ctor_set_tag(x_267, 1);
lean_ctor_set(x_267, 1, x_305);
lean_ctor_set(x_267, 0, x_1);
x_306 = l_Lean_Expr_const___override(x_304, x_267);
lean_inc_ref(x_2);
x_307 = l_Lean_Expr_app___override(x_306, x_2);
x_308 = l_Lean_Expr_app___override(x_307, x_301);
x_309 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__7;
lean_inc_ref(x_19);
x_310 = l_Lean_Expr_const___override(x_309, x_19);
lean_inc_ref(x_2);
x_311 = l_Lean_Expr_app___override(x_310, x_2);
x_312 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__10;
lean_inc_ref(x_19);
x_313 = l_Lean_Expr_const___override(x_312, x_19);
lean_inc_ref(x_2);
x_314 = l_Lean_Expr_app___override(x_313, x_2);
x_315 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__13;
lean_inc_ref(x_19);
x_316 = l_Lean_Expr_const___override(x_315, x_19);
lean_inc_ref(x_2);
x_317 = l_Lean_Expr_app___override(x_316, x_2);
x_318 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__16;
lean_inc_ref(x_19);
x_319 = l_Lean_Expr_const___override(x_318, x_19);
lean_inc_ref(x_2);
x_320 = l_Lean_Expr_app___override(x_319, x_2);
x_321 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__18;
lean_inc_ref(x_19);
x_322 = l_Lean_Expr_const___override(x_321, x_19);
lean_inc_ref(x_2);
x_323 = l_Lean_Expr_app___override(x_322, x_2);
x_324 = lp_mathlib_CancelDenoms_mkProdPrf___closed__34;
lean_inc_ref(x_19);
x_325 = l_Lean_Expr_const___override(x_324, x_19);
lean_inc_ref(x_2);
x_326 = l_Lean_Expr_app___override(x_325, x_2);
x_327 = l_Lean_Expr_app___override(x_326, x_3);
x_328 = l_Lean_Expr_app___override(x_323, x_327);
x_329 = l_Lean_Expr_app___override(x_320, x_328);
x_330 = l_Lean_Expr_app___override(x_317, x_329);
x_331 = l_Lean_Expr_app___override(x_314, x_330);
x_332 = l_Lean_Expr_app___override(x_311, x_331);
x_333 = l_Lean_Expr_app___override(x_308, x_332);
x_334 = l_Lean_Expr_app___override(x_303, x_333);
lean_inc(x_281);
lean_inc_ref(x_334);
x_335 = l_Lean_Expr_app___override(x_334, x_281);
lean_inc(x_275);
x_336 = l_Lean_Expr_app___override(x_335, x_275);
lean_inc_ref(x_297);
x_337 = l_Lean_Expr_app___override(x_297, x_336);
x_338 = l_Lean_Expr_app___override(x_244, x_337);
lean_inc_ref(x_5);
x_339 = l_Lean_Expr_app___override(x_338, x_5);
x_340 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum(x_339, x_253, x_254, x_255, x_256);
if (lean_obj_tag(x_340) == 0)
{
uint8_t x_341; 
x_341 = !lean_is_exclusive(x_340);
if (x_341 == 0)
{
lean_object* x_342; lean_object* x_343; lean_object* x_344; lean_object* x_345; lean_object* x_346; lean_object* x_347; lean_object* x_348; lean_object* x_349; lean_object* x_350; lean_object* x_351; lean_object* x_352; lean_object* x_353; lean_object* x_354; lean_object* x_355; lean_object* x_356; lean_object* x_357; 
x_342 = lean_ctor_get(x_340, 0);
lean_inc_ref(x_286);
x_343 = l_Lean_Expr_app___override(x_334, x_286);
lean_inc(x_275);
x_344 = l_Lean_Expr_app___override(x_343, x_275);
x_345 = l_Lean_Expr_app___override(x_297, x_344);
x_346 = lp_mathlib_CancelDenoms_mkProdPrf___closed__36;
x_347 = l_Lean_Expr_const___override(x_346, x_19);
x_348 = l_Lean_Expr_app___override(x_347, x_2);
x_349 = l_Lean_Expr_app___override(x_348, x_246);
x_350 = l_Lean_Expr_app___override(x_349, x_281);
x_351 = l_Lean_Expr_app___override(x_350, x_273);
x_352 = l_Lean_Expr_app___override(x_351, x_286);
x_353 = l_Lean_Expr_app___override(x_352, x_5);
x_354 = l_Lean_Expr_app___override(x_353, x_294);
x_355 = l_Lean_Expr_app___override(x_354, x_275);
x_356 = l_Lean_Expr_app___override(x_355, x_287);
x_357 = l_Lean_Expr_app___override(x_356, x_342);
lean_ctor_set(x_284, 1, x_357);
lean_ctor_set(x_284, 0, x_345);
lean_ctor_set(x_340, 0, x_284);
return x_340;
}
else
{
lean_object* x_358; lean_object* x_359; lean_object* x_360; lean_object* x_361; lean_object* x_362; lean_object* x_363; lean_object* x_364; lean_object* x_365; lean_object* x_366; lean_object* x_367; lean_object* x_368; lean_object* x_369; lean_object* x_370; lean_object* x_371; lean_object* x_372; lean_object* x_373; lean_object* x_374; 
x_358 = lean_ctor_get(x_340, 0);
lean_inc(x_358);
lean_dec(x_340);
lean_inc_ref(x_286);
x_359 = l_Lean_Expr_app___override(x_334, x_286);
lean_inc(x_275);
x_360 = l_Lean_Expr_app___override(x_359, x_275);
x_361 = l_Lean_Expr_app___override(x_297, x_360);
x_362 = lp_mathlib_CancelDenoms_mkProdPrf___closed__36;
x_363 = l_Lean_Expr_const___override(x_362, x_19);
x_364 = l_Lean_Expr_app___override(x_363, x_2);
x_365 = l_Lean_Expr_app___override(x_364, x_246);
x_366 = l_Lean_Expr_app___override(x_365, x_281);
x_367 = l_Lean_Expr_app___override(x_366, x_273);
x_368 = l_Lean_Expr_app___override(x_367, x_286);
x_369 = l_Lean_Expr_app___override(x_368, x_5);
x_370 = l_Lean_Expr_app___override(x_369, x_294);
x_371 = l_Lean_Expr_app___override(x_370, x_275);
x_372 = l_Lean_Expr_app___override(x_371, x_287);
x_373 = l_Lean_Expr_app___override(x_372, x_358);
lean_ctor_set(x_284, 1, x_373);
lean_ctor_set(x_284, 0, x_361);
x_374 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_374, 0, x_284);
return x_374;
}
}
else
{
uint8_t x_375; 
lean_dec_ref(x_334);
lean_dec_ref(x_297);
lean_dec(x_294);
lean_free_object(x_284);
lean_dec_ref(x_287);
lean_dec_ref(x_286);
lean_dec(x_281);
lean_dec(x_275);
lean_dec(x_273);
lean_dec_ref(x_246);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
x_375 = !lean_is_exclusive(x_340);
if (x_375 == 0)
{
return x_340;
}
else
{
lean_object* x_376; lean_object* x_377; 
x_376 = lean_ctor_get(x_340, 0);
lean_inc(x_376);
lean_dec(x_340);
x_377 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_377, 0, x_376);
return x_377;
}
}
}
else
{
lean_object* x_378; lean_object* x_379; lean_object* x_380; lean_object* x_381; lean_object* x_382; lean_object* x_383; lean_object* x_384; lean_object* x_385; lean_object* x_386; lean_object* x_387; lean_object* x_388; lean_object* x_389; lean_object* x_390; lean_object* x_391; lean_object* x_392; lean_object* x_393; lean_object* x_394; lean_object* x_395; lean_object* x_396; lean_object* x_397; lean_object* x_398; lean_object* x_399; lean_object* x_400; lean_object* x_401; lean_object* x_402; lean_object* x_403; lean_object* x_404; lean_object* x_405; lean_object* x_406; lean_object* x_407; lean_object* x_408; lean_object* x_409; lean_object* x_410; lean_object* x_411; lean_object* x_412; lean_object* x_413; lean_object* x_414; lean_object* x_415; lean_object* x_416; lean_object* x_417; lean_object* x_418; lean_object* x_419; lean_object* x_420; lean_object* x_421; lean_object* x_422; lean_object* x_423; lean_object* x_424; 
x_378 = lean_ctor_get(x_292, 0);
lean_inc(x_378);
lean_dec(x_292);
x_379 = lean_box(0);
lean_inc(x_378);
x_380 = l_Lean_Expr_app___override(x_249, x_378);
x_381 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__2;
lean_inc_ref(x_19);
x_382 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_382, 0, x_379);
lean_ctor_set(x_382, 1, x_19);
lean_inc(x_1);
lean_ctor_set_tag(x_279, 1);
lean_ctor_set(x_279, 1, x_382);
lean_ctor_set(x_279, 0, x_1);
x_383 = l_Lean_Expr_const___override(x_381, x_279);
lean_inc_ref(x_2);
x_384 = l_Lean_Expr_app___override(x_383, x_2);
x_385 = lp_mathlib_CancelDenoms_mkProdPrf___closed__32;
x_386 = l_Lean_Expr_app___override(x_384, x_385);
lean_inc_ref(x_2);
x_387 = l_Lean_Expr_app___override(x_386, x_2);
x_388 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__4;
x_389 = lp_mathlib_CancelDenoms_mkProdPrf___closed__33;
lean_ctor_set_tag(x_267, 1);
lean_ctor_set(x_267, 1, x_389);
lean_ctor_set(x_267, 0, x_1);
x_390 = l_Lean_Expr_const___override(x_388, x_267);
lean_inc_ref(x_2);
x_391 = l_Lean_Expr_app___override(x_390, x_2);
x_392 = l_Lean_Expr_app___override(x_391, x_385);
x_393 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__7;
lean_inc_ref(x_19);
x_394 = l_Lean_Expr_const___override(x_393, x_19);
lean_inc_ref(x_2);
x_395 = l_Lean_Expr_app___override(x_394, x_2);
x_396 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__10;
lean_inc_ref(x_19);
x_397 = l_Lean_Expr_const___override(x_396, x_19);
lean_inc_ref(x_2);
x_398 = l_Lean_Expr_app___override(x_397, x_2);
x_399 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__13;
lean_inc_ref(x_19);
x_400 = l_Lean_Expr_const___override(x_399, x_19);
lean_inc_ref(x_2);
x_401 = l_Lean_Expr_app___override(x_400, x_2);
x_402 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__16;
lean_inc_ref(x_19);
x_403 = l_Lean_Expr_const___override(x_402, x_19);
lean_inc_ref(x_2);
x_404 = l_Lean_Expr_app___override(x_403, x_2);
x_405 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__18;
lean_inc_ref(x_19);
x_406 = l_Lean_Expr_const___override(x_405, x_19);
lean_inc_ref(x_2);
x_407 = l_Lean_Expr_app___override(x_406, x_2);
x_408 = lp_mathlib_CancelDenoms_mkProdPrf___closed__34;
lean_inc_ref(x_19);
x_409 = l_Lean_Expr_const___override(x_408, x_19);
lean_inc_ref(x_2);
x_410 = l_Lean_Expr_app___override(x_409, x_2);
x_411 = l_Lean_Expr_app___override(x_410, x_3);
x_412 = l_Lean_Expr_app___override(x_407, x_411);
x_413 = l_Lean_Expr_app___override(x_404, x_412);
x_414 = l_Lean_Expr_app___override(x_401, x_413);
x_415 = l_Lean_Expr_app___override(x_398, x_414);
x_416 = l_Lean_Expr_app___override(x_395, x_415);
x_417 = l_Lean_Expr_app___override(x_392, x_416);
x_418 = l_Lean_Expr_app___override(x_387, x_417);
lean_inc(x_281);
lean_inc_ref(x_418);
x_419 = l_Lean_Expr_app___override(x_418, x_281);
lean_inc(x_275);
x_420 = l_Lean_Expr_app___override(x_419, x_275);
lean_inc_ref(x_380);
x_421 = l_Lean_Expr_app___override(x_380, x_420);
x_422 = l_Lean_Expr_app___override(x_244, x_421);
lean_inc_ref(x_5);
x_423 = l_Lean_Expr_app___override(x_422, x_5);
x_424 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum(x_423, x_253, x_254, x_255, x_256);
if (lean_obj_tag(x_424) == 0)
{
lean_object* x_425; lean_object* x_426; lean_object* x_427; lean_object* x_428; lean_object* x_429; lean_object* x_430; lean_object* x_431; lean_object* x_432; lean_object* x_433; lean_object* x_434; lean_object* x_435; lean_object* x_436; lean_object* x_437; lean_object* x_438; lean_object* x_439; lean_object* x_440; lean_object* x_441; lean_object* x_442; 
x_425 = lean_ctor_get(x_424, 0);
lean_inc(x_425);
if (lean_is_exclusive(x_424)) {
 lean_ctor_release(x_424, 0);
 x_426 = x_424;
} else {
 lean_dec_ref(x_424);
 x_426 = lean_box(0);
}
lean_inc_ref(x_286);
x_427 = l_Lean_Expr_app___override(x_418, x_286);
lean_inc(x_275);
x_428 = l_Lean_Expr_app___override(x_427, x_275);
x_429 = l_Lean_Expr_app___override(x_380, x_428);
x_430 = lp_mathlib_CancelDenoms_mkProdPrf___closed__36;
x_431 = l_Lean_Expr_const___override(x_430, x_19);
x_432 = l_Lean_Expr_app___override(x_431, x_2);
x_433 = l_Lean_Expr_app___override(x_432, x_246);
x_434 = l_Lean_Expr_app___override(x_433, x_281);
x_435 = l_Lean_Expr_app___override(x_434, x_273);
x_436 = l_Lean_Expr_app___override(x_435, x_286);
x_437 = l_Lean_Expr_app___override(x_436, x_5);
x_438 = l_Lean_Expr_app___override(x_437, x_378);
x_439 = l_Lean_Expr_app___override(x_438, x_275);
x_440 = l_Lean_Expr_app___override(x_439, x_287);
x_441 = l_Lean_Expr_app___override(x_440, x_425);
lean_ctor_set(x_284, 1, x_441);
lean_ctor_set(x_284, 0, x_429);
if (lean_is_scalar(x_426)) {
 x_442 = lean_alloc_ctor(0, 1, 0);
} else {
 x_442 = x_426;
}
lean_ctor_set(x_442, 0, x_284);
return x_442;
}
else
{
lean_object* x_443; lean_object* x_444; lean_object* x_445; 
lean_dec_ref(x_418);
lean_dec_ref(x_380);
lean_dec(x_378);
lean_free_object(x_284);
lean_dec_ref(x_287);
lean_dec_ref(x_286);
lean_dec(x_281);
lean_dec(x_275);
lean_dec(x_273);
lean_dec_ref(x_246);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
x_443 = lean_ctor_get(x_424, 0);
lean_inc(x_443);
if (lean_is_exclusive(x_424)) {
 lean_ctor_release(x_424, 0);
 x_444 = x_424;
} else {
 lean_dec_ref(x_424);
 x_444 = lean_box(0);
}
if (lean_is_scalar(x_444)) {
 x_445 = lean_alloc_ctor(1, 1, 0);
} else {
 x_445 = x_444;
}
lean_ctor_set(x_445, 0, x_443);
return x_445;
}
}
}
else
{
uint8_t x_446; 
lean_free_object(x_284);
lean_dec_ref(x_287);
lean_dec_ref(x_286);
lean_free_object(x_279);
lean_dec(x_281);
lean_free_object(x_267);
lean_dec(x_275);
lean_dec(x_273);
lean_dec(x_256);
lean_dec_ref(x_255);
lean_dec(x_254);
lean_dec_ref(x_253);
lean_dec_ref(x_249);
lean_dec_ref(x_246);
lean_dec_ref(x_244);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_446 = !lean_is_exclusive(x_291);
if (x_446 == 0)
{
return x_291;
}
else
{
lean_object* x_447; lean_object* x_448; 
x_447 = lean_ctor_get(x_291, 0);
lean_inc(x_447);
lean_dec(x_291);
x_448 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_448, 0, x_447);
return x_448;
}
}
}
else
{
lean_object* x_449; lean_object* x_450; lean_object* x_451; lean_object* x_452; lean_object* x_453; lean_object* x_454; 
x_449 = lean_ctor_get(x_284, 0);
x_450 = lean_ctor_get(x_284, 1);
lean_inc(x_450);
lean_inc(x_449);
lean_dec(x_284);
x_451 = lean_nat_pow(x_251, x_252);
lean_dec(x_252);
lean_dec(x_251);
x_452 = lean_nat_div(x_4, x_451);
lean_dec(x_451);
lean_dec(x_4);
x_453 = l_Lean_mkRawNatLit(x_452);
lean_inc_ref(x_2);
lean_inc(x_1);
x_454 = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(x_1, x_2, x_47, x_453, x_253, x_254, x_255, x_256);
if (lean_obj_tag(x_454) == 0)
{
lean_object* x_455; lean_object* x_456; lean_object* x_457; lean_object* x_458; lean_object* x_459; lean_object* x_460; lean_object* x_461; lean_object* x_462; lean_object* x_463; lean_object* x_464; lean_object* x_465; lean_object* x_466; lean_object* x_467; lean_object* x_468; lean_object* x_469; lean_object* x_470; lean_object* x_471; lean_object* x_472; lean_object* x_473; lean_object* x_474; lean_object* x_475; lean_object* x_476; lean_object* x_477; lean_object* x_478; lean_object* x_479; lean_object* x_480; lean_object* x_481; lean_object* x_482; lean_object* x_483; lean_object* x_484; lean_object* x_485; lean_object* x_486; lean_object* x_487; lean_object* x_488; lean_object* x_489; lean_object* x_490; lean_object* x_491; lean_object* x_492; lean_object* x_493; lean_object* x_494; lean_object* x_495; lean_object* x_496; lean_object* x_497; lean_object* x_498; lean_object* x_499; lean_object* x_500; lean_object* x_501; lean_object* x_502; lean_object* x_503; 
x_455 = lean_ctor_get(x_454, 0);
lean_inc(x_455);
lean_dec_ref(x_454);
x_456 = lean_ctor_get(x_455, 0);
lean_inc(x_456);
if (lean_is_exclusive(x_455)) {
 lean_ctor_release(x_455, 0);
 lean_ctor_release(x_455, 1);
 x_457 = x_455;
} else {
 lean_dec_ref(x_455);
 x_457 = lean_box(0);
}
x_458 = lean_box(0);
lean_inc(x_456);
x_459 = l_Lean_Expr_app___override(x_249, x_456);
x_460 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__2;
lean_inc_ref(x_19);
if (lean_is_scalar(x_457)) {
 x_461 = lean_alloc_ctor(1, 2, 0);
} else {
 x_461 = x_457;
 lean_ctor_set_tag(x_461, 1);
}
lean_ctor_set(x_461, 0, x_458);
lean_ctor_set(x_461, 1, x_19);
lean_inc(x_1);
lean_ctor_set_tag(x_279, 1);
lean_ctor_set(x_279, 1, x_461);
lean_ctor_set(x_279, 0, x_1);
x_462 = l_Lean_Expr_const___override(x_460, x_279);
lean_inc_ref(x_2);
x_463 = l_Lean_Expr_app___override(x_462, x_2);
x_464 = lp_mathlib_CancelDenoms_mkProdPrf___closed__32;
x_465 = l_Lean_Expr_app___override(x_463, x_464);
lean_inc_ref(x_2);
x_466 = l_Lean_Expr_app___override(x_465, x_2);
x_467 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__4;
x_468 = lp_mathlib_CancelDenoms_mkProdPrf___closed__33;
lean_ctor_set_tag(x_267, 1);
lean_ctor_set(x_267, 1, x_468);
lean_ctor_set(x_267, 0, x_1);
x_469 = l_Lean_Expr_const___override(x_467, x_267);
lean_inc_ref(x_2);
x_470 = l_Lean_Expr_app___override(x_469, x_2);
x_471 = l_Lean_Expr_app___override(x_470, x_464);
x_472 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__7;
lean_inc_ref(x_19);
x_473 = l_Lean_Expr_const___override(x_472, x_19);
lean_inc_ref(x_2);
x_474 = l_Lean_Expr_app___override(x_473, x_2);
x_475 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__10;
lean_inc_ref(x_19);
x_476 = l_Lean_Expr_const___override(x_475, x_19);
lean_inc_ref(x_2);
x_477 = l_Lean_Expr_app___override(x_476, x_2);
x_478 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__13;
lean_inc_ref(x_19);
x_479 = l_Lean_Expr_const___override(x_478, x_19);
lean_inc_ref(x_2);
x_480 = l_Lean_Expr_app___override(x_479, x_2);
x_481 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__16;
lean_inc_ref(x_19);
x_482 = l_Lean_Expr_const___override(x_481, x_19);
lean_inc_ref(x_2);
x_483 = l_Lean_Expr_app___override(x_482, x_2);
x_484 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__18;
lean_inc_ref(x_19);
x_485 = l_Lean_Expr_const___override(x_484, x_19);
lean_inc_ref(x_2);
x_486 = l_Lean_Expr_app___override(x_485, x_2);
x_487 = lp_mathlib_CancelDenoms_mkProdPrf___closed__34;
lean_inc_ref(x_19);
x_488 = l_Lean_Expr_const___override(x_487, x_19);
lean_inc_ref(x_2);
x_489 = l_Lean_Expr_app___override(x_488, x_2);
x_490 = l_Lean_Expr_app___override(x_489, x_3);
x_491 = l_Lean_Expr_app___override(x_486, x_490);
x_492 = l_Lean_Expr_app___override(x_483, x_491);
x_493 = l_Lean_Expr_app___override(x_480, x_492);
x_494 = l_Lean_Expr_app___override(x_477, x_493);
x_495 = l_Lean_Expr_app___override(x_474, x_494);
x_496 = l_Lean_Expr_app___override(x_471, x_495);
x_497 = l_Lean_Expr_app___override(x_466, x_496);
lean_inc(x_281);
lean_inc_ref(x_497);
x_498 = l_Lean_Expr_app___override(x_497, x_281);
lean_inc(x_275);
x_499 = l_Lean_Expr_app___override(x_498, x_275);
lean_inc_ref(x_459);
x_500 = l_Lean_Expr_app___override(x_459, x_499);
x_501 = l_Lean_Expr_app___override(x_244, x_500);
lean_inc_ref(x_5);
x_502 = l_Lean_Expr_app___override(x_501, x_5);
x_503 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum(x_502, x_253, x_254, x_255, x_256);
if (lean_obj_tag(x_503) == 0)
{
lean_object* x_504; lean_object* x_505; lean_object* x_506; lean_object* x_507; lean_object* x_508; lean_object* x_509; lean_object* x_510; lean_object* x_511; lean_object* x_512; lean_object* x_513; lean_object* x_514; lean_object* x_515; lean_object* x_516; lean_object* x_517; lean_object* x_518; lean_object* x_519; lean_object* x_520; lean_object* x_521; lean_object* x_522; 
x_504 = lean_ctor_get(x_503, 0);
lean_inc(x_504);
if (lean_is_exclusive(x_503)) {
 lean_ctor_release(x_503, 0);
 x_505 = x_503;
} else {
 lean_dec_ref(x_503);
 x_505 = lean_box(0);
}
lean_inc_ref(x_449);
x_506 = l_Lean_Expr_app___override(x_497, x_449);
lean_inc(x_275);
x_507 = l_Lean_Expr_app___override(x_506, x_275);
x_508 = l_Lean_Expr_app___override(x_459, x_507);
x_509 = lp_mathlib_CancelDenoms_mkProdPrf___closed__36;
x_510 = l_Lean_Expr_const___override(x_509, x_19);
x_511 = l_Lean_Expr_app___override(x_510, x_2);
x_512 = l_Lean_Expr_app___override(x_511, x_246);
x_513 = l_Lean_Expr_app___override(x_512, x_281);
x_514 = l_Lean_Expr_app___override(x_513, x_273);
x_515 = l_Lean_Expr_app___override(x_514, x_449);
x_516 = l_Lean_Expr_app___override(x_515, x_5);
x_517 = l_Lean_Expr_app___override(x_516, x_456);
x_518 = l_Lean_Expr_app___override(x_517, x_275);
x_519 = l_Lean_Expr_app___override(x_518, x_450);
x_520 = l_Lean_Expr_app___override(x_519, x_504);
x_521 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_521, 0, x_508);
lean_ctor_set(x_521, 1, x_520);
if (lean_is_scalar(x_505)) {
 x_522 = lean_alloc_ctor(0, 1, 0);
} else {
 x_522 = x_505;
}
lean_ctor_set(x_522, 0, x_521);
return x_522;
}
else
{
lean_object* x_523; lean_object* x_524; lean_object* x_525; 
lean_dec_ref(x_497);
lean_dec_ref(x_459);
lean_dec(x_456);
lean_dec_ref(x_450);
lean_dec_ref(x_449);
lean_dec(x_281);
lean_dec(x_275);
lean_dec(x_273);
lean_dec_ref(x_246);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
x_523 = lean_ctor_get(x_503, 0);
lean_inc(x_523);
if (lean_is_exclusive(x_503)) {
 lean_ctor_release(x_503, 0);
 x_524 = x_503;
} else {
 lean_dec_ref(x_503);
 x_524 = lean_box(0);
}
if (lean_is_scalar(x_524)) {
 x_525 = lean_alloc_ctor(1, 1, 0);
} else {
 x_525 = x_524;
}
lean_ctor_set(x_525, 0, x_523);
return x_525;
}
}
else
{
lean_object* x_526; lean_object* x_527; lean_object* x_528; 
lean_dec_ref(x_450);
lean_dec_ref(x_449);
lean_free_object(x_279);
lean_dec(x_281);
lean_free_object(x_267);
lean_dec(x_275);
lean_dec(x_273);
lean_dec(x_256);
lean_dec_ref(x_255);
lean_dec(x_254);
lean_dec_ref(x_253);
lean_dec_ref(x_249);
lean_dec_ref(x_246);
lean_dec_ref(x_244);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_526 = lean_ctor_get(x_454, 0);
lean_inc(x_526);
if (lean_is_exclusive(x_454)) {
 lean_ctor_release(x_454, 0);
 x_527 = x_454;
} else {
 lean_dec_ref(x_454);
 x_527 = lean_box(0);
}
if (lean_is_scalar(x_527)) {
 x_528 = lean_alloc_ctor(1, 1, 0);
} else {
 x_528 = x_527;
}
lean_ctor_set(x_528, 0, x_526);
return x_528;
}
}
}
else
{
lean_free_object(x_279);
lean_dec(x_281);
lean_free_object(x_267);
lean_dec(x_275);
lean_dec(x_273);
lean_dec(x_256);
lean_dec_ref(x_255);
lean_dec(x_254);
lean_dec_ref(x_253);
lean_dec(x_252);
lean_dec(x_251);
lean_dec_ref(x_249);
lean_dec_ref(x_246);
lean_dec_ref(x_244);
lean_dec_ref(x_47);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_283;
}
}
else
{
lean_object* x_529; lean_object* x_530; 
x_529 = lean_ctor_get(x_279, 0);
lean_inc(x_529);
lean_dec(x_279);
lean_inc(x_256);
lean_inc_ref(x_255);
lean_inc(x_254);
lean_inc_ref(x_253);
lean_inc(x_273);
lean_inc(x_529);
lean_inc(x_251);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc(x_1);
x_530 = lp_mathlib_CancelDenoms_mkProdPrf(x_1, x_2, x_3, x_251, x_529, x_250, x_273, x_253, x_254, x_255, x_256);
if (lean_obj_tag(x_530) == 0)
{
lean_object* x_531; lean_object* x_532; lean_object* x_533; lean_object* x_534; lean_object* x_535; lean_object* x_536; lean_object* x_537; lean_object* x_538; 
x_531 = lean_ctor_get(x_530, 0);
lean_inc(x_531);
lean_dec_ref(x_530);
x_532 = lean_ctor_get(x_531, 0);
lean_inc_ref(x_532);
x_533 = lean_ctor_get(x_531, 1);
lean_inc_ref(x_533);
if (lean_is_exclusive(x_531)) {
 lean_ctor_release(x_531, 0);
 lean_ctor_release(x_531, 1);
 x_534 = x_531;
} else {
 lean_dec_ref(x_531);
 x_534 = lean_box(0);
}
x_535 = lean_nat_pow(x_251, x_252);
lean_dec(x_252);
lean_dec(x_251);
x_536 = lean_nat_div(x_4, x_535);
lean_dec(x_535);
lean_dec(x_4);
x_537 = l_Lean_mkRawNatLit(x_536);
lean_inc_ref(x_2);
lean_inc(x_1);
x_538 = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(x_1, x_2, x_47, x_537, x_253, x_254, x_255, x_256);
if (lean_obj_tag(x_538) == 0)
{
lean_object* x_539; lean_object* x_540; lean_object* x_541; lean_object* x_542; lean_object* x_543; lean_object* x_544; lean_object* x_545; lean_object* x_546; lean_object* x_547; lean_object* x_548; lean_object* x_549; lean_object* x_550; lean_object* x_551; lean_object* x_552; lean_object* x_553; lean_object* x_554; lean_object* x_555; lean_object* x_556; lean_object* x_557; lean_object* x_558; lean_object* x_559; lean_object* x_560; lean_object* x_561; lean_object* x_562; lean_object* x_563; lean_object* x_564; lean_object* x_565; lean_object* x_566; lean_object* x_567; lean_object* x_568; lean_object* x_569; lean_object* x_570; lean_object* x_571; lean_object* x_572; lean_object* x_573; lean_object* x_574; lean_object* x_575; lean_object* x_576; lean_object* x_577; lean_object* x_578; lean_object* x_579; lean_object* x_580; lean_object* x_581; lean_object* x_582; lean_object* x_583; lean_object* x_584; lean_object* x_585; lean_object* x_586; lean_object* x_587; lean_object* x_588; 
x_539 = lean_ctor_get(x_538, 0);
lean_inc(x_539);
lean_dec_ref(x_538);
x_540 = lean_ctor_get(x_539, 0);
lean_inc(x_540);
if (lean_is_exclusive(x_539)) {
 lean_ctor_release(x_539, 0);
 lean_ctor_release(x_539, 1);
 x_541 = x_539;
} else {
 lean_dec_ref(x_539);
 x_541 = lean_box(0);
}
x_542 = lean_box(0);
lean_inc(x_540);
x_543 = l_Lean_Expr_app___override(x_249, x_540);
x_544 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__2;
lean_inc_ref(x_19);
if (lean_is_scalar(x_541)) {
 x_545 = lean_alloc_ctor(1, 2, 0);
} else {
 x_545 = x_541;
 lean_ctor_set_tag(x_545, 1);
}
lean_ctor_set(x_545, 0, x_542);
lean_ctor_set(x_545, 1, x_19);
lean_inc(x_1);
x_546 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_546, 0, x_1);
lean_ctor_set(x_546, 1, x_545);
x_547 = l_Lean_Expr_const___override(x_544, x_546);
lean_inc_ref(x_2);
x_548 = l_Lean_Expr_app___override(x_547, x_2);
x_549 = lp_mathlib_CancelDenoms_mkProdPrf___closed__32;
x_550 = l_Lean_Expr_app___override(x_548, x_549);
lean_inc_ref(x_2);
x_551 = l_Lean_Expr_app___override(x_550, x_2);
x_552 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__4;
x_553 = lp_mathlib_CancelDenoms_mkProdPrf___closed__33;
lean_ctor_set_tag(x_267, 1);
lean_ctor_set(x_267, 1, x_553);
lean_ctor_set(x_267, 0, x_1);
x_554 = l_Lean_Expr_const___override(x_552, x_267);
lean_inc_ref(x_2);
x_555 = l_Lean_Expr_app___override(x_554, x_2);
x_556 = l_Lean_Expr_app___override(x_555, x_549);
x_557 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__7;
lean_inc_ref(x_19);
x_558 = l_Lean_Expr_const___override(x_557, x_19);
lean_inc_ref(x_2);
x_559 = l_Lean_Expr_app___override(x_558, x_2);
x_560 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__10;
lean_inc_ref(x_19);
x_561 = l_Lean_Expr_const___override(x_560, x_19);
lean_inc_ref(x_2);
x_562 = l_Lean_Expr_app___override(x_561, x_2);
x_563 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__13;
lean_inc_ref(x_19);
x_564 = l_Lean_Expr_const___override(x_563, x_19);
lean_inc_ref(x_2);
x_565 = l_Lean_Expr_app___override(x_564, x_2);
x_566 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__16;
lean_inc_ref(x_19);
x_567 = l_Lean_Expr_const___override(x_566, x_19);
lean_inc_ref(x_2);
x_568 = l_Lean_Expr_app___override(x_567, x_2);
x_569 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__18;
lean_inc_ref(x_19);
x_570 = l_Lean_Expr_const___override(x_569, x_19);
lean_inc_ref(x_2);
x_571 = l_Lean_Expr_app___override(x_570, x_2);
x_572 = lp_mathlib_CancelDenoms_mkProdPrf___closed__34;
lean_inc_ref(x_19);
x_573 = l_Lean_Expr_const___override(x_572, x_19);
lean_inc_ref(x_2);
x_574 = l_Lean_Expr_app___override(x_573, x_2);
x_575 = l_Lean_Expr_app___override(x_574, x_3);
x_576 = l_Lean_Expr_app___override(x_571, x_575);
x_577 = l_Lean_Expr_app___override(x_568, x_576);
x_578 = l_Lean_Expr_app___override(x_565, x_577);
x_579 = l_Lean_Expr_app___override(x_562, x_578);
x_580 = l_Lean_Expr_app___override(x_559, x_579);
x_581 = l_Lean_Expr_app___override(x_556, x_580);
x_582 = l_Lean_Expr_app___override(x_551, x_581);
lean_inc(x_529);
lean_inc_ref(x_582);
x_583 = l_Lean_Expr_app___override(x_582, x_529);
lean_inc(x_275);
x_584 = l_Lean_Expr_app___override(x_583, x_275);
lean_inc_ref(x_543);
x_585 = l_Lean_Expr_app___override(x_543, x_584);
x_586 = l_Lean_Expr_app___override(x_244, x_585);
lean_inc_ref(x_5);
x_587 = l_Lean_Expr_app___override(x_586, x_5);
x_588 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum(x_587, x_253, x_254, x_255, x_256);
if (lean_obj_tag(x_588) == 0)
{
lean_object* x_589; lean_object* x_590; lean_object* x_591; lean_object* x_592; lean_object* x_593; lean_object* x_594; lean_object* x_595; lean_object* x_596; lean_object* x_597; lean_object* x_598; lean_object* x_599; lean_object* x_600; lean_object* x_601; lean_object* x_602; lean_object* x_603; lean_object* x_604; lean_object* x_605; lean_object* x_606; lean_object* x_607; 
x_589 = lean_ctor_get(x_588, 0);
lean_inc(x_589);
if (lean_is_exclusive(x_588)) {
 lean_ctor_release(x_588, 0);
 x_590 = x_588;
} else {
 lean_dec_ref(x_588);
 x_590 = lean_box(0);
}
lean_inc_ref(x_532);
x_591 = l_Lean_Expr_app___override(x_582, x_532);
lean_inc(x_275);
x_592 = l_Lean_Expr_app___override(x_591, x_275);
x_593 = l_Lean_Expr_app___override(x_543, x_592);
x_594 = lp_mathlib_CancelDenoms_mkProdPrf___closed__36;
x_595 = l_Lean_Expr_const___override(x_594, x_19);
x_596 = l_Lean_Expr_app___override(x_595, x_2);
x_597 = l_Lean_Expr_app___override(x_596, x_246);
x_598 = l_Lean_Expr_app___override(x_597, x_529);
x_599 = l_Lean_Expr_app___override(x_598, x_273);
x_600 = l_Lean_Expr_app___override(x_599, x_532);
x_601 = l_Lean_Expr_app___override(x_600, x_5);
x_602 = l_Lean_Expr_app___override(x_601, x_540);
x_603 = l_Lean_Expr_app___override(x_602, x_275);
x_604 = l_Lean_Expr_app___override(x_603, x_533);
x_605 = l_Lean_Expr_app___override(x_604, x_589);
if (lean_is_scalar(x_534)) {
 x_606 = lean_alloc_ctor(0, 2, 0);
} else {
 x_606 = x_534;
}
lean_ctor_set(x_606, 0, x_593);
lean_ctor_set(x_606, 1, x_605);
if (lean_is_scalar(x_590)) {
 x_607 = lean_alloc_ctor(0, 1, 0);
} else {
 x_607 = x_590;
}
lean_ctor_set(x_607, 0, x_606);
return x_607;
}
else
{
lean_object* x_608; lean_object* x_609; lean_object* x_610; 
lean_dec_ref(x_582);
lean_dec_ref(x_543);
lean_dec(x_540);
lean_dec(x_534);
lean_dec_ref(x_533);
lean_dec_ref(x_532);
lean_dec(x_529);
lean_dec(x_275);
lean_dec(x_273);
lean_dec_ref(x_246);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
x_608 = lean_ctor_get(x_588, 0);
lean_inc(x_608);
if (lean_is_exclusive(x_588)) {
 lean_ctor_release(x_588, 0);
 x_609 = x_588;
} else {
 lean_dec_ref(x_588);
 x_609 = lean_box(0);
}
if (lean_is_scalar(x_609)) {
 x_610 = lean_alloc_ctor(1, 1, 0);
} else {
 x_610 = x_609;
}
lean_ctor_set(x_610, 0, x_608);
return x_610;
}
}
else
{
lean_object* x_611; lean_object* x_612; lean_object* x_613; 
lean_dec(x_534);
lean_dec_ref(x_533);
lean_dec_ref(x_532);
lean_dec(x_529);
lean_free_object(x_267);
lean_dec(x_275);
lean_dec(x_273);
lean_dec(x_256);
lean_dec_ref(x_255);
lean_dec(x_254);
lean_dec_ref(x_253);
lean_dec_ref(x_249);
lean_dec_ref(x_246);
lean_dec_ref(x_244);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_611 = lean_ctor_get(x_538, 0);
lean_inc(x_611);
if (lean_is_exclusive(x_538)) {
 lean_ctor_release(x_538, 0);
 x_612 = x_538;
} else {
 lean_dec_ref(x_538);
 x_612 = lean_box(0);
}
if (lean_is_scalar(x_612)) {
 x_613 = lean_alloc_ctor(1, 1, 0);
} else {
 x_613 = x_612;
}
lean_ctor_set(x_613, 0, x_611);
return x_613;
}
}
else
{
lean_dec(x_529);
lean_free_object(x_267);
lean_dec(x_275);
lean_dec(x_273);
lean_dec(x_256);
lean_dec_ref(x_255);
lean_dec(x_254);
lean_dec_ref(x_253);
lean_dec(x_252);
lean_dec(x_251);
lean_dec_ref(x_249);
lean_dec_ref(x_246);
lean_dec_ref(x_244);
lean_dec_ref(x_47);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_530;
}
}
}
else
{
uint8_t x_614; 
lean_free_object(x_267);
lean_dec(x_275);
lean_dec(x_273);
lean_dec(x_256);
lean_dec_ref(x_255);
lean_dec(x_254);
lean_dec_ref(x_253);
lean_dec(x_252);
lean_dec(x_251);
lean_dec(x_250);
lean_dec_ref(x_249);
lean_dec_ref(x_246);
lean_dec_ref(x_244);
lean_dec_ref(x_47);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_614 = !lean_is_exclusive(x_278);
if (x_614 == 0)
{
return x_278;
}
else
{
lean_object* x_615; lean_object* x_616; 
x_615 = lean_ctor_get(x_278, 0);
lean_inc(x_615);
lean_dec(x_278);
x_616 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_616, 0, x_615);
return x_616;
}
}
}
else
{
lean_object* x_617; lean_object* x_618; lean_object* x_619; 
x_617 = lean_ctor_get(x_267, 0);
lean_inc(x_617);
lean_dec(x_267);
lean_inc(x_251);
x_618 = l_Lean_mkRawNatLit(x_251);
lean_inc_ref(x_47);
lean_inc_ref(x_2);
lean_inc(x_1);
x_619 = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(x_1, x_2, x_47, x_618, x_253, x_254, x_255, x_256);
if (lean_obj_tag(x_619) == 0)
{
lean_object* x_620; lean_object* x_621; lean_object* x_622; lean_object* x_623; 
x_620 = lean_ctor_get(x_619, 0);
lean_inc(x_620);
lean_dec_ref(x_619);
x_621 = lean_ctor_get(x_620, 0);
lean_inc(x_621);
if (lean_is_exclusive(x_620)) {
 lean_ctor_release(x_620, 0);
 lean_ctor_release(x_620, 1);
 x_622 = x_620;
} else {
 lean_dec_ref(x_620);
 x_622 = lean_box(0);
}
lean_inc(x_256);
lean_inc_ref(x_255);
lean_inc(x_254);
lean_inc_ref(x_253);
lean_inc(x_273);
lean_inc(x_621);
lean_inc(x_251);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc(x_1);
x_623 = lp_mathlib_CancelDenoms_mkProdPrf(x_1, x_2, x_3, x_251, x_621, x_250, x_273, x_253, x_254, x_255, x_256);
if (lean_obj_tag(x_623) == 0)
{
lean_object* x_624; lean_object* x_625; lean_object* x_626; lean_object* x_627; lean_object* x_628; lean_object* x_629; lean_object* x_630; lean_object* x_631; 
x_624 = lean_ctor_get(x_623, 0);
lean_inc(x_624);
lean_dec_ref(x_623);
x_625 = lean_ctor_get(x_624, 0);
lean_inc_ref(x_625);
x_626 = lean_ctor_get(x_624, 1);
lean_inc_ref(x_626);
if (lean_is_exclusive(x_624)) {
 lean_ctor_release(x_624, 0);
 lean_ctor_release(x_624, 1);
 x_627 = x_624;
} else {
 lean_dec_ref(x_624);
 x_627 = lean_box(0);
}
x_628 = lean_nat_pow(x_251, x_252);
lean_dec(x_252);
lean_dec(x_251);
x_629 = lean_nat_div(x_4, x_628);
lean_dec(x_628);
lean_dec(x_4);
x_630 = l_Lean_mkRawNatLit(x_629);
lean_inc_ref(x_2);
lean_inc(x_1);
x_631 = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(x_1, x_2, x_47, x_630, x_253, x_254, x_255, x_256);
if (lean_obj_tag(x_631) == 0)
{
lean_object* x_632; lean_object* x_633; lean_object* x_634; lean_object* x_635; lean_object* x_636; lean_object* x_637; lean_object* x_638; lean_object* x_639; lean_object* x_640; lean_object* x_641; lean_object* x_642; lean_object* x_643; lean_object* x_644; lean_object* x_645; lean_object* x_646; lean_object* x_647; lean_object* x_648; lean_object* x_649; lean_object* x_650; lean_object* x_651; lean_object* x_652; lean_object* x_653; lean_object* x_654; lean_object* x_655; lean_object* x_656; lean_object* x_657; lean_object* x_658; lean_object* x_659; lean_object* x_660; lean_object* x_661; lean_object* x_662; lean_object* x_663; lean_object* x_664; lean_object* x_665; lean_object* x_666; lean_object* x_667; lean_object* x_668; lean_object* x_669; lean_object* x_670; lean_object* x_671; lean_object* x_672; lean_object* x_673; lean_object* x_674; lean_object* x_675; lean_object* x_676; lean_object* x_677; lean_object* x_678; lean_object* x_679; lean_object* x_680; lean_object* x_681; lean_object* x_682; 
x_632 = lean_ctor_get(x_631, 0);
lean_inc(x_632);
lean_dec_ref(x_631);
x_633 = lean_ctor_get(x_632, 0);
lean_inc(x_633);
if (lean_is_exclusive(x_632)) {
 lean_ctor_release(x_632, 0);
 lean_ctor_release(x_632, 1);
 x_634 = x_632;
} else {
 lean_dec_ref(x_632);
 x_634 = lean_box(0);
}
x_635 = lean_box(0);
lean_inc(x_633);
x_636 = l_Lean_Expr_app___override(x_249, x_633);
x_637 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__2;
lean_inc_ref(x_19);
if (lean_is_scalar(x_634)) {
 x_638 = lean_alloc_ctor(1, 2, 0);
} else {
 x_638 = x_634;
 lean_ctor_set_tag(x_638, 1);
}
lean_ctor_set(x_638, 0, x_635);
lean_ctor_set(x_638, 1, x_19);
lean_inc(x_1);
if (lean_is_scalar(x_622)) {
 x_639 = lean_alloc_ctor(1, 2, 0);
} else {
 x_639 = x_622;
 lean_ctor_set_tag(x_639, 1);
}
lean_ctor_set(x_639, 0, x_1);
lean_ctor_set(x_639, 1, x_638);
x_640 = l_Lean_Expr_const___override(x_637, x_639);
lean_inc_ref(x_2);
x_641 = l_Lean_Expr_app___override(x_640, x_2);
x_642 = lp_mathlib_CancelDenoms_mkProdPrf___closed__32;
x_643 = l_Lean_Expr_app___override(x_641, x_642);
lean_inc_ref(x_2);
x_644 = l_Lean_Expr_app___override(x_643, x_2);
x_645 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__4;
x_646 = lp_mathlib_CancelDenoms_mkProdPrf___closed__33;
x_647 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_647, 0, x_1);
lean_ctor_set(x_647, 1, x_646);
x_648 = l_Lean_Expr_const___override(x_645, x_647);
lean_inc_ref(x_2);
x_649 = l_Lean_Expr_app___override(x_648, x_2);
x_650 = l_Lean_Expr_app___override(x_649, x_642);
x_651 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__7;
lean_inc_ref(x_19);
x_652 = l_Lean_Expr_const___override(x_651, x_19);
lean_inc_ref(x_2);
x_653 = l_Lean_Expr_app___override(x_652, x_2);
x_654 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__10;
lean_inc_ref(x_19);
x_655 = l_Lean_Expr_const___override(x_654, x_19);
lean_inc_ref(x_2);
x_656 = l_Lean_Expr_app___override(x_655, x_2);
x_657 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__13;
lean_inc_ref(x_19);
x_658 = l_Lean_Expr_const___override(x_657, x_19);
lean_inc_ref(x_2);
x_659 = l_Lean_Expr_app___override(x_658, x_2);
x_660 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__16;
lean_inc_ref(x_19);
x_661 = l_Lean_Expr_const___override(x_660, x_19);
lean_inc_ref(x_2);
x_662 = l_Lean_Expr_app___override(x_661, x_2);
x_663 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__18;
lean_inc_ref(x_19);
x_664 = l_Lean_Expr_const___override(x_663, x_19);
lean_inc_ref(x_2);
x_665 = l_Lean_Expr_app___override(x_664, x_2);
x_666 = lp_mathlib_CancelDenoms_mkProdPrf___closed__34;
lean_inc_ref(x_19);
x_667 = l_Lean_Expr_const___override(x_666, x_19);
lean_inc_ref(x_2);
x_668 = l_Lean_Expr_app___override(x_667, x_2);
x_669 = l_Lean_Expr_app___override(x_668, x_3);
x_670 = l_Lean_Expr_app___override(x_665, x_669);
x_671 = l_Lean_Expr_app___override(x_662, x_670);
x_672 = l_Lean_Expr_app___override(x_659, x_671);
x_673 = l_Lean_Expr_app___override(x_656, x_672);
x_674 = l_Lean_Expr_app___override(x_653, x_673);
x_675 = l_Lean_Expr_app___override(x_650, x_674);
x_676 = l_Lean_Expr_app___override(x_644, x_675);
lean_inc(x_621);
lean_inc_ref(x_676);
x_677 = l_Lean_Expr_app___override(x_676, x_621);
lean_inc(x_617);
x_678 = l_Lean_Expr_app___override(x_677, x_617);
lean_inc_ref(x_636);
x_679 = l_Lean_Expr_app___override(x_636, x_678);
x_680 = l_Lean_Expr_app___override(x_244, x_679);
lean_inc_ref(x_5);
x_681 = l_Lean_Expr_app___override(x_680, x_5);
x_682 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum(x_681, x_253, x_254, x_255, x_256);
if (lean_obj_tag(x_682) == 0)
{
lean_object* x_683; lean_object* x_684; lean_object* x_685; lean_object* x_686; lean_object* x_687; lean_object* x_688; lean_object* x_689; lean_object* x_690; lean_object* x_691; lean_object* x_692; lean_object* x_693; lean_object* x_694; lean_object* x_695; lean_object* x_696; lean_object* x_697; lean_object* x_698; lean_object* x_699; lean_object* x_700; lean_object* x_701; 
x_683 = lean_ctor_get(x_682, 0);
lean_inc(x_683);
if (lean_is_exclusive(x_682)) {
 lean_ctor_release(x_682, 0);
 x_684 = x_682;
} else {
 lean_dec_ref(x_682);
 x_684 = lean_box(0);
}
lean_inc_ref(x_625);
x_685 = l_Lean_Expr_app___override(x_676, x_625);
lean_inc(x_617);
x_686 = l_Lean_Expr_app___override(x_685, x_617);
x_687 = l_Lean_Expr_app___override(x_636, x_686);
x_688 = lp_mathlib_CancelDenoms_mkProdPrf___closed__36;
x_689 = l_Lean_Expr_const___override(x_688, x_19);
x_690 = l_Lean_Expr_app___override(x_689, x_2);
x_691 = l_Lean_Expr_app___override(x_690, x_246);
x_692 = l_Lean_Expr_app___override(x_691, x_621);
x_693 = l_Lean_Expr_app___override(x_692, x_273);
x_694 = l_Lean_Expr_app___override(x_693, x_625);
x_695 = l_Lean_Expr_app___override(x_694, x_5);
x_696 = l_Lean_Expr_app___override(x_695, x_633);
x_697 = l_Lean_Expr_app___override(x_696, x_617);
x_698 = l_Lean_Expr_app___override(x_697, x_626);
x_699 = l_Lean_Expr_app___override(x_698, x_683);
if (lean_is_scalar(x_627)) {
 x_700 = lean_alloc_ctor(0, 2, 0);
} else {
 x_700 = x_627;
}
lean_ctor_set(x_700, 0, x_687);
lean_ctor_set(x_700, 1, x_699);
if (lean_is_scalar(x_684)) {
 x_701 = lean_alloc_ctor(0, 1, 0);
} else {
 x_701 = x_684;
}
lean_ctor_set(x_701, 0, x_700);
return x_701;
}
else
{
lean_object* x_702; lean_object* x_703; lean_object* x_704; 
lean_dec_ref(x_676);
lean_dec_ref(x_636);
lean_dec(x_633);
lean_dec(x_627);
lean_dec_ref(x_626);
lean_dec_ref(x_625);
lean_dec(x_621);
lean_dec(x_617);
lean_dec(x_273);
lean_dec_ref(x_246);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
x_702 = lean_ctor_get(x_682, 0);
lean_inc(x_702);
if (lean_is_exclusive(x_682)) {
 lean_ctor_release(x_682, 0);
 x_703 = x_682;
} else {
 lean_dec_ref(x_682);
 x_703 = lean_box(0);
}
if (lean_is_scalar(x_703)) {
 x_704 = lean_alloc_ctor(1, 1, 0);
} else {
 x_704 = x_703;
}
lean_ctor_set(x_704, 0, x_702);
return x_704;
}
}
else
{
lean_object* x_705; lean_object* x_706; lean_object* x_707; 
lean_dec(x_627);
lean_dec_ref(x_626);
lean_dec_ref(x_625);
lean_dec(x_622);
lean_dec(x_621);
lean_dec(x_617);
lean_dec(x_273);
lean_dec(x_256);
lean_dec_ref(x_255);
lean_dec(x_254);
lean_dec_ref(x_253);
lean_dec_ref(x_249);
lean_dec_ref(x_246);
lean_dec_ref(x_244);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_705 = lean_ctor_get(x_631, 0);
lean_inc(x_705);
if (lean_is_exclusive(x_631)) {
 lean_ctor_release(x_631, 0);
 x_706 = x_631;
} else {
 lean_dec_ref(x_631);
 x_706 = lean_box(0);
}
if (lean_is_scalar(x_706)) {
 x_707 = lean_alloc_ctor(1, 1, 0);
} else {
 x_707 = x_706;
}
lean_ctor_set(x_707, 0, x_705);
return x_707;
}
}
else
{
lean_dec(x_622);
lean_dec(x_621);
lean_dec(x_617);
lean_dec(x_273);
lean_dec(x_256);
lean_dec_ref(x_255);
lean_dec(x_254);
lean_dec_ref(x_253);
lean_dec(x_252);
lean_dec(x_251);
lean_dec_ref(x_249);
lean_dec_ref(x_246);
lean_dec_ref(x_244);
lean_dec_ref(x_47);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_623;
}
}
else
{
lean_object* x_708; lean_object* x_709; lean_object* x_710; 
lean_dec(x_617);
lean_dec(x_273);
lean_dec(x_256);
lean_dec_ref(x_255);
lean_dec(x_254);
lean_dec_ref(x_253);
lean_dec(x_252);
lean_dec(x_251);
lean_dec(x_250);
lean_dec_ref(x_249);
lean_dec_ref(x_246);
lean_dec_ref(x_244);
lean_dec_ref(x_47);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_708 = lean_ctor_get(x_619, 0);
lean_inc(x_708);
if (lean_is_exclusive(x_619)) {
 lean_ctor_release(x_619, 0);
 x_709 = x_619;
} else {
 lean_dec_ref(x_619);
 x_709 = lean_box(0);
}
if (lean_is_scalar(x_709)) {
 x_710 = lean_alloc_ctor(1, 1, 0);
} else {
 x_710 = x_709;
}
lean_ctor_set(x_710, 0, x_708);
return x_710;
}
}
}
}
else
{
lean_object* x_711; lean_object* x_712; lean_object* x_713; uint8_t x_714; 
x_711 = lean_ctor_get(x_264, 0);
lean_inc(x_711);
lean_dec(x_264);
x_712 = lean_ctor_get(x_711, 1);
lean_inc(x_712);
x_713 = lean_ctor_get(x_712, 1);
x_714 = lean_unbox(x_713);
if (x_714 == 0)
{
lean_dec(x_712);
lean_dec(x_711);
lean_dec(x_252);
lean_dec(x_251);
lean_dec(x_250);
lean_dec_ref(x_246);
if (lean_obj_tag(x_6) == 1)
{
lean_object* x_715; 
x_715 = lean_ctor_get(x_6, 1);
if (lean_obj_tag(x_715) == 0)
{
lean_object* x_716; 
x_716 = lean_ctor_get(x_6, 2);
lean_inc(x_716);
lean_dec_ref(x_6);
if (lean_obj_tag(x_716) == 1)
{
lean_object* x_717; 
x_717 = lean_ctor_get(x_716, 0);
lean_inc(x_717);
lean_dec_ref(x_716);
x_48 = x_244;
x_49 = x_245;
x_50 = x_249;
x_51 = x_248;
x_52 = x_247;
x_53 = x_717;
x_54 = x_253;
x_55 = x_254;
x_56 = x_255;
x_57 = x_256;
x_58 = lean_box(0);
goto block_243;
}
else
{
lean_object* x_718; 
lean_dec(x_716);
lean_dec(x_256);
lean_dec_ref(x_255);
lean_dec(x_254);
lean_dec_ref(x_253);
lean_dec_ref(x_249);
lean_dec_ref(x_248);
lean_dec_ref(x_247);
lean_dec_ref(x_244);
lean_dec_ref(x_47);
lean_dec_ref(x_24);
lean_dec_ref(x_19);
lean_dec_ref(x_7);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_718 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_718, 0, x_245);
return x_718;
}
}
else
{
lean_object* x_719; 
lean_dec(x_256);
lean_dec_ref(x_255);
lean_dec(x_254);
lean_dec_ref(x_253);
lean_dec_ref(x_249);
lean_dec_ref(x_248);
lean_dec_ref(x_247);
lean_dec_ref(x_244);
lean_dec_ref(x_47);
lean_dec_ref(x_24);
lean_dec_ref(x_19);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_719 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_719, 0, x_245);
return x_719;
}
}
else
{
lean_object* x_720; 
lean_dec(x_256);
lean_dec_ref(x_255);
lean_dec(x_254);
lean_dec_ref(x_253);
lean_dec_ref(x_249);
lean_dec_ref(x_248);
lean_dec_ref(x_247);
lean_dec_ref(x_244);
lean_dec_ref(x_47);
lean_dec_ref(x_24);
lean_dec_ref(x_19);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_720 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_720, 0, x_245);
return x_720;
}
}
else
{
lean_object* x_721; lean_object* x_722; lean_object* x_723; lean_object* x_724; lean_object* x_725; 
lean_dec_ref(x_248);
lean_dec_ref(x_247);
lean_dec_ref(x_245);
lean_dec_ref(x_24);
lean_dec_ref(x_7);
lean_dec(x_6);
x_721 = lean_ctor_get(x_711, 0);
lean_inc(x_721);
lean_dec(x_711);
x_722 = lean_ctor_get(x_712, 0);
lean_inc(x_722);
if (lean_is_exclusive(x_712)) {
 lean_ctor_release(x_712, 0);
 lean_ctor_release(x_712, 1);
 x_723 = x_712;
} else {
 lean_dec_ref(x_712);
 x_723 = lean_box(0);
}
lean_inc(x_251);
x_724 = l_Lean_mkRawNatLit(x_251);
lean_inc_ref(x_47);
lean_inc_ref(x_2);
lean_inc(x_1);
x_725 = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(x_1, x_2, x_47, x_724, x_253, x_254, x_255, x_256);
if (lean_obj_tag(x_725) == 0)
{
lean_object* x_726; lean_object* x_727; lean_object* x_728; lean_object* x_729; 
x_726 = lean_ctor_get(x_725, 0);
lean_inc(x_726);
lean_dec_ref(x_725);
x_727 = lean_ctor_get(x_726, 0);
lean_inc(x_727);
if (lean_is_exclusive(x_726)) {
 lean_ctor_release(x_726, 0);
 lean_ctor_release(x_726, 1);
 x_728 = x_726;
} else {
 lean_dec_ref(x_726);
 x_728 = lean_box(0);
}
lean_inc(x_256);
lean_inc_ref(x_255);
lean_inc(x_254);
lean_inc_ref(x_253);
lean_inc(x_721);
lean_inc(x_727);
lean_inc(x_251);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc(x_1);
x_729 = lp_mathlib_CancelDenoms_mkProdPrf(x_1, x_2, x_3, x_251, x_727, x_250, x_721, x_253, x_254, x_255, x_256);
if (lean_obj_tag(x_729) == 0)
{
lean_object* x_730; lean_object* x_731; lean_object* x_732; lean_object* x_733; lean_object* x_734; lean_object* x_735; lean_object* x_736; lean_object* x_737; 
x_730 = lean_ctor_get(x_729, 0);
lean_inc(x_730);
lean_dec_ref(x_729);
x_731 = lean_ctor_get(x_730, 0);
lean_inc_ref(x_731);
x_732 = lean_ctor_get(x_730, 1);
lean_inc_ref(x_732);
if (lean_is_exclusive(x_730)) {
 lean_ctor_release(x_730, 0);
 lean_ctor_release(x_730, 1);
 x_733 = x_730;
} else {
 lean_dec_ref(x_730);
 x_733 = lean_box(0);
}
x_734 = lean_nat_pow(x_251, x_252);
lean_dec(x_252);
lean_dec(x_251);
x_735 = lean_nat_div(x_4, x_734);
lean_dec(x_734);
lean_dec(x_4);
x_736 = l_Lean_mkRawNatLit(x_735);
lean_inc_ref(x_2);
lean_inc(x_1);
x_737 = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(x_1, x_2, x_47, x_736, x_253, x_254, x_255, x_256);
if (lean_obj_tag(x_737) == 0)
{
lean_object* x_738; lean_object* x_739; lean_object* x_740; lean_object* x_741; lean_object* x_742; lean_object* x_743; lean_object* x_744; lean_object* x_745; lean_object* x_746; lean_object* x_747; lean_object* x_748; lean_object* x_749; lean_object* x_750; lean_object* x_751; lean_object* x_752; lean_object* x_753; lean_object* x_754; lean_object* x_755; lean_object* x_756; lean_object* x_757; lean_object* x_758; lean_object* x_759; lean_object* x_760; lean_object* x_761; lean_object* x_762; lean_object* x_763; lean_object* x_764; lean_object* x_765; lean_object* x_766; lean_object* x_767; lean_object* x_768; lean_object* x_769; lean_object* x_770; lean_object* x_771; lean_object* x_772; lean_object* x_773; lean_object* x_774; lean_object* x_775; lean_object* x_776; lean_object* x_777; lean_object* x_778; lean_object* x_779; lean_object* x_780; lean_object* x_781; lean_object* x_782; lean_object* x_783; lean_object* x_784; lean_object* x_785; lean_object* x_786; lean_object* x_787; lean_object* x_788; 
x_738 = lean_ctor_get(x_737, 0);
lean_inc(x_738);
lean_dec_ref(x_737);
x_739 = lean_ctor_get(x_738, 0);
lean_inc(x_739);
if (lean_is_exclusive(x_738)) {
 lean_ctor_release(x_738, 0);
 lean_ctor_release(x_738, 1);
 x_740 = x_738;
} else {
 lean_dec_ref(x_738);
 x_740 = lean_box(0);
}
x_741 = lean_box(0);
lean_inc(x_739);
x_742 = l_Lean_Expr_app___override(x_249, x_739);
x_743 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__2;
lean_inc_ref(x_19);
if (lean_is_scalar(x_740)) {
 x_744 = lean_alloc_ctor(1, 2, 0);
} else {
 x_744 = x_740;
 lean_ctor_set_tag(x_744, 1);
}
lean_ctor_set(x_744, 0, x_741);
lean_ctor_set(x_744, 1, x_19);
lean_inc(x_1);
if (lean_is_scalar(x_728)) {
 x_745 = lean_alloc_ctor(1, 2, 0);
} else {
 x_745 = x_728;
 lean_ctor_set_tag(x_745, 1);
}
lean_ctor_set(x_745, 0, x_1);
lean_ctor_set(x_745, 1, x_744);
x_746 = l_Lean_Expr_const___override(x_743, x_745);
lean_inc_ref(x_2);
x_747 = l_Lean_Expr_app___override(x_746, x_2);
x_748 = lp_mathlib_CancelDenoms_mkProdPrf___closed__32;
x_749 = l_Lean_Expr_app___override(x_747, x_748);
lean_inc_ref(x_2);
x_750 = l_Lean_Expr_app___override(x_749, x_2);
x_751 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__4;
x_752 = lp_mathlib_CancelDenoms_mkProdPrf___closed__33;
if (lean_is_scalar(x_723)) {
 x_753 = lean_alloc_ctor(1, 2, 0);
} else {
 x_753 = x_723;
 lean_ctor_set_tag(x_753, 1);
}
lean_ctor_set(x_753, 0, x_1);
lean_ctor_set(x_753, 1, x_752);
x_754 = l_Lean_Expr_const___override(x_751, x_753);
lean_inc_ref(x_2);
x_755 = l_Lean_Expr_app___override(x_754, x_2);
x_756 = l_Lean_Expr_app___override(x_755, x_748);
x_757 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__7;
lean_inc_ref(x_19);
x_758 = l_Lean_Expr_const___override(x_757, x_19);
lean_inc_ref(x_2);
x_759 = l_Lean_Expr_app___override(x_758, x_2);
x_760 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__10;
lean_inc_ref(x_19);
x_761 = l_Lean_Expr_const___override(x_760, x_19);
lean_inc_ref(x_2);
x_762 = l_Lean_Expr_app___override(x_761, x_2);
x_763 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__13;
lean_inc_ref(x_19);
x_764 = l_Lean_Expr_const___override(x_763, x_19);
lean_inc_ref(x_2);
x_765 = l_Lean_Expr_app___override(x_764, x_2);
x_766 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__16;
lean_inc_ref(x_19);
x_767 = l_Lean_Expr_const___override(x_766, x_19);
lean_inc_ref(x_2);
x_768 = l_Lean_Expr_app___override(x_767, x_2);
x_769 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__18;
lean_inc_ref(x_19);
x_770 = l_Lean_Expr_const___override(x_769, x_19);
lean_inc_ref(x_2);
x_771 = l_Lean_Expr_app___override(x_770, x_2);
x_772 = lp_mathlib_CancelDenoms_mkProdPrf___closed__34;
lean_inc_ref(x_19);
x_773 = l_Lean_Expr_const___override(x_772, x_19);
lean_inc_ref(x_2);
x_774 = l_Lean_Expr_app___override(x_773, x_2);
x_775 = l_Lean_Expr_app___override(x_774, x_3);
x_776 = l_Lean_Expr_app___override(x_771, x_775);
x_777 = l_Lean_Expr_app___override(x_768, x_776);
x_778 = l_Lean_Expr_app___override(x_765, x_777);
x_779 = l_Lean_Expr_app___override(x_762, x_778);
x_780 = l_Lean_Expr_app___override(x_759, x_779);
x_781 = l_Lean_Expr_app___override(x_756, x_780);
x_782 = l_Lean_Expr_app___override(x_750, x_781);
lean_inc(x_727);
lean_inc_ref(x_782);
x_783 = l_Lean_Expr_app___override(x_782, x_727);
lean_inc(x_722);
x_784 = l_Lean_Expr_app___override(x_783, x_722);
lean_inc_ref(x_742);
x_785 = l_Lean_Expr_app___override(x_742, x_784);
x_786 = l_Lean_Expr_app___override(x_244, x_785);
lean_inc_ref(x_5);
x_787 = l_Lean_Expr_app___override(x_786, x_5);
x_788 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum(x_787, x_253, x_254, x_255, x_256);
if (lean_obj_tag(x_788) == 0)
{
lean_object* x_789; lean_object* x_790; lean_object* x_791; lean_object* x_792; lean_object* x_793; lean_object* x_794; lean_object* x_795; lean_object* x_796; lean_object* x_797; lean_object* x_798; lean_object* x_799; lean_object* x_800; lean_object* x_801; lean_object* x_802; lean_object* x_803; lean_object* x_804; lean_object* x_805; lean_object* x_806; lean_object* x_807; 
x_789 = lean_ctor_get(x_788, 0);
lean_inc(x_789);
if (lean_is_exclusive(x_788)) {
 lean_ctor_release(x_788, 0);
 x_790 = x_788;
} else {
 lean_dec_ref(x_788);
 x_790 = lean_box(0);
}
lean_inc_ref(x_731);
x_791 = l_Lean_Expr_app___override(x_782, x_731);
lean_inc(x_722);
x_792 = l_Lean_Expr_app___override(x_791, x_722);
x_793 = l_Lean_Expr_app___override(x_742, x_792);
x_794 = lp_mathlib_CancelDenoms_mkProdPrf___closed__36;
x_795 = l_Lean_Expr_const___override(x_794, x_19);
x_796 = l_Lean_Expr_app___override(x_795, x_2);
x_797 = l_Lean_Expr_app___override(x_796, x_246);
x_798 = l_Lean_Expr_app___override(x_797, x_727);
x_799 = l_Lean_Expr_app___override(x_798, x_721);
x_800 = l_Lean_Expr_app___override(x_799, x_731);
x_801 = l_Lean_Expr_app___override(x_800, x_5);
x_802 = l_Lean_Expr_app___override(x_801, x_739);
x_803 = l_Lean_Expr_app___override(x_802, x_722);
x_804 = l_Lean_Expr_app___override(x_803, x_732);
x_805 = l_Lean_Expr_app___override(x_804, x_789);
if (lean_is_scalar(x_733)) {
 x_806 = lean_alloc_ctor(0, 2, 0);
} else {
 x_806 = x_733;
}
lean_ctor_set(x_806, 0, x_793);
lean_ctor_set(x_806, 1, x_805);
if (lean_is_scalar(x_790)) {
 x_807 = lean_alloc_ctor(0, 1, 0);
} else {
 x_807 = x_790;
}
lean_ctor_set(x_807, 0, x_806);
return x_807;
}
else
{
lean_object* x_808; lean_object* x_809; lean_object* x_810; 
lean_dec_ref(x_782);
lean_dec_ref(x_742);
lean_dec(x_739);
lean_dec(x_733);
lean_dec_ref(x_732);
lean_dec_ref(x_731);
lean_dec(x_727);
lean_dec(x_722);
lean_dec(x_721);
lean_dec_ref(x_246);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
x_808 = lean_ctor_get(x_788, 0);
lean_inc(x_808);
if (lean_is_exclusive(x_788)) {
 lean_ctor_release(x_788, 0);
 x_809 = x_788;
} else {
 lean_dec_ref(x_788);
 x_809 = lean_box(0);
}
if (lean_is_scalar(x_809)) {
 x_810 = lean_alloc_ctor(1, 1, 0);
} else {
 x_810 = x_809;
}
lean_ctor_set(x_810, 0, x_808);
return x_810;
}
}
else
{
lean_object* x_811; lean_object* x_812; lean_object* x_813; 
lean_dec(x_733);
lean_dec_ref(x_732);
lean_dec_ref(x_731);
lean_dec(x_728);
lean_dec(x_727);
lean_dec(x_723);
lean_dec(x_722);
lean_dec(x_721);
lean_dec(x_256);
lean_dec_ref(x_255);
lean_dec(x_254);
lean_dec_ref(x_253);
lean_dec_ref(x_249);
lean_dec_ref(x_246);
lean_dec_ref(x_244);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_811 = lean_ctor_get(x_737, 0);
lean_inc(x_811);
if (lean_is_exclusive(x_737)) {
 lean_ctor_release(x_737, 0);
 x_812 = x_737;
} else {
 lean_dec_ref(x_737);
 x_812 = lean_box(0);
}
if (lean_is_scalar(x_812)) {
 x_813 = lean_alloc_ctor(1, 1, 0);
} else {
 x_813 = x_812;
}
lean_ctor_set(x_813, 0, x_811);
return x_813;
}
}
else
{
lean_dec(x_728);
lean_dec(x_727);
lean_dec(x_723);
lean_dec(x_722);
lean_dec(x_721);
lean_dec(x_256);
lean_dec_ref(x_255);
lean_dec(x_254);
lean_dec_ref(x_253);
lean_dec(x_252);
lean_dec(x_251);
lean_dec_ref(x_249);
lean_dec_ref(x_246);
lean_dec_ref(x_244);
lean_dec_ref(x_47);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_729;
}
}
else
{
lean_object* x_814; lean_object* x_815; lean_object* x_816; 
lean_dec(x_723);
lean_dec(x_722);
lean_dec(x_721);
lean_dec(x_256);
lean_dec_ref(x_255);
lean_dec(x_254);
lean_dec_ref(x_253);
lean_dec(x_252);
lean_dec(x_251);
lean_dec(x_250);
lean_dec_ref(x_249);
lean_dec_ref(x_246);
lean_dec_ref(x_244);
lean_dec_ref(x_47);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_814 = lean_ctor_get(x_725, 0);
lean_inc(x_814);
if (lean_is_exclusive(x_725)) {
 lean_ctor_release(x_725, 0);
 x_815 = x_725;
} else {
 lean_dec_ref(x_725);
 x_815 = lean_box(0);
}
if (lean_is_scalar(x_815)) {
 x_816 = lean_alloc_ctor(1, 1, 0);
} else {
 x_816 = x_815;
}
lean_ctor_set(x_816, 0, x_814);
return x_816;
}
}
}
}
else
{
uint8_t x_817; 
lean_dec(x_256);
lean_dec_ref(x_255);
lean_dec(x_254);
lean_dec_ref(x_253);
lean_dec(x_252);
lean_dec(x_251);
lean_dec(x_250);
lean_dec_ref(x_249);
lean_dec_ref(x_248);
lean_dec_ref(x_247);
lean_dec_ref(x_246);
lean_dec_ref(x_245);
lean_dec_ref(x_244);
lean_dec_ref(x_47);
lean_dec_ref(x_24);
lean_dec_ref(x_19);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_817 = !lean_is_exclusive(x_264);
if (x_817 == 0)
{
return x_264;
}
else
{
lean_object* x_818; lean_object* x_819; 
x_818 = lean_ctor_get(x_264, 0);
lean_inc(x_818);
lean_dec(x_264);
x_819 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_819, 0, x_818);
return x_819;
}
}
}
block_837:
{
if (lean_obj_tag(x_6) == 1)
{
lean_object* x_831; 
x_831 = lean_ctor_get(x_6, 1);
if (lean_obj_tag(x_831) == 0)
{
lean_object* x_832; 
x_832 = lean_ctor_get(x_6, 2);
lean_inc(x_832);
lean_dec_ref(x_6);
if (lean_obj_tag(x_832) == 1)
{
lean_object* x_833; 
lean_dec(x_16);
x_833 = lean_ctor_get(x_832, 0);
lean_inc(x_833);
lean_dec_ref(x_832);
x_48 = x_821;
x_49 = x_822;
x_50 = x_825;
x_51 = x_824;
x_52 = x_823;
x_53 = x_833;
x_54 = x_826;
x_55 = x_827;
x_56 = x_828;
x_57 = x_829;
x_58 = lean_box(0);
goto block_243;
}
else
{
lean_object* x_834; 
lean_dec(x_832);
lean_dec(x_829);
lean_dec_ref(x_828);
lean_dec(x_827);
lean_dec_ref(x_826);
lean_dec_ref(x_825);
lean_dec_ref(x_824);
lean_dec_ref(x_823);
lean_dec_ref(x_821);
lean_dec_ref(x_47);
lean_dec_ref(x_24);
lean_dec_ref(x_19);
lean_dec_ref(x_7);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
if (lean_is_scalar(x_16)) {
 x_834 = lean_alloc_ctor(0, 1, 0);
} else {
 x_834 = x_16;
}
lean_ctor_set(x_834, 0, x_822);
return x_834;
}
}
else
{
lean_object* x_835; 
lean_dec(x_829);
lean_dec_ref(x_828);
lean_dec(x_827);
lean_dec_ref(x_826);
lean_dec_ref(x_825);
lean_dec_ref(x_824);
lean_dec_ref(x_823);
lean_dec_ref(x_821);
lean_dec_ref(x_47);
lean_dec_ref(x_24);
lean_dec_ref(x_19);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
if (lean_is_scalar(x_16)) {
 x_835 = lean_alloc_ctor(0, 1, 0);
} else {
 x_835 = x_16;
}
lean_ctor_set(x_835, 0, x_822);
return x_835;
}
}
else
{
lean_object* x_836; 
lean_dec(x_829);
lean_dec_ref(x_828);
lean_dec(x_827);
lean_dec_ref(x_826);
lean_dec_ref(x_825);
lean_dec_ref(x_824);
lean_dec_ref(x_823);
lean_dec_ref(x_821);
lean_dec_ref(x_47);
lean_dec_ref(x_24);
lean_dec_ref(x_19);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
if (lean_is_scalar(x_16)) {
 x_836 = lean_alloc_ctor(0, 1, 0);
} else {
 x_836 = x_16;
}
lean_ctor_set(x_836, 0, x_822);
return x_836;
}
}
block_1173:
{
lean_object* x_855; 
lean_inc(x_853);
lean_inc_ref(x_852);
lean_inc(x_851);
lean_inc_ref(x_850);
x_855 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00CancelDenoms_mkProdPrf_spec__2___redArg(x_838, x_839, x_850, x_851, x_852, x_853);
if (lean_obj_tag(x_855) == 0)
{
lean_object* x_856; lean_object* x_857; lean_object* x_858; uint8_t x_859; 
x_856 = lean_ctor_get(x_855, 0);
lean_inc(x_856);
lean_dec_ref(x_855);
x_857 = lean_ctor_get(x_856, 1);
lean_inc(x_857);
x_858 = lean_ctor_get(x_857, 1);
x_859 = lean_unbox(x_858);
if (x_859 == 0)
{
lean_object* x_860; 
lean_dec(x_857);
lean_dec(x_856);
lean_dec(x_849);
lean_dec(x_848);
lean_dec(x_842);
lean_dec_ref(x_43);
lean_inc(x_853);
lean_inc_ref(x_852);
lean_inc(x_851);
lean_inc_ref(x_850);
x_860 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00CancelDenoms_mkProdPrf_spec__2___redArg(x_840, x_839, x_850, x_851, x_852, x_853);
if (lean_obj_tag(x_860) == 0)
{
lean_object* x_861; lean_object* x_862; uint8_t x_863; 
x_861 = lean_ctor_get(x_860, 0);
lean_inc(x_861);
lean_dec_ref(x_860);
x_862 = lean_ctor_get(x_861, 1);
x_863 = lean_unbox(x_862);
if (x_863 == 0)
{
lean_dec(x_861);
lean_dec_ref(x_44);
if (lean_obj_tag(x_6) == 1)
{
lean_object* x_864; 
x_864 = lean_ctor_get(x_6, 1);
if (lean_obj_tag(x_864) == 1)
{
lean_object* x_865; 
x_865 = lean_ctor_get(x_6, 2);
if (lean_obj_tag(x_865) == 1)
{
lean_object* x_866; 
x_866 = lean_ctor_get(x_865, 1);
if (lean_obj_tag(x_866) == 0)
{
lean_object* x_867; 
x_867 = lean_ctor_get(x_865, 2);
if (lean_obj_tag(x_867) == 0)
{
lean_object* x_868; lean_object* x_869; 
lean_dec(x_16);
x_868 = lean_ctor_get(x_864, 0);
x_869 = lean_ctor_get(x_865, 0);
lean_inc(x_869);
lean_inc(x_868);
lean_inc_ref(x_864);
x_244 = x_841;
x_245 = x_843;
x_246 = x_844;
x_247 = x_847;
x_248 = x_846;
x_249 = x_845;
x_250 = x_864;
x_251 = x_868;
x_252 = x_869;
x_253 = x_850;
x_254 = x_851;
x_255 = x_852;
x_256 = x_853;
x_257 = lean_box(0);
goto block_820;
}
else
{
lean_dec_ref(x_844);
x_821 = x_841;
x_822 = x_843;
x_823 = x_847;
x_824 = x_846;
x_825 = x_845;
x_826 = x_850;
x_827 = x_851;
x_828 = x_852;
x_829 = x_853;
x_830 = lean_box(0);
goto block_837;
}
}
else
{
lean_dec_ref(x_844);
x_821 = x_841;
x_822 = x_843;
x_823 = x_847;
x_824 = x_846;
x_825 = x_845;
x_826 = x_850;
x_827 = x_851;
x_828 = x_852;
x_829 = x_853;
x_830 = lean_box(0);
goto block_837;
}
}
else
{
lean_dec_ref(x_844);
x_821 = x_841;
x_822 = x_843;
x_823 = x_847;
x_824 = x_846;
x_825 = x_845;
x_826 = x_850;
x_827 = x_851;
x_828 = x_852;
x_829 = x_853;
x_830 = lean_box(0);
goto block_837;
}
}
else
{
lean_dec_ref(x_844);
x_821 = x_841;
x_822 = x_843;
x_823 = x_847;
x_824 = x_846;
x_825 = x_845;
x_826 = x_850;
x_827 = x_851;
x_828 = x_852;
x_829 = x_853;
x_830 = lean_box(0);
goto block_837;
}
}
else
{
lean_dec_ref(x_844);
x_821 = x_841;
x_822 = x_843;
x_823 = x_847;
x_824 = x_846;
x_825 = x_845;
x_826 = x_850;
x_827 = x_851;
x_828 = x_852;
x_829 = x_853;
x_830 = lean_box(0);
goto block_837;
}
}
else
{
lean_object* x_870; lean_object* x_871; 
lean_dec_ref(x_847);
lean_dec_ref(x_846);
lean_dec_ref(x_845);
lean_dec_ref(x_844);
lean_dec_ref(x_843);
lean_dec_ref(x_841);
lean_dec_ref(x_47);
lean_dec_ref(x_24);
lean_dec(x_16);
lean_dec_ref(x_7);
x_870 = lean_ctor_get(x_861, 0);
lean_inc(x_870);
lean_dec(x_861);
lean_inc(x_870);
lean_inc_ref(x_5);
lean_inc_ref(x_2);
x_871 = lp_mathlib_CancelDenoms_mkProdPrf(x_1, x_2, x_3, x_4, x_5, x_6, x_870, x_850, x_851, x_852, x_853);
if (lean_obj_tag(x_871) == 0)
{
uint8_t x_872; 
x_872 = !lean_is_exclusive(x_871);
if (x_872 == 0)
{
lean_object* x_873; uint8_t x_874; 
x_873 = lean_ctor_get(x_871, 0);
x_874 = !lean_is_exclusive(x_873);
if (x_874 == 0)
{
lean_object* x_875; lean_object* x_876; lean_object* x_877; lean_object* x_878; lean_object* x_879; lean_object* x_880; lean_object* x_881; lean_object* x_882; lean_object* x_883; lean_object* x_884; lean_object* x_885; lean_object* x_886; lean_object* x_887; lean_object* x_888; lean_object* x_889; lean_object* x_890; lean_object* x_891; lean_object* x_892; lean_object* x_893; lean_object* x_894; lean_object* x_895; lean_object* x_896; lean_object* x_897; lean_object* x_898; lean_object* x_899; lean_object* x_900; lean_object* x_901; lean_object* x_902; lean_object* x_903; lean_object* x_904; lean_object* x_905; lean_object* x_906; lean_object* x_907; lean_object* x_908; lean_object* x_909; lean_object* x_910; lean_object* x_911; lean_object* x_912; lean_object* x_913; 
x_875 = lean_ctor_get(x_873, 0);
x_876 = lean_ctor_get(x_873, 1);
x_877 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__0;
lean_inc_ref(x_19);
x_878 = l_Lean_Expr_const___override(x_877, x_19);
lean_inc_ref(x_2);
x_879 = l_Lean_Expr_app___override(x_878, x_2);
x_880 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__3;
lean_inc_ref(x_19);
x_881 = l_Lean_Expr_const___override(x_880, x_19);
lean_inc_ref(x_2);
x_882 = l_Lean_Expr_app___override(x_881, x_2);
x_883 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__6;
lean_inc_ref(x_19);
x_884 = l_Lean_Expr_const___override(x_883, x_19);
lean_inc_ref(x_2);
x_885 = l_Lean_Expr_app___override(x_884, x_2);
x_886 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__9;
lean_inc_ref(x_19);
x_887 = l_Lean_Expr_const___override(x_886, x_19);
lean_inc_ref(x_2);
x_888 = l_Lean_Expr_app___override(x_887, x_2);
x_889 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__12;
lean_inc_ref(x_19);
x_890 = l_Lean_Expr_const___override(x_889, x_19);
lean_inc_ref(x_2);
x_891 = l_Lean_Expr_app___override(x_890, x_2);
x_892 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__15;
lean_inc_ref(x_19);
x_893 = l_Lean_Expr_const___override(x_892, x_19);
lean_inc_ref(x_2);
x_894 = l_Lean_Expr_app___override(x_893, x_2);
x_895 = lp_mathlib_CancelDenoms_mkProdPrf___closed__37;
lean_inc_ref(x_19);
x_896 = l_Lean_Expr_const___override(x_895, x_19);
lean_inc_ref(x_2);
x_897 = l_Lean_Expr_app___override(x_896, x_2);
lean_inc_ref(x_44);
x_898 = l_Lean_Expr_app___override(x_897, x_44);
x_899 = l_Lean_Expr_app___override(x_894, x_898);
x_900 = l_Lean_Expr_app___override(x_891, x_899);
x_901 = l_Lean_Expr_app___override(x_888, x_900);
x_902 = l_Lean_Expr_app___override(x_885, x_901);
x_903 = l_Lean_Expr_app___override(x_882, x_902);
x_904 = l_Lean_Expr_app___override(x_879, x_903);
lean_inc_ref(x_875);
x_905 = l_Lean_Expr_app___override(x_904, x_875);
x_906 = lp_mathlib_CancelDenoms_mkProdPrf___closed__39;
x_907 = l_Lean_Expr_const___override(x_906, x_19);
x_908 = l_Lean_Expr_app___override(x_907, x_2);
x_909 = l_Lean_Expr_app___override(x_908, x_44);
x_910 = l_Lean_Expr_app___override(x_909, x_5);
x_911 = l_Lean_Expr_app___override(x_910, x_870);
x_912 = l_Lean_Expr_app___override(x_911, x_875);
x_913 = l_Lean_Expr_app___override(x_912, x_876);
lean_ctor_set(x_873, 1, x_913);
lean_ctor_set(x_873, 0, x_905);
return x_871;
}
else
{
lean_object* x_914; lean_object* x_915; lean_object* x_916; lean_object* x_917; lean_object* x_918; lean_object* x_919; lean_object* x_920; lean_object* x_921; lean_object* x_922; lean_object* x_923; lean_object* x_924; lean_object* x_925; lean_object* x_926; lean_object* x_927; lean_object* x_928; lean_object* x_929; lean_object* x_930; lean_object* x_931; lean_object* x_932; lean_object* x_933; lean_object* x_934; lean_object* x_935; lean_object* x_936; lean_object* x_937; lean_object* x_938; lean_object* x_939; lean_object* x_940; lean_object* x_941; lean_object* x_942; lean_object* x_943; lean_object* x_944; lean_object* x_945; lean_object* x_946; lean_object* x_947; lean_object* x_948; lean_object* x_949; lean_object* x_950; lean_object* x_951; lean_object* x_952; lean_object* x_953; 
x_914 = lean_ctor_get(x_873, 0);
x_915 = lean_ctor_get(x_873, 1);
lean_inc(x_915);
lean_inc(x_914);
lean_dec(x_873);
x_916 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__0;
lean_inc_ref(x_19);
x_917 = l_Lean_Expr_const___override(x_916, x_19);
lean_inc_ref(x_2);
x_918 = l_Lean_Expr_app___override(x_917, x_2);
x_919 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__3;
lean_inc_ref(x_19);
x_920 = l_Lean_Expr_const___override(x_919, x_19);
lean_inc_ref(x_2);
x_921 = l_Lean_Expr_app___override(x_920, x_2);
x_922 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__6;
lean_inc_ref(x_19);
x_923 = l_Lean_Expr_const___override(x_922, x_19);
lean_inc_ref(x_2);
x_924 = l_Lean_Expr_app___override(x_923, x_2);
x_925 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__9;
lean_inc_ref(x_19);
x_926 = l_Lean_Expr_const___override(x_925, x_19);
lean_inc_ref(x_2);
x_927 = l_Lean_Expr_app___override(x_926, x_2);
x_928 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__12;
lean_inc_ref(x_19);
x_929 = l_Lean_Expr_const___override(x_928, x_19);
lean_inc_ref(x_2);
x_930 = l_Lean_Expr_app___override(x_929, x_2);
x_931 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__15;
lean_inc_ref(x_19);
x_932 = l_Lean_Expr_const___override(x_931, x_19);
lean_inc_ref(x_2);
x_933 = l_Lean_Expr_app___override(x_932, x_2);
x_934 = lp_mathlib_CancelDenoms_mkProdPrf___closed__37;
lean_inc_ref(x_19);
x_935 = l_Lean_Expr_const___override(x_934, x_19);
lean_inc_ref(x_2);
x_936 = l_Lean_Expr_app___override(x_935, x_2);
lean_inc_ref(x_44);
x_937 = l_Lean_Expr_app___override(x_936, x_44);
x_938 = l_Lean_Expr_app___override(x_933, x_937);
x_939 = l_Lean_Expr_app___override(x_930, x_938);
x_940 = l_Lean_Expr_app___override(x_927, x_939);
x_941 = l_Lean_Expr_app___override(x_924, x_940);
x_942 = l_Lean_Expr_app___override(x_921, x_941);
x_943 = l_Lean_Expr_app___override(x_918, x_942);
lean_inc_ref(x_914);
x_944 = l_Lean_Expr_app___override(x_943, x_914);
x_945 = lp_mathlib_CancelDenoms_mkProdPrf___closed__39;
x_946 = l_Lean_Expr_const___override(x_945, x_19);
x_947 = l_Lean_Expr_app___override(x_946, x_2);
x_948 = l_Lean_Expr_app___override(x_947, x_44);
x_949 = l_Lean_Expr_app___override(x_948, x_5);
x_950 = l_Lean_Expr_app___override(x_949, x_870);
x_951 = l_Lean_Expr_app___override(x_950, x_914);
x_952 = l_Lean_Expr_app___override(x_951, x_915);
x_953 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_953, 0, x_944);
lean_ctor_set(x_953, 1, x_952);
lean_ctor_set(x_871, 0, x_953);
return x_871;
}
}
else
{
lean_object* x_954; lean_object* x_955; lean_object* x_956; lean_object* x_957; lean_object* x_958; lean_object* x_959; lean_object* x_960; lean_object* x_961; lean_object* x_962; lean_object* x_963; lean_object* x_964; lean_object* x_965; lean_object* x_966; lean_object* x_967; lean_object* x_968; lean_object* x_969; lean_object* x_970; lean_object* x_971; lean_object* x_972; lean_object* x_973; lean_object* x_974; lean_object* x_975; lean_object* x_976; lean_object* x_977; lean_object* x_978; lean_object* x_979; lean_object* x_980; lean_object* x_981; lean_object* x_982; lean_object* x_983; lean_object* x_984; lean_object* x_985; lean_object* x_986; lean_object* x_987; lean_object* x_988; lean_object* x_989; lean_object* x_990; lean_object* x_991; lean_object* x_992; lean_object* x_993; lean_object* x_994; lean_object* x_995; lean_object* x_996; 
x_954 = lean_ctor_get(x_871, 0);
lean_inc(x_954);
lean_dec(x_871);
x_955 = lean_ctor_get(x_954, 0);
lean_inc_ref(x_955);
x_956 = lean_ctor_get(x_954, 1);
lean_inc_ref(x_956);
if (lean_is_exclusive(x_954)) {
 lean_ctor_release(x_954, 0);
 lean_ctor_release(x_954, 1);
 x_957 = x_954;
} else {
 lean_dec_ref(x_954);
 x_957 = lean_box(0);
}
x_958 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__0;
lean_inc_ref(x_19);
x_959 = l_Lean_Expr_const___override(x_958, x_19);
lean_inc_ref(x_2);
x_960 = l_Lean_Expr_app___override(x_959, x_2);
x_961 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__3;
lean_inc_ref(x_19);
x_962 = l_Lean_Expr_const___override(x_961, x_19);
lean_inc_ref(x_2);
x_963 = l_Lean_Expr_app___override(x_962, x_2);
x_964 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__6;
lean_inc_ref(x_19);
x_965 = l_Lean_Expr_const___override(x_964, x_19);
lean_inc_ref(x_2);
x_966 = l_Lean_Expr_app___override(x_965, x_2);
x_967 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__9;
lean_inc_ref(x_19);
x_968 = l_Lean_Expr_const___override(x_967, x_19);
lean_inc_ref(x_2);
x_969 = l_Lean_Expr_app___override(x_968, x_2);
x_970 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__12;
lean_inc_ref(x_19);
x_971 = l_Lean_Expr_const___override(x_970, x_19);
lean_inc_ref(x_2);
x_972 = l_Lean_Expr_app___override(x_971, x_2);
x_973 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__15;
lean_inc_ref(x_19);
x_974 = l_Lean_Expr_const___override(x_973, x_19);
lean_inc_ref(x_2);
x_975 = l_Lean_Expr_app___override(x_974, x_2);
x_976 = lp_mathlib_CancelDenoms_mkProdPrf___closed__37;
lean_inc_ref(x_19);
x_977 = l_Lean_Expr_const___override(x_976, x_19);
lean_inc_ref(x_2);
x_978 = l_Lean_Expr_app___override(x_977, x_2);
lean_inc_ref(x_44);
x_979 = l_Lean_Expr_app___override(x_978, x_44);
x_980 = l_Lean_Expr_app___override(x_975, x_979);
x_981 = l_Lean_Expr_app___override(x_972, x_980);
x_982 = l_Lean_Expr_app___override(x_969, x_981);
x_983 = l_Lean_Expr_app___override(x_966, x_982);
x_984 = l_Lean_Expr_app___override(x_963, x_983);
x_985 = l_Lean_Expr_app___override(x_960, x_984);
lean_inc_ref(x_955);
x_986 = l_Lean_Expr_app___override(x_985, x_955);
x_987 = lp_mathlib_CancelDenoms_mkProdPrf___closed__39;
x_988 = l_Lean_Expr_const___override(x_987, x_19);
x_989 = l_Lean_Expr_app___override(x_988, x_2);
x_990 = l_Lean_Expr_app___override(x_989, x_44);
x_991 = l_Lean_Expr_app___override(x_990, x_5);
x_992 = l_Lean_Expr_app___override(x_991, x_870);
x_993 = l_Lean_Expr_app___override(x_992, x_955);
x_994 = l_Lean_Expr_app___override(x_993, x_956);
if (lean_is_scalar(x_957)) {
 x_995 = lean_alloc_ctor(0, 2, 0);
} else {
 x_995 = x_957;
}
lean_ctor_set(x_995, 0, x_986);
lean_ctor_set(x_995, 1, x_994);
x_996 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_996, 0, x_995);
return x_996;
}
}
else
{
lean_dec(x_870);
lean_dec_ref(x_44);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
return x_871;
}
}
}
else
{
uint8_t x_997; 
lean_dec(x_853);
lean_dec_ref(x_852);
lean_dec(x_851);
lean_dec_ref(x_850);
lean_dec_ref(x_847);
lean_dec_ref(x_846);
lean_dec_ref(x_845);
lean_dec_ref(x_844);
lean_dec_ref(x_843);
lean_dec_ref(x_841);
lean_dec_ref(x_47);
lean_dec_ref(x_44);
lean_dec_ref(x_24);
lean_dec_ref(x_19);
lean_dec(x_16);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_997 = !lean_is_exclusive(x_860);
if (x_997 == 0)
{
return x_860;
}
else
{
lean_object* x_998; lean_object* x_999; 
x_998 = lean_ctor_get(x_860, 0);
lean_inc(x_998);
lean_dec(x_860);
x_999 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_999, 0, x_998);
return x_999;
}
}
}
else
{
lean_object* x_1000; lean_object* x_1001; lean_object* x_1002; lean_object* x_1003; 
lean_dec_ref(x_847);
lean_dec_ref(x_846);
lean_dec_ref(x_844);
lean_dec_ref(x_843);
lean_dec_ref(x_840);
lean_dec_ref(x_44);
lean_dec_ref(x_24);
lean_dec(x_16);
lean_dec_ref(x_7);
lean_dec(x_6);
x_1000 = lean_ctor_get(x_856, 0);
lean_inc(x_1000);
lean_dec(x_856);
x_1001 = lean_ctor_get(x_857, 0);
lean_inc(x_1001);
lean_dec(x_857);
lean_inc(x_849);
x_1002 = l_Lean_mkRawNatLit(x_849);
lean_inc_ref(x_47);
lean_inc_ref(x_2);
lean_inc(x_1);
x_1003 = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(x_1, x_2, x_47, x_1002, x_850, x_851, x_852, x_853);
if (lean_obj_tag(x_1003) == 0)
{
lean_object* x_1004; lean_object* x_1005; lean_object* x_1006; lean_object* x_1007; 
x_1004 = lean_ctor_get(x_1003, 0);
lean_inc(x_1004);
lean_dec_ref(x_1003);
x_1005 = lean_nat_div(x_4, x_849);
lean_dec(x_849);
lean_dec(x_4);
lean_inc(x_1005);
x_1006 = l_Lean_mkRawNatLit(x_1005);
lean_inc_ref(x_47);
lean_inc_ref(x_2);
lean_inc(x_1);
x_1007 = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(x_1, x_2, x_47, x_1006, x_850, x_851, x_852, x_853);
if (lean_obj_tag(x_1007) == 0)
{
lean_object* x_1008; lean_object* x_1009; lean_object* x_1010; 
x_1008 = lean_ctor_get(x_1007, 0);
lean_inc(x_1008);
lean_dec_ref(x_1007);
x_1009 = lean_ctor_get(x_1008, 0);
lean_inc(x_1009);
lean_dec(x_1008);
lean_inc(x_853);
lean_inc_ref(x_852);
lean_inc(x_851);
lean_inc_ref(x_850);
lean_inc(x_1000);
lean_inc(x_1009);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
x_1010 = lp_mathlib_CancelDenoms_mkProdPrf(x_1, x_2, x_3, x_1005, x_1009, x_848, x_1000, x_850, x_851, x_852, x_853);
if (lean_obj_tag(x_1010) == 0)
{
lean_object* x_1011; uint8_t x_1012; 
x_1011 = lean_ctor_get(x_1010, 0);
lean_inc(x_1011);
lean_dec_ref(x_1010);
x_1012 = !lean_is_exclusive(x_1011);
if (x_1012 == 0)
{
lean_object* x_1013; lean_object* x_1014; lean_object* x_1015; lean_object* x_1016; lean_object* x_1017; lean_object* x_1018; lean_object* x_1019; lean_object* x_1020; lean_object* x_1021; lean_object* x_1022; lean_object* x_1023; lean_object* x_1024; lean_object* x_1025; lean_object* x_1026; lean_object* x_1027; lean_object* x_1028; lean_object* x_1029; lean_object* x_1030; lean_object* x_1031; lean_object* x_1032; lean_object* x_1033; lean_object* x_1034; lean_object* x_1035; lean_object* x_1036; lean_object* x_1037; lean_object* x_1038; lean_object* x_1039; lean_object* x_1040; lean_object* x_1041; lean_object* x_1042; lean_object* x_1043; lean_object* x_1044; lean_object* x_1045; lean_object* x_1046; lean_object* x_1047; lean_object* x_1048; lean_object* x_1049; lean_object* x_1050; lean_object* x_1051; lean_object* x_1052; 
x_1013 = lean_ctor_get(x_1011, 0);
x_1014 = lean_ctor_get(x_1011, 1);
x_1015 = lean_ctor_get(x_1004, 0);
lean_inc(x_1015);
lean_dec(x_1004);
x_1016 = lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__0;
x_1017 = l_Lean_Expr_const___override(x_1016, x_842);
lean_inc_ref(x_2);
x_1018 = l_Lean_Expr_app___override(x_1017, x_2);
lean_inc_ref(x_2);
x_1019 = l_Lean_Expr_app___override(x_1018, x_2);
lean_inc_ref(x_2);
x_1020 = l_Lean_Expr_app___override(x_1019, x_2);
x_1021 = lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__2;
lean_inc_ref(x_19);
x_1022 = l_Lean_Expr_const___override(x_1021, x_19);
lean_inc_ref(x_2);
x_1023 = l_Lean_Expr_app___override(x_1022, x_2);
x_1024 = lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__5;
lean_inc_ref(x_19);
x_1025 = l_Lean_Expr_const___override(x_1024, x_19);
lean_inc_ref(x_2);
x_1026 = l_Lean_Expr_app___override(x_1025, x_2);
x_1027 = lp_mathlib_CancelDenoms_mkProdPrf___closed__40;
lean_inc_ref(x_19);
x_1028 = l_Lean_Expr_const___override(x_1027, x_19);
lean_inc_ref(x_2);
x_1029 = l_Lean_Expr_app___override(x_1028, x_2);
x_1030 = l_Lean_Expr_app___override(x_1029, x_43);
x_1031 = l_Lean_Expr_app___override(x_1026, x_1030);
x_1032 = l_Lean_Expr_app___override(x_1023, x_1031);
x_1033 = l_Lean_Expr_app___override(x_1020, x_1032);
lean_inc(x_1015);
x_1034 = l_Lean_Expr_app___override(x_1033, x_1015);
lean_inc(x_1001);
x_1035 = l_Lean_Expr_app___override(x_1034, x_1001);
lean_inc_ref(x_841);
x_1036 = l_Lean_Expr_app___override(x_841, x_1035);
x_1037 = lp_mathlib_CancelDenoms_mkProdPrf___closed__20;
lean_inc_ref(x_19);
x_1038 = l_Lean_Expr_const___override(x_1037, x_19);
lean_inc_ref(x_2);
x_1039 = l_Lean_Expr_app___override(x_1038, x_2);
x_1040 = lp_mathlib_CancelDenoms_mkProdPrf___closed__42;
x_1041 = l_Lean_Expr_app___override(x_1039, x_1040);
x_1042 = lp_mathlib_CancelDenoms_mkProdPrf___closed__45;
lean_inc_ref(x_19);
x_1043 = l_Lean_Expr_const___override(x_1042, x_19);
lean_inc_ref(x_2);
x_1044 = l_Lean_Expr_app___override(x_1043, x_2);
x_1045 = lp_mathlib_CancelDenoms_mkProdPrf___closed__47;
lean_inc_ref(x_19);
x_1046 = l_Lean_Expr_const___override(x_1045, x_19);
lean_inc_ref(x_2);
x_1047 = l_Lean_Expr_app___override(x_1046, x_2);
x_1048 = l_Lean_Expr_app___override(x_1047, x_47);
x_1049 = l_Lean_Expr_app___override(x_1044, x_1048);
x_1050 = l_Lean_Expr_app___override(x_1041, x_1049);
x_1051 = l_Lean_Expr_app___override(x_1036, x_1050);
lean_inc(x_853);
lean_inc_ref(x_852);
lean_inc(x_851);
lean_inc_ref(x_850);
x_1052 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum(x_1051, x_850, x_851, x_852, x_853);
if (lean_obj_tag(x_1052) == 0)
{
lean_object* x_1053; lean_object* x_1054; lean_object* x_1055; lean_object* x_1056; lean_object* x_1057; lean_object* x_1058; 
x_1053 = lean_ctor_get(x_1052, 0);
lean_inc(x_1053);
lean_dec_ref(x_1052);
lean_inc(x_1009);
x_1054 = l_Lean_Expr_app___override(x_845, x_1009);
lean_inc(x_1015);
x_1055 = l_Lean_Expr_app___override(x_1054, x_1015);
x_1056 = l_Lean_Expr_app___override(x_841, x_1055);
lean_inc_ref(x_5);
x_1057 = l_Lean_Expr_app___override(x_1056, x_5);
x_1058 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum(x_1057, x_850, x_851, x_852, x_853);
if (lean_obj_tag(x_1058) == 0)
{
uint8_t x_1059; 
x_1059 = !lean_is_exclusive(x_1058);
if (x_1059 == 0)
{
lean_object* x_1060; lean_object* x_1061; lean_object* x_1062; lean_object* x_1063; lean_object* x_1064; lean_object* x_1065; lean_object* x_1066; lean_object* x_1067; lean_object* x_1068; lean_object* x_1069; lean_object* x_1070; lean_object* x_1071; lean_object* x_1072; lean_object* x_1073; 
x_1060 = lean_ctor_get(x_1058, 0);
x_1061 = lp_mathlib_CancelDenoms_mkProdPrf___closed__49;
x_1062 = l_Lean_Expr_const___override(x_1061, x_19);
x_1063 = l_Lean_Expr_app___override(x_1062, x_2);
x_1064 = l_Lean_Expr_app___override(x_1063, x_3);
x_1065 = l_Lean_Expr_app___override(x_1064, x_1009);
x_1066 = l_Lean_Expr_app___override(x_1065, x_1015);
x_1067 = l_Lean_Expr_app___override(x_1066, x_5);
x_1068 = l_Lean_Expr_app___override(x_1067, x_1000);
x_1069 = l_Lean_Expr_app___override(x_1068, x_1001);
lean_inc_ref(x_1013);
x_1070 = l_Lean_Expr_app___override(x_1069, x_1013);
x_1071 = l_Lean_Expr_app___override(x_1070, x_1014);
x_1072 = l_Lean_Expr_app___override(x_1071, x_1053);
x_1073 = l_Lean_Expr_app___override(x_1072, x_1060);
lean_ctor_set(x_1011, 1, x_1073);
lean_ctor_set(x_1058, 0, x_1011);
return x_1058;
}
else
{
lean_object* x_1074; lean_object* x_1075; lean_object* x_1076; lean_object* x_1077; lean_object* x_1078; lean_object* x_1079; lean_object* x_1080; lean_object* x_1081; lean_object* x_1082; lean_object* x_1083; lean_object* x_1084; lean_object* x_1085; lean_object* x_1086; lean_object* x_1087; lean_object* x_1088; 
x_1074 = lean_ctor_get(x_1058, 0);
lean_inc(x_1074);
lean_dec(x_1058);
x_1075 = lp_mathlib_CancelDenoms_mkProdPrf___closed__49;
x_1076 = l_Lean_Expr_const___override(x_1075, x_19);
x_1077 = l_Lean_Expr_app___override(x_1076, x_2);
x_1078 = l_Lean_Expr_app___override(x_1077, x_3);
x_1079 = l_Lean_Expr_app___override(x_1078, x_1009);
x_1080 = l_Lean_Expr_app___override(x_1079, x_1015);
x_1081 = l_Lean_Expr_app___override(x_1080, x_5);
x_1082 = l_Lean_Expr_app___override(x_1081, x_1000);
x_1083 = l_Lean_Expr_app___override(x_1082, x_1001);
lean_inc_ref(x_1013);
x_1084 = l_Lean_Expr_app___override(x_1083, x_1013);
x_1085 = l_Lean_Expr_app___override(x_1084, x_1014);
x_1086 = l_Lean_Expr_app___override(x_1085, x_1053);
x_1087 = l_Lean_Expr_app___override(x_1086, x_1074);
lean_ctor_set(x_1011, 1, x_1087);
x_1088 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_1088, 0, x_1011);
return x_1088;
}
}
else
{
uint8_t x_1089; 
lean_dec(x_1053);
lean_dec(x_1015);
lean_free_object(x_1011);
lean_dec_ref(x_1014);
lean_dec_ref(x_1013);
lean_dec(x_1009);
lean_dec(x_1001);
lean_dec(x_1000);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
x_1089 = !lean_is_exclusive(x_1058);
if (x_1089 == 0)
{
return x_1058;
}
else
{
lean_object* x_1090; lean_object* x_1091; 
x_1090 = lean_ctor_get(x_1058, 0);
lean_inc(x_1090);
lean_dec(x_1058);
x_1091 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_1091, 0, x_1090);
return x_1091;
}
}
}
else
{
uint8_t x_1092; 
lean_dec(x_1015);
lean_free_object(x_1011);
lean_dec_ref(x_1014);
lean_dec_ref(x_1013);
lean_dec(x_1009);
lean_dec(x_1001);
lean_dec(x_1000);
lean_dec(x_853);
lean_dec_ref(x_852);
lean_dec(x_851);
lean_dec_ref(x_850);
lean_dec_ref(x_845);
lean_dec_ref(x_841);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
x_1092 = !lean_is_exclusive(x_1052);
if (x_1092 == 0)
{
return x_1052;
}
else
{
lean_object* x_1093; lean_object* x_1094; 
x_1093 = lean_ctor_get(x_1052, 0);
lean_inc(x_1093);
lean_dec(x_1052);
x_1094 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_1094, 0, x_1093);
return x_1094;
}
}
}
else
{
lean_object* x_1095; lean_object* x_1096; lean_object* x_1097; lean_object* x_1098; lean_object* x_1099; lean_object* x_1100; lean_object* x_1101; lean_object* x_1102; lean_object* x_1103; lean_object* x_1104; lean_object* x_1105; lean_object* x_1106; lean_object* x_1107; lean_object* x_1108; lean_object* x_1109; lean_object* x_1110; lean_object* x_1111; lean_object* x_1112; lean_object* x_1113; lean_object* x_1114; lean_object* x_1115; lean_object* x_1116; lean_object* x_1117; lean_object* x_1118; lean_object* x_1119; lean_object* x_1120; lean_object* x_1121; lean_object* x_1122; lean_object* x_1123; lean_object* x_1124; lean_object* x_1125; lean_object* x_1126; lean_object* x_1127; lean_object* x_1128; lean_object* x_1129; lean_object* x_1130; lean_object* x_1131; lean_object* x_1132; lean_object* x_1133; lean_object* x_1134; 
x_1095 = lean_ctor_get(x_1011, 0);
x_1096 = lean_ctor_get(x_1011, 1);
lean_inc(x_1096);
lean_inc(x_1095);
lean_dec(x_1011);
x_1097 = lean_ctor_get(x_1004, 0);
lean_inc(x_1097);
lean_dec(x_1004);
x_1098 = lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__0;
x_1099 = l_Lean_Expr_const___override(x_1098, x_842);
lean_inc_ref(x_2);
x_1100 = l_Lean_Expr_app___override(x_1099, x_2);
lean_inc_ref(x_2);
x_1101 = l_Lean_Expr_app___override(x_1100, x_2);
lean_inc_ref(x_2);
x_1102 = l_Lean_Expr_app___override(x_1101, x_2);
x_1103 = lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__2;
lean_inc_ref(x_19);
x_1104 = l_Lean_Expr_const___override(x_1103, x_19);
lean_inc_ref(x_2);
x_1105 = l_Lean_Expr_app___override(x_1104, x_2);
x_1106 = lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__5;
lean_inc_ref(x_19);
x_1107 = l_Lean_Expr_const___override(x_1106, x_19);
lean_inc_ref(x_2);
x_1108 = l_Lean_Expr_app___override(x_1107, x_2);
x_1109 = lp_mathlib_CancelDenoms_mkProdPrf___closed__40;
lean_inc_ref(x_19);
x_1110 = l_Lean_Expr_const___override(x_1109, x_19);
lean_inc_ref(x_2);
x_1111 = l_Lean_Expr_app___override(x_1110, x_2);
x_1112 = l_Lean_Expr_app___override(x_1111, x_43);
x_1113 = l_Lean_Expr_app___override(x_1108, x_1112);
x_1114 = l_Lean_Expr_app___override(x_1105, x_1113);
x_1115 = l_Lean_Expr_app___override(x_1102, x_1114);
lean_inc(x_1097);
x_1116 = l_Lean_Expr_app___override(x_1115, x_1097);
lean_inc(x_1001);
x_1117 = l_Lean_Expr_app___override(x_1116, x_1001);
lean_inc_ref(x_841);
x_1118 = l_Lean_Expr_app___override(x_841, x_1117);
x_1119 = lp_mathlib_CancelDenoms_mkProdPrf___closed__20;
lean_inc_ref(x_19);
x_1120 = l_Lean_Expr_const___override(x_1119, x_19);
lean_inc_ref(x_2);
x_1121 = l_Lean_Expr_app___override(x_1120, x_2);
x_1122 = lp_mathlib_CancelDenoms_mkProdPrf___closed__42;
x_1123 = l_Lean_Expr_app___override(x_1121, x_1122);
x_1124 = lp_mathlib_CancelDenoms_mkProdPrf___closed__45;
lean_inc_ref(x_19);
x_1125 = l_Lean_Expr_const___override(x_1124, x_19);
lean_inc_ref(x_2);
x_1126 = l_Lean_Expr_app___override(x_1125, x_2);
x_1127 = lp_mathlib_CancelDenoms_mkProdPrf___closed__47;
lean_inc_ref(x_19);
x_1128 = l_Lean_Expr_const___override(x_1127, x_19);
lean_inc_ref(x_2);
x_1129 = l_Lean_Expr_app___override(x_1128, x_2);
x_1130 = l_Lean_Expr_app___override(x_1129, x_47);
x_1131 = l_Lean_Expr_app___override(x_1126, x_1130);
x_1132 = l_Lean_Expr_app___override(x_1123, x_1131);
x_1133 = l_Lean_Expr_app___override(x_1118, x_1132);
lean_inc(x_853);
lean_inc_ref(x_852);
lean_inc(x_851);
lean_inc_ref(x_850);
x_1134 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum(x_1133, x_850, x_851, x_852, x_853);
if (lean_obj_tag(x_1134) == 0)
{
lean_object* x_1135; lean_object* x_1136; lean_object* x_1137; lean_object* x_1138; lean_object* x_1139; lean_object* x_1140; 
x_1135 = lean_ctor_get(x_1134, 0);
lean_inc(x_1135);
lean_dec_ref(x_1134);
lean_inc(x_1009);
x_1136 = l_Lean_Expr_app___override(x_845, x_1009);
lean_inc(x_1097);
x_1137 = l_Lean_Expr_app___override(x_1136, x_1097);
x_1138 = l_Lean_Expr_app___override(x_841, x_1137);
lean_inc_ref(x_5);
x_1139 = l_Lean_Expr_app___override(x_1138, x_5);
x_1140 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum(x_1139, x_850, x_851, x_852, x_853);
if (lean_obj_tag(x_1140) == 0)
{
lean_object* x_1141; lean_object* x_1142; lean_object* x_1143; lean_object* x_1144; lean_object* x_1145; lean_object* x_1146; lean_object* x_1147; lean_object* x_1148; lean_object* x_1149; lean_object* x_1150; lean_object* x_1151; lean_object* x_1152; lean_object* x_1153; lean_object* x_1154; lean_object* x_1155; lean_object* x_1156; lean_object* x_1157; 
x_1141 = lean_ctor_get(x_1140, 0);
lean_inc(x_1141);
if (lean_is_exclusive(x_1140)) {
 lean_ctor_release(x_1140, 0);
 x_1142 = x_1140;
} else {
 lean_dec_ref(x_1140);
 x_1142 = lean_box(0);
}
x_1143 = lp_mathlib_CancelDenoms_mkProdPrf___closed__49;
x_1144 = l_Lean_Expr_const___override(x_1143, x_19);
x_1145 = l_Lean_Expr_app___override(x_1144, x_2);
x_1146 = l_Lean_Expr_app___override(x_1145, x_3);
x_1147 = l_Lean_Expr_app___override(x_1146, x_1009);
x_1148 = l_Lean_Expr_app___override(x_1147, x_1097);
x_1149 = l_Lean_Expr_app___override(x_1148, x_5);
x_1150 = l_Lean_Expr_app___override(x_1149, x_1000);
x_1151 = l_Lean_Expr_app___override(x_1150, x_1001);
lean_inc_ref(x_1095);
x_1152 = l_Lean_Expr_app___override(x_1151, x_1095);
x_1153 = l_Lean_Expr_app___override(x_1152, x_1096);
x_1154 = l_Lean_Expr_app___override(x_1153, x_1135);
x_1155 = l_Lean_Expr_app___override(x_1154, x_1141);
x_1156 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_1156, 0, x_1095);
lean_ctor_set(x_1156, 1, x_1155);
if (lean_is_scalar(x_1142)) {
 x_1157 = lean_alloc_ctor(0, 1, 0);
} else {
 x_1157 = x_1142;
}
lean_ctor_set(x_1157, 0, x_1156);
return x_1157;
}
else
{
lean_object* x_1158; lean_object* x_1159; lean_object* x_1160; 
lean_dec(x_1135);
lean_dec(x_1097);
lean_dec_ref(x_1096);
lean_dec_ref(x_1095);
lean_dec(x_1009);
lean_dec(x_1001);
lean_dec(x_1000);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
x_1158 = lean_ctor_get(x_1140, 0);
lean_inc(x_1158);
if (lean_is_exclusive(x_1140)) {
 lean_ctor_release(x_1140, 0);
 x_1159 = x_1140;
} else {
 lean_dec_ref(x_1140);
 x_1159 = lean_box(0);
}
if (lean_is_scalar(x_1159)) {
 x_1160 = lean_alloc_ctor(1, 1, 0);
} else {
 x_1160 = x_1159;
}
lean_ctor_set(x_1160, 0, x_1158);
return x_1160;
}
}
else
{
lean_object* x_1161; lean_object* x_1162; lean_object* x_1163; 
lean_dec(x_1097);
lean_dec_ref(x_1096);
lean_dec_ref(x_1095);
lean_dec(x_1009);
lean_dec(x_1001);
lean_dec(x_1000);
lean_dec(x_853);
lean_dec_ref(x_852);
lean_dec(x_851);
lean_dec_ref(x_850);
lean_dec_ref(x_845);
lean_dec_ref(x_841);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
x_1161 = lean_ctor_get(x_1134, 0);
lean_inc(x_1161);
if (lean_is_exclusive(x_1134)) {
 lean_ctor_release(x_1134, 0);
 x_1162 = x_1134;
} else {
 lean_dec_ref(x_1134);
 x_1162 = lean_box(0);
}
if (lean_is_scalar(x_1162)) {
 x_1163 = lean_alloc_ctor(1, 1, 0);
} else {
 x_1163 = x_1162;
}
lean_ctor_set(x_1163, 0, x_1161);
return x_1163;
}
}
}
else
{
lean_dec(x_1009);
lean_dec(x_1004);
lean_dec(x_1001);
lean_dec(x_1000);
lean_dec(x_853);
lean_dec_ref(x_852);
lean_dec(x_851);
lean_dec_ref(x_850);
lean_dec_ref(x_845);
lean_dec(x_842);
lean_dec_ref(x_841);
lean_dec_ref(x_47);
lean_dec_ref(x_43);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_1010;
}
}
else
{
uint8_t x_1164; 
lean_dec(x_1005);
lean_dec(x_1004);
lean_dec(x_1001);
lean_dec(x_1000);
lean_dec(x_853);
lean_dec_ref(x_852);
lean_dec(x_851);
lean_dec_ref(x_850);
lean_dec(x_848);
lean_dec_ref(x_845);
lean_dec(x_842);
lean_dec_ref(x_841);
lean_dec_ref(x_47);
lean_dec_ref(x_43);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_1164 = !lean_is_exclusive(x_1007);
if (x_1164 == 0)
{
return x_1007;
}
else
{
lean_object* x_1165; lean_object* x_1166; 
x_1165 = lean_ctor_get(x_1007, 0);
lean_inc(x_1165);
lean_dec(x_1007);
x_1166 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_1166, 0, x_1165);
return x_1166;
}
}
}
else
{
uint8_t x_1167; 
lean_dec(x_1001);
lean_dec(x_1000);
lean_dec(x_853);
lean_dec_ref(x_852);
lean_dec(x_851);
lean_dec_ref(x_850);
lean_dec(x_849);
lean_dec(x_848);
lean_dec_ref(x_845);
lean_dec(x_842);
lean_dec_ref(x_841);
lean_dec_ref(x_47);
lean_dec_ref(x_43);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_1167 = !lean_is_exclusive(x_1003);
if (x_1167 == 0)
{
return x_1003;
}
else
{
lean_object* x_1168; lean_object* x_1169; 
x_1168 = lean_ctor_get(x_1003, 0);
lean_inc(x_1168);
lean_dec(x_1003);
x_1169 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_1169, 0, x_1168);
return x_1169;
}
}
}
}
else
{
uint8_t x_1170; 
lean_dec(x_853);
lean_dec_ref(x_852);
lean_dec(x_851);
lean_dec_ref(x_850);
lean_dec(x_849);
lean_dec(x_848);
lean_dec_ref(x_847);
lean_dec_ref(x_846);
lean_dec_ref(x_845);
lean_dec_ref(x_844);
lean_dec_ref(x_843);
lean_dec(x_842);
lean_dec_ref(x_841);
lean_dec_ref(x_840);
lean_dec_ref(x_47);
lean_dec_ref(x_44);
lean_dec_ref(x_43);
lean_dec_ref(x_24);
lean_dec_ref(x_19);
lean_dec(x_16);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_1170 = !lean_is_exclusive(x_855);
if (x_1170 == 0)
{
return x_855;
}
else
{
lean_object* x_1171; lean_object* x_1172; 
x_1171 = lean_ctor_get(x_855, 0);
lean_inc(x_1171);
lean_dec(x_855);
x_1172 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_1172, 0, x_1171);
return x_1172;
}
}
}
block_1327:
{
lean_object* x_1187; 
lean_inc(x_1185);
lean_inc_ref(x_1184);
lean_inc(x_1183);
lean_inc_ref(x_1182);
x_1187 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00CancelDenoms_mkProdPrf_spec__2___redArg(x_1175, x_1174, x_1182, x_1183, x_1184, x_1185);
if (lean_obj_tag(x_1187) == 0)
{
lean_object* x_1188; lean_object* x_1189; uint8_t x_1190; 
x_1188 = lean_ctor_get(x_1187, 0);
lean_inc(x_1188);
lean_dec_ref(x_1187);
x_1189 = lean_ctor_get(x_1188, 1);
x_1190 = lean_unbox(x_1189);
if (x_1190 == 0)
{
lean_dec(x_1188);
lean_dec_ref(x_44);
if (lean_obj_tag(x_6) == 1)
{
lean_object* x_1191; 
x_1191 = lean_ctor_get(x_6, 1);
if (lean_obj_tag(x_1191) == 1)
{
lean_object* x_1192; 
x_1192 = lean_ctor_get(x_6, 2);
if (lean_obj_tag(x_1192) == 1)
{
lean_object* x_1193; 
x_1193 = lean_ctor_get(x_1192, 1);
if (lean_obj_tag(x_1193) == 0)
{
lean_object* x_1194; 
x_1194 = lean_ctor_get(x_1192, 2);
if (lean_obj_tag(x_1194) == 0)
{
lean_object* x_1195; lean_object* x_1196; 
lean_dec(x_16);
x_1195 = lean_ctor_get(x_1191, 0);
x_1196 = lean_ctor_get(x_1192, 0);
lean_inc(x_1196);
lean_inc(x_1195);
lean_inc_ref(x_1191);
x_244 = x_1176;
x_245 = x_1177;
x_246 = x_1178;
x_247 = x_1181;
x_248 = x_1180;
x_249 = x_1179;
x_250 = x_1191;
x_251 = x_1195;
x_252 = x_1196;
x_253 = x_1182;
x_254 = x_1183;
x_255 = x_1184;
x_256 = x_1185;
x_257 = lean_box(0);
goto block_820;
}
else
{
lean_dec_ref(x_1178);
x_821 = x_1176;
x_822 = x_1177;
x_823 = x_1181;
x_824 = x_1180;
x_825 = x_1179;
x_826 = x_1182;
x_827 = x_1183;
x_828 = x_1184;
x_829 = x_1185;
x_830 = lean_box(0);
goto block_837;
}
}
else
{
lean_dec_ref(x_1178);
x_821 = x_1176;
x_822 = x_1177;
x_823 = x_1181;
x_824 = x_1180;
x_825 = x_1179;
x_826 = x_1182;
x_827 = x_1183;
x_828 = x_1184;
x_829 = x_1185;
x_830 = lean_box(0);
goto block_837;
}
}
else
{
lean_dec_ref(x_1178);
x_821 = x_1176;
x_822 = x_1177;
x_823 = x_1181;
x_824 = x_1180;
x_825 = x_1179;
x_826 = x_1182;
x_827 = x_1183;
x_828 = x_1184;
x_829 = x_1185;
x_830 = lean_box(0);
goto block_837;
}
}
else
{
lean_dec_ref(x_1178);
x_821 = x_1176;
x_822 = x_1177;
x_823 = x_1181;
x_824 = x_1180;
x_825 = x_1179;
x_826 = x_1182;
x_827 = x_1183;
x_828 = x_1184;
x_829 = x_1185;
x_830 = lean_box(0);
goto block_837;
}
}
else
{
lean_dec_ref(x_1178);
x_821 = x_1176;
x_822 = x_1177;
x_823 = x_1181;
x_824 = x_1180;
x_825 = x_1179;
x_826 = x_1182;
x_827 = x_1183;
x_828 = x_1184;
x_829 = x_1185;
x_830 = lean_box(0);
goto block_837;
}
}
else
{
lean_object* x_1197; lean_object* x_1198; 
lean_dec_ref(x_1181);
lean_dec_ref(x_1180);
lean_dec_ref(x_1179);
lean_dec_ref(x_1178);
lean_dec_ref(x_1177);
lean_dec_ref(x_1176);
lean_dec_ref(x_47);
lean_dec_ref(x_24);
lean_dec(x_16);
lean_dec_ref(x_7);
x_1197 = lean_ctor_get(x_1188, 0);
lean_inc(x_1197);
lean_dec(x_1188);
lean_inc(x_1197);
lean_inc_ref(x_5);
lean_inc_ref(x_2);
x_1198 = lp_mathlib_CancelDenoms_mkProdPrf(x_1, x_2, x_3, x_4, x_5, x_6, x_1197, x_1182, x_1183, x_1184, x_1185);
if (lean_obj_tag(x_1198) == 0)
{
uint8_t x_1199; 
x_1199 = !lean_is_exclusive(x_1198);
if (x_1199 == 0)
{
lean_object* x_1200; uint8_t x_1201; 
x_1200 = lean_ctor_get(x_1198, 0);
x_1201 = !lean_is_exclusive(x_1200);
if (x_1201 == 0)
{
lean_object* x_1202; lean_object* x_1203; lean_object* x_1204; lean_object* x_1205; lean_object* x_1206; lean_object* x_1207; lean_object* x_1208; lean_object* x_1209; lean_object* x_1210; lean_object* x_1211; lean_object* x_1212; lean_object* x_1213; lean_object* x_1214; lean_object* x_1215; lean_object* x_1216; lean_object* x_1217; lean_object* x_1218; lean_object* x_1219; lean_object* x_1220; lean_object* x_1221; lean_object* x_1222; lean_object* x_1223; lean_object* x_1224; lean_object* x_1225; lean_object* x_1226; lean_object* x_1227; lean_object* x_1228; lean_object* x_1229; lean_object* x_1230; lean_object* x_1231; lean_object* x_1232; lean_object* x_1233; lean_object* x_1234; lean_object* x_1235; lean_object* x_1236; lean_object* x_1237; lean_object* x_1238; lean_object* x_1239; lean_object* x_1240; 
x_1202 = lean_ctor_get(x_1200, 0);
x_1203 = lean_ctor_get(x_1200, 1);
x_1204 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__0;
lean_inc_ref(x_19);
x_1205 = l_Lean_Expr_const___override(x_1204, x_19);
lean_inc_ref(x_2);
x_1206 = l_Lean_Expr_app___override(x_1205, x_2);
x_1207 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__3;
lean_inc_ref(x_19);
x_1208 = l_Lean_Expr_const___override(x_1207, x_19);
lean_inc_ref(x_2);
x_1209 = l_Lean_Expr_app___override(x_1208, x_2);
x_1210 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__6;
lean_inc_ref(x_19);
x_1211 = l_Lean_Expr_const___override(x_1210, x_19);
lean_inc_ref(x_2);
x_1212 = l_Lean_Expr_app___override(x_1211, x_2);
x_1213 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__9;
lean_inc_ref(x_19);
x_1214 = l_Lean_Expr_const___override(x_1213, x_19);
lean_inc_ref(x_2);
x_1215 = l_Lean_Expr_app___override(x_1214, x_2);
x_1216 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__12;
lean_inc_ref(x_19);
x_1217 = l_Lean_Expr_const___override(x_1216, x_19);
lean_inc_ref(x_2);
x_1218 = l_Lean_Expr_app___override(x_1217, x_2);
x_1219 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__15;
lean_inc_ref(x_19);
x_1220 = l_Lean_Expr_const___override(x_1219, x_19);
lean_inc_ref(x_2);
x_1221 = l_Lean_Expr_app___override(x_1220, x_2);
x_1222 = lp_mathlib_CancelDenoms_mkProdPrf___closed__37;
lean_inc_ref(x_19);
x_1223 = l_Lean_Expr_const___override(x_1222, x_19);
lean_inc_ref(x_2);
x_1224 = l_Lean_Expr_app___override(x_1223, x_2);
lean_inc_ref(x_44);
x_1225 = l_Lean_Expr_app___override(x_1224, x_44);
x_1226 = l_Lean_Expr_app___override(x_1221, x_1225);
x_1227 = l_Lean_Expr_app___override(x_1218, x_1226);
x_1228 = l_Lean_Expr_app___override(x_1215, x_1227);
x_1229 = l_Lean_Expr_app___override(x_1212, x_1228);
x_1230 = l_Lean_Expr_app___override(x_1209, x_1229);
x_1231 = l_Lean_Expr_app___override(x_1206, x_1230);
lean_inc_ref(x_1202);
x_1232 = l_Lean_Expr_app___override(x_1231, x_1202);
x_1233 = lp_mathlib_CancelDenoms_mkProdPrf___closed__39;
x_1234 = l_Lean_Expr_const___override(x_1233, x_19);
x_1235 = l_Lean_Expr_app___override(x_1234, x_2);
x_1236 = l_Lean_Expr_app___override(x_1235, x_44);
x_1237 = l_Lean_Expr_app___override(x_1236, x_5);
x_1238 = l_Lean_Expr_app___override(x_1237, x_1197);
x_1239 = l_Lean_Expr_app___override(x_1238, x_1202);
x_1240 = l_Lean_Expr_app___override(x_1239, x_1203);
lean_ctor_set(x_1200, 1, x_1240);
lean_ctor_set(x_1200, 0, x_1232);
return x_1198;
}
else
{
lean_object* x_1241; lean_object* x_1242; lean_object* x_1243; lean_object* x_1244; lean_object* x_1245; lean_object* x_1246; lean_object* x_1247; lean_object* x_1248; lean_object* x_1249; lean_object* x_1250; lean_object* x_1251; lean_object* x_1252; lean_object* x_1253; lean_object* x_1254; lean_object* x_1255; lean_object* x_1256; lean_object* x_1257; lean_object* x_1258; lean_object* x_1259; lean_object* x_1260; lean_object* x_1261; lean_object* x_1262; lean_object* x_1263; lean_object* x_1264; lean_object* x_1265; lean_object* x_1266; lean_object* x_1267; lean_object* x_1268; lean_object* x_1269; lean_object* x_1270; lean_object* x_1271; lean_object* x_1272; lean_object* x_1273; lean_object* x_1274; lean_object* x_1275; lean_object* x_1276; lean_object* x_1277; lean_object* x_1278; lean_object* x_1279; lean_object* x_1280; 
x_1241 = lean_ctor_get(x_1200, 0);
x_1242 = lean_ctor_get(x_1200, 1);
lean_inc(x_1242);
lean_inc(x_1241);
lean_dec(x_1200);
x_1243 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__0;
lean_inc_ref(x_19);
x_1244 = l_Lean_Expr_const___override(x_1243, x_19);
lean_inc_ref(x_2);
x_1245 = l_Lean_Expr_app___override(x_1244, x_2);
x_1246 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__3;
lean_inc_ref(x_19);
x_1247 = l_Lean_Expr_const___override(x_1246, x_19);
lean_inc_ref(x_2);
x_1248 = l_Lean_Expr_app___override(x_1247, x_2);
x_1249 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__6;
lean_inc_ref(x_19);
x_1250 = l_Lean_Expr_const___override(x_1249, x_19);
lean_inc_ref(x_2);
x_1251 = l_Lean_Expr_app___override(x_1250, x_2);
x_1252 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__9;
lean_inc_ref(x_19);
x_1253 = l_Lean_Expr_const___override(x_1252, x_19);
lean_inc_ref(x_2);
x_1254 = l_Lean_Expr_app___override(x_1253, x_2);
x_1255 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__12;
lean_inc_ref(x_19);
x_1256 = l_Lean_Expr_const___override(x_1255, x_19);
lean_inc_ref(x_2);
x_1257 = l_Lean_Expr_app___override(x_1256, x_2);
x_1258 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__15;
lean_inc_ref(x_19);
x_1259 = l_Lean_Expr_const___override(x_1258, x_19);
lean_inc_ref(x_2);
x_1260 = l_Lean_Expr_app___override(x_1259, x_2);
x_1261 = lp_mathlib_CancelDenoms_mkProdPrf___closed__37;
lean_inc_ref(x_19);
x_1262 = l_Lean_Expr_const___override(x_1261, x_19);
lean_inc_ref(x_2);
x_1263 = l_Lean_Expr_app___override(x_1262, x_2);
lean_inc_ref(x_44);
x_1264 = l_Lean_Expr_app___override(x_1263, x_44);
x_1265 = l_Lean_Expr_app___override(x_1260, x_1264);
x_1266 = l_Lean_Expr_app___override(x_1257, x_1265);
x_1267 = l_Lean_Expr_app___override(x_1254, x_1266);
x_1268 = l_Lean_Expr_app___override(x_1251, x_1267);
x_1269 = l_Lean_Expr_app___override(x_1248, x_1268);
x_1270 = l_Lean_Expr_app___override(x_1245, x_1269);
lean_inc_ref(x_1241);
x_1271 = l_Lean_Expr_app___override(x_1270, x_1241);
x_1272 = lp_mathlib_CancelDenoms_mkProdPrf___closed__39;
x_1273 = l_Lean_Expr_const___override(x_1272, x_19);
x_1274 = l_Lean_Expr_app___override(x_1273, x_2);
x_1275 = l_Lean_Expr_app___override(x_1274, x_44);
x_1276 = l_Lean_Expr_app___override(x_1275, x_5);
x_1277 = l_Lean_Expr_app___override(x_1276, x_1197);
x_1278 = l_Lean_Expr_app___override(x_1277, x_1241);
x_1279 = l_Lean_Expr_app___override(x_1278, x_1242);
x_1280 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_1280, 0, x_1271);
lean_ctor_set(x_1280, 1, x_1279);
lean_ctor_set(x_1198, 0, x_1280);
return x_1198;
}
}
else
{
lean_object* x_1281; lean_object* x_1282; lean_object* x_1283; lean_object* x_1284; lean_object* x_1285; lean_object* x_1286; lean_object* x_1287; lean_object* x_1288; lean_object* x_1289; lean_object* x_1290; lean_object* x_1291; lean_object* x_1292; lean_object* x_1293; lean_object* x_1294; lean_object* x_1295; lean_object* x_1296; lean_object* x_1297; lean_object* x_1298; lean_object* x_1299; lean_object* x_1300; lean_object* x_1301; lean_object* x_1302; lean_object* x_1303; lean_object* x_1304; lean_object* x_1305; lean_object* x_1306; lean_object* x_1307; lean_object* x_1308; lean_object* x_1309; lean_object* x_1310; lean_object* x_1311; lean_object* x_1312; lean_object* x_1313; lean_object* x_1314; lean_object* x_1315; lean_object* x_1316; lean_object* x_1317; lean_object* x_1318; lean_object* x_1319; lean_object* x_1320; lean_object* x_1321; lean_object* x_1322; lean_object* x_1323; 
x_1281 = lean_ctor_get(x_1198, 0);
lean_inc(x_1281);
lean_dec(x_1198);
x_1282 = lean_ctor_get(x_1281, 0);
lean_inc_ref(x_1282);
x_1283 = lean_ctor_get(x_1281, 1);
lean_inc_ref(x_1283);
if (lean_is_exclusive(x_1281)) {
 lean_ctor_release(x_1281, 0);
 lean_ctor_release(x_1281, 1);
 x_1284 = x_1281;
} else {
 lean_dec_ref(x_1281);
 x_1284 = lean_box(0);
}
x_1285 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__0;
lean_inc_ref(x_19);
x_1286 = l_Lean_Expr_const___override(x_1285, x_19);
lean_inc_ref(x_2);
x_1287 = l_Lean_Expr_app___override(x_1286, x_2);
x_1288 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__3;
lean_inc_ref(x_19);
x_1289 = l_Lean_Expr_const___override(x_1288, x_19);
lean_inc_ref(x_2);
x_1290 = l_Lean_Expr_app___override(x_1289, x_2);
x_1291 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__6;
lean_inc_ref(x_19);
x_1292 = l_Lean_Expr_const___override(x_1291, x_19);
lean_inc_ref(x_2);
x_1293 = l_Lean_Expr_app___override(x_1292, x_2);
x_1294 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__9;
lean_inc_ref(x_19);
x_1295 = l_Lean_Expr_const___override(x_1294, x_19);
lean_inc_ref(x_2);
x_1296 = l_Lean_Expr_app___override(x_1295, x_2);
x_1297 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__12;
lean_inc_ref(x_19);
x_1298 = l_Lean_Expr_const___override(x_1297, x_19);
lean_inc_ref(x_2);
x_1299 = l_Lean_Expr_app___override(x_1298, x_2);
x_1300 = lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__15;
lean_inc_ref(x_19);
x_1301 = l_Lean_Expr_const___override(x_1300, x_19);
lean_inc_ref(x_2);
x_1302 = l_Lean_Expr_app___override(x_1301, x_2);
x_1303 = lp_mathlib_CancelDenoms_mkProdPrf___closed__37;
lean_inc_ref(x_19);
x_1304 = l_Lean_Expr_const___override(x_1303, x_19);
lean_inc_ref(x_2);
x_1305 = l_Lean_Expr_app___override(x_1304, x_2);
lean_inc_ref(x_44);
x_1306 = l_Lean_Expr_app___override(x_1305, x_44);
x_1307 = l_Lean_Expr_app___override(x_1302, x_1306);
x_1308 = l_Lean_Expr_app___override(x_1299, x_1307);
x_1309 = l_Lean_Expr_app___override(x_1296, x_1308);
x_1310 = l_Lean_Expr_app___override(x_1293, x_1309);
x_1311 = l_Lean_Expr_app___override(x_1290, x_1310);
x_1312 = l_Lean_Expr_app___override(x_1287, x_1311);
lean_inc_ref(x_1282);
x_1313 = l_Lean_Expr_app___override(x_1312, x_1282);
x_1314 = lp_mathlib_CancelDenoms_mkProdPrf___closed__39;
x_1315 = l_Lean_Expr_const___override(x_1314, x_19);
x_1316 = l_Lean_Expr_app___override(x_1315, x_2);
x_1317 = l_Lean_Expr_app___override(x_1316, x_44);
x_1318 = l_Lean_Expr_app___override(x_1317, x_5);
x_1319 = l_Lean_Expr_app___override(x_1318, x_1197);
x_1320 = l_Lean_Expr_app___override(x_1319, x_1282);
x_1321 = l_Lean_Expr_app___override(x_1320, x_1283);
if (lean_is_scalar(x_1284)) {
 x_1322 = lean_alloc_ctor(0, 2, 0);
} else {
 x_1322 = x_1284;
}
lean_ctor_set(x_1322, 0, x_1313);
lean_ctor_set(x_1322, 1, x_1321);
x_1323 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_1323, 0, x_1322);
return x_1323;
}
}
else
{
lean_dec(x_1197);
lean_dec_ref(x_44);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
return x_1198;
}
}
}
else
{
uint8_t x_1324; 
lean_dec(x_1185);
lean_dec_ref(x_1184);
lean_dec(x_1183);
lean_dec_ref(x_1182);
lean_dec_ref(x_1181);
lean_dec_ref(x_1180);
lean_dec_ref(x_1179);
lean_dec_ref(x_1178);
lean_dec_ref(x_1177);
lean_dec_ref(x_1176);
lean_dec_ref(x_47);
lean_dec_ref(x_44);
lean_dec_ref(x_24);
lean_dec_ref(x_19);
lean_dec(x_16);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_1324 = !lean_is_exclusive(x_1187);
if (x_1324 == 0)
{
return x_1187;
}
else
{
lean_object* x_1325; lean_object* x_1326; 
x_1325 = lean_ctor_get(x_1187, 0);
lean_inc(x_1325);
lean_dec(x_1187);
x_1326 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_1326, 0, x_1325);
return x_1326;
}
}
}
block_1346:
{
if (lean_obj_tag(x_6) == 1)
{
lean_object* x_1343; 
x_1343 = lean_ctor_get(x_6, 2);
if (lean_obj_tag(x_1343) == 1)
{
lean_object* x_1344; lean_object* x_1345; 
x_1344 = lean_ctor_get(x_6, 1);
x_1345 = lean_ctor_get(x_1343, 0);
lean_inc(x_1345);
lean_inc(x_1344);
x_838 = x_1328;
x_839 = x_1329;
x_840 = x_1330;
x_841 = x_1331;
x_842 = x_1333;
x_843 = x_1332;
x_844 = x_1334;
x_845 = x_1337;
x_846 = x_1336;
x_847 = x_1335;
x_848 = x_1344;
x_849 = x_1345;
x_850 = x_1338;
x_851 = x_1339;
x_852 = x_1340;
x_853 = x_1341;
x_854 = lean_box(0);
goto block_1173;
}
else
{
lean_dec(x_1333);
lean_dec_ref(x_1328);
lean_dec_ref(x_43);
x_1174 = x_1329;
x_1175 = x_1330;
x_1176 = x_1331;
x_1177 = x_1332;
x_1178 = x_1334;
x_1179 = x_1337;
x_1180 = x_1336;
x_1181 = x_1335;
x_1182 = x_1338;
x_1183 = x_1339;
x_1184 = x_1340;
x_1185 = x_1341;
x_1186 = lean_box(0);
goto block_1327;
}
}
else
{
lean_dec(x_1333);
lean_dec_ref(x_1328);
lean_dec_ref(x_43);
x_1174 = x_1329;
x_1175 = x_1330;
x_1176 = x_1331;
x_1177 = x_1332;
x_1178 = x_1334;
x_1179 = x_1337;
x_1180 = x_1336;
x_1181 = x_1335;
x_1182 = x_1338;
x_1183 = x_1339;
x_1184 = x_1340;
x_1185 = x_1341;
x_1186 = lean_box(0);
goto block_1327;
}
}
block_1458:
{
lean_object* x_1360; lean_object* x_1361; 
lean_inc(x_1349);
x_1360 = l_Lean_mkRawNatLit(x_1349);
lean_inc_ref(x_47);
lean_inc_ref(x_2);
lean_inc(x_1);
x_1361 = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(x_1, x_2, x_47, x_1360, x_1355, x_1356, x_1357, x_1358);
if (lean_obj_tag(x_1361) == 0)
{
lean_object* x_1362; lean_object* x_1363; lean_object* x_1364; lean_object* x_1365; 
x_1362 = lean_ctor_get(x_1361, 0);
lean_inc(x_1362);
lean_dec_ref(x_1361);
x_1363 = lean_nat_div(x_4, x_1349);
lean_dec(x_4);
lean_inc(x_1363);
x_1364 = l_Lean_mkRawNatLit(x_1363);
lean_inc_ref(x_2);
lean_inc(x_1);
x_1365 = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(x_1, x_2, x_47, x_1364, x_1355, x_1356, x_1357, x_1358);
if (lean_obj_tag(x_1365) == 0)
{
lean_object* x_1366; lean_object* x_1367; lean_object* x_1368; 
x_1366 = lean_ctor_get(x_1365, 0);
lean_inc(x_1366);
lean_dec_ref(x_1365);
x_1367 = lean_ctor_get(x_1362, 0);
lean_inc(x_1367);
lean_dec(x_1362);
lean_inc(x_1358);
lean_inc_ref(x_1357);
lean_inc(x_1356);
lean_inc_ref(x_1355);
lean_inc(x_1347);
lean_inc(x_1367);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc(x_1);
x_1368 = lp_mathlib_CancelDenoms_mkProdPrf(x_1, x_2, x_3, x_1349, x_1367, x_1351, x_1347, x_1355, x_1356, x_1357, x_1358);
if (lean_obj_tag(x_1368) == 0)
{
lean_object* x_1369; lean_object* x_1370; lean_object* x_1371; lean_object* x_1372; lean_object* x_1373; 
x_1369 = lean_ctor_get(x_1368, 0);
lean_inc(x_1369);
lean_dec_ref(x_1368);
x_1370 = lean_ctor_get(x_1369, 0);
lean_inc_ref(x_1370);
x_1371 = lean_ctor_get(x_1369, 1);
lean_inc_ref(x_1371);
lean_dec(x_1369);
x_1372 = lean_ctor_get(x_1366, 0);
lean_inc(x_1372);
lean_dec(x_1366);
lean_inc(x_1358);
lean_inc_ref(x_1357);
lean_inc(x_1356);
lean_inc_ref(x_1355);
lean_inc(x_1348);
lean_inc(x_1372);
lean_inc_ref(x_2);
x_1373 = lp_mathlib_CancelDenoms_mkProdPrf(x_1, x_2, x_3, x_1363, x_1372, x_1350, x_1348, x_1355, x_1356, x_1357, x_1358);
if (lean_obj_tag(x_1373) == 0)
{
lean_object* x_1374; uint8_t x_1375; 
x_1374 = lean_ctor_get(x_1373, 0);
lean_inc(x_1374);
lean_dec_ref(x_1373);
x_1375 = !lean_is_exclusive(x_1374);
if (x_1375 == 0)
{
lean_object* x_1376; lean_object* x_1377; lean_object* x_1378; lean_object* x_1379; lean_object* x_1380; lean_object* x_1381; lean_object* x_1382; 
x_1376 = lean_ctor_get(x_1374, 0);
x_1377 = lean_ctor_get(x_1374, 1);
lean_inc(x_1367);
lean_inc_ref(x_1354);
x_1378 = l_Lean_Expr_app___override(x_1354, x_1367);
lean_inc(x_1372);
x_1379 = l_Lean_Expr_app___override(x_1378, x_1372);
x_1380 = l_Lean_Expr_app___override(x_1352, x_1379);
lean_inc_ref(x_5);
x_1381 = l_Lean_Expr_app___override(x_1380, x_5);
x_1382 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum(x_1381, x_1355, x_1356, x_1357, x_1358);
if (lean_obj_tag(x_1382) == 0)
{
uint8_t x_1383; 
x_1383 = !lean_is_exclusive(x_1382);
if (x_1383 == 0)
{
lean_object* x_1384; lean_object* x_1385; lean_object* x_1386; lean_object* x_1387; lean_object* x_1388; lean_object* x_1389; lean_object* x_1390; lean_object* x_1391; lean_object* x_1392; lean_object* x_1393; lean_object* x_1394; lean_object* x_1395; lean_object* x_1396; lean_object* x_1397; lean_object* x_1398; lean_object* x_1399; lean_object* x_1400; 
x_1384 = lean_ctor_get(x_1382, 0);
lean_inc_ref(x_1370);
x_1385 = l_Lean_Expr_app___override(x_1354, x_1370);
lean_inc_ref(x_1376);
x_1386 = l_Lean_Expr_app___override(x_1385, x_1376);
x_1387 = lp_mathlib_CancelDenoms_mkProdPrf___closed__51;
x_1388 = l_Lean_Expr_const___override(x_1387, x_19);
x_1389 = l_Lean_Expr_app___override(x_1388, x_2);
x_1390 = l_Lean_Expr_app___override(x_1389, x_1353);
x_1391 = l_Lean_Expr_app___override(x_1390, x_1367);
x_1392 = l_Lean_Expr_app___override(x_1391, x_1372);
x_1393 = l_Lean_Expr_app___override(x_1392, x_5);
x_1394 = l_Lean_Expr_app___override(x_1393, x_1347);
x_1395 = l_Lean_Expr_app___override(x_1394, x_1348);
x_1396 = l_Lean_Expr_app___override(x_1395, x_1370);
x_1397 = l_Lean_Expr_app___override(x_1396, x_1376);
x_1398 = l_Lean_Expr_app___override(x_1397, x_1371);
x_1399 = l_Lean_Expr_app___override(x_1398, x_1377);
x_1400 = l_Lean_Expr_app___override(x_1399, x_1384);
lean_ctor_set(x_1374, 1, x_1400);
lean_ctor_set(x_1374, 0, x_1386);
lean_ctor_set(x_1382, 0, x_1374);
return x_1382;
}
else
{
lean_object* x_1401; lean_object* x_1402; lean_object* x_1403; lean_object* x_1404; lean_object* x_1405; lean_object* x_1406; lean_object* x_1407; lean_object* x_1408; lean_object* x_1409; lean_object* x_1410; lean_object* x_1411; lean_object* x_1412; lean_object* x_1413; lean_object* x_1414; lean_object* x_1415; lean_object* x_1416; lean_object* x_1417; lean_object* x_1418; 
x_1401 = lean_ctor_get(x_1382, 0);
lean_inc(x_1401);
lean_dec(x_1382);
lean_inc_ref(x_1370);
x_1402 = l_Lean_Expr_app___override(x_1354, x_1370);
lean_inc_ref(x_1376);
x_1403 = l_Lean_Expr_app___override(x_1402, x_1376);
x_1404 = lp_mathlib_CancelDenoms_mkProdPrf___closed__51;
x_1405 = l_Lean_Expr_const___override(x_1404, x_19);
x_1406 = l_Lean_Expr_app___override(x_1405, x_2);
x_1407 = l_Lean_Expr_app___override(x_1406, x_1353);
x_1408 = l_Lean_Expr_app___override(x_1407, x_1367);
x_1409 = l_Lean_Expr_app___override(x_1408, x_1372);
x_1410 = l_Lean_Expr_app___override(x_1409, x_5);
x_1411 = l_Lean_Expr_app___override(x_1410, x_1347);
x_1412 = l_Lean_Expr_app___override(x_1411, x_1348);
x_1413 = l_Lean_Expr_app___override(x_1412, x_1370);
x_1414 = l_Lean_Expr_app___override(x_1413, x_1376);
x_1415 = l_Lean_Expr_app___override(x_1414, x_1371);
x_1416 = l_Lean_Expr_app___override(x_1415, x_1377);
x_1417 = l_Lean_Expr_app___override(x_1416, x_1401);
lean_ctor_set(x_1374, 1, x_1417);
lean_ctor_set(x_1374, 0, x_1403);
x_1418 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_1418, 0, x_1374);
return x_1418;
}
}
else
{
uint8_t x_1419; 
lean_free_object(x_1374);
lean_dec_ref(x_1377);
lean_dec_ref(x_1376);
lean_dec(x_1372);
lean_dec_ref(x_1371);
lean_dec_ref(x_1370);
lean_dec(x_1367);
lean_dec_ref(x_1354);
lean_dec_ref(x_1353);
lean_dec(x_1348);
lean_dec(x_1347);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
x_1419 = !lean_is_exclusive(x_1382);
if (x_1419 == 0)
{
return x_1382;
}
else
{
lean_object* x_1420; lean_object* x_1421; 
x_1420 = lean_ctor_get(x_1382, 0);
lean_inc(x_1420);
lean_dec(x_1382);
x_1421 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_1421, 0, x_1420);
return x_1421;
}
}
}
else
{
lean_object* x_1422; lean_object* x_1423; lean_object* x_1424; lean_object* x_1425; lean_object* x_1426; lean_object* x_1427; lean_object* x_1428; 
x_1422 = lean_ctor_get(x_1374, 0);
x_1423 = lean_ctor_get(x_1374, 1);
lean_inc(x_1423);
lean_inc(x_1422);
lean_dec(x_1374);
lean_inc(x_1367);
lean_inc_ref(x_1354);
x_1424 = l_Lean_Expr_app___override(x_1354, x_1367);
lean_inc(x_1372);
x_1425 = l_Lean_Expr_app___override(x_1424, x_1372);
x_1426 = l_Lean_Expr_app___override(x_1352, x_1425);
lean_inc_ref(x_5);
x_1427 = l_Lean_Expr_app___override(x_1426, x_5);
x_1428 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum(x_1427, x_1355, x_1356, x_1357, x_1358);
if (lean_obj_tag(x_1428) == 0)
{
lean_object* x_1429; lean_object* x_1430; lean_object* x_1431; lean_object* x_1432; lean_object* x_1433; lean_object* x_1434; lean_object* x_1435; lean_object* x_1436; lean_object* x_1437; lean_object* x_1438; lean_object* x_1439; lean_object* x_1440; lean_object* x_1441; lean_object* x_1442; lean_object* x_1443; lean_object* x_1444; lean_object* x_1445; lean_object* x_1446; lean_object* x_1447; lean_object* x_1448; 
x_1429 = lean_ctor_get(x_1428, 0);
lean_inc(x_1429);
if (lean_is_exclusive(x_1428)) {
 lean_ctor_release(x_1428, 0);
 x_1430 = x_1428;
} else {
 lean_dec_ref(x_1428);
 x_1430 = lean_box(0);
}
lean_inc_ref(x_1370);
x_1431 = l_Lean_Expr_app___override(x_1354, x_1370);
lean_inc_ref(x_1422);
x_1432 = l_Lean_Expr_app___override(x_1431, x_1422);
x_1433 = lp_mathlib_CancelDenoms_mkProdPrf___closed__51;
x_1434 = l_Lean_Expr_const___override(x_1433, x_19);
x_1435 = l_Lean_Expr_app___override(x_1434, x_2);
x_1436 = l_Lean_Expr_app___override(x_1435, x_1353);
x_1437 = l_Lean_Expr_app___override(x_1436, x_1367);
x_1438 = l_Lean_Expr_app___override(x_1437, x_1372);
x_1439 = l_Lean_Expr_app___override(x_1438, x_5);
x_1440 = l_Lean_Expr_app___override(x_1439, x_1347);
x_1441 = l_Lean_Expr_app___override(x_1440, x_1348);
x_1442 = l_Lean_Expr_app___override(x_1441, x_1370);
x_1443 = l_Lean_Expr_app___override(x_1442, x_1422);
x_1444 = l_Lean_Expr_app___override(x_1443, x_1371);
x_1445 = l_Lean_Expr_app___override(x_1444, x_1423);
x_1446 = l_Lean_Expr_app___override(x_1445, x_1429);
x_1447 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_1447, 0, x_1432);
lean_ctor_set(x_1447, 1, x_1446);
if (lean_is_scalar(x_1430)) {
 x_1448 = lean_alloc_ctor(0, 1, 0);
} else {
 x_1448 = x_1430;
}
lean_ctor_set(x_1448, 0, x_1447);
return x_1448;
}
else
{
lean_object* x_1449; lean_object* x_1450; lean_object* x_1451; 
lean_dec_ref(x_1423);
lean_dec_ref(x_1422);
lean_dec(x_1372);
lean_dec_ref(x_1371);
lean_dec_ref(x_1370);
lean_dec(x_1367);
lean_dec_ref(x_1354);
lean_dec_ref(x_1353);
lean_dec(x_1348);
lean_dec(x_1347);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
x_1449 = lean_ctor_get(x_1428, 0);
lean_inc(x_1449);
if (lean_is_exclusive(x_1428)) {
 lean_ctor_release(x_1428, 0);
 x_1450 = x_1428;
} else {
 lean_dec_ref(x_1428);
 x_1450 = lean_box(0);
}
if (lean_is_scalar(x_1450)) {
 x_1451 = lean_alloc_ctor(1, 1, 0);
} else {
 x_1451 = x_1450;
}
lean_ctor_set(x_1451, 0, x_1449);
return x_1451;
}
}
}
else
{
lean_dec(x_1372);
lean_dec_ref(x_1371);
lean_dec_ref(x_1370);
lean_dec(x_1367);
lean_dec(x_1358);
lean_dec_ref(x_1357);
lean_dec(x_1356);
lean_dec_ref(x_1355);
lean_dec_ref(x_1354);
lean_dec_ref(x_1353);
lean_dec_ref(x_1352);
lean_dec(x_1348);
lean_dec(x_1347);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
return x_1373;
}
}
else
{
lean_dec(x_1367);
lean_dec(x_1366);
lean_dec(x_1363);
lean_dec(x_1358);
lean_dec_ref(x_1357);
lean_dec(x_1356);
lean_dec_ref(x_1355);
lean_dec_ref(x_1354);
lean_dec_ref(x_1353);
lean_dec_ref(x_1352);
lean_dec(x_1350);
lean_dec(x_1348);
lean_dec(x_1347);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_1368;
}
}
else
{
uint8_t x_1452; 
lean_dec(x_1363);
lean_dec(x_1362);
lean_dec(x_1358);
lean_dec_ref(x_1357);
lean_dec(x_1356);
lean_dec_ref(x_1355);
lean_dec_ref(x_1354);
lean_dec_ref(x_1353);
lean_dec_ref(x_1352);
lean_dec(x_1351);
lean_dec(x_1350);
lean_dec(x_1349);
lean_dec(x_1348);
lean_dec(x_1347);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_1452 = !lean_is_exclusive(x_1365);
if (x_1452 == 0)
{
return x_1365;
}
else
{
lean_object* x_1453; lean_object* x_1454; 
x_1453 = lean_ctor_get(x_1365, 0);
lean_inc(x_1453);
lean_dec(x_1365);
x_1454 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_1454, 0, x_1453);
return x_1454;
}
}
}
else
{
uint8_t x_1455; 
lean_dec(x_1358);
lean_dec_ref(x_1357);
lean_dec(x_1356);
lean_dec_ref(x_1355);
lean_dec_ref(x_1354);
lean_dec_ref(x_1353);
lean_dec_ref(x_1352);
lean_dec(x_1351);
lean_dec(x_1350);
lean_dec(x_1349);
lean_dec(x_1348);
lean_dec(x_1347);
lean_dec_ref(x_47);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_1455 = !lean_is_exclusive(x_1361);
if (x_1455 == 0)
{
return x_1361;
}
else
{
lean_object* x_1456; lean_object* x_1457; 
x_1456 = lean_ctor_get(x_1361, 0);
lean_inc(x_1456);
lean_dec(x_1361);
x_1457 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_1457, 0, x_1456);
return x_1457;
}
}
}
block_1502:
{
lean_object* x_1478; 
lean_inc(x_1476);
lean_inc_ref(x_1475);
lean_inc(x_1474);
lean_inc_ref(x_1473);
x_1478 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00CancelDenoms_mkProdPrf_spec__2___redArg(x_1461, x_1460, x_1473, x_1474, x_1475, x_1476);
if (lean_obj_tag(x_1478) == 0)
{
lean_object* x_1479; lean_object* x_1480; lean_object* x_1481; uint8_t x_1482; 
x_1479 = lean_ctor_get(x_1478, 0);
lean_inc(x_1479);
lean_dec_ref(x_1478);
x_1480 = lean_ctor_get(x_1479, 1);
lean_inc(x_1480);
x_1481 = lean_ctor_get(x_1480, 1);
x_1482 = lean_unbox(x_1481);
if (x_1482 == 0)
{
lean_dec(x_1480);
lean_dec(x_1479);
lean_dec(x_1472);
lean_dec(x_1471);
lean_dec(x_1470);
if (lean_obj_tag(x_6) == 1)
{
lean_object* x_1483; 
x_1483 = lean_ctor_get(x_6, 2);
if (lean_obj_tag(x_1483) == 1)
{
lean_object* x_1484; lean_object* x_1485; 
x_1484 = lean_ctor_get(x_6, 1);
x_1485 = lean_ctor_get(x_1483, 0);
lean_inc(x_1485);
lean_inc(x_1484);
x_838 = x_1459;
x_839 = x_1460;
x_840 = x_1462;
x_841 = x_1463;
x_842 = x_1465;
x_843 = x_1464;
x_844 = x_1466;
x_845 = x_1469;
x_846 = x_1468;
x_847 = x_1467;
x_848 = x_1484;
x_849 = x_1485;
x_850 = x_1473;
x_851 = x_1474;
x_852 = x_1475;
x_853 = x_1476;
x_854 = lean_box(0);
goto block_1173;
}
else
{
lean_dec(x_1465);
lean_dec_ref(x_1459);
lean_dec_ref(x_43);
x_1174 = x_1460;
x_1175 = x_1462;
x_1176 = x_1463;
x_1177 = x_1464;
x_1178 = x_1466;
x_1179 = x_1469;
x_1180 = x_1468;
x_1181 = x_1467;
x_1182 = x_1473;
x_1183 = x_1474;
x_1184 = x_1475;
x_1185 = x_1476;
x_1186 = lean_box(0);
goto block_1327;
}
}
else
{
lean_dec(x_1465);
lean_dec_ref(x_1459);
lean_dec_ref(x_43);
x_1174 = x_1460;
x_1175 = x_1462;
x_1176 = x_1463;
x_1177 = x_1464;
x_1178 = x_1466;
x_1179 = x_1469;
x_1180 = x_1468;
x_1181 = x_1467;
x_1182 = x_1473;
x_1183 = x_1474;
x_1184 = x_1475;
x_1185 = x_1476;
x_1186 = lean_box(0);
goto block_1327;
}
}
else
{
lean_object* x_1486; lean_object* x_1487; lean_object* x_1488; 
lean_dec_ref(x_1468);
lean_dec_ref(x_1467);
lean_dec(x_1465);
lean_dec_ref(x_1464);
lean_dec_ref(x_1462);
lean_dec_ref(x_1459);
lean_dec_ref(x_44);
lean_dec_ref(x_43);
lean_dec_ref(x_24);
lean_dec(x_16);
lean_dec_ref(x_7);
lean_dec(x_6);
x_1486 = lean_ctor_get(x_1479, 0);
lean_inc(x_1486);
lean_dec(x_1479);
x_1487 = lean_ctor_get(x_1480, 0);
lean_inc(x_1487);
lean_dec(x_1480);
x_1488 = lp_mathlib_Lean_isTracingEnabledFor___at___00CancelDenoms_mkProdPrf_spec__0___redArg(x_13, x_1475);
if (lean_obj_tag(x_1488) == 0)
{
lean_object* x_1489; uint8_t x_1490; 
x_1489 = lean_ctor_get(x_1488, 0);
lean_inc(x_1489);
lean_dec_ref(x_1488);
x_1490 = lean_unbox(x_1489);
lean_dec(x_1489);
if (x_1490 == 0)
{
x_1347 = x_1486;
x_1348 = x_1487;
x_1349 = x_1471;
x_1350 = x_1472;
x_1351 = x_1470;
x_1352 = x_1463;
x_1353 = x_1466;
x_1354 = x_1469;
x_1355 = x_1473;
x_1356 = x_1474;
x_1357 = x_1475;
x_1358 = x_1476;
x_1359 = lean_box(0);
goto block_1458;
}
else
{
lean_object* x_1491; lean_object* x_1492; 
x_1491 = lp_mathlib_CancelDenoms_mkProdPrf___closed__53;
x_1492 = lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3(x_13, x_1491, x_1473, x_1474, x_1475, x_1476);
if (lean_obj_tag(x_1492) == 0)
{
lean_dec_ref(x_1492);
x_1347 = x_1486;
x_1348 = x_1487;
x_1349 = x_1471;
x_1350 = x_1472;
x_1351 = x_1470;
x_1352 = x_1463;
x_1353 = x_1466;
x_1354 = x_1469;
x_1355 = x_1473;
x_1356 = x_1474;
x_1357 = x_1475;
x_1358 = x_1476;
x_1359 = lean_box(0);
goto block_1458;
}
else
{
uint8_t x_1493; 
lean_dec(x_1487);
lean_dec(x_1486);
lean_dec(x_1476);
lean_dec_ref(x_1475);
lean_dec(x_1474);
lean_dec_ref(x_1473);
lean_dec(x_1472);
lean_dec(x_1471);
lean_dec(x_1470);
lean_dec_ref(x_1469);
lean_dec_ref(x_1466);
lean_dec_ref(x_1463);
lean_dec_ref(x_47);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_1493 = !lean_is_exclusive(x_1492);
if (x_1493 == 0)
{
return x_1492;
}
else
{
lean_object* x_1494; lean_object* x_1495; 
x_1494 = lean_ctor_get(x_1492, 0);
lean_inc(x_1494);
lean_dec(x_1492);
x_1495 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_1495, 0, x_1494);
return x_1495;
}
}
}
}
else
{
uint8_t x_1496; 
lean_dec(x_1487);
lean_dec(x_1486);
lean_dec(x_1476);
lean_dec_ref(x_1475);
lean_dec(x_1474);
lean_dec_ref(x_1473);
lean_dec(x_1472);
lean_dec(x_1471);
lean_dec(x_1470);
lean_dec_ref(x_1469);
lean_dec_ref(x_1466);
lean_dec_ref(x_1463);
lean_dec_ref(x_47);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_1496 = !lean_is_exclusive(x_1488);
if (x_1496 == 0)
{
return x_1488;
}
else
{
lean_object* x_1497; lean_object* x_1498; 
x_1497 = lean_ctor_get(x_1488, 0);
lean_inc(x_1497);
lean_dec(x_1488);
x_1498 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_1498, 0, x_1497);
return x_1498;
}
}
}
}
else
{
uint8_t x_1499; 
lean_dec(x_1476);
lean_dec_ref(x_1475);
lean_dec(x_1474);
lean_dec_ref(x_1473);
lean_dec(x_1472);
lean_dec(x_1471);
lean_dec(x_1470);
lean_dec_ref(x_1469);
lean_dec_ref(x_1468);
lean_dec_ref(x_1467);
lean_dec_ref(x_1466);
lean_dec(x_1465);
lean_dec_ref(x_1464);
lean_dec_ref(x_1463);
lean_dec_ref(x_1462);
lean_dec_ref(x_1459);
lean_dec_ref(x_47);
lean_dec_ref(x_44);
lean_dec_ref(x_43);
lean_dec_ref(x_24);
lean_dec_ref(x_19);
lean_dec(x_16);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_1499 = !lean_is_exclusive(x_1478);
if (x_1499 == 0)
{
return x_1478;
}
else
{
lean_object* x_1500; lean_object* x_1501; 
x_1500 = lean_ctor_get(x_1478, 0);
lean_inc(x_1500);
lean_dec(x_1478);
x_1501 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_1501, 0, x_1500);
return x_1501;
}
}
}
block_1659:
{
lean_object* x_1522; 
lean_inc(x_1520);
lean_inc_ref(x_1519);
lean_inc(x_1518);
lean_inc_ref(x_1517);
x_1522 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00CancelDenoms_mkProdPrf_spec__2___redArg(x_1503, x_1506, x_1517, x_1518, x_1519, x_1520);
if (lean_obj_tag(x_1522) == 0)
{
lean_object* x_1523; lean_object* x_1524; lean_object* x_1525; uint8_t x_1526; 
x_1523 = lean_ctor_get(x_1522, 0);
lean_inc(x_1523);
lean_dec_ref(x_1522);
x_1524 = lean_ctor_get(x_1523, 1);
lean_inc(x_1524);
x_1525 = lean_ctor_get(x_1524, 1);
x_1526 = lean_unbox(x_1525);
if (x_1526 == 0)
{
lean_dec(x_1524);
lean_dec(x_1523);
lean_dec(x_1516);
lean_dec(x_1515);
lean_dec_ref(x_45);
if (lean_obj_tag(x_6) == 1)
{
lean_object* x_1527; 
x_1527 = lean_ctor_get(x_6, 1);
if (lean_obj_tag(x_1527) == 1)
{
lean_object* x_1528; lean_object* x_1529; 
x_1528 = lean_ctor_get(x_6, 2);
x_1529 = lean_ctor_get(x_1527, 0);
lean_inc(x_1528);
lean_inc(x_1529);
lean_inc_ref(x_1527);
x_1459 = x_1504;
x_1460 = x_1506;
x_1461 = x_1505;
x_1462 = x_1507;
x_1463 = x_1508;
x_1464 = x_1510;
x_1465 = x_1509;
x_1466 = x_1511;
x_1467 = x_1514;
x_1468 = x_1513;
x_1469 = x_1512;
x_1470 = x_1527;
x_1471 = x_1529;
x_1472 = x_1528;
x_1473 = x_1517;
x_1474 = x_1518;
x_1475 = x_1519;
x_1476 = x_1520;
x_1477 = lean_box(0);
goto block_1502;
}
else
{
lean_dec_ref(x_1505);
x_1328 = x_1504;
x_1329 = x_1506;
x_1330 = x_1507;
x_1331 = x_1508;
x_1332 = x_1510;
x_1333 = x_1509;
x_1334 = x_1511;
x_1335 = x_1514;
x_1336 = x_1513;
x_1337 = x_1512;
x_1338 = x_1517;
x_1339 = x_1518;
x_1340 = x_1519;
x_1341 = x_1520;
x_1342 = lean_box(0);
goto block_1346;
}
}
else
{
lean_dec_ref(x_1505);
x_1328 = x_1504;
x_1329 = x_1506;
x_1330 = x_1507;
x_1331 = x_1508;
x_1332 = x_1510;
x_1333 = x_1509;
x_1334 = x_1511;
x_1335 = x_1514;
x_1336 = x_1513;
x_1337 = x_1512;
x_1338 = x_1517;
x_1339 = x_1518;
x_1340 = x_1519;
x_1341 = x_1520;
x_1342 = lean_box(0);
goto block_1346;
}
}
else
{
lean_object* x_1530; lean_object* x_1531; lean_object* x_1532; 
lean_dec_ref(x_1514);
lean_dec_ref(x_1513);
lean_dec_ref(x_1512);
lean_dec_ref(x_1511);
lean_dec_ref(x_1510);
lean_dec_ref(x_1508);
lean_dec_ref(x_1507);
lean_dec_ref(x_1505);
lean_dec_ref(x_1504);
lean_dec_ref(x_47);
lean_dec_ref(x_43);
lean_dec_ref(x_24);
lean_dec(x_16);
lean_dec_ref(x_7);
lean_dec(x_6);
x_1530 = lean_ctor_get(x_1523, 0);
lean_inc(x_1530);
lean_dec(x_1523);
x_1531 = lean_ctor_get(x_1524, 0);
lean_inc(x_1531);
lean_dec(x_1524);
lean_inc(x_1520);
lean_inc_ref(x_1519);
lean_inc(x_1518);
lean_inc_ref(x_1517);
lean_inc(x_1530);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc(x_1);
x_1532 = lp_mathlib_CancelDenoms_mkProdPrf(x_1, x_2, x_3, x_4, x_5, x_1515, x_1530, x_1517, x_1518, x_1519, x_1520);
if (lean_obj_tag(x_1532) == 0)
{
lean_object* x_1533; lean_object* x_1534; lean_object* x_1535; lean_object* x_1536; 
x_1533 = lean_ctor_get(x_1532, 0);
lean_inc(x_1533);
lean_dec_ref(x_1532);
x_1534 = lean_ctor_get(x_1533, 0);
lean_inc_ref(x_1534);
x_1535 = lean_ctor_get(x_1533, 1);
lean_inc_ref(x_1535);
lean_dec(x_1533);
lean_inc(x_1531);
lean_inc_ref(x_5);
lean_inc_ref(x_2);
x_1536 = lp_mathlib_CancelDenoms_mkProdPrf(x_1, x_2, x_3, x_4, x_5, x_1516, x_1531, x_1517, x_1518, x_1519, x_1520);
if (lean_obj_tag(x_1536) == 0)
{
uint8_t x_1537; 
x_1537 = !lean_is_exclusive(x_1536);
if (x_1537 == 0)
{
lean_object* x_1538; uint8_t x_1539; 
x_1538 = lean_ctor_get(x_1536, 0);
x_1539 = !lean_is_exclusive(x_1538);
if (x_1539 == 0)
{
lean_object* x_1540; lean_object* x_1541; lean_object* x_1542; lean_object* x_1543; lean_object* x_1544; lean_object* x_1545; lean_object* x_1546; lean_object* x_1547; lean_object* x_1548; lean_object* x_1549; lean_object* x_1550; lean_object* x_1551; lean_object* x_1552; lean_object* x_1553; lean_object* x_1554; lean_object* x_1555; lean_object* x_1556; lean_object* x_1557; lean_object* x_1558; lean_object* x_1559; lean_object* x_1560; lean_object* x_1561; lean_object* x_1562; lean_object* x_1563; lean_object* x_1564; lean_object* x_1565; lean_object* x_1566; lean_object* x_1567; lean_object* x_1568; lean_object* x_1569; lean_object* x_1570; lean_object* x_1571; lean_object* x_1572; lean_object* x_1573; lean_object* x_1574; lean_object* x_1575; lean_object* x_1576; 
x_1540 = lean_ctor_get(x_1538, 0);
x_1541 = lean_ctor_get(x_1538, 1);
x_1542 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__0;
x_1543 = l_Lean_Expr_const___override(x_1542, x_1509);
lean_inc_ref(x_2);
x_1544 = l_Lean_Expr_app___override(x_1543, x_2);
lean_inc_ref(x_2);
x_1545 = l_Lean_Expr_app___override(x_1544, x_2);
lean_inc_ref(x_2);
x_1546 = l_Lean_Expr_app___override(x_1545, x_2);
x_1547 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__2;
lean_inc_ref(x_19);
x_1548 = l_Lean_Expr_const___override(x_1547, x_19);
lean_inc_ref(x_2);
x_1549 = l_Lean_Expr_app___override(x_1548, x_2);
x_1550 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__5;
lean_inc_ref(x_19);
x_1551 = l_Lean_Expr_const___override(x_1550, x_19);
lean_inc_ref(x_2);
x_1552 = l_Lean_Expr_app___override(x_1551, x_2);
x_1553 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__8;
lean_inc_ref(x_19);
x_1554 = l_Lean_Expr_const___override(x_1553, x_19);
lean_inc_ref(x_2);
x_1555 = l_Lean_Expr_app___override(x_1554, x_2);
x_1556 = lp_mathlib_CancelDenoms_mkProdPrf___closed__54;
lean_inc_ref(x_19);
x_1557 = l_Lean_Expr_const___override(x_1556, x_19);
lean_inc_ref(x_2);
x_1558 = l_Lean_Expr_app___override(x_1557, x_2);
x_1559 = l_Lean_Expr_app___override(x_1558, x_45);
x_1560 = l_Lean_Expr_app___override(x_1555, x_1559);
x_1561 = l_Lean_Expr_app___override(x_1552, x_1560);
x_1562 = l_Lean_Expr_app___override(x_1549, x_1561);
x_1563 = l_Lean_Expr_app___override(x_1546, x_1562);
lean_inc_ref(x_1534);
x_1564 = l_Lean_Expr_app___override(x_1563, x_1534);
lean_inc_ref(x_1540);
x_1565 = l_Lean_Expr_app___override(x_1564, x_1540);
x_1566 = lp_mathlib_CancelDenoms_mkProdPrf___closed__56;
x_1567 = l_Lean_Expr_const___override(x_1566, x_19);
x_1568 = l_Lean_Expr_app___override(x_1567, x_2);
x_1569 = l_Lean_Expr_app___override(x_1568, x_44);
x_1570 = l_Lean_Expr_app___override(x_1569, x_5);
x_1571 = l_Lean_Expr_app___override(x_1570, x_1530);
x_1572 = l_Lean_Expr_app___override(x_1571, x_1531);
x_1573 = l_Lean_Expr_app___override(x_1572, x_1534);
x_1574 = l_Lean_Expr_app___override(x_1573, x_1540);
x_1575 = l_Lean_Expr_app___override(x_1574, x_1535);
x_1576 = l_Lean_Expr_app___override(x_1575, x_1541);
lean_ctor_set(x_1538, 1, x_1576);
lean_ctor_set(x_1538, 0, x_1565);
return x_1536;
}
else
{
lean_object* x_1577; lean_object* x_1578; lean_object* x_1579; lean_object* x_1580; lean_object* x_1581; lean_object* x_1582; lean_object* x_1583; lean_object* x_1584; lean_object* x_1585; lean_object* x_1586; lean_object* x_1587; lean_object* x_1588; lean_object* x_1589; lean_object* x_1590; lean_object* x_1591; lean_object* x_1592; lean_object* x_1593; lean_object* x_1594; lean_object* x_1595; lean_object* x_1596; lean_object* x_1597; lean_object* x_1598; lean_object* x_1599; lean_object* x_1600; lean_object* x_1601; lean_object* x_1602; lean_object* x_1603; lean_object* x_1604; lean_object* x_1605; lean_object* x_1606; lean_object* x_1607; lean_object* x_1608; lean_object* x_1609; lean_object* x_1610; lean_object* x_1611; lean_object* x_1612; lean_object* x_1613; lean_object* x_1614; 
x_1577 = lean_ctor_get(x_1538, 0);
x_1578 = lean_ctor_get(x_1538, 1);
lean_inc(x_1578);
lean_inc(x_1577);
lean_dec(x_1538);
x_1579 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__0;
x_1580 = l_Lean_Expr_const___override(x_1579, x_1509);
lean_inc_ref(x_2);
x_1581 = l_Lean_Expr_app___override(x_1580, x_2);
lean_inc_ref(x_2);
x_1582 = l_Lean_Expr_app___override(x_1581, x_2);
lean_inc_ref(x_2);
x_1583 = l_Lean_Expr_app___override(x_1582, x_2);
x_1584 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__2;
lean_inc_ref(x_19);
x_1585 = l_Lean_Expr_const___override(x_1584, x_19);
lean_inc_ref(x_2);
x_1586 = l_Lean_Expr_app___override(x_1585, x_2);
x_1587 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__5;
lean_inc_ref(x_19);
x_1588 = l_Lean_Expr_const___override(x_1587, x_19);
lean_inc_ref(x_2);
x_1589 = l_Lean_Expr_app___override(x_1588, x_2);
x_1590 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__8;
lean_inc_ref(x_19);
x_1591 = l_Lean_Expr_const___override(x_1590, x_19);
lean_inc_ref(x_2);
x_1592 = l_Lean_Expr_app___override(x_1591, x_2);
x_1593 = lp_mathlib_CancelDenoms_mkProdPrf___closed__54;
lean_inc_ref(x_19);
x_1594 = l_Lean_Expr_const___override(x_1593, x_19);
lean_inc_ref(x_2);
x_1595 = l_Lean_Expr_app___override(x_1594, x_2);
x_1596 = l_Lean_Expr_app___override(x_1595, x_45);
x_1597 = l_Lean_Expr_app___override(x_1592, x_1596);
x_1598 = l_Lean_Expr_app___override(x_1589, x_1597);
x_1599 = l_Lean_Expr_app___override(x_1586, x_1598);
x_1600 = l_Lean_Expr_app___override(x_1583, x_1599);
lean_inc_ref(x_1534);
x_1601 = l_Lean_Expr_app___override(x_1600, x_1534);
lean_inc_ref(x_1577);
x_1602 = l_Lean_Expr_app___override(x_1601, x_1577);
x_1603 = lp_mathlib_CancelDenoms_mkProdPrf___closed__56;
x_1604 = l_Lean_Expr_const___override(x_1603, x_19);
x_1605 = l_Lean_Expr_app___override(x_1604, x_2);
x_1606 = l_Lean_Expr_app___override(x_1605, x_44);
x_1607 = l_Lean_Expr_app___override(x_1606, x_5);
x_1608 = l_Lean_Expr_app___override(x_1607, x_1530);
x_1609 = l_Lean_Expr_app___override(x_1608, x_1531);
x_1610 = l_Lean_Expr_app___override(x_1609, x_1534);
x_1611 = l_Lean_Expr_app___override(x_1610, x_1577);
x_1612 = l_Lean_Expr_app___override(x_1611, x_1535);
x_1613 = l_Lean_Expr_app___override(x_1612, x_1578);
x_1614 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_1614, 0, x_1602);
lean_ctor_set(x_1614, 1, x_1613);
lean_ctor_set(x_1536, 0, x_1614);
return x_1536;
}
}
else
{
lean_object* x_1615; lean_object* x_1616; lean_object* x_1617; lean_object* x_1618; lean_object* x_1619; lean_object* x_1620; lean_object* x_1621; lean_object* x_1622; lean_object* x_1623; lean_object* x_1624; lean_object* x_1625; lean_object* x_1626; lean_object* x_1627; lean_object* x_1628; lean_object* x_1629; lean_object* x_1630; lean_object* x_1631; lean_object* x_1632; lean_object* x_1633; lean_object* x_1634; lean_object* x_1635; lean_object* x_1636; lean_object* x_1637; lean_object* x_1638; lean_object* x_1639; lean_object* x_1640; lean_object* x_1641; lean_object* x_1642; lean_object* x_1643; lean_object* x_1644; lean_object* x_1645; lean_object* x_1646; lean_object* x_1647; lean_object* x_1648; lean_object* x_1649; lean_object* x_1650; lean_object* x_1651; lean_object* x_1652; lean_object* x_1653; lean_object* x_1654; lean_object* x_1655; 
x_1615 = lean_ctor_get(x_1536, 0);
lean_inc(x_1615);
lean_dec(x_1536);
x_1616 = lean_ctor_get(x_1615, 0);
lean_inc_ref(x_1616);
x_1617 = lean_ctor_get(x_1615, 1);
lean_inc_ref(x_1617);
if (lean_is_exclusive(x_1615)) {
 lean_ctor_release(x_1615, 0);
 lean_ctor_release(x_1615, 1);
 x_1618 = x_1615;
} else {
 lean_dec_ref(x_1615);
 x_1618 = lean_box(0);
}
x_1619 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__0;
x_1620 = l_Lean_Expr_const___override(x_1619, x_1509);
lean_inc_ref(x_2);
x_1621 = l_Lean_Expr_app___override(x_1620, x_2);
lean_inc_ref(x_2);
x_1622 = l_Lean_Expr_app___override(x_1621, x_2);
lean_inc_ref(x_2);
x_1623 = l_Lean_Expr_app___override(x_1622, x_2);
x_1624 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__2;
lean_inc_ref(x_19);
x_1625 = l_Lean_Expr_const___override(x_1624, x_19);
lean_inc_ref(x_2);
x_1626 = l_Lean_Expr_app___override(x_1625, x_2);
x_1627 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__5;
lean_inc_ref(x_19);
x_1628 = l_Lean_Expr_const___override(x_1627, x_19);
lean_inc_ref(x_2);
x_1629 = l_Lean_Expr_app___override(x_1628, x_2);
x_1630 = lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__8;
lean_inc_ref(x_19);
x_1631 = l_Lean_Expr_const___override(x_1630, x_19);
lean_inc_ref(x_2);
x_1632 = l_Lean_Expr_app___override(x_1631, x_2);
x_1633 = lp_mathlib_CancelDenoms_mkProdPrf___closed__54;
lean_inc_ref(x_19);
x_1634 = l_Lean_Expr_const___override(x_1633, x_19);
lean_inc_ref(x_2);
x_1635 = l_Lean_Expr_app___override(x_1634, x_2);
x_1636 = l_Lean_Expr_app___override(x_1635, x_45);
x_1637 = l_Lean_Expr_app___override(x_1632, x_1636);
x_1638 = l_Lean_Expr_app___override(x_1629, x_1637);
x_1639 = l_Lean_Expr_app___override(x_1626, x_1638);
x_1640 = l_Lean_Expr_app___override(x_1623, x_1639);
lean_inc_ref(x_1534);
x_1641 = l_Lean_Expr_app___override(x_1640, x_1534);
lean_inc_ref(x_1616);
x_1642 = l_Lean_Expr_app___override(x_1641, x_1616);
x_1643 = lp_mathlib_CancelDenoms_mkProdPrf___closed__56;
x_1644 = l_Lean_Expr_const___override(x_1643, x_19);
x_1645 = l_Lean_Expr_app___override(x_1644, x_2);
x_1646 = l_Lean_Expr_app___override(x_1645, x_44);
x_1647 = l_Lean_Expr_app___override(x_1646, x_5);
x_1648 = l_Lean_Expr_app___override(x_1647, x_1530);
x_1649 = l_Lean_Expr_app___override(x_1648, x_1531);
x_1650 = l_Lean_Expr_app___override(x_1649, x_1534);
x_1651 = l_Lean_Expr_app___override(x_1650, x_1616);
x_1652 = l_Lean_Expr_app___override(x_1651, x_1535);
x_1653 = l_Lean_Expr_app___override(x_1652, x_1617);
if (lean_is_scalar(x_1618)) {
 x_1654 = lean_alloc_ctor(0, 2, 0);
} else {
 x_1654 = x_1618;
}
lean_ctor_set(x_1654, 0, x_1642);
lean_ctor_set(x_1654, 1, x_1653);
x_1655 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_1655, 0, x_1654);
return x_1655;
}
}
else
{
lean_dec_ref(x_1535);
lean_dec_ref(x_1534);
lean_dec(x_1531);
lean_dec(x_1530);
lean_dec(x_1509);
lean_dec_ref(x_45);
lean_dec_ref(x_44);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
return x_1536;
}
}
else
{
lean_dec(x_1531);
lean_dec(x_1530);
lean_dec(x_1520);
lean_dec_ref(x_1519);
lean_dec(x_1518);
lean_dec_ref(x_1517);
lean_dec(x_1516);
lean_dec(x_1509);
lean_dec_ref(x_45);
lean_dec_ref(x_44);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_1532;
}
}
}
else
{
uint8_t x_1656; 
lean_dec(x_1520);
lean_dec_ref(x_1519);
lean_dec(x_1518);
lean_dec_ref(x_1517);
lean_dec(x_1516);
lean_dec(x_1515);
lean_dec_ref(x_1514);
lean_dec_ref(x_1513);
lean_dec_ref(x_1512);
lean_dec_ref(x_1511);
lean_dec_ref(x_1510);
lean_dec(x_1509);
lean_dec_ref(x_1508);
lean_dec_ref(x_1507);
lean_dec_ref(x_1505);
lean_dec_ref(x_1504);
lean_dec_ref(x_47);
lean_dec_ref(x_45);
lean_dec_ref(x_44);
lean_dec_ref(x_43);
lean_dec_ref(x_24);
lean_dec_ref(x_19);
lean_dec(x_16);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_1656 = !lean_is_exclusive(x_1522);
if (x_1656 == 0)
{
return x_1522;
}
else
{
lean_object* x_1657; lean_object* x_1658; 
x_1657 = lean_ctor_get(x_1522, 0);
lean_inc(x_1657);
lean_dec(x_1522);
x_1658 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_1658, 0, x_1657);
return x_1658;
}
}
}
block_1873:
{
lean_object* x_1665; lean_object* x_1666; lean_object* x_1667; lean_object* x_1668; lean_object* x_1669; lean_object* x_1670; lean_object* x_1671; lean_object* x_1672; lean_object* x_1673; lean_object* x_1674; lean_object* x_1675; lean_object* x_1676; lean_object* x_1677; lean_object* x_1678; lean_object* x_1679; lean_object* x_1680; lean_object* x_1681; lean_object* x_1682; lean_object* x_1683; lean_object* x_1684; lean_object* x_1685; lean_object* x_1686; lean_object* x_1687; lean_object* x_1688; lean_object* x_1689; lean_object* x_1690; lean_object* x_1691; lean_object* x_1692; lean_object* x_1693; lean_object* x_1694; lean_object* x_1695; lean_object* x_1696; lean_object* x_1697; lean_object* x_1698; lean_object* x_1699; lean_object* x_1700; lean_object* x_1701; lean_object* x_1702; lean_object* x_1703; lean_object* x_1704; lean_object* x_1705; lean_object* x_1706; lean_object* x_1707; lean_object* x_1708; lean_object* x_1709; lean_object* x_1710; lean_object* x_1711; lean_object* x_1712; lean_object* x_1713; lean_object* x_1714; lean_object* x_1715; lean_object* x_1716; lean_object* x_1717; lean_object* x_1718; lean_object* x_1719; lean_object* x_1720; lean_object* x_1721; lean_object* x_1722; lean_object* x_1723; lean_object* x_1724; lean_object* x_1725; lean_object* x_1726; uint8_t x_1727; lean_object* x_1728; lean_object* x_1729; lean_object* x_1730; uint8_t x_1731; lean_object* x_1732; lean_object* x_1733; lean_object* x_1734; lean_object* x_1735; lean_object* x_1736; lean_object* x_1737; lean_object* x_1738; lean_object* x_1739; lean_object* x_1740; 
x_1665 = lp_mathlib_CancelDenoms_mkProdPrf___closed__58;
lean_inc_ref(x_19);
x_1666 = l_Lean_Expr_const___override(x_1665, x_19);
lean_inc_ref(x_2);
x_1667 = l_Lean_Expr_app___override(x_1666, x_2);
x_1668 = l_Lean_Expr_app___override(x_25, x_1667);
x_1669 = lp_mathlib_CancelDenoms_mkProdPrf___closed__59;
x_1670 = lp_mathlib_CancelDenoms_mkProdPrf___closed__61;
lean_inc_ref(x_19);
x_1671 = l_Lean_Expr_const___override(x_1670, x_19);
lean_inc_ref(x_2);
x_1672 = l_Lean_Expr_app___override(x_1671, x_2);
x_1673 = lp_mathlib_CancelDenoms_mkProdPrf___closed__62;
x_1674 = lp_mathlib_CancelDenoms_mkProdPrf___closed__64;
lean_inc_ref(x_19);
x_1675 = l_Lean_Expr_const___override(x_1674, x_19);
lean_inc_ref(x_2);
x_1676 = l_Lean_Expr_app___override(x_1675, x_2);
x_1677 = lp_mathlib_CancelDenoms_mkProdPrf___closed__67;
lean_inc_ref(x_19);
x_1678 = l_Lean_Expr_const___override(x_1677, x_19);
lean_inc_ref(x_2);
x_1679 = l_Lean_Expr_app___override(x_1678, x_2);
x_1680 = lp_mathlib_CancelDenoms_mkProdPrf___closed__70;
lean_inc_ref(x_19);
x_1681 = l_Lean_Expr_const___override(x_1680, x_19);
lean_inc_ref(x_2);
x_1682 = l_Lean_Expr_app___override(x_1681, x_2);
x_1683 = lp_mathlib_CancelDenoms_mkProdPrf___closed__73;
lean_inc_ref(x_19);
x_1684 = l_Lean_Expr_const___override(x_1683, x_19);
lean_inc_ref(x_2);
x_1685 = l_Lean_Expr_app___override(x_1684, x_2);
x_1686 = lp_mathlib_CancelDenoms_mkProdPrf___closed__76;
lean_inc_ref(x_19);
x_1687 = l_Lean_Expr_const___override(x_1686, x_19);
lean_inc_ref(x_2);
x_1688 = l_Lean_Expr_app___override(x_1687, x_2);
x_1689 = lp_mathlib_CancelDenoms_mkProdPrf___closed__78;
lean_inc_ref(x_19);
x_1690 = l_Lean_Expr_const___override(x_1689, x_19);
lean_inc_ref(x_2);
x_1691 = l_Lean_Expr_app___override(x_1690, x_2);
lean_inc_ref(x_3);
x_1692 = l_Lean_Expr_app___override(x_1691, x_3);
lean_inc_ref(x_1692);
x_1693 = l_Lean_Expr_app___override(x_1688, x_1692);
x_1694 = l_Lean_Expr_app___override(x_1685, x_1693);
x_1695 = l_Lean_Expr_app___override(x_1682, x_1694);
x_1696 = l_Lean_Expr_app___override(x_1679, x_1695);
lean_inc_ref(x_1696);
x_1697 = l_Lean_Expr_app___override(x_1676, x_1696);
lean_inc_ref(x_1697);
x_1698 = l_Lean_Expr_app___override(x_1672, x_1697);
lean_inc_ref(x_1698);
x_1699 = l_Lean_Expr_app___override(x_1668, x_1698);
x_1700 = lp_mathlib_CancelDenoms_mkProdPrf___closed__79;
lean_inc_ref(x_19);
lean_inc(x_1);
x_1701 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_1701, 0, x_1);
lean_ctor_set(x_1701, 1, x_19);
lean_inc(x_1);
x_1702 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_1702, 0, x_1);
lean_ctor_set(x_1702, 1, x_1701);
lean_inc_ref(x_1702);
x_1703 = l_Lean_Expr_const___override(x_1700, x_1702);
lean_inc_ref(x_2);
x_1704 = l_Lean_Expr_app___override(x_1703, x_2);
lean_inc_ref(x_2);
x_1705 = l_Lean_Expr_app___override(x_1704, x_2);
lean_inc_ref(x_2);
x_1706 = l_Lean_Expr_app___override(x_1705, x_2);
x_1707 = lp_mathlib_CancelDenoms_mkProdPrf___closed__81;
lean_inc_ref(x_19);
x_1708 = l_Lean_Expr_const___override(x_1707, x_19);
lean_inc_ref(x_2);
x_1709 = l_Lean_Expr_app___override(x_1708, x_2);
lean_inc_ref(x_1709);
x_1710 = l_Lean_Expr_app___override(x_1709, x_1698);
lean_inc_ref(x_1706);
x_1711 = l_Lean_Expr_app___override(x_1706, x_1710);
lean_inc_ref(x_5);
lean_inc_ref(x_1711);
x_1712 = l_Lean_Expr_app___override(x_1711, x_5);
lean_inc_ref(x_7);
x_1713 = l_Lean_Expr_app___override(x_1712, x_7);
x_1714 = lp_mathlib_CancelDenoms_mkProdPrf___closed__83;
lean_inc_ref(x_24);
x_1715 = l_Lean_Expr_const___override(x_1714, x_24);
lean_inc_ref(x_2);
x_1716 = l_Lean_Expr_app___override(x_1715, x_2);
x_1717 = lp_mathlib_CancelDenoms_mkProdPrf___closed__85;
lean_inc_ref(x_24);
x_1718 = l_Lean_Expr_const___override(x_1717, x_24);
lean_inc_ref(x_2);
x_1719 = l_Lean_Expr_app___override(x_1718, x_2);
x_1720 = l_Lean_Expr_app___override(x_1709, x_1699);
x_1721 = l_Lean_Expr_app___override(x_1706, x_1720);
lean_inc_ref(x_5);
x_1722 = l_Lean_Expr_app___override(x_1721, x_5);
lean_inc_ref(x_7);
x_1723 = l_Lean_Expr_app___override(x_1722, x_7);
x_1724 = l_Lean_Expr_app___override(x_1719, x_1723);
x_1725 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_1725, 0, x_1713);
lean_ctor_set(x_1725, 1, x_1724);
lean_inc_ref(x_2);
x_1726 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_1726, 0, x_2);
x_1727 = 0;
x_1728 = lean_box(0);
x_1729 = lean_box(x_1727);
lean_inc_ref(x_7);
lean_inc_ref(x_44);
lean_inc_ref(x_2);
lean_inc_ref(x_19);
lean_inc_ref(x_1726);
x_1730 = lean_alloc_closure((void*)(lp_mathlib_CancelDenoms_mkProdPrf___lam__2___boxed), 13, 8);
lean_closure_set(x_1730, 0, x_1726);
lean_closure_set(x_1730, 1, x_1729);
lean_closure_set(x_1730, 2, x_1728);
lean_closure_set(x_1730, 3, x_19);
lean_closure_set(x_1730, 4, x_2);
lean_closure_set(x_1730, 5, x_31);
lean_closure_set(x_1730, 6, x_44);
lean_closure_set(x_1730, 7, x_7);
x_1731 = 0;
x_1732 = lean_box(x_1727);
x_1733 = lean_box(x_1731);
lean_inc_ref(x_7);
lean_inc_ref(x_43);
lean_inc_ref(x_19);
lean_inc_ref(x_2);
lean_inc_ref(x_1702);
lean_inc_ref(x_1726);
x_1734 = lean_alloc_closure((void*)(lp_mathlib_CancelDenoms_mkProdPrf___lam__3___boxed), 15, 10);
lean_closure_set(x_1734, 0, x_1726);
lean_closure_set(x_1734, 1, x_1732);
lean_closure_set(x_1734, 2, x_1728);
lean_closure_set(x_1734, 3, x_1702);
lean_closure_set(x_1734, 4, x_2);
lean_closure_set(x_1734, 5, x_19);
lean_closure_set(x_1734, 6, x_35);
lean_closure_set(x_1734, 7, x_43);
lean_closure_set(x_1734, 8, x_7);
lean_closure_set(x_1734, 9, x_1733);
x_1735 = lean_box(x_1727);
x_1736 = lean_box(x_1731);
lean_inc_ref(x_7);
lean_inc_ref(x_1711);
lean_inc_ref(x_1726);
x_1737 = lean_alloc_closure((void*)(lp_mathlib_CancelDenoms_mkProdPrf___lam__4___boxed), 11, 6);
lean_closure_set(x_1737, 0, x_1726);
lean_closure_set(x_1737, 1, x_1735);
lean_closure_set(x_1737, 2, x_1728);
lean_closure_set(x_1737, 3, x_1711);
lean_closure_set(x_1737, 4, x_7);
lean_closure_set(x_1737, 5, x_1736);
x_1738 = lean_box(x_1727);
x_1739 = lean_box(x_1731);
lean_inc_ref(x_7);
lean_inc_ref(x_45);
lean_inc_ref(x_19);
lean_inc_ref(x_2);
lean_inc_ref(x_1702);
lean_inc_ref(x_1726);
x_1740 = lean_alloc_closure((void*)(lp_mathlib_CancelDenoms_mkProdPrf___lam__5___boxed), 15, 10);
lean_closure_set(x_1740, 0, x_1726);
lean_closure_set(x_1740, 1, x_1738);
lean_closure_set(x_1740, 2, x_1728);
lean_closure_set(x_1740, 3, x_1702);
lean_closure_set(x_1740, 4, x_2);
lean_closure_set(x_1740, 5, x_19);
lean_closure_set(x_1740, 6, x_27);
lean_closure_set(x_1740, 7, x_45);
lean_closure_set(x_1740, 8, x_7);
lean_closure_set(x_1740, 9, x_1739);
if (lean_obj_tag(x_6) == 1)
{
lean_object* x_1741; lean_object* x_1742; lean_object* x_1743; lean_object* x_1744; lean_object* x_1745; lean_object* x_1746; 
x_1741 = lean_ctor_get(x_6, 1);
x_1742 = lean_ctor_get(x_6, 2);
x_1743 = lean_box(x_1727);
x_1744 = lean_box(x_1731);
lean_inc_ref(x_7);
lean_inc_ref(x_1697);
lean_inc_ref(x_19);
lean_inc_ref(x_2);
lean_inc_ref(x_1702);
x_1745 = lean_alloc_closure((void*)(lp_mathlib_CancelDenoms_mkProdPrf___lam__6___boxed), 15, 10);
lean_closure_set(x_1745, 0, x_1726);
lean_closure_set(x_1745, 1, x_1743);
lean_closure_set(x_1745, 2, x_1728);
lean_closure_set(x_1745, 3, x_1702);
lean_closure_set(x_1745, 4, x_2);
lean_closure_set(x_1745, 5, x_19);
lean_closure_set(x_1745, 6, x_1669);
lean_closure_set(x_1745, 7, x_1697);
lean_closure_set(x_1745, 8, x_7);
lean_closure_set(x_1745, 9, x_1744);
lean_inc(x_1663);
lean_inc_ref(x_1662);
lean_inc(x_1661);
lean_inc_ref(x_1660);
x_1746 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00CancelDenoms_mkProdPrf_spec__2___redArg(x_1745, x_1731, x_1660, x_1661, x_1662, x_1663);
if (lean_obj_tag(x_1746) == 0)
{
lean_object* x_1747; lean_object* x_1748; lean_object* x_1749; uint8_t x_1750; 
x_1747 = lean_ctor_get(x_1746, 0);
lean_inc(x_1747);
lean_dec_ref(x_1746);
x_1748 = lean_ctor_get(x_1747, 1);
lean_inc(x_1748);
x_1749 = lean_ctor_get(x_1748, 1);
x_1750 = lean_unbox(x_1749);
if (x_1750 == 0)
{
lean_dec(x_1748);
lean_dec(x_1747);
lean_dec_ref(x_1697);
lean_inc(x_1742);
lean_inc(x_1741);
x_1503 = x_1740;
x_1504 = x_1734;
x_1505 = x_1737;
x_1506 = x_1731;
x_1507 = x_1730;
x_1508 = x_1716;
x_1509 = x_1702;
x_1510 = x_1725;
x_1511 = x_1692;
x_1512 = x_1711;
x_1513 = x_1696;
x_1514 = x_1673;
x_1515 = x_1741;
x_1516 = x_1742;
x_1517 = x_1660;
x_1518 = x_1661;
x_1519 = x_1662;
x_1520 = x_1663;
x_1521 = lean_box(0);
goto block_1659;
}
else
{
lean_object* x_1751; lean_object* x_1752; lean_object* x_1753; 
lean_inc(x_1742);
lean_inc(x_1741);
lean_dec_ref(x_1740);
lean_dec_ref(x_1737);
lean_dec_ref(x_1734);
lean_dec_ref(x_1730);
lean_dec_ref(x_1725);
lean_dec_ref(x_1716);
lean_dec_ref(x_1711);
lean_dec_ref(x_1696);
lean_dec_ref(x_47);
lean_dec_ref(x_45);
lean_dec_ref(x_44);
lean_dec_ref(x_43);
lean_dec_ref(x_24);
lean_dec(x_16);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
x_1751 = lean_ctor_get(x_1747, 0);
lean_inc(x_1751);
lean_dec(x_1747);
x_1752 = lean_ctor_get(x_1748, 0);
lean_inc(x_1752);
lean_dec(x_1748);
lean_inc(x_1663);
lean_inc_ref(x_1662);
lean_inc(x_1661);
lean_inc_ref(x_1660);
lean_inc(x_1751);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc(x_1);
x_1753 = lp_mathlib_CancelDenoms_mkProdPrf(x_1, x_2, x_3, x_4, x_5, x_1741, x_1751, x_1660, x_1661, x_1662, x_1663);
if (lean_obj_tag(x_1753) == 0)
{
lean_object* x_1754; lean_object* x_1755; lean_object* x_1756; lean_object* x_1757; 
x_1754 = lean_ctor_get(x_1753, 0);
lean_inc(x_1754);
lean_dec_ref(x_1753);
x_1755 = lean_ctor_get(x_1754, 0);
lean_inc_ref(x_1755);
x_1756 = lean_ctor_get(x_1754, 1);
lean_inc_ref(x_1756);
lean_dec(x_1754);
lean_inc(x_1752);
lean_inc_ref(x_5);
lean_inc_ref(x_2);
x_1757 = lp_mathlib_CancelDenoms_mkProdPrf(x_1, x_2, x_3, x_4, x_5, x_1742, x_1752, x_1660, x_1661, x_1662, x_1663);
if (lean_obj_tag(x_1757) == 0)
{
uint8_t x_1758; 
x_1758 = !lean_is_exclusive(x_1757);
if (x_1758 == 0)
{
lean_object* x_1759; uint8_t x_1760; 
x_1759 = lean_ctor_get(x_1757, 0);
x_1760 = !lean_is_exclusive(x_1759);
if (x_1760 == 0)
{
lean_object* x_1761; lean_object* x_1762; lean_object* x_1763; lean_object* x_1764; lean_object* x_1765; lean_object* x_1766; lean_object* x_1767; lean_object* x_1768; lean_object* x_1769; lean_object* x_1770; lean_object* x_1771; lean_object* x_1772; lean_object* x_1773; lean_object* x_1774; lean_object* x_1775; lean_object* x_1776; lean_object* x_1777; lean_object* x_1778; lean_object* x_1779; lean_object* x_1780; lean_object* x_1781; lean_object* x_1782; lean_object* x_1783; lean_object* x_1784; lean_object* x_1785; lean_object* x_1786; lean_object* x_1787; lean_object* x_1788; lean_object* x_1789; lean_object* x_1790; lean_object* x_1791; lean_object* x_1792; lean_object* x_1793; 
x_1761 = lean_ctor_get(x_1759, 0);
x_1762 = lean_ctor_get(x_1759, 1);
x_1763 = lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__0;
x_1764 = l_Lean_Expr_const___override(x_1763, x_1702);
lean_inc_ref(x_2);
x_1765 = l_Lean_Expr_app___override(x_1764, x_2);
lean_inc_ref(x_2);
x_1766 = l_Lean_Expr_app___override(x_1765, x_2);
lean_inc_ref(x_2);
x_1767 = l_Lean_Expr_app___override(x_1766, x_2);
x_1768 = lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__2;
lean_inc_ref(x_19);
x_1769 = l_Lean_Expr_const___override(x_1768, x_19);
lean_inc_ref(x_2);
x_1770 = l_Lean_Expr_app___override(x_1769, x_2);
x_1771 = lp_mathlib_CancelDenoms_mkProdPrf___closed__86;
lean_inc_ref(x_19);
x_1772 = l_Lean_Expr_const___override(x_1771, x_19);
lean_inc_ref(x_2);
x_1773 = l_Lean_Expr_app___override(x_1772, x_2);
x_1774 = l_Lean_Expr_app___override(x_1773, x_1697);
x_1775 = l_Lean_Expr_app___override(x_1770, x_1774);
x_1776 = l_Lean_Expr_app___override(x_1767, x_1775);
lean_inc_ref(x_1755);
x_1777 = l_Lean_Expr_app___override(x_1776, x_1755);
lean_inc_ref(x_1761);
x_1778 = l_Lean_Expr_app___override(x_1777, x_1761);
x_1779 = lp_mathlib_CancelDenoms_mkProdPrf___closed__88;
lean_inc_ref(x_19);
x_1780 = l_Lean_Expr_const___override(x_1779, x_19);
lean_inc_ref(x_2);
x_1781 = l_Lean_Expr_app___override(x_1780, x_2);
x_1782 = lp_mathlib_CancelDenoms_mkProdPrf___closed__89;
x_1783 = l_Lean_Expr_const___override(x_1782, x_19);
x_1784 = l_Lean_Expr_app___override(x_1783, x_2);
x_1785 = l_Lean_Expr_app___override(x_1784, x_1692);
x_1786 = l_Lean_Expr_app___override(x_1781, x_1785);
x_1787 = l_Lean_Expr_app___override(x_1786, x_5);
x_1788 = l_Lean_Expr_app___override(x_1787, x_1751);
x_1789 = l_Lean_Expr_app___override(x_1788, x_1752);
x_1790 = l_Lean_Expr_app___override(x_1789, x_1755);
x_1791 = l_Lean_Expr_app___override(x_1790, x_1761);
x_1792 = l_Lean_Expr_app___override(x_1791, x_1756);
x_1793 = l_Lean_Expr_app___override(x_1792, x_1762);
lean_ctor_set(x_1759, 1, x_1793);
lean_ctor_set(x_1759, 0, x_1778);
return x_1757;
}
else
{
lean_object* x_1794; lean_object* x_1795; lean_object* x_1796; lean_object* x_1797; lean_object* x_1798; lean_object* x_1799; lean_object* x_1800; lean_object* x_1801; lean_object* x_1802; lean_object* x_1803; lean_object* x_1804; lean_object* x_1805; lean_object* x_1806; lean_object* x_1807; lean_object* x_1808; lean_object* x_1809; lean_object* x_1810; lean_object* x_1811; lean_object* x_1812; lean_object* x_1813; lean_object* x_1814; lean_object* x_1815; lean_object* x_1816; lean_object* x_1817; lean_object* x_1818; lean_object* x_1819; lean_object* x_1820; lean_object* x_1821; lean_object* x_1822; lean_object* x_1823; lean_object* x_1824; lean_object* x_1825; lean_object* x_1826; lean_object* x_1827; 
x_1794 = lean_ctor_get(x_1759, 0);
x_1795 = lean_ctor_get(x_1759, 1);
lean_inc(x_1795);
lean_inc(x_1794);
lean_dec(x_1759);
x_1796 = lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__0;
x_1797 = l_Lean_Expr_const___override(x_1796, x_1702);
lean_inc_ref(x_2);
x_1798 = l_Lean_Expr_app___override(x_1797, x_2);
lean_inc_ref(x_2);
x_1799 = l_Lean_Expr_app___override(x_1798, x_2);
lean_inc_ref(x_2);
x_1800 = l_Lean_Expr_app___override(x_1799, x_2);
x_1801 = lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__2;
lean_inc_ref(x_19);
x_1802 = l_Lean_Expr_const___override(x_1801, x_19);
lean_inc_ref(x_2);
x_1803 = l_Lean_Expr_app___override(x_1802, x_2);
x_1804 = lp_mathlib_CancelDenoms_mkProdPrf___closed__86;
lean_inc_ref(x_19);
x_1805 = l_Lean_Expr_const___override(x_1804, x_19);
lean_inc_ref(x_2);
x_1806 = l_Lean_Expr_app___override(x_1805, x_2);
x_1807 = l_Lean_Expr_app___override(x_1806, x_1697);
x_1808 = l_Lean_Expr_app___override(x_1803, x_1807);
x_1809 = l_Lean_Expr_app___override(x_1800, x_1808);
lean_inc_ref(x_1755);
x_1810 = l_Lean_Expr_app___override(x_1809, x_1755);
lean_inc_ref(x_1794);
x_1811 = l_Lean_Expr_app___override(x_1810, x_1794);
x_1812 = lp_mathlib_CancelDenoms_mkProdPrf___closed__88;
lean_inc_ref(x_19);
x_1813 = l_Lean_Expr_const___override(x_1812, x_19);
lean_inc_ref(x_2);
x_1814 = l_Lean_Expr_app___override(x_1813, x_2);
x_1815 = lp_mathlib_CancelDenoms_mkProdPrf___closed__89;
x_1816 = l_Lean_Expr_const___override(x_1815, x_19);
x_1817 = l_Lean_Expr_app___override(x_1816, x_2);
x_1818 = l_Lean_Expr_app___override(x_1817, x_1692);
x_1819 = l_Lean_Expr_app___override(x_1814, x_1818);
x_1820 = l_Lean_Expr_app___override(x_1819, x_5);
x_1821 = l_Lean_Expr_app___override(x_1820, x_1751);
x_1822 = l_Lean_Expr_app___override(x_1821, x_1752);
x_1823 = l_Lean_Expr_app___override(x_1822, x_1755);
x_1824 = l_Lean_Expr_app___override(x_1823, x_1794);
x_1825 = l_Lean_Expr_app___override(x_1824, x_1756);
x_1826 = l_Lean_Expr_app___override(x_1825, x_1795);
x_1827 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_1827, 0, x_1811);
lean_ctor_set(x_1827, 1, x_1826);
lean_ctor_set(x_1757, 0, x_1827);
return x_1757;
}
}
else
{
lean_object* x_1828; lean_object* x_1829; lean_object* x_1830; lean_object* x_1831; lean_object* x_1832; lean_object* x_1833; lean_object* x_1834; lean_object* x_1835; lean_object* x_1836; lean_object* x_1837; lean_object* x_1838; lean_object* x_1839; lean_object* x_1840; lean_object* x_1841; lean_object* x_1842; lean_object* x_1843; lean_object* x_1844; lean_object* x_1845; lean_object* x_1846; lean_object* x_1847; lean_object* x_1848; lean_object* x_1849; lean_object* x_1850; lean_object* x_1851; lean_object* x_1852; lean_object* x_1853; lean_object* x_1854; lean_object* x_1855; lean_object* x_1856; lean_object* x_1857; lean_object* x_1858; lean_object* x_1859; lean_object* x_1860; lean_object* x_1861; lean_object* x_1862; lean_object* x_1863; lean_object* x_1864; 
x_1828 = lean_ctor_get(x_1757, 0);
lean_inc(x_1828);
lean_dec(x_1757);
x_1829 = lean_ctor_get(x_1828, 0);
lean_inc_ref(x_1829);
x_1830 = lean_ctor_get(x_1828, 1);
lean_inc_ref(x_1830);
if (lean_is_exclusive(x_1828)) {
 lean_ctor_release(x_1828, 0);
 lean_ctor_release(x_1828, 1);
 x_1831 = x_1828;
} else {
 lean_dec_ref(x_1828);
 x_1831 = lean_box(0);
}
x_1832 = lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__0;
x_1833 = l_Lean_Expr_const___override(x_1832, x_1702);
lean_inc_ref(x_2);
x_1834 = l_Lean_Expr_app___override(x_1833, x_2);
lean_inc_ref(x_2);
x_1835 = l_Lean_Expr_app___override(x_1834, x_2);
lean_inc_ref(x_2);
x_1836 = l_Lean_Expr_app___override(x_1835, x_2);
x_1837 = lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__2;
lean_inc_ref(x_19);
x_1838 = l_Lean_Expr_const___override(x_1837, x_19);
lean_inc_ref(x_2);
x_1839 = l_Lean_Expr_app___override(x_1838, x_2);
x_1840 = lp_mathlib_CancelDenoms_mkProdPrf___closed__86;
lean_inc_ref(x_19);
x_1841 = l_Lean_Expr_const___override(x_1840, x_19);
lean_inc_ref(x_2);
x_1842 = l_Lean_Expr_app___override(x_1841, x_2);
x_1843 = l_Lean_Expr_app___override(x_1842, x_1697);
x_1844 = l_Lean_Expr_app___override(x_1839, x_1843);
x_1845 = l_Lean_Expr_app___override(x_1836, x_1844);
lean_inc_ref(x_1755);
x_1846 = l_Lean_Expr_app___override(x_1845, x_1755);
lean_inc_ref(x_1829);
x_1847 = l_Lean_Expr_app___override(x_1846, x_1829);
x_1848 = lp_mathlib_CancelDenoms_mkProdPrf___closed__88;
lean_inc_ref(x_19);
x_1849 = l_Lean_Expr_const___override(x_1848, x_19);
lean_inc_ref(x_2);
x_1850 = l_Lean_Expr_app___override(x_1849, x_2);
x_1851 = lp_mathlib_CancelDenoms_mkProdPrf___closed__89;
x_1852 = l_Lean_Expr_const___override(x_1851, x_19);
x_1853 = l_Lean_Expr_app___override(x_1852, x_2);
x_1854 = l_Lean_Expr_app___override(x_1853, x_1692);
x_1855 = l_Lean_Expr_app___override(x_1850, x_1854);
x_1856 = l_Lean_Expr_app___override(x_1855, x_5);
x_1857 = l_Lean_Expr_app___override(x_1856, x_1751);
x_1858 = l_Lean_Expr_app___override(x_1857, x_1752);
x_1859 = l_Lean_Expr_app___override(x_1858, x_1755);
x_1860 = l_Lean_Expr_app___override(x_1859, x_1829);
x_1861 = l_Lean_Expr_app___override(x_1860, x_1756);
x_1862 = l_Lean_Expr_app___override(x_1861, x_1830);
if (lean_is_scalar(x_1831)) {
 x_1863 = lean_alloc_ctor(0, 2, 0);
} else {
 x_1863 = x_1831;
}
lean_ctor_set(x_1863, 0, x_1847);
lean_ctor_set(x_1863, 1, x_1862);
x_1864 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_1864, 0, x_1863);
return x_1864;
}
}
else
{
lean_dec_ref(x_1756);
lean_dec_ref(x_1755);
lean_dec(x_1752);
lean_dec(x_1751);
lean_dec_ref(x_1702);
lean_dec_ref(x_1697);
lean_dec_ref(x_1692);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec_ref(x_2);
return x_1757;
}
}
else
{
lean_dec(x_1752);
lean_dec(x_1751);
lean_dec(x_1742);
lean_dec_ref(x_1702);
lean_dec_ref(x_1697);
lean_dec_ref(x_1692);
lean_dec(x_1663);
lean_dec_ref(x_1662);
lean_dec(x_1661);
lean_dec_ref(x_1660);
lean_dec_ref(x_19);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_1753;
}
}
}
else
{
uint8_t x_1865; 
lean_dec_ref(x_1740);
lean_dec_ref(x_1737);
lean_dec_ref(x_1734);
lean_dec_ref(x_1730);
lean_dec_ref(x_1725);
lean_dec_ref(x_1716);
lean_dec_ref(x_1711);
lean_dec_ref(x_1702);
lean_dec_ref(x_1697);
lean_dec_ref(x_1696);
lean_dec_ref(x_1692);
lean_dec(x_1663);
lean_dec_ref(x_1662);
lean_dec(x_1661);
lean_dec_ref(x_1660);
lean_dec_ref(x_47);
lean_dec_ref(x_45);
lean_dec_ref(x_44);
lean_dec_ref(x_43);
lean_dec_ref(x_24);
lean_dec_ref(x_19);
lean_dec(x_16);
lean_dec_ref(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_1865 = !lean_is_exclusive(x_1746);
if (x_1865 == 0)
{
return x_1746;
}
else
{
lean_object* x_1866; lean_object* x_1867; 
x_1866 = lean_ctor_get(x_1746, 0);
lean_inc(x_1866);
lean_dec(x_1746);
x_1867 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_1867, 0, x_1866);
return x_1867;
}
}
}
else
{
lean_dec_ref(x_1726);
lean_dec_ref(x_1697);
if (lean_obj_tag(x_6) == 1)
{
lean_object* x_1868; lean_object* x_1869; 
x_1868 = lean_ctor_get(x_6, 1);
x_1869 = lean_ctor_get(x_6, 2);
lean_inc(x_1869);
lean_inc(x_1868);
x_1503 = x_1740;
x_1504 = x_1734;
x_1505 = x_1737;
x_1506 = x_1731;
x_1507 = x_1730;
x_1508 = x_1716;
x_1509 = x_1702;
x_1510 = x_1725;
x_1511 = x_1692;
x_1512 = x_1711;
x_1513 = x_1696;
x_1514 = x_1673;
x_1515 = x_1868;
x_1516 = x_1869;
x_1517 = x_1660;
x_1518 = x_1661;
x_1519 = x_1662;
x_1520 = x_1663;
x_1521 = lean_box(0);
goto block_1659;
}
else
{
lean_dec_ref(x_1740);
lean_dec_ref(x_45);
if (lean_obj_tag(x_6) == 1)
{
lean_object* x_1870; 
x_1870 = lean_ctor_get(x_6, 1);
if (lean_obj_tag(x_1870) == 1)
{
lean_object* x_1871; lean_object* x_1872; 
x_1871 = lean_ctor_get(x_6, 2);
x_1872 = lean_ctor_get(x_1870, 0);
lean_inc(x_1871);
lean_inc(x_1872);
lean_inc_ref(x_1870);
x_1459 = x_1734;
x_1460 = x_1731;
x_1461 = x_1737;
x_1462 = x_1730;
x_1463 = x_1716;
x_1464 = x_1725;
x_1465 = x_1702;
x_1466 = x_1692;
x_1467 = x_1673;
x_1468 = x_1696;
x_1469 = x_1711;
x_1470 = x_1870;
x_1471 = x_1872;
x_1472 = x_1871;
x_1473 = x_1660;
x_1474 = x_1661;
x_1475 = x_1662;
x_1476 = x_1663;
x_1477 = lean_box(0);
goto block_1502;
}
else
{
lean_dec_ref(x_1737);
x_1328 = x_1734;
x_1329 = x_1731;
x_1330 = x_1730;
x_1331 = x_1716;
x_1332 = x_1725;
x_1333 = x_1702;
x_1334 = x_1692;
x_1335 = x_1673;
x_1336 = x_1696;
x_1337 = x_1711;
x_1338 = x_1660;
x_1339 = x_1661;
x_1340 = x_1662;
x_1341 = x_1663;
x_1342 = lean_box(0);
goto block_1346;
}
}
else
{
lean_dec_ref(x_1737);
x_1328 = x_1734;
x_1329 = x_1731;
x_1330 = x_1730;
x_1331 = x_1716;
x_1332 = x_1725;
x_1333 = x_1702;
x_1334 = x_1692;
x_1335 = x_1673;
x_1336 = x_1696;
x_1337 = x_1711;
x_1338 = x_1660;
x_1339 = x_1661;
x_1340 = x_1662;
x_1341 = x_1663;
x_1342 = lean_box(0);
goto block_1346;
}
}
}
}
}
else
{
uint8_t x_1888; 
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_1888 = !lean_is_exclusive(x_14);
if (x_1888 == 0)
{
return x_14;
}
else
{
lean_object* x_1889; lean_object* x_1890; 
x_1889 = lean_ctor_get(x_14, 0);
lean_inc(x_1889);
lean_dec(x_14);
x_1890 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_1890, 0, x_1889);
return x_1890;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00CancelDenoms_mkProdPrf_spec__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
uint8_t x_9; lean_object* x_10; 
x_9 = lean_unbox(x_3);
x_10 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00CancelDenoms_mkProdPrf_spec__2(x_1, x_2, x_9, x_4, x_5, x_6, x_7);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_isTracingEnabledFor___at___00CancelDenoms_mkProdPrf_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Lean_isTracingEnabledFor___at___00CancelDenoms_mkProdPrf_spec__0(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_isTracingEnabledFor___at___00CancelDenoms_mkProdPrf_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Lean_isTracingEnabledFor___at___00CancelDenoms_mkProdPrf_spec__0___redArg(x_1, x_2);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00CancelDenoms_mkProdPrf_spec__2___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
uint8_t x_8; lean_object* x_9; 
x_8 = lean_unbox(x_2);
x_9 = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00CancelDenoms_mkProdPrf_spec__2___redArg(x_1, x_8, x_3, x_4, x_5, x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Lean_instantiateMVars___at___00CancelDenoms_mkProdPrf_spec__1___redArg(x_1, x_2);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_mkProdPrf___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_CancelDenoms_mkProdPrf(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
return x_13;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_deriveThms___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("div_div_eq_mul_div", 18, 18);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_deriveThms___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_deriveThms___closed__0;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_deriveThms___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("div_neg", 7, 7);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_deriveThms___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_deriveThms___closed__2;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_deriveThms___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_CancelDenoms_deriveThms___closed__3;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_deriveThms___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_deriveThms___closed__4;
x_2 = lp_mathlib_CancelDenoms_deriveThms___closed__1;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_deriveThms() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_CancelDenoms_deriveThms___closed__5;
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_derive___lam__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("derive_trans", 12, 12);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_derive___lam__0___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("derive_trans₂", 15, 13);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_derive___lam__0___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(3u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_derive___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_13; lean_object* x_14; lean_object* x_20; 
if (lean_obj_tag(x_5) == 0)
{
if (lean_obj_tag(x_6) == 0)
{
lean_object* x_33; 
lean_dec(x_11);
lean_dec_ref(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_3);
x_33 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_33);
lean_dec_ref(x_2);
x_13 = x_33;
x_14 = lean_box(0);
goto block_19;
}
else
{
lean_object* x_34; 
x_34 = lean_ctor_get(x_6, 0);
lean_inc(x_34);
lean_dec_ref(x_6);
x_20 = x_34;
goto block_32;
}
}
else
{
if (lean_obj_tag(x_6) == 0)
{
lean_object* x_35; 
x_35 = lean_ctor_get(x_5, 0);
lean_inc(x_35);
lean_dec_ref(x_5);
x_20 = x_35;
goto block_32;
}
else
{
lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; 
x_36 = lean_ctor_get(x_5, 0);
lean_inc(x_36);
lean_dec_ref(x_5);
x_37 = lean_ctor_get(x_6, 0);
lean_inc(x_37);
lean_dec_ref(x_6);
x_38 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_38);
lean_dec_ref(x_2);
x_39 = lp_mathlib_CancelDenoms_derive___lam__0___closed__1;
x_40 = l_Lean_Name_mkStr2(x_3, x_39);
x_41 = lp_mathlib_CancelDenoms_derive___lam__0___closed__2;
x_42 = lean_array_push(x_41, x_36);
x_43 = lean_array_push(x_42, x_37);
x_44 = lean_array_push(x_43, x_38);
x_45 = l_Lean_Meta_mkAppM(x_40, x_44, x_8, x_9, x_10, x_11);
if (lean_obj_tag(x_45) == 0)
{
lean_object* x_46; 
x_46 = lean_ctor_get(x_45, 0);
lean_inc(x_46);
lean_dec_ref(x_45);
x_13 = x_46;
x_14 = lean_box(0);
goto block_19;
}
else
{
uint8_t x_47; 
lean_dec(x_1);
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
block_19:
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_1);
lean_ctor_set(x_15, 1, x_13);
x_16 = lean_box(0);
x_17 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_17, 0, x_15);
lean_ctor_set(x_17, 1, x_16);
x_18 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_18, 0, x_17);
return x_18;
}
block_32:
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; 
x_21 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_21);
lean_dec_ref(x_2);
x_22 = lp_mathlib_CancelDenoms_derive___lam__0___closed__0;
x_23 = l_Lean_Name_mkStr2(x_3, x_22);
x_24 = lean_mk_empty_array_with_capacity(x_4);
x_25 = lean_array_push(x_24, x_20);
x_26 = lean_array_push(x_25, x_21);
x_27 = l_Lean_Meta_mkAppM(x_23, x_26, x_8, x_9, x_10, x_11);
if (lean_obj_tag(x_27) == 0)
{
lean_object* x_28; 
x_28 = lean_ctor_get(x_27, 0);
lean_inc(x_28);
lean_dec_ref(x_27);
x_13 = x_28;
x_14 = lean_box(0);
goto block_19;
}
else
{
uint8_t x_29; 
lean_dec(x_1);
x_29 = !lean_is_exclusive(x_27);
if (x_29 == 0)
{
return x_27;
}
else
{
lean_object* x_30; lean_object* x_31; 
x_30 = lean_ctor_get(x_27, 0);
lean_inc(x_30);
lean_dec(x_27);
x_31 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_31, 0, x_30);
return x_31;
}
}
}
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_derive___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("CancelDenoms.derive failed to normalize ", 40, 40);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_derive___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_derive___closed__0;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_derive___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(".\n", 2, 2);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_derive___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_derive___closed__2;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_derive___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__13;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_derive___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("pf : ", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_derive___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_derive___closed__5;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_derive___closed__7() {
_start:
{
lean_object* x_1; 
x_1 = l_Lean_Meta_Simp_defaultMaxSteps;
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_derive___closed__8() {
_start:
{
uint8_t x_1; uint8_t x_2; uint8_t x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_1 = 0;
x_2 = 1;
x_3 = 0;
x_4 = lean_unsigned_to_nat(2u);
x_5 = lp_mathlib_CancelDenoms_derive___closed__7;
x_6 = lean_alloc_ctor(0, 2, 27);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_4);
lean_ctor_set_uint8(x_6, sizeof(void*)*2, x_3);
lean_ctor_set_uint8(x_6, sizeof(void*)*2 + 1, x_2);
lean_ctor_set_uint8(x_6, sizeof(void*)*2 + 2, x_3);
lean_ctor_set_uint8(x_6, sizeof(void*)*2 + 3, x_2);
lean_ctor_set_uint8(x_6, sizeof(void*)*2 + 4, x_2);
lean_ctor_set_uint8(x_6, sizeof(void*)*2 + 5, x_2);
lean_ctor_set_uint8(x_6, sizeof(void*)*2 + 6, x_1);
lean_ctor_set_uint8(x_6, sizeof(void*)*2 + 7, x_2);
lean_ctor_set_uint8(x_6, sizeof(void*)*2 + 8, x_2);
lean_ctor_set_uint8(x_6, sizeof(void*)*2 + 9, x_3);
lean_ctor_set_uint8(x_6, sizeof(void*)*2 + 10, x_3);
lean_ctor_set_uint8(x_6, sizeof(void*)*2 + 11, x_3);
lean_ctor_set_uint8(x_6, sizeof(void*)*2 + 12, x_2);
lean_ctor_set_uint8(x_6, sizeof(void*)*2 + 13, x_2);
lean_ctor_set_uint8(x_6, sizeof(void*)*2 + 14, x_3);
lean_ctor_set_uint8(x_6, sizeof(void*)*2 + 15, x_3);
lean_ctor_set_uint8(x_6, sizeof(void*)*2 + 16, x_3);
lean_ctor_set_uint8(x_6, sizeof(void*)*2 + 17, x_2);
lean_ctor_set_uint8(x_6, sizeof(void*)*2 + 18, x_2);
lean_ctor_set_uint8(x_6, sizeof(void*)*2 + 19, x_2);
lean_ctor_set_uint8(x_6, sizeof(void*)*2 + 20, x_2);
lean_ctor_set_uint8(x_6, sizeof(void*)*2 + 21, x_2);
lean_ctor_set_uint8(x_6, sizeof(void*)*2 + 22, x_2);
lean_ctor_set_uint8(x_6, sizeof(void*)*2 + 23, x_2);
lean_ctor_set_uint8(x_6, sizeof(void*)*2 + 24, x_2);
lean_ctor_set_uint8(x_6, sizeof(void*)*2 + 25, x_2);
lean_ctor_set_uint8(x_6, sizeof(void*)*2 + 26, x_3);
return x_6;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_derive___closed__9() {
_start:
{
lean_object* x_1; 
x_1 = l_Array_empty(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_derive___closed__10() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lean_unsigned_to_nat(16u);
x_3 = lean_mk_array(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_derive___closed__11() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_derive___closed__10;
x_2 = lean_unsigned_to_nat(0u);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_derive___closed__12() {
_start:
{
lean_object* x_1; 
x_1 = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_derive___closed__13() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_derive___closed__12;
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_derive___closed__14() {
_start:
{
lean_object* x_1; lean_object* x_2; uint8_t x_3; lean_object* x_4; 
x_1 = lp_mathlib_CancelDenoms_derive___closed__13;
x_2 = lp_mathlib_CancelDenoms_derive___closed__11;
x_3 = 1;
x_4 = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_1);
lean_ctor_set_uint8(x_4, sizeof(void*)*2, x_3);
return x_4;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_derive___closed__15() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("e norm_num'd = ", 15, 15);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_derive___closed__16() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_derive___closed__15;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_derive___closed__17() {
_start:
{
lean_object* x_1; 
x_1 = l_Lean_Meta_Simp_neutralConfig;
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_derive___closed__18() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("e simplified = ", 15, 15);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_derive___closed__19() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_derive___closed__18;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_derive___closed__20() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("e = ", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_derive___closed__21() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_derive___closed__20;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_derive(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_280; lean_object* x_281; lean_object* x_282; lean_object* x_283; lean_object* x_284; lean_object* x_285; lean_object* x_318; lean_object* x_319; lean_object* x_320; lean_object* x_321; lean_object* x_322; lean_object* x_342; lean_object* x_343; uint8_t x_344; 
x_49 = lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_50 = lp_mathlib_initFn___closed__1_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_342 = lp_mathlib_Lean_isTracingEnabledFor___at___00CancelDenoms_mkProdPrf_spec__0___redArg(x_50, x_4);
x_343 = lean_ctor_get(x_342, 0);
lean_inc(x_343);
lean_dec_ref(x_342);
x_344 = lean_unbox(x_343);
lean_dec(x_343);
if (x_344 == 0)
{
x_318 = x_2;
x_319 = x_3;
x_320 = x_4;
x_321 = x_5;
x_322 = lean_box(0);
goto block_341;
}
else
{
lean_object* x_345; lean_object* x_346; lean_object* x_347; lean_object* x_348; 
x_345 = lp_mathlib_CancelDenoms_derive___closed__21;
lean_inc_ref(x_1);
x_346 = l_Lean_MessageData_ofExpr(x_1);
x_347 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_347, 0, x_345);
lean_ctor_set(x_347, 1, x_346);
x_348 = lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3(x_50, x_347, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_348) == 0)
{
lean_dec_ref(x_348);
x_318 = x_2;
x_319 = x_3;
x_320 = x_4;
x_321 = x_5;
x_322 = lean_box(0);
goto block_341;
}
else
{
uint8_t x_349; 
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_349 = !lean_is_exclusive(x_348);
if (x_349 == 0)
{
return x_348;
}
else
{
lean_object* x_350; lean_object* x_351; 
x_350 = lean_ctor_get(x_348, 0);
lean_inc(x_350);
lean_dec(x_348);
x_351 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_351, 0, x_350);
return x_351;
}
}
}
block_24:
{
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_15 = lp_mathlib_CancelDenoms_derive___closed__1;
x_16 = l_Lean_MessageData_ofExpr(x_9);
x_17 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_17, 0, x_15);
lean_ctor_set(x_17, 1, x_16);
x_18 = lp_mathlib_CancelDenoms_derive___closed__3;
x_19 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_19, 0, x_17);
lean_ctor_set(x_19, 1, x_18);
x_20 = l_Lean_Exception_toMessageData(x_7);
x_21 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_21, 0, x_19);
lean_ctor_set(x_21, 1, x_20);
x_22 = lp_mathlib_Lean_throwError___at___00CancelDenoms_synthesizeUsingNormNum_spec__0___redArg(x_21, x_11, x_13, x_12, x_10);
lean_dec(x_10);
lean_dec_ref(x_12);
lean_dec(x_13);
lean_dec_ref(x_11);
return x_22;
}
else
{
lean_object* x_23; 
lean_dec(x_13);
lean_dec_ref(x_12);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec_ref(x_9);
x_23 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_23, 0, x_7);
return x_23;
}
}
block_34:
{
uint8_t x_32; 
x_32 = l_Lean_Exception_isInterrupt(x_30);
if (x_32 == 0)
{
uint8_t x_33; 
lean_inc_ref(x_30);
x_33 = l_Lean_Exception_isRuntime(x_30);
x_7 = x_30;
x_8 = lean_box(0);
x_9 = x_25;
x_10 = x_26;
x_11 = x_27;
x_12 = x_28;
x_13 = x_29;
x_14 = x_33;
goto block_24;
}
else
{
x_7 = x_30;
x_8 = lean_box(0);
x_9 = x_25;
x_10 = x_26;
x_11 = x_27;
x_12 = x_28;
x_13 = x_29;
x_14 = x_32;
goto block_24;
}
}
block_48:
{
if (lean_obj_tag(x_40) == 0)
{
uint8_t x_41; 
lean_dec(x_39);
lean_dec_ref(x_38);
lean_dec_ref(x_37);
lean_dec(x_36);
lean_dec_ref(x_35);
x_41 = !lean_is_exclusive(x_40);
if (x_41 == 0)
{
lean_object* x_42; lean_object* x_43; 
x_42 = lean_ctor_get(x_40, 0);
x_43 = lean_ctor_get(x_42, 0);
lean_inc(x_43);
lean_dec(x_42);
lean_ctor_set(x_40, 0, x_43);
return x_40;
}
else
{
lean_object* x_44; lean_object* x_45; lean_object* x_46; 
x_44 = lean_ctor_get(x_40, 0);
lean_inc(x_44);
lean_dec(x_40);
x_45 = lean_ctor_get(x_44, 0);
lean_inc(x_45);
lean_dec(x_44);
x_46 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_46, 0, x_45);
return x_46;
}
}
else
{
lean_object* x_47; 
x_47 = lean_ctor_get(x_40, 0);
lean_inc(x_47);
lean_dec_ref(x_40);
x_25 = x_35;
x_26 = x_36;
x_27 = x_37;
x_28 = x_38;
x_29 = x_39;
x_30 = x_47;
x_31 = lean_box(0);
goto block_34;
}
}
block_279:
{
lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; 
lean_inc_ref(x_53);
x_60 = lp_mathlib_CancelDenoms_findCancelFactor(x_53);
x_61 = lean_ctor_get(x_60, 0);
lean_inc(x_61);
x_62 = lean_ctor_get(x_60, 1);
lean_inc(x_62);
lean_dec_ref(x_60);
lean_inc(x_58);
lean_inc_ref(x_57);
lean_inc(x_56);
lean_inc_ref(x_55);
x_63 = lp_mathlib_Qq_inferTypeQ_x27(x_53, x_55, x_56, x_57, x_58);
if (lean_obj_tag(x_63) == 0)
{
lean_object* x_64; uint8_t x_65; 
x_64 = lean_ctor_get(x_63, 0);
lean_inc(x_64);
lean_dec_ref(x_63);
x_65 = !lean_is_exclusive(x_64);
if (x_65 == 0)
{
lean_object* x_66; uint8_t x_67; 
x_66 = lean_ctor_get(x_64, 1);
x_67 = !lean_is_exclusive(x_66);
if (x_67 == 0)
{
lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; 
x_68 = lean_ctor_get(x_64, 0);
x_69 = lean_ctor_get(x_66, 0);
x_70 = lean_ctor_get(x_66, 1);
x_71 = lp_mathlib_CancelDenoms_derive___closed__4;
x_72 = lean_box(0);
lean_inc(x_68);
lean_ctor_set_tag(x_66, 1);
lean_ctor_set(x_66, 1, x_72);
lean_ctor_set(x_66, 0, x_68);
lean_inc_ref(x_66);
x_73 = l_Lean_Expr_const___override(x_71, x_66);
lean_inc(x_69);
x_74 = l_Lean_Expr_app___override(x_73, x_69);
lean_inc(x_58);
lean_inc_ref(x_57);
lean_inc(x_56);
lean_inc_ref(x_55);
x_75 = lp_Qq_Qq_synthInstanceQ___redArg(x_74, x_55, x_56, x_57, x_58);
if (lean_obj_tag(x_75) == 0)
{
lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; 
x_76 = lean_ctor_get(x_75, 0);
lean_inc(x_76);
lean_dec_ref(x_75);
lean_inc(x_68);
x_77 = l_Lean_Level_succ___override(x_68);
x_78 = lp_mathlib_CancelDenoms_mkProdPrf___closed__1;
lean_inc_ref(x_66);
x_79 = l_Lean_Expr_const___override(x_78, x_66);
lean_inc(x_69);
x_80 = l_Lean_Expr_app___override(x_79, x_69);
x_81 = lp_mathlib_CancelDenoms_mkProdPrf___closed__3;
lean_ctor_set_tag(x_64, 1);
lean_ctor_set(x_64, 1, x_72);
lean_ctor_set(x_64, 0, x_77);
x_82 = l_Lean_Expr_const___override(x_81, x_64);
x_83 = l_Lean_Expr_app___override(x_82, x_80);
x_84 = lp_mathlib_CancelDenoms_mkProdPrf___closed__6;
lean_inc_ref(x_66);
x_85 = l_Lean_Expr_const___override(x_84, x_66);
lean_inc(x_69);
x_86 = l_Lean_Expr_app___override(x_85, x_69);
x_87 = lp_mathlib_CancelDenoms_mkProdPrf___closed__9;
lean_inc_ref(x_66);
x_88 = l_Lean_Expr_const___override(x_87, x_66);
lean_inc(x_69);
x_89 = l_Lean_Expr_app___override(x_88, x_69);
x_90 = lp_mathlib_CancelDenoms_mkProdPrf___closed__12;
lean_inc_ref(x_66);
x_91 = l_Lean_Expr_const___override(x_90, x_66);
lean_inc(x_69);
x_92 = l_Lean_Expr_app___override(x_91, x_69);
x_93 = lp_mathlib_CancelDenoms_mkProdPrf___closed__15;
x_94 = l_Lean_Expr_const___override(x_93, x_66);
lean_inc(x_69);
x_95 = l_Lean_Expr_app___override(x_94, x_69);
lean_inc(x_76);
x_96 = l_Lean_Expr_app___override(x_95, x_76);
x_97 = l_Lean_Expr_app___override(x_92, x_96);
x_98 = l_Lean_Expr_app___override(x_89, x_97);
x_99 = l_Lean_Expr_app___override(x_86, x_98);
x_100 = l_Lean_Expr_app___override(x_83, x_99);
lean_inc(x_61);
x_101 = l_Lean_mkRawNatLit(x_61);
lean_inc(x_69);
lean_inc(x_68);
x_102 = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(x_68, x_69, x_100, x_101, x_55, x_56, x_57, x_58);
if (lean_obj_tag(x_102) == 0)
{
lean_object* x_103; uint8_t x_104; 
x_103 = lean_ctor_get(x_102, 0);
lean_inc(x_103);
lean_dec_ref(x_102);
x_104 = !lean_is_exclusive(x_103);
if (x_104 == 0)
{
lean_object* x_105; lean_object* x_106; lean_object* x_107; 
x_105 = lean_ctor_get(x_103, 0);
x_106 = lean_ctor_get(x_103, 1);
lean_dec(x_106);
lean_inc(x_58);
lean_inc_ref(x_57);
lean_inc(x_56);
lean_inc_ref(x_55);
lean_inc(x_70);
lean_inc(x_61);
x_107 = lp_mathlib_CancelDenoms_mkProdPrf(x_68, x_69, x_76, x_61, x_105, x_62, x_70, x_55, x_56, x_57, x_58);
if (lean_obj_tag(x_107) == 0)
{
lean_object* x_108; lean_object* x_109; lean_object* x_110; uint8_t x_111; 
x_108 = lean_ctor_get(x_107, 0);
lean_inc(x_108);
lean_dec_ref(x_107);
x_109 = lp_mathlib_Lean_isTracingEnabledFor___at___00CancelDenoms_mkProdPrf_spec__0___redArg(x_50, x_57);
x_110 = lean_ctor_get(x_109, 0);
lean_inc(x_110);
lean_dec_ref(x_109);
x_111 = lean_unbox(x_110);
lean_dec(x_110);
if (x_111 == 0)
{
lean_object* x_112; lean_object* x_113; 
lean_free_object(x_103);
x_112 = lean_box(0);
lean_inc(x_58);
lean_inc_ref(x_57);
lean_inc(x_56);
lean_inc_ref(x_55);
x_113 = lp_mathlib_CancelDenoms_derive___lam__0(x_61, x_108, x_49, x_51, x_52, x_54, x_112, x_55, x_56, x_57, x_58);
x_35 = x_70;
x_36 = x_58;
x_37 = x_55;
x_38 = x_57;
x_39 = x_56;
x_40 = x_113;
goto block_48;
}
else
{
lean_object* x_114; lean_object* x_115; 
x_114 = lean_ctor_get(x_108, 1);
lean_inc(x_58);
lean_inc_ref(x_57);
lean_inc(x_56);
lean_inc_ref(x_55);
lean_inc_ref(x_114);
x_115 = lean_infer_type(x_114, x_55, x_56, x_57, x_58);
if (lean_obj_tag(x_115) == 0)
{
lean_object* x_116; lean_object* x_117; lean_object* x_118; lean_object* x_119; 
x_116 = lean_ctor_get(x_115, 0);
lean_inc(x_116);
lean_dec_ref(x_115);
x_117 = lp_mathlib_CancelDenoms_derive___closed__6;
x_118 = l_Lean_MessageData_ofExpr(x_116);
lean_ctor_set_tag(x_103, 7);
lean_ctor_set(x_103, 1, x_118);
lean_ctor_set(x_103, 0, x_117);
x_119 = lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3(x_50, x_103, x_55, x_56, x_57, x_58);
if (lean_obj_tag(x_119) == 0)
{
lean_object* x_120; lean_object* x_121; 
x_120 = lean_ctor_get(x_119, 0);
lean_inc(x_120);
lean_dec_ref(x_119);
lean_inc(x_58);
lean_inc_ref(x_57);
lean_inc(x_56);
lean_inc_ref(x_55);
x_121 = lp_mathlib_CancelDenoms_derive___lam__0(x_61, x_108, x_49, x_51, x_52, x_54, x_120, x_55, x_56, x_57, x_58);
x_35 = x_70;
x_36 = x_58;
x_37 = x_55;
x_38 = x_57;
x_39 = x_56;
x_40 = x_121;
goto block_48;
}
else
{
lean_object* x_122; 
lean_dec(x_108);
lean_dec(x_61);
lean_dec(x_54);
lean_dec(x_52);
x_122 = lean_ctor_get(x_119, 0);
lean_inc(x_122);
lean_dec_ref(x_119);
x_25 = x_70;
x_26 = x_58;
x_27 = x_55;
x_28 = x_57;
x_29 = x_56;
x_30 = x_122;
x_31 = lean_box(0);
goto block_34;
}
}
else
{
lean_object* x_123; 
lean_dec(x_108);
lean_free_object(x_103);
lean_dec(x_61);
lean_dec(x_54);
lean_dec(x_52);
x_123 = lean_ctor_get(x_115, 0);
lean_inc(x_123);
lean_dec_ref(x_115);
x_25 = x_70;
x_26 = x_58;
x_27 = x_55;
x_28 = x_57;
x_29 = x_56;
x_30 = x_123;
x_31 = lean_box(0);
goto block_34;
}
}
}
else
{
lean_object* x_124; 
lean_free_object(x_103);
lean_dec(x_61);
lean_dec(x_54);
lean_dec(x_52);
x_124 = lean_ctor_get(x_107, 0);
lean_inc(x_124);
lean_dec_ref(x_107);
x_25 = x_70;
x_26 = x_58;
x_27 = x_55;
x_28 = x_57;
x_29 = x_56;
x_30 = x_124;
x_31 = lean_box(0);
goto block_34;
}
}
else
{
lean_object* x_125; lean_object* x_126; 
x_125 = lean_ctor_get(x_103, 0);
lean_inc(x_125);
lean_dec(x_103);
lean_inc(x_58);
lean_inc_ref(x_57);
lean_inc(x_56);
lean_inc_ref(x_55);
lean_inc(x_70);
lean_inc(x_61);
x_126 = lp_mathlib_CancelDenoms_mkProdPrf(x_68, x_69, x_76, x_61, x_125, x_62, x_70, x_55, x_56, x_57, x_58);
if (lean_obj_tag(x_126) == 0)
{
lean_object* x_127; lean_object* x_128; lean_object* x_129; uint8_t x_130; 
x_127 = lean_ctor_get(x_126, 0);
lean_inc(x_127);
lean_dec_ref(x_126);
x_128 = lp_mathlib_Lean_isTracingEnabledFor___at___00CancelDenoms_mkProdPrf_spec__0___redArg(x_50, x_57);
x_129 = lean_ctor_get(x_128, 0);
lean_inc(x_129);
lean_dec_ref(x_128);
x_130 = lean_unbox(x_129);
lean_dec(x_129);
if (x_130 == 0)
{
lean_object* x_131; lean_object* x_132; 
x_131 = lean_box(0);
lean_inc(x_58);
lean_inc_ref(x_57);
lean_inc(x_56);
lean_inc_ref(x_55);
x_132 = lp_mathlib_CancelDenoms_derive___lam__0(x_61, x_127, x_49, x_51, x_52, x_54, x_131, x_55, x_56, x_57, x_58);
x_35 = x_70;
x_36 = x_58;
x_37 = x_55;
x_38 = x_57;
x_39 = x_56;
x_40 = x_132;
goto block_48;
}
else
{
lean_object* x_133; lean_object* x_134; 
x_133 = lean_ctor_get(x_127, 1);
lean_inc(x_58);
lean_inc_ref(x_57);
lean_inc(x_56);
lean_inc_ref(x_55);
lean_inc_ref(x_133);
x_134 = lean_infer_type(x_133, x_55, x_56, x_57, x_58);
if (lean_obj_tag(x_134) == 0)
{
lean_object* x_135; lean_object* x_136; lean_object* x_137; lean_object* x_138; lean_object* x_139; 
x_135 = lean_ctor_get(x_134, 0);
lean_inc(x_135);
lean_dec_ref(x_134);
x_136 = lp_mathlib_CancelDenoms_derive___closed__6;
x_137 = l_Lean_MessageData_ofExpr(x_135);
x_138 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_138, 0, x_136);
lean_ctor_set(x_138, 1, x_137);
x_139 = lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3(x_50, x_138, x_55, x_56, x_57, x_58);
if (lean_obj_tag(x_139) == 0)
{
lean_object* x_140; lean_object* x_141; 
x_140 = lean_ctor_get(x_139, 0);
lean_inc(x_140);
lean_dec_ref(x_139);
lean_inc(x_58);
lean_inc_ref(x_57);
lean_inc(x_56);
lean_inc_ref(x_55);
x_141 = lp_mathlib_CancelDenoms_derive___lam__0(x_61, x_127, x_49, x_51, x_52, x_54, x_140, x_55, x_56, x_57, x_58);
x_35 = x_70;
x_36 = x_58;
x_37 = x_55;
x_38 = x_57;
x_39 = x_56;
x_40 = x_141;
goto block_48;
}
else
{
lean_object* x_142; 
lean_dec(x_127);
lean_dec(x_61);
lean_dec(x_54);
lean_dec(x_52);
x_142 = lean_ctor_get(x_139, 0);
lean_inc(x_142);
lean_dec_ref(x_139);
x_25 = x_70;
x_26 = x_58;
x_27 = x_55;
x_28 = x_57;
x_29 = x_56;
x_30 = x_142;
x_31 = lean_box(0);
goto block_34;
}
}
else
{
lean_object* x_143; 
lean_dec(x_127);
lean_dec(x_61);
lean_dec(x_54);
lean_dec(x_52);
x_143 = lean_ctor_get(x_134, 0);
lean_inc(x_143);
lean_dec_ref(x_134);
x_25 = x_70;
x_26 = x_58;
x_27 = x_55;
x_28 = x_57;
x_29 = x_56;
x_30 = x_143;
x_31 = lean_box(0);
goto block_34;
}
}
}
else
{
lean_object* x_144; 
lean_dec(x_61);
lean_dec(x_54);
lean_dec(x_52);
x_144 = lean_ctor_get(x_126, 0);
lean_inc(x_144);
lean_dec_ref(x_126);
x_25 = x_70;
x_26 = x_58;
x_27 = x_55;
x_28 = x_57;
x_29 = x_56;
x_30 = x_144;
x_31 = lean_box(0);
goto block_34;
}
}
}
else
{
lean_object* x_145; 
lean_dec(x_76);
lean_dec(x_69);
lean_dec(x_68);
lean_dec(x_62);
lean_dec(x_61);
lean_dec(x_54);
lean_dec(x_52);
x_145 = lean_ctor_get(x_102, 0);
lean_inc(x_145);
lean_dec_ref(x_102);
x_25 = x_70;
x_26 = x_58;
x_27 = x_55;
x_28 = x_57;
x_29 = x_56;
x_30 = x_145;
x_31 = lean_box(0);
goto block_34;
}
}
else
{
uint8_t x_146; 
lean_dec_ref(x_66);
lean_dec(x_70);
lean_dec(x_69);
lean_free_object(x_64);
lean_dec(x_68);
lean_dec(x_62);
lean_dec(x_61);
lean_dec(x_58);
lean_dec_ref(x_57);
lean_dec(x_56);
lean_dec_ref(x_55);
lean_dec(x_54);
lean_dec(x_52);
x_146 = !lean_is_exclusive(x_75);
if (x_146 == 0)
{
return x_75;
}
else
{
lean_object* x_147; lean_object* x_148; 
x_147 = lean_ctor_get(x_75, 0);
lean_inc(x_147);
lean_dec(x_75);
x_148 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_148, 0, x_147);
return x_148;
}
}
}
else
{
lean_object* x_149; lean_object* x_150; lean_object* x_151; lean_object* x_152; lean_object* x_153; lean_object* x_154; lean_object* x_155; lean_object* x_156; lean_object* x_157; 
x_149 = lean_ctor_get(x_64, 0);
x_150 = lean_ctor_get(x_66, 0);
x_151 = lean_ctor_get(x_66, 1);
lean_inc(x_151);
lean_inc(x_150);
lean_dec(x_66);
x_152 = lp_mathlib_CancelDenoms_derive___closed__4;
x_153 = lean_box(0);
lean_inc(x_149);
x_154 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_154, 0, x_149);
lean_ctor_set(x_154, 1, x_153);
lean_inc_ref(x_154);
x_155 = l_Lean_Expr_const___override(x_152, x_154);
lean_inc(x_150);
x_156 = l_Lean_Expr_app___override(x_155, x_150);
lean_inc(x_58);
lean_inc_ref(x_57);
lean_inc(x_56);
lean_inc_ref(x_55);
x_157 = lp_Qq_Qq_synthInstanceQ___redArg(x_156, x_55, x_56, x_57, x_58);
if (lean_obj_tag(x_157) == 0)
{
lean_object* x_158; lean_object* x_159; lean_object* x_160; lean_object* x_161; lean_object* x_162; lean_object* x_163; lean_object* x_164; lean_object* x_165; lean_object* x_166; lean_object* x_167; lean_object* x_168; lean_object* x_169; lean_object* x_170; lean_object* x_171; lean_object* x_172; lean_object* x_173; lean_object* x_174; lean_object* x_175; lean_object* x_176; lean_object* x_177; lean_object* x_178; lean_object* x_179; lean_object* x_180; lean_object* x_181; lean_object* x_182; lean_object* x_183; lean_object* x_184; 
x_158 = lean_ctor_get(x_157, 0);
lean_inc(x_158);
lean_dec_ref(x_157);
lean_inc(x_149);
x_159 = l_Lean_Level_succ___override(x_149);
x_160 = lp_mathlib_CancelDenoms_mkProdPrf___closed__1;
lean_inc_ref(x_154);
x_161 = l_Lean_Expr_const___override(x_160, x_154);
lean_inc(x_150);
x_162 = l_Lean_Expr_app___override(x_161, x_150);
x_163 = lp_mathlib_CancelDenoms_mkProdPrf___closed__3;
lean_ctor_set_tag(x_64, 1);
lean_ctor_set(x_64, 1, x_153);
lean_ctor_set(x_64, 0, x_159);
x_164 = l_Lean_Expr_const___override(x_163, x_64);
x_165 = l_Lean_Expr_app___override(x_164, x_162);
x_166 = lp_mathlib_CancelDenoms_mkProdPrf___closed__6;
lean_inc_ref(x_154);
x_167 = l_Lean_Expr_const___override(x_166, x_154);
lean_inc(x_150);
x_168 = l_Lean_Expr_app___override(x_167, x_150);
x_169 = lp_mathlib_CancelDenoms_mkProdPrf___closed__9;
lean_inc_ref(x_154);
x_170 = l_Lean_Expr_const___override(x_169, x_154);
lean_inc(x_150);
x_171 = l_Lean_Expr_app___override(x_170, x_150);
x_172 = lp_mathlib_CancelDenoms_mkProdPrf___closed__12;
lean_inc_ref(x_154);
x_173 = l_Lean_Expr_const___override(x_172, x_154);
lean_inc(x_150);
x_174 = l_Lean_Expr_app___override(x_173, x_150);
x_175 = lp_mathlib_CancelDenoms_mkProdPrf___closed__15;
x_176 = l_Lean_Expr_const___override(x_175, x_154);
lean_inc(x_150);
x_177 = l_Lean_Expr_app___override(x_176, x_150);
lean_inc(x_158);
x_178 = l_Lean_Expr_app___override(x_177, x_158);
x_179 = l_Lean_Expr_app___override(x_174, x_178);
x_180 = l_Lean_Expr_app___override(x_171, x_179);
x_181 = l_Lean_Expr_app___override(x_168, x_180);
x_182 = l_Lean_Expr_app___override(x_165, x_181);
lean_inc(x_61);
x_183 = l_Lean_mkRawNatLit(x_61);
lean_inc(x_150);
lean_inc(x_149);
x_184 = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(x_149, x_150, x_182, x_183, x_55, x_56, x_57, x_58);
if (lean_obj_tag(x_184) == 0)
{
lean_object* x_185; lean_object* x_186; lean_object* x_187; lean_object* x_188; 
x_185 = lean_ctor_get(x_184, 0);
lean_inc(x_185);
lean_dec_ref(x_184);
x_186 = lean_ctor_get(x_185, 0);
lean_inc(x_186);
if (lean_is_exclusive(x_185)) {
 lean_ctor_release(x_185, 0);
 lean_ctor_release(x_185, 1);
 x_187 = x_185;
} else {
 lean_dec_ref(x_185);
 x_187 = lean_box(0);
}
lean_inc(x_58);
lean_inc_ref(x_57);
lean_inc(x_56);
lean_inc_ref(x_55);
lean_inc(x_151);
lean_inc(x_61);
x_188 = lp_mathlib_CancelDenoms_mkProdPrf(x_149, x_150, x_158, x_61, x_186, x_62, x_151, x_55, x_56, x_57, x_58);
if (lean_obj_tag(x_188) == 0)
{
lean_object* x_189; lean_object* x_190; lean_object* x_191; uint8_t x_192; 
x_189 = lean_ctor_get(x_188, 0);
lean_inc(x_189);
lean_dec_ref(x_188);
x_190 = lp_mathlib_Lean_isTracingEnabledFor___at___00CancelDenoms_mkProdPrf_spec__0___redArg(x_50, x_57);
x_191 = lean_ctor_get(x_190, 0);
lean_inc(x_191);
lean_dec_ref(x_190);
x_192 = lean_unbox(x_191);
lean_dec(x_191);
if (x_192 == 0)
{
lean_object* x_193; lean_object* x_194; 
lean_dec(x_187);
x_193 = lean_box(0);
lean_inc(x_58);
lean_inc_ref(x_57);
lean_inc(x_56);
lean_inc_ref(x_55);
x_194 = lp_mathlib_CancelDenoms_derive___lam__0(x_61, x_189, x_49, x_51, x_52, x_54, x_193, x_55, x_56, x_57, x_58);
x_35 = x_151;
x_36 = x_58;
x_37 = x_55;
x_38 = x_57;
x_39 = x_56;
x_40 = x_194;
goto block_48;
}
else
{
lean_object* x_195; lean_object* x_196; 
x_195 = lean_ctor_get(x_189, 1);
lean_inc(x_58);
lean_inc_ref(x_57);
lean_inc(x_56);
lean_inc_ref(x_55);
lean_inc_ref(x_195);
x_196 = lean_infer_type(x_195, x_55, x_56, x_57, x_58);
if (lean_obj_tag(x_196) == 0)
{
lean_object* x_197; lean_object* x_198; lean_object* x_199; lean_object* x_200; lean_object* x_201; 
x_197 = lean_ctor_get(x_196, 0);
lean_inc(x_197);
lean_dec_ref(x_196);
x_198 = lp_mathlib_CancelDenoms_derive___closed__6;
x_199 = l_Lean_MessageData_ofExpr(x_197);
if (lean_is_scalar(x_187)) {
 x_200 = lean_alloc_ctor(7, 2, 0);
} else {
 x_200 = x_187;
 lean_ctor_set_tag(x_200, 7);
}
lean_ctor_set(x_200, 0, x_198);
lean_ctor_set(x_200, 1, x_199);
x_201 = lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3(x_50, x_200, x_55, x_56, x_57, x_58);
if (lean_obj_tag(x_201) == 0)
{
lean_object* x_202; lean_object* x_203; 
x_202 = lean_ctor_get(x_201, 0);
lean_inc(x_202);
lean_dec_ref(x_201);
lean_inc(x_58);
lean_inc_ref(x_57);
lean_inc(x_56);
lean_inc_ref(x_55);
x_203 = lp_mathlib_CancelDenoms_derive___lam__0(x_61, x_189, x_49, x_51, x_52, x_54, x_202, x_55, x_56, x_57, x_58);
x_35 = x_151;
x_36 = x_58;
x_37 = x_55;
x_38 = x_57;
x_39 = x_56;
x_40 = x_203;
goto block_48;
}
else
{
lean_object* x_204; 
lean_dec(x_189);
lean_dec(x_61);
lean_dec(x_54);
lean_dec(x_52);
x_204 = lean_ctor_get(x_201, 0);
lean_inc(x_204);
lean_dec_ref(x_201);
x_25 = x_151;
x_26 = x_58;
x_27 = x_55;
x_28 = x_57;
x_29 = x_56;
x_30 = x_204;
x_31 = lean_box(0);
goto block_34;
}
}
else
{
lean_object* x_205; 
lean_dec(x_189);
lean_dec(x_187);
lean_dec(x_61);
lean_dec(x_54);
lean_dec(x_52);
x_205 = lean_ctor_get(x_196, 0);
lean_inc(x_205);
lean_dec_ref(x_196);
x_25 = x_151;
x_26 = x_58;
x_27 = x_55;
x_28 = x_57;
x_29 = x_56;
x_30 = x_205;
x_31 = lean_box(0);
goto block_34;
}
}
}
else
{
lean_object* x_206; 
lean_dec(x_187);
lean_dec(x_61);
lean_dec(x_54);
lean_dec(x_52);
x_206 = lean_ctor_get(x_188, 0);
lean_inc(x_206);
lean_dec_ref(x_188);
x_25 = x_151;
x_26 = x_58;
x_27 = x_55;
x_28 = x_57;
x_29 = x_56;
x_30 = x_206;
x_31 = lean_box(0);
goto block_34;
}
}
else
{
lean_object* x_207; 
lean_dec(x_158);
lean_dec(x_150);
lean_dec(x_149);
lean_dec(x_62);
lean_dec(x_61);
lean_dec(x_54);
lean_dec(x_52);
x_207 = lean_ctor_get(x_184, 0);
lean_inc(x_207);
lean_dec_ref(x_184);
x_25 = x_151;
x_26 = x_58;
x_27 = x_55;
x_28 = x_57;
x_29 = x_56;
x_30 = x_207;
x_31 = lean_box(0);
goto block_34;
}
}
else
{
lean_object* x_208; lean_object* x_209; lean_object* x_210; 
lean_dec_ref(x_154);
lean_dec(x_151);
lean_dec(x_150);
lean_free_object(x_64);
lean_dec(x_149);
lean_dec(x_62);
lean_dec(x_61);
lean_dec(x_58);
lean_dec_ref(x_57);
lean_dec(x_56);
lean_dec_ref(x_55);
lean_dec(x_54);
lean_dec(x_52);
x_208 = lean_ctor_get(x_157, 0);
lean_inc(x_208);
if (lean_is_exclusive(x_157)) {
 lean_ctor_release(x_157, 0);
 x_209 = x_157;
} else {
 lean_dec_ref(x_157);
 x_209 = lean_box(0);
}
if (lean_is_scalar(x_209)) {
 x_210 = lean_alloc_ctor(1, 1, 0);
} else {
 x_210 = x_209;
}
lean_ctor_set(x_210, 0, x_208);
return x_210;
}
}
}
else
{
lean_object* x_211; lean_object* x_212; lean_object* x_213; lean_object* x_214; lean_object* x_215; lean_object* x_216; lean_object* x_217; lean_object* x_218; lean_object* x_219; lean_object* x_220; lean_object* x_221; 
x_211 = lean_ctor_get(x_64, 1);
x_212 = lean_ctor_get(x_64, 0);
lean_inc(x_211);
lean_inc(x_212);
lean_dec(x_64);
x_213 = lean_ctor_get(x_211, 0);
lean_inc(x_213);
x_214 = lean_ctor_get(x_211, 1);
lean_inc(x_214);
if (lean_is_exclusive(x_211)) {
 lean_ctor_release(x_211, 0);
 lean_ctor_release(x_211, 1);
 x_215 = x_211;
} else {
 lean_dec_ref(x_211);
 x_215 = lean_box(0);
}
x_216 = lp_mathlib_CancelDenoms_derive___closed__4;
x_217 = lean_box(0);
lean_inc(x_212);
if (lean_is_scalar(x_215)) {
 x_218 = lean_alloc_ctor(1, 2, 0);
} else {
 x_218 = x_215;
 lean_ctor_set_tag(x_218, 1);
}
lean_ctor_set(x_218, 0, x_212);
lean_ctor_set(x_218, 1, x_217);
lean_inc_ref(x_218);
x_219 = l_Lean_Expr_const___override(x_216, x_218);
lean_inc(x_213);
x_220 = l_Lean_Expr_app___override(x_219, x_213);
lean_inc(x_58);
lean_inc_ref(x_57);
lean_inc(x_56);
lean_inc_ref(x_55);
x_221 = lp_Qq_Qq_synthInstanceQ___redArg(x_220, x_55, x_56, x_57, x_58);
if (lean_obj_tag(x_221) == 0)
{
lean_object* x_222; lean_object* x_223; lean_object* x_224; lean_object* x_225; lean_object* x_226; lean_object* x_227; lean_object* x_228; lean_object* x_229; lean_object* x_230; lean_object* x_231; lean_object* x_232; lean_object* x_233; lean_object* x_234; lean_object* x_235; lean_object* x_236; lean_object* x_237; lean_object* x_238; lean_object* x_239; lean_object* x_240; lean_object* x_241; lean_object* x_242; lean_object* x_243; lean_object* x_244; lean_object* x_245; lean_object* x_246; lean_object* x_247; lean_object* x_248; lean_object* x_249; 
x_222 = lean_ctor_get(x_221, 0);
lean_inc(x_222);
lean_dec_ref(x_221);
lean_inc(x_212);
x_223 = l_Lean_Level_succ___override(x_212);
x_224 = lp_mathlib_CancelDenoms_mkProdPrf___closed__1;
lean_inc_ref(x_218);
x_225 = l_Lean_Expr_const___override(x_224, x_218);
lean_inc(x_213);
x_226 = l_Lean_Expr_app___override(x_225, x_213);
x_227 = lp_mathlib_CancelDenoms_mkProdPrf___closed__3;
x_228 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_228, 0, x_223);
lean_ctor_set(x_228, 1, x_217);
x_229 = l_Lean_Expr_const___override(x_227, x_228);
x_230 = l_Lean_Expr_app___override(x_229, x_226);
x_231 = lp_mathlib_CancelDenoms_mkProdPrf___closed__6;
lean_inc_ref(x_218);
x_232 = l_Lean_Expr_const___override(x_231, x_218);
lean_inc(x_213);
x_233 = l_Lean_Expr_app___override(x_232, x_213);
x_234 = lp_mathlib_CancelDenoms_mkProdPrf___closed__9;
lean_inc_ref(x_218);
x_235 = l_Lean_Expr_const___override(x_234, x_218);
lean_inc(x_213);
x_236 = l_Lean_Expr_app___override(x_235, x_213);
x_237 = lp_mathlib_CancelDenoms_mkProdPrf___closed__12;
lean_inc_ref(x_218);
x_238 = l_Lean_Expr_const___override(x_237, x_218);
lean_inc(x_213);
x_239 = l_Lean_Expr_app___override(x_238, x_213);
x_240 = lp_mathlib_CancelDenoms_mkProdPrf___closed__15;
x_241 = l_Lean_Expr_const___override(x_240, x_218);
lean_inc(x_213);
x_242 = l_Lean_Expr_app___override(x_241, x_213);
lean_inc(x_222);
x_243 = l_Lean_Expr_app___override(x_242, x_222);
x_244 = l_Lean_Expr_app___override(x_239, x_243);
x_245 = l_Lean_Expr_app___override(x_236, x_244);
x_246 = l_Lean_Expr_app___override(x_233, x_245);
x_247 = l_Lean_Expr_app___override(x_230, x_246);
lean_inc(x_61);
x_248 = l_Lean_mkRawNatLit(x_61);
lean_inc(x_213);
lean_inc(x_212);
x_249 = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(x_212, x_213, x_247, x_248, x_55, x_56, x_57, x_58);
if (lean_obj_tag(x_249) == 0)
{
lean_object* x_250; lean_object* x_251; lean_object* x_252; lean_object* x_253; 
x_250 = lean_ctor_get(x_249, 0);
lean_inc(x_250);
lean_dec_ref(x_249);
x_251 = lean_ctor_get(x_250, 0);
lean_inc(x_251);
if (lean_is_exclusive(x_250)) {
 lean_ctor_release(x_250, 0);
 lean_ctor_release(x_250, 1);
 x_252 = x_250;
} else {
 lean_dec_ref(x_250);
 x_252 = lean_box(0);
}
lean_inc(x_58);
lean_inc_ref(x_57);
lean_inc(x_56);
lean_inc_ref(x_55);
lean_inc(x_214);
lean_inc(x_61);
x_253 = lp_mathlib_CancelDenoms_mkProdPrf(x_212, x_213, x_222, x_61, x_251, x_62, x_214, x_55, x_56, x_57, x_58);
if (lean_obj_tag(x_253) == 0)
{
lean_object* x_254; lean_object* x_255; lean_object* x_256; uint8_t x_257; 
x_254 = lean_ctor_get(x_253, 0);
lean_inc(x_254);
lean_dec_ref(x_253);
x_255 = lp_mathlib_Lean_isTracingEnabledFor___at___00CancelDenoms_mkProdPrf_spec__0___redArg(x_50, x_57);
x_256 = lean_ctor_get(x_255, 0);
lean_inc(x_256);
lean_dec_ref(x_255);
x_257 = lean_unbox(x_256);
lean_dec(x_256);
if (x_257 == 0)
{
lean_object* x_258; lean_object* x_259; 
lean_dec(x_252);
x_258 = lean_box(0);
lean_inc(x_58);
lean_inc_ref(x_57);
lean_inc(x_56);
lean_inc_ref(x_55);
x_259 = lp_mathlib_CancelDenoms_derive___lam__0(x_61, x_254, x_49, x_51, x_52, x_54, x_258, x_55, x_56, x_57, x_58);
x_35 = x_214;
x_36 = x_58;
x_37 = x_55;
x_38 = x_57;
x_39 = x_56;
x_40 = x_259;
goto block_48;
}
else
{
lean_object* x_260; lean_object* x_261; 
x_260 = lean_ctor_get(x_254, 1);
lean_inc(x_58);
lean_inc_ref(x_57);
lean_inc(x_56);
lean_inc_ref(x_55);
lean_inc_ref(x_260);
x_261 = lean_infer_type(x_260, x_55, x_56, x_57, x_58);
if (lean_obj_tag(x_261) == 0)
{
lean_object* x_262; lean_object* x_263; lean_object* x_264; lean_object* x_265; lean_object* x_266; 
x_262 = lean_ctor_get(x_261, 0);
lean_inc(x_262);
lean_dec_ref(x_261);
x_263 = lp_mathlib_CancelDenoms_derive___closed__6;
x_264 = l_Lean_MessageData_ofExpr(x_262);
if (lean_is_scalar(x_252)) {
 x_265 = lean_alloc_ctor(7, 2, 0);
} else {
 x_265 = x_252;
 lean_ctor_set_tag(x_265, 7);
}
lean_ctor_set(x_265, 0, x_263);
lean_ctor_set(x_265, 1, x_264);
x_266 = lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3(x_50, x_265, x_55, x_56, x_57, x_58);
if (lean_obj_tag(x_266) == 0)
{
lean_object* x_267; lean_object* x_268; 
x_267 = lean_ctor_get(x_266, 0);
lean_inc(x_267);
lean_dec_ref(x_266);
lean_inc(x_58);
lean_inc_ref(x_57);
lean_inc(x_56);
lean_inc_ref(x_55);
x_268 = lp_mathlib_CancelDenoms_derive___lam__0(x_61, x_254, x_49, x_51, x_52, x_54, x_267, x_55, x_56, x_57, x_58);
x_35 = x_214;
x_36 = x_58;
x_37 = x_55;
x_38 = x_57;
x_39 = x_56;
x_40 = x_268;
goto block_48;
}
else
{
lean_object* x_269; 
lean_dec(x_254);
lean_dec(x_61);
lean_dec(x_54);
lean_dec(x_52);
x_269 = lean_ctor_get(x_266, 0);
lean_inc(x_269);
lean_dec_ref(x_266);
x_25 = x_214;
x_26 = x_58;
x_27 = x_55;
x_28 = x_57;
x_29 = x_56;
x_30 = x_269;
x_31 = lean_box(0);
goto block_34;
}
}
else
{
lean_object* x_270; 
lean_dec(x_254);
lean_dec(x_252);
lean_dec(x_61);
lean_dec(x_54);
lean_dec(x_52);
x_270 = lean_ctor_get(x_261, 0);
lean_inc(x_270);
lean_dec_ref(x_261);
x_25 = x_214;
x_26 = x_58;
x_27 = x_55;
x_28 = x_57;
x_29 = x_56;
x_30 = x_270;
x_31 = lean_box(0);
goto block_34;
}
}
}
else
{
lean_object* x_271; 
lean_dec(x_252);
lean_dec(x_61);
lean_dec(x_54);
lean_dec(x_52);
x_271 = lean_ctor_get(x_253, 0);
lean_inc(x_271);
lean_dec_ref(x_253);
x_25 = x_214;
x_26 = x_58;
x_27 = x_55;
x_28 = x_57;
x_29 = x_56;
x_30 = x_271;
x_31 = lean_box(0);
goto block_34;
}
}
else
{
lean_object* x_272; 
lean_dec(x_222);
lean_dec(x_213);
lean_dec(x_212);
lean_dec(x_62);
lean_dec(x_61);
lean_dec(x_54);
lean_dec(x_52);
x_272 = lean_ctor_get(x_249, 0);
lean_inc(x_272);
lean_dec_ref(x_249);
x_25 = x_214;
x_26 = x_58;
x_27 = x_55;
x_28 = x_57;
x_29 = x_56;
x_30 = x_272;
x_31 = lean_box(0);
goto block_34;
}
}
else
{
lean_object* x_273; lean_object* x_274; lean_object* x_275; 
lean_dec_ref(x_218);
lean_dec(x_214);
lean_dec(x_213);
lean_dec(x_212);
lean_dec(x_62);
lean_dec(x_61);
lean_dec(x_58);
lean_dec_ref(x_57);
lean_dec(x_56);
lean_dec_ref(x_55);
lean_dec(x_54);
lean_dec(x_52);
x_273 = lean_ctor_get(x_221, 0);
lean_inc(x_273);
if (lean_is_exclusive(x_221)) {
 lean_ctor_release(x_221, 0);
 x_274 = x_221;
} else {
 lean_dec_ref(x_221);
 x_274 = lean_box(0);
}
if (lean_is_scalar(x_274)) {
 x_275 = lean_alloc_ctor(1, 1, 0);
} else {
 x_275 = x_274;
}
lean_ctor_set(x_275, 0, x_273);
return x_275;
}
}
}
else
{
uint8_t x_276; 
lean_dec(x_62);
lean_dec(x_61);
lean_dec(x_58);
lean_dec_ref(x_57);
lean_dec(x_56);
lean_dec_ref(x_55);
lean_dec(x_54);
lean_dec(x_52);
x_276 = !lean_is_exclusive(x_63);
if (x_276 == 0)
{
return x_63;
}
else
{
lean_object* x_277; lean_object* x_278; 
x_277 = lean_ctor_get(x_63, 0);
lean_inc(x_277);
lean_dec(x_63);
x_278 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_278, 0, x_277);
return x_278;
}
}
}
block_317:
{
lean_object* x_286; uint8_t x_287; lean_object* x_288; lean_object* x_289; lean_object* x_290; lean_object* x_291; 
x_286 = lean_unsigned_to_nat(2u);
x_287 = 0;
x_288 = lp_mathlib_CancelDenoms_derive___closed__8;
x_289 = lp_mathlib_CancelDenoms_derive___closed__9;
x_290 = lp_mathlib_CancelDenoms_derive___closed__14;
x_291 = l_Lean_Meta_Simp_mkContext___redArg(x_288, x_289, x_290, x_281, x_284);
if (lean_obj_tag(x_291) == 0)
{
lean_object* x_292; lean_object* x_293; lean_object* x_294; lean_object* x_295; 
x_292 = lean_ctor_get(x_291, 0);
lean_inc(x_292);
lean_dec_ref(x_291);
x_293 = lean_ctor_get(x_280, 0);
lean_inc_ref(x_293);
x_294 = lean_ctor_get(x_280, 1);
lean_inc(x_294);
lean_dec_ref(x_280);
lean_inc(x_284);
lean_inc_ref(x_283);
lean_inc(x_282);
lean_inc_ref(x_281);
x_295 = lp_mathlib_Mathlib_Meta_NormNum_deriveSimp(x_292, x_287, x_293, x_281, x_282, x_283, x_284);
if (lean_obj_tag(x_295) == 0)
{
lean_object* x_296; lean_object* x_297; lean_object* x_298; uint8_t x_299; 
x_296 = lean_ctor_get(x_295, 0);
lean_inc(x_296);
lean_dec_ref(x_295);
x_297 = lp_mathlib_Lean_isTracingEnabledFor___at___00CancelDenoms_mkProdPrf_spec__0___redArg(x_50, x_283);
x_298 = lean_ctor_get(x_297, 0);
lean_inc(x_298);
lean_dec_ref(x_297);
x_299 = lean_unbox(x_298);
lean_dec(x_298);
if (x_299 == 0)
{
lean_object* x_300; lean_object* x_301; 
x_300 = lean_ctor_get(x_296, 0);
lean_inc_ref(x_300);
x_301 = lean_ctor_get(x_296, 1);
lean_inc(x_301);
lean_dec(x_296);
x_51 = x_286;
x_52 = x_294;
x_53 = x_300;
x_54 = x_301;
x_55 = x_281;
x_56 = x_282;
x_57 = x_283;
x_58 = x_284;
x_59 = lean_box(0);
goto block_279;
}
else
{
lean_object* x_302; lean_object* x_303; lean_object* x_304; lean_object* x_305; lean_object* x_306; lean_object* x_307; 
x_302 = lean_ctor_get(x_296, 0);
lean_inc_ref(x_302);
x_303 = lean_ctor_get(x_296, 1);
lean_inc(x_303);
lean_dec(x_296);
x_304 = lp_mathlib_CancelDenoms_derive___closed__16;
lean_inc_ref(x_302);
x_305 = l_Lean_MessageData_ofExpr(x_302);
x_306 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_306, 0, x_304);
lean_ctor_set(x_306, 1, x_305);
x_307 = lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3(x_50, x_306, x_281, x_282, x_283, x_284);
if (lean_obj_tag(x_307) == 0)
{
lean_dec_ref(x_307);
x_51 = x_286;
x_52 = x_294;
x_53 = x_302;
x_54 = x_303;
x_55 = x_281;
x_56 = x_282;
x_57 = x_283;
x_58 = x_284;
x_59 = lean_box(0);
goto block_279;
}
else
{
uint8_t x_308; 
lean_dec(x_303);
lean_dec_ref(x_302);
lean_dec(x_294);
lean_dec(x_284);
lean_dec_ref(x_283);
lean_dec(x_282);
lean_dec_ref(x_281);
x_308 = !lean_is_exclusive(x_307);
if (x_308 == 0)
{
return x_307;
}
else
{
lean_object* x_309; lean_object* x_310; 
x_309 = lean_ctor_get(x_307, 0);
lean_inc(x_309);
lean_dec(x_307);
x_310 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_310, 0, x_309);
return x_310;
}
}
}
}
else
{
uint8_t x_311; 
lean_dec(x_294);
lean_dec(x_284);
lean_dec_ref(x_283);
lean_dec(x_282);
lean_dec_ref(x_281);
x_311 = !lean_is_exclusive(x_295);
if (x_311 == 0)
{
return x_295;
}
else
{
lean_object* x_312; lean_object* x_313; 
x_312 = lean_ctor_get(x_295, 0);
lean_inc(x_312);
lean_dec(x_295);
x_313 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_313, 0, x_312);
return x_313;
}
}
}
else
{
uint8_t x_314; 
lean_dec(x_284);
lean_dec_ref(x_283);
lean_dec(x_282);
lean_dec_ref(x_281);
lean_dec_ref(x_280);
x_314 = !lean_is_exclusive(x_291);
if (x_314 == 0)
{
return x_291;
}
else
{
lean_object* x_315; lean_object* x_316; 
x_315 = lean_ctor_get(x_291, 0);
lean_inc(x_315);
lean_dec(x_291);
x_316 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_316, 0, x_315);
return x_316;
}
}
}
block_341:
{
lean_object* x_323; lean_object* x_324; lean_object* x_325; 
x_323 = lp_mathlib_CancelDenoms_deriveThms;
x_324 = lp_mathlib_CancelDenoms_derive___closed__17;
lean_inc(x_321);
lean_inc_ref(x_320);
lean_inc(x_319);
lean_inc_ref(x_318);
x_325 = lp_mathlib_Lean_Meta_simpOnlyNames(x_323, x_1, x_324, x_318, x_319, x_320, x_321);
if (lean_obj_tag(x_325) == 0)
{
lean_object* x_326; lean_object* x_327; lean_object* x_328; uint8_t x_329; 
x_326 = lean_ctor_get(x_325, 0);
lean_inc(x_326);
lean_dec_ref(x_325);
x_327 = lp_mathlib_Lean_isTracingEnabledFor___at___00CancelDenoms_mkProdPrf_spec__0___redArg(x_50, x_320);
x_328 = lean_ctor_get(x_327, 0);
lean_inc(x_328);
lean_dec_ref(x_327);
x_329 = lean_unbox(x_328);
lean_dec(x_328);
if (x_329 == 0)
{
x_280 = x_326;
x_281 = x_318;
x_282 = x_319;
x_283 = x_320;
x_284 = x_321;
x_285 = lean_box(0);
goto block_317;
}
else
{
lean_object* x_330; lean_object* x_331; lean_object* x_332; lean_object* x_333; lean_object* x_334; 
x_330 = lean_ctor_get(x_326, 0);
x_331 = lp_mathlib_CancelDenoms_derive___closed__19;
lean_inc_ref(x_330);
x_332 = l_Lean_MessageData_ofExpr(x_330);
x_333 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_333, 0, x_331);
lean_ctor_set(x_333, 1, x_332);
x_334 = lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3(x_50, x_333, x_318, x_319, x_320, x_321);
if (lean_obj_tag(x_334) == 0)
{
lean_dec_ref(x_334);
x_280 = x_326;
x_281 = x_318;
x_282 = x_319;
x_283 = x_320;
x_284 = x_321;
x_285 = lean_box(0);
goto block_317;
}
else
{
uint8_t x_335; 
lean_dec(x_326);
lean_dec(x_321);
lean_dec_ref(x_320);
lean_dec(x_319);
lean_dec_ref(x_318);
x_335 = !lean_is_exclusive(x_334);
if (x_335 == 0)
{
return x_334;
}
else
{
lean_object* x_336; lean_object* x_337; 
x_336 = lean_ctor_get(x_334, 0);
lean_inc(x_336);
lean_dec(x_334);
x_337 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_337, 0, x_336);
return x_337;
}
}
}
}
else
{
uint8_t x_338; 
lean_dec(x_321);
lean_dec_ref(x_320);
lean_dec(x_319);
lean_dec_ref(x_318);
x_338 = !lean_is_exclusive(x_325);
if (x_338 == 0)
{
return x_325;
}
else
{
lean_object* x_339; lean_object* x_340; 
x_339 = lean_ctor_get(x_325, 0);
lean_inc(x_339);
lean_dec(x_325);
x_340 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_340, 0, x_339);
return x_340;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_derive___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_CancelDenoms_derive___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_4);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_derive___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CancelDenoms_derive(x_1, x_2, x_3, x_4, x_5);
return x_7;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCompLemma___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("LT", 2, 2);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCompLemma___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("LE", 2, 2);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCompLemma___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("GE", 2, 2);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCompLemma___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("GT", 2, 2);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCompLemma___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("gt", 2, 2);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCompLemma___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("cancel_factors_lt", 17, 17);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCompLemma___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_findCompLemma___closed__5;
x_2 = lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCompLemma___closed__7() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("ge", 2, 2);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCompLemma___closed__8() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("cancel_factors_le", 17, 17);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCompLemma___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_findCompLemma___closed__8;
x_2 = lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCompLemma___closed__10() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("le", 2, 2);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCompLemma___closed__11() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("lt", 2, 2);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCompLemma___closed__12() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Not", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCompLemma___closed__13() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("cancel_factors_ne", 17, 17);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCompLemma___closed__14() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_findCompLemma___closed__13;
x_2 = lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCompLemma___closed__15() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("cancel_factors_eq", 17, 17);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCompLemma___closed__16() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_findCompLemma___closed__15;
x_2 = lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_findCompLemma___closed__17() {
_start:
{
uint8_t x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = 0;
x_2 = lp_mathlib_CancelDenoms_findCompLemma___closed__16;
x_3 = lean_box(x_1);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_findCompLemma(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; lean_object* x_11; lean_object* x_15; 
lean_inc(x_5);
lean_inc_ref(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
x_15 = l_Lean_Meta_whnfR(x_1, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_15) == 0)
{
uint8_t x_16; 
x_16 = !lean_is_exclusive(x_15);
if (x_16 == 0)
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_17 = lean_ctor_get(x_15, 0);
x_18 = l_Lean_Expr_getAppFnArgs(x_17);
x_19 = lean_ctor_get(x_18, 0);
lean_inc(x_19);
if (lean_obj_tag(x_19) == 1)
{
lean_object* x_20; 
x_20 = lean_ctor_get(x_19, 0);
switch (lean_obj_tag(x_20)) {
case 1:
{
lean_object* x_21; 
lean_inc_ref(x_20);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_21 = lean_ctor_get(x_20, 0);
if (lean_obj_tag(x_21) == 0)
{
uint8_t x_22; 
x_22 = !lean_is_exclusive(x_18);
if (x_22 == 0)
{
lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; uint8_t x_28; 
x_23 = lean_ctor_get(x_18, 1);
x_24 = lean_ctor_get(x_18, 0);
lean_dec(x_24);
x_25 = lean_ctor_get(x_19, 1);
lean_inc_ref(x_25);
lean_dec_ref(x_19);
x_26 = lean_ctor_get(x_20, 1);
lean_inc_ref(x_26);
lean_dec_ref(x_20);
x_27 = lp_mathlib_CancelDenoms_findCompLemma___closed__0;
x_28 = lean_string_dec_eq(x_26, x_27);
if (x_28 == 0)
{
lean_object* x_29; uint8_t x_30; 
x_29 = lp_mathlib_CancelDenoms_findCompLemma___closed__1;
x_30 = lean_string_dec_eq(x_26, x_29);
if (x_30 == 0)
{
lean_object* x_31; uint8_t x_32; 
x_31 = lp_mathlib_CancelDenoms_findCompLemma___closed__2;
x_32 = lean_string_dec_eq(x_26, x_31);
if (x_32 == 0)
{
lean_object* x_33; uint8_t x_34; 
x_33 = lp_mathlib_CancelDenoms_findCompLemma___closed__3;
x_34 = lean_string_dec_eq(x_26, x_33);
lean_dec_ref(x_26);
if (x_34 == 0)
{
lean_dec_ref(x_25);
lean_free_object(x_18);
lean_dec(x_23);
lean_free_object(x_15);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_35; uint8_t x_36; 
x_35 = lp_mathlib_CancelDenoms_findCompLemma___closed__4;
x_36 = lean_string_dec_eq(x_25, x_35);
lean_dec_ref(x_25);
if (x_36 == 0)
{
lean_free_object(x_18);
lean_dec(x_23);
lean_free_object(x_15);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_37; lean_object* x_38; uint8_t x_39; 
x_37 = lean_array_get_size(x_23);
x_38 = lean_unsigned_to_nat(4u);
x_39 = lean_nat_dec_eq(x_37, x_38);
if (x_39 == 0)
{
lean_free_object(x_18);
lean_dec(x_23);
lean_free_object(x_15);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; 
x_40 = lean_unsigned_to_nat(2u);
x_41 = lean_array_fget(x_23, x_40);
x_42 = lean_unsigned_to_nat(3u);
x_43 = lean_array_fget(x_23, x_42);
lean_dec(x_23);
x_44 = lp_mathlib_CancelDenoms_findCompLemma___closed__6;
x_45 = lean_box(x_39);
lean_ctor_set(x_18, 1, x_45);
lean_ctor_set(x_18, 0, x_44);
x_46 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_46, 0, x_41);
lean_ctor_set(x_46, 1, x_18);
x_47 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_47, 0, x_43);
lean_ctor_set(x_47, 1, x_46);
x_48 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_48, 0, x_47);
lean_ctor_set(x_15, 0, x_48);
return x_15;
}
}
}
}
else
{
lean_object* x_49; uint8_t x_50; 
lean_dec_ref(x_26);
x_49 = lp_mathlib_CancelDenoms_findCompLemma___closed__7;
x_50 = lean_string_dec_eq(x_25, x_49);
lean_dec_ref(x_25);
if (x_50 == 0)
{
lean_free_object(x_18);
lean_dec(x_23);
lean_free_object(x_15);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_51; lean_object* x_52; uint8_t x_53; 
x_51 = lean_array_get_size(x_23);
x_52 = lean_unsigned_to_nat(4u);
x_53 = lean_nat_dec_eq(x_51, x_52);
if (x_53 == 0)
{
lean_free_object(x_18);
lean_dec(x_23);
lean_free_object(x_15);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; 
x_54 = lean_unsigned_to_nat(2u);
x_55 = lean_array_fget(x_23, x_54);
x_56 = lean_unsigned_to_nat(3u);
x_57 = lean_array_fget(x_23, x_56);
lean_dec(x_23);
x_58 = lp_mathlib_CancelDenoms_findCompLemma___closed__9;
x_59 = lean_box(x_53);
lean_ctor_set(x_18, 1, x_59);
lean_ctor_set(x_18, 0, x_58);
x_60 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_60, 0, x_55);
lean_ctor_set(x_60, 1, x_18);
x_61 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_61, 0, x_57);
lean_ctor_set(x_61, 1, x_60);
x_62 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_62, 0, x_61);
lean_ctor_set(x_15, 0, x_62);
return x_15;
}
}
}
}
else
{
lean_object* x_63; uint8_t x_64; 
lean_dec_ref(x_26);
x_63 = lp_mathlib_CancelDenoms_findCompLemma___closed__10;
x_64 = lean_string_dec_eq(x_25, x_63);
lean_dec_ref(x_25);
if (x_64 == 0)
{
lean_free_object(x_18);
lean_dec(x_23);
lean_free_object(x_15);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_65; lean_object* x_66; uint8_t x_67; 
x_65 = lean_array_get_size(x_23);
x_66 = lean_unsigned_to_nat(4u);
x_67 = lean_nat_dec_eq(x_65, x_66);
if (x_67 == 0)
{
lean_free_object(x_18);
lean_dec(x_23);
lean_free_object(x_15);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; 
x_68 = lean_unsigned_to_nat(2u);
x_69 = lean_array_fget(x_23, x_68);
x_70 = lean_unsigned_to_nat(3u);
x_71 = lean_array_fget(x_23, x_70);
lean_dec(x_23);
x_72 = lp_mathlib_CancelDenoms_findCompLemma___closed__9;
x_73 = lean_box(x_67);
lean_ctor_set(x_18, 1, x_73);
lean_ctor_set(x_18, 0, x_72);
x_74 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_74, 0, x_71);
lean_ctor_set(x_74, 1, x_18);
x_75 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_75, 0, x_69);
lean_ctor_set(x_75, 1, x_74);
x_76 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_76, 0, x_75);
lean_ctor_set(x_15, 0, x_76);
return x_15;
}
}
}
}
else
{
lean_object* x_77; uint8_t x_78; 
lean_dec_ref(x_26);
x_77 = lp_mathlib_CancelDenoms_findCompLemma___closed__11;
x_78 = lean_string_dec_eq(x_25, x_77);
lean_dec_ref(x_25);
if (x_78 == 0)
{
lean_free_object(x_18);
lean_dec(x_23);
lean_free_object(x_15);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_79; lean_object* x_80; uint8_t x_81; 
x_79 = lean_array_get_size(x_23);
x_80 = lean_unsigned_to_nat(4u);
x_81 = lean_nat_dec_eq(x_79, x_80);
if (x_81 == 0)
{
lean_free_object(x_18);
lean_dec(x_23);
lean_free_object(x_15);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; 
x_82 = lean_unsigned_to_nat(2u);
x_83 = lean_array_fget(x_23, x_82);
x_84 = lean_unsigned_to_nat(3u);
x_85 = lean_array_fget(x_23, x_84);
lean_dec(x_23);
x_86 = lp_mathlib_CancelDenoms_findCompLemma___closed__6;
x_87 = lean_box(x_81);
lean_ctor_set(x_18, 1, x_87);
lean_ctor_set(x_18, 0, x_86);
x_88 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_88, 0, x_85);
lean_ctor_set(x_88, 1, x_18);
x_89 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_89, 0, x_83);
lean_ctor_set(x_89, 1, x_88);
x_90 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_90, 0, x_89);
lean_ctor_set(x_15, 0, x_90);
return x_15;
}
}
}
}
else
{
lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; uint8_t x_95; 
x_91 = lean_ctor_get(x_18, 1);
lean_inc(x_91);
lean_dec(x_18);
x_92 = lean_ctor_get(x_19, 1);
lean_inc_ref(x_92);
lean_dec_ref(x_19);
x_93 = lean_ctor_get(x_20, 1);
lean_inc_ref(x_93);
lean_dec_ref(x_20);
x_94 = lp_mathlib_CancelDenoms_findCompLemma___closed__0;
x_95 = lean_string_dec_eq(x_93, x_94);
if (x_95 == 0)
{
lean_object* x_96; uint8_t x_97; 
x_96 = lp_mathlib_CancelDenoms_findCompLemma___closed__1;
x_97 = lean_string_dec_eq(x_93, x_96);
if (x_97 == 0)
{
lean_object* x_98; uint8_t x_99; 
x_98 = lp_mathlib_CancelDenoms_findCompLemma___closed__2;
x_99 = lean_string_dec_eq(x_93, x_98);
if (x_99 == 0)
{
lean_object* x_100; uint8_t x_101; 
x_100 = lp_mathlib_CancelDenoms_findCompLemma___closed__3;
x_101 = lean_string_dec_eq(x_93, x_100);
lean_dec_ref(x_93);
if (x_101 == 0)
{
lean_dec_ref(x_92);
lean_dec(x_91);
lean_free_object(x_15);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_102; uint8_t x_103; 
x_102 = lp_mathlib_CancelDenoms_findCompLemma___closed__4;
x_103 = lean_string_dec_eq(x_92, x_102);
lean_dec_ref(x_92);
if (x_103 == 0)
{
lean_dec(x_91);
lean_free_object(x_15);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_104; lean_object* x_105; uint8_t x_106; 
x_104 = lean_array_get_size(x_91);
x_105 = lean_unsigned_to_nat(4u);
x_106 = lean_nat_dec_eq(x_104, x_105);
if (x_106 == 0)
{
lean_dec(x_91);
lean_free_object(x_15);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; lean_object* x_116; 
x_107 = lean_unsigned_to_nat(2u);
x_108 = lean_array_fget(x_91, x_107);
x_109 = lean_unsigned_to_nat(3u);
x_110 = lean_array_fget(x_91, x_109);
lean_dec(x_91);
x_111 = lp_mathlib_CancelDenoms_findCompLemma___closed__6;
x_112 = lean_box(x_106);
x_113 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_113, 0, x_111);
lean_ctor_set(x_113, 1, x_112);
x_114 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_114, 0, x_108);
lean_ctor_set(x_114, 1, x_113);
x_115 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_115, 0, x_110);
lean_ctor_set(x_115, 1, x_114);
x_116 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_116, 0, x_115);
lean_ctor_set(x_15, 0, x_116);
return x_15;
}
}
}
}
else
{
lean_object* x_117; uint8_t x_118; 
lean_dec_ref(x_93);
x_117 = lp_mathlib_CancelDenoms_findCompLemma___closed__7;
x_118 = lean_string_dec_eq(x_92, x_117);
lean_dec_ref(x_92);
if (x_118 == 0)
{
lean_dec(x_91);
lean_free_object(x_15);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_119; lean_object* x_120; uint8_t x_121; 
x_119 = lean_array_get_size(x_91);
x_120 = lean_unsigned_to_nat(4u);
x_121 = lean_nat_dec_eq(x_119, x_120);
if (x_121 == 0)
{
lean_dec(x_91);
lean_free_object(x_15);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_127; lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; 
x_122 = lean_unsigned_to_nat(2u);
x_123 = lean_array_fget(x_91, x_122);
x_124 = lean_unsigned_to_nat(3u);
x_125 = lean_array_fget(x_91, x_124);
lean_dec(x_91);
x_126 = lp_mathlib_CancelDenoms_findCompLemma___closed__9;
x_127 = lean_box(x_121);
x_128 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_128, 0, x_126);
lean_ctor_set(x_128, 1, x_127);
x_129 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_129, 0, x_123);
lean_ctor_set(x_129, 1, x_128);
x_130 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_130, 0, x_125);
lean_ctor_set(x_130, 1, x_129);
x_131 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_131, 0, x_130);
lean_ctor_set(x_15, 0, x_131);
return x_15;
}
}
}
}
else
{
lean_object* x_132; uint8_t x_133; 
lean_dec_ref(x_93);
x_132 = lp_mathlib_CancelDenoms_findCompLemma___closed__10;
x_133 = lean_string_dec_eq(x_92, x_132);
lean_dec_ref(x_92);
if (x_133 == 0)
{
lean_dec(x_91);
lean_free_object(x_15);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_134; lean_object* x_135; uint8_t x_136; 
x_134 = lean_array_get_size(x_91);
x_135 = lean_unsigned_to_nat(4u);
x_136 = lean_nat_dec_eq(x_134, x_135);
if (x_136 == 0)
{
lean_dec(x_91);
lean_free_object(x_15);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_137; lean_object* x_138; lean_object* x_139; lean_object* x_140; lean_object* x_141; lean_object* x_142; lean_object* x_143; lean_object* x_144; lean_object* x_145; lean_object* x_146; 
x_137 = lean_unsigned_to_nat(2u);
x_138 = lean_array_fget(x_91, x_137);
x_139 = lean_unsigned_to_nat(3u);
x_140 = lean_array_fget(x_91, x_139);
lean_dec(x_91);
x_141 = lp_mathlib_CancelDenoms_findCompLemma___closed__9;
x_142 = lean_box(x_136);
x_143 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_143, 0, x_141);
lean_ctor_set(x_143, 1, x_142);
x_144 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_144, 0, x_140);
lean_ctor_set(x_144, 1, x_143);
x_145 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_145, 0, x_138);
lean_ctor_set(x_145, 1, x_144);
x_146 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_146, 0, x_145);
lean_ctor_set(x_15, 0, x_146);
return x_15;
}
}
}
}
else
{
lean_object* x_147; uint8_t x_148; 
lean_dec_ref(x_93);
x_147 = lp_mathlib_CancelDenoms_findCompLemma___closed__11;
x_148 = lean_string_dec_eq(x_92, x_147);
lean_dec_ref(x_92);
if (x_148 == 0)
{
lean_dec(x_91);
lean_free_object(x_15);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_149; lean_object* x_150; uint8_t x_151; 
x_149 = lean_array_get_size(x_91);
x_150 = lean_unsigned_to_nat(4u);
x_151 = lean_nat_dec_eq(x_149, x_150);
if (x_151 == 0)
{
lean_dec(x_91);
lean_free_object(x_15);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_152; lean_object* x_153; lean_object* x_154; lean_object* x_155; lean_object* x_156; lean_object* x_157; lean_object* x_158; lean_object* x_159; lean_object* x_160; lean_object* x_161; 
x_152 = lean_unsigned_to_nat(2u);
x_153 = lean_array_fget(x_91, x_152);
x_154 = lean_unsigned_to_nat(3u);
x_155 = lean_array_fget(x_91, x_154);
lean_dec(x_91);
x_156 = lp_mathlib_CancelDenoms_findCompLemma___closed__6;
x_157 = lean_box(x_151);
x_158 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_158, 0, x_156);
lean_ctor_set(x_158, 1, x_157);
x_159 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_159, 0, x_155);
lean_ctor_set(x_159, 1, x_158);
x_160 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_160, 0, x_153);
lean_ctor_set(x_160, 1, x_159);
x_161 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_161, 0, x_160);
lean_ctor_set(x_15, 0, x_161);
return x_15;
}
}
}
}
}
else
{
lean_dec_ref(x_20);
lean_dec_ref(x_19);
lean_dec_ref(x_18);
lean_free_object(x_15);
x_7 = lean_box(0);
goto block_10;
}
}
case 0:
{
uint8_t x_162; 
x_162 = !lean_is_exclusive(x_18);
if (x_162 == 0)
{
lean_object* x_163; lean_object* x_164; lean_object* x_165; lean_object* x_166; uint8_t x_167; 
x_163 = lean_ctor_get(x_18, 1);
x_164 = lean_ctor_get(x_18, 0);
lean_dec(x_164);
x_165 = lean_ctor_get(x_19, 1);
lean_inc_ref(x_165);
lean_dec_ref(x_19);
x_166 = lp_mathlib_CancelDenoms_mkProdPrf___closed__82;
x_167 = lean_string_dec_eq(x_165, x_166);
if (x_167 == 0)
{
lean_object* x_168; uint8_t x_169; 
lean_free_object(x_15);
x_168 = lp_mathlib_CancelDenoms_findCompLemma___closed__12;
x_169 = lean_string_dec_eq(x_165, x_168);
lean_dec_ref(x_165);
if (x_169 == 0)
{
lean_free_object(x_18);
lean_dec(x_163);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_170; lean_object* x_171; uint8_t x_172; 
x_170 = lean_array_get_size(x_163);
x_171 = lean_unsigned_to_nat(1u);
x_172 = lean_nat_dec_eq(x_170, x_171);
if (x_172 == 0)
{
lean_free_object(x_18);
lean_dec(x_163);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_173; lean_object* x_174; lean_object* x_175; 
x_173 = lean_unsigned_to_nat(0u);
x_174 = lean_array_fget(x_163, x_173);
lean_dec(x_163);
x_175 = l_Lean_Meta_whnfR(x_174, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_175) == 0)
{
uint8_t x_176; 
x_176 = !lean_is_exclusive(x_175);
if (x_176 == 0)
{
lean_object* x_177; lean_object* x_178; lean_object* x_179; 
x_177 = lean_ctor_get(x_175, 0);
x_178 = l_Lean_Expr_getAppFnArgs(x_177);
x_179 = lean_ctor_get(x_178, 0);
lean_inc(x_179);
if (lean_obj_tag(x_179) == 1)
{
lean_object* x_180; 
x_180 = lean_ctor_get(x_179, 0);
if (lean_obj_tag(x_180) == 0)
{
uint8_t x_181; 
x_181 = !lean_is_exclusive(x_178);
if (x_181 == 0)
{
lean_object* x_182; lean_object* x_183; lean_object* x_184; uint8_t x_185; 
x_182 = lean_ctor_get(x_178, 1);
x_183 = lean_ctor_get(x_178, 0);
lean_dec(x_183);
x_184 = lean_ctor_get(x_179, 1);
lean_inc_ref(x_184);
lean_dec_ref(x_179);
x_185 = lean_string_dec_eq(x_184, x_166);
lean_dec_ref(x_184);
if (x_185 == 0)
{
lean_free_object(x_178);
lean_dec(x_182);
lean_free_object(x_175);
lean_free_object(x_18);
x_11 = lean_box(0);
goto block_14;
}
else
{
lean_object* x_186; lean_object* x_187; uint8_t x_188; 
x_186 = lean_array_get_size(x_182);
x_187 = lean_unsigned_to_nat(3u);
x_188 = lean_nat_dec_eq(x_186, x_187);
if (x_188 == 0)
{
lean_free_object(x_178);
lean_dec(x_182);
lean_free_object(x_175);
lean_free_object(x_18);
x_11 = lean_box(0);
goto block_14;
}
else
{
lean_object* x_189; lean_object* x_190; lean_object* x_191; lean_object* x_192; lean_object* x_193; lean_object* x_194; lean_object* x_195; 
x_189 = lean_array_fget(x_182, x_171);
x_190 = lean_unsigned_to_nat(2u);
x_191 = lean_array_fget(x_182, x_190);
lean_dec(x_182);
x_192 = lp_mathlib_CancelDenoms_findCompLemma___closed__14;
x_193 = lean_box(x_167);
lean_ctor_set(x_178, 1, x_193);
lean_ctor_set(x_178, 0, x_192);
lean_ctor_set(x_18, 1, x_178);
lean_ctor_set(x_18, 0, x_191);
x_194 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_194, 0, x_189);
lean_ctor_set(x_194, 1, x_18);
x_195 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_195, 0, x_194);
lean_ctor_set(x_175, 0, x_195);
return x_175;
}
}
}
else
{
lean_object* x_196; lean_object* x_197; uint8_t x_198; 
x_196 = lean_ctor_get(x_178, 1);
lean_inc(x_196);
lean_dec(x_178);
x_197 = lean_ctor_get(x_179, 1);
lean_inc_ref(x_197);
lean_dec_ref(x_179);
x_198 = lean_string_dec_eq(x_197, x_166);
lean_dec_ref(x_197);
if (x_198 == 0)
{
lean_dec(x_196);
lean_free_object(x_175);
lean_free_object(x_18);
x_11 = lean_box(0);
goto block_14;
}
else
{
lean_object* x_199; lean_object* x_200; uint8_t x_201; 
x_199 = lean_array_get_size(x_196);
x_200 = lean_unsigned_to_nat(3u);
x_201 = lean_nat_dec_eq(x_199, x_200);
if (x_201 == 0)
{
lean_dec(x_196);
lean_free_object(x_175);
lean_free_object(x_18);
x_11 = lean_box(0);
goto block_14;
}
else
{
lean_object* x_202; lean_object* x_203; lean_object* x_204; lean_object* x_205; lean_object* x_206; lean_object* x_207; lean_object* x_208; lean_object* x_209; 
x_202 = lean_array_fget(x_196, x_171);
x_203 = lean_unsigned_to_nat(2u);
x_204 = lean_array_fget(x_196, x_203);
lean_dec(x_196);
x_205 = lp_mathlib_CancelDenoms_findCompLemma___closed__14;
x_206 = lean_box(x_167);
x_207 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_207, 0, x_205);
lean_ctor_set(x_207, 1, x_206);
lean_ctor_set(x_18, 1, x_207);
lean_ctor_set(x_18, 0, x_204);
x_208 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_208, 0, x_202);
lean_ctor_set(x_208, 1, x_18);
x_209 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_209, 0, x_208);
lean_ctor_set(x_175, 0, x_209);
return x_175;
}
}
}
}
else
{
lean_dec_ref(x_179);
lean_dec_ref(x_178);
lean_free_object(x_175);
lean_free_object(x_18);
x_11 = lean_box(0);
goto block_14;
}
}
else
{
lean_dec(x_179);
lean_dec_ref(x_178);
lean_free_object(x_175);
lean_free_object(x_18);
x_11 = lean_box(0);
goto block_14;
}
}
else
{
lean_object* x_210; lean_object* x_211; lean_object* x_212; 
x_210 = lean_ctor_get(x_175, 0);
lean_inc(x_210);
lean_dec(x_175);
x_211 = l_Lean_Expr_getAppFnArgs(x_210);
x_212 = lean_ctor_get(x_211, 0);
lean_inc(x_212);
if (lean_obj_tag(x_212) == 1)
{
lean_object* x_213; 
x_213 = lean_ctor_get(x_212, 0);
if (lean_obj_tag(x_213) == 0)
{
lean_object* x_214; lean_object* x_215; lean_object* x_216; uint8_t x_217; 
x_214 = lean_ctor_get(x_211, 1);
lean_inc(x_214);
if (lean_is_exclusive(x_211)) {
 lean_ctor_release(x_211, 0);
 lean_ctor_release(x_211, 1);
 x_215 = x_211;
} else {
 lean_dec_ref(x_211);
 x_215 = lean_box(0);
}
x_216 = lean_ctor_get(x_212, 1);
lean_inc_ref(x_216);
lean_dec_ref(x_212);
x_217 = lean_string_dec_eq(x_216, x_166);
lean_dec_ref(x_216);
if (x_217 == 0)
{
lean_dec(x_215);
lean_dec(x_214);
lean_free_object(x_18);
x_11 = lean_box(0);
goto block_14;
}
else
{
lean_object* x_218; lean_object* x_219; uint8_t x_220; 
x_218 = lean_array_get_size(x_214);
x_219 = lean_unsigned_to_nat(3u);
x_220 = lean_nat_dec_eq(x_218, x_219);
if (x_220 == 0)
{
lean_dec(x_215);
lean_dec(x_214);
lean_free_object(x_18);
x_11 = lean_box(0);
goto block_14;
}
else
{
lean_object* x_221; lean_object* x_222; lean_object* x_223; lean_object* x_224; lean_object* x_225; lean_object* x_226; lean_object* x_227; lean_object* x_228; lean_object* x_229; 
x_221 = lean_array_fget(x_214, x_171);
x_222 = lean_unsigned_to_nat(2u);
x_223 = lean_array_fget(x_214, x_222);
lean_dec(x_214);
x_224 = lp_mathlib_CancelDenoms_findCompLemma___closed__14;
x_225 = lean_box(x_167);
if (lean_is_scalar(x_215)) {
 x_226 = lean_alloc_ctor(0, 2, 0);
} else {
 x_226 = x_215;
}
lean_ctor_set(x_226, 0, x_224);
lean_ctor_set(x_226, 1, x_225);
lean_ctor_set(x_18, 1, x_226);
lean_ctor_set(x_18, 0, x_223);
x_227 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_227, 0, x_221);
lean_ctor_set(x_227, 1, x_18);
x_228 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_228, 0, x_227);
x_229 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_229, 0, x_228);
return x_229;
}
}
}
else
{
lean_dec_ref(x_212);
lean_dec_ref(x_211);
lean_free_object(x_18);
x_11 = lean_box(0);
goto block_14;
}
}
else
{
lean_dec(x_212);
lean_dec_ref(x_211);
lean_free_object(x_18);
x_11 = lean_box(0);
goto block_14;
}
}
}
else
{
uint8_t x_230; 
lean_free_object(x_18);
x_230 = !lean_is_exclusive(x_175);
if (x_230 == 0)
{
return x_175;
}
else
{
lean_object* x_231; lean_object* x_232; 
x_231 = lean_ctor_get(x_175, 0);
lean_inc(x_231);
lean_dec(x_175);
x_232 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_232, 0, x_231);
return x_232;
}
}
}
}
}
else
{
lean_object* x_233; lean_object* x_234; uint8_t x_235; 
lean_dec_ref(x_165);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_233 = lean_array_get_size(x_163);
x_234 = lean_unsigned_to_nat(3u);
x_235 = lean_nat_dec_eq(x_233, x_234);
if (x_235 == 0)
{
lean_free_object(x_18);
lean_dec(x_163);
lean_free_object(x_15);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_236; lean_object* x_237; lean_object* x_238; lean_object* x_239; lean_object* x_240; lean_object* x_241; lean_object* x_242; 
x_236 = lean_unsigned_to_nat(1u);
x_237 = lean_array_fget(x_163, x_236);
x_238 = lean_unsigned_to_nat(2u);
x_239 = lean_array_fget(x_163, x_238);
lean_dec(x_163);
x_240 = lp_mathlib_CancelDenoms_findCompLemma___closed__17;
lean_ctor_set(x_18, 1, x_240);
lean_ctor_set(x_18, 0, x_239);
x_241 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_241, 0, x_237);
lean_ctor_set(x_241, 1, x_18);
x_242 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_242, 0, x_241);
lean_ctor_set(x_15, 0, x_242);
return x_15;
}
}
}
else
{
lean_object* x_243; lean_object* x_244; lean_object* x_245; uint8_t x_246; 
x_243 = lean_ctor_get(x_18, 1);
lean_inc(x_243);
lean_dec(x_18);
x_244 = lean_ctor_get(x_19, 1);
lean_inc_ref(x_244);
lean_dec_ref(x_19);
x_245 = lp_mathlib_CancelDenoms_mkProdPrf___closed__82;
x_246 = lean_string_dec_eq(x_244, x_245);
if (x_246 == 0)
{
lean_object* x_247; uint8_t x_248; 
lean_free_object(x_15);
x_247 = lp_mathlib_CancelDenoms_findCompLemma___closed__12;
x_248 = lean_string_dec_eq(x_244, x_247);
lean_dec_ref(x_244);
if (x_248 == 0)
{
lean_dec(x_243);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_249; lean_object* x_250; uint8_t x_251; 
x_249 = lean_array_get_size(x_243);
x_250 = lean_unsigned_to_nat(1u);
x_251 = lean_nat_dec_eq(x_249, x_250);
if (x_251 == 0)
{
lean_dec(x_243);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_252; lean_object* x_253; lean_object* x_254; 
x_252 = lean_unsigned_to_nat(0u);
x_253 = lean_array_fget(x_243, x_252);
lean_dec(x_243);
x_254 = l_Lean_Meta_whnfR(x_253, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_254) == 0)
{
lean_object* x_255; lean_object* x_256; lean_object* x_257; lean_object* x_258; 
x_255 = lean_ctor_get(x_254, 0);
lean_inc(x_255);
if (lean_is_exclusive(x_254)) {
 lean_ctor_release(x_254, 0);
 x_256 = x_254;
} else {
 lean_dec_ref(x_254);
 x_256 = lean_box(0);
}
x_257 = l_Lean_Expr_getAppFnArgs(x_255);
x_258 = lean_ctor_get(x_257, 0);
lean_inc(x_258);
if (lean_obj_tag(x_258) == 1)
{
lean_object* x_259; 
x_259 = lean_ctor_get(x_258, 0);
if (lean_obj_tag(x_259) == 0)
{
lean_object* x_260; lean_object* x_261; lean_object* x_262; uint8_t x_263; 
x_260 = lean_ctor_get(x_257, 1);
lean_inc(x_260);
if (lean_is_exclusive(x_257)) {
 lean_ctor_release(x_257, 0);
 lean_ctor_release(x_257, 1);
 x_261 = x_257;
} else {
 lean_dec_ref(x_257);
 x_261 = lean_box(0);
}
x_262 = lean_ctor_get(x_258, 1);
lean_inc_ref(x_262);
lean_dec_ref(x_258);
x_263 = lean_string_dec_eq(x_262, x_245);
lean_dec_ref(x_262);
if (x_263 == 0)
{
lean_dec(x_261);
lean_dec(x_260);
lean_dec(x_256);
x_11 = lean_box(0);
goto block_14;
}
else
{
lean_object* x_264; lean_object* x_265; uint8_t x_266; 
x_264 = lean_array_get_size(x_260);
x_265 = lean_unsigned_to_nat(3u);
x_266 = lean_nat_dec_eq(x_264, x_265);
if (x_266 == 0)
{
lean_dec(x_261);
lean_dec(x_260);
lean_dec(x_256);
x_11 = lean_box(0);
goto block_14;
}
else
{
lean_object* x_267; lean_object* x_268; lean_object* x_269; lean_object* x_270; lean_object* x_271; lean_object* x_272; lean_object* x_273; lean_object* x_274; lean_object* x_275; lean_object* x_276; 
x_267 = lean_array_fget(x_260, x_250);
x_268 = lean_unsigned_to_nat(2u);
x_269 = lean_array_fget(x_260, x_268);
lean_dec(x_260);
x_270 = lp_mathlib_CancelDenoms_findCompLemma___closed__14;
x_271 = lean_box(x_246);
if (lean_is_scalar(x_261)) {
 x_272 = lean_alloc_ctor(0, 2, 0);
} else {
 x_272 = x_261;
}
lean_ctor_set(x_272, 0, x_270);
lean_ctor_set(x_272, 1, x_271);
x_273 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_273, 0, x_269);
lean_ctor_set(x_273, 1, x_272);
x_274 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_274, 0, x_267);
lean_ctor_set(x_274, 1, x_273);
x_275 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_275, 0, x_274);
if (lean_is_scalar(x_256)) {
 x_276 = lean_alloc_ctor(0, 1, 0);
} else {
 x_276 = x_256;
}
lean_ctor_set(x_276, 0, x_275);
return x_276;
}
}
}
else
{
lean_dec_ref(x_258);
lean_dec_ref(x_257);
lean_dec(x_256);
x_11 = lean_box(0);
goto block_14;
}
}
else
{
lean_dec(x_258);
lean_dec_ref(x_257);
lean_dec(x_256);
x_11 = lean_box(0);
goto block_14;
}
}
else
{
lean_object* x_277; lean_object* x_278; lean_object* x_279; 
x_277 = lean_ctor_get(x_254, 0);
lean_inc(x_277);
if (lean_is_exclusive(x_254)) {
 lean_ctor_release(x_254, 0);
 x_278 = x_254;
} else {
 lean_dec_ref(x_254);
 x_278 = lean_box(0);
}
if (lean_is_scalar(x_278)) {
 x_279 = lean_alloc_ctor(1, 1, 0);
} else {
 x_279 = x_278;
}
lean_ctor_set(x_279, 0, x_277);
return x_279;
}
}
}
}
else
{
lean_object* x_280; lean_object* x_281; uint8_t x_282; 
lean_dec_ref(x_244);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_280 = lean_array_get_size(x_243);
x_281 = lean_unsigned_to_nat(3u);
x_282 = lean_nat_dec_eq(x_280, x_281);
if (x_282 == 0)
{
lean_dec(x_243);
lean_free_object(x_15);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_283; lean_object* x_284; lean_object* x_285; lean_object* x_286; lean_object* x_287; lean_object* x_288; lean_object* x_289; lean_object* x_290; 
x_283 = lean_unsigned_to_nat(1u);
x_284 = lean_array_fget(x_243, x_283);
x_285 = lean_unsigned_to_nat(2u);
x_286 = lean_array_fget(x_243, x_285);
lean_dec(x_243);
x_287 = lp_mathlib_CancelDenoms_findCompLemma___closed__17;
x_288 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_288, 0, x_286);
lean_ctor_set(x_288, 1, x_287);
x_289 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_289, 0, x_284);
lean_ctor_set(x_289, 1, x_288);
x_290 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_290, 0, x_289);
lean_ctor_set(x_15, 0, x_290);
return x_15;
}
}
}
}
default: 
{
lean_dec_ref(x_19);
lean_dec_ref(x_18);
lean_free_object(x_15);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_7 = lean_box(0);
goto block_10;
}
}
}
else
{
lean_dec(x_19);
lean_dec_ref(x_18);
lean_free_object(x_15);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_7 = lean_box(0);
goto block_10;
}
}
else
{
lean_object* x_291; lean_object* x_292; lean_object* x_293; 
x_291 = lean_ctor_get(x_15, 0);
lean_inc(x_291);
lean_dec(x_15);
x_292 = l_Lean_Expr_getAppFnArgs(x_291);
x_293 = lean_ctor_get(x_292, 0);
lean_inc(x_293);
if (lean_obj_tag(x_293) == 1)
{
lean_object* x_294; 
x_294 = lean_ctor_get(x_293, 0);
switch (lean_obj_tag(x_294)) {
case 1:
{
lean_object* x_295; 
lean_inc_ref(x_294);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_295 = lean_ctor_get(x_294, 0);
if (lean_obj_tag(x_295) == 0)
{
lean_object* x_296; lean_object* x_297; lean_object* x_298; lean_object* x_299; lean_object* x_300; uint8_t x_301; 
x_296 = lean_ctor_get(x_292, 1);
lean_inc(x_296);
if (lean_is_exclusive(x_292)) {
 lean_ctor_release(x_292, 0);
 lean_ctor_release(x_292, 1);
 x_297 = x_292;
} else {
 lean_dec_ref(x_292);
 x_297 = lean_box(0);
}
x_298 = lean_ctor_get(x_293, 1);
lean_inc_ref(x_298);
lean_dec_ref(x_293);
x_299 = lean_ctor_get(x_294, 1);
lean_inc_ref(x_299);
lean_dec_ref(x_294);
x_300 = lp_mathlib_CancelDenoms_findCompLemma___closed__0;
x_301 = lean_string_dec_eq(x_299, x_300);
if (x_301 == 0)
{
lean_object* x_302; uint8_t x_303; 
x_302 = lp_mathlib_CancelDenoms_findCompLemma___closed__1;
x_303 = lean_string_dec_eq(x_299, x_302);
if (x_303 == 0)
{
lean_object* x_304; uint8_t x_305; 
x_304 = lp_mathlib_CancelDenoms_findCompLemma___closed__2;
x_305 = lean_string_dec_eq(x_299, x_304);
if (x_305 == 0)
{
lean_object* x_306; uint8_t x_307; 
x_306 = lp_mathlib_CancelDenoms_findCompLemma___closed__3;
x_307 = lean_string_dec_eq(x_299, x_306);
lean_dec_ref(x_299);
if (x_307 == 0)
{
lean_dec_ref(x_298);
lean_dec(x_297);
lean_dec(x_296);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_308; uint8_t x_309; 
x_308 = lp_mathlib_CancelDenoms_findCompLemma___closed__4;
x_309 = lean_string_dec_eq(x_298, x_308);
lean_dec_ref(x_298);
if (x_309 == 0)
{
lean_dec(x_297);
lean_dec(x_296);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_310; lean_object* x_311; uint8_t x_312; 
x_310 = lean_array_get_size(x_296);
x_311 = lean_unsigned_to_nat(4u);
x_312 = lean_nat_dec_eq(x_310, x_311);
if (x_312 == 0)
{
lean_dec(x_297);
lean_dec(x_296);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_313; lean_object* x_314; lean_object* x_315; lean_object* x_316; lean_object* x_317; lean_object* x_318; lean_object* x_319; lean_object* x_320; lean_object* x_321; lean_object* x_322; lean_object* x_323; 
x_313 = lean_unsigned_to_nat(2u);
x_314 = lean_array_fget(x_296, x_313);
x_315 = lean_unsigned_to_nat(3u);
x_316 = lean_array_fget(x_296, x_315);
lean_dec(x_296);
x_317 = lp_mathlib_CancelDenoms_findCompLemma___closed__6;
x_318 = lean_box(x_312);
if (lean_is_scalar(x_297)) {
 x_319 = lean_alloc_ctor(0, 2, 0);
} else {
 x_319 = x_297;
}
lean_ctor_set(x_319, 0, x_317);
lean_ctor_set(x_319, 1, x_318);
x_320 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_320, 0, x_314);
lean_ctor_set(x_320, 1, x_319);
x_321 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_321, 0, x_316);
lean_ctor_set(x_321, 1, x_320);
x_322 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_322, 0, x_321);
x_323 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_323, 0, x_322);
return x_323;
}
}
}
}
else
{
lean_object* x_324; uint8_t x_325; 
lean_dec_ref(x_299);
x_324 = lp_mathlib_CancelDenoms_findCompLemma___closed__7;
x_325 = lean_string_dec_eq(x_298, x_324);
lean_dec_ref(x_298);
if (x_325 == 0)
{
lean_dec(x_297);
lean_dec(x_296);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_326; lean_object* x_327; uint8_t x_328; 
x_326 = lean_array_get_size(x_296);
x_327 = lean_unsigned_to_nat(4u);
x_328 = lean_nat_dec_eq(x_326, x_327);
if (x_328 == 0)
{
lean_dec(x_297);
lean_dec(x_296);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_329; lean_object* x_330; lean_object* x_331; lean_object* x_332; lean_object* x_333; lean_object* x_334; lean_object* x_335; lean_object* x_336; lean_object* x_337; lean_object* x_338; lean_object* x_339; 
x_329 = lean_unsigned_to_nat(2u);
x_330 = lean_array_fget(x_296, x_329);
x_331 = lean_unsigned_to_nat(3u);
x_332 = lean_array_fget(x_296, x_331);
lean_dec(x_296);
x_333 = lp_mathlib_CancelDenoms_findCompLemma___closed__9;
x_334 = lean_box(x_328);
if (lean_is_scalar(x_297)) {
 x_335 = lean_alloc_ctor(0, 2, 0);
} else {
 x_335 = x_297;
}
lean_ctor_set(x_335, 0, x_333);
lean_ctor_set(x_335, 1, x_334);
x_336 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_336, 0, x_330);
lean_ctor_set(x_336, 1, x_335);
x_337 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_337, 0, x_332);
lean_ctor_set(x_337, 1, x_336);
x_338 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_338, 0, x_337);
x_339 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_339, 0, x_338);
return x_339;
}
}
}
}
else
{
lean_object* x_340; uint8_t x_341; 
lean_dec_ref(x_299);
x_340 = lp_mathlib_CancelDenoms_findCompLemma___closed__10;
x_341 = lean_string_dec_eq(x_298, x_340);
lean_dec_ref(x_298);
if (x_341 == 0)
{
lean_dec(x_297);
lean_dec(x_296);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_342; lean_object* x_343; uint8_t x_344; 
x_342 = lean_array_get_size(x_296);
x_343 = lean_unsigned_to_nat(4u);
x_344 = lean_nat_dec_eq(x_342, x_343);
if (x_344 == 0)
{
lean_dec(x_297);
lean_dec(x_296);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_345; lean_object* x_346; lean_object* x_347; lean_object* x_348; lean_object* x_349; lean_object* x_350; lean_object* x_351; lean_object* x_352; lean_object* x_353; lean_object* x_354; lean_object* x_355; 
x_345 = lean_unsigned_to_nat(2u);
x_346 = lean_array_fget(x_296, x_345);
x_347 = lean_unsigned_to_nat(3u);
x_348 = lean_array_fget(x_296, x_347);
lean_dec(x_296);
x_349 = lp_mathlib_CancelDenoms_findCompLemma___closed__9;
x_350 = lean_box(x_344);
if (lean_is_scalar(x_297)) {
 x_351 = lean_alloc_ctor(0, 2, 0);
} else {
 x_351 = x_297;
}
lean_ctor_set(x_351, 0, x_349);
lean_ctor_set(x_351, 1, x_350);
x_352 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_352, 0, x_348);
lean_ctor_set(x_352, 1, x_351);
x_353 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_353, 0, x_346);
lean_ctor_set(x_353, 1, x_352);
x_354 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_354, 0, x_353);
x_355 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_355, 0, x_354);
return x_355;
}
}
}
}
else
{
lean_object* x_356; uint8_t x_357; 
lean_dec_ref(x_299);
x_356 = lp_mathlib_CancelDenoms_findCompLemma___closed__11;
x_357 = lean_string_dec_eq(x_298, x_356);
lean_dec_ref(x_298);
if (x_357 == 0)
{
lean_dec(x_297);
lean_dec(x_296);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_358; lean_object* x_359; uint8_t x_360; 
x_358 = lean_array_get_size(x_296);
x_359 = lean_unsigned_to_nat(4u);
x_360 = lean_nat_dec_eq(x_358, x_359);
if (x_360 == 0)
{
lean_dec(x_297);
lean_dec(x_296);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_361; lean_object* x_362; lean_object* x_363; lean_object* x_364; lean_object* x_365; lean_object* x_366; lean_object* x_367; lean_object* x_368; lean_object* x_369; lean_object* x_370; lean_object* x_371; 
x_361 = lean_unsigned_to_nat(2u);
x_362 = lean_array_fget(x_296, x_361);
x_363 = lean_unsigned_to_nat(3u);
x_364 = lean_array_fget(x_296, x_363);
lean_dec(x_296);
x_365 = lp_mathlib_CancelDenoms_findCompLemma___closed__6;
x_366 = lean_box(x_360);
if (lean_is_scalar(x_297)) {
 x_367 = lean_alloc_ctor(0, 2, 0);
} else {
 x_367 = x_297;
}
lean_ctor_set(x_367, 0, x_365);
lean_ctor_set(x_367, 1, x_366);
x_368 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_368, 0, x_364);
lean_ctor_set(x_368, 1, x_367);
x_369 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_369, 0, x_362);
lean_ctor_set(x_369, 1, x_368);
x_370 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_370, 0, x_369);
x_371 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_371, 0, x_370);
return x_371;
}
}
}
}
else
{
lean_dec_ref(x_294);
lean_dec_ref(x_293);
lean_dec_ref(x_292);
x_7 = lean_box(0);
goto block_10;
}
}
case 0:
{
lean_object* x_372; lean_object* x_373; lean_object* x_374; lean_object* x_375; uint8_t x_376; 
x_372 = lean_ctor_get(x_292, 1);
lean_inc(x_372);
if (lean_is_exclusive(x_292)) {
 lean_ctor_release(x_292, 0);
 lean_ctor_release(x_292, 1);
 x_373 = x_292;
} else {
 lean_dec_ref(x_292);
 x_373 = lean_box(0);
}
x_374 = lean_ctor_get(x_293, 1);
lean_inc_ref(x_374);
lean_dec_ref(x_293);
x_375 = lp_mathlib_CancelDenoms_mkProdPrf___closed__82;
x_376 = lean_string_dec_eq(x_374, x_375);
if (x_376 == 0)
{
lean_object* x_377; uint8_t x_378; 
x_377 = lp_mathlib_CancelDenoms_findCompLemma___closed__12;
x_378 = lean_string_dec_eq(x_374, x_377);
lean_dec_ref(x_374);
if (x_378 == 0)
{
lean_dec(x_373);
lean_dec(x_372);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_379; lean_object* x_380; uint8_t x_381; 
x_379 = lean_array_get_size(x_372);
x_380 = lean_unsigned_to_nat(1u);
x_381 = lean_nat_dec_eq(x_379, x_380);
if (x_381 == 0)
{
lean_dec(x_373);
lean_dec(x_372);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_382; lean_object* x_383; lean_object* x_384; 
x_382 = lean_unsigned_to_nat(0u);
x_383 = lean_array_fget(x_372, x_382);
lean_dec(x_372);
x_384 = l_Lean_Meta_whnfR(x_383, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_384) == 0)
{
lean_object* x_385; lean_object* x_386; lean_object* x_387; lean_object* x_388; 
x_385 = lean_ctor_get(x_384, 0);
lean_inc(x_385);
if (lean_is_exclusive(x_384)) {
 lean_ctor_release(x_384, 0);
 x_386 = x_384;
} else {
 lean_dec_ref(x_384);
 x_386 = lean_box(0);
}
x_387 = l_Lean_Expr_getAppFnArgs(x_385);
x_388 = lean_ctor_get(x_387, 0);
lean_inc(x_388);
if (lean_obj_tag(x_388) == 1)
{
lean_object* x_389; 
x_389 = lean_ctor_get(x_388, 0);
if (lean_obj_tag(x_389) == 0)
{
lean_object* x_390; lean_object* x_391; lean_object* x_392; uint8_t x_393; 
x_390 = lean_ctor_get(x_387, 1);
lean_inc(x_390);
if (lean_is_exclusive(x_387)) {
 lean_ctor_release(x_387, 0);
 lean_ctor_release(x_387, 1);
 x_391 = x_387;
} else {
 lean_dec_ref(x_387);
 x_391 = lean_box(0);
}
x_392 = lean_ctor_get(x_388, 1);
lean_inc_ref(x_392);
lean_dec_ref(x_388);
x_393 = lean_string_dec_eq(x_392, x_375);
lean_dec_ref(x_392);
if (x_393 == 0)
{
lean_dec(x_391);
lean_dec(x_390);
lean_dec(x_386);
lean_dec(x_373);
x_11 = lean_box(0);
goto block_14;
}
else
{
lean_object* x_394; lean_object* x_395; uint8_t x_396; 
x_394 = lean_array_get_size(x_390);
x_395 = lean_unsigned_to_nat(3u);
x_396 = lean_nat_dec_eq(x_394, x_395);
if (x_396 == 0)
{
lean_dec(x_391);
lean_dec(x_390);
lean_dec(x_386);
lean_dec(x_373);
x_11 = lean_box(0);
goto block_14;
}
else
{
lean_object* x_397; lean_object* x_398; lean_object* x_399; lean_object* x_400; lean_object* x_401; lean_object* x_402; lean_object* x_403; lean_object* x_404; lean_object* x_405; lean_object* x_406; 
x_397 = lean_array_fget(x_390, x_380);
x_398 = lean_unsigned_to_nat(2u);
x_399 = lean_array_fget(x_390, x_398);
lean_dec(x_390);
x_400 = lp_mathlib_CancelDenoms_findCompLemma___closed__14;
x_401 = lean_box(x_376);
if (lean_is_scalar(x_391)) {
 x_402 = lean_alloc_ctor(0, 2, 0);
} else {
 x_402 = x_391;
}
lean_ctor_set(x_402, 0, x_400);
lean_ctor_set(x_402, 1, x_401);
if (lean_is_scalar(x_373)) {
 x_403 = lean_alloc_ctor(0, 2, 0);
} else {
 x_403 = x_373;
}
lean_ctor_set(x_403, 0, x_399);
lean_ctor_set(x_403, 1, x_402);
x_404 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_404, 0, x_397);
lean_ctor_set(x_404, 1, x_403);
x_405 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_405, 0, x_404);
if (lean_is_scalar(x_386)) {
 x_406 = lean_alloc_ctor(0, 1, 0);
} else {
 x_406 = x_386;
}
lean_ctor_set(x_406, 0, x_405);
return x_406;
}
}
}
else
{
lean_dec_ref(x_388);
lean_dec_ref(x_387);
lean_dec(x_386);
lean_dec(x_373);
x_11 = lean_box(0);
goto block_14;
}
}
else
{
lean_dec(x_388);
lean_dec_ref(x_387);
lean_dec(x_386);
lean_dec(x_373);
x_11 = lean_box(0);
goto block_14;
}
}
else
{
lean_object* x_407; lean_object* x_408; lean_object* x_409; 
lean_dec(x_373);
x_407 = lean_ctor_get(x_384, 0);
lean_inc(x_407);
if (lean_is_exclusive(x_384)) {
 lean_ctor_release(x_384, 0);
 x_408 = x_384;
} else {
 lean_dec_ref(x_384);
 x_408 = lean_box(0);
}
if (lean_is_scalar(x_408)) {
 x_409 = lean_alloc_ctor(1, 1, 0);
} else {
 x_409 = x_408;
}
lean_ctor_set(x_409, 0, x_407);
return x_409;
}
}
}
}
else
{
lean_object* x_410; lean_object* x_411; uint8_t x_412; 
lean_dec_ref(x_374);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_410 = lean_array_get_size(x_372);
x_411 = lean_unsigned_to_nat(3u);
x_412 = lean_nat_dec_eq(x_410, x_411);
if (x_412 == 0)
{
lean_dec(x_373);
lean_dec(x_372);
x_7 = lean_box(0);
goto block_10;
}
else
{
lean_object* x_413; lean_object* x_414; lean_object* x_415; lean_object* x_416; lean_object* x_417; lean_object* x_418; lean_object* x_419; lean_object* x_420; lean_object* x_421; 
x_413 = lean_unsigned_to_nat(1u);
x_414 = lean_array_fget(x_372, x_413);
x_415 = lean_unsigned_to_nat(2u);
x_416 = lean_array_fget(x_372, x_415);
lean_dec(x_372);
x_417 = lp_mathlib_CancelDenoms_findCompLemma___closed__17;
if (lean_is_scalar(x_373)) {
 x_418 = lean_alloc_ctor(0, 2, 0);
} else {
 x_418 = x_373;
}
lean_ctor_set(x_418, 0, x_416);
lean_ctor_set(x_418, 1, x_417);
x_419 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_419, 0, x_414);
lean_ctor_set(x_419, 1, x_418);
x_420 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_420, 0, x_419);
x_421 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_421, 0, x_420);
return x_421;
}
}
}
default: 
{
lean_dec_ref(x_293);
lean_dec_ref(x_292);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_7 = lean_box(0);
goto block_10;
}
}
}
else
{
lean_dec(x_293);
lean_dec_ref(x_292);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_7 = lean_box(0);
goto block_10;
}
}
}
else
{
uint8_t x_422; 
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_422 = !lean_is_exclusive(x_15);
if (x_422 == 0)
{
return x_15;
}
else
{
lean_object* x_423; lean_object* x_424; 
x_423 = lean_ctor_get(x_15, 0);
lean_inc(x_423);
lean_dec(x_15);
x_424 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_424, 0, x_423);
return x_424;
}
}
block_10:
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_box(0);
x_9 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_9, 0, x_8);
return x_9;
}
block_14:
{
lean_object* x_12; lean_object* x_13; 
x_12 = lean_box(0);
x_13 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_13, 0, x_12);
return x_13;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_findCompLemma___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CancelDenoms_findCompLemma(x_1, x_2, x_3, x_4, x_5);
return x_7;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(5u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("_inhabitedExprDummy", 19, 19);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__1;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__2;
x_3 = l_Lean_Expr_const___override(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_mkProdPrf___closed__29;
x_2 = lp_mathlib_CancelDenoms_mkProdPrf___closed__62;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("LinearOrder", 11, 11);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__5;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__7() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("IsStrictOrderedRing", 19, 19);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__8() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__7;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__9() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("SemilatticeInf", 14, 14);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__10() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toPartialOrder", 14, 14);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__11() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__10;
x_2 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__9;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__12() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Lattice", 7, 7);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__13() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toSemilatticeInf", 16, 16);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__14() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__13;
x_2 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__12;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__15() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("DistribLattice", 14, 14);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__16() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toLattice", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__17() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__16;
x_2 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__15;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__18() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("instDistribLatticeOfLinearOrder", 31, 31);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__19() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__18;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__20() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_findCompLemma___closed__11;
x_2 = lp_mathlib_CancelDenoms_findCompLemma___closed__0;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__21() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Preorder", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__22() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toLT", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__23() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__22;
x_2 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__21;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__24() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("PartialOrder", 12, 12);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__25() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("toPreorder", 10, 10);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__26() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__25;
x_2 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__24;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__27() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("cannot kill factors", 19, 19);
return x_1;
}
}
static lean_object* _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__28() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__27;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_13; 
lean_inc(x_5);
lean_inc_ref(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
x_13 = lp_mathlib_CancelDenoms_findCompLemma(x_1, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_13) == 0)
{
lean_object* x_14; 
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
lean_dec_ref(x_13);
if (lean_obj_tag(x_14) == 1)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
lean_dec_ref(x_14);
x_16 = lean_ctor_get(x_15, 1);
lean_inc(x_16);
x_17 = lean_ctor_get(x_16, 1);
lean_inc(x_17);
x_18 = lean_ctor_get(x_15, 0);
lean_inc(x_18);
lean_dec(x_15);
x_19 = lean_ctor_get(x_16, 0);
lean_inc(x_19);
lean_dec(x_16);
x_20 = lean_ctor_get(x_17, 0);
lean_inc(x_20);
x_21 = lean_ctor_get(x_17, 1);
lean_inc(x_21);
lean_dec(x_17);
lean_inc(x_5);
lean_inc_ref(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
lean_inc(x_18);
x_22 = lp_mathlib_CancelDenoms_derive(x_18, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_22) == 0)
{
lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; 
x_23 = lean_ctor_get(x_22, 0);
lean_inc(x_23);
lean_dec_ref(x_22);
x_24 = lean_ctor_get(x_23, 0);
lean_inc(x_24);
x_25 = lean_ctor_get(x_23, 1);
lean_inc(x_25);
lean_dec(x_23);
lean_inc(x_5);
lean_inc_ref(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
x_26 = lp_mathlib_Qq_inferTypeQ_x27(x_18, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_26) == 0)
{
lean_object* x_27; lean_object* x_28; lean_object* x_29; uint8_t x_30; 
x_27 = lean_ctor_get(x_26, 0);
lean_inc(x_27);
lean_dec_ref(x_26);
x_28 = lean_ctor_get(x_27, 1);
lean_inc(x_28);
x_29 = lean_ctor_get(x_27, 0);
lean_inc(x_29);
lean_dec(x_27);
x_30 = !lean_is_exclusive(x_28);
if (x_30 == 0)
{
lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; 
x_31 = lean_ctor_get(x_28, 0);
x_32 = lean_ctor_get(x_28, 1);
lean_dec(x_32);
x_33 = lp_mathlib_CancelDenoms_mkProdPrf___closed__1;
x_34 = lean_box(0);
lean_inc(x_29);
lean_ctor_set_tag(x_28, 1);
lean_ctor_set(x_28, 1, x_34);
lean_ctor_set(x_28, 0, x_29);
lean_inc_ref(x_28);
x_35 = l_Lean_Expr_const___override(x_33, x_28);
lean_inc(x_31);
x_36 = l_Lean_Expr_app___override(x_35, x_31);
lean_inc(x_5);
lean_inc_ref(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
x_37 = lp_Qq_Qq_synthInstanceQ___redArg(x_36, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_37) == 0)
{
lean_object* x_38; lean_object* x_39; 
x_38 = lean_ctor_get(x_37, 0);
lean_inc(x_38);
lean_dec_ref(x_37);
lean_inc(x_5);
lean_inc_ref(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
x_39 = lp_mathlib_CancelDenoms_derive(x_19, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_39) == 0)
{
lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_92; lean_object* x_93; 
x_40 = lean_ctor_get(x_39, 0);
lean_inc(x_40);
lean_dec_ref(x_39);
x_41 = lean_ctor_get(x_40, 0);
lean_inc(x_41);
x_42 = lean_ctor_get(x_40, 1);
lean_inc(x_42);
lean_dec(x_40);
lean_inc(x_24);
x_92 = l_Lean_mkRawNatLit(x_24);
lean_inc(x_38);
lean_inc(x_31);
lean_inc(x_29);
x_93 = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(x_29, x_31, x_38, x_92, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_93) == 0)
{
lean_object* x_94; lean_object* x_95; lean_object* x_96; 
x_94 = lean_ctor_get(x_93, 0);
lean_inc(x_94);
lean_dec_ref(x_93);
lean_inc(x_41);
x_95 = l_Lean_mkRawNatLit(x_41);
lean_inc(x_38);
lean_inc(x_31);
lean_inc(x_29);
x_96 = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(x_29, x_31, x_38, x_95, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_96) == 0)
{
lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; 
x_97 = lean_ctor_get(x_96, 0);
lean_inc(x_97);
lean_dec_ref(x_96);
x_98 = lean_nat_gcd(x_24, x_41);
lean_dec(x_41);
lean_dec(x_24);
x_99 = l_Lean_mkRawNatLit(x_98);
lean_inc(x_31);
lean_inc(x_29);
x_100 = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(x_29, x_31, x_38, x_99, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_100) == 0)
{
lean_object* x_101; uint8_t x_102; 
x_101 = lean_ctor_get(x_100, 0);
lean_inc(x_101);
lean_dec_ref(x_100);
x_102 = lean_unbox(x_21);
lean_dec(x_21);
if (x_102 == 0)
{
lean_object* x_103; lean_object* x_104; uint8_t x_105; 
x_103 = lean_ctor_get(x_94, 0);
lean_inc(x_103);
lean_dec(x_94);
x_104 = lean_ctor_get(x_97, 0);
lean_inc(x_104);
lean_dec(x_97);
x_105 = !lean_is_exclusive(x_101);
if (x_105 == 0)
{
lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; lean_object* x_110; lean_object* x_111; 
x_106 = lean_ctor_get(x_101, 0);
x_107 = lean_ctor_get(x_101, 1);
lean_dec(x_107);
x_108 = lp_mathlib_CancelDenoms_derive___closed__4;
lean_inc_ref(x_28);
x_109 = l_Lean_Expr_const___override(x_108, x_28);
lean_inc(x_31);
x_110 = l_Lean_Expr_app___override(x_109, x_31);
lean_inc(x_5);
lean_inc_ref(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
x_111 = lp_Qq_Qq_synthInstanceQ___redArg(x_110, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_111) == 0)
{
lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; lean_object* x_116; lean_object* x_117; lean_object* x_118; lean_object* x_119; lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_127; lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; lean_object* x_132; lean_object* x_133; lean_object* x_134; lean_object* x_135; lean_object* x_136; lean_object* x_137; lean_object* x_138; lean_object* x_139; lean_object* x_140; lean_object* x_141; lean_object* x_142; lean_object* x_143; lean_object* x_144; lean_object* x_145; lean_object* x_146; lean_object* x_147; lean_object* x_148; lean_object* x_149; lean_object* x_150; lean_object* x_151; lean_object* x_152; lean_object* x_153; lean_object* x_154; lean_object* x_155; lean_object* x_156; lean_object* x_157; lean_object* x_158; lean_object* x_159; lean_object* x_160; 
x_112 = lean_ctor_get(x_111, 0);
lean_inc(x_112);
lean_dec_ref(x_111);
x_113 = l_Lean_Level_succ___override(x_29);
x_114 = lp_mathlib_CancelDenoms_mkProdPrf___closed__17;
lean_ctor_set_tag(x_101, 1);
lean_ctor_set(x_101, 1, x_34);
lean_ctor_set(x_101, 0, x_113);
x_115 = l_Lean_Expr_const___override(x_114, x_101);
lean_inc(x_31);
x_116 = l_Lean_Expr_app___override(x_115, x_31);
lean_inc_ref(x_116);
x_117 = l_Lean_Expr_app___override(x_116, x_103);
x_118 = lp_mathlib_CancelDenoms_mkProdPrf___closed__20;
lean_inc_ref(x_28);
x_119 = l_Lean_Expr_const___override(x_118, x_28);
lean_inc(x_31);
x_120 = l_Lean_Expr_app___override(x_119, x_31);
x_121 = lp_mathlib_CancelDenoms_mkProdPrf___closed__22;
x_122 = l_Lean_Expr_app___override(x_120, x_121);
x_123 = lp_mathlib_CancelDenoms_mkProdPrf___closed__25;
lean_inc_ref(x_28);
x_124 = l_Lean_Expr_const___override(x_123, x_28);
lean_inc(x_31);
x_125 = l_Lean_Expr_app___override(x_124, x_31);
x_126 = lp_mathlib_CancelDenoms_mkProdPrf___closed__28;
lean_inc_ref(x_28);
x_127 = l_Lean_Expr_const___override(x_126, x_28);
lean_inc(x_31);
x_128 = l_Lean_Expr_app___override(x_127, x_31);
x_129 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__4;
lean_inc_ref(x_28);
x_130 = l_Lean_Expr_const___override(x_129, x_28);
lean_inc(x_31);
x_131 = l_Lean_Expr_app___override(x_130, x_31);
x_132 = lp_mathlib_CancelDenoms_mkProdPrf___closed__67;
lean_inc_ref(x_28);
x_133 = l_Lean_Expr_const___override(x_132, x_28);
lean_inc(x_31);
x_134 = l_Lean_Expr_app___override(x_133, x_31);
x_135 = lp_mathlib_CancelDenoms_mkProdPrf___closed__70;
lean_inc_ref(x_28);
x_136 = l_Lean_Expr_const___override(x_135, x_28);
lean_inc(x_31);
x_137 = l_Lean_Expr_app___override(x_136, x_31);
x_138 = lp_mathlib_CancelDenoms_mkProdPrf___closed__73;
lean_inc_ref(x_28);
x_139 = l_Lean_Expr_const___override(x_138, x_28);
lean_inc(x_31);
x_140 = l_Lean_Expr_app___override(x_139, x_31);
x_141 = lp_mathlib_CancelDenoms_mkProdPrf___closed__76;
lean_inc_ref(x_28);
x_142 = l_Lean_Expr_const___override(x_141, x_28);
lean_inc(x_31);
x_143 = l_Lean_Expr_app___override(x_142, x_31);
x_144 = lp_mathlib_CancelDenoms_mkProdPrf___closed__78;
x_145 = l_Lean_Expr_const___override(x_144, x_28);
x_146 = l_Lean_Expr_app___override(x_145, x_31);
x_147 = l_Lean_Expr_app___override(x_146, x_112);
x_148 = l_Lean_Expr_app___override(x_143, x_147);
x_149 = l_Lean_Expr_app___override(x_140, x_148);
x_150 = l_Lean_Expr_app___override(x_137, x_149);
x_151 = l_Lean_Expr_app___override(x_134, x_150);
x_152 = l_Lean_Expr_app___override(x_131, x_151);
x_153 = l_Lean_Expr_app___override(x_128, x_152);
x_154 = l_Lean_Expr_app___override(x_125, x_153);
x_155 = l_Lean_Expr_app___override(x_122, x_154);
lean_inc_ref(x_155);
x_156 = l_Lean_Expr_app___override(x_117, x_155);
lean_inc_ref(x_116);
x_157 = l_Lean_Expr_app___override(x_116, x_104);
lean_inc_ref(x_155);
x_158 = l_Lean_Expr_app___override(x_157, x_155);
x_159 = l_Lean_Expr_app___override(x_116, x_106);
x_160 = l_Lean_Expr_app___override(x_159, x_155);
x_43 = x_156;
x_44 = x_158;
x_45 = x_160;
x_46 = x_2;
x_47 = x_3;
x_48 = x_4;
x_49 = x_5;
x_50 = lean_box(0);
goto block_91;
}
else
{
uint8_t x_161; 
lean_free_object(x_101);
lean_dec(x_106);
lean_dec(x_104);
lean_dec(x_103);
lean_dec(x_42);
lean_dec_ref(x_28);
lean_dec(x_31);
lean_dec(x_29);
lean_dec(x_25);
lean_dec(x_20);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_161 = !lean_is_exclusive(x_111);
if (x_161 == 0)
{
return x_111;
}
else
{
lean_object* x_162; lean_object* x_163; 
x_162 = lean_ctor_get(x_111, 0);
lean_inc(x_162);
lean_dec(x_111);
x_163 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_163, 0, x_162);
return x_163;
}
}
}
else
{
lean_object* x_164; lean_object* x_165; lean_object* x_166; lean_object* x_167; lean_object* x_168; 
x_164 = lean_ctor_get(x_101, 0);
lean_inc(x_164);
lean_dec(x_101);
x_165 = lp_mathlib_CancelDenoms_derive___closed__4;
lean_inc_ref(x_28);
x_166 = l_Lean_Expr_const___override(x_165, x_28);
lean_inc(x_31);
x_167 = l_Lean_Expr_app___override(x_166, x_31);
lean_inc(x_5);
lean_inc_ref(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
x_168 = lp_Qq_Qq_synthInstanceQ___redArg(x_167, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_168) == 0)
{
lean_object* x_169; lean_object* x_170; lean_object* x_171; lean_object* x_172; lean_object* x_173; lean_object* x_174; lean_object* x_175; lean_object* x_176; lean_object* x_177; lean_object* x_178; lean_object* x_179; lean_object* x_180; lean_object* x_181; lean_object* x_182; lean_object* x_183; lean_object* x_184; lean_object* x_185; lean_object* x_186; lean_object* x_187; lean_object* x_188; lean_object* x_189; lean_object* x_190; lean_object* x_191; lean_object* x_192; lean_object* x_193; lean_object* x_194; lean_object* x_195; lean_object* x_196; lean_object* x_197; lean_object* x_198; lean_object* x_199; lean_object* x_200; lean_object* x_201; lean_object* x_202; lean_object* x_203; lean_object* x_204; lean_object* x_205; lean_object* x_206; lean_object* x_207; lean_object* x_208; lean_object* x_209; lean_object* x_210; lean_object* x_211; lean_object* x_212; lean_object* x_213; lean_object* x_214; lean_object* x_215; lean_object* x_216; lean_object* x_217; lean_object* x_218; 
x_169 = lean_ctor_get(x_168, 0);
lean_inc(x_169);
lean_dec_ref(x_168);
x_170 = l_Lean_Level_succ___override(x_29);
x_171 = lp_mathlib_CancelDenoms_mkProdPrf___closed__17;
x_172 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_172, 0, x_170);
lean_ctor_set(x_172, 1, x_34);
x_173 = l_Lean_Expr_const___override(x_171, x_172);
lean_inc(x_31);
x_174 = l_Lean_Expr_app___override(x_173, x_31);
lean_inc_ref(x_174);
x_175 = l_Lean_Expr_app___override(x_174, x_103);
x_176 = lp_mathlib_CancelDenoms_mkProdPrf___closed__20;
lean_inc_ref(x_28);
x_177 = l_Lean_Expr_const___override(x_176, x_28);
lean_inc(x_31);
x_178 = l_Lean_Expr_app___override(x_177, x_31);
x_179 = lp_mathlib_CancelDenoms_mkProdPrf___closed__22;
x_180 = l_Lean_Expr_app___override(x_178, x_179);
x_181 = lp_mathlib_CancelDenoms_mkProdPrf___closed__25;
lean_inc_ref(x_28);
x_182 = l_Lean_Expr_const___override(x_181, x_28);
lean_inc(x_31);
x_183 = l_Lean_Expr_app___override(x_182, x_31);
x_184 = lp_mathlib_CancelDenoms_mkProdPrf___closed__28;
lean_inc_ref(x_28);
x_185 = l_Lean_Expr_const___override(x_184, x_28);
lean_inc(x_31);
x_186 = l_Lean_Expr_app___override(x_185, x_31);
x_187 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__4;
lean_inc_ref(x_28);
x_188 = l_Lean_Expr_const___override(x_187, x_28);
lean_inc(x_31);
x_189 = l_Lean_Expr_app___override(x_188, x_31);
x_190 = lp_mathlib_CancelDenoms_mkProdPrf___closed__67;
lean_inc_ref(x_28);
x_191 = l_Lean_Expr_const___override(x_190, x_28);
lean_inc(x_31);
x_192 = l_Lean_Expr_app___override(x_191, x_31);
x_193 = lp_mathlib_CancelDenoms_mkProdPrf___closed__70;
lean_inc_ref(x_28);
x_194 = l_Lean_Expr_const___override(x_193, x_28);
lean_inc(x_31);
x_195 = l_Lean_Expr_app___override(x_194, x_31);
x_196 = lp_mathlib_CancelDenoms_mkProdPrf___closed__73;
lean_inc_ref(x_28);
x_197 = l_Lean_Expr_const___override(x_196, x_28);
lean_inc(x_31);
x_198 = l_Lean_Expr_app___override(x_197, x_31);
x_199 = lp_mathlib_CancelDenoms_mkProdPrf___closed__76;
lean_inc_ref(x_28);
x_200 = l_Lean_Expr_const___override(x_199, x_28);
lean_inc(x_31);
x_201 = l_Lean_Expr_app___override(x_200, x_31);
x_202 = lp_mathlib_CancelDenoms_mkProdPrf___closed__78;
x_203 = l_Lean_Expr_const___override(x_202, x_28);
x_204 = l_Lean_Expr_app___override(x_203, x_31);
x_205 = l_Lean_Expr_app___override(x_204, x_169);
x_206 = l_Lean_Expr_app___override(x_201, x_205);
x_207 = l_Lean_Expr_app___override(x_198, x_206);
x_208 = l_Lean_Expr_app___override(x_195, x_207);
x_209 = l_Lean_Expr_app___override(x_192, x_208);
x_210 = l_Lean_Expr_app___override(x_189, x_209);
x_211 = l_Lean_Expr_app___override(x_186, x_210);
x_212 = l_Lean_Expr_app___override(x_183, x_211);
x_213 = l_Lean_Expr_app___override(x_180, x_212);
lean_inc_ref(x_213);
x_214 = l_Lean_Expr_app___override(x_175, x_213);
lean_inc_ref(x_174);
x_215 = l_Lean_Expr_app___override(x_174, x_104);
lean_inc_ref(x_213);
x_216 = l_Lean_Expr_app___override(x_215, x_213);
x_217 = l_Lean_Expr_app___override(x_174, x_164);
x_218 = l_Lean_Expr_app___override(x_217, x_213);
x_43 = x_214;
x_44 = x_216;
x_45 = x_218;
x_46 = x_2;
x_47 = x_3;
x_48 = x_4;
x_49 = x_5;
x_50 = lean_box(0);
goto block_91;
}
else
{
lean_object* x_219; lean_object* x_220; lean_object* x_221; 
lean_dec(x_164);
lean_dec(x_104);
lean_dec(x_103);
lean_dec(x_42);
lean_dec_ref(x_28);
lean_dec(x_31);
lean_dec(x_29);
lean_dec(x_25);
lean_dec(x_20);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_219 = lean_ctor_get(x_168, 0);
lean_inc(x_219);
if (lean_is_exclusive(x_168)) {
 lean_ctor_release(x_168, 0);
 x_220 = x_168;
} else {
 lean_dec_ref(x_168);
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
}
else
{
lean_object* x_222; lean_object* x_223; lean_object* x_224; lean_object* x_225; lean_object* x_226; lean_object* x_227; lean_object* x_228; 
lean_dec(x_29);
x_222 = lean_ctor_get(x_94, 0);
lean_inc(x_222);
lean_dec(x_94);
x_223 = lean_ctor_get(x_97, 0);
lean_inc(x_223);
lean_dec(x_97);
x_224 = lean_ctor_get(x_101, 0);
lean_inc(x_224);
lean_dec(x_101);
x_225 = lp_mathlib_CancelDenoms_derive___closed__4;
lean_inc_ref(x_28);
x_226 = l_Lean_Expr_const___override(x_225, x_28);
lean_inc(x_31);
x_227 = l_Lean_Expr_app___override(x_226, x_31);
lean_inc(x_5);
lean_inc_ref(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
x_228 = lp_Qq_Qq_synthInstanceQ___redArg(x_227, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_228) == 0)
{
lean_object* x_229; lean_object* x_230; lean_object* x_231; lean_object* x_232; lean_object* x_233; 
x_229 = lean_ctor_get(x_228, 0);
lean_inc(x_229);
lean_dec_ref(x_228);
x_230 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__6;
lean_inc_ref(x_28);
x_231 = l_Lean_Expr_const___override(x_230, x_28);
lean_inc(x_31);
x_232 = l_Lean_Expr_app___override(x_231, x_31);
lean_inc(x_5);
lean_inc_ref(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
x_233 = lp_Qq_Qq_synthInstanceQ___redArg(x_232, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_233) == 0)
{
lean_object* x_234; lean_object* x_235; lean_object* x_236; lean_object* x_237; lean_object* x_238; lean_object* x_239; lean_object* x_240; lean_object* x_241; lean_object* x_242; lean_object* x_243; lean_object* x_244; lean_object* x_245; lean_object* x_246; lean_object* x_247; lean_object* x_248; lean_object* x_249; lean_object* x_250; lean_object* x_251; lean_object* x_252; lean_object* x_253; lean_object* x_254; lean_object* x_255; lean_object* x_256; lean_object* x_257; lean_object* x_258; lean_object* x_259; lean_object* x_260; lean_object* x_261; lean_object* x_262; lean_object* x_263; lean_object* x_264; lean_object* x_265; lean_object* x_266; lean_object* x_267; lean_object* x_268; 
x_234 = lean_ctor_get(x_233, 0);
lean_inc(x_234);
lean_dec_ref(x_233);
x_235 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__8;
lean_inc_ref(x_28);
x_236 = l_Lean_Expr_const___override(x_235, x_28);
lean_inc(x_31);
x_237 = l_Lean_Expr_app___override(x_236, x_31);
x_238 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__16;
lean_inc_ref(x_28);
x_239 = l_Lean_Expr_const___override(x_238, x_28);
lean_inc(x_31);
x_240 = l_Lean_Expr_app___override(x_239, x_31);
x_241 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__18;
lean_inc_ref(x_28);
x_242 = l_Lean_Expr_const___override(x_241, x_28);
lean_inc(x_31);
x_243 = l_Lean_Expr_app___override(x_242, x_31);
x_244 = lp_mathlib_CancelDenoms_mkProdPrf___closed__34;
lean_inc_ref(x_28);
x_245 = l_Lean_Expr_const___override(x_244, x_28);
lean_inc(x_31);
x_246 = l_Lean_Expr_app___override(x_245, x_31);
lean_inc(x_229);
x_247 = l_Lean_Expr_app___override(x_246, x_229);
x_248 = l_Lean_Expr_app___override(x_243, x_247);
x_249 = l_Lean_Expr_app___override(x_240, x_248);
x_250 = l_Lean_Expr_app___override(x_237, x_249);
x_251 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__11;
lean_inc_ref(x_28);
x_252 = l_Lean_Expr_const___override(x_251, x_28);
lean_inc(x_31);
x_253 = l_Lean_Expr_app___override(x_252, x_31);
x_254 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__14;
lean_inc_ref(x_28);
x_255 = l_Lean_Expr_const___override(x_254, x_28);
lean_inc(x_31);
x_256 = l_Lean_Expr_app___override(x_255, x_31);
x_257 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__17;
lean_inc_ref(x_28);
x_258 = l_Lean_Expr_const___override(x_257, x_28);
lean_inc(x_31);
x_259 = l_Lean_Expr_app___override(x_258, x_31);
x_260 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__19;
lean_inc_ref(x_28);
x_261 = l_Lean_Expr_const___override(x_260, x_28);
lean_inc(x_31);
x_262 = l_Lean_Expr_app___override(x_261, x_31);
x_263 = l_Lean_Expr_app___override(x_262, x_234);
x_264 = l_Lean_Expr_app___override(x_259, x_263);
x_265 = l_Lean_Expr_app___override(x_256, x_264);
x_266 = l_Lean_Expr_app___override(x_253, x_265);
lean_inc_ref(x_266);
x_267 = l_Lean_Expr_app___override(x_250, x_266);
lean_inc(x_5);
lean_inc_ref(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
x_268 = lp_Qq_Qq_synthInstanceQ___redArg(x_267, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_268) == 0)
{
lean_object* x_269; lean_object* x_270; lean_object* x_271; lean_object* x_272; lean_object* x_273; lean_object* x_274; lean_object* x_275; lean_object* x_276; lean_object* x_277; lean_object* x_278; lean_object* x_279; lean_object* x_280; lean_object* x_281; lean_object* x_282; lean_object* x_283; lean_object* x_284; lean_object* x_285; lean_object* x_286; lean_object* x_287; lean_object* x_288; lean_object* x_289; lean_object* x_290; lean_object* x_291; lean_object* x_292; lean_object* x_293; lean_object* x_294; lean_object* x_295; lean_object* x_296; lean_object* x_297; lean_object* x_298; lean_object* x_299; lean_object* x_300; lean_object* x_301; lean_object* x_302; lean_object* x_303; lean_object* x_304; lean_object* x_305; lean_object* x_306; lean_object* x_307; lean_object* x_308; lean_object* x_309; lean_object* x_310; lean_object* x_311; lean_object* x_312; lean_object* x_313; lean_object* x_314; lean_object* x_315; lean_object* x_316; lean_object* x_317; lean_object* x_318; lean_object* x_319; lean_object* x_320; lean_object* x_321; lean_object* x_322; 
lean_dec_ref(x_268);
x_269 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__20;
lean_inc_ref(x_28);
x_270 = l_Lean_Expr_const___override(x_269, x_28);
lean_inc(x_31);
x_271 = l_Lean_Expr_app___override(x_270, x_31);
x_272 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__23;
lean_inc_ref(x_28);
x_273 = l_Lean_Expr_const___override(x_272, x_28);
lean_inc(x_31);
x_274 = l_Lean_Expr_app___override(x_273, x_31);
x_275 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__26;
lean_inc_ref(x_28);
x_276 = l_Lean_Expr_const___override(x_275, x_28);
lean_inc(x_31);
x_277 = l_Lean_Expr_app___override(x_276, x_31);
x_278 = l_Lean_Expr_app___override(x_277, x_266);
x_279 = l_Lean_Expr_app___override(x_274, x_278);
x_280 = l_Lean_Expr_app___override(x_271, x_279);
x_281 = lp_mathlib_CancelDenoms_mkProdPrf___closed__20;
lean_inc_ref(x_28);
x_282 = l_Lean_Expr_const___override(x_281, x_28);
lean_inc(x_31);
x_283 = l_Lean_Expr_app___override(x_282, x_31);
x_284 = lp_mathlib_CancelDenoms_mkProdPrf___closed__22;
x_285 = l_Lean_Expr_app___override(x_283, x_284);
x_286 = lp_mathlib_CancelDenoms_mkProdPrf___closed__25;
lean_inc_ref(x_28);
x_287 = l_Lean_Expr_const___override(x_286, x_28);
lean_inc(x_31);
x_288 = l_Lean_Expr_app___override(x_287, x_31);
x_289 = lp_mathlib_CancelDenoms_mkProdPrf___closed__28;
lean_inc_ref(x_28);
x_290 = l_Lean_Expr_const___override(x_289, x_28);
lean_inc(x_31);
x_291 = l_Lean_Expr_app___override(x_290, x_31);
x_292 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__4;
lean_inc_ref(x_28);
x_293 = l_Lean_Expr_const___override(x_292, x_28);
lean_inc(x_31);
x_294 = l_Lean_Expr_app___override(x_293, x_31);
x_295 = lp_mathlib_CancelDenoms_mkProdPrf___closed__67;
lean_inc_ref(x_28);
x_296 = l_Lean_Expr_const___override(x_295, x_28);
lean_inc(x_31);
x_297 = l_Lean_Expr_app___override(x_296, x_31);
x_298 = lp_mathlib_CancelDenoms_mkProdPrf___closed__70;
lean_inc_ref(x_28);
x_299 = l_Lean_Expr_const___override(x_298, x_28);
lean_inc(x_31);
x_300 = l_Lean_Expr_app___override(x_299, x_31);
x_301 = lp_mathlib_CancelDenoms_mkProdPrf___closed__73;
lean_inc_ref(x_28);
x_302 = l_Lean_Expr_const___override(x_301, x_28);
lean_inc(x_31);
x_303 = l_Lean_Expr_app___override(x_302, x_31);
x_304 = lp_mathlib_CancelDenoms_mkProdPrf___closed__76;
lean_inc_ref(x_28);
x_305 = l_Lean_Expr_const___override(x_304, x_28);
lean_inc(x_31);
x_306 = l_Lean_Expr_app___override(x_305, x_31);
x_307 = lp_mathlib_CancelDenoms_mkProdPrf___closed__78;
x_308 = l_Lean_Expr_const___override(x_307, x_28);
x_309 = l_Lean_Expr_app___override(x_308, x_31);
x_310 = l_Lean_Expr_app___override(x_309, x_229);
x_311 = l_Lean_Expr_app___override(x_306, x_310);
x_312 = l_Lean_Expr_app___override(x_303, x_311);
x_313 = l_Lean_Expr_app___override(x_300, x_312);
x_314 = l_Lean_Expr_app___override(x_297, x_313);
x_315 = l_Lean_Expr_app___override(x_294, x_314);
x_316 = l_Lean_Expr_app___override(x_291, x_315);
x_317 = l_Lean_Expr_app___override(x_288, x_316);
x_318 = l_Lean_Expr_app___override(x_285, x_317);
x_319 = l_Lean_Expr_app___override(x_280, x_318);
lean_inc_ref(x_319);
x_320 = l_Lean_Expr_app___override(x_319, x_222);
lean_inc_ref(x_319);
x_321 = l_Lean_Expr_app___override(x_319, x_223);
x_322 = l_Lean_Expr_app___override(x_319, x_224);
x_43 = x_320;
x_44 = x_321;
x_45 = x_322;
x_46 = x_2;
x_47 = x_3;
x_48 = x_4;
x_49 = x_5;
x_50 = lean_box(0);
goto block_91;
}
else
{
uint8_t x_323; 
lean_dec_ref(x_266);
lean_dec(x_229);
lean_dec(x_224);
lean_dec(x_223);
lean_dec(x_222);
lean_dec(x_42);
lean_dec_ref(x_28);
lean_dec(x_31);
lean_dec(x_25);
lean_dec(x_20);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_323 = !lean_is_exclusive(x_268);
if (x_323 == 0)
{
return x_268;
}
else
{
lean_object* x_324; lean_object* x_325; 
x_324 = lean_ctor_get(x_268, 0);
lean_inc(x_324);
lean_dec(x_268);
x_325 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_325, 0, x_324);
return x_325;
}
}
}
else
{
uint8_t x_326; 
lean_dec(x_229);
lean_dec(x_224);
lean_dec(x_223);
lean_dec(x_222);
lean_dec(x_42);
lean_dec_ref(x_28);
lean_dec(x_31);
lean_dec(x_25);
lean_dec(x_20);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_326 = !lean_is_exclusive(x_233);
if (x_326 == 0)
{
return x_233;
}
else
{
lean_object* x_327; lean_object* x_328; 
x_327 = lean_ctor_get(x_233, 0);
lean_inc(x_327);
lean_dec(x_233);
x_328 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_328, 0, x_327);
return x_328;
}
}
}
else
{
uint8_t x_329; 
lean_dec(x_224);
lean_dec(x_223);
lean_dec(x_222);
lean_dec(x_42);
lean_dec_ref(x_28);
lean_dec(x_31);
lean_dec(x_25);
lean_dec(x_20);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_329 = !lean_is_exclusive(x_228);
if (x_329 == 0)
{
return x_228;
}
else
{
lean_object* x_330; lean_object* x_331; 
x_330 = lean_ctor_get(x_228, 0);
lean_inc(x_330);
lean_dec(x_228);
x_331 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_331, 0, x_330);
return x_331;
}
}
}
}
else
{
uint8_t x_332; 
lean_dec(x_97);
lean_dec(x_94);
lean_dec(x_42);
lean_dec_ref(x_28);
lean_dec(x_31);
lean_dec(x_29);
lean_dec(x_25);
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_332 = !lean_is_exclusive(x_100);
if (x_332 == 0)
{
return x_100;
}
else
{
lean_object* x_333; lean_object* x_334; 
x_333 = lean_ctor_get(x_100, 0);
lean_inc(x_333);
lean_dec(x_100);
x_334 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_334, 0, x_333);
return x_334;
}
}
}
else
{
uint8_t x_335; 
lean_dec(x_94);
lean_dec(x_42);
lean_dec(x_41);
lean_dec(x_38);
lean_dec_ref(x_28);
lean_dec(x_31);
lean_dec(x_29);
lean_dec(x_25);
lean_dec(x_24);
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_335 = !lean_is_exclusive(x_96);
if (x_335 == 0)
{
return x_96;
}
else
{
lean_object* x_336; lean_object* x_337; 
x_336 = lean_ctor_get(x_96, 0);
lean_inc(x_336);
lean_dec(x_96);
x_337 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_337, 0, x_336);
return x_337;
}
}
}
else
{
uint8_t x_338; 
lean_dec(x_42);
lean_dec(x_41);
lean_dec(x_38);
lean_dec_ref(x_28);
lean_dec(x_31);
lean_dec(x_29);
lean_dec(x_25);
lean_dec(x_24);
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_338 = !lean_is_exclusive(x_93);
if (x_338 == 0)
{
return x_93;
}
else
{
lean_object* x_339; lean_object* x_340; 
x_339 = lean_ctor_get(x_93, 0);
lean_inc(x_339);
lean_dec(x_93);
x_340 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_340, 0, x_339);
return x_340;
}
}
block_91:
{
lean_object* x_51; 
lean_inc(x_49);
lean_inc_ref(x_48);
lean_inc(x_47);
lean_inc_ref(x_46);
x_51 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum(x_43, x_46, x_47, x_48, x_49);
if (lean_obj_tag(x_51) == 0)
{
lean_object* x_52; lean_object* x_53; 
x_52 = lean_ctor_get(x_51, 0);
lean_inc(x_52);
lean_dec_ref(x_51);
lean_inc(x_49);
lean_inc_ref(x_48);
lean_inc(x_47);
lean_inc_ref(x_46);
x_53 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum(x_44, x_46, x_47, x_48, x_49);
if (lean_obj_tag(x_53) == 0)
{
lean_object* x_54; lean_object* x_55; 
x_54 = lean_ctor_get(x_53, 0);
lean_inc(x_54);
lean_dec_ref(x_53);
lean_inc(x_49);
lean_inc_ref(x_48);
lean_inc(x_47);
lean_inc_ref(x_46);
x_55 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum(x_45, x_46, x_47, x_48, x_49);
if (lean_obj_tag(x_55) == 0)
{
lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; 
x_56 = lean_ctor_get(x_55, 0);
lean_inc(x_56);
lean_dec_ref(x_55);
x_57 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__0;
x_58 = lean_array_push(x_57, x_25);
x_59 = lean_array_push(x_58, x_42);
x_60 = lean_array_push(x_59, x_52);
x_61 = lean_array_push(x_60, x_54);
x_62 = lean_array_push(x_61, x_56);
lean_inc(x_49);
lean_inc_ref(x_48);
lean_inc(x_47);
lean_inc_ref(x_46);
x_63 = l_Lean_Meta_mkAppM(x_20, x_62, x_46, x_47, x_48, x_49);
if (lean_obj_tag(x_63) == 0)
{
lean_object* x_64; lean_object* x_65; 
x_64 = lean_ctor_get(x_63, 0);
lean_inc(x_64);
lean_dec_ref(x_63);
lean_inc(x_49);
lean_inc_ref(x_48);
lean_inc(x_47);
lean_inc_ref(x_46);
lean_inc(x_64);
x_65 = lean_infer_type(x_64, x_46, x_47, x_48, x_49);
if (lean_obj_tag(x_65) == 0)
{
lean_object* x_66; lean_object* x_67; 
x_66 = lean_ctor_get(x_65, 0);
lean_inc(x_66);
lean_dec_ref(x_65);
x_67 = lp_mathlib_CancelDenoms_findCompLemma(x_66, x_46, x_47, x_48, x_49);
if (lean_obj_tag(x_67) == 0)
{
lean_object* x_68; 
x_68 = lean_ctor_get(x_67, 0);
lean_inc(x_68);
lean_dec_ref(x_67);
if (lean_obj_tag(x_68) == 0)
{
lean_object* x_69; 
x_69 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__3;
x_7 = x_64;
x_8 = lean_box(0);
x_9 = x_69;
goto block_12;
}
else
{
lean_object* x_70; lean_object* x_71; lean_object* x_72; 
x_70 = lean_ctor_get(x_68, 0);
lean_inc(x_70);
lean_dec_ref(x_68);
x_71 = lean_ctor_get(x_70, 1);
lean_inc(x_71);
lean_dec(x_70);
x_72 = lean_ctor_get(x_71, 0);
lean_inc(x_72);
lean_dec(x_71);
x_7 = x_64;
x_8 = lean_box(0);
x_9 = x_72;
goto block_12;
}
}
else
{
uint8_t x_73; 
lean_dec(x_64);
x_73 = !lean_is_exclusive(x_67);
if (x_73 == 0)
{
return x_67;
}
else
{
lean_object* x_74; lean_object* x_75; 
x_74 = lean_ctor_get(x_67, 0);
lean_inc(x_74);
lean_dec(x_67);
x_75 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_75, 0, x_74);
return x_75;
}
}
}
else
{
uint8_t x_76; 
lean_dec(x_64);
lean_dec(x_49);
lean_dec_ref(x_48);
lean_dec(x_47);
lean_dec_ref(x_46);
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
lean_dec(x_49);
lean_dec_ref(x_48);
lean_dec(x_47);
lean_dec_ref(x_46);
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
lean_dec(x_54);
lean_dec(x_52);
lean_dec(x_49);
lean_dec_ref(x_48);
lean_dec(x_47);
lean_dec_ref(x_46);
lean_dec(x_42);
lean_dec(x_25);
lean_dec(x_20);
x_82 = !lean_is_exclusive(x_55);
if (x_82 == 0)
{
return x_55;
}
else
{
lean_object* x_83; lean_object* x_84; 
x_83 = lean_ctor_get(x_55, 0);
lean_inc(x_83);
lean_dec(x_55);
x_84 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_84, 0, x_83);
return x_84;
}
}
}
else
{
uint8_t x_85; 
lean_dec(x_52);
lean_dec(x_49);
lean_dec_ref(x_48);
lean_dec(x_47);
lean_dec_ref(x_46);
lean_dec_ref(x_45);
lean_dec(x_42);
lean_dec(x_25);
lean_dec(x_20);
x_85 = !lean_is_exclusive(x_53);
if (x_85 == 0)
{
return x_53;
}
else
{
lean_object* x_86; lean_object* x_87; 
x_86 = lean_ctor_get(x_53, 0);
lean_inc(x_86);
lean_dec(x_53);
x_87 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_87, 0, x_86);
return x_87;
}
}
}
else
{
uint8_t x_88; 
lean_dec(x_49);
lean_dec_ref(x_48);
lean_dec(x_47);
lean_dec_ref(x_46);
lean_dec_ref(x_45);
lean_dec_ref(x_44);
lean_dec(x_42);
lean_dec(x_25);
lean_dec(x_20);
x_88 = !lean_is_exclusive(x_51);
if (x_88 == 0)
{
return x_51;
}
else
{
lean_object* x_89; lean_object* x_90; 
x_89 = lean_ctor_get(x_51, 0);
lean_inc(x_89);
lean_dec(x_51);
x_90 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_90, 0, x_89);
return x_90;
}
}
}
}
else
{
uint8_t x_341; 
lean_dec(x_38);
lean_dec_ref(x_28);
lean_dec(x_31);
lean_dec(x_29);
lean_dec(x_25);
lean_dec(x_24);
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_341 = !lean_is_exclusive(x_39);
if (x_341 == 0)
{
return x_39;
}
else
{
lean_object* x_342; lean_object* x_343; 
x_342 = lean_ctor_get(x_39, 0);
lean_inc(x_342);
lean_dec(x_39);
x_343 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_343, 0, x_342);
return x_343;
}
}
}
else
{
uint8_t x_344; 
lean_dec_ref(x_28);
lean_dec(x_31);
lean_dec(x_29);
lean_dec(x_25);
lean_dec(x_24);
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_19);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_344 = !lean_is_exclusive(x_37);
if (x_344 == 0)
{
return x_37;
}
else
{
lean_object* x_345; lean_object* x_346; 
x_345 = lean_ctor_get(x_37, 0);
lean_inc(x_345);
lean_dec(x_37);
x_346 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_346, 0, x_345);
return x_346;
}
}
}
else
{
lean_object* x_347; lean_object* x_348; lean_object* x_349; lean_object* x_350; lean_object* x_351; lean_object* x_352; lean_object* x_353; 
x_347 = lean_ctor_get(x_28, 0);
lean_inc(x_347);
lean_dec(x_28);
x_348 = lp_mathlib_CancelDenoms_mkProdPrf___closed__1;
x_349 = lean_box(0);
lean_inc(x_29);
x_350 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_350, 0, x_29);
lean_ctor_set(x_350, 1, x_349);
lean_inc_ref(x_350);
x_351 = l_Lean_Expr_const___override(x_348, x_350);
lean_inc(x_347);
x_352 = l_Lean_Expr_app___override(x_351, x_347);
lean_inc(x_5);
lean_inc_ref(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
x_353 = lp_Qq_Qq_synthInstanceQ___redArg(x_352, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_353) == 0)
{
lean_object* x_354; lean_object* x_355; 
x_354 = lean_ctor_get(x_353, 0);
lean_inc(x_354);
lean_dec_ref(x_353);
lean_inc(x_5);
lean_inc_ref(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
x_355 = lp_mathlib_CancelDenoms_derive(x_19, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_355) == 0)
{
lean_object* x_356; lean_object* x_357; lean_object* x_358; lean_object* x_359; lean_object* x_360; lean_object* x_361; lean_object* x_362; lean_object* x_363; lean_object* x_364; lean_object* x_365; lean_object* x_366; lean_object* x_408; lean_object* x_409; 
x_356 = lean_ctor_get(x_355, 0);
lean_inc(x_356);
lean_dec_ref(x_355);
x_357 = lean_ctor_get(x_356, 0);
lean_inc(x_357);
x_358 = lean_ctor_get(x_356, 1);
lean_inc(x_358);
lean_dec(x_356);
lean_inc(x_24);
x_408 = l_Lean_mkRawNatLit(x_24);
lean_inc(x_354);
lean_inc(x_347);
lean_inc(x_29);
x_409 = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(x_29, x_347, x_354, x_408, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_409) == 0)
{
lean_object* x_410; lean_object* x_411; lean_object* x_412; 
x_410 = lean_ctor_get(x_409, 0);
lean_inc(x_410);
lean_dec_ref(x_409);
lean_inc(x_357);
x_411 = l_Lean_mkRawNatLit(x_357);
lean_inc(x_354);
lean_inc(x_347);
lean_inc(x_29);
x_412 = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(x_29, x_347, x_354, x_411, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_412) == 0)
{
lean_object* x_413; lean_object* x_414; lean_object* x_415; lean_object* x_416; 
x_413 = lean_ctor_get(x_412, 0);
lean_inc(x_413);
lean_dec_ref(x_412);
x_414 = lean_nat_gcd(x_24, x_357);
lean_dec(x_357);
lean_dec(x_24);
x_415 = l_Lean_mkRawNatLit(x_414);
lean_inc(x_347);
lean_inc(x_29);
x_416 = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(x_29, x_347, x_354, x_415, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_416) == 0)
{
lean_object* x_417; uint8_t x_418; 
x_417 = lean_ctor_get(x_416, 0);
lean_inc(x_417);
lean_dec_ref(x_416);
x_418 = lean_unbox(x_21);
lean_dec(x_21);
if (x_418 == 0)
{
lean_object* x_419; lean_object* x_420; lean_object* x_421; lean_object* x_422; lean_object* x_423; lean_object* x_424; lean_object* x_425; lean_object* x_426; 
x_419 = lean_ctor_get(x_410, 0);
lean_inc(x_419);
lean_dec(x_410);
x_420 = lean_ctor_get(x_413, 0);
lean_inc(x_420);
lean_dec(x_413);
x_421 = lean_ctor_get(x_417, 0);
lean_inc(x_421);
if (lean_is_exclusive(x_417)) {
 lean_ctor_release(x_417, 0);
 lean_ctor_release(x_417, 1);
 x_422 = x_417;
} else {
 lean_dec_ref(x_417);
 x_422 = lean_box(0);
}
x_423 = lp_mathlib_CancelDenoms_derive___closed__4;
lean_inc_ref(x_350);
x_424 = l_Lean_Expr_const___override(x_423, x_350);
lean_inc(x_347);
x_425 = l_Lean_Expr_app___override(x_424, x_347);
lean_inc(x_5);
lean_inc_ref(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
x_426 = lp_Qq_Qq_synthInstanceQ___redArg(x_425, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_426) == 0)
{
lean_object* x_427; lean_object* x_428; lean_object* x_429; lean_object* x_430; lean_object* x_431; lean_object* x_432; lean_object* x_433; lean_object* x_434; lean_object* x_435; lean_object* x_436; lean_object* x_437; lean_object* x_438; lean_object* x_439; lean_object* x_440; lean_object* x_441; lean_object* x_442; lean_object* x_443; lean_object* x_444; lean_object* x_445; lean_object* x_446; lean_object* x_447; lean_object* x_448; lean_object* x_449; lean_object* x_450; lean_object* x_451; lean_object* x_452; lean_object* x_453; lean_object* x_454; lean_object* x_455; lean_object* x_456; lean_object* x_457; lean_object* x_458; lean_object* x_459; lean_object* x_460; lean_object* x_461; lean_object* x_462; lean_object* x_463; lean_object* x_464; lean_object* x_465; lean_object* x_466; lean_object* x_467; lean_object* x_468; lean_object* x_469; lean_object* x_470; lean_object* x_471; lean_object* x_472; lean_object* x_473; lean_object* x_474; lean_object* x_475; lean_object* x_476; 
x_427 = lean_ctor_get(x_426, 0);
lean_inc(x_427);
lean_dec_ref(x_426);
x_428 = l_Lean_Level_succ___override(x_29);
x_429 = lp_mathlib_CancelDenoms_mkProdPrf___closed__17;
if (lean_is_scalar(x_422)) {
 x_430 = lean_alloc_ctor(1, 2, 0);
} else {
 x_430 = x_422;
 lean_ctor_set_tag(x_430, 1);
}
lean_ctor_set(x_430, 0, x_428);
lean_ctor_set(x_430, 1, x_349);
x_431 = l_Lean_Expr_const___override(x_429, x_430);
lean_inc(x_347);
x_432 = l_Lean_Expr_app___override(x_431, x_347);
lean_inc_ref(x_432);
x_433 = l_Lean_Expr_app___override(x_432, x_419);
x_434 = lp_mathlib_CancelDenoms_mkProdPrf___closed__20;
lean_inc_ref(x_350);
x_435 = l_Lean_Expr_const___override(x_434, x_350);
lean_inc(x_347);
x_436 = l_Lean_Expr_app___override(x_435, x_347);
x_437 = lp_mathlib_CancelDenoms_mkProdPrf___closed__22;
x_438 = l_Lean_Expr_app___override(x_436, x_437);
x_439 = lp_mathlib_CancelDenoms_mkProdPrf___closed__25;
lean_inc_ref(x_350);
x_440 = l_Lean_Expr_const___override(x_439, x_350);
lean_inc(x_347);
x_441 = l_Lean_Expr_app___override(x_440, x_347);
x_442 = lp_mathlib_CancelDenoms_mkProdPrf___closed__28;
lean_inc_ref(x_350);
x_443 = l_Lean_Expr_const___override(x_442, x_350);
lean_inc(x_347);
x_444 = l_Lean_Expr_app___override(x_443, x_347);
x_445 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__4;
lean_inc_ref(x_350);
x_446 = l_Lean_Expr_const___override(x_445, x_350);
lean_inc(x_347);
x_447 = l_Lean_Expr_app___override(x_446, x_347);
x_448 = lp_mathlib_CancelDenoms_mkProdPrf___closed__67;
lean_inc_ref(x_350);
x_449 = l_Lean_Expr_const___override(x_448, x_350);
lean_inc(x_347);
x_450 = l_Lean_Expr_app___override(x_449, x_347);
x_451 = lp_mathlib_CancelDenoms_mkProdPrf___closed__70;
lean_inc_ref(x_350);
x_452 = l_Lean_Expr_const___override(x_451, x_350);
lean_inc(x_347);
x_453 = l_Lean_Expr_app___override(x_452, x_347);
x_454 = lp_mathlib_CancelDenoms_mkProdPrf___closed__73;
lean_inc_ref(x_350);
x_455 = l_Lean_Expr_const___override(x_454, x_350);
lean_inc(x_347);
x_456 = l_Lean_Expr_app___override(x_455, x_347);
x_457 = lp_mathlib_CancelDenoms_mkProdPrf___closed__76;
lean_inc_ref(x_350);
x_458 = l_Lean_Expr_const___override(x_457, x_350);
lean_inc(x_347);
x_459 = l_Lean_Expr_app___override(x_458, x_347);
x_460 = lp_mathlib_CancelDenoms_mkProdPrf___closed__78;
x_461 = l_Lean_Expr_const___override(x_460, x_350);
x_462 = l_Lean_Expr_app___override(x_461, x_347);
x_463 = l_Lean_Expr_app___override(x_462, x_427);
x_464 = l_Lean_Expr_app___override(x_459, x_463);
x_465 = l_Lean_Expr_app___override(x_456, x_464);
x_466 = l_Lean_Expr_app___override(x_453, x_465);
x_467 = l_Lean_Expr_app___override(x_450, x_466);
x_468 = l_Lean_Expr_app___override(x_447, x_467);
x_469 = l_Lean_Expr_app___override(x_444, x_468);
x_470 = l_Lean_Expr_app___override(x_441, x_469);
x_471 = l_Lean_Expr_app___override(x_438, x_470);
lean_inc_ref(x_471);
x_472 = l_Lean_Expr_app___override(x_433, x_471);
lean_inc_ref(x_432);
x_473 = l_Lean_Expr_app___override(x_432, x_420);
lean_inc_ref(x_471);
x_474 = l_Lean_Expr_app___override(x_473, x_471);
x_475 = l_Lean_Expr_app___override(x_432, x_421);
x_476 = l_Lean_Expr_app___override(x_475, x_471);
x_359 = x_472;
x_360 = x_474;
x_361 = x_476;
x_362 = x_2;
x_363 = x_3;
x_364 = x_4;
x_365 = x_5;
x_366 = lean_box(0);
goto block_407;
}
else
{
lean_object* x_477; lean_object* x_478; lean_object* x_479; 
lean_dec(x_422);
lean_dec(x_421);
lean_dec(x_420);
lean_dec(x_419);
lean_dec(x_358);
lean_dec_ref(x_350);
lean_dec(x_347);
lean_dec(x_29);
lean_dec(x_25);
lean_dec(x_20);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_477 = lean_ctor_get(x_426, 0);
lean_inc(x_477);
if (lean_is_exclusive(x_426)) {
 lean_ctor_release(x_426, 0);
 x_478 = x_426;
} else {
 lean_dec_ref(x_426);
 x_478 = lean_box(0);
}
if (lean_is_scalar(x_478)) {
 x_479 = lean_alloc_ctor(1, 1, 0);
} else {
 x_479 = x_478;
}
lean_ctor_set(x_479, 0, x_477);
return x_479;
}
}
else
{
lean_object* x_480; lean_object* x_481; lean_object* x_482; lean_object* x_483; lean_object* x_484; lean_object* x_485; lean_object* x_486; 
lean_dec(x_29);
x_480 = lean_ctor_get(x_410, 0);
lean_inc(x_480);
lean_dec(x_410);
x_481 = lean_ctor_get(x_413, 0);
lean_inc(x_481);
lean_dec(x_413);
x_482 = lean_ctor_get(x_417, 0);
lean_inc(x_482);
lean_dec(x_417);
x_483 = lp_mathlib_CancelDenoms_derive___closed__4;
lean_inc_ref(x_350);
x_484 = l_Lean_Expr_const___override(x_483, x_350);
lean_inc(x_347);
x_485 = l_Lean_Expr_app___override(x_484, x_347);
lean_inc(x_5);
lean_inc_ref(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
x_486 = lp_Qq_Qq_synthInstanceQ___redArg(x_485, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_486) == 0)
{
lean_object* x_487; lean_object* x_488; lean_object* x_489; lean_object* x_490; lean_object* x_491; 
x_487 = lean_ctor_get(x_486, 0);
lean_inc(x_487);
lean_dec_ref(x_486);
x_488 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__6;
lean_inc_ref(x_350);
x_489 = l_Lean_Expr_const___override(x_488, x_350);
lean_inc(x_347);
x_490 = l_Lean_Expr_app___override(x_489, x_347);
lean_inc(x_5);
lean_inc_ref(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
x_491 = lp_Qq_Qq_synthInstanceQ___redArg(x_490, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_491) == 0)
{
lean_object* x_492; lean_object* x_493; lean_object* x_494; lean_object* x_495; lean_object* x_496; lean_object* x_497; lean_object* x_498; lean_object* x_499; lean_object* x_500; lean_object* x_501; lean_object* x_502; lean_object* x_503; lean_object* x_504; lean_object* x_505; lean_object* x_506; lean_object* x_507; lean_object* x_508; lean_object* x_509; lean_object* x_510; lean_object* x_511; lean_object* x_512; lean_object* x_513; lean_object* x_514; lean_object* x_515; lean_object* x_516; lean_object* x_517; lean_object* x_518; lean_object* x_519; lean_object* x_520; lean_object* x_521; lean_object* x_522; lean_object* x_523; lean_object* x_524; lean_object* x_525; lean_object* x_526; 
x_492 = lean_ctor_get(x_491, 0);
lean_inc(x_492);
lean_dec_ref(x_491);
x_493 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__8;
lean_inc_ref(x_350);
x_494 = l_Lean_Expr_const___override(x_493, x_350);
lean_inc(x_347);
x_495 = l_Lean_Expr_app___override(x_494, x_347);
x_496 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__16;
lean_inc_ref(x_350);
x_497 = l_Lean_Expr_const___override(x_496, x_350);
lean_inc(x_347);
x_498 = l_Lean_Expr_app___override(x_497, x_347);
x_499 = lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__18;
lean_inc_ref(x_350);
x_500 = l_Lean_Expr_const___override(x_499, x_350);
lean_inc(x_347);
x_501 = l_Lean_Expr_app___override(x_500, x_347);
x_502 = lp_mathlib_CancelDenoms_mkProdPrf___closed__34;
lean_inc_ref(x_350);
x_503 = l_Lean_Expr_const___override(x_502, x_350);
lean_inc(x_347);
x_504 = l_Lean_Expr_app___override(x_503, x_347);
lean_inc(x_487);
x_505 = l_Lean_Expr_app___override(x_504, x_487);
x_506 = l_Lean_Expr_app___override(x_501, x_505);
x_507 = l_Lean_Expr_app___override(x_498, x_506);
x_508 = l_Lean_Expr_app___override(x_495, x_507);
x_509 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__11;
lean_inc_ref(x_350);
x_510 = l_Lean_Expr_const___override(x_509, x_350);
lean_inc(x_347);
x_511 = l_Lean_Expr_app___override(x_510, x_347);
x_512 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__14;
lean_inc_ref(x_350);
x_513 = l_Lean_Expr_const___override(x_512, x_350);
lean_inc(x_347);
x_514 = l_Lean_Expr_app___override(x_513, x_347);
x_515 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__17;
lean_inc_ref(x_350);
x_516 = l_Lean_Expr_const___override(x_515, x_350);
lean_inc(x_347);
x_517 = l_Lean_Expr_app___override(x_516, x_347);
x_518 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__19;
lean_inc_ref(x_350);
x_519 = l_Lean_Expr_const___override(x_518, x_350);
lean_inc(x_347);
x_520 = l_Lean_Expr_app___override(x_519, x_347);
x_521 = l_Lean_Expr_app___override(x_520, x_492);
x_522 = l_Lean_Expr_app___override(x_517, x_521);
x_523 = l_Lean_Expr_app___override(x_514, x_522);
x_524 = l_Lean_Expr_app___override(x_511, x_523);
lean_inc_ref(x_524);
x_525 = l_Lean_Expr_app___override(x_508, x_524);
lean_inc(x_5);
lean_inc_ref(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
x_526 = lp_Qq_Qq_synthInstanceQ___redArg(x_525, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_526) == 0)
{
lean_object* x_527; lean_object* x_528; lean_object* x_529; lean_object* x_530; lean_object* x_531; lean_object* x_532; lean_object* x_533; lean_object* x_534; lean_object* x_535; lean_object* x_536; lean_object* x_537; lean_object* x_538; lean_object* x_539; lean_object* x_540; lean_object* x_541; lean_object* x_542; lean_object* x_543; lean_object* x_544; lean_object* x_545; lean_object* x_546; lean_object* x_547; lean_object* x_548; lean_object* x_549; lean_object* x_550; lean_object* x_551; lean_object* x_552; lean_object* x_553; lean_object* x_554; lean_object* x_555; lean_object* x_556; lean_object* x_557; lean_object* x_558; lean_object* x_559; lean_object* x_560; lean_object* x_561; lean_object* x_562; lean_object* x_563; lean_object* x_564; lean_object* x_565; lean_object* x_566; lean_object* x_567; lean_object* x_568; lean_object* x_569; lean_object* x_570; lean_object* x_571; lean_object* x_572; lean_object* x_573; lean_object* x_574; lean_object* x_575; lean_object* x_576; lean_object* x_577; lean_object* x_578; lean_object* x_579; lean_object* x_580; 
lean_dec_ref(x_526);
x_527 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__20;
lean_inc_ref(x_350);
x_528 = l_Lean_Expr_const___override(x_527, x_350);
lean_inc(x_347);
x_529 = l_Lean_Expr_app___override(x_528, x_347);
x_530 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__23;
lean_inc_ref(x_350);
x_531 = l_Lean_Expr_const___override(x_530, x_350);
lean_inc(x_347);
x_532 = l_Lean_Expr_app___override(x_531, x_347);
x_533 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__26;
lean_inc_ref(x_350);
x_534 = l_Lean_Expr_const___override(x_533, x_350);
lean_inc(x_347);
x_535 = l_Lean_Expr_app___override(x_534, x_347);
x_536 = l_Lean_Expr_app___override(x_535, x_524);
x_537 = l_Lean_Expr_app___override(x_532, x_536);
x_538 = l_Lean_Expr_app___override(x_529, x_537);
x_539 = lp_mathlib_CancelDenoms_mkProdPrf___closed__20;
lean_inc_ref(x_350);
x_540 = l_Lean_Expr_const___override(x_539, x_350);
lean_inc(x_347);
x_541 = l_Lean_Expr_app___override(x_540, x_347);
x_542 = lp_mathlib_CancelDenoms_mkProdPrf___closed__22;
x_543 = l_Lean_Expr_app___override(x_541, x_542);
x_544 = lp_mathlib_CancelDenoms_mkProdPrf___closed__25;
lean_inc_ref(x_350);
x_545 = l_Lean_Expr_const___override(x_544, x_350);
lean_inc(x_347);
x_546 = l_Lean_Expr_app___override(x_545, x_347);
x_547 = lp_mathlib_CancelDenoms_mkProdPrf___closed__28;
lean_inc_ref(x_350);
x_548 = l_Lean_Expr_const___override(x_547, x_350);
lean_inc(x_347);
x_549 = l_Lean_Expr_app___override(x_548, x_347);
x_550 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__4;
lean_inc_ref(x_350);
x_551 = l_Lean_Expr_const___override(x_550, x_350);
lean_inc(x_347);
x_552 = l_Lean_Expr_app___override(x_551, x_347);
x_553 = lp_mathlib_CancelDenoms_mkProdPrf___closed__67;
lean_inc_ref(x_350);
x_554 = l_Lean_Expr_const___override(x_553, x_350);
lean_inc(x_347);
x_555 = l_Lean_Expr_app___override(x_554, x_347);
x_556 = lp_mathlib_CancelDenoms_mkProdPrf___closed__70;
lean_inc_ref(x_350);
x_557 = l_Lean_Expr_const___override(x_556, x_350);
lean_inc(x_347);
x_558 = l_Lean_Expr_app___override(x_557, x_347);
x_559 = lp_mathlib_CancelDenoms_mkProdPrf___closed__73;
lean_inc_ref(x_350);
x_560 = l_Lean_Expr_const___override(x_559, x_350);
lean_inc(x_347);
x_561 = l_Lean_Expr_app___override(x_560, x_347);
x_562 = lp_mathlib_CancelDenoms_mkProdPrf___closed__76;
lean_inc_ref(x_350);
x_563 = l_Lean_Expr_const___override(x_562, x_350);
lean_inc(x_347);
x_564 = l_Lean_Expr_app___override(x_563, x_347);
x_565 = lp_mathlib_CancelDenoms_mkProdPrf___closed__78;
x_566 = l_Lean_Expr_const___override(x_565, x_350);
x_567 = l_Lean_Expr_app___override(x_566, x_347);
x_568 = l_Lean_Expr_app___override(x_567, x_487);
x_569 = l_Lean_Expr_app___override(x_564, x_568);
x_570 = l_Lean_Expr_app___override(x_561, x_569);
x_571 = l_Lean_Expr_app___override(x_558, x_570);
x_572 = l_Lean_Expr_app___override(x_555, x_571);
x_573 = l_Lean_Expr_app___override(x_552, x_572);
x_574 = l_Lean_Expr_app___override(x_549, x_573);
x_575 = l_Lean_Expr_app___override(x_546, x_574);
x_576 = l_Lean_Expr_app___override(x_543, x_575);
x_577 = l_Lean_Expr_app___override(x_538, x_576);
lean_inc_ref(x_577);
x_578 = l_Lean_Expr_app___override(x_577, x_480);
lean_inc_ref(x_577);
x_579 = l_Lean_Expr_app___override(x_577, x_481);
x_580 = l_Lean_Expr_app___override(x_577, x_482);
x_359 = x_578;
x_360 = x_579;
x_361 = x_580;
x_362 = x_2;
x_363 = x_3;
x_364 = x_4;
x_365 = x_5;
x_366 = lean_box(0);
goto block_407;
}
else
{
lean_object* x_581; lean_object* x_582; lean_object* x_583; 
lean_dec_ref(x_524);
lean_dec(x_487);
lean_dec(x_482);
lean_dec(x_481);
lean_dec(x_480);
lean_dec(x_358);
lean_dec_ref(x_350);
lean_dec(x_347);
lean_dec(x_25);
lean_dec(x_20);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_581 = lean_ctor_get(x_526, 0);
lean_inc(x_581);
if (lean_is_exclusive(x_526)) {
 lean_ctor_release(x_526, 0);
 x_582 = x_526;
} else {
 lean_dec_ref(x_526);
 x_582 = lean_box(0);
}
if (lean_is_scalar(x_582)) {
 x_583 = lean_alloc_ctor(1, 1, 0);
} else {
 x_583 = x_582;
}
lean_ctor_set(x_583, 0, x_581);
return x_583;
}
}
else
{
lean_object* x_584; lean_object* x_585; lean_object* x_586; 
lean_dec(x_487);
lean_dec(x_482);
lean_dec(x_481);
lean_dec(x_480);
lean_dec(x_358);
lean_dec_ref(x_350);
lean_dec(x_347);
lean_dec(x_25);
lean_dec(x_20);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_584 = lean_ctor_get(x_491, 0);
lean_inc(x_584);
if (lean_is_exclusive(x_491)) {
 lean_ctor_release(x_491, 0);
 x_585 = x_491;
} else {
 lean_dec_ref(x_491);
 x_585 = lean_box(0);
}
if (lean_is_scalar(x_585)) {
 x_586 = lean_alloc_ctor(1, 1, 0);
} else {
 x_586 = x_585;
}
lean_ctor_set(x_586, 0, x_584);
return x_586;
}
}
else
{
lean_object* x_587; lean_object* x_588; lean_object* x_589; 
lean_dec(x_482);
lean_dec(x_481);
lean_dec(x_480);
lean_dec(x_358);
lean_dec_ref(x_350);
lean_dec(x_347);
lean_dec(x_25);
lean_dec(x_20);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_587 = lean_ctor_get(x_486, 0);
lean_inc(x_587);
if (lean_is_exclusive(x_486)) {
 lean_ctor_release(x_486, 0);
 x_588 = x_486;
} else {
 lean_dec_ref(x_486);
 x_588 = lean_box(0);
}
if (lean_is_scalar(x_588)) {
 x_589 = lean_alloc_ctor(1, 1, 0);
} else {
 x_589 = x_588;
}
lean_ctor_set(x_589, 0, x_587);
return x_589;
}
}
}
else
{
lean_object* x_590; lean_object* x_591; lean_object* x_592; 
lean_dec(x_413);
lean_dec(x_410);
lean_dec(x_358);
lean_dec_ref(x_350);
lean_dec(x_347);
lean_dec(x_29);
lean_dec(x_25);
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_590 = lean_ctor_get(x_416, 0);
lean_inc(x_590);
if (lean_is_exclusive(x_416)) {
 lean_ctor_release(x_416, 0);
 x_591 = x_416;
} else {
 lean_dec_ref(x_416);
 x_591 = lean_box(0);
}
if (lean_is_scalar(x_591)) {
 x_592 = lean_alloc_ctor(1, 1, 0);
} else {
 x_592 = x_591;
}
lean_ctor_set(x_592, 0, x_590);
return x_592;
}
}
else
{
lean_object* x_593; lean_object* x_594; lean_object* x_595; 
lean_dec(x_410);
lean_dec(x_358);
lean_dec(x_357);
lean_dec(x_354);
lean_dec_ref(x_350);
lean_dec(x_347);
lean_dec(x_29);
lean_dec(x_25);
lean_dec(x_24);
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_593 = lean_ctor_get(x_412, 0);
lean_inc(x_593);
if (lean_is_exclusive(x_412)) {
 lean_ctor_release(x_412, 0);
 x_594 = x_412;
} else {
 lean_dec_ref(x_412);
 x_594 = lean_box(0);
}
if (lean_is_scalar(x_594)) {
 x_595 = lean_alloc_ctor(1, 1, 0);
} else {
 x_595 = x_594;
}
lean_ctor_set(x_595, 0, x_593);
return x_595;
}
}
else
{
lean_object* x_596; lean_object* x_597; lean_object* x_598; 
lean_dec(x_358);
lean_dec(x_357);
lean_dec(x_354);
lean_dec_ref(x_350);
lean_dec(x_347);
lean_dec(x_29);
lean_dec(x_25);
lean_dec(x_24);
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_596 = lean_ctor_get(x_409, 0);
lean_inc(x_596);
if (lean_is_exclusive(x_409)) {
 lean_ctor_release(x_409, 0);
 x_597 = x_409;
} else {
 lean_dec_ref(x_409);
 x_597 = lean_box(0);
}
if (lean_is_scalar(x_597)) {
 x_598 = lean_alloc_ctor(1, 1, 0);
} else {
 x_598 = x_597;
}
lean_ctor_set(x_598, 0, x_596);
return x_598;
}
block_407:
{
lean_object* x_367; 
lean_inc(x_365);
lean_inc_ref(x_364);
lean_inc(x_363);
lean_inc_ref(x_362);
x_367 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum(x_359, x_362, x_363, x_364, x_365);
if (lean_obj_tag(x_367) == 0)
{
lean_object* x_368; lean_object* x_369; 
x_368 = lean_ctor_get(x_367, 0);
lean_inc(x_368);
lean_dec_ref(x_367);
lean_inc(x_365);
lean_inc_ref(x_364);
lean_inc(x_363);
lean_inc_ref(x_362);
x_369 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum(x_360, x_362, x_363, x_364, x_365);
if (lean_obj_tag(x_369) == 0)
{
lean_object* x_370; lean_object* x_371; 
x_370 = lean_ctor_get(x_369, 0);
lean_inc(x_370);
lean_dec_ref(x_369);
lean_inc(x_365);
lean_inc_ref(x_364);
lean_inc(x_363);
lean_inc_ref(x_362);
x_371 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum(x_361, x_362, x_363, x_364, x_365);
if (lean_obj_tag(x_371) == 0)
{
lean_object* x_372; lean_object* x_373; lean_object* x_374; lean_object* x_375; lean_object* x_376; lean_object* x_377; lean_object* x_378; lean_object* x_379; 
x_372 = lean_ctor_get(x_371, 0);
lean_inc(x_372);
lean_dec_ref(x_371);
x_373 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__0;
x_374 = lean_array_push(x_373, x_25);
x_375 = lean_array_push(x_374, x_358);
x_376 = lean_array_push(x_375, x_368);
x_377 = lean_array_push(x_376, x_370);
x_378 = lean_array_push(x_377, x_372);
lean_inc(x_365);
lean_inc_ref(x_364);
lean_inc(x_363);
lean_inc_ref(x_362);
x_379 = l_Lean_Meta_mkAppM(x_20, x_378, x_362, x_363, x_364, x_365);
if (lean_obj_tag(x_379) == 0)
{
lean_object* x_380; lean_object* x_381; 
x_380 = lean_ctor_get(x_379, 0);
lean_inc(x_380);
lean_dec_ref(x_379);
lean_inc(x_365);
lean_inc_ref(x_364);
lean_inc(x_363);
lean_inc_ref(x_362);
lean_inc(x_380);
x_381 = lean_infer_type(x_380, x_362, x_363, x_364, x_365);
if (lean_obj_tag(x_381) == 0)
{
lean_object* x_382; lean_object* x_383; 
x_382 = lean_ctor_get(x_381, 0);
lean_inc(x_382);
lean_dec_ref(x_381);
x_383 = lp_mathlib_CancelDenoms_findCompLemma(x_382, x_362, x_363, x_364, x_365);
if (lean_obj_tag(x_383) == 0)
{
lean_object* x_384; 
x_384 = lean_ctor_get(x_383, 0);
lean_inc(x_384);
lean_dec_ref(x_383);
if (lean_obj_tag(x_384) == 0)
{
lean_object* x_385; 
x_385 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__3;
x_7 = x_380;
x_8 = lean_box(0);
x_9 = x_385;
goto block_12;
}
else
{
lean_object* x_386; lean_object* x_387; lean_object* x_388; 
x_386 = lean_ctor_get(x_384, 0);
lean_inc(x_386);
lean_dec_ref(x_384);
x_387 = lean_ctor_get(x_386, 1);
lean_inc(x_387);
lean_dec(x_386);
x_388 = lean_ctor_get(x_387, 0);
lean_inc(x_388);
lean_dec(x_387);
x_7 = x_380;
x_8 = lean_box(0);
x_9 = x_388;
goto block_12;
}
}
else
{
lean_object* x_389; lean_object* x_390; lean_object* x_391; 
lean_dec(x_380);
x_389 = lean_ctor_get(x_383, 0);
lean_inc(x_389);
if (lean_is_exclusive(x_383)) {
 lean_ctor_release(x_383, 0);
 x_390 = x_383;
} else {
 lean_dec_ref(x_383);
 x_390 = lean_box(0);
}
if (lean_is_scalar(x_390)) {
 x_391 = lean_alloc_ctor(1, 1, 0);
} else {
 x_391 = x_390;
}
lean_ctor_set(x_391, 0, x_389);
return x_391;
}
}
else
{
lean_object* x_392; lean_object* x_393; lean_object* x_394; 
lean_dec(x_380);
lean_dec(x_365);
lean_dec_ref(x_364);
lean_dec(x_363);
lean_dec_ref(x_362);
x_392 = lean_ctor_get(x_381, 0);
lean_inc(x_392);
if (lean_is_exclusive(x_381)) {
 lean_ctor_release(x_381, 0);
 x_393 = x_381;
} else {
 lean_dec_ref(x_381);
 x_393 = lean_box(0);
}
if (lean_is_scalar(x_393)) {
 x_394 = lean_alloc_ctor(1, 1, 0);
} else {
 x_394 = x_393;
}
lean_ctor_set(x_394, 0, x_392);
return x_394;
}
}
else
{
lean_object* x_395; lean_object* x_396; lean_object* x_397; 
lean_dec(x_365);
lean_dec_ref(x_364);
lean_dec(x_363);
lean_dec_ref(x_362);
x_395 = lean_ctor_get(x_379, 0);
lean_inc(x_395);
if (lean_is_exclusive(x_379)) {
 lean_ctor_release(x_379, 0);
 x_396 = x_379;
} else {
 lean_dec_ref(x_379);
 x_396 = lean_box(0);
}
if (lean_is_scalar(x_396)) {
 x_397 = lean_alloc_ctor(1, 1, 0);
} else {
 x_397 = x_396;
}
lean_ctor_set(x_397, 0, x_395);
return x_397;
}
}
else
{
lean_object* x_398; lean_object* x_399; lean_object* x_400; 
lean_dec(x_370);
lean_dec(x_368);
lean_dec(x_365);
lean_dec_ref(x_364);
lean_dec(x_363);
lean_dec_ref(x_362);
lean_dec(x_358);
lean_dec(x_25);
lean_dec(x_20);
x_398 = lean_ctor_get(x_371, 0);
lean_inc(x_398);
if (lean_is_exclusive(x_371)) {
 lean_ctor_release(x_371, 0);
 x_399 = x_371;
} else {
 lean_dec_ref(x_371);
 x_399 = lean_box(0);
}
if (lean_is_scalar(x_399)) {
 x_400 = lean_alloc_ctor(1, 1, 0);
} else {
 x_400 = x_399;
}
lean_ctor_set(x_400, 0, x_398);
return x_400;
}
}
else
{
lean_object* x_401; lean_object* x_402; lean_object* x_403; 
lean_dec(x_368);
lean_dec(x_365);
lean_dec_ref(x_364);
lean_dec(x_363);
lean_dec_ref(x_362);
lean_dec_ref(x_361);
lean_dec(x_358);
lean_dec(x_25);
lean_dec(x_20);
x_401 = lean_ctor_get(x_369, 0);
lean_inc(x_401);
if (lean_is_exclusive(x_369)) {
 lean_ctor_release(x_369, 0);
 x_402 = x_369;
} else {
 lean_dec_ref(x_369);
 x_402 = lean_box(0);
}
if (lean_is_scalar(x_402)) {
 x_403 = lean_alloc_ctor(1, 1, 0);
} else {
 x_403 = x_402;
}
lean_ctor_set(x_403, 0, x_401);
return x_403;
}
}
else
{
lean_object* x_404; lean_object* x_405; lean_object* x_406; 
lean_dec(x_365);
lean_dec_ref(x_364);
lean_dec(x_363);
lean_dec_ref(x_362);
lean_dec_ref(x_361);
lean_dec_ref(x_360);
lean_dec(x_358);
lean_dec(x_25);
lean_dec(x_20);
x_404 = lean_ctor_get(x_367, 0);
lean_inc(x_404);
if (lean_is_exclusive(x_367)) {
 lean_ctor_release(x_367, 0);
 x_405 = x_367;
} else {
 lean_dec_ref(x_367);
 x_405 = lean_box(0);
}
if (lean_is_scalar(x_405)) {
 x_406 = lean_alloc_ctor(1, 1, 0);
} else {
 x_406 = x_405;
}
lean_ctor_set(x_406, 0, x_404);
return x_406;
}
}
}
else
{
lean_object* x_599; lean_object* x_600; lean_object* x_601; 
lean_dec(x_354);
lean_dec_ref(x_350);
lean_dec(x_347);
lean_dec(x_29);
lean_dec(x_25);
lean_dec(x_24);
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_599 = lean_ctor_get(x_355, 0);
lean_inc(x_599);
if (lean_is_exclusive(x_355)) {
 lean_ctor_release(x_355, 0);
 x_600 = x_355;
} else {
 lean_dec_ref(x_355);
 x_600 = lean_box(0);
}
if (lean_is_scalar(x_600)) {
 x_601 = lean_alloc_ctor(1, 1, 0);
} else {
 x_601 = x_600;
}
lean_ctor_set(x_601, 0, x_599);
return x_601;
}
}
else
{
lean_object* x_602; lean_object* x_603; lean_object* x_604; 
lean_dec_ref(x_350);
lean_dec(x_347);
lean_dec(x_29);
lean_dec(x_25);
lean_dec(x_24);
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_19);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_602 = lean_ctor_get(x_353, 0);
lean_inc(x_602);
if (lean_is_exclusive(x_353)) {
 lean_ctor_release(x_353, 0);
 x_603 = x_353;
} else {
 lean_dec_ref(x_353);
 x_603 = lean_box(0);
}
if (lean_is_scalar(x_603)) {
 x_604 = lean_alloc_ctor(1, 1, 0);
} else {
 x_604 = x_603;
}
lean_ctor_set(x_604, 0, x_602);
return x_604;
}
}
}
else
{
uint8_t x_605; 
lean_dec(x_25);
lean_dec(x_24);
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_19);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_605 = !lean_is_exclusive(x_26);
if (x_605 == 0)
{
return x_26;
}
else
{
lean_object* x_606; lean_object* x_607; 
x_606 = lean_ctor_get(x_26, 0);
lean_inc(x_606);
lean_dec(x_26);
x_607 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_607, 0, x_606);
return x_607;
}
}
}
else
{
uint8_t x_608; 
lean_dec(x_21);
lean_dec(x_20);
lean_dec(x_19);
lean_dec(x_18);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_608 = !lean_is_exclusive(x_22);
if (x_608 == 0)
{
return x_22;
}
else
{
lean_object* x_609; lean_object* x_610; 
x_609 = lean_ctor_get(x_22, 0);
lean_inc(x_609);
lean_dec(x_22);
x_610 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_610, 0, x_609);
return x_610;
}
}
}
else
{
lean_object* x_611; lean_object* x_612; 
lean_dec(x_14);
x_611 = lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__28;
x_612 = lp_mathlib_Lean_throwError___at___00CancelDenoms_synthesizeUsingNormNum_spec__0___redArg(x_611, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_612;
}
}
else
{
uint8_t x_613; 
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_613 = !lean_is_exclusive(x_13);
if (x_613 == 0)
{
return x_13;
}
else
{
lean_object* x_614; lean_object* x_615; 
x_614 = lean_ctor_get(x_13, 0);
lean_inc(x_614);
lean_dec(x_13);
x_615 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_615, 0, x_614);
return x_615;
}
}
block_12:
{
lean_object* x_10; lean_object* x_11; 
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_9);
lean_ctor_set(x_10, 1, x_7);
x_11 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_11, 0, x_10);
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CancelDenoms_cancelDenominatorsInType___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CancelDenoms_cancelDenominatorsInType(x_1, x_2, x_3, x_4, x_5);
return x_7;
}
}
static lean_object* _init_lp_mathlib_cancelDenoms___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("cancelDenoms", 12, 12);
return x_1;
}
}
static lean_object* _init_lp_mathlib_cancelDenoms___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_cancelDenoms___closed__0;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_cancelDenoms___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("andthen", 7, 7);
return x_1;
}
}
static lean_object* _init_lp_mathlib_cancelDenoms___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_cancelDenoms___closed__2;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_cancelDenoms___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("cancel_denoms", 13, 13);
return x_1;
}
}
static lean_object* _init_lp_mathlib_cancelDenoms___closed__5() {
_start:
{
uint8_t x_1; lean_object* x_2; lean_object* x_3; 
x_1 = 0;
x_2 = lp_mathlib_cancelDenoms___closed__4;
x_3 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set_uint8(x_3, sizeof(void*)*1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_cancelDenoms___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("optional", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_cancelDenoms___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_cancelDenoms___closed__6;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_cancelDenoms___closed__8() {
_start:
{
lean_object* x_1; 
x_1 = l_Lean_Parser_Tactic_location;
return x_1;
}
}
static lean_object* _init_lp_mathlib_cancelDenoms___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_cancelDenoms___closed__8;
x_2 = lp_mathlib_cancelDenoms___closed__7;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_cancelDenoms___closed__10() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_cancelDenoms___closed__9;
x_2 = lp_mathlib_cancelDenoms___closed__5;
x_3 = lp_mathlib_cancelDenoms___closed__3;
x_4 = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_cancelDenoms___closed__11() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_cancelDenoms___closed__10;
x_2 = lean_unsigned_to_nat(1022u);
x_3 = lp_mathlib_cancelDenoms___closed__1;
x_4 = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_cancelDenoms() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_cancelDenoms___closed__11;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00cancelDenominatorsAt_spec__0___redArg(lean_object* x_1, lean_object* x_2) {
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00cancelDenominatorsAt_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_Lean_instantiateMVars___at___00cancelDenominatorsAt_spec__0___redArg(x_1, x_7);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_cancelDenominatorsAt___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; 
x_10 = l___private_Lean_Meta_Tactic_Replace_0__Lean_Meta_replaceLocalDeclCore(x_4, x_1, x_2, x_3, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_10) == 0)
{
uint8_t x_11; 
x_11 = !lean_is_exclusive(x_10);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; 
x_12 = lean_ctor_get(x_10, 0);
x_13 = lean_ctor_get(x_12, 1);
lean_inc(x_13);
lean_dec(x_12);
lean_ctor_set(x_10, 0, x_13);
return x_10;
}
else
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_14 = lean_ctor_get(x_10, 0);
lean_inc(x_14);
lean_dec(x_10);
x_15 = lean_ctor_get(x_14, 1);
lean_inc(x_15);
lean_dec(x_14);
x_16 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_16, 0, x_15);
return x_16;
}
}
else
{
uint8_t x_17; 
x_17 = !lean_is_exclusive(x_10);
if (x_17 == 0)
{
return x_10;
}
else
{
lean_object* x_18; lean_object* x_19; 
x_18 = lean_ctor_get(x_10, 0);
lean_inc(x_18);
lean_dec(x_10);
x_19 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_19, 0, x_18);
return x_19;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_cancelDenominatorsAt___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_cancelDenominatorsAt___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_cancelDenominatorsAt(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; 
lean_inc_ref(x_6);
lean_inc(x_1);
x_11 = l_Lean_FVarId_getDecl___redArg(x_1, x_6, x_8, x_9);
if (lean_obj_tag(x_11) == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
lean_dec_ref(x_11);
x_13 = l_Lean_LocalDecl_type(x_12);
lean_dec(x_12);
x_14 = lp_mathlib_Lean_instantiateMVars___at___00cancelDenominatorsAt_spec__0___redArg(x_13, x_7);
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
lean_dec_ref(x_14);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
x_16 = lp_mathlib_CancelDenoms_cancelDenominatorsInType(x_15, x_6, x_7, x_8, x_9);
if (lean_obj_tag(x_16) == 0)
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_17 = lean_ctor_get(x_16, 0);
lean_inc(x_17);
lean_dec_ref(x_16);
x_18 = lean_ctor_get(x_17, 0);
lean_inc(x_18);
x_19 = lean_ctor_get(x_17, 1);
lean_inc(x_19);
lean_dec(x_17);
x_20 = lean_alloc_closure((void*)(lp_mathlib_cancelDenominatorsAt___lam__0___boxed), 9, 3);
lean_closure_set(x_20, 0, x_1);
lean_closure_set(x_20, 1, x_18);
lean_closure_set(x_20, 2, x_19);
x_21 = lp_mathlib_Lean_Elab_Tactic_liftMetaTactic_x27(x_20, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_21;
}
else
{
uint8_t x_22; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_22 = !lean_is_exclusive(x_16);
if (x_22 == 0)
{
return x_16;
}
else
{
lean_object* x_23; lean_object* x_24; 
x_23 = lean_ctor_get(x_16, 0);
lean_inc(x_23);
lean_dec(x_16);
x_24 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_24, 0, x_23);
return x_24;
}
}
}
else
{
uint8_t x_25; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_25 = !lean_is_exclusive(x_11);
if (x_25 == 0)
{
return x_11;
}
else
{
lean_object* x_26; lean_object* x_27; 
x_26 = lean_ctor_get(x_11, 0);
lean_inc(x_26);
lean_dec(x_11);
x_27 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_27, 0, x_26);
return x_27;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00cancelDenominatorsAt_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_Lean_instantiateMVars___at___00cancelDenominatorsAt_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_cancelDenominatorsAt___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_cancelDenominatorsAt(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00cancelDenominatorsAt_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Lean_instantiateMVars___at___00cancelDenominatorsAt_spec__0___redArg(x_1, x_2);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_cancelDenominatorsTarget___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_9; 
x_9 = l_Lean_MVarId_replaceTargetEq(x_3, x_1, x_2, x_4, x_5, x_6, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_cancelDenominatorsTarget___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_cancelDenominatorsTarget___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_cancelDenominatorsTarget(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; 
x_10 = l_Lean_Elab_Tactic_getMainTarget(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_10) == 0)
{
lean_object* x_11; lean_object* x_12; 
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
lean_dec_ref(x_10);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
x_12 = lp_mathlib_CancelDenoms_cancelDenominatorsInType(x_11, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_12) == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
x_15 = lean_ctor_get(x_13, 1);
lean_inc(x_15);
lean_dec(x_13);
x_16 = lean_alloc_closure((void*)(lp_mathlib_cancelDenominatorsTarget___lam__0___boxed), 8, 2);
lean_closure_set(x_16, 0, x_14);
lean_closure_set(x_16, 1, x_15);
x_17 = lp_mathlib_Lean_Elab_Tactic_liftMetaTactic_x27(x_16, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
return x_17;
}
else
{
uint8_t x_18; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
x_18 = !lean_is_exclusive(x_12);
if (x_18 == 0)
{
return x_12;
}
else
{
lean_object* x_19; lean_object* x_20; 
x_19 = lean_ctor_get(x_12, 0);
lean_inc(x_19);
lean_dec(x_12);
x_20 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_20, 0, x_19);
return x_20;
}
}
}
else
{
uint8_t x_21; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
x_21 = !lean_is_exclusive(x_10);
if (x_21 == 0)
{
return x_10;
}
else
{
lean_object* x_22; lean_object* x_23; 
x_22 = lean_ctor_get(x_10, 0);
lean_inc(x_22);
lean_dec(x_10);
x_23 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_23, 0, x_22);
return x_23;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_cancelDenominatorsTarget___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_cancelDenominatorsTarget(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00cancelDenominators_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_7 = lean_ctor_get(x_4, 5);
x_8 = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00CancelDenoms_synthesizeUsingNormNum_spec__0_spec__0(x_1, x_2, x_3, x_4, x_5);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00cancelDenominators_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Lean_throwError___at___00cancelDenominators_spec__0___redArg(x_2, x_7, x_8, x_9, x_10);
return x_12;
}
}
static lean_object* _init_lp_mathlib_cancelDenominators___lam__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Failed to cancel any denominators", 33, 33);
return x_1;
}
}
static lean_object* _init_lp_mathlib_cancelDenominators___lam__0___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_cancelDenominators___lam__0___closed__0;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_cancelDenominators___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; lean_object* x_12; 
x_11 = lp_mathlib_cancelDenominators___lam__0___closed__1;
x_12 = lp_mathlib_Lean_throwError___at___00cancelDenominators_spec__0___redArg(x_11, x_6, x_7, x_8, x_9);
return x_12;
}
}
static lean_object* _init_lp_mathlib_cancelDenominators___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_cancelDenominatorsAt___boxed), 10, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_cancelDenominators___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_cancelDenominators___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_cancelDenominators(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_11 = lean_alloc_closure((void*)(lp_mathlib_cancelDenominators___lam__0___boxed), 10, 0);
x_12 = lp_mathlib_cancelDenominators___closed__0;
x_13 = lean_alloc_closure((void*)(lp_mathlib_cancelDenominatorsTarget___boxed), 9, 0);
x_14 = l_Lean_Elab_Tactic_withLocation(x_1, x_12, x_13, x_11, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00cancelDenominators_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_Lean_throwError___at___00cancelDenominators_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_cancelDenominators___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_cancelDenominators(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_1);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00cancelDenominators_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Lean_throwError___at___00cancelDenominators_spec__0___redArg(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_7;
}
}
static lean_object* _init_lp_mathlib_tacticCancel__denoms___00__closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("tacticCancel_denoms_", 20, 20);
return x_1;
}
}
static lean_object* _init_lp_mathlib_tacticCancel__denoms___00__closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_tacticCancel__denoms___00__closed__0;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_tacticCancel__denoms___00__closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_cancelDenoms___closed__10;
x_2 = lean_unsigned_to_nat(1022u);
x_3 = lp_mathlib_tacticCancel__denoms___00__closed__1;
x_4 = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_tacticCancel__denoms__() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_tacticCancel__denoms___00__closed__2;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1_spec__0___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = l_Lean_Elab_unsupportedSyntaxExceptionId;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1_spec__0___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1_spec__0___redArg___closed__0;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1_spec__0___redArg() {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1_spec__0___redArg___closed__1;
x_3 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1_spec__0___redArg();
return x_11;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("tacticTry_", 10, 10);
return x_1;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__0;
x_2 = lp_mathlib_initFn___closed__8_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_3 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__4;
x_4 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__3;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("try", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("tacticSeq", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__3;
x_2 = lp_mathlib_initFn___closed__8_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_3 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__4;
x_4 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__3;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("tacticSeq1Indented", 18, 18);
return x_1;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__5;
x_2 = lp_mathlib_initFn___closed__8_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_3 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__4;
x_4 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__3;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__7() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("simpArgs", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__8() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__7;
x_2 = lp_mathlib_initFn___closed__8_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_3 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__4;
x_4 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__3;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__9() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("[", 1, 1);
return x_1;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__10() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("simpLemma", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__11() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__10;
x_2 = lp_mathlib_initFn___closed__8_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_;
x_3 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__4;
x_4 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__3;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__12() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("patternIgnore", 13, 13);
return x_1;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__13() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__12;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__14() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("token", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__15() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("← ", 4, 2);
return x_1;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__16() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__15;
x_2 = lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__14;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__17() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("←", 3, 1);
return x_1;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__18() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("mul_assoc", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__19() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__18;
x_2 = l_String_toRawSubstring_x27(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__20() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__18;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__21() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__20;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__22() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__21;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__23() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("]", 1, 1);
return x_1;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__24() {
_start:
{
lean_object* x_1; 
x_1 = l_Array_empty(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_34; lean_object* x_35; lean_object* x_83; uint8_t x_84; 
x_83 = lp_mathlib_tacticCancel__denoms___00__closed__1;
lean_inc(x_1);
x_84 = l_Lean_Syntax_isOfKind(x_1, x_83);
if (x_84 == 0)
{
lean_object* x_85; 
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_85 = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1_spec__0___redArg();
return x_85;
}
else
{
lean_object* x_86; lean_object* x_87; lean_object* x_88; 
x_86 = lean_unsigned_to_nat(1u);
x_87 = l_Lean_Syntax_getArg(x_1, x_86);
lean_dec(x_1);
x_88 = l_Lean_Syntax_getOptional_x3f(x_87);
lean_dec(x_87);
if (lean_obj_tag(x_88) == 0)
{
lean_object* x_89; 
x_89 = lean_box(0);
x_34 = x_89;
x_35 = x_89;
goto block_82;
}
else
{
uint8_t x_90; 
x_90 = !lean_is_exclusive(x_88);
if (x_90 == 0)
{
lean_inc_ref(x_88);
x_34 = x_88;
x_35 = x_88;
goto block_82;
}
else
{
lean_object* x_91; lean_object* x_92; 
x_91 = lean_ctor_get(x_88, 0);
lean_inc(x_91);
lean_dec(x_88);
x_92 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_92, 0, x_91);
lean_inc_ref(x_92);
x_34 = x_92;
x_35 = x_92;
goto block_82;
}
}
}
block_33:
{
lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; 
x_25 = l_Array_append___redArg(x_11, x_24);
lean_dec_ref(x_24);
lean_inc(x_23);
lean_inc(x_15);
x_26 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_26, 0, x_15);
lean_ctor_set(x_26, 1, x_23);
lean_ctor_set(x_26, 2, x_25);
lean_inc(x_15);
x_27 = l_Lean_Syntax_node5(x_15, x_13, x_18, x_14, x_19, x_22, x_26);
lean_inc(x_15);
x_28 = l_Lean_Syntax_node1(x_15, x_23, x_27);
lean_inc(x_15);
x_29 = l_Lean_Syntax_node1(x_15, x_21, x_28);
lean_inc(x_15);
x_30 = l_Lean_Syntax_node1(x_15, x_16, x_29);
x_31 = l_Lean_Syntax_node2(x_15, x_17, x_12, x_30);
x_32 = l_Lean_Elab_Tactic_evalTactic(x_31, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_32;
}
block_82:
{
lean_object* x_36; lean_object* x_37; lean_object* x_38; 
x_36 = l_Lean_mkOptionalNode(x_35);
x_37 = l_Lean_Elab_Tactic_expandOptLocation(x_36);
lean_dec(x_36);
lean_inc(x_9);
lean_inc_ref(x_8);
lean_inc(x_7);
lean_inc_ref(x_6);
lean_inc(x_5);
lean_inc_ref(x_4);
lean_inc(x_3);
lean_inc_ref(x_2);
x_38 = lp_mathlib_cancelDenominators(x_37, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_37);
if (lean_obj_tag(x_38) == 0)
{
lean_object* x_39; lean_object* x_40; lean_object* x_41; uint8_t x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; 
lean_dec_ref(x_38);
x_39 = lean_ctor_get(x_8, 5);
x_40 = lean_ctor_get(x_8, 10);
x_41 = lean_ctor_get(x_8, 11);
x_42 = 0;
x_43 = l_Lean_SourceInfo_fromRef(x_39, x_42);
x_44 = lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__1;
x_45 = lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__2;
lean_inc(x_43);
x_46 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_46, 0, x_43);
lean_ctor_set(x_46, 1, x_45);
x_47 = lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__4;
x_48 = lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__6;
x_49 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__8;
x_50 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__1;
x_51 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__2;
lean_inc(x_43);
x_52 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_52, 0, x_43);
lean_ctor_set(x_52, 1, x_51);
x_53 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__6;
x_54 = lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__9;
lean_inc(x_43);
x_55 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_55, 0, x_43);
lean_ctor_set(x_55, 1, x_49);
lean_ctor_set(x_55, 2, x_54);
lean_inc_ref(x_55);
lean_inc(x_43);
x_56 = l_Lean_Syntax_node1(x_43, x_53, x_55);
x_57 = lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__8;
x_58 = lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__9;
lean_inc(x_43);
x_59 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_59, 0, x_43);
lean_ctor_set(x_59, 1, x_58);
x_60 = lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__11;
x_61 = lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__13;
x_62 = lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__16;
x_63 = lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__17;
lean_inc(x_43);
x_64 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_64, 0, x_43);
lean_ctor_set(x_64, 1, x_63);
lean_inc(x_43);
x_65 = l_Lean_Syntax_node1(x_43, x_62, x_64);
lean_inc(x_43);
x_66 = l_Lean_Syntax_node1(x_43, x_61, x_65);
lean_inc(x_43);
x_67 = l_Lean_Syntax_node1(x_43, x_49, x_66);
x_68 = lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__19;
x_69 = lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__20;
lean_inc(x_41);
lean_inc(x_40);
x_70 = l_Lean_addMacroScope(x_40, x_69, x_41);
x_71 = lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__22;
lean_inc(x_43);
x_72 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_72, 0, x_43);
lean_ctor_set(x_72, 1, x_68);
lean_ctor_set(x_72, 2, x_70);
lean_ctor_set(x_72, 3, x_71);
lean_inc_ref(x_55);
lean_inc(x_43);
x_73 = l_Lean_Syntax_node3(x_43, x_60, x_55, x_67, x_72);
lean_inc(x_43);
x_74 = l_Lean_Syntax_node1(x_43, x_49, x_73);
x_75 = lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__23;
lean_inc(x_43);
x_76 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_76, 0, x_43);
lean_ctor_set(x_76, 1, x_75);
lean_inc(x_43);
x_77 = l_Lean_Syntax_node3(x_43, x_57, x_59, x_74, x_76);
lean_inc(x_43);
x_78 = l_Lean_Syntax_node1(x_43, x_49, x_77);
if (lean_obj_tag(x_34) == 1)
{
lean_object* x_79; lean_object* x_80; 
x_79 = lean_ctor_get(x_34, 0);
lean_inc(x_79);
lean_dec_ref(x_34);
x_80 = l_Array_mkArray1___redArg(x_79);
x_11 = x_54;
x_12 = x_46;
x_13 = x_50;
x_14 = x_56;
x_15 = x_43;
x_16 = x_47;
x_17 = x_44;
x_18 = x_52;
x_19 = x_55;
x_20 = lean_box(0);
x_21 = x_48;
x_22 = x_78;
x_23 = x_49;
x_24 = x_80;
goto block_33;
}
else
{
lean_object* x_81; 
lean_dec(x_34);
x_81 = lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__24;
x_11 = x_54;
x_12 = x_46;
x_13 = x_50;
x_14 = x_56;
x_15 = x_43;
x_16 = x_47;
x_17 = x_44;
x_18 = x_52;
x_19 = x_55;
x_20 = lean_box(0);
x_21 = x_48;
x_22 = x_78;
x_23 = x_49;
x_24 = x_81;
goto block_33;
}
}
else
{
lean_dec(x_34);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_38;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1_spec__0___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1_spec__0___redArg();
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_11;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Tree_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_Core(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_SynthesizeUsing(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_Qq(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_CancelDenoms_Core(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Tree_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NormNum_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_SynthesizeUsing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = _init_lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_();
lean_mark_persistent(lp_mathlib_initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_);
lp_mathlib_initFn___closed__1_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = _init_lp_mathlib_initFn___closed__1_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_();
lean_mark_persistent(lp_mathlib_initFn___closed__1_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_);
lp_mathlib_initFn___closed__2_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = _init_lp_mathlib_initFn___closed__2_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_();
lean_mark_persistent(lp_mathlib_initFn___closed__2_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_);
lp_mathlib_initFn___closed__3_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = _init_lp_mathlib_initFn___closed__3_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_();
lean_mark_persistent(lp_mathlib_initFn___closed__3_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_);
lp_mathlib_initFn___closed__4_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = _init_lp_mathlib_initFn___closed__4_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_();
lean_mark_persistent(lp_mathlib_initFn___closed__4_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_);
lp_mathlib_initFn___closed__5_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = _init_lp_mathlib_initFn___closed__5_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_();
lean_mark_persistent(lp_mathlib_initFn___closed__5_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_);
lp_mathlib_initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = _init_lp_mathlib_initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_();
lean_mark_persistent(lp_mathlib_initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_);
lp_mathlib_initFn___closed__7_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = _init_lp_mathlib_initFn___closed__7_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_();
lean_mark_persistent(lp_mathlib_initFn___closed__7_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_);
lp_mathlib_initFn___closed__8_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = _init_lp_mathlib_initFn___closed__8_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_();
lean_mark_persistent(lp_mathlib_initFn___closed__8_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_);
lp_mathlib_initFn___closed__9_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = _init_lp_mathlib_initFn___closed__9_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_();
lean_mark_persistent(lp_mathlib_initFn___closed__9_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_);
lp_mathlib_initFn___closed__10_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = _init_lp_mathlib_initFn___closed__10_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_();
lean_mark_persistent(lp_mathlib_initFn___closed__10_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_);
lp_mathlib_initFn___closed__11_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = _init_lp_mathlib_initFn___closed__11_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_();
lean_mark_persistent(lp_mathlib_initFn___closed__11_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_);
lp_mathlib_initFn___closed__12_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = _init_lp_mathlib_initFn___closed__12_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_();
lean_mark_persistent(lp_mathlib_initFn___closed__12_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_);
lp_mathlib_initFn___closed__13_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = _init_lp_mathlib_initFn___closed__13_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_();
lean_mark_persistent(lp_mathlib_initFn___closed__13_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_);
lp_mathlib_initFn___closed__14_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = _init_lp_mathlib_initFn___closed__14_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_();
lean_mark_persistent(lp_mathlib_initFn___closed__14_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_);
lp_mathlib_initFn___closed__15_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = _init_lp_mathlib_initFn___closed__15_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_();
lean_mark_persistent(lp_mathlib_initFn___closed__15_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_);
lp_mathlib_initFn___closed__16_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = _init_lp_mathlib_initFn___closed__16_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_();
lean_mark_persistent(lp_mathlib_initFn___closed__16_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_);
lp_mathlib_initFn___closed__17_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = _init_lp_mathlib_initFn___closed__17_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_();
lean_mark_persistent(lp_mathlib_initFn___closed__17_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_);
lp_mathlib_initFn___closed__18_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = _init_lp_mathlib_initFn___closed__18_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_();
lean_mark_persistent(lp_mathlib_initFn___closed__18_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_);
res = lp_mathlib_initFn_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_CancelDenoms_findCancelFactor___closed__0 = _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__0();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCancelFactor___closed__0);
lp_mathlib_CancelDenoms_findCancelFactor___closed__1 = _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__1();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCancelFactor___closed__1);
lp_mathlib_CancelDenoms_findCancelFactor___closed__2 = _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__2();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCancelFactor___closed__2);
lp_mathlib_CancelDenoms_findCancelFactor___closed__3 = _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__3();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCancelFactor___closed__3);
lp_mathlib_CancelDenoms_findCancelFactor___closed__4 = _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__4();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCancelFactor___closed__4);
lp_mathlib_CancelDenoms_findCancelFactor___closed__5 = _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__5();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCancelFactor___closed__5);
lp_mathlib_CancelDenoms_findCancelFactor___closed__6 = _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__6();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCancelFactor___closed__6);
lp_mathlib_CancelDenoms_findCancelFactor___closed__7 = _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__7();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCancelFactor___closed__7);
lp_mathlib_CancelDenoms_findCancelFactor___closed__8 = _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__8();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCancelFactor___closed__8);
lp_mathlib_CancelDenoms_findCancelFactor___closed__9 = _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__9();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCancelFactor___closed__9);
lp_mathlib_CancelDenoms_findCancelFactor___closed__10 = _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__10();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCancelFactor___closed__10);
lp_mathlib_CancelDenoms_findCancelFactor___closed__11 = _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__11();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCancelFactor___closed__11);
lp_mathlib_CancelDenoms_findCancelFactor___closed__12 = _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__12();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCancelFactor___closed__12);
lp_mathlib_CancelDenoms_findCancelFactor___closed__13 = _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__13();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCancelFactor___closed__13);
lp_mathlib_CancelDenoms_findCancelFactor___closed__14 = _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__14();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCancelFactor___closed__14);
lp_mathlib_CancelDenoms_findCancelFactor___closed__15 = _init_lp_mathlib_CancelDenoms_findCancelFactor___closed__15();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCancelFactor___closed__15);
lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__0 = _init_lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__0();
lean_mark_persistent(lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__0);
lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__1 = _init_lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__1();
lean_mark_persistent(lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__1);
lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__2 = _init_lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__2();
lean_mark_persistent(lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__2);
lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__3 = _init_lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__3();
lean_mark_persistent(lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__3);
lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__4 = _init_lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__4();
lean_mark_persistent(lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__4);
lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__5 = _init_lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__5();
lean_mark_persistent(lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__5);
lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__6 = _init_lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__6();
lean_mark_persistent(lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__6);
lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__7 = _init_lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__7();
lean_mark_persistent(lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__7);
lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__8 = _init_lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__8();
lean_mark_persistent(lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__8);
lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__9 = _init_lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__9();
lean_mark_persistent(lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__9);
lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__10 = _init_lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__10();
lean_mark_persistent(lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__10);
lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__11 = _init_lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__11();
lean_mark_persistent(lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__11);
lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__12 = _init_lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__12();
lean_mark_persistent(lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__12);
lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__13 = _init_lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__13();
lean_mark_persistent(lp_mathlib_CancelDenoms_synthesizeUsingNormNum___closed__13);
lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3___closed__0 = _init_lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3___closed__0();
lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3___closed__1 = _init_lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3___closed__1();
lean_mark_persistent(lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3___closed__1);
lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3___closed__2 = _init_lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3___closed__2();
lean_mark_persistent(lp_mathlib_Lean_addTrace___at___00CancelDenoms_mkProdPrf_spec__3___closed__2);
lp_mathlib_CancelDenoms_mkProdPrf___lam__4___closed__0 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__4___closed__0();
lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__0 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__0();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__0);
lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__1 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__1();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__1);
lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__2 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__2();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__2);
lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__3 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__3();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__6___closed__3);
lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__0 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__0();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__0);
lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__1 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__1();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__1);
lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__2 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__2();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__2);
lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__3 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__3();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__3);
lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__4 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__4();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__4);
lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__5 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__5();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__5);
lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__6 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__6();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__3___closed__6);
lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__0 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__0();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__0);
lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__1 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__1();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__1);
lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__2 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__2();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__2);
lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__3 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__3();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__3);
lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__4 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__4();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__4);
lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__5 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__5();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__5);
lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__6 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__6();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__6);
lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__7 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__7();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__7);
lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__8 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__8();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__8);
lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__9 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__9();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__9);
lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__10 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__10();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__10);
lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__11 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__11();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__11);
lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__12 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__12();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__12);
lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__13 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__13();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__13);
lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__14 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__14();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__14);
lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__15 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__15();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__15);
lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__16 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__16();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__2___closed__16);
lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__0 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__0();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__0);
lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__1 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__1();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__1);
lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__2 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__2();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__2);
lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__3 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__3();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__3);
lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__4 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__4();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__4);
lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__5 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__5();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__5);
lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__6 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__6();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__6);
lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__7 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__7();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__7);
lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__8 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__8();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__8);
lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__9 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__9();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__5___closed__9);
lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__0 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__0();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__0);
lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__1 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__1();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__1);
lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__2 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__2();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__2);
lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__3 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__3();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__3);
lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__4 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__4();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__4);
lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__5 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__5();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__5);
lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__6 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__6();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__6);
lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__7 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__7();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__7);
lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__8 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__8();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__8);
lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__9 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__9();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__9);
lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__10 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__10();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__10);
lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__11 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__11();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__11);
lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__12 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__12();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__12);
lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__13 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__13();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__13);
lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__14 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__14();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__14);
lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__15 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__15();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__15);
lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__16 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__16();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__16);
lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__17 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__17();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__17);
lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__18 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__18();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__18);
lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__19 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__19();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__0___closed__19);
lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__0 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__0();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__0);
lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__1 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__1();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__1);
lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__2 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__2();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__2);
lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__3 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__3();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__3);
lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__4 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__4();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__4);
lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__5 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__5();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__5);
lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__6 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__6();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__6);
lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__7 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__7();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__7);
lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__8 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__8();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__8);
lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__9 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__9();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__9);
lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__10 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__10();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__10);
lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__11 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__11();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__11);
lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__12 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__12();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__12);
lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__13 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__13();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__13);
lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__14 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__14();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__14);
lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__15 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__15();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__15);
lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__16 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__16();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__16);
lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__17 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__17();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__17);
lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__18 = _init_lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__18();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___lam__1___closed__18);
lp_mathlib_CancelDenoms_mkProdPrf___closed__0 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__0();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__0);
lp_mathlib_CancelDenoms_mkProdPrf___closed__1 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__1();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__1);
lp_mathlib_CancelDenoms_mkProdPrf___closed__2 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__2();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__2);
lp_mathlib_CancelDenoms_mkProdPrf___closed__3 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__3();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__3);
lp_mathlib_CancelDenoms_mkProdPrf___closed__4 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__4();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__4);
lp_mathlib_CancelDenoms_mkProdPrf___closed__5 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__5();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__5);
lp_mathlib_CancelDenoms_mkProdPrf___closed__6 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__6();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__6);
lp_mathlib_CancelDenoms_mkProdPrf___closed__7 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__7();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__7);
lp_mathlib_CancelDenoms_mkProdPrf___closed__8 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__8();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__8);
lp_mathlib_CancelDenoms_mkProdPrf___closed__9 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__9();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__9);
lp_mathlib_CancelDenoms_mkProdPrf___closed__10 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__10();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__10);
lp_mathlib_CancelDenoms_mkProdPrf___closed__11 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__11();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__11);
lp_mathlib_CancelDenoms_mkProdPrf___closed__12 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__12();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__12);
lp_mathlib_CancelDenoms_mkProdPrf___closed__13 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__13();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__13);
lp_mathlib_CancelDenoms_mkProdPrf___closed__14 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__14();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__14);
lp_mathlib_CancelDenoms_mkProdPrf___closed__15 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__15();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__15);
lp_mathlib_CancelDenoms_mkProdPrf___closed__16 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__16();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__16);
lp_mathlib_CancelDenoms_mkProdPrf___closed__17 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__17();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__17);
lp_mathlib_CancelDenoms_mkProdPrf___closed__18 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__18();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__18);
lp_mathlib_CancelDenoms_mkProdPrf___closed__19 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__19();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__19);
lp_mathlib_CancelDenoms_mkProdPrf___closed__20 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__20();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__20);
lp_mathlib_CancelDenoms_mkProdPrf___closed__21 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__21();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__21);
lp_mathlib_CancelDenoms_mkProdPrf___closed__22 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__22();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__22);
lp_mathlib_CancelDenoms_mkProdPrf___closed__23 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__23();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__23);
lp_mathlib_CancelDenoms_mkProdPrf___closed__24 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__24();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__24);
lp_mathlib_CancelDenoms_mkProdPrf___closed__25 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__25();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__25);
lp_mathlib_CancelDenoms_mkProdPrf___closed__26 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__26();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__26);
lp_mathlib_CancelDenoms_mkProdPrf___closed__27 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__27();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__27);
lp_mathlib_CancelDenoms_mkProdPrf___closed__28 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__28();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__28);
lp_mathlib_CancelDenoms_mkProdPrf___closed__29 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__29();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__29);
lp_mathlib_CancelDenoms_mkProdPrf___closed__30 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__30();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__30);
lp_mathlib_CancelDenoms_mkProdPrf___closed__31 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__31();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__31);
lp_mathlib_CancelDenoms_mkProdPrf___closed__32 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__32();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__32);
lp_mathlib_CancelDenoms_mkProdPrf___closed__33 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__33();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__33);
lp_mathlib_CancelDenoms_mkProdPrf___closed__34 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__34();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__34);
lp_mathlib_CancelDenoms_mkProdPrf___closed__35 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__35();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__35);
lp_mathlib_CancelDenoms_mkProdPrf___closed__36 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__36();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__36);
lp_mathlib_CancelDenoms_mkProdPrf___closed__37 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__37();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__37);
lp_mathlib_CancelDenoms_mkProdPrf___closed__38 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__38();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__38);
lp_mathlib_CancelDenoms_mkProdPrf___closed__39 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__39();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__39);
lp_mathlib_CancelDenoms_mkProdPrf___closed__40 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__40();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__40);
lp_mathlib_CancelDenoms_mkProdPrf___closed__41 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__41();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__41);
lp_mathlib_CancelDenoms_mkProdPrf___closed__42 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__42();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__42);
lp_mathlib_CancelDenoms_mkProdPrf___closed__43 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__43();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__43);
lp_mathlib_CancelDenoms_mkProdPrf___closed__44 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__44();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__44);
lp_mathlib_CancelDenoms_mkProdPrf___closed__45 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__45();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__45);
lp_mathlib_CancelDenoms_mkProdPrf___closed__46 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__46();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__46);
lp_mathlib_CancelDenoms_mkProdPrf___closed__47 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__47();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__47);
lp_mathlib_CancelDenoms_mkProdPrf___closed__48 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__48();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__48);
lp_mathlib_CancelDenoms_mkProdPrf___closed__49 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__49();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__49);
lp_mathlib_CancelDenoms_mkProdPrf___closed__50 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__50();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__50);
lp_mathlib_CancelDenoms_mkProdPrf___closed__51 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__51();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__51);
lp_mathlib_CancelDenoms_mkProdPrf___closed__52 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__52();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__52);
lp_mathlib_CancelDenoms_mkProdPrf___closed__53 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__53();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__53);
lp_mathlib_CancelDenoms_mkProdPrf___closed__54 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__54();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__54);
lp_mathlib_CancelDenoms_mkProdPrf___closed__55 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__55();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__55);
lp_mathlib_CancelDenoms_mkProdPrf___closed__56 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__56();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__56);
lp_mathlib_CancelDenoms_mkProdPrf___closed__57 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__57();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__57);
lp_mathlib_CancelDenoms_mkProdPrf___closed__58 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__58();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__58);
lp_mathlib_CancelDenoms_mkProdPrf___closed__59 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__59();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__59);
lp_mathlib_CancelDenoms_mkProdPrf___closed__60 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__60();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__60);
lp_mathlib_CancelDenoms_mkProdPrf___closed__61 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__61();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__61);
lp_mathlib_CancelDenoms_mkProdPrf___closed__62 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__62();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__62);
lp_mathlib_CancelDenoms_mkProdPrf___closed__63 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__63();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__63);
lp_mathlib_CancelDenoms_mkProdPrf___closed__64 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__64();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__64);
lp_mathlib_CancelDenoms_mkProdPrf___closed__65 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__65();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__65);
lp_mathlib_CancelDenoms_mkProdPrf___closed__66 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__66();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__66);
lp_mathlib_CancelDenoms_mkProdPrf___closed__67 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__67();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__67);
lp_mathlib_CancelDenoms_mkProdPrf___closed__68 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__68();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__68);
lp_mathlib_CancelDenoms_mkProdPrf___closed__69 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__69();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__69);
lp_mathlib_CancelDenoms_mkProdPrf___closed__70 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__70();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__70);
lp_mathlib_CancelDenoms_mkProdPrf___closed__71 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__71();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__71);
lp_mathlib_CancelDenoms_mkProdPrf___closed__72 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__72();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__72);
lp_mathlib_CancelDenoms_mkProdPrf___closed__73 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__73();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__73);
lp_mathlib_CancelDenoms_mkProdPrf___closed__74 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__74();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__74);
lp_mathlib_CancelDenoms_mkProdPrf___closed__75 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__75();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__75);
lp_mathlib_CancelDenoms_mkProdPrf___closed__76 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__76();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__76);
lp_mathlib_CancelDenoms_mkProdPrf___closed__77 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__77();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__77);
lp_mathlib_CancelDenoms_mkProdPrf___closed__78 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__78();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__78);
lp_mathlib_CancelDenoms_mkProdPrf___closed__79 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__79();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__79);
lp_mathlib_CancelDenoms_mkProdPrf___closed__80 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__80();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__80);
lp_mathlib_CancelDenoms_mkProdPrf___closed__81 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__81();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__81);
lp_mathlib_CancelDenoms_mkProdPrf___closed__82 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__82();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__82);
lp_mathlib_CancelDenoms_mkProdPrf___closed__83 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__83();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__83);
lp_mathlib_CancelDenoms_mkProdPrf___closed__84 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__84();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__84);
lp_mathlib_CancelDenoms_mkProdPrf___closed__85 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__85();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__85);
lp_mathlib_CancelDenoms_mkProdPrf___closed__86 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__86();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__86);
lp_mathlib_CancelDenoms_mkProdPrf___closed__87 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__87();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__87);
lp_mathlib_CancelDenoms_mkProdPrf___closed__88 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__88();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__88);
lp_mathlib_CancelDenoms_mkProdPrf___closed__89 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__89();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__89);
lp_mathlib_CancelDenoms_mkProdPrf___closed__90 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__90();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__90);
lp_mathlib_CancelDenoms_mkProdPrf___closed__91 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__91();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__91);
lp_mathlib_CancelDenoms_mkProdPrf___closed__92 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__92();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__92);
lp_mathlib_CancelDenoms_mkProdPrf___closed__93 = _init_lp_mathlib_CancelDenoms_mkProdPrf___closed__93();
lean_mark_persistent(lp_mathlib_CancelDenoms_mkProdPrf___closed__93);
lp_mathlib_CancelDenoms_deriveThms___closed__0 = _init_lp_mathlib_CancelDenoms_deriveThms___closed__0();
lean_mark_persistent(lp_mathlib_CancelDenoms_deriveThms___closed__0);
lp_mathlib_CancelDenoms_deriveThms___closed__1 = _init_lp_mathlib_CancelDenoms_deriveThms___closed__1();
lean_mark_persistent(lp_mathlib_CancelDenoms_deriveThms___closed__1);
lp_mathlib_CancelDenoms_deriveThms___closed__2 = _init_lp_mathlib_CancelDenoms_deriveThms___closed__2();
lean_mark_persistent(lp_mathlib_CancelDenoms_deriveThms___closed__2);
lp_mathlib_CancelDenoms_deriveThms___closed__3 = _init_lp_mathlib_CancelDenoms_deriveThms___closed__3();
lean_mark_persistent(lp_mathlib_CancelDenoms_deriveThms___closed__3);
lp_mathlib_CancelDenoms_deriveThms___closed__4 = _init_lp_mathlib_CancelDenoms_deriveThms___closed__4();
lean_mark_persistent(lp_mathlib_CancelDenoms_deriveThms___closed__4);
lp_mathlib_CancelDenoms_deriveThms___closed__5 = _init_lp_mathlib_CancelDenoms_deriveThms___closed__5();
lean_mark_persistent(lp_mathlib_CancelDenoms_deriveThms___closed__5);
lp_mathlib_CancelDenoms_deriveThms = _init_lp_mathlib_CancelDenoms_deriveThms();
lean_mark_persistent(lp_mathlib_CancelDenoms_deriveThms);
lp_mathlib_CancelDenoms_derive___lam__0___closed__0 = _init_lp_mathlib_CancelDenoms_derive___lam__0___closed__0();
lean_mark_persistent(lp_mathlib_CancelDenoms_derive___lam__0___closed__0);
lp_mathlib_CancelDenoms_derive___lam__0___closed__1 = _init_lp_mathlib_CancelDenoms_derive___lam__0___closed__1();
lean_mark_persistent(lp_mathlib_CancelDenoms_derive___lam__0___closed__1);
lp_mathlib_CancelDenoms_derive___lam__0___closed__2 = _init_lp_mathlib_CancelDenoms_derive___lam__0___closed__2();
lean_mark_persistent(lp_mathlib_CancelDenoms_derive___lam__0___closed__2);
lp_mathlib_CancelDenoms_derive___closed__0 = _init_lp_mathlib_CancelDenoms_derive___closed__0();
lean_mark_persistent(lp_mathlib_CancelDenoms_derive___closed__0);
lp_mathlib_CancelDenoms_derive___closed__1 = _init_lp_mathlib_CancelDenoms_derive___closed__1();
lean_mark_persistent(lp_mathlib_CancelDenoms_derive___closed__1);
lp_mathlib_CancelDenoms_derive___closed__2 = _init_lp_mathlib_CancelDenoms_derive___closed__2();
lean_mark_persistent(lp_mathlib_CancelDenoms_derive___closed__2);
lp_mathlib_CancelDenoms_derive___closed__3 = _init_lp_mathlib_CancelDenoms_derive___closed__3();
lean_mark_persistent(lp_mathlib_CancelDenoms_derive___closed__3);
lp_mathlib_CancelDenoms_derive___closed__4 = _init_lp_mathlib_CancelDenoms_derive___closed__4();
lean_mark_persistent(lp_mathlib_CancelDenoms_derive___closed__4);
lp_mathlib_CancelDenoms_derive___closed__5 = _init_lp_mathlib_CancelDenoms_derive___closed__5();
lean_mark_persistent(lp_mathlib_CancelDenoms_derive___closed__5);
lp_mathlib_CancelDenoms_derive___closed__6 = _init_lp_mathlib_CancelDenoms_derive___closed__6();
lean_mark_persistent(lp_mathlib_CancelDenoms_derive___closed__6);
lp_mathlib_CancelDenoms_derive___closed__7 = _init_lp_mathlib_CancelDenoms_derive___closed__7();
lean_mark_persistent(lp_mathlib_CancelDenoms_derive___closed__7);
lp_mathlib_CancelDenoms_derive___closed__8 = _init_lp_mathlib_CancelDenoms_derive___closed__8();
lean_mark_persistent(lp_mathlib_CancelDenoms_derive___closed__8);
lp_mathlib_CancelDenoms_derive___closed__9 = _init_lp_mathlib_CancelDenoms_derive___closed__9();
lean_mark_persistent(lp_mathlib_CancelDenoms_derive___closed__9);
lp_mathlib_CancelDenoms_derive___closed__10 = _init_lp_mathlib_CancelDenoms_derive___closed__10();
lean_mark_persistent(lp_mathlib_CancelDenoms_derive___closed__10);
lp_mathlib_CancelDenoms_derive___closed__11 = _init_lp_mathlib_CancelDenoms_derive___closed__11();
lean_mark_persistent(lp_mathlib_CancelDenoms_derive___closed__11);
lp_mathlib_CancelDenoms_derive___closed__12 = _init_lp_mathlib_CancelDenoms_derive___closed__12();
lean_mark_persistent(lp_mathlib_CancelDenoms_derive___closed__12);
lp_mathlib_CancelDenoms_derive___closed__13 = _init_lp_mathlib_CancelDenoms_derive___closed__13();
lean_mark_persistent(lp_mathlib_CancelDenoms_derive___closed__13);
lp_mathlib_CancelDenoms_derive___closed__14 = _init_lp_mathlib_CancelDenoms_derive___closed__14();
lean_mark_persistent(lp_mathlib_CancelDenoms_derive___closed__14);
lp_mathlib_CancelDenoms_derive___closed__15 = _init_lp_mathlib_CancelDenoms_derive___closed__15();
lean_mark_persistent(lp_mathlib_CancelDenoms_derive___closed__15);
lp_mathlib_CancelDenoms_derive___closed__16 = _init_lp_mathlib_CancelDenoms_derive___closed__16();
lean_mark_persistent(lp_mathlib_CancelDenoms_derive___closed__16);
lp_mathlib_CancelDenoms_derive___closed__17 = _init_lp_mathlib_CancelDenoms_derive___closed__17();
lean_mark_persistent(lp_mathlib_CancelDenoms_derive___closed__17);
lp_mathlib_CancelDenoms_derive___closed__18 = _init_lp_mathlib_CancelDenoms_derive___closed__18();
lean_mark_persistent(lp_mathlib_CancelDenoms_derive___closed__18);
lp_mathlib_CancelDenoms_derive___closed__19 = _init_lp_mathlib_CancelDenoms_derive___closed__19();
lean_mark_persistent(lp_mathlib_CancelDenoms_derive___closed__19);
lp_mathlib_CancelDenoms_derive___closed__20 = _init_lp_mathlib_CancelDenoms_derive___closed__20();
lean_mark_persistent(lp_mathlib_CancelDenoms_derive___closed__20);
lp_mathlib_CancelDenoms_derive___closed__21 = _init_lp_mathlib_CancelDenoms_derive___closed__21();
lean_mark_persistent(lp_mathlib_CancelDenoms_derive___closed__21);
lp_mathlib_CancelDenoms_findCompLemma___closed__0 = _init_lp_mathlib_CancelDenoms_findCompLemma___closed__0();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCompLemma___closed__0);
lp_mathlib_CancelDenoms_findCompLemma___closed__1 = _init_lp_mathlib_CancelDenoms_findCompLemma___closed__1();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCompLemma___closed__1);
lp_mathlib_CancelDenoms_findCompLemma___closed__2 = _init_lp_mathlib_CancelDenoms_findCompLemma___closed__2();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCompLemma___closed__2);
lp_mathlib_CancelDenoms_findCompLemma___closed__3 = _init_lp_mathlib_CancelDenoms_findCompLemma___closed__3();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCompLemma___closed__3);
lp_mathlib_CancelDenoms_findCompLemma___closed__4 = _init_lp_mathlib_CancelDenoms_findCompLemma___closed__4();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCompLemma___closed__4);
lp_mathlib_CancelDenoms_findCompLemma___closed__5 = _init_lp_mathlib_CancelDenoms_findCompLemma___closed__5();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCompLemma___closed__5);
lp_mathlib_CancelDenoms_findCompLemma___closed__6 = _init_lp_mathlib_CancelDenoms_findCompLemma___closed__6();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCompLemma___closed__6);
lp_mathlib_CancelDenoms_findCompLemma___closed__7 = _init_lp_mathlib_CancelDenoms_findCompLemma___closed__7();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCompLemma___closed__7);
lp_mathlib_CancelDenoms_findCompLemma___closed__8 = _init_lp_mathlib_CancelDenoms_findCompLemma___closed__8();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCompLemma___closed__8);
lp_mathlib_CancelDenoms_findCompLemma___closed__9 = _init_lp_mathlib_CancelDenoms_findCompLemma___closed__9();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCompLemma___closed__9);
lp_mathlib_CancelDenoms_findCompLemma___closed__10 = _init_lp_mathlib_CancelDenoms_findCompLemma___closed__10();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCompLemma___closed__10);
lp_mathlib_CancelDenoms_findCompLemma___closed__11 = _init_lp_mathlib_CancelDenoms_findCompLemma___closed__11();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCompLemma___closed__11);
lp_mathlib_CancelDenoms_findCompLemma___closed__12 = _init_lp_mathlib_CancelDenoms_findCompLemma___closed__12();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCompLemma___closed__12);
lp_mathlib_CancelDenoms_findCompLemma___closed__13 = _init_lp_mathlib_CancelDenoms_findCompLemma___closed__13();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCompLemma___closed__13);
lp_mathlib_CancelDenoms_findCompLemma___closed__14 = _init_lp_mathlib_CancelDenoms_findCompLemma___closed__14();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCompLemma___closed__14);
lp_mathlib_CancelDenoms_findCompLemma___closed__15 = _init_lp_mathlib_CancelDenoms_findCompLemma___closed__15();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCompLemma___closed__15);
lp_mathlib_CancelDenoms_findCompLemma___closed__16 = _init_lp_mathlib_CancelDenoms_findCompLemma___closed__16();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCompLemma___closed__16);
lp_mathlib_CancelDenoms_findCompLemma___closed__17 = _init_lp_mathlib_CancelDenoms_findCompLemma___closed__17();
lean_mark_persistent(lp_mathlib_CancelDenoms_findCompLemma___closed__17);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__0 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__0();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__0);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__1 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__1();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__1);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__2 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__2();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__2);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__3 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__3();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__3);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__4 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__4();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__4);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__5 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__5();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__5);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__6 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__6();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__6);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__7 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__7();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__7);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__8 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__8();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__8);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__9 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__9();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__9);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__10 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__10();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__10);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__11 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__11();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__11);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__12 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__12();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__12);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__13 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__13();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__13);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__14 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__14();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__14);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__15 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__15();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__15);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__16 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__16();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__16);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__17 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__17();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__17);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__18 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__18();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__18);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__19 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__19();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__19);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__20 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__20();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__20);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__21 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__21();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__21);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__22 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__22();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__22);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__23 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__23();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__23);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__24 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__24();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__24);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__25 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__25();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__25);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__26 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__26();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__26);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__27 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__27();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__27);
lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__28 = _init_lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__28();
lean_mark_persistent(lp_mathlib_CancelDenoms_cancelDenominatorsInType___closed__28);
lp_mathlib_cancelDenoms___closed__0 = _init_lp_mathlib_cancelDenoms___closed__0();
lean_mark_persistent(lp_mathlib_cancelDenoms___closed__0);
lp_mathlib_cancelDenoms___closed__1 = _init_lp_mathlib_cancelDenoms___closed__1();
lean_mark_persistent(lp_mathlib_cancelDenoms___closed__1);
lp_mathlib_cancelDenoms___closed__2 = _init_lp_mathlib_cancelDenoms___closed__2();
lean_mark_persistent(lp_mathlib_cancelDenoms___closed__2);
lp_mathlib_cancelDenoms___closed__3 = _init_lp_mathlib_cancelDenoms___closed__3();
lean_mark_persistent(lp_mathlib_cancelDenoms___closed__3);
lp_mathlib_cancelDenoms___closed__4 = _init_lp_mathlib_cancelDenoms___closed__4();
lean_mark_persistent(lp_mathlib_cancelDenoms___closed__4);
lp_mathlib_cancelDenoms___closed__5 = _init_lp_mathlib_cancelDenoms___closed__5();
lean_mark_persistent(lp_mathlib_cancelDenoms___closed__5);
lp_mathlib_cancelDenoms___closed__6 = _init_lp_mathlib_cancelDenoms___closed__6();
lean_mark_persistent(lp_mathlib_cancelDenoms___closed__6);
lp_mathlib_cancelDenoms___closed__7 = _init_lp_mathlib_cancelDenoms___closed__7();
lean_mark_persistent(lp_mathlib_cancelDenoms___closed__7);
lp_mathlib_cancelDenoms___closed__8 = _init_lp_mathlib_cancelDenoms___closed__8();
lean_mark_persistent(lp_mathlib_cancelDenoms___closed__8);
lp_mathlib_cancelDenoms___closed__9 = _init_lp_mathlib_cancelDenoms___closed__9();
lean_mark_persistent(lp_mathlib_cancelDenoms___closed__9);
lp_mathlib_cancelDenoms___closed__10 = _init_lp_mathlib_cancelDenoms___closed__10();
lean_mark_persistent(lp_mathlib_cancelDenoms___closed__10);
lp_mathlib_cancelDenoms___closed__11 = _init_lp_mathlib_cancelDenoms___closed__11();
lean_mark_persistent(lp_mathlib_cancelDenoms___closed__11);
lp_mathlib_cancelDenoms = _init_lp_mathlib_cancelDenoms();
lean_mark_persistent(lp_mathlib_cancelDenoms);
lp_mathlib_cancelDenominators___lam__0___closed__0 = _init_lp_mathlib_cancelDenominators___lam__0___closed__0();
lean_mark_persistent(lp_mathlib_cancelDenominators___lam__0___closed__0);
lp_mathlib_cancelDenominators___lam__0___closed__1 = _init_lp_mathlib_cancelDenominators___lam__0___closed__1();
lean_mark_persistent(lp_mathlib_cancelDenominators___lam__0___closed__1);
lp_mathlib_cancelDenominators___closed__0 = _init_lp_mathlib_cancelDenominators___closed__0();
lean_mark_persistent(lp_mathlib_cancelDenominators___closed__0);
lp_mathlib_tacticCancel__denoms___00__closed__0 = _init_lp_mathlib_tacticCancel__denoms___00__closed__0();
lean_mark_persistent(lp_mathlib_tacticCancel__denoms___00__closed__0);
lp_mathlib_tacticCancel__denoms___00__closed__1 = _init_lp_mathlib_tacticCancel__denoms___00__closed__1();
lean_mark_persistent(lp_mathlib_tacticCancel__denoms___00__closed__1);
lp_mathlib_tacticCancel__denoms___00__closed__2 = _init_lp_mathlib_tacticCancel__denoms___00__closed__2();
lean_mark_persistent(lp_mathlib_tacticCancel__denoms___00__closed__2);
lp_mathlib_tacticCancel__denoms__ = _init_lp_mathlib_tacticCancel__denoms__();
lean_mark_persistent(lp_mathlib_tacticCancel__denoms__);
lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1_spec__0___redArg___closed__0 = _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1_spec__0___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1_spec__0___redArg___closed__0);
lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1_spec__0___redArg___closed__1 = _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1_spec__0___redArg___closed__1();
lean_mark_persistent(lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1_spec__0___redArg___closed__1);
lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__0 = _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__0();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__0);
lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__1 = _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__1();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__1);
lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__2 = _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__2();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__2);
lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__3 = _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__3();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__3);
lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__4 = _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__4();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__4);
lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__5 = _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__5();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__5);
lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__6 = _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__6();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__6);
lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__7 = _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__7();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__7);
lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__8 = _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__8();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__8);
lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__9 = _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__9();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__9);
lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__10 = _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__10();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__10);
lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__11 = _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__11();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__11);
lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__12 = _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__12();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__12);
lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__13 = _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__13();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__13);
lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__14 = _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__14();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__14);
lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__15 = _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__15();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__15);
lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__16 = _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__16();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__16);
lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__17 = _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__17();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__17);
lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__18 = _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__18();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__18);
lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__19 = _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__19();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__19);
lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__20 = _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__20();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__20);
lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__21 = _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__21();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__21);
lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__22 = _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__22();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__22);
lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__23 = _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__23();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__23);
lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__24 = _init_lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__24();
lean_mark_persistent(lp_mathlib___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__tacticCancel__denoms____1___closed__24);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
