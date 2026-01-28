// Lean compiler output
// Module: Mathlib.Tactic.Common
// Imports: public import Init public meta import Aesop public meta import Qq public meta import Plausible public meta import ImportGraph.Imports public meta import Batteries.Tactic.Basic public meta import Batteries.Tactic.Case public meta import Batteries.Tactic.HelpCmd public meta import Batteries.Tactic.Alias public meta import Batteries.Tactic.GeneralizeProofs public meta import LeanSearchClient public meta import Mathlib.Tactic.Linter.Lint public meta import Mathlib.Tactic.ApplyCongr public meta import Mathlib.Tactic.ApplyAt public meta import Mathlib.Tactic.ApplyWith public meta import Mathlib.Tactic.Basic public meta import Mathlib.Tactic.ByCases public meta import Mathlib.Tactic.ByContra public meta import Mathlib.Tactic.CasesM public meta import Mathlib.Tactic.Check public meta import Mathlib.Tactic.Choose public meta import Mathlib.Tactic.ClearExclamation public meta import Mathlib.Tactic.ClearExcept public meta import Mathlib.Tactic.Clear_ public meta import Mathlib.Tactic.Coe public meta import Mathlib.Tactic.CongrExclamation public meta import Mathlib.Tactic.CongrM public meta import Mathlib.Tactic.Constructor public meta import Mathlib.Tactic.Contrapose public meta import Mathlib.Tactic.Conv public meta import Mathlib.Tactic.Convert public meta import Mathlib.Tactic.DefEqTransformations public meta import Mathlib.Tactic.DeprecateTo public meta import Mathlib.Tactic.ErwQuestion public meta import Mathlib.Tactic.Eqns public meta import Mathlib.Tactic.ExistsI public meta import Mathlib.Tactic.ExtractGoal public meta import Mathlib.Tactic.FailIfNoProgress public meta import Mathlib.Tactic.Find public meta import Mathlib.Tactic.FunProp public meta import Mathlib.Tactic.GCongr public meta import Mathlib.Tactic.GRewrite public meta import Mathlib.Tactic.GuardGoalNums public meta import Mathlib.Tactic.GuardHypNums public meta import Mathlib.Tactic.HigherOrder public meta import Mathlib.Tactic.Hint public meta import Mathlib.Tactic.InferParam public meta import Mathlib.Tactic.Inhabit public meta import Mathlib.Tactic.IrreducibleDef public meta import Mathlib.Tactic.Lift public meta import Mathlib.Tactic.Linter public meta import Mathlib.Tactic.MkIffOfInductiveProp public meta import Mathlib.Tactic.NthRewrite public meta import Mathlib.Tactic.Observe public meta import Mathlib.Tactic.OfNat public meta import Mathlib.Tactic.Push public meta import Mathlib.Tactic.RSuffices public meta import Mathlib.Tactic.Recover public meta import Mathlib.Tactic.Relation.Rfl public meta import Mathlib.Tactic.Rename public meta import Mathlib.Tactic.RenameBVar public meta import Mathlib.Tactic.Says public meta import Mathlib.Tactic.ScopedNS public meta import Mathlib.Tactic.Set public meta import Mathlib.Tactic.SimpIntro public meta import Mathlib.Tactic.SimpRw public meta import Mathlib.Tactic.Simps.Basic public meta import Mathlib.Tactic.SplitIfs public meta import Mathlib.Tactic.Spread public meta import Mathlib.Tactic.Subsingleton public meta import Mathlib.Tactic.Substs public meta import Mathlib.Tactic.SuccessIfFailWithMsg public meta import Mathlib.Tactic.SudoSetOption public meta import Mathlib.Tactic.SwapVar public meta import Mathlib.Tactic.Tauto public meta import Mathlib.Tactic.ToFun public meta import Mathlib.Tactic.TermCongr public meta import Mathlib.Tactic.ToExpr public meta import Mathlib.Tactic.ToLevel public meta import Mathlib.Tactic.Trace public meta import Mathlib.Tactic.TypeCheck public meta import Mathlib.Tactic.UnsetOption public meta import Mathlib.Tactic.Use public meta import Mathlib.Tactic.Variable public meta import Mathlib.Tactic.Widget.Calc public meta import Mathlib.Tactic.Widget.CongrM public meta import Mathlib.Tactic.Widget.Conv public meta import Mathlib.Tactic.Widget.LibraryRewrite public meta import Mathlib.Tactic.WLOG public meta import Mathlib.Util.AssertExists public meta import Mathlib.Util.CountHeartbeats public meta import Mathlib.Util.PrintSorries public meta import Mathlib.Util.TransImports public meta import Mathlib.Util.WhatsNew
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
static lean_object* lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__0;
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic17471413729771528806___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic3259365975255618868___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic17471413729771528806(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__4;
static lean_object* lp_mathlib___auxTryTactic3259365975255618868___redArg___closed__0;
lean_object* lean_st_ref_get(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic3259365975255618868___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic3259365975255618868___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic11400386961666083988___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___auxTryTactic17471413729771528806___redArg___closed__0;
lean_object* l_Lean_Parser_runParserCategory(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic11400386961666083988___redArg___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__3;
lean_object* l_Lean_Name_mkStr1(lean_object*);
static lean_object* lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic3259365975255618868(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic11400386961666083988___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic11400386961666083988(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic17471413729771528806___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic17471413729771528806___redArg___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__2;
static lean_object* _init_lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("tactic", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__0;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("tauto", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("<input>", 7, 7);
return x_1;
}
}
static lean_object* _init_lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic11400386961666083988___redArg(lean_object* x_1) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_3 = lean_st_ref_get(x_1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec(x_3);
x_5 = lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__1;
x_6 = lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__2;
x_7 = lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__3;
x_8 = l_Lean_Parser_runParserCategory(x_4, x_5, x_6, x_7);
if (lean_obj_tag(x_8) == 0)
{
uint8_t x_9; 
x_9 = !lean_is_exclusive(x_8);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; 
x_10 = lean_ctor_get(x_8, 0);
lean_dec(x_10);
x_11 = lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__4;
lean_ctor_set(x_8, 0, x_11);
return x_8;
}
else
{
lean_object* x_12; lean_object* x_13; 
lean_dec(x_8);
x_12 = lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__4;
x_13 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_13, 0, x_12);
return x_13;
}
}
else
{
uint8_t x_14; 
x_14 = !lean_is_exclusive(x_8);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_15 = lean_ctor_get(x_8, 0);
x_16 = lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__5;
x_17 = lean_array_push(x_16, x_15);
lean_ctor_set_tag(x_8, 0);
lean_ctor_set(x_8, 0, x_17);
return x_8;
}
else
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_18 = lean_ctor_get(x_8, 0);
lean_inc(x_18);
lean_dec(x_8);
x_19 = lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__5;
x_20 = lean_array_push(x_19, x_18);
x_21 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_21, 0, x_20);
return x_21;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic11400386961666083988(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib___auxTryTactic11400386961666083988___redArg(x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic11400386961666083988___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib___auxTryTactic11400386961666083988(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic11400386961666083988___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib___auxTryTactic11400386961666083988___redArg(x_1);
lean_dec(x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib___auxTryTactic17471413729771528806___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("aesop", 5, 5);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic17471413729771528806___redArg(lean_object* x_1) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_3 = lean_st_ref_get(x_1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec(x_3);
x_5 = lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__1;
x_6 = lp_mathlib___auxTryTactic17471413729771528806___redArg___closed__0;
x_7 = lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__3;
x_8 = l_Lean_Parser_runParserCategory(x_4, x_5, x_6, x_7);
if (lean_obj_tag(x_8) == 0)
{
uint8_t x_9; 
x_9 = !lean_is_exclusive(x_8);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; 
x_10 = lean_ctor_get(x_8, 0);
lean_dec(x_10);
x_11 = lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__4;
lean_ctor_set(x_8, 0, x_11);
return x_8;
}
else
{
lean_object* x_12; lean_object* x_13; 
lean_dec(x_8);
x_12 = lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__4;
x_13 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_13, 0, x_12);
return x_13;
}
}
else
{
uint8_t x_14; 
x_14 = !lean_is_exclusive(x_8);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_15 = lean_ctor_get(x_8, 0);
x_16 = lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__5;
x_17 = lean_array_push(x_16, x_15);
lean_ctor_set_tag(x_8, 0);
lean_ctor_set(x_8, 0, x_17);
return x_8;
}
else
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_18 = lean_ctor_get(x_8, 0);
lean_inc(x_18);
lean_dec(x_8);
x_19 = lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__5;
x_20 = lean_array_push(x_19, x_18);
x_21 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_21, 0, x_20);
return x_21;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic17471413729771528806(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib___auxTryTactic17471413729771528806___redArg(x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic17471413729771528806___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib___auxTryTactic17471413729771528806(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic17471413729771528806___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib___auxTryTactic17471413729771528806___redArg(x_1);
lean_dec(x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib___auxTryTactic3259365975255618868___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("fun_prop", 8, 8);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic3259365975255618868___redArg(lean_object* x_1) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_3 = lean_st_ref_get(x_1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec(x_3);
x_5 = lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__1;
x_6 = lp_mathlib___auxTryTactic3259365975255618868___redArg___closed__0;
x_7 = lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__3;
x_8 = l_Lean_Parser_runParserCategory(x_4, x_5, x_6, x_7);
if (lean_obj_tag(x_8) == 0)
{
uint8_t x_9; 
x_9 = !lean_is_exclusive(x_8);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; 
x_10 = lean_ctor_get(x_8, 0);
lean_dec(x_10);
x_11 = lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__4;
lean_ctor_set(x_8, 0, x_11);
return x_8;
}
else
{
lean_object* x_12; lean_object* x_13; 
lean_dec(x_8);
x_12 = lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__4;
x_13 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_13, 0, x_12);
return x_13;
}
}
else
{
uint8_t x_14; 
x_14 = !lean_is_exclusive(x_8);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_15 = lean_ctor_get(x_8, 0);
x_16 = lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__5;
x_17 = lean_array_push(x_16, x_15);
lean_ctor_set_tag(x_8, 0);
lean_ctor_set(x_8, 0, x_17);
return x_8;
}
else
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_18 = lean_ctor_get(x_8, 0);
lean_inc(x_18);
lean_dec(x_8);
x_19 = lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__5;
x_20 = lean_array_push(x_19, x_18);
x_21 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_21, 0, x_20);
return x_21;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic3259365975255618868(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib___auxTryTactic3259365975255618868___redArg(x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic3259365975255618868___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib___auxTryTactic3259365975255618868(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic3259365975255618868___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib___auxTryTactic3259365975255618868___redArg(x_1);
lean_dec(x_1);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_aesop_Aesop(uint8_t builtin);
lean_object* initialize_Qq_Qq(uint8_t builtin);
lean_object* initialize_plausible_Plausible(uint8_t builtin);
lean_object* initialize_importGraph_ImportGraph_Imports(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Basic(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Case(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_HelpCmd(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Alias(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_GeneralizeProofs(uint8_t builtin);
lean_object* initialize_LeanSearchClient_LeanSearchClient(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Lint(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ApplyCongr(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ApplyAt(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ApplyWith(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ByCases(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ByContra(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_CasesM(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Check(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Choose(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ClearExclamation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ClearExcept(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Clear__(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Coe(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_CongrExclamation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_CongrM(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Constructor(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Contrapose(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Conv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Convert(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_DefEqTransformations(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_DeprecateTo(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ErwQuestion(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Eqns(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ExistsI(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ExtractGoal(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FailIfNoProgress(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Find(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FunProp(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_GCongr(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_GRewrite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_GuardGoalNums(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_GuardHypNums(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_HigherOrder(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Hint(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_InferParam(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Inhabit(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_IrreducibleDef(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Lift(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NthRewrite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Observe(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_OfNat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Push(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_RSuffices(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Recover(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Relation_Rfl(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Rename(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_RenameBVar(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Says(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ScopedNS(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Set(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_SimpIntro(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_SimpRw(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Simps_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_SplitIfs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Subsingleton(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Substs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_SuccessIfFailWithMsg(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_SudoSetOption(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_SwapVar(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Tauto(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToFun(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_TermCongr(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToExpr(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToLevel(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Trace(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_TypeCheck(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_UnsetOption(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Use(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Variable(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Widget_Calc(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Widget_CongrM(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Widget_Conv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Widget_LibraryRewrite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_WLOG(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_AssertExists(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_CountHeartbeats(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_PrintSorries(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_TransImports(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_WhatsNew(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Qq_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_plausible_Plausible(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_importGraph_ImportGraph_Imports(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Case(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_HelpCmd(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Alias(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_GeneralizeProofs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_LeanSearchClient_LeanSearchClient(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_Lint(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ApplyCongr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ApplyAt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ApplyWith(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ByCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ByContra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_CasesM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Check(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Choose(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ClearExclamation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ClearExcept(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Clear__(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Coe(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_CongrExclamation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_CongrM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Constructor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Contrapose(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Conv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Convert(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_DefEqTransformations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_DeprecateTo(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ErwQuestion(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Eqns(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ExistsI(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ExtractGoal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FailIfNoProgress(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Find(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FunProp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_GCongr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_GRewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_GuardGoalNums(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_GuardHypNums(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_HigherOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Hint(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_InferParam(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Inhabit(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_IrreducibleDef(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Lift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NthRewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Observe(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_OfNat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Push(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_RSuffices(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Recover(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Relation_Rfl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Rename(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_RenameBVar(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Says(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ScopedNS(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_SimpIntro(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_SimpRw(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Simps_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_SplitIfs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Spread(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Subsingleton(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Substs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_SuccessIfFailWithMsg(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_SudoSetOption(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_SwapVar(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Tauto(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToFun(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_TermCongr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToExpr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToLevel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Trace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_TypeCheck(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_UnsetOption(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Use(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Variable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Widget_Calc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Widget_CongrM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Widget_Conv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Widget_LibraryRewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_WLOG(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_AssertExists(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_CountHeartbeats(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_PrintSorries(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_TransImports(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_WhatsNew(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__0 = _init_lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__0();
lean_mark_persistent(lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__0);
lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__1 = _init_lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__1();
lean_mark_persistent(lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__1);
lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__2 = _init_lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__2();
lean_mark_persistent(lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__2);
lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__3 = _init_lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__3();
lean_mark_persistent(lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__3);
lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__4 = _init_lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__4();
lean_mark_persistent(lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__4);
lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__5 = _init_lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__5();
lean_mark_persistent(lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__5);
lp_mathlib___auxTryTactic17471413729771528806___redArg___closed__0 = _init_lp_mathlib___auxTryTactic17471413729771528806___redArg___closed__0();
lean_mark_persistent(lp_mathlib___auxTryTactic17471413729771528806___redArg___closed__0);
lp_mathlib___auxTryTactic3259365975255618868___redArg___closed__0 = _init_lp_mathlib___auxTryTactic3259365975255618868___redArg___closed__0();
lean_mark_persistent(lp_mathlib___auxTryTactic3259365975255618868___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
