// Lean compiler output
// Module: Aesop.RuleTac.RuleTerm
// Imports: public import Init public import Aesop.Rule.Name
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToMessageDataRuleTerm___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_toRuleTerm(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedRuleTerm_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_expr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_name___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedElabRuleTerm;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_const_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_const_elim___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_instToMessageData;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_const_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_ctorIdx(lean_object*);
lean_object* l_Lean_Expr_constName_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_name(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_term_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_scope___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_term_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_const_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_ctorElim___redArg(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_getRuleNameForExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToMessageDataRuleTerm;
LEAN_EXPORT uint8_t lp_aesop_Aesop_ElabRuleTerm_scope(lean_object*);
static lean_object* lp_aesop_Aesop_instInhabitedElabRuleTerm_default___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedElabRuleTerm_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_expr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedRuleTerm;
static lean_object* lp_aesop_Aesop_instInhabitedRuleTerm_default___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_term_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_ofElaboratedTerm(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkConstWithFreshMVarLevels(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_term_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_instToMessageData___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_ctorIdx(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_ctorIdx___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_RuleTerm_ctorIdx(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_ctorElim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_ctorElim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_aesop_Aesop_RuleTerm_ctorElim___redArg(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_ctorElim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_aesop_Aesop_RuleTerm_ctorElim(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_const_elim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop_Aesop_RuleTerm_ctorElim___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_const_elim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_RuleTerm_ctorElim___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_term_elim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop_Aesop_RuleTerm_ctorElim___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_term_elim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_RuleTerm_ctorElim___redArg(x_1, x_2);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedRuleTerm_default___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_box(0);
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedRuleTerm_default() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instInhabitedRuleTerm_default___closed__0;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedRuleTerm() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instInhabitedRuleTerm_default;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToMessageDataRuleTerm___lam__0(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
lean_dec_ref(x_1);
x_3 = l_Lean_MessageData_ofName(x_2);
return x_3;
}
else
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = l_Lean_MessageData_ofSyntax(x_4);
return x_5;
}
}
}
static lean_object* _init_lp_aesop_Aesop_instToMessageDataRuleTerm() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_instToMessageDataRuleTerm___lam__0), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_ctorIdx(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_ctorIdx___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_ElabRuleTerm_ctorIdx(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_ctorElim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_2, x_3);
return x_4;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_6);
lean_dec_ref(x_1);
x_7 = lean_apply_2(x_2, x_5, x_6);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_ctorElim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_aesop_Aesop_ElabRuleTerm_ctorElim___redArg(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_ctorElim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_aesop_Aesop_ElabRuleTerm_ctorElim(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_const_elim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop_Aesop_ElabRuleTerm_ctorElim___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_const_elim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_ElabRuleTerm_ctorElim___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_term_elim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_aesop_Aesop_ElabRuleTerm_ctorElim___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_term_elim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_ElabRuleTerm_ctorElim___redArg(x_1, x_2);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedElabRuleTerm_default___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_box(0);
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedElabRuleTerm_default() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instInhabitedElabRuleTerm_default___closed__0;
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedElabRuleTerm() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_instInhabitedElabRuleTerm_default;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_instToMessageData___lam__0(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
lean_dec_ref(x_1);
x_3 = l_Lean_MessageData_ofName(x_2);
return x_3;
}
else
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = l_Lean_MessageData_ofSyntax(x_4);
return x_5;
}
}
}
static lean_object* _init_lp_aesop_Aesop_ElabRuleTerm_instToMessageData() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_ElabRuleTerm_instToMessageData___lam__0), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_expr(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_ctor_get(x_1, 0);
lean_inc(x_7);
lean_dec_ref(x_1);
x_8 = l_Lean_Meta_mkConstWithFreshMVarLevels(x_7, x_2, x_3, x_4, x_5);
return x_8;
}
else
{
lean_object* x_9; lean_object* x_10; 
lean_dec_ref(x_4);
x_9 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_9);
lean_dec_ref(x_1);
x_10 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_10, 0, x_9);
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_expr___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_aesop_Aesop_ElabRuleTerm_expr(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_ElabRuleTerm_scope(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
uint8_t x_2; 
x_2 = 0;
return x_2;
}
else
{
uint8_t x_3; 
x_3 = 1;
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_scope___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_aesop_Aesop_ElabRuleTerm_scope(x_1);
lean_dec_ref(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_name(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
uint8_t x_7; 
lean_dec_ref(x_2);
x_7 = !lean_is_exclusive(x_1);
if (x_7 == 0)
{
return x_1;
}
else
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_1, 0);
lean_inc(x_8);
lean_dec(x_1);
x_9 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_9, 0, x_8);
return x_9;
}
}
else
{
lean_object* x_10; lean_object* x_11; 
x_10 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_10);
lean_dec_ref(x_1);
x_11 = lp_aesop_Aesop_getRuleNameForExpr(x_10, x_2, x_3, x_4, x_5);
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_name___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_aesop_Aesop_ElabRuleTerm_name(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_toRuleTerm(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
return x_1;
}
else
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec(x_1);
x_4 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_4, 0, x_3);
return x_4;
}
}
else
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_6, 0, x_5);
return x_6;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_ofElaboratedTerm(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = l_Lean_Expr_constName_x3f(x_2);
if (lean_obj_tag(x_3) == 1)
{
uint8_t x_4; 
lean_dec_ref(x_2);
lean_dec(x_1);
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_ctor_set_tag(x_3, 0);
return x_3;
}
else
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_ctor_get(x_3, 0);
lean_inc(x_5);
lean_dec(x_3);
x_6 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_6, 0, x_5);
return x_6;
}
}
else
{
lean_object* x_7; 
lean_dec(x_3);
x_7 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_7, 0, x_1);
lean_ctor_set(x_7, 1, x_2);
return x_7;
}
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Rule_Name(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_RuleTac_RuleTerm(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Rule_Name(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instInhabitedRuleTerm_default___closed__0 = _init_lp_aesop_Aesop_instInhabitedRuleTerm_default___closed__0();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedRuleTerm_default___closed__0);
lp_aesop_Aesop_instInhabitedRuleTerm_default = _init_lp_aesop_Aesop_instInhabitedRuleTerm_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedRuleTerm_default);
lp_aesop_Aesop_instInhabitedRuleTerm = _init_lp_aesop_Aesop_instInhabitedRuleTerm();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedRuleTerm);
lp_aesop_Aesop_instToMessageDataRuleTerm = _init_lp_aesop_Aesop_instToMessageDataRuleTerm();
lean_mark_persistent(lp_aesop_Aesop_instToMessageDataRuleTerm);
lp_aesop_Aesop_instInhabitedElabRuleTerm_default___closed__0 = _init_lp_aesop_Aesop_instInhabitedElabRuleTerm_default___closed__0();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedElabRuleTerm_default___closed__0);
lp_aesop_Aesop_instInhabitedElabRuleTerm_default = _init_lp_aesop_Aesop_instInhabitedElabRuleTerm_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedElabRuleTerm_default);
lp_aesop_Aesop_instInhabitedElabRuleTerm = _init_lp_aesop_Aesop_instInhabitedElabRuleTerm();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedElabRuleTerm);
lp_aesop_Aesop_ElabRuleTerm_instToMessageData = _init_lp_aesop_Aesop_ElabRuleTerm_instToMessageData();
lean_mark_persistent(lp_aesop_Aesop_ElabRuleTerm_instToMessageData);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
