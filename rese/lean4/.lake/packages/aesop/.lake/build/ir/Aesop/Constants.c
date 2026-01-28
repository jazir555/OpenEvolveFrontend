// Lean compiler output
// Module: Aesop.Constants
// Imports: public import Init public import Aesop.Percent
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
LEAN_EXPORT double lp_aesop_Aesop_postponedSafeRuleSuccessProbability;
static double lp_aesop_Aesop_postponedSafeRuleSuccessProbability___closed__0;
double l_Float_ofScientific(lean_object*, uint8_t, lean_object*);
static double lp_aesop_Aesop_unificationGoalPenalty___closed__0;
LEAN_EXPORT double lp_aesop_Aesop_unificationGoalPenalty;
static double _init_lp_aesop_Aesop_unificationGoalPenalty___closed__0() {
_start:
{
lean_object* x_1; uint8_t x_2; lean_object* x_3; double x_4; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = 1;
x_3 = lean_unsigned_to_nat(8u);
x_4 = l_Float_ofScientific(x_3, x_2, x_1);
return x_4;
}
}
static double _init_lp_aesop_Aesop_unificationGoalPenalty() {
_start:
{
double x_1; 
x_1 = lp_aesop_Aesop_unificationGoalPenalty___closed__0;
return x_1;
}
}
static double _init_lp_aesop_Aesop_postponedSafeRuleSuccessProbability___closed__0() {
_start:
{
lean_object* x_1; uint8_t x_2; lean_object* x_3; double x_4; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = 1;
x_3 = lean_unsigned_to_nat(9u);
x_4 = l_Float_ofScientific(x_3, x_2, x_1);
return x_4;
}
}
static double _init_lp_aesop_Aesop_postponedSafeRuleSuccessProbability() {
_start:
{
double x_1; 
x_1 = lp_aesop_Aesop_postponedSafeRuleSuccessProbability___closed__0;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Percent(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Constants(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Percent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_unificationGoalPenalty___closed__0 = _init_lp_aesop_Aesop_unificationGoalPenalty___closed__0();
lp_aesop_Aesop_unificationGoalPenalty = _init_lp_aesop_Aesop_unificationGoalPenalty();
lp_aesop_Aesop_postponedSafeRuleSuccessProbability___closed__0 = _init_lp_aesop_Aesop_postponedSafeRuleSuccessProbability___closed__0();
lp_aesop_Aesop_postponedSafeRuleSuccessProbability = _init_lp_aesop_Aesop_postponedSafeRuleSuccessProbability();
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
