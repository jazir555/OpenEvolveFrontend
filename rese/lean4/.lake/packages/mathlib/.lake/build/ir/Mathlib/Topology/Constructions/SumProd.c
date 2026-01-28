// Lean compiler output
// Module: Mathlib.Topology.Constructions.SumProd
// Imports: public import Init public import Mathlib.Topology.Homeomorph.Defs public import Mathlib.Topology.Maps.Basic public import Mathlib.Topology.Separation.SeparatedNhds
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
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_sumComm(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Homeomorph_prodAssoc___closed__0;
static lean_object* lp_mathlib_Homeomorph_emptySum___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodPUnit(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodSumDistrib___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Homeomorph_sumEmpty___closed__0;
static lean_object* lp_mathlib_instTopologicalSpaceProd___closed__0;
lean_object* lp_mathlib_Equiv_prodComm(lean_object*, lean_object*);
static lean_object* lp_mathlib_Homeomorph_prodProdProdComm___closed__0;
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTopologicalSpaceProd(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_instTopologicalSpaceSum___closed__0;
lean_object* lp_mathlib_Equiv_sumComm(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sumSumSumComm(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Homeomorph_sumProdDistrib___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_sumProdDistrib(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Homeomorph_prodSumDistrib___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_sumCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodComm(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_prodCongr___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_Homeomorph_punitProd___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_emptySum(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodSumDistrib(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTopologicalSpaceSum(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_sumSumSumComm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Homeomorph_sumProdDistrib___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodProdProdComm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_sumEmpty(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Homeomorph_sumAssoc___closed__0;
lean_object* lp_mathlib_Equiv_sumCongr___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_Homeomorph_sumProdDistrib___closed__1;
lean_object* lp_mathlib_Equiv_sumProdDistrib(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_TopologicalSpace_gciGenerateFrom___lam__0(lean_object*, lean_object*);
static lean_object* lp_mathlib_Homeomorph_sumComm___closed__0;
lean_object* lp_mathlib_Equiv_sumAssoc(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_sumAssoc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Homeomorph_prodComm___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_sumCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Homeomorph_sumSumSumComm___closed__0;
lean_object* lp_mathlib_Equiv_prodProdProdComm(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_punitProd(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodAssoc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_prodAssoc(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Homeomorph_prodPUnit___closed__0;
lean_object* lp_mathlib_Equiv_prodPUnit(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sumEmpty(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_instTopologicalSpaceSum___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_TopologicalSpace_gciGenerateFrom___lam__0(lean_box(0), lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTopologicalSpaceSum(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_instTopologicalSpaceSum___closed__0;
return x_5;
}
}
static lean_object* _init_lp_mathlib_instTopologicalSpaceProd___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_refl(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTopologicalSpaceProd(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lp_mathlib_instTopologicalSpaceProd___closed__0;
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
x_7 = lean_box(0);
x_8 = lean_apply_1(x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodCongr(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_Equiv_prodCongr___redArg(x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodCongr___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_prodCongr___redArg(x_1, x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Homeomorph_prodComm___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_prodComm(lean_box(0), lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodComm(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Homeomorph_prodComm___closed__0;
return x_5;
}
}
static lean_object* _init_lp_mathlib_Homeomorph_prodAssoc___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_prodAssoc(lean_box(0), lean_box(0), lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodAssoc(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Homeomorph_prodAssoc___closed__0;
return x_7;
}
}
static lean_object* _init_lp_mathlib_Homeomorph_prodProdProdComm___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_prodProdProdComm(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodProdProdComm(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Homeomorph_prodProdProdComm___closed__0;
return x_9;
}
}
static lean_object* _init_lp_mathlib_Homeomorph_prodPUnit___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_prodPUnit(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodPUnit(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Homeomorph_prodPUnit___closed__0;
return x_3;
}
}
static lean_object* _init_lp_mathlib_Homeomorph_punitProd___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Homeomorph_prodPUnit___closed__0;
x_2 = lp_mathlib_Homeomorph_prodComm___closed__0;
x_3 = lp_mathlib_Equiv_trans___redArg(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_punitProd(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Homeomorph_punitProd___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_sumCongr(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_Equiv_sumCongr___redArg(x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_sumCongr___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_sumCongr___redArg(x_1, x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Homeomorph_sumComm___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_sumComm(lean_box(0), lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_sumComm(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Homeomorph_sumComm___closed__0;
return x_5;
}
}
static lean_object* _init_lp_mathlib_Homeomorph_sumAssoc___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_sumAssoc(lean_box(0), lean_box(0), lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_sumAssoc(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Homeomorph_sumAssoc___closed__0;
return x_7;
}
}
static lean_object* _init_lp_mathlib_Homeomorph_sumSumSumComm___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_sumSumSumComm(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_sumSumSumComm(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Homeomorph_sumSumSumComm___closed__0;
return x_9;
}
}
static lean_object* _init_lp_mathlib_Homeomorph_sumEmpty___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_sumEmpty(lean_box(0), lean_box(0), lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_sumEmpty(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Homeomorph_sumEmpty___closed__0;
return x_6;
}
}
static lean_object* _init_lp_mathlib_Homeomorph_emptySum___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Homeomorph_sumEmpty___closed__0;
x_2 = lp_mathlib_Homeomorph_sumComm___closed__0;
x_3 = lp_mathlib_Equiv_trans___redArg(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_emptySum(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Homeomorph_emptySum___closed__0;
return x_6;
}
}
static lean_object* _init_lp_mathlib_Homeomorph_sumProdDistrib___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_sumProdDistrib(lean_box(0), lean_box(0), lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Homeomorph_sumProdDistrib___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Homeomorph_sumProdDistrib___closed__0;
x_2 = lp_mathlib_Equiv_symm___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Homeomorph_sumProdDistrib___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Homeomorph_sumProdDistrib___closed__1;
x_2 = lp_mathlib_Equiv_symm___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_sumProdDistrib(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Homeomorph_sumProdDistrib___closed__2;
return x_7;
}
}
static lean_object* _init_lp_mathlib_Homeomorph_prodSumDistrib___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Homeomorph_prodComm___closed__0;
x_2 = lp_mathlib_Equiv_sumCongr___redArg(x_1, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodSumDistrib___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lp_mathlib_Homeomorph_prodComm___closed__0;
x_5 = lp_mathlib_Homeomorph_sumProdDistrib(lean_box(0), lean_box(0), lean_box(0), x_2, x_3, x_1);
x_6 = lp_mathlib_Homeomorph_prodSumDistrib___redArg___closed__0;
x_7 = lp_mathlib_Equiv_trans___redArg(x_5, x_6);
x_8 = lp_mathlib_Equiv_trans___redArg(x_4, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Homeomorph_prodSumDistrib(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Homeomorph_prodSumDistrib___redArg(x_4, x_5, x_6);
return x_7;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Homeomorph_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Maps_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Separation_SeparatedNhds(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_Constructions_SumProd(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Homeomorph_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Maps_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Separation_SeparatedNhds(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_instTopologicalSpaceSum___closed__0 = _init_lp_mathlib_instTopologicalSpaceSum___closed__0();
lean_mark_persistent(lp_mathlib_instTopologicalSpaceSum___closed__0);
lp_mathlib_instTopologicalSpaceProd___closed__0 = _init_lp_mathlib_instTopologicalSpaceProd___closed__0();
lean_mark_persistent(lp_mathlib_instTopologicalSpaceProd___closed__0);
lp_mathlib_Homeomorph_prodComm___closed__0 = _init_lp_mathlib_Homeomorph_prodComm___closed__0();
lean_mark_persistent(lp_mathlib_Homeomorph_prodComm___closed__0);
lp_mathlib_Homeomorph_prodAssoc___closed__0 = _init_lp_mathlib_Homeomorph_prodAssoc___closed__0();
lean_mark_persistent(lp_mathlib_Homeomorph_prodAssoc___closed__0);
lp_mathlib_Homeomorph_prodProdProdComm___closed__0 = _init_lp_mathlib_Homeomorph_prodProdProdComm___closed__0();
lean_mark_persistent(lp_mathlib_Homeomorph_prodProdProdComm___closed__0);
lp_mathlib_Homeomorph_prodPUnit___closed__0 = _init_lp_mathlib_Homeomorph_prodPUnit___closed__0();
lean_mark_persistent(lp_mathlib_Homeomorph_prodPUnit___closed__0);
lp_mathlib_Homeomorph_punitProd___closed__0 = _init_lp_mathlib_Homeomorph_punitProd___closed__0();
lean_mark_persistent(lp_mathlib_Homeomorph_punitProd___closed__0);
lp_mathlib_Homeomorph_sumComm___closed__0 = _init_lp_mathlib_Homeomorph_sumComm___closed__0();
lean_mark_persistent(lp_mathlib_Homeomorph_sumComm___closed__0);
lp_mathlib_Homeomorph_sumAssoc___closed__0 = _init_lp_mathlib_Homeomorph_sumAssoc___closed__0();
lean_mark_persistent(lp_mathlib_Homeomorph_sumAssoc___closed__0);
lp_mathlib_Homeomorph_sumSumSumComm___closed__0 = _init_lp_mathlib_Homeomorph_sumSumSumComm___closed__0();
lean_mark_persistent(lp_mathlib_Homeomorph_sumSumSumComm___closed__0);
lp_mathlib_Homeomorph_sumEmpty___closed__0 = _init_lp_mathlib_Homeomorph_sumEmpty___closed__0();
lean_mark_persistent(lp_mathlib_Homeomorph_sumEmpty___closed__0);
lp_mathlib_Homeomorph_emptySum___closed__0 = _init_lp_mathlib_Homeomorph_emptySum___closed__0();
lean_mark_persistent(lp_mathlib_Homeomorph_emptySum___closed__0);
lp_mathlib_Homeomorph_sumProdDistrib___closed__0 = _init_lp_mathlib_Homeomorph_sumProdDistrib___closed__0();
lean_mark_persistent(lp_mathlib_Homeomorph_sumProdDistrib___closed__0);
lp_mathlib_Homeomorph_sumProdDistrib___closed__1 = _init_lp_mathlib_Homeomorph_sumProdDistrib___closed__1();
lean_mark_persistent(lp_mathlib_Homeomorph_sumProdDistrib___closed__1);
lp_mathlib_Homeomorph_sumProdDistrib___closed__2 = _init_lp_mathlib_Homeomorph_sumProdDistrib___closed__2();
lean_mark_persistent(lp_mathlib_Homeomorph_sumProdDistrib___closed__2);
lp_mathlib_Homeomorph_prodSumDistrib___redArg___closed__0 = _init_lp_mathlib_Homeomorph_prodSumDistrib___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Homeomorph_prodSumDistrib___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
