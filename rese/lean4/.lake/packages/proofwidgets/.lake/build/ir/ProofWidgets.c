// Lean compiler output
// Module: ProofWidgets
// Imports: public import Init public import ProofWidgets.Compat public import ProofWidgets.Component.Basic public import ProofWidgets.Component.FilterDetails public import ProofWidgets.Component.GraphDisplay public import ProofWidgets.Component.HtmlDisplay public import ProofWidgets.Component.InteractiveSvg public import ProofWidgets.Component.MakeEditLink public import ProofWidgets.Component.OfRpcMethod public import ProofWidgets.Component.Panel.Basic public import ProofWidgets.Component.Panel.GoalTypePanel public import ProofWidgets.Component.Panel.SelectionPanel public import ProofWidgets.Component.PenroseDiagram public import ProofWidgets.Component.Recharts public import ProofWidgets.Data.Html public import ProofWidgets.Data.Svg public import ProofWidgets.Presentation.Expr public import ProofWidgets.Extra.CheckHighlight
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
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Compat(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_Basic(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_FilterDetails(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_GraphDisplay(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_HtmlDisplay(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_InteractiveSvg(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_MakeEditLink(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_OfRpcMethod(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_Panel_Basic(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_Panel_GoalTypePanel(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_Panel_SelectionPanel(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_PenroseDiagram(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_Recharts(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Data_Html(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Data_Svg(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Presentation_Expr(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Extra_CheckHighlight(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_proofwidgets_ProofWidgets(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Compat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_FilterDetails(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_GraphDisplay(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_HtmlDisplay(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_InteractiveSvg(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_MakeEditLink(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_OfRpcMethod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_Panel_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_Panel_GoalTypePanel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_Panel_SelectionPanel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_PenroseDiagram(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_Recharts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Data_Html(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Data_Svg(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Presentation_Expr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Extra_CheckHighlight(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
