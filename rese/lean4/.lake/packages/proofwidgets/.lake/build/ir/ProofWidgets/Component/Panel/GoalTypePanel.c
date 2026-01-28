// Lean compiler output
// Module: ProofWidgets.Component.Panel.GoalTypePanel
// Imports: public import Init public meta import ProofWidgets.Component.Panel.Basic
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
static lean_object* lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__4;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_GoalTypePanel;
uint64_t lean_string_hash(lean_object*);
static lean_object* lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__3;
static uint64_t lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__1;
static lean_object* lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__2;
static lean_object* lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__0;
static lean_object* _init_lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("window;import{jsxs as e,jsx as r,Fragment as n}from\"react/jsx-runtime\";import{useRpcSession as t,EnvPosContext as a,useAsyncPersistent as o,mapRpcError as s,importWidgetModule as i,InteractiveCode as l}from\"@leanprover/infoview\";import*as c from\"react\";async function m(t,a,o){if(\"text\"in o)return r(n,{children:o.text});if(\"element\"in o){const[e,n,s]=o.element,i={};for(const[e,r]of n)i[e]=r;const l=await Promise.all(s.map((async e=>await m(t,a,e))));return\"hr\"===e\?r(\"hr\",{}):0===l.length\?c.createElement(e,i):c.createElement(e,i,l)}if(\"component\"in o){const[e,r,n,s]=o.component,l=await Promise.all(s.map((async e=>await m(t,a,e)))),u={...n,pos:a},d=await i(t,a,e);if(!(r in d))throw new Error(`Module '${e}' does not export '${r}'`);return 0===l.length\?c.createElement(d[r],u):c.createElement(d[r],u,l)}return e(\"span\",{className:\"red\",children:[\"Unknown HTML variant: \",JSON.stringify(o)]})}function u({html:i}){const l=t(),u=c.useContext(a),d=o((()=>m(l,u,i)),[l,u,i]);return\"resolved\"===d.state\?d.value:\"rejected\"===d.state\?e(\"span\",{className:\"red\",children:[\"Error rendering HTML: \",s(d.error).message]}):r(n,{})}function d({expr:a}){const i=t(),c=o((()=>i.call(\"ProofWidgets.ppExprTagged\",{expr:a})),[a]);return\"resolved\"===c.state\?r(l,{fmt:c.value}):\"rejected\"===c.state\?e(n,{children:[\"Error: $\",s(c.error).message]}):r(n,{children:\"Loading..\"})}function f({expr:a}){const i=t(),[l,m]=c.useState({tag:\"auto\"}),f=o((async()=>{const e=await async function(e,r){return(await e.call(\"ProofWidgets.getExprPresentations\",{expr:r})).presentations}(i,a);return m((r=>\"manual\"!==r.tag||void 0===r.name||e.some((e=>e.name===r.name))\?r:{tag:\"auto\"})),new Map(e.map((e=>[e.name,e])))}),[i,a]);if(\"rejected\"===f.state)return e(n,{children:[\"Error: \",s(f.error).message]});if(\"resolved\"===f.state){let n=\"none\";return\"auto\"===l.tag&&0<f.value.size\?n=Array.from(f.value.values())[0].name:\"manual\"!==l.tag||\"none\"!==l.name&&!f.value.has(l.name)||(n=l.name),e(\"div\",{style:{display:\"flow-root\"},children:[\"none\"!==n&&r(u,{html:f.value.get(n).html}),\"none\"===n&&r(d,{expr:a}),e(\"select\",{className:\"fr\",value:n,onChange:e=>{m({tag:\"manual\",name:e.target.value})},children:[Array.from(f.value.values(),(e=>r(\"option\",{value:e.name,children:e.userName},e.name))),r(\"option\",{value:\"none\",children:\"Default\"},\"none\")]})]})}return r(d,{expr:a})}function p({pos:a,goals:i,loc:l}){const c=t(),m=o((async()=>{const e=function(e,r){for(const n of e)if(n.mvarId===r.mvarId)return n;throw new Error(`Could not find goal for location ${JSON.stringify(r)}`)}(i,l);if(void 0===e.ctx)throw new Error(\"Lean server 1.1.2 or newer is required.\");return(await c.call(\"ProofWidgets.goalsLocationsToExprs\",{locations:[[e.ctx,l]]})).exprs[0]}),[c,i,l]);return\"loading\"===m.state\?r(n,{children:\"Loading..\"}):\"rejected\"===m.state\?e(n,{children:[\"Error: \",s(m.error).message]}):r(f,{expr:m.value})}function g(t){if(0===t.goals.length)return r(n,{});const a=t.goals[0];if(!a.mvarId)throw new Error(\"Lean server 1.1.2 or newer is required.\");return e(\"details\",{open:!0,children:[r(\"summary\",{className:\"mv2 pointer\",children:\"Main goal type\"}),r(p,{pos:t.pos,goals:t.goals,loc:{mvarId:a.mvarId,loc:{target:\"/\"}}})]})}export{g as default};", 3208, 3208);
return x_1;
}
}
static uint64_t _init_lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__1() {
_start:
{
lean_object* x_1; uint64_t x_2; 
x_1 = lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__0;
x_2 = lean_string_hash(x_1);
return x_2;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__2() {
_start:
{
uint64_t x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__1;
x_2 = lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__0;
x_3 = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set_uint64(x_3, sizeof(void*)*1, x_1);
return x_3;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("default", 7, 7);
return x_1;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__3;
x_2 = lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__2;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_GoalTypePanel() {
_start:
{
lean_object* x_1; 
x_1 = lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__4;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_Panel_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_proofwidgets_ProofWidgets_Component_Panel_GoalTypePanel(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_Panel_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__0 = _init_lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__0();
lean_mark_persistent(lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__0);
lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__1 = _init_lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__1();
lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__2 = _init_lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__2();
lean_mark_persistent(lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__2);
lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__3 = _init_lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__3();
lean_mark_persistent(lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__3);
lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__4 = _init_lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__4();
lean_mark_persistent(lp_proofwidgets_ProofWidgets_GoalTypePanel___closed__4);
lp_proofwidgets_ProofWidgets_GoalTypePanel = _init_lp_proofwidgets_ProofWidgets_GoalTypePanel();
lean_mark_persistent(lp_proofwidgets_ProofWidgets_GoalTypePanel);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
