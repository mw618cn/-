# 模型实现范围与数据接口 — 2026-11-09

User asks for a complete, runnable Python research prototype with detailed Chinese documentation and a paper-style Word model/case report. User accepted editable Word OMML equations plus LaTeX source (native MathType OLE is not available). All parameters are illustrative and replaceable. Full study horizon is 24 h, default step 15 min (96 continuous-dispatch intervals). This must not be represented as a completed full-scale IEEE system, AC/transient black-start validation, or proof of novelty.

## Scope and core choices

Six buses: T0 external source + target non-black-start generator; T1 transmission loads; D0 distribution loads; M1/M2/M3 microgrids. Edges T0–T1, T1–D0 via mutually exclusive weak existing L0 or repaired higher-capacity B, D0–M1, D0–M2, D0–M3 through damaged A. LinDistFlow-type lossless active/reactive nodal balances, linear squared-voltage drops with line status, inscribed apparent-power polygons. Ideal nominal voltage transformations represented in pu. Microgrids initially self-started/grid-forming as appropriate. Main-grid initial support alpha may be zero. Energization transients, protection/inrush, communication attacks are outside scope.

Protected microgrid critical loads are hard served; other priority and ordinary loads are curtailable with different outage costs. Storage uses charge/discharge efficiencies and exclusivity binaries (mode default held for 1 h, configurable down to 15 min). Resource commands and SOC retain the whole 24 h horizon and a terminal reserve. PV forecasts deterministic in first version. Four uncertain hidden damage types are A light/heavy × B light/heavy with explicit joint prior.

One inspection/repair crew with travel-time matrix. Finite menu of 8 causal contingent crew policies: inspect A or B first; after observing first site's light/heavy type, either repair it before visiting the other, or inspect/repair the other before returning. Policies are irrevocable before their observation trigger. Travel, inspection and nonpreemptive repair are scheduled; damage known only on completed inspection. Each policy induces scenario-dependent observation and availability times. Select best policy by enumerating the menu; this is exact over this declared menu, NOT unrestricted VRP. The repair policy menu and grid can be extended.

For each candidate policy solve a four-scenario MILP. Explicit non-anticipativity constraints bind scenario controls that have the same actual observation history at each t. Availability is a scenario parameter conditional on that physical crew policy; inspection policy choices affect information timing. Start indicator is one-shot and permitted on configurable default one-hour grid within first 8 h; fixed stage auxiliary power, startup duration, net generation and ramp follow the selected start. A start is not assumed successful without supporting its complete auxiliary profile. No abort/restart in version 1. Online output min/max, ramp and Q capability; threshold-based first usable net output reported.

Reconfiguration switches A, B are monotone recovery switches on event grid; B replaces L0 (not both closed) and A connects M3. Fixed edges always available. Grid power availability scales with alpha, independently of target generator restoration. No exogenous 'grid returns at hour 6'.

Compare current_start (start slot 0), current_wait (slot 0 forbidden, future allowed) and free action, all over same menu and same constraints. J expected monetary outage/fuel/start/repair/grid/terminal cost. D=J_wait*-J_start* only when both branches have valid feasible costs; preserve solver lower/upper bounds and inconclusive intervals. Optional coordination contains no-start when feasible, so never claim negative *optimal optional-coordination gain*.

## API contract

Package src/tdm_restore. The package provides config.py, crew.py, linear.py, model.py, solution.py, experiments.py, __main__.py / cli.py.

`config.load_config(path=None, overrides=None) -> dict`; config includes `horizon_hours`, `step_minutes`, `n_steps` derived, grid/node/edge/microgrid/generator/crew/scenario/solver settings. JSON canonical configuration. `config.build_profiles(cfg) -> dict` with time_hours (N,), dt_hours float, pv_mw (3,N), load_critical_mw (6,N), load_general_mw (6,N), price and loss-cost arrays.

`crew.build_scenarios(cfg) -> list[dict]` keys id, name, probability, damage {A:light|heavy,B:...}. `crew.enumerate_policies(cfg) -> list[dict]` keys id,label,first,repair_first_if_light,repair_first_if_heavy. `crew.schedule_policy(cfg,policy,scenarios) -> list[dict]` each scenario keys scenario_id,tasks (records site,kind,start_hour,end_hour), inspection_hours {A,B}, repair_hours {A,B}, observation_history_by_step (list of tuples), availability (A/B arrays N). Time event boundaries align to dispatch grid. Same observation history determines controls; e.g before inspection all scenarios same, after first result only compatible pair shares controls.

`model.build_model(cfg, policy, branch='free', information='causal', coordination=True, fixed_switches=False) -> LinearModel`; branch start|wait|free; perfect_information uses no NA as ideal reference only. `LinearModel` exposes c, integrality, lb, ub, A (scipy CSR), row_lb,row_ub, var_indices dict, metadata dict, objective_constant float. `linear.solve(model, solver_cfg=None) -> dict` keys success, status, message, x (array or None), objective (cost with constant, inf if no incumbent), lower_bound (solver dual bound+constant), mip_gap, solve_seconds, node_count. `linear.freeze_discrete(model,x) -> LinearModel` fixes integer entries at verified incumbent and uses continuous LP, retaining all other constraints including NA. `linear.solve_lp(model, objective=None, extra_equalities=None) -> dict` supports sparse affine constraints for interface fixing; minimizes given vector, returns original objective if objective None, bound etc.

var_indices maps names to shaped arrays of variable columns. Important series: battery_charge (S,3,N), battery_discharge (S,3,N), energy (S,3,N+1), pv_used (S,3,N), served_critical (S,6,N), served_general (S,6,N), dg_power (S,3,N), inverter_q (S,3,N), target_power (S,N), target_q (S,N), target_start (S,K), switch_A (S,E), switch_B (S,E), flow_p (S,6,N), flow_q (S,6,N), voltage_sq (S,6,N), grid_p (S,N), grid_q (S,N). Edges order T0T1,L0,B,D0M1,D0M2,A. metadata includes profiles, scenarios, schedules, policy, branch, start_slots, event_slots, switch_index_for_t, target_aux_coeff (N,K), online_coeff, generation_cap_coeff, handover threshold. All row names optional.

Interface to TSO is `flow_p[:,1,:]+flow_p[:,2,:]` oriented T1→D0, so UPWARD support is minus this sum; upward Q analogous. MG export is minus edge D0→Mm. Do not confuse generation with interface export.

`solution.extract_solution(model,solver_result,cfg) -> dict` JSON serializable schema:
```
{
 policy_id,policy_label,branch,information,horizon_hours,dt_hours,time_hours,scenario_names,probabilities,
 solver:{status,message,objective,lower_bound,mip_gap,solve_seconds,...},
 summary:{expected_cost,expected_weighted_eens_mwh,expected_eens_mwh,expected_restored_energy_mwh,start_probability,handover_probability,expected_handover_hour,...},
 series:{
   mg_charge_mw:(S,3,N),mg_discharge_mw,mg_energy_mwh:(S,3,N+1),mg_soc:(S,3,N+1),mg_pv_used_mw:(S,3,N),mg_dg_mw:(S,3,N),mg_export_mw:(S,3,N),
   upward_p_mw:(S,N),upward_q_mvar:(S,N),target_power_mw:(S,N),target_aux_mw:(S,N),target_online:(S,N),target_start:(S,K),
   served_critical_mw:(S,6,N),served_general_mw:(S,6,N),demand_critical_mw:(6,N),demand_general_mw:(6,N),
   voltage_pu:(S,6,N),flow_p_mw:(S,6,N),flow_q_mvar:(S,6,N),switch_A:(S,N),switch_B:(S,N)
 },
 schedules:[...], cost_components:{expected categories}, diagnostics:{...}, parameter_digest:{...}
}
```
Export as solution.json (lists); Numpy .npz optional. Root supports extracting a failed result with status and no series; plots must skip gracefully.

Experiments CLI outputs `results/run_name/config_resolved.json`, `policy_screening.csv`, `solution.json`, `branch_comparison.json`, scenario/series CSVs, `validation.json`, `figures/`, `capability/` optional. Screen menu free action; if winner has current action start consistently can reuse its incumbent for start branch, otherwise solve conditions; preserve per-policy lower bounds to bound menu optimum. Baselines fixed crew vs contingent, earliest feasible start (tie objective), no upward collaboration, perfect information optionally; never label a bound/heuristic as certified optimum.


## Explanation and reproducibility

Every delivered module includes Chinese docstrings and purpose-labelled sections. Detailed explanations are in README.md and docs/code_walkthrough.md. The Word model and equations/ use the same variables, units and constraints as the code. All supplied result curves were produced by the included solver pipeline; configuration and solver bounds are stored with each case.
