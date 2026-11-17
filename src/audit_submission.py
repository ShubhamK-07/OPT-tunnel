"""Check final submission integrity without re-running optimization.

Run: python -m src.audit_submission
"""
from __future__ import annotations
import json
import re
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from pptx import Presentation
from . import report_html

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'
b = json.loads((R / 'baseline.json').read_text())
checks = []
def check(name, condition):
    checks.append((name, bool(condition)))
    print(('PASS  ' if condition else 'FAIL  ') + name)

p = pd.read_csv(R / 'control_points_hard.csv')
e = pd.read_csv(R / 'route_easy.csv')
h = pd.read_csv(R / 'route_hard.csv')
cols = ['easting_m', 'northing_m']
P, E, H = (x[cols].to_numpy() for x in (p, e, h))
check('Nine control points P0–P8, seven interior', len(P) == 9 and list(p.point) == [f'P{i}' for i in range(9)])
check('P0/A and P8/B match both exported route endpoints',
      np.allclose(P[[0,-1]], [b['endpoints']['A_xy_m'], b['endpoints']['B_xy_m']], atol=1e-7, rtol=0)
      and np.allclose(E[[0,-1]], P[[0,-1]], atol=1e-7, rtol=0)
      and np.allclose(H[[0,-1]], P[[0,-1]], atol=1e-7, rtol=0))
check('CSV coordinates match baseline JSON geometry',
      np.allclose(P, [[q['easting_m'],q['northing_m']] for q in b['geometry']['control_points_hard']], atol=1e-7, rtol=0)
      and np.allclose(E, b['geometry']['route_easy_xy_m'], atol=1e-7, rtol=0)
      and np.allclose(H, b['geometry']['route_hard_xy_m'], atol=1e-7, rtol=0))
n, degree = len(P), 3
knots = np.r_[np.zeros(degree+1), np.arange(1,n-degree)/(n-degree), np.ones(degree+1)]
check('Control points reconstruct all sampled spline coordinates',
      np.allclose(BSpline(knots, P, degree)(h.parameter_t.to_numpy()), H, atol=1e-6, rtol=0))
def length(xy): return np.hypot(*np.diff(xy, axis=0).T).sum()/1000
check('Route CSV lengths reproduce baseline reported lengths',
      abs(length(E)-b['comparison']['L_easy_km']) < 1e-8 and abs(length(H)-b['comparison']['L_hard_km']) < 1e-8)
snap = b['verification']['dijkstra_crosscheck']
check('Graph distance plus exact endpoint connectors equals full route',
      abs(snap['own_length_m']+snap['connector_A_m']+snap['connector_B_m']-snap['own_length_with_exact_endpoints_m']) < 1e-6
      and snap['difference_m'] < 1e-6)
check('Both adopted routes independently feasible', all(v['feasible'] for v in b['route_feasibility'].values()))
run = json.loads((R/'hard_runs.json').read_text())['hard_runs']
m = b['verification']['multistart']
check('All solver executions classified and counts match baseline',
      len(run)==m['n_solver_executions'] and all(x.get('independent_feasibility') is not None for x in run)
      and sum(x['independent_feasibility']['feasible'] for x in run)==m['n_feasible'])
check('All seven toy tests passed', all(x['pass'] for x in json.loads((R/'toy_tests.json').read_text())['tests']))
for name in ('score','full','secondary','beta'):
    df = pd.read_csv(R / f'sensitivity_{name}.csv')
    invalid = df[~df.valid_solution.astype(bool)]
    check(f'{name} sensitivity invalid pairs have no improvement metrics',
          invalid[['price_of_realism_pct','risk_reduction_pct','price_of_realism_pct_frozen','risk_reduction_pct_frozen']].isna().all().all()
          and (~df.valid_solution.astype(bool) | (df.easy_feasible.astype(bool) & df.hard_feasible.astype(bool))).all())
csv = pd.read_csv(R/'comparison.csv').iloc[0]
t = b['verification']['runtimes']
check('Explicit runtime CSV fields match current baseline',
      abs(csv.runtime_easy_baseline_graph_search_s-t['easy_single_dijkstra_s'])<1e-12
      and abs(csv.runtime_easy_repeated_solve_median_s-t['easy_repeated_median_s'])<1e-12
      and abs(csv.runtime_hard_adopted_penalty_continuation_s-t['hard_adopted_penalty_continuation_s'])<1e-12
      and abs(csv.runtime_hard_repeated_refinement_median_s-t['hard_repeated_median_s'])<1e-12
      and abs(csv.runtime_hard_full_multistart_budget_s-t['hard_full_multistart_budget_s'])<1e-12)
report = (ROOT/'report.md').read_text()
refs = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', report)
check('Report figure references exist', all((ROOT/x).is_file() for x in refs))
with tempfile.TemporaryDirectory() as td:
    target = Path(td)/'report.html'
    report_html.build(out=target)
    check('HTML is the exact current rendering of Markdown', target.read_bytes() == (ROOT/'report.html').read_bytes())
prs = Presentation(ROOT/'deck.pptx')
check('PowerPoint package opens and has 12 slides and 12 speaker notes',
      len(prs.slides)==12 and all(s.has_notes_slide and len(s.notes_slide.notes_text_frame.text.strip())>25 for s in prs.slides))
notes = prs.slides[5].notes_slide.notes_text_frame.text
check('Slide 6 speaker notes state snapped endpoints and graph cost',
      'snapped' in notes.lower() and f"{snap['own_length_m']/1000:.3f}" in notes and 'connector' in notes.lower())
check('Control points appear in report appendix', all(f"{v:.3f}" in report for v in P.ravel()))
check('Citation, configuration, metadata and inputs present',
      '10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa' in report and all((ROOT/x).exists() for x in
      ('config.yaml','requirements.txt','data/processed/baseline_meta.json','data/raw','src')))
print(f'\n{sum(v for _,v in checks)}/{len(checks)} checks passed')
if not all(v for _,v in checks): raise SystemExit(1)
