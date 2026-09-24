"""Pinned source, bounded views and reconstructed ex-ante command plans.

No function in this module reads episode 7. There is no data download fallback.
The only allowed future input is regenerated from the already-published schedule
algorithm, then checked against its stored hash and the applied-control record.
"""
from __future__ import annotations
from dataclasses import dataclass
import csv, gzip, io, json, math, random
from pathlib import Path
import numpy as np
from .common import require, ContractError, file_hash, sha_bytes, write_json, utc

TARGETS = ('reaTSup_y', 'reaTZon_y')
CONTROL = 'oveHeaPumY_u'
ARMS = ('CURRENT_PLAN', 'HISTORY_SUMMARY_PLAN', 'HISTORY_LAGS_PLAN')


def regenerate_schedule(settings, ep):
    rng = random.Random(settings['seed_base'] + ep)
    levels = settings['levels']; dwell = settings['dwell_steps']
    current = levels[(settings['seed_base'] + ep) % len(levels)]
    out = []
    while len(out) < settings['n_control_steps']:
        current = rng.choice([v for v in levels if v != current])
        out.extend([current] * rng.choice(dwell))
    return np.asarray(out[:settings['n_control_steps']], dtype=np.float64)


def schedule_hash(schedule):
    # Historical generator used json.dumps(list_of_Python_floats), including spaces.
    return sha_bytes(json.dumps(schedule.tolist()).encode())


def plan_for_endpoints(schedule, offsets_s, step_s):
    """Stored measurement at t sees the command used in the preceding interval.

    For t in (k*step, (k+1)*step], use schedule[k]. t=0 is undefined.
    This is NOT a free shift parameter: misalignment aborts, never auto-searches.
    """
    offsets_s = np.asarray(offsets_s, dtype=np.float64)
    require(np.all(offsets_s > 0), 'Plan at initial measurement is undefined')
    require(np.all(offsets_s <= len(schedule)*step_s), 'Plan request beyond schedule')
    indices = np.ceil(offsets_s / float(step_s)).astype(np.int64) - 1
    return np.asarray(schedule)[indices]


@dataclass
class Episode:
    index: int
    time: np.ndarray
    fields: dict
    schedule: np.ndarray
    start_time: float
    step_s: float

    @property
    def change_times(self):
        k = np.flatnonzero(np.abs(np.diff(self.schedule)) > 1e-8) + 1
        return self.start_time + k*self.step_s


class PinnedSource:
    def __init__(self, repo, config, public):
        self.repo = Path(repo).resolve(); self.config = config; self.public = Path(public)
        self.allowed = set(config['train_episodes']); self.phase = 'TRAIN_ONLY'
        self.records = []; self.touched = {}
        self.opened_episodes = []

    def _path(self, ep):
        require(ep not in self.config['never_open_episodes'], 'RESERVE_ACCESS_FORBIDDEN')
        require(ep in self.allowed, 'DEV_ACCESS_BEFORE_SEAL_FORBIDDEN')
        require(str(ep) in self.config['source_data'], 'No locked source for episode')
        path = self.repo / self.config['source_data'][str(ep)]['relative_path']
        require(path.is_file() and not path.is_symlink(), 'Missing/symlink source: ' + str(path))
        require(path.resolve().is_relative_to(self.repo), 'Source path escaped repository')
        return path

    def load(self, ep):
        path = self._path(ep); lock = self.config['source_data'][str(ep)]
        require(path.stat().st_size <= self.config['budgets']['max_episode_compressed_mib']*(1<<20), 'Oversized episode gzip')
        digest = file_hash(path)
        require(digest == lock['sha256'], f'SOURCE_HASH_MISMATCH episode {ep}; do not replace or regenerate')
        with gzip.open(path, 'rb') as f:
            raw = f.read(self.config['budgets']['max_episode_decompressed_mib']*(1<<20) + 1)
        require(len(raw) <= self.config['budgets']['max_episode_decompressed_mib']*(1<<20), 'Oversized decompressed episode')
        rows = list(csv.reader(io.StringIO(raw.decode('utf-8'))))
        require(rows and len(rows[0]) == len(set(rows[0])), 'Duplicate/missing CSV columns')
        head = rows[0]; a = np.asarray(rows[1:], dtype=np.float64)
        require(a.ndim == 2 and a.shape == (lock['n_samples'], len(head)), 'CSV dimensions changed')
        fields = {h: a[:, i] for i, h in enumerate(head)}
        needed = ['time', *TARGETS, CONTROL, 'ovePum_u', 'oveFan_u']
        require(all(k in fields for k in needed), 'Required measurement/control column missing')
        require(all(np.isfinite(fields[k]).all() for k in needed), 'Non-finite required observations')
        t = fields['time']; start = (lock['start_day']-1)*86400.0
        require(float(t[0]) == start, 'Episode start differs from recorded contract')
        require(np.allclose(np.diff(t), lock['dt_s'], rtol=0, atol=1e-7), 'Nonuniform stored grid')
        schedule = regenerate_schedule(self.config['schedule'], ep)
        require(schedule_hash(schedule) == lock['schedule_sha256'], 'SCHEDULE_HASH_MISMATCH')
        step = self.config['schedule']['control_step_s']
        require(abs(t[-1]-start-len(schedule)*step) < 1e-7, 'Episode length mismatch')
        planned = plan_for_endpoints(schedule, t[1:]-start, step)
        mismatch = np.flatnonzero(np.abs(planned - fields[CONTROL][1:]) > 1e-8)
        if len(mismatch):
            j = int(mismatch[0])+1
            raise ContractError(f'COMMAND_TIME_ALIGNMENT_MISMATCH ep={ep}, row={j}, t={t[j]}, '
                                f'planned={planned[j-1]}, applied={fields[CONTROL][j]}; no automatic shift allowed')
        for k in ('ovePum_u', 'oveFan_u'):
            require(np.allclose(fields[k][1:], (planned>0).astype(float), rtol=0, atol=1e-8),
                    'Auxiliary control is not the pinned compressor>0 mapping: ' + k)
        self.touched[ep] = digest; self.opened_episodes.append(ep)
        self.records.append({'utc': utc(), 'phase': self.phase, 'episode': ep, 'sha256': digest,
                             'schedule_sha256': schedule_hash(schedule), 'command_alignment': 'EXACT_PRECEDING_INTERVAL',
                             'zero_time_command': 'initial measurement excluded from planned-control equality',
                             'samples': len(t), 'start': float(t[0]), 'end': float(t[-1]),
                             'schedule_provenance': 'reconstructed deterministic preschedule; not a wall-clock issuance log'})
        self._save_audit()
        return Episode(ep, t, fields, schedule, start, step)

    def enable_dev(self, seal_path):
        require(self.phase == 'TRAIN_ONLY', 'DEV already enabled')
        from .common import read_json, canonical_hash
        seal = read_json(seal_path); body = dict(seal); digest = body.pop('seal_sha256')
        require(canonical_hash(body) == digest, 'Invalid pre-DEV seal')
        require(seal['target_selection'] == 'NONE_ALL_FIVE_RETAINED', 'Candidate selection violates contract')
        require(seal['train_episodes'] == self.config['train_episodes'], 'Seal split mismatch')
        self.allowed |= set(self.config['dev_episodes']); self.phase = 'AFTER_PRE_DEV_SEAL'
        self.records.append({'utc': utc(), 'phase': self.phase, 'event': 'DEV_AUTHORIZED', 'seal_sha256': digest})
        self._save_audit()

    def verify_unchanged(self):
        for ep, expected in self.touched.items():
            require(file_hash(self._path(ep)) == expected, 'Source changed during re-evaluation')
        self._save_audit()

    def _save_audit(self):
        write_json(self.public/'DATA_ACCESS.json', {'records': self.records,
                   'opened_episodes': self.opened_episodes, 'reserve_measurements_opened': False})


@dataclass
class ForecastView:
    time_context: np.ndarray
    target_context: np.ndarray
    command_context: np.ndarray
    plan_times: np.ndarray
    command_plan: np.ndarray


def make_view(ep: Episode, cc, origin_time):
    C, dt, H = cc['context_s'], cc['sample_s'], cc['horizon_s']
    require(C % dt == 0 and H % dt == 0, 'Context/horizon not divisible by sample interval')
    tc = origin_time - C + np.arange(C//dt+1, dtype=np.float64)*dt
    tp = origin_time + np.arange(1,H//dt+1, dtype=np.float64)*dt
    indices = np.searchsorted(ep.time, tc)
    require(np.all(indices < len(ep.time)) and np.array_equal(ep.time[indices], tc), 'Context timestamps not on stored grid')
    require(tp[-1] <= ep.time[-1], 'Forecast extends beyond episode')
    # Future target and future executed command are deliberately NOT read here.
    plan = plan_for_endpoints(ep.schedule, tp-ep.start_time, ep.step_s)
    return ForecastView(tc.copy(), ep.fields[cc['target']][indices].copy(),
                        ep.fields[CONTROL][indices].copy(), tp, plan)


def _slope(values, dt):
    t = np.arange(len(values), dtype=np.float64)*dt; t -= t.mean()
    den = float(t@t)
    return float(t@(values-values.mean())/den) if den else 0.0


def features(view: ForecastView, cc, config):
    y, u, plan = view.target_context, view.command_context, view.command_plan
    dt, C = cc['sample_s'], cc['context_s']
    require(len(y) == C//dt+1 and len(u) == len(y), 'Bounded context shape mismatch')
    require(np.isfinite(y).all() and np.isfinite(u).all() and np.isfinite(plan).all(), 'Invalid view values')
    common = [float(y[-1]), float(u[-1]), float(u[-1]>0)]
    names = ['target_now','command_now','command_on_now']
    for lag in config['target_lags_s']:
        if 0 < lag <= C and lag % dt == 0:
            common.append(float(y[-1-lag//dt])); names.append(f'y_lag_{lag}s')
    for span in config['target_summary_windows_s']:
        if span <= C and span % dt == 0:
            v = y[-(span//dt+1):]
            common += [float(v.mean()), float(v.std()), _slope(v,dt)]
            names += [f'y_{span}s_mean', f'y_{span}s_std', f'y_{span}s_slope']
    # Same FULL scheduled future input in all three arms, not a coarse summary.
    plan_start = len(common)
    common += plan.tolist() + (plan>0).astype(float).tolist()
    names += [f'planned_command_t+{(i+1)*dt}s' for i in range(len(plan))]
    names += [f'planned_on_t+{(i+1)*dt}s' for i in range(len(plan))]
    hist, hn = [], []
    for span in config['command_summary_windows_s']:
        if span <= C and span % dt == 0:
            v = u[-(span//dt+1):]; d=np.diff(v)
            hist += [float(v.mean()),float(v.std()),abs(float(u[-1]-v.mean())),
                     float(np.sum(np.abs(d)>1e-8)),float(np.abs(d).sum())]
            hn += [f'u_{span}s_{x}' for x in ('mean','std','now_minus_mean_abs','changes','total_variation')]
    changes = np.flatnonzero(np.abs(np.diff(u))>1e-8)+1
    # The true pre-context last-change time is forbidden; absence is right-censored at C.
    age = (len(u)-1-int(changes[-1]))*dt if len(changes) else C
    hist += [float(age), float(not len(changes))]
    hn += ['command_age_capped_at_context_s','command_age_left_censored']
    lag_names=[f'u_lag_{(len(u)-1-i)*dt}s' for i in range(len(u)-1)]
    lag_on_names=[f'u_on_lag_{(len(u)-1-i)*dt}s' for i in range(len(u)-1)]
    lag_values=u[:-1].tolist()+(u[:-1]>0).astype(float).tolist()
    arrays={ARMS[0]:np.asarray(common),ARMS[1]:np.asarray(common+hist),ARMS[2]:np.asarray(common+lag_values)}
    schema={ARMS[0]:names,ARMS[1]:names+hn,ARMS[2]:names+lag_names+lag_on_names}
    return arrays, schema, {'shared_prefix_features':len(common),'plan_start_feature':plan_start,
                            'plan_feature_count':2*len(plan),'physical_context_span_s':C}


def label_panels(view, cc, config):
    """Command-defined strata only. Never use target values/errors or an estimated tau."""
    u = view.command_context; dt = cc['sample_s']; C=cc['context_s']
    changes=np.flatnonzero(np.abs(np.diff(u))>1e-8)+1
    age=(len(u)-1-int(changes[-1]))*dt if len(changes) else None
    w=config['recent_window_s'][cc['name']]
    recent=age is not None and age<=w
    future_changes=int(np.sum(np.abs(np.diff(np.r_[u[-1],view.command_plan]))>1e-8))
    upcoming=future_changes>0
    flags={'ALL':True,'RECENT_OR_PLANNED_CHANGE':bool(recent or upcoming),
           'RECENT_PAST_CHANGE':bool(recent),'CHANGE_IN_PLAN':bool(upcoming),
           'NO_RECENT_OR_PLANNED_CHANGE':bool(not recent and not upcoming),
           'MULTIPLE_CHANGES_IN_CONTEXT':bool(len(changes)>=2)}
    return flags, {'past_changes_in_context':int(len(changes)), 'plan_changes':future_changes,
                   'last_observed_change_age_s':age,'recent_window_s':w,
                   'age_resolution_s':dt,'no_change_does_not_prove_thermal_settling':True}


@dataclass
class Prepared:
    episode: int
    X: dict
    Y: np.ndarray
    origins: np.ndarray
    target_times: np.ndarray
    masks: dict
    metadata: list
    schema: dict
    audit: dict
    persistence: np.ndarray


def prepare(ep, cc, config):
    # Control plans can only be known at/after the input timestamp. No interpolation.
    # Preserve v1's forecast-origin phase. Drop its first incomplete C-second view
    # instead of shifting every origin to a different control phase.
    first=ep.time[0]+cc['context_s']-cc['sample_s']; last=ep.time[-1]-cc['horizon_s']
    original_grid=np.arange(first,last+1e-7,cc['origin_stride_s'])
    origins=original_grid[original_grid>=ep.time[0]+cc['context_s']]
    require(len(origins)>0, 'No forecast origins for task')
    xs={a:[] for a in ARMS}; Y=[]; times=[]; ps=[]; metadata=[]
    masks={p:[] for p in config['panels']}; schema=None; audit=None
    for t in origins:
        v=make_view(ep,cc,t); fx,sc,au=features(v,cc,config)
        flags,meta=label_panels(v,cc,config)
        yi=np.searchsorted(ep.time,v.plan_times)
        require(np.array_equal(ep.time[yi],v.plan_times),'Target grid mismatch')
        # Only the label extraction below reads future target values.
        Y.append(ep.fields[cc['target']][yi]); times.append(v.plan_times)
        ps.append(np.full(len(v.plan_times),v.target_context[-1]))
        for a in ARMS:xs[a].append(fx[a])
        for p in masks:masks[p].append(flags[p])
        metadata.append(meta)
        if schema is not None:require(schema==sc,'Feature schema changed within episode')
        schema,audit=sc,au
    arrays={k:np.stack(v) for k,v in xs.items()}
    audit={**audit,'origin_first':float(origins[0]),'origin_last':float(origins[-1]),
           'source_start':float(ep.time[0]),'source_end':float(ep.time[-1]),'episode':ep.index,
           'n_origins':len(origins),'horizon_points':len(Y[0]),
           'v1_origin_phase_preserved':True,'origins_removed_incomplete_context':int(len(original_grid)-len(origins)),
           'feature_counts':{a:arrays[a].shape[1] for a in ARMS},
           'group': 'one simulator/one building; episode is not independent building',
           'target_unit':'K','temperature_error_K_equals_degC_difference':True}
    return Prepared(ep.index,arrays,np.stack(Y),origins,np.stack(times),
                    {k:np.asarray(v,bool) for k,v in masks.items()},metadata,schema,audit,np.stack(ps))
