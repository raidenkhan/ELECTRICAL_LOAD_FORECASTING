'use client';

import React, { useState, useEffect, useCallback } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts';
import {
  Activity, BarChart3, TrendingUp, ChevronDown, ChevronUp, Info, Loader2,
} from 'lucide-react';
import { api } from '@/lib/api';

type Horizon = '24h' | '7d' | '30d' | '90d';

const HORIZON_LABELS: Record<Horizon, string> = {
  '24h': 'Day Ahead',
  '7d': '7-Day',
  '30d': '30-Day',
  '90d': '90-Day',
};

const HORIZON_NOTES: Record<Horizon, string> = {
  '24h': 'DLinear+TIDE — 6-fold ensemble (NLinear) + TIDE bias corrector (EMA α=0.3) + per-hour uncertainty (P10/P90)',
  '7d': 'Recursive D+1 calls · TIDE corrector applied only at D+1 · error accumulates (~250 MW at D+7)',
  '30d': 'Recursive rolling · TIDE offline after D+1 · pure DLinear ensemble output · high drift',
  '90d': 'Recursive 90-day · no TIDE correction · ensemble divergence + regime drift',
};

const NOMINAL: Record<Horizon, { mae: number; mape: number; best: number; worst: number }> = {
  '24h': { mae: 67, mape: 2.8, best: 58, worst: 88 },
  '7d': { mae: 251, mape: 7.6, best: 210, worst: 310 },
  '30d': { mae: 302, mape: 9.2, best: 260, worst: 380 },
  '90d': { mae: 420, mape: 12.8, best: 360, worst: 520 },
};

function dowColor(d: string): string {
  return d === 'Sat' || d === 'Sun' ? '#FFBD2E' : '#4B6FFF';
}

export function ModelAccuracyPanel({ horizon: initialHorizon = '24h' }: { horizon?: Horizon }) {
  const [expanded, setExpanded] = useState(false);
  const [horizon, setHorizon] = useState<Horizon>(initialHorizon);
  const [byHour, setByHour] = useState<{ hour: number; mae: number | null }[]>([]);
  const [byDow, setByDow] = useState<{ day: string; mae: number | null }[]>([]);
  const [overview, setOverview] = useState<{ folds: any[]; rolling: any }>({ folds: [], rolling: null });
  const [loading, setLoading] = useState(false);

  useEffect(() => { setHorizon(initialHorizon); }, [initialHorizon]);

  const fetchData = useCallback(async (h: Horizon) => {
    setLoading(true);
    try {
      const [hourRes, dowRes, overviewRes] = await Promise.all([
        api.get(`/forecast/metrics/by-hour?horizon=${h}&window_days=90`),
        api.get(`/forecast/metrics/by-dow?horizon=${h}&window_days=90`),
        api.get(`/forecast/metrics/overview?horizon=${h}`),
      ]);
      setByHour(hourRes.data.by_hour || []);
      setByDow(dowRes.data.by_dow || []);
      setOverview({ folds: overviewRes.data.folds || [], rolling: overviewRes.data.rolling || null });
    } catch {
      setByHour([]);
      setByDow([]);
      setOverview({ folds: [], rolling: null });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { if (expanded) fetchData(horizon); }, [horizon, expanded, fetchData]);

  const nom = NOMINAL[horizon];
  const liveMae = overview.rolling?.mae_avg ?? nom.mae;
  const liveMape = overview.rolling?.mape_avg ?? nom.mape;

  // Build by_hour display: prefer live data, fall back to empty
  const byHourDisplay = byHour.length > 0
    ? byHour.map((h) => ({ h: h.hour, mae: h.mae ?? 0 }))
    : Array.from({ length: 24 }, (_, i) => ({ h: i + 1, mae: 0 }));

  const byDowDisplay = byDow.length > 0
    ? byDow.map((d) => ({ d: d.day, mae: d.mae ?? 0 }))
    : [];

  const foldsDisplay = overview.folds.length > 0 ? overview.folds : [];

  return (
    <div className="panel-card mt-6">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between p-4 hover:bg-[var(--color-surface-alt)]/50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <Activity className="w-4 h-4 text-[var(--color-accent)]" />
          <h3 className="text-sm font-semibold text-[var(--color-text)]">Model Accuracy & Error Insights</h3>
          <span className="eyebrow" style={{ fontSize: 10, padding: '2px 8px' }}>
            {HORIZON_LABELS[horizon]} · MAPE {nom.mape}% · MAE {nom.mae} MW
          </span>
          {loading && <Loader2 className="w-3 h-3 animate-spin text-[var(--color-accent)]" />}
        </div>
        {expanded ? <ChevronUp className="w-4 h-4 text-[var(--color-text-muted)]" /> : <ChevronDown className="w-4 h-4 text-[var(--color-text-muted)]" />}
      </button>

      {expanded && (
        <div className="px-4 pb-4 space-y-5 animate-fade-in">
          <div className="flex gap-1 border-b border-[var(--color-border)] pb-0">
            {(Object.keys(HORIZON_LABELS) as Horizon[]).map((h) => (
              <button
                key={h}
                onClick={() => setHorizon(h)}
                className={`px-3 py-2 text-[11px] font-semibold uppercase tracking-wider transition-colors border-b-2 -mb-px ${
                  horizon === h
                    ? 'border-[var(--color-accent)] text-[var(--color-accent)]'
                    : 'border-transparent text-[var(--color-text-muted)] hover:text-[var(--color-text)]'
                }`}
              >
                {HORIZON_LABELS[h]}
              </button>
            ))}
          </div>

          <div className="flex items-start gap-2 text-[11px] text-[var(--color-text-muted)] bg-[var(--color-surface-alt)] px-3 py-2">
            <Info className="w-3 h-3 mt-0.5 shrink-0 text-[var(--color-accent)]" />
            <span>{HORIZON_NOTES[horizon]}. {overview.rolling?.count > 0 ? `Live rolling: ${overview.rolling.mae_avg?.toFixed(1) ?? nom.mae} MW MAE (n=${overview.rolling.count})` : `Nominal: ${nom.mae} MW MAE`}</span>
          </div>

          <div className="stat-grid">
            <div className="stat-cell">
              <div className="stat-cell-label">Mean MAE</div>
              <div className="stat-cell-val">{Math.round(liveMae)}</div>
              <div className="stat-cell-sub">MW · {liveMape.toFixed(1)}% MAPE {overview.rolling?.count > 0 ? '(live)' : '(nominal)'}</div>
            </div>
            <div className="stat-cell">
              <div className="stat-cell-label">Best Fold</div>
              <div className="stat-cell-val stat-up">{nom.best}</div>
              <div className="stat-cell-sub">MW · training benchmark</div>
            </div>
            <div className="stat-cell">
              <div className="stat-cell-label">Worst Fold</div>
              <div className="stat-cell-val" style={{ color: 'var(--color-error)' }}>{nom.worst}</div>
              <div className="stat-cell-sub">MW · growth discontinuity</div>
            </div>
            <div className="stat-cell">
              <div className="stat-cell-label">Confidence 80%</div>
              <div className="stat-cell-val" style={{ color: 'var(--color-success)' }}>±{Math.round(1.28 * liveMae)}</div>
              <div className="stat-cell-sub">MW · z=1.28</div>
            </div>
          </div>

          <div>
            <h4 className="caption mb-2 flex items-center gap-1.5">
              <BarChart3 className="w-3.5 h-3.5 text-[var(--color-accent)]" />
              MAE by Hour of Day
              {byHour.length === 0 && <span className="text-[9px] text-[var(--color-text-muted)] italic">(nominal — live data pending)</span>}
            </h4>
            <div className="h-40">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={byHourDisplay} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                  <XAxis dataKey="h" tick={{ fontSize: 9, fill: 'var(--color-text-muted)' }} />
                  <YAxis tick={{ fontSize: 9, fill: 'var(--color-text-muted)' }} unit=" MW" />
                  <Tooltip
                    contentStyle={{ background: 'var(--color-surface-card)', border: '1px solid var(--color-border)', fontSize: 11 }}
                    formatter={(v: number) => [`${v} MW`, 'MAE']}
                  />
                  <Bar dataKey="mae" radius={[2, 2, 0, 0]}>
                    {byHourDisplay.map((_, i) => (
                      <Cell key={i} fill={i >= 21 || i < 6 ? '#FFBD2E' : '#4B6FFF'} fillOpacity={0.8} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="caption mb-2 flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5 text-[var(--color-accent)]" />
                MAE by Day of Week
              </h4>
              <div className="h-32">
                {byDowDisplay.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={byDowDisplay} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                      <XAxis dataKey="d" tick={{ fontSize: 10, fill: 'var(--color-text-muted)' }} />
                      <YAxis tick={{ fontSize: 9, fill: 'var(--color-text-muted)' }} unit=" MW" domain={['dataMin - 10', 'dataMax + 10']} />
                      <Tooltip
                        contentStyle={{ background: 'var(--color-surface-card)', border: '1px solid var(--color-border)', fontSize: 11 }}
                        formatter={(v: number) => [`${v} MW`, 'MAE']}
                      />
                      <Bar dataKey="mae" radius={[2, 2, 0, 0]}>
                        {byDowDisplay.map((e) => (
                          <Cell key={e.d} fill={dowColor(e.d)} fillOpacity={0.8} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex items-center justify-center h-full text-[10px] text-[var(--color-text-muted)] italic">No DOW data yet — needs actuals to compare</div>
                )}
              </div>
            </div>

            <div>
              <h4 className="caption mb-2 flex items-center gap-1.5">
                <TrendingUp className="w-3.5 h-3.5 text-[var(--color-accent)]" />
                Per-Fold Performance
              </h4>
              <div className="overflow-x-auto">
                {foldsDisplay.length > 0 ? (
                  <table className="w-full text-[11px]">
                    <thead>
                      <tr className="text-[var(--color-text-muted)] border-b border-[var(--color-border)]">
                        <th className="text-left py-1.5 pr-2">Fold</th>
                        <th className="text-left py-1.5 pr-2">Period</th>
                        <th className="text-right py-1.5 pr-2">MAE</th>
                        <th className="text-right py-1.5 pr-2">MAPE</th>
                        <th className="text-left py-1.5">Context</th>
                      </tr>
                    </thead>
                    <tbody>
                      {foldsDisplay.map((row: any) => (
                        <tr key={row.fold} className="border-b border-[var(--color-border)]/50">
                          <td className="py-1.5 pr-2 font-medium">{row.fold}</td>
                          <td className="py-1.5 pr-2 text-[var(--color-text-muted)]">{row.period}</td>
                          <td className={`py-1.5 pr-2 text-right font-mono ${row.mae >= (nom.worst * 0.85) ? 'text-[var(--color-error)]' : row.mae <= (nom.best * 1.15) ? 'text-[var(--color-success)]' : ''}`}>{row.mae} MW</td>
                          <td className="py-1.5 pr-2 text-right font-mono">{row.mape}%</td>
                          <td className="py-1.5 text-[var(--color-text-muted)]">{row.context}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <div className="flex items-center justify-center h-20 text-[10px] text-[var(--color-text-muted)] italic">No fold data available</div>
                )}
              </div>
            </div>
          </div>

          <div className="flex items-start gap-2 text-[10px] text-[var(--color-text-muted)] bg-[var(--color-surface-alt)] p-2.5">
            <Info className="w-3 h-3 mt-0.5 shrink-0 text-[var(--color-accent)]" />
            <span>
              {HORIZON_LABELS[horizon]} MAE of {Math.round(liveMae)} MW ({liveMape.toFixed(1)}% MAPE) using DLinear+TIDE — 6-fold ensemble of NLinear models
              with TIDE bias corrector (EMA α=0.3). D+1 is the most reliable (67 MW / 2.8% MAPE) with TIDE active.
              Longer horizons use recursive day-by-day prediction without TIDE correction, so error propagates:
              D+7 ~250 MW, D+30 ~300 MW, D+90 ~420 MW. Per-hour uncertainty bands (P10/P90) are computed from
              combined TIDE residual variance and ensemble disagreement. Worst errors coincide with growth discontinuities.
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
