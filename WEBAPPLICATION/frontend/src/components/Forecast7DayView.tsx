'use client';

import React, { useState, useEffect, useMemo } from 'react';
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, BarChart, Bar,
} from 'recharts';
import { Loader2, AlertTriangle, RefreshCw, TrendingUp, ChevronLeft } from 'lucide-react';
import { toast } from 'sonner';
import { dispatchForecastService, Forecast7DayResponse } from '@/services/dispatchForecastService';
import { baselineForecastService, Baseline7DayResponse } from '@/services/baselineForecastService';

const DAY_COLORS = ['#4B6FFF', '#22D47A', '#FFBD2E', '#FF5F57', '#8B5CF6', '#EC4899', '#06B6D4'];
const DAY_LABELS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const DAY_MAE = [67, 89, 112, 136, 161, 187, 214];

function formatHour(h: number) {
  return `${h.toString().padStart(2, '0')}:00`;
}

function getDayName(dateStr: string): string {
  const d = new Date(dateStr);
  return DAY_LABELS[d.getDay()] || '';
}

export function Forecast7DayView({ engineType = 'decom' }: { engineType?: 'decom' | 'baseline' }) {
  const [data, setData] = useState<Forecast7DayResponse | Baseline7DayResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedDay, setSelectedDay] = useState<number | null>(null);
  const [chartMode, setChartMode] = useState<'overlay' | 'bars' | 'table'>('overlay');

  const fetchData = async (forceRefresh = false) => {
    setLoading(true);
    setError(null);
    try {
      const result = engineType === 'decom'
        ? await dispatchForecastService.get7Day(forceRefresh)
        : await baselineForecastService.get7Day(forceRefresh);
      setData(result);
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to load 7-day forecast');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, [engineType]);

  const handleRefresh = () => fetchData(true);

  const overlayData = useMemo(() => {
    if (!data || !data.hourly_mw.length) return [];
    const days: { hour: string; [key: string]: string | number }[] = [];
    for (let d = 0; d < 7; d++) {
      const dayKey = `day${d}`;
      for (let h = 0; h < 24; h++) {
        const idx = d * 24 + h;
        if (idx >= data.hourly_mw.length) break;
        if (!days[h]) days[h] = { hour: formatHour(h + 1) };
        days[h][dayKey] = Math.round(data.hourly_mw[idx]);
      }
    }
    return days;
  }, [data]);

  const singleDayData = useMemo(() => {
    if (selectedDay === null || !data?.hourly_mw) return [];
    const mae = DAY_MAE[selectedDay] || 67;
    const band80 = Math.round(1.28 * mae);
    const band95 = Math.round(1.96 * mae);
    const useP10P90 = 'p10_mw' in data && 'p90_mw' in data && data.p10_mw && data.p90_mw;
    const result: { hour: string; value: number; p10: number; p90: number; p5: number; p95: number; band80: [number, number]; band95: [number, number] }[] = [];
    for (let h = 0; h < 24; h++) {
      const idx = selectedDay * 24 + h;
      if (idx >= data.hourly_mw.length) break;
      const val = Math.round(data.hourly_mw[idx]);
      const p10Val = useP10P90 && data.p10_mw ? Math.round(data.p10_mw[idx]) : val - band80;
      const p90Val = useP10P90 && data.p90_mw ? Math.round(data.p90_mw[idx]) : val + band80;
      result.push({
        hour: formatHour(h + 1),
        value: val,
        p10: p10Val,
        p90: p90Val,
        p5: val - band95,
        p95: val + band95,
        band80: [p10Val, p90Val],
        band95: [val - band95, val + band95],
      });
    }
    return result;
  }, [data, selectedDay]);

  const dailyChartData = useMemo(() => {
    if (!data?.daily_aggregates) return [];
    return data.daily_aggregates.map((d, i) => ({
      dayLabel: DAY_LABELS[i % 7],
      date: d.date,
      peak: Math.round(d.peak_mw),
      mean: Math.round(d.mean_mw),
      min: Math.round(d.min_mw),
      mae: DAY_MAE[i % 7],
    }));
  }, [data]);

  const totalEnergy = data?.daily_aggregates ? Math.round(data.daily_aggregates.reduce((s, d) => s + d.total_energy_mwh, 0)) : 0;
  const avgPeak = data?.daily_aggregates ? Math.round(data.daily_aggregates.reduce((s, d) => s + d.peak_mw, 0) / data.daily_aggregates.length) : 0;
  const avgMean = data?.daily_aggregates ? Math.round(data.daily_aggregates.reduce((s, d) => s + d.mean_mw, 0) / data.daily_aggregates.length) : 0;

  const toggleDay = (dayIdx: number) => {
    setSelectedDay(prev => prev === dayIdx ? null : dayIdx);
  };

  if (loading) {
    return <div className="flex items-center justify-center py-20"><Loader2 className="w-6 h-6 text-[var(--color-accent)] animate-spin" /></div>;
  }

  if (error) {
    return <div className="flex flex-col items-center justify-center py-20 gap-4">
      <AlertTriangle className="w-10 h-10 text-[var(--color-error)]" />
      <p className="text-sm text-[var(--color-text-muted)]">{error}</p>
      <button onClick={handleRefresh} className="px-4 py-2 text-xs font-semibold border border-[var(--color-border)] text-[var(--color-text-muted)] hover:bg-[var(--color-surface-alt)] transition-colors"><RefreshCw className="w-3.5 h-3.5 inline mr-1.5" /> Refresh</button>
    </div>;
  }

  if (!data) return null;

  return (
    <div className="flex flex-col gap-4 animate-fade-in">
      {/* Summary Cards (bento-style stat grid) */}
      <div className="stat-grid">
        <div className="stat-cell">
          <div className="stat-cell-label">Avg Daily Peak</div>
          <div className="stat-cell-val">{avgPeak.toLocaleString()}</div>
          <div className="stat-cell-sub stat-up">7-day forecast</div>
        </div>
        <div className="stat-cell">
          <div className="stat-cell-label">Avg Daily Mean</div>
          <div className="stat-cell-val">{avgMean.toLocaleString()}</div>
          <div className="stat-cell-sub">MW</div>
        </div>
        <div className="stat-cell">
          <div className="stat-cell-label">Total Energy</div>
          <div className="stat-cell-val">{totalEnergy.toLocaleString()}</div>
          <div className="stat-cell-sub">MWh (7 days)</div>
        </div>
        <div className="stat-cell">
          <div className="stat-cell-label">Horizon MAE</div>
          <div className="stat-cell-val">{DAY_MAE[6]} MW</div>
          <div className="stat-cell-sub">Day 7 · expands daily</div>
        </div>
      </div>

      {/* Chart Section */}
      <div className="panel-card p-0">
        <div className="flex items-center justify-between px-5 pt-4 pb-2">
          <div className="flex items-center gap-3">
            {selectedDay !== null ? (
              <button onClick={() => setSelectedDay(null)} className="btn-ghost p-1">
                <ChevronLeft className="w-4 h-4" />
              </button>
            ) : null}
            <h3 className="text-sm font-semibold text-[var(--color-text)]">
              {selectedDay !== null
                ? `Day ${selectedDay + 1}: ${getDayName(data.daily_aggregates[selectedDay]?.date || '')} — ${data.daily_aggregates[selectedDay]?.date || ''}`
                : '7-Day Hourly Profiles'}
            </h3>
          </div>
          <div className="flex gap-1">
            {selectedDay === null && (
              <>
                <button
                  onClick={() => setChartMode('overlay')}
                  className={`px-2 py-1 text-[10px] font-semibold uppercase tracking-wider transition-colors ${chartMode === 'overlay' ? 'bg-[var(--color-accent)] text-white' : 'text-[var(--color-text-muted)] hover:text-[var(--color-text)]'}`}
                >Overlay</button>
                <button
                  onClick={() => setChartMode('bars')}
                  className={`px-2 py-1 text-[10px] font-semibold uppercase tracking-wider transition-colors ${chartMode === 'bars' ? 'bg-[var(--color-accent)] text-white' : 'text-[var(--color-text-muted)] hover:text-[var(--color-text)]'}`}
                >Bars</button>
                <button
                  onClick={() => setChartMode('table')}
                  className={`px-2 py-1 text-[10px] font-semibold uppercase tracking-wider transition-colors ${chartMode === 'table' ? 'bg-[var(--color-accent)] text-white' : 'text-[var(--color-text-muted)] hover:text-[var(--color-text)]'}`}
                >Table</button>
              </>
            )}
          </div>
        </div>

        {selectedDay !== null ? (
          /* Single day with uncertainty bands */
          <div className="px-5 pb-5">
            <ResponsiveContainer width="100%" height={380}>
              <AreaChart data={singleDayData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis dataKey="hour" tick={{ fontSize: 11, fill: 'var(--color-text-muted)' }} />
                <YAxis tick={{ fontSize: 11, fill: 'var(--color-text-muted)' }} domain={['auto', 'auto']} />
                <Tooltip
                  contentStyle={{ background: 'var(--color-surface-card)', border: '1px solid var(--color-border)', borderRadius: 0, fontSize: 12 }}
                  formatter={(v: number) => [`${v.toLocaleString()} MW`, undefined]}
                />
                <Area type="monotone" dataKey="band95" stroke="none" fill="#4B6FFF" fillOpacity={0.12} legendType="none" />
                <Area type="monotone" dataKey="band80" stroke="none" fill="#4B6FFF" fillOpacity={0.35} legendType="none" />
                <Line type="monotone" dataKey="value" stroke={DAY_COLORS[selectedDay]} strokeWidth={2.5} dot={{ r: 3 }} name="Forecast" />
              </AreaChart>
            </ResponsiveContainer>
            <div className="flex gap-4 text-[10px] text-[var(--color-text-muted)] mt-2">
              <span><span className="inline-block w-3 h-[2px] bg-[var(--color-accent)] mr-1" style={{ opacity: 0.2 }} /> 95% CI (±{Math.round(1.96 * DAY_MAE[selectedDay])} MW)</span>
              <span><span className="inline-block w-3 h-[2px] bg-[var(--color-accent)] mr-1" style={{ opacity: 0.4 }} /> 80% CI (±{Math.round(1.28 * DAY_MAE[selectedDay])} MW)</span>
              <span className="ml-auto">MAE: {DAY_MAE[selectedDay]} MW</span>
            </div>
          </div>
        ) : chartMode === 'overlay' ? (
          /* Overlaid 7-day profiles with click-to-select */
          <div className="px-5 pb-5">
            <ResponsiveContainer width="100%" height={380}>
              <LineChart data={overlayData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis dataKey="hour" tick={{ fontSize: 11, fill: 'var(--color-text-muted)' }} />
                <YAxis tick={{ fontSize: 11, fill: 'var(--color-text-muted)' }} domain={['auto', 'auto']} />
                <Tooltip
                  contentStyle={{ background: 'var(--color-surface-card)', border: '1px solid var(--color-border)', borderRadius: 0, fontSize: 12 }}
                />
                <Legend
                  wrapperStyle={{ fontSize: 11, cursor: 'pointer' }}
                  onClick={(e) => {
                    const idx = DAY_LABELS.indexOf(e.value as string);
                    if (idx >= 0) toggleDay(idx);
                  }}
                />
                {Array.from({ length: 7 }, (_, i) => (
                  <Line
                    key={i}
                    type="monotone"
                    dataKey={`day${i}`}
                    stroke={DAY_COLORS[i]}
                    strokeWidth={1.5}
                    dot={false}
                    name={DAY_LABELS[i]}
                    activeDot={{ r: 4, onClick: () => toggleDay(i) }}
                    style={{ cursor: 'pointer' }}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
            <p className="text-[10px] text-[var(--color-text-faint)] mt-2 text-center">Click a day in the legend or on a line to isolate it with uncertainty bands</p>
          </div>
        ) : chartMode === 'bars' ? (
          /* Daily bar chart */
          <div className="px-5 pb-5">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={dailyChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis dataKey="dayLabel" tick={{ fontSize: 11, fill: 'var(--color-text-muted)' }} />
                <YAxis tick={{ fontSize: 11, fill: 'var(--color-text-muted)' }} domain={['auto', 'auto']} />
                <Tooltip
                  contentStyle={{ background: 'var(--color-surface-card)', border: '1px solid var(--color-border)', borderRadius: 0, fontSize: 12 }}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="peak" fill="var(--color-warning)" radius={[2, 2, 0, 0]} name="Peak" />
                <Bar dataKey="mean" fill="var(--color-accent)" radius={[2, 2, 0, 0]} name="Mean" />
                <Bar dataKey="min" fill="var(--color-success)" radius={[2, 2, 0, 0]} name="Min" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          /* Hourly data table */
          <div className="px-5 pb-5 overflow-x-auto">
            <table className="w-full text-xs border-collapse" style={{ fontVariantNumeric: 'tabular-nums', minWidth: 600 }}>
              <thead>
                <tr>
                  <th className="text-left px-2 py-1.5 text-[11px] uppercase tracking-[0.4px] text-[var(--color-text-muted)] font-semibold border border-[var(--color-border)]">Hour</th>
                  {data.daily_aggregates.map((d, i) => (
                    <th key={i} className="px-2 py-1.5 text-center text-[11px] uppercase tracking-[0.4px] text-[var(--color-text-muted)] font-semibold border border-[var(--color-border)]" style={{ color: DAY_COLORS[i] }}>{d.date.slice(5)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Array.from({ length: 24 }, (_, h) => {
                  const vals = Array.from({ length: 7 }, (_, d) => {
                    const idx = d * 24 + h;
                    return idx < data.hourly_mw.length ? Math.round(data.hourly_mw[idx]) : null;
                  });
                  const maxVal = Math.max(...vals.filter((v): v is number => v !== null));
                  return (
                    <tr key={h} className="border border-[var(--color-border)] hover:bg-[var(--color-surface-alt)] transition-colors">
                      <td className="px-2 py-1 font-medium text-[var(--color-text)] border border-[var(--color-border)]">{formatHour(h + 1)}</td>
                      {vals.map((v, d) => (
                        <td key={d} className={`px-2 py-1 text-right font-mono border border-[var(--color-border)] ${v === maxVal ? 'text-[var(--color-warning)] font-semibold' : 'text-[var(--color-text)]'}`}>
                          {v !== null ? v.toLocaleString() : '-'}
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="text-[10px] text-[var(--color-text-faint)] mt-2">Daily peak hour highlighted in amber · values in MW</p>
          </div>
        )}
      </div>

      {/* Daily Table */}
      <div className="panel-card overflow-x-auto">
        <table className="w-full text-xs border-collapse" style={{ fontVariantNumeric: 'tabular-nums' }}>
          <thead>
            <tr>
              <th className="text-left px-3 py-2 text-[11px] uppercase tracking-[0.4px] text-[var(--color-text-muted)] font-semibold border border-[var(--color-border)]">Date</th>
              <th className="text-left px-3 py-2 text-[11px] uppercase tracking-[0.4px] text-[var(--color-text-muted)] font-semibold border border-[var(--color-border)]">Day</th>
              <th className="text-right px-3 py-2 text-[11px] uppercase tracking-[0.4px] text-[var(--color-text-muted)] font-semibold border border-[var(--color-border)]">Peak (MW)</th>
              <th className="text-right px-3 py-2 text-[11px] uppercase tracking-[0.4px] text-[var(--color-text-muted)] font-semibold border border-[var(--color-border)]">Mean (MW)</th>
              <th className="text-right px-3 py-2 text-[11px] uppercase tracking-[0.4px] text-[var(--color-text-muted)] font-semibold border border-[var(--color-border)]">Min (MW)</th>
              <th className="text-right px-3 py-2 text-[11px] uppercase tracking-[0.4px] text-[var(--color-text-muted)] font-semibold border border-[var(--color-border)]">Energy (MWh)</th>
              <th className="text-right px-3 py-2 text-[11px] uppercase tracking-[0.4px] text-[var(--color-text-muted)] font-semibold border border-[var(--color-border)]">MAE est.</th>
            </tr>
          </thead>
          <tbody>
            {data.daily_aggregates.map((d, i) => {
              const isSelected = selectedDay === i;
              return (
                <tr
                  key={d.date}
                  onClick={() => toggleDay(i)}
                  className={`border border-[var(--color-border)] transition-colors cursor-pointer ${isSelected ? 'bg-[var(--color-accent-muted)]' : i % 2 === 0 ? 'bg-[var(--color-surface)]' : ''} hover:bg-[var(--color-surface-alt)]`}
                >
                  <td className="px-3 py-2 font-medium text-[var(--color-text)] border border-[var(--color-border)]">{d.date}</td>
                  <td className="px-3 py-2 text-[var(--color-text-muted)] border border-[var(--color-border)]">{DAY_LABELS[i % 7]}</td>
                  <td className="px-3 py-2 text-right font-mono text-[var(--color-warning)] border border-[var(--color-border)]">{Math.round(d.peak_mw).toLocaleString()}</td>
                  <td className="px-3 py-2 text-right font-mono text-[var(--color-accent)] border border-[var(--color-border)]">{Math.round(d.mean_mw).toLocaleString()}</td>
                  <td className="px-3 py-2 text-right font-mono text-[var(--color-text-muted)] border border-[var(--color-border)]">{Math.round(d.min_mw).toLocaleString()}</td>
                  <td className="px-3 py-2 text-right font-mono text-[var(--color-text)] border border-[var(--color-border)]">{Math.round(d.total_energy_mwh).toLocaleString()}</td>
                  <td className="px-3 py-2 text-right font-mono text-[var(--color-text-faint)] border border-[var(--color-border)]">±{DAY_MAE[i]} MW</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
