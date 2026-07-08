'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, LineChart, Line } from 'recharts';
import { Loader2, AlertTriangle, RefreshCw, TrendingUp, TrendingDown } from 'lucide-react';
import { dispatchForecastService, Forecast90DayResponse } from '@/services/dispatchForecastService';
import { baselineForecastService, Baseline90DayResponse } from '@/services/baselineForecastService';

export function Forecast90DayView({ engineType = 'decom' }: { engineType?: 'decom' | 'baseline' }) {
  const [data, setData] = useState<Forecast90DayResponse | Baseline90DayResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async (forceRefresh = false) => {
    setLoading(true);
    setError(null);
    try {
      const result = engineType === 'decom'
        ? await dispatchForecastService.get90Day(forceRefresh)
        : await baselineForecastService.get90Day(forceRefresh);
      setData(result);
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to load 90-day forecast');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, [engineType]);

  const handleRefresh = () => fetchData(true);

  const chartData = useMemo(() => {
    if (!data?.weekly_aggregates) return [];
    return data.weekly_aggregates.map((w) => ({
      label: w.week_start.slice(5),
      week: `${w.week_start.slice(5)}-${w.week_end.slice(5)}`,
      mean: Math.round(w.mean_mw),
      peak: Math.round(w.peak_mw),
      min: Math.round(w.min_mw),
    }));
  }, [data]);

  const firstWeek = data?.weekly_aggregates?.[0];
  const lastWeek = data?.weekly_aggregates?.[data.weekly_aggregates.length - 1];
  const trend = firstWeek && lastWeek ? lastWeek.mean_mw - firstWeek.mean_mw : 0;
  const totalEnergy = data?.weekly_aggregates ? Math.round(data.weekly_aggregates.reduce((s, w) => s + w.total_energy_mwh, 0)) : 0;

  if (loading) {
    return <div className="flex items-center justify-center py-20"><Loader2 className="w-6 h-6 text-[var(--brand-blue)] animate-spin" /></div>;
  }

  if (error) {
    return <div className="flex flex-col items-center justify-center py-20 gap-4">
      <AlertTriangle className="w-10 h-10 text-[var(--status-crimson)]" />
      <p className="text-sm text-[var(--text-secondary)]">{error}</p>
      <button onClick={handleRefresh} className="px-4 py-2 text-xs font-semibold border border-[var(--border-card)] text-[var(--text-secondary)] hover:bg-[var(--surface-secondary)] transition-colors"><RefreshCw className="w-3.5 h-3.5 inline mr-1.5" /> Refresh</button>
    </div>;
  }

  if (!data) return null;

  return (
    <div className="flex flex-col gap-4">
      {/* Summary Cards */}
      <div className="grid grid-cols-4 gap-3">
        <div className="bg-[var(--bg-card)] border border-[var(--border-card)] px-4 py-3">
          <p className="text-[11px] uppercase tracking-[0.8px] text-[var(--text-muted)] font-semibold">Start Week Avg</p>
          <p className="text-2xl font-bold text-[var(--text-primary)] mt-1" style={{ fontVariantNumeric: 'tabular-nums' }}>{firstWeek ? Math.round(firstWeek.mean_mw).toLocaleString() : '-'} MW</p>
        </div>
        <div className="bg-[var(--bg-card)] border border-[var(--border-card)] px-4 py-3">
          <p className="text-[11px] uppercase tracking-[0.8px] text-[var(--text-muted)] font-semibold">End Week Avg</p>
          <p className="text-2xl font-bold text-[var(--text-primary)] mt-1" style={{ fontVariantNumeric: 'tabular-nums' }}>{lastWeek ? Math.round(lastWeek.mean_mw).toLocaleString() : '-'} MW</p>
        </div>
        <div className="bg-[var(--bg-card)] border border-[var(--border-card)] px-4 py-3">
          <p className="text-[11px] uppercase tracking-[0.8px] text-[var(--text-muted)] font-semibold">Trend</p>
          <div className="flex items-center gap-1.5 mt-1">
            {trend > 0 ? <TrendingUp className="w-5 h-5 text-[var(--status-amber)]" /> : <TrendingDown className="w-5 h-5 text-[var(--status-crimson)]" />}
            <p className={`text-2xl font-bold ${trend > 0 ? 'text-[var(--status-amber)]' : 'text-[var(--status-crimson)]'}`} style={{ fontVariantNumeric: 'tabular-nums' }}>
              {trend > 0 ? '+' : ''}{Math.round(trend).toLocaleString()} MW
            </p>
          </div>
        </div>
        <div className="bg-[var(--bg-card)] border border-[var(--border-card)] px-4 py-3">
          <p className="text-[11px] uppercase tracking-[0.8px] text-[var(--text-muted)] font-semibold">Total Energy (90d)</p>
          <p className="text-2xl font-bold text-[var(--text-primary)] mt-1" style={{ fontVariantNumeric: 'tabular-nums' }}>{totalEnergy.toLocaleString()} MWh</p>
        </div>
      </div>

      {/* Weekly Trend Chart */}
      <div className="bg-[var(--bg-card)] border border-[var(--border-card)] p-5">
        <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-4">13-Week Trend &mdash; Weekly Average Demand</h3>
        <ResponsiveContainer width="100%" height={380}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
            <XAxis dataKey="label" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
            <YAxis tick={{ fontSize: 12, fill: 'var(--text-muted)' }} domain={['auto', 'auto']} />
            <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-card)', borderRadius: 0, fontSize: 12 }} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="peak" stroke="var(--status-amber)" strokeWidth={2} dot={{ r: 3 }} name="Weekly Peak" />
            <Line type="monotone" dataKey="mean" stroke="var(--brand-blue)" strokeWidth={2.5} dot={{ r: 3 }} name="Weekly Mean" />
            <Line type="monotone" dataKey="min" stroke="var(--status-emerald)" strokeWidth={2} dot={{ r: 3 }} name="Weekly Min" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Weekly Table */}
      <div className="bg-[var(--bg-card)] border border-[var(--border-card)] overflow-x-auto">
        <table className="w-full text-xs border-collapse" style={{ fontVariantNumeric: 'tabular-nums' }}>
          <thead>
            <tr>
              <th className="text-left px-3 py-2 text-[11px] uppercase tracking-[0.4px] text-[var(--text-muted)] font-semibold border border-[var(--border-card)]">Week</th>
              <th className="text-right px-3 py-2 text-[11px] uppercase tracking-[0.4px] text-[var(--text-muted)] font-semibold border border-[var(--border-card)]">Mean (MW)</th>
              <th className="text-right px-3 py-2 text-[11px] uppercase tracking-[0.4px] text-[var(--text-muted)] font-semibold border border-[var(--border-card)]">Peak (MW)</th>
              <th className="text-right px-3 py-2 text-[11px] uppercase tracking-[0.4px] text-[var(--text-muted)] font-semibold border border-[var(--border-card)]">Min (MW)</th>
              <th className="text-right px-3 py-2 text-[11px] uppercase tracking-[0.4px] text-[var(--text-muted)] font-semibold border border-[var(--border-card)]">Energy (MWh)</th>
            </tr>
          </thead>
          <tbody>
            {data.weekly_aggregates.map((w, i) => (
              <tr key={w.week_start} className={i % 2 === 0 ? 'bg-[var(--surface-primary)]' : ''}>
                <td className="px-3 py-2 font-medium text-[var(--text-primary)] border border-[var(--border-card)]">{w.week_start.slice(5)}-{w.week_end.slice(5)}</td>
                <td className="px-3 py-2 text-right text-[var(--brand-blue)] border border-[var(--border-card)]">{Math.round(w.mean_mw).toLocaleString()}</td>
                <td className="px-3 py-2 text-right text-[var(--status-amber)] border border-[var(--border-card)]">{Math.round(w.peak_mw).toLocaleString()}</td>
                <td className="px-3 py-2 text-right text-[var(--text-secondary)] border border-[var(--border-card)]">{Math.round(w.min_mw).toLocaleString()}</td>
                <td className="px-3 py-2 text-right text-[var(--text-primary)] border border-[var(--border-card)]">{Math.round(w.total_energy_mwh).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
