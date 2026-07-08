'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { Loader2, AlertTriangle, RefreshCw } from 'lucide-react';
import { dispatchForecastService, Forecast30DayResponse } from '@/services/dispatchForecastService';
import { baselineForecastService, Baseline30DayResponse } from '@/services/baselineForecastService';

export function Forecast30DayView({ engineType = 'decom' }: { engineType?: 'decom' | 'baseline' }) {
  const [data, setData] = useState<Forecast30DayResponse | Baseline30DayResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async (forceRefresh = false) => {
    setLoading(true);
    setError(null);
    try {
      const result = engineType === 'decom'
        ? await dispatchForecastService.get30Day(forceRefresh)
        : await baselineForecastService.get30Day(forceRefresh);
      setData(result);
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to load 30-day forecast');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, [engineType]);

  const handleRefresh = () => fetchData(true);

  const chartData = useMemo(() => {
    if (!data?.daily_aggregates) return [];
    return data.daily_aggregates.map((d) => ({
      date: d.date,
      peak: Math.round(d.peak_mw),
      mean: Math.round(d.mean_mw),
      min: Math.round(d.min_mw),
    }));
  }, [data]);

  const totalEnergy = data?.daily_aggregates ? Math.round(data.daily_aggregates.reduce((s, d) => s + d.total_energy_mwh, 0)) : 0;
  const avgPeak = data?.daily_aggregates ? Math.round(data.daily_aggregates.reduce((s, d) => s + d.peak_mw, 0) / data.daily_aggregates.length) : 0;
  const avgMean = data?.daily_aggregates ? Math.round(data.daily_aggregates.reduce((s, d) => s + d.mean_mw, 0) / data.daily_aggregates.length) : 0;

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
      <div className="grid grid-cols-3 gap-3">
        <div className="bg-[var(--bg-card)] border border-[var(--border-card)] px-4 py-3">
          <p className="text-[11px] uppercase tracking-[0.8px] text-[var(--text-muted)] font-semibold">Avg Daily Peak</p>
          <p className="text-2xl font-bold text-[var(--status-amber)] mt-1" style={{ fontVariantNumeric: 'tabular-nums' }}>{avgPeak.toLocaleString()} MW</p>
        </div>
        <div className="bg-[var(--bg-card)] border border-[var(--border-card)] px-4 py-3">
          <p className="text-[11px] uppercase tracking-[0.8px] text-[var(--text-muted)] font-semibold">Avg Daily Mean</p>
          <p className="text-2xl font-bold text-[var(--brand-blue)] mt-1" style={{ fontVariantNumeric: 'tabular-nums' }}>{avgMean.toLocaleString()} MW</p>
        </div>
        <div className="bg-[var(--bg-card)] border border-[var(--border-card)] px-4 py-3">
          <p className="text-[11px] uppercase tracking-[0.8px] text-[var(--text-muted)] font-semibold">Total Energy (30d)</p>
          <p className="text-2xl font-bold text-[var(--text-primary)] mt-1" style={{ fontVariantNumeric: 'tabular-nums' }}>{totalEnergy.toLocaleString()} MWh</p>
        </div>
      </div>

      {/* Daily Aggregates Chart */}
      <div className="bg-[var(--bg-card)] border border-[var(--border-card)] p-5">
        <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-4">30-Day Daily Peak &amp; Average</h3>
        <ResponsiveContainer width="100%" height={380}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
            <XAxis dataKey="date" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} interval={4} />
            <YAxis tick={{ fontSize: 12, fill: 'var(--text-muted)' }} domain={['auto', 'auto']} />
            <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-card)', borderRadius: 0, fontSize: 12 }} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="peak" stroke="var(--status-amber)" strokeWidth={2} dot={false} name="Daily Peak" />
            <Line type="monotone" dataKey="mean" stroke="var(--brand-blue)" strokeWidth={2} dot={false} name="Daily Mean" />
            <Line type="monotone" dataKey="min" stroke="var(--status-emerald)" strokeWidth={1.5} dot={false} name="Daily Min" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Aggregated Monthly Stats */}
      <div className="bg-[var(--bg-card)] border border-[var(--border-card)] p-5">
        <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-4">Monthly Summary</h3>
        <div className="grid grid-cols-4 gap-3">
          <div className="px-3 py-2 border border-[var(--border-card)]">
            <p className="text-[11px] text-[var(--text-muted)]">Max Peak</p>
            <p className="text-lg font-bold text-[var(--status-amber)]">{Math.round(Math.max(...data.daily_aggregates.map(d => d.peak_mw))).toLocaleString()} MW</p>
          </div>
          <div className="px-3 py-2 border border-[var(--border-card)]">
            <p className="text-[11px] text-[var(--text-muted)]">Min Trough</p>
            <p className="text-lg font-bold text-[var(--text-primary)]">{Math.round(Math.min(...data.daily_aggregates.map(d => d.min_mw))).toLocaleString()} MW</p>
          </div>
          <div className="px-3 py-2 border border-[var(--border-card)]">
            <p className="text-[11px] text-[var(--text-muted)]">Avg Peak-to-Trough</p>
            <p className="text-lg font-bold text-[var(--brand-blue)]">{Math.round(data.daily_aggregates.reduce((s, d) => s + (d.peak_mw - d.min_mw), 0) / data.daily_aggregates.length).toLocaleString()} MW</p>
          </div>
          <div className="px-3 py-2 border border-[var(--border-card)]">
            <p className="text-[11px] text-[var(--text-muted)]">Days Forecast</p>
            <p className="text-lg font-bold text-[var(--text-primary)]">{data.daily_aggregates.length}</p>
          </div>
        </div>
      </div>

      {/* Daily Table (scrollable) */}
      <div className="bg-[var(--bg-card)] border border-[var(--border-card)] overflow-x-auto max-h-[400px] overflow-y-auto">
        <table className="w-full text-xs border-collapse" style={{ fontVariantNumeric: 'tabular-nums' }}>
          <thead className="sticky top-0 bg-[var(--bg-card)]">
            <tr>
              <th className="text-left px-3 py-2 text-[11px] uppercase tracking-[0.4px] text-[var(--text-muted)] font-semibold border border-[var(--border-card)]">Date</th>
              <th className="text-right px-3 py-2 text-[11px] uppercase tracking-[0.4px] text-[var(--text-muted)] font-semibold border border-[var(--border-card)]">Peak</th>
              <th className="text-right px-3 py-2 text-[11px] uppercase tracking-[0.4px] text-[var(--text-muted)] font-semibold border border-[var(--border-card)]">Mean</th>
              <th className="text-right px-3 py-2 text-[11px] uppercase tracking-[0.4px] text-[var(--text-muted)] font-semibold border border-[var(--border-card)]">Min</th>
              <th className="text-right px-3 py-2 text-[11px] uppercase tracking-[0.4px] text-[var(--text-muted)] font-semibold border border-[var(--border-card)]">Energy</th>
            </tr>
          </thead>
          <tbody>
            {data.daily_aggregates.map((d, i) => (
              <tr key={d.date} className={i % 2 === 0 ? 'bg-[var(--surface-primary)]' : ''}>
                <td className="px-3 py-1.5 font-medium text-[var(--text-primary)] border border-[var(--border-card)]">{d.date}</td>
                <td className="px-3 py-1.5 text-right text-[var(--status-amber)] border border-[var(--border-card)]">{Math.round(d.peak_mw).toLocaleString()}</td>
                <td className="px-3 py-1.5 text-right text-[var(--brand-blue)] border border-[var(--border-card)]">{Math.round(d.mean_mw).toLocaleString()}</td>
                <td className="px-3 py-1.5 text-right text-[var(--text-secondary)] border border-[var(--border-card)]">{Math.round(d.min_mw).toLocaleString()}</td>
                <td className="px-3 py-1.5 text-right text-[var(--text-primary)] border border-[var(--border-card)]">{Math.round(d.total_energy_mwh).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
