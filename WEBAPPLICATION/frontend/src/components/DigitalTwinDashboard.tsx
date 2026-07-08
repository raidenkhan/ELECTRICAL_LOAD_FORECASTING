'use client';

import React, { useState, useEffect, useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area, ComposedChart, Line, Legend,
} from 'recharts';
import {
  Zap, Activity, TrendingUp, Calendar, Thermometer, Clock,
  ArrowUp, ArrowDown, Loader2, AlertTriangle, Layers, Building2,
} from 'lucide-react';
import { dispatchForecastService, DispatchForecastResponse } from '@/services/dispatchForecastService';
import { scheduleService, ScheduleDetail } from '@/services/scheduleService';

function formatHour(h: number) {
  return `${h.toString().padStart(2, '0')}:00`;
}

const ENTITY_COLORS: Record<string, string> = {
  ECG: '#58a6ff',
  NEDCo: '#3fb950',
  VALCO: '#facc15',
  Mines: '#f85149',
  Export: '#bc8cff',
};

const ENTITY_LABELS: Record<string, string> = {
  ECG: 'ECG Demand',
  NEDCo: 'NEDCo',
  VALCO: 'VALCO',
  Mines: 'Mines',
  Export: 'Export',
};

export function DigitalTwinDashboard() {
  const [forecast, setForecast] = useState<DispatchForecastResponse | null>(null);
  const [schedule, setSchedule] = useState<ScheduleDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [chartMode, setChartMode] = useState<'stacked' | 'entities' | 'total'>('stacked');

  useEffect(() => {
    async function fetch() {
      try {
        setLoading(true);
        const [forecastData, scheduleData] = await Promise.all([
          dispatchForecastService.getTomorrow().catch(() => null),
          scheduleService.getLatestSchedule().catch(() => null),
        ]);
        setForecast(forecastData);
        setSchedule(scheduleData);
      } catch (err: any) {
        setError(err?.response?.data?.detail || 'Failed to load data');
      } finally {
        setLoading(false);
      }
    }
    fetch();
  }, []);

  const entityNames = useMemo(() => {
    if (!schedule) return [];
    return [...new Set(schedule.demand.map(d => d.entity_name))]
      .filter(e => e !== 'NITS_Total');
  }, [schedule]);

  const scheduleData = useMemo(() => {
    if (!schedule) return null;
    const entities = ['ECG', 'NEDCo', 'VALCO', 'Mines', 'Export'].filter(e => entityNames.includes(e));
    if (entities.length === 0) return null;

    // Build per-entity per-hour map
    const entityMap: Record<string, Record<number, number>> = {};
    for (const e of entities) entityMap[e] = {};

    for (const d of schedule.demand) {
      if (entityMap[d.entity_name]) {
        entityMap[d.entity_name][d.hour] = d.demand_mw;
      }
    }

    const nitsTotal: Record<number, number> = {};
    for (const d of schedule.demand) {
      if (d.entity_name === 'NITS_Total') {
        nitsTotal[d.hour] = d.demand_mw;
      }
    }

    return Array.from({ length: 24 }, (_, i) => {
      const h = i + 1;
      const point: Record<string, any> = {
        hour: formatHour(h),
        hourNum: h,
      };
      for (const e of entities) {
        point[e] = entityMap[e][h] ?? 0;
      }
      point.NITS_Total = nitsTotal[h] ?? 0;
      return point;
    });
  }, [schedule, entityNames]);

  const combinedChartData = useMemo(() => {
    if (scheduleData) return scheduleData;

    // Fallback: use forecast data only
    if (!forecast) return [];
    return forecast.forecast_mw.map((mw, i) => {
      const h = i + 1;
      return {
        hour: formatHour(h),
        hourNum: h,
        ECG: Math.round(mw),
      };
    });
  }, [forecast, scheduleData]);

  const peak = useMemo(() => {
    if (combinedChartData.length === 0) return null;
    return combinedChartData.reduce((a: any, b: any) => {
      const valA = a.NITS_Total || a.ECG || 0;
      const valB = (b as any).NITS_Total || (b as any).ECG || 0;
      return valA > valB ? a : b;
    });
  }, [combinedChartData]);

  const totalDayDemand = useMemo(() => {
    if (combinedChartData.length === 0) return 0;
    return combinedChartData.reduce((s, d: any) => s + (d.NITS_Total || d.ECG || 0), 0);
  }, [combinedChartData]);

  const avgDemand = useMemo(() => {
    if (combinedChartData.length === 0) return 0;
    return Math.round(totalDayDemand / combinedChartData.length);
  }, [combinedChartData, totalDayDemand]);

  const supplyEntities = useMemo(() => {
    if (!schedule) return [];
    return [...new Set(schedule.supply.map(s => s.plant_name))];
  }, [schedule]);

  const supplyChartData = useMemo(() => {
    if (!schedule) return null;
    return Array.from({ length: 24 }, (_, i) => {
      const h = i + 1;
      const point: Record<string, any> = { hour: formatHour(h), hourNum: h };
      for (const plant of supplyEntities) {
        point[plant] = schedule.supply.filter(s => s.plant_name === plant && s.hour === h).reduce((a, b) => a + b.supply_mw, 0);
      }
      return point;
    });
  }, [schedule, supplyEntities]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="w-8 h-8 text-[var(--brand-blue)] animate-spin" />
          <span className="text-sm font-semibold text-[var(--text-muted)] uppercase tracking-widest">Loading Control Room...</span>
        </div>
      </div>
    );
  }

  if (error && !forecast && !schedule) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="flex flex-col items-center gap-4 max-w-md text-center">
          <AlertTriangle className="w-10 h-10 text-[var(--status-crimson)]" />
          <p className="text-sm font-bold text-[var(--status-crimson)]">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 text-xs font-bold border border-[var(--border-card)] text-[var(--text-secondary)] hover:bg-[var(--surface-secondary)] transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const hasFullSchedule = scheduleData !== null && scheduleData.length > 0;

  return (
    <div className="flex flex-col gap-5 h-full">

      {/* Top Bar */}
      <div className="flex items-center justify-between px-4 py-3 bg-[var(--bg-card)] border border-[var(--border-card)]">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-[var(--status-amber)]" />
            <h1 className="text-base font-bold text-[var(--text-primary)]">Control Room</h1>
          </div>
          <span className="text-xs text-[var(--text-muted)]">|</span>
          <span className="text-xs text-[var(--text-secondary)]">
            <Calendar className="w-3.5 h-3.5 inline mr-1" />
            {schedule?.date ?? forecast?.forecast_date ?? '—'}
          </span>
        </div>
        <div className="flex items-center gap-3">
          {hasFullSchedule && (
            <span className="flex items-center gap-1.5 text-xs text-[var(--status-emerald)]">
              <span className="w-2 h-2 rounded-full bg-[var(--status-emerald)] animate-pulse" />
              Schedule Loaded
            </span>
          )}
          <span className="text-xs px-2 py-1 border border-[var(--border-card)] text-[var(--text-muted)]">
            24-Hour Dispatch
          </span>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-4 gap-3">
        {[
          {
            label: 'Peak Demand',
            value: peak ? Math.round((peak as any).NITS_Total || (peak as any).ECG || 0).toLocaleString() : '—',
            unit: 'MW',
            sub: peak ? `Hour ${peak.hourNum} (${peak.hour})` : '—',
            accent: 'var(--status-amber)',
          },
          {
            label: 'Total Energy',
            value: Math.round(totalDayDemand).toLocaleString(),
            unit: 'MWh',
            sub: '24-hour total',
            accent: 'var(--brand-blue)',
          },
          {
            label: 'Average Load',
            value: avgDemand.toLocaleString(),
            unit: 'MW',
            sub: '24h mean',
            accent: 'var(--text-primary)',
          },
          {
            label: 'Entities',
            value: entityNames.length > 0 ? entityNames.length.toString() : (forecast ? '1' : '—'),
            unit: entityNames.length > 0 ? 'demand groups' : 'ECG only',
            sub: schedule ? `${supplyEntities.length} supply sources` : '',
            accent: 'var(--status-emerald)',
          },
        ].map((kpi, i) => (
          <div key={i} className="bg-[var(--bg-card)] border border-[var(--border-card)] px-4 py-3">
            <p className="text-[11px] uppercase tracking-[0.8px] text-[var(--text-muted)] font-semibold">{kpi.label}</p>
            <p className="text-2xl font-bold mt-1" style={{ color: kpi.accent, fontVariantNumeric: 'tabular-nums' }}>
              {kpi.value} <span className="text-sm font-semibold text-[var(--text-muted)]">{kpi.unit}</span>
            </p>
            <p className="text-[11px] text-[var(--text-muted)] mt-0.5">{kpi.sub}</p>
          </div>
        ))}
      </div>

      {/* Main Content */}
      <div className="flex flex-1 gap-5 min-h-[400px]">

        {/* LEFT PANEL: Snapshot */}
        <div className="w-[240px] flex-shrink-0 flex flex-col gap-4">
          {hasFullSchedule && entityNames.length > 0 && (
            <div className="bg-[var(--bg-card)] border border-[var(--border-card)] p-4">
              <h3 className="text-[11px] font-bold text-[var(--text-primary)] uppercase tracking-widest mb-3 flex items-center gap-2">
                <Building2 className="w-3.5 h-3.5 text-[var(--brand-blue)]" /> Demand Mix
              </h3>
              <div className="space-y-2">
                {entityNames.map((name) => {
                  const total = schedule!.demand
                    .filter(d => d.entity_name === name)
                    .reduce((a, b) => a + b.demand_mw, 0);
                  const avg = Math.round(total / 24);
                  const share = totalDayDemand > 0 ? Math.round((total / totalDayDemand) * 100) : 0;
                  return (
                    <div key={name} className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-1.5">
                        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: ENTITY_COLORS[name] || 'var(--text-muted)' }} />
                        <span className="text-[var(--text-secondary)]">{ENTITY_LABELS[name] || name}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-[var(--text-primary)]" style={{ fontVariantNumeric: 'tabular-nums' }}>{avg.toLocaleString()}</span>
                        <span className="text-[10px] text-[var(--text-muted)] w-8 text-right">{share}%</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {supplyEntities.length > 0 && (
            <div className="bg-[var(--bg-card)] border border-[var(--border-card)] p-4">
              <h3 className="text-[11px] font-bold text-[var(--text-primary)] uppercase tracking-widest mb-3 flex items-center gap-2">
                <Zap className="w-3.5 h-3.5 text-[var(--status-emerald)]" /> Supply Sources
              </h3>
              <div className="space-y-2">
                {supplyEntities.map((plant) => {
                  const total = schedule!.supply
                    .filter(s => s.plant_name === plant)
                    .reduce((a, b) => a + b.supply_mw, 0);
                  const avg = Math.round(total / 24);
                  return (
                    <div key={plant} className="flex items-center justify-between text-xs">
                      <span className="text-[var(--text-secondary)]">{plant}</span>
                      <span className="font-bold text-[var(--text-primary)]" style={{ fontVariantNumeric: 'tabular-nums' }}>{avg.toLocaleString()} MW</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {forecast?.factors && (
            <div className="bg-[var(--bg-card)] border border-[var(--border-card)] p-4">
              <h3 className="text-[11px] font-bold text-[var(--text-primary)] uppercase tracking-widest mb-3 flex items-center gap-2">
                <Activity className="w-3.5 h-3.5 text-[var(--status-amber)]" /> Forecast Factors
              </h3>
              <div className="space-y-1.5 text-xs">
                {[
                  { label: 'Seasonal Range', value: `${(Math.max(...forecast.factors.seasonal_ratio) * 100 - 100).toFixed(1)}% / ${(Math.min(...forecast.factors.seasonal_ratio) * 100 - 100).toFixed(1)}%` },
                  { label: 'Temp Sensitivity', value: `${(Math.max(...forecast.factors.temp_ratio) * 100 - 100).toFixed(1)}%` },
                  { label: 'Growth Mult', value: forecast.factors.growth_ratio[0].toFixed(4) },
                ].map((f, i) => (
                  <div key={i} className="flex justify-between py-1 border-b border-[var(--divider)] last:border-0">
                    <span className="text-[var(--text-muted)]">{f.label}</span>
                    <span className="font-bold text-[var(--text-primary)]">{f.value}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* CENTER: Main Chart */}
        <div className="flex-1 min-w-0 bg-[var(--bg-card)] border border-[var(--border-card)] p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-sm font-bold text-[var(--text-primary)]">
                {hasFullSchedule ? 'System Demand Breakdown' : 'ECG Demand Forecast'}
              </h2>
              <p className="text-[11px] text-[var(--text-muted)] mt-0.5">
                {hasFullSchedule ? 'All entities · Hourly MW' : '24-hour forecast · Hourly MW'}
              </p>
            </div>
            {hasFullSchedule && entityNames.length > 0 && (
              <div className="flex border border-[var(--border-card)]">
                {(['stacked', 'entities', 'total'] as const).map(mode => (
                  <button
                    key={mode}
                    onClick={() => setChartMode(mode)}
                    className={`px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider transition-colors ${
                      chartMode === mode
                        ? 'bg-[var(--surface-secondary)] text-[var(--text-primary)]'
                        : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
                    }`}
                  >
                    {mode === 'stacked' ? 'Stacked' : mode === 'entities' ? 'Grouped' : 'Total'}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="h-[380px]">
            <ResponsiveContainer width="100%" height="100%">
              {chartMode === 'stacked' && hasFullSchedule && entityNames.length > 0 ? (
                <AreaChart data={combinedChartData} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="0" stroke="var(--chart-grid)" vertical={false} strokeWidth={0.5} />
                  <XAxis dataKey="hour" stroke="var(--text-muted)" fontSize={10} tickMargin={8} axisLine={false} tickLine={false} interval={0} />
                  <YAxis stroke="var(--text-muted)" fontSize={11} tickMargin={8} axisLine={false} tickLine={false} />
                  <Tooltip
                    contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-card)', borderRadius: 0, fontSize: 12 }}
                    formatter={(value: number, name: string) => [`${Math.round(value).toLocaleString()} MW`, ENTITY_LABELS[name] || name]}
                    labelFormatter={(label: string) => `Hour ${label}`}
                  />
                  <Legend
                    wrapperStyle={{ fontSize: 11, paddingTop: 8 }}
                    formatter={(value: string) => ENTITY_LABELS[value] || value}
                  />
                  {entityNames.map(name => (
                    <Area
                      key={name}
                      type="monotone"
                      dataKey={name}
                      stackId="1"
                      stroke={ENTITY_COLORS[name]}
                      fill={ENTITY_COLORS[name]}
                      fillOpacity={0.6}
                      name={name}
                    />
                  ))}
                </AreaChart>
              ) : chartMode === 'entities' && hasFullSchedule ? (
                <BarChart data={combinedChartData} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="0" stroke="var(--chart-grid)" vertical={false} strokeWidth={0.5} />
                  <XAxis dataKey="hour" stroke="var(--text-muted)" fontSize={10} tickMargin={8} axisLine={false} tickLine={false} interval={0} />
                  <YAxis stroke="var(--text-muted)" fontSize={11} tickMargin={8} axisLine={false} tickLine={false} />
                  <Tooltip
                    contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-card)', borderRadius: 0, fontSize: 12 }}
                    formatter={(value: number) => [`${Math.round(value).toLocaleString()} MW`]}
                    labelFormatter={(label: string) => `Hour ${label}`}
                  />
                  <Legend wrapperStyle={{ fontSize: 11, paddingTop: 8 }}
                    formatter={(value: string) => ENTITY_LABELS[value] || value} />
                  {entityNames.map(name => (
                    <Bar
                      key={name}
                      dataKey={name}
                      fill={ENTITY_COLORS[name]}
                      stackId="a"
                      maxBarSize={20}
                      name={name}
                    />
                  ))}
                </BarChart>
              ) : (
                <ComposedChart data={combinedChartData} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="0" stroke="var(--chart-grid)" vertical={false} strokeWidth={0.5} />
                  <XAxis dataKey="hour" stroke="var(--text-muted)" fontSize={10} tickMargin={8} axisLine={false} tickLine={false} interval={0} />
                  <YAxis stroke="var(--text-muted)" fontSize={11} tickMargin={8} axisLine={false} tickLine={false} />
                  <Tooltip
                    contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-card)', borderRadius: 0, fontSize: 12 }}
                    formatter={(value: number) => [`${Math.round(value).toLocaleString()} MW`]}
                    labelFormatter={(label: string) => `Hour ${label}`}
                  />
                  <Bar
                    dataKey="NITS_Total"
                    fill="var(--brand-blue)"
                    radius={[2, 2, 0, 0]}
                    maxBarSize={28}
                    name="NITS Total"
                    shape={(props: any) => {
                      const { x, y, width, height, payload } = props;
                      const isPeak = payload.hourNum === peak?.hourNum;
                      return <rect x={x} y={y} width={width} height={height} fill={isPeak ? 'var(--status-amber)' : 'var(--brand-blue)'} rx={2} />;
                    }}
                  />
                  {hasFullSchedule && (
                    <Line
                      type="monotone"
                      dataKey="NITS_Total"
                      stroke="var(--brand-blue)"
                      strokeWidth={0}
                      dot={false}
                      activeDot={false}
                      name="NITS Total"
                    />
                  )}
                </ComposedChart>
              )}
            </ResponsiveContainer>
          </div>
        </div>

        {/* RIGHT PANEL: Hourly values */}
        <div className="w-[200px] flex-shrink-0 flex flex-col gap-4">
          <div className="bg-[var(--bg-card)] border border-[var(--border-card)] p-4 flex-1">
            <h3 className="text-[11px] font-bold text-[var(--text-primary)] uppercase tracking-widest mb-3">Hourly Values</h3>
            <div className="space-y-0.5 max-h-[420px] overflow-y-auto">
              {combinedChartData.map((d) => {
                const val = (d as any).NITS_Total || (d as any).ECG || 0;
                const isPeak = d.hourNum === peak?.hourNum;
                const maxVal = peak ? ((peak as any).NITS_Total || (peak as any).ECG || 1) : 1;
                return (
                  <div
                    key={d.hourNum}
                    className={`flex items-center justify-between text-xs px-2 py-1 ${
                      isPeak ? 'bg-[var(--status-amber)]/10 border-l-2 border-[var(--status-amber)]' : 'hover:bg-[var(--surface-secondary)]'
                    }`}
                  >
                    <span className="text-[var(--text-muted)] w-10">{d.hour}</span>
                    <div className="flex items-center gap-1.5 flex-1 justify-end">
                      <div className="h-1.5 bg-[var(--brand-blue)]/25 rounded-full" style={{ width: `${Math.max(4, (val / maxVal) * 50)}px` }} />
                      <span
                        className={`font-bold w-14 text-right ${isPeak ? 'text-[var(--status-amber)]' : 'text-[var(--text-primary)]'}`}
                        style={{ fontVariantNumeric: 'tabular-nums' }}
                      >
                        {Math.round(val).toLocaleString()}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
