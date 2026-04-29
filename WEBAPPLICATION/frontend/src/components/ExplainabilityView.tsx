'use client';

import { useState, useEffect } from 'react';
import {
    BarChart3,
    Activity,
    AlertTriangle,
    Info,
    Fingerprint,
    Loader2,
    RefreshCcw,
    Download,
    Cpu,
} from 'lucide-react';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Cell,
} from 'recharts';
import { forecastService } from '@/services/forecastService';
import { MetricCard } from './MetricCard';
import { StatusBadge } from './StatusBadge';

export function ExplainabilityView() {
    const [decompData, setDecompData] = useState<{ 
        peak_mw: number, 
        peak_timestamp: string, 
        components: { name: string, value: number, color: string }[] 
    } | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fetchData = async () => {
        try {
            setIsLoading(true);
            setError(null);
            const data = await forecastService.getPeakDecomposition();
            setDecompData(data);
        } catch (err) {
            console.error('Failed to fetch Decomposition:', err);
            setError('Structural analysis failed. The decomposition engine is calibrating.');
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchData();
    }, []);

    const peakLoad = decompData?.peak_mw || 0;
    const chartData = decompData?.components || [];

    // Component descriptions for Ghanaian Grid context
    const componentDescriptions: Record<string, string> = {
        'Base Trend': 'Long-term load trajectory reflecting organic industrial and population growth in the Nayagina-82 catchment.',
        'Seasonal Rhythm': 'Daily and weekly cyclic patterns driven by institutional schedules and residential behavior.',
        'Temperature Impact': 'Additional load attributed to HVAC systems and cooling demand based on regional temperature gradients.',
        'Holiday Adjustment': 'Reduction in commercial and industrial load due to standard Ghanaian public holidays.',
        'Rain Suppression': 'Suppression of AC and heating demand during precipitation events as observed in operator interviews.',
        'Line Efficiency': 'Gain in system throughput due to lower resistive losses in transition lines during cooler weather.',
        'Short-term Bias': 'Real-time error correction managed by the adaptive Kalman layer to sync with latest SCADA.'
    };

    return (
        <div className="flex flex-col gap-10 h-full font-sans">

            {/* HEADER SECTION */}
            <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-6 pb-6 border-b border-[var(--divider)]">
                <div className="flex flex-col gap-1">
                    <div className="flex items-center gap-3 text-[var(--brand-indigo)]">
                        <Cpu className="w-5 h-5" />
                        <span className="micro-num uppercase tracking-[0.4em]">Engine ID: DECOMP_V1_GH</span>
                    </div>
                    <h1 className="display-num text-[var(--text-primary)]">Structural Analysis</h1>
                    <p className="body-text text-[var(--text-secondary)] max-w-2xl">
                        Physical load breakdown for peak demand. This view decomposes the forecast into its 
                        <span className="text-[var(--text-primary)] font-medium"> trend, seasonal, and environmental </span> drivers.
                    </p>
                </div>

                <div className="flex items-center gap-3">
                    <button
                        onClick={fetchData}
                        disabled={isLoading}
                        className="h-10 px-4 border border-[var(--divider)] bg-[var(--surface-secondary)] hover:border-[var(--brand-indigo)] text-[var(--text-primary)] transition-all flex items-center gap-2 group rounded"
                    >
                        <RefreshCcw className={`w-4 h-4 ${isLoading ? 'animate-spin text-[var(--brand-indigo)]' : 'group-hover:text-[var(--brand-indigo)]'}`} />
                        <span className="text-[11px] font-bold tracking-widest uppercase">Deep Scan</span>
                    </button>
                    <button className="h-10 px-4 border border-[var(--divider)] bg-[var(--surface-secondary)] hover:border-[var(--brand-indigo)] text-[var(--text-primary)] transition-all flex items-center gap-2 group rounded">
                        <Download className="w-4 h-4 group-hover:text-[var(--brand-indigo)]" />
                        <span className="text-[11px] font-bold tracking-widest uppercase">Export PDF</span>
                    </button>
                </div>
            </div>

            {/* TOP METRICS GRID */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <MetricCard
                    label="Peak Forecast"
                    value={peakLoad || '---'}
                    unit="MW"
                    status="amber"
                    subtext={`At ${decompData?.peak_timestamp ? new Date(decompData.peak_timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) : '---'}`}
                />
                <MetricCard
                    label="Base Load"
                    value={chartData.find(c => c.name === 'Base Trend')?.value.toFixed(1) || '---'}
                    unit="MW"
                    subtext="De-seasonalized Trend"
                />
                <MetricCard
                    label="Temp. Sensitivity"
                    value={chartData.find(c => c.name === 'Temperature')?.value.toFixed(1) || '0.0'}
                    unit="MW"
                    status="none"
                    subtext="Thermal Driver Output"
                />
                <div className="glass-panel p-5 flex flex-col justify-between">
                   <span className="text-[12px] font-bold uppercase tracking-widest text-[var(--text-muted)]">Analysis Mode</span>
                   <div className="mt-2">
                      <StatusBadge status="stable" label="Multiplicative" pulse />
                   </div>
                   <p className="text-[11px] text-[var(--text-muted)] mt-2 italic font-medium">Physical Decomposition</p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

                {/* DECOMPOSITION BAR CHART */}
                <div className="lg:col-span-2 glass-panel p-8 flex flex-col gap-8">
                    <div className="flex items-center justify-between">
                        <div className="flex flex-col gap-1">
                            <h3 className="headline text-[var(--text-primary)] flex items-center gap-2">
                                <BarChart3 className="w-4 h-4 text-[var(--brand-indigo)]" /> Physical Component Breakdown
                            </h3>
                            <span className="micro-num text-[var(--text-muted)] uppercase tracking-widest">MW CONTRIBUTION TO PEAK LOAD</span>
                        </div>
                    </div>

                    <div className="h-[450px] w-full relative">
                        {isLoading && (
                            <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-[var(--bg-card)]/40 backdrop-blur-sm rounded">
                                <Loader2 className="w-10 h-10 animate-spin text-[var(--brand-indigo)] mb-4" />
                                <span className="micro-num animate-pulse uppercase tracking-[0.3em]">Decomposing Signal...</span>
                            </div>
                        )}

                        {!isLoading && chartData.length > 0 && (
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                                    <CartesianGrid strokeDasharray="0" stroke="var(--divider)" horizontal={false} />
                                    <XAxis type="number" stroke="var(--text-muted)" fontSize={10} fontFamily="JetBrains Mono" />
                                    <YAxis
                                        dataKey="name"
                                        type="category"
                                        stroke="var(--text-muted)"
                                        fontSize={11}
                                        width={140}
                                        axisLine={false}
                                        tickLine={false}
                                        tick={{ fontWeight: 700, fontFamily: 'JetBrains Mono' }}
                                    />
                                    <Tooltip
                                        cursor={{ fill: 'var(--divider)', opacity: 0.3 }}
                                        contentStyle={{ backgroundColor: 'var(--bg-card)', borderColor: 'var(--border-card)', borderRadius: '4px', fontFamily: 'JetBrains Mono' }}
                                    />
                                    <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={40}>
                                        {chartData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={entry.color} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        )}
                    </div>
                </div>

                {/* COMPONENT DESCRIPTIONS */}
                <div className="flex flex-col gap-6">
                   <div className="p-6 border-l-4 border-l-[var(--brand-indigo)] bg-[var(--brand-indigo)]/5 rounded-r-lg">
                      <h3 className="title text-[var(--text-primary)] uppercase tracking-widest font-bold mb-6">Component Insights</h3>
                      <div className="space-y-8">
                         {chartData.map((d, i) => (
                            <div key={i} className="flex flex-col gap-2">
                               <div className="flex justify-between items-center">
                                  <span className="caption text-[var(--text-muted)] uppercase font-bold tracking-widest">Physical Driver</span>
                                  <span className="micro-num font-bold text-[var(--text-primary)]" style={{ color: d.color }}>
                                    {d.value.toFixed(1)} MW
                                  </span>
                               </div>
                               <h4 className="text-[13px] font-bold text-[var(--text-primary)]">{d.name}</h4>
                               <p className="text-[11px] text-[var(--text-secondary)] italic border-l border-[var(--divider)] pl-3 leading-relaxed">
                                  {componentDescriptions[d.name] || 'Operational factor affecting Nayagina-82 substation dynamics.'}
                                </p>
                            </div>
                         ))}
                      </div>
                   </div>

                   <div className="glass-panel p-6 flex flex-col gap-4">
                      <h3 className="caption text-[var(--text-muted)] uppercase tracking-widest font-bold">Model Parameters</h3>
                      <div className="space-y-3">
                         {[
                            { label: 'Trend Growth', val: '+2.1% /yr', status: 'STABLE' },
                            { label: 'Temp Sensitivity', val: '2.4 MW/°C', status: 'HIGH' },
                            { label: 'Residual Error', val: '1.24 MW', status: 'LOW' },
                            { label: 'Kalman Gain', val: '0.85', status: 'OPTI' },
                         ].map((item, i) => (
                            <div key={i} className="flex justify-between items-center border-b border-[var(--divider)] pb-2 last:border-0">
                               <span className="micro-num text-[var(--text-muted)]">{item.label}</span>
                               <div className="flex items-center gap-2">
                                  <span className="micro-num text-[var(--text-primary)] font-bold">{item.val}</span>
                                  {item.status && <span className="text-[8px] px-1 bg-[var(--surface-secondary)] border border-[var(--divider)] rounded">{item.status}</span>}
                                </div>
                            </div>
                         ))}
                      </div>
                   </div>
                </div>

            </div>
        </div>
    );
}
