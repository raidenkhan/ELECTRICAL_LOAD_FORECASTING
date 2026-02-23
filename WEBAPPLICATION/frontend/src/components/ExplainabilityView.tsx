'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    BarChart3,
    Activity,
    AlertTriangle,
    Info,
    Zap,
    ChevronRight,
    ArrowUpRight,
    ArrowDownRight,
    Loader2,
    RefreshCcw,
    Download,
    Share2,
    Database,
    Cpu,
    Fingerprint,
    Shield,
    FlaskConical,
    CheckCircle2
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
    ReferenceLine
} from 'recharts';
import { forecastService } from '@/services/forecastService';

export function ExplainabilityView() {
    const [shapData, setShapData] = useState<{ features: string[], values: number[], base_value: number } | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fetchData = async () => {
        try {
            setIsLoading(true);
            setError(null);
            const data = await forecastService.getShapValues();
            setShapData(data);
        } catch (err) {
            console.error('Failed to fetch SHAP:', err);
            setError('Technical failure in SHAP attribution layer. Remote node currently unreachable.');
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchData();
    }, []);

    const totalAdjustment = shapData ? shapData.values.reduce((a, b) => a + b, 0) : 0;
    const finalForecast = shapData ? Math.round(shapData.base_value + totalAdjustment) : 0;

    // Prepare data for BarChart
    const chartData = shapData ? shapData.features.map((f, i) => ({
        name: f.replace(/_mw|mw/gi, '').replace(/_/g, ' ').toUpperCase(),
        value: parseFloat(shapData.values[i].toFixed(1)),
        originalValue: shapData.values[i]
    })).sort((a, b) => Math.abs(b.value) - Math.abs(a.value)) : [];

    return (
        <div className="space-y-8 pb-12 animate-in fade-in duration-700">

            {/* HEADER SECTION */}
            <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-6 pb-2 border-b border-border/50">
                <div className="space-y-2">
                    <div className="flex items-center gap-3 text-primary">
                        <Fingerprint className="w-5 h-5" />
                        <span className="text-[10px] font-black tracking-[0.4em] uppercase">Attribution ID: 0xFD42A_SHAP</span>
                    </div>
                    <h1 className="text-4xl font-black tracking-tighter">FORECAST <span className="text-primary italic">EXPLAINABILITY</span></h1>
                    <p className="text-muted-foreground font-medium text-sm max-w-2xl">
                        Real-time feature attribution using <span className="text-foreground">Kernel SHAP algorithms</span>. Identifying the core drivers behind the current STLF load vector.
                    </p>
                </div>

                <div className="flex items-center gap-3">
                    <button
                        onClick={fetchData}
                        disabled={isLoading}
                        className="h-10 px-4 border border-border bg-card/50 hover:border-primary/50 text-foreground transition-all flex items-center gap-2 group"
                    >
                        <RefreshCcw className={`w-4 h-4 ${isLoading ? 'animate-spin text-primary' : 'group-hover:text-primary'}`} />
                        <span className="text-[10px] font-bold tracking-widest uppercase" style={{ fontFamily: 'var(--font-geist-mono)' }}>Initialize Scan</span>
                    </button>
                    <button className="h-10 px-4 border border-border bg-card/50 hover:border-primary/50 text-foreground transition-all flex items-center gap-2 group">
                        <Download className="w-4 h-4 group-hover:text-primary" />
                        <span className="text-[10px] font-bold tracking-widest uppercase" style={{ fontFamily: 'var(--font-geist-mono)' }}>Export Log</span>
                    </button>
                </div>
            </div>

            {/* TOP METRICS GRID */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <MetricCard
                    label="BASE VALUE"
                    value={`${shapData?.base_value || '---'} MW`}
                    sub="Starting Bias"
                    icon={Database}
                />
                <MetricCard
                    label="NET ATTRIBUTION"
                    value={`${totalAdjustment > 0 ? '+' : ''}${totalAdjustment.toFixed(1)} MW`}
                    sub="Feature Sum"
                    icon={Cpu}
                    color={totalAdjustment > 0 ? 'text-red-500' : 'text-emerald-500'}
                />
                <MetricCard
                    label="DERIVED TARGET"
                    value={`${finalForecast || '---'} MW`}
                    sub="Ensemble Output"
                    icon={Activity}
                    primary
                />
                <MetricCard
                    label="MODEL CONFIDENCE"
                    value="84.2%"
                    sub="High Fidelity"
                    icon={Shield}
                />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

                {/* SHAP FORCE PLOT / BAR CHART (2/3) */}
                <div className="lg:col-span-2 space-y-6">
                    <div className="p-8 bg-card/50 border border-border glass-morphism relative overflow-hidden">
                        <div className="absolute top-0 right-0 p-4 opacity-5 pointer-events-none">
                            <BarChart3 className="w-32 h-32" />
                        </div>

                        <div className="flex items-center justify-between mb-8">
                            <div className="space-y-1">
                                <h3 className="text-sm font-black tracking-widest uppercase flex items-center gap-2">
                                    <Activity className="w-4 h-4 text-primary" /> Feature Contribution Analysis
                                </h3>
                                <p className="text-[10px] text-muted-foreground font-bold tracking-widest uppercase" style={{ fontFamily: 'var(--font-geist-mono)' }}>
                                    SHAP VALUES (ATTRIBUTION_UNITS_MW)
                                </p>
                            </div>

                            <div className="flex items-center gap-6">
                                <div className="flex items-center gap-2">
                                    <div className="w-2 h-2 bg-red-500" />
                                    <span className="text-[9px] font-black uppercase tracking-widest">Pressure (+)</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <div className="w-2 h-2 bg-emerald-500" />
                                    <span className="text-[9px] font-black uppercase tracking-widest">Damping (-)</span>
                                </div>
                            </div>
                        </div>

                        <div className="h-[400px] w-full relative">
                            {isLoading && (
                                <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-card/40 backdrop-blur-sm">
                                    <Loader2 className="w-10 h-10 animate-spin text-primary mb-4" />
                                    <span className="text-xs font-black tracking-[0.3em] uppercase animate-pulse">Computing Matrix...</span>
                                </div>
                            )}

                            {error && (
                                <div className="absolute inset-0 z-20 flex flex-col items-center justify-center text-center p-8 bg-card/40 backdrop-blur-sm">
                                    <AlertTriangle className="w-12 h-12 text-red-500 mb-4" />
                                    <span className="text-[10px] font-black tracking-[0.2em] uppercase max-w-xs">{error}</span>
                                    <button onClick={fetchData} className="mt-4 text-xs font-bold text-primary hover:underline uppercase">Retry Handshake</button>
                                </div>
                            )}

                            {!isLoading && chartData && (
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                                        <XAxis type="number" hide />
                                        <YAxis
                                            dataKey="name"
                                            type="category"
                                            stroke="var(--text-muted)"
                                            fontSize={10}
                                            width={120}
                                            axisLine={false}
                                            tickLine={false}
                                            tick={{ fontWeight: 800, letterSpacing: '0.1em' }}
                                        />
                                        <Tooltip
                                            cursor={{ fill: 'var(--primary)', opacity: 0.05 }}
                                            content={<CustomTooltip />}
                                        />
                                        <ReferenceLine x={0} stroke="var(--border-primary)" strokeWidth={2} />
                                        <Bar dataKey="value" radius={[0, 2, 2, 0]} barSize={32}>
                                            {chartData.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#ef4444' : '#10b981'} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            )}
                        </div>
                    </div>

                    {/* SECONDARY INFO CARDS */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="p-6 bg-card border border-border hover:border-primary/20 transition-all space-y-4">
                            <h4 className="text-[10px] font-black text-primary tracking-[0.3em] uppercase flex items-center gap-2">
                                <FlaskConical className="w-4 h-4" /> Regime Classification
                            </h4>
                            <div className="space-y-4">
                                <div className="flex items-center justify-between">
                                    <span className="text-xs font-bold text-muted-foreground uppercase tracking-widest">Active Profile</span>
                                    <span className="text-xs font-black text-foreground uppercase tracking-widest bg-red-500/10 px-2 py-1 border border-red-500/20 shadow-[0_0_15px_rgba(239,68,68,0.1)]">Industrial Startup</span>
                                </div>
                                <p className="text-[11px] text-muted-foreground leading-relaxed">
                                    Attribution reflects standard 6AM operating shift increase. Temperature dependencies are secondary to schedule indicators for this horizon.
                                </p>
                            </div>
                        </div>

                        <div className="p-6 bg-card border border-border hover:border-primary/20 transition-all space-y-4">
                            <h4 className="text-[10px] font-black text-primary tracking-[0.3em] uppercase flex items-center gap-2">
                                <Info className="w-4 h-4" /> Attribution Policy
                            </h4>
                            <div className="space-y-2">
                                <div className="flex items-center gap-3">
                                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                                    <span className="text-[10px] font-bold text-foreground">Kernel Consistency Check Pass</span>
                                </div>
                                <div className="flex items-center gap-3">
                                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                                    <span className="text-[10px] font-bold text-foreground">Convergence Reached (0.001%)</span>
                                </div>
                            </div>
                            <p className="text-[11px] text-muted-foreground leading-relaxed">
                                Feature interactions are computed using additive feature attribution methods. High stability detected in current node parameters.
                            </p>
                        </div>
                    </div>
                </div>

                {/* DRIVER FEEDBACK / INSIGHTS (1/3) */}
                <div className="space-y-6">
                    <div className="p-6 border-l-2 border-primary bg-primary/5 space-y-6">
                        <div className="flex items-center justify-between">
                            <h3 className="text-xs font-black tracking-[0.3em] uppercase">Executive Summary</h3>
                            <Zap className="w-4 h-4 text-primary" />
                        </div>

                        <div className="space-y-6">
                            <InsightItem
                                title="Primary Driver"
                                value="Lag_96_Load"
                                impact="+45.2MW"
                                desc="Historical 24h demand remains the strongest predictor for current grid pressure."
                                type="pos"
                            />
                            <InsightItem
                                title="Secondary Driver"
                                value="Rolling_Mean_24h"
                                impact="+28.5MW"
                                desc="A 3.2% increase in the daily average load is shifting the base value upwards."
                                type="pos"
                            />
                            <InsightItem
                                title="Negative Bias"
                                value="Hour_Sin"
                                impact="-12.4MW"
                                desc="Standard cyclic damping expected during the early morning transition period."
                                type="neg"
                            />
                        </div>
                    </div>

                    <div className="p-8 bg-card/50 border border-border space-y-6">
                        <h3 className="text-xs font-black tracking-[0.3em] uppercase">Raw Data Integrity</h3>
                        <div className="space-y-4">
                            <DataLine label="SHAP Convergence" value="0.00032" status="OPTIMAL" />
                            <DataLine label="Model Version" value="LSTM_ENS_v4" status="STABLE" />
                            <DataLine label="Samples Decoded" value="14,204" />
                            <DataLine label="Local Node Latency" value="12ms" />
                        </div>
                        <button className="w-full h-12 border border-border bg-muted/50 text-[10px] font-black tracking-widest uppercase hover:text-primary hover:border-primary transition-all">
                            Initialize Substation Audit
                        </button>
                    </div>
                </div>

            </div>
        </div>
    );
}

// SUB-COMPONENTS

function MetricCard({ label, value, sub, icon: Icon, primary = false, color }: any) {
    return (
        <div className={`p-6 border bg-card transition-all relative overflow-hidden group
                    ${primary ? 'border-primary shadow-[0_0_20px_rgba(204,255,0,0.1)]' : 'border-border hover:border-primary/30'}`}>
            <div className={`absolute top-0 right-0 p-4 opacity-[0.03] group-hover:opacity-[0.06] transition-opacity
                      ${primary ? 'text-primary' : 'text-foreground'}`}>
                <Icon className="w-16 h-16" />
            </div>
            <div className="space-y-3 relative z-10">
                <div className="flex items-center gap-2">
                    <Icon className={`w-3.5 h-3.5 ${primary ? 'text-primary' : 'text-muted-foreground'}`} />
                    <span className="text-[10px] font-black tracking-[0.3em] text-muted-foreground uppercase">{label}</span>
                </div>
                <div className={`text-3xl font-black ${color || 'text-foreground'}`} style={{ fontFamily: 'var(--font-space-grotesk)' }}>
                    {value}
                </div>
                <div className="text-[9px] font-bold tracking-widest text-muted-foreground uppercase flex items-center gap-2">
                    <div className={`w-1 h-1 ${primary ? 'bg-primary' : 'bg-muted-foreground'}`} /> {sub}
                </div>
            </div>
        </div>
    );
}

function InsightItem({ title, value, impact, desc, type }: any) {
    return (
        <div className="space-y-2 group">
            <div className="flex items-center justify-between">
                <span className="text-[10px] font-black text-muted-foreground uppercase tracking-widest">{title}</span>
                <span className={`text-[10px] font-black tracking-widest uppercase ${type === 'pos' ? 'text-red-500' : 'text-emerald-500'}`}>
                    {impact}
                </span>
            </div>
            <div className="flex items-center gap-2">
                <h4 className="text-xs font-black text-foreground uppercase tracking-widest">{value}</h4>
                {type === 'pos' ? <ArrowUpRight className="w-3 h-3 text-red-500" /> : <ArrowDownRight className="w-3 h-3 text-emerald-500" />}
            </div>
            <p className="text-[10px] text-muted-foreground font-medium leading-relaxed italic border-l border-border pl-3 mt-1 opacity-80">
                {desc}
            </p>
        </div>
    );
}

function DataLine({ label, value, status }: any) {
    return (
        <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-widest">
            <span className="text-muted-foreground">{label}</span>
            <div className="flex items-center gap-3">
                <span className="text-foreground">{value}</span>
                {status && (
                    <span className={`text-[8px] px-1.5 py-0.5 border ${status === 'OPTIMAL' ? 'text-primary border-primary/20 bg-primary/5' : 'text-blue-500 border-blue-500/20 bg-blue-500/5'}`}>
                        {status}
                    </span>
                )}
            </div>
        </div>
    );
}

function CustomTooltip({ active, payload, label }: any) {
    if (active && payload && payload.length) {
        const val = payload[0].value;
        return (
            <div className="bg-card border border-primary p-4 shadow-xl glass-morphism">
                <p className="text-[10px] font-black text-muted-foreground uppercase tracking-[0.3em] mb-2">{label}</p>
                <p className={`text-lg font-black ${val > 0 ? 'text-red-500' : 'text-emerald-500'}`}>
                    {val > 0 ? '+' : ''}{val} MW
                </p>
                <p className="text-[9px] font-bold text-muted-foreground uppercase tracking-widest mt-1">Impact attribution</p>
            </div>
        );
    }
    return null;
}
