'use client';

import { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, RotateCcw, Play, Loader2 } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { forecastService } from '@/services/forecastService';

interface ScenarioParams {
  temperature: number;
  gridInflow: number;
  industrialLoad: number;
}

export function ScenarioSimulator() {
  const [params, setParams] = useState<ScenarioParams>({
    temperature: 0,
    gridInflow: 0,
    industrialLoad: 0
  });

  const [simulationData, setSimulationData] = useState<any[]>([]);
  const [isSimulating, setIsSimulating] = useState(false);
  const [basePeak, setBasePeak] = useState(1580);
  const [adjustedPeak, setAdjustedPeak] = useState(1580);

  // Fetch base forecast on mount
  useEffect(() => {
    handleApplyScenario();
  }, []);

  const handleApplyScenario = async () => {
    try {
      setIsSimulating(true);
      const res = await forecastService.runSimulation(
        params.temperature,
        params.gridInflow,
        params.industrialLoad
      );

      // Map to chart format
      const chartData = res.timestamps.slice(0, 10).map((t, i) => ({
        time: new Date(t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        base: Math.round(res.forecast_mw[i] * 0.95), // Mock base comparison
        scenario: Math.round(res.forecast_mw[i])
      }));

      setSimulationData(chartData);
      setAdjustedPeak(Math.max(...res.forecast_mw));
      if (params.temperature === 0 && params.gridInflow === 0 && params.industrialLoad === 0) {
        setBasePeak(Math.max(...res.forecast_mw));
      }
    } catch (err) {
      console.error('Simulation failed:', err);
    } finally {
      setIsSimulating(false);
    }
  };

  const handleReset = () => {
    setParams({
      temperature: 0,
      gridInflow: 0,
      industrialLoad: 0
    });
  };

  const difference = Math.round(adjustedPeak - basePeak);

  return (
    <div
      className="glass-morphism"
    >

      {/* Header */}
      <div className="px-6 py-4 border-b" style={{ borderColor: 'var(--border-primary)', opacity: 0.5 }}>

        <h3 className="text-base font-semibold" style={{ color: 'var(--text-primary)' }}>
          What-If Scenario Builder
        </h3>
        <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>
          Base Forecast Peak: {Math.round(basePeak).toLocaleString()} MW
        </p>
      </div>

      <div className="p-6">
        {/* Parameters */}
        <div className="space-y-6 mb-6">
          <SliderControl
            label="Temperature Change"
            min={-5}
            max={5}
            step={0.5}
            value={params.temperature}
            onChange={(value) => setParams({ ...params, temperature: value })}
            unit="°C"
          />
          <SliderControl
            label="Grid Inflow (NY6ZA)"
            min={-20}
            max={20}
            step={1}
            value={params.gridInflow}
            onChange={(value) => setParams({ ...params, gridInflow: value })}
            unit="%"
          />
          <SliderControl
            label="Industrial Load"
            min={-30}
            max={30}
            step={1}
            value={params.industrialLoad}
            onChange={(value) => setParams({ ...params, industrialLoad: value })}
            unit="%"
          />
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-3 mb-6">
          <button
            onClick={handleReset}
            className="px-4 py-2 text-[11px] border font-bold uppercase tracking-widest flex items-center gap-2 transition-all active:scale-95"
            style={{
              borderColor: 'var(--border-primary)',
              color: 'var(--text-tertiary)',
              backgroundColor: 'var(--bg-surface)',
              fontFamily: 'var(--font-geist-mono)'
            }}
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset
          </button>
          <button
            onClick={handleApplyScenario}
            disabled={isSimulating}
            className="px-4 py-2 text-[11px] font-bold uppercase tracking-widest flex items-center gap-2 disabled:opacity-50 transition-all active:scale-95"
            style={{
              backgroundColor: 'var(--status-info)',
              color: 'white',
              fontFamily: 'var(--font-geist-mono)'
            }}
          >

            {isSimulating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            Apply Scenario
          </button>
        </div>

        {/* Result */}
        <div
          className="px-4 py-3 mb-6 flex items-center justify-between border-l-4 glass-morphism"
          style={{
            backgroundColor: difference >= 0 ? 'color-mix(in srgb, var(--status-warn), transparent 90%)' : 'color-mix(in srgb, var(--status-ok), transparent 90%)',
            borderLeftColor: difference >= 0 ? 'var(--status-warn)' : 'var(--status-ok)'
          }}
        >

          <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            {isSimulating ? 'Recalculating...' : `Adjusted Peak: ${Math.round(adjustedPeak).toLocaleString()} MW`}
          </span>
          <div className="flex items-center gap-1 text-sm font-semibold" style={{
            color: difference >= 0 ? 'var(--status-warn)' : 'var(--status-ok)'
          }}>

            {!isSimulating && (
              <>
                {difference >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                {difference >= 0 ? '+' : ''}{difference} MW from base
              </>
            )}
          </div>
        </div>

        {/* Comparison Chart */}
        <div className="relative">
          {isSimulating && (
            <div className="absolute inset-0 flex items-center justify-center bg-white/5 backdrop-blur-sm z-10">
              <Loader2 className="w-8 h-8 animate-spin" style={{ color: 'var(--status-info)' }} />
            </div>
          )}

          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={simulationData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-primary)" strokeOpacity={0.1} vertical={false} />

              <XAxis
                dataKey="time"
                stroke="var(--text-tertiary)"
                style={{ fontSize: '11px', fontWeight: 700 }}
                tickLine={false}
                axisLine={false}

              />
              <YAxis
                stroke="var(--text-tertiary)"
                style={{ fontSize: '11px', fontWeight: 700 }}
                tickLine={false}
                axisLine={false}

                domain={['auto', 'auto']}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'var(--bg-surface)',
                  border: '1px solid var(--border-primary)',
                  borderRadius: '0px',
                  fontSize: '11px',
                  boxShadow: 'var(--glass-shadow)',
                  color: 'var(--text-primary)',
                  fontFamily: 'var(--font-geist-mono)'
                }}
              />

              <Legend
                wrapperStyle={{ fontSize: '12px', fontWeight: 500 }}
                iconType="line"
              />
              <Line
                type="monotone"
                dataKey="base"
                stroke="var(--text-muted)"
                strokeWidth={2}
                strokeDasharray="5 5"
                name="Base Forecast"
                dot={{ fill: 'var(--text-muted)', r: 3 }}
              />

              <Line
                type="monotone"
                dataKey="scenario"
                stroke="var(--regime-transition)"
                strokeWidth={3}
                name="Scenario Forecast"
                dot={{ fill: 'var(--regime-transition)', r: 3 }}
              />

            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

interface SliderControlProps {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (value: number) => void;
  unit: string;
}

function SliderControl({ label, min, max, step, value, onChange, unit }: SliderControlProps) {
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <label className="text-sm font-semibold" style={{ color: 'var(--text-secondary)' }}>
          {label}
        </label>
        <span className="text-[10px] font-bold uppercase tracking-widest px-2 py-1 glass-morphism" style={{
          color: value === 0 ? 'var(--text-tertiary)' : value > 0 ? 'var(--status-warn)' : 'var(--status-ok)',
          fontFamily: 'var(--font-geist-mono)'
        }}>

          {value > 0 ? '+' : ''}{value}{unit}
        </span>

      </div>
      <div className="relative">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="w-full h-1 appearance-none cursor-pointer"
          style={{
            backgroundColor: 'var(--border-primary)',
            background: `linear-gradient(to right, var(--status-info) 0%, var(--status-info) ${((value - min) / (max - min)) * 100}%, var(--border-primary) ${((value - min) / (max - min)) * 100}%, var(--border-primary) 100%)`
          }}

        />
        <style jsx>{`
          input[type="range"]::-webkit-slider-thumb {
            appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: #2563EB;
            cursor: pointer;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            border: 2px solid white;
          }
          input[type="range"]::-moz-range-thumb {
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: #2563EB;
            cursor: pointer;
            border: 2px solid white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
          }
        `}</style>
        <div className="flex justify-between mt-1">
          <span className="text-[10px] font-bold" style={{ color: 'var(--text-tertiary)', fontFamily: 'var(--font-geist-mono)' }}>{min}{unit}</span>
          <span className="text-[10px] font-bold" style={{ color: 'var(--text-tertiary)', fontFamily: 'var(--font-geist-mono)' }}>+{max}{unit}</span>

        </div>
      </div>
    </div>
  );
}
