import { useState } from 'react';
import { TrendingUp } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface ScenarioParams {
  temperature: number;
  gridInflow: number;
  industrialLoad: number;
}

const baseScenarioData = [
  { time: '12:00', base: 85, scenario: 85 },
  { time: '13:00', base: 92, scenario: 95 },
  { time: '14:00', base: 102, scenario: 108 },
  { time: '15:00', base: 98, scenario: 104 },
  { time: '16:00', base: 88, scenario: 92 }
];

export function ScenarioSimulator() {
  const [params, setParams] = useState<ScenarioParams>({
    temperature: 2,
    gridInflow: 10,
    industrialLoad: -5
  });

  const handleReset = () => {
    setParams({
      temperature: 0,
      gridInflow: 0,
      industrialLoad: 0
    });
  };

  // Calculate adjusted peak based on parameters
  const basePeak = 102;
  const tempImpact = params.temperature * 1.5;
  const inflowImpact = params.gridInflow * 0.2;
  const loadImpact = params.industrialLoad * 0.3;
  const adjustedPeak = Math.round(basePeak + tempImpact + inflowImpact + loadImpact);
  const difference = adjustedPeak - basePeak;

  return (
    <div 
      className="rounded-lg p-6"
      style={{
        backgroundColor: 'var(--bg-secondary)',
        border: '1px solid var(--border-default)',
        boxShadow: '0 2px 8px rgba(0,0,0,0.06)'
      }}
    >
      {/* Header */}
      <h3 className="mb-1" style={{ 
        fontSize: 'var(--text-lg)',
        fontWeight: 'var(--font-weight-semibold)',
        color: 'var(--text-primary)'
      }}>
        What-If Scenario Builder
      </h3>
      <p className="text-sm mb-6" style={{ color: 'var(--text-secondary)' }}>
        Base Forecast: 102 MW on Feb 4 at 14:30
      </p>

      {/* Parameters */}
      <div className="space-y-6 mb-6">
        <SliderControl
          label="Temperature"
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
          className="px-4 py-2 rounded-md text-sm transition-colors"
          style={{
            border: '1px solid var(--border-default)',
            color: 'var(--text-secondary)',
            fontWeight: 'var(--font-weight-medium)'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = 'var(--bg-primary)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = 'transparent';
          }}
        >
          Reset
        </button>
        <button
          className="px-4 py-2 rounded-md text-sm transition-colors"
          style={{
            backgroundColor: 'var(--primary-blue)',
            color: 'white',
            fontWeight: 'var(--font-weight-medium)'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = '#1D4ED8';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = 'var(--primary-blue)';
          }}
        >
          Apply Scenario
        </button>
      </div>

      {/* Result */}
      <div 
        className="px-4 py-3 rounded-md mb-6 flex items-center justify-between"
        style={{
          backgroundColor: difference >= 0 ? '#FEF3C7' : '#ECFDF5',
          border: `1px solid ${difference >= 0 ? 'var(--warning-orange)' : 'var(--success-green)'}`
        }}
      >
        <span className="text-sm" style={{ 
          fontWeight: 'var(--font-weight-semibold)',
          color: 'var(--text-primary)'
        }}>
          Adjusted Peak: {adjustedPeak} MW
        </span>
        <div className="flex items-center gap-1 text-sm" style={{ 
          color: difference >= 0 ? 'var(--warning-orange)' : 'var(--success-green)',
          fontWeight: 'var(--font-weight-medium)'
        }}>
          <TrendingUp className="w-4 h-4" />
          {difference >= 0 ? '+' : ''}{difference} MW from base
        </div>
      </div>

      {/* Comparison Chart */}
      <ResponsiveContainer width="100%" height={250}>
        <LineChart data={baseScenarioData}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)" />
          <XAxis 
            dataKey="time" 
            stroke="var(--text-muted)"
            style={{ fontSize: '12px' }}
          />
          <YAxis 
            stroke="var(--text-muted)"
            style={{ fontSize: '12px' }}
            label={{ value: 'Load (MW)', angle: -90, position: 'insideLeft', style: { fill: 'var(--text-secondary)' } }}
          />
          <Tooltip 
            contentStyle={{
              backgroundColor: 'var(--bg-secondary)',
              border: '1px solid var(--border-default)',
              borderRadius: '6px',
              fontSize: '12px'
            }}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="base" 
            stroke="#94A3B8" 
            strokeWidth={2}
            strokeDasharray="5 5"
            name="Base Forecast"
            dot={{ fill: '#94A3B8', r: 4 }}
          />
          <Line 
            type="monotone" 
            dataKey="scenario" 
            stroke="var(--warning-orange)" 
            strokeWidth={2}
            name="Scenario Forecast"
            dot={{ fill: 'var(--warning-orange)', r: 4 }}
          />
        </LineChart>
      </ResponsiveContainer>
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
        <label className="text-sm" style={{ 
          fontWeight: 'var(--font-weight-medium)',
          color: 'var(--text-secondary)'
        }}>
          {label}
        </label>
        <span className="text-sm" style={{ 
          fontWeight: 'var(--font-weight-semibold)',
          color: 'var(--text-primary)'
        }}>
          Current: {value > 0 ? '+' : ''}{value}{unit}
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
          className="w-full h-1 rounded-lg appearance-none cursor-pointer"
          style={{
            backgroundColor: '#E2E8F0',
            background: `linear-gradient(to right, var(--primary-blue) 0%, var(--primary-blue) ${((value - min) / (max - min)) * 100}%, #E2E8F0 ${((value - min) / (max - min)) * 100}%, #E2E8F0 100%)`
          }}
        />
        <style>{`
          input[type="range"]::-webkit-slider-thumb {
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: var(--primary-blue);
            cursor: pointer;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
          }
          input[type="range"]::-moz-range-thumb {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: var(--primary-blue);
            cursor: pointer;
            border: none;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
          }
        `}</style>
        <div className="flex justify-between mt-1">
          <span className="text-xs" style={{ color: '#64748B' }}>{min}{unit}</span>
          <span className="text-xs" style={{ color: '#64748B' }}>+{max}{unit}</span>
        </div>
      </div>
    </div>
  );
}
