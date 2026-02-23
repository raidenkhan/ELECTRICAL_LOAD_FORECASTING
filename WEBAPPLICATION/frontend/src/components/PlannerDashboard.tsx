import { WeeklyForecast } from './WeeklyForecast';
import { MonthlyComparison } from './MonthlyComparison';
import { ScenarioSimulator } from './ScenarioSimulator';
import { useState, useEffect } from 'react';
import { forecastService, ForecastResponse } from '@/services/forecastService';
import { AlertTriangle, Clock } from 'lucide-react';

export function PlannerDashboard() {
  const [ltlf, setLtlf] = useState<ForecastResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        setIsLoading(true);
        // Fetch 30-day forecast (720 hours)
        const data = await forecastService.getLTLF(720);
        setLtlf(data);
        setError(null);
      } catch (err: any) {
        console.error('Failed to fetch LTLF:', err);
        setError(err.message || 'Failed to load long-term forecast data');
      } finally {
        setIsLoading(false);
      }
    }
    fetchData();
  }, []);

  return (
    <div className="space-y-6">
      {error && (
        <div
          className="p-6 border flex items-center gap-3 glass-morphism"
          style={{
            backgroundColor: 'color-mix(in srgb, var(--status-error), transparent 95%)',
            borderColor: 'var(--status-error)',
            color: 'var(--status-error)'
          }}
        >
          <AlertTriangle className="w-5 h-5" />
          <span className="font-bold tracking-tight uppercase" style={{ fontFamily: 'var(--font-geist-mono)' }}>{error}</span>
        </div>
      )}


      {isLoading && !ltlf && (
        <div
          className="p-12 flex flex-col items-center justify-center border glass-morphism"
          style={{
            backgroundColor: 'var(--bg-secondary)',
            borderColor: 'var(--border-primary)'
          }}
        >
          <Clock className="w-10 h-10 animate-spin mb-6" style={{ color: 'var(--status-info)' }} />
          <p className="text-sm font-bold tracking-widest uppercase mb-1" style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-geist-mono)' }}>
            Generating Long-Term Forecast...
          </p>
          <p className="text-[11px] font-medium" style={{ color: 'var(--text-tertiary)', fontFamily: 'var(--font-geist-mono)' }}>
            This may take a moment for 30-day horizons
          </p>
        </div>
      )}


      {/* Weekly Forecast Calendar */}
      <WeeklyForecast data={ltlf} isLoading={isLoading} />

      {/* Two Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Monthly Comparison */}
        <MonthlyComparison data={ltlf} isLoading={isLoading} />

        {/* Scenario Simulator */}
        <ScenarioSimulator />
      </div>
    </div>
  );
}
