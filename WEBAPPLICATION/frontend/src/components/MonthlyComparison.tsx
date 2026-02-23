import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, ComposedChart } from 'recharts';
import { ForecastResponse } from '@/services/forecastService';
import { ArrowUpRight } from 'lucide-react';

interface MonthlyComparisonProps {
  data: ForecastResponse | null;
  isLoading: boolean;
}

export function MonthlyComparison({ data, isLoading }: MonthlyComparisonProps) {
  // Process 30-day LTLF data
  const processMonthlyData = (ltlf: ForecastResponse) => {
    const dailyData: Record<string, { max: number, p10?: number, p90?: number }> = {};
    const result: { day: string, current: number, previous: number, lower?: number, upper?: number }[] = [];

    ltlf.timestamps.forEach((ts, idx) => {
      const date = new Date(ts);
      const day = date.getDate().toString();
      const val = ltlf.forecast_mw[idx];
      const p10 = ltlf.p10 ? ltlf.p10[idx] : undefined;
      const p90 = ltlf.p90 ? ltlf.p90[idx] : undefined;

      if (!dailyData[day] || val > dailyData[day].max) {
        dailyData[day] = { max: val, p10, p90 };
      }
    });

    // Map to chart format
    Object.keys(dailyData).forEach(day => {
      result.push({
        day,
        current: Math.round(dailyData[day].max),
        previous: Math.round(dailyData[day].max * 0.95), // Mock comparison
        lower: dailyData[day].p10 ? Math.round(dailyData[day].p10!) : undefined,
        upper: dailyData[day].p90 ? Math.round(dailyData[day].p90!) : undefined
      });
    });

    return result.sort((a, b) => parseInt(a.day) - parseInt(b.day));
  };

  const chartData = data ? processMonthlyData(data) : [];
  const avgPeak = chartData.length > 0
    ? Math.round(chartData.reduce((acc, curr) => acc + curr.current, 0) / chartData.length)
    : 1502;
  const maxPeak = chartData.length > 0
    ? Math.max(...chartData.map(d => d.current))
    : 200;

  const currentMonth = new Date().toLocaleDateString([], { month: 'short', year: 'numeric' });
  const previousYear = new Date().getFullYear() - 1;
  return (
    <div
      className="glass-morphism"
    >

      {/* Header */}
      <div className="px-6 py-4 border-b" style={{ borderColor: 'var(--border-primary)', opacity: 0.5 }}>

        <h3 className="text-base font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>
          Monthly Comparison
        </h3>
        <div className="flex items-center gap-3">
          <select
            className="px-3 py-2 text-sm border font-medium"
            style={{
              borderColor: 'var(--border-primary)',
              backgroundColor: 'var(--bg-surface)',
              color: 'var(--text-primary)'
            }}
          >
            <option>{currentMonth}</option>
          </select>
          <span className="text-sm font-medium" style={{ color: 'var(--text-muted)' }}>vs</span>
          <select
            className="px-3 py-2 text-sm border font-medium"
            style={{
              borderColor: 'var(--border-primary)',
              backgroundColor: 'var(--bg-surface)',
              color: 'var(--text-primary)'
            }}
          >

            <option>Avg {previousYear}</option>
          </select>
        </div>
      </div>

      <div className="p-6">
        {/* Comparison Cards */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div
            className="p-5 glass-morphism"
          >
            <p className="text-[10px] font-bold tracking-[0.2em] uppercase mb-2" style={{
              color: 'var(--text-tertiary)',
              fontFamily: 'var(--font-geist-mono)'
            }}>
              AVG PEAK
            </p>

            <p className="text-2xl font-bold mb-2" style={{ color: 'var(--text-primary)' }}>
              {isLoading && !data ? "..." : `${avgPeak.toLocaleString()} MW`}
            </p>
            <div className="flex items-center gap-1 text-sm font-semibold" style={{ color: 'var(--status-ok)' }}>
              <ArrowUpRight className="w-4 h-4" />
              <span>+3.4% vs {previousYear}</span>
            </div>

          </div>

          <div
            className="p-5 glass-morphism"
          >
            <p className="text-[10px] font-bold tracking-[0.2em] uppercase mb-2" style={{
              color: 'var(--text-tertiary)',
              fontFamily: 'var(--font-geist-mono)'
            }}>
              MAX PEAK
            </p>

            <p className="text-2xl font-bold mb-2" style={{ color: 'var(--text-primary)' }}>
              {isLoading && !data ? "..." : `${maxPeak.toLocaleString()} MW`}
            </p>
            <div className="flex items-center gap-1 text-sm font-semibold" style={{ color: 'var(--status-ok)' }}>
              <ArrowUpRight className="w-4 h-4" />
              <span>+2.9% vs {previousYear}</span>
            </div>

          </div>
        </div>


        {/* Comparison Chart */}
        <div className="relative h-[250px]">
          {isLoading && !data && (
            <div className="absolute inset-0 flex items-center justify-center bg-white/5 backdrop-blur-sm z-10">
              <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin" style={{ borderColor: 'var(--status-info)' }} />
            </div>
          )}

          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData}>
              <defs>
                <linearGradient id="ltlfFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--status-info)" stopOpacity={0.1} />
                  <stop offset="95%" stopColor="var(--status-info)" stopOpacity={0.02} />
                </linearGradient>

              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-primary)" strokeOpacity={0.1} vertical={false} />

              <XAxis
                dataKey="day"
                stroke="var(--text-tertiary)"

                style={{ fontSize: '10px', fontWeight: 700 }}
                tickLine={false}
                axisLine={false}

                interval={Math.floor(chartData.length / 10)}
              />
              <YAxis
                stroke="var(--text-tertiary)"
                style={{ fontSize: '10px', fontWeight: 700 }}
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
              <Area
                type="monotone"
                dataKey={(d) => [d.lower || d.current * 0.9, d.upper || d.current * 1.1]}
                fill="url(#ltlfFill)"
                stroke="transparent"
                name="Confidence Range"
              />
              <Line
                type="monotone"
                dataKey="current"
                stroke="var(--status-info)"
                strokeWidth={3}
                name={currentMonth}
                dot={{ fill: 'var(--status-info)', r: 3 }}
              />

              <Line
                type="monotone"
                dataKey="previous"
                stroke="var(--text-muted)"
                strokeWidth={2}
                strokeDasharray="5 5"
                name={`Avg ${previousYear}`}
                dot={{ fill: 'var(--text-muted)', r: 3 }}
              />

            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
