import { ForecastResponse } from '@/services/forecastService';

interface DayForecast {
  day: string;
  date: string;
  peakMW: number;
  time: string;
  exceedsThreshold: boolean;
}

interface WeeklyForecastProps {
  data: ForecastResponse | null;
  isLoading: boolean;
}

export function WeeklyForecast({ data, isLoading }: WeeklyForecastProps) {
  // Process LTLF data into 7-day view
  const processWeeklyData = (ltlf: ForecastResponse): DayForecast[] => {
    const days: DayForecast[] = [];
    const threshold = 1600;

    // Group by date (first 7 unique days)
    const groupedByDate: Record<string, { max: number, time: string }> = {};
    const uniqueDates: string[] = [];

    ltlf.timestamps.forEach((ts, idx) => {
      const dateObj = new Date(ts);
      const dateKey = dateObj.toLocaleDateString();
      const val = ltlf.forecast_mw[idx];

      if (!groupedByDate[dateKey]) {
        groupedByDate[dateKey] = { max: val, time: ts };
        uniqueDates.push(dateKey);
      } else if (val > groupedByDate[dateKey].max) {
        groupedByDate[dateKey] = { max: val, time: ts };
      }
    });

    // Take the first 7 days
    uniqueDates.slice(0, 7).forEach(dateKey => {
      const dateObj = new Date(groupedByDate[dateKey].time);
      days.push({
        day: dateObj.toLocaleDateString([], { weekday: 'short' }).toUpperCase(),
        date: `${dateObj.getMonth() + 1}/${dateObj.getDate()}`,
        peakMW: Math.round(groupedByDate[dateKey].max),
        time: dateObj.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        exceedsThreshold: groupedByDate[dateKey].max > threshold
      });
    });

    return days;
  };

  const displayData = data ? processWeeklyData(data) : [];
  const startDay = displayData.length > 0 ? displayData[0].date : '...';
  const endDay = displayData.length > 0 ? displayData[displayData.length - 1].date : '...';
  const year = new Date().getFullYear();
  return (
    <div
      className="glass-morphism"
    >

      {/* Header */}
      <div className="px-6 py-4 border-b" style={{ borderColor: 'var(--border-primary)', opacity: 0.5 }}>

        <h3 className="text-base font-semibold" style={{ color: 'var(--text-primary)' }}>
          Weekly Load Forecast
        </h3>
        <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>
          Week of {startDay} - {endDay}, {year}
        </p>
      </div>

      {/* Day Cards Grid */}
      <div className="p-6">
        <div className="grid grid-cols-7 gap-3">
          {isLoading && displayData.length === 0 ? (
            Array(7).fill(0).map((_, i) => (
              <div key={i} className="h-40 animate-pulse border" style={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border-primary)', opacity: 0.5 }} />
            ))

          ) : (
            displayData.map((day) => (
              <DayCard key={day.date} {...day} />
            ))
          )}
        </div>

        {/* Legend */}
        <div className="flex items-center gap-2 pt-6 mt-6 border-t" style={{ borderColor: 'var(--border-primary)', opacity: 0.5 }}>
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: 'var(--status-error)' }} />

          <span className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            Exceeds 1,600 MW threshold
          </span>
        </div>
      </div>
    </div>
  );
}

function DayCard({ day, date, peakMW, time, exceedsThreshold }: DayForecast) {
  return (
    <div
      className="glass-morphism transition-all duration-200 cursor-pointer relative"
      style={{
        borderColor: exceedsThreshold ? 'var(--status-error)' : 'var(--border-primary)'
      }}

      onMouseEnter={(e) => {
        e.currentTarget.style.backgroundColor = 'var(--hover-bg)';
        e.currentTarget.style.transform = 'translateY(-2px)';
      }}

      onMouseLeave={(e) => {
        e.currentTarget.style.backgroundColor = 'var(--bg-secondary)';
        e.currentTarget.style.transform = 'translateY(0)';
      }}
    >
      {/* Alert Indicator */}
      {exceedsThreshold && (
        <div
          className="absolute top-2 right-2 w-2 h-2 rounded-full"
          style={{ backgroundColor: 'var(--status-error)' }}
        />
      )}


      {/* Day Header */}
      <div
        className="px-3 py-2 border-b"
        style={{
          borderColor: 'var(--border-primary)',
          opacity: 0.8
        }}
      >


        <p className="text-xs font-bold text-center tracking-widest uppercase" style={{
          color: 'var(--text-tertiary)',
          fontFamily: 'var(--font-geist-mono)'
        }}>
          {day}
        </p>

        <p className="text-sm font-medium text-center mt-0.5" style={{ color: 'var(--text-primary)' }}>
          {date}
        </p>
      </div>

      {/* Card Body */}
      <div className="flex flex-col items-center justify-center p-4" style={{ minHeight: '140px' }}>
        <p className="text-[10px] font-bold tracking-widest uppercase mb-2" style={{
          color: 'var(--text-tertiary)',
          fontFamily: 'var(--font-geist-mono)'
        }}>
          PEAK
        </p>

        <p className="text-2xl font-black tracking-tighter mb-1" style={{
          color: exceedsThreshold ? 'var(--status-error)' : 'var(--text-primary)'
        }}>
          {peakMW.toLocaleString()}
        </p>

        <p className="text-[10px] font-bold uppercase tracking-widest px-2 py-1" style={{
          color: 'var(--text-tertiary)',
          backgroundColor: 'var(--bg-surface)',
          border: '1px solid var(--border-primary)',
          fontFamily: 'var(--font-geist-mono)'
        }}>
          {time}
        </p>

      </div>
    </div>
  );
}
