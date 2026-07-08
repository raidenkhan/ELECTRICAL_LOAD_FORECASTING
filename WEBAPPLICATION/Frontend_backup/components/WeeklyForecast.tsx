interface DayForecast {
  day: string;
  date: string;
  peakMW: number;
  time: string;
  exceedsThreshold: boolean;
}

const weeklyData: DayForecast[] = [
  { day: 'MON', date: '2/3', peakMW: 98, time: '14:30', exceedsThreshold: false },
  { day: 'TUE', date: '2/4', peakMW: 102, time: '14:15', exceedsThreshold: true },
  { day: 'WED', date: '2/5', peakMW: 95, time: '15:00', exceedsThreshold: false },
  { day: 'THU', date: '2/6', peakMW: 99, time: '14:45', exceedsThreshold: false },
  { day: 'FRI', date: '2/7', peakMW: 105, time: '13:30', exceedsThreshold: true },
  { day: 'SAT', date: '2/8', peakMW: 88, time: '12:00', exceedsThreshold: false },
  { day: 'SUN', date: '2/9', peakMW: 82, time: '11:30', exceedsThreshold: false }
];

export function WeeklyForecast() {
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
      <div className="flex items-center justify-between mb-6">
        <h3 style={{ 
          fontSize: 'var(--text-lg)',
          fontWeight: 'var(--font-weight-semibold)',
          color: 'var(--text-primary)'
        }}>
          Weekly Load Forecast
        </h3>
        <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
          Week of Feb 3 - Feb 9
        </span>
      </div>

      {/* Day Cards Grid */}
      <div className="grid grid-cols-7 gap-4 mb-4">
        {weeklyData.map((day) => (
          <DayCard key={day.day} {...day} />
        ))}
      </div>

      {/* Legend */}
      <div className="flex items-center gap-2 pt-4" style={{ borderTop: '1px solid var(--border-default)' }}>
        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: 'var(--danger-red)' }} />
        <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
          Exceeds 100 MW threshold
        </span>
      </div>
    </div>
  );
}

function DayCard({ day, date, peakMW, time, exceedsThreshold }: DayForecast) {
  return (
    <div 
      className="rounded-lg overflow-hidden transition-all duration-200 cursor-pointer relative"
      style={{
        backgroundColor: 'var(--bg-secondary)',
        border: '1px solid var(--border-default)',
        height: '200px'
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.12)';
        e.currentTarget.style.transform = 'translateY(-2px)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.boxShadow = 'none';
        e.currentTarget.style.transform = 'translateY(0)';
      }}
    >
      {/* Alert Indicator */}
      {exceedsThreshold && (
        <div 
          className="absolute top-3 right-3 w-3 h-3 rounded-full"
          style={{ backgroundColor: 'var(--danger-red)' }}
        />
      )}

      {/* Day Header */}
      <div 
        className="px-3 py-2 border-b"
        style={{
          backgroundColor: '#F8FAFC',
          borderColor: 'var(--border-default)',
          height: '40px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center'
        }}
      >
        <p className="text-xs" style={{ 
          fontWeight: 'var(--font-weight-semibold)',
          color: '#64748B'
        }}>
          {day}
        </p>
        <p className="text-sm" style={{ color: 'var(--text-primary)' }}>
          {date}
        </p>
      </div>

      {/* Card Body */}
      <div className="flex flex-col items-center justify-center p-4" style={{ height: 'calc(200px - 40px)' }}>
        <p className="text-xs mb-2" style={{ 
          fontWeight: 'var(--font-weight-medium)',
          color: '#64748B'
        }}>
          Peak
        </p>
        <p className="text-3xl mb-2" style={{ 
          fontWeight: 'var(--font-weight-bold)',
          color: exceedsThreshold ? 'var(--danger-red)' : 'var(--text-primary)'
        }}>
          {peakMW}
        </p>
        <p className="text-xs" style={{ 
          fontWeight: 'var(--font-weight-medium)',
          color: '#64748B'
        }}>
          MW
        </p>
        <div className="flex-1" />
        <p className="text-xs mt-2" style={{ color: '#64748B' }}>
          {time}
        </p>
      </div>
    </div>
  );
}
