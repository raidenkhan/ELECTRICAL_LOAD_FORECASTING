import { WeeklyForecast } from './WeeklyForecast';
import { MonthlyComparison } from './MonthlyComparison';
import { ScenarioSimulator } from './ScenarioSimulator';

export function PlannerDashboard() {
  return (
    <div className="space-y-6">
      {/* Weekly Forecast Calendar */}
      <WeeklyForecast />

      {/* Two Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Monthly Comparison */}
        <MonthlyComparison />

        {/* Scenario Simulator */}
        <ScenarioSimulator />
      </div>
    </div>
  );
}
