import { api } from '@/lib/api';

export interface DispatchForecastResponse {
  forecast_date: string;
  forecast_mw: number[];
  p10_mw?: number[];
  p90_mw?: number[];
  uncertainty_mw?: number[];
  temperature_c?: number[];
  components?: {
    trend: number[];
    temp_effect: number[];
    holiday_effect: number[];
    growth_effect: number[];
    kalman_bias: number[];
  };
  factors?: {
    trend_mw: number[];
    seasonal_ratio: number[];
    temp_ratio: number[];
    holiday_ratio: number[];
    growth_ratio: number[];
  };
}

export interface DailyAggregate {
  date: string;
  peak_mw: number;
  mean_mw: number;
  min_mw: number;
  total_energy_mwh: number;
}

export interface WeeklyAggregate {
  week_start: string;
  week_end: string;
  mean_mw: number;
  peak_mw: number;
  min_mw: number;
  total_energy_mwh: number;
}

export interface Forecast7DayResponse {
  forecast_date: string;
  hourly_mw: number[];
  p10_mw?: number[];
  p90_mw?: number[];
  uncertainty_mw?: number[];
  temperature_c?: number[];
  daily_aggregates: DailyAggregate[];
}

export interface Forecast30DayResponse {
  forecast_date: string;
  daily_aggregates: DailyAggregate[];
}

export interface Forecast90DayResponse {
  forecast_date: string;
  weekly_aggregates: WeeklyAggregate[];
}

export const dispatchForecastService = {
  async getTomorrow(forceRefresh = false): Promise<DispatchForecastResponse> {
    const params = forceRefresh ? '?force_refresh=true' : '';
    const response = await api.get<DispatchForecastResponse>(`/forecast/dispatch/tomorrow${params}`);
    return response.data;
  },

  async getForDate(targetDate: string): Promise<DispatchForecastResponse> {
    const response = await api.post<DispatchForecastResponse>('/forecast/dispatch', {
      target_date: targetDate,
    });
    return response.data;
  },

  async get7Day(forceRefresh = false): Promise<Forecast7DayResponse> {
    const params = forceRefresh ? '?force_refresh=true' : '';
    const response = await api.get<Forecast7DayResponse>(`/forecast/dispatch/7day${params}`);
    return response.data;
  },

  async get30Day(forceRefresh = false): Promise<Forecast30DayResponse> {
    const params = forceRefresh ? '?force_refresh=true' : '';
    const response = await api.get<Forecast30DayResponse>(`/forecast/dispatch/30day${params}`);
    return response.data;
  },

  async get90Day(forceRefresh = false): Promise<Forecast90DayResponse> {
    const params = forceRefresh ? '?force_refresh=true' : '';
    const response = await api.get<Forecast90DayResponse>(`/forecast/dispatch/90day${params}`);
    return response.data;
  },

  async getHealth(): Promise<any> {
    const response = await api.get('/models/metrics');
    return response.data;
  }};

