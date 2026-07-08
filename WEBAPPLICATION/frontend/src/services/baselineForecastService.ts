import { api } from '@/lib/api';

export interface BaselineForecastResponse {
  forecast_date: string;
  forecast_mw: number[];
  factors?: {
    level_mw: number;
    dow_offset: number;
    profile: number[];
    dow: number;
    month: number;
  };
}

export interface DailyAggregate {
  date: string;
  peak_mw: number;
  mean_mw: number;
  min_mw: number;
  total_energy_mwh: number;
}

export interface Baseline7DayResponse {
  forecast_date: string;
  hourly_mw: number[];
  daily_aggregates: DailyAggregate[];
}

export interface Baseline30DayResponse {
  forecast_date: string;
  daily_aggregates: DailyAggregate[];
}

export interface Baseline90DayResponse {
  forecast_date: string;
  weekly_aggregates: WeeklyAggregate[];
}

export interface WeeklyAggregate {
  week_start: string;
  week_end: string;
  mean_mw: number;
  peak_mw: number;
  min_mw: number;
  total_energy_mwh: number;
}

export interface DataFreshnessInfo {
  latest_date: string;
  days_stale: number;
  status: 'fresh' | 'stale' | 'old' | 'unknown';
}

export interface BaselineUploadResponse {
  status: string;
  records_loaded: number;
  latest_date: string;
  days_loaded: number;
  forecast: BaselineForecastResponse;
  freshness: DataFreshnessInfo;
}

export const baselineForecastService = {
  async getTomorrow(forceRefresh = false): Promise<BaselineForecastResponse> {
    const params = forceRefresh ? '?force_refresh=true' : '';
    const response = await api.get<BaselineForecastResponse>(`/forecast/baseline/tomorrow${params}`);
    return response.data;
  },

  async get7Day(forceRefresh = false): Promise<Baseline7DayResponse> {
    const params = forceRefresh ? '?force_refresh=true' : '';
    const response = await api.get<Baseline7DayResponse>(`/forecast/baseline/7day${params}`);
    return response.data;
  },

  async get30Day(forceRefresh = false): Promise<Baseline30DayResponse> {
    const params = forceRefresh ? '?force_refresh=true' : '';
    const response = await api.get<Baseline30DayResponse>(`/forecast/baseline/30day${params}`);
    return response.data;
  },

  async get90Day(forceRefresh = false): Promise<Baseline90DayResponse> {
    const params = forceRefresh ? '?force_refresh=true' : '';
    const response = await api.get<Baseline90DayResponse>(`/forecast/baseline/90day${params}`);
    return response.data;
  },

  async getFreshness(): Promise<DataFreshnessInfo> {
    const response = await api.get<DataFreshnessInfo>('/forecast/baseline/freshness');
    return response.data;
  },

  async uploadScadaData(file: File): Promise<BaselineUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post<BaselineUploadResponse>('/forecast/baseline/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },
};
