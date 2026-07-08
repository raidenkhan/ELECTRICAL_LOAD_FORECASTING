import { api } from '@/lib/api';

export interface HourlyDemandItem {
  hour: number;
  entity_name: string;
  demand_mw: number;
  is_forecasted: boolean;
}

export interface HourlySupplyItem {
  hour: number;
  plant_name: string;
  supply_mw: number;
  is_baseload?: boolean;
  category?: string;
}

export interface ScheduleDetail {
  id: number;
  date: string;
  status: string;
  source_filename: string;
  operator_notes?: string;
  created_at: string;
  updated_at: string;
  demand: HourlyDemandItem[];
  supply: HourlySupplyItem[];
}

export interface HourlyAggregation {
  hour: number;
  ecg_forecast_mw: number;
  nedco_mw: number;
  valco_mw: number;
  mines_mw: number;
  export_mw: number;
  total_demand_mw: number;
  total_supply_mw: number;
  reserve_mw: number;
  reserve_pct: number;
}

export interface AggregatedSchedule {
  schedule_id: number;
  schedule_date: string;
  status: string;
  source_filename: string;
  operator_notes?: string;
  hourly: HourlyAggregation[];
  peak_demand_mw: number;
  peak_demand_hour: number;
  total_energy_demand_mwh: number;
  total_energy_supply_mwh: number;
  avg_demand_mw: number;
  avg_supply_mw: number;
  min_reserve_mw: number;
  min_reserve_hour: number;
  using_forecast: boolean;
  computed_at: string;
}

export interface ScheduleUploadResponse {
  id: number;
  date: string;
  status: string;
  source_filename: string;
  demand_count: number;
  supply_count: number;
  message: string;
}

export const scheduleService = {
  async uploadSchedule(file: File): Promise<ScheduleUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post<ScheduleUploadResponse>('/schedule/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  async getSchedule(id: number): Promise<ScheduleDetail> {
    const response = await api.get<ScheduleDetail>(`/schedule/${id}`);
    return response.data;
  },

  async getLatestSchedule(): Promise<ScheduleDetail> {
    const response = await api.get<ScheduleDetail>('/schedule/latest');
    return response.data;
  },

  async updateCell(
    scheduleId: number,
    table: 'demand' | 'supply',
    entityName: string,
    hour: number,
    value: number,
  ): Promise<ScheduleDetail> {
    const response = await api.patch<ScheduleDetail>(`/schedule/${scheduleId}/cell`, {
      table,
      entity_name: entityName,
      hour,
      value,
    });
    return response.data;
  },

  async confirmSchedule(scheduleId: number, notes?: string): Promise<ScheduleDetail> {
    const response = await api.post<ScheduleDetail>(`/schedule/${scheduleId}/confirm`, {
      operator_notes: notes || null,
    });
    return response.data;
  },

  async reviseSchedule(scheduleId: number, notes: string): Promise<ScheduleDetail> {
    const response = await api.post<ScheduleDetail>(`/schedule/${scheduleId}/revise`, {
      operator_notes: notes,
    });
    return response.data;
  },

  async autoFillForecast(scheduleId: number): Promise<ScheduleDetail> {
    const response = await api.post<ScheduleDetail>(`/schedule/${scheduleId}/auto-fill-forecast`);
    return response.data;
  },

  async getAggregated(scheduleId: number, useForecast: boolean = true): Promise<AggregatedSchedule> {
    const response = await api.get<AggregatedSchedule>(`/schedule/${scheduleId}/aggregated`, {
      params: { use_forecast: useForecast },
    });
    return response.data;
  },
};

export interface AuditLogEntry {
  id: number;
  schedule_id: number;
  action: string;
  description: string;
  details?: Record<string, any>;
  user_id?: number | null;
  created_at: string;
  hash: string;
  previous_hash: string;
}

export const auditService = {
  async getForSchedule(scheduleId: number): Promise<AuditLogEntry[]> {
    const response = await api.get<AuditLogEntry[]>(`/schedule/${scheduleId}/audit-logs`);
    return response.data;
  },
  async verifyChain(scheduleId: number): Promise<{ valid: boolean; count: number; failures: any[] }> {
    const response = await api.get(`/schedule/${scheduleId}/audit-logs/verify`);
    return response.data;
  },
};
