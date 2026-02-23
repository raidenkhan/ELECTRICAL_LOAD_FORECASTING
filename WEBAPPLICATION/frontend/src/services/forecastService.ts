import { api } from '@/lib/api';

export interface ForecastRequest {
    horizon_hours: number;
    model_type: 'stlf' | 'ltlf';
}

export interface RegimeBin {
    hour: string;
    regime0: number;
    regime1: number;
    regime2: number;
}

export interface ForecastResponse {
    forecast_id: string;
    timestamp: string;
    horizon_hours: number;
    model_type: string;
    timestamps: string[];
    forecast_mw: number[];
    p10?: number[];
    p90?: number[];
    regime_distribution?: RegimeBin[];
    metadata?: Record<string, any>;
}

export const forecastService = {
    async getSTLF(horizonHours: number = 24): Promise<ForecastResponse> {
        const response = await api.post<ForecastResponse>('/forecast/stlf', {
            horizon_hours: horizonHours,
            model_type: 'stlf',
        });
        return response.data;
    },

    async getLTLF(horizonHours: number = 720): Promise<ForecastResponse> {
        const response = await api.post<ForecastResponse>('/forecast/ltlf', {
            horizon_hours: horizonHours,
            model_type: 'ltlf',
        });
        return response.data;
    },

    async getLatestMetrics() {
        // This is a helper to get current load and recent performance
        const response = await api.get('/data/uploads');
        return response.data;
    },

    async getLatestData(limit: number = 100): Promise<any[]> {
        const response = await api.get('/data/latest', {
            params: { limit }
        });
        return response.data;
    },

    async runSimulation(
        tempOffset: number,
        inflowOffset: number,
        industrialOffset: number,
        horizonHours: number = 24
    ): Promise<ForecastResponse> {
        const response = await api.post<ForecastResponse>('/forecast/simulate', {
            horizon_hours: horizonHours,
            temp_offset: tempOffset,
            inflow_offset_pct: inflowOffset,
            industrial_load_offset_pct: industrialOffset
        });
        return response.data;
    },

    async getShapValues(): Promise<{ features: string[], values: number[], base_value: number }> {
        const response = await api.get('/explain/shap');
        return response.data;
    },

    async getModelMetrics(): Promise<any[]> {
        const response = await api.get('/models/metrics');
        return response.data;
    }
};
