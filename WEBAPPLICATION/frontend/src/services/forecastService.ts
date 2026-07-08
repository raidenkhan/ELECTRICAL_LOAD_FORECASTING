import { api } from '@/lib/api';

export const forecastService = {
    async getLatestData(limit: number = 100): Promise<any[]> {
        const response = await api.get('/data/latest', {
            params: { limit }
        });
        return response.data;
    },

    async getPeakDecomposition(): Promise<{ peak_mw: number, peak_hour: number, mean_mw: number, components: { name: string, value: number, color: string }[] }> {
        const response = await api.get('/explain/peak-decomposition');
        return response.data;
    },

    async getModelMetrics(): Promise<any> {
        const response = await api.get<any>('/models/metrics');
        return response.data;
    },

    async getAlerts(): Promise<any[]> {
        const response = await api.get<any[]>('/alerts/');
        return response.data;
    }
};
