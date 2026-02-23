import { api } from '@/lib/api';

export interface DataUploadResponse {
    upload_id: number;
    filename: string;
    file_size_bytes: number;
    row_count: number;
    status: string;
    message: string;
}

export interface RawDataUpload {
    id: number;
    filename: string;
    upload_timestamp: string;
    row_count: number;
    status: string;
}

export interface ValidationSummary {
    check_name: string;
    passed: boolean;
    details: any;
}

export interface ValidationReportResponse {
    report_id: number;
    upload_id: number;
    created_at: string;
    total_rows: number;
    valid_rows: number;
    invalid_rows: number;
    anomaly_count: number;
    passed: boolean;
    validation_checks: ValidationSummary[];
    error_messages: string;
}

export const dataService = {
    async uploadData(file: File): Promise<DataUploadResponse> {
        const formData = new FormData();
        formData.append('file', file);

        const response = await api.post<DataUploadResponse>('/data/upload', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    },

    async getUploads(skip: number = 0, limit: number = 10): Promise<{ uploads: RawDataUpload[] }> {
        const response = await api.get('/data/uploads', {
            params: { skip, limit }
        });
        return response.data;
    },

    async getValidationReport(uploadId: number): Promise<ValidationReportResponse> {
        const response = await api.get(`/data/validation/${uploadId}`);
        return response.data;
    },

    async getLatestData(limit: number = 100): Promise<any[]> {
        const response = await api.get('/data/latest', {
            params: { limit }
        });
        return response.data;
    }
};
