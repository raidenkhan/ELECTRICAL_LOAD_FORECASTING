import { Upload, Database, Cloud, Settings as SettingsIcon, CheckCircle, XCircle, Clock, BarChart3, RefreshCw } from 'lucide-react';
import { useState, useEffect } from 'react';
import { dataService, RawDataUpload } from '@/services/dataService';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface DataSource {
  id: string;
  icon: React.ReactNode;
  title: string;
  status: 'connected' | 'disconnected';
  lastSync?: string;
}

const dataSources: DataSource[] = [
  {
    id: 'internal-db',
    icon: <Database className="w-7 h-7" />,
    title: 'Internal Database',
    status: 'connected',
    lastSync: 'Live'
  },
  {
    id: 'weather-api',
    icon: <Cloud className="w-7 h-7" />,
    title: 'Weather API',
    status: 'connected',
    lastSync: '10 mins ago'
  }
];

export function DataManagement() {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadHistory, setUploadHistory] = useState<RawDataUpload[]>([]);
  const [previewData, setPreviewData] = useState<any[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [isLoadingPreview, setIsLoadingPreview] = useState(false);
  const [message, setMessage] = useState<{ text: string, type: 'success' | 'error' } | null>(null);

  useEffect(() => {
    fetchHistory();
    fetchPreview();
  }, []);

  const fetchHistory = async () => {
    try {
      setIsLoadingHistory(true);
      const data = await dataService.getUploads(0, 5);
      setUploadHistory(data.uploads);
    } catch (err) {
      console.error('Failed to fetch history:', err);
    } finally {
      setIsLoadingHistory(false);
    }
  };

  const fetchPreview = async () => {
    try {
      setIsLoadingPreview(true);
      const data = await dataService.getLatestData(50);
      // Map for Recharts
      const mapped = data.map(d => ({
        time: new Date(d.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        load: d.total_load_mw
      })).reverse();
      setPreviewData(mapped);
    } catch (err) {
      console.error('Failed to fetch preview:', err);
    } finally {
      setIsLoadingPreview(false);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    try {
      setIsUploading(true);
      setMessage(null);
      const response = await dataService.uploadData(selectedFile);
      setMessage({ text: response.message, type: response.status === 'validated' ? 'success' : 'error' });
      setSelectedFile(null);
      fetchHistory();
      fetchPreview();
    } catch (err: any) {
      setMessage({ text: err.response?.data?.detail || 'Upload failed', type: 'error' });
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Section: Upload & Preview */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Upload Column */}
        <div className="lg:col-span-1 p-6 glass-morphism">
          <h3 className="mb-2" style={{
            fontSize: 'var(--text-lg)',
            fontWeight: 'var(--font-weight-semibold)',
            color: 'var(--text-primary)'
          }}>
            Upload SCADA Data
          </h3>
          <p className="text-sm mb-6" style={{ color: 'var(--text-secondary)' }}>
            Import 15-min interval CSV measurements.
          </p>

          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            className="mb-4 flex flex-col items-center justify-center py-10 px-4 transition-all duration-200"
            style={{
              border: `2px dashed ${dragActive ? 'var(--status-info)' : 'var(--border-primary)'}`,
              backgroundColor: dragActive ? 'var(--hover-bg)' : 'transparent'
            }}
          >
            <Upload className="w-10 h-10 mb-3" style={{ color: '#94A3B8' }} />
            <p className="text-xs mb-2 text-center" style={{ color: 'var(--text-secondary)' }}>
              Drag files here or click
            </p>
            <input
              type="file"
              id="file-upload"
              className="hidden"
              accept=".csv"
              onChange={handleFileSelect}
            />
            <label
              htmlFor="file-upload"
              className="px-4 py-2 text-xs rounded transition-all duration-200"
              style={{
                backgroundColor: 'var(--input-bg)',
                border: '1px solid var(--input-border)',
                color: 'var(--text-primary)',
                cursor: 'pointer'
              }}
            >
              Choose File
            </label>

            {selectedFile && (
              <p className="text-xs mt-3 font-semibold" style={{ color: 'var(--primary-blue)' }}>
                {selectedFile.name}
              </p>
            )}
          </div>

          <button
            onClick={handleUpload}
            disabled={!selectedFile || isUploading}
            className="w-full py-3 text-sm font-bold tracking-widest uppercase border transition-all active:scale-95 disabled:opacity-50"
            style={{
              borderColor: 'var(--lime-primary)',
              color: 'var(--lime-primary)',
              fontFamily: 'var(--font-geist-mono)',
            }}
          >
            {isUploading ? 'Processing...' : 'Upload & Validate'}
          </button>

          {message && (
            <div className={`mt-4 p-3 border text-xs flex items-center gap-2 ${message.type === 'success' ? 'bg-green-50' : 'bg-red-50'}`}
              style={{ borderColor: message.type === 'success' ? 'var(--success-green)' : '#DC2626' }}>
              {message.type === 'success' ? <CheckCircle className="w-4 h-4 text-green-600" /> : <XCircle className="w-4 h-4 text-red-600" />}
              <span className={message.type === 'success' ? 'text-green-700' : 'text-red-700'}>
                {message.text}
              </span>
            </div>
          )}
        </div>

        {/* Live Preview Chart */}
        <div className="lg:col-span-2 p-6 glass-morphism">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="mb-1" style={{
                fontSize: 'var(--text-lg)',
                fontWeight: 'var(--font-weight-semibold)',
                color: 'var(--text-primary)'
              }}>
                SCADA Data Preview
              </h3>
              <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                Last 50 measurements (Real-time stream)
              </p>
            </div>
            <button
              onClick={fetchPreview}
              className="p-2 border rounded-md transition-colors"
              style={{ borderColor: 'var(--input-border)', backgroundColor: 'var(--input-bg)' }}
            >
              <RefreshCw className={`w-4 h-4 ${isLoadingPreview ? 'animate-spin' : ''}`} style={{ color: 'var(--text-tertiary)' }} />
            </button>

          </div>

          <div className="h-[250px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={previewData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis
                  dataKey="time"
                  hide={false}
                  stroke="#94a3b8"
                  fontSize={10}
                  tickLine={false}
                  axisLine={{ stroke: '#e2e8f0' }}
                  interval={5}
                />
                <YAxis
                  stroke="#94a3b8"
                  fontSize={10}
                  tickLine={false}
                  axisLine={{ stroke: '#e2e8f0' }}
                  domain={['auto', 'auto']}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: 'white', border: '1px solid var(--border-default)', fontSize: '12px' }}
                />
                <Line
                  type="monotone"
                  dataKey="load"
                  stroke="var(--primary-blue)"
                  strokeWidth={2}
                  dot={false}
                  animationDuration={1000}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Bottom Section: Sources & History */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Data Sources */}
        <div className="p-6 glass-morphism">
          <h3 className="mb-4" style={{
            fontSize: 'var(--text-lg)',
            fontWeight: 'var(--font-weight-semibold)',
            color: 'var(--text-primary)'
          }}>
            Data Connectors
          </h3>
          <div className="space-y-2">
            {dataSources.map((source, index) => (
              <DataSourceItem key={source.id} {...source} isLast={index === dataSources.length - 1} />
            ))}
          </div>
        </div>

        {/* Recent Uploads */}
        <div className="p-6 glass-morphism">
          <h3 className="mb-4" style={{
            fontSize: 'var(--text-lg)',
            fontWeight: 'var(--font-weight-semibold)',
            color: 'var(--text-primary)'
          }}>
            Recent Activity
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b" style={{ borderColor: 'var(--border-primary)' }}>
                  <th className="text-left py-2 text-[10px] font-bold text-gray-400 uppercase tracking-widest">File</th>
                  <th className="text-left py-2 text-[10px] font-bold text-gray-400 uppercase tracking-widest">Rows</th>
                  <th className="text-left py-2 text-[10px] font-bold text-gray-400 uppercase tracking-widest">Date</th>
                  <th className="text-left py-2 text-[10px] font-bold text-gray-400 uppercase tracking-widest">Status</th>
                </tr>
              </thead>
              <tbody>
                {isLoadingHistory ? (
                  Array(3).fill(0).map((_, i) => (
                    <tr key={i} className="animate-pulse">
                      <td colSpan={4} className="h-10 bg-gray-50/50 rounded mt-2" />
                    </tr>
                  ))
                ) : (
                  uploadHistory.map((record) => (
                    <TableRow key={record.id} {...record} />
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function DataSourceItem({ icon, title, status, lastSync, isLast }: DataSource & { isLast: boolean }) {
  const config = {
    connected: { bg: '#DCFCE7', text: '#059669', label: 'Connected' },
    disconnected: { bg: '#F1F5F9', text: '#64748B', label: 'Disconnected' }
  }[status];

  return (
    <div className="flex items-center justify-between py-3" style={{ borderBottom: isLast ? 'none' : '1px solid var(--border-primary)' }}>
      <div className="flex items-center gap-3">
        <div className="p-2 rounded" style={{ backgroundColor: 'var(--input-bg)', color: 'var(--text-secondary)' }}>
          {icon}
        </div>

        <div>
          <p className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>{title}</p>
          <p className="text-[10px] text-gray-400">Sync: {lastSync}</p>
        </div>
      </div>
      <div className="px-2 py-1 rounded-full text-[10px] font-bold uppercase" style={{ backgroundColor: config.bg, color: config.text }}>
        {config.label}
      </div>
    </div>
  );
}

function TableRow({ filename, row_count, upload_timestamp, status }: RawDataUpload) {
  const config = {
    validated: { bg: '#DCFCE7', text: '#059669', label: 'Validated' },
    failed: { bg: '#FEE2E2', text: '#DC2626', label: 'Failed' },
    validating: { bg: '#FEF3C7', text: '#F59E0B', label: 'Checking' }
  }[status] || { bg: '#F1F5F9', text: '#64748B', label: status };

  return (
    <tr className="border-b hover:bg-gray-50/50 transition-colors" style={{ borderColor: 'var(--border-primary)' }}>
      <td className="py-3 text-xs font-medium" style={{ color: 'var(--text-primary)' }}>{filename}</td>
      <td className="py-3 text-xs text-gray-500">{row_count.toLocaleString()}</td>
      <td className="py-3 text-xs text-gray-500">{new Date(upload_timestamp).toLocaleDateString()}</td>
      <td className="py-3">
        <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase" style={{ backgroundColor: config.bg, color: config.text }}>
          {config.label}
        </span>
      </td>
    </tr>
  );
}
