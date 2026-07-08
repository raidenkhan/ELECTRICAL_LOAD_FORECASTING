import { Upload, Database, Cloud, Settings as SettingsIcon, CheckCircle, XCircle, Clock } from 'lucide-react';
import { useState } from 'react';

interface DataSource {
  id: string;
  icon: React.ReactNode;
  title: string;
  status: 'connected' | 'disconnected';
  lastSync?: string;
}

interface UploadRecord {
  fileName: string;
  type: string;
  size: string;
  uploadDate: string;
  status: 'processed' | 'processing' | 'failed';
}

const dataSources: DataSource[] = [
  {
    id: 'internal-db',
    icon: <Database className="w-7 h-7" />,
    title: 'Internal Database',
    status: 'connected',
    lastSync: '2 hours ago'
  },
  {
    id: 'weather-api',
    icon: <Cloud className="w-7 h-7" />,
    title: 'Weather API',
    status: 'connected',
    lastSync: '1 hour ago'
  },
  {
    id: 'maintenance',
    icon: <SettingsIcon className="w-7 h-7" />,
    title: 'Maintenance System',
    status: 'disconnected',
    lastSync: undefined
  }
];

const uploadHistory: UploadRecord[] = [
  {
    fileName: 'outage_data.csv',
    type: 'History',
    size: '2.3 MB',
    uploadDate: '2 days ago',
    status: 'processed'
  },
  {
    fileName: 'sensor_feb.csv',
    type: 'Sensor',
    size: '1.8 MB',
    uploadDate: '5 days ago',
    status: 'processed'
  }
];

export function DataManagement() {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

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

  return (
    <div className="space-y-6">
      {/* Two Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload Section */}
        <div 
          className="rounded-lg p-6"
          style={{
            backgroundColor: 'var(--bg-secondary)',
            border: '1px solid var(--border-default)',
            boxShadow: '0 2px 8px rgba(0,0,0,0.06)'
          }}
        >
          <h3 className="mb-2" style={{ 
            fontSize: 'var(--text-lg)',
            fontWeight: 'var(--font-weight-semibold)',
            color: 'var(--text-primary)'
          }}>
            Upload Historical Data
          </h3>
          <p className="text-sm mb-6" style={{ color: 'var(--text-secondary)' }}>
            Import historical outage data for model training
          </p>

          {/* Upload Zone */}
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            className="rounded-lg mb-4 flex flex-col items-center justify-center py-16 px-8 transition-all duration-200"
            style={{
              border: `2px dashed ${dragActive ? 'var(--primary-blue)' : '#CBD5E1'}`,
              backgroundColor: dragActive ? '#EFF6FF' : '#F8FAFC'
            }}
          >
            <Upload className="w-12 h-12 mb-4" style={{ color: '#94A3B8' }} />
            <p className="text-sm mb-2 text-center" style={{ color: 'var(--text-secondary)' }}>
              Drag and drop your files here, or click to browse
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
              className="mt-4 px-6 py-3 rounded-md text-sm cursor-pointer transition-colors"
              style={{
                backgroundColor: '#64748B',
                color: 'white',
                fontWeight: 'var(--font-weight-medium)'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = '#475569';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = '#64748B';
              }}
            >
              Choose File
            </label>
            <p className="text-xs mt-3" style={{ color: 'var(--text-muted)' }}>
              Supported formats: CSV (Max 100MB)
            </p>
            {selectedFile && (
              <p className="text-sm mt-3" style={{ 
                color: 'var(--success-green)',
                fontWeight: 'var(--font-weight-medium)'
              }}>
                ✓ Selected: {selectedFile.name}
              </p>
            )}
          </div>

          {/* Upload Button */}
          <button
            className="w-full py-3 rounded-md text-sm transition-colors"
            style={{
              backgroundColor: 'var(--primary-blue)',
              color: 'white',
              fontWeight: 'var(--font-weight-medium)',
              height: '48px'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#1D4ED8';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'var(--primary-blue)';
            }}
          >
            Upload and Train
          </button>

          {/* Training Status */}
          <div className="flex items-center gap-2 mt-4 pt-4" style={{ borderTop: '1px solid var(--border-default)' }}>
            <CheckCircle className="w-5 h-5" style={{ color: 'var(--success-green)' }} />
            <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
              Training Status: <strong style={{ color: 'var(--success-green)' }}>completed</strong>
            </span>
          </div>
        </div>

        {/* Data Sources */}
        <div 
          className="rounded-lg p-6"
          style={{
            backgroundColor: 'var(--bg-secondary)',
            border: '1px solid var(--border-default)',
            boxShadow: '0 2px 8px rgba(0,0,0,0.06)'
          }}
        >
          <h3 className="mb-2" style={{ 
            fontSize: 'var(--text-lg)',
            fontWeight: 'var(--font-weight-semibold)',
            color: 'var(--text-primary)'
          }}>
            Data Sources
          </h3>
          <p className="text-sm mb-6" style={{ color: 'var(--text-secondary)' }}>
            Manage your connected data sources
          </p>

          <div className="space-y-0">
            {dataSources.map((source, index) => (
              <DataSourceItem 
                key={source.id} 
                {...source}
                isLast={index === dataSources.length - 1}
              />
            ))}

            {/* Add Data Source */}
            <button
              className="w-full py-4 text-left flex items-center gap-3 transition-colors rounded-md"
              style={{
                color: 'var(--primary-blue)',
                fontWeight: 'var(--font-weight-medium)'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = '#EFF6FF';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = 'transparent';
              }}
            >
              <div className="w-7 h-7 rounded-full flex items-center justify-center" style={{ backgroundColor: '#EFF6FF' }}>
                <span className="text-lg">+</span>
              </div>
              <span className="text-sm">Add Data Source</span>
            </button>
          </div>
        </div>
      </div>

      {/* Recent Uploads Table */}
      <div 
        className="rounded-lg p-6"
        style={{
          backgroundColor: 'var(--bg-secondary)',
          border: '1px solid var(--border-default)',
          boxShadow: '0 2px 8px rgba(0,0,0,0.06)'
        }}
      >
        <h3 className="mb-2" style={{ 
          fontSize: 'var(--text-lg)',
          fontWeight: 'var(--font-weight-semibold)',
          color: 'var(--text-primary)'
        }}>
          Recent Uploads
        </h3>
        <p className="text-sm mb-6" style={{ color: 'var(--text-secondary)' }}>
          Track your recent data uploads and processing status
        </p>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr style={{ borderBottom: '1px solid #F1F5F9' }}>
                <th className="text-left pb-3 pr-4" style={{ 
                  fontSize: '12px',
                  fontWeight: 'var(--font-weight-semibold)',
                  color: '#64748B',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}>
                  File Name
                </th>
                <th className="text-left pb-3 px-4" style={{ 
                  fontSize: '12px',
                  fontWeight: 'var(--font-weight-semibold)',
                  color: '#64748B',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}>
                  Type
                </th>
                <th className="text-left pb-3 px-4" style={{ 
                  fontSize: '12px',
                  fontWeight: 'var(--font-weight-semibold)',
                  color: '#64748B',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}>
                  Size
                </th>
                <th className="text-left pb-3 px-4" style={{ 
                  fontSize: '12px',
                  fontWeight: 'var(--font-weight-semibold)',
                  color: '#64748B',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}>
                  Upload Date
                </th>
                <th className="text-left pb-3 pl-4" style={{ 
                  fontSize: '12px',
                  fontWeight: 'var(--font-weight-semibold)',
                  color: '#64748B',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}>
                  Status
                </th>
              </tr>
            </thead>
            <tbody>
              {uploadHistory.map((record, index) => (
                <TableRow key={index} {...record} />
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="flex items-center justify-end gap-2 mt-4">
          {[1, 2, 3, '...', 5].map((page, index) => (
            <button
              key={index}
              className="w-8 h-8 rounded text-sm transition-colors"
              style={{
                backgroundColor: page === 1 ? 'var(--primary-blue)' : 'transparent',
                color: page === 1 ? 'white' : 'var(--text-secondary)',
                border: page === 1 ? 'none' : '1px solid var(--border-default)'
              }}
              onMouseEnter={(e) => {
                if (page !== 1 && page !== '...') {
                  e.currentTarget.style.backgroundColor = '#F8FAFC';
                }
              }}
              onMouseLeave={(e) => {
                if (page !== 1) {
                  e.currentTarget.style.backgroundColor = 'transparent';
                }
              }}
            >
              {page}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function DataSourceItem({ icon, title, status, lastSync, isLast }: DataSource & { isLast: boolean }) {
  const statusConfig = {
    connected: { bg: '#DCFCE7', text: '#059669', label: 'Connected' },
    disconnected: { bg: '#F1F5F9', text: '#64748B', label: 'Disconnected' }
  };

  const config = statusConfig[status];

  return (
    <div 
      className="flex items-center justify-between py-4 transition-colors"
      style={{
        height: '72px',
        borderBottom: isLast ? 'none' : '1px solid #E2E8F0'
      }}
    >
      <div className="flex items-center gap-3">
        <div style={{ color: 'var(--text-secondary)' }}>
          {icon}
        </div>
        <div>
          <p className="text-base mb-0.5" style={{ 
            fontWeight: 'var(--font-weight-semibold)',
            color: 'var(--text-primary)'
          }}>
            {title}
          </p>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {lastSync ? `Last sync: ${lastSync}` : 'Not configured'}
          </p>
        </div>
      </div>
      <div 
        className="px-3 py-1 rounded-full text-xs"
        style={{
          backgroundColor: config.bg,
          color: config.text,
          fontWeight: 'var(--font-weight-semibold)'
        }}
      >
        {config.label}
      </div>
    </div>
  );
}

function TableRow({ fileName, type, size, uploadDate, status }: UploadRecord) {
  const statusConfig = {
    processed: { bg: '#DCFCE7', text: '#059669', label: 'Processed' },
    processing: { bg: '#FEF3C7', text: '#F59E0B', label: 'Processing' },
    failed: { bg: '#FEE2E2', text: '#DC2626', label: 'Failed' }
  };

  const config = statusConfig[status];

  return (
    <tr 
      className="transition-colors"
      style={{
        height: '56px',
        borderBottom: '1px solid #F1F5F9'
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.backgroundColor = '#F8FAFC';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.backgroundColor = 'transparent';
      }}
    >
      <td className="pr-4 text-sm" style={{ color: 'var(--text-primary)' }}>
        {fileName}
      </td>
      <td className="px-4 text-sm" style={{ color: 'var(--text-secondary)' }}>
        {type}
      </td>
      <td className="px-4 text-sm" style={{ color: 'var(--text-secondary)' }}>
        {size}
      </td>
      <td className="px-4 text-sm" style={{ color: 'var(--text-secondary)' }}>
        {uploadDate}
      </td>
      <td className="pl-4">
        <div 
          className="inline-block px-3 py-1 rounded-xl text-xs"
          style={{
            backgroundColor: config.bg,
            color: config.text,
            fontWeight: 'var(--font-weight-semibold)'
          }}
        >
          {config.label}
        </div>
      </td>
    </tr>
  );
}
