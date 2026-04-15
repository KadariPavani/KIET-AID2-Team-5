import React, { useState, useEffect } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, ArcElement, Title, Tooltip, Legend, Filler } from 'chart.js';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
import { useAuth } from '../context/AuthContext';
import { useConfig } from '../context/ConfigContext';
import { formatTime } from '../utils/helpers';
import '../styles/analytics.css';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, ArcElement, Title, Tooltip, Legend, Filler);

const Analytics = () => {
  const { getAuthHeaders } = useAuth();
  const { API_BASE_URL } = useConfig();
  const [dateRange, setDateRange] = useState('all');
  const [customDate, setCustomDate] = useState('');
  const [violations, setViolations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const rowsPerPage = 8;

  const [totalCount, setTotalCount] = useState(0);
  const [summary, setSummary] = useState({
    total: 0, speed: 0, red_light: 0, stop_line: 0, lane_change: 0, unsafe_distance: 0
  });

  useEffect(() => { fetchViolations(); }, [dateRange, customDate]);

  const fetchViolations = async () => {
    setLoading(true);
    let query = '?limit=500';
    if (dateRange === 'custom' && customDate) {
      query += `&specific_date=${encodeURIComponent(customDate)}`;
    } else if (dateRange !== 'all') {
      query += `&date_range=${encodeURIComponent(dateRange)}`;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/api/db/violations${query}`, {
        headers: getAuthHeaders()
      });
      if (response.ok) {
        const data = await response.json();
        const list = data.violations || [];
        const actualTotal = data.total ?? list.length;
        setViolations(list);
        setTotalCount(actualTotal);
        setCurrentPage(1);

        const counts = { total: actualTotal, speed: 0, red_light: 0, stop_line: 0, lane_change: 0, unsafe_distance: 0 };
        list.forEach(v => { if (counts.hasOwnProperty(v.violation_type)) counts[v.violation_type]++; });
        setSummary(counts);
      }
    } catch (error) {
      console.error('Error fetching violations:', error);
    } finally {
      setLoading(false);
    }
  };

  // Pagination
  const totalPages = Math.ceil(violations.length / rowsPerPage);
  const startIndex = (currentPage - 1) * rowsPerPage;
  const endIndex = startIndex + rowsPerPage;
  const currentViolations = violations.slice(startIndex, endIndex);
  const handlePageChange = (page) => { if (page >= 1 && page <= totalPages) setCurrentPage(page); };

  // Helpers
  const pct = (n) => summary.total > 0 ? Math.round((n / summary.total) * 100) : 0;

  const violationTypes = [
    { key: 'red_light', label: 'Red Light', icon: 'fa-traffic-light', color: '#ef4444' },
    { key: 'stop_line', label: 'Stop Line', icon: 'fa-stop', color: '#f97316' },
    { key: 'lane_change', label: 'Lane Change', icon: 'fa-right-left', color: '#eab308' },
    { key: 'unsafe_distance', label: 'Unsafe Distance', icon: 'fa-car-burst', color: '#8b5cf6' },
  ];

  const badgeColors = {
    red_light: { bg: '#fef2f2', text: '#dc2626', border: '#fecaca' },
    stop_line: { bg: '#fff7ed', text: '#ea580c', border: '#fed7aa' },
    lane_change: { bg: '#fefce8', text: '#a16207', border: '#fef08a' },
    unsafe_distance: { bg: '#f5f3ff', text: '#7c3aed', border: '#ddd6fe' },
    speed: { bg: '#f9fafb', text: '#6b7280', border: '#e5e7eb' },
  };

  // ── Charts ──

  const chartTooltip = {
    backgroundColor: 'rgba(33,33,33,0.95)', titleFont: { size: 13, weight: '600' },
    bodyFont: { size: 12 }, padding: 12, cornerRadius: 8,
    borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1,
  };

  // Timeline — single smooth area line
  const timelineData = () => {
    const buckets = {};
    violations.forEach(v => {
      if (!v.timestamp) return;
      const d = new Date(v.timestamp);
      if (isNaN(d.getTime())) return;
      const key = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      buckets[key] = (buckets[key] || 0) + 1;
    });
    const labels = Object.keys(buckets).sort();
    return {
      labels,
      datasets: [{
        label: 'Violations',
        data: labels.map(k => buckets[k]),
        borderColor: '#ef4444',
        backgroundColor: 'rgba(239, 68, 68, 0.12)',
        fill: true,
        tension: 0.3,
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 0,
      }]
    };
  };

  const lineOptions = {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: chartTooltip,
    },
    scales: {
      y: { beginAtZero: true, ticks: { precision: 0, color: '#9e9e9e', font: { size: 11 } }, grid: { color: 'rgba(0,0,0,0.04)', drawBorder: false }, border: { display: false } },
      x: { ticks: { color: '#9e9e9e', font: { size: 10 }, maxRotation: 45, autoSkip: true, maxTicksLimit: 12 }, grid: { display: false }, border: { display: false } }
    },
    interaction: { intersect: false, mode: 'index' },
  };

  // Bar by type
  const barData = {
    labels: violationTypes.map(t => t.label),
    datasets: [{
      data: violationTypes.map(t => summary[t.key]),
      backgroundColor: violationTypes.map(t => t.color + 'cc'),
      borderColor: violationTypes.map(t => t.color),
      borderWidth: 2, borderRadius: 8, borderSkipped: false,
      barPercentage: 0.65, categoryPercentage: 0.7,
    }]
  };

  const barOptions = {
    responsive: true, maintainAspectRatio: false,
    indexAxis: 'y',
    plugins: { legend: { display: false }, tooltip: chartTooltip },
    scales: {
      x: { beginAtZero: true, ticks: { precision: 0, color: '#9e9e9e', font: { size: 11 } }, grid: { color: 'rgba(0,0,0,0.04)', drawBorder: false }, border: { display: false } },
      y: { ticks: { color: '#424242', font: { size: 12, weight: '600' } }, grid: { display: false }, border: { display: false } }
    }
  };

  // Doughnut
  const activeTypes = violationTypes.filter(t => summary[t.key] > 0);
  const doughnutData = {
    labels: activeTypes.length ? activeTypes.map(t => t.label) : ['No Data'],
    datasets: [{
      data: activeTypes.length ? activeTypes.map(t => summary[t.key]) : [1],
      backgroundColor: activeTypes.length ? activeTypes.map(t => t.color) : ['#e5e7eb'],
      borderWidth: 3, borderColor: '#ffffff', hoverOffset: 8,
    }]
  };

  const doughnutOptions = {
    responsive: true, maintainAspectRatio: false, cutout: '68%',
    plugins: {
      legend: { position: 'bottom', labels: { padding: 16, usePointStyle: true, pointStyleWidth: 8, font: { size: 12, weight: '500' } } },
      tooltip: chartTooltip,
    }
  };

  // Stream stacked bar — each camera broken down by violation type
  const streamData = () => {
    const cameras = {};
    violations.forEach(v => {
      const cam = `Camera ${v.stream_id + 1}`;
      if (!cameras[cam]) cameras[cam] = {};
      cameras[cam][v.violation_type] = (cameras[cam][v.violation_type] || 0) + 1;
    });
    const camLabels = Object.keys(cameras).sort();
    return {
      labels: camLabels,
      datasets: violationTypes.map(t => ({
        label: t.label,
        data: camLabels.map(c => cameras[c]?.[t.key] || 0),
        backgroundColor: t.color + 'cc',
        borderColor: t.color,
        borderWidth: 1, borderRadius: 4, borderSkipped: false,
      }))
    };
  };

  const streamBarOptions = {
    responsive: true, maintainAspectRatio: false,
    indexAxis: 'y',
    plugins: {
      legend: { position: 'top', align: 'end', labels: { boxWidth: 12, boxHeight: 12, padding: 14, usePointStyle: true, font: { size: 11, weight: '500' } } },
      tooltip: { ...chartTooltip, mode: 'index', intersect: false },
    },
    scales: {
      x: { stacked: true, beginAtZero: true, ticks: { precision: 0, color: '#9e9e9e', font: { size: 11 } }, grid: { color: 'rgba(0,0,0,0.04)', drawBorder: false }, border: { display: false } },
      y: { stacked: true, ticks: { color: '#424242', font: { size: 12, weight: '600' } }, grid: { display: false }, border: { display: false } }
    }
  };

  // Hourly heatmap-style chart — violations per hour of day
  const hourlyData = () => {
    const hours = Array(24).fill(0);
    const hourTypes = Array.from({ length: 24 }, () => ({}));
    violations.forEach(v => {
      if (!v.timestamp) return;
      const d = new Date(v.timestamp);
      if (isNaN(d.getTime())) return;
      const h = d.getHours();
      hours[h]++;
      hourTypes[h][v.violation_type] = (hourTypes[h][v.violation_type] || 0) + 1;
    });
    const labels = Array.from({ length: 24 }, (_, i) => {
      const ampm = i >= 12 ? 'PM' : 'AM';
      const hr = i % 12 || 12;
      return `${hr}${ampm}`;
    });
    return {
      labels,
      datasets: violationTypes.map(t => ({
        label: t.label,
        data: hourTypes.map(ht => ht[t.key] || 0),
        backgroundColor: t.color + 'cc',
        borderColor: t.color,
        borderWidth: 1, borderRadius: 3, borderSkipped: false,
      }))
    };
  };

  const hourlyOptions = {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { position: 'top', align: 'end', labels: { boxWidth: 12, boxHeight: 12, padding: 14, usePointStyle: true, font: { size: 11, weight: '500' } } },
      tooltip: { ...chartTooltip, mode: 'index', intersect: false },
    },
    scales: {
      x: { stacked: true, ticks: { color: '#9e9e9e', font: { size: 10 }, maxRotation: 0 }, grid: { display: false }, border: { display: false } },
      y: { stacked: true, beginAtZero: true, ticks: { precision: 0, color: '#9e9e9e', font: { size: 11 } }, grid: { color: 'rgba(0,0,0,0.04)', drawBorder: false }, border: { display: false } }
    }
  };

  // Page numbers
  const getPageNumbers = () => {
    if (totalPages <= 5) return Array.from({ length: totalPages }, (_, i) => i + 1);
    if (currentPage <= 3) return [1, 2, 3, 4, 5];
    if (currentPage >= totalPages - 2) return [totalPages - 4, totalPages - 3, totalPages - 2, totalPages - 1, totalPages];
    return [currentPage - 2, currentPage - 1, currentPage, currentPage + 1, currentPage + 2];
  };

  return (
    <div className="dashboard-body">
      <Navbar isDashboard={true} />

      <div className="a-page">
        {/* ── Header ── */}
        <div className="a-top">
          <div>
            <h1 className="a-heading">Traffic Analytics</h1>
            <p className="a-subtext">Violation insights and trends across all monitored streams</p>
          </div>
          <div className="a-filters">
            <div className="a-select-wrap">
              <i className="fas fa-calendar-alt"></i>
              <select value={dateRange} onChange={(e) => setDateRange(e.target.value)}>
                <option value="all">All Time</option>
                <option value="today">Today</option>
                <option value="yesterday">Yesterday</option>
                <option value="last_week">Last 7 Days</option>
                <option value="last_month">Last 30 Days</option>
                <option value="last_year">Last Year</option>
                <option value="custom">Pick Date</option>
              </select>
            </div>
            {dateRange === 'custom' && (
              <input type="date" className="a-date-pick" value={customDate} onChange={(e) => setCustomDate(e.target.value)} />
            )}
            <button className="a-refresh-btn" onClick={fetchViolations} title="Refresh">
              <i className={`fas fa-sync-alt ${loading ? 'fa-spin' : ''}`}></i>
            </button>
          </div>
        </div>

        {/* ── Metric Cards ── */}
        <div className="a-metrics">
          {/* Big total card */}
          <div className="a-metric-hero">
            <div className="a-metric-hero-icon">
              <i className="fas fa-shield-halved"></i>
            </div>
            <div className="a-metric-hero-data">
              <span className="a-metric-hero-num">{summary.total}</span>
              <span className="a-metric-hero-label">Total Violations</span>
            </div>
          </div>

          {/* Individual type cards */}
          {violationTypes.map(t => (
            <div className="a-metric-card" key={t.key}>
              <div className="a-metric-top">
                <div className="a-metric-icon-sm" style={{ background: t.color + '15', color: t.color }}>
                  <i className={`fas ${t.icon}`}></i>
                </div>
                <span className="a-metric-pct" style={{ color: t.color }}>{pct(summary[t.key])}%</span>
              </div>
              <span className="a-metric-num">{summary[t.key]}</span>
              <span className="a-metric-label">{t.label}</span>
            </div>
          ))}
        </div>

        {/* ── Charts Row 1 ── */}
        <div className="a-row">
          <div className="a-card a-card-grow">
            <div className="a-card-head">
              <h3><i className="fas fa-chart-area"></i> Violation Timeline</h3>
            </div>
            <div className="a-card-body a-chart-tall">
              <Line data={timelineData()} options={lineOptions} />
            </div>
          </div>
        </div>

        {/* ── Charts Row 2 ── */}
        <div className="a-row a-row-2col">
          <div className="a-card">
            <div className="a-card-head">
              <h3><i className="fas fa-chart-bar"></i> Violations by Type</h3>
            </div>
            <div className="a-card-body a-chart-mid">
              <Bar data={barData} options={barOptions} />
            </div>
          </div>
          <div className="a-card">
            <div className="a-card-head">
              <h3><i className="fas fa-chart-pie"></i> Distribution</h3>
            </div>
            <div className="a-card-body a-chart-mid a-donut-wrap">
              <Doughnut data={doughnutData} options={doughnutOptions} />
            </div>
          </div>
        </div>

        {/* ── Charts Row 3: Camera Breakdown ── */}
        <div className="a-row">
          <div className="a-card a-card-grow">
            <div className="a-card-head">
              <h3><i className="fas fa-video"></i> Violations by Camera</h3>
            </div>
            <div className="a-card-body a-chart-mid">
              <Bar data={streamData()} options={streamBarOptions} />
            </div>
          </div>
        </div>

        {/* ── Charts Row 4: Peak Hours ── */}
        <div className="a-row">
          <div className="a-card a-card-grow">
            <div className="a-card-head">
              <h3><i className="fas fa-clock"></i> Peak Violation Hours</h3>
            </div>
            <div className="a-card-body a-chart-mid">
              <Bar data={hourlyData()} options={hourlyOptions} />
            </div>
          </div>
        </div>

        {/* ── Table ── */}
        <div className="a-card">
          <div className="a-card-head a-card-head-between">
            <h3><i className="fas fa-table-list"></i> Incident Log</h3>
            <span className="a-record-count">{totalCount} total</span>
          </div>
          <div className="a-table-scroll">
            <table className="a-table a-table-simple">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Stream</th>
                  <th>Violation Type</th>
                  <th>Signal State</th>
                  <th>Vehicle</th>
                  <th>Time</th>
                </tr>
              </thead>
              <tbody>
                {currentViolations.length === 0 ? (
                  <tr>
                    <td colSpan="6" className="a-table-empty">
                      <div>
                        <i className="fas fa-inbox"></i>
                        <p>{loading ? 'Loading...' : 'No violations found'}</p>
                      </div>
                    </td>
                  </tr>
                ) : (
                  currentViolations.map((v, i) => (
                    <tr key={i}>
                      <td>{startIndex + i + 1}</td>
                      <td>Stream {v.stream_id + 1}</td>
                      <td>{(v.violation_type || '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</td>
                      <td>{v.signal_state || 'N/A'}</td>
                      <td>{v.vehicle_class || 'N/A'}</td>
                      <td>{formatTime(v.timestamp)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
          {totalPages > 1 && (
            <div className="a-pager">
              <span className="a-pager-info">
                Showing {startIndex + 1}–{Math.min(endIndex, violations.length)} of {totalCount}
              </span>
              <div className="a-pager-btns">
                <button onClick={() => handlePageChange(currentPage - 1)} disabled={currentPage === 1} className="a-pg-btn">
                  <i className="fas fa-chevron-left"></i>
                </button>
                {getPageNumbers().map(p => (
                  <button key={p} className={`a-pg-btn ${p === currentPage ? 'a-pg-active' : ''}`} onClick={() => handlePageChange(p)}>
                    {p}
                  </button>
                ))}
                <button onClick={() => handlePageChange(currentPage + 1)} disabled={currentPage === totalPages} className="a-pg-btn">
                  <i className="fas fa-chevron-right"></i>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      <Footer />
    </div>
  );
};

export default Analytics;
