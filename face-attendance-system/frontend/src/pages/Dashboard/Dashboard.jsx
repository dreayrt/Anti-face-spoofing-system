import React, { useState, useEffect } from 'react';
import { apiService } from '../../services/api';

const Dashboard = () => {
  const [stats, setStats] = useState({ present: 0, absent: 0, total: 0 });
  const [logs, setLogs] = useState([]);

  useEffect(() => {
    // Generate some mock dummy logs for visual showcase
    setStats({ present: 142, absent: 8, total: 150 });
    
    setLogs([
      { id: 1, name: "Alice Developer", empId: "emp_101", time: "08:15 AM", status: "success", score: 0.98 },
      { id: 2, name: "Bob Designer", empId: "emp_102", time: "08:22 AM", status: "success", score: 0.95 },
      { id: 3, name: "Unknown User", empId: "---", time: "08:45 AM", status: "failed", score: 0.21 },
      { id: 4, name: "Charlie Manager", empId: "emp_103", time: "08:50 AM", status: "success", score: 0.99 },
    ]);
  }, []);

  return (
    <div className="w-full space-y-8 animate-fade-in-up">
      <div className="flex items-end justify-between mb-8">
        <div>
          <h2 className="text-3xl font-black text-slate-800">System Overview</h2>
          <p className="text-slate-500 mt-1">Real-time attendance metrics and security logs</p>
        </div>
        <div className="text-right hidden sm:block">
          <p className="text-sm text-slate-400 font-mono tracking-widest uppercase">System Status</p>
          <div className="flex items-center space-x-2 mt-1">
             <div className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
            </div>
            <span className="text-emerald-400 font-semibold text-sm">ONLINE</span>
          </div>
        </div>
      </div>
      
      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Total Card */}
        <div className="group relative overflow-hidden rounded-3xl bg-white border border-slate-200 shadow-sm hover:shadow-md transition-all duration-300">
          <div className="relative h-full p-6 flex flex-col justify-between">
            <div className="flex justify-between items-start mb-4">
              <h3 className="text-slate-500 text-xs font-bold tracking-widest uppercase">Total Employees</h3>
              <div className="p-2 bg-slate-50 rounded-lg border border-slate-100 text-slate-400 group-hover:text-blue-500 transition-colors">
                 <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"></path></svg>
              </div>
            </div>
            <div>
              <p className="text-5xl font-black text-slate-800 tracking-tight">{stats.total}</p>
              <p className="text-slate-500 text-sm mt-2 flex items-center">
                 <svg className="w-4 h-4 mr-1 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path></svg>
                 <span className="text-emerald-600 font-medium">+2%</span> vs last month
              </p>
            </div>
          </div>
        </div>

        {/* Present Card */}
        <div className="group relative overflow-hidden rounded-3xl bg-white border border-emerald-100 shadow-sm hover:shadow-emerald-100 transition-all duration-300">
          <div className="absolute inset-0 bg-gradient-to-br from-emerald-50/50 to-transparent"></div>
          <div className="relative h-full p-6 flex flex-col justify-between">
            <div className="flex justify-between items-start mb-4">
              <h3 className="text-emerald-600/80 text-xs font-bold tracking-widest uppercase">Present Today</h3>
              <div className="p-2 bg-emerald-50 rounded-lg border border-emerald-100 text-emerald-500 group-hover:bg-emerald-100/50 transition-colors">
                 <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
              </div>
            </div>
            <div>
              <p className="text-5xl font-black text-emerald-600 tracking-tight">{stats.present}</p>
              <div className="w-full bg-slate-100 rounded-full h-1.5 mt-4 overflow-hidden border border-slate-200">
                 <div className="bg-emerald-500 h-1.5 rounded-full" style={{width: `${(stats.present/stats.total)*100}%`}}></div>
              </div>
            </div>
          </div>
        </div>

        {/* Absent Card */}
        <div className="group relative overflow-hidden rounded-3xl bg-white border border-rose-100 shadow-sm hover:shadow-rose-100 transition-all duration-300">
          <div className="absolute inset-0 bg-gradient-to-br from-rose-50/50 to-transparent"></div>
          <div className="relative h-full p-6 flex flex-col justify-between">
            <div className="flex justify-between items-start mb-4">
              <h3 className="text-rose-600/80 text-xs font-bold tracking-widest uppercase">Absent</h3>
              <div className="p-2 bg-rose-50 rounded-lg border border-rose-100 text-rose-500 group-hover:bg-rose-100/50 transition-colors">
                 <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
              </div>
            </div>
            <div>
              <p className="text-5xl font-black text-rose-600 tracking-tight">{stats.absent}</p>
              <div className="w-full bg-slate-100 rounded-full h-1.5 mt-4 overflow-hidden border border-slate-200">
                 <div className="bg-rose-500 h-1.5 rounded-full" style={{width: `${(stats.absent/stats.total)*100}%`}}></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Logs Table */}
      <div className="relative overflow-hidden rounded-3xl bg-white border border-slate-200 shadow-sm">
        <div className="relative h-full overflow-hidden">
          <div className="p-6 border-b border-slate-100 flex justify-between items-center bg-slate-50">
            <h3 className="text-lg font-bold text-slate-800 flex items-center">
              <svg className="w-5 h-5 mr-2 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
              Live Recognition Logs
            </h3>
            <button className="text-xs text-indigo-600 hover:text-indigo-700 font-semibold uppercase tracking-wider transition-colors">View All History &rarr;</button>
          </div>
          
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-white text-slate-400 text-xs uppercase tracking-widest border-b border-slate-100">
                  <th className="py-4 px-6 font-medium">Employee</th>
                  <th className="py-4 px-6 font-medium">Emp ID</th>
                  <th className="py-4 px-6 font-medium">Time (Today)</th>
                  <th className="py-4 px-6 font-medium">Liveness Score</th>
                  <th className="py-4 px-6 font-medium text-right">Status</th>
                </tr>
              </thead>
              <tbody className="text-sm">
                {logs.map((log) => (
                  <tr key={log.id} className="border-b border-slate-50 hover:bg-slate-50 transition-colors group">
                    <td className="py-4 px-6 font-medium text-slate-800 flex items-center">
                      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-100 to-blue-100 flex items-center justify-center text-indigo-700 font-bold text-xs mr-3 shadow-sm group-hover:scale-110 transition-transform">
                        {log.name.charAt(0)}
                      </div>
                      {log.name}
                    </td>
                    <td className="py-4 px-6 text-slate-500 font-mono text-xs">{log.empId}</td>
                    <td className="py-4 px-6 text-slate-600">{log.time}</td>
                    <td className="py-4 px-6">
                      <div className="flex items-center">
                         <div className="w-16 bg-slate-100 rounded-full h-1.5 mr-2 overflow-hidden border border-slate-200">
                            <div className={`h-full rounded-full ${log.score > 0.8 ? 'bg-emerald-500' : 'bg-rose-500'}`} style={{width: `${log.score * 100}%`}}></div>
                         </div>
                         <span className={`text-xs font-mono font-medium ${log.score > 0.8 ? 'text-emerald-600' : 'text-rose-600'}`}>
                           {(log.score * 100).toFixed(0)}%
                         </span>
                      </div>
                    </td>
                    <td className="py-4 px-6 text-right">
                      <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold capitalize border ${
                        log.status === 'success' 
                          ? 'bg-emerald-50 text-emerald-600 border-emerald-200' 
                          : 'bg-rose-50 text-rose-600 border-rose-200'
                     }`}>
                        {log.status === 'success' ? (
                          <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path></svg>
                        ) : (
                          <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
                        )}
                        {log.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
