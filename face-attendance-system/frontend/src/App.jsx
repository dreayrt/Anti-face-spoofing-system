import { useMemo, useState } from 'react';
import CameraPage from './pages/Camera/CameraPage';
import RegisterEmployee from './pages/Register/RegisterEmployee';

export default function App() {
  const [activePage, setActivePage] = useState('login');

  const pageTitle = useMemo(() => {
    return activePage === 'login' ? 'Đăng nhập / Chấm công' : 'Đăng ký nhân viên';
  }, [activePage]);

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      <header className="border-b border-white/10 bg-slate-900/60 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-4 py-4 sm:px-6 lg:px-8">
          <div>
            <p className="text-xs uppercase tracking-[0.25em] text-slate-400">Face Attendance System</p>
            <h1 className="mt-1 text-xl font-semibold text-white">{pageTitle}</h1>
          </div>

          <nav className="inline-flex rounded-2xl border border-white/10 bg-slate-900/70 p-1">
            <button
              type="button"
              onClick={() => setActivePage('login')}
              className={`rounded-xl px-4 py-2 text-sm font-medium transition ${
                activePage === 'login'
                  ? 'bg-blue-600 text-white shadow-md shadow-blue-900/30'
                  : 'text-slate-300 hover:bg-white/5'
              }`}
            >
              Login
            </button>
            <button
              type="button"
              onClick={() => setActivePage('register')}
              className={`rounded-xl px-4 py-2 text-sm font-medium transition ${
                activePage === 'register'
                  ? 'bg-violet-600 text-white shadow-md shadow-violet-900/30'
                  : 'text-slate-300 hover:bg-white/5'
              }`}
            >
              Register
            </button>
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        {activePage === 'login' ? <CameraPage /> : <RegisterEmployee />}
      </main>
    </div>
  );
}
