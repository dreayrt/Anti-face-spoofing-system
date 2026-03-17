import React from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Link,
  useLocation,
} from "react-router-dom";
import Dashboard from "./pages/Dashboard/Dashboard";
import CameraPage from "./pages/Camera/CameraPage";

const NavLink = ({ to, children }) => {
  const location = useLocation();
  const isActive = location.pathname === to;

  return (
    <Link
      to={to}
      className={`relative px-4 py-2 rounded-xl text-sm font-semibold transition-all duration-300 ease-in-out ${
        isActive
          ? "text-slate-800 bg-white shadow-sm ring-1 ring-slate-200"
          : "text-slate-500 hover:text-slate-800 hover:bg-slate-200/50"
      }`}
    >
      {children}
    </Link>
  );
};

function AppContent() {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 flex flex-col font-sans selection:bg-blue-200 selection:text-blue-900">
      {/* Dynamic Animated Background Orbs - Elegant Bright Mode */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden -z-10 pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-96 h-96 bg-blue-200 rounded-full mix-blend-multiply filter blur-[100px] opacity-70 animate-blob"></div>
        <div className="absolute top-[20%] right-[-10%] w-96 h-96 bg-amber-100 rounded-full mix-blend-multiply filter blur-[100px] opacity-70 animate-blob animation-delay-2000"></div>
        <div className="absolute bottom-[-10%] left-[20%] w-96 h-96 bg-emerald-100 rounded-full mix-blend-multiply filter blur-[100px] opacity-70 animate-blob animation-delay-4000"></div>
      </div>

      {/* Glassmorphic Navigation Bar */}
      <nav className="sticky top-0 z-50 backdrop-blur-xl bg-white/70 border-b border-slate-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 h-20 flex justify-between items-center">
          <div className="flex items-center space-x-3 group cursor-pointer">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg group-hover:scale-105 transition-transform duration-300">
              <svg
                className="w-6 h-6 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M5.121 17.804A13.937 13.937 0 0112 16c2.5 0 4.847.655 6.879 1.804M15 10a3 3 0 11-6 0 3 3 0 016 0zm6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                ></path>
              </svg>
            </div>
            <h1 className="text-2xl font-black tracking-tight text-slate-800">
              FaceDetect<span className="text-blue-600">.AI</span>
            </h1>
          </div>
          <div className="space-x-2 flex p-1 bg-slate-100 rounded-2xl border border-slate-200">
            <NavLink to="/">Dashboard</NavLink>
            <NavLink to="/camera">Live </NavLink>
          </div>
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="flex-grow p-6 flex flex-col items-center justify-center relative z-10 w-full">
        <div className="max-w-7xl mx-auto w-full animate-fade-in-up">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/camera" element={<CameraPage />} />
          </Routes>
        </div>
      </main>

      {/* Footer */}
      <footer className="py-6 text-center text-slate-400 text-sm backdrop-blur-md bg-white/50 border-t border-slate-200/60 mt-auto">
        &copy; 2026 FaceDetect.AI System.
      </footer>
    </div>
  );
}

function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}

export default App;
