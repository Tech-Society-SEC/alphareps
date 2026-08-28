import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { useAuthStore } from './store/authStore'

// Pages
import Landing from './pages/Landing'
import Login from './pages/Login'
import AdminDashboard from './pages/admin/AdminDashboard'
import UserDashboard from './pages/user/UserDashboard'
import WorkoutSession from './pages/user/WorkoutSession'
import Profile from './pages/user/Profile'
import Analytics from './pages/user/Analytics'

// Protected Route Component
const ProtectedRoute = ({ children, requireAdmin = false }) => {
  const { user, isAuthenticated } = useAuthStore()

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />
  }

  if (requireAdmin && user?.role !== 'admin') {
    return <Navigate to="/user/dashboard" replace />
  }

  return children
}

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-dark-900">
        <Routes>
          {/* Public Routes */}
          <Route path="/" element={<Landing />} />
          <Route path="/login" element={<Login />} />

          {/* Admin Routes */}
          <Route
            path="/admin/*"
            element={
              <ProtectedRoute requireAdmin>
                <AdminDashboard />
              </ProtectedRoute>
            }
          />

          {/* User Routes */}
          <Route
            path="/user/dashboard"
            element={
              <ProtectedRoute>
                <UserDashboard />
              </ProtectedRoute>
            }
          />
          <Route
            path="/user/workout"
            element={
              <ProtectedRoute>
                <WorkoutSession />
              </ProtectedRoute>
            }
          />
          <Route
            path="/user/profile"
            element={
              <ProtectedRoute>
                <Profile />
              </ProtectedRoute>
            }
          />
          <Route
            path="/user/analytics"
            element={
              <ProtectedRoute>
                <Analytics />
              </ProtectedRoute>
            }
          />

          {/* Fallback */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </Router>
  )
}

export default App
