import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  Play, LogOut, User, Clock, Flame, Trophy
} from 'lucide-react'
import { useAuthStore } from '../../store/authStore'

const UserDashboard = () => {
  const { user, logout } = useAuthStore()

  const exercises = [
    { name: 'Push-ups', icon: '💪', color: 'from-primary-500 to-primary-600' },
    { name: 'Squats', icon: '🦵', color: 'from-accent-500 to-accent-600' },
    { name: 'Bicep Curls', icon: '💪', color: 'from-warning-500 to-warning-600' },
    { name: 'Shoulder Press', icon: '🏋️', color: 'from-purple-500 to-purple-600' },
    { name: 'Hammer Curls', icon: '🔨', color: 'from-primary-400 to-accent-400' },
  ]

  return (
    <div className="min-h-screen bg-dark-900">
      {/* Header */}
      <div className="bg-dark-800/50 backdrop-blur-sm border-b border-dark-700 sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <img src="/assets/logo.svg" alt="AlphaReps" className="w-10 h-10 rounded-xl" />
              <div>
                <h1 className="text-xl font-black text-white">ALPHAREPS</h1>
                <p className="text-xs text-gray-400">AI Training Assistant</p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <Link to="/user/profile">
                <button className="flex items-center gap-2 bg-dark-700 hover:bg-dark-600 px-4 py-2 rounded-lg transition-colors">
                  <User className="w-4 h-4 text-gray-400" />
                  <span className="text-sm text-gray-300">{user?.name}</span>
                </button>
              </Link>
              <button
                onClick={logout}
                className="p-2 text-gray-400 hover:text-white hover:bg-dark-700 rounded-lg transition-colors"
              >
                <LogOut className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 py-8">
        {/* Welcome Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h2 className="text-3xl font-black text-white mb-2">
            Welcome back, <span className="text-primary-400">{user?.name?.split(' ')[0]}</span>
          </h2>
          <p className="text-gray-400">Ready to crush your workout today?</p>
        </motion.div>

        {/* Main CTA - Start Workout */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mb-8"
        >
          <Link to="/user/workout">
            <div className="relative overflow-hidden bg-gradient-to-br from-dark-800 to-dark-900 rounded-2xl p-8 border border-primary-500/30 hover:border-primary-500/60 transition-all group cursor-pointer">
              {/* Background Glow */}
              <div className="absolute inset-0 bg-gradient-to-r from-primary-500/10 via-transparent to-accent-500/10 opacity-0 group-hover:opacity-100 transition-opacity" />

              <div className="relative flex items-center justify-between">
                <div>
                  <h3 className="text-2xl font-black text-white mb-2">Start Training Session</h3>
                  <p className="text-gray-400 mb-6">AI-powered form correction • Real-time rep counting</p>

                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className="bg-accent-500 hover:bg-accent-600 text-white font-bold py-4 px-8 rounded-xl flex items-center gap-3 transition-colors"
                  >
                    <Play className="w-6 h-6" />
                    <span className="text-lg">Begin Workout</span>
                  </motion.button>
                </div>

                <div className="hidden md:block">
                  <img src="/assets/logo.svg" alt="AlphaReps" className="w-32 h-32 opacity-50" />
                </div>
              </div>
            </div>
          </Link>
        </motion.div>

        {/* Supported Exercises */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mb-8"
        >
          <h3 className="text-lg font-bold text-white mb-4">Supported Exercises</h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {exercises.map((exercise, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.3 + i * 0.05 }}
                className="bg-dark-800 rounded-xl p-4 border border-dark-700 hover:border-primary-500/30 transition-colors text-center"
              >
                <div className="text-2xl mb-2">{exercise.icon}</div>
                <p className="text-sm font-medium text-gray-300">{exercise.name}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Features Grid */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <h3 className="text-lg font-bold text-white mb-4">How It Works</h3>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-dark-800 rounded-xl p-5 border border-dark-700">
              <div className="w-10 h-10 bg-primary-500/20 rounded-lg flex items-center justify-center mb-3">
                <img src="/assets/logo.svg" alt="" className="w-5 h-5" />
              </div>
              <h4 className="font-bold text-white mb-1">AI Detection</h4>
              <p className="text-sm text-gray-400">Automatically identifies your exercise using advanced pose detection</p>
            </div>

            <div className="bg-dark-800 rounded-xl p-5 border border-dark-700">
              <div className="w-10 h-10 bg-accent-500/20 rounded-lg flex items-center justify-center mb-3">
                <Trophy className="w-5 h-5 text-accent-400" />
              </div>
              <h4 className="font-bold text-white mb-1">Rep Counting</h4>
              <p className="text-sm text-gray-400">Smart rep counting that only counts reps with proper form</p>
            </div>

            <div className="bg-dark-800 rounded-xl p-5 border border-dark-700">
              <div className="w-10 h-10 bg-warning-500/20 rounded-lg flex items-center justify-center mb-3">
                <Flame className="w-5 h-5 text-warning-400" />
              </div>
              <h4 className="font-bold text-white mb-1">Form Feedback</h4>
              <p className="text-sm text-gray-400">Real-time corrections to help you maintain perfect form</p>
            </div>
          </div>
        </motion.div>

        {/* Quick Stats (simplified) */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="mt-8 bg-gradient-to-r from-primary-500/10 to-accent-500/10 rounded-xl p-6 border border-primary-500/20"
        >
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center gap-3">
              <Clock className="w-5 h-5 text-primary-400" />
              <div>
                <p className="text-xs text-gray-400">Powered by</p>
                <p className="font-bold text-white">MediaPipe + BiLSTM AI</p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-xs text-gray-400">Real-time Processing</p>
              <p className="font-bold text-accent-400">~150ms latency</p>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}

export default UserDashboard
