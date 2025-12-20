import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowLeft, Play } from 'lucide-react'

const Analytics = () => {
  return (
    <div className="min-h-screen bg-dark-900">
      {/* Header */}
      <div className="bg-dark-800/50 backdrop-blur-sm border-b border-dark-700">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <Link to="/user/dashboard" className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors">
              <ArrowLeft className="w-5 h-5" />
              <span>Back</span>
            </Link>
            <h1 className="text-xl font-bold text-white">Analytics</h1>
            <div className="w-16"></div>
          </div>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-4 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          {/* Coming Soon Card */}
          <div className="bg-dark-800 rounded-2xl p-12 border border-dark-700 text-center">
            <img src="/assets/logo.svg" alt="AlphaReps" className="w-20 h-20 mx-auto mb-6 rounded-2xl opacity-50" />

            <h2 className="text-2xl font-black text-white mb-3">
              Analytics Coming Soon
            </h2>
            <p className="text-gray-400 mb-8 max-w-md mx-auto">
              Track your workout history, progress, and performance metrics. This feature is under development.
            </p>

            <Link to="/user/workout">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="bg-accent-500 hover:bg-accent-600 text-white font-bold py-4 px-8 rounded-xl transition-colors inline-flex items-center gap-2"
              >
                <Play className="w-5 h-5" />
                Start Training Now
              </motion.button>
            </Link>
          </div>

          {/* Placeholder Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: 'Workouts', value: '—' },
              { label: 'Total Reps', value: '—' },
              { label: 'Best Streak', value: '—' },
              { label: 'This Week', value: '—' },
            ].map((stat, i) => (
              <div key={i} className="bg-dark-800 rounded-xl p-4 border border-dark-700 text-center">
                <p className="text-2xl font-black text-gray-600">{stat.value}</p>
                <p className="text-xs text-gray-500">{stat.label}</p>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  )
}

export default Analytics
