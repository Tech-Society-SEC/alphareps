import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowLeft, User, Mail, Calendar } from 'lucide-react'
import { useAuthStore } from '../../store/authStore'

const Profile = () => {
  const { user } = useAuthStore()

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
            <h1 className="text-xl font-bold text-white">Profile</h1>
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
          {/* Profile Card */}
          <div className="bg-dark-800 rounded-2xl p-8 border border-dark-700">
            <div className="flex items-center gap-6">
              <div className="w-20 h-20 bg-gradient-to-br from-primary-500 to-accent-500 rounded-2xl flex items-center justify-center">
                <User className="w-10 h-10 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-black text-white">{user?.name || 'User'}</h2>
                <p className="text-gray-400 capitalize">{user?.role || 'Member'}</p>
              </div>
            </div>
          </div>

          {/* Details */}
          <div className="bg-dark-800 rounded-xl p-6 border border-dark-700 space-y-4">
            <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-4">Account Details</h3>

            <div className="flex items-center gap-4 p-4 bg-dark-700/50 rounded-lg">
              <Mail className="w-5 h-5 text-primary-400" />
              <div>
                <p className="text-xs text-gray-500">Email</p>
                <p className="text-white">{user?.email || 'user@alphareps.com'}</p>
              </div>
            </div>

            <div className="flex items-center gap-4 p-4 bg-dark-700/50 rounded-lg">
              <Calendar className="w-5 h-5 text-accent-400" />
              <div>
                <p className="text-xs text-gray-500">Member Since</p>
                <p className="text-white">{user?.joinDate || 'Dec 2024'}</p>
              </div>
            </div>

            <div className="flex items-center gap-4 p-4 bg-dark-700/50 rounded-lg">
              <img src="/assets/logo.svg" alt="" className="w-5 h-5" />
              <div>
                <p className="text-xs text-gray-500">Account Type</p>
                <p className="text-white capitalize">{user?.role || 'User'}</p>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-4">
            <Link to="/user/workout" className="flex-1">
              <button className="w-full bg-accent-500 hover:bg-accent-600 text-white font-bold py-4 rounded-xl transition-colors">
                Start Workout
              </button>
            </Link>
          </div>
        </motion.div>
      </div>
    </div>
  )
}

export default Profile
