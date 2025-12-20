import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { User, Shield, Loader2 } from 'lucide-react'
import { useAuthStore } from '../store/authStore'

const Login = () => {
  const navigate = useNavigate()
  const { login } = useAuthStore()

  const [userName, setUserName] = useState('')
  const [userRole, setUserRole] = useState('user')
  const [isLoading, setIsLoading] = useState(false)

  const handleLogin = async (e) => {
    e.preventDefault()

    if (!userName.trim()) return

    setIsLoading(true)

    // Simulate login delay
    setTimeout(() => {
      const userData = {
        name: userName,
        role: userRole,
        email: `${userName.toLowerCase().replace(/\s+/g, '')}@alphareps.com`,
        joinDate: new Date().toLocaleDateString('en-US', { month: 'short', year: 'numeric' })
      }

      login(userData, null)
      navigate(userRole === 'admin' ? '/admin' : '/user/dashboard')
    }, 800)
  }

  return (
    <div className="min-h-screen bg-dark-900 flex items-center justify-center p-4">
      {/* Subtle Background */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute w-[600px] h-[600px] bg-primary-500/5 rounded-full blur-3xl top-0 left-1/2 -translate-x-1/2 -translate-y-1/2" />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative z-10 w-full max-w-md"
      >
        {/* Logo */}
        <div className="text-center mb-8">
          <img src="/assets/logo.svg" alt="AlphaReps" className="w-16 h-16 mx-auto mb-6 rounded-2xl" />

          <h1 className="text-4xl font-black mb-2">
            <span className="bg-gradient-to-r from-primary-400 to-accent-500 bg-clip-text text-transparent">ALPHA</span>
            <span className="text-white">REPS</span>
          </h1>
          <p className="text-gray-400">Sign in to continue</p>
        </div>

        {/* Login Card */}
        <div className="bg-dark-800 rounded-2xl p-8 border border-dark-700">
          <form onSubmit={handleLogin} className="space-y-6">
            {/* Name Input */}
            <div>
              <label className="block text-sm font-medium mb-2 text-gray-300">
                Your Name
              </label>
              <input
                type="text"
                value={userName}
                onChange={(e) => setUserName(e.target.value)}
                placeholder="Enter your name"
                className="w-full bg-dark-700 border border-dark-600 focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 rounded-xl px-4 py-3 text-white placeholder-gray-500 transition-all outline-none"
                disabled={isLoading}
                autoFocus
              />
            </div>

            {/* Role Selection */}
            <div>
              <label className="block text-sm font-medium mb-3 text-gray-300">
                Login As
              </label>
              <div className="grid grid-cols-2 gap-3">
                <button
                  type="button"
                  onClick={() => setUserRole('user')}
                  disabled={isLoading}
                  className={`p-4 rounded-xl border-2 transition-all ${userRole === 'user'
                      ? 'border-primary-500 bg-primary-500/10'
                      : 'border-dark-600 bg-dark-700 hover:border-dark-500'
                    }`}
                >
                  <User className={`w-6 h-6 mx-auto mb-2 ${userRole === 'user' ? 'text-primary-400' : 'text-gray-400'}`} />
                  <p className={`text-sm font-semibold ${userRole === 'user' ? 'text-white' : 'text-gray-400'}`}>User</p>
                </button>

                <button
                  type="button"
                  onClick={() => setUserRole('admin')}
                  disabled={isLoading}
                  className={`p-4 rounded-xl border-2 transition-all ${userRole === 'admin'
                      ? 'border-accent-500 bg-accent-500/10'
                      : 'border-dark-600 bg-dark-700 hover:border-dark-500'
                    }`}
                >
                  <Shield className={`w-6 h-6 mx-auto mb-2 ${userRole === 'admin' ? 'text-accent-400' : 'text-gray-400'}`} />
                  <p className={`text-sm font-semibold ${userRole === 'admin' ? 'text-white' : 'text-gray-400'}`}>Admin</p>
                </button>
              </div>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isLoading || !userName.trim()}
              className="w-full bg-accent-500 hover:bg-accent-600 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-bold py-4 rounded-xl transition-colors flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Signing in...
                </>
              ) : (
                'Continue'
              )}
            </button>
          </form>

          {/* Quick Login */}
          <div className="mt-6 pt-6 border-t border-dark-700">
            <p className="text-xs text-gray-500 text-center mb-3">Quick login:</p>
            <div className="flex gap-2 justify-center">
              {['John', 'Sarah', 'Alex'].map(name => (
                <button
                  key={name}
                  onClick={() => setUserName(name)}
                  disabled={isLoading}
                  className="text-xs px-4 py-2 bg-dark-700 hover:bg-dark-600 rounded-lg transition-colors text-gray-300"
                >
                  {name}
                </button>
              ))}
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  )
}

export default Login
