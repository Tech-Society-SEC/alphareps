import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { Zap, Brain, Activity, Target, ChevronRight } from 'lucide-react'
import Navbar from '../components/Navbar'

const Landing = () => {
  const exercises = [
    'Push-ups', 'Squats', 'Bicep Curls', 'Hammer Curls', 'Shoulder Press'
  ]

  return (
    <div className="min-h-screen bg-dark-900 overflow-hidden">
      <Navbar />

      {/* Subtle Background */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute w-[800px] h-[800px] bg-primary-500/5 rounded-full blur-3xl -top-40 -left-40" />
        <div className="absolute w-[600px] h-[600px] bg-accent-500/5 rounded-full blur-3xl bottom-0 right-0" />
      </div>

      {/* Hero Section */}
      <div className="relative z-10 min-h-[90vh] flex items-center justify-center px-4">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2 }}
              className="inline-flex items-center gap-2 bg-primary-500/10 border border-primary-500/20 rounded-full px-4 py-2 mb-8"
            >
              <Zap className="w-4 h-4 text-primary-400" />
              <span className="text-sm text-gray-300">AI-Powered Fitness Training</span>
            </motion.div>

            {/* Main Title */}
            <h1 className="text-6xl md:text-8xl font-black mb-6 tracking-tight">
              <span className="bg-gradient-to-r from-primary-400 via-primary-500 to-accent-500 bg-clip-text text-transparent">
                ALPHA
              </span>
              <span className="text-white">REPS</span>
            </h1>

            <p className="text-xl md:text-2xl text-gray-400 mb-4 font-medium">
              Your personal AI gym trainer
            </p>

            <p className="text-gray-500 mb-10 max-w-xl mx-auto">
              Real-time exercise detection, automatic rep counting, and instant form correction — all powered by computer vision AI.
            </p>

            {/* CTA Button */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              <Link to="/login">
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="bg-accent-500 hover:bg-accent-600 text-white font-bold py-4 px-10 rounded-xl text-lg transition-colors inline-flex items-center gap-2"
                >
                  Get Started
                  <ChevronRight className="w-5 h-5" />
                </motion.button>
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </div>

      {/* Features Section */}
      <div className="relative z-10 py-20 px-4 bg-dark-800/50">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-black text-white mb-4">
              How It Works
            </h2>
            <p className="text-gray-400 max-w-xl mx-auto">
              Just position yourself in front of your camera and start exercising. Our AI handles the rest.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                icon: Brain,
                title: 'AI Detection',
                desc: 'Automatically identifies which exercise you\'re performing in real-time',
                color: 'primary'
              },
              {
                icon: Target,
                title: 'Rep Counting',
                desc: 'Accurately counts your reps using advanced pose estimation',
                color: 'accent'
              },
              {
                icon: Activity,
                title: 'Form Feedback',
                desc: 'Instant corrections to help you maintain proper technique',
                color: 'warning'
              }
            ].map((feature, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="bg-dark-800 rounded-xl p-6 border border-dark-700 hover:border-primary-500/30 transition-colors"
              >
                <div className={`w-12 h-12 bg-${feature.color}-500/20 rounded-xl flex items-center justify-center mb-4`}>
                  <feature.icon className={`w-6 h-6 text-${feature.color}-400`} />
                </div>
                <h3 className="text-xl font-bold text-white mb-2">{feature.title}</h3>
                <p className="text-gray-400 text-sm">{feature.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      {/* Exercises Section */}
      <div className="relative z-10 py-20 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-3xl font-black text-white mb-8">
              5 Supported Exercises
            </h2>

            <div className="flex flex-wrap justify-center gap-3">
              {exercises.map((exercise, i) => (
                <motion.span
                  key={i}
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.05 }}
                  className="bg-dark-800 border border-dark-700 px-5 py-2.5 rounded-lg text-gray-300 font-medium"
                >
                  {exercise}
                </motion.span>
              ))}
            </div>
          </motion.div>
        </div>
      </div>

      {/* Final CTA */}
      <div className="relative z-10 py-20 px-4">
        <div className="max-w-2xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="bg-gradient-to-br from-primary-500/10 to-accent-500/10 rounded-2xl p-10 border border-primary-500/20"
          >
            <img src="/assets/logo.svg" alt="AlphaReps" className="w-16 h-16 mx-auto mb-4 rounded-xl" />
            <h2 className="text-2xl font-black text-white mb-3">
              Ready to train smarter?
            </h2>
            <p className="text-gray-400 mb-6">
              Start your AI-powered workout session now.
            </p>
            <Link to="/login">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="bg-accent-500 hover:bg-accent-600 text-white font-bold py-3 px-8 rounded-xl transition-colors"
              >
                Start Training
              </motion.button>
            </Link>
          </motion.div>
        </div>
      </div>

      {/* Footer */}
      <div className="relative z-10 py-8 px-4 border-t border-dark-800">
        <div className="max-w-5xl mx-auto flex items-center justify-between text-sm text-gray-500">
          <div className="flex items-center gap-2">
            <img src="/assets/logo.svg" alt="AlphaReps" className="w-5 h-5 rounded" />
            <span>AlphaReps</span>
          </div>
          <div>Powered by MediaPipe + TensorFlow</div>
        </div>
      </div>
    </div>
  )
}

export default Landing
