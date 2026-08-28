import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Webcam from 'react-webcam'
import axios from 'axios'
import {
  Play, Pause, RotateCcw, ArrowLeft, CheckCircle, AlertTriangle,
  Timer, Flame
} from 'lucide-react'
import { Link } from 'react-router-dom'

const WorkoutSession = () => {
  const webcamRef = useRef(null)
  const [isActive, setIsActive] = useState(false)
  const [exercise, setExercise] = useState('DETECTING...')
  const [reps, setReps] = useState(0)
  const [stage, setStage] = useState('ready')
  const [feedback, setFeedback] = useState('Position yourself in frame')
  const [formQuality, setFormQuality] = useState('GOOD')
  const [sessionStats, setSessionStats] = useState({
    totalReps: 0,
    goodFormReps: 0,
  })
  const [elapsedTime, setElapsedTime] = useState(0)

  const intervalRef = useRef(null)
  const timerRef = useRef(null)
  const isProcessingRef = useRef(false)
  const lastRepCountRef = useRef(0)

  useEffect(() => {
    if (isActive) {
      startWorkout()
      // Start timer
      timerRef.current = setInterval(() => {
        setElapsedTime(prev => prev + 1)
      }, 1000)
    } else {
      stopWorkout()
      if (timerRef.current) clearInterval(timerRef.current)
    }

    return () => {
      stopWorkout()
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [isActive])

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  const startWorkout = () => {
    intervalRef.current = setInterval(async () => {
      if (isProcessingRef.current) return
      if (webcamRef.current) {
        const imageSrc = webcamRef.current.getScreenshot()
        if (imageSrc) await sendFrame(imageSrc)
      }
    }, 100)  // Faster frame rate for more responsive counting
  }

  const stopWorkout = () => {
    if (intervalRef.current) clearInterval(intervalRef.current)
  }

  const sendFrame = async (imageData) => {
    if (isProcessingRef.current) return
    isProcessingRef.current = true

    try {
      const response = await axios.post('/api/workout/analyze', {
        frame: imageData
      }, { timeout: 3000 })

      const data = response.data
      if (data.exercise) setExercise(data.exercise.toUpperCase())
      if (data.reps !== undefined) setReps(data.reps)
      if (data.stage) setStage(data.stage)
      if (data.feedback) setFeedback(data.feedback)
      if (data.formQuality) setFormQuality(data.formQuality)

      if (typeof data.reps === 'number') {
        const currentReps = data.reps
        const prevReps = lastRepCountRef.current
        const delta = Math.max(0, currentReps - prevReps)
        lastRepCountRef.current = currentReps

        if (delta > 0) {
          setSessionStats(prev => ({
            ...prev,
            totalReps: prev.totalReps + delta,
            goodFormReps: data.formQuality === 'GOOD' ? prev.goodFormReps + delta : prev.goodFormReps
          }))
        }
      }
    } catch (error) {
      console.error('Error analyzing frame:', error)
    } finally {
      isProcessingRef.current = false
    }
  }

  const resetSession = async () => {
    setIsActive(false)
    setReps(0)
    setElapsedTime(0)
    setSessionStats({ totalReps: 0, goodFormReps: 0 })
    setExercise('DETECTING...')
    setFeedback('Position yourself in frame')
    lastRepCountRef.current = 0
    try {
      await axios.post('/api/workout/reset')
    } catch (err) {
      console.error('Error resetting:', err)
    }
  }

  const isGoodForm = formQuality === 'GOOD'
  const isPerfectForm = feedback.includes('PERFECT')

  return (
    <div className="min-h-screen bg-dark-900">
      {/* Minimal Header */}
      <div className="bg-dark-800/50 backdrop-blur-sm border-b border-dark-700 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <Link to="/user/dashboard" className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors">
              <ArrowLeft className="w-5 h-5" />
              <span className="hidden sm:inline">Exit</span>
            </Link>

            <div className="flex items-center gap-2">
              <img src="/assets/logo.svg" alt="AlphaReps" className="w-6 h-6 rounded" />
              <span className="font-bold text-white">ALPHAREPS</span>
            </div>

            <div className="flex items-center gap-4 text-sm">
              <div className="flex items-center gap-2 text-gray-400">
                <Timer className="w-4 h-4" />
                <span className="font-mono font-bold text-white">{formatTime(elapsedTime)}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid lg:grid-cols-4 gap-6">

          {/* Main Video Area - Takes up more space */}
          <div className="lg:col-span-3">
            <div className="relative rounded-2xl overflow-hidden bg-dark-800 border border-dark-700">
              {/* Video Feed */}
              <div className="relative aspect-video">
                <Webcam
                  ref={webcamRef}
                  audio={false}
                  screenshotFormat="image/jpeg"
                  className="w-full h-full object-cover"
                  videoConstraints={{
                    facingMode: 'user',
                    width: 1280,
                    height: 720,
                  }}
                />

                {/* Minimal Overlay - Only Essential Info */}
                <div className="absolute inset-0 pointer-events-none">

                  {/* Live Indicator */}
                  {isActive && (
                    <div className="absolute top-4 left-4 flex items-center gap-2 bg-red-500/90 backdrop-blur-sm px-3 py-1.5 rounded-full">
                      <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                      <span className="text-xs font-bold text-white">LIVE</span>
                    </div>
                  )}

                  {/* Exercise Badge - Top Right */}
                  <div className="absolute top-4 right-4">
                    <div className="bg-dark-900/80 backdrop-blur-sm rounded-lg px-4 py-2 border border-primary-500/30">
                      <p className="text-xs text-gray-400 uppercase tracking-wider">Exercise</p>
                      <p className="text-lg font-black text-white">{exercise}</p>
                    </div>
                  </div>

                  {/* Large Rep Counter - Center Bottom */}
                  <div className="absolute bottom-20 left-1/2 -translate-x-1/2">
                    <motion.div
                      key={reps}
                      initial={{ scale: 1.3 }}
                      animate={{ scale: 1 }}
                      className="bg-dark-900/90 backdrop-blur-sm rounded-2xl px-10 py-4 border-2 border-accent-500/50"
                    >
                      <p className="text-6xl font-black text-accent-400 text-center">{reps}</p>
                      <p className="text-xs text-gray-400 text-center uppercase tracking-widest mt-1">REPS</p>
                    </motion.div>
                  </div>

                  {/* Form Feedback Bar - Bottom */}
                  <div className="absolute bottom-0 left-0 right-0 p-4">
                    <AnimatePresence mode="wait">
                      <motion.div
                        key={feedback}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className={`flex items-center justify-center gap-3 rounded-xl py-3 px-6 backdrop-blur-sm ${isPerfectForm
                          ? 'bg-accent-500/90'
                          : isGoodForm
                            ? 'bg-accent-500/20 border border-accent-500/50'
                            : 'bg-primary-500/90'
                          }`}
                      >
                        {isGoodForm ? (
                          <CheckCircle className="w-5 h-5 text-white" />
                        ) : (
                          <AlertTriangle className="w-5 h-5 text-white" />
                        )}
                        <span className="font-bold text-white text-lg">{feedback}</span>
                      </motion.div>
                    </AnimatePresence>
                  </div>
                </div>
              </div>

              {/* Control Bar */}
              <div className="p-4 bg-dark-800 border-t border-dark-700">
                <div className="flex items-center justify-center gap-4">
                  {!isActive ? (
                    <motion.button
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => setIsActive(true)}
                      className="bg-accent-500 hover:bg-accent-600 text-white font-bold py-4 px-12 rounded-xl flex items-center gap-3 transition-colors"
                    >
                      <Play className="w-6 h-6" />
                      <span className="text-lg">Start Training</span>
                    </motion.button>
                  ) : (
                    <>
                      <motion.button
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={() => setIsActive(false)}
                        className="bg-dark-700 hover:bg-dark-600 text-white font-bold py-3 px-8 rounded-xl flex items-center gap-2 transition-colors border border-dark-600"
                      >
                        <Pause className="w-5 h-5" />
                        Pause
                      </motion.button>
                      <motion.button
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={resetSession}
                        className="bg-primary-500/20 hover:bg-primary-500/30 text-primary-400 font-bold py-3 px-8 rounded-xl flex items-center gap-2 transition-colors border border-primary-500/50"
                      >
                        <RotateCcw className="w-5 h-5" />
                        Reset
                      </motion.button>
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Sidebar - Compact Stats */}
          <div className="space-y-4">
            {/* Session Summary */}
            <div className="bg-dark-800 rounded-xl p-5 border border-dark-700">
              <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-4">Session</h3>

              <div className="space-y-4">
                {/* Total Reps */}
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Total Reps</span>
                  <span className="text-2xl font-black text-white">{sessionStats.totalReps}</span>
                </div>

                {/* Good Form */}
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Good Form</span>
                  <span className="text-2xl font-black text-accent-400">{sessionStats.goodFormReps}</span>
                </div>

                {/* Duration */}
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Duration</span>
                  <span className="text-xl font-bold text-white">{formatTime(elapsedTime)}</span>
                </div>
              </div>
            </div>

            {/* Current State */}
            <div className="bg-dark-800 rounded-xl p-5 border border-dark-700">
              <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-4">Status</h3>

              {/* Form Quality Indicator */}
              <div className={`rounded-lg p-4 text-center ${isGoodForm
                ? 'bg-accent-500/20 border border-accent-500/30'
                : 'bg-primary-500/20 border border-primary-500/30'
                }`}>
                <div className="flex items-center justify-center gap-2 mb-1">
                  {isGoodForm ? (
                    <CheckCircle className="w-6 h-6 text-accent-400" />
                  ) : (
                    <AlertTriangle className="w-6 h-6 text-primary-400" />
                  )}
                  <span className={`text-xl font-black ${isGoodForm ? 'text-accent-400' : 'text-primary-400'}`}>
                    {formQuality}
                  </span>
                </div>
                <p className="text-xs text-gray-400 uppercase tracking-wider">Form Quality</p>
              </div>

              {/* Stage */}
              <div className="mt-4 bg-dark-700/50 rounded-lg p-3 text-center">
                <p className="text-lg font-bold text-warning-400 uppercase">{stage}</p>
                <p className="text-xs text-gray-500">Current Stage</p>
              </div>
            </div>

            {/* Tips */}
            <div className="bg-gradient-to-br from-primary-500/10 to-accent-500/10 rounded-xl p-5 border border-primary-500/20">
              <div className="flex items-center gap-2 mb-3">
                <Flame className="w-4 h-4 text-warning-400" />
                <h3 className="text-sm font-bold text-white">Pro Tips</h3>
              </div>
              <ul className="space-y-2 text-xs text-gray-300">
                <li className="flex items-start gap-2">
                  <span className="text-accent-400">•</span>
                  Full body visible in frame
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-accent-400">•</span>
                  Good lighting helps detection
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-accent-400">•</span>
                  Follow form feedback
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default WorkoutSession
