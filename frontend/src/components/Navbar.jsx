import { Link } from 'react-router-dom'

const Navbar = () => {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-dark-900/90 backdrop-blur-sm border-b border-dark-800">
      <div className="max-w-6xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3">
            <img src="/assets/logo.svg" alt="AlphaReps" className="w-9 h-9 rounded-lg" />
            <span className="text-xl font-black">
              <span className="bg-gradient-to-r from-primary-400 to-accent-500 bg-clip-text text-transparent">ALPHA</span>
              <span className="text-white">REPS</span>
            </span>
          </Link>

          {/* CTA */}
          <Link to="/login">
            <button className="bg-accent-500 hover:bg-accent-600 text-white px-5 py-2 rounded-lg font-bold text-sm transition-colors">
              Get Started
            </button>
          </Link>
        </div>
      </div>
    </nav>
  )
}

export default Navbar
