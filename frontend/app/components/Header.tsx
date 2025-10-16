"use client"

import Link from "next/link"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { User } from "lucide-react"
import AuthModal from "./AuthModal";
import { ShieldCheck } from "lucide-react"
export default function Header() {
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [username, setUsername] = useState<string | null>(null);

  return (
    <header className="bg-[#0f0f1a] rounded-b-[40px] relative overflow-hidden">
      <div className="container mx-auto px-4 py-6">
        {/* Navbar */}
        <nav className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-full flex items-center justify-center text-white">
              <span className="text-xl"><img src="/1.jpeg" alt="TE" /></span>
            </div>
            <span className="font-bold text-white">TE CONNECTIVITY</span>
          </div>

          <div className="hidden md:flex items-center space-x-8 text-gray-300">
            <Link href="/" className="text-white font-medium">Home</Link>
            <Link href="/upload" className="text-gray-400 hover:text-white">Predict</Link>
            <Link href="/train" className="text-gray-400 hover:text-white">Upload Training Images</Link>
            <Link href="#features" className="text-gray-400 hover:text-white">Features</Link>
            <Link href="#about" className="text-gray-400 hover:text-white">About TE</Link>
          </div>

          {/* User icon and username */}
          <div 
            className="flex items-center gap-2 cursor-pointer"
            onClick={() => setIsAuthModalOpen(true)} // 👈 open modal on click
          >
            <User className="text-white w-6 h-6" />
            {username && <span className="text-white">{username}</span>}
          </div>
        </nav>
      </div>

      {/* Hero */}
      <div className="container mx-auto px-4 py-24 relative">
        <div className="text-center max-w-3xl mx-auto relative z-10">
          <div className="mb-8 flex items-center justify-center gap-2 text-orange-500">
            <ShieldCheck className="w-6 h-6" />
            <p className="uppercase tracking-wider font-medium">Industrial-Grade Part Recognition</p>
          </div>

          <h1 className="text-4xl md:text-6xl font-bold text-white mb-6 leading-tight">
            TE Part Number Recognition System
          </h1>

          <p className="text-xl text-gray-300 mb-8 max-w-2xl mx-auto">
            Automatically identify TE Connectivity's parts through part numbers from images with our AI-powered recognition system.
          </p>
        </div>
      </div>

            {/* Auth Modal */}
            <AuthModal isOpen={isAuthModalOpen} 
            onClose={() => setIsAuthModalOpen(false)}
              onLogin={(username) => setUsername(username)} />
    </header>
  )
}
