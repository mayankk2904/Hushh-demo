"use client";

import { useState } from "react";
import AuthModal from "./AuthModal";

export default function HeroSection() {
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [username, setUsername] = useState<string | null>(null); // 👈 add username state (optional)

  return (
    <section className="flex flex-col items-center justify-center text-center py-24 px-6">
      <h1 className="text-5xl font-bold mb-6">Build Your Model in Minutes</h1>
      <p className="text-lg text-gray-700 mb-8 max-w-2xl">
        Easily upload datasets, train, and deploy with our intuitive dashboard.
      </p>
      <button
        onClick={() => setIsAuthModalOpen(true)}
        className="bg-black text-white rounded-xl py-3 px-8 text-lg hover:bg-gray-800 transition"
      >
        Start Building Your Model!
      </button>

      {/* Auth Modal */}
      <AuthModal 
        isOpen={isAuthModalOpen} 
        onClose={() => setIsAuthModalOpen(false)} 
        onLogin={(username) => {
          setUsername(username); // save username (if you want)
          setIsAuthModalOpen(false); // 👈 Auto-close modal after login (good UX)
        }} 
      />
    </section>
  );
}
