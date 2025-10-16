import { useState } from "react";
import { motion } from "framer-motion";
import LoginForm from "./LoginForm";
import RegisterForm from "./RegisterForm";

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  onLogin: (username: string) => void; // 👈 you already added this
}

export default function AuthModal({ isOpen, onClose, onLogin }: AuthModalProps) {
  const [isLogin, setIsLogin] = useState(true);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
      <motion.div
        initial={{ scale: 0.8 }}
        animate={{ scale: 1 }}
        className="bg-white rounded-2xl shadow-2xl p-8 w-[90%] max-w-md relative"
      >
        <button
          onClick={onClose}
          className="absolute top-3 right-3 text-gray-500 hover:text-black"
        >
          ✖
        </button>

        <div className="mb-6 text-center">
          <h2 className="text-2xl font-bold">
            {isLogin ? "Welcome Back" : "Create an Account"}
          </h2>
          <p className="text-gray-500 text-sm">
            {isLogin ? "Login to continue" : "Register to get started"}
          </p>
        </div>

        {/* 🔥 pass the onLogin prop to LoginForm */}
        {isLogin ? <LoginForm onLogin={onLogin} /> : <RegisterForm  onLogin={onLogin} />}

        <div className="mt-4 text-center text-sm text-gray-600">
          {isLogin ? "Don't have an account?" : "Already have an account?"}
          <button
            onClick={() => setIsLogin(!isLogin)}
            className="ml-1 text-blue-600 hover:underline"
          >
            {isLogin ? "Register" : "Login"}
          </button>
        </div>
      </motion.div>
    </div>
  );
}
