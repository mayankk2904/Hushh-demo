"use client"

import { useState } from "react"

export default function RegisterForm({ onLogin }: { onLogin: (username: string) => void }) {
  const [email, setEmail] = useState("")
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)

    try {
      const res = await fetch("http://127.0.0.1:8000/auth/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, email, password })
      })

      const data = await res.json()

      if (!res.ok) {
        throw new Error(data.detail || "Registration failed")
      }

      // On successful registration, automatically log the user in
      // You could either:
      // 1. Call onLogin directly (as shown below)
      // 2. Or make another API call to login after registration
      onLogin(username)

    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4">
     
      <input 
        type="text" 
        placeholder="Username" 
        value={username}
        onChange={(e) => setUsername(e.target.value)}
        required
        className="border p-2 rounded"
      />
       <input 
        type="email" 
        placeholder="Email" 
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        required
        className="border p-2 rounded"
      />
      <input 
        type="password" 
        placeholder="Password" 
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        required
        className="border p-2 rounded"
      />

      {error && <p className="text-red-500">{error}</p>}

      <button 
        type="submit" 
        className="bg-orange-500 text-white py-2 rounded hover:bg-orange-600"
        disabled={loading}
      >
        {loading ? "Registering..." : "Register"}
      </button>
    </form>
  )
}