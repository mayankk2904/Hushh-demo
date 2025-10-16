import { Button } from "@/components/ui/button"
import { CheckCircle } from "lucide-react"

export default function AboutTE() {
  const points = [
    "75+ years of industrial expertise",
    "14,000+ engineers worldwide",
    "140+ countries served"
  ]

  const whyMatters = [
    "Reduce identification errors in inventory management",
    "Speed up part lookup processes in warehouses",
    "Maintain accurate digital records of physical components"
  ]

  return (
    <section id="about" className="py-16 bg-white">
      <div className="container mx-auto px-4">
        <div className="grid md:grid-cols-2 gap-12 items-center">
          <div className="space-y-6">
            <h2 className="text-3xl font-bold mb-4">About TE Connectivity</h2>
            <p className="text-gray-600">TE Connectivity is a global industrial technology leader with a proven history of innovation, serving customers in harsh environments for more than 75 years.</p>
            <p className="text-gray-600">Our solutions enable the electrification of everything, from electric vehicles and aircraft to factories and homes.</p>
            <ul className="space-y-4">
              {points.map((point, index) => (
                <li key={index} className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-orange-100 rounded-full flex items-center justify-center">
                    <CheckCircle className="w-4 h-4 text-orange-600" />
                  </div>
                  <span className="font-medium">{point}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="bg-gradient-to-br from-[#0f0f1a] to-[#1a1a2f] rounded-2xl p-8 text-white">
            <div className="space-y-6">
              <h3 className="text-2xl font-bold">Why This Tool Matters</h3>
              <div className="space-y-4">
                {whyMatters.map((reason, index) => (
                  <div key={index} className="flex items-start gap-3">
                    <CheckCircle className="w-5 h-5 text-orange-400 flex-shrink-0 mt-1" />
                    <span>{reason}</span>
                  </div>
                ))}
              </div>
              <Button className="w-full mt-6" variant="outline">
                Learn More About TE
              </Button>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
