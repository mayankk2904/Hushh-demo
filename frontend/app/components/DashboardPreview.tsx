import { Button } from "@/components/ui/button"
import { Barcode } from "lucide-react"

export default function DashboardPreview() {
  const parts = [
    { part: "1-2314-K56", status: "Verified" },
    { part: "5-8821-MN9", status: "Pending" },
    { part: "3-4490-PQ2", status: "Verified" },
  ]

  return (
    <section className="container mx-auto px-4 pb-24 relative z-10">
      <div className="bg-white rounded-3xl shadow-2xl overflow-hidden max-w-6xl mx-auto border border-gray-200">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 p-8">
          <div className="space-y-6">
            <h3 className="text-xl font-bold mb-4">System Performance</h3>
            <div className="space-y-4">
              <div className="p-4 bg-gray-50 rounded-lg">
                <p className="text-sm font-medium text-gray-600">TE Parts in Database</p>
                <p className="text-3xl font-bold">25,000+</p>
              </div>
              <div className="p-4 bg-gray-50 rounded-lg">
                <p className="text-sm font-medium text-gray-600">Recognition Accuracy</p>
                <p className="text-3xl font-bold text-green-600">99.2%</p>
              </div>
            </div>
          </div>

          <div className="lg:col-span-2 bg-gradient-to-br from-orange-500 to-amber-600 rounded-xl p-6 text-white">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-lg font-semibold">Recent TE Part Identifications</h3>
              <Button variant="ghost" className="text-white/80 hover:text-white">View All</Button>
            </div>
            <div className="space-y-4">
              {parts.map((item, index) => (
                <div key={index} className="flex justify-between items-center p-3 bg-orange-400/20 rounded-lg">
                  <div className="flex items-center gap-3">
                    <Barcode className="w-5 h-5" />
                    <span className="font-medium">{item.part}</span>
                  </div>
                  <span className={`px-2 py-1 rounded-full text-sm ${item.status === 'Verified' ? 'bg-green-100/20' : 'bg-amber-100/20'}`}>
                    {item.status}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
