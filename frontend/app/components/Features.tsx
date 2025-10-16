import { ScanEye, PackageCheck, Factory, LinkIcon } from "lucide-react"

export default function Features() {
  const features = [
    {
      icon: <ScanEye className="w-5 h-5 text-orange-500" />,
      title: "Instant Recognition",
      description: "Identify TE part numbers from images in seconds with our optimized machine learning models."
    },
    {
      icon: <PackageCheck className="w-5 h-5 text-orange-500" />,
      title: "TE-Specific Database",
      description: "Comprehensive database of TE Connectivity parts with detailed specifications."
    },
    {
      icon: <Factory className="w-5 h-5 text-orange-500" />,
      title: "Industrial Focus",
      description: "Designed for industrial environments with high accuracy on stamped or printed part numbers."
    },
    {
      icon: <LinkIcon className="w-5 h-5 text-orange-500" />,
      title: "Inventory Integration",
      description: "Connect directly with your inventory management system for seamless updates."
    }
  ]

  return (
    <section id="features" className="py-16">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">Key Features</h2>
          <p className="text-gray-700 max-w-2xl mx-auto">
            Our specialized tool is designed specifically for TE Connectivity part number recognition
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature, index) => (
            <div key={index} className="bg-white rounded-xl p-6 shadow-md hover:shadow-lg transition-shadow">
              <div className="mb-4 flex items-center gap-2">
                <div className="p-2 bg-orange-100 rounded-lg">
                  {feature.icon}
                </div>
                <h3 className="font-semibold">{feature.title}</h3>
              </div>
              <p className="text-sm text-gray-600 mb-4">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
