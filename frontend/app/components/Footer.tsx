import Link from "next/link"

export default function Footer() {
  return (
    <footer className="py-16 bg-[#0f0f1a] text-gray-300">
      <div className="container mx-auto px-4">
        <div className="grid md:grid-cols-4 gap-8">
          <div>
            <h3 className="text-white font-medium mb-4">TE Connectivity</h3>
            <p className="text-sm">Creating a safer, sustainable, productive, and connected future.</p>
          </div>

          <div>
            <h3 className="text-white font-medium mb-4">Quick Links</h3>
            <ul className="space-y-2 text-sm">
              <li><Link href="/upload" className="hover:text-white transition-colors">Part Recognition</Link></li>
              <li><Link href="/train" className="hover:text-white transition-colors">Upload Training Data</Link></li>
              <li><Link href="#" className="hover:text-white transition-colors">TE Product Catalog</Link></li>
            </ul>
          </div>

          <div>
            <h3 className="text-white font-medium mb-4">Resources</h3>
            <ul className="space-y-2 text-sm">
              <li><Link href="#" className="hover:text-white transition-colors">TE Support</Link></li>
              <li><Link href="#" className="hover:text-white transition-colors">Technical Documents</Link></li>
              <li><Link href="#" className="hover:text-white transition-colors">API Documentation</Link></li>
            </ul>
          </div>

          <div>
            <h3 className="text-white font-medium mb-4">Contact</h3>
            <ul className="space-y-2 text-sm">
              <li><Link href="#" className="hover:text-white transition-colors">Technical Support</Link></li>
              <li><Link href="#" className="hover:text-white transition-colors">Sales Inquiries</Link></li>
              <li><Link href="#" className="hover:text-white transition-colors">Feedback</Link></li>
            </ul>
          </div>
        </div>

        <div className="mt-12 pt-8 border-t border-gray-800 text-center text-sm">
          <p>© 2024 TE Connectivity. All rights reserved.</p>
        </div>
      </div>
    </footer>
  )
}
